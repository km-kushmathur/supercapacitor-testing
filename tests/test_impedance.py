import unittest
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

try:
    from tests.test_gamry_loaders import load_gamry_dta
except ImportError:
    from test_gamry_loaders import load_gamry_dta


def fit_series_rc(frequencies, z_real, z_imag):
    """
    Fits a series RC model: Z = Rs - j / (2 * pi * f * C)
    Rs ~ mean(Z_real at high frequency)
    C ~ -1 / (2 * pi * f * Z_imag at low frequency)
    """
    frequencies = np.asarray(frequencies, dtype=float)
    z_real = np.asarray(z_real, dtype=float)
    z_imag = np.asarray(z_imag, dtype=float)

    # Rs from highest frequencies
    rs = float(np.median(z_real[frequencies >= np.percentile(frequencies, 90)]))

    # C from lowest non-zero frequencies
    low_mask = (frequencies <= np.percentile(frequencies, 20)) & (z_imag < 0)
    if low_mask.any():
        c_vals = -1.0 / (2.0 * np.pi * frequencies[low_mask] * z_imag[low_mask])
        c = float(np.median(c_vals))
    else:
        c = float(-1.0 / (2.0 * np.pi * frequencies[-1] * z_imag[-1]))

    return {"Rs_ohm": rs, "Capacitance_F": c}


def fit_parallel_rc(frequencies, z, guess=(10.0, 10.0, 25.0)):
    """
    Fits 1Rs/1Rct model: Z = Rs + Rct / (1 + 1j * 2 * pi * f * Rct * C)
    Uses non-linear least squares.
    """
    from scipy.optimize import curve_fit

    f = np.asarray(frequencies, dtype=float)
    omega = 2.0 * np.pi * f

    def model_func(_, rs, rct, cap):
        denom = 1.0 + (omega * rct * cap) ** 2
        zr = rs + rct / denom
        zi = -omega * (rct**2) * cap / denom
        return np.concatenate([zr, zi])

    ydata = np.concatenate([z.real, z.imag])
    popt, _ = curve_fit(
        model_func,
        f,
        ydata,
        p0=guess,
        bounds=(0, [np.inf, np.inf, np.inf]),
        maxfev=10000,
    )
    rs, rct, cap = popt
    return {"Rs_ohm": float(rs), "Rct_ohm": float(rct), "Capacitance_F": float(cap)}


class TestImpedance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = DATA_DIR / "gamry_impedance_sample.DTA"
        if not cls.sample_path.is_file():
            # Fallback to gamry_eis_sample.DTA if present
            cls.sample_path = DATA_DIR / "gamry_eis_sample.DTA"

    def test_impedance_file_exists(self):
        self.assertTrue(self.sample_path.is_file(), f"Missing impedance data file: {self.sample_path}")

    def test_impedance_metadata(self):
        header, tables = load_gamry_dta(self.sample_path)
        self.assertIn("TAG", header)
        self.assertEqual(header["TAG"], "EISPOT")
        self.assertIn("ZCURVE", tables)

    def test_impedance_data_validity(self):
        _, tables = load_gamry_dta(self.sample_path)
        zcurve = tables["ZCURVE"]
        self.assertGreater(len(zcurve), 0)

        required_cols = ["Freq", "Zreal", "Zimag"]
        for col in required_cols:
            self.assertIn(col, zcurve.columns)

        f = zcurve["Freq"].astype(float).values
        zr = zcurve["Zreal"].astype(float).values
        zi = zcurve["Zimag"].astype(float).values

        self.assertTrue((f > 0).all(), "All frequencies must be positive")
        self.assertTrue((zr > 0).all(), "Real impedance must be positive")
        self.assertTrue(np.isfinite(zi).all(), "Imaginary impedance must be finite")

    def test_impedance_series_rc_fit(self):
        _, tables = load_gamry_dta(self.sample_path)
        zcurve = tables["ZCURVE"]
        f = zcurve["Freq"].astype(float).values
        zr = zcurve["Zreal"].astype(float).values
        zi = zcurve["Zimag"].astype(float).values

        fit = fit_series_rc(f, zr, zi)
        self.assertGreater(fit["Rs_ohm"], 0.0)
        self.assertGreater(fit["Capacitance_F"], 0.0)

    def test_impedance_parallel_rc_fit(self):
        _, tables = load_gamry_dta(self.sample_path)
        zcurve = tables["ZCURVE"]
        f = zcurve["Freq"].astype(float).values
        zr = zcurve["Zreal"].astype(float).values
        zi = zcurve["Zimag"].astype(float).values
        z = zr + 1j * zi

        try:
            fit = fit_parallel_rc(f, z)
            self.assertGreater(fit["Rs_ohm"], 0.0)
            self.assertGreater(fit["Rct_ohm"], 0.0)
            self.assertGreater(fit["Capacitance_F"], 0.0)
        except ImportError:
            self.skipTest("scipy not installed; skipping non-linear fit test")

    def test_impedance_package_if_available(self):
        """Optional test that runs if 'impedance' (impedance.py) is installed."""
        try:
            from impedance import preprocessing
            from impedance.models.circuits import CustomCircuit
        except ImportError:
            self.skipTest("impedance.py package not installed in environment")

        f, z = preprocessing.readGamry(str(self.sample_path))
        self.assertGreater(len(f), 0)
        circuit = CustomCircuit("R0-C1", initial_guess=[10, 25])
        circuit.fit(f, z)
        self.assertGreater(len(circuit.parameters_), 0)


if __name__ == "__main__":
    unittest.main()

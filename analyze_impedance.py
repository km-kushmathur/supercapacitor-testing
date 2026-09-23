"""
Impedance analysis and Nyquist plotting for Gamry .DTA electrochemical files.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# Default impedance data file
DEFAULT_DATA_FILE = BASE_DIR / "data" / "gamry_impedance_sample.DTA"


def load_gamry_impedance(filename):
    """
    Loads frequency and complex impedance from a Gamry .DTA file.
    Prefers impedance.preprocessing.readGamry if installed, otherwise uses built-in loader.
    """
    try:
        from impedance import preprocessing
        frequencies, z = preprocessing.readGamry(str(filename))
        return frequencies, z
    except ImportError:
        from tests.test_gamry_loaders import load_gamry_dta
        _, tables = load_gamry_dta(filename)
        zcurve = tables["ZCURVE"]
        frequencies = zcurve["Freq"].astype(float).values
        z_real = zcurve["Zreal"].astype(float).values
        z_imag = zcurve["Zimag"].astype(float).values
        return frequencies, z_real + 1j * z_imag


def fit_equivalent_circuit(frequencies, z, circuit_string="R0-C1", initial_guesses=(10.0, 25.0)):
    """
    Fits equivalent circuit model to impedance data.
    """
    try:
        from impedance.models.circuits import CustomCircuit
        circuit = CustomCircuit(circuit_string, initial_guess=list(initial_guesses))
        circuit.fit(frequencies, z)
        z_fit = circuit.predict(frequencies)
        return circuit, z_fit
    except ImportError:
        # Fallback using pure numpy series RC calculation
        from tests.test_impedance import fit_series_rc
        fit_results = fit_series_rc(frequencies, z.real, z.imag)
        rs = fit_results["Rs_ohm"]
        cap = fit_results["Capacitance_F"]
        omega = 2.0 * np.pi * frequencies
        z_fit = rs - 1j / (omega * cap)
        return fit_results, z_fit


def plot_nyquist_squared(z_raw, z_fit=None, title="Nyquist Plot"):
    """
    Plots Nyquist curve with equal 1:1 aspect ratio and squared axis limits.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(z_raw.real, -z_raw.imag, "o", label="Raw Gamry Data", markersize=5)
    if z_fit is not None:
        ax.plot(z_fit.real, -z_fit.imag, "-", label="Fitted Circuit", linewidth=2)

    # Make Nyquist geometry visually correct: 1 ohm on x equals 1 ohm on y.
    all_real = [val.real for val in z_raw] + ([val.real for val in z_fit] if z_fit is not None else [])
    all_neg_imag = [-val.imag for val in z_raw] + ([-val.imag for val in z_fit] if z_fit is not None else [])

    real_min, real_max = min(all_real), max(all_real)
    imag_min, imag_max = min(all_neg_imag), max(all_neg_imag)

    max_span = max(real_max - real_min, imag_max - imag_min)
    padding = 0.05 * max_span if max_span > 0 else 1.0

    real_center = 0.5 * (real_min + real_max)
    imag_center = 0.5 * (imag_min + imag_max)
    half_span = 0.5 * max_span + padding

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(real_center - half_span, real_center + half_span)
    ax.set_ylim(imag_center - half_span, imag_center + half_span)

    ax.set_xlabel(r"$Z_{real}\ (\Omega)$", fontsize=12)
    ax.set_ylabel(r"$-Z_{imag}\ (\Omega)$", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)

    return fig, ax


def main():
    data_path = DEFAULT_DATA_FILE
    print(f"Loading Gamry impedance file: {data_path}")
    frequencies, z = load_gamry_impedance(data_path)
    print(f"Loaded {len(frequencies)} points (Freq: {frequencies.min():.4g} to {frequencies.max():.4g} Hz)")

    circuit, z_fit = fit_equivalent_circuit(frequencies, z)
    print("Circuit fit results:")
    print(circuit)

    fig, _ = plot_nyquist_squared(z, z_fit, title="Gamry Impedance Fit")
    output_png = BASE_DIR / "nyquist_plot.png"
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Nyquist plot saved to: {output_png}")


if __name__ == "__main__":
    main()

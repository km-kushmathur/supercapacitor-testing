import unittest
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestImpedance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = DATA_DIR / "gamry_impedance_sample.DTA"

    def test_impedance_file_exists(self):
        self.assertTrue(self.sample_path.is_file(), f"Missing impedance data file: {self.sample_path}")

    def test_impedance_read_gamry(self):
        """Test loading impedance data via official impedance package (impedance.preprocessing)."""
        try:
            from impedance import preprocessing
        except ImportError:
            self.skipTest("impedance package not installed; run: pip install -r requirements.txt")

        frequencies, Z = preprocessing.readGamry(str(self.sample_path))
        self.assertEqual(len(frequencies), 86)
        self.assertEqual(len(Z), 86)
        self.assertTrue((frequencies > 0).all())
        self.assertTrue(np.isfinite(Z).all())

    def test_impedance_custom_circuit_fit(self):
        """Test fitting equivalent circuit R0-C1 using impedance.models.circuits.CustomCircuit."""
        try:
            from impedance import preprocessing
            from impedance.models.circuits import CustomCircuit
        except ImportError:
            self.skipTest("impedance package not installed; run: pip install -r requirements.txt")

        frequencies, Z = preprocessing.readGamry(str(self.sample_path))
        circuit = CustomCircuit("R0-C1", initial_guess=[10, 25])
        circuit.fit(frequencies, Z)

        # Ensure parameters were fitted and positive
        self.assertEqual(len(circuit.parameters_), 2)
        r0, c1 = circuit.parameters_
        self.assertGreater(r0, 0)
        self.assertGreater(c1, 0)

        # Predict and verify predictions
        z_fit = circuit.predict(frequencies)
        self.assertEqual(len(z_fit), len(frequencies))
        self.assertTrue(np.isfinite(z_fit).all())


if __name__ == "__main__":
    unittest.main()

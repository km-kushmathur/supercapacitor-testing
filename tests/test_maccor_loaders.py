import unittest
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_maccor_excel(file_path):
    """
    Parses a MACCOR exported Excel (.xlsx) file into a cleaned pandas DataFrame.

    Parameters:
        file_path (str or Path): Path to the Excel file.

    Returns:
        pd.DataFrame: Cleaned cycling data.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path)
    required = ["Cyc#", "Amps", "Volts", "Test (Min)"]
    df = df.dropna(subset=[col for col in required if col in df.columns])

    numeric_cols = ["Cyc#", "Step", "Test (Min)", "Step (Min)", "Amps", "Volts"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_maccor_csv(file_path):
    """
    Parses a MACCOR exported CSV file into a cleaned pandas DataFrame.

    Parameters:
        file_path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned cycling data.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    required = ["Cyc#", "Amps", "Volts", "Test (Min)"]
    df = df.dropna(subset=[col for col in required if col in df.columns])

    numeric_cols = ["Cyc#", "Step", "Test (Min)", "Step (Min)", "Amps", "Volts"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


class TestMaccorLoaders(unittest.TestCase):
    def test_load_maccor_excel(self):
        sample_path = DATA_DIR / "maccor_cycling_sample.xlsx"
        self.assertTrue(sample_path.is_file(), f"Missing file: {sample_path}")

        df = load_maccor_excel(sample_path)
        self.assertGreater(len(df), 0)

        expected_columns = ["Rec#", "Cyc#", "Step", "Test (Min)", "Amps", "Volts"]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Validate that cycle numbers and voltage ranges are sensible
        cycles = df["Cyc#"].dropna().unique()
        self.assertGreater(len(cycles), 0)
        self.assertFalse(df["Volts"].dropna().empty)

    def test_load_maccor_csv(self):
        sample_path = DATA_DIR / "maccor_cycling_sample.csv"
        self.assertTrue(sample_path.is_file(), f"Missing file: {sample_path}")

        df = load_maccor_csv(sample_path)
        self.assertGreater(len(df), 0)

        expected_columns = ["Rec#", "Cyc#", "Step", "Test (Min)", "Amps", "Volts"]
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_cycle_filtering(self):
        sample_path = DATA_DIR / "maccor_cycling_sample.csv"
        df = load_maccor_csv(sample_path)

        # Check cycle filtering
        first_cycle = df[df["Cyc#"] == 1]
        self.assertGreater(len(first_cycle), 0)
        self.assertTrue((first_cycle["Cyc#"] == 1).all())


if __name__ == "__main__":
    unittest.main()

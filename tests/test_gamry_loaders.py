import io
import unittest
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_gamry_dta(file_path):
    """
    Parses a Gamry .DTA file into header metadata and extracted tables.

    Parameters:
        file_path (str or Path): Path to the Gamry .DTA file.

    Returns:
        tuple[dict, dict[str, pd.DataFrame]]: (metadata dict, dict of table DataFrames)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="latin1") as f:
        lines = f.readlines()

    header = {}
    tables = {}
    current_table = None
    table_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = line.split("\t")
        clean_parts = [p.strip() for p in parts if p.strip()]

        if len(clean_parts) >= 2 and clean_parts[1] == "TABLE":
            if current_table and table_lines:
                tables[current_table] = pd.read_csv(
                    io.StringIO("".join(table_lines)), sep=r"\s+", skiprows=[1]
                )
            current_table = clean_parts[0]
            table_lines = []
            continue

        if current_table is not None:
            if len(clean_parts) >= 2 and clean_parts[1] in (
                "LABEL",
                "QUANT",
                "IQUANT",
                "TOGGLE",
                "SELECTOR",
                "TWOPARAM",
            ):
                if table_lines:
                    tables[current_table] = pd.read_csv(
                        io.StringIO("".join(table_lines)), sep=r"\s+", skiprows=[1]
                    )
                current_table = None
                table_lines = []
                header[clean_parts[0]] = clean_parts[2] if len(clean_parts) > 2 else ""
            else:
                table_lines.append(line)
        else:
            if len(clean_parts) == 2:
                header[clean_parts[0]] = clean_parts[1]
            elif len(clean_parts) >= 3:
                header[clean_parts[0]] = clean_parts[2]

    if current_table and table_lines:
        tables[current_table] = pd.read_csv(
            io.StringIO("".join(table_lines)), sep=r"\s+", skiprows=[1]
        )

    return header, tables


def find_gamry_cv_header(path):
    """Find the zero-indexed line number where tabular CV data begins."""
    with Path(path).open(encoding="utf-8-sig") as file:
        for line_num, line in enumerate(file):
            columns = [col.strip() for col in line.rstrip("\n").split("\t")]
            columns = [col for col in columns if col]
            if len(columns) >= 4 and columns[:4] == ["#", "s", "V vs. Ref.", "A"]:
                return line_num
    raise ValueError("Could not locate Gamry CV data header row.")


def load_gamry_cv_csv(file_path):
    """
    Parses a Gamry exported CV .csv file into a cleaned pandas DataFrame.

    Parameters:
        file_path (str or Path): Path to the CV CSV file.

    Returns:
        pd.DataFrame: Tabular CV data.
    """
    path = Path(file_path)
    header_row = find_gamry_cv_header(path)
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_row,
        engine="python",
        encoding="utf-8-sig",
    )
    df.columns = df.columns.str.strip()
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed)
    df["V vs. Ref."] = pd.to_numeric(df["V vs. Ref."], errors="coerce")
    df["A"] = pd.to_numeric(df["A"], errors="coerce")
    return df


class TestGamryLoaders(unittest.TestCase):
    def test_load_gamry_eis_dta(self):
        sample_path = DATA_DIR / "gamry_eis_sample.DTA"
        self.assertTrue(sample_path.is_file(), f"Missing file: {sample_path}")

        header, tables = load_gamry_dta(sample_path)
        self.assertIn("TAG", header)
        self.assertEqual(header["TAG"], "EISPOT")
        self.assertIn("ZCURVE", tables)

        zcurve = tables["ZCURVE"]
        self.assertGreater(len(zcurve), 0)
        for col in ["Freq", "Zreal", "Zimag"]:
            self.assertIn(col, zcurve.columns)

    def test_load_gamry_cv_dta(self):
        sample_path = DATA_DIR / "gamry_cv_sample.DTA"
        self.assertTrue(sample_path.is_file(), f"Missing file: {sample_path}")

        header, tables = load_gamry_dta(sample_path)
        self.assertIn("TAG", header)
        self.assertEqual(header["TAG"], "PWR800_CV")
        self.assertIn("CURVE1", tables)

        curve1 = tables["CURVE1"]
        self.assertGreater(len(curve1), 0)
        for col in ["T", "Vf", "Im"]:
            self.assertIn(col, curve1.columns)

    def test_load_gamry_cv_csv(self):
        sample_path = DATA_DIR / "gamry_cv_sample.csv"
        self.assertTrue(sample_path.is_file(), f"Missing file: {sample_path}")

        df = load_gamry_cv_csv(sample_path)
        self.assertGreater(len(df), 0)
        self.assertIn("V vs. Ref.", df.columns)
        self.assertIn("A", df.columns)
        self.assertFalse(df["V vs. Ref."].dropna().empty)
        self.assertFalse(df["A"].dropna().empty)


if __name__ == "__main__":
    unittest.main()

import pandas as pd
import time
from pathlib import Path

start_time = time.time()
BASE_DIR = Path(__file__).resolve().parent

scan_rate = 5  # Scan rate set for the test in units of mV/s
V = 3          # Input defined voltage in units of V

# Load the test data file
file_path = BASE_DIR / "data" / "gamry_cv_sample.csv"


def find_data_header(path):
    with path.open(encoding='utf-8-sig') as file:
        for line_num, line in enumerate(file):
            columns = [column.strip() for column in line.rstrip('\n').split('\t')]
            columns = [column for column in columns if column]
            if len(columns) >= 4 and columns[:4] == ['#', 's', 'V vs. Ref.', 'A']:
                return line_num
    raise ValueError("Could not find the CV data header row.")


def load_cv_data(path):
    header_row = find_data_header(path)
    data = pd.read_csv(path, sep='\t', skiprows=header_row, engine='python', encoding='utf-8-sig')
    data.columns = data.columns.str.strip()
    unnamed_columns = [column for column in data.columns if column.startswith('Unnamed')]
    return data.drop(columns=unnamed_columns)


try:
    df = load_cv_data(file_path)
except FileNotFoundError:
    raise SystemExit(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    raise SystemExit(f"An error occurred: {e}")

# Clean the data
df['V vs. Ref.'] = pd.to_numeric(df['V vs. Ref.'], errors='coerce')
df['A'] = pd.to_numeric(df['A'], errors='coerce')
df = df.dropna(subset=['V vs. Ref.', 'A']).reset_index(drop=True)


def capacitance_calc(data):
    # Grab data for charge and discharge cycle based on positive and negative current index
    is_pos = data['A'] > 0
    if not is_pos.any():
        raise ValueError("No positive-current charging data was found.")

    first_pos_idx = is_pos.idxmax()
    cycle_data = data.loc[first_pos_idx:]

    is_neg = cycle_data['A'] < 0
    if not is_neg.any():
        raise ValueError("No negative-current discharging data was found.")

    first_neg_idx = is_neg.idxmax()
    pos_data = cycle_data.loc[:first_neg_idx].iloc[:-1]

    neg_2_end = cycle_data.loc[first_neg_idx:]
    is_pos = neg_2_end['A'] > 0
    if is_pos.any():
        first_pos_idx = is_pos.idxmax()
        neg_data = data.loc[first_neg_idx:first_pos_idx].iloc[:-1]
    else:
        neg_data = neg_2_end

    if pos_data.empty or neg_data.empty:
        raise ValueError("Could not split the data into charge and discharge segments.")

    # Charging capacitance
    pos_len = len(pos_data)
    pos_sum = pos_data['A'].sum()
    ch_integration = 0.5 * (V / pos_len) * (2 * pos_sum)
    C_ch = ch_integration / (V * (scan_rate / 1000))

    # Discharging capacitance
    neg_len = len(neg_data)
    neg_sum = neg_data['A'].sum()
    dch_integration = 0.5 * (V / neg_len) * (2 * neg_sum)
    C_dch = -dch_integration / (V * (scan_rate / 1000))

    return C_ch, C_dch


# Display capacitance values
C_ch, C_dch = capacitance_calc(df)
print(f'C_ch (F) = {C_ch:.2f}')
print(f'C_dch (F) = {C_dch:.2f}')

end_time = time.time()
print(f'Runtime: {end_time - start_time:.2f} seconds')

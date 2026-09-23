"""
Initial cleaning includes removing the top two rows of the raw file given by the MACCOR
in order to have data headers (current, voltage, time) be the first row for access
through pandas
"""

import pandas as pd
import time
from pathlib import Path
from tabulate import tabulate

start_time = time.time()
BASE_DIR = Path(__file__).resolve().parent

# Load the excel test data file
file_path = BASE_DIR / "data" / "maccor_cycling_sample.xlsx"

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    raise SystemExit(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    raise SystemExit(f"An error occurred: {e}")

# Clean the data
df = df.dropna(subset=['Cyc#', 'Amps', 'Volts', 'Test (Min)'])
df['Cyc#'] = pd.to_numeric(df['Cyc#'], errors='coerce')
df['Test (Min)'] = pd.to_numeric(df['Test (Min)'], errors='coerce')
df['Amps'] = pd.to_numeric(df['Amps'], errors='coerce')
df['Volts'] = pd.to_numeric(df['Volts'], errors='coerce')


def capacitance_resistance_calc(data):
    I_median = data['Amps'].median()
    I_median_low = I_median * 0.95
    I_median_high = I_median * 1.05

    filtered_data = data[(data['Amps'] >= I_median_low) & (data['Amps'] <= I_median_high)]

    first_row = filtered_data.iloc[0]
    last_row = filtered_data.iloc[-1]

    I = last_row['Amps']

    t1 = first_row['Test (Min)'] * 60
    t2 = last_row['Test (Min)'] * 60

    V1 = first_row['Volts']
    V2 = last_row['Volts']

    capacitance = (I * (t2 - t1)) / (V2 - V1)
    resistance = (V2 - V1) / I
    return capacitance, resistance, I_median


# Data storage
cycle_nums = df['Cyc#'].unique()
C_ch_values = []
C_dch_values = []
R_ch_values = []
R_dch_values = []
I_values = []

# Gather data
for cycle in cycle_nums:
    if cycle == 0:
        continue
    else:
        charge_data = df[(df['Cyc#'] == cycle) & (df['State'] == 'C')]
        discharge_data = df[(df['Cyc#'] == cycle) & (df['State'] == 'D')]

        C_ch, R_ch, I_m_ch = capacitance_resistance_calc(charge_data)
        C_dch, R_dch, I_m_dch = capacitance_resistance_calc(discharge_data)

        C_ch_values.append(C_ch)
        C_dch_values.append(C_dch * -1)
        R_ch_values.append(R_ch)
        R_dch_values.append(R_dch * -1)
        I_values.append(I_m_ch)

# Display values
cycle_data = {
    'Cycle': cycle_nums[1:],
    'Current (A)': I_values,
    'C_ch (F)': C_ch_values,
    'R_ch (Ω)': R_ch_values,
    'C_dch (F)': C_dch_values,
    'R_dch (Ω)': R_dch_values,
}
cycle_table = pd.DataFrame(cycle_data)

# Averages and standard deviations
grouped_table = cycle_table.copy(deep=True)
grouped_table['Current (A)'] = grouped_table['Current (A)'].round(2)
cycle_change = grouped_table['Current (A)'] != grouped_table['Current (A)'].shift()
grouped_table['ID'] = cycle_change.cumsum()
summary_table = grouped_table.groupby('ID').agg(
    current=('Current (A)', 'first'),
    avg_C_ch=('C_ch (F)', 'mean'),
    std_C_ch=('C_ch (F)', 'std'),
    avg_R_ch=('R_ch (Ω)', 'mean'),
    std_R_ch=('R_ch (Ω)', 'std'),
    avg_C_dch=('C_dch (F)', 'mean'),
    std_C_dch=('C_dch (F)', 'std'),
    avg_R_dch=('R_dch (Ω)', 'mean'),
    std_R_dch=('R_dch (Ω)', 'std'),
)

# Display averages and stds in standardized format
disp_avg_std = pd.DataFrame()
disp_avg_std['Current (A)'] = summary_table['current']

disp_avg_C_ch = summary_table['avg_C_ch'].map('{:.2f}'.format)
disp_std_C_ch = summary_table['std_C_ch'].map('{:.3f}'.format)

disp_avg_R_ch = summary_table['avg_R_ch'].map('{:.2f}'.format)
disp_std_R_ch = summary_table['std_R_ch'].map('{:.3f}'.format)

disp_avg_C_dch = summary_table['avg_C_dch'].map('{:.2f}'.format)
disp_std_C_dch = summary_table['std_C_dch'].map('{:.3f}'.format)

disp_avg_R_dch = summary_table['avg_R_dch'].map('{:.2f}'.format)
disp_std_R_dch = summary_table['std_R_dch'].map('{:.3f}'.format)

disp_avg_std['C_ch (F)'] = disp_avg_C_ch + " ± " + disp_std_C_ch
disp_avg_std['R_ch (Ω)'] = disp_avg_R_ch + " ± " + disp_std_R_ch
disp_avg_std['C_dch (F)'] = disp_avg_C_dch + " ± " + disp_std_C_dch
disp_avg_std['R_dch (Ω)'] = disp_avg_R_dch + " ± " + disp_std_R_dch

centered = ["center" for _ in range(len(disp_avg_std) + 1)]
print(tabulate(disp_avg_std, headers='keys', tablefmt='fancy_grid', showindex=False, colalign=centered))

end_time = time.time()
print(f'Runtime: {end_time - start_time:.2f} seconds')

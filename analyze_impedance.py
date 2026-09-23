import matplotlib.pyplot as plt
from pathlib import Path
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist

BASE_DIR = Path(__file__).resolve().parent

# 1. Load your Gamry .DTA file directly
filename = BASE_DIR / "data" / "gamry_impedance_sample.DTA"
frequencies, Z = preprocessing.readGamry(filename)

# 2. Define the Equivalent Circuit Model
# Option A (Series RC): 'R0-C1'
# Option B (Randles): 'R0-p(R1,C1)'

circuit_string = 'R0-C1'
initial_guesses = [10, 25]  # Provide one guess per element in your string

circuit = CustomCircuit(circuit_string, initial_guess=initial_guesses)

# 3. Fit the model to your data
circuit.fit(frequencies, Z)
print(circuit)

# 4. Predict and plot the fit
Z_fit = circuit.predict(frequencies)

fig, ax = plt.subplots()
plot_nyquist(Z, fmt='o', ax=ax, label='Raw Gamry Data')
plot_nyquist(Z_fit, fmt='-', ax=ax, label='Fitted Circuit')

# Make Nyquist geometry visually correct: 1 ohm on x equals 1 ohm on y.
# Also enforce square plotting limits so a true semicircle is not stretched.
all_real_impedance_values = [z.real for z in Z] + [z.real for z in Z_fit]
all_negative_imag_impedance_values = [-z.imag for z in Z] + [-z.imag for z in Z_fit]

real_axis_min, real_axis_max = min(all_real_impedance_values), max(all_real_impedance_values)
imag_axis_min, imag_axis_max = min(all_negative_imag_impedance_values), max(all_negative_imag_impedance_values)

max_axis_span = max(real_axis_max - real_axis_min, imag_axis_max - imag_axis_min)
axis_padding = 0.05 * max_axis_span if max_axis_span > 0 else 1.0

real_axis_center = 0.5 * (real_axis_min + real_axis_max)
imag_axis_center = 0.5 * (imag_axis_min + imag_axis_max)
half_square_span = 0.5 * max_axis_span + axis_padding

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(real_axis_center - half_square_span, real_axis_center + half_square_span)
ax.set_ylim(imag_axis_center - half_square_span, imag_axis_center + half_square_span)

plt.legend()
plt.show()

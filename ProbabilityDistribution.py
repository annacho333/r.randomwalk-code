import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Step 1: Load the CSV file
# Replace 'landslide_analysis.csv' with the actual path to your file
file_path = '/home/igv/Documents/landslide_analysis.csv'
data = pd.read_csv(file_path)

# Ensure the 'ratio' column exists
if 'ratio' not in data.columns:
    raise ValueError("The CSV file must contain a 'ratio' column.")

# Step 2: Calculate the reach angle (omega = atan(ratio))
data['omega'] = np.degrees(np.arctan(data['ratio']))

# Step 2.1: Remove rows with non-finite values in 'omega'
data = data[np.isfinite(data['omega'])]

# Ensure there are valid data points left after cleaning
if data.empty:
    raise ValueError("No valid data points left after removing non-finite values from 'omega'.")

# Step 3: Fit a Gaussian distribution to omega values
omega_values = data['omega']
mean_omega, std_omega = norm.fit(omega_values)

# Step 4: Create a range of omega values for plotting
grid = np.linspace(omega_values.min(), omega_values.max(), 500)
pdf = norm.pdf(grid, mean_omega, std_omega)

# Step 5: Normalize the histogram by bin width
hist_values, bin_edges = np.histogram(omega_values, bins=30, density=False)
bin_width = bin_edges[1] - bin_edges[0]  # Calculate bin width
hist_area = np.sum(hist_values * bin_width)  # Integrate the histogram area

# Step 6: Adjust Gaussian scaling to match histogram area
scale_factor = hist_area / np.trapz(pdf, grid)  # Scale based on total area
pdf_scaled = pdf * scale_factor

# Step 7: Plot the Gaussian distribution
plt.figure(figsize=(8, 6))
plt.plot(grid, pdf_scaled, label=f'Gaussian Fit\nMean: {mean_omega:.2f}, Std: {std_omega:.2f}', color='blue')
plt.hist(omega_values, bins=30, alpha=0.6, color='gray', label='Histogram of F(Ω)', density=False)
plt.title('Gaussian Distribution of Reach Angle (F(Ω))', fontsize=14)
plt.xlabel('Reach Angle (Ω, degrees)', fontsize=12)
plt.ylabel('F(Ω)', fontsize=12)  # Match Marchesini's axis label
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# Optional: Save the plot
# plt.savefig('reach_angle_distribution.png')


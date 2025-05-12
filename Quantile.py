import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor

# File paths
flow_accum_path = '/home/igv/Documents/Correct/PERMANENT/combined_flow_accum.tif'
slope_path = '/home/igv/Documents/Correct/PERMANENT/combined_slope.tif'
output_path = '/home/igv/Documents/Correct/PERMANENT/classified_cells.tif'
quantile_results_path = '/home/igv/Documents/QuantileLines/quantile_results.npy'  # Save path for quantile results

# Read the flow accumulation raster
with rasterio.open(flow_accum_path) as flow_accum_src:
    flow_accum_raster = flow_accum_src.read(1)
    flow_affine = flow_accum_src.transform
    flow_crs = flow_accum_src.crs
    flow_nodata = flow_accum_src.nodata

# Read the slope raster
with rasterio.open(slope_path) as slope_src:
    slope_raster = slope_src.read(1)
    slope_nodata = slope_src.nodata

# Replace nodata with NaN for processing
flow_accum_raster = np.where(flow_accum_raster == flow_nodata, np.nan, flow_accum_raster)
slope_raster = np.where(slope_raster == slope_nodata, np.nan, slope_raster)

# Create a mask for valid data
valid_mask = (~np.isnan(flow_accum_raster)) & (~np.isnan(slope_raster)) & (slope_raster > 0) & (slope_raster < 90)

# Extract valid data for quantile regression
flow_accum_valid = flow_accum_raster[valid_mask]
slope_valid = slope_raster[valid_mask]

# DEBUGGING
print("Flow Accum Valid Values:")
print(flow_accum_valid)

# Convert flow accumulation to km² and slope to tan(beta)
flow_accum_valid_km2 = flow_accum_valid * 100 / 1e6  # Assuming 10x10m cells
tan_slope_valid = np.tan(np.radians(slope_valid))

# Log-transform data
log_flow_accum = np.log(flow_accum_valid_km2)
log_tan_slope = np.log(tan_slope_valid)

# Perform Quantile Regression
quantiles = [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]
quantile_results = {}

# Plot figure
plt.figure(figsize=(10, 6))
x_vals = np.linspace(min(log_flow_accum), max(log_flow_accum), 100)
flow_accum_vals = np.exp(x_vals)  # Convert back to original flow accumulation

# Fit regression models and plot quantile lines
for q in quantiles:
    model = QuantileRegressor(quantile=q, alpha=0.01, solver='highs').fit(log_flow_accum.reshape(-1, 1), log_tan_slope)
    b = model.coef_[0]
    log_c = model.intercept_
    c = np.exp(log_c)
    quantile_results[q] = {'c': c, 'b': b}
    tan_slope_pred = c * (flow_accum_vals ** b)
    slope_deg = np.degrees(np.arctan(tan_slope_pred))
    plt.plot(flow_accum_vals, slope_deg, label=f"Q{int(q*100)}")

# Scatter plot of the data points
plt.scatter(flow_accum_valid_km2, slope_valid, color='gray', alpha=0.5, s=10, label="Data points")
plt.xscale("log")
plt.xlabel("Flow Accumulation (km²)")
plt.ylabel("Slope (degrees)")
plt.title("Quantile Regression: Slope vs Flow Accumulation")
plt.legend(title="Quantiles")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# Function to save quantile regression results (for sharing between scripts)
def save_quantile_results(quantile_results, filepath):
    np.save(filepath, quantile_results)

# Save the quantile results
save_quantile_results(quantile_results, quantile_results_path)
print(f"Quantile regression results saved to {quantile_results_path}")

    ###DEBUGGING###
# Print the quantile line equations
print("Quantile Line Functions:")
for q, results in quantile_results.items():
    c, b = results['c'], results['b']
    print(f"Q{int(q * 100)}: tan(beta) = {c:.5f} * A^{b:.5f}")

    ###DEBUGGING###
#print("Flow Accumulation in km²:", flow_accum_valid_km2)
#print("Log(Flow Accumulation):", log_flow_accum)

# Initialize classified raster
classified_cells = np.full(flow_accum_raster.shape, np.nan)

# Apply classification logic
for row in range(flow_accum_raster.shape[0]):
    for col in range(flow_accum_raster.shape[1]):
        if not valid_mask[row, col]:
            continue  # Skip invalid cells

        # Get flow accumulation and slope for the current cell
        flow_accum = flow_accum_raster[row, col] * 100 / 1e6  # Convert to km²
        slope = slope_raster[row, col]
        tan_slope = np.tan(np.radians(slope))

        # Calculate thresholds for the current flow_accum
        thresholds = {}
        for q, results in quantile_results.items():
            c, b = results['c'], results['b']
            tan_beta = c * (flow_accum ** b)
            thresholds[q] = np.degrees(np.arctan(tan_beta))

        # Classify the cell based on slope thresholds
        if thresholds[0.05] <= slope < thresholds[0.1]:
            classified_cells[row, col] = 5
        elif thresholds[0.1] <= slope < thresholds[0.25]:
            classified_cells[row, col] = 15
        elif thresholds[0.25] <= slope < thresholds[0.75]:
            classified_cells[row, col] = 50
        elif thresholds[0.75] <= slope < thresholds[0.9]:
            classified_cells[row, col] = 15
        elif thresholds[0.9] <= slope <= thresholds[0.95]:
            classified_cells[row, col] = 5

# Save the classified raster
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=classified_cells.shape[0],
    width=classified_cells.shape[1],
    count=1,
    dtype=rasterio.float32,
    crs=flow_crs,
    transform=flow_affine,
    nodata=-9999
) as dst:
    classified_cells[np.isnan(classified_cells)] = -9999  # Replace NaN with nodata value
    dst.write(classified_cells, 1)

print(f"Classified raster saved to {output_path}.")


import grass.script as gs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import QuantileRegressor
import rasterio

# Set region to match the DEM
gs.run_command('g.region', raster='dtm10m')

# Calculate flow accumulation with the specified parameters
gs.run_command('r.watershed', elevation='dtm10m', accumulation='flow_accum_temp', threshold=10000, flags='ab', overwrite=True)

# Additional setup: calculate slope
gs.run_command('r.slope.aspect', elevation='dtm10m', slope='slope_dtm10m', overwrite=True)

# Get the list of all categories (polygons) from the landslide polygons vector
landslide_polygons = gs.vector_db_select('LandslidesRestricted')['values']

# Initialize lists to store raster names for merging
flow_accum_rasters = []
slope_rasters = []

# Loop through each polygon category
for poly_id, attributes in landslide_polygons.items():
	print(f"Processing polygon ID: {poly_id}, attributes: {attributes}")
    
	# Extract a single polygon from the landslide layer
	gs.run_command('v.extract', input='LandslidesRestricted', where=f"cat = {poly_id}", output=f'single_polygon_{poly_id}', overwrite=True)

	# Mask the DEM by the current single polygon
	gs.run_command('r.mask', vector=f'single_polygon_{poly_id}', overwrite=True)

	# Calculate the 90th percentile using r.quantile
	quantile_result = gs.read_command('r.quantile', input='dtm10m', percentiles='90')
	elev90 = float(quantile_result.split(':')[2])  # Extract the elevation value at the 90th percentile
	print(f"90th percentile elevation for polygon {poly_id}: {elev90}")

	# Select cells where the elevation is above the 90th percentile
	gs.mapcalc(f"elevation_above_90_{poly_id} = if(dtm10m > {elev90}, 1, null())", overwrite=True)

	# Clip flow accumulation to areas where elevation is above the 90th percentile
	gs.mapcalc(f"flow_accum_clipped_{poly_id} = if(elevation_above_90_{poly_id} == 1, flow_accum_temp, null())", overwrite=True)
	flow_accum_rasters.append(f"flow_accum_clipped_{poly_id}")

	# Clip slope to areas where elevation is above the 90th percentile
	gs.mapcalc(f"slope_clipped_{poly_id} = if(elevation_above_90_{poly_id} == 1, slope_dtm10m, null())", overwrite=True)
	slope_rasters.append(f"slope_clipped_{poly_id}")

	# Remove the mask before moving to the next polygon
	gs.run_command('r.mask', flags='r')

# Combine all clipped flow accumulation rasters into one
gs.run_command('r.patch', input=','.join(flow_accum_rasters), output='combined_flow_accum', overwrite=True)

# Combine all clipped slope rasters into one
gs.run_command('r.patch', input=','.join(slope_rasters), output='combined_slope', overwrite=True)

# Export the combined flow accumulation raster
gs.run_command('r.out.gdal', input='combined_flow_accum', output='/home/igv/Documents/Correct/PERMANENT/combined_flow_accum.tif', format='GTiff', type='Float64', overwrite=True)

# Export the combined slope raster
gs.run_command('r.out.gdal', input='combined_slope', output='/home/igv/Documents/Correct/PERMANENT/combined_slope.tif', format='GTiff', type='Float64', overwrite=True)


print("Combined flow accumulation and slope rasters have been created.")


# Export combined rasters to ASCII
gs.run_command('r.out.ascii', input='combined_slope', output='/tmp/combined_slope.txt', overwrite=True)

gs.run_command('r.out.ascii', input='combined_slope', output='/home/igv/Documents/Python/combined_slope.txt', overwrite=True)
gs.run_command('r.out.ascii', input='combined_flow_accum', output='/home/igv/Documents/Python/combined_flow_accum.txt', overwrite=True)

gs.run_command('r.out.ascii', input='combined_flow_accum', output='/tmp/combined_flow_accum.txt', overwrite=True)

# Load raster data into numpy arrays
def load_ascii(filepath):
    with open(filepath) as f:
        return np.array([float(value) for line in f for value in line.split() if value.replace('.', '', 1).isdigit()])

slope_values = load_ascii('/tmp/combined_slope.txt')
flow_accum_values = load_ascii('/tmp/combined_flow_accum.txt')

# Filter slope and flow accumulation for valid values
valid_indices = (slope_values > 0) & (flow_accum_values > 0) & (slope_values < 90)
slope_values = slope_values[valid_indices]
flow_accum_values = flow_accum_values[valid_indices]

# Ensure valid data exists after filtering
if slope_values.size == 0 or flow_accum_values.size == 0:
	raise ValueError("Filtered arrays are empty. Check input data.")

# Debug: Print statistics
print("Filtered Slope Values:", np.min(slope_values), np.max(slope_values), np.mean(slope_values))
print("Filtered Flow Accumulation Values:", np.min(flow_accum_values), np.max(flow_accum_values), np.mean(flow_accum_values))

# Convert slope to tan(beta)
tan_slope_values = np.tan(np.radians(slope_values))

# Debug: Check for invalid tan(beta)
if np.isnan(tan_slope_values).any() or np.isinf(tan_slope_values).any():
	raise ValueError("Invalid values in tan_slope_values after conversion.")

# Log-transform the filtered arrays
log_tan_slope = np.log(tan_slope_values)
log_flow_accum = np.log(flow_accum_values)

# Ensure no NaN or Inf values exist after log-transform
if np.isnan(log_tan_slope).any() or np.isinf(log_tan_slope).any():
	raise ValueError("NaN or Inf values encountered in log_tan_slope.")
if np.isnan(log_flow_accum).any() or np.isinf(log_flow_accum).any():
	raise ValueError("NaN or Inf values encountered in log_flow_accum.")

# Step 3: Perform Linear Quantile Regression in Log-Log Space

# Perform Quantile Regression in Log-Log Space
quantiles = [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]
quantile_results = {}

plt.figure(figsize=(10, 6))

# Define range of flow accumulation (log scale for smoother curve)
x_vals = np.linspace(min(log_flow_accum), max(log_flow_accum), 100)
flow_accum_vals = np.exp(x_vals)  # Convert back to original flow accumulation

for q in quantiles:
	# Perform quantile regression
	model = QuantileRegressor(quantile=q, alpha=0.01, solver='highs').fit(log_flow_accum.reshape(-1, 1), log_tan_slope)
	b = model.coef_[0]  # Slope of the linear equation
	log_c = model.intercept_  # Intercept of the linear equation
	c = np.exp(log_c)
	quantile_results[q] = {'c': c, 'b': b}
    
	# Convert regression results back to slope degrees
	tan_slope_pred = c * (flow_accum_vals ** b)  # From log-log space: tan(beta) = c * A^b
	slope_deg = np.degrees(np.arctan(tan_slope_pred))  # Convert tan(beta) to slope in degrees
    
	# Plot regression line
	plt.plot(flow_accum_vals / 1e6, slope_deg, label=f"Q{int(q*100)}")

# Scatter plot of actual data points
plt.scatter(flow_accum_values / 1e6, slope_values, color='gray', alpha=0.5, s=10, label="Data points")

# Configure the axes
plt.xscale("log")  # Logarithmic scale for flow accumulation
plt.xlabel("Flow Accumulation (km²)")
plt.ylabel("Slope (degrees)")
plt.title("Slope as a Function of Flow Accumulation (Quantile Regression)")
plt.legend(title="Quantiles")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

import grass.script as gs
import numpy as np
from sklearn.linear_model import QuantileRegressor

# Step 1: Set region and input rasters
gs.run_command('g.region', raster='combined_flow_accum')
flow_accum_raster = 'combined_flow_accum'
slope_raster = 'combined_slope'

# Step 2: Export the rasters to ASCII for processing in Python
flow_accum_file = "/tmp/combined_flow_accum.txt"
slope_file = "/tmp/combined_slope.txt"

gs.run_command('r.out.ascii', input=flow_accum_raster, output=flow_accum_file, overwrite=True)
gs.run_command('r.out.ascii', input=slope_raster, output=slope_file, overwrite=True)

# Load ASCII files into numpy arrays
def load_ascii(filepath):
    with open(filepath) as f:
        return np.array([
            float(value) for line in f for value in line.split() if value.replace('.', '', 1).isdigit()
        ])

flow_accum_values = load_ascii(flow_accum_file)
slope_values = load_ascii(slope_file)

# Filter valid data (remove missing or invalid values)
valid_indices = (flow_accum_values > 0) & (slope_values > 0) & (slope_values < 90)
flow_accum_values = flow_accum_values[valid_indices]
slope_values = slope_values[valid_indices]

# Ensure valid data exists
if flow_accum_values.size == 0 or slope_values.size == 0:
    raise ValueError("No valid data found in rasters. Check the input rasters.")

# Log-transform the data
log_flow_accum = np.log(flow_accum_values)
log_tan_slope = np.log(np.tan(np.radians(slope_values)))

# Step 3: Perform Quantile Regression
quantiles = [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]
quantile_results = {}

for q in quantiles:
    model = QuantileRegressor(quantile=q, alpha=0.01, solver='highs').fit(log_flow_accum.reshape(-1, 1), log_tan_slope)
    b = model.coef_[0]
    log_c = model.intercept_
    c = np.exp(log_c)
    quantile_results[q] = {'c': c, 'b': b}

# Step 4: Create classification thresholds
classification_thresholds = {}
flow_accum_vals = np.exp(log_flow_accum)  # Back to original flow accumulation
for q, results in quantile_results.items():
    c, b = results['c'], results['b']
    tan_slope_thresholds = c * (flow_accum_vals ** b)  # tan(beta) = c * A^b
    classification_thresholds[q] = np.degrees(np.arctan(tan_slope_thresholds))  # Convert to degrees

# Extract thresholds
q05 = np.min(classification_thresholds[0.05])
q10 = np.max(classification_thresholds[0.1])
q25 = np.min(classification_thresholds[0.25])
q75 = np.max(classification_thresholds[0.75])
q90 = np.min(classification_thresholds[0.9])
q95 = np.max(classification_thresholds[0.95])

# Step 5: Create classification expression in GRASS GIS
classification_expr = (
    f"SA_final = if("
    f"(tan(slope({slope_raster} * 3.14159 / 180)) >= {np.tan(np.radians(q05))} && "
    f"tan(slope({slope_raster} * 3.14159 / 180)) < {np.tan(np.radians(q10))}), 5, "
    f"if(tan(slope({slope_raster} * 3.14159 / 180)) >= {np.tan(np.radians(q10))} && "
    f"tan(slope({slope_raster} * 3.14159 / 180)) < {np.tan(np.radians(q25))}), 15, "
    f"if(tan(slope({slope_raster} * 3.14159 / 180)) >= {np.tan(np.radians(q25))} && "
    f"tan(slope({slope_raster} * 3.14159 / 180)) <= {np.tan(np.radians(q75))}), 50, null())))"
)


print("Applying classification...")
gs.mapcalc(expression=classification_expr, overwrite=True)

# Step 6: Export the classified raster
gs.run_command(
    'r.out.gdal',
    input='SA_final',
    output='/home/igv/Documents/Correct/PERMANENT/SA_final.tif',
    format='GTiff',
    type='Int32',
    overwrite=True
)

print("Final classified raster (SA_final) created and exported.")


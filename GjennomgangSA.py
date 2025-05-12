import grass.script as gs
import numpy as np
import rasterio

# Set region to match the DEM
gs.run_command('g.region', raster='dtm10m')

# Calculate flow accumulation with the specified parameters
gs.run_command('r.watershed', elevation='dtm10m', accumulation='flow_accum_temp', threshold=10000, flags='ab', overwrite=True)

# Additional setup: calculate slope
gs.run_command('r.slope.aspect', elevation='dtm10m', slope='slope_dtm10m', overwrite=True)

# Get the list of all categories (polygons) from the landslide polygons vector
landslide_polygons = gs.vector_db_select('LandslidesInArea')['values']

# Initialize lists to store raster names for merging
flow_accum_rasters = []
slope_rasters = []

# Loop through each polygon category
for poly_id, attributes in landslide_polygons.items():
    print(f"Processing polygon ID: {poly_id}, attributes: {attributes}")
    
    # Extract a single polygon from the landslide layer
    gs.run_command('v.extract', input='LandslidesInArea', where=f"cat = {poly_id}", output=f'single_polygon_{poly_id}', overwrite=True)

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
#tan_slope_values = np.tan(np.radians(slope_values))

# Debug: Check for invalid tan(beta)
#if np.isnan(tan_slope_values).any() or np.isinf(tan_slope_values).any():
    #raise ValueError("Invalid values in tan_slope_values after conversion.")

# Log-transform the filtered arrays
#log_tan_slope = np.log(tan_slope_values)
#log_flow_accum = np.log(flow_accum_values)

# Ensure no NaN or Inf values exist after log-transform
#if np.isnan(log_tan_slope).any() or np.isinf(log_tan_slope).any():
    #raise ValueError("NaN or Inf values encountered in log_tan_slope.")
#if np.isnan(log_flow_accum).any() or np.isinf(log_flow_accum).any():
    #raise ValueError("NaN or Inf values encountered in log_flow_accum.")

print("Script execution completed.")


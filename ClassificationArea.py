import grass.script as gs
import numpy as np
import rasterio
from collections import Counter
import matplotlib.pyplot as plt

# Paths
grass_db_path = '/home/igv/Documents/Correct/PERMANENT'  # GRASS GIS database path
input_raster = 'dtm10m'  # Input DEM raster name in GRASS GIS
slope_output = 'slope_clipped'  # Slope output raster name
flow_accum_output = 'flow_accum_clipped'  # Flow accumulation output raster name
slope_tif = '/home/igv/Documents/Correct/slope_clipped.tif'  # Exported slope file
flow_accum_tif = '/home/igv/Documents/Correct/flow_accum_clipped.tif'  # Exported flow accumulation file
quantile_results_path = '/home/igv/Documents/QuantileLines/quantile_results.npy'  # Path to saved quantile regression results

# Step 1: Calculate slope using GRASS GIS
print("Calculating slope using GRASS GIS...")
gs.run_command(
    'r.slope.aspect',
    elevation=input_raster,
    slope=slope_output,
    overwrite=True
)
print(f"Slope raster '{slope_output}' created.")

    ###DEBUGGING###
# Step 2: Export the slope raster to GeoTIFF
print("Exporting slope raster to GeoTIFF...")
gs.run_command(
    'r.out.gdal',
    input=slope_output,
    output=slope_tif,
    format='GTiff',
    createopt="COMPRESS=DEFLATE",
    overwrite=True
)
print(f"Slope raster exported to '{slope_tif}'.")

# Step 3: Load the slope raster
with rasterio.open(slope_tif) as slope_src:
    slope_raster = slope_src.read(1)
    slope_nodata = slope_src.nodata

# Replace nodata with NaN
slope_raster = np.where(slope_raster == slope_nodata, np.nan, slope_raster)

# Step 4: Round slope values to integers for grouping
slope_rounded = np.round(slope_raster, decimals=0)

# Step 5: Count the number of cells for each slope value
# Mask out NaN values and flatten the array for counting
valid_slope_values = slope_rounded[~np.isnan(slope_rounded)].astype(int)
slope_counts = Counter(valid_slope_values)

# Step 6: Print statistics
print("\nSlope Value Statistics:")
print(f"{'Slope (degrees)':>15} | {'Number of Cells':>15}")
print("-" * 35)
for slope, count in sorted(slope_counts.items()):
    print(f"{slope:>15} | {count:>15}")

# Step 7: Plot a histogram of slope values
plt.figure(figsize=(10, 6))
plt.bar(slope_counts.keys(), slope_counts.values(), color='blue', alpha=0.7)
plt.xlabel("Slope (degrees)")
plt.ylabel("Number of Cells")
plt.title("Distribution of Slope Values")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

    ###DEBUGGING FINISHED###
    
# Step 2: Calculate flow accumulation using GRASS GIS
print("Calculating flow accumulation using GRASS GIS...")
gs.run_command(
    'r.watershed',
    elevation=input_raster,
    accumulation=flow_accum_output,
    threshold=10000,  # Adjust the threshold as needed
    flags='ab',  # a: calculate accumulation, b: skip basin delineation
    overwrite=True
)
print(f"Flow accumulation raster '{flow_accum_output}' created.")

# Conversion factor for cell area to km²
cell_area_km2 = 0.0001  # Assuming 10m × 10m cells
    
    ###DEBUGGING FLOW ACCUM###

# Step 1: Calculate flow accumulation using GRASS GIS
print("Calculating flow accumulation using GRASS GIS...")
gs.run_command(
    'r.watershed',
    elevation=input_raster,
    accumulation=flow_accum_output,
    threshold=10000,  # Adjust the threshold if needed
    flags='ab',  # 'a' for accumulation map, 'b' to skip basin delineation
    overwrite=True
)
print(f"Flow accumulation raster '{flow_accum_output}' created.")

# Step 2: Export the flow accumulation raster to GeoTIFF
print("Exporting flow accumulation raster to GeoTIFF...")
gs.run_command(
    'r.out.gdal',
    input=flow_accum_output,
    output=flow_accum_tif,
    format='GTiff',
    createopt="COMPRESS=DEFLATE",
    overwrite=True
)
print(f"Flow accumulation raster exported to '{flow_accum_tif}'.")

# Step 3: Load the flow accumulation raster
with rasterio.open(flow_accum_tif) as flow_accum_src:
    flow_accum_raster = flow_accum_src.read(1)
    flow_accum_nodata = flow_accum_src.nodata

# Replace nodata with NaN
flow_accum_raster = np.where(flow_accum_raster == flow_accum_nodata, np.nan, flow_accum_raster)

# Step 4: Convert flow accumulation values to km²
flow_accum_km2 = flow_accum_raster * cell_area_km2

# Step 5: Round flow accumulation values to 2 decimal places for grouping
flow_accum_rounded = np.round(flow_accum_km2, decimals=2)

# Step 6: Count the number of cells for each flow accumulation value
# Mask out NaN values and flatten the array for counting
valid_flow_accum_values = flow_accum_rounded[~np.isnan(flow_accum_rounded)]
flow_accum_counts = Counter(valid_flow_accum_values)

# Step 7: Print statistics
print("\nFlow Accumulation Value Statistics (in km²):")
print(f"{'Flow Accum (km²)':>20} | {'Number of Cells':>15}")
print("-" * 40)
for value, count in sorted(flow_accum_counts.items()):
    print(f"{value:>20.2f} | {count:>15}")

# Step 8: Plot a histogram of flow accumulation values
plt.figure(figsize=(10, 6))
plt.bar(flow_accum_counts.keys(), flow_accum_counts.values(), color='green', alpha=0.7)
plt.xscale("log")  # Use a log scale for better visualization if values span many orders of magnitude
plt.xlabel("Flow Accumulation (km², log scale)")
plt.ylabel("Number of Cells")
plt.title("Distribution of Flow Accumulation Values (in km²)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



    ###DEBUGGING FINISHED###
    
# Step 3: Export GRASS GIS rasters to GeoTIFF
print("Exporting GRASS GIS rasters to GeoTIFF...")
gs.run_command(
    'r.out.gdal',
    input=slope_output,
    output=slope_tif,
    format='GTiff',
    createopt="COMPRESS=DEFLATE",
    overwrite=True
)
print(f"Slope raster exported to '{slope_tif}'.")

gs.run_command(
    'r.out.gdal',
    input=flow_accum_output,
    output=flow_accum_tif,
    format='GTiff',
    createopt="COMPRESS=DEFLATE",
    overwrite=True
)
print(f"Flow accumulation raster exported to '{flow_accum_tif}'.")

# Step 4: Load quantile regression results
def load_quantile_results(filepath):
    return np.load(filepath, allow_pickle=True).item()

quantile_results = load_quantile_results(quantile_results_path)
print(f"Quantile regression results loaded: {quantile_results}")

# Step 5: Load the slope and flow accumulation GeoTIFFs
with rasterio.open(slope_tif) as slope_src:
    slope_raster = slope_src.read(1)
    slope_nodata = slope_src.nodata

with rasterio.open(flow_accum_tif) as flow_accum_src:
    flow_accum_raster = flow_accum_src.read(1)
    flow_accum_nodata = flow_accum_src.nodata

# Replace nodata with NaN
slope_raster = np.where(slope_raster == slope_nodata, np.nan, slope_raster)
flow_accum_raster = np.where(flow_accum_raster == flow_accum_nodata, np.nan, flow_accum_raster)

# Step 6: Prepare data for plotting
valid_mask = ~np.isnan(slope_raster) & ~np.isnan(flow_accum_raster)
slope_valid = slope_raster[valid_mask]
flow_accum_valid = flow_accum_raster[valid_mask]

# Convert flow accumulation to km²
flow_accum_valid_km2 = flow_accum_valid * 100 / 1e6

# Step 7: Plot slope vs flow accumulation
plt.figure(figsize=(10, 6))
plt.scatter(flow_accum_valid_km2, slope_valid, color='gray', alpha=0.5, s=10, label="Clipped Raster Cells")
plt.xscale("log")
plt.xlabel("Flow Accumulation (km²)")
plt.ylabel("Slope (degrees)")
plt.title("Slope vs Flow Accumulation")

# Step 8: Overlay quantile regression lines
x_vals = np.linspace(np.log(min(flow_accum_valid_km2)), np.log(max(flow_accum_valid_km2)), 100)
flow_accum_vals = np.exp(x_vals)

for q, results in quantile_results.items():
    c = results['c']
    b = results['b']
    tan_slope_pred = c * (flow_accum_vals ** b)
    slope_deg = np.degrees(np.arctan(tan_slope_pred))
    plt.plot(flow_accum_vals, slope_deg, label=f"Q{int(q*100)}")

plt.legend(title="Quantiles")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# File paths
classified_output_path = '/home/igv/Documents/Correct//PERMANENT/classified_area.tif'  # Output classified raster path

# Step 1: Prepare an empty array for classification
classified_area = np.full(flow_accum_raster.shape, np.nan)

    ###DEBUGGING###
# Print the quantile line equations
print("Quantile Line Functions:")
for q, results in quantile_results.items():
    c, b = results['c'], results['b']
    print(f"Q{int(q * 100)}: tan(beta) = {c:.5f} * A^{b:.5f}")
    
# Step 2: Loop through each cell and classify based on quantile regression lines
for row in range(flow_accum_raster.shape[0]):
    for col in range(flow_accum_raster.shape[1]):
        if np.isnan(flow_accum_raster[row, col]) or np.isnan(slope_raster[row, col]):
            continue  # Skip invalid cells

        # Get flow accumulation and slope for the current cell
        flow_accum = flow_accum_raster[row, col] * 100 / 1e6  # Convert to km²
        slope = slope_raster[row, col]
        tan_slope = np.tan(np.radians(slope))

        # Calculate thresholds for the current flow_accum based on quantile regression
        thresholds = {}
        for q, results in quantile_results.items():
            c, b = results['c'], results['b']
            tan_beta = c * (flow_accum ** b)
            thresholds[q] = np.degrees(np.arctan(tan_beta))  # Convert back to slope degrees

        # Classify the cell based on slope thresholds
        if thresholds[0.05] <= slope < thresholds[0.1]:
            classified_area[row, col] = 5
        elif thresholds[0.1] <= slope < thresholds[0.25]:
            classified_area[row, col] = 15
        elif thresholds[0.25] <= slope < thresholds[0.75]:
            classified_area[row, col] = 50
        elif thresholds[0.75] <= slope < thresholds[0.9]:
            classified_area[row, col] = 15
        elif thresholds[0.9] <= slope <= thresholds[0.95]:
            classified_area[row, col] = 5

# Step 3: Save the classified raster
with rasterio.open(
    classified_output_path,
    'w',
    driver='GTiff',
    height=classified_area.shape[0],
    width=classified_area.shape[1],
    count=1,
    dtype=rasterio.float32,
    crs=slope_src.crs,  # Assuming slope_src is already loaded with rasterio
    transform=slope_src.transform,  # Assuming slope_src is already loaded with rasterio
    nodata=-9999
) as dst:
    classified_area[np.isnan(classified_area)] = -9999  # Replace NaN with nodata value
    dst.write(classified_area, 1)

print(f"Classified raster saved to {classified_output_path}.")


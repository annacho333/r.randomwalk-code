import grass.script as gs
import csv

# Input Parameters
input_raster = 'dtm10m'  # DTM raster layer
landslide_vector = 'Landslides200'  # Landslide polygons vector
output_csv = '/home/igv/Documents/landslide_analysis.csv'

# Ensure the input data exists
if input_raster not in gs.list_grouped('raster')[gs.gisenv()['MAPSET']]:
    raise FileNotFoundError(f"Raster map '{input_raster}' not found!")
if landslide_vector not in gs.list_grouped('vector')[gs.gisenv()['MAPSET']]:
    raise FileNotFoundError(f"Vector map '{landslide_vector}' not found!")

# Retrieve category IDs (cat) of the polygons
categories = gs.read_command('v.category', input=landslide_vector, option='print').splitlines()

results = []

###### DEBUGGING STARTING ######

#for cat in categories:
    #print(f"Processing polygon with category: {repr(cat)}")
    #safe_cat = str(cat).strip().replace("/", "_").replace("\\", "_")

    #if not safe_cat.isdigit():
        #print(f"Skipping invalid category: {repr(cat)}")
        #continue

    #try:
        #mask_raster = f"temp_mask_{safe_cat}"
        #print(f"Processing with mask: {mask_raster}")  # Debugging step
        #gs.run_command('v.to.rast', input=landslide_vector, output=mask_raster, 
                       #use='attr', attribute_column='cat', where=f"cat = '{cat}'", overwrite=True)
    #except Exception as e:
       # print(f"Error processing category {cat}: {e}")

#for cat in categories:
    #safe_cat = str(cat).strip().replace("/", "_").replace("\\", "_")

    #if not safe_cat.isdigit():
        #print(f"Skipping invalid category: {repr(cat)}")
        #continue  # Ensures the script skips invalid categories and moves to the next iteration

    #try:
        #mask_raster = f"temp_mask_{safe_cat}"
        #gs.run_command('v.to.rast', input=landslide_vector, output=mask_raster, use='attr', attribute_column='cat', where=f"cat = '{cat}'", overwrite=True)
        #print(f"Successfully processed polygon with category {cat}")
   # except Exception as e:
       # print(f"Error processing category {cat}: {e}")
###### DEBUGGING ENDING ######

for cat in categories:
    print(f"Processing polygon with category {cat}...")

    # Create a temporary mask for the current polygon
    mask_raster = f"temp_mask_{cat}"
    gs.run_command('v.to.rast', input=landslide_vector, output=mask_raster, use='attr', attribute_column='cat', where=f"cat = {cat}", overwrite=True)

    # Mask the DTM with the polygon
    masked_dtm = f"masked_dtm_{cat}"
    gs.mapcalc(f"{masked_dtm} = if({mask_raster}, {input_raster}, null())", overwrite=True)

    # Get min and max values
    stats = gs.parse_command('r.univar', map=masked_dtm, flags='g')
    min_val = float(stats['min'])
    max_val = float(stats['max'])
    height = max_val - min_val

    # List all cells and filter for min and max values with a small tolerance
    cell_data = gs.read_command('r.stats', input=masked_dtm, flags='1gn').strip().splitlines()
    min_coords, max_coords = None, None
    tolerance = 0.001  # Adjust tolerance if needed

    for line in cell_data:
        x, y, value = line.split()
        value = float(value)
        if abs(value - min_val) <= tolerance and not min_coords:
            min_coords = (x, y)
        if abs(value - max_val) <= tolerance and not max_coords:
            max_coords = (x, y)
        if min_coords and max_coords:
            break

    # Default to None if coordinates not found
    min_x, min_y = min_coords if min_coords else (None, None)
    max_x, max_y = max_coords if max_coords else (None, None)

    # Calculate the length (distance) between min and max coordinates using m.measure
    length = None
    if min_coords and max_coords:
        coords = f"{min_x},{min_y},{max_x},{max_y}"
        length_output = gs.read_command('m.measure', flags='g', coordinates=coords).strip()
        length_line = [line for line in length_output.splitlines() if "length" in line.lower()]
        if length_line:
            length = float(length_line[0].split('=')[-1].strip())

    # Calculate ratio
    ratio = height / length if length and length > 0 else None

    # Append results
    results.append({
        'cat': cat,
        'min_value': min_val,
        'min_coords': f"({min_x}, {min_y})" if min_x and min_y else "None",
        'max_value': max_val,
        'max_coords': f"({max_x}, {max_y})" if max_x and max_y else "None",
        'height': height,
        'length': length,
        'ratio': ratio
    })

    # Clean up temporary layers
    gs.run_command('g.remove', type='raster', name=[mask_raster, masked_dtm], flags='f')

# Save results to CSV
with open(output_csv, mode='w', newline='') as csvfile:
    fieldnames = ['cat', 'min_value', 'min_coords', 'max_value', 'max_coords', 'height', 'length', 'ratio']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {output_csv}.")


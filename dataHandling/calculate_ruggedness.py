import numpy as np

# Specify the file name (update the path as needed)
#filename = "data/Ruggedness/15N015E_LAND_30S.ACE2"  # For the height dataset, usually "BOTH" is used
#filename = "data/Ruggedness/30N060E_LAND_30S.ACE2"  # For the height dataset, usually "BOTH" is used
#filename = "data/Ruggedness/30S075W_LAND_30S.ACE2"  # For the height dataset, usually "BOTH" is used
filename = "data/Ruggedness/15S060W_LAND_30S.ACE2"  # For the height dataset, usually "BOTH" is used




# Define the number of rows and columns
nrows, ncols = 1800, 1800

# Read the binary file as little-endian float32
# '<f4' means little-endian ('<') 4-byte float ('f4')
data = np.fromfile(filename, dtype='<f4')

# Check that the size matches what you expect
if data.size != nrows * ncols:
    raise ValueError("Unexpected file size. Check the file and dimensions.")

# Reshape the flat array into a 2D array (row-major order)
# Note: The first row in the array corresponds to the northernmost row.
dem = data.reshape((nrows, ncols))

# Optionally, mask out the nodata values
# Here we mask both -500 (land/sea mask) and -32768 (voids)
dem_masked = np.ma.masked_where((dem == -500) | (dem == -32768), dem)

# Now 'dem_masked' holds the elevation data ready for analysis
print("DEM shape:", dem_masked.shape)

def compute_TRI(dem):
    """
    Compute the Terrain Ruggedness Index (TRI) for a DEM.
    The TRI is calculated as the sum of the absolute differences between
    each cell and its eight neighboring cells.
    
    Parameters:
        dem (2D numpy array): DEM data with np.nan for missing values.
        
    Returns:
        tri (2D numpy array): TRI value for each cell.
    """
    # Define shifts for the 8 neighbors: N, NE, E, SE, S, SW, W, NW
    shifts = [(-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    # List to store the difference arrays
    diff_list = []
    
    for dy, dx in shifts:
        # Roll the array to shift by (dy, dx)
        shifted = np.roll(dem, shift=(dy, dx), axis=(0, 1))
        
        # To avoid wrap-around effects, set the wrapped edges to np.nan.
        if dy == -1:
            shifted[-1, :] = np.nan
        elif dy == 1:
            shifted[0, :] = np.nan
        if dx == -1:
            shifted[:, -1] = np.nan
        elif dx == 1:
            shifted[:, 0] = np.nan
        
        # Compute the absolute difference between the original and shifted array
        diff = np.abs(dem - shifted)
        diff_list.append(diff)
    
    # Sum the differences across all 8 neighbors, ignoring np.nan values.
    tri = np.nansum(np.array(diff_list), axis=0)
    return tri

# Convert the masked array to a regular array with np.nan for missing values.
dem_filled = dem_masked.filled(np.nan)

# Compute the TRI for each cell.
tri_array = compute_TRI(dem_filled)

# Now, calculate an average ruggedness for the entire area.
# This gives you a single "ruggedness number" (TRI) for the tile.
avg_ruggedness = np.nanmean(tri_array)
print("Average Terrain Ruggedness Index (TRI):", avg_ruggedness)
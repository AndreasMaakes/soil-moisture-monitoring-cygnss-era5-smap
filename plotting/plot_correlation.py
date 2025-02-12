import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Import your SMAP and CYGNSS routines.
from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from CYGNSS.import_data import importData

def SMAP_CYGNSS_correlation_plot(smap_folder, cygnss_folder, sigma, grid_size, window_size):
    """
    Imports SMAP and CYGNSS data, regrids them to a common grid, applies Gaussian blur,
    and computes a local Pearson correlation coefficient (using a moving window) between
    the two datasets. The resulting correlation map is then plotted.

    Parameters:
      smap_folder : str
          Folder path for the SMAP data.
      cygnss_folder : str
          Folder path for the CYGNSS data.
      sigma : float
          Sigma value for the Gaussian blur (applied to both regridded fields).
      grid_size : int
          Number of grid points along each dimension of the common grid.
      window_size : int
          Size (in grid cells) of the moving window used to compute the local correlation.
          (A typical choice is an odd number such as 5 so that there is a central cell.)
    """
    # -------------------------------
    # 1. Import and prepare the data
    # -------------------------------
    # SMAP data: Import, concatenate, and average
    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap = pd.concat(dfs_smap)
    df_smap_avg = SMAP_averaging_soil_moisture(df_smap)
    
    # Extract SMAP values
    smap_lat = df_smap_avg["latitude"].values
    smap_lon = df_smap_avg["longitude"].values
    smap_val = df_smap_avg["soil_moisture_avg"].values

    # CYGNSS data: Import and concatenate
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    
    # Extract CYGNSS values (note the column names for CYGNSS)
    cygnss_lat = df_cygnss["sp_lat"].values
    cygnss_lon = df_cygnss["sp_lon"].values
    cygnss_val = df_cygnss["sr"].values

    # ----------------------------------------
    # 2. Create a common grid for both datasets
    # ----------------------------------------
    lat_min = min(smap_lat.min(), cygnss_lat.min())
    lat_max = max(smap_lat.max(), cygnss_lat.max())
    lon_min = min(smap_lon.min(), cygnss_lon.min())
    lon_max = max(smap_lon.max(), cygnss_lon.max())
    
    # Create grid vectors and a meshgrid for plotting
    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # --------------------------------------
    # 3. Regrid each dataset and apply Gaussian blur
    # --------------------------------------
    # SMAP regridding
    smap_grid = griddata(
        (smap_lon, smap_lat),
        smap_val,
        (lon_mesh, lat_mesh),
        method='linear'
    )
    smap_grid_blur = gaussian_filter(smap_grid, sigma=sigma)
    
    # CYGNSS regridding
    cygnss_grid = griddata(
        (cygnss_lon, cygnss_lat),
        cygnss_val,
        (lon_mesh, lat_mesh),
        method='linear'
    )
    cygnss_grid_blur = gaussian_filter(cygnss_grid, sigma=sigma)

    # -------------------------------------------------
    # 4. Compute a local Pearson correlation in a moving window
    # -------------------------------------------------
    nrows, ncols = smap_grid_blur.shape
    half_win = window_size // 2  # for a symmetric window
    # Prepare an array to store the correlation coefficient for each grid cell
    corr_map = np.full((nrows, ncols), np.nan)
    
    def local_pearson(x, y):
        """Compute the Pearson correlation coefficient between x and y,
        ignoring any NaN values."""
        valid = (~np.isnan(x)) & (~np.isnan(y))
        if np.sum(valid) < 2:
            return np.nan
        x_valid = x[valid]
        y_valid = y[valid]
        return np.corrcoef(x_valid, y_valid)[0, 1]
    
    # Loop over each grid cell and compute the local correlation
    for i in range(nrows):
        for j in range(ncols):
            # Define window boundaries (handle edge cases)
            i_min = max(i - half_win, 0)
            i_max = min(i + half_win + 1, nrows)
            j_min = max(j - half_win, 0)
            j_max = min(j + half_win + 1, ncols)
            
            # Extract local window data from both fields (flattened into 1D arrays)
            smap_window = smap_grid_blur[i_min:i_max, j_min:j_max].flatten()
            cygnss_window = cygnss_grid_blur[i_min:i_max, j_min:j_max].flatten()
            
            # Compute the correlation for this window
            corr_map[i, j] = local_pearson(smap_window, cygnss_window)
    
    # ---------------------------
    # 5. Plot the correlation map
    # ---------------------------
    plt.figure(figsize=(10, 8))
    # pcolormesh will plot the grid; using a diverging colormap from -1 to 1.
    mesh = plt.pcolormesh(lon_mesh, lat_mesh, corr_map, shading='auto',
                          cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(mesh, label='Local Pearson Correlation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Local Correlation between SMAP Soil Moisture and CYGNSS Surface Reflectivity')
    plt.axis('equal')
    plt.show()

# =============================================================================
# Example usage:
#
# smap_folder = "Your/SMAP/folder/path"
# cygnss_folder = "Your/CYGNSS/folder/path"
# sigma = 1            # for Gaussian blur; set to 0 for no smoothing
# grid_size = 200      # number of grid points in each dimension
# window_size = 5      # moving window size (in grid cells) for computing correlation
#
# SMAP_CYGNSS_correlation_plot(smap_folder, cygnss_folder, sigma, grid_size, window_size)
# =============================================================================

SMAP_CYGNSS_correlation_plot("Western-Australia", "Western-Australia/Western-Australia-20201001-20201007", 0, 50, 5)
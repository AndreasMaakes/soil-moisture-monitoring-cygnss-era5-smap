import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# Import functions for SMAP and CYGNSS data (adjust the import paths as needed)
from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from CYGNSS.import_data import importData as importDataCYGNSS

def SMAP_CYGNSS_correlation_matrix(smap_folder, cygnss_folder, 
                                  fine_grid_size=200, coarse_block_size=10, 
                                  smap_sigma=0, cygnss_sigma=0):
    """
    Create a 2D correlation matrix between SMAP and CYGNSS data.
    
    This function:
      1. Imports SMAP and CYGNSS data and regrids each onto a common fine grid.
      2. Optionally applies a Gaussian blur to each fine-grid field.
      3. Aggregates the fine grid into coarse cells (whose size is adjustable).
      4. For each coarse cell, computes the Pearson correlation coefficient between 
         all fine-grid SMAP and CYGNSS values inside that cell.
      5. Computes an overall correlation value using all valid fine-grid points.
      6. Plots the coarse correlation matrix with longitude on the x-axis, latitude on 
         the y-axis, writes the correlation value inside each cell, and displays the
         overall correlation value in the title.
    
    Parameters:
      smap_folder       : str
          Folder (or identifier) for SMAP data.
      cygnss_folder     : str
          Folder (or identifier) for CYGNSS data.
      fine_grid_size    : int, optional
          Resolution (number of points along each axis) of the fine common grid.
      coarse_block_size : int, optional
          Number of fine-grid cells to aggregate into one coarse cell.
      smap_sigma        : float, optional
          Sigma for Gaussian blur applied to the SMAP fine grid.
      cygnss_sigma      : float, optional
          Sigma for Gaussian blur applied to the CYGNSS fine grid.
    """
    # -----------------------------------------------------------
    # 1. Import and Preprocess the Data
    # -----------------------------------------------------------
    # --- SMAP data ---
    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap = pd.concat(dfs_smap)
    df_smap_avg = SMAP_averaging_soil_moisture(df_smap)
    # Expected columns: 'latitude', 'longitude', 'soil_moisture_avg'
    
    # --- CYGNSS data ---
    dfs_cygnss = importDataCYGNSS(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    # Expected columns: 'sp_lat', 'sp_lon', 'sr'
    
    # -----------------------------------------------------------
    # 2. Define a Common Fine Grid for Interpolation
    # -----------------------------------------------------------
    # Determine the overall spatial extents (union of both datasets)
    lat_min = min(df_smap_avg["latitude"].min(), df_cygnss["sp_lat"].min())
    lat_max = max(df_smap_avg["latitude"].max(), df_cygnss["sp_lat"].max())
    lon_min = min(df_smap_avg["longitude"].min(), df_cygnss["sp_lon"].min())
    lon_max = max(df_smap_avg["longitude"].max(), df_cygnss["sp_lon"].max())
    
    # Create fine grid vectors
    lat_fine = np.linspace(lat_min, lat_max, fine_grid_size)
    lon_fine = np.linspace(lon_min, lon_max, fine_grid_size)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)
    
    # -----------------------------------------------------------
    # 3. Interpolate & Optionally Smooth Each Dataset onto the Fine Grid
    # -----------------------------------------------------------
    # --- SMAP ---
    smap_points = (df_smap_avg["longitude"].values, df_smap_avg["latitude"].values)
    smap_vals = df_smap_avg["soil_moisture_avg"].values
    smap_fine = griddata(smap_points, smap_vals, (lon_mesh, lat_mesh), method='linear')
    if smap_sigma:
        smap_fine = gaussian_filter(smap_fine, sigma=smap_sigma)
    
    # --- CYGNSS ---
    cygnss_points = (df_cygnss["sp_lon"].values, df_cygnss["sp_lat"].values)
    cygnss_vals = df_cygnss["sr"].values
    cygnss_fine = griddata(cygnss_points, cygnss_vals, (lon_mesh, lat_mesh), method='linear')
    if cygnss_sigma:
        cygnss_fine = gaussian_filter(cygnss_fine, sigma=cygnss_sigma)
    
    # -----------------------------------------------------------
    # 4. Aggregate the Fine Grid into Coarse Cells & Compute Correlation
    # -----------------------------------------------------------
    # Determine number of coarse cells along each axis.
    coarse_n = fine_grid_size // coarse_block_size
    corr_matrix = np.full((coarse_n, coarse_n), np.nan)
    
    # The spacing (in degrees) of the fine grid:
    dlon = (lon_max - lon_min) / (fine_grid_size - 1)
    dlat = (lat_max - lat_min) / (fine_grid_size - 1)
    
    # Loop over each coarse cell (non-overlapping blocks)
    for i in range(coarse_n):
        for j in range(coarse_n):
            # Indices for this coarse block
            i_start = i * coarse_block_size
            i_end = (i + 1) * coarse_block_size
            j_start = j * coarse_block_size
            j_end = (j + 1) * coarse_block_size
            
            # Extract the block from each fine grid and flatten to 1D arrays
            block_smap = smap_fine[i_start:i_end, j_start:j_end].flatten()
            block_cygnss = cygnss_fine[i_start:i_end, j_start:j_end].flatten()
            
            # Compute correlation if there are at least two paired non-NaN values.
            valid = ~np.isnan(block_smap) & ~np.isnan(block_cygnss)
            if np.sum(valid) >= 2:
                corr = np.corrcoef(block_smap[valid], block_cygnss[valid])[0, 1]
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = np.nan
    
    # -----------------------------------------------------------
    # 5. Create Coordinates for the Coarse Cells and Plot the Matrix
    # -----------------------------------------------------------
    # Compute coarse cell centers in physical coordinates.
    coarse_lon = np.linspace(lon_min + (coarse_block_size * dlon) / 2,
                             lon_max - (coarse_block_size * dlon) / 2,
                             coarse_n)
    coarse_lat = np.linspace(lat_min + (coarse_block_size * dlat) / 2,
                             lat_max - (coarse_block_size * dlat) / 2,
                             coarse_n)
    coarse_lon_mesh, coarse_lat_mesh = np.meshgrid(coarse_lon, coarse_lat)
    
    plt.figure(figsize=(10, 8))
    # Plot the coarse correlation matrix.
    mesh = plt.pcolormesh(coarse_lon_mesh, coarse_lat_mesh, corr_matrix, 
                          shading='none', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(mesh, label='Pearson Correlation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    
    # -----------------------------------------------------------
    # 6. Annotate Each Cell with the Correlation Value
    # -----------------------------------------------------------
    for i in range(coarse_n):
        for j in range(coarse_n):
            corr_val = corr_matrix[i, j]
            if not np.isnan(corr_val):
                # Use the corresponding coarse cell center coordinates.
                x_coord = coarse_lon[j]
                y_coord = coarse_lat[i]
                plt.text(x_coord, y_coord, f"{corr_val:.2f}", 
                         ha='center', va='center', color='black', fontsize=10)
    
    # -----------------------------------------------------------
    # 7. Compute Overall Correlation using all fine-grid values that are valid for both datasets.
    # -----------------------------------------------------------
    valid = ~np.isnan(smap_fine) & ~np.isnan(cygnss_fine)
    if np.sum(valid) >= 2:
        overall_corr = np.corrcoef(smap_fine[valid].flatten(), 
                                   cygnss_fine[valid].flatten())[0, 1]
    else:
        overall_corr = np.nan
    
    # -----------------------------------------------------------
    # 8. Add the overall correlation to the plot title (e.g., below the main title)
    # -----------------------------------------------------------
    plt.title(f'Correlation Matrix between SMAP and CYGNSS\nOverall Correlation: {overall_corr:.2f}', 
              fontsize=14, pad=20)
    
    plt.show()

# =============================================================================
# Example Usage:
#
# smap_folder = "Your_SMAP_Folder_Name"
# cygnss_folder = "Your/CYGNSS/Folder_Name"
#
# SMAP_CYGNSS_correlation_matrix(smap_folder, cygnss_folder,
#                                fine_grid_size=200, coarse_block_size=10,
#                                smap_sigma=1, cygnss_sigma=1)
# =============================================================================

SMAP_CYGNSS_correlation_matrix("India2", "India2/India2-20200101-20200107", 100, 10, 0, 0)

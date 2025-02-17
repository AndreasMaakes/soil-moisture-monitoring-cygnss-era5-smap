import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from scipy.stats import pearsonr
import xarray as xr
from ERA5.ERA5_utils import apply_land_sea_mask, averaging_soil_moisture

from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from CYGNSS.import_data import importData

# Function to regrid a dataframe by binning lat and lon
def regrid_dataframe(df, lat_bins, lon_bins, data_source):
    if data_source == "CYGNSS":
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['sp_lat'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['sp_lon'], bins=lon_bins, right=False)
        # Aggregate soil moisture within each bin (using the mean here)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['sr'].mean().reset_index()
    elif data_source == "SMAP":
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        # Aggregate soil moisture within each bin (using the mean here)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['soil_moisture_avg'].mean().reset_index()
    elif data_source == "ERA5":
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        # Aggregate soil moisture within each bin (using the mean here)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['average_moisture'].mean().reset_index()
    else:
        print("Invalid data source provided")
    
    # Compute the center of each bin for plotting
    df_grid['lat_center'] = df_grid['lat_bin'].apply(lambda x: x.left + 0.5)
    df_grid['lon_center'] = df_grid['lon_bin'].apply(lambda x: x.left + 0.5)
    
    return df_grid

def merged_dataframe(smap_folder, cygnss_folder, era5_folder, lsm_threshold):

    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap1 = pd.concat(dfs_smap)
    df_smap = SMAP_averaging_soil_moisture(df_smap1)

    # CYGNSS data: Import and concatenate
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    
    #Adjusting ddm_snr and sp_rx_gain max limits to increase correlation with SMAP
    df_cygnss = df_cygnss[df_cygnss['ddm_snr'] >= 2]
    df_cygnss = df_cygnss[df_cygnss['sp_rx_gain'] >= 13]

    df_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}').to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)
    
    # Determine the overall spatial domain
    lat_min = min(df_cygnss['sp_lat'].min(), df_smap['latitude'].min(), df_era5_lsm['latitude'].min())
    lat_max = max(df_cygnss['sp_lat'].max(), df_smap['latitude'].max(), df_era5_lsm['latitude'].max())
    lon_min = min(df_cygnss['sp_lon'].min(), df_smap['longitude'].min(), df_era5_lsm['longitude'].min())
    lon_max = max(df_cygnss['sp_lon'].max(), df_smap['longitude'].max(), df_era5_lsm['longitude'].max())

    # Create bins with a 0.5° resolution
    lat_bins = np.arange(lat_min, lat_max + 0.5, 0.5)
    lon_bins = np.arange(lon_min, lon_max + 0.5, 0.5)

    df_cygnss_grid = regrid_dataframe(df_cygnss, lat_bins, lon_bins, "CYGNSS")    
    df_smap_grid   = regrid_dataframe(df_smap, lat_bins, lon_bins, "SMAP")
    df_era5_grid = regrid_dataframe(df_era5_lsm, lat_bins, lon_bins, "ERA5")

    df_merged_CYGNSS_SMAP = pd.merge(df_cygnss_grid, df_smap_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_cygnss', '_smap'))

    df_merged_CYGNSS_ERA5 = pd.merge(df_cygnss_grid, df_era5_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_cygnss', '_era5'))

    df_merged_SMAP_ERA5 = pd.merge(df_smap_grid, df_era5_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_smap', '_era5'))

    return df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5




def correlation_plot(smap_folder, cygnss_folder, era5_folder, lsm_threshold):
    # Get the merged data
    df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5 = merged_dataframe(smap_folder,cygnss_folder,era5_folder,lsm_threshold)

    # Remove rows with NaN values in the relevant soil moisture columns
    df_valid_CYGNSS_SMAP = df_merged_CYGNSS_SMAP.dropna(subset=['sr', 'soil_moisture_avg'])
    df_valid_CYGNSS_ERA5 = df_merged_CYGNSS_ERA5.dropna(subset=['sr', 'average_moisture'])
    df_valid_SMAP_ERA5   = df_merged_SMAP_ERA5.dropna(subset=['soil_moisture_avg', 'average_moisture'])

    # Calculate Pearson correlations
    r_CYGNSS_SMAP, p_value_CYGNSS_SMAP = pearsonr(df_valid_CYGNSS_SMAP['sr'], df_valid_CYGNSS_SMAP['soil_moisture_avg'])
    print(f'Overall Pearson correlation for CYGNSS/SMAP: r = {r_CYGNSS_SMAP:.2f}, p = {p_value_CYGNSS_SMAP:.3f}')

    r_CYGNSS_ERA5, p_value_CYGNSS_ERA5 = pearsonr(df_valid_CYGNSS_ERA5['sr'], df_valid_CYGNSS_ERA5['average_moisture'])
    print(f'Overall Pearson correlation for CYGNSS/ERA5: r = {r_CYGNSS_ERA5:.2f}, p = {p_value_CYGNSS_ERA5:.3f}')

    r_SMAP_ERA5, p_value_SMAP_ERA5 = pearsonr(df_valid_SMAP_ERA5['soil_moisture_avg'], df_valid_SMAP_ERA5['average_moisture'])
    print(f'Overall Pearson correlation for SMAP/ERA5: r = {r_SMAP_ERA5:.2f}, p = {p_value_SMAP_ERA5:.3f}')

    # -----------------------------#
    # Plot CYGNSS vs SMAP
    # -----------------------------#
    plt.figure(figsize=(8, 6))
    plt.scatter(df_valid_CYGNSS_SMAP['sr'], df_valid_CYGNSS_SMAP['soil_moisture_avg'], 
                alpha=0.7, label='Data Points')
    plt.xlabel('CYGNSS Soil Moisture')
    plt.ylabel('SMAP Soil Moisture')
    plt.title('Scatter Plot of Soil Moisture (CYGNSS vs SMAP)')

    # Compute and plot best-fit line
    x = df_valid_CYGNSS_SMAP['sr']
    y = df_valid_CYGNSS_SMAP['soil_moisture_avg']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Best Fit Line')
    plt.legend()
    plt.show()

    # -----------------------------#
    # Plot CYGNSS vs ERA5
    # -----------------------------#
    plt.figure(figsize=(8, 6))
    plt.scatter(df_valid_CYGNSS_ERA5['sr'], df_valid_CYGNSS_ERA5['average_moisture'], 
                alpha=0.7, label='Data Points')
    plt.xlabel('CYGNSS Soil Moisture')
    plt.ylabel('ERA5 Soil Moisture')
    plt.title('Scatter Plot of Soil Moisture (CYGNSS vs ERA5)')

    # Compute and plot best-fit line
    x = df_valid_CYGNSS_ERA5['sr']
    y = df_valid_CYGNSS_ERA5['average_moisture']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Best Fit Line')
    plt.legend()
    plt.show()

    # -----------------------------#
    # Plot SMAP vs ERA5
    # -----------------------------#
    plt.figure(figsize=(8, 6))
    plt.scatter(df_valid_SMAP_ERA5['soil_moisture_avg'], df_valid_SMAP_ERA5['average_moisture'], 
                alpha=0.7, label='Data Points')
    plt.xlabel('SMAP Soil Moisture')
    plt.ylabel('ERA5 Soil Moisture')
    plt.title('Scatter Plot of Soil Moisture (SMAP vs ERA5)')

    # Compute and plot best-fit line
    x = df_valid_SMAP_ERA5['soil_moisture_avg']
    y = df_valid_SMAP_ERA5['average_moisture']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Best Fit Line')
    plt.legend()
    plt.show()


def correlation_matrix(smap_folder, cygnss_folder, era5_folder,
                                  fine_grid_size, coarse_block_size, lsm_threshold):
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
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    
    # Remove the ddm_snr below 2, and max_sp_rx_gain below 13 (We can adjust these values)
    df_cygnss = df_cygnss[df_cygnss['ddm_snr'] > 2]
    df_cygnss = df_cygnss[df_cygnss['sp_rx_gain'] > 13]
    # Expected columns: 'sp_lat', 'sp_lon', 'sr'
    
    # --- ERA5 data --- 
    df_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}').to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)
    # Expected columns: 'latitude', 'longitude', 'average_moisture'
    
    # -----------------------------------------------------------
    # 2. Define a Common Fine Grid for Interpolation
    # -----------------------------------------------------------
    # Determine the overall spatial extents (union of the two datasets. ERA5 is not included because the Land Sea Mask might move the boundaries)
    lat_min = min(df_smap_avg["latitude"].min(), df_cygnss["sp_lat"].min())  # df_era5_lsm["latitude"].min())
    lat_max = max(df_smap_avg["latitude"].max(), df_cygnss["sp_lat"].max())  # df_era5_lsm["latitude"].max())
    lon_min = min(df_smap_avg["longitude"].min(), df_cygnss["sp_lon"].min()) # df_era5_lsm["longitude"].min())
    lon_max = max(df_smap_avg["longitude"].max(), df_cygnss["sp_lon"].max()) #df_era5_lsm["longitude"].max())
    
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
    
    
    # --- CYGNSS ---
    cygnss_points = (df_cygnss["sp_lon"].values, df_cygnss["sp_lat"].values)
    cygnss_vals = df_cygnss["sr"].values
    cygnss_fine = griddata(cygnss_points, cygnss_vals, (lon_mesh, lat_mesh), method='linear')
    
    # --- ERA5 ---
    era5_points = (df_era5_lsm["longitude"].values, df_era5_lsm["latitude"].values)
    era5_vals = df_era5_lsm["average_moisture"].values
    era5_fine = griddata(era5_points, era5_vals, (lon_mesh, lat_mesh), method='linear')
    
    
    # -----------------------------------------------------------
    # 4. Aggregate the Fine Grid into Coarse Cells & Compute Correlation
    # -----------------------------------------------------------
    # Determine number of coarse cells along each axis.
    coarse_n = fine_grid_size // coarse_block_size
    corr_matrix_smap_cygnss = np.full((coarse_n, coarse_n), np.nan)
    corr_matrix_era5_cygnss = np.full((coarse_n, coarse_n), np.nan)
    corr_matrix_smap_era5 = np.full((coarse_n, coarse_n), np.nan)
    
    # The spacing (in degrees) of the fine grid:
    dlon = (lon_max - lon_min) / (fine_grid_size - 1)
    dlat = (lat_max - lat_min) / (fine_grid_size - 1)
    
    # Loop over each coarse cell (non-overlapping blocks) for SMAP and CYGNSS.
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
                corr_matrix_smap_cygnss[i, j] = corr
            else:
                corr_matrix_smap_cygnss[i, j] = np.nan
    # Loop over each coarse cell (non-overlapping blocks) for CYGNSS and ERA5.
    
    for i in range(coarse_n):
        for j in range(coarse_n):
            # Indices for this coarse block
            i_start = i * coarse_block_size
            i_end = (i + 1) * coarse_block_size
            j_start = j * coarse_block_size
            j_end = (j + 1) * coarse_block_size
            
            # Extract the block from each fine grid and flatten to 1D arrays
            block_era5 = era5_fine[i_start:i_end, j_start:j_end].flatten()
            block_cygnss = cygnss_fine[i_start:i_end, j_start:j_end].flatten()
            
            # Compute correlation if there are at least two paired non-NaN values.
            valid = ~np.isnan(block_era5) & ~np.isnan(block_cygnss) 
            if np.sum(valid) >= 2:
                corr = np.corrcoef(block_era5[valid], block_cygnss[valid])[0, 1]
                corr_matrix_era5_cygnss[i, j] = corr
            else:
                corr_matrix_era5_cygnss[i, j] = np.nan
                
    # Loop over each coarse cell (non-overlapping blocks) for SMAP and ERA5.
    for i in range(coarse_n):
        for j in range(coarse_n):
            # Indices for this coarse block
            i_start = i * coarse_block_size
            i_end = (i + 1) * coarse_block_size
            j_start = j * coarse_block_size
            j_end = (j + 1) * coarse_block_size
            
            # Extract the block from each fine grid and flatten to 1D arrays
            block_smap = smap_fine[i_start:i_end, j_start:j_end].flatten()
            block_era5 = era5_fine[i_start:i_end, j_start:j_end].flatten()
            
            # Compute correlation if there are at least two paired non-NaN values.
            valid = ~np.isnan(block_smap) & ~np.isnan(block_era5) 
            if np.sum(valid) >= 2:
                corr = np.corrcoef(block_smap[valid], block_era5[valid])[0, 1]
                corr_matrix_smap_era5[i, j] = corr
            else:
                corr_matrix_smap_era5[i, j] = np.nan
    
    # -----------------------------------------------------------
    # 5. Create Coordinates for the Coarse Cells and Plot the Matrix for SMAP and CYGNSS
    # -----------------------------------------------------------
    # Compute coarse cell centers in physical coordinates.
    coarse_lon_smap_cygnss = np.linspace(lon_min + (coarse_block_size * dlon) / 2,
                             lon_max - (coarse_block_size * dlon) / 2,
                             coarse_n)
    coarse_lat_smap_cygnss = np.linspace(lat_min + (coarse_block_size * dlat) / 2,
                             lat_max - (coarse_block_size * dlat) / 2,
                             coarse_n)
    coarse_lon_mesh_smap_cygnss, coarse_lat_mesh_smap_cygnss = np.meshgrid(coarse_lon_smap_cygnss, coarse_lat_smap_cygnss)
    
    plt.figure(figsize=(10, 8))
    # Plot the coarse correlation matrix.
    mesh = plt.pcolormesh(coarse_lon_mesh_smap_cygnss, coarse_lat_mesh_smap_cygnss, corr_matrix_smap_cygnss, 
                          shading='auto', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(mesh, label='Pearson Correlation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    
    # -----------------------------------------------------------
    # 6. Annotate Each Cell with the Correlation Value
    # -----------------------------------------------------------
    for i in range(coarse_n):
        for j in range(coarse_n):
            corr_val = corr_matrix_smap_cygnss[i, j]
            if not np.isnan(corr_val):
                # Use the corresponding coarse cell center coordinates.
                x_coord = coarse_lon_smap_cygnss[j]
                y_coord = coarse_lat_smap_cygnss[i]
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
    # -----------------------------------------------------------
    # 9. Create Coordinates for the Coarse Cells and Plot the Matrix for CYGNSS and ERA5
    # -----------------------------------------------------------
    # Compute coarse cell centers in physical coordinates.
    coarse_lon_era5_cygnss = np.linspace(lon_min + (coarse_block_size * dlon) / 2,
                             lon_max - (coarse_block_size * dlon) / 2,
                             coarse_n)
    coarse_lat_era5_cygnss = np.linspace(lat_min + (coarse_block_size * dlat) / 2,
                                lat_max - (coarse_block_size * dlat) / 2,
                                coarse_n)
    coarse_lon_mesh_era5_cygnss, coarse_lat_mesh_era5_cygnss = np.meshgrid(coarse_lon_era5_cygnss, coarse_lat_era5_cygnss)
    
    plt.figure(figsize=(10, 8))
    # Plot the coarse correlation matrix.
    mesh = plt.pcolormesh(coarse_lon_mesh_era5_cygnss, coarse_lat_mesh_era5_cygnss, corr_matrix_era5_cygnss, 
                          shading='auto', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(mesh, label='Pearson Correlation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    
    # -----------------------------------------------------------
    # 10. Annotate Each Cell with the Correlation Value
    # -----------------------------------------------------------
    for i in range(coarse_n):
        for j in range(coarse_n):
            corr_val = corr_matrix_era5_cygnss[i, j]
            if not np.isnan(corr_val):
                # Use the corresponding coarse cell center coordinates.
                x_coord = coarse_lon_era5_cygnss[j]
                y_coord = coarse_lat_era5_cygnss[i]
                plt.text(x_coord, y_coord, f"{corr_val:.2f}", 
                         ha='center', va='center', color='black', fontsize=10)
                
    # -----------------------------------------------------------
    # 11. Compute Overall Correlation using all fine-grid values that are valid for both datasets.
    # -----------------------------------------------------------
    valid = ~np.isnan(era5_fine) & ~np.isnan(cygnss_fine)
    if np.sum(valid) >= 2:
        overall_corr = np.corrcoef(era5_fine[valid].flatten(), 
                                   cygnss_fine[valid].flatten())[0, 1]
    else:
        overall_corr = np.nan
    
    # -----------------------------------------------------------
    # 12. Add the overall correlation to the plot title (e.g., below the main title)
    # -----------------------------------------------------------
    plt.title(f'Correlation Matrix between ERA5 and CYGNSS\nOverall Correlation: {overall_corr:.2f}', 
              fontsize=14, pad=20)
    plt.show()
    
    # -----------------------------------------------------------
    # 13. Create Coordinates for the Coarse Cells and Plot the Matrix for SMAP and ERA5
    # -----------------------------------------------------------
    # Compute coarse cell centers in physical coordinates.
    coarse_lon_smap_era5 = np.linspace(lon_min + (coarse_block_size * dlon) / 2,
                             lon_max - (coarse_block_size * dlon) / 2,
                             coarse_n)
    coarse_lat_smap_era5 = np.linspace(lat_min + (coarse_block_size * dlat) / 2,
                                lat_max - (coarse_block_size * dlat) / 2,
                                coarse_n)
    coarse_lon_mesh_smap_era5, coarse_lat_mesh_smap_era5 = np.meshgrid(coarse_lon_smap_era5, coarse_lat_smap_era5)
    
    plt.figure(figsize=(10, 8))
    # Plot the coarse correlation matrix.
    mesh = plt.pcolormesh(coarse_lon_mesh_smap_era5, coarse_lat_mesh_smap_era5, corr_matrix_smap_era5, 
                          shading='auto', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(mesh, label='Pearson Correlation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    
    # -----------------------------------------------------------
    # 14. Annotate Each Cell with the Correlation Value
    # -----------------------------------------------------------
    
    for i in range(coarse_n):
        for j in range(coarse_n):
            corr_val = corr_matrix_smap_era5[i, j]
            if not np.isnan(corr_val):
                # Use the corresponding coarse cell center coordinates.
                x_coord = coarse_lon_smap_era5[j]
                y_coord = coarse_lat_smap_era5[i]
                plt.text(x_coord, y_coord, f"{corr_val:.2f}", 
                         ha='center', va='center', color='black', fontsize=10)
    
    # -----------------------------------------------------------
    # 15. Compute Overall Correlation using all fine-grid values that are valid for both datasets.
    # -----------------------------------------------------------
    valid = ~np.isnan(smap_fine) & ~np.isnan(era5_fine)
    if np.sum(valid) >= 2:
        overall_corr = np.corrcoef(smap_fine[valid].flatten(), 
                                   era5_fine[valid].flatten())[0, 1]
    else:
        overall_corr = np.nan
    
    # -----------------------------------------------------------
    # 16. Add the overall correlation to the plot title (e.g., below the main title)
    # -----------------------------------------------------------
    plt.title(f'Correlation Matrix between SMAP and ERA5\nOverall Correlation: {overall_corr:.2f}', 
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

#correlation_matrix("India2", "India2/India2-20200101-20200107", "India2/ERA5_India2_2020_01_01_07.nc", 100, 10, 0.95)
#correlation_plot( "India2", "India2/India2-20200101-20200107", "India2/ERA5_India2_2020_01_01_07.nc", 0.95)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import xarray as xr
from scipy.ndimage import gaussian_filter

from ERA5.ERA5_utils import apply_land_sea_mask, averaging_soil_moisture
from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from CYGNSS.import_data import importData

#####################################
# Helper functions for Gaussian Blur
#####################################

def apply_gaussian_blur(array, sigma):
    """
    Applies a Gaussian blur to a 2D numpy array while handling NaNs.
    NaNs are temporarily replaced with 0 and a weight mask is used for proper normalization.
    """
    arr = np.copy(array)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 0
    weight = np.ones_like(array)
    weight[nan_mask] = 0
    arr_blur = gaussian_filter(arr, sigma=sigma)
    weight_blur = gaussian_filter(weight, sigma=sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        blurred = arr_blur / weight_blur
    blurred[weight_blur == 0] = np.nan
    return blurred

def apply_gaussian_to_grid(df_grid, value_column, lat_bins, lon_bins, sigma):
    """
    Given a gridded DataFrame (with columns 'lat_center' and 'lon_center'),
    pivot to a full grid (using the bin centers computed from lat_bins and lon_bins),
    apply a Gaussian blur to the grid of values (value_column), and return a long-format DataFrame.
    """
    # Compute bin widths and full grid centers
    lat_bin_width = lat_bins[1] - lat_bins[0]
    lon_bin_width = lon_bins[1] - lon_bins[0]
    lat_centers = lat_bins[:-1] + lat_bin_width / 2
    lon_centers = lon_bins[:-1] + lon_bin_width / 2
    
    # Create a pivot table from the provided dataframe
    pivot = df_grid.pivot(index='lat_center', columns='lon_center', values=value_column)
    # Ensure the pivot has a full grid by reindexing (missing bins become NaN)
    pivot = pivot.reindex(index=lat_centers, columns=lon_centers)
    
    # Apply Gaussian blur to the 2D array of values
    blurred_array = apply_gaussian_blur(pivot.values, sigma=sigma)
    
    # Convert the blurred array back into a DataFrame with the same index/columns
    pivot_blurred = pd.DataFrame(blurred_array, index=lat_centers, columns=lon_centers)
    # Set the index name so that reset_index produces a 'lat_center' column
    pivot_blurred.index.name = 'lat_center'
    
    # Melt back to long format: columns become 'lon_center' and the blurred values in value_column
    df_blurred = pivot_blurred.reset_index().melt(id_vars='lat_center', var_name='lon_center', value_name=value_column)
    
    return df_blurred

    
#####################################
# Gridding and Merging Functions
#####################################

def regrid_dataframe(df, lat_bins, lon_bins, data_source):
    """
    Regrid the provided dataframe by binning lat and lon.
    """
    df = df.copy()  # avoid modifying the original dataframe
    
    if data_source == "CYGNSS":
        df['lat_bin'] = pd.cut(df['sp_lat'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['sp_lon'], bins=lon_bins, right=False)
        # Aggregate using mean
        df_grid = df.groupby(['lat_bin', 'lon_bin'], observed=True)['sr'].mean().reset_index()
    elif data_source == "SMAP":
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'], observed=True)['soil_moisture_avg'].mean().reset_index()
    elif data_source == "ERA5":
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'], observed=True)['average_moisture'].mean().reset_index()
    else:
        raise ValueError("Invalid data source provided")
    
    # Compute bin widths and centers for each bin
    lat_bin_width = lat_bins[1] - lat_bins[0]
    lon_bin_width = lon_bins[1] - lon_bins[0]
    df_grid['lat_center'] = df_grid['lat_bin'].apply(lambda x: float(x.left) + lat_bin_width / 2)
    df_grid['lon_center'] = df_grid['lon_bin'].apply(lambda x: float(x.left) + lon_bin_width / 2)
    
    return df_grid

def merged_dataframe(smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step, gaussian_sigma=0):
    """
    Imports and regrids data from SMAP, CYGNSS, and ERA5. If gaussian_sigma > 0, applies a Gaussian blur
    to each sensor's gridded data before merging.
    """
    # --- SMAP ---
    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap1 = pd.concat(dfs_smap)
    df_smap = SMAP_averaging_soil_moisture(df_smap1)
    
    #df_smap = df_smap[(df_smap["longitude"] >= 68) & (df_smap["longitude"] <= 68.5) & (df_smap["latitude"] >= 27) & (df_smap["latitude"] <= 27.5)]
    


    # --- CYGNSS ---
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    df_cygnss = df_cygnss[df_cygnss['ddm_snr'] >= 2]  
    df_cygnss = df_cygnss[df_cygnss["sp_rx_gain"] >= 0]
    df_cygnss = df_cygnss[df_cygnss["sp_rx_gain"] <= 13]
    df_cygnss = df_cygnss[df_cygnss["sp_inc_angle"] <= 45]
    #df_cygnss = df_cygnss[(df_cygnss["sp_lon"] >= 68) & (df_cygnss["sp_lon"] <= 68.5) & (df_cygnss["sp_lat"] >= 27) & (df_cygnss["sp_lat"] <= 27.5)]

    # --- ERA5 ---
    df_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}').to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)
    
    # Determine overall spatial domain using union of all data
    lat_min = min(df_cygnss['sp_lat'].min(), df_smap['latitude'].min(), df_era5_lsm['latitude'].min())
    lat_max = max(df_cygnss['sp_lat'].max(), df_smap['latitude'].max(), df_era5_lsm['latitude'].max())
    lon_min = min(df_cygnss['sp_lon'].min(), df_smap['longitude'].min(), df_era5_lsm['longitude'].min())
    lon_max = max(df_cygnss['sp_lon'].max(), df_smap['longitude'].max(), df_era5_lsm['longitude'].max())

    # Create bins using provided step sizes
    lat_bins = np.arange(lat_min, lat_max + lat_step, lat_step)
    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)

    # Regrid each dataset
    df_cygnss_grid = regrid_dataframe(df_cygnss, lat_bins, lon_bins, "CYGNSS")    
    df_smap_grid   = regrid_dataframe(df_smap, lat_bins, lon_bins, "SMAP")
    df_era5_grid   = regrid_dataframe(df_era5_lsm, lat_bins, lon_bins, "ERA5")

    # Apply Gaussian blur to each grid if desired
    if gaussian_sigma > 0:
        df_cygnss_grid = apply_gaussian_to_grid(df_cygnss_grid, 'sr', lat_bins, lon_bins, sigma=gaussian_sigma)
        df_smap_grid   = apply_gaussian_to_grid(df_smap_grid, 'soil_moisture_avg', lat_bins, lon_bins, sigma=gaussian_sigma)
        df_era5_grid   = apply_gaussian_to_grid(df_era5_grid, 'average_moisture', lat_bins, lon_bins, sigma=gaussian_sigma)

    # Merge the gridded data based on grid cell centers
    df_merged_CYGNSS_SMAP = pd.merge(df_cygnss_grid, df_smap_grid, on=['lat_center', 'lon_center'],
                                     suffixes=('_cygnss', '_smap'))
    df_merged_CYGNSS_ERA5 = pd.merge(df_cygnss_grid, df_era5_grid, on=['lat_center', 'lon_center'],
                                     suffixes=('_cygnss', '_era5'))
    df_merged_SMAP_ERA5 = pd.merge(df_smap_grid, df_era5_grid, on=['lat_center', 'lon_center'],
                                   suffixes=('_smap', '_era5'))

    return df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5

#####################################
# Plotting Functions
#####################################

def correlation_plot(smap_folder, cygnss_folder, era5_folder, lat_step, lon_step, lsm_threshold, gaussian_sigma=0):
    """
    Plots scatter plots and best-fit lines for the three sensor pairings.
    If gaussian_sigma > 0, the gridded data are blurred before computing correlations.
    """
    df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5 = merged_dataframe(
        smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step, gaussian_sigma=gaussian_sigma)

    # Remove rows with NaNs in relevant columns
    df_valid_CYGNSS_SMAP = df_merged_CYGNSS_SMAP.dropna(subset=['sr', 'soil_moisture_avg'])
    df_valid_CYGNSS_ERA5 = df_merged_CYGNSS_ERA5.dropna(subset=['sr', 'average_moisture'])
    df_valid_SMAP_ERA5 = df_merged_SMAP_ERA5.dropna(subset=['soil_moisture_avg', 'average_moisture'])

    # Calculate and print Pearson correlations
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
    x = df_valid_CYGNSS_SMAP['sr']
    y = df_valid_CYGNSS_SMAP['soil_moisture_avg']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_fit, slope * x_fit + intercept, color='red', linewidth=2, label='Best Fit Line')
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
    x = df_valid_CYGNSS_ERA5['sr']
    y = df_valid_CYGNSS_ERA5['average_moisture']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_fit, slope * x_fit + intercept, color='red', linewidth=2, label='Best Fit Line')
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
    x = df_valid_SMAP_ERA5['soil_moisture_avg']
    y = df_valid_SMAP_ERA5['average_moisture']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_fit, slope * x_fit + intercept, color='red', linewidth=2, label='Best Fit Line')
    plt.legend()
    plt.show()

def spatial_correlation_matrix_old(smap_folder, cygnss_folder, era5_folder,
                               lat_step, lon_step, lsm_threshold,
                               window=1, min_points=3, gaussian_sigma=0):
    """
    Computes local (moving-window) spatial Pearson correlations for:
      - CYGNSS vs SMAP and 
      - CYGNSS vs ERA5.
    If gaussian_sigma > 0, the gridded data are blurred before calculating the correlations.
    """
    # Get gridded (and optionally blurred) merged data.
    df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, _ = merged_dataframe(
        smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step, gaussian_sigma=gaussian_sigma)
    
    # Ensure lat/lon centers are floats.
    df_merged_CYGNSS_SMAP['lat_center'] = df_merged_CYGNSS_SMAP['lat_center'].astype(float)
    df_merged_CYGNSS_SMAP['lon_center'] = df_merged_CYGNSS_SMAP['lon_center'].astype(float)
    df_merged_CYGNSS_ERA5['lat_center'] = df_merged_CYGNSS_ERA5['lat_center'].astype(float)
    df_merged_CYGNSS_ERA5['lon_center'] = df_merged_CYGNSS_ERA5['lon_center'].astype(float)
    
    # Drop grid cells with missing sensor values.
    df_valid_CS = df_merged_CYGNSS_SMAP.dropna(subset=['sr', 'soil_moisture_avg']).reset_index(drop=True)
    df_valid_CE = df_merged_CYGNSS_ERA5.dropna(subset=['sr', 'average_moisture']).reset_index(drop=True)
    
    # Compute local correlation for CYGNSS vs SMAP.
    local_corr_CS = []
    for idx, row in df_valid_CS.iterrows():
        center_lat = row['lat_center']
        center_lon = row['lon_center']
        lat_min_win = center_lat - window * lat_step
        lat_max_win = center_lat + window * lat_step
        lon_min_win = center_lon - window * lon_step
        lon_max_win = center_lon + window * lon_step
        neighborhood = df_valid_CS[
            (df_valid_CS['lat_center'] >= lat_min_win) &
            (df_valid_CS['lat_center'] <= lat_max_win) &
            (df_valid_CS['lon_center'] >= lon_min_win) &
            (df_valid_CS['lon_center'] <= lon_max_win)
        ]
        if len(neighborhood) >= min_points:
            r_local, _ = pearsonr(neighborhood['sr'], neighborhood['soil_moisture_avg'])
        else:
            r_local = np.nan
        local_corr_CS.append(r_local)
    df_valid_CS['local_corr'] = local_corr_CS

    # Compute local correlation for CYGNSS vs ERA5.
    local_corr_CE = []
    for idx, row in df_valid_CE.iterrows():
        center_lat = row['lat_center']
        center_lon = row['lon_center']
        lat_min_win = center_lat - window * lat_step
        lat_max_win = center_lat + window * lat_step
        lon_min_win = center_lon - window * lon_step
        lon_max_win = center_lon + window * lon_step
        neighborhood = df_valid_CE[
            (df_valid_CE['lat_center'] >= lat_min_win) &
            (df_valid_CE['lat_center'] <= lat_max_win) &
            (df_valid_CE['lon_center'] >= lon_min_win) &
            (df_valid_CE['lon_center'] <= lon_max_win)
        ]
        if len(neighborhood) >= min_points:
            r_local, _ = pearsonr(neighborhood['sr'], neighborhood['average_moisture'])
        else:
            r_local = np.nan
        local_corr_CE.append(r_local)
    df_valid_CE['local_corr'] = local_corr_CE

    # Create pivot tables for local correlations.
    pivot_CS = df_valid_CS.pivot(index='lat_center', columns='lon_center', values='local_corr')
    pivot_CE = df_valid_CE.pivot(index='lat_center', columns='lon_center', values='local_corr')
    
    pivot_CS = pivot_CS.sort_index(ascending=True)
    pivot_CE = pivot_CE.sort_index(ascending=True)
    
    # ---------------------------
    # Plot for CYGNSS vs SMAP
    # ---------------------------
    plt.figure(figsize=(8, 6))
    extent_CS = [pivot_CS.columns.min()-lon_step/2, pivot_CS.columns.max()+lon_step/2,
                 pivot_CS.index.min()-lat_step/2, pivot_CS.index.max()+lat_step/2]
    plt.imshow(pivot_CS, origin='lower', aspect='auto', extent=extent_CS, cmap='coolwarm')
    for lat in pivot_CS.index:
        for lon in pivot_CS.columns:
            value = pivot_CS.loc[lat, lon]
            if pd.notna(value):
                plt.text(lon, lat, f'{value:.2f}', ha='center', va='center', fontsize=10, color='black')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Local Correlation: CYGNSS vs SMAP')
    plt.show()
    
    # ---------------------------
    # Plot for CYGNSS vs ERA5
    # ---------------------------
    plt.figure(figsize=(8, 6))
    extent_CE = [pivot_CE.columns.min()-lon_step/2, pivot_CE.columns.max()+lon_step/2,
                 pivot_CE.index.min()-lat_step/2, pivot_CE.index.max()+lat_step/2]
    plt.imshow(pivot_CE, origin='lower', aspect='auto', extent=extent_CE, cmap='coolwarm')
    for lat in pivot_CE.index:
        for lon in pivot_CE.columns:
            value = pivot_CE.loc[lat, lon]
            if pd.notna(value):
                plt.text(lon, lat, f'{value:.2f}', ha='center', va='center', fontsize=10, color='black')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Local Correlation: CYGNSS vs ERA5')
    plt.show()
    
    return pivot_CS, pivot_CE

#####################################
# Example Usage
#####################################
# To run without any blur, you can call:
# spatial_correlation_matrix("Bolivia", "Bolivia/Bolivia-20240701-20240707", 
#                            "Bolivia/ERA5_Bolivia_2024_07_01_07.nc", 0.5, 0.5, 0.9,
#                            window=1, min_points=3, gaussian_sigma=0)
#
# To test with a Gaussian blur (e.g., sigma=1), call:

'''
spatial_correlation_matrix_old("India2", 
                           "India2/India2-20200101-20200131", 
                           "India2/ERA5_India2_2020_01_01_31.nc", 
                           0.5, 0.5, 0.9, window=1, min_points=3, gaussian_sigma=3)
'''






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

###############################################
# Helper Functions for Local Correlation
###############################################

def compute_local_correlation(merged_df, col1, col2, window_size=3):
    """
    Compute a spatial map of Pearson correlation coefficients using a moving window.

    Parameters:
      merged_df : DataFrame with columns 'lat_center', 'lon_center', col1, col2.
      col1, col2: Names of the columns to correlate (e.g., 'sr' and 'soil_moisture_avg').
      window_size: Size of the moving window (should be odd, e.g., 3 means a 3x3 window).

    Returns:
      lat_centers : Sorted unique latitude centers.
      lon_centers : Sorted unique longitude centers.
      corr_matrix: 2D numpy array with the local correlation coefficients.
    """
    # Pivot the merged dataframe into 2D arrays
    pivot1 = merged_df.pivot(index='lat_center', columns='lon_center', values=col1)
    pivot2 = merged_df.pivot(index='lat_center', columns='lon_center', values=col2)

    lat_centers = pivot1.index.values
    lon_centers = pivot1.columns.values
    n_lat, n_lon = pivot1.shape

    # Initialize the output correlation matrix with NaNs
    corr_matrix = np.full((n_lat, n_lon), np.nan)
    half_window = window_size // 2

    # Loop over each grid cell
    for i in range(n_lat):
        for j in range(n_lon):
            # Determine indices for the local window
            i_min = max(0, i - half_window)
            i_max = min(n_lat, i + half_window + 1)
            j_min = max(0, j - half_window)
            j_max = min(n_lon, j + half_window + 1)

            # Extract local window values for each sensor
            window1 = pivot1.iloc[i_min:i_max, j_min:j_max].values.flatten()
            window2 = pivot2.iloc[i_min:i_max, j_min:j_max].values.flatten()

            # Filter out pairs where either value is NaN
            valid = (~np.isnan(window1)) & (~np.isnan(window2))
            if np.sum(valid) >= 3:  # Need at least two valid pairs for Pearson correlation
                r = np.corrcoef(window1[valid], window2[valid])[0, 1]
                corr_matrix[i, j] = r
            else:
                corr_matrix[i, j] = np.nan

    return lat_centers, lon_centers, corr_matrix

def plot_spatial_correlation(lat_centers, lon_centers, corr_matrix, title):
    """
    Plot the spatial correlation matrix with latitude and longitude on the axes,
    and print the correlation value in each grid cell.

    Parameters:
      lat_centers : 1D array of latitude centers.
      lon_centers : 1D array of longitude centers.
      corr_matrix : 2D array of correlation coefficients.
      title       : Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    # Calculate spacing (assumes uniform spacing)
    if len(lat_centers) > 1:
        lat_spacing = lat_centers[1] - lat_centers[0]
    else:
        lat_spacing = 1
    if len(lon_centers) > 1:
        lon_spacing = lon_centers[1] - lon_centers[0]
    else:
        lon_spacing = 1

    # Compute extent for imshow so that cell centers match lat_centers and lon_centers.
    extent = [lon_centers[0] - lon_spacing/2, lon_centers[-1] + lon_spacing/2,
              lat_centers[0] - lat_spacing/2, lat_centers[-1] + lat_spacing/2]

    # Display the correlation matrix using imshow.
    img = plt.imshow(corr_matrix, origin='lower', extent=extent, cmap='Spectral', aspect='auto')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.colorbar(img, label='Pearson r')

    # Annotate each cell with its correlation value.
    for i, lat in enumerate(lat_centers):
        for j, lon in enumerate(lon_centers):
            value = corr_matrix[i, j]
            if not np.isnan(value):
                plt.text(lon, lat, f"{value:.2f}", ha='center', va='center', color='black', fontsize=8)
    plt.show()

###############################################
# Main Function: Spatial Correlation Matrix
###############################################

def spatial_correlation_matrix(smap_folder, cygnss_folder, era5_folder,
                               lsm_threshold, lat_step, lon_step,
                               gaussian_sigma=0, window_size=3):
    """
    Create and plot spatial correlation matrices for:
      - CYGNSS vs SMAP
      - CYGNSS vs ERA5

    This function:
      1. Loads and regrids data from the three sensors using your merged_dataframe function.
      2. Computes and prints the overall (global) Pearson correlation coefficients.
      3. Computes local Pearson correlations using a moving window (adjustable via window_size).
      4. Plots the spatial correlation matrices with latitude and longitude axes.
         Each grid cell is annotated with its correlation value.

    Parameters:
      smap_folder, cygnss_folder, era5_folder: Paths to your data folders.
      lsm_threshold : Land-sea mask threshold.
      lat_step, lon_step: Binning step sizes.
      gaussian_sigma: Sigma for optional Gaussian blur applied to the gridded data.
      window_size : Size of the moving window (e.g., 3 for a 3x3 grid).
    """
    # Regrid and merge the datasets
    df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, _ = merged_dataframe(
        smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step, gaussian_sigma=gaussian_sigma)

    # --- CYGNSS vs SMAP ---
    df_valid_smap = df_merged_CYGNSS_SMAP.dropna(subset=['sr', 'soil_moisture_avg'])
    r_global_smap, p_global_smap = pearsonr(df_valid_smap['sr'], df_valid_smap['soil_moisture_avg'])
    print(f'Overall Pearson correlation for CYGNSS/SMAP: r = {r_global_smap:.2f}, p = {p_global_smap:.3f}')

    lat_centers, lon_centers, corr_matrix_smap = compute_local_correlation(
        df_merged_CYGNSS_SMAP, 'sr', 'soil_moisture_avg', window_size=window_size)
    plot_spatial_correlation(lat_centers, lon_centers, corr_matrix_smap,
                             f'Local Pearson Correlation (CYGNSS vs SMAP) - Window {window_size}x{window_size}')

    # --- CYGNSS vs ERA5 ---
    df_valid_era5 = df_merged_CYGNSS_ERA5.dropna(subset=['sr', 'average_moisture'])
    r_global_era5, p_global_era5 = pearsonr(df_valid_era5['sr'], df_valid_era5['average_moisture'])
    print(f'Overall Pearson correlation for CYGNSS/ERA5: r = {r_global_era5:.2f}, p = {p_global_era5:.3f}')

    lat_centers_era5, lon_centers_era5, corr_matrix_era5 = compute_local_correlation(
        df_merged_CYGNSS_ERA5, 'sr', 'average_moisture', window_size=window_size)
    plot_spatial_correlation(lat_centers_era5, lon_centers_era5, corr_matrix_era5,
                             f'Local Pearson Correlation (CYGNSS vs ERA5) - Window {window_size}x{window_size}')

# Example call:
# spatial_correlation_matrix(smap_folder="path_to_smap_data",
#                            cygnss_folder="path_to_cygnss_data",
#                            era5_folder="path_to_era5_data",
#                            lsm_threshold=0.5,
#                            lat_step=0.1,
#                            lon_step=0.1,
#                            gaussian_sigma=1,
#                            window_size=3)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    pivot to a full grid, apply a Gaussian blur to the grid of values, and return a long-format DataFrame.
    """
    lat_bin_width = lat_bins[1] - lat_bins[0]
    lon_bin_width = lon_bins[1] - lon_bins[0]
    lat_centers = lat_bins[:-1] + lat_bin_width / 2
    lon_centers = lon_bins[:-1] + lon_bin_width / 2
    
    pivot = df_grid.pivot(index='lat_center', columns='lon_center', values=value_column)
    pivot = pivot.reindex(index=lat_centers, columns=lon_centers)
    
    blurred_array = apply_gaussian_blur(pivot.values, sigma=sigma)
    
    pivot_blurred = pd.DataFrame(blurred_array, index=lat_centers, columns=lon_centers)
    pivot_blurred.index.name = 'lat_center'
    
    df_blurred = pivot_blurred.reset_index().melt(id_vars='lat_center', var_name='lon_center', value_name=value_column)
    
    return df_blurred

#####################################
# Gridding and Merging Functions
#####################################

def regrid_dataframe(df, lat_bins, lon_bins, data_source):
    """
    Regrid the provided dataframe by binning lat and lon.
    For vegetation data, we assume the dataframe has columns 'lat', 'lon', and 'veg'.
    """
    df = df.copy()  # avoid modifying the original dataframe
    
    if data_source == "CYGNSS":
        df['lat_bin'] = pd.cut(df['sp_lat'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['sp_lon'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['sr'].mean().reset_index()
    elif data_source == "SMAP":
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['soil_moisture_avg'].mean().reset_index()
    elif data_source == "ERA5":
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['average_moisture'].mean().reset_index()
    elif data_source == "VEG":
        # For vegetation: assume columns 'lat', 'lon', and 'veg'
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['cvl'].mean().reset_index()
    else:
        raise ValueError("Invalid data source provided")
    
    # Compute bin widths and centers
    lat_bin_width = lat_bins[1] - lat_bins[0]
    lon_bin_width = lon_bins[1] - lon_bins[0]
    df_grid['lat_center'] = df_grid['lat_bin'].apply(lambda x: float(x.left) + lat_bin_width / 2)
    df_grid['lon_center'] = df_grid['lon_bin'].apply(lambda x: float(x.left) + lon_bin_width / 2)
    
    return df_grid

def merged_dataframe(smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step, gaussian_sigma=0):
    """
    Imports and regrids data from SMAP, CYGNSS, and ERA5.
    If gaussian_sigma > 0, applies a Gaussian blur to the gridded data before merging.
    """
    # --- SMAP ---
    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap1 = pd.concat(dfs_smap)
    df_smap = SMAP_averaging_soil_moisture(df_smap1)
    
    # --- CYGNSS ---
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    df_cygnss = df_cygnss[df_cygnss['ddm_snr'] >= 2]  # apply ddm_snr threshold

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

    return df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5, lat_bins, lon_bins

###############################################
# Helper Functions for Local Correlation
###############################################

def compute_local_correlation(merged_df, col1, col2, window_size=3):
    """
    Compute a spatial map of Pearson correlation coefficients using a moving window.
    """
    # Pivot the merged dataframe into 2D arrays
    pivot1 = merged_df.pivot(index='lat_center', columns='lon_center', values=col1)
    pivot2 = merged_df.pivot(index='lat_center', columns='lon_center', values=col2)

    lat_centers = pivot1.index.values
    lon_centers = pivot1.columns.values
    n_lat, n_lon = pivot1.shape

    corr_matrix = np.full((n_lat, n_lon), np.nan)
    half_window = window_size // 2

    for i in range(n_lat):
        for j in range(n_lon):
            i_min = max(0, i - half_window)
            i_max = min(n_lat, i + half_window + 1)
            j_min = max(0, j - half_window)
            j_max = min(n_lon, j + half_window + 1)

            window1 = pivot1.iloc[i_min:i_max, j_min:j_max].values.flatten()
            window2 = pivot2.iloc[i_min:i_max, j_min:j_max].values.flatten()

            valid = (~np.isnan(window1)) & (~np.isnan(window2))
            if np.sum(valid) >= 2:
                r = np.corrcoef(window1[valid], window2[valid])[0, 1]
                corr_matrix[i, j] = r
            else:
                corr_matrix[i, j] = np.nan

    return lat_centers, lon_centers, corr_matrix

###############################################
# New Function: Scatter Plot of Correlation vs Vegetation
###############################################

def scatter_corr_vs_veg(smap_folder, cygnss_folder, era5_folder, veg_file,
                        lsm_threshold, lat_step, lon_step, gaussian_sigma=0, window_size=3):
    """
    1. Merges CYGNSS vs SMAP and CYGNSS vs ERA5 data and computes local spatial correlations using a moving window.
    2. Loads and regrids vegetation data from a netCDF file.
    3. Merges the local correlation with vegetation percent for each grid cell.
    4. Plots scatter plots of local correlation (y-axis) vs. vegetation percent (x-axis) for both sensor pairs.
    """
    # Merge CYGNSS with SMAP and ERA5 data and get grid bins
    df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, _, lat_bins, lon_bins = merged_dataframe(
        smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step, gaussian_sigma=gaussian_sigma)
    
    # Load vegetation data from netCDF file.
    ds_veg = xr.open_dataset(veg_file)
    # Assume the vegetation variable is named 'cvl'; adjust if necessary.
    veg = ds_veg['cvl']
    df_veg = veg.to_dataframe().reset_index()  # Should contain columns 'lat', 'lon', and 'cvl'
    
    # Regrid vegetation data using the same bins.
    df_veg_grid = regrid_dataframe(df_veg, lat_bins, lon_bins, "VEG")
    
    # --- CYGNSS vs SMAP ---
    # Compute local spatial correlation using the moving window
    lat_centers, lon_centers, corr_matrix_smap = compute_local_correlation(
        df_merged_CYGNSS_SMAP, 'sr', 'soil_moisture_avg', window_size=window_size)
    
    # Convert the correlation matrix into a long-format DataFrame.
    corr_rows_smap = []
    for i, lat in enumerate(lat_centers):
        for j, lon in enumerate(lon_centers):
            corr_rows_smap.append({'lat_center': lat, 'lon_center': lon, 'local_corr': corr_matrix_smap[i, j]})
    df_corr_smap = pd.DataFrame(corr_rows_smap)
    
    # Merge with vegetation grid for CYGNSS vs SMAP
    df_scatter_smap = pd.merge(df_corr_smap, df_veg_grid, on=['lat_center', 'lon_center'], how='inner')
    
    # --- CYGNSS vs ERA5 ---
    # Compute local spatial correlation using the moving window
    lat_centers_era5, lon_centers_era5, corr_matrix_era5 = compute_local_correlation(
        df_merged_CYGNSS_ERA5, 'sr', 'average_moisture', window_size=window_size)
    
    # Convert the correlation matrix into a long-format DataFrame.
    corr_rows_era5 = []
    for i, lat in enumerate(lat_centers_era5):
        for j, lon in enumerate(lon_centers_era5):
            corr_rows_era5.append({'lat_center': lat, 'lon_center': lon, 'local_corr': corr_matrix_era5[i, j]})
    df_corr_era5 = pd.DataFrame(corr_rows_era5)
    
    # Merge with vegetation grid for CYGNSS vs ERA5
    df_scatter_era5 = pd.merge(df_corr_era5, df_veg_grid, on=['lat_center', 'lon_center'], how='inner')
    
    # --- Plotting ---
    # Create two subplots for the scatter plots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot for CYGNSS vs SMAP
    ax1.scatter(df_scatter_smap['cvl'] * 100, df_scatter_smap['local_corr'])
    ax1.set_xlabel('Vegetation Percent (%)')
    ax1.set_ylabel('Local Spatial Correlation')
    ax1.set_title('CYGNSS vs SMAP')
    ax1.grid(True)
    
    # Scatter plot for CYGNSS vs ERA5
    ax2.scatter(df_scatter_era5['cvl'] * 100, df_scatter_era5['local_corr'])
    ax2.set_xlabel('Vegetation Percent (%)')
    ax2.set_ylabel('Local Spatial Correlation')
    ax2.set_title('CYGNSS vs ERA5')
    ax2.grid(True)
    
    plt.suptitle('Local Correlation vs. Vegetation Percent')
    plt.show()


###############################################
# Example Call
###############################################

# Replace the folder/file paths and parameters as needed.
scatter_corr_vs_veg(
    smap_folder="India2",
    cygnss_folder="India2/India2-20200101-20200131",
    era5_folder="India2/ERA5_India2_2020_01_01_31.nc",
    veg_file="data/Vegetation/vegetation.nc",  # path to your ERA5 vegetation file
    lsm_threshold=0.9,
    lat_step=0.5,
    lon_step=0.5,
    gaussian_sigma=2.5,
    window_size=3
)

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

def compute_tri_neighbour(elevation_fine, kernel_size=3):
    """
    Compute a neighborhood-based TRI for the DEM fine grid.
    For each cell, calculate the mean absolute difference between the cell's value 
    and those of its neighbors in a kernel window.
    
    Parameters:
      elevation_fine: 2D array of DEM values.
      kernel_size: size of the neighborhood window (e.g., 3 for a 3x3 window).
      
    Returns:
      A 2D array (same shape as elevation_fine) containing the local TRI values.
    """
    from scipy.ndimage import generic_filter
    
    def neighborhood_diff(values):
        # The center cell is at the middle index of the flattened array.
        central = values[len(values) // 2]
        # If the center is NaN, return NaN.
        if np.isnan(central):
            return np.nan
        # Return the mean absolute difference between the center and all values.
        return np.nanmean(np.abs(values - central))
    
    tri_fine = generic_filter(elevation_fine, neighborhood_diff, 
                              size=kernel_size, mode='constant', cval=np.nan)
    return tri_fine

def correlation_vs_tri(smap_folder, cygnss_folder, era5_folder, lsm_threshold, dem_file, lat_step, lon_step):
    """
    Compute the correlation between SMAP and CYGNSS and between ERA5 and CYGNSS,
    and compare these correlations to the Terrain Ruggedness Index (TRI) using a common fine grid.
    
    Parameters:
      smap_folder: str, location of SMAP data.
      cygnss_folder: str, location of CYGNSS data.
      dem_file: str, path to the DEM .nc file.
      lat_step: float, bin size in latitude (degrees).
      lon_step: float, bin size in longitude (degrees).
      era5_folder: str, folder or filename for ERA5 data.
      lsm_threshold: float, threshold for applying the land–sea mask to ERA5.
    """
    # --- Load SMAP & CYGNSS Data ---
    df_smap = pd.concat(importDataSMAP(False, smap_folder))
    df_smap_avg = SMAP_averaging_soil_moisture(df_smap)

    df_cygnss = pd.concat(importData(cygnss_folder))
    df_cygnss = df_cygnss[(df_cygnss['ddm_snr'] > 2) & (df_cygnss['sp_rx_gain'] > 13)]

    # --- Determine spatial extents ---
    lat_min = min(df_smap_avg["latitude"].min(), df_cygnss["sp_lat"].min())
    lat_max = max(df_smap_avg["latitude"].max(), df_cygnss["sp_lat"].max())
    lon_min = min(df_smap_avg["longitude"].min(), df_cygnss["sp_lon"].min())
    lon_max = max(df_smap_avg["longitude"].max(), df_cygnss["sp_lon"].max())

    # --- Create Fine Grid ---
    # Use a grid that is 10× finer than the bin (i.e. lat_step, lon_step)
    interp_factor = 10
    fine_lat_step = lat_step / interp_factor
    fine_lon_step = lon_step / interp_factor
    lat_fine = np.arange(lat_min, lat_max + fine_lat_step, fine_lat_step)
    lon_fine = np.arange(lon_min, lon_max + fine_lon_step, fine_lon_step)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)

    # --- Interpolate SMAP & CYGNSS onto Fine Grid ---
    smap_fine = griddata(
        (df_smap_avg["longitude"], df_smap_avg["latitude"]),
        df_smap_avg["soil_moisture_avg"],
        (lon_mesh, lat_mesh),
        method='nearest'
    )
    cygnss_fine = griddata(
        (df_cygnss["sp_lon"], df_cygnss["sp_lat"]),
        df_cygnss["sr"],
        (lon_mesh, lat_mesh),
        method='nearest'
    )
    
    # --- Load and Interpolate DEM Data ---
    dem_ds = xr.open_dataset(dem_file)
    elevation = dem_ds["ASTER_GDEM_DEM"].values
    dem_lat = dem_ds["lat"].values
    dem_lon = dem_ds["lon"].values
    dem_lon_mesh, dem_lat_mesh = np.meshgrid(dem_lon, dem_lat)
    dem_fine = griddata(
        (dem_lon_mesh.flatten(), dem_lat_mesh.flatten()),
        elevation.flatten(),
        (lon_mesh, lat_mesh),
        method='nearest'
    )

    # --- Load ERA5 Data ---
    df_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}').to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)
    era5_fine = griddata(
        (df_era5_lsm["longitude"], df_era5_lsm["latitude"]),
        df_era5_lsm["average_moisture"],
        (lon_mesh, lat_mesh),
        method='nearest'
    )

    # --- Define Coarse Bins Using lat_step and lon_step ---
    lat_edges = np.arange(lat_min, lat_max, lat_step)
    lon_edges = np.arange(lon_min, lon_max, lon_step)
    n_lat_bins = len(lat_edges)
    n_lon_bins = len(lon_edges)
    # Optionally compute bin centers:
    lat_centers = lat_edges + lat_step/2
    lon_centers = lon_edges + lon_step/2

    # --- Compute TRI using the neighborhood-based method ---
    # First compute a fine-grid TRI map using a moving window (e.g., 3×3)
    tri_fine = compute_tri_neighbour(dem_fine, kernel_size=3)
    
    # Now aggregate (average) the local TRI values within each bin.
    tri_matrix = np.full((n_lat_bins, n_lon_bins), np.nan)
    for i in range(n_lat_bins):
        for j in range(n_lon_bins):
            lat_lower = lat_edges[i]
            lat_upper = lat_lower + lat_step
            lon_lower = lon_edges[j]
            lon_upper = lon_lower + lon_step
            mask = ((lat_mesh >= lat_lower) & (lat_mesh < lat_upper) &
                    (lon_mesh >= lon_lower) & (lon_mesh < lon_upper))
            bin_tri_values = tri_fine[mask].flatten()
            valid_values = bin_tri_values[~np.isnan(bin_tri_values)]
            if len(valid_values) >= 5:
                tri_matrix[i, j] = np.nanmean(valid_values)

    # --- Compute Correlation (SMAP vs. CYGNSS) in Each Bin ---
    corr_matrix_smap = np.full((n_lat_bins, n_lon_bins), np.nan)
    for i in range(n_lat_bins):
        for j in range(n_lon_bins):
            lat_lower = lat_edges[i]
            lat_upper = lat_lower + lat_step
            lon_lower = lon_edges[j]
            lon_upper = lon_lower + lon_step
            mask = ((lat_mesh >= lat_lower) & (lat_mesh < lat_upper) &
                    (lon_mesh >= lon_lower) & (lon_mesh < lon_upper))
            block_smap = smap_fine[mask].flatten()
            block_cygnss = cygnss_fine[mask].flatten()
            valid = ~np.isnan(block_smap) & ~np.isnan(block_cygnss)
            if np.sum(valid) >= 2:
                corr_matrix_smap[i, j] = pearsonr(block_smap[valid], block_cygnss[valid])[0]

    # --- Compute Correlation (ERA5 vs. CYGNSS) in Each Bin ---
    corr_matrix_era5 = np.full((n_lat_bins, n_lon_bins), np.nan)
    for i in range(n_lat_bins):
        for j in range(n_lon_bins):
            lat_lower = lat_edges[i]
            lat_upper = lat_lower + lat_step
            lon_lower = lon_edges[j]
            lon_upper = lon_lower + lon_step
            mask = ((lat_mesh >= lat_lower) & (lat_mesh < lat_upper) &
                    (lon_mesh >= lon_lower) & (lon_mesh < lon_upper))
            block_era5 = era5_fine[mask].flatten()
            block_cygnss = cygnss_fine[mask].flatten()
            valid = ~np.isnan(block_era5) & ~np.isnan(block_cygnss)
            if np.sum(valid) >= 2:
                corr_matrix_era5[i, j] = pearsonr(block_era5[valid], block_cygnss[valid])[0]

    # --- Prepare Data for Scatter Plots ---
    # For SMAP vs CYGNSS:
    tri_values = tri_matrix.flatten()
    corr_values_smap = corr_matrix_smap.flatten()
    valid_idx_smap = ~np.isnan(tri_values) & ~np.isnan(corr_values_smap)
    tri_values_smap = tri_values[valid_idx_smap]
    corr_values_smap = corr_values_smap[valid_idx_smap]

    # For ERA5 vs CYGNSS:
    corr_values_era5 = corr_matrix_era5.flatten()
    valid_idx_era5 = ~np.isnan(tri_values) & ~np.isnan(corr_values_era5)
    tri_values_era5 = tri_values[valid_idx_era5]
    corr_values_era5 = corr_values_era5[valid_idx_era5]

    if tri_values_smap.size == 0:
        print("No valid TRI values for SMAP correlation. Check your DEM and bin settings.")
        return
    if corr_values_smap.size == 0:
        print("No valid correlation values for SMAP. Check your data and bin settings.")
        return
    if corr_values_era5.size == 0:
        print("No valid correlation values for ERA5. Check your data and bin settings.")
        return

    print("SMAP vs CYGNSS: TRI min:", np.nanmin(tri_values_smap), "max:", np.nanmax(tri_values_smap))
    print("SMAP vs CYGNSS: Correlation min:", np.nanmin(corr_values_smap), "max:", np.nanmax(corr_values_smap))
    print("ERA5 vs CYGNSS: Correlation min:", np.nanmin(corr_values_era5), "max:", np.nanmax(corr_values_era5))
    print(f"Valid SMAP data points: {len(tri_values_smap)}")
    print(f"Valid ERA5 data points: {len(tri_values_era5)}")

    # --- Scatter Plot: SMAP vs CYGNSS ---
    plt.figure(figsize=(8, 6))
    plt.scatter(tri_values_smap, corr_values_smap, alpha=0.7, edgecolors='k')
    plt.xlabel("Terrain Ruggedness Index (TRI)")
    plt.ylabel("Pearson Correlation (SMAP vs. CYGNSS)")
    plt.title("Correlation vs. TRI (SMAP vs. CYGNSS)")
    plt.grid(True)
    plt.show()

    # --- Scatter Plot: ERA5 vs CYGNSS ---
    plt.figure(figsize=(8, 6))
    plt.scatter(tri_values_era5, corr_values_era5, alpha=0.7, edgecolors='k', color='orange')
    plt.xlabel("Terrain Ruggedness Index (TRI)")
    plt.ylabel("Pearson Correlation (ERA5 vs. CYGNSS)")
    plt.title("Correlation vs. TRI (ERA5 vs. CYGNSS)")
    plt.grid(True)
    plt.show()
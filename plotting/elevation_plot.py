import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D

from ERA5.ERA5_utils import apply_land_sea_mask, averaging_soil_moisture
from CYGNSS.import_data import importData


def correlation_3d_terrain(cygnss_folder, era5_folder, lsm_threshold, dem_file, lat_step, lon_step):
    # --- Load CYGNSS Data ---
    df_cygnss = pd.concat(importData(cygnss_folder))
    df_cygnss = df_cygnss[(df_cygnss['ddm_snr'] > 2) & (df_cygnss['sp_rx_gain'] > 13)]

    # --- Load ERA5 Data ---
    df_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}').to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)

    # --- Determine spatial extent ---
    lat_min = min(df_cygnss["sp_lat"].min(), df_era5_lsm["latitude"].min())
    lat_max = max(df_cygnss["sp_lat"].max(), df_era5_lsm["latitude"].max())
    lon_min = min(df_cygnss["sp_lon"].min(), df_era5_lsm["longitude"].min())
    lon_max = max(df_cygnss["sp_lon"].max(), df_era5_lsm["longitude"].max())

    # --- Create fine grid ---
    interp_factor = 10
    fine_lat_step = lat_step / interp_factor
    fine_lon_step = lon_step / interp_factor
    lat_fine = np.arange(lat_min, lat_max + fine_lat_step, fine_lat_step)
    lon_fine = np.arange(lon_min, lon_max + fine_lon_step, fine_lon_step)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)

    # --- Interpolate CYGNSS and ERA5 onto fine grid ---
    cygnss_fine = griddata(
        (df_cygnss["sp_lon"], df_cygnss["sp_lat"]),
        df_cygnss["sr"],
        (lon_mesh, lat_mesh),
        method='nearest'
    )

    era5_fine = griddata(
        (df_era5_lsm["longitude"], df_era5_lsm["latitude"]),
        df_era5_lsm["average_moisture"],
        (lon_mesh, lat_mesh),
        method='nearest'
    )

    # --- Load and interpolate DEM ---
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

    # --- Define coarse bins ---
    lat_edges = np.arange(lat_min, lat_max, lat_step)
    lon_edges = np.arange(lon_min, lon_max, lon_step)
    n_lat_bins = len(lat_edges)
    n_lon_bins = len(lon_edges)
    lat_centers = lat_edges + lat_step / 2
    lon_centers = lon_edges + lon_step / 2

    # --- Compute correlation matrix (ERA5 vs CYGNSS) ---
    corr_matrix = np.full((n_lat_bins, n_lon_bins), np.nan)
    elevation_binned = np.full((n_lat_bins, n_lon_bins), np.nan)
    for i in range(n_lat_bins):
        for j in range(n_lon_bins):
            lat_lower, lat_upper = lat_edges[i], lat_edges[i] + lat_step
            lon_lower, lon_upper = lon_edges[j], lon_edges[j] + lon_step
            mask = ((lat_mesh >= lat_lower) & (lat_mesh < lat_upper) &
                    (lon_mesh >= lon_lower) & (lon_mesh < lon_upper))

            block_era5 = era5_fine[mask].flatten()
            block_cygnss = cygnss_fine[mask].flatten()
            block_elev = dem_fine[mask].flatten()

            valid = ~np.isnan(block_era5) & ~np.isnan(block_cygnss)
            if np.sum(valid) >= 2:
                corr_matrix[i, j] = pearsonr(block_era5[valid], block_cygnss[valid])[0]
            if np.sum(~np.isnan(block_elev)) > 0:
                elevation_binned[i, j] = np.nanmean(block_elev)

    # --- Create mesh for plotting ---
    lon_centers_mesh, lat_centers_mesh = np.meshgrid(lon_centers, lat_centers)

    # --- Normalize correlation for color mapping ---
    corr_min = np.nanmin(corr_matrix)
    corr_max = np.nanmax(corr_matrix)
    norm_corr = (corr_matrix - corr_min) / (corr_max - corr_min)

    # --- Plot 3D Terrain ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        lon_centers_mesh, lat_centers_mesh, elevation_binned,
        facecolors=plt.cm.viridis(norm_corr),
        linewidth=0, antialiased=False, shade=False
    )

    # --- Add colorbar ---
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(corr_matrix)
    plt.colorbar(mappable, ax=ax, shrink=0.5, label='Correlation (ERA5 vs CYGNSS)')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Elevation (m)")
    ax.set_title("3D Terrain Colored by ERA5 vs CYGNSS Correlation")
    plt.tight_layout()
    plt.show()

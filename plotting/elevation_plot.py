import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D

from ERA5.ERA5_utils import apply_land_sea_mask, averaging_soil_moisture
from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from CYGNSS.import_data import importData


def correlation_3d_terrain(smap_folder, cygnss_folder, era5_file, dem_file,
                                     lsm_threshold, lat_step, lon_step,
                                     dataset='ERA5'):
    assert dataset in ['ERA5', 'SMAP'], "dataset must be either 'ERA5' or 'SMAP'"

    # --- Load CYGNSS Data ---
    df_cygnss = pd.concat(importData(cygnss_folder))
    df_cygnss = df_cygnss[(df_cygnss['ddm_snr'] > 2) & (df_cygnss['sp_rx_gain'] < 13)]

    # --- Load SMAP Data ---
    df_smap = pd.concat(importDataSMAP(False, smap_folder))
    df_smap_avg = SMAP_averaging_soil_moisture(df_smap)

    # --- Load ERA5 Data ---
    df_era5 = xr.open_dataset(era5_file).to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)

    # --- Determine spatial extent ---
    lat_min = min(df_cygnss["sp_lat"].min(), df_smap_avg["latitude"].min(), df_era5_lsm["latitude"].min())
    lat_max = max(df_cygnss["sp_lat"].max(), df_smap_avg["latitude"].max(), df_era5_lsm["latitude"].max())
    lon_min = min(df_cygnss["sp_lon"].min(), df_smap_avg["longitude"].min(), df_era5_lsm["longitude"].min())
    lon_max = max(df_cygnss["sp_lon"].max(), df_smap_avg["longitude"].max(), df_era5_lsm["longitude"].max())

    # --- Create fine grid ---
    interp_factor = 10
    fine_lat_step = lat_step / interp_factor
    fine_lon_step = lon_step / interp_factor
    lat_fine = np.arange(lat_min, lat_max + fine_lat_step, fine_lat_step)
    lon_fine = np.arange(lon_min, lon_max + fine_lon_step, fine_lon_step)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)

    # --- Interpolate datasets ---
    cygnss_fine = griddata(
        (df_cygnss["sp_lon"], df_cygnss["sp_lat"]),
        df_cygnss["sr"],
        (lon_mesh, lat_mesh),
        method='linear'
    )

    era5_fine = griddata(
        (df_era5_lsm["longitude"], df_era5_lsm["latitude"]),
        df_era5_lsm["average_moisture"],
        (lon_mesh, lat_mesh),
        method='linear'
    )

    smap_fine = griddata(
        (df_smap_avg["longitude"], df_smap_avg["latitude"]),
        df_smap_avg["soil_moisture_avg"],
        (lon_mesh, lat_mesh),
        method='linear'
    )

    # --- Load and interpolate DEM ---
    dem_ds = xr.open_dataset(dem_file)
    elevation = dem_ds["__xarray_dataarray_variable__"].values
    dem_lon = dem_ds["x"].values
    dem_lat = dem_ds["y"].values
    dem_lon_mesh, dem_lat_mesh = np.meshgrid(dem_lon, dem_lat)
    dem_fine = griddata(
        (dem_lon_mesh.flatten(), dem_lat_mesh.flatten()),
        elevation.flatten(),
        (lon_mesh, lat_mesh),
        method='linear'
    )

    # --- Define coarse bins ---
    lat_edges = np.arange(lat_min, lat_max, lat_step)
    lon_edges = np.arange(lon_min, lon_max, lon_step)
    n_lat_bins = len(lat_edges)
    n_lon_bins = len(lon_edges)
    lat_centers = lat_edges + lat_step / 2
    lon_centers = lon_edges + lon_step / 2

    # --- Compute correlation matrices ---
    corr_matrix_smap = np.full((n_lat_bins, n_lon_bins), np.nan)
    corr_matrix_era5 = np.full((n_lat_bins, n_lon_bins), np.nan)
    elevation_binned = np.full((n_lat_bins, n_lon_bins), np.nan)

    for i in range(n_lat_bins):
        for j in range(n_lon_bins):
            lat_lower = lat_edges[i]
            lat_upper = lat_lower + lat_step
            lon_lower = lon_edges[j]
            lon_upper = lon_lower + lon_step
            mask = ((lat_mesh >= lat_lower) & (lat_mesh < lat_upper) &
                    (lon_mesh >= lon_lower) & (lon_mesh < lon_upper))

            block_cygnss = cygnss_fine[mask].flatten()
            block_era5 = era5_fine[mask].flatten()
            block_smap = smap_fine[mask].flatten()
            block_elev = dem_fine[mask].flatten()

            # Correlation with ERA5
            valid_era5 = ~np.isnan(block_era5) & ~np.isnan(block_cygnss)
            if np.sum(valid_era5) >= 2:
                corr_matrix_era5[i, j] = pearsonr(block_era5[valid_era5], block_cygnss[valid_era5])[0]

            # Correlation with SMAP
            valid_smap = ~np.isnan(block_smap) & ~np.isnan(block_cygnss)
            if np.sum(valid_smap) >= 2:
                corr_matrix_smap[i, j] = pearsonr(block_smap[valid_smap], block_cygnss[valid_smap])[0]

            # Mean elevation
            if np.sum(~np.isnan(block_elev)) > 0:
                elevation_binned[i, j] = np.nanmean(block_elev)

    # --- Choose dataset for visualization ---
    corr_matrix = corr_matrix_era5 if dataset == 'ERA5' else corr_matrix_smap

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
    label = f"Correlation ({dataset} vs CYGNSS)"
    plt.colorbar(mappable, ax=ax, shrink=0.5, label=label)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Elevation (m)")
    ax.set_title(f"3D Terrain Colored by {label}")
    plt.tight_layout()
    plt.show()

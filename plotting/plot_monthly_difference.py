import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, binned_statistic_2d
from scipy.interpolate import griddata

# Data import utilities
from CYGNSS.import_data import importData
from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from ERA5.ERA5_utils import averaging_soil_moisture, apply_land_sea_mask



def _load_cygnss(
    folder, sigma=0, step_lon=1.0, step_lat=1.0, min_count=1
):
    """
    Load CYGNSS data, compute mean SR per grid cell, mask cells with fewer
    than min_count samples, interpolate missing, and optionally smooth.
    Returns lon_edges, lat_edges, stat_grid, lon_mesh, lat_mesh.
    """
    # Load and filter
    dfs = importData(folder)
    df = pd.concat(dfs)
    df = df[df['ddm_snr'] >= 2]
    df = df[(df['sp_rx_gain'] >= 0) & (df['sp_rx_gain'] <= 13)]
    df = df[df['sp_inc_angle'] <= 45]

    lon = df['sp_lon'].values
    lat = df['sp_lat'].values
    sr  = df['sr'].values

    # Define grid edges
    lon_edges = np.arange(lon.min(), lon.max() + step_lon, step_lon)
    lat_edges = np.arange(lat.min(), lat.max() + step_lat, step_lat)

    # Compute mean and count per cell
    mean_stat, _, _, _ = binned_statistic_2d(
        lon, lat, sr, statistic='mean', bins=[lon_edges, lat_edges]
    )
    count_stat, _, _, _ = binned_statistic_2d(
        lon, lat, sr, statistic='count', bins=[lon_edges, lat_edges]
    )
    mean_stat = mean_stat.T
    count_stat = count_stat.T

    # Mask cells below min_count
    mean_stat[count_stat < min_count] = np.nan

    # Set up mesh centers for interpolation
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)

    # Interpolate remaining missing values
    mask = ~np.isnan(mean_stat)
    if np.any(mask):
        points = np.column_stack((lon_mesh[mask], lat_mesh[mask]))
        values = mean_stat[mask]
        mean_stat = griddata(points, values, (lon_mesh, lat_mesh), method='linear')

    # Optional Gaussian smoothing
    if sigma > 0:
        mean_stat = gaussian_filter(mean_stat, sigma=sigma)

    return lon_edges, lat_edges, mean_stat, lon_mesh, lat_mesh


def _load_smap(folder, sigma=0, step_lon=1.0, step_lat=1.0):
    # Load SMAP and interpolate onto grid
    dfs = importDataSMAP(False, folder)
    df = pd.concat(dfs)
    df = SMAP_averaging_soil_moisture(df)

    lon = df['longitude'].values
    lat = df['latitude'].values
    sm  = df['soil_moisture_avg'].values

    lon_edges = np.arange(lon.min(), lon.max() + step_lon, step_lon)
    lat_edges = np.arange(lat.min(), lat.max() + step_lat, step_lat)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)

    grid = griddata((lon, lat), sm, (lon_mesh, lat_mesh), method='linear')
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)

    return lon_edges, lat_edges, grid, lon_mesh, lat_mesh


def _load_era5(
    filepath, sigma=0, threshold=0.5, step_lon=1.0, step_lat=1.0
):
    # Load ERA5, mask ocean, interpolate onto grid
    ds = xr.open_dataset(filepath, engine='netcdf4')
    df = ds.to_dataframe().reset_index()
    df = averaging_soil_moisture(df)
    df = apply_land_sea_mask(df, threshold)

    lon = df['longitude'].values
    lat = df['latitude'].values
    sm  = df['average_moisture'].values

    lon_edges = np.arange(lon.min(), lon.max() + step_lon, step_lon)
    lat_edges = np.arange(lat.min(), lat.max() + step_lat, step_lat)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)

    grid = griddata((lon, lat), sm, (lon_mesh, lat_mesh), method='linear')
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)

    return lon_edges, lat_edges, grid, lon_mesh, lat_mesh


def monthly_difference_analysis(
    cygnss_jan_folder, cygnss_jun_folder,
    smap_jan_folder,   smap_jun_folder,
    era5_jan_file,     era5_jun_file,
    sigma_cygnss=0, sigma_smap=0, sigma_era5=0,
    step_lon=1.0, step_lat=1.0,
    era5_threshold=0.5,
    min_count_cygnss=1,
    absolute_plot=False
):
    """
    Compute Jan−Jun differences for CYGNSS, SMAP, and ERA5;
    mask CYGNSS cells with < min_count samples;
    plot maps (optionally as absolute values);
    return Pearson r/p for CYGNSS vs SMAP and vs ERA5.
    """
    # CYGNSS Jan & Jun with masking
    lon_e_c, lat_e_c, grid_c_jan, lon_mesh_c, lat_mesh_c = _load_cygnss(
        cygnss_jan_folder, sigma_cygnss, step_lon, step_lat, min_count=min_count_cygnss
    )
    _,      _,      grid_c_jun, _,          _         = _load_cygnss(
        cygnss_jun_folder, sigma_cygnss, step_lon, step_lat, min_count=min_count_cygnss
    )
    diff_c = grid_c_jan - grid_c_jun

    # SMAP
    lon_e_s, lat_e_s, grid_s_jan, lon_mesh_s, lat_mesh_s = _load_smap(
        smap_jan_folder, sigma_smap, step_lon, step_lat
    )
    _,       _,       grid_s_jun, _,          _         = _load_smap(
        smap_jun_folder, sigma_smap, step_lon, step_lat
    )
    diff_s = grid_s_jan - grid_s_jun

    # ERA5
    lon_e_e, lat_e_e, grid_e_jan, lon_mesh_e, lat_mesh_e = _load_era5(
        era5_jan_file, sigma_era5, era5_threshold, step_lon, step_lat
    )
    _,       _,       grid_e_jun, _,          _         = _load_era5(
        era5_jun_file, sigma_era5, era5_threshold, step_lon, step_lat
    )
    diff_e = grid_e_jan - grid_e_jun

    # Plot differences
    for title, lon_edges, lat_edges, diff in [
        ('CYGNSS Jan−Jun', lon_e_c, lat_e_c, diff_c),
        ('SMAP Jan−Jun',   lon_e_s, lat_e_s, diff_s),
        ('ERA5 Jan−Jun',   lon_e_e, lat_e_e, diff_e),
    ]:
        plot_data = np.abs(diff) if absolute_plot else diff
        plot_title = f"{'Absolute ' if absolute_plot else ''}{title}"
        cmap_label = 'Absolute Jan−Jun' if absolute_plot else 'Jan−Jun'

        plt.figure(figsize=(10, 8))
        mesh = plt.pcolormesh(lon_edges, lat_edges, plot_data, shading='auto', cmap='RdBu_r')
        plt.colorbar(mesh, label=cmap_label)
        plt.title(plot_title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        plt.show()

    # Interpolate for correlation (signed diffs)
    s_on_c = griddata(
        (lon_mesh_s.flatten(), lat_mesh_s.flatten()),
        diff_s.flatten(),
        (lon_mesh_c, lat_mesh_c),
        method='linear'
    )
    e_on_c = griddata(
        (lon_mesh_e.flatten(), lat_mesh_e.flatten()),
        diff_e.flatten(),
        (lon_mesh_c, lat_mesh_c),
        method='linear'
    )

    mask = ~np.isnan(diff_c) & ~np.isnan(s_on_c) & ~np.isnan(e_on_c)
    vals_c = diff_c[mask]
    vals_s = s_on_c[mask]
    vals_e = e_on_c[mask]

    r_s, p_s = pearsonr(vals_c, vals_s)
    r_e, p_e = pearsonr(vals_c, vals_e)

    print(f"Pearson CYGNSS vs SMAP: r = {r_s:.3f}, p = {p_s:.3e}")
    print(f"Pearson CYGNSS vs ERA5: r = {r_e:.3f}, p = {p_e:.3e}")

    return {'CYGNSS_SMAP': (r_s, p_s), 'CYGNSS_ERA5': (r_e, p_e)}


jan_cygnss_folder = 'Pakistan\Pakistan-20200101-20200131'
jun_cygnss_folder = 'Pakistan\Pakistan-20200601-20200630'

jan_smap_folder   = 'Pakistan_20200101_20200131'
jun_smap_folder   = 'Pakistan_20200601_20200630'

jan_era5_filepath  = 'data\ERA5\Pakistan\ERA5_Pakistan_20200101_20200131.nc' # ERa5 stored as single file
jun_era5_filepath  = 'data\ERA5\Pakistan\ERA5_Pakistan_20200601_20200630.nc' # ERa5 stored as single file

monthly_difference_analysis(
    jan_cygnss_folder, jun_cygnss_folder,
    jan_smap_folder,   jun_smap_folder, 
    jan_era5_filepath, jun_era5_filepath,
    sigma_cygnss=1.5, sigma_smap=1.5, sigma_era5=1.5,
    step_lon=0.08, step_lat=0.08,
    era5_threshold=0.9, min_count_cygnss=0
)
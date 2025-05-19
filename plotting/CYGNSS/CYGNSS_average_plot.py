import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .import_data import importData
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata  # Import for interpolation
from matplotlib.ticker import MultipleLocator, FuncFormatter


def CYGNSS_average_plot(folder_name, sigma, step_size_lon, step_size_lat, smooth):
    # Create a title based on whether smoothing is applied.
    name = folder_name.split("/")[0]
    if sigma == 0:
        title = f'CYGNSS Surface Reflectivity - {name} - September 2024'
    else:
        title = f'Smoothed CYGNSS Surface Reflectivity - {name} - September 2024 - σ={sigma}'

    # Import and concatenate data from the folder.
    dfs = importData(folder_name)
    df = pd.concat(dfs)



    min_lat, max_lat = -30.25, -28.25
    min_lon, max_lon = 118.5, 120.5
    df = df[
        (df['sp_lat'] >= min_lat) & (df['sp_lat'] <= max_lat) &
        (df['sp_lon'] >= min_lon) & (df['sp_lon'] <= max_lon)
    ]

    # Extract the latitude, longitude, and surface reflectivity values.
    latitudes = df["sp_lat"].values
    longitudes = df["sp_lon"].values
    sr_values = df["sr"].values

    # Dropping DDM_SNR and SP_RX_GAIN below thresholds.
    df = df[df['ddm_snr'] > 2]
    df = df[df['sp_rx_gain'] < 13]

    # Define grid bin edges based on the desired step sizes.
    lat_edges = np.arange(min_lat, max_lat + step_size_lat, step_size_lat)
    lon_edges = np.arange(min_lon, max_lon + step_size_lon, step_size_lon)

    #lat_edges = np.arange(latitudes.min(), latitudes.max() + step_size_lat, step_size_lat)
    #lon_edges = np.arange(longitudes.min(), longitudes.max() + step_size_lon, step_size_lon)

    # Bin the data: average the 'sr' values that fall within each grid cell.
    stat, _, _, _ = binned_statistic_2d(
        longitudes, latitudes, sr_values, statistic='mean', bins=[lon_edges, lat_edges]
    )
    # Transpose so that the array has shape (n_lat_bins, n_lon_bins)
    stat = stat.T

    # Create grid centers for interpolation.
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)

    # Identify bins with valid data.
    mask = ~np.isnan(stat)
    if np.any(mask):
        # Points with data.
        points = np.column_stack((lon_mesh[mask], lat_mesh[mask]))
        values = stat[mask]
        # Interpolate missing values using nearest-neighbor.
        stat = griddata(points, values, (lon_mesh, lat_mesh), method='linear')
    
    # Optionally apply Gaussian smoothing if sigma > 0.
    if sigma > 0:
        stat = gaussian_filter(stat, sigma=sigma)

    fig, ax = plt.subplots(figsize=(12, 10))

    if smooth:
        mesh = ax.imshow(
            stat,
            origin='lower',
            extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
            cmap='viridis',
            interpolation='bilinear',
            aspect='auto'
        )
    else:
        mesh = ax.pcolormesh(
            lon_edges, lat_edges, stat, shading='auto', cmap="viridis"
        )

    # Colorbar below or beside plot
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label('SR [dB]', fontsize=32, labelpad=24)
    cbar.ax.tick_params(labelsize=28)

    # Axis labels
    ax.set_xlabel('Longitude', fontsize=32, labelpad=20)
    ax.set_ylabel('Latitude', fontsize=32, labelpad=20)

    # Tick label font size and padding
    ax.tick_params(axis='x', labelsize=28, pad=10)  # pad increases distance from axis
    ax.tick_params(axis='y', labelsize=28, pad=10)

    # Title
    ax.set_title("CYGNSS SR - Lake Barlee- January & February 2020", fontsize=32, pad=30)
    ax.set_aspect('equal', adjustable='box')


    def format_lon(x, _):
        direction = 'E' if x >= 0 else 'W'
        return f"{abs(x):.1f}°{direction}"

    def format_lat(y, _):
        direction = 'N' if y >= 0 else 'S'
        return f"{abs(y):.1f}°{direction}"

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    fig.tight_layout()
    plt.show()


    plt.xlabel('Longitude [degrees]', fontsize=32, labelpad=20)
    plt.ylabel('Latitude [degrees]', fontsize=32, labelpad=20)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.title("CYGNSS SR - Lake Barlee - January & February 2020", fontsize=35, pad=20)

    plt.tight_layout()  # <- Also helps remove white space
    plt.show()
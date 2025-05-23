import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .import_data import importData
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata
import matplotlib.ticker as mticker

def CYGNSS_average_plot(folder_name, sigma, step_size_lon, step_size_lat, smooth):
    # Fixed title
    title = "CYGNSS SR - Lake Barlee - Jan & Feb 2020"

    # Import and concatenate data
    dfs = importData(folder_name)
    df = pd.concat(dfs)

    # Apply filters
    df = df[df['ddm_snr'] >= 2]
    df = df[df['sp_rx_gain'].between(0, 13)]
    df = df[df["sp_inc_angle"] <= 65]

    # Extract lat/lon/SR
    latitudes = df["sp_lat"].values
    longitudes = df["sp_lon"].values
    sr_values = df["sr"].values

    # Fixed Lake Barlee bounds
    lat_min, lat_max = latitudes.min(), latitudes.max()
    lon_min, lon_max = longitudes.min(), longitudes.max()

    # Define bin edges
    lat_edges = np.arange(lat_min, lat_max, step_size_lat)
    if lat_edges.size == 0 or lat_edges[-1] < lat_max:
        lat_edges = np.append(lat_edges, lat_max)

    lon_edges = np.arange(lon_min, lon_max, step_size_lon)
    if lon_edges.size == 0 or lon_edges[-1] < lon_max:
        lon_edges = np.append(lon_edges, lon_max)

    # Bin data to compute mean SR
    stat, _, _, _ = binned_statistic_2d(
        longitudes, latitudes, sr_values,
        statistic='mean',
        bins=[lon_edges, lat_edges]
    )
    stat = stat.T

    # (Optional) Check for empty bins
    count, _, _, _ = binned_statistic_2d(
        longitudes, latitudes, sr_values,
        statistic='count',
        bins=[lon_edges, lat_edges]
    )
    empty = (count.T == 0)
    print("Empty bins in your grid:", np.argwhere(empty))

    # Grid centers for interpolation
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)

    # Two-stage interpolation: linear then nearest to fill holes
    mask = ~np.isnan(stat)
    if mask.any():
        points = np.column_stack((lon_mesh[mask], lat_mesh[mask]))
        values = stat[mask]

        # 1) Linear interpolation for smooth values
        stat_lin = griddata(points, values,
                            (lon_mesh, lat_mesh),
                            method='linear')

        # 2) Nearest-neighbour for any NaNs left by linear
        stat_nn = griddata(points, values,
                           (lon_mesh, lat_mesh),
                           method='nearest')

        # 3) Combine: use linear where available, else nearest
        stat = np.where(np.isnan(stat_lin), stat_nn, stat_lin)

    # Smooth with Gaussian filter if requested
    if sigma > 0:
        stat = gaussian_filter(stat, sigma=sigma)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, stat,
        shading='auto', cmap='viridis'
    )

    # Labels, title, colorbar
    ax.set_title(title, fontsize=32, pad=30)
    ax.set_xlabel("Longitude", fontsize=32, labelpad=20)
    ax.set_ylabel("Latitude", fontsize=32, labelpad=20)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label("SR [dB]", fontsize=32, labelpad=24)
    cbar.ax.tick_params(labelsize=28)

    # Tick formatting
    ax.tick_params(axis='x', labelsize=28, pad=10)
    ax.tick_params(axis='y', labelsize=28, pad=10)

    def format_lon(x, _):
        return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"
    def format_lat(y, _):
        return f"{abs(y):.0f}°{'N' if y >= 0 else 'S'}"

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    fig.tight_layout()
    plt.show()

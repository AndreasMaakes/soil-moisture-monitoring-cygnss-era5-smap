import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as mticker

from .ERA5_utils import averaging_soil_moisture, apply_land_sea_mask

def ERA5_gaussian_blur_plot(folder_name, sigma, threshold, lat_step, lon_step):
    ds = xr.open_dataset(f'data/ERA5/{folder_name}', engine='netcdf4') 
    df = ds.to_dataframe().reset_index()

    averaged_df = averaging_soil_moisture(df)
    lsm_df = apply_land_sea_mask(averaged_df, threshold)

    latitudes = lsm_df["latitude"].values
    longitudes = lsm_df["longitude"].values
    moisture_values = lsm_df["average_moisture"].values

    # Set region bounds (Lake Barlee example)
    lat_min, lat_max = -30.25, -28.25
    lon_min, lon_max = 118.5, 120.5

    # Create grid edges
    lat_edges = np.arange(lat_min, lat_max, lat_step)
    if lat_edges.size == 0 or lat_edges[-1] < lat_max:
        lat_edges = np.append(lat_edges, lat_max)

    lon_edges = np.arange(lon_min, lon_max, lon_step)
    if lon_edges.size == 0 or lon_edges[-1] < lon_max:
        lon_edges = np.append(lon_edges, lon_max)

    # Compute grid centers
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    # Interpolate to center grid
    moisture_grid = griddata(
        (longitudes, latitudes),
        moisture_values,
        (lon_grid, lat_grid),
        method="linear"
    )

    # Apply Gaussian blur
    smoothed_data = gaussian_filter(moisture_grid, sigma=sigma)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, smoothed_data,
        shading='auto', cmap='viridis'
    )

    ax.set_title("ERA5 SM - Lake Barlee - Jan & Feb 2020", fontsize=32, pad=30)
    #ax.set_xlabel("Longitude", fontsize=32, labelpad=20)
    #ax.set_ylabel("Latitude", fontsize=32, labelpad=20)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label("Soil Moisture [m$^3$/m$^3$]", fontsize=32, labelpad=24)
    cbar.ax.tick_params(labelsize=28)

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

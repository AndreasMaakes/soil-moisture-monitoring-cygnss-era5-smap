import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from .SMAP_import_data import importDataSMAP
from .SMAP_utils import SMAP_averaging_soil_moisture
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

def SMAP_gaussian_blur_plot(folder_name, sigma, step_size_lon, step_size_lat):
    # Import the data as a list of DataFrames and concatenate them
    df_list = importDataSMAP(False, folder_name)
    df = pd.concat(df_list)
    
    # Average the soil moisture values
    df = SMAP_averaging_soil_moisture(df)
    
    # Extract latitude, longitude, and moisture values
    latitudes = df["latitude"].values
    longitudes = df["longitude"].values
    moisture_values = df["soil_moisture_avg"].values
    
    # Set the plot boundaries to the original data boundaries.
    # For your Thailand example these should be: 14, 18, 99, and 105.
    lat_min, lat_max = -30.25, -28.25
    lon_min, lon_max = 118.5, 120.5
    
    # Create grid cell edges using the desired step sizes.
    # Use np.arange from the minimum up to (but not including) the maximum,
    # then append the maximum so that the grid exactly spans the interval.
    lat_edges = np.arange(lat_min, lat_max, step_size_lat)
    if lat_edges.size == 0 or lat_edges[-1] < lat_max:
        lat_edges = np.append(lat_edges, lat_max)
    
    lon_edges = np.arange(lon_min, lon_max, step_size_lon)
    if lon_edges.size == 0 or lon_edges[-1] < lon_max:
        lon_edges = np.append(lon_edges, lon_max)
    
    # Compute the cell centers from the edges.
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    
    # Interpolate the soil moisture data onto the grid defined by the cell centers.
    moisture_grid = griddata(
        (longitudes, latitudes),
        moisture_values,
        (lon_grid, lat_grid),
        method='linear'
    )
    
    # Apply Gaussian smoothing
    smoothed_data = gaussian_filter(moisture_grid, sigma=sigma)
    
    import matplotlib.ticker as mticker  # put this at the top if not already

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot using pcolormesh with correct bounds
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, smoothed_data,
        shading='auto', cmap='viridis'
    )

    # Title, labels, and colorbar
    ax.set_title(f"SMAP SM - Lake Barlee - January & February 2020", fontsize=32, pad=30)
    ax.set_xlabel("Longitude", fontsize=32, labelpad=20)
    ax.set_ylabel("Latitude", fontsize=32, labelpad=20)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label("Soil Moisture [m$^3$/m$^3$]", fontsize=32, labelpad=24)
    cbar.ax.tick_params(labelsize=28)

    # Ticks & format
    ax.tick_params(axis='x', labelsize=28, pad=10)
    ax.tick_params(axis='y', labelsize=28, pad=10)

    def format_lon(x, _):
        return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

    def format_lat(y, _):
        return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))

    # Aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(118.5, 120.5)
    ax.set_ylim(-30.25, -28.25)

    fig.tight_layout()
    plt.show()


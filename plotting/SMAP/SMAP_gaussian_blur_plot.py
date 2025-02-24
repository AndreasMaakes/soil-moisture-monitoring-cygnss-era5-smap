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
    lat_min, lat_max = latitudes.min(), latitudes.max()
    lon_min, lon_max = longitudes.min(), longitudes.max()
    
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
    
    # Plot with pcolormesh using the cell edges so that the boundaries are exactly preserved.
    plt.figure(figsize=(10, 8))
    mesh = plt.pcolormesh(lon_edges, lat_edges, smoothed_data,
                          shading='auto', cmap='viridis')
    plt.colorbar(mesh, label='Soil Moisture (Smoothed)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Smoothed SMAP Soil Moisture for {folder_name}')
    
    # Set the plot limits to exactly match the original boundaries.
    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.axis('equal')
    plt.show()

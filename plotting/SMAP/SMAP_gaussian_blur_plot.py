import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from .SMAP_import_data import importDataSMAP
from .SMAP_utils import SMAP_averaging_soil_moisture
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

def SMAP_gaussian_blur_plot(folder_name, sigma, grid_size):

    # Importing the data as a list of dataframes
    df = importDataSMAP(folder_name)

    # Concatenating the dataframes into one single dataframe
    df = pd.concat(df)

    # Averaging the soil moisture values
    df = SMAP_averaging_soil_moisture(df)

    # Extract latitude, longitude, and soil moisture values
    latitudes = df["latitude"].values
    longitudes = df["longitude"].values
    moisture_values = df["soil_moisture_avg"].values

    # Define new finer grid resolution
    lat_min, lat_max = latitudes.min(), latitudes.max()
    lon_min, lon_max = longitudes.min(), longitudes.max()

    lat_new = np.linspace(lat_min, lat_max, grid_size)
    lon_new = np.linspace(lon_min, lon_max, grid_size)
    lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)

    # Interpolate soil moisture data onto new grid
    moisture_grid = griddata(
        (longitudes, latitudes), 
        moisture_values, 
        (lon_grid, lat_grid), 
        method="linear"
    )

    # Apply Gaussian blur
    smoothed_data = gaussian_filter(moisture_grid, sigma=sigma)

    # Plot the smoothed soil moisture data
    plt.figure(figsize=(10, 8))
    mesh = plt.pcolormesh(
        lon_grid, lat_grid, smoothed_data, 
        shading='auto', cmap='viridis'
    )
    plt.colorbar(mesh, label='Soil Moisture (Smoothed)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Smoothed SMAP Soil Moisture for {folder_name}')

    # Set aspect ratio to equal to ensure squares are not distorted
    plt.axis('equal')


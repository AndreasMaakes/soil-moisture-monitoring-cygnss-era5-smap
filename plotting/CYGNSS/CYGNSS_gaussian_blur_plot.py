import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .import_data import importData
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

def CYGNSS_gaussian_blur_plot(folder_name, sigma, grid_size):
    title = ""
    name = folder_name.split("/")[0]

    # Define the plot title based on sigma value
    if sigma == 0:
        title = f'CYGNSS Surface Reflectivity - {name} - September 2024'
    else:
        title = f'Smoothed CYGNSS Surface Reflectivity - {name} - September 2024 - σ={sigma}'

    # Import data from the folder
    dfs = importData(folder_name)
    
    df = pd.concat(dfs)


    # Extract latitude, longitude, and surface reflectivity
    latitudes = df["sp_lat"].values
    longitudes = df["sp_lon"].values
    sr_values = df["sr"].values

    # Define the structured grid for interpolation
    lat_grid = np.linspace(latitudes.min(), latitudes.max(), grid_size)  # 200 points in latitude
    lon_grid = np.linspace(longitudes.min(), longitudes.max(),grid_size)  # 200 points in longitude
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate data onto the structured grid
    sr_grid = griddata(
        (longitudes, latitudes), sr_values, (lon_mesh, lat_mesh), method='linear'
    )

    # Apply Gaussian blur
    smoothed_data = gaussian_filter(sr_grid, sigma=sigma)

    # Plot the smoothed surface reflectivity data
    plt.figure(figsize=(10, 8))
    mesh = plt.pcolormesh(lon_mesh, lat_mesh, smoothed_data, shading='auto', cmap='viridis')
    plt.colorbar(mesh, label='SR')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)

    # Ensure the aspect ratio is equal
    plt.axis('equal')

    plt.show()

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from .ERA5_utils import averaging_soil_moisture
from .ERA5_utils import apply_land_sea_mask



def ERA5_gaussian_blur_plot(folder_name, sigma, threshold, grid_size):

    ds = xr.open_dataset(f'data/ERA5/{folder_name}', engine='netcdf4') 
    df = ds.to_dataframe().reset_index()

    averaged_df = averaging_soil_moisture(df)
    lsm_df = apply_land_sea_mask(averaged_df, threshold)

# Extract latitude, longitude, and soil moisture values
    latitudes = lsm_df["latitude"].values
    longitudes = lsm_df["longitude"].values
    moisture_values = lsm_df["average_moisture"].values

    # Define new grid resolution
    lat_min, lat_max = latitudes.min(), latitudes.max()
    lon_min, lon_max = longitudes.min(), longitudes.max()

    # Create new grid with specified resolution
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
    plt.title('Smoothed ERA5 Soil Moisture')

    # Set aspect ratio to equal
    plt.axis('equal')

    plt.show()

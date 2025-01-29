import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from .ERA5_utils import averaging_soil_moisture
from .ERA5_utils import apply_land_sea_mask



def ERA5_gaussian_blur_plot(folder_name, sigma, threshold):

    ds = xr.open_dataset(f'data/ERA5/{folder_name}', engine='netcdf4') 
    df = ds.to_dataframe().reset_index()

    averaged_df = averaging_soil_moisture(df)
    lsm_df = apply_land_sea_mask(averaged_df, threshold)

    #Hei

    # Create a pivot table with latitude as rows and longitude as columns
    pivoted_data = lsm_df.pivot_table(
        index="latitude", 
        columns="longitude", 
        values="average_moisture"
    )

    # Apply a Gaussian filter to the data
    smoothed_data = gaussian_filter(pivoted_data.values, sigma=sigma)

    # Create a meshgrid of latitudes and longitudes
    latitudes = pivoted_data.index.values
    longitudes = pivoted_data.columns.values
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Plot the smoothed soil moisture data using pcolormesh
    plt.figure(figsize=(10, 8))
    mesh = plt.pcolormesh(
        lon_grid, lat_grid, smoothed_data, 
        shading='auto', cmap='viridis'
    )
    plt.colorbar(mesh, label='Soil Moisture (Smoothed)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Smoothed ERA5 Soil Moisture')

    # Set aspect ratio to equal to ensure squares are not distorted
    plt.axis('equal')

    plt.show()
    
    
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import xarray as xr
from .SMAP_import_data import importDataSMAP
from .SMAP_utils import SMAP_averaging_soil_moisture
from scipy.ndimage import gaussian_filter

def SMAP_gaussian_blur_plot(folder_name, sigma):

    #Importing the data as a list of dataframes
    df = importDataSMAP(folder_name)

    #Concating the dataframes to one single dataframe
    df = pd.concat(df)

    #Averaging the soil moisture values
    df = SMAP_averaging_soil_moisture(df)

    # Create a pivot table with latitude as rows and longitude as columns
    pivoted_data = df.pivot_table(
        index="latitude", 
        columns="longitude", 
        values="soil_moisture_avg"
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
    plt.title(f'Smoothed SMAP Soil Moisture for {folder_name}')

    # Set aspect ratio to equal to ensure squares are not distorted
    plt.axis('equal')

    plt.show() 
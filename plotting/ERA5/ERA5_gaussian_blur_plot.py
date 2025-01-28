import os
import xarray as xr
import numpy as np
import pandas as pd

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from .ERA5_utils import averaging_soil_moisture
from .ERA5_utils import apply_land_sea_mask


def ERA5_gaussian_blur_plot(folder_name, sigma):

    ds = xr.open_dataset(f'data/ERA5/{folder_name}', engine='netcdf4') 
    df = ds.to_dataframe().reset_index()

    averaged_df = averaging_soil_moisture(df)
    lsm_df = apply_land_sea_mask(averaged_df, 0.95)


    lats = np.array(lsm_df['latitude'])
    lons = np.array(lsm_df['longitude'])
    soil_moisture = np.array(lsm_df['average_moisture'])

    max_lat = np.max(lats)
    min_lat = np.min(lats)
    max_lon = np.max(lons)
    min_lon = np.min(lons)


    # Creating a grid
    lat_step = 0.1  # Grid resolution for latitude
    lon_step = 0.1 # Grid resolution for longitude
    lat_grid = np.arange(min_lat, max_lat, lat_step)
    lon_grid = np.arange(min_lon, max_lon, lon_step)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    #Interpolating using nearest neighbor
    grid_values = griddata((lats, lons), soil_moisture, (lat_grid, lon_grid), method='nearest')

    # Apply Gaussian filter
    grid_values_blurred = gaussian_filter(grid_values, sigma=sigma)

    # Flatten the grid to create the necessary structure
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    z_flat = grid_values_blurred.flatten()

    hovertext = [f'VSWL: {sr_val:.2f}' for sr_val in z_flat]

    # Create a scatter plot
    scatter = go.Scatter(
    x=lon_flat,
    y=lat_flat,
    mode='markers',
    text=hovertext,
    hoverinfo='text',
    marker=dict(
        size=13,  # Adjust marker size
        color=z_flat,
        colorscale='portland_r', # Choose a colorscale
        colorbar=dict(title='VSWL  m^3 * m^(-3)'),
        opacity=1,
    )
    )

    # Layout with Cartesian axes and centered title
    layout = go.Layout(
    title=dict(
        text="Volumetric soil water level 1 (0-7 cm)",
        x=0.5,  # Center the title
        xanchor='center',
        font=dict(size=35)  # Adjust title font size
    ),
    xaxis=dict(
        title="Longitude",
        showgrid=True,
        zeroline=False,
    ),
    yaxis=dict(
        title="Latitude",
        showgrid=True,
        zeroline=False,
    ),
    height=1000,
    width=1800,
    plot_bgcolor='white',  # Set the plot background to white
    paper_bgcolor='white',  # Set the outer paper background to white
    font=dict(size=25)

    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

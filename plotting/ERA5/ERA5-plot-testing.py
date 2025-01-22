import os
import xarray as xr
import numpy as np
import pandas as pd

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go


'''
def readERA5Data(folder_name):
    
    #Data folder
    basePath = f'/data/ERA5/{folder_name}'
   
    #Iterating through all files in the folder, and adding the data to the dataFrames array
    for file in os.listdir(basePath):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath, file)
            ds = xr.open_dataset(filePath, engine='netcdf4') #Opening the file using xarray
            df = ds.to_dataframe().reset_index()
            min_sr = min(min_sr, df["sr"].min()) #Finding the minimum sr value of all the data
            df.name = file #Adding name of file to dataframe
            dataFrames.append(df)
    
    return dataFrames
'''

ds = xr.open_dataset('data/ERA5/Brazil/ERA5_Brazil_2024_07_24_24.nc', engine='netcdf4') 
df = ds.to_dataframe().reset_index()

df['valid_time'] = pd.to_datetime(df['valid_time'])

averaged_df = (
    df.groupby(['latitude', 'longitude'])
    .agg(average_moisture=('swvl1', 'mean'))
    .reset_index()
)

print(averaged_df)

lats = np.array(averaged_df['latitude'])
lons = np.array(averaged_df['longitude'])
soil_moisture = np.array(averaged_df['average_moisture'])




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
grid_values_blurred = gaussian_filter(grid_values, sigma=1.0)

# Flatten the grid to create the necessary structure
lat_flat = lat_grid.flatten()
lon_flat = lon_grid.flatten()
z_flat = grid_values_blurred.flatten()


hovertext = [f'SR: {sr_val:.2f}' for sr_val in z_flat]

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
    colorbar=dict(title='SR (dB)'),
    opacity=1,
)
)

# Layout with Cartesian axes and centered title
layout = go.Layout(
title=dict(
    text="Hello",
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
width=1450,
plot_bgcolor='white',  # Set the plot background to white
paper_bgcolor='white',  # Set the outer paper background to white
font=dict(size=25)

)

fig = go.Figure(data=[scatter], layout=layout)

fig.show()
'''

hovertext = [f'SR: {sr_val:.2f}' for sr_val in soil_moisture]

# Create a scatter plot
scatter = go.Scatter(
    x=lons,
    y=lats,
    mode='markers',
    text=hovertext,
    hoverinfo='text',
    marker=dict(
        size=6,
        color=soil_moisture,
        colorscale='portland_r',
        colorbar=dict(title='SR (dB)'),
        opacity=1,
    )
)

# Layout with Cartesian axes
layout = go.Layout(
    title=dict(
        text="Ye",
        x=0.5,  # Center the title
        xanchor='center',
        font=dict(size=35)
    ), 
    xaxis=dict(
        title="Longitude",
        showgrid=True,
        zeroline=False
    ),
    yaxis=dict(
        title="Latitude",
        showgrid=True,
        zeroline=False
    ),
    height=1000,
    width=1450,
    font=dict(size=25),
    plot_bgcolor='white',  # White background
    paper_bgcolor='white'  # White outer background
)

fig = go.Figure(data=[scatter], layout=layout)

# Show the figure
fig.show()
'''

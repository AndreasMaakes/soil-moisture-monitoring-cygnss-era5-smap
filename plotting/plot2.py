import plotly.graph_objects as go
from functions import importData
from scipy.interpolate import griddata
import numpy as np
import sys
import os



dataFrames = importData()
lats = np.array([])
lons = np.array([])
srs = np.array([])

for df in dataFrames:
    lat = np.array(df['sp_lat'])
    lon = np.array(df['sp_lon'])
    sr = np.array(df['sr'])
    
    lats = np.append(lats, lat)
    lons = np.append(lons, lon)
    srs = np.append(srs, sr)
    
max_lat = 36
min_lat = 28

max_lon = 65
min_lon = 54

#Creating a grid
lat_grid = np.arange(28, 36, 0.1)
lon_grid = np.arange(54, 65, 0.1)
lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

#Interpolating
grid_values = griddata((lats, lons), srs, (lat_grid, lon_grid), method = 'nearest')

#Trying heatmeap
heatmap = go.Heatmap(
    z = grid_values,
    x = np.arange(min_lon, max_lon, 0.1),
    y = np.arange(min_lat, max_lat, 0.1),
    colorscale = 'Viridis',
    colorbar = dict(title = 'SR')
    )
layout = go.Layout(
    title = 'Interpolated spatial distribution of SR',
    xaxis = dict(title = 'Longitude'),
    yaxis = dict(title = 'Latitude'),
    height = 600,
    width = 800
)
fig = go.Figure(data = [heatmap], layout = layout)

fig.show()
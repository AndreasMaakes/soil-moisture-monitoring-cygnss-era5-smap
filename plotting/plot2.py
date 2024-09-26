import plotly.graph_objects as go
from functions import importData
from scipy.interpolate import griddata
import numpy as np

# Your data import and preparation remains the same
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

# Creating a grid
lat_step = 0.1  # Grid resolution for latitude
lon_step = 0.1  # Grid resolution for longitude
lat_grid = np.arange(min_lat, max_lat, lat_step)
lon_grid = np.arange(min_lon, max_lon, lon_step)
lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

# Interpolating
grid_values = griddata((lats, lons), srs, (lat_grid, lon_grid), method='nearest')

# Flatten the grid to create the necessary structure for mapbox plotting
lat_flat = lat_grid.flatten()
lon_flat = lon_grid.flatten()
z_flat = grid_values.flatten()

# Calculate the average latitude (for scaling marker size proportionally)
mean_lat = np.mean(lat_flat)

# Calculate marker size: in degrees, size should reflect grid cell size (lon_step, lat_step)
# Convert the degrees to approximate kilometers for scaling
# Latitude changes ~111 km per degree, Longitude scales by cos(latitude)
lat_km_per_deg = 111.0  # Rough approximation for latitude (km/degree)
lon_km_per_deg = 111.0 * np.cos(np.radians(mean_lat))  # Adjust for longitude shrinking with latitude

# Convert lat/lon steps into approximate sizes in kilometers for marker scaling
marker_size_lat = lat_step * lat_km_per_deg  # Scale latitude step to kilometers
marker_size_lon = lon_step * lon_km_per_deg  # Scale longitude step to kilometers

# Now, use the larger of the two dimensions to determine the marker size in kilometers
marker_size_km = max(marker_size_lat, marker_size_lon)

hovertext = [f'SR: {sr_val:.2f}' for sr_val in z_flat]

# Create the Mapbox heatmap
#Color scales and syntax found here
#https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scattermapbox.html

heatmap = go.Scattermapbox(
    lat=lat_flat,
    lon=lon_flat,
    mode='markers',
    hovertext = hovertext,
    hoverinfo = 'text',
    marker=go.scattermapbox.Marker(
        size=5,  # Adjust the marker size to match grid dimensions
        color=z_flat,
        colorscale='RdYlBu',
        colorbar=dict(title='SR'),
        showscale=True,
        opacity=1,
    ),
)

# Layout with a mapbox background
layout = go.Layout(
    title='Interpolated Spatial Distribution of SR',
    mapbox=dict(
        style='outdoors',  # You can choose different styles such as 'stamen-terrain', 'carto-positron', etc.
        center=dict(lat=(min_lat + max_lat) / 2, lon=(min_lon + max_lon) / 2),
        zoom=5,  # Adjust zoom level based on your region
    ),
    height=800,
    width=1000
)

fig = go.Figure(data=[heatmap], layout=layout)

# Add your Mapbox access token here
mapbox_access_token = 'pk.eyJ1Ijoib2xlZXZjYSIsImEiOiJjbTFldmt6aGIyeWN4MmxzamFrYTV3dTNxIn0.bbVpqBfsIl_Y0W7YGRXCgQ'
fig.update_layout(mapbox_accesstoken=mapbox_access_token)

fig.show()

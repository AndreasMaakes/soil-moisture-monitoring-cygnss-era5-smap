import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy as np
from import_data import importData 

def gaussian_blur_plot_2(folder_name, saveplot, sigma, color_min, color_max):  # Added color_min and color_max
    dataFrames = importData(folder_name)
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
    
    max_lat = np.max(lats)
    min_lat = np.min(lats)
    max_lon = np.max(lons)
    min_lon = np.min(lons)

    # Creating a grid
    lat_step = 0.1  # Grid resolution for latitude
    lon_step = 0.1  # Grid resolution for longitude
    lat_grid = np.arange(min_lat, max_lat, lat_step)
    lon_grid = np.arange(min_lon, max_lon, lon_step)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Interpolating
    grid_values = griddata((lats, lons), srs, (lat_grid, lon_grid), method='nearest')

    # Apply Gaussian filter
    grid_values_blurred = gaussian_filter(grid_values, sigma=sigma)

    # Flatten the grid to create the necessary structure for mapbox plotting
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    z_flat = grid_values_blurred.flatten()

    hovertext = [f'SR: {sr_val:.2f}' for sr_val in z_flat]

    # Create the Mapbox heatmap
    heatmap = go.Scattermapbox(
        lat=lat_flat,
        lon=lon_flat,
        mode='markers',
        hovertext=hovertext,
        hoverinfo='text',
        marker=go.scattermapbox.Marker(
            size=8,  # Adjust the marker size to match grid dimensions
            color=z_flat,
            colorscale='RdYlBu',
            cmin=color_min,  # Set minimum color scale value
            cmax=color_max,  # Set maximum color scale value
            colorbar=dict(title='SR'),
            showscale=True,
            opacity=0.6,
        ),
    )

    # Layout with a mapbox background
    layout = go.Layout(
        title={'text': f'Scatter Mapbox Plot for {folder_name}', 
               'x': 0.5, 'xanchor': 'center', 'font': {'size': 30}},
        mapbox=dict(
            style='basic',
            center=dict(lat=(min_lat + max_lat) / 2, lon=(min_lon + max_lon) / 2),
            zoom=6.75,  # Adjust zoom level based on your region
        ),
        height=1000,
        width=1000
    )

    fig = go.Figure(data=[heatmap], layout=layout)

    # Add your Mapbox access token here
    mapbox_access_token = 'pk.eyJ1Ijoib2xlZXZjYSIsImEiOiJjbTFldmt6aGIyeWN4MmxzamFrYTV3dTNxIn0.bbVpqBfsIl_Y0W7YGRXCgQ'
    fig.update_layout(mapbox_accesstoken=mapbox_access_token)
    fig.show()
    
    if saveplot:
        fig.write_html(f'plotting/plots/{folder_name}.html')

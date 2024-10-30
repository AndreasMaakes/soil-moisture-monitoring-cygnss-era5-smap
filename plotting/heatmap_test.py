import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy as np
from import_data import importData 

def rasterized_heatmap(folder_name, saveplot, sigma):
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

    # Create a high-resolution grid
    lat_step = 0.05  # Higher resolution grid step
    lon_step = 0.05
    lat_grid = np.arange(min_lat, max_lat, lat_step)
    lon_grid = np.arange(min_lon, max_lon, lon_step)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Interpolating the data onto the grid
    grid_values = griddata((lats, lons), srs, (lat_grid, lon_grid), method='nearest')

    # Apply Gaussian blur to smooth the data
    grid_values_blurred = gaussian_filter(grid_values, sigma=sigma)

    # Create the Contour plot
    heatmap = go.Contour(
        z=grid_values_blurred,
        x=lon_grid[0],  # Longitude values
        y=lat_grid[:, 0],  # Latitude values
        colorscale='RdYlBu',
        colorbar=dict(title='SR'),
        contours=dict(
            coloring='heatmap',  # Use solid coloring for heatmap effect
            showlabels=False  # Hide contour labels for cleaner appearance
        ),
    )

    # Create a layout
    layout = go.Layout(
        title={'text': f'Rasterized Heatmap for {folder_name}', 
               'x': 0.5, 'xanchor': 'center', 'font': {'size': 30}},
        xaxis=dict(title='Longitude', showgrid=False),
        yaxis=dict(title='Latitude', showgrid=False),
        height=800,
        width=1200
    )

    fig = go.Figure(data=[heatmap], layout=layout)

    fig.show()
    
    if saveplot:
        fig.write_html(f'plotting/plots/{folder_name}_rasterized_heatmap.html')

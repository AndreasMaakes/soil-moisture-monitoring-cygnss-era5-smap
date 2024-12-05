import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy as np
from import_data import importData 

def gaussian_blur_plot(folder_name, saveplot, sigma):  # Add a sigma parameter to control blur level
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
            colorscale='RdYlBu',
            colorbar=dict(title='SR (dB)'),
            opacity=0.8,
        )
    )

    # Layout with Cartesian axes
    layout = go.Layout(
        title=f"Gaussian Blurred Scatter Plot (σ={sigma})",
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
        plot_bgcolor='white',  # Set the plot background to white
        paper_bgcolor='white'  # Set the outer paper background to white
    )

    fig = go.Figure(data=[scatter], layout=layout)

    fig.show()
    
    if saveplot:
        fig.write_html(f'plotting/plots/{folder_name}.html')

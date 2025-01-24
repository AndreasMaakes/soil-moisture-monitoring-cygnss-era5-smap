import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy as np
from CYGNSS.import_data import importData 

'''
This function creates a scatter plot of the surface reflectivity data with the possibility to apply a Gaussian blur to it.

Parameters:
- folder_name: The name of the folder containing the data to be plotted.
- saveplot: A boolean indicating whether the plot should be saved as an HTML file.
- sigma: The standard deviation of the Gaussian filter. If sigma is 0, no blur is applied.

Returns:
Nothing. The function displays the plot in the browser and saves it as an HTML file if saveplot is True.

'''

def CYGNSS_gaussian_blur_plot(folder_name, saveplot, sigma):
    
    title = ""
    name = folder_name.split("/")[0]

    '''Title of plot depends on the sigma value'''
    
    if(sigma == 0):
        title = f'CYGNSS Surface Reflectivity - {name} - September 2024'
    else:
        title = f'Smoothed CYGNSS Surface Reflectivity - {name} - September 2024 - σ={sigma}'

    '''Import data from the folder'''
    
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

    # Interpolating using nearest neighbor
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
            size=12,  # Adjust marker size
            color=z_flat,
            colorscale='portland_r', # Choose a colorscale
            colorbar=dict(title='SR (dB)'),
            opacity=1,
        )
    )

    # Layout with Cartesian axes and centered title
    layout = go.Layout(
        title=dict(
            text=title,
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


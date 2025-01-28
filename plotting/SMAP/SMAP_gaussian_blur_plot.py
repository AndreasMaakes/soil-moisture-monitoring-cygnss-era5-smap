import pandas as pd
import numpy as np

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from SMAP_import_data import importDataSMAP
from SMAP_utils import SMAP_averaging_soil_moisture




def SMAP_gaussian_blur_plot(folder_name, sigma):

    #Importing the data
    dataframes = importDataSMAP(folder_name)
    
    #Concating the dataframes to one single dataframe
    df = pd.concat(dataframes)
    #Averaging the soil moisture values
    df = SMAP_averaging_soil_moisture(df)
    
    lats = np.array(df['latitude']) 
    lons = np.array(df['longitude'])
    soil_moisture = np.array(df['soil_moisture_avg'])
    
    #Hovertext to show the soil moisture values
    hovertext = [f'SM: {sm:.2f}' for sm in soil_moisture]
    
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
            text="Hello",
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
    
 
    
SMAP_gaussian_blur_plot('Brazil', 1)
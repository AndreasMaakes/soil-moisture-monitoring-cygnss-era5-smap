import plotly.graph_objects as go
from scipy.interpolate import griddata
import numpy as np
from import_data import importData 

# Your data import and preparation remains the same
def trace_plot(folder_name, saveplot):
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

    

    hovertext = [f'SR: {sr_val:.2f}' for sr_val in srs]

    # Create the Mapbox heatmap
    #Color scales and syntax found here
    #https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scattermapbox.html

    heatmap = go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        hovertext=hovertext,
        hoverinfo = 'text',
        marker=go.scattermapbox.Marker(
            size=5,  
            colorscale='RdYlBu',
            color = srs,
            colorbar=dict(title='SR (dB)'),
            showscale=True,
            opacity=1,
        ),
    )

    # Layout with a mapbox background
    layout = go.Layout(
        
        mapbox=dict(
            style='satellite', #mapbox://styles/oleevca/cm20jbhca002t01qv2jxfe7sh is the custom map ole designed 
            center=dict(lat=(min_lat + max_lat) / 2, lon=(min_lon + max_lon) / 2),
            zoom=5.75,  # Adjust zoom level based on your region
        ),
        height=1000,
        width=1450, 
        font=dict(size=25)
    )

    fig = go.Figure(data=[heatmap], layout=layout) 

    # Add your Mapbox access token here
    mapbox_access_token = 'pk.eyJ1Ijoib2xlZXZjYSIsImEiOiJjbTFldmt6aGIyeWN4MmxzamFrYTV3dTNxIn0.bbVpqBfsIl_Y0W7YGRXCgQ'
    fig.update_layout(mapbox_accesstoken=mapbox_access_token)
    fig.show()
    
    if saveplot:
        fig.write_html(f'plotting/plots/{folder_name}.html')

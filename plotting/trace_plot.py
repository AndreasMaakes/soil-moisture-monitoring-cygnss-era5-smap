import plotly.graph_objects as go
import numpy as np
from import_data import importData

def trace_plot(folder_name, saveplot):

    name = folder_name.split("/")[0]
    title = f'Smoothed CYGNSS Surface Reflectivity - {name} - September 2024'

    # Import data
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

    # Create a scatter plot
    scatter = go.Scatter(
        x=lons,
        y=lats,
        mode='markers',
        text=hovertext,
        hoverinfo='text',
        marker=dict(
            size=5,
            color=srs,
            colorscale='rdylbu',
            colorbar=dict(title='SR (dB)'),
            opacity=1,
        )
    )

    # Layout with Cartesian axes
    layout = go.Layout(
        title=title,
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

    if saveplot:
        fig.write_html(f'plotting/plots/{folder_name}.html')

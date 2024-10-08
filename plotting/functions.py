import plotly.graph_objects as go
import pandas as pd
import xarray as xr
import plotly.express as px
import os

#Function to import data from the data folder
def importData():
    '''Data folder'''
    basePath = "../Prosjektoppgave/data/Chad-20230612-20230613"
    dataFrames = []
    for file in os.listdir(basePath):
        filePath = os.path.join(basePath, file)
        ds = xr.open_dataset(filePath, engine='netcdf4')
        df = ds.to_dataframe().reset_index()
        df.name = file #adding name of file to dataframe
        dataFrames.append(df)
    return dataFrames



#Function to create traces to plot and visualize variables from CYGNSS data
def createTraces(dataFrames):
    traces = []
    colors = ['blue', 'red', 'green']
    for df in dataFrames:
        name = df.name
        if name.endswith('11.nc'):
            color = colors[0]
        elif name.endswith('12.nc'):
            color = colors[1]
        else:
            color = colors[2]
        trace = go.Scattermapbox(
            lon = df['sp_lon'],
            lat = df['sp_lat'],
            text = df['ddm_snr'],
            marker = dict(
                color = color,
            ),
            name = df.name
        )
        traces.append(trace)
    return traces



import plotly.graph_objects as go
import pandas as pd
import xarray as xr
import plotly.express as px
import os

#Function to import data
def importData():
    '''Data folder'''
    basePath = "../Prosjektoppgave/data/"
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
    for df in dataFrames:
        trace = go.Scattermapbox(
            lon = df['sp_lon'],
            lat = df['sp_lat'],
            text = df['ddm_snr'],
            marker = dict(
                color = 'blue',
            ),
            name = df.name
        )
        traces.append(trace)
    return traces



import xarray as xr
import os

#Function to import data from the data folder
def importData(folder_name):
    '''Data folder'''
    basePath = f'../Prosjektoppgave/data/{folder_name}'
    dataFrames = []
    for file in os.listdir(basePath):
        filePath = os.path.join(basePath, file)
        ds = xr.open_dataset(filePath, engine='netcdf4')
        df = ds.to_dataframe().reset_index()
        df.name = file #adding name of file to dataframe
        dataFrames.append(df)
    return dataFrames

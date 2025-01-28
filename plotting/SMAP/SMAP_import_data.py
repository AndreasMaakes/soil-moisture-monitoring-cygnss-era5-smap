import os
import xarray as xr

'''Function to import data from the data folder, using a given folder name. It returns an array of dataframes with the data in the folders.'''
def importDataSMAP(folder_name):
    
    '''Data folder'''
    basePath = f'data/SMAP/{folder_name}'
    dataFrames = []
    
    '''Iterating through all files in the folder, and adding the data to the dataFrames array'''
    for file in os.listdir(basePath):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath, file)
            ds = xr.open_dataset(filePath, engine='netcdf4') #Opening the file using xarray
            df = ds.to_dataframe().reset_index()
            df.name = file #Adding name of file to dataframe
            dataFrames.append(df)
    return dataFrames


import xarray as xr
import os

#Function to adjust SR. Takes in an array of dataframes and subtracts the min_sr value from the sr column
def adjustSR(dataFrames, min_sr):
    for df in dataFrames:
        df["sr"] = df["sr"] - min_sr
    
#Function to import data from the data folder
def importData(folder_name):
    '''Data folder'''
    basePath = f'../Prosjektoppgave/data/{folder_name}'
    dataFrames = []
    #Subtracting the lowest value from the SR to get more intuitive plots
    min_sr = 100000
    for file in os.listdir(basePath):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath, file)
            ds = xr.open_dataset(filePath, engine='netcdf4')
            df = ds.to_dataframe().reset_index()
            min_sr = min(min_sr, df["sr"].min())
            df.name = file #adding name of file to dataframe
            dataFrames.append(df)
    #Running the adjustSR function
    adjustSR(dataFrames, min_sr)
    return dataFrames

import xarray as xr
import os

'''
This function is used to adjust the SR values of the data to make the plots more intuitive.
Takes in an array of dataframes and subtracts the min_sr value from the sr column in each dataframe.
'''
def adjustSR(dataFrames, min_sr):
    for df in dataFrames:
        df["sr"] = df["sr"] - min_sr
    
'''Function to import data from the data folder, using a given folder name. It returns an array of dataframes with the data in the folders.'''
def importData(folder_name):
    
    '''Data folder'''
    basePath = f'../Prosjektoppgave/data/{folder_name}'
    
    dataFrames = []
    '''Min SR is set to something ridiculously high to make sure that the first value is always lower'''    
    min_sr = 100000
    
    '''Iterating through all files in the folder, and adding the data to the dataFrames array'''
    for file in os.listdir(basePath):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath, file)
            ds = xr.open_dataset(filePath, engine='netcdf4') #Opening the file using xarray
            df = ds.to_dataframe().reset_index()
            min_sr = min(min_sr, df["sr"].min()) #Finding the minimum sr value of all the data
            df.name = file #Adding name of file to dataframe
            dataFrames.append(df)
    '''Running the adjustSR function'''
    adjustSR(dataFrames, min_sr)
    return dataFrames

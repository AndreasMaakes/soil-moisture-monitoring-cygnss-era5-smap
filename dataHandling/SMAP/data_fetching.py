import earthaccess
import xarray as xr
import numpy as np
import h5py
import pandas as pd
from .data_filtering import data_filtering_SMAP
import os
from datetime import datetime, timedelta

'''
This function creates a time and date array to easier handle the dates in the data fetching function
'''
def create_dates_array_SMAP(startDate: str, endDate: str):
    # Convert string dates to datetime objects. This uses this format: "2019-02-01"
    start = datetime.strptime(startDate, "%Y-%m-%d")
    end = datetime.strptime(endDate, "%Y-%m-%d")
    
    # Generate all dates in the range
    dates = []
    current_date = start
    while current_date <= end:
        # Append the date in the desired format
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return dates


'''
Function to fetch SMAP data

API date format: "2019-02-01" - "2019-02-02"

'''
def data_fetching_smap(start_date: str, end_date: str, max_lat: float, min_lat: float, max_lon: float, min_lon: float, name: str):
    #Log into earthaccess using username and password stored in .evn file
    earthaccess.login()

    #Searching for the results using earthaccess API. 
    results = earthaccess.search_data(
        short_name='SPL3SMP', #L3 36 km gridded SMAP short name
        temporal=(start_date, end_date), #Temporal filter
        count=-1,
        provider="NSIDC_CPRD" #Specifying the cloud based provider
        
    )
    #Opening the result - this yields a list of HTTP file system objects 
    dataset = earthaccess.open(results)
    
    #Extracting the year, month and days from the start and end date
    dates = create_dates_array_SMAP(start_date, end_date)
    count = 0
    #Iterating through the objects
    for ds in dataset:
        print(f"Processing file: {ds}")
        with h5py.File(ds, 'r') as f:
            #Fetching both AM and PM data
            group_AM = f['Soil_Moisture_Retrieval_Data_AM']
            group_PM = f['Soil_Moisture_Retrieval_Data_PM']
            
            #Fetching variables from AM group
            latitude_AM = group_AM['latitude'][...]  
            longitude_AM = group_AM['longitude'][...]
            soil_moisture_AM= group_AM['soil_moisture'][...]
            soil_moisture_dca_AM = group_AM['soil_moisture_dca'][...]
            
            #Fetching variables from PM group
            latitude_PM = group_PM['latitude_pm'][...]
            longitude_PM = group_PM['longitude_pm'][...]
            soil_moisture_PM = group_PM['soil_moisture_pm'][...]
            soil_moisture_dca_PM= group_PM['soil_moisture_dca_pm'][...]
            
            
            # Flatten the arrays - is this necessary? 
            latitude_AM = latitude_AM.flatten()
            longitude_AM = longitude_AM.flatten()
            soil_moisture_AM = soil_moisture_AM.flatten()
            soil_moisture_dca_AM = soil_moisture_dca_AM.flatten()
            
            latitude_PM = latitude_PM.flatten()
            longitude_PM = longitude_PM.flatten()
            soil_moisture_PM = soil_moisture_PM.flatten()
            soil_moisture_dca_PM = soil_moisture_dca_PM.flatten()

            #Creating two seperate dataframes for the AM and PM groups
            df_AM = pd.DataFrame({
                'latitude': latitude_AM, 
                'longitude': longitude_AM,
                'soil_moisture': soil_moisture_AM,
                'soil_moisture_dca': soil_moisture_dca_AM
            })
            
            df_PM = pd.DataFrame({
                'latitude': latitude_PM, 
                'longitude': longitude_PM,
                'soil_moisture': soil_moisture_PM,
                'soil_moisture_dca': soil_moisture_dca_PM
            })
            
            #Combing the two dataframes into one
            df_combined = pd.concat([df_AM, df_PM])
            
            #Filtering the data based on the spatial filter and removing NaN/filler values
            df_filtered = data_filtering_SMAP(df_combined, max_lat, min_lat, max_lon, min_lon).reset_index(drop=True)
            
            
            '''Define the base path for the data directory'''
            base_data_path = "data/SMAP"
            
            '''Create or locate the area-specific folder'''
            area_folder_path = os.path.join(base_data_path, name)
            if not os.path.exists(area_folder_path):
                try:
                    os.mkdir(area_folder_path)
                    print(f"Directory {area_folder_path} created successfully.")
                except OSError:
                    print(f"Creation of the directory {area_folder_path} failed.")
            else:
                print(f"Directory {area_folder_path} already exists.")

            '''Create a subfolder for this specific run inside the area-specific folder. Name it after the area and the date'''
            
            file_name = f"SMAP_{name}_{dates[count]}"
            count += 1
            
            ds = xr.Dataset.from_dataframe(df_filtered)
            ds.to_netcdf(os.path.join(area_folder_path, file_name + ".nc"))
            
            print(f"File {file_name} created successfully.")
            
            
import earthaccess
import xarray as xr
import numpy as np
import h5py
import pandas as pd
from .data_filtering import data_filtering_SMAP
import os
from datetime import datetime, timedelta
from create_dates_array import create_dates_array

def data_fetching_smap(Timeseries: bool, startDate: str, endDate: str, max_lat: float, min_lat: float, max_lon: float, min_lon: float, name: str):
    # Log into earthaccess using username and password stored in .evn file
    earthaccess.login()
    
    # Extracting the year, month, and days from the start and end date
    dates = create_dates_array(startDate, endDate, "smap")
    
    # Searching for the results using earthaccess API
    results = earthaccess.search_data(
        short_name='SPL3SMP',  # L3 36 km gridded SMAP short name
        temporal=(dates[0], dates[1]),  # Temporal filter
        count=-1,
        provider="NSIDC_CPRD"  # Specifying the cloud-based provider
    )
    
    # Opening the result - this yields a list of HTTP file system objects
    dataset = earthaccess.open(results)
    
    # Counting files processed
    count = 0
    
    df_timeseries = pd.DataFrame({})
    
    # Iterating through the objects
    for ds in dataset:
        print(f"Processing file: {ds}")
        with h5py.File(ds, 'r') as f:
            # Fetching both AM and PM data
            group_AM = f['Soil_Moisture_Retrieval_Data_AM']
            group_PM = f['Soil_Moisture_Retrieval_Data_PM']
            
            # Fetching variables from AM group
            latitude_AM = group_AM['latitude'][...].flatten()
            longitude_AM = group_AM['longitude'][...].flatten()
            soil_moisture_AM = group_AM['soil_moisture'][...].flatten()
            soil_moisture_dca_AM = group_AM['soil_moisture_dca'][...].flatten()
            
            # Fetching variables from PM group
            latitude_PM = group_PM['latitude_pm'][...].flatten()
            longitude_PM = group_PM['longitude_pm'][...].flatten()
            soil_moisture_PM = group_PM['soil_moisture_pm'][...].flatten()
            soil_moisture_dca_PM = group_PM['soil_moisture_dca_pm'][...].flatten()
            
            # Creating two separate dataframes for the AM and PM groups
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
            
            # Combining the two dataframes into one
            df_combined = pd.concat([df_AM, df_PM])
            
            # Filtering the data based on the spatial filter and removing NaN/filler values
            df_filtered = data_filtering_SMAP(df_combined, max_lat, min_lat, max_lon, min_lon).reset_index(drop=True)
            
            # If the run is part of a timeseries, concatenate the dataframes and return it for further processing
            if Timeseries:
                df_timeseries = pd.concat([df_timeseries, df_filtered])
            else:
                # Define base data path
                base_data_path = "data/SMAP"
                
                # Create or locate the area-specific folder
                area_folder_path = os.path.join(base_data_path, name)
                if not os.path.exists(area_folder_path):
                    try:
                        os.mkdir(area_folder_path)
                        print(f"Directory {area_folder_path} created successfully.")
                    except OSError:
                        print(f"Creation of the directory {area_folder_path} failed.")
                else:
                    print(f"Directory {area_folder_path} already exists.")
                
                # Create a subfolder for this specific run inside the area-specific folder
                file_name = f'{name}_{dates[count]}'
                folder_path = os.path.join(area_folder_path, file_name)
                
                if not os.path.exists(folder_path):
                    try:
                        os.mkdir(folder_path)
                        print(f"Directory {folder_path} created successfully.")
                    except OSError:
                        print(f"Creation of the directory {folder_path} failed.")
                else:
                    print(f"Directory {folder_path} already exists.")
                
                # Convert dataframe to xarray dataset and save to NetCDF
                ds_xr = xr.Dataset.from_dataframe(df_filtered)
                ds_xr.to_netcdf(os.path.join(folder_path, file_name + ".nc"))
                print(f"File {file_name} created successfully.")
                
            count += 1
    
    if Timeseries:
        df_timeseries = df_timeseries.reset_index(drop=True)
        return df_timeseries

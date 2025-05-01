import time
import earthaccess
import xarray as xr
import numpy as np
import h5py
import pandas as pd
from .data_filtering import data_filtering_SMAP
import os
from datetime import datetime, timedelta
from create_dates_array import create_dates_array
from dotenv import load_dotenv
from requests.exceptions import HTTPError

load_dotenv()


def data_fetching_smap(Timeseries: bool, startDate: str, endDate: str, max_lat: float, min_lat: float, max_lon: float, min_lon: float, name: str):
    """
    Fetches SMAP data using Earthaccess with a 5-minute wait between retries if a timeout occurs.
    
    :param Timeseries: Boolean, if True, returns a combined DataFrame for all time intervals.
    :param startDate: Start date in "YYYYMMDD" format.
    :param endDate: End date in "YYYYMMDD" format.
    :param max_lat: Maximum latitude.
    :param min_lat: Minimum latitude.
    :param max_lon: Maximum longitude.
    :param min_lon: Minimum longitude.
    :param name: Name identifier for saving the files.
    :param max_retries: Maximum number of retry attempts in case of a failure.
    :param wait_time: Wait time in seconds before retrying (default: 300 seconds / 5 minutes).
    :return: DataFrame if Timeseries is True, otherwise None.
    """

    # Log into Earthaccess using credentials from .env file
    earthaccess.login(strategy="environment")
    
    # Extracting the year, month, and days from the start and end date
    dates = create_dates_array(startDate, endDate, "smap")
    
    # Searching for the results using earthaccess API
    results = earthaccess.search_data(
        short_name='SPL3SMP_E',  # L3 36 km gridded SMAP short name SPL3SMP
        temporal=(dates[0], dates[-1]),  # Temporal filter
        count=-1,
        provider="NSIDC_CPRD"  # Specifying the cloud-based provider
    )
    
    # Opening the result - this yields a list of HTTP file system objects
    dataset = earthaccess.open(results)
    
    # Counting files processed
    count = 0
    df_timeseries = pd.DataFrame({})

    for ds in dataset:
        print(f"Processing file: {ds}")
        with h5py.File(ds, 'r') as f:
            group_AM = f['Soil_Moisture_Retrieval_Data_AM']
            group_PM = f['Soil_Moisture_Retrieval_Data_PM']

            latitude_AM = group_AM['latitude'][...].flatten()
            longitude_AM = group_AM['longitude'][...].flatten()
            soil_moisture_AM = group_AM['soil_moisture'][...].flatten()
            soil_moisture_dca_AM = group_AM['soil_moisture_dca'][...].flatten()

            latitude_PM = group_PM['latitude_pm'][...].flatten()
            longitude_PM = group_PM['longitude_pm'][...].flatten()
            soil_moisture_PM = group_PM['soil_moisture_pm'][...].flatten()
            soil_moisture_dca_PM = group_PM['soil_moisture_dca_pm'][...].flatten()

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

            df_combined = pd.concat([df_AM, df_PM])
            df_filtered = data_filtering_SMAP(df_combined, max_lat, min_lat, max_lon, min_lon).reset_index(drop=True)

            if Timeseries:
                df_timeseries = pd.concat([df_timeseries, df_filtered])
            else:
                base_data_path = "data/SMAP"
                area_folder_path = os.path.join(base_data_path, name)
                if not os.path.exists(area_folder_path):
                    os.mkdir(area_folder_path)

                file_name = f'{name}_{dates[count]}'
                folder_path = os.path.join(area_folder_path, file_name)

                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

                ds_xr = xr.Dataset.from_dataframe(df_filtered)
                ds_xr.to_netcdf(os.path.join(folder_path, file_name + ".nc"))
                print(f"File {file_name} created successfully.")

            count += 1

    if Timeseries:
        return df_timeseries.reset_index(drop=True)


#data_fetching_smap(False, "20200101", "20200131",  28.5, 25, 73, 67, "Paki_smap_9km")

import pandas as pd
import matplotlib.pyplot as plt
import os
import xarray as xr
import datetime
from SMAP.SMAP_import_data import importDataSMAP
import numpy as np


def plot_time_series(folder_name, min_lat, min_lon, max_lat, max_lon):
    '''Data folder paths'''
    basePath_ERA5 = f'{folder_name}/ERA5'
    basePath_SMAP = f'{folder_name}/SMAP'
    basePath_CYGNSS = f'{folder_name}/CYGNSS'
    
    # Lists to store weekly average values and corresponding weeks
    avg_moisture_values_ERA5 = []
    avg_moisture_values_SMAP = []
    avg_moisture_values_CYGNSS = []
    min_sr = 10000
    week_count = 0

    # Loop over CYGNSS files 
    for file in os.listdir(basePath_CYGNSS):
        if file.endswith(".nc"):
            # Open the NetCDF file using xarray
            filePath = os.path.join(basePath_CYGNSS, file)
            ds = xr.open_dataset(filePath, engine='netcdf4')
            df = ds.to_dataframe().reset_index()
            
            # Geospatial filtering
            df_filtered = df[
                (df["sp_lat"] >= min_lat) & (df["sp_lat"] <= max_lat) &
                (df["sp_lon"] >= min_lon) & (df["sp_lon"] <= max_lon)
            ]
            # Filtering ddm_snr and sp_rx_gain
            df_filtered = df_filtered[df_filtered['ddm_snr'] >= 2]
            df_filtered = df_filtered[df_filtered['sp_rx_gain'] >= 13]
            
            # Adjust sr values
            min_sr = min(min_sr, df_filtered["sr"].min())  # Find the minimum sr value among all data
            df_filtered["sr"] = df_filtered["sr"] - min_sr

            # Append the weekly average and update week count
            avg_moisture_values_CYGNSS.append(df_filtered["sr"].mean())
            week_count += 1

    # Convert the CYGNSS values to a NumPy array for boolean indexing
    avg_moisture_values_CYGNSS = np.array(avg_moisture_values_CYGNSS)
    valid = ~np.isnan(avg_moisture_values_CYGNSS)
    avg_moisture_values_CYGNSS = avg_moisture_values_CYGNSS[valid]

    # Create the weeks array and filter it to match valid indices
    weeks = np.arange(1, week_count + 1)
    weeks = weeks[valid]
    
    print(avg_moisture_values_CYGNSS)
    
    # Loop over ERA5 files
    for file in os.listdir(basePath_ERA5):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath_ERA5, file)
            ds = xr.open_dataset(filePath, engine='netcdf4')
            df = ds.to_dataframe().reset_index()
            
            # Check if expected column exists
            if "swvl1" in df.columns:
                # Spatial filtering
                df_filtered = df[
                    (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
                    (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
                ]
                avg_moisture = df_filtered["swvl1"].mean()  # Compute mean soil moisture
                avg_moisture_values_ERA5.append(avg_moisture)
    
    #Loop over SMAP files
    for file in os.listdir(basePath_SMAP):
        if file.endswith(".nc"):
            filePath = os.path.join(basePath_SMAP, file)
            ds = xr.open_dataset(filePath, engine='netcdf4')
            df = ds.to_dataframe().reset_index()
            
            
            # Spatial filtering
            df_filtered = df[
                (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat) &
                (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
            ]
            avg_moisture = df_filtered["soil_moisture"].mean()
            avg_moisture_values_SMAP.append(avg_moisture)
      
    # Parse folder name to extract start and end dates
    parts = folder_name.split('-')
    start_date_str = parts[2]
    end_date_str = parts[3]

    start_date = pd.to_datetime(start_date_str, format="%Y%m%d")
    end_date = pd.to_datetime(end_date_str, format="%Y%m%d")

    # Generate month start dates and labels
    month_starts = pd.date_range(start=start_date, end=end_date, freq='MS')
    month_labels = month_starts.strftime('%b')
    tick_positions = [(m - start_date).days // 7 + 1 for m in month_starts]

    # Create figure and primary y-axis for ERA5 (and optionally SMAP) data
    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Plot ERA5 data; note that we assume ERA5 values align with week numbers 1, 2, ...
    ax1.plot(np.arange(1, len(avg_moisture_values_ERA5) + 1), avg_moisture_values_ERA5,
              linestyle='-', color='g', label="ERA5")
     
    ax1.plot(np.arange(1, len(avg_moisture_values_SMAP) + 1), avg_moisture_values_SMAP,
              linestyle='-', color='r', label="SMAP")
    
    ax1.set_xlabel("Week Start Date")
    ax1.set_ylabel("ERA5 & SMAP Soil Moisture (0-1)")
    
    # Create secondary y-axis for CYGNSS data (native scale)
    ax2 = ax1.twinx()
    ax2.plot(weeks, avg_moisture_values_CYGNSS,  linestyle='-', color='b', label="CYGNSS")
    ax2.set_ylabel("CYGNSS Soil Moisture")
    
   
    # Add legends, title, and grid
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Soil Moisture Time Series (ERA5 and CYGNSS)")
    plt.xticks(tick_positions, month_labels, rotation=45)
    ax1.grid(True)
    plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import os
import xarray as xr
import datetime
from SMAP.SMAP_import_data import importDataSMAP
import numpy as np

def plot_time_series(folder_name, min_lat, max_lat, min_lon, max_lon):
    '''Data folder paths'''
    basePath_ERA5 = f'{folder_name}/ERA5'
    basePath_SMAP = f'{folder_name}/SMAP'
    
    # Lists to store weekly average values and corresponding weeks
    weeks = []
    avg_moisture_values_ERA5 = []
    avg_moisture_values_SMAP = []
    # Example CYGNSS values on a different scale (e.g., not 0–1)
    avg_moisture_values_CYGNSS = [4, 6, 3, 7, 12, 3, 6]

    # Loop over ERA5 files
    for file in os.listdir(basePath_ERA5):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath_ERA5, file)
            # Extract date from filename (assuming format like "ERA5_20240101_20240102.nc")
            date_str = file.split("_")[1]
            first_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            
            # Open the NetCDF file using xarray
            ds = xr.open_dataset(filePath, engine='netcdf4')
            df = ds.to_dataframe().reset_index()
            
            # Ensure the expected column exists
            if "swvl1" in df.columns:
                # Filter data spatially
                df_filtered = df[
                    (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
                    (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
                ]
                avg_moisture = df_filtered["swvl1"].mean()  # Compute mean soil moisture
                weeks.append(first_date)
                avg_moisture_values_ERA5.append(avg_moisture)
            else:
                print(f"Warning: 'swvl1' column not found in {file}")
    
    # Import SMAP data
    smap_dfs = importDataSMAP(True, basePath_SMAP)
    for df in smap_dfs:
        if 'soil_moisture' in df.columns:
            df_filtered = df[
                (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
                (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
            ]
            avg_moisture = df_filtered["soil_moisture"].mean()
            avg_moisture_values_SMAP.append(avg_moisture)
        else:
            print(f'Warning: "soil_moisture" column not found in {df.name}')
    
    # Create figure and primary y-axis for ERA5 and SMAP data
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(weeks, avg_moisture_values_SMAP, marker='s', linestyle='-', color='r', label="SMAP")
    ax1.plot(weeks, avg_moisture_values_ERA5, marker='^', linestyle='-', color='g', label="ERA5")
    ax1.set_xlabel("Week Start Date")
    ax1.set_ylabel("ERA5 & SMAP Soil Moisture (0-1)")
    
    # Create secondary y-axis for CYGNSS data (native scale)
    ax2 = ax1.twinx()
    ax2.plot(weeks, avg_moisture_values_CYGNSS, marker='o', linestyle='-', color='b', label="CYGNSS")
    ax2.set_ylabel("CYGNSS Soil Moisture (Original Scale)")
    
    # Add legends, title, and grid
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Soil Moisture Time Series (Multiple Sources)")
    plt.xticks(rotation=45)
    ax1.grid(True)
    plt.show()




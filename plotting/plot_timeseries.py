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
    avg_moisture_values_CYGNSS = [4, 6, 3, 7, 12, 3, 6]  # Example CYGNSS values

    # Dictionary to align SMAP data with ERA5 weeks
    smap_moisture_dict = {}

    # Loop over ERA5 files to extract weeks
    for file in os.listdir(basePath_ERA5):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath_ERA5, file)
            date_str = file.split("_")[1]
            first_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            
            # Open the NetCDF file using xarray
            ds = xr.open_dataset(filePath, engine='netcdf4')
            df = ds.to_dataframe().reset_index()
            
            if "swvl1" in df.columns:
                df_filtered = df[
                    (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
                    (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
                ]
                avg_moisture = df_filtered["swvl1"].mean()  
                weeks.append(first_date)
                avg_moisture_values_ERA5.append(avg_moisture)
            else:
                print(f"Warning: 'swvl1' column not found in {file}")

    # Import SMAP data and align with weeks
    smap_dfs = importDataSMAP(True, basePath_SMAP)
    for df in smap_dfs:
        if 'soil_moisture' in df.columns:
            df_filtered = df[
                (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
                (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
            ]
            avg_moisture = df_filtered["soil_moisture"].mean()
            file_date = datetime.datetime.strptime(df.name.split("_")[1], "%Y%m%d").date()
            smap_moisture_dict[file_date] = avg_moisture
        else:
            print(f'Warning: "soil_moisture" column not found in {df.name}')

    # Ensure SMAP values match ERA5 weeks (fill missing weeks with NaN)
    for week in weeks:
        avg_moisture_values_SMAP.append(smap_moisture_dict.get(week, np.nan))

    # Create figure and primary y-axis for ERA5 and SMAP data
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(weeks, avg_moisture_values_SMAP, marker='s', linestyle='-', color='r', label="SMAP")
    ax1.plot(weeks, avg_moisture_values_ERA5, marker='^', linestyle='-', color='g', label="ERA5")
    ax1.set_xlabel("Week Start Date")
    ax1.set_ylabel("ERA5 & SMAP Soil Moisture (0-1)")

    # Create secondary y-axis for CYGNSS data (native scale)
    ax2 = ax1.twinx()
    ax2.plot(weeks[:len(avg_moisture_values_CYGNSS)], avg_moisture_values_CYGNSS, marker='o', linestyle='-', color='b', label="CYGNSS")
    ax2.set_ylabel("CYGNSS Soil Moisture (Original Scale)")

    # Add legends, title, and grid
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Soil Moisture Time Series (Multiple Sources)")
    plt.xticks(rotation=45)
    ax1.grid(True)
    plt.show()

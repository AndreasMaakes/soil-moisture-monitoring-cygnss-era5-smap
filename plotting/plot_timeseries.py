import pandas as pd
import matplotlib.pyplot as plt
import os
import xarray as xr
import numpy as np
import matplotlib.dates as mdates

def plot_time_series(folder_name, min_lat, min_lon, max_lat, max_lon):
    '''Data folder paths'''
    basePath_ERA5 = f'{folder_name}/ERA5'
    basePath_SMAP = f'{folder_name}/SMAP'
    basePath_CYGNSS = f'{folder_name}/CYGNSS'
    
    # We'll collect ERA5 and SMAP data in dataframes with date indexes
    era5_data = []
    smap_data = []
    cygnss_data = []

    # ========== ERA5 ========== 
    # (We still need real dates for ERA5 if you want them on the same date axis)
    # If your ERA5 files have no explicit time dimension, parse from filename, e.g.:
    # "ERA5_20220101.nc" -> date_str = "20220101"
    # Adjust as needed for your actual ERA5 filename structure.
    if os.path.exists(basePath_ERA5):
        for file in os.listdir(basePath_ERA5):
            if file.endswith(".nc"):
                filePath = os.path.join(basePath_ERA5, file)
                ds = xr.open_dataset(filePath, engine='netcdf4')
                
                # 1) Attempt to parse date from filename (adjust if your naming differs)
                base_name = os.path.splitext(file)[0]  # e.g. "ERA5_20220101"
                parts = base_name.split("_")           # ["ERA5", "20220101"]
                if len(parts) >= 2:
                    date_str = parts[1]               # "20220101"
                    date_val = pd.to_datetime(date_str, format="%Y%m%d")
                else:
                    # If no date in filename, you might try ds.time.values
                    # date_val = pd.to_datetime(ds.time.values[0])
                    raise ValueError(f"Cannot parse date from filename: {file}")

                df = ds.to_dataframe().reset_index()

                # 2) Spatial filtering
                df_filtered = df[
                    (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
                    (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
                ]

                # 3) Check if swvl1 is present
                if "swvl1" in df_filtered.columns:
                    avg_moisture = df_filtered["swvl1"].mean()
                    era5_data.append((date_val, avg_moisture))

    # ========== SMAP ==========
    if os.path.exists(basePath_SMAP):
        for file in os.listdir(basePath_SMAP):
            if file.endswith(".nc"):
                filePath = os.path.join(basePath_SMAP, file)

                # e.g. "SMAP_20220101_20220103.nc" -> parse the second part "20220101"
                base_name = os.path.splitext(file)[0]      # "SMAP_20220101_20220103"
                parts = base_name.split("_")              # ["SMAP", "20220101", "20220103"]
                date_str = parts[1]                       # "20220101"
                date_val = pd.to_datetime(date_str, format="%Y%m%d")

                ds = xr.open_dataset(filePath, engine='netcdf4')
                if ds.dims["index"] == 0:
                    print(f"Warning: {file} is empty. Skipping...")
                    continue

                df = ds.to_dataframe().reset_index()

                if "latitude" not in df.columns or "longitude" not in df.columns:
                    print(f"Warning: 'latitude' or 'longitude' not found in {file}. Skipping...")
                    continue

                # Spatial filtering
                df_filtered = df[
                    (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat) &
                    (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon)
                ]
                avg_moisture = df_filtered["soil_moisture"].mean()
                smap_data.append((date_val, avg_moisture))

    # ========== CYGNSS ==========
    if os.path.exists(basePath_CYGNSS):
        for file in os.listdir(basePath_CYGNSS):
            if file.endswith(".nc"):
                filePath = os.path.join(basePath_CYGNSS, file)
                
                # "CYGNSS_20220101_20220103.nc" -> parse second part "20220101"
                base_name = os.path.splitext(file)[0]     # "CYGNSS_20220101_20220103"
                parts = base_name.split("_")             # ["CYGNSS", "20220101", "20220103"]
                date_str = parts[1]                      # "20220101"
                date_val = pd.to_datetime(date_str, format="%Y%m%d")

                ds = xr.open_dataset(filePath, engine='netcdf4')
                df = ds.to_dataframe().reset_index()

                # Geospatial filtering
                df_filtered = df[
                    (df["sp_lat"] >= min_lat) & (df["sp_lat"] <= max_lat) &
                    (df["sp_lon"] >= min_lon) & (df["sp_lon"] <= max_lon)
                ]
                # Additional filter
                df_filtered = df_filtered[df_filtered['ddm_snr'] >= 4]

                # Optionally adjust sr by subtracting the file-level min
                # If you want a global min across *all* files, you'd have to do a 2-pass approach.
                min_sr_file = df_filtered["sr"].min()
                df_filtered["sr"] = df_filtered["sr"] - min_sr_file

                avg_sr = df_filtered["sr"].mean()
                cygnss_data.append((date_val, avg_sr))

    # ========== Convert to DataFrames ========== 
    df_era5 = pd.DataFrame(era5_data, columns=["date", "era5_moisture"])
    df_era5.set_index("date", inplace=True)
    df_era5.sort_index(inplace=True)

    df_smap = pd.DataFrame(smap_data, columns=["date", "smap_moisture"])
    df_smap.set_index("date", inplace=True)
    df_smap.sort_index(inplace=True)

    df_cygnss = pd.DataFrame(cygnss_data, columns=["date", "cygnss_sr"])
    df_cygnss.set_index("date", inplace=True)
    df_cygnss.sort_index(inplace=True)

    # ========== Plotting ==========
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot ERA5
    #if not df_era5.empty:
        #ax1.plot(df_era5.index, df_era5["era5_moisture"], color='g', label="ERA5", marker="o")

    # Plot SMAP
    if not df_smap.empty:
        ax1.plot(df_smap.index, df_smap["smap_moisture"], color='r', label="SMAP", marker="o")

    ax1.set_ylabel("ERA5 & SMAP Soil Moisture (0-1)")

    # Create a secondary y-axis for CYGNSS
    ax2 = ax1.twinx()
    if not df_cygnss.empty:
        ax2.plot(df_cygnss.index, df_cygnss["cygnss_sr"], color='b', label="CYGNSS", marker="o")
    ax2.set_ylabel("CYGNSS Soil Moisture")

    # Format x-axis as dates
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # every 3 months
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax1.set_title("Soil Moisture Time Series (ERA5, SMAP, CYGNSS)")
    ax1.grid(True)
    plt.show()

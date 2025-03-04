import pandas as pd
import matplotlib.pyplot as plt
import os
import xarray as xr
import numpy as np
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter1d  # Import gaussian_filter1d

def plot_time_series(folder_name, min_lat, min_lon, max_lat, max_lon, gaussian_sigma=0):
    '''Data folder paths'''
    basePath_ERA5 = f'{folder_name}/ERA5'
    basePath_SMAP = f'{folder_name}/SMAP'
    basePath_CYGNSS = f'{folder_name}/CYGNSS'
    
    # We'll collect ERA5 and SMAP data in dataframes with date indexes
    era5_data = []
    smap_data = []
    cygnss_data = []

    # ========== ERA5 ==========
    if os.path.exists(basePath_ERA5):
        for file in os.listdir(basePath_ERA5):
            if file.endswith(".nc"):
                filePath = os.path.join(basePath_ERA5, file)
                ds = xr.open_dataset(filePath, engine='netcdf4')
                
                base_name = os.path.splitext(file)[0]  # e.g. "ERA5_20220101"
                parts = base_name.split("_")           
                if len(parts) >= 2:
                    date_str = parts[1]               # "20220101"
                    date_val = pd.to_datetime(date_str, format="%Y%m%d")
                else:
                    raise ValueError(f"Cannot parse date from filename: {file}")

                df = ds.to_dataframe().reset_index()

                # Spatial filtering
                df_filtered = df[
                    (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
                    (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
                ]

                if "swvl1" in df_filtered.columns:
                    avg_moisture = df_filtered["swvl1"].mean()
                    era5_data.append((date_val, avg_moisture))

    # ========== SMAP ==========
    if os.path.exists(basePath_SMAP):
        for file in os.listdir(basePath_SMAP):
            if file.endswith(".nc"):
                filePath = os.path.join(basePath_SMAP, file)
                base_name = os.path.splitext(file)[0]      # e.g. "SMAP_20220101_20220103"
                parts = base_name.split("_")
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
                base_name = os.path.splitext(file)[0]     # e.g. "CYGNSS_20220101_20220103"
                parts = base_name.split("_")
                date_str = parts[1]                      # "20220101"
                date_val = pd.to_datetime(date_str, format="%Y%m%d")

                ds = xr.open_dataset(filePath, engine='netcdf4')
                df = ds.to_dataframe().reset_index()

                df_filtered = df[
                    (df["sp_lat"] >= min_lat) & (df["sp_lat"] <= max_lat) &
                    (df["sp_lon"] >= min_lon) & (df["sp_lon"] <= max_lon)
                ]
                df_filtered = df_filtered[df_filtered['ddm_snr'] >= 4]

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

    # ========== Apply Gaussian Blur ==========
    # The gaussian_sigma parameter controls the standard deviation of the blur.
    # A value of 0 means no blurring is applied.
    if gaussian_sigma > 0:
        if not df_era5.empty:
            df_era5["era5_moisture"] = gaussian_filter1d(df_era5["era5_moisture"].values, sigma=gaussian_sigma)
        if not df_smap.empty:
            df_smap["smap_moisture"] = gaussian_filter1d(df_smap["smap_moisture"].values, sigma=gaussian_sigma)
        if not df_cygnss.empty:
            df_cygnss["cygnss_sr"] = gaussian_filter1d(df_cygnss["cygnss_sr"].values, sigma=gaussian_sigma)

    # ========== Plotting ==========
    fig, ax1 = plt.subplots(figsize=(10, 5))

    if not df_era5.empty:
        ax1.plot(df_era5.index, df_era5["era5_moisture"], color='g', label="ERA5", marker="o")
    if not df_smap.empty:
        ax1.plot(df_smap.index, df_smap["smap_moisture"], color='r', label="SMAP", marker="o")
    ax1.set_ylabel("ERA5 & SMAP Soil Moisture (0-1)")

    ax2 = ax1.twinx()
    if not df_cygnss.empty:
        ax2.plot(df_cygnss.index, df_cygnss["cygnss_sr"], color='b', label="CYGNSS", marker="o")
    ax2.set_ylabel("CYGNSS Soil Moisture")

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax1.set_title("Soil Moisture Time Series (ERA5, SMAP, CYGNSS)")
    ax1.grid(True)
    plt.show()

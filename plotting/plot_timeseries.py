import pandas as pd
import matplotlib.pyplot as plt
import os
import xarray as xr
import numpy as np
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter1d  # Import gaussian_filter1d

def plot_time_series(folder_name, min_lat, min_lon, max_lat, max_lon, gaussian_sigma=0):
    '''Data folder paths'''
    basePath_CYGNSS = f'{folder_name}/CYGNSS'
    cygnss_data = []

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

    # Convert to DataFrame
    df_cygnss = pd.DataFrame(cygnss_data, columns=["date", "cygnss_sr"])
    df_cygnss.set_index("date", inplace=True)
    df_cygnss.sort_index(inplace=True)

    # Apply Gaussian Blur
    if gaussian_sigma > 0 and not df_cygnss.empty:
        df_cygnss["cygnss_sr"] = gaussian_filter1d(df_cygnss["cygnss_sr"].values, sigma=gaussian_sigma)

    # Plotting
    plt.figure(figsize=(10, 5))
    if not df_cygnss.empty:
        plt.plot(df_cygnss.index, df_cygnss["cygnss_sr"], color='b', label="CYGNSS", marker="o")
        plt.ylabel("CYGNSS Soil Moisture")
        plt.title("CYGNSS Soil Moisture Time Series")
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gcf().autofmt_xdate()
        plt.show()
    else:
        print("No CYGNSS data found for the given criteria.")

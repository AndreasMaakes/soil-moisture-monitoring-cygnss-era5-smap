from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from scipy.signal import correlate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import xarray as xr
import numpy as np

def time_series_correlation(folder_name, min_lat, min_lon, max_lat, max_lon, gaussian_sigma=0):
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

    

# ---- Interpolation for CYGNSS ----
    if not df_cygnss.empty:
        full_range = pd.date_range(df_cygnss.index.min(), df_cygnss.index.max(), freq='D')
        df_cygnss = df_cygnss.reindex(full_range)
        df_cygnss["cygnss_sr"] = df_cygnss["cygnss_sr"].interpolate(method='linear', limit_direction='both', limit_area='inside')

    # ---- Apply Gaussian smoothing (optional) ----
    if gaussian_sigma > 0:
        if not df_era5.empty:
            df_era5["era5_moisture"] = gaussian_filter1d(df_era5["era5_moisture"].values, sigma=gaussian_sigma)
        if not df_smap.empty:
            df_smap["smap_moisture"] = gaussian_filter1d(df_smap["smap_moisture"].values, sigma=gaussian_sigma)
        if not df_cygnss.empty:
            df_cygnss["cygnss_sr"] = gaussian_filter1d(df_cygnss["cygnss_sr"].values, sigma=gaussian_sigma)

    # ---- Merge for correlation ----
    df_merged = df_cygnss.join([df_smap, df_era5], how='inner')
    df_valid = df_merged.dropna()

    if df_valid.empty:
        print("Insufficient overlapping data for correlation analysis.")
        return

    # ---- Pearson Correlation ----
    r_cs, p_cs = pearsonr(df_valid["cygnss_sr"], df_valid["smap_moisture"])
    r_ce, p_ce = pearsonr(df_valid["cygnss_sr"], df_valid["era5_moisture"])
    r_se, p_se = pearsonr(df_valid["smap_moisture"], df_valid["era5_moisture"])
    print(f"Temporal Pearson Correlation:")
    print(f"  CYGNSS vs SMAP:  r = {r_cs:.2f}, p = {p_cs:.3e}")
    print(f"  CYGNSS vs ERA5:  r = {r_ce:.2f}, p = {p_ce:.3e}")
    print(f"  SMAP vs ERA5:    r = {r_se:.2f}, p = {p_se:.3e}")

    # ---- Rolling Correlation ----
    window = 14  # days
    rolling_corr_cs = df_valid['cygnss_sr'].rolling(window).corr(df_valid['smap_moisture'])
    rolling_corr_ce = df_valid['cygnss_sr'].rolling(window).corr(df_valid['era5_moisture'])

    plt.figure(figsize=(10, 4))
    plt.plot(df_valid.index, rolling_corr_cs, label='Rolling Corr: CYGNSS vs SMAP', color='purple')
    plt.plot(df_valid.index, rolling_corr_ce, label='Rolling Corr: CYGNSS vs ERA5', color='orange')
    plt.title("14-Day Rolling Pearson Correlation")
    plt.ylabel("Correlation Coefficient (r)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Cross-Correlation ----
    def compute_cross_corr(x, y):
        x = x - np.mean(x)
        y = y - np.mean(y)
        corr = correlate(x, y, mode='full')
        lags = np.arange(-len(x) + 1, len(y))
        corr /= (np.std(x) * np.std(y) * len(x))  # Normalize
        return lags, corr

    lags_cs, xcorr_cs = compute_cross_corr(df_valid['cygnss_sr'], df_valid['smap_moisture'])
    lags_ce, xcorr_ce = compute_cross_corr(df_valid['cygnss_sr'], df_valid['era5_moisture'])

    plt.figure(figsize=(10, 4))
    plt.plot(lags_cs, xcorr_cs, label="Cross-Corr: CYGNSS vs SMAP", color='blue')
    plt.plot(lags_ce, xcorr_ce, label="Cross-Corr: CYGNSS vs ERA5", color='green')
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("Lag (days)")
    plt.ylabel("Cross-Correlation")
    plt.title("Cross-Correlation with Lag")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import xarray as xr
import numpy as np
import glob

def time_series_correlation_with_ismn(folder_name, ismn_folder, gaussian_sigma=0):
    '''Data folder paths'''
    basePath_CYGNSS = f'{folder_name}/CYGNSS'

    cygnss_data = []

    # ========== CYGNSS ==========
    if os.path.exists(basePath_CYGNSS):
        for file in os.listdir(basePath_CYGNSS):
            if file.endswith(".nc"):
                filePath = os.path.join(basePath_CYGNSS, file)
                base_name = os.path.splitext(file)[0]
                parts = base_name.split("_")
                date_str = parts[1]
                date_val = pd.to_datetime(date_str, format="%Y%m%d")

                ds = xr.open_dataset(filePath, engine='netcdf4')
                df = ds.to_dataframe().reset_index()


                df_filtered = df[
                    (df['ddm_snr'] >= 4) &
                    (df['sp_rx_gain'] >= 0) &
                    (df['sp_rx_gain'] <= 13) &
                    (df['sp_inc_angle'] <= 45)
                ]

                if not df_filtered.empty:
                    min_sr_file = df_filtered["sr"].min()
                    df_filtered["sr"] = df_filtered["sr"] - min_sr_file
                    avg_sr = df_filtered["sr"].mean()
                    cygnss_data.append((date_val, avg_sr))

    df_cygnss = pd.DataFrame(cygnss_data, columns=["date", "cygnss_sr"]).set_index("date").sort_index()

    # ========== ISMN ==========
    file_list = glob.glob(os.path.join(ismn_folder, "*.stm"))
    weekly_series_list = []

    for file in file_list:
        df = pd.read_csv(file, delim_whitespace=True, skiprows=1, header=None,
                         names=["date", "time", "moisture", "unit1", "unit2"])
        df = df[df["moisture"] >= 0]
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M")
        df.set_index("datetime", inplace=True)
        weekly_avg = df["moisture"].resample("W").mean()
        weekly_series_list.append(weekly_avg)

    combined = pd.concat(weekly_series_list, axis=1)
    combined["mean_moisture"] = combined.mean(axis=1)
    df_ismn = combined[combined["mean_moisture"] <= 1][["mean_moisture"]]

    # ========== Time Alignment ==========
    if df_cygnss.empty or df_ismn.empty:
        print("CYGNSS or ISMN data is empty. Exiting.")
        return

    common_start = max(df_cygnss.index.min(), df_ismn.index.min())
    common_end = min(df_cygnss.index.max(), df_ismn.index.max())
    df_cygnss = df_cygnss[(df_cygnss.index >= common_start) & (df_cygnss.index <= common_end)]
    df_ismn = df_ismn[(df_ismn.index >= common_start) & (df_ismn.index <= common_end)]

    # Interpolate CYGNSS to match ISMN weekly resolution
    df_cygnss_interp = df_cygnss.resample("W").mean().interpolate(method="linear")

    # Combine datasets
    df_combined = df_ismn.join(df_cygnss_interp, how="inner").dropna()

    if df_combined.empty:
        print("No overlapping weekly data for ISMN and CYGNSS.")
        return

    # Apply Gaussian Smoothing (optional)
    if gaussian_sigma > 0:
        df_combined["mean_moisture"] = gaussian_filter1d(df_combined["mean_moisture"].values, sigma=gaussian_sigma)
        df_combined["cygnss_sr"] = gaussian_filter1d(df_combined["cygnss_sr"].values, sigma=gaussian_sigma)

    # ========== Correlation ==========
    r, p = pearsonr(df_combined["mean_moisture"], df_combined["cygnss_sr"])
    print(f"\nISMN vs CYGNSS Pearson correlation: r = {r:.3f}, p = {p:.3e}")

    # ========== Normalized RMSE ==========
    ismn_norm = (df_combined["mean_moisture"] - df_combined["mean_moisture"].mean()) / df_combined["mean_moisture"].std()
    cygnss_norm = (df_combined["cygnss_sr"] - df_combined["cygnss_sr"].mean()) / df_combined["cygnss_sr"].std()

    rmse = np.sqrt(np.mean((ismn_norm - cygnss_norm)**2))
    nrmse = rmse / (ismn_norm.max() - ismn_norm.min())

    print(f"RMSE (normalized): {rmse:.3f}")
    print(f"NRMSE: {nrmse:.3f}")

    # ========== Optional Scatter Plot ==========
    plt.figure(figsize=(6, 6))
    x = df_combined["mean_moisture"]
    y = df_combined["cygnss_sr"]

    # Scatter points
    plt.scatter(x, y, alpha=0.6, label="Data")

    # Regression line
    coeffs = np.polyfit(x, y, 1)  # Linear fit
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = coeffs[0] * x_fit + coeffs[1]
    plt.plot(x_fit, y_fit, color="red", linestyle="--", linewidth=2, label="Linear Fit")

    # Labels and styling
    plt.xlabel("ISMN Soil Moisture")
    plt.ylabel("CYGNSS SR (dB)")
    plt.title("Scatter Plot: ISMN vs CYGNSS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



time_series_correlation_with_ismn(
    folder_name="data\Timeseries\TimeSeries-Australia-20180801-20200801",
    ismn_folder="data/ISMN/Australia",
    gaussian_sigma=2
)

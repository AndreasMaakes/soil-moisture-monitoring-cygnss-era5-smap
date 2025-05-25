import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
import tqdm
from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from CYGNSS.import_data import importData
from plot_correlation import regrid_dataframe  # adjust import as needed


def composite_smap(dates, smap_folder, lat_bins, lon_bins):
    dfs = importDataSMAP(False, smap_folder)
    dates_smap = [d[:4] + '-' + d[4:6] + '-' + d[6:] for d in dates]
    dfs_sel = [df for df in dfs if any(ds in df.name for ds in dates_smap)]
    if not dfs_sel:
        return pd.DataFrame(columns=['lat_center', 'lon_center', 'soil_moisture_avg'])
    df_all = pd.concat(dfs_sel, ignore_index=True)
    df_avg = SMAP_averaging_soil_moisture(df_all)
    return regrid_dataframe(df_avg, lat_bins, lon_bins, "SMAP")


def composite_cygnss(dates, cygnss_folder, lat_bins, lon_bins):
    dfs = importData(cygnss_folder)
    dfs_sel = [df for df in dfs if any(date in df.name for date in dates)]
    if not dfs_sel:
        return pd.DataFrame(columns=['lat_center', 'lon_center', 'sr'])
    df_all = pd.concat(dfs_sel, ignore_index=True)
    df_all = df_all[df_all['ddm_snr'] >= 2]
    df_all = df_all[(df_all['sp_rx_gain'] >= 0) & (df_all['sp_rx_gain'] <= 13)]
    df_all = df_all[df_all['sp_inc_angle'] <= 45]
    return regrid_dataframe(df_all, lat_bins, lon_bins, "CYGNSS")


def smooth_dataframe(df, value_col, lat_bins, lon_bins):
    df = df.copy()
    df['lat_center'] = df['lat_center'].astype(float)
    df['lon_center'] = df['lon_center'].astype(float)

    grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
    lat_indices = ((df['lat_center'] - lat_bins[0]) / (lat_bins[1] - lat_bins[0])).astype(int)
    lon_indices = ((df['lon_center'] - lon_bins[0]) / (lon_bins[1] - lon_bins[0])).astype(int)

    for idx, val in zip(zip(lat_indices, lon_indices), df[value_col]):
        i, j = idx
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            grid[i, j] = val

    smoothed = gaussian_filter(grid, sigma=1.5, mode='nearest', truncate=1.0)

    smoothed_values = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            lat = lat_bins[i] + (lat_bins[1] - lat_bins[0]) / 2
            lon = lon_bins[j] + (lon_bins[1] - lon_bins[0]) / 2
            val = smoothed[i, j]
            if not np.isnan(val):
                smoothed_values.append({
                    'lat_center': round(lat, 4),
                    'lon_center': round(lon, 4),
                    value_col: val
                })

    return pd.DataFrame(smoothed_values)


def cumulative_correlations(smap_folder, cygnss_folder, date_list, lat_step, lon_step):
    all_smap = pd.concat(importDataSMAP(False, smap_folder), ignore_index=True)
    all_cyg  = pd.concat(importData(cygnss_folder), ignore_index=True)

    lat_min = min(all_smap['latitude'].min(), all_cyg['sp_lat'].min())
    lat_max = max(all_smap['latitude'].max(), all_cyg['sp_lat'].max())
    lon_min = min(all_smap['longitude'].min(), all_cyg['sp_lon'].min())
    lon_max = max(all_smap['longitude'].max(), all_cyg['sp_lon'].max())

    lat_bins = np.arange(lat_min, lat_max + lat_step, lat_step)
    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)

    results = []
    for k in tqdm.tqdm(range(1, len(date_list) + 1), desc="Cumulative windows", unit="day"):
        window_dates = date_list[:k]
        grid_cyg  = composite_cygnss(window_dates, cygnss_folder, lat_bins, lon_bins)
        grid_smap = composite_smap(window_dates, smap_folder, lat_bins, lon_bins)

        if grid_cyg.empty or grid_smap.empty:
            results.append((k, np.nan))
            continue

        #grid_cyg = smooth_dataframe(grid_cyg, 'sr', lat_bins, lon_bins)  # Disable smoothing for comparison
        grid_smap['lat_center'] = grid_smap['lat_center'].round(4)
        grid_smap['lon_center'] = grid_smap['lon_center'].round(4)

        df = pd.merge(
            grid_cyg, grid_smap,
            on=['lat_center', 'lon_center'],
            suffixes=('_cygnss', '_smap')
        ).dropna(subset=['sr', 'soil_moisture_avg'])

        r = pearsonr(df['sr'], df['soil_moisture_avg'])[0] if not df.empty else np.nan
        results.append((k, r))

    sorted_by_r = sorted(results, key=lambda x: (np.nan_to_num(x[1], nan=-np.inf), x[0]))
    bottom3 = sorted_by_r[:3]
    top3    = sorted_by_r[-3:]

    bottom3_fmt = [(k, round(float(r), 3)) for k, r in bottom3]
    top3_fmt    = [(k, round(float(r), 3)) for k, r in top3]

    print("Bottom 3 window sizes (days) and r:", bottom3_fmt)
    print("Top 3 window sizes (days) and r:", top3_fmt)
    return results



from datetime import datetime, timedelta

# Define your two-month period here
start = datetime(2020, 1, 1)
num_days = 31  # two months
dates = [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range(num_days)]

# Parameters
SMAP_FOLDER   = 'Western-Australia_20200101_20200131'
CYGNSS_FOLDER = 'Western-Australia/Western-Australia-20200101-20200131'
#SMAP_FOLDER   = 'Pakistan_20200101_20200131'
#CYGNSS_FOLDER = 'Pakistan/Pakistan-20200101-20200131'
#SMAP_FOLDER   = 'Uruguay_20200101_20200131'
#CYGNSS_FOLDER = 'Uruguay/Uruguay-20200101-20200131'
LAT_STEP      = 0.08
LON_STEP      = 0.08

cumulative_correlations(SMAP_FOLDER, CYGNSS_FOLDER, dates, LAT_STEP, LON_STEP)
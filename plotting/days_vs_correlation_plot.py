import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    for k in tqdm.tqdm(range(1, len(date_list) + 1), desc=f"{smap_folder}", unit="day"):
        window_dates = date_list[:k]
        grid_cyg  = composite_cygnss(window_dates, cygnss_folder, lat_bins, lon_bins)
        grid_smap = composite_smap(window_dates, smap_folder, lat_bins, lon_bins)

        if grid_cyg.empty or grid_smap.empty:
            results.append((k, np.nan))
            continue

        grid_smap['lat_center'] = grid_smap['lat_center'].round(4)
        grid_smap['lon_center'] = grid_smap['lon_center'].round(4)

        df = pd.merge(
            grid_cyg, grid_smap,
            on=['lat_center', 'lon_center'],
            suffixes=('_cygnss', '_smap')
        ).dropna(subset=['sr', 'soil_moisture_avg'])

        r = pearsonr(df['sr'], df['soil_moisture_avg'])[0] if not df.empty else np.nan
        results.append((k, r))

    return results

from datetime import datetime, timedelta

start = datetime(2020, 1, 1)
num_days = 31
dates = [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range(num_days)]

LAT_STEP = 0.08
LON_STEP = 0.08

regions = {
    'Pakistan': {
        'smap': 'Pakistan_20200101_20200131',
        'cygnss': 'Pakistan/Pakistan-20200101-20200131'
    },
    'Uruguay': {
        'smap': 'Uruguay_20200101_20200131',
        'cygnss': 'Uruguay/Uruguay-20200101-20200131'
    },
    'Sudan': {
        'smap': 'Sudan_20200101_20200131',
        'cygnss': 'Sudan/Sudan-20200101-20200131'
    },
    'South-Africa': {
        'smap': 'South-Africa_20200101_20200131',
        'cygnss': 'South-Africa-20200101-20200131'
    }
}

plt.figure(figsize=(10, 6))

for region, folders in regions.items():
    results = cumulative_correlations(folders['smap'], folders['cygnss'], dates, LAT_STEP, LON_STEP)
    ks, rs = zip(*results)
    plt.plot(ks, rs, label=region, linewidth=2, marker='o', markersize=4)

plt.xlabel("Days of CYGNSS Averaging")
plt.ylabel("Pearson Correlation with SMAP")
plt.title("Correlation vs. CYGNSS Averaging Window Length")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''
'Western Australia': {
        'smap': 'Western-Australia_20200101_20200131',
        'cygnss': 'Western-Australia/Western-Australia-20200101-20200131'
    }
'''
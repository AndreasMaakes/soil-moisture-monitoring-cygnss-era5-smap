import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.ndimage import gaussian_filter
from CYGNSS.import_data import importData
from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from ERA5.ERA5_utils import apply_land_sea_mask, averaging_soil_moisture
from scipy.stats import pearsonr

# ----------------------------------------
# tweak regrid_dataframe to support INC
# ----------------------------------------
def regrid_dataframe(df, lat_bins, lon_bins, data_source):
    df = df.copy()
    # choose value column
    if data_source == "CYGNSS":
        vals = df['sr']
    elif data_source == "SMAP":
        vals = df['soil_moisture_avg']
    elif data_source == "ERA5":
        vals = df['average_moisture']
    elif data_source == "INC":
        vals = df['sp_inc_angle']
    else:
        raise ValueError(f"Unknown source {data_source}")

    # assign lat/lon
    if data_source in ["CYGNSS", "INC"]:
        lat_col, lon_col = 'sp_lat', 'sp_lon'
    elif data_source == "SMAP":
        lat_col, lon_col = 'latitude', 'longitude'
    else:  # ERA5
        lat_col, lon_col = 'latitude', 'longitude'

    df['lat_bin'] = pd.cut(df[lat_col], bins=lat_bins, right=False)
    df['lon_bin'] = pd.cut(df[lon_col], bins=lon_bins, right=False)

    # mean in each cell
    df_grid = df.groupby(['lat_bin','lon_bin'])[vals.name].mean().reset_index()
    
    # compute centers
    lat_w = lat_bins[1] - lat_bins[0]
    lon_w = lon_bins[1] - lon_bins[0]
    df_grid['lat_center'] = df_grid['lat_bin'].apply(lambda b: b.left + lat_w/2)
    df_grid['lon_center'] = df_grid['lon_bin'].apply(lambda b: b.left + lon_w/2)
    return df_grid.rename(columns={vals.name: vals.name})

# ----------------------------------------
# rest of your machinery, unchanged:
# merged_dataframe, apply_gaussian_blur,...
# compute_local_correlation(...)
# (I’m assuming you already have these from before)
# ----------------------------------------

def scatter_corr_vs_incidence(
    smap_folder,
    cygnss_folder,
    era5_folder,
    lsm_threshold,
    lat_step,
    lon_step,
    gaussian_sigma=0,
    window_size=3
):
    # --- assemble dataframes ---
    # SMAP
    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap = SMAP_averaging_soil_moisture(pd.concat(dfs_smap))
    # CYGNSS
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss  = pd.concat(dfs_cygnss)
    df_cygnss  = df_cygnss[df_cygnss['ddm_snr'] >= 2]
    # ERA5
    df_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}').to_dataframe().reset_index()
    df_era5 = apply_land_sea_mask(averaging_soil_moisture(df_era5), lsm_threshold)

    # define domain & bins
    lat_min = min(df_cygnss.sp_lat.min(), df_smap.latitude.min(), df_era5.latitude.min())
    lat_max = max(df_cygnss.sp_lat.max(), df_smap.latitude.max(), df_era5.latitude.max())
    lon_min = min(df_cygnss.sp_lon.min(), df_smap.longitude.min(), df_era5.longitude.min())
    lon_max = max(df_cygnss.sp_lon.max(), df_smap.longitude.max(), df_era5.longitude.max())
    lat_bins = np.arange(lat_min, lat_max+lat_step, lat_step)
    lon_bins = np.arange(lon_min, lon_max+lon_step, lon_step)

    # regrid CYGNSS, SMAP
    grid_cyg = regrid_dataframe(df_cygnss, lat_bins, lon_bins, "CYGNSS")
    grid_inc = regrid_dataframe(df_cygnss, lat_bins, lon_bins, "INC")  # mean incidence
    grid_smp = regrid_dataframe(df_smap,  lat_bins, lon_bins, "SMAP")

    # optional blur
    if gaussian_sigma>0:
        from scipy.ndimage import gaussian_filter
        def blur(df, col):
            arr = df.pivot('lat_center','lon_center',col).values
            w   = ~np.isnan(arr)
            arr[np.isnan(arr)] = 0
            blurred = gaussian_filter(arr, sigma=gaussian_sigma) / gaussian_filter(w.astype(float), sigma=gaussian_sigma)
            df2 = (pd.DataFrame(blurred, index=np.unique(df.lat_center), columns=np.unique(df.lon_center))
                    .stack().reset_index())
            df2.columns = ['lat_center','lon_center',col]
            return df2
        grid_cyg = blur(grid_cyg,'sr')
        grid_inc = blur(grid_inc,'sp_inc_angle')
        grid_smp = blur(grid_smp,'soil_moisture_avg')

    # merge for correlation
    merged = pd.merge(grid_cyg, grid_smp, on=['lat_center','lon_center'],
                      suffixes=('_cyg','_smp'))
    # compute local corr
    lat_c, lon_c, corr_mat = compute_local_correlation(merged,'sr','soil_moisture_avg',window_size)
    # flatten
    df_corr = pd.DataFrame([
        {'lat_center':lat_c[i],'lon_center':lon_c[j],'corr':corr_mat[i,j]}
        for i in range(len(lat_c)) for j in range(len(lon_c))
    ])

    # merge with incidence
    df_plot = pd.merge(df_corr, grid_inc, on=['lat_center','lon_center'], how='inner')

    # now scatter: incidence vs correlation
    plt.figure(figsize=(8,6))
    plt.scatter(df_plot['sp_inc_angle'], df_plot['corr'], s=30, alpha=0.7)
    plt.xlabel('Mean Incidence Angle (°)')
    plt.ylabel('Local Spatial Correlation (CYGNSS vs SMAP)')
    plt.title('Local Correlation vs. Incidence Angle')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


scatter_corr_vs_incidence(
    smap_folder="India2",
    cygnss_folder="India2/India2-20200101-20200131",
    era5_folder="India2/ERA5_India2_2020_01_01_31.nc",
    lsm_threshold=0.9,
    lat_step=0.5,
    lon_step=0.5,
    gaussian_sigma=2.5,
    window_size=3
)
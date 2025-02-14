import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import xarray as xr
from ERA5.ERA5_utils import apply_land_sea_mask, averaging_soil_moisture

# Import your SMAP and CYGNSS routines.
from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from CYGNSS.import_data import importData

# Function to regrid a dataframe by binning lat and lon
def regrid_dataframe(df, lat_bins, lon_bins, data_source):
    if data_source == "CYGNSS":
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['sp_lat'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['sp_lon'], bins=lon_bins, right=False)
        # Aggregate soil moisture within each bin (using the mean here)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['sr'].mean().reset_index()
    elif data_source == "SMAP":
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        # Aggregate soil moisture within each bin (using the mean here)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['soil_moisture_avg'].mean().reset_index()
    elif data_source == "ERA5":
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        # Aggregate soil moisture within each bin (using the mean here)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['average_moisture'].mean().reset_index()
    else:
        print("Invalid data source provided")
    
    # Compute the center of each bin for plotting
    df_grid['lat_center'] = df_grid['lat_bin'].apply(lambda x: x.left + 0.5)
    df_grid['lon_center'] = df_grid['lon_bin'].apply(lambda x: x.left + 0.5)
    
    return df_grid

def merged_dataframe(smap_folder, cygnss_folder, era5_folder, lsm_threshold):

    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap1 = pd.concat(dfs_smap)
    df_smap = SMAP_averaging_soil_moisture(df_smap1)

    # CYGNSS data: Import and concatenate
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    
    #Adjusting ddm_snr and sp_rx_gain max limits to increase correlation with SMAP
    df_cygnss = df_cygnss[df_cygnss['ddm_snr'] >= 2]
    df_cygnss = df_cygnss[df_cygnss['sp_rx_gain'] >= 13]

    df_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}').to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)
    
    # Determine the overall spatial domain
    lat_min = min(df_cygnss['sp_lat'].min(), df_smap['latitude'].min(), df_era5_lsm['latitude'].min())
    lat_max = max(df_cygnss['sp_lat'].max(), df_smap['latitude'].max(), df_era5_lsm['latitude'].max())
    lon_min = min(df_cygnss['sp_lon'].min(), df_smap['longitude'].min(), df_era5_lsm['longitude'].min())
    lon_max = max(df_cygnss['sp_lon'].max(), df_smap['longitude'].max(), df_era5_lsm['longitude'].max())

    # Create bins with a 0.5° resolution
    lat_bins = np.arange(lat_min, lat_max + 0.5, 0.5)
    lon_bins = np.arange(lon_min, lon_max + 0.5, 0.5)

    df_cygnss_grid = regrid_dataframe(df_cygnss, lat_bins, lon_bins, "CYGNSS")    
    df_smap_grid   = regrid_dataframe(df_smap, lat_bins, lon_bins, "SMAP")
    df_era5_grid = regrid_dataframe(df_era5_lsm, lat_bins, lon_bins, "ERA5")

    df_merged_CYGNSS_SMAP = pd.merge(df_cygnss_grid, df_smap_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_cygnss', '_smap'))

    df_merged_CYGNSS_ERA5 = pd.merge(df_cygnss_grid, df_era5_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_cygnss', '_era5'))

    df_merged_SMAP_ERA5 = pd.merge(df_smap_grid, df_era5_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_smap', '_era5'))

    return df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5




def correlation_plot(smap_folder, cygnss_folder, era5_folder, lsm_threshold):
    # Get the merged data
    df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5 = merged_dataframe(smap_folder,cygnss_folder,era5_folder,lsm_threshold)

    # Remove rows with NaN values in the relevant soil moisture columns
    df_valid_CYGNSS_SMAP = df_merged_CYGNSS_SMAP.dropna(subset=['sr', 'soil_moisture_avg'])
    df_valid_CYGNSS_ERA5 = df_merged_CYGNSS_ERA5.dropna(subset=['sr', 'average_moisture'])
    df_valid_SMAP_ERA5   = df_merged_SMAP_ERA5.dropna(subset=['soil_moisture_avg', 'average_moisture'])

    # Calculate Pearson correlations
    r_CYGNSS_SMAP, p_value_CYGNSS_SMAP = pearsonr(df_valid_CYGNSS_SMAP['sr'], df_valid_CYGNSS_SMAP['soil_moisture_avg'])
    print(f'Overall Pearson correlation for CYGNSS/SMAP: r = {r_CYGNSS_SMAP:.2f}, p = {p_value_CYGNSS_SMAP:.3f}')

    r_CYGNSS_ERA5, p_value_CYGNSS_ERA5 = pearsonr(df_valid_CYGNSS_ERA5['sr'], df_valid_CYGNSS_ERA5['average_moisture'])
    print(f'Overall Pearson correlation for CYGNSS/ERA5: r = {r_CYGNSS_ERA5:.2f}, p = {p_value_CYGNSS_ERA5:.3f}')

    r_SMAP_ERA5, p_value_SMAP_ERA5 = pearsonr(df_valid_SMAP_ERA5['soil_moisture_avg'], df_valid_SMAP_ERA5['average_moisture'])
    print(f'Overall Pearson correlation for SMAP/ERA5: r = {r_SMAP_ERA5:.2f}, p = {p_value_SMAP_ERA5:.3f}')

    # -----------------------------#
    # Plot CYGNSS vs SMAP
    # -----------------------------#
    plt.figure(figsize=(8, 6))
    plt.scatter(df_valid_CYGNSS_SMAP['sr'], df_valid_CYGNSS_SMAP['soil_moisture_avg'], 
                alpha=0.7, label='Data Points')
    plt.xlabel('CYGNSS Soil Moisture')
    plt.ylabel('SMAP Soil Moisture')
    plt.title('Scatter Plot of Soil Moisture (CYGNSS vs SMAP)')

    # Compute and plot best-fit line
    x = df_valid_CYGNSS_SMAP['sr']
    y = df_valid_CYGNSS_SMAP['soil_moisture_avg']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Best Fit Line')
    plt.legend()
    plt.show()

    # -----------------------------#
    # Plot CYGNSS vs ERA5
    # -----------------------------#
    plt.figure(figsize=(8, 6))
    plt.scatter(df_valid_CYGNSS_ERA5['sr'], df_valid_CYGNSS_ERA5['average_moisture'], 
                alpha=0.7, label='Data Points')
    plt.xlabel('CYGNSS Soil Moisture')
    plt.ylabel('ERA5 Soil Moisture')
    plt.title('Scatter Plot of Soil Moisture (CYGNSS vs ERA5)')

    # Compute and plot best-fit line
    x = df_valid_CYGNSS_ERA5['sr']
    y = df_valid_CYGNSS_ERA5['average_moisture']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Best Fit Line')
    plt.legend()
    plt.show()

    # -----------------------------#
    # Plot SMAP vs ERA5
    # -----------------------------#
    plt.figure(figsize=(8, 6))
    plt.scatter(df_valid_SMAP_ERA5['soil_moisture_avg'], df_valid_SMAP_ERA5['average_moisture'], 
                alpha=0.7, label='Data Points')
    plt.xlabel('SMAP Soil Moisture')
    plt.ylabel('ERA5 Soil Moisture')
    plt.title('Scatter Plot of Soil Moisture (SMAP vs ERA5)')

    # Compute and plot best-fit line
    x = df_valid_SMAP_ERA5['soil_moisture_avg']
    y = df_valid_SMAP_ERA5['average_moisture']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Best Fit Line')
    plt.legend()
    plt.show()


correlation_plot( "India2", "India2/India2-20200101-20200107", "India2/ERA5_India2_2020_01_01_07.nc", 0.95)


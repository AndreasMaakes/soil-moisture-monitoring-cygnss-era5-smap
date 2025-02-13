import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

# Import your SMAP and CYGNSS routines.
from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from CYGNSS.import_data import importData

# Function to regrid a dataframe by binning lat and lon
def regrid_dataframe(df, lat_bins, lon_bins, cygnss):
    if cygnss:
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['sp_lat'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['sp_lon'], bins=lon_bins, right=False)
        # Aggregate soil moisture within each bin (using the mean here)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['sr'].mean().reset_index()
    else:
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        # Aggregate soil moisture within each bin (using the mean here)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['soil_moisture_avg'].mean().reset_index()
    
    # Compute the center of each bin for plotting
    df_grid['lat_center'] = df_grid['lat_bin'].apply(lambda x: x.left + 0.5)
    df_grid['lon_center'] = df_grid['lon_bin'].apply(lambda x: x.left + 0.5)
    
    return df_grid

def SMAP_CYGNSS_correlation_plot(smap_folder, cygnss_folder):
    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap1 = pd.concat(dfs_smap)
    df_smap = SMAP_averaging_soil_moisture(df_smap1)

    # CYGNSS data: Import and concatenate
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    
    #Adjusting ddm_snr and sp_rx_gain max limits to increase correlation with SMAP
    df_cygnss = df_cygnss[df_cygnss['ddm_snr'] >= 2]
    df_cygnss = df_cygnss[df_cygnss['sp_rx_gain'] >= 13]
    #df_cygnss = df_cygnss[df_cygnss['sp_inc_angle'] <= 40]
    
    

    # Determine the overall spatial domain
    lat_min = min(df_cygnss['sp_lat'].min(), df_smap['latitude'].min())
    lat_max = max(df_cygnss['sp_lat'].max(), df_smap['latitude'].max())
    lon_min = min(df_cygnss['sp_lon'].min(), df_smap['longitude'].min())
    lon_max = max(df_cygnss['sp_lon'].max(), df_smap['longitude'].max())

    # Create bins with a 0.5° resolution
    lat_bins = np.arange(lat_min, lat_max + 0.5, 0.5)
    lon_bins = np.arange(lon_min, lon_max + 0.5, 0.5)

    df_cygnss_grid = regrid_dataframe(df_cygnss, lat_bins, lon_bins, True)    
    df_smap_grid   = regrid_dataframe(df_smap, lat_bins, lon_bins, False)

    df_merged = pd.merge(df_cygnss_grid, df_smap_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_cygnss', '_smap'))

    return df_merged

# Get the merged data
df_merged = SMAP_CYGNSS_correlation_plot("India2", "India2/India2-20200101-20200107")

# Remove rows with NaN values in either soil moisture column
df_valid = df_merged.dropna(subset=['sr', 'soil_moisture_avg'])

# Calculate Pearson correlation on the valid data
r, p_value = pearsonr(df_valid['sr'], df_valid['soil_moisture_avg'])
print(f'Overall Pearson correlation: r = {r:.2f}, p = {p_value:.3f}')

# Create the scatter plot with valid data only
plt.figure(figsize=(8,6))
plt.scatter(df_valid['sr'], df_valid['soil_moisture_avg'], alpha=0.7, label='Data Points')
plt.xlabel('CYGNSS Soil Moisture')
plt.ylabel('SMAP Soil Moisture')
plt.title('Scatter Plot of Soil Moisture (CYGNSS vs SMAP)')

# Use polyfit to compute the best fit line using the valid data
x = df_valid['sr']
y = df_valid['soil_moisture_avg']
slope, intercept = np.polyfit(x, y, 1)
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = slope * x_fit + intercept

# Plot the best fit line
plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Best Fit Line')
plt.legend()
plt.show()



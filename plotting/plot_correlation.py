import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from scipy.stats import pearsonr
import xarray as xr
from ERA5.ERA5_utils import apply_land_sea_mask, averaging_soil_moisture

from SMAP.SMAP_import_data import importDataSMAP
from SMAP.SMAP_utils import SMAP_averaging_soil_moisture
from CYGNSS.import_data import importData

# Function to regrid a dataframe by binning lat and lon using the provided step sizes
def regrid_dataframe(df, lat_bins, lon_bins, data_source):
    if data_source == "CYGNSS":
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['sp_lat'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['sp_lon'], bins=lon_bins, right=False)
        # Aggregate the soil moisture (or related) value within each bin using the mean
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['sr'].mean().reset_index()
    elif data_source == "SMAP":
        df = df.copy()
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['soil_moisture_avg'].mean().reset_index()
    elif data_source == "ERA5":
        df = df.copy()
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['average_moisture'].mean().reset_index()
    else:
        print("Invalid data source provided")
    
    # Compute the center of each bin for plotting.
    # The bin width is determined from the provided bin edges.
    lat_bin_width = lat_bins[1] - lat_bins[0]
    lon_bin_width = lon_bins[1] - lon_bins[0]
    df_grid['lat_center'] = df_grid['lat_bin'].apply(lambda x: x.left + lat_bin_width / 2)
    df_grid['lon_center'] = df_grid['lon_bin'].apply(lambda x: x.left + lon_bin_width / 2)
    
    return df_grid

def merged_dataframe(smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step):
    # --- SMAP ---
    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap1 = pd.concat(dfs_smap)
    df_smap = SMAP_averaging_soil_moisture(df_smap1)

    # --- CYGNSS ---
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    # Adjust ddm_snr to be >= 2 or whatever threshold is desired. This could be done in datafetching but we have chose to rather to it 
    # here to see how the correlation changes when we change the ddm_snr threshold
    df_cygnss = df_cygnss[df_cygnss['ddm_snr'] >= 2]
    

    # --- ERA5 ---
    df_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}').to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)
    
    # Determine the overall spatial domain using the union of all data
    lat_min = min(df_cygnss['sp_lat'].min(), df_smap['latitude'].min(), df_era5_lsm['latitude'].min())
    lat_max = max(df_cygnss['sp_lat'].max(), df_smap['latitude'].max(), df_era5_lsm['latitude'].max())
    lon_min = min(df_cygnss['sp_lon'].min(), df_smap['longitude'].min(), df_era5_lsm['longitude'].min())
    lon_max = max(df_cygnss['sp_lon'].max(), df_smap['longitude'].max(), df_era5_lsm['longitude'].max())

    # Create bins using the provided step sizes
    lat_bins = np.arange(lat_min, lat_max + lat_step, lat_step)
    lon_bins = np.arange(lon_min, lon_max + lon_step, lon_step)

    df_cygnss_grid = regrid_dataframe(df_cygnss, lat_bins, lon_bins, "CYGNSS")    
    df_smap_grid   = regrid_dataframe(df_smap, lat_bins, lon_bins, "SMAP")
    df_era5_grid   = regrid_dataframe(df_era5_lsm, lat_bins, lon_bins, "ERA5")

    df_merged_CYGNSS_SMAP = pd.merge(df_cygnss_grid, df_smap_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_cygnss', '_smap'))
    df_merged_CYGNSS_ERA5 = pd.merge(df_cygnss_grid, df_era5_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_cygnss', '_era5'))
    df_merged_SMAP_ERA5   = pd.merge(df_smap_grid, df_era5_grid, on=['lat_center', 'lon_center'],
                         suffixes=('_smap', '_era5'))

    return df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5

def correlation_plot(smap_folder, cygnss_folder, era5_folder,  lat_step, lon_step, lsm_threshold):
    # Get the merged data using the provided step sizes
    df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5 = merged_dataframe(
        smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step)

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

    x = df_valid_SMAP_ERA5['soil_moisture_avg']
    y = df_valid_SMAP_ERA5['average_moisture']
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Best Fit Line')
    plt.legend()
    plt.show()

def regrid_dataframe(df, lat_bins, lon_bins, data_source):
    if data_source == "CYGNSS":
        df = df.copy()  # avoid modifying the original dataframe
        df['lat_bin'] = pd.cut(df['sp_lat'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['sp_lon'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['sr'].mean().reset_index()
    elif data_source == "SMAP":
        df = df.copy()
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['soil_moisture_avg'].mean().reset_index()
    elif data_source == "ERA5":
        df = df.copy()
        df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, right=False)
        df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, right=False)
        df_grid = df.groupby(['lat_bin', 'lon_bin'])['average_moisture'].mean().reset_index()
    else:
        print("Invalid data source provided")
        return None

    # Compute the bin width from the provided bin edges.
    lat_bin_width = lat_bins[1] - lat_bins[0]
    lon_bin_width = lon_bins[1] - lon_bins[0]
    # Compute the center of each bin
    df_grid['lat_center'] = df_grid['lat_bin'].apply(lambda x: float(x.left) + lat_bin_width / 2)
    df_grid['lon_center'] = df_grid['lon_bin'].apply(lambda x: float(x.left) + lon_bin_width / 2)
    
    return df_grid

def spatial_correlation_matrix(smap_folder, cygnss_folder, era5_folder,
                               lat_step, lon_step, lsm_threshold,
                               window=1, min_points=3):
    """
    Computes local (moving-window) spatial Pearson correlations for:
      - CYGNSS vs SMAP and 
      - CYGNSS vs ERA5
    using your gridded data (via merged_dataframe).

    The data are first gridded (using your existing regrid_dataframe function via merged_dataframe)
    and then for each grid cell a neighborhood (defined by 'window' cells in each direction) is used
    to compute a local Pearson correlation. The result is pivoted into matrices (with lat/lon axes)
    and then each is plotted in a separate figure with the correlation printed inside each cell.

    Parameters:
      smap_folder   : Folder containing SMAP data.
      cygnss_folder : Folder containing CYGNSS data.
      era5_folder   : Folder containing ERA5 data.
      lat_step      : Grid resolution (degrees) in latitude.
      lon_step      : Grid resolution (degrees) in longitude.
      lsm_threshold : Land-sea mask threshold for ERA5.
      window        : Number of grid cells (in each direction) for the moving window.
                      (window=1 gives a 3x3 neighborhood)
      min_points    : Minimum number of grid cells in the neighborhood required to compute a correlation.
    
    Returns:
      Two pivot DataFrames (matrices) for the local correlation values:
         - pivot_CS: CYGNSS vs SMAP
         - pivot_CE: CYGNSS vs ERA5
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    # Get gridded data using your existing merged_dataframe function.
    # (It returns three DataFrames: CYGNSS vs SMAP, CYGNSS vs ERA5, and SMAP vs ERA5)
    df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, _ = merged_dataframe(
        smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step)
    
    # Convert the grid cell centers to float (if they are still categorical)
    df_merged_CYGNSS_SMAP['lat_center'] = df_merged_CYGNSS_SMAP['lat_center'].astype(float)
    df_merged_CYGNSS_SMAP['lon_center'] = df_merged_CYGNSS_SMAP['lon_center'].astype(float)
    df_merged_CYGNSS_ERA5['lat_center'] = df_merged_CYGNSS_ERA5['lat_center'].astype(float)
    df_merged_CYGNSS_ERA5['lon_center'] = df_merged_CYGNSS_ERA5['lon_center'].astype(float)
    
    # Drop grid cells where either sensor's value is missing
    df_valid_CS = df_merged_CYGNSS_SMAP.dropna(subset=['sr', 'soil_moisture_avg']).reset_index(drop=True)
    df_valid_CE = df_merged_CYGNSS_ERA5.dropna(subset=['sr', 'average_moisture']).reset_index(drop=True)
    
    # Compute local correlation for CYGNSS vs SMAP
    local_corr_CS = []
    for idx, row in df_valid_CS.iterrows():
        center_lat = row['lat_center']
        center_lon = row['lon_center']
        # Define neighborhood boundaries
        lat_min_win = center_lat - window * lat_step
        lat_max_win = center_lat + window * lat_step
        lon_min_win = center_lon - window * lon_step
        lon_max_win = center_lon + window * lon_step
        
        neighborhood = df_valid_CS[
            (df_valid_CS['lat_center'] >= lat_min_win) &
            (df_valid_CS['lat_center'] <= lat_max_win) &
            (df_valid_CS['lon_center'] >= lon_min_win) &
            (df_valid_CS['lon_center'] <= lon_max_win)
        ]
        
        if len(neighborhood) >= min_points:
            r_local, _ = pearsonr(neighborhood['sr'], neighborhood['soil_moisture_avg'])
        else:
            r_local = np.nan
        local_corr_CS.append(r_local)
    df_valid_CS['local_corr'] = local_corr_CS

    # Compute local correlation for CYGNSS vs ERA5
    local_corr_CE = []
    for idx, row in df_valid_CE.iterrows():
        center_lat = row['lat_center']
        center_lon = row['lon_center']
        lat_min_win = center_lat - window * lat_step
        lat_max_win = center_lat + window * lat_step
        lon_min_win = center_lon - window * lon_step
        lon_max_win = center_lon + window * lon_step
        
        neighborhood = df_valid_CE[
            (df_valid_CE['lat_center'] >= lat_min_win) &
            (df_valid_CE['lat_center'] <= lat_max_win) &
            (df_valid_CE['lon_center'] >= lon_min_win) &
            (df_valid_CE['lon_center'] <= lon_max_win)
        ]
        
        if len(neighborhood) >= min_points:
            r_local, _ = pearsonr(neighborhood['sr'], neighborhood['average_moisture'])
        else:
            r_local = np.nan
        local_corr_CE.append(r_local)
    df_valid_CE['local_corr'] = local_corr_CE

    # Create pivot tables (matrices) with lat_center as rows and lon_center as columns.
    pivot_CS = df_valid_CS.pivot(index='lat_center', columns='lon_center', values='local_corr')
    pivot_CE = df_valid_CE.pivot(index='lat_center', columns='lon_center', values='local_corr')
    
    pivot_CS = pivot_CS.sort_index(ascending=True)
    pivot_CE = pivot_CE.sort_index(ascending=True)
    
    # ---------------------------
    # Plot for CYGNSS vs SMAP
    # ---------------------------
    plt.figure(figsize=(8, 6))
    # Calculate extent so that grid cells are centered properly
    extent_CS = [pivot_CS.columns.min()-lon_step/2, pivot_CS.columns.max()+lon_step/2,
                 pivot_CS.index.min()-lat_step/2, pivot_CS.index.max()+lat_step/2]
    plt.imshow(pivot_CS, origin='lower', aspect='auto', extent=extent_CS, cmap='coolwarm')
    # Annotate each cell with its correlation value
    for lat in pivot_CS.index:
        for lon in pivot_CS.columns:
            value = pivot_CS.loc[lat, lon]
            if pd.notna(value):
                plt.text(lon, lat, f'{value:.2f}', ha='center', va='center', fontsize=10, color='black')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Local Correlation: CYGNSS vs SMAP')
    plt.show()
    
    # ---------------------------
    # Plot for CYGNSS vs ERA5
    # ---------------------------
    plt.figure(figsize=(8, 6))
    extent_CE = [pivot_CE.columns.min()-lon_step/2, pivot_CE.columns.max()+lon_step/2,
                 pivot_CE.index.min()-lat_step/2, pivot_CE.index.max()+lat_step/2]
    plt.imshow(pivot_CE, origin='lower', aspect='auto', extent=extent_CE, cmap='coolwarm')
    for lat in pivot_CE.index:
        for lon in pivot_CE.columns:
            value = pivot_CE.loc[lat, lon]
            if pd.notna(value):
                plt.text(lon, lat, f'{value:.2f}', ha='center', va='center', fontsize=10, color='black')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Local Correlation: CYGNSS vs ERA5')
    plt.show()
    
    return pivot_CS, pivot_CE

# Example usage:
# pivot_CS, pivot_CE = spatial_correlation_matrix("India2", 
#                                                  "India2/India2-20200101-20200131", 
#                                                  "India2/ERA5_India2_2020_01_01_31.nc", 
#                                                  lat_step=0.5, lon_step=0.5, lsm_threshold=0.9,
#                                                  window=1, min_points=3)



spatial_correlation_matrix("Bolivia", "Bolivia/Bolivia-20240701-20240707", "Bolivia/ERA5_Bolivia_2024_07_01_07.nc", 0.5, 0.5, 0.9, window=1, min_points=3)


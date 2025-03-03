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
    # Adjust quality control limits
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

def correlation_matrix(smap_folder, cygnss_folder, era5_folder, lat_step, lon_step, lsm_threshold):
    """
    Create 2D correlation matrices between SMAP, CYGNSS, and ERA5 data using a user‐specified 
    latitude and longitude step size (in degrees) to define the spatial bins.
    
    This function:
      1. Imports SMAP, CYGNSS, and ERA5 data and preprocesses them.
      2. Interpolates each dataset onto a common, “fine” grid (with a resolution 10× finer 
         than the bin (cell) size).
      3. Divides the domain into bins (starting at the minimum lat/lon) of size lat_step x lon_step.
      4. In each bin, computes the Pearson correlation coefficient between the two datasets,
         using all the fine-grid points that fall within the bin.
      5. Computes an overall correlation using binned data (to be consistent with your scatter plot).
      6. Plots a pcolormesh of the correlation matrix with the correlation value printed inside 
         each bin and the overall correlation in the title.
    
    Parameters:
      smap_folder : str
          Folder (or identifier) for SMAP data.
      cygnss_folder : str
          Folder (or identifier) for CYGNSS data.
      era5_folder : str
          Identifier for the ERA5 dataset (assumed to be in data/ERA5/).
      lat_step : float
          Step size in latitude (degrees) for the correlation bins.
      lon_step : float
          Step size in longitude (degrees) for the correlation bins.
      lsm_threshold : float
          Threshold value used by apply_land_sea_mask() for ERA5 data.
    """
    
    # ============================================================
    # 1. Import and Preprocess the Data
    # ============================================================
    # --- SMAP data ---
    dfs_smap = importDataSMAP(False, smap_folder)
    df_smap = pd.concat(dfs_smap)
    df_smap_avg = SMAP_averaging_soil_moisture(df_smap)
    
    # --- CYGNSS data ---
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    #df_cygnss = df_cygnss[df_cygnss['ddm_snr'] > 2]
    #df_cygnss = df_cygnss[df_cygnss['sp_rx_gain'] < 13]
    
    # --- ERA5 data ---
    ds_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}')
    df_era5 = ds_era5.to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)
    
    # ============================================================
    # 2. Define the Overall Spatial Extents and Create a Fine Grid
    # ============================================================
    # Use union of SMAP, CYGNSS, and ERA5
    lat_min = min(df_smap_avg["latitude"].min(), df_cygnss["sp_lat"].min(), df_era5_lsm["latitude"].min())
    lat_max = max(df_smap_avg["latitude"].max(), df_cygnss["sp_lat"].max(), df_era5_lsm["latitude"].max())
    lon_min = min(df_smap_avg["longitude"].min(), df_cygnss["sp_lon"].min(), df_era5_lsm["longitude"].min())
    lon_max = max(df_smap_avg["longitude"].max(), df_cygnss["sp_lon"].max(), df_era5_lsm["longitude"].max())
    
    # Create a fine grid with resolution 10× finer than the bin size
    interp_factor = 10
    fine_lat_step = lat_step / interp_factor
    fine_lon_step = lon_step / interp_factor
    lat_fine = np.arange(lat_min, lat_max + fine_lat_step, fine_lat_step)
    lon_fine = np.arange(lon_min, lon_max + fine_lon_step, fine_lon_step)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)
    
    # ============================================================
    # 3. Interpolate Each Dataset onto the Fine Grid
    # ============================================================
    smap_points = (df_smap_avg["longitude"].values, df_smap_avg["latitude"].values)
    smap_vals = df_smap_avg["soil_moisture_avg"].values
    smap_fine = griddata(smap_points, smap_vals, (lon_mesh, lat_mesh), method='linear')
    
    cygnss_points = (df_cygnss["sp_lon"].values, df_cygnss["sp_lat"].values)
    cygnss_vals = df_cygnss["sr"].values
    cygnss_fine = griddata(cygnss_points, cygnss_vals, (lon_mesh, lat_mesh), method='linear')
    
    era5_points = (df_era5_lsm["longitude"].values, df_era5_lsm["latitude"].values)
    era5_vals = df_era5_lsm["average_moisture"].values
    era5_fine = griddata(era5_points, era5_vals, (lon_mesh, lat_mesh), method='linear')
    
    # ============================================================
    # 4. Define Correlation Bins Using the Specified Step Size
    # ============================================================
    lat_edges = np.arange(lat_min, lat_max, lat_step)
    lon_edges = np.arange(lon_min, lon_max, lon_step)
    n_lat_bins = len(lat_edges)
    n_lon_bins = len(lon_edges)
    lat_centers = lat_edges + lat_step/2
    lon_centers = lon_edges + lon_step/2
    
    # Initialize correlation matrices (per bin)
    corr_matrix_smap_cygnss = np.full((n_lat_bins, n_lon_bins), np.nan)
    corr_matrix_era5_cygnss = np.full((n_lat_bins, n_lon_bins), np.nan)
    corr_matrix_smap_era5   = np.full((n_lat_bins, n_lon_bins), np.nan)
    
    # ============================================================
    # 5. Compute the Correlation in Each Bin (using fine grid interpolation)
    # ============================================================
    for i in range(n_lat_bins):
        for j in range(n_lon_bins):
            lat_lower = lat_edges[i]
            lat_upper = lat_lower + lat_step
            lon_lower = lon_edges[j]
            lon_upper = lon_lower + lon_step
            mask = ((lat_mesh >= lat_lower) & (lat_mesh < lat_upper) &
                    (lon_mesh >= lon_lower) & (lon_mesh < lon_upper))
            
            # SMAP vs CYGNSS
            block_smap = smap_fine[mask].flatten()
            block_cygnss = cygnss_fine[mask].flatten()
            valid = ~np.isnan(block_smap) & ~np.isnan(block_cygnss)
            if np.sum(valid) >= 2:
                corr_matrix_smap_cygnss[i, j] = np.corrcoef(block_smap[valid], block_cygnss[valid])[0, 1]
            
            # ERA5 vs CYGNSS
            block_era5 = era5_fine[mask].flatten()
            block_cygnss = cygnss_fine[mask].flatten()
            valid = ~np.isnan(block_era5) & ~np.isnan(block_cygnss)
            if np.sum(valid) >= 2:
                corr_matrix_era5_cygnss[i, j] = np.corrcoef(block_era5[valid], block_cygnss[valid])[0, 1]
            
            # SMAP vs ERA5
            block_smap = smap_fine[mask].flatten()
            block_era5 = era5_fine[mask].flatten()
            valid = ~np.isnan(block_smap) & ~np.isnan(block_era5)
            if np.sum(valid) >= 2:
                corr_matrix_smap_era5[i, j] = np.corrcoef(block_smap[valid], block_era5[valid])[0, 1]
    
    # ============================================================
    # 6. Compute Overall Correlations from the Binned Data (to match correlation_plot)
    # ============================================================
      # assuming merged_dataframe is accessible
    df_merged_CYGNSS_SMAP, df_merged_CYGNSS_ERA5, df_merged_SMAP_ERA5 = merged_dataframe(
        smap_folder, cygnss_folder, era5_folder, lsm_threshold, lat_step, lon_step)
    
    df_valid_CYGNSS_SMAP = df_merged_CYGNSS_SMAP.dropna(subset=['sr', 'soil_moisture_avg'])
    overall_smap_cygnss = np.corrcoef(df_valid_CYGNSS_SMAP['sr'], df_valid_CYGNSS_SMAP['soil_moisture_avg'])[0,1]
    
    df_valid_CYGNSS_ERA5 = df_merged_CYGNSS_ERA5.dropna(subset=['sr', 'average_moisture'])
    overall_era5_cygnss = np.corrcoef(df_valid_CYGNSS_ERA5['sr'], df_valid_CYGNSS_ERA5['average_moisture'])[0,1]
    
    df_valid_SMAP_ERA5 = df_merged_SMAP_ERA5.dropna(subset=['soil_moisture_avg', 'average_moisture'])
    overall_smap_era5 = np.corrcoef(df_valid_SMAP_ERA5['soil_moisture_avg'], df_valid_SMAP_ERA5['average_moisture'])[0,1]
    
    # ============================================================
    # 7. Plotting Function for a Given Correlation Matrix
    # ============================================================
    def plot_corr_matrix(corr_matrix, lon_edges, lat_edges, lon_centers, lat_centers, title_str, overall_val):
        plt.figure(figsize=(10, 8))
        mesh = plt.pcolormesh(lon_edges, lat_edges, corr_matrix, shading='auto', 
                              cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(mesh, label='Pearson Correlation')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                corr_val = corr_matrix[i, j]
                if not np.isnan(corr_val):
                    plt.text(lon_centers[j], lat_centers[i], f"{corr_val:.2f}",
                             ha='center', va='center', color='black', fontsize=10)
        plt.title(f'{title_str}\nOverall Correlation: {overall_val:.2f}', fontsize=14, pad=20)
        plt.show()
    
    lat_edges_plot = np.append(lat_edges, lat_edges[-1] + lat_step)
    lon_edges_plot = np.append(lon_edges, lon_edges[-1] + lon_step)
    
    # ============================================================
    # 8. Plot the Three Correlation Matrices
    # ============================================================
    plot_corr_matrix(corr_matrix_smap_cygnss, lon_edges_plot, lat_edges_plot, 
                     lon_centers, lat_centers,
                     'Correlation Matrix between SMAP and CYGNSS', overall_smap_cygnss)
    
    plot_corr_matrix(corr_matrix_era5_cygnss, lon_edges_plot, lat_edges_plot, 
                     lon_centers, lat_centers,
                     'Correlation Matrix between ERA5 and CYGNSS', overall_era5_cygnss)
    
    plot_corr_matrix(corr_matrix_smap_era5, lon_edges_plot, lat_edges_plot, 
                     lon_centers, lat_centers,
                     'Correlation Matrix between SMAP and ERA5', overall_smap_era5)




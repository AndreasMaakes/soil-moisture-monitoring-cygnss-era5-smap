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
      5. Computes an overall correlation using all valid fine-grid points.
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
    # Expected columns: 'latitude', 'longitude', 'soil_moisture_avg'
    
    # --- CYGNSS data ---
    dfs_cygnss = importData(cygnss_folder)
    df_cygnss = pd.concat(dfs_cygnss)
    # Remove low quality measurements
    df_cygnss = df_cygnss[df_cygnss['ddm_snr'] > 2]
    df_cygnss = df_cygnss[df_cygnss['sp_rx_gain'] > 13]
    # Expected columns: 'sp_lat', 'sp_lon', 'sr'
    
    # --- ERA5 data ---
    ds_era5 = xr.open_dataset(f'data/ERA5/{era5_folder}')
    df_era5 = ds_era5.to_dataframe().reset_index()
    df_era5_avg = averaging_soil_moisture(df_era5)
    df_era5_lsm = apply_land_sea_mask(df_era5_avg, lsm_threshold)
    # Expected columns: 'latitude', 'longitude', 'average_moisture'
    
    # ============================================================
    # 2. Define the Overall Spatial Extents and Create a Fine Grid
    # ============================================================
    # Use union of SMAP and CYGNSS (similar to original code; ERA5 may be masked)
    lat_min = min(df_smap_avg["latitude"].min(), df_cygnss["sp_lat"].min())
    lat_max = max(df_smap_avg["latitude"].max(), df_cygnss["sp_lat"].max())
    lon_min = min(df_smap_avg["longitude"].min(), df_cygnss["sp_lon"].min())
    lon_max = max(df_smap_avg["longitude"].max(), df_cygnss["sp_lon"].max())
    
    # --- Create a fine grid for interpolation ---
    # We choose an interpolation resolution that is 10× finer than the correlation bin size.
    interp_factor = 10
    fine_lat_step = lat_step / interp_factor
    fine_lon_step = lon_step / interp_factor
    
    lat_fine = np.arange(lat_min, lat_max + fine_lat_step, fine_lat_step)
    lon_fine = np.arange(lon_min, lon_max + fine_lon_step, fine_lon_step)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)
    
    # ============================================================
    # 3. Interpolate Each Dataset onto the Fine Grid
    # ============================================================
    # --- SMAP ---
    smap_points = (df_smap_avg["longitude"].values, df_smap_avg["latitude"].values)
    smap_vals = df_smap_avg["soil_moisture_avg"].values
    smap_fine = griddata(smap_points, smap_vals, (lon_mesh, lat_mesh), method='linear')
    
    # --- CYGNSS ---
    cygnss_points = (df_cygnss["sp_lon"].values, df_cygnss["sp_lat"].values)
    cygnss_vals = df_cygnss["sr"].values
    cygnss_fine = griddata(cygnss_points, cygnss_vals, (lon_mesh, lat_mesh), method='linear')
    
    # --- ERA5 ---
    era5_points = (df_era5_lsm["longitude"].values, df_era5_lsm["latitude"].values)
    era5_vals = df_era5_lsm["average_moisture"].values
    era5_fine = griddata(era5_points, era5_vals, (lon_mesh, lat_mesh), method='linear')
    
    # ============================================================
    # 4. Define Correlation Bins Using the Specified Step Size
    # ============================================================
    # Bins (edges) starting at lat_min and lon_min.
    lat_edges = np.arange(lat_min, lat_max, lat_step)
    lon_edges = np.arange(lon_min, lon_max, lon_step)
    n_lat_bins = len(lat_edges)
    n_lon_bins = len(lon_edges)
    
    # Compute cell centers for annotation
    lat_centers = lat_edges + lat_step/2
    lon_centers = lon_edges + lon_step/2
    
    # Initialize correlation matrices for the three pairs
    corr_matrix_smap_cygnss = np.full((n_lat_bins, n_lon_bins), np.nan)
    corr_matrix_era5_cygnss = np.full((n_lat_bins, n_lon_bins), np.nan)
    corr_matrix_smap_era5   = np.full((n_lat_bins, n_lon_bins), np.nan)
    
    # ============================================================
    # 5. Compute the Correlation in Each Bin
    # ============================================================
    # For each bin, find the indices of the fine grid that fall within the bin boundaries,
    # then compute the correlation coefficient between the two datasets.
    for i in range(n_lat_bins):
        for j in range(n_lon_bins):
            # Define bin boundaries
            lat_lower = lat_edges[i]
            lat_upper = lat_lower + lat_step
            lon_lower = lon_edges[j]
            lon_upper = lon_lower + lon_step
            
            # Create a boolean mask for points within this bin
            mask = ((lat_mesh >= lat_lower) & (lat_mesh < lat_upper) &
                    (lon_mesh >= lon_lower) & (lon_mesh < lon_upper))
            
            # ---- SMAP vs CYGNSS ----
            block_smap = smap_fine[mask].flatten()
            block_cygnss = cygnss_fine[mask].flatten()
            valid = ~np.isnan(block_smap) & ~np.isnan(block_cygnss)
            if np.sum(valid) >= 2:
                corr = np.corrcoef(block_smap[valid], block_cygnss[valid])[0, 1]
                corr_matrix_smap_cygnss[i, j] = corr
            
            # ---- ERA5 vs CYGNSS ----
            block_era5 = era5_fine[mask].flatten()
            block_cygnss = cygnss_fine[mask].flatten()
            valid = ~np.isnan(block_era5) & ~np.isnan(block_cygnss)
            if np.sum(valid) >= 2:
                corr = np.corrcoef(block_era5[valid], block_cygnss[valid])[0, 1]
                corr_matrix_era5_cygnss[i, j] = corr
            
            # ---- SMAP vs ERA5 ----
            block_smap = smap_fine[mask].flatten()
            block_era5 = era5_fine[mask].flatten()
            valid = ~np.isnan(block_smap) & ~np.isnan(block_era5)
            if np.sum(valid) >= 2:
                corr = np.corrcoef(block_smap[valid], block_era5[valid])[0, 1]
                corr_matrix_smap_era5[i, j] = corr
    
    # ============================================================
    # 6. Compute Overall Correlations (using all fine-grid points)
    # ============================================================
    def overall_corr(field1, field2):
        valid = ~np.isnan(field1) & ~np.isnan(field2)
        if np.sum(valid) >= 2:
            return np.corrcoef(field1[valid].flatten(), field2[valid].flatten())[0, 1]
        else:
            return np.nan
    
    overall_smap_cygnss = overall_corr(smap_fine, cygnss_fine)
    overall_era5_cygnss = overall_corr(era5_fine, cygnss_fine)
    overall_smap_era5   = overall_corr(smap_fine, era5_fine)
    
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
        
        # Annotate each bin with the correlation value
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                corr_val = corr_matrix[i, j]
                if not np.isnan(corr_val):
                    plt.text(lon_centers[j], lat_centers[i], f"{corr_val:.2f}",
                             ha='center', va='center', color='black', fontsize=10)
        plt.title(f'{title_str}\nOverall Correlation: {overall_val:.2f}', fontsize=14, pad=20)
        plt.show()
    
    # Create grid edges for pcolormesh (add one extra edge at the end)
    lat_edges_plot = np.append(lat_edges, lat_edges[-1] + lat_step)
    lon_edges_plot = np.append(lon_edges, lon_edges[-1] + lon_step)
    
    # ============================================================
    # 8. Plot the Three Correlation Matrices
    # ============================================================
    # SMAP vs CYGNSS
    plot_corr_matrix(corr_matrix_smap_cygnss, lon_edges_plot, lat_edges_plot, 
                     lon_centers, lat_centers,
                     'Correlation Matrix between SMAP and CYGNSS', overall_smap_cygnss)
    
    # ERA5 vs CYGNSS
    plot_corr_matrix(corr_matrix_era5_cygnss, lon_edges_plot, lat_edges_plot, 
                     lon_centers, lat_centers,
                     'Correlation Matrix between ERA5 and CYGNSS', overall_era5_cygnss)
    
    # SMAP vs ERA5
    plot_corr_matrix(corr_matrix_smap_era5, lon_edges_plot, lat_edges_plot, 
                     lon_centers, lat_centers,
                     'Correlation Matrix between SMAP and ERA5', overall_smap_era5)


correlation_matrix("India2", "India2/India2-20200101-20200107", "India2/ERA5_India2_2020_01_01_07.nc", 0.5, 0.5, 0.95)
#correlation_plot( "India2", "India2/India2-20200101-20200107", "India2/ERA5_India2_2020_01_01_07.nc", 0.95)


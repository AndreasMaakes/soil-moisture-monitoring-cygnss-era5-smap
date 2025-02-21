import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .import_data import importData
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

def CYGNSS_average_plot(folder_name, sigma, step_size_lon, step_size_lat, smooth):
    # Create a title based on whether smoothing is applied.
    name = folder_name.split("/")[0]
    if sigma == 0:
        title = f'CYGNSS Surface Reflectivity - {name} - September 2024'
    else:
        title = f'Smoothed CYGNSS Surface Reflectivity - {name} - September 2024 - σ={sigma}'

    # Import and concatenate data from the folder.
    dfs = importData(folder_name)
    df = pd.concat(dfs)

    # Extract the latitude, longitude, and surface reflectivity values.
    latitudes = df["sp_lat"].values
    longitudes = df["sp_lon"].values
    sr_values = df["sr"].values
    
    #dropping DDM_SNR and SP_RX_GAIN below 2 and 13
    df = df[df['ddm_snr'] > 2]
    df = df[df['sp_rx_gain'] < 13]


    # Define grid bin edges based on the desired step sizes.
    lat_edges = np.arange(latitudes.min(), latitudes.max() + step_size_lat, step_size_lat)
    lon_edges = np.arange(longitudes.min(), longitudes.max() + step_size_lon, step_size_lon)

    # Bin the data: average the 'sr' values that fall within each grid cell.
    # Note: binned_statistic_2d returns an array with shape 
    #       (len(lon_edges)-1, len(lat_edges)-1) where the first axis corresponds to longitude.
    stat, _, _, _ = binned_statistic_2d(
        longitudes, latitudes, sr_values, statistic='mean', bins=[lon_edges, lat_edges]
    )
    # Transpose so that the array has shape (n_lat_bins, n_lon_bins)
    stat = stat.T

    # Optionally apply Gaussian smoothing if sigma > 0.
    if sigma > 0:
        stat = gaussian_filter(stat, sigma=sigma)

    plt.figure(figsize=(10, 8))

    if smooth:
        # Use imshow with bilinear interpolation.
        mesh = plt.imshow(
            stat, 
            origin='lower', 
            extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]], 
            cmap='viridis', 
            interpolation='bilinear'
        )
    else:
        # Use pcolormesh with the bin edges. Note that stat.shape should be 
        # (len(lat_edges)-1, len(lon_edges)-1) which is consistent with pcolormesh.
        mesh = plt.pcolormesh(
            lon_edges, lat_edges, stat, shading='auto', cmap="viridis"
        )
        
    plt.colorbar(mesh, label='SR')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    
    # Ensure the aspect ratio is equal.
    plt.axis('equal')
    plt.show()


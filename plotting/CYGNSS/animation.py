import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata


import xarray as xr
import os

'''
This function is used to adjust the SR values of the data to make the plots more intuitive.
Takes in an array of dataframes and subtracts the min_sr value from the sr column in each dataframe.
'''
def adjustSR(dataFrames, min_sr):
    for df in dataFrames:
        df["sr"] = df["sr"] - min_sr
    
'''Function to import data from the data folder, using a given folder name. It returns an array of dataframes with the data in the folders.'''
def importData(folder_name):
    
    '''Data folder'''
    basePath = folder_name
    
    dataFrames = []
    '''Min SR is set to something ridiculously high to make sure that the first value is always lower'''    
    min_sr = 100000
    
    '''Iterating through all files in the folder, and adding the data to the dataFrames array'''
    for file in os.listdir(basePath):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath, file)
            ds = xr.open_dataset(filePath, engine='netcdf4') #Opening the file using xarray
            df = ds.to_dataframe().reset_index()
            min_sr = min(min_sr, df["sr"].min()) #Finding the minimum sr value of all the data
            # Extract date from filename like "CYGNSS_20220101_20220103"
            date_str = file.split("_")[1]
            df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
            dataFrames.append(df)
    '''Running the adjustSR function'''
    adjustSR(dataFrames, min_sr)
    dataFrames.sort(key=lambda df: df["date"].iloc[0])
    return dataFrames


def process_frame(df, step_size_lon, step_size_lat, sigma):
    # Filter
    df = df[(df['ddm_snr'] > 2) & (df['sp_rx_gain'] < 13)]

    if df.empty:
        return None, None, None

    latitudes = df["sp_lat"].values
    longitudes = df["sp_lon"].values
    sr_values = df["sr"].values

    lat_edges = np.arange(latitudes.min(), latitudes.max() + step_size_lat, step_size_lat)
    lon_edges = np.arange(longitudes.min(), longitudes.max() + step_size_lon, step_size_lon)

    stat, _, _, _ = binned_statistic_2d(
        longitudes, latitudes, sr_values, statistic='mean', bins=[lon_edges, lat_edges]
    )
    stat = stat.T

    # Interpolate missing values
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)

    mask = ~np.isnan(stat)
    if np.any(mask):
        points = np.column_stack((lon_mesh[mask], lat_mesh[mask]))
        values = stat[mask]
        stat = griddata(points, values, (lon_mesh, lat_mesh), method='nearest')
    else:
        return None, None, None

    if sigma > 0:
        stat = gaussian_filter(stat, sigma=sigma)

    return stat, lon_edges, lat_edges

def animate_cygnss(folder_name, sigma=1, step_size_lon=0.25, step_size_lat=0.25, smooth=True):
    dfs = importData(folder_name)
    all_frames = []
    dates = []
    lon_edges = lat_edges = None  # Placeholder for later

    for df in dfs:
        date = None
        if "date" in df.columns:
            date = df["date"].iloc[0]
        stat, le, lae = process_frame(df, step_size_lon, step_size_lat, sigma)

        if stat is not None:
            all_frames.append(stat)
            dates.append(date)
            if lon_edges is None or lat_edges is None:
                lon_edges, lat_edges = le, lae  # Set extent only once

    if not all_frames:
        print("No valid data to animate.")
        return

    # === Create Animation ===
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]]
    im = ax.imshow(all_frames[0], origin='lower', extent=extent, cmap='viridis', interpolation='bilinear' if smooth else 'nearest')
    title = ax.set_title(f"CYGNSS SR - {dates[0].strftime('%Y-%m-%d') if dates[0] else 'Unknown'}")
    plt.colorbar(im, ax=ax, label='SR')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    def update(frame):
        im.set_array(all_frames[frame])
        title.set_text(f"CYGNSS SR - {dates[frame].strftime('%Y-%m-%d') if dates[frame] else 'Unknown'}")
        return [im, title]

    
    
    ani = animation.FuncAnimation(fig, update, frames=len(all_frames), blit=True)
    ani.save("cygnss_animation6.gif", writer="pillow", fps=4)
    plt.close()
    print("Animation saved as 'cygnss_animation.gif'")





animate_cygnss(folder_name="data/Timeseries/TimeSeries-Pakistan-20220101-20241231/CYGNSS",  # change to your actual folder path
        sigma=1.5,             # set to 0 to turn off smoothing
        step_size_lon=0.05,
        step_size_lat=0.05,
        smooth=False)

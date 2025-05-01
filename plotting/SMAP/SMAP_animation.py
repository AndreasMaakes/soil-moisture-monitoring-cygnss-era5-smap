import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from SMAP_utils import SMAP_averaging_soil_moisture



import os
import xarray as xr

'''Function to import data from the data folder, using a given folder name. It returns an array of dataframes with the data in the folders.'''
def importDataSMAP(Timeseries, folder_name):
    basePath = folder_name
    dataFrames = []

    for file in os.listdir(basePath):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath, file)
            ds = xr.open_dataset(filePath, engine='netcdf4')
            df = ds.to_dataframe().reset_index()
            if df.empty:
                continue  # Skip empty DataFrames
            date_str = file.split("_")[1]
            df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
            df.name = file  # Save filename for later use
            dataFrames.append(df)

    # Filter out empty dataframes again just in case
    dataFrames = [df for df in dataFrames if not df.empty and "date" in df.columns]

    # Safe sorting
    dataFrames.sort(key=lambda df: df["date"].iloc[0])
    return dataFrames


def process_smap_frame(df, step_size_lon, step_size_lat, sigma):
    if df.empty:
        return None, None, None

    df = SMAP_averaging_soil_moisture(df)

    latitudes = df["latitude"].values
    longitudes = df["longitude"].values
    moisture_values = df["soil_moisture_avg"].values

    lat_edges = np.arange(latitudes.min(), latitudes.max(), step_size_lat)
    if lat_edges.size == 0 or lat_edges[-1] < latitudes.max():
        lat_edges = np.append(lat_edges, latitudes.max())

    lon_edges = np.arange(longitudes.min(), longitudes.max(), step_size_lon)
    if lon_edges.size == 0 or lon_edges[-1] < longitudes.max():
        lon_edges = np.append(lon_edges, longitudes.max())

    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    moisture_grid = griddata(
        (longitudes, latitudes),
        moisture_values,
        (lon_grid, lat_grid),
        method="linear"
    )

    if np.isnan(moisture_grid).all():
        return None, None, None

    # Fill missing values using nearest neighbor
    nan_mask = np.isnan(moisture_grid)
    if np.any(nan_mask):
        moisture_grid[nan_mask] = griddata(
            (longitudes, latitudes),
            moisture_values,
            (lon_grid, lat_grid),
            method="nearest"
        )[nan_mask]

    if sigma > 0:
        moisture_grid = gaussian_filter(moisture_grid, sigma=sigma)

    return moisture_grid, lon_edges, lat_edges


def animate_smap(folder_name, sigma=1, step_size_lon=0.25, step_size_lat=0.25, smooth=True):
    dfs = importDataSMAP(Timeseries=True, folder_name=folder_name)
    all_frames = []
    dates = []
    lon_edges = lat_edges = None

    for df in dfs:
        # Extract date from filename (e.g., SMAP_20220101_20220103)
        filename = getattr(df, "name", "")
        if "_" in filename:
            date_str = filename.split("_")[1]
            try:
                date = pd.to_datetime(date_str, format="%Y%m%d")
            except ValueError:
                date = None
        else:
            date = None

        stat, le, lae = process_smap_frame(df, step_size_lon, step_size_lat, sigma)

        if stat is not None:
            all_frames.append(stat)
            dates.append(date)
            if lon_edges is None or lat_edges is None:
                lon_edges, lat_edges = le, lae

    if not all_frames:
        print("No valid SMAP data to animate.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]]
    im = ax.imshow(all_frames[0], origin='lower', extent=extent, cmap='viridis', interpolation='bilinear' if smooth else 'nearest')
    title = ax.set_title(f"SMAP Soil Moisture - {dates[0].strftime('%Y-%m-%d') if dates[0] else 'Unknown'}")
    plt.colorbar(im, ax=ax, label='Soil Moisture')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    def update(frame):
        im.set_array(all_frames[frame])
        title.set_text(f"SMAP Soil Moisture - {dates[frame].strftime('%Y-%m-%d') if dates[frame] else 'Unknown'}")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(all_frames), blit=True)
    ani.save("smap_animation2.gif", writer="pillow", fps=3)
    plt.close()
    print("Animation saved as 'smap_animation.gif'")


animate_smap(folder_name="data/Timeseries/TimeSeries-Pakistan-20220101-20241231/SMAP",  # change to your actual folder path
        sigma=1.5,             # set to 0 to turn off smoothing
        step_size_lon=0.05,
        step_size_lat=0.05,
        smooth=False)

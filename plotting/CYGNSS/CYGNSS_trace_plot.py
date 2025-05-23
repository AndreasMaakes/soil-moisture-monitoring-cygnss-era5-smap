import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
from .import_data import importData
import matplotlib.ticker as mticker
import contextily as ctx
import geopandas as gpd
from shapely.geometry import box




# High-res satellite tiles from ESRI
class ESRIImagery(cimgt.GoogleTiles):
    def __init__(self):
        super().__init__()
        self.tile_size = 256
        self.max_zoom = 19
        self.min_zoom = 1

    def _image_url(self, tile):
        x, y, z = tile
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def CYGNSS_raw_plot_satellite(folder_name):
    title = f'CYGNSS Surface Reflectivity - Ground Tracks'

    # Import and concatenate data
    dfs = importData(folder_name)
    df = pd.concat(dfs)

    # Filtering
    df = df[df['ddm_snr'] >= 2]
    df = df[df['sp_rx_gain'] <= 13]
    df = df[df['sp_rx_gain'] >= 0]
    df = df[df["sp_inc_angle"] <= 45]

    # Coordinates and values
    lons = df["sp_lon"].values
    lats = df["sp_lat"].values
    sr_values = df["sr"].values

    # Define map bounds with buffer
    buffer = 1
    min_lon, max_lon = np.min(lons) - buffer, np.max(lons) + buffer
    min_lat, max_lat = np.min(lats) - buffer, np.max(lats) + buffer

    # Setup satellite basemap
    tiler = ESRIImagery()
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=tiler.crs)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.add_image(tiler, 8)

    # Optional: national borders
    #ax.add_feature(cfeature.BORDERS, edgecolor='white', linewidth=0.75)
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.75)

    # Scatter plot of raw SR data
    sc = ax.scatter(lons, lats, c=sr_values, cmap='viridis', s=10,
                    transform=ccrs.PlateCarree(), alpha=1.0)

    # Colorbar and title
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('CYGNSS SR (dB)', fontsize=22)
    cbar.ax.tick_params(labelsize=22)

    ax.set_title(title, fontsize=34, pad=20)

    # Optional: gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 24}
    gl.ylabel_style = {'size': 24}

    plt.show()



def CYGNSS_raw_plot(folder_name):
    # Title
    title = "CYGNSS SR Ground Tracks - Pakistan - January 2020"

    # Import and concatenate data
    dfs = importData(folder_name)
    df = pd.concat(dfs)

    # Apply same filters as average plot
    df = df[df['ddm_snr'] >= 1]
    df = df[df['sp_rx_gain'].between(0, 13)]
    df = df[df["sp_inc_angle"] <= 65]

    # Extract arrays
    lons = df["sp_lon"].values
    lats = df["sp_lat"].values
    sr  = df["sr"].values

    # Figure limits (with a tiny buffer if you like)
    buffer = 0.05  # degrees
    lon_min, lon_max = lons.min() - buffer, lons.max() + buffer
    lat_min, lat_max = lats.min() - buffer, lats.max() + buffer

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(
        lons, lats,
        c=sr,
        cmap='viridis',
        s=17,
        alpha=1.0,
        edgecolors='none'
    )

    # Title and colorbar
    ax.set_title(title, fontsize=32, pad=30)
    ax.set_xlabel("Longitude", fontsize=32, labelpad=20)
    ax.set_ylabel("Latitude", fontsize=32, labelpad=20)
    cbar = fig.colorbar(sc, ax=ax, pad=0.04, shrink=0.7)
    cbar.set_label("SR [dB]", fontsize=32, labelpad=24)
    cbar.ax.tick_params(labelsize=28)

    # Tick formatting
    ax.tick_params(axis='x', labelsize=28, pad=10)
    ax.tick_params(axis='y', labelsize=28, pad=10)

    def format_lon(x, _):
        return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"
    def format_lat(y, _):
        return f"{abs(y):.0f}°{'N' if y >= 0 else 'S'}"

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))

    # Geo‐aspect and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    fig.tight_layout()
    plt.show()
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from shapely.geometry import box
from .import_data import importData  # adjust import path as needed

# High-res ESRI World Imagery tile source
class ESRIImagery(cimgt.GoogleTiles):
    def __init__(self):
        super().__init__()
        self.tile_size = 256
        self.max_zoom = 19
        self.min_zoom = 1

    def _image_url(self, tile):
        x, y, z = tile
        return (
            f"https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
        )

def plot_satellite_only(folder_name, zoom_level=8):
    # --- 1) load your CYGNSS points to get the bounds ---
    dfs = importData(folder_name)
    df  = pd.concat(dfs)

    # same filters as your CYGNSS_raw_plot
    df = df[df['ddm_snr'] >= 2]
    df = df[df['sp_rx_gain'].between(0, 13)]
    df = df[df['sp_inc_angle'] <= 45]

    lons = df['sp_lon'].values
    lats = df['sp_lat'].values

    # buffer in degrees
    buffer = 1.0
    min_lon, max_lon = lons.min() - buffer, lons.max() + buffer
    min_lat, max_lat = lats.min() - buffer, lats.max() + buffer

    # --- 2) set up the map ---
    tiler = ESRIImagery()
    fig  = plt.figure(figsize=(12, 8))
    ax   = plt.axes(projection=tiler.crs)

    # restrict to your study area's lon/lat
    ax.set_extent([min_lon, max_lon, min_lat, max_lat],
                  crs=ccrs.PlateCarree())

    # draw only the satellite tiles
    ax.add_image(tiler, zoom_level)

    # optional: coastlines/borders for reference
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.75)

    ax.set_title("ESRI World Imagery – Study Area", fontsize=24, pad=15)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

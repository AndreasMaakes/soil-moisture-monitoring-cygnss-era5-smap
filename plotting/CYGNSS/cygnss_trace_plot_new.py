import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
from .import_data import importData

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
    df = df[df['ddm_snr'] >= 0]
    df = df[df['sp_rx_gain'] <= 13]
    df = df[df['sp_rx_gain'] >= 0]
    df = df[df["sp_inc_angle"] <= 65]

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

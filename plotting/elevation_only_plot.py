import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

from ERA5.ERA5_utils import apply_land_sea_mask, averaging_soil_moisture
from CYGNSS.import_data import importData


def terrain_3d_plot(cygnss_folder, era5_folder, lsm_threshold, dem_file, lat_step, lon_step):
    
    lat_min, lat_max = 25, 28.5
    lon_min, lon_max = 67, 73

    # --- Create fine grid ---
    interp_factor = 1
    fine_lat_step = lat_step / interp_factor
    fine_lon_step = lon_step / interp_factor
    lat_fine = np.arange(lat_min, lat_max + fine_lat_step, fine_lat_step)
    lon_fine = np.arange(lon_min, lon_max + fine_lon_step, fine_lon_step)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)

    # --- Load and interpolate DEM ---
    dem_ds = xr.open_dataset(dem_file)
    #print(dem_ds)
    elevation = dem_ds["__xarray_dataarray_variable__"].values
    print("Elevation min:", np.nanmin(elevation))
    print("Elevation max:", np.nanmax(elevation))
    dem_lon = dem_ds["x"].values
    dem_lat = dem_ds["y"].values
    dem_lon_mesh, dem_lat_mesh = np.meshgrid(dem_lon, dem_lat)

    print("lon_mesh shape:", lon_mesh.shape)
    print("dem_lon shape:", dem_lon.shape)
    print("elevation shape:", elevation.shape)

    dem_fine = griddata(
    (dem_lon_mesh.flatten(), dem_lat_mesh.flatten()),
    elevation.flatten(),
    (lon_mesh, lat_mesh),
    method='linear'  # or 'linear'
)

    # --- Plot 3D Terrain ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        lon_mesh, lat_mesh, dem_fine,
        cmap='terrain',
        linewidth=0, antialiased=True,
        edgecolor="none"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Elevation (m)")
    ax.set_title("3D Terrain Elevation")
    plt.tight_layout()
    plt.show()

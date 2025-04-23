import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from .ERA5_utils import averaging_soil_moisture, apply_land_sea_mask

def ERA5_regrid_and_blur(folder_name: str,
                         sigma: float,
                         lsm_threshold: float,
                         lat_step: float,
                         lon_step: float):
    """
    - folder_name: the .nc filename under data/ERA5
    - sigma: gaussian blur
    - lsm_threshold: keep grid‐cells where mask ≥ this
    - lat_step, lon_step: your desired output resolution (degrees)
    """

    # 1) Load & melt to DataFrame
    ds = xr.open_dataset(f"data/ERA5/{folder_name}", engine="netcdf4")
    df = ds.to_dataframe().reset_index()

    # 2) Average over time and compute per‐point mask & moisture
    avg = averaging_soil_moisture(df)
    # (avg has columns: latitude, longitude, lsm, average_moisture)

    # 3) Build your output lat/lon grid
    lats = avg["latitude"].values
    lons = avg["longitude"].values
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    lat_new = np.arange(lat_min, lat_max + lat_step, lat_step)
    lon_new = np.arange(lon_min, lon_max + lon_step, lon_step)
    lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)

    # 4) Interpolate water‐content and mask separately
    moisture_vals = avg["average_moisture"].values
    lsm_vals      = avg["lsm"].values

    moisture_grid = griddata(
        (lons, lats),
        moisture_vals,
        (lon_grid, lat_grid),
        method="linear"
    )
    lsm_grid = griddata(
        (lons, lats),
        lsm_vals,
        (lon_grid, lat_grid),
        method="nearest"
    )

    # 5) Now blank out the sea
    moisture_grid[lsm_grid < lsm_threshold] = np.nan

    # 6) Gaussian‐blur the result
    smoothed = gaussian_filter(moisture_grid, sigma=sigma)

    # 7) Plot
    plt.figure(figsize=(10, 8))
    mesh = plt.pcolormesh(
        lon_grid, lat_grid, smoothed,
        shading="auto", cmap="viridis"
    )
    plt.colorbar(mesh, label="Soil Moisture (Smoothed)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(
        f"Regridded & Smoothed Soil Moisture\n"
        f"Grid: {lat_step}°×{lon_step}°, σ={sigma}, mask_thr={lsm_threshold}"
    )
    plt.axis("equal")
    plt.show()

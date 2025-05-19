import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter



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

    # 2.5) Filter geospatially before interpolation
    lat_min_bound, lat_max_bound = -30.25, -28.25
    lon_min_bound, lon_max_bound = 118.5, 120.5

    avg = avg[
        (avg["latitude"] >= lat_min_bound) & (avg["latitude"] <= lat_max_bound) &
        (avg["longitude"] >= lon_min_bound) & (avg["longitude"] <= lon_max_bound)
    ]

    # 3) Build your output lat/lon grid
    lats = avg["latitude"].values
    lons = avg["longitude"].values
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    lat_new = np.round(np.linspace(lat_min, lat_max, int((lat_max - lat_min) / lat_step) + 1), 6)
    lon_new = np.round(np.linspace(lon_min, lon_max, int((lon_max - lon_min) / lon_step) + 1), 6)
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
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot using pcolormesh
    mesh = ax.pcolormesh(
        lon_grid, lat_grid, smoothed,
        shading="auto", cmap="viridis"
    )

    # Colorbar (beside or below)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.03)
    cbar.set_label("Soil Moisture [m$^3$/m$^3$]", fontsize=32, labelpad=26)
    cbar.ax.tick_params(labelsize=28)

    # Axis labels
    ax.set_xlabel("Longitude", fontsize=32, labelpad=20)
    ax.set_ylabel("Latitude", fontsize=32, labelpad=20)

    # Tick font size and padding
    ax.tick_params(axis='x', labelsize=28, pad=10)
    ax.tick_params(axis='y', labelsize=28, pad=10)

    ax.set_title("ERA5 SM - Lake Barlee - January & February 2020", fontsize=32, pad=30)
    ax.set_aspect('equal', adjustable='box')


    # Optional: Tick spacing every 1 degree
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    def format_lon(x, _):
        direction = 'E' if x >= 0 else 'W'
        return f"{abs(x):.1f}°{direction}"

    def format_lat(y, _):
        direction = 'N' if y >= 0 else 'S'
        return f"{abs(y):.1f}°{direction}"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Final layout adjustments
    fig.tight_layout()
    plt.show()



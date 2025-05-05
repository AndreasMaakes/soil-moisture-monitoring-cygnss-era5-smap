import matplotlib.pyplot as plt
import numpy as np
from CYGNSS.import_data import importData

def trace_plot_2(folder_name, saveplot=False):
    name = folder_name.split("/")[0]
    title = f'CYGNSS Ground Tracks – {name} – September 2024'

    # Import data
    dataFrames = importData(folder_name)
    lats = np.array([])
    lons = np.array([])

    for df in dataFrames:
        lats = np.append(lats, df['sp_lat'].to_numpy())
        lons = np.append(lons, df['sp_lon'].to_numpy())

    # Determine bounds
    min_lat, max_lat = np.min(lats), np.max(lats)
    min_lon, max_lon = np.min(lons), np.max(lons)

    # Approximate 9 km in degrees
    grid_deg = 9 / 111  # ~0.081°

    # Snap axis bounds to grid
    lat_start = np.floor(min_lat / grid_deg) * grid_deg
    lat_end = np.ceil(max_lat / grid_deg) * grid_deg
    lon_start = np.floor(min_lon / grid_deg) * grid_deg
    lon_end = np.ceil(max_lon / grid_deg) * grid_deg

    # Create grid ticks
    lat_ticks = np.arange(lat_start, lat_end + grid_deg, grid_deg)
    lon_ticks = np.arange(lon_start, lon_end + grid_deg, grid_deg)

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(lons, lats, s=20, color='steelblue', alpha=0.7)

    # Emphasized gridlines
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.grid(True, linestyle='-', linewidth=1.2, color='gray', alpha=0.5)

    # Set tight bounding box
    ax.set_xlim(lon_start, lon_end)
    ax.set_ylim(lat_start, lat_end)

    # Remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)

    # Title only
    ax.set_title(title, fontsize=18, pad=20)

    # Save or show
    if saveplot:
        plt.savefig(f'plotting/plots/{folder_name}_9kmgrid_clean.png', dpi=300, bbox_inches='tight')
    plt.show()

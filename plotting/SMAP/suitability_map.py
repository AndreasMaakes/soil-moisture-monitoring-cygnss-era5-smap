import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .SMAP_import_data import importDataSMAP
from .SMAP_utils import SMAP_averaging_soil_moisture
import pandas as pd

def SMAP_surface_flags_suitability(
    folder_name,
    step_size_lon,
    step_size_lat,
    sigma=None,
    weights=None
):
    """
    Compute & plot gridded suitability from SMAP data:
    - Based on roughness, vegetation opacity, static water body fraction
    - Also includes binary urban and mountainous flags
    - All inputs inverted to produce suitability score (0–1)
    """
    # 1) Load & average
    df_list = importDataSMAP(False, folder_name)
    df = pd.concat(df_list, ignore_index=True)
    df = SMAP_averaging_soil_moisture(df)

    # 2) Extract bit flags (urban, mountainous)
    flags = df['surface_flag'].astype(int).values
    df['urban'] = (flags >> 3) & 1
    df['mountainous'] = (flags >> 9) & 1

    # 3) Extract continuous variables (already in the dataframe)
    # Remove NaNs just in case
    df = df.dropna(subset=[
        'urban', 'mountainous', 'vegetation_opacity',
        'static_water_body_fraction', 'roughness_coefficient'
    ])

    # 4) Normalize continuous variables (0 = low, 1 = high) then invert
    for col in ['vegetation_opacity', 'static_water_body_fraction', 'roughness_coefficient']:
        vmin, vmax = df[col].min(), df[col].max()
        df[col + '_norm'] = 1 - ((df[col] - vmin) / (vmax - vmin))

    # 5) Invert binary flags
    df['urban_inv'] = 1 - df['urban']
    df['mountainous_inv'] = 1 - df['mountainous']

    # 6) Combine into a score
    features = [
        'urban_inv',
        'mountainous_inv',
        'vegetation_opacity_norm',
        'static_water_body_fraction_norm',
        'roughness_coefficient_norm'
    ]
    if weights is None:
        weights = np.ones(len(features)) / len(features)
    else:
        weights = np.array(weights, float)
        weights /= weights.sum()

    df['suitability'] = df[features].values.dot(weights)

    # 7) Grid definition
    lons = df['longitude'].values
    lats = df['latitude'].values
    scores = df['suitability'].values

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    lon_edges = np.arange(lon_min, lon_max + step_size_lon, step_size_lon)
    lat_edges = np.arange(lat_min, lat_max + step_size_lat, step_size_lat)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    # 8) Interpolate score
    score_grid = griddata(
        (lons, lats), scores, (lon_grid, lat_grid), method='linear'
    )

    # 9) Mask high static water body fraction as "water"
    water_grid = griddata(
        (lons, lats), 1 - df['static_water_body_fraction_norm'].values,
        (lon_grid, lat_grid), method='nearest'
    )
    mask = (water_grid < 0.5)  # low "non-water" value = high actual water fraction
    score_masked = np.ma.masked_where(mask, score_grid)

    # 10) Optional smoothing
    if sigma is not None:
        score_masked = gaussian_filter(score_masked, sigma=sigma)

    # 11) Plot
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')

    mesh = ax.pcolormesh(
        lon_edges, lat_edges, score_masked,
        shading='auto', cmap='RdYlGn', vmin=0, vmax=1,
        transform=ccrs.PlateCarree()
    )
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', label='Suitability score (0–1)')

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(f"New SMAP Suitability Score for {folder_name}")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.show()

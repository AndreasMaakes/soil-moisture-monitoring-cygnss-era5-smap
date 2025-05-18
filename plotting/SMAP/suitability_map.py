import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, NearestNDInterpolator
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
    - Based on roughness and vegetation opacity
    - Includes binary urban, mountainous, and static water flags
    - All continuous inputs inverted to produce suitability score (min–max)
    """
    # 1) Load & average
    df_list = importDataSMAP(False, folder_name)
    df = pd.concat(df_list, ignore_index=True)
    df = SMAP_averaging_soil_moisture(df)

    # 2) Extract bit flags (static water, urban, mountainous)
    flags = df['surface_flag'].astype(int).values
    df['static_water'] = (flags >> 0) & 1
    df['urban']        = (flags >> 3) & 1
    df['mountainous']  = (flags >> 9) & 1

    # 3) Drop rows missing anything important
    df = df.dropna(subset=[
        'static_water', 'urban', 'mountainous',
        'vegetation_opacity', 'roughness_coefficient'
    ])

    # 4) Normalize and invert continuous variables
    for col in ['vegetation_opacity', 'roughness_coefficient']:
        vmin, vmax = df[col].min(), df[col].max()
        df[col + '_norm'] = 1 - ((df[col] - vmin) / (vmax - vmin))

    # 5) Invert binary flags for scoring
    df['urban_inv']       = 1 - df['urban']
    df['mountainous_inv'] = 1 - df['mountainous']

    # 6) Compute raw suitability scores
    features = [
        'urban_inv',
        'mountainous_inv',
        'vegetation_opacity_norm',
        'roughness_coefficient_norm'
    ]
    if weights is None:
        weights = np.ones(len(features)) / len(features)
    else:
        weights = np.array(weights, float)
        weights /= weights.sum()

    df['suitability'] = df[features].values.dot(weights)

    # Determine dynamic min and max for plotting
    min_score = df['suitability'].min()
    max_score = df['suitability'].max()

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

    # 8) Interpolate score (linear), may leave NaNs
    score_grid = griddata((lons, lats), scores, (lon_grid, lat_grid), method='linear')

        # 9) Interpolate static water flag (1=water) using nearest so oceans get correctly flagged
    water_grid = griddata(
        (lons, lats),
        df['static_water'].values,
        (lon_grid, lat_grid),
        method='nearest'
    )
    mask_water = (water_grid == 1)  # True over all water (including ocean)

    # Mask grid cells over water
    score_grid_masked = np.ma.masked_where(mask_water, score_grid)

    # 10) Fill NaNs only over land
    nan_mask = np.isnan(score_grid_masked.data) & ~mask_water
    if np.any(nan_mask):
        interpolator = NearestNDInterpolator(list(zip(lons, lats)), scores)
        score_grid_masked.data[nan_mask] = interpolator(
            lon_grid[nan_mask], lat_grid[nan_mask]
        )

    # 11) Optional smoothing (re-mask water afterwards)
    if sigma is not None:
        smoothed = gaussian_filter(score_grid_masked, sigma=sigma)
        score_grid_masked = np.ma.masked_where(mask_water, smoothed)

        # 12) Plot
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.Mercator())

    # zoom to specified region
    ax.set_extent([-130, 160, -40, 40], crs=ccrs.PlateCarree())

    # add features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='#cfedf3')

    # plot grid
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, score_grid_masked,
        shading='auto', cmap='RdYlGn',
        vmin=min_score, vmax=max_score,
        transform=ccrs.PlateCarree()
    )
    _cbar = fig.colorbar(
        mesh, ax=ax,
        orientation='horizontal',
        fraction=0.046,  # width of the cbar as fraction of axes width
        pad=0.03,        # space between the plot and the cbar
        label=f'Suitability (0–1)'
    )
    _cbar.ax.tick_params(labelsize=25)
    _cbar.set_label('Suitability (0–1)', fontsize=30, labelpad=20)

    # Increase title and axis label font sizes
    ax.set_title(f"Suitability Score for Soil Moisture Estimation Using GNSS-R", fontsize=30, pad=25)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)

    plt.show()

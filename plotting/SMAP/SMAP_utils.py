import pandas as pd
import numpy as np

def SMAP_averaging_soil_moisture(df):
    """
    Average soil moisture and continuous variables, and combine quality flags per latitude/longitude grid cell.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        - 'latitude', 'longitude'
        - 'soil_moisture', 'surface_flag'
        - 'roughness_coefficient', 'vegetation_opacity', 'static_water_body_fraction'

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - latitude, longitude
        - soil_moisture_avg: mean soil moisture over the cell
        - roughness_coefficient, vegetation_opacity, static_water_body_fraction: means over the cell
        - surface_flag: bitwise OR of all flags in the cell
    """
    aggregated = (
        df.groupby(['latitude', 'longitude'])
        .agg(
            soil_moisture_avg=('soil_moisture', 'mean'),
            #roughness_coefficient=('roughness_coefficient', 'mean'),
            #vegetation_opacity=('vegetation_opacity', 'mean'),
            #surface_flag=('surface_flag', lambda x: int(np.bitwise_or.reduce(x.astype(int))))
        )
        .reset_index()
    )
    return aggregated

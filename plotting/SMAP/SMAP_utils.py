import pandas as pd
import numpy as np

def SMAP_averaging_soil_moisture(df):
    """
    Average soil moisture and combine quality flags per latitude/longitude grid cell.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'latitude', 'longitude', 'soil_moisture', and 'surface_flag'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - latitude, longitude
        - soil_moisture_avg: mean soil moisture over the cell
        - surface_flag: bitwise OR of all flags in the cell
    """
    # Group by location and compute mean soil moisture
    # and bitwise OR of the quality flags
    aggregated = (
        df.groupby(['latitude', 'longitude'])
        .agg(
            soil_moisture_avg=('soil_moisture', 'mean'),
            surface_flag=('surface_flag', lambda x: int(np.bitwise_or.reduce(x.astype(int))))
        )
        .reset_index()
    )
    return aggregated

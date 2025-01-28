import pandas as pd

def SMAP_averaging_soil_moisture(df):
    #Averaging the soil moisture values
    averaged_df = (
        df.groupby(['latitude', 'longitude'])
        .agg(soil_moisture_avg=('soil_moisture', 'mean'))
        .reset_index()
    )
    return averaged_df
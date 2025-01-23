import pandas as pd

def averaging_soil_moisture(df):

    df['valid_time'] = pd.to_datetime(df['valid_time'])

    averaged_df = (
        df.groupby(['latitude', 'longitude'])
        .agg(lsm=('lsm', 'mean'), average_moisture=('swvl1', 'mean'))
        .reset_index()
    )
    return averaged_df


def apply_land_sea_mask(df, lsm_threshold):
    df_filtered_land_sea_mask = df[(df["lsm"] >= lsm_threshold)]
    return df_filtered_land_sea_mask
import time
import earthaccess
import xarray as xr
import numpy as np
import h5py
import pandas as pd
from .data_filtering import data_filtering_SMAP
import os
from datetime import datetime, timedelta
from create_dates_array import create_dates_array
from dotenv import load_dotenv
from requests.exceptions import HTTPError

load_dotenv()

def scale(h5var):
    """Read raw bytes, apply CF scale_factor & add_offset (if present)."""
    raw = h5var[...].astype(float).flatten()
    sf  = h5var.attrs.get('scale_factor', 1.0)
    ao  = h5var.attrs.get('add_offset',   0.0)
    return raw * sf + ao

def data_fetching_smap(
    Timeseries: bool,
    startDate: str,
    endDate: str,
    max_lat: float,
    min_lat: float,
    max_lon: float,
    min_lon: float,
    name: str
):
    """
    Fetches SMAP data using Earthaccess with a 5-minute wait between retries if a timeout occurs.

    :param Timeseries: Boolean, if True, returns a combined DataFrame for all time intervals.
    :param startDate: Start date in "YYYYMMDD" format.
    :param endDate: End date in "YYYYMMDD" format.
    :param max_lat: Maximum latitude.
    :param min_lat: Minimum latitude.
    :param max_lon: Maximum longitude.
    :param min_lon: Minimum longitude.
    :param name: Name identifier for saving the files.
    :return: DataFrame if Timeseries is True, otherwise None.
    """

    earthaccess.login(strategy="environment")
    dates = create_dates_array(startDate, endDate, "smap")
    results = earthaccess.search_data(
        short_name='SPL3SMP_E',
        temporal=(dates[0], dates[-1]),
        count=-1,
        provider="NSIDC_CPRD"
    )
    dataset = earthaccess.open(results)

    df_timeseries = pd.DataFrame({})

    if not Timeseries:
        # Single output folder for all date files
        base_data_path = "data/SMAP"
        folder_name = f"{name}-{startDate}-{endDate}"
        output_folder = os.path.join(base_data_path, folder_name)
        os.makedirs(output_folder, exist_ok=True)

    for idx, ds in enumerate(dataset):
        print(f"Processing file: {ds}")
        with h5py.File(ds, 'r') as f:
            group_AM = f['Soil_Moisture_Retrieval_Data_AM']
            group_PM = f['Soil_Moisture_Retrieval_Data_PM']

            # AM arrays
            latitude_AM          = group_AM['latitude'][...].flatten()
            longitude_AM         = group_AM['longitude'][...].flatten()
            soil_moisture_AM     = group_AM['soil_moisture'][...].flatten()
            soil_moisture_dca_AM = group_AM['soil_moisture_dca'][...].flatten()
            surface_flag_AM      = group_AM['surface_flag'][...].flatten()
            roughness_AM          = scale(group_AM['roughness_coefficient'])
            vegetation_opacity_AM = scale(group_AM['vegetation_opacity'])
            static_water_fraction_AM = scale(group_AM['static_water_body_fraction'])

            # PM arrays (rename to same column name)
            latitude_PM          = group_PM['latitude_pm'][...].flatten()
            longitude_PM         = group_PM['longitude_pm'][...].flatten()
            soil_moisture_PM     = group_PM['soil_moisture_pm'][...].flatten()
            soil_moisture_dca_PM = group_PM['soil_moisture_dca_pm'][...].flatten()
            surface_flag_PM      = group_PM['surface_flag_pm'][...].flatten()
            roughness_PM          = scale(group_PM['roughness_coefficient_pm'])
            vegetation_opacity_PM = scale(group_PM['vegetation_opacity_pm'])
            static_water_fraction_PM = scale(group_PM['static_water_body_fraction_pm'])

            # Build DataFrames with consistent column names
            df_AM = pd.DataFrame({
                'latitude': latitude_AM,
                'longitude': longitude_AM,
                'soil_moisture': soil_moisture_AM,
                'soil_moisture_dca': soil_moisture_dca_AM,
                'surface_flag': surface_flag_AM,
                'roughness_coefficient': roughness_AM,
                'vegetation_opacity': vegetation_opacity_AM,
                'static_water_body_fraction': static_water_fraction_AM
            })

            df_PM = pd.DataFrame({
                'latitude': latitude_PM,
                'longitude': longitude_PM,
                'soil_moisture': soil_moisture_PM,
                'soil_moisture_dca': soil_moisture_dca_PM,
                'surface_flag': surface_flag_PM,
                'roughness_coefficient': roughness_PM,
                'vegetation_opacity': vegetation_opacity_PM,
                'static_water_body_fraction': static_water_fraction_PM
            })

            # Combine and filter
            df_combined = pd.concat([df_AM, df_PM], ignore_index=True)
            df_filtered = data_filtering_SMAP(
                df_combined, max_lat, min_lat, max_lon, min_lon
            ).reset_index(drop=True)

            if Timeseries:
                df_timeseries = pd.concat([df_timeseries, df_filtered], ignore_index=True)
            else:
                # Save each date to a single folder
                current_date = dates[idx]
                file_name = f"{name}_{current_date}.nc"
                ds_xr = xr.Dataset.from_dataframe(df_filtered)
                ds_xr.to_netcdf(os.path.join(output_folder, file_name))
                print(f"File {file_name} created in {output_folder}.")

    if Timeseries:
        return df_timeseries.reset_index(drop=True)

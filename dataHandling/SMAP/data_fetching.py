import os
import time
import earthaccess
import h5py
import xarray as xr
import numpy as np
import pandas as pd
from .data_filtering import data_filtering_SMAP
from create_dates_array import create_dates_array
from dotenv import load_dotenv

load_dotenv()

def scale(h5var):
    """
    Read raw HDF5 variable and flatten, masking out values <0 or >1 as NaN.
    """
    raw = h5var[...].astype(float).flatten()
    raw[(raw < 0) | (raw > 1)] = np.nan
    return raw


def data_fetching_smap(
    timeseries: bool,
    startDate: str,
    endDate: str,
    max_lat: float,
    min_lat: float,
    max_lon: float,
    min_lon: float,
    name: str
) -> pd.DataFrame:
    """
    Fetches SMAP Enhanced L3 data, filters out invalid roughness values,
    and returns a DataFrame (if timeseries=True) or writes NetCDF files.
    """
    earthaccess.login(strategy="environment")
    dates = create_dates_array(startDate, endDate, "smap")
    results = earthaccess.search_data(
        short_name='SPL3SMP_E',
        temporal=(dates[0], dates[-1]),
        count=-1,
        provider='NSIDC_CPRD'
    )
    dataset = earthaccess.open(results)

    df_timeseries = pd.DataFrame()

    if not timeseries:
        base_path = 'data/SMAP'
        folder = f"{name}-{startDate}-{endDate}"
        output_folder = os.path.join(base_path, folder)
        os.makedirs(output_folder, exist_ok=True)

    for idx, ds in enumerate(dataset):
        print(f"Processing file: {ds}")
        with h5py.File(ds, 'r') as f:
            am = f['Soil_Moisture_Retrieval_Data_AM']
            pm = f['Soil_Moisture_Retrieval_Data_PM']

            # AM arrays
            lat_AM = am['latitude'][...].flatten()
            lon_AM = am['longitude'][...].flatten()
            sm_AM  = am['soil_moisture'][...].flatten()
            sma_AM = am['soil_moisture_dca'][...].flatten()
            flag_AM = am['surface_flag'][...].flatten().astype(int)

            # Roughness: filter invalid
            rough_AM = scale(am['roughness_coefficient'])
            # Vegetation opacity: keep full range
            veg_AM = am['vegetation_opacity'][...].flatten()

            print(f"Filtered roughness range: {np.nanmin(rough_AM):.3f}–{np.nanmax(rough_AM):.3f}")

            df_AM = pd.DataFrame({
                'latitude': lat_AM,
                'longitude': lon_AM,
                'soil_moisture': sm_AM,
                'soil_moisture_dca': sma_AM,
                'surface_flag': flag_AM,
                'roughness_coefficient': rough_AM,
                'vegetation_opacity': veg_AM
            })

            # PM arrays
            lat_PM = pm['latitude_pm'][...].flatten()
            lon_PM = pm['longitude_pm'][...].flatten()
            sm_PM  = pm['soil_moisture_pm'][...].flatten()
            sma_PM = pm['soil_moisture_dca_pm'][...].flatten()
            flag_PM = pm['surface_flag_pm'][...].flatten().astype(int)

            rough_PM = scale(pm['roughness_coefficient_pm'])
            veg_PM = pm['vegetation_opacity_pm'][...].flatten()

            df_PM = pd.DataFrame({
                'latitude': lat_PM,
                'longitude': lon_PM,
                'soil_moisture': sm_PM,
                'soil_moisture_dca': sma_PM,
                'surface_flag': flag_PM,
                'roughness_coefficient': rough_PM,
                'vegetation_opacity': veg_PM
            })

            # Combine and geographic filter
            df_all = pd.concat([df_AM, df_PM], ignore_index=True)
            df_filtered = data_filtering_SMAP(
                df_all, max_lat, min_lat, max_lon, min_lon
            ).reset_index(drop=True)

            if timeseries:
                df_timeseries = pd.concat([df_timeseries, df_filtered], ignore_index=True)
            else:
                date = dates[idx]
                fname = f"{name}_{date}.nc"
                xr.Dataset.from_dataframe(df_filtered).to_netcdf(
                    os.path.join(output_folder, fname)
                )
                print(f"Saved {fname} to {output_folder}")

    return df_timeseries.reset_index(drop=True) if timeseries else None

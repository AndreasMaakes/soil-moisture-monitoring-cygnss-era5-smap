from SMAP.data_fetching import data_fetching_smap
from ERA5.data_fetching import data_fetching_era5
from CYGNSS.data_fetching import data_fetching_CYGNSS
import os
import xarray as xr


def data_fetching_time_series(startDate, endDate, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain):

    basePath = f'data/TimeSeries/TimeSeries-{name}-{startDate}-{endDate}'

    if not os.path.exists(basePath):
            try:
                os.makedirs(basePath)  # Create base directory if it doesn't exist
                os.mkdir(os.path.join(basePath, "CYGNSS"))
                os.mkdir(os.path.join(basePath, "SMAP"))
                os.mkdir(os.path.join(basePath, "ERA5"))
                print(f"Directory {basePath} created successfully.")
            except OSError:
                print(f"Creation of the directory {basePath} failed.")
            else:
                print(f"Directory {basePath} already exists.")
                
    df_cyg = data_fetching_CYGNSS(True, startDate, endDate, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)
    ds_cyg = xr.Dataset.from_dataframe(df_cyg)
    ds_cyg.to_netcdf(f'{basePath}/CYGNSS/CYGNSS_{startDate}_{endDate}.nc')

    #data_fetching_era5(True, startDate,endDate, min_lat, max_lat, min_lon, max_lon, name)
    #data_fetching_smap(True, startDate, endDate,  max_lat, min_lat, max_lon, min_lon, name)            
    
    
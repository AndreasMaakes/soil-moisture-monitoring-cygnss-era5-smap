from SMAP.data_fetching import data_fetching_smap
from ERA5.data_fetching import data_fetching_era5
from CYGNSS.data_fetching import data_fetching_CYGNSS
import os
import xarray as xr
import datetime



def generate_weekly_intervals(start_date, end_date, num_days):
    """
    Generates a list of (startDate, endDate) tuples where each range consists of 'num_days' consecutive days.
    The intervals repeat weekly until 'end_date' is reached.

    :param start_date: Start date in "YYYY-MM-DD" format
    :param end_date: End date in "YYYY-MM-DD" format
    :param num_days: Number of consecutive days to fetch data
    :return: List of (start_date, end_date) tuples
    """
    start = datetime.datetime.strptime(start_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")
    
    intervals = []
    
    current_start = start
    while current_start <= end:
        current_end = current_start + datetime.timedelta(days=num_days - 1)

        # Ensure the end date does not exceed the overall limit
        if current_end > end:
            break
        
        intervals.append((current_start.strftime("%Y%m%d"), current_end.strftime("%Y%m%d")))

        # Move to the same weekday next week
        current_start += datetime.timedelta(weeks=2)

    return intervals


def data_fetching_time_series(startDate, endDate, num_days, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain):
    
    '''Basepath to save the timeseries data in a seperate folder'''
    basePath = f'data/TimeSeries/TimeSeries-{name}-{startDate}-{endDate}'
    
    '''Creating the folders if they don't exist'''
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
                     
    '''Generate weekly intervals based on user input'''
    intervals = generate_weekly_intervals(startDate, endDate, num_days)
    
    for days in intervals:
            
        start_date = days[0]

        if len(days) == 1:
            end_date = start_date
        else:
            end_date = days[1]
       
        '''
         #SMAP data fetching

         #Fetching the data from the current run as a dataframe
        df_cyg = data_fetching_CYGNSS(True, start_date, end_date, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)
        #Converting the dataframe to a xarray dataset
        ds_cyg = xr.Dataset.from_dataframe(df_cyg)    
        #Saving the data to a NETCDF4 file in the correct folder
        ds_cyg.to_netcdf(f'{basePath}/CYGNSS/CYGNSS_{start_date}_{end_date}.nc')
        
        
        df_SMAP = data_fetching_smap(True, start_date, end_date,  max_lat, min_lat, max_lon, min_lon, name)
        ds_SMAP = xr.Dataset.from_dataframe(df_SMAP)
        ds_SMAP.to_netcdf(f'{basePath}/SMAP/SMAP_{start_date}_{end_date}.nc')
        '''
        
        #ERA5 data fetching

        data_fetching_era5(True, start_date, end_date, min_lat, max_lat, min_lon, max_lon, name, basePath)
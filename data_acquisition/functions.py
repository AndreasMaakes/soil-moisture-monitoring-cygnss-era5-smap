from webob.exc import HTTPError  # Import HTTPError to catch 404 errors
from pydap.client import open_url
from pydap.cas.urs import setup_session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import xarray as xr


'''
This functions lets the user input the from and to date, and returns the date in the format that the API requires.

The date format is as follows: YYYYMMDD, for example 20240701
'''

from datetime import datetime, timedelta

def create_dates_array(startDate: str, endDate: str):
    # Convert string dates to datetime objects
    start = datetime.strptime(startDate, "%Y%m%d")
    end = datetime.strptime(endDate, "%Y%m%d")
    
    # Generate all dates in the range
    dates = []
    current_date = start
    while current_date <= end:
        # Append the date in the desired format
        dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    return dates


'''
This function filters the data based on geographical location, quality flag and inicident angle 
quality flags 2, 4, 5, 8, 16, and 17
'''
def data_filtering(df, max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float):
    
    df_filtered_spatial = df[(df["sp_lon"] >= min_lon) & (df["sp_lon"] <= max_lon) & (df["sp_lat"] >= min_lat) & (df["sp_lat"] <= max_lat)]
    df_filtered_inclination = df_filtered_spatial[(df["sp_inc_angle"] <= inc_angle)]
    
    # Bitmasks to exclude rows with certain quality flags set
    bitmask_exclude = (
        0x00000002 |  # S-Band powered up (qf 2)
        0x00000008 |  # Small SC attitude error (qf 4)
        0x00000010 |  # Black body DDM (qf 5)
        0x00000080 |  #  ddm_is_test_pattern (qf 8)
        0x00008000 |  #  direct_signal_in_ddm (qf 15)
        0x00010000    # low_confdence_gps_eirp_estimate (qf 16)
        
    )
    
    # Use bitwise AND to filter out rows with these quality flag bits set
    df_filtered_qf = df_filtered_inclination[(df_filtered_inclination["quality_flags"] & bitmask_exclude) == 0]
    
    
    
    return df_filtered_qf



'''This function downloads the data'''

def data_fetching(startDate: str, endDate: str, username: str, password: str, max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float):
    dates = create_dates_array(startDate, endDate)
    
    # List of satellite identifiers
    satellites = [f'cyg0{i}' for i in range(1, 9)]

# Iterate over each satellite and date
    for sat in tqdm(satellites):
        for date in tqdm(dates):
            try:
                # Construct the URL for the current satellite and date
                url = f"https://opendap.earthdata.nasa.gov/collections/C2832195379-POCLOUD/granules/{sat}.ddmi.s{date}-000000-e{date}-235959.l1.power-brcs.a32.d33"
            
                # Attempt to open the dataset
                dataset = open_url(url, session=setup_session(username, password), protocol='dap4')

                # Fetch data from the dataset 
                print("Fetching data - 0/11 completed")
                ddm_snr = np.array(dataset['ddm_snr'][:, 0]).ravel()
                print("ddm_snr fetched - 1/11 completed")
                sp_lon = np.array(dataset['sp_lon'][:, 0]).ravel()
                print("sp_lon fetched - 2/11 completed")
                sp_lon[sp_lon > 180] -= 360 #Adjusting the longitude to the correct values for plotting
                sp_lat = np.array(dataset['sp_lat'][:, 0]).ravel()
                print("sp_lat fetched - 3/11 completed")
                gps_tx_power_db_w = np.array(dataset['gps_tx_power_db_w'][:, 0]).ravel()
                print("gps_tx_power_db_w fetched - 4/11 completed")
                gps_ant_gain_db_i = np.array(dataset['gps_ant_gain_db_i'][:, 0]).ravel()
                print("gps_ant_gain_db_i fetched - 5/11 completed")
                sp_rx_gain = np.array(dataset['sp_rx_gain'][:, 0]).ravel()
                print("sp_rx_gain fetched - 6/11 completed")
                tx_to_sp_range = np.array(dataset['tx_to_sp_range'][:, 0]).ravel()
                print("tx_to_sp_range fetched - 7/11 completed")
                rx_to_sp_range = np.array(dataset['rx_to_sp_range'][:, 0]).ravel()
                print("rx_to_sp_range fetched - 8/11 completed")
                prn_code = np.array(dataset['prn_code'][:, 0]).ravel()
                print("prn_code fetched - 9/11 completed")
                sp_inc_angle = np.array(dataset['sp_inc_angle'][:, 0]).ravel()
                print("sp_inc_angle fetched - 10/11 completed")
                quality_flags = np.array(dataset['quality_flags'][:, 0]).ravel()
                print("quality_flags fetched - 11/11 completed")

                print("⿐ⱐ⻘⢌⡩␿␍⫋❣⚆⃷␟Ⰺ⼆␩ⶋ⃽Ⱋ⫄⑝ⓣ┿⍡⫠Ⳋ⬵☵")
                print("CONGRATULATIONS, ALL DATA FETCHED SUCCESSFULLY")
                print("∜♕⭫⋭∳⁄≬┃⼻⠎▅⊡⅝⨅ⰰ⦷⌍⨓▊⛗⛠✕⠫ⱴ⫁⭅Ⲡ⿄")
                print("These are the shapes")
                print(ddm_snr.shape)
                print(sp_lon.shape)
                print(sp_lat.shape)
                print(gps_tx_power_db_w.shape)
                print(gps_ant_gain_db_i.shape)
                print(sp_rx_gain.shape)
                print(tx_to_sp_range.shape)
                print(rx_to_sp_range.shape)
                print(prn_code.shape)
                print(sp_inc_angle.shape)
                print(quality_flags.shape)
                # Create a dataframe with the data
                
                df = pd.DataFrame({
                    'ddm_snr': ddm_snr,
                    'sp_lon': sp_lon,
                    'sp_lat': sp_lat,
                    'gps_tx_power_db_w': gps_tx_power_db_w,
                    'gps_ant_gain_db_i': gps_ant_gain_db_i,
                    'sp_rx_gain': sp_rx_gain,
                    'tx_to_sp_range': tx_to_sp_range,
                    'rx_to_sp_range': rx_to_sp_range,
                    'prn_code': prn_code,
                    'sp_inc_angle': sp_inc_angle,
                    'quality_flags': quality_flags
                })    
                            
                '''
                Filter parameters
                Max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float
                '''
                
                df_filtered = data_filtering(df, max_lat, min_lat, max_lon, min_lon, inc_angle)
                #Reseting the index to start at zero again
                df_filtered = df_filtered.reset_index(drop=True)
                ds = xr.Dataset.from_dataframe(df_filtered)
                ds.to_netcdf(f'./data/{sat}_{date}.nc')
                print(f"Data for {sat} on {date}:")
                

            except HTTPError as e:
                # If the URL is invalid (404 error), print a message and skip this satellite/date
                print(f"No data available for {sat} on {date}, skipping. Error: {e}")

    

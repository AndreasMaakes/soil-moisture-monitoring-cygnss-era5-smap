from webob.exc import HTTPError  # Import HTTPError to catch 404 errors
from pydap.client import open_url
from pydap.cas.urs import setup_session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


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
    df_filtered_qf = df_filtered_inclination[(df["quality_flags"] != 2) | (df["quality_flags"] != 4) | (df["quality_flags"] != 5) | (df["quality_flags"] != 8) |(df["quality_flags"] != 16) | (df["quality_flags"] != 17)]
    
    return df_filtered_qf



'''This function downloads the data'''

def data_fetching(startDate: str, endDate: str, username: str, password: str, max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float):
    dates = create_dates_array(startDate, endDate)
    
    # List of satellite identifiers
    satellites = [f'cyg0{i}' for i in range(1, 2)]

# Iterate over each satellite and date
    for sat in tqdm(satellites):
        for date in dates:
            try:
                # Construct the URL for the current satellite and date
                url = f"https://opendap.earthdata.nasa.gov/collections/C2832195379-POCLOUD/granules/{sat}.ddmi.s{date}-000000-e{date}-235959.l1.power-brcs.a32.d33"
            
                # Attempt to open the dataset
                dataset = open_url(url, session=setup_session(username, password), protocol='dap4')

                # Fetch data from the dataset 
                ddm_snr = np.array(dataset['ddm_snr'][:, 0]) 
                sp_lon = np.array(dataset['sp_lon'][:, 0])
                sp_lon[sp_lon > 180] -= 360 #Adjusting the longitude to the correct values for plotting
                sp_lat = np.array(dataset['sp_lat'][:, 0])
                gps_tx_power_db_w = np.array(dataset['gps_tx_power_db_w'][:, 0])
                gps_ant_gain_db_i = np.array(dataset['gps_ant_gain_db_i'][:, 0])
                sp_rx_gain = np.array(dataset['sp_rx_gain'][:, 0])
                tx_to_sp_range = np.array(dataset['tx_to_sp_range'][:, 0])
                rx_to_sp_range = np.array(dataset['rx_to_sp_range'][:, 0])
                prn_code = np.array(dataset['prn_code'][:, 0])
                sp_inc_angle = np.array(dataset['sp_inc_angle'][:, 0]) 
                quality_flags = np.array(dataset['quality_flags'][:, 0])

                # Create a dataframe with the data
                
                df = pd.DataFrame({
                    'sp_lon': sp_lon,
                    'sp_lat': sp_lat,
                    'ddm_snr': ddm_snr,
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
                #Printing 45 first rows to check
                print(df_filtered.head(10))
                print(f"Data for {sat} on {date}:")
                

            except HTTPError as e:
                # If the URL is invalid (404 error), print a message and skip this satellite/date
                print(f"No data available for {sat} on {date}, skipping. Error: {e}")

    print("Data fetching complete.")

from webob.exc import HTTPError  # Import HTTPError to catch 404 errors
from pydap.client import open_url
from pydap.cas.urs import setup_session
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import os


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
quality flags 2, 4, 5, 8, 16, and 17. We also need to filter out ddm_snr below 2, and sp_rx_gain gain below 0 and over 13.
'''
def data_filtering(df, max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float):
    
    print("Filtering data. Dataframe length before geospatial filtering: ", len(df))
    df_filtered_spatial = df[(df["sp_lon"] >= min_lon) & (df["sp_lon"] <= max_lon) & (df["sp_lat"] >= min_lat) & (df["sp_lat"] <= max_lat)]
    print("Dataframe length after geospatial filtering: ", len(df_filtered_spatial))
    
    print("Filtering data based on inclination angle")
    df_filtered_inclination = df_filtered_spatial[(df["sp_inc_angle"] <= inc_angle)]
    print("Dataframe length after inclination angle filtering: ", len(df_filtered_inclination))
    
    print("Filtering data based on ddm_snr")
    df_filtered_ddm_snr = df_filtered_inclination[(df_filtered_inclination["ddm_snr"] >= 1)]
    print("Dataframe length after ddm_snr filtering: ", len(df_filtered_ddm_snr))
    
    print("Filtering data based on sp_rx_gain")
    df_filtered_sp_rx_gain = df_filtered_ddm_snr[(df_filtered_ddm_snr["sp_rx_gain"] >= 0) & (df_filtered_ddm_snr["sp_rx_gain"] <= 15)]
    print("Dataframe length after sp_rx_gain filtering: ", len(df_filtered_sp_rx_gain))
    
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
    print("Filtering data based on quality flags")
    df_filtered_qf = df_filtered_sp_rx_gain[(df_filtered_sp_rx_gain["quality_flags"] & bitmask_exclude) == 0]
    print("Dataframe length after quality flag filtering: ", len(df_filtered_qf))
    
    return df_filtered_qf



'''This function downloads the data'''

def data_fetching(startDate: str, endDate: str, username: str, password: str, max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float, area: str):
    dates = create_dates_array(startDate, endDate)
    
    # List of satellite identifiers
    satellites = [f'cyg0{i}' for i in range(1, 9)]
    
    #Creating a new folder for the data for this specific run
    name = f'{area}-{startDate}-{endDate}'
    folder_path = f'./data/{name}'
    
    try:
        os.mkdir(folder_path)
    except OSError:
        print ("Creation of the directory %s failed" % folder_path)
    
# Iterate over each satellite and date
    for sat in tqdm(satellites):
        for date in tqdm(dates):
            try:
                # Construct the URL for the current satellite and date
                url = f"https://opendap.earthdata.nasa.gov/collections/C2832195379-POCLOUD/granules/{sat}.ddmi.s{date}-000000-e{date}-235959.l1.power-brcs.a32.d33"
            
                # Attempt to open the dataset
                dataset = open_url(url, session=setup_session(username, password), protocol='dap4')

                # Fetch data from the dataset 
                print()
                print("Fetching data - 0/11 completed")
                ddm_snr = np.array(dataset['ddm_snr'][:, 0]).ravel()
                ddm_snr = np.append(ddm_snr, np.array(dataset['ddm_snr'][:, 1]).ravel())
                ddm_snr = np.append(ddm_snr, np.array(dataset['ddm_snr'][:, 2]).ravel())
                ddm_snr = np.append(ddm_snr, np.array(dataset['ddm_snr'][:, 3]).ravel())
                print("ddm_snr fetched - 1/11 completed")
                sp_lon = np.array(dataset['sp_lon'][:, 0]).ravel()
                sp_lon = np.append(sp_lon, np.array(dataset['sp_lon'][:, 1]).ravel())
                sp_lon = np.append(sp_lon, np.array(dataset['sp_lon'][:, 2]).ravel())
                sp_lon = np.append(sp_lon, np.array(dataset['sp_lon'][:, 3]).ravel())
                print("sp_lon fetched - 2/11 completed")
                sp_lon[sp_lon > 180] -= 360 #Adjusting the longitude to the correct values for plotting
                sp_lat = np.array(dataset['sp_lat'][:, 0]).ravel()
                sp_lat = np.append(sp_lat, np.array(dataset['sp_lat'][:, 1]).ravel())   
                sp_lat = np.append(sp_lat, np.array(dataset['sp_lat'][:, 2]).ravel())
                sp_lat = np.append(sp_lat, np.array(dataset['sp_lat'][:, 3]).ravel())
                print("sp_lat fetched - 3/11 completed")
                gps_tx_power_db_w = np.array(dataset['gps_tx_power_db_w'][:, 0]).ravel()
                gps_tx_power_db_w = np.append(gps_tx_power_db_w, np.array(dataset['gps_tx_power_db_w'][:, 1]).ravel())
                gps_tx_power_db_w = np.append(gps_tx_power_db_w, np.array(dataset['gps_tx_power_db_w'][:, 2]).ravel())
                gps_tx_power_db_w = np.append(gps_tx_power_db_w, np.array(dataset['gps_tx_power_db_w'][:, 3]).ravel())
                print("gps_tx_power_db_w fetched - 4/11 completed")
                gps_ant_gain_db_i = np.array(dataset['gps_ant_gain_db_i'][:, 0]).ravel()
                gps_ant_gain_db_i = np.append(gps_ant_gain_db_i, np.array(dataset['gps_ant_gain_db_i'][:, 1]).ravel())
                gps_ant_gain_db_i = np.append(gps_ant_gain_db_i, np.array(dataset['gps_ant_gain_db_i'][:, 2]).ravel())
                gps_ant_gain_db_i = np.append(gps_ant_gain_db_i, np.array(dataset['gps_ant_gain_db_i'][:, 3]).ravel())
                print("gps_ant_gain_db_i fetched - 5/11 completed")
                sp_rx_gain = np.array(dataset['sp_rx_gain'][:, 0]).ravel()
                sp_rx_gain = np.append(sp_rx_gain, np.array(dataset['sp_rx_gain'][:, 1]).ravel())
                sp_rx_gain = np.append(sp_rx_gain, np.array(dataset['sp_rx_gain'][:, 2]).ravel())
                sp_rx_gain = np.append(sp_rx_gain, np.array(dataset['sp_rx_gain'][:, 3]).ravel())
                print("sp_rx_gain fetched - 6/11 completed")
                tx_to_sp_range = np.array(dataset['tx_to_sp_range'][:, 0]).ravel()
                tx_to_sp_range = np.append(tx_to_sp_range, np.array(dataset['tx_to_sp_range'][:, 1]).ravel())
                tx_to_sp_range = np.append(tx_to_sp_range, np.array(dataset['tx_to_sp_range'][:, 2]).ravel())
                tx_to_sp_range = np.append(tx_to_sp_range, np.array(dataset['tx_to_sp_range'][:, 3]).ravel())
                print("tx_to_sp_range fetched - 7/11 completed")
                rx_to_sp_range = np.array(dataset['rx_to_sp_range'][:, 0]).ravel()
                rx_to_sp_range = np.append(rx_to_sp_range, np.array(dataset['rx_to_sp_range'][:, 1]).ravel())
                rx_to_sp_range = np.append(rx_to_sp_range, np.array(dataset['rx_to_sp_range'][:, 2]).ravel())
                rx_to_sp_range = np.append(rx_to_sp_range, np.array(dataset['rx_to_sp_range'][:, 3]).ravel())
                print("rx_to_sp_range fetched - 8/11 completed")
                prn_code = np.array(dataset['prn_code'][:, 0]).ravel()
                prn_code = np.append(prn_code, np.array(dataset['prn_code'][:, 1]).ravel())
                prn_code = np.append(prn_code, np.array(dataset['prn_code'][:, 2]).ravel())
                prn_code = np.append(prn_code, np.array(dataset['prn_code'][:, 3]).ravel())
                print("prn_code fetched - 9/11 completed")
                sp_inc_angle = np.array(dataset['sp_inc_angle'][:, 0]).ravel()
                sp_inc_angle = np.append(sp_inc_angle, np.array(dataset['sp_inc_angle'][:, 1]).ravel())
                sp_inc_angle = np.append(sp_inc_angle, np.array(dataset['sp_inc_angle'][:, 2]).ravel())
                sp_inc_angle = np.append(sp_inc_angle, np.array(dataset['sp_inc_angle'][:, 3]).ravel())
                print("sp_inc_angle fetched - 10/11 completed")
                quality_flags = np.array(dataset['quality_flags'][:, 0]).ravel()
                quality_flags = np.append(quality_flags, np.array(dataset['quality_flags'][:, 1]).ravel())
                quality_flags = np.append(quality_flags, np.array(dataset['quality_flags'][:, 2]).ravel())
                quality_flags = np.append(quality_flags, np.array(dataset['quality_flags'][:, 3]).ravel())
                print("quality_flags fetched - 11/11 completed")
                
                print("CONGRATULATIONS, ALL DATA FETCHED SUCCESSFULLY")
                
                
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
                            
                
                df_filtered = data_filtering(df, max_lat, min_lat, max_lon, min_lon, inc_angle)
                #Reseting the index to start at zero again
                df_filtered = df_filtered.reset_index(drop=True)
                ds = xr.Dataset.from_dataframe(df_filtered)
                
                
                ds.to_netcdf(f'{folder_path}/{sat}_{date}.nc')
                print(f"Data fetched for {sat} on {date}")
                

            except HTTPError as e:
                # If the URL is invalid (404 error), print a message and skip this satellite/date
                print(f"No data available for {sat} on {date}, skipping. Error: {e}")

    

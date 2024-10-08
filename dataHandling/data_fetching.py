from webob.exc import HTTPError  # Import HTTPError to catch 404 errors
from pydap.client import open_url
from pydap.cas.urs import setup_session
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import os
from data_filtering import data_filtering
from calculate_sr import sr
from create_dates_array import create_dates_array

'''
This functions lets the user input the from and to date, and returns the date in the format that the API requires.

The date format is as follows: YYYYMMDD, for example 20240701
'''



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
                            
                '''
                The data_filtering function is being run on the dataframe here
                '''
                
                df_filtered = data_filtering(df, max_lat, min_lat, max_lon, min_lon, inc_angle)
                
                
                #Reseting the index to start at zero again
                df_filtered = df_filtered.reset_index(drop=True)
                
                #Calculate the SR value and add it to the dataframe
                df_filtered["sr"] = df_filtered.apply(sr, axis = 1)
                
                ds = xr.Dataset.from_dataframe(df_filtered)
                ds.to_netcdf(f'{folder_path}/{sat}_{date}.nc')
                print(f"Data fetched for {sat} on {date}")
                

            except HTTPError as e:
                # If the URL is invalid (404 error), print a message and skip this satellite/date
                print(f"No data available for {sat} on {date}, skipping. Error: {e}")

    

from webob.exc import HTTPError  # Import HTTPError to catch 404 errors
from pydap.client import open_url
from pydap.cas.urs import setup_session
from datetime import datetime
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



def data_fetching(startDate: str, endDate: str, username: str, password: str, max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float, name: str, min_ddm_snr: float, min_sp_rx_gain: float, max_sp_rx_gain: float):
    dates = create_dates_array(startDate, endDate)
    
    # List of satellite identifiers
    satellites = [f'cyg0{i}' for i in range(1, 9)]
    
    #Creating a new folder for the data for this specific run
    file_name = f'{name}-{startDate}-{endDate}'
    folder_path = f'./data/{file_name}'
    
    try:
        os.mkdir(folder_path)
    except OSError:
        print(f"Creation of the directory {folder_path} failed")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    parameter_txt = open(f'./data/{file_name}/{name}-{startDate}-{endDate}.txt', "x")
    parameter_txt.write("Area of interest: " + name + "\n" + "\nStart date: " + str(startDate) + "\nEnd date: " + str(endDate) + "\n" + "\nMinimum latitude: " + str(min_lat) + "\nMinimum longitude: " + str(min_lon) + "\nMaximum latitude: " + str(max_lat) + "\nMaximum longitude: " + str(max_lon) + "\n" + "\nMaximum inclination angle: " + str(inc_angle) + "\nMinimum ddm_snr: " + str(min_ddm_snr) + "\nMinimum sp_rx_gain: " + str(min_sp_rx_gain) + "\nMaximum sp_rx_gain: " + str(max_sp_rx_gain) + "\n" + "\nData fetching started: " + current_time)
    parameter_txt.close()



    # Calculate the total number of iterations for progress tracking
    total_iterations = len(satellites) * len(dates)

    # Use tqdm to monitor the progress of the whole process (satellites and dates combined)
    with tqdm(total=total_iterations, desc="Progress: ", unit="files", colour = "green") as pbar:
        for sat in satellites:
            for date in dates:
                try:
                    # Construct the URL for the current satellite and date
                    url = f"https://opendap.earthdata.nasa.gov/collections/C2832195379-POCLOUD/granules/{sat}.ddmi.s{date}-000000-e{date}-235959.l1.power-brcs.a32.d33"
                    
                    # Attempt to open the dataset
                    dataset = open_url(url, session=setup_session(username, password), protocol='dap4')
                    
                    # Fetching data with progress inside each step
                    with tqdm(total=11, desc=f"Fetching data for {sat} on {date}", unit="steps", leave=False, colour = "yellow") as step_pbar:
                        ddm_snr = np.array(dataset['ddm_snr'][:, 0]).ravel()
                        ddm_snr = np.append(ddm_snr, np.array(dataset['ddm_snr'][:, 1]).ravel())
                        ddm_snr = np.append(ddm_snr, np.array(dataset['ddm_snr'][:, 2]).ravel())
                        ddm_snr = np.append(ddm_snr, np.array(dataset['ddm_snr'][:, 3]).ravel())
                        step_pbar.update(1)

                        sp_lon = np.array(dataset['sp_lon'][:, 0]).ravel()
                        sp_lon = np.append(sp_lon, np.array(dataset['sp_lon'][:, 1]).ravel())
                        sp_lon = np.append(sp_lon, np.array(dataset['sp_lon'][:, 2]).ravel())
                        sp_lon = np.append(sp_lon, np.array(dataset['sp_lon'][:, 3]).ravel())
                        sp_lon[sp_lon > 180] -= 360  # Adjusting the longitude to the correct values for plotting
                        step_pbar.update(1)

                        sp_lat = np.array(dataset['sp_lat'][:, 0]).ravel()
                        sp_lat = np.append(sp_lat, np.array(dataset['sp_lat'][:, 1]).ravel())   
                        sp_lat = np.append(sp_lat, np.array(dataset['sp_lat'][:, 2]).ravel())
                        sp_lat = np.append(sp_lat, np.array(dataset['sp_lat'][:, 3]).ravel())
                        step_pbar.update(1)

                        gps_tx_power_db_w = np.array(dataset['gps_tx_power_db_w'][:, 0]).ravel()
                        gps_tx_power_db_w = np.append(gps_tx_power_db_w, np.array(dataset['gps_tx_power_db_w'][:, 1]).ravel())
                        gps_tx_power_db_w = np.append(gps_tx_power_db_w, np.array(dataset['gps_tx_power_db_w'][:, 2]).ravel())
                        gps_tx_power_db_w = np.append(gps_tx_power_db_w, np.array(dataset['gps_tx_power_db_w'][:, 3]).ravel())
                        step_pbar.update(1)

                        gps_ant_gain_db_i = np.array(dataset['gps_ant_gain_db_i'][:, 0]).ravel()
                        gps_ant_gain_db_i = np.append(gps_ant_gain_db_i, np.array(dataset['gps_ant_gain_db_i'][:, 1]).ravel())
                        gps_ant_gain_db_i = np.append(gps_ant_gain_db_i, np.array(dataset['gps_ant_gain_db_i'][:, 2]).ravel())
                        gps_ant_gain_db_i = np.append(gps_ant_gain_db_i, np.array(dataset['gps_ant_gain_db_i'][:, 3]).ravel())
                        step_pbar.update(1)

                        sp_rx_gain = np.array(dataset['sp_rx_gain'][:, 0]).ravel()
                        sp_rx_gain = np.append(sp_rx_gain, np.array(dataset['sp_rx_gain'][:, 1]).ravel())
                        sp_rx_gain = np.append(sp_rx_gain, np.array(dataset['sp_rx_gain'][:, 2]).ravel())
                        sp_rx_gain = np.append(sp_rx_gain, np.array(dataset['sp_rx_gain'][:, 3]).ravel())
                        step_pbar.update(1)

                        tx_to_sp_range = np.array(dataset['tx_to_sp_range'][:, 0]).ravel()
                        tx_to_sp_range = np.append(tx_to_sp_range, np.array(dataset['tx_to_sp_range'][:, 1]).ravel())
                        tx_to_sp_range = np.append(tx_to_sp_range, np.array(dataset['tx_to_sp_range'][:, 2]).ravel())
                        tx_to_sp_range = np.append(tx_to_sp_range, np.array(dataset['tx_to_sp_range'][:, 3]).ravel())
                        step_pbar.update(1)

                        rx_to_sp_range = np.array(dataset['rx_to_sp_range'][:, 0]).ravel()
                        rx_to_sp_range = np.append(rx_to_sp_range, np.array(dataset['rx_to_sp_range'][:, 1]).ravel())
                        rx_to_sp_range = np.append(rx_to_sp_range, np.array(dataset['rx_to_sp_range'][:, 2]).ravel())
                        rx_to_sp_range = np.append(rx_to_sp_range, np.array(dataset['rx_to_sp_range'][:, 3]).ravel())
                        step_pbar.update(1)

                        prn_code = np.array(dataset['prn_code'][:, 0]).ravel()
                        prn_code = np.append(prn_code, np.array(dataset['prn_code'][:, 1]).ravel())
                        prn_code = np.append(prn_code, np.array(dataset['prn_code'][:, 2]).ravel())
                        prn_code = np.append(prn_code, np.array(dataset['prn_code'][:, 3]).ravel())
                        step_pbar.update(1)

                        sp_inc_angle = np.array(dataset['sp_inc_angle'][:, 0]).ravel()
                        sp_inc_angle = np.append(sp_inc_angle, np.array(dataset['sp_inc_angle'][:, 1]).ravel())
                        sp_inc_angle = np.append(sp_inc_angle, np.array(dataset['sp_inc_angle'][:, 2]).ravel())
                        sp_inc_angle = np.append(sp_inc_angle, np.array(dataset['sp_inc_angle'][:, 3]).ravel())
                        step_pbar.update(1)

                        quality_flags = np.array(dataset['quality_flags'][:, 0]).ravel()
                        quality_flags = np.append(quality_flags, np.array(dataset['quality_flags'][:, 1]).ravel())
                        quality_flags = np.append(quality_flags, np.array(dataset['quality_flags'][:, 2]).ravel())
                        quality_flags = np.append(quality_flags, np.array(dataset['quality_flags'][:, 3]).ravel())
                        step_pbar.update(1)

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
                                                    
                        # Filter and process the data
                        df_filtered = data_filtering(df, max_lat, min_lat, max_lon, min_lon, inc_angle, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)
                        df_filtered = df_filtered.reset_index(drop=True)
                        df_filtered["sr"] = df_filtered.apply(sr, axis=1)
                        
                        # Save the data
                        ds = xr.Dataset.from_dataframe(df_filtered)
                        ds.to_netcdf(f'{folder_path}/{sat}_{date}.nc')                        
                        
                        # Update the main progress bar
                        pbar.update(1)
                        print()
                        print(f"Data fetched and filtered for {sat} on {date}")

                except HTTPError as e:
                    print(f"No data available for {sat} on {date}, skipping. Error: {e}")
                    pbar.update(1)  # Even if skipped, progress the overall bar
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    parameter_txt = open(f'./data/{file_name}/{name}-{startDate}-{endDate}.txt', "a")
    parameter_txt.write("\nData fetching completed: " + current_time)
    parameter_txt.close()


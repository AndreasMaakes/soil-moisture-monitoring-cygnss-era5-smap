from webob.exc import HTTPError  # Import HTTPError to catch 404 errors
from pydap.client import open_url
from pydap.cas.urs import setup_session
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import os
from .data_filtering import data_filtering
from .calculate_sr import sr
from create_dates_array import create_dates_array


'''This is the main function that handles the downloading of the data. This function also calls the data_filtering function to filter the data.

Inputs:
    startDate (str): The start date of the data fetching in the format YYYYMMDD
    endDate (str): The end date of the data fetching in the format YYYYMMDD
    username (str): The username of the user for the API 
    password (str): The password of the user for the API (Note: Storing it as cleartext is not secure and should be handled with care)
    max_lat (float): The maximum latitude of the area of interest
    min_lat (float): The minimum latitude of the area of interest
    max_lon (float): The maximum longitude of the area of interest
    min_lon (float): The minimum longitude of the area of interest
    inc_angle (float): The maximum inclination angle for the CYGNSS L1 data
    name (str): The name of the area of interest, used for naming folders and files
    min_ddm_snr (float): The minimum ddm_snr value for the CYGNSS L1 data
    min_sp_rx_gain (float): The minimum sp_rx_gain value for the CYGNSS L1 data
    max_sp_rx_gain (float): The maximum sp_rx_gain value for the CYGNSS L1 data

Returns: 
    The function returns nothing, but saves the data in the data folder, along with a txt files with metadata.
'''

def data_fetching_CYGNSS(timeSeries: bool, startDate: str, endDate: str, username: str, password: str, max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float, name: str, min_ddm_snr: float, min_sp_rx_gain: float, max_sp_rx_gain: float):
    
    '''Create a list of dates between the start and end date using the create_dates_array function.'''
    dates = create_dates_array(startDate, endDate, "cygnss")
    
    '''Create a list of the CYGNSS satellites'''
    satellites = [f'cyg0{i}' for i in range(1, 9)]
    
    '''Define the base path for the data directory'''
    if timeSeries:
        base_data_path = f'data/TimeSeries/TimeSeries-{name}-{startDate}-{endDate}/CYGNSS'
    else:
        base_data_path = 'data/CYGNSS'
        
        #Create or locate the area-specific folder
        area_folder_path = os.path.join(base_data_path, name)
        if not os.path.exists(area_folder_path):
            try:
                os.mkdir(area_folder_path)
                print(f"Directory {area_folder_path} created successfully.")
            except OSError:
                print(f"Creation of the directory {area_folder_path} failed.")
        else:
            print(f"Directory {area_folder_path} already exists.")

        '''Create a subfolder for this specific run inside the area-specific folder. Name it after the area, start date, and end date.'''
        file_name = f'{name}-{startDate}-{endDate}'
        folder_path = os.path.join(area_folder_path, file_name)

        if not os.path.exists(folder_path):
            try:
                os.mkdir(folder_path)
                print(f"Directory {folder_path} created successfully.")
            except OSError:
                print(f"Creation of the directory {folder_path} failed.")
        else:
            print(f"Directory {folder_path} already exists.")

        '''Log the start of data fetching using the current time'''
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        '''Write parameters to a text file in the subfolder (metadata)'''
        parameter_txt_path = os.path.join(folder_path, f'{name}-{startDate}-{endDate}.txt')
        with open(parameter_txt_path, "w") as parameter_txt:
            parameter_txt.write(
                f"Area of interest: {name}\n"
                f"\nStart date: {startDate}\nEnd date: {endDate}\n"
                f"\nMinimum latitude: {min_lat}\nMinimum longitude: {min_lon}\n"
                f"Maximum latitude: {max_lat}\nMaximum longitude: {max_lon}\n"
                f"\nMaximum inclination angle: {inc_angle}\n"
                f"Minimum ddm_snr: {min_ddm_snr}\n"
                f"Minimum sp_rx_gain: {min_sp_rx_gain}\n"
                f"Maximum sp_rx_gain: {max_sp_rx_gain}\n"
                f"\nData fetching started: {current_time}"
            )

    '''Calculate the total number of iterations for progress tracking'''
    total_iterations = len(satellites) * len(dates)

    '''
    This is the main loop of the function.
    It iterates over all the satellites and dates, fetching the data for each combination.
    It uses the tqdm library to monitor the progress of the whole process, visualized as a progress bar.
    '''
    
    '''Dataframe to store the data in if the timeseries variable is active'''
    df_timeseries = pd.DataFrame({})

    with tqdm(total=total_iterations, desc="Progress: ", unit="files", colour = "green") as pbar:
        for sat in satellites:
            for date in dates:
                try:
                    '''Construct the URL for the current satellite and date'''
                    url = f"https://opendap.earthdata.nasa.gov/collections/C2832195379-POCLOUD/granules/{sat}.ddmi.s{date}-000000-e{date}-235959.l1.power-brcs.a32.d33"
                    
                    '''Attempt to open the dataset'''
                    dataset = open_url(url, session=setup_session(username, password), protocol='dap4')
                    
                    '''
                    Fetching all 11 variable from the dataset
                    Note: The tqdm progressbar is hardcoded to 11 variables, if additional variables are added the progressbar 
                    needs to be updated accordingly.
                    '''
                    
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
                        sp_lon[sp_lon > 180] -= 360  # Adjusting the longitude since it is in the range [-180, 180]
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

                        '''Create a dataframe with the data'''
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
                                                    
                        '''Filter and process the data'''
                        df_filtered = data_filtering(df, max_lat, min_lat, max_lon, min_lon, inc_angle, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)
                        df_filtered = df_filtered.reset_index(drop=True)
                        
                        '''Check if df_filtered is empty, if so, skip to the next iteration'''
                        if df_filtered.empty:
                            print(f"No data available after filtering for {sat} on {date}, skipping.")
                            pbar.update(1)  # Update progress even if no data saved
                            continue  # Skip to the next iteration
                        
                        '''Calculate the surface reflectivity using the sr function'''
                        df_filtered["sr"] = df_filtered.apply(sr, axis=1)
                        
                        '''If timeseries booliean is True, append the dataframe instead of saving it'''
                        if timeSeries:
                        
                            df_timeseries = pd.concat([df_timeseries,df_filtered])
                            
                            pbar.update(1)
                            print(f"Data fetched and filtered for {sat} on {date}")
                        else:
                            '''Save the data'''
                            ds = xr.Dataset.from_dataframe(df_filtered)
                            
                            '''Saving the data to the correct folder'''
                            
                            ds.to_netcdf(f'{folder_path}/{sat}_{date}.nc')                        
                            
                            '''Update the main progress bar'''
                            pbar.update(1)
                            print()
                            print(f"Data fetched and filtered for {sat} on {date}")
                        

                except HTTPError as e:
                    print(f"No data available for {sat} on {date}, skipping. Error: {e}")
                    pbar.update(1)  # Even if skipped, progressbar needs to be updated
    if timeSeries:
        df_timeseries = df_timeseries.reset_index(drop=True)
        return df_timeseries
    else:
        '''Log the completion time to the parameter file'''
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        # Construct the path for the metadata file
        parameter_txt_path = os.path.join(folder_path, f'{name}-{startDate}-{endDate}.txt')

        # Append the completion time to the parameter file
        with open(parameter_txt_path, "a") as parameter_txt:
            parameter_txt.write(f"\nData fetching completed: {current_time}")


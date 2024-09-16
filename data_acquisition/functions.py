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

'''This function downloads the data blabla'''

def data_fetching(startDate: str, endDate: str, username: str, password: str):
    dates = create_dates_array(startDate, endDate)
    
    # List of satellite identifiers
    satellites = [f'cyg0{i}' for i in range(1, 9)]

# Iterate over each satellite and date
    for sat in tqdm(satellites):
        for date in dates:
            try:
                # Construct the URL for the current satellite and date
                url = f"https://opendap.earthdata.nasa.gov/collections/C2832195379-POCLOUD/granules/{sat}.ddmi.s{date}-000000-e{date}-235959.l1.power-brcs.a32.d33"
            
                # Attempt to open the dataset
                dataset = open_url(url, session=setup_session(username, password), protocol='dap4')

                # Fetch data from the dataset if it exists
                ddm_snr = np.array(dataset['ddm_snr'][:, 0][:45]) 
                sp_lon = np.array(dataset['sp_lon'][:, 0][:45])
                sp_lat = np.array(dataset['sp_lat'][:, 0][:45])

                # Create a dataframe with the data
                df = pd.DataFrame(sp_lon, columns=['sp_lon'])
                df['sp_lat'] = sp_lat
                df['ddm_snr'] = ddm_snr
                print(f"Data for {sat} on {date}:")
                print(df)

            except HTTPError as e:
                # If the URL is invalid (404 error), print a message and skip this satellite/date
                print(f"No data available for {sat} on {date}, skipping. Error: {e}")

    print("Data fetching complete.")

from SMAP.data_fetching import data_fetching_smap
from ERA5.data_fetching import data_fetching_era5
from CYGNSS.data_fetching import data_fetching_CYGNSS
from data_fetching_time_series import data_fetching_time_series



'''
This is the main function of the program. 

This file is simply used to call the data_fetching function with the desired parameters.

The parameters are as follows:
username: str
    The username for the Earthdata server. Should be set as an environment variable or inserted 
password: str
    The password for the Earthdata server. Should be set as an environment variable or inserted 
max_lat: float
    The maximum latitude of the area of interest.
min_lat: float
    The minimum latitude of the area of interest.
max_lon: float
    The maximum longitude of the area of interest.
min_lon: float
    The minimum longitude of the area of interest.
inc_angle: float
    The maximum inclination angle of the CYGNSS observation
name: str
    The name of the area of interest. This is used to create the folder structure for the downloaded data.
min_ddm_snr: int
    The minimum DDM SNR value for the CYGNSS data. Default is 2 (Empirical value)
min_sp_rx_gain: int
    The minimum SP RX gain value for the CYGNSS data. Default is 0 (Empirical value)
max_sp_rx_gain: int
    The maximum SP RX gain value for the CYGNSS data. Default is 13 (Empirical value)



Login to the server is required to access the data from the Earthdata server.
The username and password should either be set as environment variables or inserted here as cleartext (not recommended).

'''

#Cleartext example with filler values:
username = "your_username_here"  # Set your Earthdata username here or as an environment variable
password = "your_password_here"  # Set your Earthdata password here or as an environment variable


'''
To fetch data for CYGNSS, SMAP or ERA5, simply call the respective data_fetching function with the desired parameters.
To fetch time series data, call the data_fetching_time_series function with the desired parameters.

'''


'''CYGNSS parameters   - Change these as needed '''
#DDM SNR and SP RX gain
min_ddm_snr = 2
min_sp_rx_gain = 0
max_sp_rx_gain = 13
#Maximum inclination angle
inc_angle = 65

'''Below are some examples of how to set up coordinates and names for the data fetching functions.'''

#Example - Pakistan
'''
name = "Pakistan"
min_lat = 25
min_lon = 67
max_lat = 28.5
max_lon = 73
'''
#Example - Western Australia
'''
name = "Western-Australia"
min_lat = -32
min_lon = 116
max_lat = -27
max_lon = 122
'''




'''Example calls are shown below. Uncomment the desired function call to fetch data, and adjust the parameters as needed.'''

'''CYGNSS data fetching example'''
#data_fetching_CYGNSS(False, "20200115", "20200131", username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)

'''SMAP data fetching example'''
#data_fetching_smap(False, "20200101", "20200114",  max_lat, min_lat, max_lon, min_lon, name)

'''ERA5 data fetching example'''
#data_fetching_era5(False, "20200601","20200607", min_lat, max_lat, min_lon, max_lon, name)
'''
Time series data fetching example. This function fetches data from CYGNSS, SMAP and ERA5 for a given time period and area of interest.
The number of days per week can be adjusted with the third parameter
'''
#data_fetching_time_series("20220108", "20241231", 3, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)

 
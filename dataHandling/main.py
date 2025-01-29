from SMAP.data_fetching import data_fetching_smap
from ERA5.data_fetching import data_fetching_era5
from CYGNSS.data_fetching import data_fetching_CYGNSS
'''
This is the main function of the program. 
This file is simply used to call the data_fetching function with the desired parameters, which are explained in the data_fetching.py file.
'''

'''Login to the server. '''

username = "andreasmaakes"
password = "Terrengmodell69!"



#Dates format CYGNSS "yyyymmdd"

'''CYGNSS parameters'''
#DDM SNR and SP RX gain
min_ddm_snr = 1
min_sp_rx_gain = 0
max_sp_rx_gain = 15
#Maximum inclination angle
inc_angle = 65




'''Area of interest'''

name = "Mexico"
min_lat = 20
min_lon = -105
max_lat = 26
max_lon = -98

data_fetching_smap("2024-07-04", "2024-07-05",  max_lat, min_lat, max_lon, min_lon, name)
data_fetching_era5("2024", "07", ["01","02","03"], min_lat, max_lat, min_lon, max_lon, name)
data_fetching_CYGNSS("20240701", "20240703", username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)

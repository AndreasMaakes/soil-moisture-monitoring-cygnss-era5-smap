from CYGNSS.data_fetching import data_fetching
from SMAP.data_fetching import data_fetching_smap
'''
This is the main function of the program. 
This file is simply used to call the data_fetching function with the desired parameters, which are explained in the data_fetching.py file.
'''

'''Login to the server. '''

username = "andreasmaakes"
password = "Terrengmodell69!"



#Dates format: "yyyymmdd"
start_date = "20240430"
end_date = "20240430" 
#Region name that is to be used for naming data folders
name = "Chad"
#Chad spatial filter
min_lat = 11
max_lat = 14
min_lon = 16
max_lon = 21
#DDM SNR and SP RX gain
min_ddm_snr = 1
min_sp_rx_gain = 0
max_sp_rx_gain = 15
#Maximum inclination angle
inc_angle = 65


'''
#Dates format: "yyyymmdd"
start_date = "20240430"
end_date = "20240430" 
#Region name that is to be used for naming data folders
name = "Brazil"
min_lat = -15
max_lat = -10
min_lon = -55
max_lon = -47
#DDM SNR and SP RX gain
min_ddm_snr = 2
min_sp_rx_gain = 0
max_sp_rx_gain = 15
#Maximum inclination angle
inc_angle = 65
'''

'''
#Dates format: "yyyymmdd"
start_date = "20240430"
end_date = "20240430" 
#Region name that is to be used for naming data folders
name = "Australia"
min_lat = -30
max_lat = -25
min_lon = 144
max_lon = 152
#DDM SNR and SP RX gain
min_ddm_snr = 1
min_sp_rx_gain = 0
max_sp_rx_gain = 15
#Maximum inclination angle
inc_angle = 65
'''

#data_fetching(start_date, end_date, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)

name = "Brazil"
min_lat = -15
max_lat = -10
min_lon = -55
max_lon = -47

data_fetching_smap("2024-07-24", "2024-07-24",  max_lat, min_lat, max_lon, min_lon, "Brazil")


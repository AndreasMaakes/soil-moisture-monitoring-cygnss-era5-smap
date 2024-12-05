from data_fetching import data_fetching

'''Login to the server'''
username = "andreasmaakes"
password = "Terrengmodell69!"


'''
Defining spatial filter
Max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float
'''

#Iran spatial filter
'''
max_lat = 36
min_lat = 28
max_lon = 65
min_lon = 54
inc_angle = 65
'''

'''
Dates format: "yyyymmdd"
'''

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


#Cygnss longitudes have to be corrected 
#Entire africa spatial filter
'''
name = "North-Africa"
min_lat = 2
max_lat = 20
min_lon = -19
max_lon = 38

min_ddm_snr = 1
min_sp_rx_gain = 0
max_sp_rx_gain = 15
inc_angle = 65
'''

'''
#Region name that is to be used for naming data folders
name = "Pakistan-India"
#Chad spatial filter
min_lat = 25
max_lat = 33
min_lon = 70
max_lon = 80
#DDM SNR and SP RX gain
min_ddm_snr = 1
min_sp_rx_gain = 0
max_sp_rx_gain = 15
#Maximum inclination angle
inc_angle = 65
'''

'''
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

start_date = "20240812"
end_date = "20240812" 

data_fetching(start_date, end_date, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)



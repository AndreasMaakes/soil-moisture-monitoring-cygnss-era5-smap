from SMAP.data_fetching import data_fetching_smap
from ERA5.data_fetching import data_fetching_era5
from ERA5.data_fetching import data_fetching_era5
from CYGNSS.data_fetching import data_fetching_CYGNSS
from data_fetching_time_series import data_fetching_time_series



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




'''
name = "Argentina"
min_lat = -30
min_lon = -68
max_lat = -25
max_lon = -61
'''


'''
name = "Iraq"
min_lat = 30.5
min_lon = 43
max_lat = 34.5
max_lon = 48
'''

'''
name = "India"
min_lat = 24
min_lon = 70
max_lat = 32
max_lon = 80
'''

'''
name = "South-Australia"
min_lat = -37
min_lon = 140
max_lat = -32
max_lon = 149
'''


name = "Western-Australia"
min_lat = -32
min_lon = 116
max_lat = -27
max_lon = 122


'''
name = "Thailand"
min_lat = 14
min_lon = 99
max_lat = 18
max_lon = 105
'''


name = "Pakistan_9km_smap"
min_lat = 25
min_lon = 67
max_lat = 28.5
max_lon = 73


'''
name = "Bolivia"
min_lat = -19
max_lat = -14
min_lon = -69
max_lon = -62

name = "China2"
min_lat = 42
max_lat = 46
min_lon = 123
max_lon = 129

name = "Aus1"
min_lat = -21
max_lat = -13.7
min_lon = 130
max_lon = 135

name = "Senegal"
min_lat = 12.6
max_lat = 15.7
min_lon = -16.2
max_lon = -12.5

name = "Texas"
min_lat = 30
max_lat = 33.4
min_lon = -102.2
max_lon = -94.5
'''

'''
name = "Venezuela"
min_lat = 3
max_lat = 9
min_lon = -70
max_lon = -63
'''
'''
name = "Testing"
min_lat = 4
max_lat = 19
min_lon = 16
max_lon = 33
'''

'''
name = "India3"
min_lat = 24.7
min_lon = 69.6
max_lat = 32.5
max_lon = 79.8
'''

'''
name = "Australia"
min_lat = -35.4
min_lon = 145.8
max_lat = -34.6
max_lon = 147.6
'''

#data_fetching_CYGNSS(False, "20200101", "20200131", username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)
data_fetching_smap(False, "20200101", "20200131",  max_lat, min_lat, max_lon, min_lon, name)
#data_fetching_era5(False, "20200101","20200131", min_lat, max_lat, min_lon, max_lon, name)
#data_fetching_time_series("20240701", "20240703", username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)

#data_fetching_era5(False, "20240101", "20240103", 20, 22, 20, 22, "Bombaclaat")
#data_fetching_time_series("20200511", "20200801", 3, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name, min_ddm_snr, min_sp_rx_gain, max_sp_rx_gain)



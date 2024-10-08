from functions import data_fetching

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
start_date = "20230912"
end_date = "20230913" 

#Region name that is to be used for naming data folders
name = "Chad"

#Chad spatial filter
min_lat = 10
max_lat = 15
min_lon = 16
max_lon = 21

#Inclination angle
inc_angle = 65

data_fetching(start_date, end_date, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle, name)


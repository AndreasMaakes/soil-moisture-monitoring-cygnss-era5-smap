from functions import data_fetching

'''Login to the server'''
username = "andreasmaakes"
password = "Terrengmodell69!"

'''
Dates format: "yyyymmdd"
'''

start_date = "20240101"
end_date = "20240101"

'''
Defining spatial filter
Max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float
'''
max_lat = 36
min_lat = 28
max_lon = 65
min_lon = 54
inc_angle = 65

data_fetching(start_date, end_date, username, password, max_lat, min_lat, max_lon, min_lon, inc_angle)


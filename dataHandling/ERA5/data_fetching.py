import cdsapi
import os
from create_dates_array import create_dates_array


'''
Husk å skrive dokumentasjon om at api-kallet henter data fra et gitt sett med dager og måneder, så start-og end date funksjonaliteten er ikke lik som for CYGNSS.
NB: Nå hentes det for 24 timer
'''

def data_fetching_era5(timeSeries: bool, startDate: str, endDate: str, min_lat, max_lat, min_lon, max_lon, name, basePath="data/ERA5"):
    
    '''Create an array of dates between the start and end date'''
    dates = create_dates_array(startDate, endDate, "era5")
    year = dates[0]
    month = dates[1]
    days = dates[2]
    
    dataset = "reanalysis-era5-single-levels"
    request = {
    "product_type": ["reanalysis"],
    "variable": ["volumetric_soil_water_layer_1","land_sea_mask"],
    "year": [year],
    "month": [month],
    "day": days,
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [max_lat, min_lon, min_lat, max_lon]
}
    '''Define the base path for the data directory'''

    if timeSeries:
        base_data_path = f'{basePath}/ERA5'
        file_name = f"ERA5_{year}{month}{days[0]}_{year}{month}{days[-1]}"
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(f"{base_data_path}/{file_name}.nc")


    else:
        '''Create or locate the area-specific folder'''
        area_folder_path = os.path.join(basePath, name)
        if not os.path.exists(area_folder_path):
            try:
                os.mkdir(area_folder_path)
                print(f"Directory {area_folder_path} created successfully.")
            except OSError:
                print(f"Creation of the directory {area_folder_path} failed.")
        else:
            print(f"Directory {area_folder_path} already exists.")

        '''Create a subfolder for this specific run inside the area-specific folder. Name it after the area, year, month and days'''
        file_name = f"ERA5_{name}_{year}_{month}_{days[0]}_{days[-1]}"
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(f"{area_folder_path}/{file_name}.nc")







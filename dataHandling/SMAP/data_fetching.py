import earthaccess
import xarray as xr
import numpy as np
import h5py
import pandas as pd

earthaccess.login()

results = earthaccess.search_data(
    short_name='SPL3SMP',
    bounding_box=(-10, 20, 10, 50),
    temporal=("2019-02-01", "2019-02-02"),
    count=-1,
    provider="NSIDC_CPRD" #Specifying the cloud based provider
    
)

dataset = earthaccess.open(results)


# Assuming `dataset` contains the list of HDF5 file paths
for ds in dataset:
    print(f"Processing file: {ds}")
    with h5py.File(ds, 'r') as f:
        # Navigate to the desired group
        group = f['Soil_Moisture_Retrieval_Data_AM']
        # TODO: Add PM group
        
        # Access the required variables
        latitude = group['latitude'][...]  
        longitude = group['longitude'][...]
        soil_moisture = group['soil_moisture'][...]
        soil_moisture_dca = group['soil_moisture_dca'][...]
        
        # Flatten arrays if necessary
        latitude = latitude.flatten()
        longitude = longitude.flatten()
        soil_moisture = soil_moisture.flatten()
        soil_moisture_dca = soil_moisture_dca.flatten()

        # Create a DataFrame
        df = pd.DataFrame({
            'latitude': latitude,
            'longitude': longitude,
            'soil_moisture': soil_moisture,
            'soil_moisture_dca': soil_moisture_dca
        })
        
        print(df.head())  # Preview the DataFrame



'''
for ds in dataset:
    
    xr_ds = xr.open_dataset(ds, engine='h5netcdf', group='Soil_Moisture_Retrieval_Data_AM', phony_dims='sort').sel(variable='longitude')  # Open a specific dataset
    
    # Access specific variables
    longitude = xr_ds['longitude']  # Replace with actual variable name if different
    latitude = xr_ds['latitude']
    soil_moisture = xr_ds['soil_moisture']  # Replace with actual variable name if different
    soil_moisture_dca = xr_ds['soil_moisture_dca']  # Replace with actual variable name if different

    # Print variable details (optional)
    print(longitude)
    print(latitude)
    print(soil_moisture)
    print(soil_moisture_dca)
'''
#xr_ds = xr.open_mfdataset(dataset, engine='h5netcdf') 

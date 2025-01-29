import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from SMAP_import_data import importDataSMAP
from SMAP_utils import SMAP_averaging_soil_moisture

#Experimenting with raster plots in matplotlib


#Importing the data as a list of dataframes
df = importDataSMAP('Brazil')

#Concating the dataframes to one single dataframe
df = pd.concat(df)

#Averaging the soil moisture values
df = SMAP_averaging_soil_moisture(df)

# Create a pivot table with latitude as rows and longitude as columns
pivoted_data = df.pivot_table(
    index="latitude", 
    columns="longitude", 
    values="soil_moisture_avg"
)

# Create a meshgrid of latitudes and longitudes
latitudes = pivoted_data.index.values
longitudes = pivoted_data.columns.values
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

# Plot the soil moisture data using pcolormesh
plt.figure(figsize=(10, 8))
mesh = plt.pcolormesh(
    lon_grid, lat_grid, pivoted_data.values, 
    shading='auto', cmap='viridis'
)
plt.colorbar(mesh, label='Soil Moisture')
plt.axis('equal')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Soil Moisture Map')
plt.show()


#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from plotting.CYGNSS.import_data import importData

def satellite_image_with_axes(folder_name, saveplot):
    # Simulated latitude and longitude bounds
    dataFrames = importData(folder_name)
    
    lats = np.array([])
    lons = np.array([])
    
    for df in dataFrames:
        lat = np.array(df['sp_lat'])
        lon = np.array(df['sp_lon'])
        

        lats = np.append(lats, lat)
        lons = np.append(lons, lon)
        
    max_lat = np.max(lats)
    min_lat = np.min(lats)
    max_lon = np.max(lons)
    min_lon = np.min(lons)
    
'''
    # Plotting the satellite image
    fig, ax = plt.subplots(figsize=(12, 8))
    m = Basemap(
        epsg=4326,
        llcrnrlat=min_lat,
        urcrnrlat=max_lat,
        llcrnrlon=min_lon,
        urcrnrlon=max_lon,
        resolution='i',
        ax=ax,
    )
    m.arcgisimage(service='World_Imagery', xpixels=1500, verbose=False)

   

    # Add title
    plt.title("Satellite Image with Latitude and Longitude Axes", fontsize=15)

    # Save or show the plot
    if saveplot:
        plt.savefig(f'plotting/plots/{folder_name}_satellite_with_axes.png', dpi=300)

    plt.show()
'''
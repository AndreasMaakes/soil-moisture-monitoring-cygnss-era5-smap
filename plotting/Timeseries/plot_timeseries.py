import pandas as pd
import matplotlib.pyplot as plt
import os
import xarray as xr
import datetime

def plot_time_series(folder_name, min_lat, max_lat, min_lon, max_lon):
    '''Data folder'''
    basePath = f'{folder_name}/ERA5'
    
    # Lists to store weekly average values and corresponding weeks
    weeks = []
    avg_moisture_values = []

    '''Iterating through all .nc files in the folder'''
    for file in os.listdir(basePath):
        if file.endswith(".nc"):   
            filePath = os.path.join(basePath, file)

            # Extract the first date from the filename
            date_str = file.split("_")[1]  # Extract "20240101" from "ERA5_20240101_20240102.nc"
            first_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()  # Convert to date

            # Open the NetCDF file using xarray
            ds = xr.open_dataset(filePath, engine='netcdf4')  
            df = ds.to_dataframe().reset_index()

            # Ensure the 'soil_moisture' column exists before calculating the mean
            if "swvl1" in df.columns:

                df_filtered_spatial = df[(df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) & (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)]
                avg_moisture = df_filtered_spatial["swvl1"].mean()  # Compute mean soil moisture
                
                # Append values to lists
                weeks.append(first_date)
                avg_moisture_values.append(avg_moisture)
            else:
                print(f"Warning: 'swvl1' column not found in {file}")

    # Step 3: Plot the time series
    plt.figure(figsize=(10, 5))
    plt.plot(weeks, avg_moisture_values, marker='o', linestyle='-', color='b', label="Average Soil Moisture")

    plt.xlabel("Week Start Date")
    plt.ylabel("Average Soil Moisture")
    plt.title("Weekly Average Soil Moisture Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Example usage
plot_time_series("data\Timeseries\TimeSeries-Thailand-20240101-20241201",15,16,103,104)

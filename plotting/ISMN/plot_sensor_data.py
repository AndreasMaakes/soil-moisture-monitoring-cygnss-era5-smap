import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import gaussian_filter1d  # Requires scipy

def plot_soil_moisture(folder_path, sigma=0):
    # Get list of all .stm files in the folder.
    file_list = glob.glob(os.path.join(folder_path, '*.stm'))
    
    # List to store each sensor's weekly average as a pandas Series.
    weekly_series_list = []
    
    for file in file_list:
        # Read the file, skipping the first header line.
        df = pd.read_csv(file, delim_whitespace=True, skiprows=1, header=None,
                         names=['date', 'time', 'moisture', 'unit1', 'unit2'])
        
        # Filter out rows with invalid moisture values (values below zero, e.g., -9999).
        df = df[df['moisture'] >= 0]
        
        # Combine date and time into a single datetime column.
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M')
        
        # Set the datetime as the index.
        df.set_index('datetime', inplace=True)
        
        # Resample the data weekly and compute the mean moisture for each week.
        weekly_avg = df['moisture'].resample('W').mean()
        weekly_series_list.append(weekly_avg)
    
    # Combine all sensors' weekly data into one DataFrame.
    combined = pd.concat(weekly_series_list, axis=1)
    
    # Compute the overall weekly average across sensors (ignoring NaN where data is missing).
    combined['mean_moisture'] = combined.mean(axis=1)
    
    # Remove weeks where the overall weekly average moisture is above 1.
    filtered = combined[combined['mean_moisture'] <= 1]
    
    # Optionally apply Gaussian blur if sigma > 0.
    if sigma > 0:
        blurred_values = gaussian_filter1d(filtered['mean_moisture'].values, sigma=sigma)
        # Store the blurred data in a new column.
        filtered['mean_moisture_blurred'] = blurred_values
        y_values = filtered['mean_moisture_blurred']
        title = f'Weekly Average Soil Moisture Across Sensors (Gaussian Blur, sigma={sigma})'
    else:
        y_values = filtered['mean_moisture']
        title = 'Weekly Average Soil Moisture Across Sensors'
    
    # Plot the overall weekly average soil moisture.
    plt.figure(figsize=(10, 5))
    plt.plot(filtered.index, y_values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Week')
    plt.ylabel('Soil Moisture')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#Calling the function 
plot_soil_moisture("data\ISMN\Ghana", sigma=2)
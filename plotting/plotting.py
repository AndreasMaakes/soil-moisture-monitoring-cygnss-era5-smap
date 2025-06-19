from ERA5.ERA5_gaussian_blur_plot import ERA5_gaussian_blur_plot
from SMAP.SMAP_gaussian_blur_plot import SMAP_gaussian_blur_plot
from plot_timeseries import plot_time_series
from plot_correlation import correlation_plot, spatial_correlation_matrix
from CYGNSS.CYGNSS_average_plot import CYGNSS_average_plot
from timeseries_correlation import time_series_correlation
from CYGNSS.cygnss_trace_plot_new import CYGNSS_raw_plot_satellite
from timeseries_correlation_ismn import optimize_correlation_parameters, time_series_correlation_with_ismn
from plot_timeseries import plot_cygnss_ismn_time_series


'''
This is the main plotting file for the software package.

This file is used to call the plotting functions with the desired parameters, and handles geographic data visualization,
correlation analysis, time series plotting and satellite imagery. 
'''

'''The functions CYGNSS_average_plot, SMAP_gaussian_blur_plot and ERA5_gaussian_blur_plot are used to plot the data from CYGNSS, SMAP and ERA5 respectively.
Parameters:
- folder_name: str
    The name of the folder containing the data to be plotted. The format should be consistent with the formats
    provided in the examples below.
    - gaussian_sigma: float
        The standard deviation for Gaussian blurring applied to the data. This is used to smooth the data for better visualization.
    - lat_step: float
        The step size for latitude in the plot. This determines the resolution of the plot in the latitude direction.
    - lon_step: float
        The step size for longitude in the plot. This determines the resolution of the plot in the
        
    Additionally, the ERA5 function has an additional parameter:
    - land_sea_mask: default is 0.9, as shown in the example below.
        This is used to mask out land-sea areas in the plot, depending on the value
        
    The CYGNSS_raw_plot_satellite plots the ground tracks of the CYGNSS observations. 
    
    Example calls are provided below, which can be modified to suit the desired area of interest and parameters.
'''


#CYGNSS_average_plot("Western-Australia/Western-Australia-20200101-20200228", 0, 0.001, 0.001)
#SMAP_gaussian_blur_plot("Western-Australia_20200101_20200228", 0, 0.08, 0.08)
#ERA5_gaussian_blur_plot("Western-Australia/ERA5_Western-Australia_20200601_20200607.nc", 0, 0.9, 0.08, 0.08)

#CYGNSS_raw_plot_satellite("Sudan\Sudan-20201001-20201003")


'''
The correlation analysis between CYGNSS and reference data from SMAP and ERA5 can be performed using the correlation_matrix and correlation_plot functions.

correlation_matrix and correlation_plot takes in the three folder locations, using the same syntax as the example. The order of the input folders is SMAP, CYGNSS then ERA5. 
Additionally, the parameters lat_step, lon_step and land_sea mask are used to determine the resolution of the plot and the land-sea mask for the ERA5 data.

The correlation_plot also takes in the additional parameter gaussian_sigma, which is used to apply a Gaussian blur to the data before plotting.

To plot the correlation between ISMN in situ data with CYGNSS time series data, the function time_series_correlation_with_ismn can be used.
'''

#correlation_matrix("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9)
#correlation_plot("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9, gaussian_sigma=2)
#time_series_correlation_with_ismn(folder_name="data\Timeseries\TimeSeries-Australia-20180801-20200801",ismn_folder="data/ISMN/Australia",gaussian_sigma=3)


'''
To plot time series data from CYGNSS, ERA5 and SMAP, use the plot_time_series function. This function allows for geospatial filtering of the area of 
interest to visualize the time series data for a specific region within the study area.

Parameters:
- folder_name: str
    The name of the folder containing the time series data to be plotted. The format should be consistent with the examples below.
- max_lat: float
- min_lat: float
- max_lon: float
- min_lon: float

- gaussian_sigma: float

To plot the ISMN in situ data with CYGNSS time series data, the function plot_cygnss_ismn_time_series can be used. 

Example usages:
'''

#plot_time_series("data/Timeseries/TimeSeries-Australia-20180801-20200801", -35.4, 145.8, -34.6, 147.6, gaussian_sigma=5)
#plot_cygnss_ismn_time_series(cygnss_folder="data\Timeseries\TimeSeries-Australia-20180801-20200801/CYGNSS",ismn_folder="data/ISMN/Australia",sigma=3)







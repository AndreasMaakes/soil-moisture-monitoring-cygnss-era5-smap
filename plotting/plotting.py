from CYGNSS.trace_plot import trace_plot
from CYGNSS.CYGNSS_gaussian_blur_plot import CYGNSS_gaussian_blur_plot
from CYGNSS.CYGNSS_gaussian_blur_plot_old import CYGNSS_gaussian_blur_plot as CYGGIBOI
from ERA5.ERA5_gaussian_blur_plot import ERA5_gaussian_blur_plot
from SMAP.SMAP_gaussian_blur_plot import SMAP_gaussian_blur_plot
from CYGNSS.trace_plot import trace_plot
from plot_timeseries import plot_time_series
from plot_correlation import correlation_plot, spatial_correlation_matrix
from CYGNSS.CYGNSS_average_plot import CYGNSS_average_plot
from ERA5.ERA5_gaussian_blur_plot_new import ERA5_regrid_and_blur
from timeseries_correlation import time_series_correlation



'''
This is the main plotting file in the program, which calls the plotting functions with the desired parameters.
There are two different plots available: 

#1: trace plot: 
    To track the traces or ground tracks of the CYGNSS spacecraft, simply call the trace_plot function, 
    and insert the location of the data folder. This should be on the format: Name/Name-yyyymmdd-yyyymmdd. 


#2: gaussian blur plot.
    To plot the gaussian blur plot, call the gaussian_blur_plot function, and insert the location of the data folder
    using the same format as stated above. One additional parameter is required, which is sigma value of the gaussian blur.
    
    Using sigma = 0 will simply plot the interpolated data without any gaussian blur.

Note: Both plotting options has the option to save the plot to a file. This is done by setting the save parameter to True.

'''





#SMAP_gaussian_blur_plot("India2", 0, 0.05, 0.05)
#CYGNSS_gaussian_blur_plot("Western-Australia/Western-Australia-20200101-20200131", 0, 2000, True)
#CYGNSS_average_plot("India2/India2-20200101-20200107", 2.0, 0.05, 0.05,  False)
#ERA5_gaussian_blur_plot("Pakistan/ERA5_Pakistan20200701_20200707.nc", 0, 0.9, 50)
#ERA5_regrid_and_blur("Pakistan/ERA5_Pakistan_20200101_20200107.nc", 2.0, 0.6, 0.05, 0.05)
#correlation_matrix("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9)
#correlation_plot("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9, gaussian_sigma=2)

plot_time_series("data/Timeseries/TimeSeries-Pakistan-20220101-20241231", 25, 69, 25.5, 69.5, gaussian_sigma=2)

time_series_correlation("data/Timeseries/TimeSeries-Pakistan-20220101-20241231", 25, 69, 25.5, 69.5, gaussian_sigma=1)

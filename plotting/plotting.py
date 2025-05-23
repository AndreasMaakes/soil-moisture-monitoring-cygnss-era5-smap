from CYGNSS.trace_plot import trace_plot
from CYGNSS.CYGNSS_gaussian_blur_plot import CYGNSS_gaussian_blur_plot
from CYGNSS.CYGNSS_gaussian_blur_plot_old import CYGNSS_gaussian_blur_plot as CYGGIBOI
from ERA5.ERA5_gaussian_blur_plot import ERA5_gaussian_blur_plot
from SMAP.SMAP_gaussian_blur_plot import SMAP_gaussian_blur_plot
from CYGNSS.trace_plot import trace_plot
from plot_timeseries import plot_time_series
from plot_correlation import correlation_plot, spatial_correlation_matrix, spatial_correlation_matrix_old
from CYGNSS.CYGNSS_average_plot import CYGNSS_average_plot
from ERA5.ERA5_gaussian_blur_plot_new import ERA5_regrid_and_blur
from timeseries_correlation import time_series_correlation
from CYGNSS.cygnss_trace_plot_new import CYGNSS_raw_plot_satellite
from ERA5.ERA5_gaussian_blur_plot_new import ERA5_regrid_and_blur as ERA5_gaussian_blur_plot_new


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





#SMAP_gaussian_blur_plot("Western-Australia_20200101_20200228", 0, 0.08, 0.08)
#SMAP_gaussian_blur_plot("Western-Australia/30_days", 0, 0.08, 0.08)
#CYGNSS_gaussian_blur_plot("Western-Australia/Western-Australia-20200101-20200131", 0, 2000, True)
CYGNSS_average_plot("Pakistan/Pakistan-20200601-20200630", 1.5, 0.08, 0.08,  False)
#ERA5_gaussian_blur_plot("Uruguay/ERA5_Uruguay_2020_01_01_14.nc", 0, 0.1, 0.1, 0.1)
#spatial_correlation_matrix("Pakistan_9km_smap", "India2/India2-20200101-20200107", "Pakistan/ERA5_Pakistan_20200101_20200107.nc", 0.9, 0.5, 0.5)
#correlation_plot("India2", "India2/India2-20200101-20200107", "India2/ERA5_India2_2020_01_01_07.nc", 0.5, 0.5, 0.9, gaussian_sigma=2)
#plot_time_series("data/Timeseries/TimeSeries-Australia-20180801-20200801", -35.4, 145.8, -34.6, 147.6, gaussian_sigma=5)

#CYGNSS_raw_plot_satellite("Pakistan\Pakistan-20200601-20200630")

#SMAP_gaussian_blur_plot("Pakistan_20200601_20200630", 1.5, 0.08, 0.08)
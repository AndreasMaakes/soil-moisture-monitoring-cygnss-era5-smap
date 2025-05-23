from CYGNSS.trace_plot import trace_plot
from CYGNSS.CYGNSS_gaussian_blur_plot import CYGNSS_gaussian_blur_plot
from CYGNSS.CYGNSS_gaussian_blur_plot_old import CYGNSS_gaussian_blur_plot as CYGGIBOI
from ERA5.ERA5_gaussian_blur_plot import ERA5_gaussian_blur_plot
from SMAP.SMAP_gaussian_blur_adjustable_grid_size import SMAP_gaussian_blur_plot
from CYGNSS.trace_plot import trace_plot
from plot_timeseries import plot_time_series
from plot_correlation import correlation_plot, spatial_correlation_matrix
from CYGNSS.CYGNSS_average_plot import CYGNSS_average_plot
from CYGNSS.cygnss_grid_comparison import trace_plot_2 
from SMAP.suitability_map import SMAP_surface_flags_suitability



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





#SMAP_gaussian_blur_plot("Western-Australia-20200101-20200103", 0, 0.005)
#CYGNSS_gaussian_blur_plot("Western-Australia/Western-Australia-20200101-20200131", 0, 2000, True)
#CYGNSS_average_plot("Western-Australia/Western-Australia-20200101-20200228", 0, 0.001, 0.001,  False)
#ERA5_gaussian_blur_plot("Western-Australia/ERA5_Western-Australia_2020_01_01_28.nc", 0, 0.9, 100)
#correlation_matrix("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9)
#correlation_plot("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9, gaussian_sigma=2)
#plot_time_series("data/Timeseries/TimeSeries-Australia-20180801-20200801", -35.4, 145.8, -34.6, 147.6, gaussian_sigma=5)

#trace_plot_2('India2/India2-20200101-20200107', saveplot=False)

SMAP_surface_flags_suitability("suitability_map_newwwww-20200320-20200326", 0.1, 0.1, sigma=0, weights=[0.1,0.2,0.3,0.4])
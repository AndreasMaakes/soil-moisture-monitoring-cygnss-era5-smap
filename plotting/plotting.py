from CYGNSS.trace_plot import trace_plot
from CYGNSS.CYGNSS_gaussian_blur_plot import CYGNSS_gaussian_blur_plot
from CYGNSS.CYGNSS_gaussian_blur_plot_old import CYGNSS_gaussian_blur_plot as CYGGIBOI
from ERA5.ERA5_gaussian_blur_plot import ERA5_gaussian_blur_plot
from SMAP.SMAP_gaussian_blur_plot import SMAP_gaussian_blur_plot
from CYGNSS.trace_plot import trace_plot
from plot_timeseries import plot_time_series
from plot_correlation import correlation_plot, correlation_matrix
from CYGNSS.CYGNSS_average_plot import CYGNSS_average_plot



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





#SMAP_gaussian_blur_plot("India2", 2.0, 0.1, 0.1)
#CYGNSS_gaussian_blur_plot("India2/India2-20200101-20200131", 2.0, 50, False)
#CYGNSS_average_plot("India2/India2-20200101-20200131", 2.0, 0.1, 0.1,  False)
#ERA5_gaussian_blur_plot("India2/ERA5_India2_2020_01_01_31.nc", 2.0, 0.9, 50)
#correlation_matrix("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9)
#correlation_plot("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9)
plot_time_series("data/Timeseries/TimeSeries-Pakistan-20220101-20241231", 25, 68, 28.5, 70, gaussian_sigma=1.5)


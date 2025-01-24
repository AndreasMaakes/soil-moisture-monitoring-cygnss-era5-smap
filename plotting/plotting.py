from CYGNSS.trace_plot import trace_plot
from CYGNSS.CYGNSS_gaussian_blur_plot import CYGNSS_gaussian_blur_plot
from ERA5.ERA5_gaussian_blur_plot import ERA5_gaussian_blur_plot

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

'''Example of how to plot data for Brazil - 7 days'''

#CYGNSS_gaussian_blur_plot("Brazil/Brazil-20240912-20240918", False, 1)
#trace_plot("Chad/Chad-20240430-20240430", False)
ERA5_gaussian_blur_plot("Brazil/ERA5_Brazil_2024_07_24_24.nc", 1)



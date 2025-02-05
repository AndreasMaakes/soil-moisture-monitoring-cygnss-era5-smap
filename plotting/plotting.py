from CYGNSS.trace_plot import trace_plot
from CYGNSS.CYGNSS_gaussian_blur_plot import CYGNSS_gaussian_blur_plot
from CYGNSS.CYGNSS_gaussian_blur_plot_old import CYGNSS_gaussian_blur_plot as CYGGIBOI
from ERA5.ERA5_gaussian_blur_plot import ERA5_gaussian_blur_plot
from SMAP.SMAP_gaussian_blur_plot import SMAP_gaussian_blur_plot
from CYGNSS.trace_plot import trace_plot



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


ERA5_gaussian_blur_plot("Argentina/ERA5_Argentina_2024_07_01_03.nc", 0.7, 0.9, 100) 
#SMAP_gaussian_blur_plot("Argentina", 0.5)
#CYGNSS_gaussian_blur_plot("Argentina/Argentina-20240701-20240707", 0.7)
#trace_plot("Mexico/Mexico-20240701-20240703", False)
#CYGGIBOI("Argentina/Argentina-20240701-20240703", False, 0.5)
#trace_plot("Argentina/Argentina-20240701-20240707", False)

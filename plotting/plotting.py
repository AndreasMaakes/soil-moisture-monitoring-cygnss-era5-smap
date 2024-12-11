from trace_plot import trace_plot
from gaussian_blur_plot import gaussian_blur_plot

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

#gaussian_blur_plot("Brazil/Brazil-20240912-20240918", False, 0)
#gaussian_blur_plot("Brazil/Brazil-20240912-20240918", False, 1)
#trace_plot("Brazil/Brazil-20240912-20240918", False)


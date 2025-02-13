from CYGNSS.trace_plot import trace_plot
from CYGNSS.CYGNSS_gaussian_blur_plot import CYGNSS_gaussian_blur_plot
from CYGNSS.CYGNSS_gaussian_blur_plot_old import CYGNSS_gaussian_blur_plot as CYGGIBOI
from ERA5.ERA5_gaussian_blur_plot import ERA5_gaussian_blur_plot
from SMAP.SMAP_gaussian_blur_plot import SMAP_gaussian_blur_plot
from CYGNSS.trace_plot import trace_plot
from plot_timeseries import plot_time_series
from CYGNSS.CYGNSS_testing_limits import CYGNSS_gaussian_blur_plot as LETSGO



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




#CYGNSS_gaussian_blur_plot("Iraq/Iraq-20240201-20240207", 2.5)
#CYGGIBOI("Iraq/Iraq-20240201-20240207", False, 1.0)
#SMAP_gaussian_blur_plot("Iraq", 2.0, 100)
#ERA5_gaussian_blur_plot("Iraq/ERA5_Iraq_2024_02_01_07.nc", 2.0, 0.9, 100)

#CYGNSS_gaussian_blur_plot("India/India-20200601-20200607", 0, 1000)
#CYGGIBOI("India/India-20240201-20240207", False, 1.5)
#SMAP_gaussian_blur_plot("India", 1.5, 100)
#ERA5_gaussian_blur_plot("India/ERA5_India_2020_06_01_07.nc", 1.5, 0.9, 100)

#CYGNSS_gaussian_blur_plot("Argentina/Argentina-20240701-20240707", 0, 1000)
#CYGGIBOI("Argentina/Argentina-20240701-20240707", False, 1.0)
#SMAP_gaussian_blur_plot("Argentina", 1.0, 100)
#ERA5_gaussian_blur_plot("Argentina/ERA5_Argentina_2024_07_01_07.nc", 1.0, 0.9, 100)
#trace_plot("Argentina/Argentina-20240701-20240707", True)

#CYGNSS_gaussian_blur_plot("South-Australia/South-Australia-20240201-20240207", 0, 1000)
#CYGGIBOI("South-Australia/South-Australia-20240201-20240207", False, 1.0)
#SMAP_gaussian_blur_plot("South-Australia", 1.0, 100)
#ERA5_gaussian_blur_plot("South-Australia/ERA5_South-Australia_2024_02_01_07.nc", 1.0, 0.9, 100)

#CYGNSS_gaussian_blur_plot("Sudan/Sudan-20200701-20200707", 0, 1000)
#CYGGIBOI("Sudan/Sudan-20240201-20240203", False, 1.5)
#SMAP_gaussian_blur_plot("Sudan", 2, 25)
#ERA5_gaussian_blur_plot("Sudan/ERA5_Sudan_2020_07_01_07.nc", 2, 0.9, 25)
#trace_plot("Sudan/Sudan-20240201-20240203", True)

#CYGNSS_gaussian_blur_plot("Western-Australia/Western-Australia-20201001-20201007", 0, 1000)
#trace_plot("Western-Australia/Western-Australia-20201001-20201007", False)
#CYGGIBOI("Western-Australia/Western-Australia-20201001-20201007", False, 1.5)
#SMAP_gaussian_blur_plot("Western-Australia", 2.5, 100)
#ERA5_gaussian_blur_plot("Western-Australia/ERA5_Western-Australia_2020_10_01_07.nc", 2.5, 0.9, 100)

#CYGNSS_gaussian_blur_plot("Sudan/Sudan-20200701-20200707", 1.5)
#LETSGO("Sudan/Sudan-20200701-20200707", 1.5, 100)
#trace_plot("Sudan/Sudan-20200701-20200707", True)
#SMAP_gaussian_blur_plot("Sudan", 1.5, 100)
#ERA5_gaussian_blur_plot("data/Timeseries/TimeSeries-Thailand-20240101-20241201/ERA5/ERA5_20240101_20240102.nc", 0.9, )

#CYGNSS_gaussian_blur_plot("Sudan/Sudan-20201001-20201003", 3.0)
#trace_plot("Sudan/Sudan-20201001-20201003", False)
#SMAP_gaussian_blur_plot("Sudan", 2.0, 100)
#ERA5_gaussian_blur_plot("Sudan/ERA5_Sudan_2020_10_01_03.nc", 2.0, 0.9, 100)

#CYGNSS_gaussian_blur_plot("Thailand/Thailand-20200101-20200107", 2.0, 100)
LETSGO("Thailand/Thailand-20200101-20200107", 2.0, 100)
#SMAP_gaussian_blur_plot("Thailand", 2.0, 100)
#ERA5_gaussian_blur_plot("Thailand/ERA5_Thailand_2020_01_01_07.nc", 2.0, 0.9, 100)

#CYGNSS_gaussian_blur_plot("India2/India2-20200101-20200107", 2, 100)
#LETSGO("India2/India2-20200101-20200107", 2, 100)
#SMAP_gaussian_blur_plot("India2", 2, 100)
#ERA5_gaussian_blur_plot("India2/ERA5_India2_2020_01_01_07.nc", 2, 0.9, 100)

#plot_time_series("data/Timeseries/TimeSeries-Thailand-20240101-20241201", 15, 16, 103, 104)


from CYGNSS.trace_plot import trace_plot
from CYGNSS.CYGNSS_gaussian_blur_plot import CYGNSS_gaussian_blur_plot
from CYGNSS.CYGNSS_gaussian_blur_plot_old import CYGNSS_gaussian_blur_plot as CYGGIBOI
from ERA5.ERA5_gaussian_blur_plot import ERA5_gaussian_blur_plot
from SMAP.SMAP_gaussian_blur_plot import SMAP_gaussian_blur_plot
from CYGNSS.trace_plot import trace_plot
from plot_timeseries import plot_time_series
<<<<<<< HEAD
from CYGNSS.CYGNSS_testing_limits import CYGNSS_gaussian_blur_plot as LETSGO
from plot_correlation import correlation_plot, correlation_matrix
from horse import correlation_vs_tri
from elevation_plot import correlation_3d_terrain
from elevation_only_plot import terrain_3d_plot

=======
from plot_correlation import correlation_plot, spatial_correlation_matrix
from CYGNSS.CYGNSS_average_plot import CYGNSS_average_plot
from ERA5.ERA5_gaussian_blur_plot_new import ERA5_regrid_and_blur
from timeseries_correlation import time_series_correlation
from CYGNSS.cygnss_trace_plot_new import CYGNSS_raw_plot_satellite
>>>>>>> 324c8fcc1ebba8a2a6a8acf47e673f1a7429c308


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
#SMAP_gaussian_blur_plot("Pakistan_9km_smap", 0, 0.001, 0.001)
#CYGNSS_gaussian_blur_plot("Western-Australia/Western-Australia-20200101-20200131", 0, 2000, True)
CYGNSS_average_plot("Western-Australia/Western-Australia-20200101-20200228", 0, 0.001, 0.001,  False)
#ERA5_gaussian_blur_plot("Western-Australia/ERA5_Western-Australia_20200601_20200607.nc", 0, 0.1, 0.08, 0.08)
#correlation_matrix("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9)
#correlation_plot("India2", "India2/India2-20200101-20200131", "India2/ERA5_India2_2020_01_01_31.nc", 0.5, 0.5, 0.9, gaussian_sigma=2)
#plot_time_series("data/Timeseries/TimeSeries-Australia-20180801-20200801", -35.4, 145.8, -34.6, 147.6, gaussian_sigma=5)

<<<<<<< HEAD
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
#LETSGO("Thailand/Thailand-20200101-20200107", 2.0, 100)
#SMAP_gaussian_blur_plot("South-Africa_20200101_20200131", 2.0, 100)
#ERA5_gaussian_blur_plot("South-Africa/ERA5_South-Africa_20200101_20200131.nc", 2.0, 0.9, 100)

#CYGNSS_gaussian_blur_plot("South-Africa/South-Africa-20200101-20200131", 2, 100)
#LETSGO("India2/India2-20200101-20200107", 2, 100)
#SMAP_gaussian_blur_plot("India2", 2, 100)
#ERA5_gaussian_blur_plot("India2/ERA5_India2_2020_01_01_07.nc", 2, 0.9, 100)

#plot_time_series("data/Timeseries/TimeSeries-Thailand-20240101-20241201", 15, 16, 103, 104)


#correlation_matrix("Bolivia", "Bolivia/Bolivia-20240701-20240707", "Bolivia/ERA5_Bolivia_2024_07_01_07.nc", 100, 15, 0.95)
correlation_matrix("India2", "India2/India2-20200101-20200107", "India2/ERA5_India2_2020_01_01_07.nc", 100, 10, 0.95)
#correlation_plot( "India2", "India2/India2-20200101-20200107", "India2/ERA5_India2_2020_01_01_07.nc", 0.95)

#correlation_vs_tri("Bolivia", "Bolivia/Bolivia-20240701-20240707", "data/ASTGTM/Bolivia_downsampled.nc", 0.5, 0.5)
#correlation_vs_tri("India2", "India2/India2-20200101-20200107", "India2/ERA5_India2_2020_01_01_07.nc", 0.95, "data/ASTGTM/Pakistan_downsampled.nc", 0.5, 0.5)
#correlation_vs_tri("Thailand", "Thailand/Thailand-20200101-20200107", "Thailand/ERA5_Thailand_2020_01_01_07.nc", 0.95, "data/ASTGTM/Thailand_downsampled.nc", 0.5, 0.5)
#correlation_vs_tri("India", "India/India-20240201-20240207", "India/ERA5_India_2024_02_01_07.nc", 0.95, "data/ASTGTM/India_downsampled.nc", 0.5, 0.5)

'''
correlation_3d_terrain(
    cygnss_folder="India2/India2-20200101-20200107",
    era5_folder="data/ERA5/India2/ERA5_India2_2020_01_01_07.nc",
    lsm_threshold=0.5,
    dem_file="data/ASTGTM/South_Africa_downsampled.nc",
    lat_step=0.1,
    lon_step=0.1
)
'''
'''
correlation_3d_terrain(
    smap_folder="India2",
    cygnss_folder="India2/India2-20200101-20200107",
    era5_file="data/ERA5/India2/ERA5_India2_2020_01_01_07.nc",
    dem_file="data/ASTGTM/Pakistan_downsampled.nc",
    lsm_threshold=0.5,
    lat_step=0.1,
    lon_step=0.1,
    dataset='SMAP'  # or 'ERA5'
)
'''

'''
terrain_3d_plot(cygnss_folder="South-Africa/South-Africa-20200101-20200131",
    era5_folder="data/ERA5/South-Africa/ERA5_South-Africa_20200101_20200131.nc", 
    lsm_threshold=0.5, 
    dem_file="data/ASTGTM/South_Africa_downsampled.nc", 
    lat_step=0.01, 
    lon_step=0.01
    )
'''

terrain_3d_plot(cygnss_folder="India2/India2-20200101-20200107",
    era5_folder="data/ERA5/India2/ERA5_India2_2020_01_01_07.nc", 
    lsm_threshold=0.5, 
    dem_file="data/ASTGTM/Pakistan_downsampled.nc", 
    lat_step=0.1, 
    lon_step=0.1
    )
=======
#CYGNSS_raw_plot_satellite("Sudan\Sudan-20201001-20201003")
>>>>>>> 324c8fcc1ebba8a2a6a8acf47e673f1a7429c308

from scatter_mapbox_plot import scatter_plot
from trace_plot import trace_plot
from gaussian_blur_plot import gaussian_blur_plot
from heatmap_test import rasterized_heatmap
'''
setting the boolean to true enables filesaving 
'''
#rasterized_heatmap("Pakistan-India-20200101-20200108", False, 1.0)
#scatter_plot("Pakistan-India-20200101-20200108", False)
trace_plot("Australia-20240912-20240918", False)
gaussian_blur_plot("Australia-20240912-20240918", False, 1.0)
#gaussian_blur_plot("Brazil-20240912-20240912", False, 1.5)

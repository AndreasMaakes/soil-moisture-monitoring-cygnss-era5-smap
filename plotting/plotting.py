from trace_plot import trace_plot
from gaussian_blur_plot import gaussian_blur_plot
'''
Setting the boolean to true enables filesaving 
'''

'''To plot, insert the location of the data folder. This should be on the format: Name/Name-yyyymmdd-yyyymmdd'''

'''Brazil plots - 7 days'''
#gaussian_blur_plot("Brazil/Brazil-20240912-20240918", False, 1.5)
#trace_plot("Brazil/Brazil-20240912-20240918", False)

'''Australia plots - 7 days'''  
#gaussian_blur_plot("Australia-20240912-20240918", False, 1.5)
#trace_plot("Australia-20240912-20240918", False)

'''Chad plots - 7 days'''
#gaussian_blur_plot("Chad/Chad-20240912-20240918", False, 1.5)
#trace_plot("Chad/Chad-20240912-20240914", False)

'''Gaussian blur plot - 3 days'''
#gaussian_blur_plot("Brazil/Brazil-20240912-20240912", False, 0)
#gaussian_blur_plot("Australia/Australia-20240912-20240914", False, 1)
#gaussian_blur_plot("Brazil/Brazil-20240912-20240918", False, 1.5)
trace_plot("Brazil/Brazil-20240912-20240918", False)

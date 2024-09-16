from pydap.client import open_url
from pydap.cas.urs import setup_session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import create_dates_array, data_fetching

'''Login to the server'''
username = "andreasmaakes"
password = "Terrengmodell69!"

'''
Dates format: "yyyymmdd"
'''

startDate = "20240101"
endDate = "20240101"

data_fetching(startDate, endDate, username, password)


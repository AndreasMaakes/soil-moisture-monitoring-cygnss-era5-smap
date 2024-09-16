from pydap.client import open_url
from pydap.cas.urs import setup_session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



'''Login to the server'''
username = "andreasmaakes"
password = "Terrengmodell69!"

'''Dates
format: s20180801-000000-e20180801-235959
'''
dates = ["20240701", "20240702", "20240703", "20240704", "20240705"]

'''URLS'''


#cyg03 indicates the satellite
#url= "https://opendap.earthdata.nasa.gov/collections/C2832195379-POCLOUD/granules/cyg03.ddmi.s20180801-000000-e20180801-235959.l1.power-brcs.a32.d33"

url= "https://opendap.earthdata.nasa.gov/collections/C2832195379-POCLOUD/granules/cyg03.ddmi.s"+dates[1]+"-000000-e"+dates[1]+"-235959.l1.power-brcs.a32.d33"

dataset = open_url(url, session=setup_session(username, password), protocol='dap4')


ddm_snr = np.array(dataset['ddm_snr'][:, 0][:45]) 
sp_lon = np.array(dataset['sp_lon'][:, 0][:45])
sp_lat = np.array(dataset['sp_lat'][:, 0][:45])

'''Creating a dataframe with the data'''
df = pd.DataFrame(sp_lon, columns = ['sp_lon'])
df['sp_lat'] = sp_lat
df['ddm_snr'] = ddm_snr
print(df)




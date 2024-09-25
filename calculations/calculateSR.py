import numpy as np
import pandas as pd
import xarray as xr
import sys
import os

# Add the absolute path to the 'plotting' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../plotting')))
import functions


#Constants
l1_wavelength = 0.1903
biases = [1.017,0.004,1.636,0,-0.610,0.24,-0.709,0.605,1.498,-0.783,-0.230,-1.021,0.007,-0.730,-0.376,-0.481,0.256,-0.206,-0.206,0.345,-0.909,-0.838,-0.858,1.140,0.880,0.163,0.409,-0.712,-1.032,0.877,-0.562,-0.819]

def calculateBias(prn):
    return biases[int(prn) - 1]

def sr(row):
    return row["ddm_snr"] - row["gps_tx_power_db_w"] - row["gps_ant_gain_db_i"] - row["sp_rx_gain"] - 20 * np.log10(l1_wavelength) + 20 * np.log10(row["tx_to_sp_range"] + row["rx_to_sp_range"]) + 20 * np.log10(4 * np.pi) + calculateBias(row["prn_code"]) - 140 #Subtracting 140 because why the hell not


dfs = functions.importData()
print(dfs[0].head(5))

for df in dfs:
    filename = df.name
    df["sr"] = df.apply(sr, axis = 1)
    ds = xr.Dataset.from_dataframe(df)
    ds.to_netcdf(f'./data/{filename}')

dfs2 = functions.importData()
print(dfs2[0].head(5))




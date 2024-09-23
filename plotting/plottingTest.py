import plotly.graph_objects as go
import pandas as pd
import xarray as xr


'''
# Load the data
data = xr.open_dataset("../Prosjektoppgave/data/cyg01_20240101.nc")
df = data.to_dataframe().reset_index()
df = df.drop(columns=['index', 'gps_tx_power_db_w', 'gps_ant_gain_db_i', 'sp_rx_gain', 'tx_to_sp_range', 'rx_to_sp_range', 'prn_code', 'quality_flags', 'sp_inc_angle'])
print(df.head(45))

fig = px.scatter_geo(df, lat='sp_lat', lon='sp_lon', hover_name='ddm_snr', projection = 'natural earth')
fig.show()
'''
data1 = xr.open_dataset("../Prosjektoppgave/data/cyg01_20240101.nc")
df1 = data1.to_dataframe().reset_index()

data2 = xr.open_dataset("../Prosjektoppgave/data/cyg02_20240101.nc")
df2 = data2.to_dataframe().reset_index()

data3 = xr.open_dataset("../Prosjektoppgave/data/cyg03_20240101.nc")
df3 = data3.to_dataframe().reset_index()

data4 = xr.open_dataset("../Prosjektoppgave/data/cyg04_20240101.nc")
df4 = data4.to_dataframe().reset_index()

data5 = xr.open_dataset("../Prosjektoppgave/data/cyg05_20240101.nc")
df5 = data5.to_dataframe().reset_index()

data7 = xr.open_dataset("../Prosjektoppgave/data/cyg07_20240101.nc")    
df7 = data7.to_dataframe().reset_index()

data8 = xr.open_dataset("../Prosjektoppgave/data/cyg08_20240101.nc")
df8 = data8.to_dataframe().reset_index()

trace1 = go.Scattergeo(
    lon = df1['sp_lon'],
    lat = df1['sp_lat'],
    text = df1['ddm_snr'],
    marker = dict(
        color = 'blue',
    )
)

trace2 = go.Scattergeo(
    lon = df2['sp_lon'],
    lat = df2['sp_lat'],
    text = df2['ddm_snr'],
    marker = dict(
        color = 'red',
    )
)

trace3 = go.Scattergeo(
    lon = df3['sp_lon'],
    lat = df3['sp_lat'],
    text = df3['ddm_snr'],
    marker = dict(
        color = 'green',
    )
)

trace4 = go.Scattergeo(
    lon = df4['sp_lon'],
    lat = df4['sp_lat'],
    text = df4['ddm_snr'],
    marker = dict(
        color = 'yellow',
    )
)

trace5 = go.Scattergeo(
    lon = df5['sp_lon'],
    lat = df5['sp_lat'],
    text = df5['ddm_snr'],
    marker = dict(
        color = 'orange',
    )
)

trace7 = go.Scattergeo(
    lon = df7['sp_lon'],
    lat = df7['sp_lat'],
    text = df7['ddm_snr'],
    marker = dict(
        color = 'purple',
    )
)

trace8 = go.Scattergeo(
    lon = df8['sp_lon'],
    lat = df8['sp_lat'],
    text = df8['ddm_snr'],
    marker = dict(
        color = 'black',
    )
)

fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5, trace7, trace8])

fig.update_layout(
    title = 'DDM_SNR on 20240101',
    geo = dict(
        projection_type = 'natural earth',
        scope = 'asia',
    )
)

fig.show()
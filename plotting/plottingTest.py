import plotly.graph_objects as go
import pandas as pd
import xarray as xr
import plotly.express as px

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


mapbox_token = 'pk.eyJ1Ijoib2xlZXZjYSIsImEiOiJjbTFldmt6aGIyeWN4MmxzamFrYTV3dTNxIn0.bbVpqBfsIl_Y0W7YGRXCgQ'
px.set_mapbox_access_token(mapbox_token)


trace1 = go.Scattermapbox(
    lon = df1['sp_lon'],
    lat = df1['sp_lat'],
    text = df1['ddm_snr'],
    marker = dict(
        color = 'blue',
    )
)

trace2 = go.Scattermapbox(
    lon = df2['sp_lon'],
    lat = df2['sp_lat'],
    text = df2['ddm_snr'],
    marker = dict(
        color = 'red',
    )
)

trace3 = go.Scattermapbox(
    lon = df3['sp_lon'],
    lat = df3['sp_lat'],
    text = df3['ddm_snr'],
    marker = dict(
        color = 'green',
    )
)

trace4 = go.Scattermapbox(
    lon = df4['sp_lon'],
    lat = df4['sp_lat'],
    text = df4['ddm_snr'],
    marker = dict(
        color = 'yellow',
    )
)

trace5 = go.Scattermapbox(
    lon = df5['sp_lon'],
    lat = df5['sp_lat'],
    text = df5['ddm_snr'],
    marker = dict(
        color = 'orange',
    )
)

trace7 = go.Scattermapbox(
    lon = df7['sp_lon'],
    lat = df7['sp_lat'],
    text = df7['ddm_snr'],
    marker = dict(
        color = 'purple',
    )
)

trace8 = go.Scattermapbox(
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
    mapbox = dict(
        style = 'light',
        zoom = 4,
        center = dict(lon = 59.5, lat = 32),
        accesstoken = mapbox_token
    )
)

fig.show()
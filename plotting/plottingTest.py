import plotly.graph_objects as go
import pandas as pd
import xarray as xr
import plotly.express as px

'''
# Load the data
data = xr.open_dataset("../Prosjektoppgave/data/cyg01_20240101.nc")
df = data.to_dataframe().reset_index()
df = df.drop(columns=['index', 'gps_tx_power_db_w', 'gps_ant_gain_db_i', 'gps_tx_power_db_w', 'tx_to_sp_range', 'gps_tx_power_db_w', 'prn_code', 'quality_flags', 'sp_inc_angle'])
print(df.head(45))

fig = px.scatter_geo(df, lat='sp_lat', lon='sp_lon', hover_name='gps_tx_power_db_w', projection = 'natural earth')
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

data1_2 = xr.open_dataset("../Prosjektoppgave/data/cyg01_20240102.nc")
df1_2 = data1_2.to_dataframe().reset_index()

data2_2 = xr.open_dataset("../Prosjektoppgave/data/cyg02_20240102.nc")
df2_2 = data2_2.to_dataframe().reset_index()

data3_2 = xr.open_dataset("../Prosjektoppgave/data/cyg03_20240102.nc")
df3_2 = data3_2.to_dataframe().reset_index()

data4_2 = xr.open_dataset("../Prosjektoppgave/data/cyg04_20240102.nc")
df4_2 = data4_2.to_dataframe().reset_index()

data5_2 = xr.open_dataset("../Prosjektoppgave/data/cyg05_20240102.nc")
df5_2 = data5_2.to_dataframe().reset_index()

data7_2 = xr.open_dataset("../Prosjektoppgave/data/cyg07_20240102.nc")
df7_2 = data7_2.to_dataframe().reset_index()

data8_2 = xr.open_dataset("../Prosjektoppgave/data/cyg08_20240102.nc")
df8_2 = data8_2.to_dataframe().reset_index()

data1_3 = xr.open_dataset("../Prosjektoppgave/data/cyg01_20240103.nc")
df1_3 = data1_3.to_dataframe().reset_index()

data2_3 = xr.open_dataset("../Prosjektoppgave/data/cyg02_20240103.nc")
df2_3 = data2_3.to_dataframe().reset_index()

data3_3 = xr.open_dataset("../Prosjektoppgave/data/cyg03_20240103.nc")
df3_3 = data3_3.to_dataframe().reset_index()

data4_3 = xr.open_dataset("../Prosjektoppgave/data/cyg04_20240103.nc")
df4_3 = data4_3.to_dataframe().reset_index()

data5_3 = xr.open_dataset("../Prosjektoppgave/data/cyg05_20240103.nc")
df5_3 = data5_3.to_dataframe().reset_index()

data7_3 = xr.open_dataset("../Prosjektoppgave/data/cyg07_20240103.nc")
df7_3 = data7_3.to_dataframe().reset_index()

data8_3 = xr.open_dataset("../Prosjektoppgave/data/cyg08_20240103.nc")
df8_3 = data8_3.to_dataframe().reset_index()




mapbox_token = 'pk.eyJ1Ijoib2xlZXZjYSIsImEiOiJjbTFldmt6aGIyeWN4MmxzamFrYTV3dTNxIn0.bbVpqBfsIl_Y0W7YGRXCgQ'
px.set_mapbox_access_token(mapbox_token)


trace1 = go.Scattermapbox(
    lon = df1['sp_lon'],
    lat = df1['sp_lat'],
    text = df1['gps_tx_power_db_w'],
    marker = dict(
        color = 'blue',
    )
)

trace2 = go.Scattermapbox(
    lon = df2['sp_lon'],
    lat = df2['sp_lat'],
    text = df2['gps_tx_power_db_w'],
    marker = dict(
        color = 'red',
    )
)

trace3 = go.Scattermapbox(
    lon = df3['sp_lon'],
    lat = df3['sp_lat'],
    text = df3['gps_tx_power_db_w'],
    marker = dict(
        color = 'green',
    )
)

trace4 = go.Scattermapbox(
    lon = df4['sp_lon'],
    lat = df4['sp_lat'],
    text = df4['gps_tx_power_db_w'],
    marker = dict(
        color = 'yellow',
    )
)

trace5 = go.Scattermapbox(
    lon = df5['sp_lon'],
    lat = df5['sp_lat'],
    text = df5['gps_tx_power_db_w'],
    marker = dict(
        color = 'orange',
    )
)

trace7 = go.Scattermapbox(
    lon = df7['sp_lon'],
    lat = df7['sp_lat'],
    text = df7['gps_tx_power_db_w'],
    marker = dict(
        color = 'purple',
    )
)

trace8 = go.Scattermapbox(
    lon = df8['sp_lon'],
    lat = df8['sp_lat'],
    text = df8['gps_tx_power_db_w'],
    marker = dict(
        color = 'black',
    )
)

trace1_2 = go.Scattermapbox(
    lon = df1_2['sp_lon'],
    lat = df1_2['sp_lat'],
    text = df1_2['gps_tx_power_db_w'],
    marker = dict(
        color = 'blue',
    )
)

trace2_2 = go.Scattermapbox(
    lon = df2_2['sp_lon'],
    lat = df2_2['sp_lat'],
    text = df2_2['gps_tx_power_db_w'],
    marker = dict(
        color = 'red',
    )
)

trace3_2 = go.Scattermapbox(
    lon = df3_2['sp_lon'],
    lat = df3_2['sp_lat'],
    text = df3_2['gps_tx_power_db_w'],
    marker = dict(
        color = 'green',
    )
)

trace4_2 = go.Scattermapbox(
    lon = df4_2['sp_lon'],
    lat = df4_2['sp_lat'],
    text = df4_2['gps_tx_power_db_w'],
    marker = dict(
        color = 'yellow',
    )
)

trace5_2 = go.Scattermapbox(
    lon = df5_2['sp_lon'],
    lat = df5_2['sp_lat'],
    text = df5_2['gps_tx_power_db_w'],
    marker = dict(
        color = 'orange',
    )
)

trace7_2 = go.Scattermapbox(
    lon = df7_2['sp_lon'],
    lat = df7_2['sp_lat'],
    text = df7_2['gps_tx_power_db_w'],
    marker = dict(
        color = 'purple',
    )
)

trace8_2 = go.Scattermapbox(
    lon = df8_2['sp_lon'],
    lat = df8_2['sp_lat'],
    text = df8_2['gps_tx_power_db_w'],
    marker = dict(
        color = 'black',
    )
)

trace1_3 = go.Scattermapbox(
    lon = df1_3['sp_lon'],
    lat = df1_3['sp_lat'],
    text = df1_3['gps_tx_power_db_w'],
    marker = dict(
        color = 'blue',
    )
)

trace2_3 = go.Scattermapbox(
    lon = df2_3['sp_lon'],
    lat = df2_3['sp_lat'],
    text = df2_3['gps_tx_power_db_w'],
    marker = dict(
        color = 'red',
    )
)

trace3_3 = go.Scattermapbox(
    lon = df3_3['sp_lon'],
    lat = df3_3['sp_lat'],
    text = df3_3['gps_tx_power_db_w'],
    marker = dict(
        color = 'green',
    )
)

trace4_3 = go.Scattermapbox(
    lon = df4_3['sp_lon'],
    lat = df4_3['sp_lat'],
    text = df4_3['gps_tx_power_db_w'],
    marker = dict(
        color = 'yellow',
    )
)

trace5_3 = go.Scattermapbox(
    lon = df5_3['sp_lon'],
    lat = df5_3['sp_lat'],
    text = df5_3['gps_tx_power_db_w'],
    marker = dict(
        color = 'orange',
    )
)

trace7_3 = go.Scattermapbox(
    lon = df7_3['sp_lon'],
    lat = df7_3['sp_lat'],
    text = df7_3['gps_tx_power_db_w'],
    marker = dict(
        color = 'purple',
    )
)

trace8_3 = go.Scattermapbox(
    lon = df8_3['sp_lon'],
    lat = df8_3['sp_lat'],
    text = df8_3['gps_tx_power_db_w'],
    marker = dict(
        color = 'black',
    )
)



fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5, trace7, trace8, trace1_2, trace2_2, trace3_2, trace4_2, trace5_2, trace7_2, trace8_2, trace1_3, trace2_3, trace3_3, trace4_3, trace5_3, trace7_3, trace8_3])

fig.update_layout(
    title = 'gps_tx_power_db_w on 20240101',
    mapbox = dict(
        style = 'light',
        zoom = 4,
        center = dict(lon = 59.5, lat = 32),
        accesstoken = mapbox_token
    )
)

fig.show()
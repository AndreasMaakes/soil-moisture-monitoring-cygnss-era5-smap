import plotly.express as px
import plotly.graph_objects as go
from functions import importData, createTraces

mapbox_token = 'pk.eyJ1Ijoib2xlZXZjYSIsImEiOiJjbTFldmt6aGIyeWN4MmxzamFrYTV3dTNxIn0.bbVpqBfsIl_Y0W7YGRXCgQ'
px.set_mapbox_access_token(mapbox_token)

importData()

fig = go.Figure(data=createTraces(importData()))

fig.update_layout(
    title = 'ddm_snr on 20240901-20240905',
    mapbox = dict(
        style = 'outdoors',
        zoom = 4,
        center = dict(lon = 59.5, lat = 32),
        accesstoken = mapbox_token
    )
)

fig.show()

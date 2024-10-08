import plotly.graph_objects as go

#Function to create traces to plot and visualize variables from CYGNSS data
def createTraces(dataFrames):
    traces = []
    colors = ['blue', 'red', 'green']
    for df in dataFrames:
        name = df.name
        if name.endswith('11.nc'):
            color = colors[0]
        elif name.endswith('12.nc'):
            color = colors[1]
        else:
            color = colors[2]
        trace = go.Scattermapbox(
            lon = df['sp_lon'],
            lat = df['sp_lat'],
            text = df['ddm_snr'],
            marker = dict(
                color = color,
            ),
            name = df.name
        )
        traces.append(trace)
    return traces



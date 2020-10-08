import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px

from ILAMB.ModelResult import ModelResult
from sklearn.cluster import KMeans

def GetVariables():
    m = ModelResult("/home/nate/data/ILAMB/MODELS/CMIP6/CESM2")
    t0,tf = 365*(1980-1850),365*(2000-1850)
    Vs = {}
    mask = None
    for vname in ['tas','pr','gpp','ra','nbp','hurs']:
        v = m.extractTimeSeries(vname,initial_time=t0,final_time=tf)
        v = v.integrateInTime(mean=True)
        if mask is None: mask = v.data.mask
        mask += v.data.mask
        Vs[vname] = v
    data = None
    for i,vname in enumerate(Vs):
        v = Vs[vname].data
        v = np.ma.masked_array(v,mask=mask)
        v = v.compressed()
        if data is None: data = np.zeros((v.size,len(Vs.keys())))
        data[:,i] = v        
    kmeans = KMeans(n_clusters=10,random_state=0).fit(data)
    out = {}
    for i,vname in enumerate(Vs):
        out[vname] = kmeans.cluster_centers_[:,i]
    return out

# make a sample data frame where the columns are the cluster centroids
df = pd.DataFrame(GetVariables())

# make the Dash app, setup the html structure
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.layout = html.Div([
    html.Div(
        dcc.Graph(id='g1', config={'displayModeBar': False}),
        className='four columns'
    ),
    html.Div(
        dcc.Graph(id='g2', config={'displayModeBar': False}),
        className='four columns'
        ),
    html.Div(
        dcc.Graph(id='g3', config={'displayModeBar': False}),
        className='four columns'
    )
], className='row')


def get_figure(df, x_col, y_col, selectedpoints, selectedpoints_local):

    if selectedpoints_local and selectedpoints_local['range']:
        ranges = selectedpoints_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
    else:
        selection_bounds = {'x0': np.min(df[x_col]), 'x1': np.max(df[x_col]),
                            'y0': np.min(df[y_col]), 'y1': np.max(df[y_col])}

    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute
    fig = px.scatter(df, x=df[x_col], y=df[y_col], text=df.index)
    fig.update_traces(selectedpoints=selectedpoints, 
                      customdata=df.index,
                      mode='markers+text',
                      marker={ 'color': 'rgba(0, 116, 217, 0.7)',
                               'size' : 20 },
                      unselected={'marker'  : { 'opacity': 0.30 },
                                  'textfont': { 'color'  : 'rgba(0, 0, 0, 0)' }})
    fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)
    fig.add_shape(dict({'type': 'rect', 
                        'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }}, 
                      **selection_bounds))
    return fig

# this callback defines 3 figures
# as a function of the intersection of their 3 selections
@app.callback(
    [Output('g1', 'figure'),
     Output('g2', 'figure'),
     Output('g3', 'figure')],
    [Input('g1', 'selectedData'),
     Input('g2', 'selectedData'),
     Input('g3', 'selectedData')]
)
def callback(selection1, selection2, selection3):
    selectedpoints = df.index
    for selected_data in [selection1, selection2, selection3]:
        if selected_data and selected_data['points']:
            selectedpoints = np.intersect1d(selectedpoints,
                [p['customdata'] for p in selected_data['points']])

    return [get_figure(df, "tas", "pr", selectedpoints, selection1),
            get_figure(df, "gpp", "ra", selectedpoints, selection2),
            get_figure(df, "nbp", "hurs", selectedpoints, selection3)]


if __name__ == '__main__':

    app.run_server(debug=True)

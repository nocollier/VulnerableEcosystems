import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np

# read in the clustering datafiles
names = ['variable_%d' % i for i in range(22)]
df  = pd.read_csv("seeds.out.cwe_regimes.50.final"      ,sep="\t",header=None,names=['junk',]+names)
dfu = pd.read_csv("seeds.out.cwe_regimes.50.final.unstd",sep=" " ,header=None,names=['junk',]+names)

# make the Dash app, setup the html structure
pulldown_style = dict(width='90%',display='inline-block',verticalAlign="middle",horizontalAlign="right")
options = [dict(label=n,value=n) for n in names]
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.layout = html.Div([
    html.Div(
        [html.Label(["x-axis ",dcc.Dropdown(id="p1x",value=names[0],options=options,style=pulldown_style)]),
         html.Label(["y-axis ",dcc.Dropdown(id="p1y",value=names[1],options=options,style=pulldown_style)]),
         dcc.Graph(id='g1', config={'displayModeBar': False})],className='four columns'),
    html.Div(
        [html.Label(["x-axis ",dcc.Dropdown(id="p2x",value=names[2],options=options,style=pulldown_style)]),
         html.Label(["y-axis ",dcc.Dropdown(id="p2y",value=names[3],options=options,style=pulldown_style)]),
         dcc.Graph(id='g2', config={'displayModeBar': False})],className='four columns'),
    html.Div(
        [html.Label(["x-axis ",dcc.Dropdown(id="p3x",value=names[4],options=options,style=pulldown_style)]),
         html.Label(["y-axis ",dcc.Dropdown(id="p3y",value=names[5],options=options,style=pulldown_style)]),
         dcc.Graph(id='g3', config={'displayModeBar': False})],className='four columns'),
    html.Div(
        [dcc.Checklist(id='std',value=[],options=[{'label': 'standardized data', 'value': 'standardized data'}])],
        className='four columns'),
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
                      unselected={'marker'  : { 'opacity': 0.05 },
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
     Input('g3', 'selectedData'),
     Input('p1x', 'value'),
     Input('p1y', 'value'),
     Input('p2x', 'value'),
     Input('p2y', 'value'),
     Input('p3x', 'value'),
     Input('p3y', 'value'),
     Input('std', 'value')]
)
def callback(selection1, selection2, selection3, v1x, v1y, v2x, v2y, v3x, v3y, std):
    DF = df if 'standardized data' in std else dfu
    selectedpoints = df.index
    for selected_data in [selection1, selection2, selection3]:
        if selected_data and selected_data['points']:
            selectedpoints = np.intersect1d(selectedpoints,
                [p['customdata'] for p in selected_data['points']])
    return [get_figure(DF, v1x, v1y, selectedpoints, selection1),
            get_figure(DF, v2x, v2y, selectedpoints, selection2),
            get_figure(DF, v3x, v3y, selectedpoints, selection3)]

if __name__ == '__main__':
    app.run_server(debug=True)

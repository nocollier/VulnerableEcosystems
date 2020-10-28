import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

"""
seed_file = "4seasons/k20/results/seeds.out.4seasons_historicalssp585.20.final"
cluster_file = "4seasons/k20/results/clusters.out.4seasons_historicalssp585.20"
coords_file = "4seasons/coords.4seasons_historicalssp585"
names = ["Q1 gpp","Q2 gpp","Q3 gpp","Q4 gpp",
         "Q1 pr" ,"Q2 pr" ,"Q3 pr" ,"Q4 pr" ,
         "Q1 tas","Q2 tas","Q3 tas","Q4 tas"]
t = np.asarray(range(1850,2101,10),dtype=int)
"""

# stuff from the files
seed_file = "seasons/k20/results/seeds.out.seasons_historicalssp585.20.final"
cluster_file = "seasons/k20/results/clusters.out.seasons_historicalssp585.20"
coords_file = "seasons/coords.seasons_historicalssp585"
names = ["gpp [g m-2 d-1]","seasonal pr [mm d-1]","tas [degK]"]
conversions = {"gpp [g m-2 d-1]"      : 8.640000e+00,
               "seasonal pr [mm d-1]" : 8.655580e-03,
               "tas [degK]"           : 1e-7}

t = np.asarray(range(1850,2101,10),dtype=int)

# read in the clustering datafiles
df  = pd.read_csv(seed_file           ,sep="\t",header=None,names=['junk',]+names)
dfu = pd.read_csv(seed_file + ".unstd",sep=" " ,header=None,names=['junk',]+names)
dfc = pd.read_csv(coords_file         ,sep=" " ,header=None,names=['lon','lat'])
dfi = pd.read_csv(cluster_file        ,sep=" " ,header=None,names=['id'])
for key in conversions:
    dfu[key] *= conversions[key]

# intermediate quantities
ntimes = t.size-1
ncells = int(dfi.size / ntimes)
marks = {}
for year in t: marks[int(year)] = {'label':'%d' % year}
style = 'none'
dfc = dfc.iloc[:ncells] # we only need the first time slice of coordinates
dfc.lon -= 180    

# reduce the cluster dataframes
dfi = pd.DataFrame(dict(id=dfi.id,cell=np.tile(range(ncells),ntimes),time=np.repeat(t[:-1],ncells)))
dfr = dfi.groupby(['id','time']).count().reset_index()

# make the Dash app, setup the html structure
options = [dict(label=n,value=n) for n in names]
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
panel1 = [html.Label(["x-axis ",dcc.Dropdown(id="p1x",value=names[1],options=options)],style={'width':'50%','display':'inline-block'}),
          html.Label(["y-axis ",dcc.Dropdown(id="p1y",value=names[0],options=options)],style={'width':'50%','display':'inline-block'}),
          dcc.Graph(id='g1',style={'height':'80%'})]
panel2 = [html.Label(["x-axis ",dcc.Dropdown(id="p2x",value=names[2],options=options)],style={'width':'50%','display':'inline-block'}),
          html.Label(["y-axis ",dcc.Dropdown(id="p2y",value=names[0],options=options)],style={'width':'50%','display':'inline-block'}),
          dcc.Graph(id='g2',style={'height':'80%'})]
panel3 = [html.Label(["x-axis ",dcc.Dropdown(id="p3x",value=names[2],options=options)],style={'width':'50%','display':'inline-block'}),
          html.Label(["y-axis ",dcc.Dropdown(id="p3y",value=names[1],options=options)],style={'width':'50%','display':'inline-block'}),
          dcc.Graph(id='g3',style={'height':'80%'})]
panel4 = dcc.RangeSlider(id='slide',min=t[0],max=t[-1],value=[2010,t[-1]],marks=marks,included=True)
panel5 = dcc.Graph(id='line',style={'height':'100%','width':'100%'})
panel6 = dcc.Graph(id='g4',style={'height':'100%','width':'100%'})
panel7 = dcc.Graph(id='g5',style={'height':'100%','width':'100%'})

app.layout = html.Div([
    html.Div([
        html.Div(panel1,style={'width':'33%','height':'100%','display':'inline-block','border-style':style}),
        html.Div(panel2,style={'width':'33%','height':'100%','display':'inline-block','border-style':style}),
        html.Div(panel3,style={'width':'33%','height':'100%','display':'inline-block','border-style':style}),
    ],style={'height':'40%'}),
    html.Div(panel4,style={'width':'100%','height':'5%' ,'display':'inline-block','border-style':style,'verticalAlign':'middle'}),
    html.Div([
        html.Div(panel5,style={'width':'26%' ,'height':'99%','display':'inline-block','border-style':style}),
        html.Div(panel6,style={'width':'37%' ,'height':'99%','display':'inline-block','border-style':style}),
        html.Div(panel7,style={'width':'37%' ,'height':'99%','display':'inline-block','border-style':style})
    ],style={'height':'55%'}),
],style = {'width':'100%','height': '95vh'}
)

def update_parameter_plots(df, x_col, y_col, selectedpoints, selectedpoints_local):
    if selectedpoints_local and selectedpoints_local['range']:
        ranges = selectedpoints_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
    else:
        selection_bounds = {'x0': np.min(df[x_col]), 'x1': np.max(df[x_col]),
                            'y0': np.min(df[y_col]), 'y1': np.max(df[y_col])}
    fig = px.scatter(df, x=df[x_col], y=df[y_col], text=df.index+1)
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
     Input('p3y', 'value')]
)
def callback(selection1, selection2, selection3, v1x, v1y, v2x, v2y, v3x, v3y):
    DF = dfu #df if 'standardized data' in std else dfu
    selectedpoints = DF.index
    for selected_data in [selection1, selection2, selection3]:
        if selected_data and selected_data['points']:
            selectedpoints = np.intersect1d(selectedpoints,
                [p['customdata'] for p in selected_data['points']])
    return [update_parameter_plots(DF, v1x, v1y, selectedpoints, selection1),
            update_parameter_plots(DF, v2x, v2y, selectedpoints, selection2),
            update_parameter_plots(DF, v3x, v3y, selectedpoints, selection3)]

@app.callback(
    [Output('line', 'figure'),
     Output('g4', 'figure'),
     Output('g5', 'figure')],
    Input('slide', 'value')
)
def callback(tvalue):
    
    tvalue[0] = t[t.searchsorted(tvalue[0],side='right')-1]
    tvalue[0] = tvalue[0] if tvalue[0] < t[-1] else t[-2]
    q = dfi.query("time == %d" % tvalue[0])
    q = pd.DataFrame({'id':q.id.to_numpy(),'lat':dfc.lat.to_numpy(),'lon':dfc.lon.to_numpy()})
    f1 = px.scatter(q,x='lon',y='lat',color='id')
    f1.update_traces(marker=dict(size=4))
    f1.update(layout_coloraxis_showscale=False)
    
    tvalue[1] = t[t.searchsorted(tvalue[1],side='right')-1]
    tvalue[1] = tvalue[1] if tvalue[1] < t[-1] else t[-2]
    q = dfi.query("time == %d" % tvalue[1])
    q = pd.DataFrame({'id':q.id.to_numpy(),'lat':dfc.lat.to_numpy(),'lon':dfc.lon.to_numpy()})
    f2 = px.scatter(q,x='lon',y='lat',color='id')
    f2.update_traces(marker=dict(size=4))
    f2.update(layout_coloraxis_showscale=False)

    lo = dfr.cell.min()
    hi = dfr.cell.max()
    ln = px.line(dfr,x='time',y='cell',color='id',color_discrete_sequence=px.colors.qualitative.Dark24)
    ln.add_trace(go.Scatter(x=[tvalue[0],tvalue[0],tvalue[1],tvalue[1],tvalue[1]],
                            y=[lo,hi,None,lo,hi],
                            mode='lines',line=dict(color='black',dash='dash')))
    ln.update_layout(showlegend=False)
    return ln,f1,f2

if __name__ == '__main__':
    app.run_server(debug=True)

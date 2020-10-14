import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from ILAMB.Scoreboard import ParseScoreboardConfigureFile
from ILAMB.run import ParseModelSetup
import os
from ILAMB.Confrontation import Confrontation
import ILAMB.ilamblib as il
import numpy as np
import pandas as pd
import plotly.express as px
import time,datetime
import cftime as cf
from ILAMB.constants import lbl_months
global app_ctx
app_ctx = {'rebuild':True}

def CreateFeatureList(varlist):
    """Create a nested dictionary of polygons from a list of ILAMB Variables.
    """
    var = varlist[0]
    lat_bnds = var.lat_bnds
    lon_bnds = var.lon_bnds    
    lnd = (var.data.mask==False)
    if var.temporal: raise ValueError("Not for temporal variables")
    i,j = np.where(lnd)
    n = i.size
    x = lon_bnds[j]
    y = lat_bnds[i]
    cells = {'type':'FeatureCollection','features':[]}
    for k in range(n):
        if not lnd[i[k],j[k]]: continue
        cells['features'].append({'type':'Feature',
                                  'properties': {'ID': '%05d' % k},
                                  'geometry': {'type': 'Polygon',
                                               'coordinates':[[[x[k,0],y[k,0]],
                                                               [x[k,0],y[k,1]],
                                                               [x[k,1],y[k,1]],
                                                               [x[k,1],y[k,0]],
                                                               [x[k,0],y[k,0]]]]},
                                  'id': '%05d' % k})
    df = {'id':[('{0:0%dd}' % (len(str(n)))).format(k) for k in range(n)],
          'lat':np.round(y.mean(axis=1)),
          'lon':np.round(x.mean(axis=1))}
    for var in varlist: df[var.name] = var.data[i,j]
    df = pd.DataFrame(df)
    return cells,df

def create_line_plot(var):
    if var.data.ndim == 1:
        df2 = pd.DataFrame({'time':var.time,'mean':var.data[:]})
    else:
        df2 = pd.DataFrame({'time':var.time,'mean':var.data[:,0]})
    month = np.asarray([t.month for t in df2.time],dtype=int)-1
    mean = []; low = []; high = []
    for i in np.unique(month):
        sdf = df2['mean'][month==i]
        q = sdf.quantile([0.25,0.75])
        mean.append(sdf.mean())
        low.append(mean[-1]-q.iloc[0])
        high.append(q.iloc[1]-mean[-1])
    df3 = pd.DataFrame({'month':lbl_months,'cycle':mean,'25%':low,'75%':high})
    dy = 0.05*(df2['mean'].max()-df2['mean'].min())
    ymin = df2['mean'].min()-dy
    ymax = df2['mean'].max()+dy
    
    f2 = px.line(df2,x="time",y="mean")
    f2.update_traces(line = dict(color = '#266e2e'))
    f2.update_layout(title = 'Time Series @ (%.1f,%.1f)' % (var.lat,var.lon),
                     yaxis = dict(range = [ymin,ymax]),
                     plot_bgcolor = '#bfbfbf',
                     font = dict(size=18))
    f3 = px.line(df3,x="month",y="cycle",error_y_minus="25%",error_y="75%")
    f3.update_traces(line = dict(color = '#266e2e'))
    f3.update_layout(title = 'Mean Annual Cycle',
                     yaxis = dict(range = [ymin,ymax]),
                     plot_bgcolor = '#bfbfbf',
                     font = dict(size=18))
    return f2,f3

def UpdateGlobalContext(model,variable,source,quantity):
    global app_ctx
    if not app_ctx['rebuild']: return
    t0 = time.time()
    app_ctx['model'] = model
    app_ctx['variable'] = variable
    app_ctx['source'] = source
    app_ctx['quantity'] = quantity
    
    # get the model and confrontation
    m = [m for m in M if m.name == model][0]
    c = None
    for h1 in cfg.children:
        for v in h1.children:
            if v.name != variable: continue
            for s in v.children:
                if s.name == source: c = s
    ref,mod = c.confrontation.stageData(m)
    app_ctx['ref'] = ref
    app_ctx['mod'] = mod
    app_ctx['c'] = c
    print("  + stageData  {0:>8}".format(str(datetime.timedelta(seconds=int(np.round((time.time()-t0))))))); t0 = time.time()

    # analysis phase
    ref_mean = ref.integrateInTime(mean=True)
    mod_mean = mod.integrateInTime(mean=True)
    if c.plot_unit is not None:
        ref_mean.convert(c.plot_unit)
        mod_mean.convert(c.plot_unit)
    REF_mean = ref_mean.interpolate(lat=mod_mean.lat,lon=mod_mean.lon)
    bias = REF_mean.bias(mod_mean)
    mod.time = cf.num2date(mod.time,"days since 1850-01-01")
    mod_mean.name = "mean"
    bias.name = "bias"
    app_ctx['ref_mean'] = ref_mean
    app_ctx['mod_mean'] = mod_mean
    app_ctx['bias'] = bias
    print("  + confront   {0:>8}".format(str(datetime.timedelta(seconds=int(np.round((time.time()-t0))))))); t0 = time.time()
        
    # plot map
    cells,df = CreateFeatureList([mod_mean,bias])
    app_ctx['cells'] = cells
    app_ctx['df'] = df
    
    return

cfg = ParseScoreboardConfigureFile("cmip.cfg")
M   = ParseModelSetup("models.txt")

# initialize some variables
models = [m.name for m in M]
variables = []
for h1 in cfg.children:
    for v in h1.children:
        for node in v.children:
            if node.cmap is None: node.cmap = "jet"
            node.output_path = "./"
            node.regions = ['global']
            node.source = os.path.join(os.environ["ILAMB_ROOT"],node.source)
            try:
                node.confrontation = Confrontation(**(node.__dict__))
            except:
                pass
        v.children = [n for n in v.children if n.confrontation is not None]
        if len(v.children) > 0: variables.append(v.name)
    h1.children = [v for v in h1.children if len(v.children)>0]
quantities = ['Mean','Bias']

# dash stuff
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(id='model',options=[{'label': m,'value': m} for m in models],value=models[0]),
        dcc.Dropdown(id='variable',options=[{'label': v,'value': v} for v in variables],value=variables[0]),
        dcc.Dropdown(id='source'),
        dcc.Dropdown(id='quantity',options=[{'label': q,'value': q} for q in quantities],value=quantities[0]),
    ],style = {'width':'25%','height': '67%','display':'inline-block'}),
    html.Div([
        dcc.Graph(id='map-plot',style={'height':'66vh'})
    ], style = {'width':'74%','height': '67%','display':'inline-block'}),
    html.Div([
        html.Div([
            dcc.Graph(id='line-plot')
        ], style = {'width':'66%','height':'33%','display':'inline-block'}),
        html.Div([
            dcc.Graph(id='cycle-plot')
        ], style = {'width':'33%','height':'33%','display':'inline-block'}),
    ],style = {'width':'100%','height': '33%'})
],style = {'width':'100%','height': '95vh'})

@app.callback(
    [Output('source','options'),Output('source','value')],
    Input('variable', 'value'))
def set_source_options(variable):
    sources = []
    for h1 in cfg.children:
        for v in h1.children:
            if v.name != variable: continue
            for s in v.children:
                if s.confrontation is None: continue
                sources.append(s.name)
    if len(sources) == 0: return None,None
    return [{'label': s, 'value': s} for s in sources],sources[0]

@app.callback(
    [Output('line-plot','figure'),Output('cycle-plot','figure')],
    [Input('map-plot','clickData')])
def clickme(clickData):
    global app_ctx
    if "df" not in app_ctx: raise dash.exceptions.PreventUpdate()
    df = app_ctx['df']
    mod = app_ctx['mod']
    if clickData is None:
        location = df.id[0]
    else:
        location = clickData['points'][0]['location']
    DF = df[df.id == location]
    return create_line_plot(mod.extractDatasites(DF.lat.to_numpy(),DF.lon.to_numpy()))

@app.callback(
    Output('map-plot','figure'),
    [Input('model','value'),Input('variable','value'),Input('source','value'),Input('quantity','value')])
def update_plots(model,variable,source,quantity):
    global app_ctx
    print("updating plot...")

    if 'model' in app_ctx:
        if (app_ctx['model'   ] == model and
            app_ctx['variable'] == variable and
            app_ctx['source'  ] == source): app_ctx['rebuild'] = False
    UpdateGlobalContext(model,variable,source,quantity)
    app_ctx['rebuild'] = True
    
    t0 = time.time()
    df = app_ctx['df']
    c = app_ctx['c']
    if quantity == 'Mean':
        col  = 'mean'
        low  = df['mean'].min()
        high = df['mean'].max()
        cmap = c.cmap
    elif quantity == 'Bias':
        col  = 'bias'
        q    = df['bias'].abs().quantile(0.98)
        low  = -q
        high =  q
        cmap = 'RdBu_r'
    fig = px.choropleth(df,
                        projection = 'robinson',
                        geojson = app_ctx['cells'], locations='id', color=col,
                        range_color = (low,high),
                        color_continuous_scale = cmap,
                        hover_data = ['lat','lon','mean'])
    fig.update_traces(marker_line_width=0)
    fig.update_layout(font=dict(size=18),geo=dict(landcolor='#bfbfbf',oceancolor='#bfbfbf',showocean=True,coastlinewidth=0))
    print("  + plot       {0:>8}".format(str(datetime.timedelta(seconds=int(np.round((time.time()-t0))))))); t0 = time.time()
    return fig
    
if __name__ == '__main__':
    app.run_server(debug=True)

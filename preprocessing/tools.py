import os
import pickle
import pandas as pd
import plotly.express as px
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

"""
To Do:

* find the model missing data, fix the database again to include many
  versions
* decide which models to include (2 CNRMs, 2 MPIs)

Comments:

* A shrinking area does not necesarily mean threatened, it just
  reduces in the future. How might we incorporate this into the
  analysis?
* I believe that this process is defensible if somewhat
  convoluted. Are there other ways we could get at the same
  information in a more clear path? Do other methods of estimating
  vulnerability agree with these conclusions?
* Once a series of area growth rate maps are available, how do we
  choose which cluster series to use? How do we defend that choice?

"""

def StackToDataframe(stack_file):
    """Given the stack file, assemble a DataFrame with coordinate
    information.

    """
    path = os.path.dirname(stack_file)
    df_stack = pd.read_csv(stack_file,sep=" ",header=None,names=['model','time','row'])
    df_coord = {}
    for m in df_stack.model.unique():
        df_coord[m] = pd.read_csv(os.path.join(path,"coords.%s" % m),
                                 sep=" ",header=None,names=['lon','lat','area'])
    df_stack['lat']  = [df_coord[m].lat  for m in df_stack.model]
    df_stack['lon']  = [df_coord[m].lon  for m in df_stack.model]
    df_stack['area'] = [df_coord[m].area for m in df_stack.model]
    df_stack = df_stack.explode(['lat','lon','area']).drop(columns='row').reset_index()
    df_stack['lat' ] = df_stack['lat' ].astype(float)
    df_stack['lon' ] = df_stack['lon' ].astype(float)
    df_stack['area'] = df_stack['area'].astype(float)
    return df_stack

def ClusterToDataframe(cluster_root):
    """Given the cluster code output root, return a pandas DataFrame.

    """
    df = {}
    for root,subdirs,files in os.walk(cluster_root):
        for fname in files:
            if fname.startswith("clusters.out."):
                k = 'k%s' % (fname.split(".")[-1])
                df[k] = pd.read_csv(os.path.join(root,fname),sep=" ",header=None,names=[k]).astype({k:'int'})
                df[k] = df[k][k].to_numpy()
    df = pd.DataFrame(df)
    return df

def CentroidToDataframe(centroid_root):
    """Given the cluster code output root, return a dictionary of pandas
       DataFrames which contain the centroids.

    """
    names = None
    for root,subdirs,files in os.walk(os.path.join(centroid_root,"data")):
        for fname in files:
            if fname.startswith("names."):
                with open(os.path.join(root,fname),'rb') as f: names = pickle.load(f)
    if names is None: raise ValueError("Cannot find a names file")
    dfs = []
    for root,subdirs,files in os.walk(centroid_root):
        for fname in files:
            if (fname.startswith("seeds.out") and fname.endswith(".unstd")):
                k = 'k%s' % (fname.split(".")[-3])
                df = pd.read_csv(os.path.join(root,fname),sep=" ",header=None,names=['junk',]+names).drop(columns='junk')
                df['k'] = k
                dfs.append(df)
    return pd.concat(dfs).reset_index()

def ProcessAreas(df,t0=1990,tf=2090):
    """Given the concatenated DataFrame, approximate a area growth rate
       for each cluster set found.

    The area growth rate is the slope of the best fit line of 'area'
    vs. 'time' for all time \in [t0,tf]. Note that the regression is
    done on a per model/cluster set basis. This means that while the
    clusters are defined across all models, the growth rate computed
    here is relative to the area of a given cluster id for a
    particular model, defined globally.
    """
    for model in df.model.unique():
        t = df[(df.model==model)].time.unique()[0]
        A = df[(df.model==model) & (df.time==t)].area.sum()
        df.loc[df.model==model,'area'] *= (100/A)
    def polyfit(row):
        if len(row) < 2: return 0.
        m,b = np.polyfit(row['time'],row['area'],1)
        return m
    ks = sorted([c for c in df.columns if c.startswith('k')],key=lambda x: int(x.strip("k")))
    for k in ks:
        notk = [x for x in ks if x != k]
        a = df[(df.time >= t0)&(df.time <= tf)]
        a = a.drop(columns=notk+['lat','lon']).groupby(['model','time',k]).sum().reset_index()
        g = a.groupby(['model',k]).apply(polyfit)
        t = df.set_index(keys=['model',k])
        t.loc[g.index,'agr_%s' % k] = g
        df = t.reset_index()
    return df

def DataframeToDataset(df):
    """Return a dictionary of gridded xarray dataset objects where the
       keys are the models found.

    """
    ds = {}
    for model in df.model.unique():
        dfm = df[df.model==model].drop(columns=['model','area']).set_index(['time','lat','lon'])
        ds[model] = xr.Dataset.from_dataframe(dfm)
    return ds

def ComputeMeanAGR(ds,t):
    """For a given dictionary of datasets and a time, return the mean area
       growth rate and a boolean dataset for where models agree on the
       sign(agr).

    """
    models = []
    for model in ds:
        if model == "Reference": continue
        try:
            ds[model].sel({'time':t})
            models.append(model)
        except:
            pass
    lat = []; lon = []
    for model in models:
        lat += list(ds[model].lat)
        lon += list(ds[model].lon)
    lat = np.unique(lat)
    lon = np.unique(lon)
    dsc = []
    for model in models:
        dsc.append(ds[model].sel({'time':t}).interp({'lat':lat,'lon':lon},method='nearest'))
    dsc = xr.concat(dsc,"model")
    dsc = dsc.drop_vars([c for c in dsc.variables if c.startswith("k")])
    mean = dsc.mean(dim="model")
    agree = (dsc < 0).all(dim='model') + (dsc > 0).all(dim='model')
    return mean,agree
    
if __name__ == "__main__":
    import plotly

    # run this in iPython as 'run -i tools.py' and 'df' will stay
    # persistent in memory saving parsing time as you play around with
    # plots
    if "df" not in vars(): 
        print("Recreating dataframe...")
        df_aux = StackToDataframe("data/stack.obs_mod_mix")
        df_k   = ClusterToDataframe("./")
        df     = pd.concat([df_k,df_aux],axis=1).drop(columns="index")
        df     = ProcessAreas(df)
        df_c   = CentroidToDataframe("./")
        ds     = DataframeToDataset(df)
    
    # stuff for plots
    n = int(np.sqrt(len(df.model.unique())))+1
    df_usa = df[(df.lon>-126)&(df.lon<-66)&(df.lat>24)&(df.lat<50)]
    DF = df_usa # DF will be used in plots, point to the desired DataFrame

    if 1:
        # plots the mean area growth rate with stipling to indicate
        # that models agree on the sign
        mean,agree = ComputeMeanAGR(ds,1990)        
        for v in mean:
            fig,ax = plt.subplots(figsize=(10,5),dpi=300,subplot_kw={'projection':ccrs.Robinson()})
            vmax = np.abs(mean[v]).quantile(0.95)
            mean[v].plot(ax=ax,vmin=-vmax,vmax=vmax,transform=ccrs.PlateCarree(),cmap="RdBu")
            st = agree[v].to_dataframe().reset_index()
            st = st[st[v]]
            ax.plot(st.lon,st.lat,'.k',ms=0.03,transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.NaturalEarthFeature('physical','land','110m',
                                                        edgecolor='face',
                                                        facecolor='0.875'),zorder=-1)
            ax.add_feature(cfeature.NaturalEarthFeature('physical','ocean','110m',
                                                        edgecolor='face',
                                                        facecolor='0.750'),zorder=-1)
            ax.set_title("Mean Area Growth Rate Across Models (%s, Stipling = Models Agree on Sign)" % v)
            ax.set_extent([-180,180,-90,90],ccrs.PlateCarree())
            fig.savefig("%s.png" % v)
            plt.close()
            

    if 0:
        # plots the area growth rate for a given cluster set and year,
        # removing the reference. The reference growth rate is based
        # on too little data to be reliable.
        k = 'k64'
        y = 1990
        ms = 7 if id(DF) == id(df_usa) else 3
        tmp = DF[(DF.time==y)&(DF.model!="Reference")]
        vmax = tmp['agr_%s' % k].abs().quantile(q=0.95)
        fig = px.scatter(tmp,
                         x = 'lon',
                         y = 'lat',
                         color = 'agr_%s' % k,
                         facet_col = 'model',
                         facet_col_wrap = n,
                         color_continuous_scale=px.colors.diverging.RdBu,
                         range_color = [-vmax,vmax])
        fig.update_traces(marker=dict(size=ms,symbol='square'))
        fig.update_layout(font_size = 20,
                          plot_bgcolor='rgba(0,0,0,0)',
                          hoverlabel = dict(font_size=20))
        plotly.offline.plot(fig,filename="map_%s_%s.html" % (y,k))


    if 0:
        k = 'k16'
        dfc = df_c[df_c.k==k]
        drop = ['lat','lon'] + [c for c in df_k.columns if "k" in c and c != k]
        grp  = ['model','time',k]
        tmp  = DF[DF.time >= 1950]
        g = tmp.drop(columns=drop).groupby(grp).sum().reset_index()
        g = pd.concat([g,dfc.iloc[g[k]-1].reset_index().drop(columns='index')],axis=1)
        fig = px.line(g,
                      x = 'time',
                      y = 'area',
                      color = k,
                      line_group = k,
                      facet_col = 'model',
                      facet_col_wrap = n,
                      hover_data = list(g.columns),
                      range_y = [-0.05*g.area.max(),1.05*g.area.max()])
        fig.update_layout(font_size = 28,
                          hoverlabel = dict(font_size=20))
        plotly.offline.plot(fig,filename="tas_pr_mrsos_%s.html" % k)

    # 4 panel plot of a given model's cluster IDs
    if 0: 
        k = 'k64'        
        tmp = DF[DF.model=='CESM2']
        tmp = tmp[(tmp.time==1970)|(tmp.time==2010)|(tmp.time==2050)|(tmp.time==2090)]
        fig = px.scatter(tmp,
                         x = 'lon',
                         y = 'lat',
                         color = k,
                         facet_col = 'time',
                         facet_col_wrap = 2)
        fig.update_layout(font_size = 20,
                          plot_bgcolor='rgba(0,0,0,0)',
                          hoverlabel = dict(font_size=20))
        plotly.offline.plot(fig,filename="map_%s.html" % (k))

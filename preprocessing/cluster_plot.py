import os
import pickle
import pandas as pd
import plotly.express as px
import numpy as np
import plotly
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.rcParams.update({'font.size': 14})

def GetCmap(k):
    assert k <= 70
    if k <= 10: return plt.get_cmap("tab10",k)
    if k <= 20: return plt.get_cmap("tab20",k)
    K = k
    clrs = []
    c20 = ["tab20","tab20b","tab20c"]
    while K > 0:
        if K > 10:
            clrs.append(plt.get_cmap(c20.pop(0)).colors)
            K -= 20
        else:
            clrs.append(plt.get_cmap("tab10").colors)
            K -= 10
    return colors.ListedColormap(np.vstack(clrs)[:k],'custom')

class ClusterViz():

    def __init__(self,root):
        self.root = root
        self.case = root.strip("/").split("/")[-1]
        self.seeds = {}
        self.clusters = {}
        self.names = ""
        self.models = ""
        self.model_names = []
        self.years = ""
        self.areas = {}
        self.df_centroids = None
        self.df_cluster = None
        self.period = []
        self.selected_k = None
        
    def SetStudyPeriod(self,y0,yf):
        self.period = [y0,yf]
        
    def FindDatafiles(self):
        for root,subdirs,files in os.walk(self.root):
            for fname in files:
                if (fname.startswith("seeds.out") and fname.endswith(".unstd") and self.case in fname):
                    k = int(fname.split(".")[-3])
                    self.seeds[k] = os.path.join(root,fname)
                if (fname.startswith("clusters.out.") and self.case in fname):
                    k = int(fname.split(".")[-1])
                    self.clusters[k] = os.path.join(root,fname)
                if (fname.startswith("names.") and self.case in fname):
                    self.names = os.path.join(root,fname)
                if (fname.startswith("models.") and self.case in fname):
                    self.models = os.path.join(root,fname)
                if (fname.startswith("coords.") and self.case in fname):
                    self.coords = os.path.join(root,fname)
                if (fname.startswith("areas.") and self.case in fname):
                    self.areas = os.path.join(root,fname)
                if (fname.startswith("years.") and self.case in fname):
                    self.years = os.path.join(root,fname)
        
    def LoadDatasets(self,k):
        """
        """
        assert k in self.seeds
        self.selected_k = k
        with open( self.names,'rb') as f: names  = pickle.load(f) # column names
        with open(self.models,'rb') as f: models = pickle.load(f) # model names
        self.model_names = models
        t = np.fromfile(self.years,sep=' ')                       # list of the times
        a = np.fromfile(self.areas,sep=' ').reshape((-1,2))       # areas of all cells from all models x id of each model
        mid = np.asarray(models)[a[:,1].astype(int)]              # model ids, indexes 'models'
        a = a[:,0]                                                # areas of all cells from all models
        df_centroids = pd.read_csv(self.seeds[k],sep=" ",header=None,names=['junk',]+names).drop(columns='junk')
        dfc = pd.read_csv(self.coords,sep=" " ,header=None,names=['lon','lat'])
        tmp = pd.read_csv(self.clusters[k],sep=" ",header=None,names=['id']).astype({'id':'int'})
        df_cluster = pd.DataFrame(dict(id     = tmp.id,
                                       area   = np.tile(      a,t.size),
                                       model  = np.tile(    mid,t.size),
                                       lat    = np.tile(dfc.lat,t.size),
                                       lon    = np.tile(dfc.lon,t.size),
                                       decade = np.repeat(t.astype(int),a.size)))
        gr = df_cluster.drop(columns=['lat','lon']).groupby(['id','model','decade']).sum().reset_index()
        gr.area *= 100/a.sum() # normalize areas to be % of the globe across all models
        for d in gr.decade.unique():
            # initialize to all 0 because some decades may not contain areas for each cluster
            df_centroids['A(%d)' %d] = 0
            for i in range(k):
                df_centroids.loc[i,'A(%d)' % d] = gr.loc[(gr.id==(i+1)) & (gr.decade==d)].area.sum()
                
        # compute the area growth rate aross the study period
        if len(self.period) == 0: self.period = [t.min(),t.max()]
        cols = [c for c in df_centroids.columns if "A(" in c]
        cols = [c for c in cols if (int(c.replace("A(","").replace(")","")) >= self.period[0],
                                    int(c.replace("A(","").replace(")","")) <= self.period[1])]
        A = df_centroids[cols]
        def polyfit(row):
            m,b = np.polyfit(range(len(row)),row,1)
            return m
        df_centroids['AGR [% yr-1]'] = A.apply(polyfit,axis=1)
                
        self.df_centroids = df_centroids
        self.df_cluster = df_cluster
        
    def PlotCentroids(self,xy=(0,1)):
        rem_k = self.selected_k
        ks = sorted(list(self.seeds.keys()))
        DF = []
        for k in ks:
            self.LoadDatasets(k)
            df = self.df_centroids
            df['k'] = k
            df['ks'] = df['k'].astype(str)
            DF.append(df)
        df = pd.concat(DF)
        col = list(df.columns)
        fig = px.scatter(df,
                         x = col[xy[0]],
                         y = col[xy[1]],
                         color = "ks",
                         hover_data = col)
        fig.update_traces(marker = dict(size=12))
        fig.update_layout(font_size = 28,
                          hoverlabel = dict(font_size=24))
        plotly.offline.plot(fig,filename="%s_centroid.html" % (self.root))
        if rem_k: self.LoadDatasets(rem_k)
            
    def TableCentroids(self,mean=True,count=True,std=False,export="latex"):
        """
        Use the booleans to control what is included in the table output.
        """
        df = self.df_centroids
        dfs = df.sort_values('AGR [% yr-1]')
        dfs['cluster'] = dfs.index+1
        cols  = ['cluster',]
        if mean: cols += [c for c in dfs.columns if "mean" in c]
        if count: cols += [c for c in dfs.columns if "count" in c]
        if std: cols += [c for c in dfs.columns if "std" in c]
        if len(self.period)>0:
            cols += ['A(%d)' % self.period[0],'A(%d)' % self.period[1],'AGR [% yr-1]']
        dfs = dfs[cols]
        def f1(x): return '%d' % (x)
        def f2(x): return '%.2g' % (x)
        fmt = [f1,] + [f2]*(len(cols)-1)

        funcs = [f for f in dfs.__dir__() if "to_" in f]
        if "to_%s" % export not in funcs:
            msg = "No %s in [%s]" % ("to_%s" % export,",".join(funcs))
            raise ValueError(msg)
        ex = getattr(dfs,"to_%s" % export)
        print(ex(index=False,formatters=fmt))

        
    def Plot(self):
        """
        df = self.df_cluster.groupby(['id','model','decade']).sum().reset_index()
        df = df[df.decade>=2000]
        fig = px.bar(df,
                     x = "id",
                     y = "area",
                     color = "model",
                     barmode = "group",
                     animation_frame = "decade",
                     animation_group = "id",
                     labels={'id':'Cluster ID','area':'Cluster Area [m2]'})
        fig.update_layout(font_size = 28)
        plotly.offline.plot(fig,filename="bar.html")
        fig.show()
        """
        pass

    
    def PlotMap(self,decade,model):
        
        # select the data to plot
        df = self.df_cluster[(self.df_cluster.decade==decade) & (self.df_cluster.model==model)]
        k = self.selected_k
        
        # convert 1D arrays to a 2D grid
        lat = df.lat.unique()
        lat.sort()
        lon = df.lon.unique()
        lon.sort()
        i = lat.searchsorted(df.lat)
        j = lon.searchsorted(df.lon)
        A = np.ma.masked_array(np.zeros((lat.size,lon.size)),mask=True)
        A[i,j] = df.id.to_numpy()

        # extents and projection
        difx = lon.max()-lon.min()
        lonmin = max(lon.min() - 0.05*difx,-180)
        lonmax = min(lon.max() + 0.05*difx,+180)
        dify = lat.max()-lat.min()
        latmin = max(lat.min() - 0.05*dify,-90)
        latmax = min(lat.max() + 0.05*dify,+90)
        proj = ccrs.AlbersEqualArea(central_longitude=0.5*(lonmax+lonmin),
                                    central_latitude =0.5*(latmax+latmin)) if (difx < 0.5*360 and dify < 0.5*180) else ccrs.Robinson()
        
        # plot with pcolormesh
        aspect_ratio = 14/29 
        w = 10; h = w*aspect_ratio
        cm = GetCmap(k)
        fig,ax = plt.subplots(figsize=(w,h),dpi=200,subplot_kw={'projection':proj})
        p = ax.pcolormesh(lon,lat,A,cmap=cm,vmin=0.5,vmax=k+0.5,transform=ccrs.PlateCarree())
        ax.set_title("%s %s %d's" % (self.root,model,decade))
        ax.set_extent([lonmin,lonmax,latmin,latmax],ccrs.PlateCarree())
        ax.add_feature(cfeature.NaturalEarthFeature('physical','land','110m',
                                                    edgecolor='face',
                                                    facecolor='0.875'),zorder=-1)
        ax.add_feature(cfeature.NaturalEarthFeature('physical','ocean','110m',
                                                    edgecolor='face',
                                                    facecolor='0.750'),zorder=-1)
        fig.colorbar(p,orientation='horizontal',pad=0.01,ticks=range(1,k+1))
        fig.savefig("%s_%s_%d_%d.png" % (self.root,model,decade,k))
        
        plt.close()

if __name__ == "__main__":

    """
    Sample output snippet.
    """
    cv = ClusterViz("standard_e3sm")
    cv.FindDatafiles()
    cv.LoadDatasets(16)
    cv.PlotMap(1990)
    cv.PlotMap(2090)
    cv.PlotCentroids()
    cv.TableCentroids(export="string")
    

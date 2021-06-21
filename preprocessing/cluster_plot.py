import os
import pickle
import pandas as pd
import plotly.express as px
import numpy as np
import plotly
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.rcParams.update({'font.size': 14})

# stack is models x years x cells
# the areas part of this is specific to the CMIP6 run

class ClusterViz():

    def __init__(self,root):
        self.root = root
        self.case = root.strip("/").split("/")[-1]
        self.seeds = {}
        self.clusters = {}
        self.names = ""
        self.years = ""
        self.areas = {}
        self.df_centroids = None
        self.df_cluster = None
        
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
                if (fname.startswith("coords.") and self.case in fname):
                    self.coords = os.path.join(root,fname)
                if (fname.startswith("years.") and self.case in fname):
                    self.years = os.path.join(root,fname)
                if (fname.startswith("areas.")): ### specific to CMIP6 run
                    k = fname.replace("areas.","").replace("_base","")
                    self.areas[k] = os.path.join(root,fname)
        
    def LoadDatasets(self,k):
        with open(self.names,'rb') as f: names = pickle.load(f)
        t = np.fromfile(self.years,sep=' ')
        df_centroids = pd.read_csv(self.seeds[k],sep=" ",header=None,names=['junk',]+names)
        df_centroids = df_centroids.drop(columns="junk")
        dfc = pd.read_csv(self.coords,sep=" " ,header=None,names=['lon','lat'])
        tmp = pd.read_csv(self.clusters[k],sep=" ",header=None,names=['id']).astype({'id':'int'})
        A = T = M = X = Y = None
        earths_area = 0
        for i,key in enumerate(sorted(self.areas,key=lambda key: key.lower())):
            a = np.fromfile(self.areas[key],sep=' ')
            earths_area += a.sum()
            A = np.hstack([A,np.tile(a,t.size)])
            X = np.hstack([X,np.tile(dfc.lon,t.size)])
            Y = np.hstack([Y,np.tile(dfc.lat,t.size)])
            T = np.hstack([T,np.repeat(t.astype(int),a.size)])
            M = np.hstack([M,[key]*(a.size*t.size)])
        earths_area /= len(self.areas)
        df_cluster = pd.DataFrame(dict(id=tmp.id,area=A[1:],decade=T[1:],model=M[1:],lat=Y[1:],lon=X[1:]))
        gr = df_cluster.drop(columns='model').groupby(['id','decade']).sum().reset_index()
        for d in gr.decade.unique():
            df_centroids['A(%d)' % d] = 0
            for i in range(k):
                a = gr.loc[(gr.id==(i+1)) & (gr.decade==d)].area/earths_area*100
                if(len(a)): df_centroids.loc[i,'A(%d)' % d] = a.to_numpy()
        self.df_centroids = df_centroids
        self.df_cluster = df_cluster
        
    def PlotCentroids(self):
        df = self.df_centroids        
        A = df[[c for c in df.columns if "A(" in c and int(c.replace("A(","").replace(")","")) >= 1990]] #
        print(A)
        def polyfit(row):
            m,b = np.polyfit(range(len(row)),row,1)
            return m
        df['area growth rate [% dec-1]'] = A.apply(polyfit,axis=1)
        return
        fig = px.scatter(df,
                         x = "mean(tas) [degC]",
                         y = "mean(pr) [mm d-1]",
                         color = "area growth rate [% dec-1]",
                         size = "mean(mrsos) [kg m-2]",
                         hover_data = ['mean(tas) [degC]', 'std(tas) [degC]',
                                       'mean(pr) [mm d-1]','std(pr) [mm d-1]',
                                       'mean(mrsos) [kg m-2]', 'std(mrsos) [kg m-2]',
                                       'A(1990)', 'A(2090)', 'area growth rate [% dec-1]'],
                         color_continuous_scale = "RdBu")
        fig.update_layout(font_size = 28,
                          coloraxis = dict(cmid=0),
                          hoverlabel=dict(font_size=24))
        plotly.offline.plot(fig,filename="%s_centroid.html" % self.root)
        
    def TableCentroids(self):
        df = self.df_centroids
        dfs = df.sort_values('area growth rate [% dec-1]')
        dfs['cluster'] = dfs.index+1
        cols  = ['cluster',]
        cols += [c for c in dfs.columns if "mean" in c]
        #cols += [c for c in dfs.columns if "std" in c]
        cols += ['A(1990)','A(2090)','area growth rate [% dec-1]']
        dfs = dfs[cols]
        def f1(x): return '%d' % (x)
        def f2(x): return '%.2g' % (x)
        fmt = [f1,] + [f2]*(len(cols)-1)
        print(dfs.to_latex(index=False,formatters=fmt))
        
    def Plot(self):
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

    def PlotMap(self,decade):
        k = 16
        df = self.df_cluster[self.df_cluster.decade==decade]        
        aspect_ratio = 14/29 
        w = 10; h = w*aspect_ratio
        cm = plt.get_cmap("tab20",k)
        fig,ax = plt.subplots(figsize=(w,h),dpi=200,subplot_kw={'projection':ccrs.Robinson()})
        p = ax.scatter(df.lon,df.lat,c=df.id,s=0.4,cmap=cm,vmin=0.5,vmax=k+0.5,transform=ccrs.PlateCarree())
        ax.set_title("%s %d's" % (self.root,decade))
        ax.set_extent([-180,180,-90,90],ccrs.PlateCarree())
        fig.colorbar(p,orientation='horizontal',pad=0.01,ticks=range(1,k+1))
        fig.savefig("map_%s_%d.png" % (self.root,decade))
        
        plt.close()

        
if 0:
    cv = ClusterViz("standard_e3sm")
    cv.FindDatafiles()
    cv.LoadDatasets(16)
    cv.PlotMap(1990)
    cv.PlotMap(2090)
    cv.PlotCentroids()
    cv.TableCentroids()
if 0:
    cv = ClusterViz("change_e3sm")
    cv.FindDatafiles()
    cv.LoadDatasets(16)
    cv.PlotMap(1990)
    cv.PlotMap(2090)
    cv.PlotCentroids()
    cv.TableCentroids()
if 0:
    cv = ClusterViz("drought_e3sm")
    cv.FindDatafiles()
    cv.LoadDatasets(16)
    cv.PlotMap(1950)
    cv.PlotMap(2000)
    cv.PlotCentroids()
    cv.TableCentroids()
if 1:
    cv = ClusterViz("extremes_e3sm")
    cv.FindDatafiles()
    cv.LoadDatasets(16)
    cv.PlotMap(1990)
    cv.PlotMap(2090)
    cv.PlotCentroids()
    cv.TableCentroids()

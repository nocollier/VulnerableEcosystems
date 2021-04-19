import os
import pickle
import pandas as pd
import plotly.express as px
import numpy as np
import plotly

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
        self.df_clusters = None
        
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
        tmp = pd.read_csv(self.clusters[k],sep=" ",header=None,names=['id']).astype({'id':'int'})
        A = T = M = None
        earths_area = 0
        for i,key in enumerate(sorted(self.areas,key=lambda key: key.lower())): ### specific to CMIP6 run
            a = np.fromfile(self.areas[key],sep=' ')
            earths_area += a.sum()
            A = np.hstack([A,np.tile(a,t.size)])
            T = np.hstack([T,np.repeat(t.astype(int),a.size)])
            M = np.hstack([M,[key]*(a.size*t.size)])
        earths_area /= len(self.areas)
        df_cluster = pd.DataFrame(dict(id=tmp.id,area=A[1:],decade=T[1:],model=M[1:]))
        tmp = df_cluster.drop(columns='model').groupby(['id','decade']).sum().reset_index()
        for d in tmp.decade.unique():
            df_centroids['A(%d)' % d] = tmp[tmp.decade==d].area.to_numpy()/earths_area*100
        self.df_centroids = df_centroids
        self.df_cluster = df_cluster
        
    def PlotCentroids(self):
        df = self.df_centroids
        fig = px.scatter(df,
                         x = "mean(Surface Air Temperature) [degC]",
                         y = "mean(Precipitation) [mm d-1]",
                         size = "A(2000)",
                         color = "mean(Gross Primary Productivity) [g m-2 d-1]",
                         color_continuous_scale = "viridis")
        fig.show()

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
        
cv = ClusterViz("./CMIP6_base_dec/")
cv.FindDatafiles()
cv.LoadDatasets(16)
cv.Plot()

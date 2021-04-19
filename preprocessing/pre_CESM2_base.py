"""A script for preprocessing files for clustering.

"""
import socket
import sys
import os
import pickle
import pandas as pd
from ILAMB.ModelResult import ModelResult
from ILAMB.Variable import Variable
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

if not (socket.gethostname().startswith("cori") or socket.gethostname().startswith("nid")):
    print("This script is intended to run on Cori")
    sys.exit(1)

# Inputs
MODEL = "CESM2"
CASE  = "%s_base_dec" % MODEL

# Setup models with the required variables
ROOT = "/global/cfs/projectdirs/m3522/cmip6/CMIP6"
df = pd.read_csv(os.path.join(ROOT,'cmip6_stores.csv'))
Vs = ["tas","pr","mrsos","gpp"]
V  = " | ".join(["variable_id == '%s'" % v for v in Vs + ["areacella","sftlf"]])
Q  = " & ".join(["source_id == '%s'" % MODEL,
                 "member_id == 'r1i1p1f1'",
                 "(experiment_id == 'historical' | experiment_id == 'ssp585')",
                 "(table_id == 'Amon' | table_id == 'Lmon' | table_id == 'fx')",
                 "(%s)" % V])
q = df.query(Q)

sname = {'tas'  : 'Surface Air Temperature',
         'pr'   : 'Precipitation',
         'mrsos': 'Soil Moisture',
         'gpp'  : 'Gross Primary Productivity'}
units = {'tas'  : 'degC',
         'pr'   : 'mm d-1',
         'mrsos': 'kg m-2', 
         'gpp'  : 'g m-2 d-1'}

M = {}
for e in q.experiment_id.unique():
    PATHS = [os.path.join(ROOT,p) for p in q[q.experiment_id==e].path.to_list()]
    M[e] = ModelResult("./",modelname=e,paths=PATHS)

# Compute the amount of land area
v = Variable(filename = M['historical'].variables['sftlf'][0],variable_name = "sftlf")
v.convert("1")
area = v.data.data
lat = v.lat
lon = v.lon
v = Variable(filename = M['historical'].variables['areacella'][0],variable_name = "areacella")
area *= v.data.data    
lat,lon = np.meshgrid(lat,lon,indexing='ij')

# Compute the bioclimatic variables on an annual scale
out = {}
shp = None
names = []
for vname in Vs:
    data = []
    for e in ['historical','ssp585']:
        v = M[e].extractTimeSeries(vname).convert(units[vname])
        data.append(v.data)
    data = np.ma.vstack(data)
    data = data[:(int(data.shape[0]/120)*120)]
    data = data.reshape((-1,120,)+v.data.shape[-2:])
    mean = data.mean(axis=1)
    std  = data.std (axis=1)    
    out["mean(%s)" % vname] = mean
    out["std(%s)"  % vname] = std
    names.append("mean(%s) [%s]" % (sname[vname],units[vname]))
    names.append("std(%s) [%s]" % (sname[vname],units[vname]))
    shp = mean.shape
years = np.asarray(range(1850,1850+10*shp[0],10))

# Apply a uniform mask across all variables
mask = np.zeros(mean.shape,dtype=bool) + (area < 1e-6) + (lat < -60)
for key in out: mask += out[key].mask
for key in out: out[key] = np.ma.masked_array(out[key],mask=mask)
lat  = np.ma.masked_array(lat,mask=mask[0])
lon  = np.ma.masked_array(lon,mask=mask[0])
area = np.ma.masked_array(area,mask=mask[0])

# Output files
output = np.vstack([v.compressed() for v in list(out.values())])
np.savetxt('%s.ascii' % CASE,output.T,delimiter=' ')
output = np.vstack([lon.compressed(),lat.compressed()])
np.savetxt('coords.%s' % CASE,output.T,delimiter=' ')
output = area.compressed()
np.savetxt('areas.%s' % CASE,output.T,delimiter=' ')
output = years
np.savetxt('years.%s' % CASE,output.T,delimiter=' ')
with open('names.%s' % CASE,'wb') as f:
    pickle.dump(names,f)

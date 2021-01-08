"""A script for preprocessing files for clustering.

We will put together annual values for the following bioclimatic
variables, where a '*' indicates a variable to be included in the
clustering.

* BIO1 = Annual mean temperature
  BIO2 = Mean Diurnal Range (Mean of monthly (max temp - min temp))
* BI03 = Isothermality (BIO2/BIO7)(x100)
* BIO4 = Temperature Seasonality (standard deviation x100)
  BIO5 = Max Temperature of Warmest Month
  BIO6 = Min Temperature of Coldest Month
  BIO7 = Temperature Annual Range (BIO5-BIO6)

Variables taken from: https://www.worldclim.org/data/bioclim.html
"""
import socket
import sys
import os
import pandas as pd
from ILAMB.ModelResult import ModelResult
import numpy as np
from netCDF4 import Dataset

if not (socket.gethostname().startswith("cori") or socket.gethostname().startswith("nid")):
    print("This script is intended to run on Cori")
    sys.exit(1)

# Inputs
MODEL = "CanESM5"
CASE  = "%s_bioclimatic_yr" % MODEL

# Setup models with the required variables
ROOT = "/global/cfs/projectdirs/m3522/cmip6/CMIP6"
df = pd.read_csv(os.path.join(ROOT,'cmip6_stores.csv'))
V = " | ".join(["variable_id == 'tas'",
                "variable_id == 'tasmax'",
                "variable_id == 'tasmin'",
                "variable_id == 'areacella'",
                "variable_id == 'sftlf'"])
Q = " & ".join(["source_id == '%s'" % MODEL,
                "member_id == 'r1i1p1f1'",
                "(experiment_id == 'historical' | experiment_id == 'ssp585')",
                "(table_id == 'Amon' | table_id == 'fx')",
                "(%s)" % V])
q = df.query(Q)
M = {}
for e in q.experiment_id.unique():
    PATHS = [os.path.join(ROOT,p) for p in q[q.experiment_id==e].path.to_list()]
    M[e] = ModelResult("./",modelname=e,paths=PATHS)

# Compute the amount of land area
with Dataset(M['historical'].variables['sftlf'][0]) as dset:
    area = dset.variables["sftlf"][...]
    lat  = dset.variables["lat"][...]
    lon  = dset.variables["lon"][...]
    lon  = (lon<=180)*lon + (lon>180)*(lon-360) + (lon<-180)*360
    if area.max() > 50:
        area *= 0.01
with Dataset(M['historical'].variables['areacella'][0]) as dset:
    area *= dset.variables["areacella"][...]
lat,lon = np.meshgrid(lat,lon,indexing='ij')

# Compute the bioclimatic variables on an annual scale
BIO1 = []
BIO3 = []
BIO4 = []
for e in ['historical','ssp585']:
    v = M[e].extractTimeSeries("tas").convert("degC")
    data = v.data.reshape((-1,12,)+v.data.shape[-2:])
    mean = data.mean(axis=1)
    std  = data.std (axis=1)
    amax = data.max (axis=1)
    amin = data.min (axis=1)
    del data
    v = M[e].extractTimeSeries("dtr",expression="tasmax-tasmin")
    dtr  = v.data.reshape((-1,12,)+v.data.shape[-2:]).mean(axis=1)
    BIO1.append(mean)
    BIO3.append(dtr/(amax-amin)*100)
    BIO4.append(std) # going to skip x100 here
BIO1 = np.ma.vstack(BIO1) # [degC]
BIO3 = np.ma.vstack(BIO3) # [%]
BIO4 = np.ma.vstack(BIO4) # [degC]

# Apply a uniform mask across all variables
mask = (area < 1e-6) + BIO1.mask + BIO3.mask + BIO4.mask
BIO1 = np.ma.masked_array(BIO1,mask=mask)
BIO3 = np.ma.masked_array(BIO3,mask=mask)
BIO4 = np.ma.masked_array(BIO4,mask=mask)
lat  = np.ma.masked_array(lat,mask=mask[0])
lon  = np.ma.masked_array(lon,mask=mask[0])
area = np.ma.masked_array(area,mask=mask[0])

# Output files
output = np.vstack([BIO1.compressed(),BIO3.compressed(),BIO4.compressed()])
np.savetxt('%s.ascii' % CASE,output.T,delimiter=' ')
output = np.vstack([lat.compressed(),lon.compressed()])
np.savetxt('coords.%s' % CASE,output.T,delimiter=' ')
output = area.compressed()
np.savetxt('areas.%s' % CASE,output.T,delimiter=' ')
output = np.asarray(range(1850,1850+BIO1.shape[0]))
np.savetxt('years.%s' % CASE,output.T,delimiter=' ')

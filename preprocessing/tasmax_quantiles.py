import os
import glob
import xarray as xr

root = "/gpfs/alpine/cli143/proj-shared/Deeksha/Data/CMIP6/historical/daily/tasmax/raw/"
files = sorted([f for f in glob.glob(os.path.join(root,"*.nc")) if "_day_" in f])
models = sorted(list(set([os.path.basename(f).split("_")[2] for f in files])),key = lambda f: f.lower())
done = glob.glob("*.nc")

for m in models:

    print(m)
    
    # Is this model already processed?
    if len([f for f in done if m in f]) > 0: continue
    
    # Which files do we need?
    mfiles = [f for f in files if m in f]

    # Load each file and compute its annual quantiles, assuming whole
    # years are contained in each file. We will encase this in a 'try'
    # and 'except' because the files may be too big to be read in and
    # fail.
    Q = []
    try:
        for f in mfiles:
            ds = xr.load_dataset(f)
            q = ds['tasmax'].groupby('time.year').quantile([0.10,0.30,0.50,0.70,0.90],dim='time')
            Q.append(q)
    except:
        print("Processing %s encountered an error, skipping" % m)
        continue
    Q = sorted(Q,key=lambda q: q.year.min()) # probably not needed since we sort filenames
    Q = xr.concat(Q,dim='year')

    # Before we change the time, create a new filename
    f = os.path.basename(f).split("_")
    f[ 1] = "Ayr"
    f[-1] = "%d-%d.nc" % (Q.year.min(),Q.year.max())
    f = "_".join(f)

    # Change time around so ILAMB can understand it
    Q['year'] = (Q['year'].astype(float)+0.5-1850)*365
    Q['year'].attrs['units'] = 'days since 1850-01-01'
    Q['year'].attrs['calendar'] = 'noleap'
    
    # Create a new dataset where the quantiles are each a separate variable
    Q.attrs['units'] = ds['tasmax'].attrs['units']
    out = xr.Dataset({"tasmaxQ%02d" % (q*100):Q.sel({'quantile':q}).drop('quantile') for q in Q['quantile']}).rename({"year":"time"})
    out.to_netcdf(f)
    

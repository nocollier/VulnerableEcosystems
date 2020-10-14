# Resources for the Vulnerable Ecosystems LDRD

The following repository is meant to store resources and scripts used in our LDRD.

## Obtaining model data

Model data may be obtained the traditional way via [ESGF](https://esgf-node.llnl.gov/search/cmip6/). However, if you have a NERSC account, Forrest has long been making a copy for RUBISCO/RGMA which is directly accessible for analysis in the `m3522` project. We will look into getting time on NERSC for this project, but even if copying data, we have found it faster to write queries on Cori using a database that we have built there using the python package `pandas`. Then the query can used to setup a `rsync` to CADES or wherever the analysis will need to take place.

First, on Cori with `pandas` installed, you need to load the database into memory:

```
import pandas as pd
df = pd.read_csv("/global/cfs/projectdirs/m3522/cmip6/CMIP6/cmip6_stores.csv")
```

Pandas can read this relatively quickly, and then it can be queried using SQL-like syntax. For example, to find all `SSP585` entries of monthly `tas`, you write: 

```
q = df.query('experiment_id == "ssp585" & table_id == "Amon" & variable_id == "tas"')
```

Then, for example, you could see which models have output for this selection by:

```
print(q.source_id.unique())
['CNRM-CM6-1' 'CNRM-ESM2-1' 'BCC-CSM2-MR' 'CESM2' 'FGOALS-g3'
 'UKESM1-0-LL' 'AWI-CM-1-1-MR' 'CanESM5' 'MPI-ESM1-2-HR' 'CAMS-CSM1-0'
 'MCM-UA-1-0' 'KACE-1-0-G' 'EC-Earth3' 'EC-Earth3-Veg' 'MRI-ESM2-0'
 'NESM3' 'MIROC-ES2L' 'MIROC6' 'IPSL-CM6A-LR' 'NorESM2-LM' 'FIO-ESM-2-0'
 'MPI-ESM1-2-LR']
```

The columns of the database are setup to use the CMIP vocabulary. You can query on any of these:

```
print(df.columns)                                                                               
Index(['activity_id', 'institution_id', 'source_id', 'experiment_id',
       'member_id', 'table_id', 'variable_id', 'grid_label', 'version',
       'path'],
```

Once you have reduced the database to the selection you need, then the `path` column will be the locations relative to `/global/cfs/projectdirs/m3522/cmip6/CMIP6` where you can find the netCDF4 files.

## Getting your feet wet with clustering

Particularly for those who have never worked with clustering, it may be helpful to take a look at the following [overview](https://scikit-learn.org/stable/modules/clustering.html#clustering) from `scikit-learn`. The cluster code that Forrest, et al. have developed is an implementation of the k-means algorithm. It may be helpful for you to play around with a version of the algorithm that already has a python/numpy interface. More details can be found on their well-documented [site](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn-cluster-kmeans).

## Running the production cluster code

Need help writing this.

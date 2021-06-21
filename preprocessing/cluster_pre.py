import os
import pickle
import numpy as np
from ILAMB.ModelResult import ModelResult

pref_units = {}
pref_units[ 'pr'] = 'mm d-1'
pref_units['gpp'] = 'g m-2 d-1'
pref_units['tas'] = 'degC'

def PrepareClusterInputs(casename,models,variables,times,variability=True,reference_period=None,regions=["global"],quiet=False):
    """Uses ILAMB objects to prepare clustering for use with Forrest/Jitu's code.

    The user provides the models, variables, and times through which
    he/she wishes to stack the columns. By default, the code will
    provide mean/std for each model/variable across each time span
    provided. If change is desired in place of the standard approach,
    then a reference period can be specified. ILAMB regions may also
    be used to exclude areas from the clustering.

    Parameters
    ----------
    casename : str
        the identifier to be used for this case
    models : ILAMB.ModelResult or list of ILAMB.ModelResult
        the models to include in the clustering
    variables : str or list of str
        the variables to include in the clustering
    times : array-like
        the boundaries of the intervals to be considered in years
    variability : bool, optional
        enable to include variability in the cluster columns
    quiet : bool
        enable to turnoff the debugging output

    """
    # check types
    if type(models) == ModelResult: models = [models]
    for m in models: assert(type(m) == ModelResult)
    if type(regions)==str: regions = [regions]
    years = np.asarray(times)[:-1]
    times = (np.asarray(times,dtype=float)-1850)*365
    change = False
    if reference_period is not None:
        change = True
        reference_period = (np.asarray(reference_period,dtype=float)-1850)*365
        assert reference_period.size==2

    # pop models that do not have the required variables
    V = variables
    M = [m for m in models if set(V).issubset(m.variables.keys())]
    if len(M) != len(models) and not quiet:
        pop = list(set(models).difference(set(M)))
        print("Some models do not have required variables:")
        for m in pop:
            print("  - %s: missing [%s]" % (m.name,",".join(list(set(V).difference(m.variables.keys())))))

    # if based on change, we need to compute the reference period
    if change:
        rstack = []
        for m in M:
            columns = {}
            for vname in V:
                v = m.extractTimeSeries(vname,initial_time=reference_period[0],final_time=reference_period[1])
                v.trim(t=reference_period)
                if vname in pref_units: v.convert(pref_units[vname])
                lbl = 'mean(%s) [%s]' % (vname,v.unit)
                columns[lbl] = v.integrateInTime(mean=True)
                if variability:
                    lbl = 'std(%s) [%s]' % (vname,v.unit)
                    columns[lbl] = v.variability()
        rstack.append(columns)

    # build the stack based on the aspect of the variables
    tstack = []
    for t in range(times.size-1):
        mstack = []
        for i,m in enumerate(M):
            columns = {}
            for vname in V:
                v = m.extractTimeSeries(vname,initial_time=times[t],final_time=times[t+1])
                v.trim(t=times[t:(t+2)])
                if vname in pref_units: v.convert(pref_units[vname])
                lbl = 'mean(%s) [%s]' % (vname,v.unit)
                columns[lbl] = v.integrateInTime(mean=True)
                if change: columns[lbl].data -= rstack[i][lbl].data
                if variability:
                    lbl = 'std(%s) [%s]' % (vname,v.unit)
                    columns[lbl] = v.variability()
                    if change: columns[lbl].data -= rstack[i][lbl].data
            mstack.append(columns)
        tstack.append(mstack)

    # data within each model must be uniformly masked, also build up
    # coords/areas
    LAT = []; LON = []; AREA = []
    for i,m in enumerate(M):
        v = tstack[0][i][list(columns.keys())[0]]
        mask = ilamb_region.getMask(regions[0],v)
        if len(regions) > 1:
            for region in regions[1:]:
                mask += ilamb_region.getMask(region,v)
        lat,lon = np.meshgrid(v.lat,v.lon,indexing='ij')
        area = v.area[...]
        for vname in tstack[0][i]: mask += tstack[0][i][vname].data.mask
        for t in range(len(tstack)):
            for vname in tstack[t][i]: tstack[t][i][vname].data.mask = mask
        LAT .append(np.ma.masked_array( lat,mask=mask).compressed())
        LON .append(np.ma.masked_array( lon,mask=mask).compressed())
        AREA.append(np.ma.masked_array(area,mask=mask).compressed())
    lat  = np.hstack(LAT)
    lon  = np.hstack(LON)
    area = np.hstack(AREA)

    # write output
    pathname = os.path.join(casename,"data")
    if not os.path.isdir(pathname): os.makedirs(pathname)
    output = []
    for mstack in tstack:
        stack = []
        for columns in mstack:
            stack.append(np.vstack([v.data.compressed() for v in list(columns.values())]))
        output.append(np.hstack(stack))
    np.savetxt(os.path.join(pathname,'%s.ascii' % casename),
               np.hstack(output).T,delimiter=' ')
    np.savetxt(os.path.join(pathname,'years.%s' % casename),
               years.T,delimiter=' ')
    np.savetxt(os.path.join(pathname,'coords.%s' % casename),
               np.vstack([lon,lat]).T,delimiter=' ')
    np.savetxt(os.path.join(pathname,'areas.%s' % casename),
               area.T,delimiter=' ')
    with open(os.path.join(pathname,'names.%s' % casename),'wb') as f:
        pickle.dump(list(columns.keys()),f)

if __name__ == "__main__":

    # register a ILAMB region which exlcudes Antarctica
    from ILAMB.Regions import Regions
    ilamb_region = Regions()
    ilamb_region.addRegionLatLonBounds("noant","No Antarctica",(-60,89.999),(-179.999,179.999),"")
    ilamb_region.addRegionNetCDF4(os.path.join(os.environ["ILAMB_ROOT"],"DATA/regions/GlobalLand.nc"))

    # read in the E3SM model and cache
    pkl_file = "./E3SM-1-1.pkl"
    if os.path.isfile(pkl_file):
        with open(pkl_file,'rb') as infile:
            m = pickle.load(infile)
    else:
        m = ModelResult("",modelname = "E3SM-1-1",filter = "E3SM-1-1",
                        paths = ["/gpfs/alpine/cli143/proj-shared/mxu/CMIP6/CMIP",
                                 "/gpfs/alpine/cli143/proj-shared/mxu/CMIP6/ScenarioMIP",
                                 "/gpfs/alpine/cli143/proj-shared/mxu/drought_indices_reprocess/indices_1850-2100-calib1850-2100/",
                                 "/ccs/home/nate/cli143/Deeksha/analysis/E3SM/postproc"])
        # a little hacking to only get the combined and not combined-bgc
        for v in m.variables:
            l = [a for a in m.variables[v] if '_combined_' in a]
            if len(l)==1: m.variables[v] = l
        with open(pkl_file,'wb') as out:
            pickle.dump(m,out,pickle.HIGHEST_PROTOCOL)

    # setup clustering
    if 0:
        PrepareClusterInputs("standard_e3sm",      # casename
                             m,                    # ModelResult or list of them
                             ['tas','pr','mrsos'], # variables to include
                             range(1850,2101,10),  # the time slices to use in years
                             variability = True,   # will also include stdev in the clusters
                             regions=['global','noant']) # the regions over which we want to cluster
    if 0:
        PrepareClusterInputs("change_e3sm",        # casename
                             m,                    # ModelResult or list of them
                             ['tas','pr','mrsos'], # variables to include
                             range(1960,2101,10),  # the time slices to use in years
                             variability = True,   # will also include stdev in the clusters
                             regions=['global','noant'], # the regions over which we want to cluster
                             reference_period = [1960,1990]) # if reference period used, we get change clusters
    if 0:
        PrepareClusterInputs("drought_e3sm",
                             m,
                             ['scpdsi','spi_pearson_03','spi_pearson_09','spei_pearson_03','spei_pearson_09'],
                             range(1850,2101,10),
                             variability = False,
                             regions=['global','noant'])
    if 1:
        PrepareClusterInputs("extremes_e3sm",       # casename
                             m,                     # ModelResult or list of them
                             ['pr_q90','tasmax_q90','tasmin_q05'], # variables to include
                             range(1850,2101,10),   # the time slices to use in years
                             variability = False,   # will also include stdev in the clusters
                             regions=['global','noant']) # the regions over which we want to cluster

"""
drought if < -1
threshold on the 5th percentile, 10th percentile
pr/tas percentiles?
"""

import os
import pickle
import numpy as np
from ILAMB.ModelResult import ModelResult
from ILAMB.Variable import Variable
from ILAMB.Regions import Regions

ilamb_region = Regions()

pref_units = {}
pref_units['pr'] = 'mm d-1'
pref_units['gpp'] = 'g m-2 d-1'
pref_units['tas'] = 'degC'
pref_units['tasmin'] = 'degC'
pref_units['tasmax'] = 'degC'
pref_units['tasmaxQ10'] = 'degC'
pref_units['tasmaxQ30'] = 'degC'
pref_units['tasmaxQ50'] = 'degC'
pref_units['tasmaxQ70'] = 'degC'
pref_units['tasmaxQ90'] = 'degC'

def PrepareClusterInputs(casename,models,variables,times,regions=["global"],quiet=False):
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
    variables : list of dict
        the variables to include in the clustering
    times : array-like
        the boundaries of the intervals to be considered in years
    quiet : bool
        enable to turnoff the debugging output

    """
    # check types
    if type(models) == ModelResult: models = [models]
    for m in models: assert(type(m) == ModelResult)
    if type(regions)==str: regions = [regions]
    years = np.asarray(times)[:-1]
    times = (np.asarray(times,dtype=float)-1850)*365

    # pop models that do not have the required variables
    V = []
    [V.extend(d['vars']) for d in variables]    
    M = [m for m in models if set(V).issubset(m.variables.keys())]
    if len(M) != len(models) and not quiet:
        pop = list(set(models).difference(set(M)))
        print("Some models do not have required variables:")
        for m in pop:
            print("  - %s: missing [%s]" % (m.name,",".join(list(set(V).difference(m.variables.keys())))))
    if len(M) == 0:
        print("No model contained all variables, exiting")
        return

    rstack = []
    for m in M:
        columns = {}        
        for j,dct in enumerate(variables):
            if 'reference_period' not in dct: continue
            reference_period = (np.asarray(dct['reference_period'],dtype=float)-1850)*365
            assert reference_period.size == 2
            for vname in dct['vars']:
                v = m.extractTimeSeries(vname,initial_time=reference_period[0],final_time=reference_period[1])
                v.trim(t=reference_period)
                if vname in pref_units: v.convert(pref_units[vname])
                lbl = 'mean(%s) [%s]' % (vname,v.unit)
                columns[lbl] = v.integrateInTime(mean=True)
                if 'variability' in dct and dct['variability']:
                    lbl = 'std(%s) [%s]' % (vname,v.unit)
                    columns[lbl] = v.variability()
        rstack.append(columns)
        
    # create the time slices
    tstack = []
    for t in range(times.size-1):
        mstack = []
        for i,m in enumerate(M):
            columns = {}
            for j,dct in enumerate(variables):
                change = True if 'reference_period' in dct else False                    
                for vname in dct['vars']:
                    
                    # extract the variable
                    v = m.extractTimeSeries(vname,initial_time=times[t],final_time=times[t+1])
                    v.trim(t=times[t:(t+2)])
                    if vname in pref_units: v.convert(pref_units[vname])                

                    if 'count' in dct:
                        if type(dct['count']) is not list: dct['count'] = [dct['count']]
                        lbl = 'count(%s%s) [1]' % (vname,dct['count'][0])
                        data = eval("*".join(["(v.data %s)" % i for i in dct['count']])).sum(axis=0)
                        columns[lbl] = Variable(data = data,
                                                unit = "1",
                                                lat = v.lat,
                                                lon = v.lon)
                    else:
                        lbl = 'mean(%s) [%s]' % (vname,v.unit)
                        columns[lbl] = v.integrateInTime(mean=True)
                        if change: columns[lbl].data -= rstack[i][lbl].data
                        if 'variability' in dct and dct['variability']:
                            lbl = 'std(%s) [%s]' % (vname,v.unit)
                            columns[lbl] = v.variability()
                            if change: columns[lbl].data -= rstack[i][lbl].data
            mstack.append(columns)
        tstack.append(mstack)
                        
    # data within each model must be uniformly masked, also build up
    # coords/areas
    LAT = []; LON = []; AREA = []; MODEL = []
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
        MODEL.append(np.ma.masked_array(np.ones(area.shape)*i,mask=mask).compressed())
    lat  = np.hstack(LAT)
    lon  = np.hstack(LON)
    area = np.hstack(AREA)
    area = np.vstack([area,np.hstack(MODEL)])
    
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
    with open(os.path.join(pathname,'models.%s' % casename),'wb') as f:
        pickle.dump([m.name for m in M],f)

def PrepareCluster(casename,models,variables,times,regions=["global"],quiet=False):
    """Uses ILAMB objects to prepare clustering for use with Forrest/Jitu's code.

    Parameters
    ----------
    casename : str
        the identifier to be used for this case
    models : ILAMB.ModelResult or list of ILAMB.ModelResult
        the models to include in the clustering
    variables : list of dict
        the variables to include in the clustering
    times : array-like
        the boundaries of the intervals to be considered in years
    quiet : bool
        enable to turnoff the debugging output

    """
    # check types
    if type(models) == ModelResult: models = [models]
    for m in models: assert(type(m) == ModelResult)
    if type(regions)==str: regions = [regions]
    years = np.asarray(times)[:-1]
    times = (np.asarray(times,dtype=float)-1850)*365
    pathname = os.path.join(casename,"data")
    if not os.path.isdir(pathname): os.makedirs(pathname)

    row = 0
    data = []
    stack = []
    for m in models:        
        lat = lon = area = mask = None
        
        for i,y in enumerate(years):

            complete = True
            columns  = {}
            
            for j,dct in enumerate(variables):
                for vname in dct['vars']:
                    if not complete: continue
                    
                    # extract the variable
                    try:
                        v = m.extractTimeSeries(vname,initial_time=times[i],final_time=times[i+1])
                        v.trim(t=times[i:(i+2)])
                        if vname in pref_units: v.convert(pref_units[vname])                
                    except:
                        complete = False
                        continue

                    # compute the aspect to cluster
                    lbl = 'mean(%s) [%s]' % (vname,v.unit)
                    columns[lbl] = v.integrateInTime(mean=True)
                    
            if not complete: continue
            if mask is None:

                # build up the mask
                for vname in columns:
                    v = columns[vname]
                    if mask is None: mask = v.data.mask
                    mask += v.data.mask
                for region in regions:
                    mask += ilamb_region.getMask(region,v)

                # build coordinate values
                lat,lon = np.meshgrid(v.lat,v.lon,indexing='ij')
                lat  = np.ma.masked_array(   lat,mask=mask).compressed()
                lon  = np.ma.masked_array(   lon,mask=mask).compressed()
                area = np.ma.masked_array(v.area,mask=mask).compressed()
                size = lat.size

                # write out the coordinates
                np.savetxt(os.path.join(pathname,'coords.%s' % m.name),
                           np.vstack([lon,lat,area]).T,delimiter=' ')

            # apply the mask consistently across all columns
            data.append(np.hstack([np.ma.masked_array(columns[vname].data,mask=mask).compressed() for vname in columns]))
            stack.append("%s %d %d" % (m.name,y,row))
            row += size

    np.savetxt(os.path.join(pathname,'ascii.%s' % casename),
               np.hstack(data).T,delimiter=' ')
    with open(os.path.join(pathname,'stack.%s' % casename),mode='w') as f:
        f.write("\n".join(stack))

if __name__ == "__main__":

    """ The following is just a sample of how you can setup a comparison,
    the below could sit in its own script where we just:

    from cluster_pre import PrepareClusterInputs

    The main difference is that now we pass in a dictionary with
    groups of variables and keywords of how we want things
    preprocessed. See below for samples.
    """
    # register a ILAMB region which exlcudes Antarctica
    from ILAMB.Regions import Regions
    ilamb_region = Regions()
    ilamb_region.addRegionLatLonBounds("noant","No Antarctica",(-60,89.999),(-179.999,179.999),"")
    ilamb_region.addRegionNetCDF4(os.path.join(os.environ["ILAMB_ROOT"],"DATA/regions/GlobalLand.nc"))

    # read in model
    M = []
    M.append(ModelResult("/home/nate/data/ILAMB/MODELS/CMIP6/CESM2/",modelname="CESM2"))
    M.append(ModelResult("",modelname = "Reference",paths = ["/home/nate/data/ILAMB/DATA/pr/GPCP2",
                                                             "/home/nate/data/ILAMB/DATA/tas/CRU"]))

    # setup clustering
    PrepareCluster("sample",
                   M,
                   [{'vars':['tas','pr'],'variability':False }],
                   range(1930,2011,10),
                   regions=['global','noant'])

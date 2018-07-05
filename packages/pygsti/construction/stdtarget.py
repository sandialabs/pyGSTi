""" Helper functions for standard gate set modules. """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import time as _time
import os as _os
import pickle as _pickle
import sys as _sys
import gzip as _gzip
import numpy as _np

from . import stdlists as _stdlists
from .. import objects as _objs
    
def _get_cachefile_names(std_module, param_type, sim_type):
    """ TODO: docstring (for this entire module) """
    py_version = 3 if (_sys.version_info > (3, 0)) else 2
    
    if param_type == "H+S terms":
        cachePath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "caches")

        assert(sim_type == "auto" or sim_type.startswith("termorder:")), "Invalid `sim_type` argument!"
        termOrder = 1 if sim_type == "auto" else int(sim_type.split(":")[1])
        fn = ("cacheHS%d." % termOrder) + std_module.__name__ + "_v%d" % py_version
        fn = _os.path.join(cachePath, fn)
        return fn + "_keys.pkz", fn + "_vals.npz"
    else:
        raise ValueError("No cache files used for param-type=%s" % param_type)

def _make_HScache_for_std_gateset(std_module, termOrder, maxLength, json_too=False):
    """ A utility routine to for creating the cache files for a standard gate set """
    gs_target = std_module.gs_target.copy()
    prep_fiducials = std_module.prepStrs
    effect_fiducials = std_module.effectStrs
    germs = std_module.germs

    x = 1
    maxLengths = []
    while(x <= maxLength):
        maxLengths.append(x)
        x *= 2
        
    listOfExperiments = _stdlists.make_lsgst_experiment_list(
                            gs_target, prep_fiducials, effect_fiducials, germs, maxLengths)
    print(len(listOfExperiments),"Experiments")

    gs_terms = gs_target.copy()
    gs_terms.set_all_parameterizations("H+S terms") # CPTP terms?
    calc_cache = {}
    gs_terms.set_simtype("termorder:%d" % termOrder,calc_cache)

    t0 = _time.time()
    for i,gstr in enumerate(listOfExperiments):
        print("%.2fs: Computing prob %d of %d" % (_time.time()-t0,i,len(listOfExperiments)))
        gs_terms.probs(gstr)

    #gs_terms.bulk_probs(listOfExperiments) # also fills cache, but allocs more mem at once
    print("Completed in %.2fs" % (_time.time()-t0))
    print("Cachesize = ",len(calc_cache))
    print("Num of Experiments = ", len(listOfExperiments))

    key_fn, val_fn = _get_cachefile_names(std_module, "H+S terms", "termorder:%d" % termOrder)
    _write_calccache(calc_cache, key_fn, val_fn, json_too)


def _write_calccache(calc_cache, key_fn, val_fn, json_too=False):

    keys = list(calc_cache.keys())
    def conv_key(ky): # converts key to native python objects for faster serialization (but *same* hashing)
        return (ky[0], ky[1].tonative(), ky[2].tonative(), tuple([x.tonative() for x in ky[3]]) )

    ckeys = [ conv_key(x) for x in keys ]
    with _gzip.open(key_fn,'wb') as f:
        _pickle.dump(ckeys, f, protocol=_pickle.HIGHEST_PROTOCOL)
    print("Wrote %s" % key_fn)

    if json_too: # for Python 2 & 3 compatibility
        import os as _os
        from ..io import json as _json
        key_fn_json = _os.path.splitext(key_fn)[0] + ".json" 
        with open(key_fn_json,'w') as f:
            _json.dump(ckeys, f)
        print("Wrote %s" % key_fn_json)
    
    values = [calc_cache[k] for k in keys]
    vtape = []; ctape = []
    for v in values:
        vt,ct = v.compact()
        vtape.append(vt)
        ctape.append(ct)    
    vtape = _np.concatenate(vtape)
    ctape = _np.concatenate(ctape)
    _np.savez_compressed(val_fn, vtape=vtape, ctape=ctape)
    print("Wrote %s" % val_fn)

def _load_calccache(key_fn, val_fn):
    #print("Loading cache..."); t0 = _time.time()
    with _gzip.open(key_fn,"rb") as f:            
        keys = _pickle.load(f)
    npfile = _np.load(val_fn)
    vals = _objs.polynomial.bulk_load_compact_polys((npfile['vtape'],npfile['ctape']))
    calc_cache = { k:v for k,v in zip(keys,vals) }
    #print("Done in %.1fs" % (_time.time()-t0))
    return calc_cache

    
def _copy_target(std_module, param_type, sim_type="auto", gscache=None):

    if gscache is not None:
        if (param_type,sim_type) in gscache:
            return gscache[(param_type,sim_type)].copy()

    gs = std_module.gs_target.copy()
    gs.set_all_parameterizations(param_type) # automatically sets sim_type
    if param_type == "H+S terms": 
        assert(sim_type == "auto" or sim_type.startswith("termorder:")), "Invalid `sim_type` argument!"
        simt = "termorder:1" if sim_type == "auto" else sim_type # don't update sim_type b/c gscache
        key_fn, val_fn = _get_cachefile_names(std_module, param_type, simt)
        if _os.path.exists(key_fn) and _os.path.exists(val_fn):        
            calc_cache = _load_calccache(key_fn, val_fn)
        else:
            calc_cache = {}

        gs.set_simtype(simt,calc_cache)
    else:
        if sim_type != "auto":
            gs.set_simtype(sim_type)

    if gscache is not None:
        gscache[(param_type,sim_type)] = gs
    
    return gs.copy()

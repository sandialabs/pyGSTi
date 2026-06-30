"""
Helper functions for standard model modules.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

# XXX this module should probably be deprecated with the new `pygsti.modelpacks` API

import gzip as _gzip
import itertools as _itertools
import os as _os
import pickle as _pickle

import numpy as _np

from pygsti.baseobjs import polynomial as _polynomial


def _get_cachefile_names(std_module, param_type, simulator, py_version):
    """ Get the standard cache file names for a module """
    # No more "H+S terms" parametype
    # if param_type == "H+S terms":
    #     cachePath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
    #                               "../construction/caches")
    #
    #     assert(simulator == "auto" or isinstance(simulator, _TermFSim)), "Invalid `simulator` argument!"
    #     termOrder = 1 if simulator == "auto" else simulator.max_order
    #     fn = ("cacheHS%d." % termOrder) + std_module.__name__ + "_v%d" % py_version
    #     fn = _os.path.join(cachePath, fn)
    #     return fn + "_keys.pkz", fn + "_vals.npz"
    # else:
    raise ValueError("No cache files used for param-type=%s" % param_type)


# XXX apparently only used from _make_hs_cache_for_std_model which itself looks unused
def _write_calccache(calc_cache, key_fn, val_fn, comm=None):
    """
    Write `caclcache`, a dictionary of compact polys, to disk in two files,
    one for the keys and one for the values.

    This function can be called by multiple ranks and passed `comm` to
    synchronize collecting and writing a single set of cache files.

    Parameters
    ----------
    calc_cache : dict
        The cache of calculated (compact) polynomial to save to disk.

    key_fn, val_fn : str
        key and value filenames.

    comm : mpi4py.MPI.comm
        Communicator for synchronizing across multiple ranks (each with different
        `calc_cache` args that need to be gathered.

    Returns
    -------
    None
    """
    keys = list(calc_cache.keys())

    def conv_key(ky):  # converts key to native python objects for faster serialization (but *same* hashing)
        return (ky[0], ky[1].to_native(), ky[2].to_native(), tuple([x.to_native() for x in ky[3]]))

    ckeys = [conv_key(x) for x in keys]

    #Gather keys onto rank 0 processor if necessary
    # (Note: gathering relies on .gather and .Gather using the *same* rank ordering)
    if comm is not None:
        ckeys_list = comm.gather(ckeys, root=0)
    else:
        ckeys_list = [ckeys]

    if (comm is None) or (comm.Get_rank() == 0):
        ckeys = list(_itertools.chain(*ckeys_list))
        print("Writing cache of size = ", len(ckeys))

        with _gzip.open(key_fn, 'wb') as f:
            _pickle.dump(ckeys, f, protocol=_pickle.HIGHEST_PROTOCOL)
        print("Wrote %s" % key_fn)

    if len(keys) > 0:  # some procs might have 0 keys (e.g. the "scheduler")
        values = [calc_cache[k] for k in keys]
        vtape = []; ctape = []
        for v in values:
            vt, ct = v  # .compact() # Now cache hold compact polys already
            vtape.append(vt)
            ctape.append(ct)
        vtape = _np.concatenate(vtape)
        ctape = _np.concatenate(ctape)
        if comm is not None:
            comm.allgather(vtape.dtype)
            comm.allgather(ctape.dtype)
    else:
        #Need to create vtape and ctape of length 0 and *correct type*
        if comm is not None:
            vtape_types = comm.allgather(None)
            ctape_types = comm.allgather(None)
        else:
            vtape_types = ctape_types = []  # will cause us to use default type below

        for typ in vtape_types:
            if typ is not None:
                vtape = _np.zeros(0, typ); break
        else:
            vtape = _np.zeros(0, _np.int64)  # default type = int64

        for typ in ctape_types:
            if typ is not None:
                ctape = _np.zeros(0, typ); break
        else:
            ctape = _np.zeros(0, complex)  # default type = complex

    #Gather keys onto rank 0 processor if necessary
    if comm is not None:
        sizes = comm.gather(vtape.size, root=0)
        recvbuf = (_np.empty(sum(sizes), vtape.dtype), sizes) \
            if (comm.Get_rank() == 0) else None
        comm.Gatherv(sendbuf=vtape, recvbuf=recvbuf, root=0)
        if comm.Get_rank() == 0: vtape = recvbuf[0]

        sizes = comm.gather(ctape.size, root=0)
        recvbuf = (_np.empty(sum(sizes), ctape.dtype), sizes) \
            if (comm.Get_rank() == 0) else None
        comm.Gatherv(sendbuf=ctape, recvbuf=recvbuf, root=0)
        if comm.Get_rank() == 0: ctape = recvbuf[0]

    if comm is None or comm.Get_rank() == 0:
        _np.savez_compressed(val_fn, vtape=vtape, ctape=ctape)
        print("Wrote %s" % val_fn)


def _load_calccache(key_fn, val_fn):
    """
    The complement to _write_calccache, this function loads a cache
    dictionary from key and value filenames.

    Parameters
    ----------
    key_fn, val_fn : str
        key and value filenames.

    Returns
    -------
    dict
        The cache of calculated (compact) polynomials.
    """
    #print("Loading cache..."); t0 = _time.time()
    with _gzip.open(key_fn, "rb") as f:
        keys = _pickle.load(f)
    npfile = _np.load(val_fn)
    vals = _polynomial.bulk_load_compact_polynomials(npfile['vtape'], npfile['ctape'], keep_compact=True)
    calc_cache = {k: v for k, v in zip(keys, vals)}
    #print("Done in %.1fs" % (_time.time()-t0))
    return calc_cache


def _copy_target(std_module, param_type, simulator="auto", gscache=None):
    """
    Returns a copy of `std_module._target_model` in the given parameterization.

    Parameters
    ----------
    std_module : module
        The standard model module whose target model should be
        copied and returned.

    param_type : {"TP", "CPTP", "H+S", "S", ... }
        The gate and SPAM vector parameterization type. See
        :func:`Model.set_all_parameterizations` for all allowed values.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The simulator (or type) to be used for model calculations (leave as
        "auto" if you're not sure what this is).

    gscache : dict, optional
        A dictionary for maintaining the results of past calls to
        `_copy_target`.  Keys are `(param_type, simulator)` tuples and values
        are `Model` objects.  If `gscache` contains the requested
        `param_type` and `simulator` then a copy of the cached value is
        returned instead of doing any real work.  Furthermore, if `gscache`
        is not None and a new `Model` is constructed, it will be added
        to the given `gscache` for future use.

    Returns
    -------
    Model
    """
    #TODO: to get this working we need to be able to hash forward simulators, which should be done
    # without regard to the parent model (just, e.g. the max_order, etc. of a TermForwardSimulator).
    if gscache is not None:
        if (param_type, simulator) in gscache:
            return gscache[(param_type, simulator)].copy()

    mdl = std_module._target_model.copy()
    mdl.set_all_parameterizations(param_type)  # automatically sets simulator

    # No more "H+S terms" paramtype (update in FUTURE?)
    # if param_type == "H+S terms":
    #     assert(simulator == "auto" or isinstance(simulator, _TermFSim)), "Invalid `simulator` argument!"
    #     # Note: don't update `simulator` variable here as it's used below for setting gscache element.
    #     sim = _TermFSim(mode="taylor", max_order=1) if simulator == "auto" else simulator
    #     py_version = 3 if (_sys.version_info > (3, 0)) else 2
    #     calc_cache = {}  # the default
    #
    #     key_fn, val_fn = _get_cachefile_names(std_module, param_type, sim, py_version)
    #     if _os.path.exists(key_fn) and _os.path.exists(val_fn):
    #         calc_cache = _load_calccache(key_fn, val_fn)
    #     elif py_version == 3:  # python3 will try to load python2 files as a fallback
    #         key_fn, val_fn = _get_cachefile_names(std_module, param_type, sim, 2)
    #         if _os.path.exists(key_fn) and _os.path.exists(val_fn):
    #             calc_cache = _load_calccache(key_fn, val_fn)
    #
    #     sim.set_cache(calc_cache)  # TODO
    #     mdl.sim = sim
    # else:
    if simulator != "auto":
        mdl.sim = simulator

    if gscache is not None:
        gscache[(param_type, simulator)] = mdl

    return mdl.copy()


""" Helper functions for standard model modules. """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

# XXX this module should probably be deprecated with the new `pygsti.modelpacks` API

import time as _time
import os as _os
import pickle as _pickle
import sys as _sys
import gzip as _gzip
import numpy as _np
import itertools as _itertools

from . import stdlists as _stdlists
from .. import objects as _objs
from ..tools import mpitools as _mpit


def _get_cachefile_names(std_module, param_type, sim_type, py_version):
    """ Get the standard cache file names for a module """

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


# XXX is this used?
def _make_HScache_for_std_model(std_module, termOrder, maxLength, json_too=False, comm=None):
    """
    A utility routine to for creating the term-based cache files for a standard module
    """
    target_model = std_module.target_model()
    prep_fiducials = std_module.prepStrs
    effect_fiducials = std_module.effectStrs
    germs = std_module.germs

    x = 1
    maxLengths = []
    while(x <= maxLength):
        maxLengths.append(x)
        x *= 2

    listOfExperiments = _stdlists.make_lsgst_experiment_list(
        target_model, prep_fiducials, effect_fiducials, germs, maxLengths)

    mdl_terms = target_model.copy()
    mdl_terms.set_all_parameterizations("H+S terms")  # CPTP terms?
    my_calc_cache = {}
    mdl_terms.set_simtype("termorder:%d" % termOrder, my_calc_cache)

    comm_method = "scheduler"
    if comm is not None and comm.Get_size() > 1 and comm_method == "scheduler":
        from mpi4py import MPI  # just needed for MPI.SOURCE below

        #Alternate: use rank0 as "scheduler"
        rank = 0 if (comm is None) else comm.Get_rank()
        nprocs = 1 if (comm is None) else comm.Get_size()
        N = len(listOfExperiments); cur_index = 0; active_workers = nprocs - 1
        buf = _np.zeros(1, _np.int64)  # use buffer b/c mpi4py .send/.recv seem buggy
        if rank == 0:
            # ** I am the scheduler **
            # Give each "worker" rank an initial index to compute
            for i in range(1, nprocs):
                if cur_index == N:  # there are more procs than items - just send -1 index to mean "you're done!"
                    buf[0] = -1
                    comm.Send(buf, dest=i, tag=1)  # tag == 1 => scheduler to worker
                    active_workers -= 1
                else:
                    buf[0] = cur_index
                    comm.Send(buf, dest=i, tag=1); cur_index += 1

            # while there are active workers keep dishing out indices
            while active_workers > 0:
                comm.Recv(buf, source=MPI.ANY_SOURCE, tag=2)  # worker requesting assignment
                worker_rank = buf[0]
                if cur_index == N:  # nothing more to do: just send -1 index to mean "you're done!"
                    buf[0] = -1
                    comm.Send(buf, dest=worker_rank, tag=1)  # tag == 1 => scheduler to worker
                    active_workers -= 1
                else:
                    buf[0] = cur_index
                    comm.Send(buf, dest=worker_rank, tag=1)
                    cur_index += 1

        else:
            # ** I am a worker **
            comm.Recv(buf, source=0, tag=1)
            index_to_compute = buf[0]

            while index_to_compute >= 0:
                print("Worker %d computing prob %d of %d" % (rank, index_to_compute, N))
                t0 = _time.time()
                mdl_terms.probs(listOfExperiments[index_to_compute])
                print("Worker %d finished computing prob %d in %.2fs" % (rank, index_to_compute, _time.time() - t0))

                buf[0] = rank
                comm.Send(buf, dest=0, tag=2)  # tag == 2 => worker requests next assignment
                comm.Recv(buf, source=0, tag=1)
                index_to_compute = buf[0]

        print("Rank %d at barrier" % rank)
        comm.barrier()  # wait here until all workers and scheduler are done

    else:

        #divide up strings among ranks
        my_expList, _, _ = _mpit.distribute_indices(listOfExperiments, comm, False)
        rankStr = "" if (comm is None) else "Rank%d: " % comm.Get_rank()

        if comm is not None and comm.Get_rank() == 0:
            print("%d operation sequences divided among %d processors" % (len(listOfExperiments), comm.Get_size()))

        t0 = _time.time()
        for i, opstr in enumerate(my_expList):
            print("%s%.2fs: Computing prob %d of %d" % (rankStr, _time.time() - t0, i, len(my_expList)))
            mdl_terms.probs(opstr)
        #mdl_terms.bulk_probs(my_expList) # also fills cache, but allocs more mem at once

    py_version = 3 if (_sys.version_info > (3, 0)) else 2
    key_fn, val_fn = _get_cachefile_names(std_module, "H+S terms",
                                          "termorder:%d" % termOrder, py_version)
    _write_calccache(my_calc_cache, key_fn, val_fn, json_too, comm)

    if comm is None or comm.Get_rank() == 0:
        print("Completed in %.2fs" % (_time.time() - t0))
        print("Num of Experiments = ", len(listOfExperiments))

    #if comm is None:
    #    calcc_list = [ my_calc_cache ]
    #else:
    #    calcc_list = comm.gather(my_calc_cache, root=0)
    #
    #if comm is None or comm.Get_rank() == 0:
    #    calc_cache = {}
    #    for c in calcc_list:
    #        calc_cache.update(c)
    #
    #    print("Completed in %.2fs" % (_time.time()-t0))
    #    print("Cachesize = ",len(calc_cache))
    #    print("Num of Experiments = ", len(listOfExperiments))
    #
    #    py_version = 3 if (_sys.version_info > (3, 0)) else 2
    #    key_fn, val_fn = _get_cachefile_names(std_module, "H+S terms",
    #                                          "termorder:%d" % termOrder,py_version)
    #    _write_calccache(calc_cache, key_fn, val_fn, json_too, comm)


# XXX apparently only used from _make_HScache_for_std_model which itself looks unused
def _write_calccache(calc_cache, key_fn, val_fn, json_too=False, comm=None):
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

    json_too : bool, optional
        When true, the keys are also written in JSON format (to facilitate
        python2 & 3 compatibility)

    comm : mpi4py.MPI.comm
        Communicator for synchronizing across multiple ranks (each with different
        `calc_cache` args that need to be gathered.

    Returns
    -------
    None
    """
    keys = list(calc_cache.keys())

    def conv_key(ky):  # converts key to native python objects for faster serialization (but *same* hashing)
        return (ky[0], ky[1].tonative(), ky[2].tonative(), tuple([x.tonative() for x in ky[3]]))

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

        if json_too:  # for Python 2 & 3 compatibility
            import os as _os
            from ..io import json as _json
            key_fn_json = _os.path.splitext(key_fn)[0] + ".json"
            with open(key_fn_json, 'w') as f:
                _json.dump(ckeys, f)
            print("Wrote %s" % key_fn_json)

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
    vals = _objs.polynomial.bulk_load_compact_polys(npfile['vtape'], npfile['ctape'], keep_compact=True)
    calc_cache = {k: v for k, v in zip(keys, vals)}
    #print("Done in %.1fs" % (_time.time()-t0))
    return calc_cache


def _copy_target(std_module, param_type, sim_type="auto", gscache=None):
    """
    Returns a copy of `std_module._target_model` in the given parameterization.

    Parameters
    ----------
    std_module : module
        The standard model module whose target model should be
        copied and returned.

    param_type : {"TP", "CPTP", "H+S", "S", ... }
        The gate and SPAM vector parameterization type. See
        :function:`Model.set_all_parameterizations` for all allowed values.

    sim_type : {"auto", "matrix", "map", "termorder:X" }
        The simulator type to be used for model calculations (leave as
        "auto" if you're not sure what this is).

    gscache : dict, optional
        A dictionary for maintaining the results of past calls to
        `_copy_target`.  Keys are `(param_type,sim_type)` tuples and values
        are `Model` objects.  If `gscache` contains the requested
        `param_type` and `sim_type` then a copy of the cached value is
        returned instead of doing any real work.  Furthermore, if `gscache`
        is not None and a new `Model` is constructed, it will be added
        to the given `gscache` for future use.

    Returns
    -------
    Model
    """
    if gscache is not None:
        if (param_type, sim_type) in gscache:
            return gscache[(param_type, sim_type)].copy()

    mdl = std_module._target_model.copy()
    mdl.set_all_parameterizations(param_type)  # automatically sets sim_type
    if param_type == "H+S terms":
        assert(sim_type == "auto" or sim_type.startswith("termorder:")), "Invalid `sim_type` argument!"
        simt = "termorder:1" if sim_type == "auto" else sim_type  # don't update sim_type b/c gscache
        py_version = 3 if (_sys.version_info > (3, 0)) else 2
        calc_cache = {}  # the default

        key_fn, val_fn = _get_cachefile_names(std_module, param_type, simt, py_version)
        if _os.path.exists(key_fn) and _os.path.exists(val_fn):
            calc_cache = _load_calccache(key_fn, val_fn)
        elif py_version == 3:  # python3 will try to load python2 files as a fallback
            key_fn, val_fn = _get_cachefile_names(std_module, param_type, simt, 2)
            if _os.path.exists(key_fn) and _os.path.exists(val_fn):
                calc_cache = _load_calccache(key_fn, val_fn)

        mdl.set_simtype(simt, calc_cache)
    else:
        if sim_type != "auto":
            mdl.set_simtype(sim_type)

    if gscache is not None:
        gscache[(param_type, sim_type)] = mdl

    return mdl.copy()

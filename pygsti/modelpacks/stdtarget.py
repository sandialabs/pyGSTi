"""
Helper functions for standard model modules.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

# XXX this module should probably be deprecated with the new `pygsti.modelpacks` API

import gzip as _gzip
import itertools as _itertools
import collections as _collections
import os as _os
import pickle as _pickle

import numpy as _np

from pygsti.baseobjs import polynomial as _polynomial
from pygsti.baseobjs import statespace as _statespace
from pygsti.circuits import circuitconstruction as _gsc
from pygsti.tools.legacytools import deprecate as _deprecated_fn


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


# XXX is this used?
# def _make_hs_cache_for_std_model(std_module, term_order, max_length, json_too=False, comm=None):
#     """
#     A utility routine to for creating the term-based cache files for a standard module
#     """
#     target_model = std_module.target_model()
#     prep_fiducials = std_module.prepStrs
#     effect_fiducials = std_module.effectStrs
#     germs = std_module.germs
#
#     x = 1
#     maxLengths = []
#     while(x <= max_length):
#         maxLengths.append(x)
#         x *= 2
#
#     listOfExperiments = _stdlists.create_lsgst_circuits(
#         target_model, prep_fiducials, effect_fiducials, germs, maxLengths)
#
#     mdl_terms = target_model.copy()
#     mdl_terms.set_all_parameterizations("H+S terms")  # CPTP terms?
#     my_calc_cache = {}
#     mdl_terms.sim = _TermFSim(mode="taylor", max_order=term_order, cache=my_calc_cache)
#
#     comm_method = "scheduler"
#     if comm is not None and comm.Get_size() > 1 and comm_method == "scheduler":
#         from mpi4py import MPI  # just needed for MPI.SOURCE below
#
#         #Alternate: use rank0 as "scheduler"
#         rank = 0 if (comm is None) else comm.Get_rank()
#         nprocs = 1 if (comm is None) else comm.Get_size()
#         N = len(listOfExperiments); cur_index = 0; active_workers = nprocs - 1
#         buf = _np.zeros(1, _np.int64)  # use buffer b/c mpi4py .send/.recv seem buggy
#         if rank == 0:
#             # ** I am the scheduler **
#             # Give each "worker" rank an initial index to compute
#             for i in range(1, nprocs):
#                 if cur_index == N:  # there are more procs than items - just send -1 index to mean "you're done!"
#                     buf[0] = -1
#                     comm.Send(buf, dest=i, tag=1)  # tag == 1 => scheduler to worker
#                     active_workers -= 1
#                 else:
#                     buf[0] = cur_index
#                     comm.Send(buf, dest=i, tag=1); cur_index += 1
#
#             # while there are active workers keep dishing out indices
#             while active_workers > 0:
#                 comm.Recv(buf, source=MPI.ANY_SOURCE, tag=2)  # worker requesting assignment
#                 worker_rank = buf[0]
#                 if cur_index == N:  # nothing more to do: just send -1 index to mean "you're done!"
#                     buf[0] = -1
#                     comm.Send(buf, dest=worker_rank, tag=1)  # tag == 1 => scheduler to worker
#                     active_workers -= 1
#                 else:
#                     buf[0] = cur_index
#                     comm.Send(buf, dest=worker_rank, tag=1)
#                     cur_index += 1
#
#         else:
#             # ** I am a worker **
#             comm.Recv(buf, source=0, tag=1)
#             index_to_compute = buf[0]
#
#             while index_to_compute >= 0:
#                 print("Worker %d computing prob %d of %d" % (rank, index_to_compute, N))
#                 t0 = _time.time()
#                 mdl_terms.probabilities(listOfExperiments[index_to_compute])
#                 print("Worker %d finished computing prob %d in %.2fs" % (rank, index_to_compute, _time.time() - t0))
#
#                 buf[0] = rank
#                 comm.Send(buf, dest=0, tag=2)  # tag == 2 => worker requests next assignment
#                 comm.Recv(buf, source=0, tag=1)
#                 index_to_compute = buf[0]
#
#         print("Rank %d at barrier" % rank)
#         comm.barrier()  # wait here until all workers and scheduler are done
#
#     else:
#
#         #divide up strings among ranks
#         my_expList, _, _ = _mpit.distribute_indices(listOfExperiments, comm, False)
#         rankStr = "" if (comm is None) else "Rank%d: " % comm.Get_rank()
#
#         if comm is not None and comm.Get_rank() == 0:
#             print("%d circuits divided among %d processors" % (len(listOfExperiments), comm.Get_size()))
#
#         t0 = _time.time()
#         for i, opstr in enumerate(my_expList):
#             print("%s%.2fs: Computing prob %d of %d" % (rankStr, _time.time() - t0, i, len(my_expList)))
#             mdl_terms.probabilities(opstr)
#         #mdl_terms.bulk_probs(my_expList) # also fills cache, but allocs more mem at once
#
#     py_version = 3 if (_sys.version_info > (3, 0)) else 2
#     key_fn, val_fn = _get_cachefile_names(std_module, "H+S terms",
#                                           "termorder:%d" % term_order, py_version)
#     _write_calccache(my_calc_cache, key_fn, val_fn, json_too, comm)
#
#     if comm is None or comm.Get_rank() == 0:
#         print("Completed in %.2fs" % (_time.time() - t0))
#         print("Num of Experiments = ", len(listOfExperiments))
#
#     #if comm is None:
#     #    calcc_list = [ my_calc_cache ]
#     #else:
#     #    calcc_list = comm.gather(my_calc_cache, root=0)
#     #
#     #if comm is None or comm.Get_rank() == 0:
#     #    calc_cache = {}
#     #    for c in calcc_list:
#     #        calc_cache.update(c)
#     #
#     #    print("Completed in %.2fs" % (_time.time()-t0))
#     #    print("Cachesize = ",len(calc_cache))
#     #    print("Num of Experiments = ", len(listOfExperiments))
#     #
#     #    py_version = 3 if (_sys.version_info > (3, 0)) else 2
#     #    key_fn, val_fn = _get_cachefile_names(std_module, "H+S terms",
#     #                                          "termorder:%d" % term_order,py_version)
#    #    _write_calccache(calc_cache, key_fn, val_fn, json_too, comm)


# XXX apparently only used from _make_hs_cache_for_std_model which itself looks unused
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

        if json_too:  # for Python 2 & 3 compatibility
            from pygsti.serialization import json as _json
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


@_deprecated_fn("the pre-build SMQ modelpacks under `pygsti.modelpacks`")
def stdmodule_to_smqmodule(std_module):
    """
    Converts a pyGSTi "standard module" to a "standard multi-qubit module".

    PyGSTi provides a number of 1- and 2-qubit models corrsponding to commonly
    used gate sets, along with related meta-information.  Each such
    model+metadata is stored in a "standard module" beneath `pygsti.modelpacks.legacy`
    (e.g. `pygsti.modelpacks.legacy.std1Q_XYI` is the standard module for modeling a
    single-qubit quantum processor which can perform X(pi/2), Y(pi/2) and idle
    operations).  Because they deal with just 1- and 2-qubit models, multi-qubit
    labelling conventions are not used to improve readability.  For example, a
    "X(pi/2)" gate is labelled "Gx" (in a 1Q context) or "Gix" (in a 2Q context)
    rather than "Gx:0" or "Gx:1" respectively.

    There are times, however, when you many *want* a standard module with this
    multi-qubit labelling convention (e.g. performing 1Q-GST on the 3rd qubit
    of a 5-qubit processor).  We call such a module a standard *multi-qubit*
    module, and these typically begin with `"smq"` rather than `"std"`.

    Standard multi-qubit modules are *created* by this function.  For example,
    If you want the multi-qubit version of `pygsti.modelpacks.legacy.std1Q_XYI`
    you must:

    1. import `std1Q_XYI` (`from pygsti.modelpacks.legacy import std1Q_XYI`)
    2. call this function (i.e. `stdmodule_to_smqmodule(std1Q_XYI)`)
    3. import `smq1Q_XYI` (`from pygsti.modelpacks.legacy import smq1Q_XYI`)

    The `smq1Q_XYI` module will look just like the `std1Q_XYI` module but use
    multi-qubit labelling conventions.

    .. deprecated:: v0.9.9
        `stdmodule_to_smqmodule` will be removed in future versions of
        pyGSTi. Instead, import pre-built SMQ modelpacks directly from
        `pygsti.modelpacks`.

    Parameters
    ----------
    std_module : Module
        The standard module to convert to a standard-multi-qubit module.

    Returns
    -------
    Module
        The new module, although it's better to import this using the appropriate
        "smq"-prefixed name as described above.
    """
    from types import ModuleType as _ModuleType
    import sys as _sys
    import importlib

    std_module_name_parts = std_module.__name__.split('.')
    std_module_name_parts[-1] = std_module_name_parts[-1].replace('std', 'smq')
    new_module_name = '.'.join(std_module_name_parts)

    try:
        return importlib.import_module(new_module_name)
    except ImportError:
        pass  # ok, this is what the rest of the function is for

    out_module = {}
    std_target_model = std_module.target_model()  # could use ._target_model to save a copy
    dim = std_target_model.dim
    if dim == 4:
        sslbls = [0]
        find_replace_labels = {'Gi': (), 'Gx': ('Gx', 0), 'Gy': ('Gy', 0),
                               'Gz': ('Gz', 0), 'Gn': ('Gn', 0)}
        find_replace_strs = [((oldgl,), (newgl,)) for oldgl, newgl
                             in find_replace_labels.items()]
    elif dim == 16:
        sslbls = [0, 1]
        find_replace_labels = {'Gii': (),
                               'Gxi': ('Gx', 0), 'Gyi': ('Gy', 0), 'Gzi': ('Gz', 0),
                               'Gix': ('Gx', 1), 'Giy': ('Gy', 1), 'Giz': ('Gz', 1),
                               'Gxx': ('Gxx', 0, 1), 'Gxy': ('Gxy', 0, 1),
                               'Gyx': ('Gxy', 0, 1), 'Gyy': ('Gyy', 0, 1),
                               'Gcnot': ('Gcnot', 0, 1), 'Gcphase': ('Gcphase', 0, 1)}
        find_replace_strs = [((oldgl,), (newgl,)) for oldgl, newgl
                             in find_replace_labels.items()]
        #find_replace_strs.append( (('Gxx',), (('Gx',0),('Gx',1))) )
        #find_replace_strs.append( (('Gxy',), (('Gx',0),('Gy',1))) )
        #find_replace_strs.append( (('Gyx',), (('Gy',0),('Gx',1))) )
        #find_replace_strs.append( (('Gyy',), (('Gy',0),('Gy',1))) )
    else:
        #TODO: add qutrit?
        raise ValueError("Unsupported model dimension: %d" % dim)

    def upgrade_dataset(ds):
        """
        Update DataSet `ds` in-place to use  multi-qubit style labels.
        """
        ds.process_circuits_inplace(lambda s: _gsc.manipulate_circuit(
            s, find_replace_strs, sslbls))

    out_module['find_replace_gatelabels'] = find_replace_labels
    out_module['find_replace_circuits'] = find_replace_strs
    out_module['upgrade_dataset'] = upgrade_dataset

    # gate names
    out_module['gates'] = [find_replace_labels.get(nm, nm) for nm in std_module.gates]

    #Fully-parameterized target model (update labels)
    from pygsti.models.explicitmodel import ExplicitOpModel as _ExplicitOpModel
    state_space = _statespace.ExplicitStateSpace(sslbls)
    new_target_model = _ExplicitOpModel(state_space, std_target_model.basis.copy())
    new_target_model._evotype = std_target_model._evotype
    new_target_model._default_gauge_group = std_target_model._default_gauge_group

    #Note: setting object ._state_space is a bit of a hack here, and assumes
    # that these are "simple" objects that don't contain other sub-members that
    # need to have their state spaces updated too.
    for lbl, obj in std_target_model.preps.items():
        new_lbl = find_replace_labels.get(lbl, lbl)
        new_obj = obj.copy(); new_obj._state_space = state_space
        new_target_model.preps[new_lbl] = new_obj
    for lbl, obj in std_target_model.povms.items():
        new_lbl = find_replace_labels.get(lbl, lbl)
        new_obj = obj.copy(); new_obj._state_space = state_space
        for effect in new_obj.values():
            effect._state_space = state_space
        new_target_model.povms[new_lbl] = new_obj
    for lbl, obj in std_target_model.operations.items():
        new_lbl = find_replace_labels.get(lbl, lbl)
        new_obj = obj.copy(); new_obj._state_space = state_space
        new_target_model.operations[new_lbl] = new_obj
    for lbl, obj in std_target_model.instruments.items():
        new_lbl = find_replace_labels.get(lbl, lbl)
        new_obj = obj.copy(); new_obj._state_space = state_space
        for member in new_obj.values():
            member._state_space = state_space
        new_target_model.instruments[new_lbl] = new_obj
    out_module['_target_model'] = new_target_model

    # _stdtarget and _gscache need to be *locals* as well so target_model(...) works
    _stdtarget = importlib.import_module('.stdtarget', 'pygsti.modelpacks')
    _gscache = {("full", "auto"): new_target_model}
    out_module['_stdtarget'] = _stdtarget
    out_module['_gscache'] = _gscache

    def target_model(parameterization_type="full", simulator="auto"):
        """
        Returns a copy of the target model in the given parameterization.

        Parameters
        ----------
        parameterization_type : {"TP", "CPTP", "H+S", "S", ... }
            The gate and SPAM vector parameterization type. See
            :func:`Model.set_all_parameterizations` for all allowed values.

        simulator : ForwardSimulator or {"auto", "matrix", "map"}
            The simulator (or type) to be used for model calculations (leave as
            "auto" if you're not sure what this is).

        Returns
        -------
        Model
        """
        return _stdtarget._copy_target(_sys.modules[new_module_name], parameterization_type,
                                       simulator, _gscache)
    out_module['target_model'] = target_model

    # circuit lists
    circuitlist_names = ['germs', 'germs_lite', 'prepStrs', 'effectStrs', 'fiducials']
    for nm in circuitlist_names:
        if hasattr(std_module, nm):
            out_module[nm] = _gsc.manipulate_circuits(getattr(std_module, nm), find_replace_strs, sslbls)

    # clifford compilation (keys are lists of operation labels)
    if hasattr(std_module, 'clifford_compilation'):
        new_cc = _collections.OrderedDict()
        for ky, val in std_module.clifford_compilation.items():
            new_val = [find_replace_labels.get(lbl, lbl) for lbl in val]
            new_cc[ky] = new_val

    passthrough_names = ['global_fidPairs', 'pergerm_fidPairsDict', 'global_fidPairs_lite', 'pergerm_fidPairsDict_lite']
    for nm in passthrough_names:
        if hasattr(std_module, nm):
            out_module[nm] = getattr(std_module, nm)

    #Create the new module
    new_module = _ModuleType(str(new_module_name))  # str(.) converts to native string for Python 2 compatibility
    for k, v in out_module.items():
        setattr(new_module, k, v)
    _sys.modules[new_module_name] = new_module
    return new_module

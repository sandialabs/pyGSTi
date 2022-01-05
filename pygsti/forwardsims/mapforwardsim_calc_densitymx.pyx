# encoding: utf-8
# cython: profile=False
# cython: linetrace=False

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from libc.math cimport sqrt, log
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from ..evotypes.densitymx.statereps cimport StateRep, StateCRep
from ..evotypes.densitymx.opreps cimport OpRep, OpCRep
from ..evotypes.densitymx.effectreps cimport EffectRep, EffectCRep

import time as pytime
import numpy as np
cimport numpy as np
cimport cython

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
#from ..tools import optools as _ot
from ..tools.matrixtools import _fas

#DEBUG REMOVE MEMORY PROFILING
#import os, psutil
#process = psutil.Process(os.getpid())
#def print_mem_usage(prefix):
#    print("%s: mem usage = %.3f GB" % (prefix, process.memory_info().rss / (1024.0**3)))

#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

ctypedef OpCRep* OpCRep_ptr
ctypedef StateCRep* StateCRep_ptr
ctypedef EffectCRep* EffectCRep_ptr

ctypedef double (*TD_obj_fn)(double, double, double, double, double, double, double)


def propagate_staterep(staterep, operationreps):
    # FUTURE: could use inner C-reps to do propagation
    # instead of using extension type wrappers as this does now
    ret = staterep
    for oprep in operationreps:
        ret = oprep.acton(ret)
        # DEBUG print("post-action rhorep = ",str(ret))
    return ret


# -----------------------------------------
# Python -> C Conversion functions
# -----------------------------------------

cdef vector[vector[INT]] convert_maplayout(layout_atom, operation_lookup, rho_lookup):
    # c_layout :
    # an array of INT-arrays; each INT-array is [iDest,iStart,iCache,<remainder gate indices>]
    cdef vector[INT] intarray
    cdef vector[vector[INT]] c_layout_atom = vector[vector[INT]](len(layout_atom.table))
    for kk, (iDest, iStart, remainder, iCache) in enumerate(layout_atom.table.contents):
        if iStart is None: iStart = -1 # so always an int
        if iCache is None: iCache = -1 # so always an int
        remainder = remainder.circuit_without_povm.layertup
        intarray = vector[INT](3 + len(remainder))
        intarray[0] = iDest
        intarray[1] = iStart
        intarray[2] = iCache
        if iStart == -1:  # then first element of remainder is a rholabel
            intarray[3] = rho_lookup[remainder[0]]
            for jj,gl in enumerate(remainder[1:],start=4):
                intarray[jj] = operation_lookup[gl]
        else:
            for jj,gl in enumerate(remainder, start=3):
                intarray[jj] = operation_lookup[gl]
        c_layout_atom[kk] = intarray

    return c_layout_atom

cdef vector[vector[INT]] convert_dict_of_intlists(d):
    # d is an dict of lists of integers, whose keys are integer
    # indices from 0 to len(d).  We can convert this
    # to a vector of vector[INT] elements.
    cdef INT i, j;
    cdef vector[vector[INT]] ret = vector[vector[INT]](len(d))
    for i, intlist in d.items():
        ret[i] = vector[INT](len(intlist))
        for j in range(len(intlist)):
            ret[i][j] = intlist[j]
    return ret

cdef vector[vector[INT]] convert_and_wrap_dict_of_intlists(d, wrapper):
    # d is an dict of lists of integers, whose keys are integer
    # indices from 0 to len(d).  We can convert this
    # to a vector of vector[INT] elements.
    cdef INT i, j;
    cdef vector[vector[INT]] ret = vector[vector[INT]](len(d))
    for i, intlist in d.items():
        ret[i] = vector[INT](len(intlist))
        for j in range(len(intlist)):
            ret[i][j] = wrapper[intlist[j]]
    return ret

cdef vector[StateCRep*] create_rhocache(INT cacheSize, INT state_dim):
    cdef INT i
    cdef vector[StateCRep*] rho_cache = vector[StateCRep_ptr](cacheSize)
    for i in range(cacheSize): # fill cache with empty but alloc'd states
        rho_cache[i] = new StateCRep(state_dim)
    return rho_cache

cdef void free_rhocache(vector[StateCRep*] rho_cache):
    cdef UINT i
    for i in range(rho_cache.size()): # fill cache with empty but alloc'd states
        del rho_cache[i]

cdef vector[OpCRep*] convert_opreps(operationreps):
    # c_opreps : an array of OpCReps
    cdef vector[OpCRep*] c_opreps = vector[OpCRep_ptr](len(operationreps))
    for ii,grep in operationreps.items(): # (ii = python variable)
        c_opreps[ii] = (<OpRep?>grep).c_rep
    return c_opreps

cdef StateCRep* convert_rhorep(rhorep):
    # extract c-reps from rhorep and ereps => c_rho and c_ereps
    return (<StateRep?>rhorep).c_state

cdef vector[StateCRep*] convert_rhoreps(rhoreps):
    cdef vector[StateCRep*] c_rhoreps = vector[StateCRep_ptr](len(rhoreps))
    for ii,rrep in rhoreps.items(): # (ii = python variable)
        c_rhoreps[ii] = (<StateRep?>rrep).c_state
    return c_rhoreps

cdef vector[EffectCRep*] convert_ereps(ereps):
    cdef vector[EffectCRep*] c_ereps = vector[EffectCRep_ptr](len(ereps))
    for i in range(len(ereps)):
        c_ereps[i] = (<EffectRep>ereps[i]).c_effect
    return c_ereps


# -----------------------------------------
# Mapfill functions
# -----------------------------------------

def mapfill_probs_atom(fwdsim, np.ndarray[double, mode="c", ndim=1] array_to_fill,
                       dest_indices, layout_atom, resource_alloc):

    # The required ending condition is that array_to_fill on each processor has been filled.  But if
    # memory is being shared and resource_alloc contains multiple processors on a single host, we only
    # want *one* (the rank=0) processor to perform the computation, since array_to_fill will be
    # shared memory that we don't want to have muliple procs using simultaneously to compute the
    # same thing.  Thus, we carefully guard any shared mem updates/usage
    # using "if shared_mem_leader" (and barriers, if needed) below.
    shared_mem_leader = resource_alloc.is_host_leader if (resource_alloc is not None) else True

    dest_indices = _slct.to_array(dest_indices)  # make sure this is an array and not a slice
    #dest_indices = np.ascontiguousarray(dest_indices) #unneeded

    #Get (extension-type) representation objects
    rho_lookup = { lbl:i for i,lbl in enumerate(layout_atom.rho_labels) } # rho labels -> ints for faster lookup
    rhoreps = { i: fwdsim.model._circuit_layer_operator(rholbl, 'prep')._rep for rholbl,i in rho_lookup.items() }
    operation_lookup = { lbl:i for i,lbl in enumerate(layout_atom.op_labels) } # operation labels -> ints for faster lookup
    operationreps = { i:fwdsim.model._circuit_layer_operator(lbl, 'op')._rep for lbl,i in operation_lookup.items() }
    ereps = [fwdsim.model._circuit_layer_operator(elbl, 'povm')._rep for elbl in layout_atom.full_effect_labels]  # cache these in future

    # convert to C-mode:  evaltree, operation_lookup, operationreps
    cdef vector[vector[INT]] c_layout_atom = convert_maplayout(layout_atom, operation_lookup, rho_lookup)
    cdef vector[StateCRep*] c_rhos = convert_rhoreps(rhoreps)
    cdef vector[EffectCRep*] c_ereps = convert_ereps(ereps)
    cdef vector[OpCRep*] c_opreps = convert_opreps(operationreps)

    # create rho_cache = vector of StateCReps
    #print "DB: creating rho_cache of size %d * %g GB => %g GB" % \
    #   (layout_atom.cache_size, 8.0 * fwdsim.model.dim / 1024.0**3, layout_atom.cache_size * 8.0 * fwdsim.model.dim / 1024.0**3)
    cdef vector[StateCRep*] rho_cache = create_rhocache(layout_atom.cache_size, fwdsim.model.dim)
    cdef vector[vector[INT]] elabel_indices_per_circuit = convert_dict_of_intlists(layout_atom.elbl_indices_by_expcircuit)
    cdef vector[vector[INT]] final_indices_per_circuit = convert_and_wrap_dict_of_intlists(
        layout_atom.elindices_by_expcircuit, dest_indices)

    #DEBUG REMOVE
    #print_mem_usage("MAPFILL PROBS begin")
    #for i in [1808, 419509, 691738, 497424]:
    #    from ..evotypes.densitymx.opreps import OpRepComposed
    #    op = operationreps[i]
    #    if isinstance(op.embedded_rep, OpRepComposed):
    #        extra = " factors = " + ', '.join([str(type(opp)) for opp in op.embedded_rep.factor_reps])
    #    else:
    #        extra = ""
    #    print("ID ",i,str(type(op)),str(type(op.embedded_rep)), extra)

    if shared_mem_leader:
        #Note: dm_mapfill_probs could have taken a resource_alloc to employ multiple cpus to do computation.
        # Since array_fo_fill is assumed to be shared mem it would need to only update `array_to_fill` *if*
        # it were the host leader.
        dm_mapfill_probs(array_to_fill, c_layout_atom, c_opreps, c_rhos, c_ereps, &rho_cache,
                         elabel_indices_per_circuit, final_indices_per_circuit, fwdsim.model.dim)

    free_rhocache(rho_cache)  #delete cache entries


cdef dm_mapfill_probs(double[:] array_to_fill,
                      vector[vector[INT]] c_layout_atom,
                      vector[OpCRep*] c_opreps,
                      vector[StateCRep*] c_rhoreps, vector[EffectCRep*] c_ereps,
                      vector[StateCRep*]* prho_cache,
                      vector[vector[INT]] elabel_indices_per_circuit,
                      vector[vector[INT]] final_indices_per_circuit,
                      INT dim):

    #Note: we need to take in rho_cache as a pointer b/c we may alter the values its
    # elements point to (instead of copying the states) - we just guarantee that in the end
    # all of the cache entries are filled with allocated (by 'new') states that the caller
    # can deallocate at will.
    cdef INT k,l,i,istart, icache, iFirstOp, precomp_id
    cdef double p
    cdef StateCRep *init_state
    cdef StateCRep *prop1
    cdef StateCRep *tprop
    cdef StateCRep *final_state
    cdef StateCRep *prop2 = new StateCRep(dim)
    cdef StateCRep *shelved = new StateCRep(dim)
    cdef StateCRep *precomp_state

    cdef vector[INT] final_indices
    cdef vector[INT] elabel_indices

    #Invariants required for proper memory management:
    # - upon loop entry, prop2 is allocated and prop1 is not (it doesn't "own" any memory)
    # - all rho_cache entries have been allocated via "new"
    #REMOVE print("MAPFILL PROBS begin cfn")
    for k in range(<INT>c_layout_atom.size()):
        t0 = pytime.time() # DEBUG
        intarray = c_layout_atom[k]
        i = intarray[0]
        istart = intarray[1]
        icache = intarray[2]

        #REMOVE print_mem_usage("mapfill_probs_block: BEGIN %d of %d: sz=%d" % (k, c_layout_atom.size(), intarray.size()))

        if istart == -1:
            init_state = c_rhoreps[intarray[3]]
            iFirstOp = 4
        else:
            init_state = deref(prho_cache)[istart]
            iFirstOp = 3

        #DEBUG
        #print "LOOP i=",i," istart=",istart," icache=",icache," remcnt=",(intarray.size()-3)
        #print [ init_state._dataptr[t] for t in range(4) ]

        #Propagate state rep
        # prop2 should already be alloc'd; need to "allocate" prop1 - either take from cache or from "shelf"
        prop1 = shelved if icache == -1 else deref(prho_cache)[icache]
        prop1.copy_from(init_state) # copy init_state -> prop1
        #print " prop1:";  print [ prop1._dataptr[t] for t in range(4) ]
        #t1 = pytime.time() # DEBUG
        for l in range(iFirstOp,<INT>intarray.size()): #during loop, both prop1 & prop2 are alloc'd
            #print "begin acton %d: %.2fs since last, %.2fs elapsed" % (l-2,pytime.time()-t1,pytime.time()-t0) # DEBUG
            #t1 = pytime.time() #DEBUG

            c_opreps[intarray[l]].acton(prop1,prop2)
            #REMOVE print(" -> after ", intarray[l], " mem bytes = ",process.memory_info().rss)

            #print " post-act prop2:"; print [ prop2._dataptr[t] for t in range(4) ]
            #print("Acton %d (oprep index %d): %.3fs" % (l, intarray[l], pytime.time() - t1))
            tprop = prop1; prop1 = prop2; prop2 = tprop # swap prop1 <-> prop2
        final_state = prop1 # output = prop1 (after swap from loop above)
        # Note: prop2 is the other alloc'd state and this maintains invariant
        #print " final:"; print [ final_state._dataptr[t] for t in range(4) ]

        #print "begin prob comps: %.2fs since last, %.2fs elapsed" % (pytime.time()-t1, pytime.time()-t0) # DEBUG
        final_indices = final_indices_per_circuit[i]
        elabel_indices = elabel_indices_per_circuit[i]
        #print("Op actons done - computing %d probs" % elabel_indices.size());t1 = pytime.time() # DEBUG

        precomp_state = prop2  # used as cache/scratch space
        precomp_id = 0  # this should be a number that is *never* a Python id()
        for j in range(<INT>elabel_indices.size()):
            #print("Erep prob %d of %d: elapsed = %.2fs" % (j, elabel_indices.size(), pytime.time() - t1))
            #OLD: array_to_fill[ final_indices[j] ] = c_ereps[elabel_indices[j]].probability(final_state) #outcome probability
            array_to_fill[ final_indices[j] ] = c_ereps[elabel_indices[j]].probability_using_cache(final_state, precomp_state, precomp_id) #outcome probability

        if icache != -1:
            deref(prho_cache)[icache] = final_state # store this state in the cache
        else: # our 2nd state was pulled from the shelf before; return it
            shelved = final_state
            final_state = NULL
        #print "%d of %d (i=%d,istart=%d,remlen=%d): %.1fs" % (k, c_layout_atom.size(), i, istart,
        #                                                      intarray.size()-3, pytime.time()-t0)
        #print("mapfill_probs_block: %d of %d: %.1fs" % (k, c_layout_atom.size(), pytime.time()-t0))

    #delete our temp states
    del prop2
    del shelved


def mapfill_dprobs_atom(fwdsim,
                        np.ndarray[double, ndim=2] array_to_fill,
                        dest_indices,
                        dest_param_indices,
                        layout_atom, param_indices, resource_alloc, double eps):

    #cdef double eps = 1e-7

    if param_indices is None:
        param_indices = list(range(fwdsim.model.num_params))
    if dest_param_indices is None:
        dest_param_indices = list(range(_slct.length(param_indices)))

    param_indices = _slct.to_array(param_indices)
    dest_param_indices = _slct.to_array(dest_param_indices)

    #Get (extension-type) representation objects
    # NOTE: the circuit_layer_operator(lbl) functions cache the returned operation
    # inside fwdsim.model's opcache.  This speeds up future calls, but
    # more importantly causes fwdsim.model.from_vector to be aware of these operations and to
    # re-initialize them with updated parameter vectors as is necessary for the finite difference loop.
    rho_lookup = { lbl:i for i,lbl in enumerate(layout_atom.rho_labels) } # rho labels -> ints for faster lookup
    rhoreps = { i: fwdsim.model._circuit_layer_operator(rholbl, 'prep')._rep for rholbl,i in rho_lookup.items() }
    operation_lookup = { lbl:i for i,lbl in enumerate(layout_atom.op_labels) } # operation labels -> ints for faster lookup
    operationreps = { i:fwdsim.model._circuit_layer_operator(lbl, 'op')._rep for lbl,i in operation_lookup.items() }
    ereps = [fwdsim.model._circuit_layer_operator(elbl, 'povm')._rep for elbl in layout_atom.full_effect_labels]  # cache these in future

    # convert to C-mode:  evaltree, operation_lookup, operationreps
    cdef vector[vector[INT]] c_layout_atom = convert_maplayout(layout_atom, operation_lookup, rho_lookup)
    cdef vector[StateCRep*] c_rhos = convert_rhoreps(rhoreps)
    cdef vector[EffectCRep*] c_ereps = convert_ereps(ereps)
    cdef vector[OpCRep*] c_opreps = convert_opreps(operationreps)

    # create rho_cache = vector of StateCReps
    #print "DB: creating rho_cache of size %d * %g GB => %g GB" % \
    #   (layout_atom.cache_size, 8.0 * fwdsim.model.dim / 1024.0**3, layout_atom.cache_size * 8.0 * fwdsim.model.dim / 1024.0**3)
    cdef vector[StateCRep*] rho_cache = create_rhocache(layout_atom.cache_size, fwdsim.model.dim)

    cdef vector[vector[INT]] elabel_indices_per_circuit = convert_dict_of_intlists(layout_atom.elbl_indices_by_expcircuit)
    cdef vector[vector[INT]] final_indices_per_circuit = convert_dict_of_intlists(layout_atom.elindices_by_expcircuit)

    orig_vec = fwdsim.model.to_vector().copy()
    fwdsim.model.from_vector(orig_vec, close=False)  # ensure we call with close=False first

    nEls = layout_atom.num_elements
    probs = np.empty(nEls, 'd') #must be contiguous!
    probs2 = np.empty(nEls, 'd') #must be contiguous!

    #if resource_alloc.comm_rank == 0:
    #    print("MAPFILL DPROBS ATOM 1"); t=pytime.time(); t0=pytime.time()
    dm_mapfill_probs(probs, c_layout_atom, c_opreps, c_rhos, c_ereps, &rho_cache,
                     elabel_indices_per_circuit, final_indices_per_circuit, fwdsim.model.dim)
    #if resource_alloc.comm_rank == 0:
    #    print("MAPFILL DPROBS ATOM 2 %.3fs" % (pytime.time() - t)); t=pytime.time()

    shared_mem_leader = resource_alloc.is_host_leader

    #Get a map from global parameter indices to the desired
    # final index within array_to_fill
    iParamToFinal = {i: dest_index for i, dest_index in zip(param_indices, dest_param_indices)}

    for i in range(fwdsim.model.num_params):
        #print("dprobs cache %d of %d" % (i,self.Np))
        if i in iParamToFinal:
            #if resource_alloc.comm_rank == 0:
            #    print("MAPFILL DPROBS ATOM 3 (i=%d) %.3fs elapssed=%.1fs" % (i, pytime.time() - t, pytime.time() - t0)); t=pytime.time()
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            fwdsim.model.from_vector(vec, close=True)
            #Note: dm_mapfill_probs could have taken a resource_alloc to employ multiple cpus to do computation.
            # If probs2 were shared mem (seems not benefit to this?) it would need to only update `probs2` *if*
            # it were the host leader.
            if shared_mem_leader:  # don't fill assumed-shared array-to_fill on non-mem-leaders
                dm_mapfill_probs(probs2, c_layout_atom, c_opreps, c_rhos, c_ereps, &rho_cache,
                                 elabel_indices_per_circuit, final_indices_per_circuit, fwdsim.model.dim)
                #_fas(array_to_fill, [dest_indices, iFinal], (probs2 - probs) / eps)  # I don't think this is needed
                array_to_fill[dest_indices, iFinal] = (probs2 - probs) / eps

    #if resource_alloc.comm_rank == 0:
    #    print("MAPFILL DPROBS ATOM 4 elapsed=%.1fs" % (pytime.time() - t0))
    fwdsim.model.from_vector(orig_vec, close=True)
    free_rhocache(rho_cache)  #delete cache entries


cdef double TDchi2_obj_fn(double p, double f, double n_i, double n, double omitted_p, double min_prob_clip_for_weighting, double extra):
    cdef double cp, v, omitted_cp
    cp = p if p > min_prob_clip_for_weighting else min_prob_clip_for_weighting
    cp = cp if cp < 1 - min_prob_clip_for_weighting else 1 - min_prob_clip_for_weighting
    v = (p - f) * sqrt(n / cp)

    if omitted_p != 0.0:
        # if this is the *last* outcome at this time then account for any omitted probability
        if omitted_p < min_prob_clip_for_weighting:        omitted_cp = min_prob_clip_for_weighting
        elif omitted_p > 1 - min_prob_clip_for_weighting:  omitted_cp = 1 - min_prob_clip_for_weighting
        else:                                              omitted_cp = omitted_p
        v = sqrt(v*v + n * omitted_p*omitted_p / omitted_cp)
    return v  # sqrt(the objective function term)  (the qty stored in cache)

def mapfill_TDchi2_terms(fwdsim, array_to_fill, dest_indices, num_outcomes, layout_atom, dataset_rows,
                            min_prob_clip_for_weighting, prob_clip_interval, comm, outcomes_cache):
    mapfill_TDterms(fwdsim, "chi2", array_to_fill, dest_indices, num_outcomes, layout_atom,
                       dataset_rows, comm, outcomes_cache, min_prob_clip_for_weighting, 0.0)


cdef double TDloglpp_obj_fn(double p, double f, double n_i, double n, double omitted_p, double min_p, double a):
    cdef double freq_term, S, S2, v, tmp
    cdef double pos_p = max(p, min_p)

    if n_i != 0.0:
        freq_term = n_i * (log(f) - 1.0)
    else:
        freq_term = 0.0

    S = -n_i / min_p + n
    S2 = 0.5 * n_i / (min_p*min_p)
    v = freq_term + -n_i * log(pos_p) + n * pos_p  # dims K x M (K = nSpamLabels, M = nCircuits)

    # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
    v = max(v, 0)

    # quadratic extrapolation of logl at min_p for probabilities < min_p
    if p < min_p:
        tmp = (p - min_p)
        v = v + S * tmp + S2 * tmp * tmp

    if n_i == 0.0:
        if p >= a:
            v = n * p
        else:
            v = n * ((-1.0 / (3 * a*a)) * p*p*p + p*p / a + a / 3.0)
    # special handling for f == 0 terms
    # using quadratic rounding of function with minimum: max(0,(a-p)^2)/(2a) + p

    if omitted_p != 0.0:
        # if this is the *last* outcome at this time then account for any omitted probability
        v += n * omitted_p if omitted_p >= a else \
            n * ((-1.0 / (3 * a*a)) * omitted_p*omitted_p*omitted_p + omitted_p*omitted_p / a + a / 3.0)

    return v  # objective function term (the qty stored in cache)


def mapfill_TDloglpp_terms(fwdsim, array_to_fill, dest_indices, num_outcomes, layout_atom, dataset_rows,
                              min_prob_clip, radius, prob_clip_interval, comm, outcomes_cache):
    mapfill_TDterms(fwdsim, "logl", array_to_fill, dest_indices, num_outcomes, layout_atom,
                       dataset_rows, comm, outcomes_cache, min_prob_clip, radius)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mapfill_TDterms(fwdsim, objective, array_to_fill, dest_indices, num_outcomes,
                    layout_atom, dataset_rows, comm, outcomes_cache, double fnarg1, double fnarg2):

    cdef INT i, j, k, l, kinit, nTotOutcomes
    cdef double cur_probtotal, t, t0, n_i, n, N  # note: n, N can be a floats!
    cdef TD_obj_fn objfn
    if objective == "chi2":
        objfn = TDchi2_obj_fn
    else:
        objfn = TDloglpp_obj_fn

    array_to_fill[dest_indices] = 0.0  # reset destination (we sum into it)
    dest_indices = _slct.to_array(dest_indices)  # make sure this is an array and not a slice

    cdef INT cacheSize = layout_atom.cache_size
    #cdef np.ndarray ret = np.zeros((len(layout_atom), len(elabels)), 'd')  # zeros so we can just add contributions below
    #rhoVec, EVecs = fwdsim._rho_es_from_labels(rholabel, elabels)
    EVecs = {i: fwdsim.model._circuit_layer_operator(elbl, 'povm') for i, elbl in enumerate(layout_atom.full_effect_labels)}

    #elabels_as_outcomes = [(_ot.e_label_to_outcome(e),) for e in layout_atom.full_effect_labels]
    #outcome_to_elabel_index = {outcome: i for i, outcome in enumerate(elabels_as_outcomes)}

    local_repcache = {}

    #comm is currently ignored
    #TODO: if layout_atom is split, distribute among processors
    for iDest, iStart, remainder, iCache in layout_atom.table.contents:
        remainder = remainder.circuit_without_povm.layertup
        rholabel = remainder[0]; remainder = remainder[1:]
        rhoVec = fwdsim.model._circuit_layer_operator(rholabel, 'prep')

        #print("DB: ",iDest,iStart,remainder)
        datarow = dataset_rows[iDest]
        nTotOutcomes = num_outcomes[iDest]
        N = 0; nOutcomes = 0

        if outcomes_cache is not None:  # calling dataset.outcomes can be a bottleneck
            # need to base cache on this b/c same iDest for different atoms corresponds to different circuits!
            iOrig = layout_atom.orig_indices_by_expcircuit[iDest]
            if iOrig in outcomes_cache:
                outcomes = outcomes_cache[iOrig]
            else:
                outcomes = datarow.outcomes
                outcomes_cache[iOrig] = outcomes
        else:
            outcomes = datarow.outcomes

        datarow_time = {i: tm for i, tm in enumerate(datarow.time)}  # dict for speed
        datarow_reps = {i: repcnt for i, repcnt in enumerate(datarow.reps)}  # dict for speed
        datarow_outcomes = {i: outcome for i, outcome in enumerate(outcomes)}  # dict for speed

        elbl_indices = layout_atom.elbl_indices_by_expcircuit[iDest]
        outcomes = layout_atom.outcomes_by_expcircuit[iDest]
        outcome_to_elbl_index = {outcome: elbl_index for outcome, elbl_index in zip(outcomes, elbl_indices)}
        #FUTURE: construct outcome_to_elbl_index dict in layout_atom, so we don't construct it here?
        final_indices = [dest_indices[j] for j in layout_atom.elindices_by_expcircuit[iDest]]
        elbl_to_final_index = {elbl_index: final_index for elbl_index, final_index in zip(elbl_indices, final_indices)}
        model = fwdsim.model  # just for faster inner loop performance

        #opcache = fwdsim.model._opcaches['layers']  # use knowledge of internals for faster innerloop performance

        n = len(datarow_reps) # == len(datarow.time)
        kinit = 0
        while kinit < n:
            #Process all outcomes of this datarow occuring at a single time, t0
            t0 = datarow_time[kinit]

            #Compute N, nOutcomes for t0
            N = 0; k = kinit
            while k < n and datarow_time[k] == t0:
                N += datarow_reps[k]
                k += 1
            nOutcomes = k - kinit

            #Compute each outcome's contribution
            cur_probtotal = 0.0
            for l in range(kinit, k):
                t = t0
                rhoVec.set_time(t)
                rho = rhoVec._rep
                t += rholabel.time

                n_i = datarow_reps[l]
                outcome = datarow_outcomes[l]

                for gl in remainder:
                    if (gl,t) in local_repcache:  # Note: this could cache a *lot* of reps - add flag to disable?
                        op_rep = local_repcache[(gl,t)]
                    else:
                        op = model._circuit_layer_operator(gl, 'op')  # add explicit cache check (would increase performance)
                        op.set_time(t); op_rep = op._rep
                        local_repcache[(gl, t)] = op_rep.copy()  # need to *copy* here
                    t += gl.time  # time in gate label == gate duration?
                    rho = op_rep.acton(rho)

                j = outcome_to_elbl_index[outcome]
                E = EVecs[j]; E.set_time(t)
                p = E._rep.probability(rho)  # outcome probability
                f = n_i / N
                cur_probtotal += p

                omitted_p = 1.0 - cur_probtotal if (l == k-1 and nOutcomes < nTotOutcomes) else 0.0
                # and cur_probtotal < 1.0?

                val = objfn(p, f, n_i, N, omitted_p, fnarg1, fnarg2)
                array_to_fill[elbl_to_final_index[j]] += val
                #if t0 < 20:
                #    print("DB: t0=",t0," p,f,n,N,o = ",p, f, n_i, N, omitted_p, " =>", val)

            kinit = k


def mapfill_TDdchi2_terms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes,
                             layout_atom, dataset_rows, min_prob_clip_for_weighting, prob_clip_interval,
                             wrt_slice, comm, outcomes_cache):

    def fillfn(array_to_fill, dest_indices, n_outcomes, layout_atom, dataset_rows, fill_comm, ocache):
        mapfill_TDchi2_terms(fwdsim, array_to_fill, dest_indices, n_outcomes, layout_atom, dataset_rows,
                                min_prob_clip_for_weighting, prob_clip_interval, fill_comm, ocache)

    mapfill_timedep_dterms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes,
                              layout_atom, dataset_rows, fillfn, wrt_slice, comm, outcomes_cache)


def mapfill_TDdloglpp_terms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes,
                               layout_atom, dataset_rows, min_prob_clip, radius, prob_clip_interval,
                               wrt_slice, comm, outcomes_cache):

    def fillfn(array_to_fill, dest_indices, n_outcomes, layout_atom, dataset_rows, fill_comm, ocache):
        mapfill_TDloglpp_terms(fwdsim, array_to_fill, dest_indices, n_outcomes, layout_atom, dataset_rows,
                                  min_prob_clip, radius, prob_clip_interval, fill_comm, ocache)

    mapfill_timedep_dterms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes,
                              layout_atom, dataset_rows, fillfn, wrt_slice, comm, outcomes_cache)


def mapfill_timedep_dterms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes,
                              layout_atom, dataset_rows, fillfn, wrt_slice, comm, outcomes_cache):

    cdef INT i, ii, iFinal
    cdef double eps = 1e-7  # hardcoded?

    #Compute finite difference derivatives, one parameter at a time.
    param_indices = range(fwdsim.model.num_params) if (wrt_slice is None) else _slct.indices(wrt_slice)
    #cdef INT nDerivCols = len(param_indices)  # *all*, not just locally computed ones

    #rhoVec, EVecs = fwdsim._rho_es_from_labels(rholabel, elabels)
    #cdef np.ndarray cache = np.empty((len(layout_atom), len(elabels)), 'd')
    #cdef np.ndarray dcache = np.zeros((len(layout_atom), len(elabels), nDerivCols), 'd')

    cdef INT cacheSize = layout_atom.cache_size
    cdef INT nEls = layout_atom.num_elements
    cdef np.ndarray vals = np.empty(nEls, 'd')
    cdef np.ndarray vals2 = np.empty(nEls, 'd')
    #assert(cacheSize == 0)

    orig_vec = fwdsim.model.to_vector().copy()
    fwdsim.model.from_vector(orig_vec, close=False)  # ensure we call with close=False first

    fillfn(vals, slice(0, nEls), num_outcomes, layout_atom, dataset_rows, comm, outcomes_cache)

    all_slices, my_slice, owners, subComm = \
        _mpit.distribute_slice(slice(0, len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    cdef INT st = my_slice.start  # beginning of where my_param_indices results
    # get placed into dpr_cache

    #Get a map from global parameter indices to the desired
    # final index within dpr_cache
    iParamToFinal = {i: st + ii for ii, i in enumerate(my_param_indices)}

    for i in range(fwdsim.model.num_params):
        #print("dprobs cache %d of %d" % (i,fwdsim.model.num_params))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            fwdsim.model.from_vector(vec, close=True)
            fillfn(vals2, slice(0, nEls), num_outcomes, layout_atom, dataset_rows, subComm, outcomes_cache)
            _fas(array_to_fill, [dest_indices, iFinal], (vals2 - vals) / eps)
    fwdsim.model.from_vector(orig_vec, close=True)

    #Now each processor has filled the relavant parts of dpr_cache,
    # so gather together:
    _mpit.gather_slices(all_slices, owners, array_to_fill, [], axes=1, comm=comm)

    # DEBUG LINE USED FOR MONITORING N-QUBIT GST TESTS
    #print("DEBUG TIME: dpr_cache(Np=%d, dim=%d, cachesize=%d, treesize=%d, napplies=%d) in %gs" %
    #      (fwdsim.model.num_params, fwdsim.model.dim, cacheSize, len(layout_atom), layout_atom.get_num_applies(), _time.time()-tStart)) #DEBUG


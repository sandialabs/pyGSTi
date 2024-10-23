"""Defines generic Python-version of map forward simuator calculations"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.tools import mpitools as _mpit
from pygsti.tools import sharedmemtools as _smt
from pygsti.tools import slicetools as _slct
from pygsti.tools.matrixtools import _fas


def propagate_staterep(staterep, operationreps):
    ret = staterep.actionable_staterep()
    for oprep in operationreps:
        ret = oprep.acton(ret)
    return ret


def mapfill_probs_atom(fwdsim, mx_to_fill, dest_indices, layout_atom, resource_alloc):

    # The required ending condition is that array_to_fill on each processor has been filled.  But if
    # memory is being shared and resource_alloc contains multiple processors on a single host, we only
    # want *one* (the rank=0) processor to perform the computation, since array_to_fill will be
    # shared memory that we don't want to have muliple procs using simultaneously to compute the
    # same thing.  Thus, we carefully guard any shared mem updates/usage
    # using "if shared_mem_leader" (and barriers, if needed) below.
    shared_mem_leader = resource_alloc.is_host_leader if (resource_alloc is not None) else True

    dest_indices = _slct.to_array(dest_indices)  # make sure this is an array and not a slice
    cacheSize = layout_atom.cache_size

    #Create rhoCache
    rho_cache = [None] * cacheSize  # so we can store (s,p) tuples in cache

    #Get operationreps and ereps now so we don't make unnecessary ._rep references
    rhoreps = {rholbl: fwdsim.model._circuit_layer_operator(rholbl, 'prep')._rep for rholbl in layout_atom.rho_labels}
    operationreps = {gl: fwdsim.model._circuit_layer_operator(gl, 'op')._rep for gl in layout_atom.op_labels}
    povmreps = {plbl: fwdsim.model._circuit_layer_operator(plbl, 'povm')._rep for plbl in layout_atom.povm_labels}
    if any([(povmrep is None) for povmrep in povmreps.values()]):
        effectreps = {i: fwdsim.model._circuit_layer_operator(Elbl, 'povm')._rep
                      for i, Elbl in enumerate(layout_atom.full_effect_labels)}  # cache these in future
    else:
        effectreps = None  # not needed, as we use povm reps directly

    #TODO: if layout_atom is split, distribute somehow among processors(?) instead of punting for all but rank-0 above
    for iDest, iStart, remainder, iCache in layout_atom.table.contents:

        if iStart is None:  # then first element of remainder is a state prep label
            rholabel = remainder[0]
            init_state = rhoreps[rholabel]
            remainder = remainder[1:]
        else:
            init_state = rho_cache[iStart]  # [:,None]

        #OLD final_state = self.propagate_state(init_state, remainder)
        final_state = propagate_staterep(init_state, [operationreps[gl] for gl in remainder])
        if iCache is not None: rho_cache[iCache] = final_state  # [:,0] #store this state in the cache

        final_indices = [dest_indices[j] for j in layout_atom.elindices_by_expcircuit[iDest]]

        if effectreps is None:
            povm_lbl, *effect_labels = layout_atom.povm_and_elbls_by_expcircuit[iDest]

            if shared_mem_leader:
                mx_to_fill[final_indices] = povmreps[povm_lbl].probabilities(final_state, None, effect_labels)
        else:
            ereps = [effectreps[j] for j in layout_atom.elbl_indices_by_expcircuit[iDest]]
            #print(ereps)
            if shared_mem_leader:
                for j, erep in zip(final_indices, ereps):
                    mx_to_fill[j] = erep.probability(final_state)  # outcome probability
    #raise Exception
#Version of the probability calculation that updates circuit probabilities conditionally based on
#Whether the circuit is sensitive to the parameter. If not we leave that circuit alone.
def cond_update_probs_atom(fwdsim, mx_to_fill, dest_indices, layout_atom, param_index, resource_alloc):

    # The required ending condition is that array_to_fill on each processor has been filled.  But if
    # memory is being shared and resource_alloc contains multiple processors on a single host, we only
    # want *one* (the rank=0) processor to perform the computation, since array_to_fill will be
    # shared memory that we don't want to have muliple procs using simultaneously to compute the
    # same thing.  Thus, we carefully guard any shared mem updates/usage
    # using "if shared_mem_leader" (and barriers, if needed) below.
    shared_mem_leader = resource_alloc.is_host_leader if (resource_alloc is not None) else True

    dest_indices = _slct.to_array(dest_indices)  # make sure this is an array and not a slice
    cacheSize = layout_atom.jac_table.cache_size_by_parameter[param_index]

    #Create rhoCache
    rho_cache = [None] * cacheSize  # so we can store (s,p) tuples in cache

    #Get operationreps and ereps now so we don't make unnecessary ._rep references
    rhoreps = {rholbl: fwdsim.model._circuit_layer_operator(rholbl, 'prep')._rep for rholbl in layout_atom.rho_labels}
    operationreps = {gl: fwdsim.model._circuit_layer_operator(gl, 'op')._rep for gl in layout_atom.op_labels}
    povmreps = {plbl: fwdsim.model._circuit_layer_operator(plbl, 'povm')._rep for plbl in layout_atom.povm_labels}
    if any([(povmrep is None) for povmrep in povmreps.values()]):
        effectreps = {i: fwdsim.model._circuit_layer_operator(Elbl, 'povm')._rep
                      for i, Elbl in enumerate(layout_atom.full_effect_labels)}  # cache these in future
    else:
        effectreps = None  # not needed, as we use povm reps directly


    #TODO: if layout_atom is split, distribute somehow among processors(?) instead of punting for all but rank-0 above

    for iDest, iStart, remainder, iCache in layout_atom.jac_table.contents_by_parameter[param_index]:
 
        if iStart is None:  # then first element of remainder is a state prep label
            rholabel = remainder[0]
            init_state = rhoreps[rholabel]
            remainder = remainder[1:]
        else:
            init_state = rho_cache[iStart]  # [:,None]

        final_state = propagate_staterep(init_state, [operationreps[gl] for gl in remainder])
        if iCache is not None: rho_cache[iCache] = final_state  # [:,0] #store this state in the cache

        final_indices = [dest_indices[j] for j in layout_atom.elindices_by_expcircuit[iDest]]

        if effectreps is None:
            povm_lbl, *effect_labels = layout_atom.povm_and_elbls_by_expcircuit[iDest]

            if shared_mem_leader:
                mx_to_fill[final_indices] = povmreps[povm_lbl].probabilities(final_state, None, effect_labels)
        else:
            ereps = [effectreps[j] for j in layout_atom.elbl_indices_by_expcircuit[iDest]]
            if shared_mem_leader:
                for j, erep in zip(final_indices, ereps):
                    mx_to_fill[j] = erep.probability(final_state)  # outcome probability


def mapfill_dprobs_atom(fwdsim, mx_to_fill, dest_indices, dest_param_indices, layout_atom, param_indices,
                        resource_alloc, eps):

    num_params = fwdsim.model.num_params

    if param_indices is None:
        param_indices = list(range(num_params))
    if dest_param_indices is None:
        dest_param_indices = list(range(_slct.length(param_indices)))

    param_indices = _slct.to_array(param_indices)
    dest_param_indices = _slct.to_array(dest_param_indices)

    #Get a map from global parameter indices to the desired
    # final index within mx_to_fill (fpoffset = final parameter offset)
    iParamToFinal = {i: dest_index for i, dest_index in zip(param_indices, dest_param_indices)}

    orig_vec = fwdsim.model.to_vector().copy()
    fwdsim.model.from_vector(orig_vec, close=False)  # ensure we call with close=False first

    #Note: no real need for using shared memory here except so that we can pass
    # `resource_alloc` to mapfill_probs_block and have it potentially use multiple procs.
    nEls = layout_atom.num_elements
    probs, shm = _smt.create_shared_ndarray(resource_alloc, (nEls,), 'd', memory_tracker=None)
    probs2, shm2 = _smt.create_shared_ndarray(resource_alloc, (nEls,), 'd', memory_tracker=None)
    #probs2_test, shm2_test = _smt.create_shared_ndarray(resource_alloc, (nEls,), 'd', memory_tracker=None)
    
    #mx_to_fill_test = mx_to_fill.copy()

    mapfill_probs_atom(fwdsim, probs, slice(0, nEls), layout_atom, resource_alloc)  # probs != shared

    #Split off the first finite difference step, as the pattern I want in the loop with each step
    #is to simultaneously undo the previous update and apply the new one.
    if len(param_indices)>0:
        probs2[:] = probs[:]
        first_param_idx = param_indices[0]
        iFinal = iParamToFinal[first_param_idx]
        fwdsim.model.set_parameter_value(first_param_idx, orig_vec[first_param_idx]+eps)
        #mapfill_probs_atom(fwdsim, probs2, slice(0, nEls), layout_atom, resource_alloc)
        cond_update_probs_atom(fwdsim, probs2, slice(0, nEls), layout_atom, first_param_idx, resource_alloc)
        #assert _np.linalg.norm(probs2_test-probs2) < 1e-10
        #print(f'{_np.linalg.norm(probs2_test-probs2)=}')
        _fas(mx_to_fill, [dest_indices, iFinal], (probs2 - probs) / eps)


    for i in range(1, len(param_indices)):
        probs2[:] = probs[:]
        iFinal = iParamToFinal[param_indices[i]]
        fwdsim.model.set_parameter_values([param_indices[i-1], param_indices[i]], 
                                          [orig_vec[param_indices[i-1]], orig_vec[param_indices[i]]+eps])
        #mapfill_probs_atom(fwdsim, probs2, slice(0, nEls), layout_atom, resource_alloc)
        cond_update_probs_atom(fwdsim, probs2, slice(0, nEls), layout_atom, param_indices[i], resource_alloc)
        #assert _np.linalg.norm(probs2_test-probs2) < 1e-10
        #print(f'{_np.linalg.norm(probs2_test-probs2)=}')
        _fas(mx_to_fill, [dest_indices, iFinal], (probs2 - probs) / eps)

    #reset the final model parameter we changed to it's original value.
    fwdsim.model.set_parameter_value(param_indices[-1], orig_vec[param_indices[-1]])

    _smt.cleanup_shared_ndarray(shm)
    _smt.cleanup_shared_ndarray(shm2)
    #_smt.cleanup_shared_ndarray(shm2_test)


def mapfill_TDchi2_terms(fwdsim, array_to_fill, dest_indices, num_outcomes, layout_atom, dataset_rows,
                         min_prob_clip_for_weighting, prob_clip_interval, comm, outcomes_cache):

    def obj_fn(p, f, n_i, n, omitted_p):
        cp = _np.clip(p, min_prob_clip_for_weighting, 1 - min_prob_clip_for_weighting)
        v = (p - f) * _np.sqrt(n / cp)

        if omitted_p != 0:
            # if this is the *last* outcome at this time then account for any omitted probability
            omitted_cp = _np.clip(omitted_p, min_prob_clip_for_weighting, 1 - min_prob_clip_for_weighting)
            v = _np.sqrt(v**2 + n * omitted_p**2 / omitted_cp)
        return v  # sqrt(the objective function term)  (the qty stored in cache)

    return mapfill_TDterms(fwdsim, obj_fn, array_to_fill, dest_indices, num_outcomes, layout_atom,
                           dataset_rows, comm, outcomes_cache)


def mapfill_TDloglpp_terms(fwdsim, array_to_fill, dest_indices, num_outcomes, layout_atom, dataset_rows,
                           min_prob_clip, radius, prob_clip_interval, comm, outcomes_cache):

    min_p = min_prob_clip; a = radius

    def obj_fn(p, f, n_i, n, omitted_p):
        pos_p = max(p, min_p)

        if n_i != 0:
            freq_term = n_i * (_np.log(f) - 1.0)
        else:
            freq_term = 0.0
        S = -n_i / min_p + n
        S2 = 0.5 * n_i / (min_p**2)
        v = freq_term + -n_i * _np.log(pos_p) + n * pos_p  # dims K x M (K = nSpamLabels, M = n_circuits)

        # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
        v = max(v, 0)

        # quadratic extrapolation of logl at min_p for probabilities < min_p
        if p < min_p:
            v = v + S * (p - min_p) + S2 * (p - min_p)**2

        if n_i == 0:
            if p >= a:
                v = n * p
            else:
                v = n * ((-1.0 / (3 * a**2)) * p**3 + p**2 / a + a / 3.0)
        # special handling for f == 0 terms
        # using quadratic rounding of function with minimum: max(0,(a-p)^2)/(2a) + p

        if omitted_p != 0.0:
            # if this is the *last* outcome at this time then account for any omitted probability
            v += n * omitted_p if omitted_p >= a else \
                n * ((-1.0 / (3 * a**2)) * omitted_p**3 + omitted_p**2 / a + a / 3.0)

        return v  # objective function term (the qty stored in cache)

    return mapfill_TDterms(fwdsim, obj_fn, array_to_fill, dest_indices, num_outcomes, layout_atom,
                           dataset_rows, comm, outcomes_cache)


def mapfill_TDterms(fwdsim, objfn, array_to_fill, dest_indices, num_outcomes, layout_atom, dataset_rows, comm,
                    outcomes_cache):

    dest_indices = _slct.to_array(dest_indices)  # make sure this is an array and not a slice
    cacheSize = layout_atom.cache_size

    EVecs = [fwdsim.model._circuit_layer_operator(elbl, 'povm') for elbl in layout_atom.full_effect_labels]

    assert(cacheSize == 0)  # so all elements have None as start and remainder[0] is a prep label
    #if clip_to is not None:
    #    _np.clip(array_to_fill, clip_to[0], clip_to[1], out=array_to_fill)  # in-place clip

    array_to_fill[dest_indices] = 0.0  # reset destination (we sum into it)

    #comm is currently ignored
    #TODO: if layout_atom is split, distribute among processors
    for iDest, iStart, remainder, iCache in layout_atom.table.contents:
        remainder = remainder.circuit_without_povm.layertup
        assert(iStart is None), "Cannot use trees with max-cache-size > 0 when performing time-dependent calcs!"
        rholabel = remainder[0]; remainder = remainder[1:]
        rhoVec = fwdsim.model._circuit_layer_operator(rholabel, 'prep')
        datarow = dataset_rows[iDest]
        nTotOutcomes = num_outcomes[iDest]

        totalCnts = {}  # TODO defaultdict?
        lastInds = {}; outcome_cnts = {}

        # consolidate multiple outcomes that occur at same time? or sort?
        #CHECK - should this loop filter only outcomes relevant to this expanded circuit (like below)?
        for k, (t0, Nreps) in enumerate(zip(datarow.time, datarow.reps)):
            if t0 in totalCnts:
                totalCnts[t0] += Nreps; outcome_cnts[t0] += 1
            else:
                totalCnts[t0] = Nreps; outcome_cnts[t0] = 1
            lastInds[t0] = k

        elbl_indices = layout_atom.elbl_indices_by_expcircuit[iDest]
        outcomes = layout_atom.outcomes_by_expcircuit[iDest]
        outcome_to_elbl_index = {outcome: elbl_index for outcome, elbl_index in zip(outcomes, elbl_indices)}
        #FUTURE: construct outcome_to_elbl_index dict in layout_atom, so we don't construct it here?
        final_indices = [dest_indices[j] for j in layout_atom.elindices_by_expcircuit[iDest]]
        elbl_to_final_index = {elbl_index: final_index for elbl_index, final_index in zip(elbl_indices, final_indices)}

        cur_probtotal = 0; last_t = 0
        # consolidate multiple outcomes that occur at same time? or sort?
        for k, (t0, Nreps, outcome) in enumerate(zip(datarow.time, datarow.reps, datarow.outcomes)):
            if outcome not in outcome_to_elbl_index:
                continue  # skip datarow outcomes not for this expanded circuit

            t = t0
            rhoVec.set_time(t)
            rho = rhoVec._rep.actionable_staterep()
            t += rholabel.time

            for gl in remainder:
                op = fwdsim.model._circuit_layer_operator(gl, 'op')
                op.set_time(t); t += gl.time  # time in gate label == gate duration?
                rho = op._rep.acton(rho)

            j = outcome_to_elbl_index[outcome]
            E = EVecs[j]; E.set_time(t)
            p = E._rep.probability(rho)  # outcome probability
            N = totalCnts[t0]
            f = Nreps / N

            if t0 == last_t:
                cur_probtotal += p
            else:
                last_t = t0
                cur_probtotal = p

            omitted_p = 1.0 - cur_probtotal if (lastInds[t0] == k and outcome_cnts[t0] < nTotOutcomes) else 0.0
            # and cur_probtotal < 1.0?

            array_to_fill[elbl_to_final_index[j]] += objfn(p, f, Nreps, N, omitted_p)


def mapfill_TDdchi2_terms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes, layout_atom,
                          dataset_rows, min_prob_clip_for_weighting, prob_clip_interval, wrt_slice,
                          comm, outcomes_cache):

    def fillfn(array_to_fill, dest_indices, n_outcomes, layout_atom, dataset_rows, fill_comm):
        mapfill_TDchi2_terms(fwdsim, array_to_fill, dest_indices, n_outcomes,
                             layout_atom, dataset_rows, min_prob_clip_for_weighting,
                             prob_clip_interval, fill_comm, outcomes_cache)

    return mapfill_timedep_dterms(fwdsim, array_to_fill, dest_indices, dest_param_indices,
                                  num_outcomes, layout_atom, dataset_rows, fillfn, wrt_slice, comm)


def mapfill_TDdloglpp_terms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes,
                            layout_atom, dataset_rows, min_prob_clip, radius, prob_clip_interval,
                            wrt_slice, comm, outcomes_cache):

    def fillfn(array_to_fill, dest_indices, n_outcomes, layout_atom, dataset_rows, fill_comm):
        mapfill_TDloglpp_terms(fwdsim, array_to_fill, dest_indices, n_outcomes,
                               layout_atom, dataset_rows, min_prob_clip, radius,
                               prob_clip_interval, fill_comm, outcomes_cache)

    return mapfill_timedep_dterms(fwdsim, array_to_fill, dest_indices, dest_param_indices,
                                  num_outcomes, layout_atom, dataset_rows, fillfn, wrt_slice, comm)


def mapfill_timedep_dterms(fwdsim, array_to_fill, dest_indices, dest_param_indices, num_outcomes, layout_atom,
                           dataset_rows, fillfn, wrt_slice, comm):

    eps = 1e-7  # hardcoded?

    #Compute finite difference derivatives, one parameter at a time.
    param_indices = range(fwdsim.model.num_params) if (wrt_slice is None) else _slct.indices(wrt_slice)

    nEls = layout_atom.num_elements
    vals = _np.empty(nEls, 'd')
    vals2 = _np.empty(nEls, 'd')
    assert(layout_atom.cache_size == 0)  # so all elements have None as start and remainder[0] is a prep label

    orig_vec = fwdsim.model.to_vector().copy()
    fwdsim.model.from_vector(orig_vec, close=False)  # ensure we call with close=False first

    fillfn(vals, slice(0, nEls), num_outcomes, layout_atom, dataset_rows, comm)

    all_slices, my_slice, owners, subComm = \
        _mpit.distribute_slice(slice(0, len(param_indices)), comm)

    my_param_indices = param_indices[my_slice]
    st = my_slice.start  # beginning of where my_param_indices results
    # get placed into dpr_cache

    #Get a map from global parameter indices to the desired
    # final index within dpr_cache
    iParamToFinal = {i: st + ii for ii, i in enumerate(my_param_indices)}

    for i in range(fwdsim.model.num_params):
        # print("dprobs cache %d of %d" % (i,fwdsim.model.num_params))
        if i in iParamToFinal:
            iFinal = iParamToFinal[i]
            vec = orig_vec.copy(); vec[i] += eps
            fwdsim.model.from_vector(vec, close=True)
            fillfn(vals2, slice(0, nEls), num_outcomes, layout_atom, dataset_rows, subComm)
            _fas(array_to_fill, [dest_indices, iFinal], (vals2 - vals) / eps)

    fwdsim.model.from_vector(orig_vec, close=True)

    #Now each processor has filled the relavant parts of dpr_cache,
    # so gather together:
    _mpit.gather_slices(all_slices, owners, array_to_fill, [], axes=1, comm=comm)

    # DEBUG LINE USED FOR MONITORION N-QUBIT GST TESTS
    #print("DEBUG TIME: dpr_cache(Np=%d, dim=%d, cachesize=%d, treesize=%d, napplies=%d) in %gs" %
    #      (fwdsim.model.num_params, fwdsim.model.dim, cache_size, len(layout_atom),
    #       layout_atom.num_applies(), _time.time()-tStart)) #DEBUG

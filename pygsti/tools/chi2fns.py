""" Chi-squared and related functions """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from . import listtools as _lt
from . import slicetools as _slct


def chi2_terms(model, dataset, circuits=None,
               min_prob_clip_for_weighting=1e-4, clip_to=None,
               use_freq_weighted_chi_sq=False, check=False,
               mem_limit=None, op_label_aliases=None,
               evaltree_cache=None, comm=None, smartc=None):
    """
    Computes the chi^2 contributions from a set of operation sequences.

    This function returns the same value as :func:`chi2` with
    `return_gradient=False` and `return_hessian=False`, except the
    contributions from different operation sequences and spam labels is
    not summed but returned as an array.

    Parameters
    ----------
    This function takes the same arguments as :func:`chi2` except
    for `return_gradient` and `return_hessian` (which aren't supported yet).

    Returns
    -------
    chi2 : numpy.ndarray
        Array of length either `len(circuit_list)` or `len(dataset.keys())`.
        Values are the chi2 contributions of the corresponding gate
        string aggregated over outcomes.
    """
    def smart(fn, *args, **kwargs):
        if smartc:
            return smartc.cached_compute(fn, args, kwargs)[1]
        else:
            if '_filledarrays' in kwargs: del kwargs['_filledarrays']
            return fn(*args, **kwargs)

    if use_freq_weighted_chi_sq:
        raise ValueError("frequency weighted chi2 is not implemented yet.")

    if circuits is None:
        circuits = list(dataset.keys())

    if evaltree_cache and 'evTree' in evaltree_cache:
        evTree = evaltree_cache['evTree']
        lookup = evaltree_cache['lookup']
        outcomes_lookup = evaltree_cache['outcomes_lookup']
    else:
        # Note: simplify_circuits doesn't support aliased dataset (yet)
        dstree = dataset if (op_label_aliases is None) else None
        evTree, _, _, lookup, outcomes_lookup = \
            smart(model.bulk_evaltree_from_resources,
                  circuits, None, mem_limit, "deriv", ['bulk_fill_probs'], dstree)

        #Fill cache dict if one was given
        if evaltree_cache is not None:
            evaltree_cache['evTree'] = evTree
            evaltree_cache['lookup'] = lookup
            evaltree_cache['outcomes_lookup'] = outcomes_lookup

    #Memory allocation
    nEls = evTree.num_final_elements()
    C = 1.0 / 1024.0**3

    #  Estimate & check persistent memory (from allocs directly below)
    persistentMem = 8 * (3 * nEls)  # in bytes
    if mem_limit is not None and mem_limit < persistentMem:
        raise MemoryError("Chi2 Memory limit (%g GB) is " % (mem_limit * C)
                          + "< memory required to hold final results (%g GB)"
                          % (persistentMem * C))

    #  Allocate peristent memory
    N = _np.empty(nEls, 'd')
    f = _np.empty(nEls, 'd')
    probs = _np.empty(nEls, 'd')

    dsCircuits = _lt.apply_aliases_to_circuit_list(circuits, op_label_aliases)
    for (i, opStr) in enumerate(dsCircuits):
        N[lookup[i]] = dataset[opStr].total
        f[lookup[i]] = [dataset[opStr].fraction(x) for x in outcomes_lookup[i]]

    #Detect omitted frequences (assumed to be 0) so we can compute chi2 correctly
    firsts = []; indicesOfCircuitsWithOmittedData = []
    for i, c in enumerate(circuits):
        lklen = _slct.length(lookup[i])
        if 0 < lklen < model.get_num_outcomes(c):
            firsts.append(_slct.as_array(lookup[i])[0])
            indicesOfCircuitsWithOmittedData.append(i)
    if len(firsts) > 0:
        firsts = _np.array(firsts, 'i')
        indicesOfCircuitsWithOmittedData = _np.array(indicesOfCircuitsWithOmittedData, 'i')
    else:
        firsts = None

    smart(model.bulk_fill_probs, probs, evTree, clip_to, check, comm, _filledarrays=(0,))

    cprobs = _np.clip(probs, min_prob_clip_for_weighting, 1e10)  # effectively no upper bound
    v = N * ((probs - f)**2 / cprobs)

    #account for omitted probs (sparse data)
    if firsts is not None:
        omitted_probs = 1.0 - _np.array([_np.sum(probs[lookup[i]]) for i in indicesOfCircuitsWithOmittedData])
        clipped_oprobs = _np.clip(omitted_probs, min_prob_clip_for_weighting, 1 - min_prob_clip_for_weighting)
        v[firsts] = v[firsts] + N[firsts] * omitted_probs**2 / clipped_oprobs

    #Aggregate over outcomes:
    # v[iElement] contains all chi2 contributions - now aggregate over outcomes
    # terms[iCircuit] wiil contain chi2 contributions for each original gate
    # string (aggregated over outcomes)
    nCircuits = len(circuits)
    terms = _np.empty(nCircuits, 'd')
    for i in range(nCircuits):
        terms[i] = _np.sum(v[lookup[i]], axis=0)
    return terms


def chi2(model, dataset, circuits=None,
         return_gradient=False, return_hessian=False,
         min_prob_clip_for_weighting=1e-4, clip_to=None,
         use_freq_weighted_chi_sq=False, check=False,
         mem_limit=None, op_label_aliases=None,
         evaltree_cache=None, comm=None,
         approximate_hessian=False, smartc=None):
    """
    Computes the total chi^2 for a set of operation sequences.

    The chi^2 test statistic obtained by summing up the
    contributions of a given set of operation sequences or all
    the strings available in a dataset.  Optionally,
    the gradient and/or Hessian of chi^2 can be returned too.

    Parameters
    ----------
    model : Model
        The model used to specify the probabilities and SPAM labels

    dataset : DataSet
        The data used to specify frequencies and counts

    circuits : list of Circuits or tuples, optional
        List of operation sequences whose terms will be included in chi^2 sum.
        Default value (None) means "all strings in dataset".

    return_gradient, return_hessian : bool
        Whether to compute and return the gradient and/or Hessian of chi^2.

    min_prob_clip_for_weighting : float, optional
        defines the clipping interval for the statistical weight (see chi2fn).

    clip_to : 2-tuple, optional
        (min,max) to clip probabilities to within Model probability
        computation routines (see Model.bulk_fill_probs)

    use_freq_weighted_chi_sq : bool, optional
        Whether or not frequencies (instead of probabilities) should be used
        in statistical weight factors.

    check : bool, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    mem_limit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    evaltree_cache : dict, optional
        A dictionary which server as a cache for the computed EvalTree used
        in this computation.  If an empty dictionary is supplied, it is filled
        with cached values to speed up subsequent executions of this function
        which use the *same* `model` and `circuit_list`.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    approximate_hessian : bool, optional
        Whether, when `return_hessian == True`, an *approximate* version of
        the Hessian should be returned.  This approximation neglects
        terms proportional to the Hessian of the probabilities w.r.t. the
        gate parameters (which can take a long time to compute).  See
        `logl_approximate_hessian` for details on the analogous approximation
        for the log-likelihood Hessian.

    smartc : SmartCache, optional
        A cache object to cache & use previously cached values inside this
        function.


    Returns
    -------
    chi2 : float
        chi^2 value, equal to the sum of chi^2 terms from all specified operation sequences
    dchi2 : numpy array
        Only returned if return_gradient == True. The gradient vector of
        length nModelParams, the number of model parameters.
    d2chi2 : numpy array
        Only returned if return_hessian == True. The Hessian matrix of
        shape (nModelParams, nModelParams).
    """
    def smart(fn, *args, **kwargs):
        if smartc:
            return smartc.cached_compute(fn, args, kwargs)[1]
        else:
            if '_filledarrays' in kwargs: del kwargs['_filledarrays']
            return fn(*args, **kwargs)

    # Scratch work:
    # chi^2 = sum_i N_i*(p_i-f_i)^2 / p_i  (i over circuits & spam labels)                                                                                      # noqa
    # d(chi^2)/dx = sum_i N_i * [ 2(p_i-f_i)*dp_i/dx / p_i - (p_i-f_i)^2 / p_i^2 * dp_i/dx ]                                                                    # noqa
    #             = sum_i N_i * (p_i-f_i) / p_i * [2 - (p_i-f_i)/p_i   ] * dp_i/dx                                                                              # noqa
    #             = sum_i N_i * t_i * [2 - t_i ] * dp_i/dx     where t_i = (p_i-f_i) / p_i                                                                      # noqa
    # d2(chi^2)/dydx = sum_i N_i * [ dt_i/dy * [2 - t_i ] * dp_i/dx - t_i * dt_i/dy * dp_i/dx + t_i * [2 - t_i] * d2p_i/dydx ]                                  # noqa
    #                          where dt_i/dy = [ 1/p_i - (p_i-f_i) / p_i^2 ] * dp_i/dy                                                                          # noqa
    if use_freq_weighted_chi_sq:
        raise ValueError("frequency weighted chi2 is not implemented yet.")

    vec_gs_len = model.num_params()

    if circuits is None:
        circuits = list(dataset.keys())

    if evaltree_cache and 'evTree' in evaltree_cache:
        evTree = evaltree_cache['evTree']
        lookup = evaltree_cache['lookup']
        outcomes_lookup = evaltree_cache['outcomes_lookup']
    else:
        #OLD: evTree,lookup,outcomes_lookup = smart(model.bulk_evaltree,circuits)
        evTree, _, _, lookup, outcomes_lookup = smart(model.bulk_evaltree_from_resources,
                                                      circuits, comm, dataset=dataset)

        #Fill cache dict if one was given
        if evaltree_cache is not None:
            evaltree_cache['evTree'] = evTree
            evaltree_cache['lookup'] = lookup
            evaltree_cache['outcomes_lookup'] = outcomes_lookup

    #Memory allocation
    nEls = evTree.num_final_elements()
    ng = evTree.num_final_strings()
    ne = model.num_params(); gd = model.get_dimension()
    C = 1.0 / 1024.0**3

    #  Estimate & check persistent memory (from allocs directly below)
    persistentMem = 8 * (3 * nEls)  # in bytes
    compute_hprobs = bool(return_hessian and not approximate_hessian)  # don't need hprobs for approx-Hessian
    if return_gradient or return_hessian: persistentMem += 8 * nEls * ne
    if compute_hprobs: persistentMem += 8 * nEls * ne**2
    if mem_limit is not None and mem_limit < persistentMem:
        raise MemoryError("Chi2 Memory limit (%g GB) is " % (mem_limit * C)
                          + "< memory required to hold final results (%g GB)"
                          % (persistentMem * C))

    #  Allocate peristent memory
    N = _np.empty(nEls, 'd')
    f = _np.empty(nEls, 'd')
    probs = _np.empty(nEls, 'd')

    if return_gradient or return_hessian:
        dprobs = _np.empty((nEls, vec_gs_len), 'd')
    if compute_hprobs:
        hprobs = _np.empty((nEls, vec_gs_len, vec_gs_len), 'd')

    #  Estimate & check intermediate memory
    #    - maybe make Model methods get intermediate estimates?
    intermedMem = 8 * ng * gd**2  # ~ bulk_product
    if return_gradient: intermedMem += 8 * ng * gd**2 * ne  # ~ bulk_dproduct
    if compute_hprobs: intermedMem += 8 * ng * gd**2 * ne**2  # ~ bulk_hproduct
    if mem_limit is not None and mem_limit < intermedMem:
        reductionFactor = float(intermedMem) / float(mem_limit)
        maxEvalSubTreeSize = int(ng / reductionFactor)
    else:
        maxEvalSubTreeSize = None

    if maxEvalSubTreeSize is not None:
        lookup = evTree.split(lookup, maxEvalSubTreeSize, None)

    #DEBUG - no verbosity passed in to just leave commented out
    #if mem_limit is not None:
    #    print "Chi2 Memory estimates: (%d spam labels," % ns + \
    #        "%d operation sequences, %d model params, %d gate dim)" % (ng,ne,gd)
    #    print "Peristent: %g GB " % (persistentMem*C)
    #    print "Intermediate: %g GB " % (intermedMem*C)
    #    print "Limit: %g GB" % (mem_limit*C)
    #    if maxEvalSubTreeSize is not None:
    #        print "Maximum eval sub-tree size = %d" % maxEvalSubTreeSize
    #        print "Chi2 mem limit has imposed a division of evaluation tree:"
    #  evTree.print_analysis()

    dsCircuits = _lt.apply_aliases_to_circuit_list(circuits, op_label_aliases)

    for (i, opStr) in enumerate(dsCircuits):
        N[lookup[i]] = dataset[opStr].total
        f[lookup[i]] = [dataset[opStr].fraction(x) for x in outcomes_lookup[i]]

    #Detect omitted frequences (assumed to be 0) so we can compute chi2 correctly
    firsts = []; indicesOfCircuitsWithOmittedData = []
    for i, c in enumerate(circuits):
        lklen = _slct.length(lookup[i])
        if 0 < lklen < model.get_num_outcomes(c):
            firsts.append(_slct.as_array(lookup[i])[0])
            indicesOfCircuitsWithOmittedData.append(i)
    if len(firsts) > 0:
        firsts = _np.array(firsts, 'i')
        indicesOfCircuitsWithOmittedData = _np.array(indicesOfCircuitsWithOmittedData, 'i')
        dprobs_omitted_rowsum = _np.empty((len(firsts), vec_gs_len), 'd')
    else:
        firsts = None

    if compute_hprobs:
        smart(model.bulk_fill_hprobs, hprobs, evTree,
              probs, dprobs, clip_to, check, comm, _filledarrays=(0, 2, 3))
    elif return_gradient:
        smart(model.bulk_fill_dprobs, dprobs, evTree,
              probs, clip_to, check, comm, _filledarrays=(0, 2))
        for ii, i in enumerate(indicesOfCircuitsWithOmittedData):
            dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[lookup[i], :], axis=0)
    else:
        smart(model.bulk_fill_probs, probs, evTree,
              clip_to, check, comm, _filledarrays=(0,))

    # # clipped probabilities (also clip derivs to 0?)
    # cprobs = _np.clip(probs,min_prob_clip_for_weighting,1-min_prob_clip_for_weighting)
    cprobs = _np.clip(probs, min_prob_clip_for_weighting, 1e10)  # effectively no upper bound
    v = N * ((probs - f)**2 / cprobs)
    #TODO: try to replace final N[...] multiplication with dot or einsum, or do summing sooner to reduce memory

    #account for omitted probs (sparse data)
    if firsts is not None:
        omitted_probs = 1.0 - _np.array([_np.sum(probs[lookup[i]]) for i in indicesOfCircuitsWithOmittedData])
        clipped_oprobs = _np.clip(omitted_probs, min_prob_clip_for_weighting, 1 - min_prob_clip_for_weighting)
        v[firsts] = v[firsts] + N[firsts] * omitted_probs**2 / clipped_oprobs

    chi2 = _np.sum(v, axis=0)  # Note 0 is only axis in this case

    if return_gradient:
        t = ((probs - f) / cprobs)[:, None]  # (iElement, 0) = (KM,1)
        dchi2 = N[:, None] * t * (2 - t) * dprobs  # (KM,1) * (KM,1) * (KM,N)  (K=#spam, M=#strings, N=#vec_gs)

        #account for omitted probs
        if firsts is not None:
            t_firsts = (omitted_probs / clipped_oprobs)[:, None]
            dchi2[firsts, :] -= N[firsts, None] * t_firsts * (2 - t_firsts) * dprobs_omitted_rowsum

        dchi2 = _np.sum(dchi2, axis=0)  # sum over operation sequences and spam labels => (N)

    if return_hessian:
        if firsts is not None:
            raise NotImplementedError("Chi2 hessian not implemented for sparse data (yet)")

        dprobs_p = dprobs[:, None, :]  # (KM,1,N)
        t = ((probs - f) / cprobs)[:, None, None]  # (iElement, 0,0) = (KM,1,1)
        dt = ((1.0 / cprobs - (probs - f) / cprobs**2)[:, None]
              * dprobs)[:, :, None]  # (KM,1) * (KM,N) = (KM,N) => (KM,N,1)

        if approximate_hessian:  # neglect all hprobs-proportional terms
            d2chi2 = N[:, None, None] * (dt * (2 - t) * dprobs_p - t * dt * dprobs_p)
        else:  # we have hprobs and can compute the true Hessian
            d2chi2 = N[:, None, None] * (dt * (2 - t) * dprobs_p - t * dt * dprobs_p + t * (2 - t) * hprobs)

        d2chi2 = _np.sum(d2chi2, axis=0)  # sum over operation sequences and spam labels => (N1,N2)
        # (KM,1,1) * ( (KM,N1,1) * (KM,1,1) * (KM,1,N2) + (KM,1,1) * (KM,N1,1) * \
        #  (KM,1,N2) + (KM,1,1) * (KM,1,1) * (KM,N1,N2) )

    if return_gradient:
        return (chi2, dchi2, d2chi2) if return_hessian else (chi2, dchi2)
    else:
        return (chi2, d2chi2) if return_hessian else chi2


def chi2fn_2outcome(n, p, f, min_prob_clip_for_weighting=1e-4):
    """
    Computes chi^2 for a 2-outcome measurement.

    The chi-squared function for a 2-outcome measurement using
    a clipped probability for the statistical weighting.

    Parameters
    ----------
    n : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    min_prob_clip_for_weighting : float, optional
        Defines clipping interval (see return value).

    Returns
    -------
    float or numpy array
        n(p-f)^2 / (cp(1-cp)),
        where cp is the value of p clipped to the interval
        (min_prob_clip_for_weighting, 1-min_prob_clip_for_weighting)
    """
    cp = _np.clip(p, min_prob_clip_for_weighting, 1 - min_prob_clip_for_weighting)
    return n * (p - f)**2 / (cp * (1 - cp))


def chi2fn_2outcome_wfreqs(n, p, f):
    """
    Computes chi^2 for a 2-outcome measurement using frequency-weighting.

    The chi-squared function for a 2-outcome measurement using
    the observed frequency in the statistical weight.

    Parameters
    ----------
    n : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    Returns
    -------
    float or numpy array
        n(p-f)^2 / (f*(1-f*)),
        where f* = (f*n+1)/n+2 is the frequency value used in the
        statistical weighting (prevents divide by zero errors)
    """
    f1 = (f * n + 1) / (n + 2)
    return n * (p - f)**2 / (f1 * (1 - f1))


def chi2fn(n, p, f, min_prob_clip_for_weighting=1e-4):
    """
    Computes the chi^2 term corresponding to a single outcome.

    The chi-squared term for a single outcome of a multi-outcome
    measurement using a clipped probability for the statistical
    weighting.

    Parameters
    ----------
    n : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    min_prob_clip_for_weighting : float, optional
        Defines clipping interval (see return value).

    Returns
    -------
    float or numpy array
        n(p-f)^2 / cp ,
        where cp is the value of p clipped to the interval
        (min_prob_clip_for_weighting, 1-min_prob_clip_for_weighting)
    """
    #cp = _np.clip(p,min_prob_clip_for_weighting,1-min_prob_clip_for_weighting)
    cp = _np.clip(p, min_prob_clip_for_weighting, 1e10)  # effectively no upper bound
    return n * (p - f)**2 / cp


def chi2fn_wfreqs(n, p, f, min_prob_clip_for_weighting=1e-4):
    """
    Computes the frequency-weighed chi^2 term corresponding to a single outcome.

    The chi-squared term for a single outcome of a multi-outcome
    measurement using the observed frequency in the statistical weight.

    Parameters
    ----------
    n : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    min_prob_clip_for_weighting : float, optional
        unused but present to keep the same function
        signature as chi2fn.

    Returns
    -------
    float or numpy array
        n(p-f)^2 / f*,
        where f* = (f*n+1)/n+2 is the frequency value used in the
        statistical weighting (prevents divide by zero errors)
    """
    f1 = (f * n + 1) / (n + 2)
    return n * (p - f)**2 / f1

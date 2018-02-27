""" Chi-squared and related functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from . import listtools as _lt

def chi2_terms(gateset, dataset, gateStrings=None,
               minProbClipForWeighting=1e-4, clipTo=None,
               useFreqWeightedChiSq=False, check=False,
               memLimit=None, gateLabelAliases=None):
    """
    Computes the chi^2 contributions from a set of gate strings.

    This function returns the same value as :func:`chi2` with
    `returnGradient=False` and `returnHessian=False`, except the
    contributions from different gate strings and spam labels is
    not summed but returned as an array.

    Parameters
    ----------
    This function takes the same arguments as :func:`chi2` except
    for `returnGradient` and `returnHessian` (which aren't supported yet).

    Returns
    -------
    chi2 : numpy.ndarray
        Array of length either `len(gatestring_list)` or `len(dataset.keys())`.
        Values are the chi2 contributions of the corresponding gate
        string aggregated over outcomes.
    """
    if useFreqWeightedChiSq:
        raise ValueError("frequency weighted chi2 is not implemented yet.")

    if gateStrings is None:
        gateStrings = list(dataset.keys())

    evTree, blkSize1, blkSize2, lookup, outcomes_lookup = \
        gateset.bulk_evaltree_from_resources(
            gateStrings, None, memLimit, "deriv", ['bulk_fill_probs'])

    #Memory allocation
    nEls = evTree.num_final_elements()
    ng = evTree.num_final_strings()
    gd = gateset.get_dimension()
    C = 1.0/1024.0**3

    #  Estimate & check persistent memory (from allocs directly below)
    persistentMem = 8*(3*nEls) # in bytes
    if memLimit is not None and memLimit < persistentMem:
        raise MemoryError("Chi2 Memory limit (%g GB) is " % (memLimit*C) +
                          "< memory required to hold final results (%g GB)"
                          % (persistentMem*C))

    #  Allocate peristent memory
    N      = _np.empty( nEls , 'd')
    f      = _np.empty( nEls , 'd')
    probs  = _np.empty( nEls , 'd')

    dsGateStrings = _lt.find_replace_tuple_list(
            gateStrings, gateLabelAliases)
    for (i,gateStr) in enumerate(dsGateStrings):
        N[ lookup[i] ] = dataset[gateStr].total
        f[ lookup[i] ] = [ dataset[gateStr].fraction(x) for x in outcomes_lookup[i] ]

    gateset.bulk_fill_probs(probs, evTree, clipTo, check)

    cprobs = _np.clip(probs,minProbClipForWeighting,1e10) #effectively no upper bound
    v = N * ((probs - f)**2/cprobs)

    #Aggregate over outcomes:
    # v[iElement] contains all chi2 contributions - now aggregate over outcomes
    # terms[iGateString] wiil contain chi2 contributions for each original gate
    # string (aggregated over outcomes)    
    nGateStrings = len(gateStrings)
    terms = _np.empty(nGateStrings , 'd')
    for i in range(nGateStrings):
        terms[i] = _np.sum( v[lookup[i]], axis=0 )
    return terms
    


def chi2(gateset, dataset, gateStrings=None,
         returnGradient=False, returnHessian=False,
         minProbClipForWeighting=1e-4, clipTo=None,
         useFreqWeightedChiSq=False, check=False,
         memLimit=None, gateLabelAliases=None):
    """
    Computes the total chi^2 for a set of gate strings.

    The chi^2 test statistic obtained by summing up the
    contributions of a given set of gate strings or all
    the strings available in a dataset.  Optionally,
    the gradient and/or Hessian of chi^2 can be returned too.

    Parameters
    ----------
    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    dataset : DataSet
        The data used to specify frequencies and counts

    gateStrings : list of GateStrings or tuples, optional
        List of gate strings whose terms will be included in chi^2 sum.
        Default value (None) means "all strings in dataset".

    returnGradient, returnHessian : bool
        Whether to compute and return the gradient and/or Hessian of chi^2.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight (see chi2fn).

    clipTo : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.bulk_fill_probs)

    useFreqWeightedChiSq : bool, optional
        Whether or not frequencies (instead of probabilities) should be used
        in statistical weight factors.

    check : bool, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')


    Returns
    -------
    chi2 : float
        chi^2 value, equal to the sum of chi^2 terms from all specified gate strings
    dchi2 : numpy array
        Only returned if returnGradient == True. The gradient vector of
        length nGatesetParams, the number of gateset parameters.
    d2chi2 : numpy array
        Only returned if returnHessian == True. The Hessian matrix of
        shape (nGatesetParams, nGatesetParams).
    """

    # Scratch work:
    # chi^2 = sum_i N_i*(p_i-f_i)^2 / p_i  (i over gatestrings & spam labels)
    # d(chi^2)/dx = sum_i N_i * [ 2(p_i-f_i)*dp_i/dx / p_i - (p_i-f_i)^2 / p_i^2 * dp_i/dx ]
    #             = sum_i N_i * (p_i-f_i) / p_i * [2 - (p_i-f_i)/p_i   ] * dp_i/dx
    #             = sum_i N_i * t_i * [2 - t_i ] * dp_i/dx     where t_i = (p_i-f_i) / p_i
    # d2(chi^2)/dydx = sum_i N_i * [ dt_i/dy * [2 - t_i ] * dp_i/dx - t_i * dt_i/dy * dp_i/dx + t_i * [2 - t_i] * d2p_i/dydx ]
    #                          where dt_i/dy = [ 1/p_i - (p_i-f_i) / p_i^2 ] * dp_i/dy
    if useFreqWeightedChiSq:
        raise ValueError("frequency weighted chi2 is not implemented yet.")

    vec_gs_len = gateset.num_params()

    if gateStrings is None:
        gateStrings = list(dataset.keys())

    evTree,lookup,outcomes_lookup = gateset.bulk_evaltree(gateStrings)

    #Memory allocation
    nEls = evTree.num_final_elements()
    ng = evTree.num_final_strings()
    ne = gateset.num_params(); gd = gateset.get_dimension()
    C = 1.0/1024.0**3

    #  Estimate & check persistent memory (from allocs directly below)
    persistentMem = 8* (3*nEls) # in bytes
    if returnGradient or returnHessian: persistentMem += 8*nEls*ne
    if returnHessian: persistentMem += 8*nEls*ne**2
    if memLimit is not None and memLimit < persistentMem:
        raise MemoryError("Chi2 Memory limit (%g GB) is " % (memLimit*C) +
                          "< memory required to hold final results (%g GB)"
                          % (persistentMem*C))

    #  Allocate peristent memory
    N      = _np.empty( nEls, 'd')
    f      = _np.empty( nEls, 'd')
    probs  = _np.empty( nEls, 'd')

    if returnGradient or returnHessian:
        dprobs = _np.empty( (nEls,vec_gs_len), 'd')
    if returnHessian:
        hprobs = _np.empty( (nEls,vec_gs_len,vec_gs_len), 'd')

    #  Estimate & check intermediate memory
    #    - maybe make GateSet methods get intermediate estimates?
    intermedMem = 8*ng*gd**2 # ~ bulk_product
    if returnGradient: intermedMem += 8*ng*gd**2*ne # ~ bulk_dproduct
    if returnHessian: intermedMem += 8*ng*gd**2*ne**2 # ~ bulk_hproduct
    if memLimit is not None and memLimit < intermedMem:
        reductionFactor = float(intermedMem) / float(memLimit)
        maxEvalSubTreeSize = int(ng / reductionFactor)
    else:
        maxEvalSubTreeSize = None

    if maxEvalSubTreeSize is not None:
        lookup = evTree.split(lookup,maxEvalSubTreeSize, None)

    #DEBUG - no verbosity passed in to just leave commented out
    #if memLimit is not None:
    #    print "Chi2 Memory estimates: (%d spam labels," % ns + \
    #        "%d gate strings, %d gateset params, %d gate dim)" % (ng,ne,gd)
    #    print "Peristent: %g GB " % (persistentMem*C)
    #    print "Intermediate: %g GB " % (intermedMem*C)
    #    print "Limit: %g GB" % (memLimit*C)
    #    if maxEvalSubTreeSize is not None:
    #        print "Maximum eval sub-tree size = %d" % maxEvalSubTreeSize
    #        print "Chi2 mem limit has imposed a division of evaluation tree:"
    #  evTree.print_analysis()


    dsGateStrings = _lt.find_replace_tuple_list(
        gateStrings, gateLabelAliases)
    for (i,gateStr) in enumerate(dsGateStrings):
        N[ lookup[i] ] = dataset[gateStr].total
        f[ lookup[i] ] = [ dataset[gateStr].fraction(x) for x in outcomes_lookup[i] ]

    if returnHessian:
        gateset.bulk_fill_hprobs(hprobs, evTree,
                                probs, dprobs, clipTo, check)
    elif returnGradient:
        gateset.bulk_fill_dprobs(dprobs, evTree,
                                probs, clipTo, check)
    else:
        gateset.bulk_fill_probs(probs, evTree,
                                clipTo, check)


    #cprobs = _np.clip(probs,minProbClipForWeighting,1-minProbClipForWeighting) #clipped probabilities (also clip derivs to 0?)
    cprobs = _np.clip(probs,minProbClipForWeighting,1e10) #effectively no upper bound
    chi2 = _np.sum( N * ((probs - f)**2/cprobs), axis=0) # Note 0 is only axis in this case
    #TODO: try to replace final N[...] multiplication with dot or einsum, or do summing sooner to reduce memory

    if returnGradient:
        t = ((probs - f)/cprobs)[:,None] # (iElement, 0) = (KM,1)
        dchi2 = N[:,None] * t * (2 - t) * dprobs  # (KM,1) * (KM,1) * (KM,N)  (K=#spam, M=#strings, N=#vec_gs)
        dchi2 = _np.sum( dchi2, axis=0 ) # sum over gate strings and spam labels => (N)

    if returnHessian:
        dprobs_p = dprobs[:,None,:] # (KM,1,N)
        t = ((probs - f)/cprobs)[:,None,None] # (iElement, 0,0) = (KM,1,1)
        dt = ((1.0/cprobs - (probs-f)/cprobs**2)[:,None] * dprobs)[:,:,None] # (KM,1) * (KM,N) = (KM,N) => (KM,N,1)
        d2chi2 = N[:,None,None] * (dt * (2 - t) * dprobs_p - t * dt * dprobs_p + t * (2 - t) * hprobs)
        d2chi2 = _np.sum( d2chi2, axis=0 ) # sum over gate strings and spam labels => (N1,N2)
        # (KM,1,1) * ( (KM,N1,1) * (KM,1,1) * (KM,1,N2) + (KM,1,1) * (KM,N1,1) * (KM,1,N2) + (KM,1,1) * (KM,1,1) * (KM,N1,N2) )

    if returnGradient:
        return (chi2, dchi2, d2chi2) if returnHessian else (chi2, dchi2)
    else:
        return (chi2, d2chi2) if returnHessian else chi2


def chi2fn_2outcome( N, p, f, minProbClipForWeighting=1e-4 ):
    """
    Computes chi^2 for a 2-outcome measurement.

    The chi-squared function for a 2-outcome measurement using
    a clipped probability for the statistical weighting.

    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClipForWeighting : float, optional
        Defines clipping interval (see return value).

    Returns
    -------
    float or numpy array
        N(p-f)^2 / (cp(1-cp)),
        where cp is the value of p clipped to the interval
        (minProbClipForWeighting, 1-minProbClipForWeighting)
    """
    cp = _np.clip(p,minProbClipForWeighting,1-minProbClipForWeighting)
    return N*(p-f)**2/(cp*(1-cp))


def chi2fn_2outcome_wfreqs( N, p, f ):
    """
    Computes chi^2 for a 2-outcome measurement using frequency-weighting.

    The chi-squared function for a 2-outcome measurement using
    the observed frequency in the statistical weight.

    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    Returns
    -------
    float or numpy array
        N(p-f)^2 / (f*(1-f*)),
        where f* = (f*N+1)/N+2 is the frequency value used in the
        statistical weighting (prevents divide by zero errors)
    """
    f1 = (f*N+1)/(N+2)
    return N*(p-f)**2/(f1*(1-f1))


def chi2fn( N, p, f, minProbClipForWeighting=1e-4 ):
    """
    Computes the chi^2 term corresponding to a single outcome.

    The chi-squared term for a single outcome of a multi-outcome
    measurement using a clipped probability for the statistical
    weighting.

    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClipForWeighting : float, optional
        Defines clipping interval (see return value).

    Returns
    -------
    float or numpy array
        N(p-f)^2 / cp ,
        where cp is the value of p clipped to the interval
        (minProbClipForWeighting, 1-minProbClipForWeighting)
    """
    #cp = _np.clip(p,minProbClipForWeighting,1-minProbClipForWeighting)
    cp = _np.clip(p,minProbClipForWeighting,1e10) #effectively no upper bound
    return N*(p-f)**2/cp


def chi2fn_wfreqs( N, p, f, minProbClipForWeighting=1e-4 ):
    """
    Computes the frequency-weighed chi^2 term corresponding to a single outcome.

    The chi-squared term for a single outcome of a multi-outcome
    measurement using the observed frequency in the statistical weight.

    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClipForWeighting : float, optional
        unused but present to keep the same function
        signature as chi2fn.

    Returns
    -------
    float or numpy array
        N(p-f)^2 / f*,
        where f* = (f*N+1)/N+2 is the frequency value used in the
        statistical weighting (prevents divide by zero errors)
    """
    f1 = (f*N+1)/(N+2)
    return N*(p-f)**2/f1

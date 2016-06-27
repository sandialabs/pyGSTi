from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Chi-squared and related functions """

import numpy as _np

def chi2(dataset, gateset, gateStrings=None,
         returnGradient=False, returnHessian=False,
         minProbClipForWeighting=1e-4, clipTo=None,
         useFreqWeightedChiSq=False, check=False,
         memLimit=None):
    """
    Computes the total chi^2 for a set of gate strings.

    The chi^2 test statistic obtained by summing up the
    contributions of a given set of gate strings or all
    the strings available in a dataset.  Optionally,
    the gradient and/or Hessian of chi^2 can be returned too.

    Parameters
    ----------
    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

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

    spamLabels = gateset.get_spam_labels() #this list fixes the ordering of the spam labels
    spam_lbl_rows = { sl:i for (i,sl) in enumerate(spamLabels) }
    vec_gs_len = gateset.num_params()

    if gateStrings is None:
        gateStrings = list(dataset.keys())

    nSpamLabels = len(spamLabels)
    nGateStrings = len(gateStrings)

    evTree = gateset.bulk_evaltree(gateStrings)

    #Memory allocation
    ns = nSpamLabels; ng = nGateStrings
    ne = gateset.num_params(); gd = gateset.get_dimension()
    C = 1.0/1024.0**3

    #  Estimate & check persistent memory (from allocs directly below)
    persistentMem = 8* (ng*(1 + 2*ns)) # in bytes
    if returnGradient or returnHessian: persistentMem += 8*ng*ns*ne
    if returnHessian: persistentMem += 8*ng*ns*ne**2
    if memLimit is not None and memLimit < persistentMem:
        raise MemoryError("Chi2 Memory limit (%g GB) is " % (memLimit*C) +
                          "< memory required to hold final results (%g GB)"
                          % (persistentMem*C))

    #  Allocate peristent memory
    N      = _np.empty( nGateStrings )
    f      = _np.empty( (nSpamLabels, nGateStrings) )
    probs  = _np.empty( (nSpamLabels, nGateStrings) )

    if returnGradient or returnHessian:
        dprobs = _np.empty( (nSpamLabels, nGateStrings,vec_gs_len) )
    if returnHessian:
        hprobs = _np.empty( (nSpamLabels, nGateStrings,vec_gs_len,vec_gs_len) )

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
        evTree.split(maxEvalSubTreeSize, None)

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


    for (i,gateStr) in enumerate(gateStrings):
        N[i] = float(dataset[gateStr].total())
        for k,sl in enumerate(spamLabels):
            f[k,i] = dataset[gateStr].fraction(sl)


    if returnHessian:
        gateset.bulk_fill_hprobs(hprobs, spam_lbl_rows, evTree,
                                probs, dprobs, clipTo, check)
    elif returnGradient:
        gateset.bulk_fill_dprobs(dprobs, spam_lbl_rows, evTree,
                                probs, clipTo, check)
    else:
        gateset.bulk_fill_probs(probs, spam_lbl_rows, evTree,
                                clipTo, check)


    #cprobs = _np.clip(probs,minProbClipForWeighting,1-minProbClipForWeighting) #clipped probabilities (also clip derivs to 0?)
    cprobs = _np.clip(probs,minProbClipForWeighting,1e10) #effectively no upper bound
    chi2 = _np.sum( N[None,:] * ((probs - f)**2/cprobs), axis=(0,1) ) # Note (0,1) are all axes in this case
    #TODO: try to replace final N[...] multiplication with dot or einsum, or do summing sooner to reduce memory

    if returnGradient:
        t = ((probs - f)/cprobs)[:,:,None] # (iSpamLabel, iGateString, 0) = (K,M,1)
        dchi2 = N[None,:,None] * t * (2 - t) * dprobs  # (1,M,1) * (K,M,1) * (K,M,N)  (K=#spam, M=#strings, N=#vec_gs)
        dchi2 = _np.sum( dchi2, axis=(0,1) ) # sum over gate strings and spam labels => (N)

    if returnHessian:
        dprobs_p = dprobs[:,:,None,:] # (K,M,1,N)
        t = ((probs - f)/cprobs)[:,:,None,None] # (iSpamLabel, iGateString, 0,0) = (K,M,1,1)
        dt = ((1.0/cprobs - (probs-f)/cprobs**2)[:,:,None] * dprobs)[:,:,:,None] # (K,M,1) * (K,M,N) = (K,M,N) => (K,M,N,1)
        d2chi2 = N[None,:,None,None] * (dt * (2 - t) * dprobs_p - t * dt * dprobs_p + t * (2 - t) * hprobs)
        d2chi2 = _np.sum( d2chi2, axis=(0,1) ) # sum over gate strings and spam labels => (N1,N2)
        # (1,M,1,1) * ( (K,M,N1,1) * (K,M,1,1) * (K,M,1,N2) + (K,M,1,1) * (K,M,N1,1) * (K,M,1,N2) + (K,M,1,1) * (K,M,1,1) * (K,M,N1,N2) )

    if returnGradient:
        return (chi2, dchi2, d2chi2) if returnHessian else (chi2, dchi2)
    else:
        return (chi2, d2chi2) if returnHessian else chi2



#def _oldTotalChiSquared( dataset, gateset, gateStrings=None, useFreqWeightedChiSq=False,
#                     minProbClipForWeighting=1e-4):
#    """
#    Return summed chi^2 values for given gateStrings or all gatestrings
#    in the given dataset (if gateStrings is None).
#    If useFreqWeightedChiSq is True, then use the dataset frequencies instead
#    of the (correct) gateset probabilities in the expression for "Chi^2".
#    """
#
#    if gateStrings is None:
#        return sum( [ gate_string_chi2( gatestring, dataset, gateset, useFreqWeightedChiSq,
#                                       minProbClipForWeighting) for gatestring in dataset] )
#        #return sum( [ chi2fn_2outcome( dsRow.total(),
#        #                     gateset.pr('plus', gatestring),
#        #                     dsRow.fraction('plus'),
#        #                     minProbClipForWeighting ) for gatestring,dsRow in dataset.iteritems() ] )
#    else:
#        return sum( [ gate_string_chi2( gatestring, dataset, gateset, useFreqWeightedChiSq,
#                                       minProbClipForWeighting) for gatestring in gateStrings] )
#

def gate_string_chi2( gatestring, dataset, gateset, useFreqWeightedChiSq=False,
                     minProbClipForWeighting=1e-4):
    """
    Computes the chi-squared term for a single gate string.

    The sum of the chi-squared terms for each of the possible
    outcomes (the SPAM labels in gateset).

    Parameters
    ----------
    gatestring : GateString or tuple
        The gate string to compute chi-squared for.

    dataset : Dataset
        The object from which frequencies are extracted.

    gateset : GateSet
        The object from which probabilities and SPAM labels are extracted.

    useFreqWeightedChiSq : bool, optional
        Whether or not frequencies (instead of probabilities) should be used
        in statistical weight factors.

    minProbClipForWeighting : float, optional
        If useFreqWeightedChiSq == False, defines the clipping interval
        for the statistical weight (see chi2fn).

    Returns
    -------
    float
        The sum of chi^2 terms for corresponding to gatestring
        (one term per SPAM label).
    """

    chiSqFn  = chi2fn_wfreqs if useFreqWeightedChiSq else chi2fn
    rowData  = dataset[gatestring]
    clip = minProbClipForWeighting

    N = rowData.total()
    p = gateset.probs(gatestring)
    return sum( [ chiSqFn(N, p[sl], rowData[sl]/N, clip) for sl in gateset.get_spam_labels() ] )


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


def chi2fn_2outcome_wfreqs( N, p, f, minProbClipForWeighting=1e-4 ):
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

    minProbClipForWeighting : float, optional
        unused but present to keep the same function
        signature as chi2fn_2outcome.

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

"""Functions related to computation of the log-likelihood."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import warnings as _warnings
import itertools as _itertools
import time as _time
import sys as _sys
from collections import OrderedDict as _OrderedDict
from . import basistools as _bt
from . import listtools as _lt
from . import jamiolkowski as _jam
from . import mpitools as _mpit
from ..baseobjs import smart_cached

TOL = 1e-20


 # The log(Likelihood) within the standard (non-Poisson) picture is:
 #
 # L = prod_{i,sl} p_{i,sl}^N_{i,sl}
 #
 # Where i indexes the gate string, and sl indexes the spam label.  N[i] is the total counts
 #  for the i-th gatestring, and so sum_{sl} N_{i,sl} == N[i]. We can take the log:
 #
 # log L = sum_{i,sl} N_{i,sl} log(p_{i,sl})
 #
 #   after patching (linear extrapolation below min_p and ignore f == 0 terms ( 0*log(0) == 0 ) ):
 #
 # logl = sum_{i,sl} N_{i,sl} log(p_{i,sl})                                                        if p_{i,sl} >= min_p and N_{i,sl} > 0
 #                   N_{i,sl} log(min_p)     + S * (p_{i,sl} - min_p) + S2 * (p_{i,sl} - min_p)**2 if p_{i,sl} < p_min and N_{i,sl} > 0
 #                   0                                                                             if N_{i,sl} == 0
 #
 # dlogL = sum_{i,sl} N_{i,sl} / p_{i,sl} * dp                    if p_{i,sl} >= min_p and N_{i,sl} > 0
 #                    (S + 2*S2*(p_{i,sl} - min_p)) * dp          if p_{i,sl} < p_min and N_{i,sl} > 0
 #                    0                                           if N_{i,sl} == 0
 #
 # hlogL = sum_{i,sl} -N_{i,sl} / p_{i,sl}**2 * dp1 * dp2 +  N_{i,sl} / p_{i,sl} *hp        if p_{i,sl} >= min_p and N_{i,sl} > 0
 #                    2*S2* dp1 * dp2 + (S + 2*S2*(p_{i,sl} - min_p)) * hp                  if p_{i,sl} < p_min and N_{i,sl} > 0
 #                    0                                                                     if N_{i,sl} == 0
 #
 #  where S = N_{i,sl} / min_p is the slope of the line tangent to logl at min_p
 #    and S2 = 0.5*( -N_{i,sl} / min_p**2 ) is 1/2 the 2nd derivative of the logl term at min_p
 #   and hlogL == d/d1 ( d/d2 ( logl ) )  -- i.e. dp2 is the *first* derivative performed...


 #Note: Poisson picture entered use when we allowed an EVec which was 1-{other EVecs} -- a
 # (0,-1) spam index -- instead of assuming all probabilities of a given gat string summed
 # to one -- a (-1,-1) spam index.  The poisson picture gives a correct log-likelihood
 # description when the probabilities (for a given gate string) may not sum to one, by
 # interpreting them each as rates.  In the standard picture, large gatestring probabilities
 # are not penalized (each standard logL term increases monotonically with each probability,
 # and the reason this is ok when the probabilities sum to one is that for a probabilility
 # that gets close to 1, there's another that is close to zero, and logL is very negative
 # near zero.

 # The log(Likelihood) within the Poisson picture is:
 #
 # L = prod_{i,sl} lambda_{i,sl}^N_{i,sl} e^{-lambda_{i,sl}} / N_{i,sl}!
 #
 # Where lamba_{i,sl} := p_{i,sl}*N[i] is a rate, i indexes the gate string,
 #  and sl indexes the spam label.  N[i] is the total counts for the i-th gatestring, and
 #  so sum_{sl} N_{i,sl} == N[i]. We can ignore the p-independent N_j! and take the log:
 #
 # log L = sum_{i,sl} N_{i,sl} log(N[i]*p_{i,sl}) - N[i]*p_{i,sl}
 #       = sum_{i,sl} N_{i,sl} log(p_{i,sl}) - N[i]*p_{i,sl}   (where we ignore the p-independent log(N[i]) terms)
 #
 #   after patching (linear extrapolation below min_p and "softening" f == 0 terms w/cubic below radius "a"):
 #
 # logl = sum_{i,sl} N_{i,sl} log(p_{i,sl}) - N[i]*p_{i,sl}                                                        if p_{i,sl} >= min_p and N_{i,sl} > 0
 #                   N_{i,sl} log(min_p)    - N[i]*min_p    + S * (p_{i,sl} - min_p) + S2 * (p_{i,sl} - min_p)**2  if p_{i,sl} < p_min and N_{i,sl} > 0
 #                   0                      - N[i]*p_{i,sl}                                                        if N_{i,sl} == 0 and p_{i,sl} >= a
 #                   0                      - N[i]*( -(1/(3a**2))p_{i,sl}**3 + p_{i,sl}**2/a + (1/3)*a )           if N_{i,sl} == 0 and p_{i,sl} < a
 #
 # dlogL = sum_{i,sl} [ N_{i,sl} / p_{i,sl} - N[i] ] * dp                   if p_{i,sl} >= min_p and N_{i,sl} > 0
 #                    (S + 2*S2*(p_{i,sl} - min_p)) * dp                    if p_{i,sl} < p_min and N_{i,sl} > 0
 #                    -N[i] * dp                                            if N_{i,sl} == 0 and p_{i,sl} >= a
 #                    -N[i] * ( (-1/a**2)p_{i,sl}**2 + 2*p_{i,sl}/a ) * dp  if N_{i,sl} == 0 and p_{i,sl} < a
 #
 # hlogL = sum_{i,sl} -N_{i,sl} / p_{i,sl}**2 * dp1 * dp2 + [ N_{i,sl} / p_{i,sl} - N[i] ]*hp      if p_{i,sl} >= min_p and N_{i,sl} > 0
 #                    2*S2* dp1 * dp2 + (S + 2*S2*(p_{i,sl} - min_p)) * hp                         if p_{i,sl} < p_min and N_{i,sl} > 0
 #                    -N[i] * hp                                                                   if N_{i,sl} == 0 and p_{i,sl} >= a
 #                    -N[i]*( (-2/a**2)p_{i,sl} + 2/a ) * dp1 * dp2
 #                        - N[i]*( (-1/a**2)p_{i,sl}**2 + 2*p_{i,sl}/a ) * hp                      if N_{i,sl} == 0 and p_{i,sl} < a
 #
 #  where S = N_{i,sl} / min_p - N[i] is the slope of the line tangent to logl at min_p
 #    and S2 = 0.5*( -N_{i,sl} / min_p**2 ) is 1/2 the 2nd derivative of the logl term at min_p so
 #    logL_term = logL_term(min_p) + S * (p-min_p) + S2 * (p-min_p)**2
 #   and hlogL == d/d1 ( d/d2 ( logl ) )  -- i.e. dp2 is the *first* derivative performed...
 #
 # For cubic interpolation, use function F(p) (derived by Robin: match value, 1st-deriv, 2nd-deriv at p == r, and require min at p == 0):
 #  Given a radius r << 1 (but r>0):
 #   F(p) = piecewise{ if( p>r ) then p; else -(1/3)*p^3/r^2 + p^2/r + (1/3)*r }
 #  OLD: quadratic that doesn't match 2nd-deriv:
 #   F(p) = piecewise{ if( p>r ) then p; else (r-p)^2/(2*r) + p }



#@smart_cached
def logl_terms(gateset, dataset, gatestring_list=None,
               minProbClip=1e-6, probClipInterval=(-1e6,1e6), radius=1e-4,
               poissonPicture=True, check=False, gateLabelAliases=None,
               evaltree_cache=None, comm=None):
    """
    The vector of log-likelihood contributions for each gate string, 
    aggregated over outcomes.

    Parameters
    ----------
    This function takes the same arguments as :func:`logl` except it
    doesn't perform the final sum over gate sequences and SPAM labels.

    Returns
    -------
    numpy.ndarray
        Array of length either `len(gatestring_list)` or `len(dataset.keys())`.
        Values are the log-likelihood contributions of the corresponding gate
        string aggregated over outcomes.
    """
    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    a = radius # parameterizes "roundness" of f == 0 terms
    min_p = minProbClip

    if evaltree_cache and 'evTree' in evaltree_cache:
        evalTree = evaltree_cache['evTree']
        lookup = evaltree_cache['lookup']
        outcomes_lookup = evaltree_cache['outcomes_lookup']
        #tree_gatestring_list = evalTree.generate_gatestring_list()
        # Note: this is != gatestring_list, as the tree hold *compiled* gatestrings
    else:
        evalTree,lookup,outcomes_lookup = gateset.bulk_evaltree(gatestring_list, dataset=dataset)

        #Fill cache dict if one was given
        if evaltree_cache is not None:
            evaltree_cache['evTree'] = evalTree
            evaltree_cache['lookup'] = lookup
            evaltree_cache['outcomes_lookup'] = outcomes_lookup
        
    nEls = evalTree.num_final_elements()    
    probs = _np.empty( nEls, 'd' )

    ds_gatestring_list = _lt.find_replace_tuple_list(
        gatestring_list, gateLabelAliases)

    if evaltree_cache and 'cntVecMx' in evaltree_cache:
        countVecMx = evaltree_cache['cntVecMx']
        totalCntVec = evaltree_cache['totalCntVec']
    else:
        countVecMx = _np.empty(nEls, 'd' )
        totalCntVec = _np.empty(nEls, 'd' )
        for (i,gateStr) in enumerate(ds_gatestring_list):
            cnts = dataset[gateStr].counts
            totalCntVec[ lookup[i] ] = sum(cnts.values()) #dataset[gateStr].total
            countVecMx[ lookup[i] ] = [ cnts.get(x,0) for x in outcomes_lookup[i] ]

        #could add to cache, but we don't have option of gateStringWeights
        # here yet, so let's be conservative and not do this:
        #if evaltree_cache is not None:
        #    evaltree_cache['cntVecMx'] = countVecMx
        #    evaltree_cache['totalCntVec'] = totalCntVec


    #OLD
    #freqs = countVecMx / totalCntVec[None,:]
    #freqs_nozeros = _np.where(countVecMx == 0, 1.0, freqs) # set zero freqs to 1.0 so np.log doesn't complain
    #freqTerm = countVecMx * ( _np.log(freqs_nozeros) - 1.0 )
    #freqTerm[ countVecMx == 0 ] = 0.0 # set 0 * log(0) terms explicitly to zero since numpy doesn't know this limiting behavior

    gateset.bulk_fill_probs(probs, evalTree, probClipInterval, check, comm)
    pos_probs = _np.where(probs < min_p, min_p, probs)

    if poissonPicture:
        S = countVecMx / min_p - totalCntVec # slope term that is derivative of logl at min_p
        S2 = -0.5 * countVecMx / (min_p**2)          # 2nd derivative of logl term at min_p
        v = countVecMx * _np.log(pos_probs) - totalCntVec*pos_probs # dim KM (K = nSpamLabels, M = nGateStrings)
        v = _np.minimum(v,0)  #remove small positive elements due to roundoff error (above expression *cannot* really be positive)
        v = _np.where( probs < min_p, v + S*(probs - min_p) + S2*(probs - min_p)**2, v) #quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where( countVecMx == 0, -totalCntVec * _np.where(probs >= a, probs, (-1.0/(3*a**2))*probs**3 + probs**2/a + a/3.0), v)
           #special handling for f == 0 poissonPicture terms using quadratic rounding of function with minimum: max(0,(a-p))^2/(2a) + p

    else: #(the non-poisson picture requires that the probabilities of the spam labels for a given string are constrained to sum to 1)
        S = countVecMx / min_p               # slope term that is derivative of logl at min_p
        S2 = -0.5 * countVecMx / (min_p**2)  # 2nd derivative of logl term at min_p
        v = countVecMx * _np.log(pos_probs) # dim KM (K = nSpamLabels, M = nGateStrings)
        v = _np.minimum(v,0)  #remove small positive elements due to roundoff error (above expression *cannot* really be positive)
        v = _np.where( probs < min_p, v + S*(probs - min_p) + S2*(probs - min_p)**2, v) #quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where( countVecMx == 0, 0.0, v)

    #DEBUG
    #print "num clipped = ",_np.sum(probs < min_p)," of ",probs.shape
    #print "min/max probs = ",min(probs.flatten()),",",max(probs.flatten())
    #for i in range(v.shape[1]):
    #    print "%d %.0f (%f) %.0f (%g)" % (i,v[0,i],probs[0,i],v[1,i],probs[1,i])

    #Aggregate over outcomes:
    # v[iElement] contains all logl contributions - now aggregate over outcomes
    # terms[iGateString] wiil contain logl contributions for each original gate
    # string (aggregated over outcomes)    
    nGateStrings = len(gatestring_list)
    terms = _np.empty(nGateStrings , 'd')
    for i in range(nGateStrings):
        terms[i] = _np.sum( v[lookup[i]], axis=0 )
    return terms


#@smart_cached
def logl(gateset, dataset, gatestring_list=None,
         minProbClip=1e-6, probClipInterval=(-1e6,1e6), radius=1e-4,
         poissonPicture=True, check=False, gateLabelAliases=None,
         evaltree_cache=None, comm=None):
    """
    The log-likelihood function.

    Parameters
    ----------
    gateset : GateSet
        Gateset of parameterized gates

    dataset : DataSet
        Probability data

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the log-likelihood
        sum.  Default value of None implies all the gate strings in dataset
        should be used.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by gatesets during MLEGST's
        search for an optimal gateset (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    evalTree : evaluation tree, optional
      given by a prior call to bulk_evaltree for the same gatestring_list.
      Significantly speeds up evaluation of log-likelihood, even more so
      when accompanied by countVecMx (see below).

    poissonPicture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    evaltree_cache : dict, optional
        A dictionary which server as a cache for the computed EvalTree used
        in this computation.  If an empty dictionary is supplied, it is filled
        with cached values to speed up subsequent executions of this function
        which use the *same* `gateset` and `gatestring_list`.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    float
        The log likelihood
    """
    v = logl_terms(gateset, dataset, gatestring_list,
                   minProbClip, probClipInterval, radius,
                   poissonPicture, check, gateLabelAliases, evaltree_cache, comm)
    return _np.sum(v) # sum over *all* dimensions


def logl_jacobian(gateset, dataset, gatestring_list=None,
                  minProbClip=1e-6, probClipInterval=(-1e6,1e6), radius=1e-4,
                  poissonPicture=True, check=False, comm=None,
                  memLimit=None, gateLabelAliases=None, verbosity=0):
    """
    The jacobian of the log-likelihood function.

    Parameters
    ----------
    gateset : GateSet
        Gateset of parameterized gates (including SPAM)

    dataset : DataSet
        Probability data

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the log-likelihood
        sum.  Default value of None implies all the gate strings in dataset
        should be used.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by gatesets during MLEGST's
        search for an optimal gateset (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    evalTree : evaluation tree, optional
        given by a prior call to bulk_evaltree for the same gatestring_list.
        Significantly speeds up evaluation of log-likelihood derivatives, even
        more so when accompanied by countVecMx (see below).  Defaults to None.

    poissonPicture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    verbosity : int, optional
        How much detail to print to stdout.

    Returns
    -------
    numpy array
      array of shape (M,), where M is the length of the vectorized gateset.
    """

    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    C = 1.0/1024.0**3; nP = gateset.num_params()
    persistentMem = 8*nP + 8*len(gatestring_list)*(nP+1) # in bytes

    if memLimit is not None and memLimit < persistentMem:
        raise MemoryError("DLogL Memory limit (%g GB) is " % (memLimit*C) +
                          "< memory required to hold final results (%g GB)"
                          % (persistentMem*C))


    #OLD: evalTree,lookup,outcomes_lookup = gateset.bulk_evaltree(gatestring_list)
    mlim = None if (memLimit is None) else memLimit-persistentMem
    dstree = dataset if (gateLabelAliases is None) else None #Note: compile_gatestrings doesn't support aliased dataset (yet)
    evalTree, blkSize, _, lookup, outcomes_lookup = \
        gateset.bulk_evaltree_from_resources(
            gatestring_list, comm, mlim, "deriv", ['bulk_fill_dprobs'],
            dstree, verbosity)

    a = radius # parameterizes "roundness" of f == 0 terms
    min_p = minProbClip

    #  Allocate persistent memory
    jac = _np.zeros([1,nP])
    nEls = evalTree.num_final_elements()
    probs = _np.empty( nEls, 'd' )
    dprobs = _np.empty( (nEls,nP), 'd' )

    ds_gatestring_list = _lt.find_replace_tuple_list(
        gatestring_list, gateLabelAliases)

    countVecMx = _np.empty(nEls, 'd' )
    totalCntVec = _np.empty(nEls, 'd' )
    for (i,gateStr) in enumerate(ds_gatestring_list):
        cnts = dataset[gateStr].counts
        totalCntVec[ lookup[i] ] = sum(cnts.values()) #dataset[gateStr].total
        countVecMx[ lookup[i] ] = [ cnts.get(x,0) for x in outcomes_lookup[i] ]

    #OLD
    #freqs = cntVecMx / totalCntVec[None,:]
    #freqs_nozeros = _np.where(cntVecMx == 0, 1.0, freqs) # set zero freqs to 1.0 so np.log doesn't complain
    #freqTerm = cntVecMx * ( _np.log(freqs_nozeros) - 1.0 )
    #freqTerm[ cntVecMx == 0 ] = 0.0 # set 0 * log(0) terms explicitly to zero since numpy doesn't know this limiting behavior
    #minusCntVecMx = -1.0 * cntVecMx


    gateset.bulk_fill_dprobs(dprobs, evalTree, prMxToFill=probs,
                             clipTo=probClipInterval, check=check, comm=comm,
                             wrtBlockSize=blkSize) # FUTURE: set gatherMemLimit=?


    pos_probs = _np.where(probs < min_p, min_p, probs)

    if poissonPicture:
        S = countVecMx / min_p - totalCntVec         # slope term that is derivative of logl at min_p
        S2 = -0.5 * countVecMx / (min_p**2)          # 2nd derivative of logl term at min_p
        v = countVecMx * _np.log(pos_probs) - totalCntVec*pos_probs # dim KM (K = nSpamLabels, M = nGateStrings)
        v = _np.minimum(v,0)  #remove small positive elements due to roundoff error (above expression *cannot* really be positive)
        v = _np.where( probs < min_p, v + S*(probs - min_p) + S2*(probs - min_p)**2, v) #quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where( countVecMx == 0, -totalCntVec * _np.where(probs >= a, probs, (-1.0/(3*a**2))*probs**3 + probs**2/a + a/3.0), v)
           #special handling for f == 0 poissonPicture terms using quadratic rounding of function with minimum: max(0,(a-p))^2/(2a) + p

        dprobs_factor_pos = (countVecMx / pos_probs - totalCntVec)
        dprobs_factor_neg = S + 2*S2*(probs - min_p)
        dprobs_factor_zerofreq = -totalCntVec * _np.where( probs >= a, 1.0, (-1.0/a**2)*probs**2 + 2*probs/a)
        dprobs_factor = _np.where( probs < min_p, dprobs_factor_neg, dprobs_factor_pos)
        dprobs_factor = _np.where( countVecMx == 0, dprobs_factor_zerofreq, dprobs_factor )
        jac = dprobs * dprobs_factor[:,None] # (KM,N) * (KM,1)   (N = dim of vectorized gateset)


    else: #(the non-poisson picture requires that the probabilities of the spam labels for a given string are constrained to sum to 1)
        S = countVecMx / min_p              # slope term that is derivative of logl at min_p
        S2 = -0.5 * countVecMx / (min_p**2) # 2nd derivative of logl term at min_p
        v = countVecMx * _np.log(pos_probs) # dims K x M (K = nSpamLabels, M = nGateStrings)
        v = _np.minimum(v,0)  #remove small positive elements due to roundoff error (above expression *cannot* really be positive)
        v = _np.where( probs < min_p, v + S*(probs - min_p) + S2*(probs - min_p)**2, v) #quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where( countVecMx == 0, 0.0, v)

        dprobs_factor_pos = countVecMx / pos_probs
        dprobs_factor_neg = S + 2*S2*(probs - min_p)
        dprobs_factor = _np.where( probs < min_p, dprobs_factor_neg, dprobs_factor_pos)
        dprobs_factor = _np.where( countVecMx == 0, 0.0, dprobs_factor )
        jac = dprobs * dprobs_factor[:,None] # (KM,N) * (KM,1)   (N = dim of vectorized gateset)

    # jac[iSpamLabel,iGateString,iGateSetParam] contains all d(logl)/d(gatesetParam) contributions
    return _np.sum(jac, axis=0) # sum over spam label and gate string dimensions


def logl_hessian(gateset, dataset, gatestring_list=None, minProbClip=1e-6,
                 probClipInterval=(-1e6,1e6), radius=1e-4, poissonPicture=True,
                 check=False, comm=None, memLimit=None,
                 gateLabelAliases=None, verbosity=0):
    """
    The hessian of the log-likelihood function.

    Parameters
    ----------
    gateset : GateSet
        Gateset of parameterized gates (including SPAM)

    dataset : DataSet
        Probability data

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the log-likelihood
        sum.  Default value of None implies all the gate strings in dataset
        should be used.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by
        gatesets during MLEGST's search for an optimal gateset (if not None).
        if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poissonPicture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    verbosity : int, optional
        How much detail to print to stdout.


    Returns
    -------
    numpy array
      array of shape (M,M), where M is the length of the vectorized gateset.
    """

    nP = gateset.num_params()

    if gatestring_list is None:
        gatestring_list = list(dataset.keys())
    
    #  Estimate & check persistent memory (from allocs directly below)
    C = 1.0/1024.0**3; nP = gateset.num_params()
    persistentMem = 8*nP**2 # in bytes
    if memLimit is not None and memLimit < persistentMem:
        raise MemoryError("HLogL Memory limit (%g GB) is " % (memLimit*C) +
                          "< memory required to hold final results (%g GB)"
                          % (persistentMem*C))

    #  Allocate persistent memory
    final_hessian = _np.zeros( (nP,nP), 'd')

    #  Estimate & check intermediate memory
    #  - figure out how many row & column partitions are needed
    #    to fit computation within available memory (and use all cpus)
    mlim = None if (memLimit is None) else memLimit-persistentMem
    dstree = dataset if (gateLabelAliases is None) else None #Note: compile_gatestrings doesn't support aliased dataset (yet)
    evalTree, blkSize1, blkSize2, lookup, outcomes_lookup = \
        gateset.bulk_evaltree_from_resources(
            gatestring_list, comm, mlim, "deriv", ['bulk_hprobs_by_block'],
            dstree, verbosity)
    
    rowParts = int(round(nP / blkSize1)) if (blkSize1 is not None) else 1
    colParts = int(round(nP / blkSize2)) if (blkSize2 is not None) else 1

    a = radius # parameterizes "roundness" of f == 0 terms
    min_p = minProbClip

    if poissonPicture:
        #NOTE: hessian_from_hprobs MAY modify hprobs and dprobs12 (to save mem)
        def hessian_from_hprobs(hprobs, dprobs12, cntVecMx, totalCntVec, pos_probs):
            """ Factored-out computation of hessian from raw components """
            # Notation:  (K=#spam, M=#strings, N=#wrtParams1, N'=#wrtParams2 )
            totCnts = totalCntVec  #shorthand
            S = cntVecMx / min_p - totCnts # slope term that is derivative of logl at min_p
            S2 = -0.5 * cntVecMx / (min_p**2)          # 2nd derivative of logl term at min_p

            #hprobs_pos  = (-cntVecMx / pos_probs**2)[:,:,None,None] * dprobs12   # (K,M,1,1) * (K,M,N,N')
            #hprobs_pos += (cntVecMx / pos_probs - totalCntVec[None,:])[:,:,None,None] * hprobs  # (K,M,1,1) * (K,M,N,N')
            #hprobs_neg  = (2*S2)[:,:,None,None] * dprobs12 + (S + 2*S2*(probs - min_p))[:,:,None,None] * hprobs # (K,M,1,1) * (K,M,N,N')
            #hprobs_zerofreq = _np.where( (probs >= a)[:,:,None,None],
            #                             -totalCntVec[None,:,None,None] * hprobs,
            #                             (-totalCntVec[None,:] * ( (-2.0/a**2)*probs + 2.0/a))[:,:,None,None] * dprobs12
            #                             - (totalCntVec[None,:] * ((-1.0/a**2)*probs**2 + 2*probs/a))[:,:,None,None] * hprobs )
            #hessian = _np.where( (probs < min_p)[:,:,None,None], hprobs_neg, hprobs_pos)
            #hessian = _np.where( (cntVecMx == 0)[:,:,None,None], hprobs_zerofreq, hessian) # (K,M,N,N')
            
            #Accomplish the same thing as the above commented-out lines, 
            # but with more memory effiency:
            dprobs12_coeffs = \
                _np.where(probs < min_p, 2*S2, -cntVecMx / pos_probs**2)
            zfc = _np.where(probs >= a, 0.0, -totCnts*((-2.0/a**2)*probs+2.0/a))
            dprobs12_coeffs = _np.where(cntVecMx == 0, zfc, dprobs12_coeffs)

            hprobs_coeffs = \
                _np.where(probs < min_p, S + 2*S2*(probs - min_p),
                          cntVecMx / pos_probs - totCnts)
            zfc = _np.where(probs >= a, -totCnts, 
                            -totCnts * ((-1.0/a**2)*probs**2 + 2*probs/a))
            hprobs_coeffs = _np.where(cntVecMx == 0, zfc, hprobs_coeffs)

              # hessian = hprobs_coeffs * hprobs + dprobs12_coeff * dprobs12
              #  but re-using dprobs12 and hprobs memory (which is overwritten!)
            hprobs *= hprobs_coeffs[:,None,None]
            dprobs12 *= dprobs12_coeffs[:,None,None]
            hessian = dprobs12; hessian += hprobs

            # hessian[iSpamLabel,iGateString,iGateSetParam1,iGateSetParams2] contains all
            #  d2(logl)/d(gatesetParam1)d(gatesetParam2) contributions
            return _np.sum(hessian, axis=0)
              # sum over spam label and gate string dimensions (gate strings in evalSubTree)
              # adds current subtree contribution for (N,N')-sized block of Hessian


    else:

        #(the non-poisson picture requires that the probabilities of the spam labels for a given string are constrained to sum to 1)
        #NOTE: hessian_from_hprobs MAY modify hprobs and dprobs12 (to save mem)
        def hessian_from_hprobs(hprobs, dprobs12, cntVecMx, totalCntVec, pos_probs):
            """ Factored-out computation of hessian from raw components """
            S = cntVecMx / min_p # slope term that is derivative of logl at min_p
            S2 = -0.5 * cntVecMx / (min_p**2) # 2nd derivative of logl term at min_p

            #hprobs_pos  = (-cntVecMx / pos_probs**2)[:,:,None,None] * dprobs12   # (K,M,1,1) * (K,M,N,N')
            #hprobs_pos += (cntVecMx / pos_probs)[:,:,None,None] * hprobs  # (K,M,1,1) * (K,M,N,N')
            #hprobs_neg  = (2*S2)[:,:,None,None] * dprobs12 + (S + 2*S2*(probs - min_p))[:,:,None,None] * hprobs # (K,M,1,1) * (K,M,N,N')
            #hessian = _np.where( (probs < min_p)[:,:,None,None], hprobs_neg, hprobs_pos)
            #hessian = _np.where( (cntVecMx == 0)[:,:,None,None], 0.0, hessian) # (K,M,N,N')

            #Accomplish the same thing as the above commented-out lines, 
            # but with more memory effiency:
            dprobs12_coeffs = \
                _np.where(probs < min_p, 2*S2, -cntVecMx / pos_probs**2)
            dprobs12_coeffs = _np.where(cntVecMx == 0, 0.0, dprobs12_coeffs)

            hprobs_coeffs = \
                _np.where(probs < min_p, S + 2*S2*(probs - min_p),
                          cntVecMx / pos_probs)
            hprobs_coeffs = _np.where(cntVecMx == 0, 0.0, hprobs_coeffs)

              # hessian = hprobs_coeffs * hprobs + dprobs12_coeff * dprobs12
              #  but re-using dprobs12 and hprobs memory (which is overwritten!)
            hprobs *= hprobs_coeffs[:,None,None]
            dprobs12 *= dprobs12_coeffs[:,None,None]
            hessian = dprobs12; hessian += hprobs

            return _np.sum(hessian, axis=0) #see comments as above


    #Note - we could in the future use comm to distribute over
    # subtrees here.  We currently don't because we parallelize
    # over columns and it seems that in almost all cases of
    # interest there will be more hessian columns than processors,
    # so adding the additional ability to parallelize over
    # subtrees would just add unnecessary complication.

    #get distribution across subtrees (groups if needed)
    subtrees = evalTree.get_sub_trees()
    mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

    #  Allocate memory (alloc max required & take views)
    max_nEls = max([subtrees[i].num_final_elements() for i in mySubTreeIndices])
    probs_mem  = _np.empty( max_nEls, 'd' )

    # Fill cntVecMx, totalCntVec for all elements (all subtrees)
    nEls = evalTree.num_final_elements()
    cntVecMx_all = _np.empty( nEls,'d')
    totalCntVec_all = _np.empty(nEls, 'd')

    ds_subtree_gatestring_list = _lt.find_replace_tuple_list(
        gatestring_list, gateLabelAliases)
    for (i,gateStr) in enumerate(ds_subtree_gatestring_list):
        cnts = dataset[gateStr].counts
        totalCntVec_all[ lookup[i] ] = sum(cnts.values()) #dataset[gateStr].total
        cntVecMx_all[ lookup[i] ] = [ cnts.get(x,0) for x in outcomes_lookup[i] ]

    tStart = _time.time()

    #Loop over subtrees
    for iSubTree in mySubTreeIndices:
        evalSubTree = subtrees[iSubTree]
        sub_nEls = evalSubTree.num_final_elements()

        if evalSubTree.myFinalElsToParentFinalElsMap is not None:
            #Then `evalSubTree` is a nontrivial sub-tree and its .spamtuple_indices
            # will index the *parent's* final index array space, which we
            # usually want but NOT here, where we fill arrays just big
            # enough for each subtree separately - so re-init spamtuple_indices
            evalSubTree = evalSubTree.copy()
            evalSubTree.recompute_spamtuple_indices(bLocal=True)

        # Create views into pre-allocated memory
        probs  =  probs_mem[0:sub_nEls]

        # Take portions of count arrays for this subtree
        cntVecMx = cntVecMx_all[ evalSubTree.final_element_indices(evalTree) ]
        totalCntVec = totalCntVec_all[ evalSubTree.final_element_indices(evalTree) ]
        assert(len(cntVecMx) == len(probs))

        #compute pos_probs separately
        gateset.bulk_fill_probs(probs, evalSubTree,
                                clipTo=probClipInterval, check=check,
                                comm=mySubComm)
        pos_probs = _np.where(probs < min_p, min_p, probs)

        nCols = gateset.num_params()
        blocks1 = _mpit.slice_up_range(nCols, rowParts)
        blocks2 = _mpit.slice_up_range(nCols, colParts)
        sliceTupList_all = list(_itertools.product(blocks1,blocks2))
        #cull out lower triangle blocks, which have no overlap with
        # the upper triangle of the hessian
        sliceTupList = [ (slc1,slc2) for slc1,slc2 in sliceTupList_all
                         if slc1.start <= slc2.stop ]

        loc_iBlks, blkOwners, blkComm = \
            _mpit.distribute_indices(list(range(len(sliceTupList))), mySubComm)
        mySliceTupList = [ sliceTupList[i] for i in loc_iBlks ]
       
        subtree_hessian = _np.zeros( (nP,nP), 'd')

        k,kmax = 0,len(mySliceTupList)
        for (slice1,slice2,hprobs,dprobs12) in gateset.bulk_hprobs_by_block(
            evalSubTree, mySliceTupList, True, blkComm):
            rank = comm.Get_rank() if (comm is not None) else 0

            if verbosity > 3 or (verbosity == 3 and rank == 0):
                iSub = mySubTreeIndices.index(iSubTree)
                print("rank %d: %gs: block %d/%d, sub-tree %d/%d, sub-tree-len = %d"
                      % (rank,_time.time()-tStart,k,kmax,iSub,
                         len(mySubTreeIndices), len(evalSubTree)))            
                _sys.stdout.flush(); k += 1

            subtree_hessian[slice1,slice2] = \
                hessian_from_hprobs(hprobs, dprobs12, cntVecMx,
                                        totalCntVec, pos_probs)
                #NOTE: hessian_from_hprobs MAY modify hprobs and dprobs12

        #Gather columns from different procs and add to running final hessian
        #_mpit.gather_slices_by_owner(slicesIOwn, subtree_hessian,[], (0,1), mySubComm)
        _mpit.gather_slices(sliceTupList, blkOwners, subtree_hessian,[], (0,1), mySubComm)
        final_hessian += subtree_hessian

    #gather (add together) final_hessians from different processors
    if comm is not None and len(set(subTreeOwners.values())) > 1:
        if comm.Get_rank() not in subTreeOwners.values(): 
            # this proc is not the "owner" of its subtrees and should not send a contribution to the sum
            final_hessian[:,:] = 0.0 #zero out hessian so it won't contribute
        final_hessian = comm.allreduce(final_hessian)
        
    #copy upper triangle to lower triangle (we only compute upper)
    for i in range(final_hessian.shape[0]):
        for j in range(i+1,final_hessian.shape[1]):
            final_hessian[j,i] = final_hessian[i,j]

    return final_hessian # (N,N)


def logl_approximate_hessian(gateset, dataset, gatestring_list=None,
                             minProbClip=1e-6, probClipInterval=(-1e6,1e6), radius=1e-4,
                             poissonPicture=True, check=False, comm=None,
                             memLimit=None, gateLabelAliases=None, verbosity=0):
    """
    An approximate Hessian of the log-likelihood function.

    An approximation to the true Hessian is computed using just the Jacobian
    (and *not* the Hessian) of the probabilities w.r.t. the gate set
    parameters.  Let `J = d(probs)/d(params)` and denote the Hessian of the
    log-likelihood w.r.t. the probabilities as `d2(logl)/dprobs2` (a *diagonal*
    matrix indexed by the term, i.e. probability, of the log-likelihood). Then
    this function computes:

    `H = J * d2(logl)/dprobs2 * J.T`

    Which simply neglects the `d2(probs)/d(params)2` terms of the true Hessian.
    Since this curvature is expected to be small at the MLE point, this
    approximation can be useful for computing approximate error bars.

    Parameters
    ----------
    gateset : GateSet
        Gateset of parameterized gates (including SPAM)

    dataset : DataSet
        Probability data

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the log-likelihood
        sum.  Default value of None implies all the gate strings in dataset
        should be used.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by gatesets during MLEGST's
        search for an optimal gateset (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    evalTree : evaluation tree, optional
        given by a prior call to bulk_evaltree for the same gatestring_list.
        Significantly speeds up evaluation of log-likelihood derivatives, even
        more so when accompanied by countVecMx (see below).  Defaults to None.

    poissonPicture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    verbosity : int, optional
        How much detail to print to stdout.

    Returns
    -------
    numpy array
      array of shape (M,M), where M is the length of the vectorized gateset.
    """

    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    C = 1.0/1024.0**3; nP = gateset.num_params()
    persistentMem = 8*nP**2 + 8*len(gatestring_list)*(nP+1) # in bytes

    if memLimit is not None and memLimit < persistentMem:
        raise MemoryError("DLogL Memory limit (%g GB) is " % (memLimit*C) +
                          "< memory required to hold final results (%g GB)"
                          % (persistentMem*C))


    #OLD: evalTree,lookup,outcomes_lookup = gateset.bulk_evaltree(gatestring_list)
    mlim = None if (memLimit is None) else memLimit-persistentMem
    dstree = dataset if (gateLabelAliases is None) else None #Note: compile_gatestrings doesn't support aliased dataset (yet)
    evalTree, blkSize, _, lookup, outcomes_lookup = \
        gateset.bulk_evaltree_from_resources(
            gatestring_list, comm, mlim, "deriv", ['bulk_fill_dprobs'],
            dstree, verbosity)

    a = radius # parameterizes "roundness" of f == 0 terms
    min_p = minProbClip

    #  Allocate persistent memory
    #hessian = _np.zeros( (nP,nP), 'd') # allocated below by assignment
    nEls = evalTree.num_final_elements()
    probs = _np.empty( nEls, 'd' )
    dprobs = _np.empty( (nEls,nP), 'd' )

    ds_gatestring_list = _lt.find_replace_tuple_list(
        gatestring_list, gateLabelAliases)

    cntVecMx = _np.empty(nEls, 'd' )
    totalCntVec = _np.empty(nEls, 'd' )
    for (i,gateStr) in enumerate(ds_gatestring_list):
        cnts = dataset[gateStr].counts
        totalCntVec[ lookup[i] ] = sum(cnts.values()) #dataset[gateStr].total
        cntVecMx[ lookup[i] ] = [ cnts.get(x,0) for x in outcomes_lookup[i] ]

    gateset.bulk_fill_dprobs(dprobs, evalTree, prMxToFill=probs,
                             clipTo=probClipInterval, check=check, comm=comm,
                             wrtBlockSize=blkSize) # FUTURE: set gatherMemLimit=?

    pos_probs = _np.where(probs < min_p, min_p, probs)

    #Note: these approximate-hessian formula are similar to (but simpler than) the
    # computations done by the `hessian_from_probs` functions in `logl_hessian(...)`.
    # They compute just the hessian of the log-likelihood w.r.t. the probabilities -
    # which correspond to just the `dprobs12_coeffs` variable of the aforementioned
    # functions.  This is so b/c in this case the "dp1" and "dp2" terms are delta
    # functions and "hp==0" (b/c the "params" here are just the probabilities
    # themselves) - so only the X*dp1*dp2 terms survive the general expressions
    # found above.
    if poissonPicture:
        totCnts = totalCntVec  #shorthand
        S = cntVecMx / min_p - totCnts # slope term that is derivative of logl at min_p
        S2 = -0.5 * cntVecMx / (min_p**2)          # 2nd derivative of logl term at min_p

        dprobs12_coeffs = \
            _np.where(probs < min_p, 2*S2, -cntVecMx / pos_probs**2)
        zfc = _np.where(probs >= a, 0.0, -totCnts*((-2.0/a**2)*probs+2.0/a))
        dprobs12_coeffs = _np.where(cntVecMx == 0, zfc, dprobs12_coeffs)
          # a 1D array of the diagonal of d2(logl)/dprobs2; shape = (nEls,)

    else:
        S = cntVecMx / min_p # slope term that is derivative of logl at min_p
        S2 = -0.5 * cntVecMx / (min_p**2) # 2nd derivative of logl term at min_p
        dprobs12_coeffs = \
            _np.where(probs < min_p, 2*S2, -cntVecMx / pos_probs**2)
        dprobs12_coeffs = _np.where(cntVecMx == 0, 0.0, dprobs12_coeffs)
            # a 1D array of the diagonal of d2(logl)/dprobs2; shape = (nEls,)

    # In notation in docstring:
    # J = dprobs.T (shape nEls,nP)
    # diagonal of d2(logl)/dprobs2 = dprobs12_coeffs (var name kept to preserve
    #                similarity w/functions in logl_hessian)
    # So H = J * d2(logl)/dprobs2 * J.T becomes:
    hessian = _np.dot( dprobs.T, dprobs12_coeffs[:,None]*dprobs )
    return hessian


#@smart_cached
def logl_max(gateset, dataset, gatestring_list=None, poissonPicture=True,
             check=False, gateLabelAliases=None, evaltree_cache=None):
    """
    The maximum log-likelihood possible for a DataSet.  That is, the
    log-likelihood obtained by a maximal model that can fit perfectly
    the probability of each gate string.

    Parameters
    ----------
    gateset : GateSet
        the gateset, used only for gate string compilation

    dataset : DataSet
        the data set to use.

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the max-log-likelihood
        sum.  Default value of None implies all the gate strings in dataset should
        be used.

    poissonPicture : boolean, optional
        Whether the Poisson-picture maximum log-likelihood should be returned.

    check : boolean, optional
        Whether additional check is performed which computes the max logl another
        way an compares to the faster method.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    evaltree_cache : dict, optional
        A dictionary which server as a cache for the computed EvalTree used
        in this computation.  If an empty dictionary is supplied, it is filled
        with cached values to speed up subsequent executions of this function
        which use the *same* `gateset` and `gatestring_list`.

    Returns
    -------
    float
    """
    maxLogLTerms = logl_max_terms(gateset, dataset, gatestring_list,
                                  poissonPicture, gateLabelAliases,
                                  evaltree_cache)
    
    # maxLogLTerms[iSpamLabel,iGateString] contains all logl-upper-bound contributions
    maxLogL = _np.sum(maxLogLTerms) # sum over *all* dimensions

    if check:
        L = 0
        for gateString in gatestring_list:
            dsRow = dataset[gateString]
            N = dsRow.total #sum of counts for all outcomes (all spam labels)
            for n in dsRow.counts.values():
                f = n / N
                if f < TOL and n == 0: continue # 0 * log(0) == 0
                if poissonPicture:
                    L += n * _np.log(f) - N * f
                else:
                    L += n * _np.log(f)
        if not _np.isclose(maxLogL,L):
            _warnings.warn("Log-likelihood upper bound mismatch: %g != %g (diff=%g)" % \
                               (maxLogL, L, maxLogL-L))

    return maxLogL

#@smart_cached
def logl_max_terms(gateset, dataset, gatestring_list=None,
                   poissonPicture=True, gateLabelAliases=None,
                   evaltree_cache=None):
    """
    The vector of maximum log-likelihood contributions for each gate string,
    aggregated over outcomes.

    Parameters
    ----------
    This function takes the same arguments as :func:`logl_max` except it
    doesn't perform the final sum over gate sequences and SPAM labels.

    Returns
    -------
    numpy.ndarray
        Array of length either `len(gatestring_list)` or `len(dataset.keys())`.
        Values are the maximum log-likelihood contributions of the corresponding
        gate string aggregated over outcomes.
    """

    if evaltree_cache and 'evTree' in evaltree_cache:
        evalTree = evaltree_cache['evTree']
        lookup = evaltree_cache['lookup']
        outcomes_lookup = evaltree_cache['outcomes_lookup']

        #construct raw_dict & nEls from tree (holds keys & vals separately)
        #tree_gatestring_list = evalTree.generate_gatestring_list()
        # Note: this is != gatestring_list, as the tree hold *compiled* gatestrings
        raw_dict = _OrderedDict(list(zip(gatestring_list,
                                         evalTree.compiled_gatestring_spamTuples)))
        nEls = evalTree.num_final_elements()
    else:
        if gatestring_list is None:
            gatestring_list = list(dataset.keys())

        raw_dict, lookup, outcomes_lookup, nEls = \
            gateset.compile_gatestrings(gatestring_list, dataset)
        #Note: we don't actually need an evaltree, so we
        # won't make one here and so won't fill an empty
        # evaltree_cache.
        
    gatestring_list = _lt.find_replace_tuple_list(
        gatestring_list, gateLabelAliases)

    if evaltree_cache and 'cntVecMx' in evaltree_cache:
        countVecMx = evaltree_cache['cntVecMx']
        totalCntVec = evaltree_cache['totalCntVec']
    else:
        countVecMx = _np.empty(nEls, 'd' )
        totalCntVec = _np.empty(nEls, 'd' )
        for (i,gateStr) in enumerate(gatestring_list):
            cnts = dataset[gateStr].counts
            totalCntVec[ lookup[i] ] = sum(cnts.values()) #dataset[gateStr].total
            countVecMx[ lookup[i] ] = [ cnts.get(x,0) for x in outcomes_lookup[i] ]

        #could add to cache, but we don't have option of gateStringWeights
        # here yet, so let's be conservative and not do this:
        #if evaltree_cache is not None:
        #    evaltree_cache['cntVecMx'] = countVecMx
        #    evaltree_cache['totalCntVec'] = totalCntVec

        
    freqs = countVecMx / totalCntVec
    freqs_nozeros = _np.where(countVecMx == 0, 1.0, freqs) # set zero freqs to 1.0 so np.log doesn't complain

    if poissonPicture:
        maxLogLTerms = countVecMx * ( _np.log(freqs_nozeros) - 1.0 )
    else:
        maxLogLTerms = countVecMx * _np.log(freqs_nozeros)

    maxLogLTerms[ countVecMx == 0 ] = 0.0 # set 0 * log(0) terms explicitly to zero since numpy doesn't know this limiting behavior

    #Aggregate over outcomes:
    # maxLogLTerms[iElement] contains all logl-upper-bound contributions
    # terms[iGateString] wiil contain contributions for each original gate
    # string (aggregated over outcomes)    
    nGateStrings = len(gatestring_list)
    terms = _np.empty(nGateStrings , 'd')
    for i in range(nGateStrings):
        terms[i] = _np.sum( maxLogLTerms[lookup[i]], axis=0 )
    return terms


def forbidden_prob(gateset, dataset):
    """
    Compute the sum of the out-of-range probabilities
    generated by gateset, using only those gate strings
    contained in dataset.  Non-zero value indicates
    that gateset is not in XP for the supplied dataset.

    Parameters
    ----------
    gateset : GateSet
        gate set to generate probabilities.

    dataset : DataSet
        data set to obtain gate strings.  Dataset counts are
        used to check for zero or all counts being under a
        single spam label, in which case out-of-bounds probabilities
        are ignored because they contribute zero to the logl sum.

    Returns
    -------
    float
        sum of the out-of-range probabilities.
    """
    forbidden_prob = 0

    for gs,dsRow in dataset.items():
        probs = gateset.probs(gs)
        for (spamLabel,p) in probs.items():
            if p < TOL:
                if round(dsRow[spamLabel]) == 0: continue #contributes zero to the sum
                else: forbidden_prob += abs(TOL-p) + TOL
            elif p > 1-TOL:
                if round(dsRow[spamLabel]) == dsRow.total: continue #contributes zero to the sum
                else: forbidden_prob += abs(p-(1-TOL)) + TOL


    return forbidden_prob

def prep_penalty(rhoVec, basis):
    """
    Penalty assigned to a state preparation (rho) vector rhoVec.  State
      preparation density matrices must be positive semidefinite
      and trace == 1.  A positive return value indicates an
      these criteria are not met and the rho-vector is invalid.

    Parameters
    ----------
    rhoVec : numpy array
        rho vector array of shape (N,1) for some N.

    basis : {"std", "gm", "pp", "qt"}
        The abbreviation for the basis used to interpret rhoVec
        ("gm" = Gell-Mann, "pp" = Pauli-product, "std" = matrix unit,
         "qt" = qutrit, or standard).

    Returns
    -------
    float
    """
    # rhoVec must be positive semidefinite and trace = 1
    rhoMx = _bt.vec_to_stdmx(_np.asarray(rhoVec),basis)
    evals = _np.linalg.eigvals( rhoMx )  #could use eigvalsh, but wary of this since eigh can be wrong...
    sumOfNeg = sum( [ -ev.real for ev in evals if ev.real < 0 ] )
    tracePenalty = abs(rhoVec[0,0]-(1.0/_np.sqrt(rhoMx.shape[0])))
      # 0th el is coeff of I(dxd)/sqrt(d) which has trace sqrt(d)
    #print "Sum of neg = ",sumOfNeg  #DEBUG
    #print "Trace Penalty = ",tracePenalty  #DEBUG    
    return sumOfNeg +  tracePenalty

def effect_penalty(EVec, basis):
    """
    Penalty assigned to a POVM effect vector EVec. Effects
      must have eigenvalues between 0 and 1.  A positive return
      value indicates this criterion is not met and the E-vector
      is invalid.

    Parameters
    ----------
    EVec : numpy array
         effect vector array of shape (N,1) for some N.

    basis : {"std", "gm", "pp", "qt"}
        The abbreviation for the basis used to interpret EVec
        ("gm" = Gell-Mann, "pp" = Pauli-product, "std" = matrix unit,
         "qt" = qutrit, or standard).

    Returns
    -------
    float
    """
    # EVec must have eigenvalues between 0 and 1
    EMx = _bt.vec_to_stdmx(_np.asarray(EVec),basis)
    evals = _np.linalg.eigvals( EMx )  #could use eigvalsh, but wary of this since eigh can be wrong...
    sumOfPen = 0
    for ev in evals:
        if ev.real < 0: sumOfPen += -ev.real
        if ev.real > 1: sumOfPen += ev.real-1.0
    return sumOfPen

def cptp_penalty(gateset, include_spam_penalty=True):
    """
    The sum of all negative Choi matrix eigenvalues, and
      if include_spam_penalty is True, the rho-vector and
      E-vector penalties of gateset.  A non-zero value
      indicates that the gateset is not CPTP.

    Parameters
    ----------
    gateset : GateSet
        the gate set to compute CPTP penalty for.

    include_spam_penalty : bool, optional
        if True, also test gateset for invalid SPAM
        operation(s) and return sum of CPTP penalty
        with rhoVecPenlaty(...) and effect_penalty(...)
        for each rho and E vector.

    Returns
    -------
    float
        CPTP penalty (possibly with added spam penalty).
    """
    ret = _jam.sum_of_negative_choi_evals(gateset)
    if include_spam_penalty:
        b = gateset.basis
        ret += sum([ prep_penalty(r,b) for r in gateset.preps.values() ])
        ret += sum([ effect_penalty(e,b) for povm in gateset.povms.values()
                     for e in povm.values() ])
    return ret

#@smart_cached
def two_delta_loglfn(N, p, f, minProbClip=1e-6, poissonPicture=True):
    """
    Term of the 2*[log(L)-upper-bound - log(L)] sum corresponding
     to a single gate string and spam label.

    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClip : float, optional
        Minimum probability clip point to avoid evaluating
        log(number <= zero)

    poissonPicture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    Returns
    -------
    float or numpy array
    """
    #TODO: change this function to handle nan's in the inputs without warnings, since
    # fiducial pair reduction may pass inputs with nan's legitimately and the desired
    # behavior is to just let the nan's pass through to nan's in the output.
    cp  = _np.clip(p,minProbClip,1e10) #effectively no upper bound
    
    nan_indices = _np.isnan(f) # get indices of invalid entries
    if not _np.isscalar(f): f[nan_indices] = 0.0 
      #set nan's to zero to avoid RuntimeWarnings (invalid value)

    zf  = _np.where(f < 1e-10, 0.0, f) #set zero-freqs to zero
    nzf = _np.where(f < 1e-10, 1.0, f) #set zero-freqs to one -- together
                                       # w/above line makes 0 * log(0) == 0    
    if not _np.isscalar(f): 
        zf[nan_indices] = _np.nan  #set nan indices back to nan
        nzf[nan_indices] = _np.nan #set nan indices back to nan

    if poissonPicture:
        return 2 * (N * zf * _np.log(nzf/cp) - N * (f-cp))
    else:
        return 2 * N * zf * _np.log(nzf/cp)






##############################################################################################
#   FUNCTIONS FOR HESSIAN ANALYSIS (which take derivatives of the log(likelihood) function)  #
##############################################################################################


#def dlogl_analytic(gateset, dataset):
#    nP = gateset.num_params()
#    result = _np.zeros([1,nP])
#    dPmx = dpr_plus(gateset, [gateString for gateString in dataset])
#
#    for (k,d) in enumerate(dataset.values()):
#        p = gateset.PrPlus(d.gateString)
#        if _np.fabs(p) < TOL and round(d.nPlus) == 0: continue
#        if _np.fabs(p - 1) < TOL and round(d.nMinus) == 0: continue
#
#        for i in range(nP):
#            #pre = ((1-p)*d.nPlus - p*d.nMinus) / (p*(1-p))
#            #print "%d: Pre(%s) = " % (i,d.gateString), pre, "  (p = %g, np = %g)" % (p, d.nPlus)
#            result[0,i] += ((1-p)*d.nPlus - p*d.nMinus) / (p*(1-p)) * dPmx[i,k]
#
#    return result
#
#
#def dlogl_finite_diff(gateset, dataset):
#    return numerical_deriv(logl, gateset, dataset, 1)
#
#def logl_hessian_finite_diff(gateset, dataset):
#    return numerical_deriv(dlogl_finite_diff, gateset, dataset, gateset.num_params())
#
#def logl_hessian_at_ml(gateset, gatestrings, nSamples):
#    return nSamples * logl_hessian_at_ML_per_sample(gateset, gatestrings)
#
#def logl_hessian_at_ML_per_sample(gateset, gatestrings):
#    nP = gateset.num_params()
#    result = _np.zeros([nP,nP])
#
#    dPmx = dpr_plus(gateset, gatestrings)
#
#    for (k,s) in enumerate(gatestrings):
#        p = gateset.PrPlus(s)
#        if _np.fabs(p) < TOL: continue
#        if _np.fabs(p - 1) < TOL: continue
#        for i in range(nP):
#            for j in range(nP):
#                result[i,j] += -1.0/(p*(1-p)) * dPmx[i,k] * dPmx[j,k]
#
#    return result
#
#
#
#def dpr_plus(gateset, gatestrings):
#    DELTA = 1e-7
#    nP = gateset.num_params()
#    nGateStrings = len(gatestrings)
#    result = _np.zeros([nP,nGateStrings])
#
#    for (j,s) in enumerate(gatestrings):
#        fMid = gateset.PrPlus(s)
#
#        for i in range(nP):
#            gs = gateset.copy()
#            gs.add_to_param(i,DELTA)
#            fRight = gs.PrPlus(s)
#            gs.add_to_param(i,-2*DELTA)
#            fLeft = gs.PrPlus(s)
#
#            if fRight is None and fLeft is None:
#                raise ValueError("Cannot take derivative - both sides are out of bounds!")
#            if fRight is None:
#                dP = (fMid - fLeft) / DELTA
#            elif fLeft is None:
#                dP = (fRight - fMid) / DELTA
#            else:
#                dP = (fRight - fLeft) / (2*DELTA)
#
#            result[i,j] = dP
#
#    return result
#
#
#def numerical_deriv(fnToDifferentiate, gateset, dataset, resultLen):
#    DELTA = 1e-6
#    nP = gateset.num_params()
#    result = _np.zeros([resultLen,nP])
#
#    fMid = fnToDifferentiate(gateset, dataset)
#    if fMid is None: return None
#
#    for i in range(nP):
#        gs = gateset.copy()
#        gs.add_to_param(i,DELTA)
#        fRight = fnToDifferentiate(gs, dataset)
#
#        gs = gateset.copy()
#        gs.add_to_param(i,-DELTA)
#        fLeft = fnToDifferentiate(gs, dataset)
#
#        #print "DEBUG: %d: l,m,r = " % i,(fLeft,fMid,fRight)
#        if fRight is None and fLeft is None:
#            raise ValueError("numerical_deriv cannot take derivative - both sides are out of bounds!")
#
#        if fRight is None:
#            df = (fMid - fLeft) / DELTA
#        elif fLeft is None:
#            df = (fRight - fMid) / DELTA
#        else:
#            df = (fRight - fLeft) / (2*DELTA)
#
#        #print "DEBUG: df(%d) = " % i,df
#        result[:,i] = _np.transpose(df)
#
#    return result

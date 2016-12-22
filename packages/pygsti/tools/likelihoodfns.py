from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions related to computation of the log-likelihood."""

import numpy as _np
import warnings as _warnings
import itertools as _itertools
#import time as _time
from . import basistools as _bt
from . import jamiolkowski as _jam
from . import mpitools as _mpit
from . import slicetools as _slct

TOL = 1e-20


# Functions for computing the log liklihood function and its derivatives

def create_count_vec_dict(spamLabels, dataset, gatestring_list):
    """
    Create a count-vector dictionary that is useful for speeding up multiple
    evaluations of logl(...).   The returned dictionary has keys that are
    spam labels and values that are numpy vectors containing the dataset counts
    for that spam label for each gate string in gatestring_list.

    Parameters
    ----------
    spamLabels : list of strings
        List of the spam labels to include as keys in returned dict.

    dataset : DataSet
        The dataset to extract counts from.

    gatestring_list : list of (tuples or GateStrings)
        List of the gate strings to extract counts for, which
        determines the ordering of the counts within each dictionary
        value.

    Returns
    -------
    dict
        as described above.
    """
    countVecDict = { }
    for spamLabel in spamLabels:
        countVecDict[spamLabel] = _np.array( [ dataset[gs][spamLabel] for gs in gatestring_list ] )
    return countVecDict


def fill_count_vecs(mxToFill, spam_label_rows, dataset, gatestring_list):
    """
    Fill a matrix of counts that is useful for speeding up multiple
    evaluations of logl(...).   Identical to create_count_vec_dict except
    counts for a given spam label are placed into a row of mxToFill
    instead of into a returned dictionary.

    Parameters
    ----------
    mxToFill : numpy ndarray
        an already-allocated KxS numpy array, where K is larger
        than the maximum value in spam_label_rows and S is equal
        to the number of gate strings (lenght of gatestring_list).

    spam_label_rows : dictionary
        a dictionary with keys == spam labels and values which
        are integer row indices into mxToFill, specifying the
        correspondence between rows of mxToFill and spam labels.

    dataset : DataSet
        The dataset to extract counts from.

    gatestring_list : list of (tuples or GateStrings)
        List of the gate strings to extract counts for, which
        determines the ordering of the counts within each dictionary
        value.

    Returns
    -------
    None
    """
    for spamLabel,iRow in spam_label_rows.items():
        mxToFill[iRow,:] = [ dataset[gs][spamLabel] for gs in gatestring_list ]




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
 # Where lamba_{i,sl} := p_{i,sl}/N[i] is a rate, i indexes the gate string,
 #  and sl indexes the spam label.  N[i] is the total counts for the i-th gatestring, and
 #  so sum_{sl} N_{i,sl} == N[i]. We can ignore the p-independent N_j! and take the log:
 #
 # log L = sum_{i,sl} N_{i,sl} log(N[i]*p_{i,sl}) - N[i]*p_{i,sl}
 #       = sum_{i,sl} N_{i,sl} log(p_{i,sl}) - N[i]*p_{i,sl}   (where we ignore the p-independent log(N[i]) terms)
 #
 #   after patching (linear extrapolation below min_p and "softening" f == 0 terms w/cubit below radius "a"):
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
 # For cubit interpolation, use function F(p) (derived by Robin: match value, 1st-deriv, 2nd-deriv at p == r, and require min at p == 0):
 #  Given a radius r << 1 (but r>0):
 #   F(p) = piecewise{ if( p>r ) then p; else -(1/3)*p^3/r^2 + p^2/r + (1/3)*r }
 #  OLD: quadratic that doesn't match 2nd-deriv:
 #   F(p) = piecewise{ if( p>r ) then p; else (r-p)^2/(2*r) + p }




def logl(gateset, dataset, gatestring_list=None,
         minProbClip=1e-6, probClipInterval=(-1e6,1e6), radius=1e-4,
         evalTree=None, countVecMx=None, poissonPicture=True, check=False):
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

    countVecMx : numpy array, optional
      Two-dimensional numpy array whose rows correspond to the gate's spam
      labels (i.e. gateset.get_spam_labels()).  Each row is  contains the
      dataset counts for that spam label for each gate string in gatestring_list.
      Use fill_count_vecs(...) to generate this quantity once for multiple
      evaluations of the log-likelihood function which use the same dataset.

    poissonPicture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    check : boolean, optional
      If True, perform extra checks within code to verify correctness.  Used
      for testing, and runs much slower when True.

    Returns
    -------
    float
        The log likelihood
    """

    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    spamLabels = gateset.get_spam_labels() #this list fixes the ordering of the spam labels
    spam_lbl_rows = { sl:i for (i,sl) in enumerate(spamLabels) }

    probs = _np.empty( (len(spamLabels),len(gatestring_list)), 'd' )
    if countVecMx is None:
        countVecMx = _np.empty( (len(spamLabels),len(gatestring_list)), 'd' )
        fill_count_vecs(countVecMx, spam_lbl_rows, dataset, gatestring_list)

    totalCntVec = _np.sum(countVecMx, axis=0)

    #freqs = countVecMx / totalCntVec[None,:]
    #freqs_nozeros = _np.where(countVecMx == 0, 1.0, freqs) # set zero freqs to 1.0 so np.log doesn't complain
    #freqTerm = countVecMx * ( _np.log(freqs_nozeros) - 1.0 )
    #freqTerm[ countVecMx == 0 ] = 0.0 # set 0 * log(0) terms explicitly to zero since numpy doesn't know this limiting behavior

    a = radius # parameterizes "roundness" of f == 0 terms
    min_p = minProbClip

    if evalTree is None:
        evalTree = gateset.bulk_evaltree(gatestring_list)

    gateset.bulk_fill_probs(probs, spam_lbl_rows, evalTree, probClipInterval, check)
    pos_probs = _np.where(probs < min_p, min_p, probs)

    if poissonPicture:
        S = countVecMx / min_p - totalCntVec[None,:] # slope term that is derivative of logl at min_p
        S2 = -0.5 * countVecMx / (min_p**2)          # 2nd derivative of logl term at min_p
        v = countVecMx * _np.log(pos_probs) - totalCntVec[None,:]*pos_probs # dims K x M (K = nSpamLabels, M = nGateStrings)
        v = _np.minimum(v,0)  #remove small positive elements due to roundoff error (above expression *cannot* really be positive)
        v = _np.where( probs < min_p, v + S*(probs - min_p) + S2*(probs - min_p)**2, v) #quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where( countVecMx == 0, -totalCntVec[None,:] * _np.where(probs >= a, probs, (-1.0/(3*a**2))*probs**3 + probs**2/a + a/3.0), v)
           #special handling for f == 0 poissonPicture terms using quadratic rounding of function with minimum: max(0,(a-p))^2/(2a) + p

    else: #(the non-poisson picture requires that the probabilities of the spam labels for a given string are constrained to sum to 1)
        S = countVecMx / min_p               # slope term that is derivative of logl at min_p
        S2 = -0.5 * countVecMx / (min_p**2)  # 2nd derivative of logl term at min_p
        v = countVecMx * _np.log(pos_probs) # dims K x M (K = nSpamLabels, M = nGateStrings)
        v = _np.minimum(v,0)  #remove small positive elements due to roundoff error (above expression *cannot* really be positive)
        v = _np.where( probs < min_p, v + S*(probs - min_p) + S2*(probs - min_p)**2, v) #quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where( countVecMx == 0, 0.0, v)

    #DEBUG
    #print "num clipped = ",_np.sum(probs < min_p)," of ",probs.shape
    #print "min/max probs = ",min(probs.flatten()),",",max(probs.flatten())
    #for i in range(v.shape[1]):
    #    print "%d %.0f (%f) %.0f (%g)" % (i,v[0,i],probs[0,i],v[1,i],probs[1,i])

    # v[iSpamLabel,iGateString] contains all logl contributions
    return _np.sum(v) # sum over *all* dimensions




def logl_jacobian(gateset, dataset, gatestring_list=None,
                  minProbClip=1e-6, probClipInterval=(-1e6,1e6), radius=1e-4,
                  evalTree=None, countVecMx=None, poissonPicture=True, check=False):
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

    countVecMx : numpy array, optional
      Two-dimensional numpy array whose rows correspond to the gate's spam
      labels (i.e. gateset.get_spam_labels()).  Each row is  contains the
      dataset counts for that spam label for each gate string in gatestring_list.
      Use fill_count_vecs(...) to generate this quantity once for multiple
      evaluations of the log-likelihood function which use the same dataset.

    poissonPicture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    Returns
    -------
    numpy array
      array of shape (M,), where M is the length of the vectorized gateset.
    """

    nP = gateset.num_params()
    jac = _np.zeros([1,nP])

    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    spamLabels = gateset.get_spam_labels() #this list fixes the ordering of the spam labels
    spam_lbl_rows = { sl:i for (i,sl) in enumerate(spamLabels) }

    if countVecMx is None:
        countVecMx = _np.empty( (len(spamLabels),len(gatestring_list)), 'd' )
        fill_count_vecs(countVecMx, spam_lbl_rows, dataset, gatestring_list)

    probs = _np.empty( (len(spamLabels),len(gatestring_list)), 'd' )
    dprobs = _np.empty( (len(spamLabels),len(gatestring_list),nP), 'd' )
    totalCntVec = _np.sum(countVecMx, axis=0)

    #freqs = cntVecMx / totalCntVec[None,:]
    #freqs_nozeros = _np.where(cntVecMx == 0, 1.0, freqs) # set zero freqs to 1.0 so np.log doesn't complain
    #freqTerm = cntVecMx * ( _np.log(freqs_nozeros) - 1.0 )
    #freqTerm[ cntVecMx == 0 ] = 0.0 # set 0 * log(0) terms explicitly to zero since numpy doesn't know this limiting behavior
    #minusCntVecMx = -1.0 * cntVecMx

    a = radius # parameterizes "roundness" of f == 0 terms
    min_p = minProbClip

    if evalTree is None:
        evalTree = gateset.bulk_evaltree(gatestring_list)

    gateset.bulk_fill_dprobs(dprobs, spam_lbl_rows, evalTree,
                            prMxToFill=probs, clipTo=probClipInterval, check=check)

    pos_probs = _np.where(probs < min_p, min_p, probs)

    if poissonPicture:
        S = countVecMx / min_p - totalCntVec[None,:] # slope term that is derivative of logl at min_p
        S2 = -0.5 * countVecMx / (min_p**2)          # 2nd derivative of logl term at min_p
        v = countVecMx * _np.log(pos_probs) - totalCntVec[None,:]*pos_probs # dims K x M (K = nSpamLabels, M = nGateStrings)
        v = _np.minimum(v,0)  #remove small positive elements due to roundoff error (above expression *cannot* really be positive)
        v = _np.where( probs < min_p, v + S*(probs - min_p) + S2*(probs - min_p)**2, v) #quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where( countVecMx == 0, -totalCntVec[None,:] * _np.where(probs >= a, probs, (-1.0/(3*a**2))*probs**3 + probs**2/a + a/3.0), v)
           #special handling for f == 0 poissonPicture terms using quadratic rounding of function with minimum: max(0,(a-p))^2/(2a) + p

        dprobs_factor_pos = (countVecMx / pos_probs - totalCntVec[None,:])
        dprobs_factor_neg = S + 2*S2*(probs - min_p)
        dprobs_factor_zerofreq = -totalCntVec[None,:] * _np.where( probs >= a, 1.0, (-1.0/a**2)*probs**2 + 2*probs/a)
        dprobs_factor = _np.where( probs < min_p, dprobs_factor_neg, dprobs_factor_pos)
        dprobs_factor = _np.where( countVecMx == 0, dprobs_factor_zerofreq, dprobs_factor )
        jac = dprobs * dprobs_factor[:,:,None] # (K,M,N) * (K,M,1)   (N = dim of vectorized gateset)


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
        jac = dprobs * dprobs_factor[:,:,None] # (K,M,N) * (K,M,1)   (N = dim of vectorized gateset)

    # jac[iSpamLabel,iGateString,iGateSetParam] contains all d(logl)/d(gatesetParam) contributions
    return _np.sum(jac, axis=(0,1)) # sum over spam label and gate string dimensions


def logl_hessian(gateset, dataset, gatestring_list=None, minProbClip=1e-6,
                 probClipInterval=(-1e6,1e6), radius=1e-4, poissonPicture=True,
                 check=False, comm=None, memLimit=None, verbosity=0):
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

    spamLabels = gateset.get_spam_labels() #fixes the ordering of the spam labels
    spam_lbl_rows = { sl:i for (i,sl) in enumerate(spamLabels) }
    
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
    evalTree, blkSize1, blkSize2 = gateset.bulk_evaltree_from_resources(
        gatestring_list, comm, mlim, "deriv", ['bulk_hprobs_by_block'],
        verbosity)
    
    rowParts = int(round(nP / blkSize1)) if (blkSize1 is not None) else 1
    colParts = int(round(nP / blkSize2)) if (blkSize2 is not None) else 1

    a = radius # parameterizes "roundness" of f == 0 terms
    min_p = minProbClip

    if poissonPicture:
        #NOTE: hessian_from_hprobs MAY modify hprobs and dprobs12 (to save mem)
        def hessian_from_hprobs(hprobs, dprobs12, cntVecMx, totalCntVec, pos_probs):
            # Notation:  (K=#spam, M=#strings, N=#wrtParams1, N'=#wrtParams2 )
            totCnts = totalCntVec[None,:]  #shorthand (just a view)
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
            hprobs *= hprobs_coeffs[:,:,None,None]
            dprobs12 *= dprobs12_coeffs[:,:,None,None]
            hessian = dprobs12; hessian += hprobs

            # hessian[iSpamLabel,iGateString,iGateSetParam1,iGateSetParams2] contains all
            #  d2(logl)/d(gatesetParam1)d(gatesetParam2) contributions
            return _np.sum(hessian, axis=(0,1))
              # sum over spam label and gate string dimensions (gate strings in evalSubTree)
              # adds current subtree contribution for (N,N')-sized block of Hessian


    else:

        #(the non-poisson picture requires that the probabilities of the spam labels for a given string are constrained to sum to 1)
        #NOTE: hessian_from_hprobs MAY modify hprobs and dprobs12 (to save mem)
        def hessian_from_hprobs(hprobs, dprobs12, cntVecMx, totalCntVec, pos_probs):
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
            hprobs *= hprobs_coeffs[:,:,None,None]
            dprobs12 *= dprobs12_coeffs[:,:,None,None]
            hessian = dprobs12; hessian += hprobs

            return _np.sum(hessian, axis=(0,1)) #see comments as above


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
    maxNumGatestrings = max([subtrees[i].num_final_strings() for i in mySubTreeIndices])
    cntVecMx_mem = _np.empty( (len(spamLabels),maxNumGatestrings),'d')
    probs_mem  = _np.empty( (len(spamLabels),maxNumGatestrings), 'd' )

    #DEBUG
    #import time
    #import sys
    #tStart = time.time()

    #Loop over subtrees
    for iSubTree in mySubTreeIndices:
        evalSubTree = subtrees[iSubTree]
        sub_nGateStrings = evalSubTree.num_final_strings()

        #  Create views into pre-allocated memory
        cntVecMx = cntVecMx_mem[:,0:sub_nGateStrings]
        probs  =  probs_mem[:,0:sub_nGateStrings]

        # Fill cntVecMx, totalCntVec
        fill_count_vecs(cntVecMx,spam_lbl_rows,dataset,
                            evalSubTree.generate_gatestring_list())
        totalCntVec = _np.sum(cntVecMx, axis=0)

        #compute pos_probs separately
        gateset.bulk_fill_probs(probs, spam_lbl_rows, evalSubTree,
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

        #k,kmax = 0,len(mySliceTupList) #DEBUG
        for (slice1,slice2,hprobs,dprobs12) in gateset.bulk_hprobs_by_block(
            spam_lbl_rows, evalSubTree, mySliceTupList, True, blkComm):

            #DEBUG
            #iSub = mySubTreeIndices.index(iSubTree)
            #print("DEBUG: rank%d: %gs: block %d/%d, sub-tree %d/%d, sub-tree-len = %d"
            #          % (comm.Get_rank(),time.time()-tStart,k,kmax,iSub,
            #             len(mySubTreeIndices), len(evalSubTree)))            
            #sys.stdout.flush(); k += 1

            subtree_hessian[slice1,slice2] = \
                hessian_from_hprobs(hprobs, dprobs12, cntVecMx,
                                        totalCntVec, pos_probs)
                #NOTE: hessian_from_hprobs MAY modify hprobs and dprobs12

        #Gather columns from different procs and add to running final hessian
        #_mpit.gather_slices_by_owner(slicesIOwn, subtree_hessian, (0,1), mySubComm)
        _mpit.gather_slices(sliceTupList, blkOwners, subtree_hessian, (0,1), mySubComm)
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


def logl_max(dataset, gatestring_list=None, countVecMx=None, poissonPicture=True, check=False):
    """
    The maximum log-likelihood possible for a DataSet.  That is, the
    log-likelihood obtained by a maximal model that can fit perfectly
    the probability of each gate string.

    Parameters
    ----------
    dataset : DataSet
        the data set to use.

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the max-log-likelihood
        sum.  Default value of None implies all the gate strings in dataset should
        be used.

    countVecMx : numpy array, optional
        Two-dimensional numpy array whose rows correspond to the data set's spam
        labels (i.e. dataset.get_spam_labels()).  Each row is  contains the
        dataset counts for that spam label for each gate string in gatestring_list.
        Use fill_count_vecs(...) to generate this quantity when it is useful elsewhere
        (e.g. for logl(...) calls).

    poissonPicture : boolean, optional
        Whether the Poisson-picture maximum log-likelihood should be returned.

    check : boolean, optional
        Whether additional check is performed which computes the max logl another
        way an compares to the faster method.

    Returns
    -------
    float
    """

    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    if countVecMx is None:
        spamLabels = dataset.get_spam_labels()
        spam_lbl_rows = { sl:i for (i,sl) in enumerate(spamLabels) }
        countVecMx = _np.empty( (len(spamLabels),len(gatestring_list)), 'd' )
        fill_count_vecs(countVecMx, spam_lbl_rows, dataset, gatestring_list)

    totalCntVec = _np.sum(countVecMx, axis=0)
    freqs = countVecMx / totalCntVec[None,:]
    freqs_nozeros = _np.where(countVecMx == 0, 1.0, freqs) # set zero freqs to 1.0 so np.log doesn't complain

    if poissonPicture:
        maxLogLTerms = countVecMx * ( _np.log(freqs_nozeros) - 1.0 )
    else:
        maxLogLTerms = countVecMx * _np.log(freqs_nozeros)

    maxLogLTerms[ countVecMx == 0 ] = 0.0 # set 0 * log(0) terms explicitly to zero since numpy doesn't know this limiting behavior

    # maxLogLTerms[iSpamLabel,iGateString] contains all logl-upper-bound contributions
    maxLogL = _np.sum(maxLogLTerms) # sum over *all* dimensions

    if check:
        L = 0
        for gateString in gatestring_list:
            dsRow = dataset[gateString]
            N = dsRow.total() #sum of counts for all outcomes (all spam labels)
            for n in list(dsRow.values()):
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

    for gs,dsRow in dataset.iteritems():
        probs = gateset.probs(gs)
        for (spamLabel,p) in probs.items():
            if p < TOL:
                if round(dsRow[spamLabel]) == 0: continue #contributes zero to the sum
                else: forbidden_prob += abs(TOL-p) + TOL
            elif p > 1-TOL:
                if round(dsRow[spamLabel]) == dsRow.total(): continue #contributes zero to the sum
                else: forbidden_prob += abs(p-(1-TOL)) + TOL


    return forbidden_prob

def prep_penalty(rhoVec):
    """
    Penalty assigned to a state preparation (rho) vector rhoVec.  State
      preparation density matrices must be positive semidefinite
      and trace == 1.  A positive return value indicates an
      these criteria are not met and the rho-vector is invalid.

    Parameters
    ----------
    rhoVec : numpy array
        rho vector array of shape (N,1) for some N.

    Returns
    -------
    float
    """
    # rhoVec must be positive semidefinite and trace = 1
    rhoMx = _bt.gmvec_to_stdmx(_np.asarray(rhoVec))
    evals = _np.linalg.eigvals( rhoMx )  #could use eigvalsh, but wary of this since eigh can be wrong...
    sumOfNeg = sum( [ -ev.real for ev in evals if ev.real < 0 ] )
    nQubits = _np.log2(len(rhoVec)) / 2
    tracePenalty = abs(rhoVec[0,0]-(1.0/_np.sqrt(2))**nQubits) # tensor of n I(2x2)/sqrt(2) has trace sqrt(2)**n
    #print "Sum of neg = ",sumOfNeg  #DEBUG
    #print "Trace Penalty = ",tracePenalty  #DEBUG
    return sumOfNeg +  tracePenalty

def effect_penalty(EVec):
    """
    Penalty assigned to a POVM effect vector EVec. Effects
      must have eigenvalues between 0 and 1.  A positive return
      value indicates this criterion is not met and the E-vector
      is invalid.

    Parameters
    ----------
    EVec : numpy array
         effect vector array of shape (N,1) for some N.

    Returns
    -------
    float
    """
    # EVec must have eigenvalues between 0 and 1
    EMx = _bt.gmvec_to_stdmx(_np.asarray(EVec))
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
        ret += sum([ prep_penalty(r) for r in list(gateset.preps.values()) ])
        ret += sum([ effect_penalty(e) for e in list(gateset.effects.values()) ])
    return ret


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

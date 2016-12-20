from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions related to computation of the log-likelihood."""

import numpy as _np
import warnings as _warnings
#import time as _time
from . import basistools as _bt
from . import jamiolkowski as _jam

TOL = 1e-20

#import sys #DEBUG TIMER

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


def logl_hessian(gateset, dataset, gatestring_list=None,
                 minProbClip=1e-6, probClipInterval=(-1e6,1e6), radius=1e-4,
                 evalTree=None, countVecMx=None, poissonPicture=True,
                 check=False, comm=None, memLimit=None):
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

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.


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

    if evalTree is None:
        evalTree = gateset.bulk_evaltree(gatestring_list)

    #Memory allocation
    ns = len(spamLabels); ng = len(gatestring_list)
    ne = gateset.num_params(); gd = gateset.get_dimension()
    C = 1.0/1024.0**3

    #  Estimate & check persistent memory (from allocs directly below)
    persistentMem = 8*ne**2 # in bytes
    if memLimit is not None and memLimit < persistentMem:
        raise MemoryError("HLogL Memory limit (%g GB) is " % (memLimit*C) +
                          "< memory required to hold final results (%g GB)"
                          % (persistentMem*C))

    #  Allocate persistent memory
    # Not needed yet: final_hessian = _np.zeros( (nP,nP), 'd')

    #  Estimate & check intermediate memory
    #  - check if we can fit entire hessian computation in memory; if so
    #      run in "all at once" mode
    #  - otherwise, work with a single column of the hessian at a time,
    #      which we call "by column" mode
    #  - if even in "by column" mode there's not enough memory, split the
    #      tree as needed (or raise an error if this is not possible)
    intermedMem  = 8* (ng*(2*ns + ns*ne + ns*ne**2)) # ~ local: for bulk_fill_hprods results
    intermedMem += 8*ng*gd**2*(ne**2 + ne + 1) # ~ bulk_hproduct
    if memLimit is not None and memLimit < intermedMem:
        mode = "by column"
        ne_spam = sum([v.num_params() for v in list(gateset.preps.values())] +
                      [v.num_params() for v in list(gateset.effects.values())])
        intermedMem  = 8* (ng*(2*ns)) # ~ local: for bulk_hprods_by_column
        intermedMem += 8*ns*ng*ne*(2*ne_spam+2) # ~ bulk_hprods_by_column internal - immediate
        intermedMem += 8*ng*gd**2*(ne + ne + 1) # ~ bulk_hprods_by_column internal - caches
        if memLimit < intermedMem:
            reductionFactor = float(intermedMem) / float(memLimit)
            maxEvalSubTreeSize = ng / reductionFactor # float
            minTreeSize = evalTree.get_min_tree_size()
            if maxEvalSubTreeSize < minTreeSize:
                raise MemoryError("Not enough memory to perform needed tree splitting!")
        else:
            maxEvalSubTreeSize = ng
    else:
        mode = "all at once"
        maxEvalSubTreeSize = ng

    #  Allocate memory (alloc max required & take views)
    maxNumGatestrings = maxEvalSubTreeSize
    cntVecMx_mem = _np.empty( (len(spamLabels),maxNumGatestrings),'d')
    probs_mem  = _np.empty( (len(spamLabels),maxNumGatestrings), 'd' )
    if mode == "all at once":
        dprobs_mem = _np.empty( (len(spamLabels),maxNumGatestrings,nP), 'd' )
        hprobs_mem = _np.empty( (len(spamLabels),maxNumGatestrings,nP,nP), 'd' )

    assert(not evalTree.is_split()) #assume we do all the splitting
    if maxEvalSubTreeSize < ng:
        evalTree.split(maxEvalSubTreeSize, None)
    #else:
    #    evalTree.split(None, 1) #trivial split - necessary?

    #DEBUG - no verbosity passed in to just leave commented out
    if memLimit is not None:
        print("HLogL Memory estimates: (%d spam labels," % ns + \
            "%d gate strings, %d gateset params, %d gate dim)" % (ng,ne,gd))
        print("Mode = %s" % mode)
        print("Peristent: %g GB " % (persistentMem*C))
        print("Intermediate: %g GB " % (intermedMem*C))
        print("Limit: %g GB" % (memLimit*C))
        if maxEvalSubTreeSize < ng:
            print("Maximum sub-tree size = %d" % maxEvalSubTreeSize)
            print("HLogL mem limit has imposed a division of evaluation tree.")
            print("Original tree length %d split into %d sub-trees of total length %d" % \
                (len(evalTree), len(evalTree.get_sub_trees()), sum(map(len,evalTree.get_sub_trees()))))

    a = radius # parameterizes "roundness" of f == 0 terms
    min_p = minProbClip


    #print "TEST dprobs timing"
    #t1 = _time.time()
    #for iTree,evalSubTree in enumerate(evalTree.get_sub_trees()):
    #    sub_nGateStrings = evalSubTree.num_final_strings()
    #    probs  =  probs_mem[:,0:sub_nGateStrings]
    #    dprobs = dprobs_mem[:,0:sub_nGateStrings,:]
    #
    #    gateset._calc().bulk_fill_dprobs(dprobs, spam_lbl_rows, evalSubTree,
    #                                     prMxToFill=probs,
    #                                     clipTo=probClipInterval, check=check)
    #    print "DEBUG: %gs: sub-tree %d/%d, sub-tree-len = %d" \
    #        % (_time.time()-t1,iTree, len(evalTree.get_sub_trees()), len(evalSubTree))
    #    sys.stdout.flush()
    #print "TOTAL TEST Time = ",(_time.time()-t1)

    if poissonPicture:
        def hessian_from_hprobs(hprobs, dprobs12, cntVecMx, totalCntVec, pos_probs):
            # Notation:  (K=#spam, M=#strings, N=#wrtParams1, N'=#wrtParams2 )
            S = cntVecMx / min_p - totalCntVec[None,:] # slope term that is derivative of logl at min_p
            S2 = -0.5 * cntVecMx / (min_p**2)          # 2nd derivative of logl term at min_p

            hprobs_pos  = (-cntVecMx / pos_probs**2)[:,:,None,None] * dprobs12   # (K,M,1,1) * (K,M,N,N')
            hprobs_pos += (cntVecMx / pos_probs - totalCntVec[None,:])[:,:,None,None] * hprobs  # (K,M,1,1) * (K,M,N,N')
            hprobs_neg  = (2*S2)[:,:,None,None] * dprobs12 + (S + 2*S2*(probs - min_p))[:,:,None,None] * hprobs # (K,M,1,1) * (K,M,N,N')

            hprobs_zerofreq = _np.where( (probs >= a)[:,:,None,None],
                                         -totalCntVec[None,:,None,None] * hprobs,
                                         (-totalCntVec[None,:] * ( (-2.0/a**2)*probs + 2.0/a))[:,:,None,None] * dprobs12
                                         - (totalCntVec[None,:] * ((-1.0/a**2)*probs**2 + 2*probs/a))[:,:,None,None] * hprobs )

            hessian = _np.where( (probs < min_p)[:,:,None,None], hprobs_neg, hprobs_pos)
            hessian = _np.where( (cntVecMx == 0)[:,:,None,None], hprobs_zerofreq, hessian) # (K,M,N,N)

            # hessian[iSpamLabel,iGateString,iGateSetParam1,iGateSetParams2] contains all
            #  d2(logl)/d(gatesetParam1)d(gatesetParam2) contributions
            return _np.sum(hessian, axis=(0,1))
              # sum over spam label and gate string dimensions (gate strings in evalSubTree)
              # adds current subtree contribution for (N,N')-sized block of Hessian


    else:

        #(the non-poisson picture requires that the probabilities of the spam labels for a given string are constrained to sum to 1)
        def hessian_from_hprobs(hprobs, dprobs12, cntVecMx, totalCntVec, pos_probs):
            S = cntVecMx / min_p # slope term that is derivative of logl at min_p
            S2 = -0.5 * cntVecMx / (min_p**2) # 2nd derivative of logl term at min_p

            hprobs_pos  = (-cntVecMx / pos_probs**2)[:,:,None,None] * dprobs12   # (K,M,1,1) * (K,M,N,N')
            hprobs_pos += (cntVecMx / pos_probs)[:,:,None,None] * hprobs  # (K,M,1,1) * (K,M,N,N')
            hprobs_neg  = (2*S2)[:,:,None,None] * dprobs12 + (S + 2*S2*(probs - min_p))[:,:,None,None] * hprobs # (K,M,1,1) * (K,M,N,N')

            hessian = _np.where( (probs < min_p)[:,:,None,None], hprobs_neg, hprobs_pos)
            hessian = _np.where( (cntVecMx == 0)[:,:,None,None], 0.0, hessian) # (K,M,N,N')

            return _np.sum(hessian, axis=(0,1)) #see comments as above



    # tStart = _time.time() #TIMER

    final_hessian = None #final computed quantity

    #Note - we could in the future use comm to distribute over
    # subtrees here.  We currently don't because we parallelize
    # over columns and it seems that in almost all cases of
    # interest there will be more hessian columns than processors,
    # so adding the additional ability to parallelize over
    # subtrees would just add unnecessary complication.

    #Loop over subtrees
    for evalSubTree in evalTree.get_sub_trees():
        sub_nGateStrings = evalSubTree.num_final_strings()

        #  Create views into pre-allocated memory
        cntVecMx = cntVecMx_mem[:,0:sub_nGateStrings]
        probs  =  probs_mem[:,0:sub_nGateStrings]

        # Fill cntVecMx, totalCntVec
        if countVecMx is None:
            fill_count_vecs(cntVecMx,spam_lbl_rows,dataset,
                            evalSubTree.generate_gatestring_list())
        else:
            # This local doesn't seem to exist, but the affected tests pass. However, pylint does not
            for i in myFinalToParentFinalMap:    #pylint: disable=undefined-variable
                cntVecMx[:,i] = countVecMx[:,i] #fill w/supplied countVecMx
        totalCntVec = _np.sum(cntVecMx, axis=0)

        if mode == "all at once":

            #additional memory views
            dprobs = dprobs_mem[:,0:sub_nGateStrings,:]
            hprobs = hprobs_mem[:,0:sub_nGateStrings,:,:]

            #TODO: call GateSet routine directly
            gateset.bulk_fill_hprobs(hprobs, spam_lbl_rows, evalSubTree,
                                     prMxToFill=probs, derivMxToFill=dprobs,
                                     clipTo=probClipInterval, check=check,
                                     comm=comm)

            pos_probs = _np.where(probs < min_p, min_p, probs)
            dprobs12 = dprobs[:,:,:,None] * dprobs[:,:,None,:] # (K,M,N,1) * (K,M,1,N) = (K,M,N,N)
            subtree_hessian = hessian_from_hprobs(hprobs, dprobs12, cntVecMx,
                                                  totalCntVec, pos_probs)

        elif mode == "by column":

            #compute pos_probs separately
            gateset.bulk_fill_probs(probs, spam_lbl_rows, evalSubTree,
                                    clipTo=probClipInterval, check=check,
                                    comm=comm)
            pos_probs = _np.where(probs < min_p, min_p, probs)

            # k = 0 #DEBUG


            #perform parallelization over columns
            if comm is None:
                nprocs, rank = 1, 0
            else:
                nprocs = comm.Get_size()
                rank = comm.Get_rank()

            nCols = gateset.num_params()
            if nprocs > nCols:
                raise ValueError("Too many (>%d) processors!" % nCols)
            loc_iCols = list(range(rank,nCols,nprocs))

            #  iterate over columns of hessian via bulk_hprobs_by_column
            assert(not evalSubTree.is_split()) #sub trees should not be split further
            loc_hessian_cols = [] # holds columns for this subtree (for this processor)
            for hprobs, dprobs12 in gateset.bulk_hprobs_by_column(
                spam_lbl_rows, evalSubTree, True, wrtFilter=loc_iCols):

                #DEBUG!!!
                #print "DEBUG: rank%d: %gs: column %d/%d, sub-tree %d/%d, sub-tree-len = %d" \
                #    % (rank,_time.time()-tStart,k,len(loc_iCols),iTree,
                #       len(evalTree.get_sub_trees()), len(evalSubTree))
                #sys.stdout.flush(); k += 1

                hessian_col = hessian_from_hprobs(hprobs, dprobs12, cntVecMx, totalCntVec, pos_probs)
                loc_hessian_cols.append(hessian_col)
                  #add current hessian column to list of columns on this proc

            #gather columns for this subtree (from all processors)
            if comm is None:
                proc_hessian_cols = [ loc_hessian_cols ]
            else:
                proc_hessian_cols = comm.allgather(loc_hessian_cols)
            proc_nCols = list(map(len,proc_hessian_cols)) # num cols on each proc

            #Untangle interleaved column ordering and concatenate
            max_loc_cols = max(proc_nCols) #max. cols computed on a single proc
            to_concat = [ proc_hessian_cols[rank][k] for k in range(max_loc_cols) \
                              for rank in range(nprocs) if proc_nCols[rank] > k  ]
            subtree_hessian = _np.concatenate(to_concat, axis=1)
              #same shape as final hessian, but only contribs from this subtree

            #OLD: subtree_hessian = _np.concatenate(hessian_cols, axis=1)

        #Add sub-tree contribution to final hessian
        if final_hessian is None:
            final_hessian = subtree_hessian
        else:
            final_hessian += subtree_hessian

    return final_hessian # (N,N)



#TODO: update like above
#def logl_debug(gateset, dataset, out_of_bounds_val=-1e8):
#    L = 0
#    for d in dataset.values():
#        p = gateset.PrPlus(d.gateString)
#        np = d.nPlus; nm = d.nMinus
#        if p < TOL and round(np) == 0: continue #contributes zero to the sum
#        if 1-p < TOL and round(nm) == 0: continue #contributes zero to the sum
#        if p < TOL or 1-p < TOL:
#            print "LogL out of bounds p = %g for %s" % (p,d.gateString)
#            return out_of_bounds_val #logl is undefined
#        L += logL_term(np,nm,p)
#
#    return L



#def logl_sloped_boundary(gateset, dataset):
#    L = 0
#    EPS=1e-20; S = 1 #slope reduction factor
#    for d in dataset.values():
#        p = gateset.PrPlus(d.gateString)
#        np = d.nPlus; nm = d.nMinus
#        if p < TOL and round(np) == 0: continue #contributes zero to the sum
#        if 1-p < TOL and round(nm) == 0: continue #contributes zero to the sum
#        L += np*log(p)   if p > EPS else np*(log(EPS) - (EPS-p)/(S*EPS))
#        L += nm*log(1-p) if p < 1-EPS else np*(log(EPS) - (p-(1-EPS))/(S*EPS))
#
#        #DEBUG
#        #pre = ((1-p)*d.nPlus - p*d.nMinus) / (p*(1-p))
#        #print "Pre(%s) = " % (d.gateString), pre, "  (p = %g, np = %g)" % (p, d.nPlus)
#        #END DEBUG
#
#    return L


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

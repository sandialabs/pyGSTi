from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for selecting a complete set of germs for a GST analysis."""

import numpy as _np
import numpy.linalg as _nla
import itertools as _itertools
import math as _math
import sys as _sys
import warnings as _warnings
from .. import objects as _objs

def num_non_spam_gauge_params(gateset):
    """Returns number of non-gauge parameters in a GateSet, not including SPAM
    parameters"""
    gateset = gateset.copy()
    for prepLabel in gateset.preps.keys():
        del gateset.preps[prepLabel]
    for effectLabel in gateset.effects.keys():
        del gateset.effects[effectLabel]
    return gateset.num_gauge_params()


#def _PerfectTwirl(mxToTwirl,wrt,eps):
#    """ Perform twirl on mxToTwirl with respect to wrt """
#    assert(mxToTwirl.shape[0] == mxToTwirl.shape[1]) #only square matrices allowed
#    assert(wrt.shape[0] == wrt.shape[1])
#    dim = mxToTwirl.shape[0]
#
#    #Get spectrum and eigenvectors of wrt
#    wrtEvals,wrtEvecs = _np.linalg.eig(wrt)
#    wrtEvecsInv = _np.linalg.inv( wrtEvecs )
#
#    # rotate mxToTwirl to the eigenbasis of wrt
#    rotmat = _np.dot(wrtEvecsInv, _np.dot(mxToTwirl, wrtEvecs))
#
#    #destroy coherences between non-degenerate eigenvectors (this is what twirling does)
#    for i in range(dim):
#        for j in range(dim):
#            if abs(wrtEvals[i] - wrtEvals[j]) > eps:
#                rotmat[i,j] = 0
#
#    return _np.dot(wrtEvecs, _np.dot(rotmat, wrtEvecsInv)) # rotate back to the original basis


# wrt is gate_dim x gate_dim, so is M, Minv, Proj
# so SOP is gate_dim^2 x gate_dim^2 and acts on vectorized *gates*
# Recall vectorizing identity (when vec(.) concats rows as flatten does):
#     vec( A * X * B ) = A tensor B^T * vec( X )
def _SuperOpForPerfectTwirl(wrt, eps):
    """ Return super operator for doing a perfect twirl with respect to wrt """
    assert(wrt.shape[0] == wrt.shape[1])    # only square matrices allowed
    dim = wrt.shape[0]
    SuperOp = _np.zeros( (dim**2, dim**2), 'complex' )

    # Get spectrum and eigenvectors of wrt
    wrtEvals,wrtEvecs = _np.linalg.eig(wrt)
    wrtEvecsInv = _np.linalg.inv( wrtEvecs )

    # We want to project  X -> M * (Proj_i * (Minv * X * M) * Proj_i) * Minv,
    # where M = wrtEvecs. So A = B = M * Proj_i * Minv and so
    # superop = A tensor B^T == A tensor A^T
    # NOTE: this == (A^T tensor A)^T while *Maple* germ functions seem to just
    # use A^T tensor A -> ^T difference
    for i in range(dim):
        # Create projector onto i-th eigenspace (spanned by i-th eigenvector
        # and other degenerate eigenvectors)
        Proj_i = _np.diag( [ (1 if (abs(wrtEvals[i] - wrtEvals[j]) <= eps)
                              else 0) for j in range(dim) ] )
        A = _np.dot(wrtEvecs, _np.dot(Proj_i, wrtEvecsInv) )
        # Need to normalize, because we are overcounting projectors onto
        # subspaces of dimension d > 1, giving us d * Proj_i tensor Proj_i^T.
        # We can fix this with a division by tr(Proj_i) = d.
        SuperOp += _np.kron(A, A.T) / _np.trace(Proj_i)
        # SuperOp += _np.kron(A.T,A) # Mimic Maple version (but I think this is
        # wrong... or it doesn't matter?)
    return SuperOp  # a gate_dim^2 x gate_dim^2 matrix



def twirled_deriv(gateset, gatestring, eps=1e-6):
    """
    Compute the "Twirled Derivative" of a gatestring, obtained
    by acting on the standard derivative of a gate string with
    the twirling superoperator.

    Parameters
    ----------
    gateset : Gateset object
      The GateSet which associates gate labels with operators.

    gatestring : GateString object
      The gate string to take a twirled derivative of.

    eps : float, optional
      Tolerance used for testing whether two eigenvectors
      are degenerate (i.e. abs(eval1 - eval2) < eps ? )

    Returns
    -------
    numpy array
      An array of shape (gate_dim^2, num_gateset_params)
    """
    prod  = gateset.product(gatestring)
    dProd = gateset.dproduct(gatestring, flat=True) # flattened_gate_dim x vec_gateset_dim
    twirler = _SuperOpForPerfectTwirl(prod, eps) # flattened_gate_dim x flattened_gate_dim
    return _np.dot( twirler, dProd ) # flattened_gate_dim x vec_gateset_dim


def bulk_twirled_deriv(gateset, gatestrings, eps=1e-6, check=False):
    """
    Compute the "Twirled Derivative" of a gatestring, obtained
    by acting on the standard derivative of a gate string with
    the twirling superoperator.

    Parameters
    ----------
    gateset : Gateset object
      The GateSet which associates gate labels with operators.

    gatestrings : list of GateString objects
      The gate string to take a twirled derivative of.

    eps : float, optional
      Tolerance used for testing whether two eigenvectors
      are degenerate (i.e. abs(eval1 - eval2) < eps ? )

    check : bool, optional
      Whether to perform internal consistency checks, at the
      expense of making the function slower.

    Returns
    -------
    numpy array
      An array of shape (num_gate_strings, gate_dim^2, num_gateset_params)
    """
    evalTree = gateset.bulk_evaltree(gatestrings)
    dProds, prods = gateset.bulk_dproduct(evalTree, flat=True, bReturnProds=True)#, memLimit=None)
    gate_dim = gateset.get_dimension()
    fd = gate_dim**2 # flattened gate dimension

    ret = _np.empty( (len(gatestrings), fd, dProds.shape[1]), 'complex')
    for i in range(len(gatestrings)):
        twirler = _SuperOpForPerfectTwirl(prods[i], eps) # flattened_gate_dim x flattened_gate_dim
        ret[i] = _np.dot( twirler, dProds[i*fd:(i+1)*fd] ) # flattened_gate_dim x vec_gateset_dim

    if check:
        for i in range(len(gatestrings)):
            chk_ret = twirled_deriv(gateset, gatestrings[i], eps)
            if _nla.norm(ret[i] - chk_ret) > 1e-6:
                _warnings.warn( "bulk twirled derive norm mismatch = %g - %g = %g" % \
                   (_nla.norm(ret[i]), _nla.norm(chk_ret), _nla.norm(ret[i] - chk_ret)) )

    return ret # nGateStrings x flattened_gate_dim x vec_gateset_dim



def test_germ_list_finitel(gateset, germsToTest, L, weights=None,
                         returnSpectrum=False, tol=1e-6):
    """
    Test whether a set of germs is able to amplify all of the GateSet's
    non-gauge parameters.

    Parameters
    ----------
    gateset : GateSet
        The GateSet (associates gate matrices with gate labels).

    germsToTest : list of GateStrings
        List of germs gate sequences to test for completeness.

    L : int
        The finite length to use in amplification testing.  Larger
        values take longer to compute but give more robust results.

    weights : numpy array, optional
        A 1-D array of weights with length equal len(germsToTest),
        which multiply the contribution of each germ to the total
        jacobian matrix determining parameter amplification. If
        None, a uniform weighting of 1.0/len(germsToTest) is applied.

    returnSpectrum : bool, optional
        If True, return the jacobian^T*jacobian spectrum in addition
        to the success flag.

    tol : float, optional
        Tolerance: an eigenvalue of jacobian^T*jacobian is considered
        zero and thus a parameter un-amplified when it is less than tol.

    Returns
    -------
    success : bool
        Whether all non-gauge parameters were amplified.

    spectrum : numpy array
        Only returned when `returnSpectrum` is ``True``.  Sorted array of
        eigenvalues (from small to large) of the jacobian^T * jacobian
        matrix used to determine parameter amplification.
    """

    # Remove any SPAM vectors from gateset since we only want
    # to consider the set of *gate* parameters for amplification
    # and this makes sure our parameter counting is correct
    gateset = gateset.copy()
    for prepLabel in list(gateset.preps.keys()):  del gateset.preps[prepLabel]
    for effectLabel in list(gateset.effects.keys()):  del gateset.effects[effectLabel]

    nGerms = len(germsToTest)
    germToPowL = [ germ*L for germ in germsToTest ]

    gate_dim = gateset.get_dimension()
    evt = gateset.bulk_evaltree(germToPowL)

    # shape (nGerms*flattened_gate_dim, vec_gateset_dim)
    dprods = gateset.bulk_dproduct(evt, flat=True)

    # shape (nGerms, flattened_gate_dim, vec_gateset_dim
    dprods = _np.reshape(dprods, (nGerms, gate_dim**2, dprods.shape[1]))

    germLensSq = _np.array( [ float(len(s))**2 for s in germsToTest ], 'd' )

    # shape (nGerms, vec_gateset_dim, vec_gateset_dim)
    derivDaggerDeriv = _np.einsum('ijk,ijl->ikl', _np.conjugate(dprods), dprods) / germLensSq[:,None,None]
       # result[i] = _np.dot( dprods[i].H, dprods[i] ) / len_of_ith_germString^2
       # result[i,k,l] = sum_j dprodsH[i,k,j] * dprods(i,j,l)
       # result[i,k,l] = sum_j dprods_conj[i,j,k] * dprods(i,j,l)

    # Take the average of the D^dagger*D/L^2 matrices associated with each germ
    # with optional weights.
    combineDDD = _np.average(derivDaggerDeriv, weights=weights, axis=0) / L**2
    sortedEigenvals = _np.sort(_np.real(_np.linalg.eigvalsh(combineDDD)))

    nGaugeParams = gateset.num_gauge_params()
    bSuccess = bool(sortedEigenvals[nGaugeParams] > tol)

    return (bSuccess,sortedEigenvals) if returnSpectrum else bSuccess




def test_germ_list_infl(gateset, germsToTest, scoreFunc='all', weights=None,
                           returnSpectrum=False, threshold=1e6, check=False):
    """
    Test whether a set of germs is able to amplify all of the GateSet's
    non-gauge parameters.

    Parameters
    ----------
    gateset : GateSet
        The GateSet (associates gate matrices with gate labels).

    germsToTest : list of GateString
        List of germs gate sequences to test for completeness.

    scoreFunc : {'all'm 'worst'}, optional (default is 'all')
        Sets the objective function for scoring a germ set.  If 'all', score is
        `l1Penalty`*(number of germs) + sum(1/Eigenvalues of score matrix).  If
        'worst', score is `l1Penalty`*(number of germs) + * 1/min(Eigenvalues of
        score matrix).  (Also note: because we are using a simple integer
        program to choose germs, it is possible to get stuck in a local
        minimum, and choosing one or the other objective function can help
        avoid such minima in different circumstances.)

    weights : numpy array, optional
        A 1-D array of weights with length equal len(germsToTest),
        which multiply the contribution of each germ to the total
        jacobian matrix determining parameter amplification. If
        None, a uniform weighting of 1.0/len(germsToTest) is applied.

    returnSpectrum : bool, optional
        If ``True``, return the jacobian^T*jacobian spectrum in addition
        to the success flag.

    threshold : float, optional
        An eigenvalue of jacobian^T*jacobian is considered zero and thus a
        parameter un-amplified when its reciprocal is greater than threshold.
        Also used for eigenvector degeneracy testing in twirling operation.

    check : bool, optional
      Whether to perform internal consistency checks, at the
      expense of making the function slower.


    Returns
    -------
    success : bool
        Whether all non-gauge parameters were amplified.

    spectrum : numpy array
        Only returned when `returnSpectrum` is ``True``.  Sorted array of
        eigenvalues (from small to large) of the jacobian^T * jacobian
        matrix used to determine parameter amplification.
    """

    if scoreFunc == 'all':
        def list_score(input_array):
            return sum(1./input_array)
    elif scoreFunc == 'worst':
        def list_score(input_array):
            return 1./min(input_array)


    # Remove any SPAM vectors from gateset since we only want
    # to consider the set of *gate* parameters for amplification
    # and this makes sure our parameter counting is correct
    gateset = gateset.copy()
    for prepLabel in gateset.preps.keys():  del gateset.preps[prepLabel]
    for effectLabel in gateset.effects.keys():  del gateset.effects[effectLabel]


    germLengths = _np.array( list(map(len,germsToTest)), 'i')
    twirledDeriv = bulk_twirled_deriv(gateset, germsToTest, 1./threshold, check) / germLengths[:,None,None]

    # Is conjugate needed? -- all should be real
    twirledDerivDaggerDeriv = _np.einsum('ijk,ijl->ikl', _np.conjugate(twirledDeriv), twirledDeriv)
       # result[i] = _np.dot( twirledDeriv[i].H, twirledDeriv[i] ) i.e. matrix
       # product
       # result[i,k,l] = sum_j twirledDerivH[i,k,j] * twirledDeriv(i,j,l)
       # result[i,k,l] = sum_j twirledDeriv_conj[i,j,k] * twirledDeriv(i,j,l)

    if weights is None:
        nGerms = len(germsToTest)
        # weights = _np.array( [1.0/nGerms]*nGerms, 'd')
        weights = _np.array( [1.0]*nGerms, 'd')

    combinedTDDD = _np.einsum('i,ijk', weights, twirledDerivDaggerDeriv)
    sortedEigenvals = _np.sort(_np.real(_np.linalg.eigvalsh(combinedTDDD)))

    nGaugeParams = gateset.num_gauge_params()

    bSuccess = bool(list_score(sortedEigenvals[nGaugeParams:]) < threshold)

    return (bSuccess,sortedEigenvals) if returnSpectrum else bSuccess

#@profile
def optimize_integer_germs_slack(gatesetList, germsList, randomize=True,
                                 randomizationStrength=1e-3, numCopies=None,
                                 seed=0, l1Penalty=1e-2, gatePenalty=0,
                                 initialWeights=None, scoreFunc='all',
                                 maxIter=100, fixedSlack=False,
                                 slackFrac=False, returnAll=False, tol=1e-6,
                                 check=False, forceSingletons=True,
                                 forceSingletonsScore=1e100, threshold=1e6,
                                 verbosity=1):
    """
    Find a locally optimal subset of the germs in germsList.

    Locally optimal here means that no single germ can be excluded
    without making the smallest non-gauge eigenvalue of the
    Jacobian.H*Jacobian matrix smaller, i.e. less amplified,
    by more than a fixed or variable amount of "slack", as
    specified by `fixedSlack` or `slackFrac`.

    Parameters
    ----------
    gatesetList : GateSet or list of GateSet
        The list of GateSets to be tested.  To ensure that the returned germ
        set is amplficationally complete, it is a good idea to score potential
        germ sets against a collection (~5-10) of similar gate sets.  The user
        may specify a single GateSet and a number of unitarily close copies to
        be made (set by the kwarg `numCopies`, or the user may specify their
        own list of GateSets, each of which in turn may or may not be
        randomized (set by the kwarg `randomize`).

    germsList : list of GateString
        List of all germs gate sequences to consider.
        IMPORTANT:  If `forceSingletons` is ``True``, the first k elements of
        `germsList` must be all k gates in GateSet.

    randomize : Bool, optional
        Whether or not the input GateSet(s) are subject first subject to
        unitary randomization.  If ``False``, the user should perform the
        unitary randomization themselves.  Note:  If the GateSet(s) are perfect
        (e.g. ``std1Q_XYI.gs_target``), then the germ selection output should
        not be trusted, due to accidental degeneracies in the GateSet.  If the
        GateSet(s) include stochastic (non-unitary) error, then germ selection
        will fail, as we score amplificational completeness in the limit of
        infinite sequence length (so any stochastic noise will completely
        depolarize any sequence in that limit).  Default is ``True``.

    randomizationStrength : float, optional
        The strength of the unitary noise used to randomize input GateSet(s);
        is passed to ``randomize_with_unitary``.  Default is ``1e-3``.

    numCopies : int, optional
        The number of GateSet copies to be made of the input GateSet (prior to
        unitary randomization).  If more than one GateSet is passed in,
        `numCopies` should be ``None``.  If only one GateSet is passed in and
        `numCopies` is ``None``, no extra copies are made.

    seed : float, optional
        The starting seed used for unitary randomization.  If multiple GateSets
        are to be randomized, ``gatesetList[i]`` is randomized with ``seed +
        i``.  Default is 0.

    l1Penalty : float, optional
        How strong the penalty should be for increasing the germ set list by a
        single germ.  Default is 1e-2.
    
    gatePenalty : float, optional
        How strong the penalty should be for increasing a germ in the germ set
        list by a single gate.  Default is 0. 
    
    initialWeights : list-like
        List or array of either booleans or (0 or 1) integers
        specifying which germs in `germList` comprise the initial
        germ set.  If ``None``, then starting point includes all
        germs.

    scoreFunc : {'all', 'worst'}, optional (default is 'all')
        Sets the objective function for scoring a germ set.  If 'all', score is
        `l1Penalty`*(number of germs) + sum(1/Eigenvalues of score matrix).
        If 'worst', score is `l1Penalty`*(number of germs) +
        1/min(Eigenvalues of score matrix).  (Also note: because we are using a
        simple integer program to choose germs, it is possible to get stuck in
        a local minimum, and choosing one or the other objective function can
        help avoid such minima in different circumstances.)

    maxIter : int, optional
        The maximum number of iterations before giving up.

    fixedSlack : float, optional
        If not ``None``, a floating point number which specifies that excluding
        a germ is allowed to increase 1.0/smallest-non-gauge-eigenvalue by
        `fixedSlack`.  You must specify *either* `fixedSlack` or `slackFrac`.

    slackFrac : float, optional
        If not ``None``, a floating point number which specifies that excluding
        a germ is allowed to increase 1.0/smallest-non-gauge-eigenvalue by
        `fixedFrac`*100 percent.  You must specify *either* `fixedSlack` or
        `slackFrac`.

    returnAll : bool, optional
        If ``True``, return the final ``weights`` vector and score dictionary
        in addition to the optimal germ list (see below).

    tol : float, optional
        Tolerance used for eigenvector degeneracy testing in twirling operation.

    check : bool, optional
        Whether to perform internal consistency checks, at the
        expense of making the function slower.

    forceSingletons : bool, optional (default is True)
        Whether or not to force all germ sets to contain each gate as a germ.
        IMPORTANT:  This only works if, for a gate set of k gates, the first k
        elements of germsList are the k gates.

    forceSingletonsScore : float, optional (default is 1e100)
        When `forceSingletons` is ``True``, what score to assign any germ set
        that does not contain each gate as a germ.

    threshold : float, optional (default is 1e6)
        Specifies a maximum score for the score matrix, above which the germ set
        is rejected as amplificationally incomplete.

    verbosity : int, optional
        Integer >= 0 indicating the amount of detail to print.

    Returns
    -------
    finalGermList : list
        Sublist of `germList` specifying the final, optimal set of germs.

    weights : array
        Integer array, of length ``len(germList)``, containing 0s and 1s to
        indicate which elements of `germList` were chosen as `finalGermList`.
        Only returned when `returnAll` is ``True``.

    scoreDictionary : dict
        Dictionary with keys which are tuples of 0s and 1s of length
        ``len(germList)``, specifying a subset of germs, and values ==
        1.0/smallest-non-gauge-eigenvalue "scores".

    See Also
    --------
    :class:`~pygsti.objects.GateSet`
    :class:`~pygsti.objects.GateString`
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    # Remove any SPAM vectors from gateset since we only want
    # to consider the set of *gate* parameters for amplification
    # and this makes sure our parameter counting is correct

    if not isinstance(gatesetList,list):
        gatesetList = [gatesetList]

    if (fixedSlack and slackFrac) or (not fixedSlack and not slackFrac):
        raise ValueError("Either fixedSlack *or* slackFrac should be specified")

    if len(gatesetList) > 1 and numCopies is not None:
        raise ValueError("Input multiple gate sets XOR request multiple "
                         "copies only!")

    if scoreFunc == 'all':
        def list_score(input_array):
            return sum(1./_np.abs(input_array))
    elif scoreFunc == 'worst':
        def list_score(input_array):
            return 1./min(_np.abs(input_array))
    else:
        raise ValueError("Either 'all' or 'worst' must be specified for "
                         "scoreFunc!")

    if randomize:
#        if seed:
#            _np.random.seed(seed)
        newgatesetList = []
        if len(gatesetList) > 1:
            for gatesetnum, gateset in enumerate(gatesetList):
                newgatesetList.append(gateset.randomize_with_unitary(
                                        randomizationStrength,
                                        seed=seed+gatesetnum))
#            gatesetList[gatesetnum] =
        else:
            for gatesetnum in range(numCopies if numCopies is not None else 1):
                newgatesetList.append(gatesetList[0].randomize_with_unitary(
                                        randomizationStrength,
                                        seed=seed+gatesetnum))
        gatesetList = newgatesetList

    for gatesetnum, gateset in enumerate(gatesetList):
        initial_test = test_germ_list_infl(gateset, germsList,
                                           scoreFunc=scoreFunc,
                                           threshold=threshold)
#        if initial_test:
#            print "Complete initial fiducial set succeeds."
#            print "Now searching for best fiducial set."
        if not initial_test:
            printer.log("Complete initial germ set FAILS on gateset "
                        + str(gatesetnum) + ".")
            printer.log("Aborting search.")
            if returnAll:
                return None, None, None
            else:
                return None
    printer.log("Complete initial germ set succeeds on all input gatesets.")
    printer.log("Now searching for best germ set.")

    num_gatesets = len(gatesetList)

    gateset0 = gatesetList[0].copy()
    for prepLabel in gateset0.preps.keys():  del gateset0.preps[prepLabel]
    for effectLabel in gateset0.effects.keys():  del gateset0.effects[effectLabel]

    lessWeightOnly = False  #Initially allow adding to weight. -- maybe make this an argument??

    nGaugeParams = gateset0.num_gauge_params()
    nGerms = len(germsList)

    printer.log("Starting germ set optimization. Lower score is better.", 1)
    printer.log("Gateset has %d gauge params." % nGaugeParams, 1)

    #score dictionary:
    #  keys = (gatesetNum, tuple-ized weight vector of 1's and 0's only)
    #  values = list_score
    scoreD = {}
    numGates = len(gateset0.gates.keys())
    #twirledDerivDaggerDeriv == array J.H*J contributions from each germ (J=Jacobian)
    # indexed by (iGerm, iGatesetParam1, iGatesetParam2)
    # size (nGerms, vec_gateset_dim, vec_gateset_dim)

    germLengths = _np.array(list(map(len,germsList)), 'i')

    twirledDerivDaggerDerivList = []

    for gateset in gatesetList:
        twirledDeriv = bulk_twirled_deriv(gateset, germsList, tol, check) / germLengths[:,None,None]
        twirledDerivDaggerDerivList.append(_np.einsum('ijk,ijl->ikl',
                                                _np.conjugate(twirledDeriv),
                                                twirledDeriv))

    def compute_score(wts,gateset_num):
        """Returns a germ set "score" in which smaller is better.  Also returns
        intentionally bad score if wts do not include all individual gates as
        individual germs, if forceSingletons is True.

        """
        if forceSingletons and _np.count_nonzero(wts[:numGates]) != numGates:
            score = forceSingletonsScore
        else:
            combinedTDDD = _np.einsum('i,ijk', wts,
                                      twirledDerivDaggerDerivList[gateset_num])
            sortedEigenvals = _np.sort(_np.real(_nla.eigvalsh(combinedTDDD)))
            score = (list_score(sortedEigenvals[nGaugeParams:])
                     + l1Penalty*_np.sum(wts)
                     + gatePenalty*_np.dot(germLengths, wts))
        # Side effect: calling compute_score caches result in scoreD
        scoreD[gateset_num,tuple(wts)] = score
        return score

    def get_neighbors(boolVec):
        for i in range(nGerms):
            v = boolVec.copy()
            v[i] = (v[i] + 1) % 2 # Toggle v[i] btwn 0 and 1
            yield v

    if initialWeights is not None:
        weights = _np.array( [1 if x else 0 for x in initialWeights ] )
    else:
        weights = _np.ones( nGerms, 'i' ) #default: start with all germs
#        lessWeightOnly = True #we're starting at the max-weight vector

    scoreList = [compute_score(weights,gateset_num)
                 for gateset_num in range(num_gatesets)]
    score = _np.max(scoreList)
#    print "scoreList:", scoreList
#    print score
    L1 = sum(weights) # ~ L1 norm of weights

    with printer.progress_logging(1):
        for iIter in range(maxIter):
            # List of weight tuples already computed
            scoreD_keys = scoreD.keys()

            printer.show_progress(iIter + 1, maxIter,
                                  suffix="score=%g, nGerms=%d" % (score, L1))

            bFoundBetterNeighbor = False
            for neighbor in get_neighbors(weights):
                # if force_singletons:
                #     if _np.count_nonzeros(neighbor[:numGates]) != numGates
                #         continue
                neighborScoreList = []
                for gateset_num, gateset in enumerate(gatesetList):
                    if (gateset_num,tuple(neighbor)) not in scoreD_keys:
                        neighborL1 = sum(neighbor)
                        neighborScoreList.append(compute_score(neighbor,
                                                               gateset_num))
                    else:
                        neighborL1 = sum(neighbor)
                        neighborScoreList.append(scoreD[gateset_num,
                                                        tuple(neighbor)])

                neighborScore = _np.max(neighborScoreList)#Take worst case.
                # print "neighborScore:", neighborScore
                # print "score:", score
                # Move if we've found better position; if we've relaxed, we
                # only move when L1 is improved.
                if neighborScore <= score and (neighborL1 < L1 or
                                               lessWeightOnly == False):
                    weights, score, L1 = neighbor, neighborScore, neighborL1
                    bFoundBetterNeighbor = True

                    printer.log("Found better neighbor: "
                                + "nGerms = %d score = %g" % (L1,score), 2)

            if not bFoundBetterNeighbor: # Time to relax our search.
                # From now on, don't allow increasing weight L1
                lessWeightOnly=True

                if fixedSlack==False:
                    # Note score is positive (for sum of 1/lambda)
                    slack = score*slackFrac 
                    # print "slack =", slack
                else:
                    slack = fixedSlack
                assert(slack > 0)

                printer.log("No better neighbor. Relaxing score w/slack: "
                            + "%g => %g" % (score, score+slack), 2)
                # Artificially increase score and see if any neighbor is better
                # now...
                score += slack 

                for neighborNum, neighbor in enumerate(get_neighbors(weights)):
                    scoreList = [scoreD[gateset_num,tuple(neighbor)]
                                 for gateset_num in range(len(gatesetList))]
                    maxScore = _np.max(scoreList)
                    if sum(neighbor) < L1 and maxScore < score:
                        weights, score, L1 = neighbor, maxScore, sum(neighbor)
                        bFoundBetterNeighbor = True
                        printer.log("Found better neighbor: "
                                    + "nGerms = %d score = %g" % (L1,score), 2)

                if not bFoundBetterNeighbor: # Relaxing didn't help!
                    printer.log("Stationary point found!");
                    break # end main for loop

            printer.log("Moving to better neighbor")
            # print score
        else:
            printer.log("Hit max. iterations")

    printer.log("score = %s" % score)
    printer.log("weights = %s" % weights)
    printer.log("L1(weights) = %s" % sum(weights))

    goodGermsList = []
    for index,val in enumerate(weights):
        if val==1:
            goodGermsList.append(germsList[index])

    if returnAll:
        return goodGermsList, weights, scoreD
    else:
        return goodGermsList

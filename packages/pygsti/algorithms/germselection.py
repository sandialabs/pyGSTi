from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for selecting a complete set of germs for a GST analysis."""

import warnings as _warnings

import numpy as _np
import numpy.linalg as _nla

from .. import objects as _objs
from .. import construction as _constr
from . import grasp as _grasp
from . import scoring as _scoring


def generate_germs(gs_target, randomize=True, randomizationStrength=1e-2,
                   numGSCopies=5, seed=None, maxGermLength=6,
                   force="singletons", algorithm='greedy',
                   algorithm_kwargs=None, verbosity=1):
    """Generate a germ set for doing GST with a given target gateset.

    This function provides a streamlined interface to a variety of germ
    selection algorithms. It's goal is to provide a method that typical users
    can run by simply providing a target gateset and leaving all other settings
    at their default values, while providing flexibility for users desiring
    more control to fine tune some of the general and algorithm-specific
    details.

    Currently, to break troublesome degeneracies and provide some confidence
    that the chosen germ set is amplificationally complete (AC) for all
    gatesets in a neighborhood of the target gateset (rather than only the
    target gateset), an ensemble of gatesets with random unitary perturbations
    to their gates must be provided or generated.

    Parameters
    ----------
    gs_target : GateSet or list of GateSet
        The gateset you are aiming to implement, or a list of gatesets that are
        copies of the gateset you are trying to implement (either with or
        without random unitary perturbations applied to the gatesets).

    randomize : bool, optional
        Whether or not to add random unitary perturbations to the gateset(s)
        provided.

    randomizationStrength : float, optional
        The size of the random unitary perturbations applied to gates in the
        gateset. See :meth:`~pygsti.objects.GateSet.randomize_with_unitary`
        for more details.

    numGSCopies : int, optional
        The number of copies of the original gateset that should be used.

    seed : int, optional
        Seed for generating random unitary perturbations to gatesets. Also
        passed along to stochastic germ-selection algorithms.

    maxGermsLength : int, optional
        The maximum length (in terms of gates) of any germ allowed in the germ
        set. Currently will construct a list of all non-equivalent germs of
        length up to `maxGermsLength` for the germ selection algorithms to play
        around with.

    force : str or list, optional
        A list of GateStrings which *must* be included in the final germ set.
        If set to the special string "singletons" then all length-1 strings will
        be included.  Seting to None is the same as an empty list.

    algorithm : {'greedy', 'grasp', 'slack'}, optional
        Specifies the algorithm to use to generate the germ set. Current
        options are:

        'greedy'
            Add germs one-at-a-time until the set is AC, picking the germ that
            improves the germ-set score by the largest amount at each step. See
            :func:`build_up_breadth` for more details.
        'grasp'
            Use GRASP to generate random greedy germ sets and then locally
            optimize them. See :func:`grasp_germ_set_optimization` for more
            details.
        'slack'
            From a initial set of germs, add or remove a germ at each step in
            an attempt to improve the germ-set score. Will allow moves that
            degrade the score in an attempt to escape local optima as long as
            the degredation is within some specified amount of "slack". See
            :func:`optimize_integer_germs_slack` for more details.

    algorithm_kwargs : dict
        Dictionary of ``{'keyword': keyword_arg}`` pairs providing keyword
        arguments for the specified `algorithm` function. See the documentation
        for functions referred to in the `algorithm` keyword documentation for
        what options are available for each algorithm.

    verbosity : int, optional
        The verbosity level of the :class:`~pygsti.objects.VerbosityPrinter`
        used to print log messages.

    Returns
    -------
    list of GateString
        A list containing the germs making up the germ set.

    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    gatesetList = setup_gateset_list(gs_target, randomize,
                                     randomizationStrength, numGSCopies, seed)
    gates = gs_target.gates.keys()
    availableGermsList = _constr.list_all_gatestrings_without_powers_and_cycles(
        gates, maxGermLength)

    if algorithm_kwargs is None:
        # Avoid danger of using empty dict for default value.
        algorithm_kwargs = {}

    if algorithm == 'greedy':
        printer.log('Using greedy algorithm.', 1)
        # Define defaults for parameters that currently have no default or
        # whose default we want to change.
        default_kwargs = {
            'germsList': availableGermsList,
            'randomize': False,
            'seed': seed,
            'verbosity': max(0, verbosity - 1),
            'force': force,
            'scoreFunc': 'all',
            }
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
        germList = build_up_breadth(gatesetList=gatesetList,
                                    **algorithm_kwargs)
        if germList is not None:
            germsetScore = calculate_germset_score(
                germList, neighborhood=gatesetList,
                scoreFunc=algorithm_kwargs['scoreFunc'])
            printer.log('Constructed germ set:', 1)
            printer.log(str([str(germ) for germ in germList]), 1)
            printer.log('Score: {}'.format(germsetScore.score), 1)
    elif algorithm == 'grasp':
        printer.log('Using GRASP algorithm.', 1)
        # Define defaults for parameters that currently have no default or
        # whose default we want to change.
        default_kwargs = {
            'alpha': 0.1,   # No real reason for setting this value of alpha.
            'germsList': availableGermsList,
            'randomize': False,
            'seed': seed,
            'verbosity': max(0, verbosity - 1),
            'force': force,
            'returnAll': False,
            'scoreFunc': 'all',
            }
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
        germList = grasp_germ_set_optimization(gatesetList=gatesetList,
                                               **algorithm_kwargs)
        printer.log('Constructed germ set:', 1)

        if algorithm_kwargs['returnAll'] and germList[0] is not None:
            germsetScore = calculate_germset_score(
                germList[0], neighborhood=gatesetList,
                scoreFunc=algorithm_kwargs['scoreFunc'])
            printer.log(str([str(germ) for germ in germList[0]]), 1)
            printer.log('Score: {}'.format(germsetScore.score))
        elif not algorithm_kwargs['returnAll'] and germList is not None:
            germsetScore = calculate_germset_score(germList,
                                                   neighborhood=gatesetList)
            printer.log(str([str(germ) for germ in germList]), 1)
            printer.log('Score: {}'.format(germsetScore.score), 1)
    elif algorithm == 'slack':
        printer.log('Using slack algorithm.', 1)
        # Define defaults for parameters that currently have no default or
        # whose default we want to change.
        default_kwargs = {
            'germsList': availableGermsList,
            'randomize': False,
            'seed': seed,
            'verbosity': max(0, verbosity - 1),
            'force': force,
            'scoreFunc': 'all',
            }
        if ('slackFrac' not in algorithm_kwargs
                and 'fixedSlack' not in algorithm_kwargs):
            algorithm_kwargs['slackFrac'] = 0.1
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
        germList = optimize_integer_germs_slack(gatesetList,
                                                **algorithm_kwargs)
        if germList is not None:
            germsetScore = calculate_germset_score(
                germList, neighborhood=gatesetList,
                scoreFunc=algorithm_kwargs['scoreFunc'])
            printer.log('Constructed germ set:', 1)
            printer.log(str([str(germ) for germ in germList]), 1)
            printer.log('Score: {}'.format(germsetScore.score), 1)
    else:
        raise ValueError("'{}' is not a valid algorithm "
                         "identifier.".format(algorithm))

    return germList


def calculate_germset_score(germs, gs_target=None, neighborhood=None,
                            neighborhoodSize=5,
                            randomizationStrength=1e-2, scoreFunc='all',
                            gatePenalty=0.0, l1Penalty=0.0):
    """Calculate the score of a germ set with respect to a gate set.

    """
    scoreFn = lambda x: _scoring.list_score(x, scoreFunc=scoreFunc)
    if neighborhood is None:
        neighborhood = [gs_target.randomize_with_unitary(randomizationStrength)
                        for n in range(neighborhoodSize)]
    scores = [compute_non_AC_score(scoreFn, gateset=gateset,
                                   partialGermsList=germs,
                                   gatePenalty=gatePenalty,
                                   l1Penalty=l1Penalty)
              for gateset in neighborhood]

    return max(scores)


def get_gateset_params(gatesetList):
    """Get the number of gates and gauge parameters of the gatesets in a list.

    Also verify all gatesets have the same number of gates and gauge parameters.

    Parameters
    ----------
    gatesetList : list of GateSet
        A list of gatesets for which you want an AC germ set.

    Returns
    -------
    reducedGatesetList : list of GateSet
        The original list of gatesets with SPAM removed

    numGaugeParams : int
        The number of non-SPAM gauge parameters for all gatesets.

    numNonGaugeParams : int
        The number of non-SPAM non-gauge parameters for all gatesets.

    numGates : int
        The number of gates for all gatesets.

    Raises
    ------
    ValueError
        If the number of gauge parameters or gates varies among the gatesets.

    """
    # We don't care about SPAM, since it can't be amplified.
    reducedGatesetList = [removeSPAMVectors(gateset)
                          for gateset in gatesetList]

    # All the gatesets should have the same number of parameters and gates, but
    # let's be paranoid here for the time being and make sure.
    numGaugeParamsList = [reducedGateset.num_gauge_params()
                          for reducedGateset in reducedGatesetList]
    numGaugeParams = numGaugeParamsList[0]
    if not all([numGaugeParams == otherNumGaugeParams
                for otherNumGaugeParams in numGaugeParamsList[1:]]):
        raise ValueError("All gatesets must have the same number of gauge "
                         "parameters!")

    numNonGaugeParamsList = [reducedGateset.num_nongauge_params()
                             for reducedGateset in reducedGatesetList]
    numNonGaugeParams = numNonGaugeParamsList[0]
    if not all([numNonGaugeParams == otherNumNonGaugeParams
                for otherNumNonGaugeParams in numNonGaugeParamsList[1:]]):
        raise ValueError("All gatesets must have the same number of non-gauge "
                         "parameters!")

    numGatesList = [len(reducedGateset.gates)
                    for reducedGateset in reducedGatesetList]
    numGates = numGatesList[0]
    if not all([numGates == otherNumGates
                for otherNumGates in numGatesList[1:]]):
        raise ValueError("All gatesets must have the same number of gates!")

    return reducedGatesetList, numGaugeParams, numNonGaugeParams, numGates


def setup_gateset_list(gatesetList, randomize, randomizationStrength,
                       numCopies, seed):
    if not isinstance(gatesetList, (list, tuple)):
        gatesetList = [gatesetList]
    if len(gatesetList) > 1 and numCopies is not None:
        _warnings.warn("Ignoring numCopies={} since multiple gatesets were "
                       "supplied.".format(numCopies))
        assert(False)

    if randomize:
        gatesetList = randomizeGatesetList(gatesetList, randomizationStrength,
                                           numCopies, seed)

    return gatesetList


def compute_non_AC_score(scoreFn, thresholdAC=1e6, initN=1,
                         partialDerivDaggerDeriv=None, gateset=None,
                         partialGermsList=None, eps=None, numGaugeParams=None,
                         gatePenalty=0.0, germLengths=None, l1Penalty=0.0):
    """Compute the score for a germ set when it is not AC against a gateset.

    Normally scores computed for germ sets against gatesets for which they are
    not AC will simply be astronomically large. This is fine if AC is all you
    care about, but not so useful if you want to compare partial germ sets
    against one another to see which is closer to being AC. This function
    will see if the germ set is AC for the parameters corresponding to the
    largest `N` eigenvalues for increasing `N` until it finds a value of `N`
    for which the germ set is not AC or all the non gauge parameters are
    accounted for and report the value of `N` as well as the score.

    This allows partial germ set scores to be compared against one-another
    sensibly, where a larger value of `N` always beats a smaller value of `N`,
    and ties in the value of `N` are broken by the score for that value of `N`.

    Parameters
    ----------
    scoreFn : callable
        A function that takes as input a list of sorted eigenvalues and returns
        a score for the partial germ set based on those eigenvalues, with lower
        scores indicating better germ sets. Usually some flavor of
        :func:`~pygsti.algorithms.scoring.list_score`.

    thresholdAC : float, optional
        Value which the score (before penalties are applied) must be lower than
        for the germ set to be considered AC.

    initN : int
        The number of largest eigenvalues to begin with checking.

    partialDerivDaggerDeriv : numpy.array, optional
        Array with three axes, where the first axis indexes individual germs
        within the partial germ set and the remaining axes index entries in the
        positive square of the Jacobian of each individual germ's parameters
        with respect to the gateset parameters.

        If this array is not supplied it will need to be computed from
        `germsList` and `gateset`, which will take longer, so it is recommended
        to precompute this array if this routine will be called multiple times.

    gateset : GateSet, optional
        The gateset against which the germ set is to be scored. Not needed if
        `partialDerivDaggerDeriv` is provided.

    partialGermsList : list of GateString, optional
        The list of germs in the partial germ set to be evaluated. Not needed
        if `partialDerivDaggerDeriv` (and `germLengths` when
        ``gatePenalty > 0``) are provided.

    eps : float, optional
        Used when calculating `partialDerivDaggerDeriv` to determine if two
        eigenvalues are equal (see :func:`bulk_twirled_deriv` for details). Not
        used if `partialDerivDaggerDeriv` is provided.

    numGaugeParams : int
        The number of gauge parameters of the gateset. Not needed if `gateset`
        is provided.

    gatePenalty : float, optional
        Coefficient for a penalty linear in the sum of the germ lengths.

    germLengths : numpy.array, optional
        The length of each germ. Not needed if `gatePenalty` is ``0.0`` or
        `partialGermsList` is provided.

    l1Penalty : float, optional
        Coefficient for a penalty linear in the number of germs.

    Returns
    -------
    CompositeScore
        The score for the germ set indicating how many parameters it amplifies
        and its numerical score restricted to those parameters.

    """
    if partialDerivDaggerDeriv is None:
        if gateset is None or partialGermsList is None:
            raise ValueError("Must provide either partialDerivDaggerDeriv or "
                             "(gateset, partialGermsList)!")
        else:
            pDDD_kwargs = {'gateset': gateset, 'germsList': partialGermsList}
            if eps is not None:
                pDDD_kwargs['eps'] = eps
            if germLengths is not None:
                pDDD_kwargs['germLengths'] = germLengths
            partialDerivDaggerDeriv = calc_twirled_DDD(**pDDD_kwargs)

    if numGaugeParams is None:
        if gateset is None:
            raise ValueError("Must provide either numGaugeParams or gateset!")
        else:
            numGaugeParams = removeSPAMVectors(gateset).num_gauge_params()

    # Calculate penalty scores
    numGerms = partialDerivDaggerDeriv.shape[0]
    l1Score = l1Penalty*numGerms
    gateScore = 0.0
    if gatePenalty != 0.0:
        if germLengths is None:
            if partialGermsList is None:
                raise ValueError("Must provide either germLengths or "
                                 "partialGermsList when gatePenalty != 0.0!")
            else:
                germLengths = _np.array([len(germ)
                                         for germ in partialGermsList])
        gateScore = gatePenalty*_np.sum(germLengths)

    combinedDDD = _np.sum(partialDerivDaggerDeriv, axis=0)
    sortedEigenvals = _np.sort(_np.real(_nla.eigvalsh(combinedDDD)))
    observableEigenvals = sortedEigenvals[numGaugeParams:]
    N_AC = 0
    AC_score = _np.inf
    for N in range(initN, len(observableEigenvals) + 1):
        scoredEigenvals = observableEigenvals[-N:]
        candidate_AC_score = scoreFn(scoredEigenvals)
        if candidate_AC_score > thresholdAC:
            break   # We've found a set of parameters for which the germ set
                    # is not AC.
        else:
            AC_score = candidate_AC_score
            N_AC = N
    # Apply penalties
    score = AC_score + l1Score + gateScore

    return _scoring.CompositeScore(score, N_AC)


def calc_twirled_DDD(gateset, germsList, eps=None, check=False,
                     germLengths=None):
    """Calculate the positive squares of the germ Jacobians.

    twirledDerivDaggerDeriv == array J.H*J contributions from each germ
    (J=Jacobian) indexed by (iGerm, iGatesetParam1, iGatesetParam2)
    size (nGerms, vec_gateset_dim, vec_gateset_dim)

    """
    if germLengths is None:
        germLengths = _np.array([len(germ) for germ in germsList])
    btd_kwargs = {'gateset': gateset, 'gatestrings': germsList, 'check': check}
    if eps is not None:
        btd_kwargs['eps'] = eps
    twirledDeriv = bulk_twirled_deriv(**btd_kwargs)/germLengths[:, None, None]
    twirledDerivDaggerDeriv = _np.einsum('ijk,ijl->ikl',
                                         _np.conjugate(twirledDeriv),
                                         twirledDeriv)
    return twirledDerivDaggerDeriv


def compute_score(weights, gateset_num, scoreFunc, derivDaggerDerivList,
                  forceIndices, forceScore,
                  nGaugeParams, gatePenalty, germLengths, l1Penalty=1e-2,
                  scoreDict=None):
    """Returns a germ set "score" in which smaller is better.  Also returns
    intentionally bad score (`forceScore`) if `weights` is zero on any of
    the "forced" germs (i.e. at any index in `forcedIndices`).

    This function is included for use by :func:`optimize_integer_germs_slack`,
    but is not convenient for just computing the score of a germ set. For that,
    use :func:`calculate_germset_score`.

    """
    if forceIndices and _np.any(weights[forceIndices] <= 0):
        score = forceScore
    else:
        combinedDDD = _np.einsum('i,ijk', weights,
                                 derivDaggerDerivList[gateset_num])
        sortedEigenvals = _np.sort(_np.real(_nla.eigvalsh(combinedDDD)))
        observableEigenvals = sortedEigenvals[nGaugeParams:]
        score = (_scoring.list_score(observableEigenvals, scoreFunc)
                 + l1Penalty*_np.sum(weights)
                 + gatePenalty*_np.dot(germLengths, weights))
    if scoreDict is not None:
        # Side effect: calling compute_score caches result in scoreDict
        scoreDict[gateset_num, tuple(weights)] = score
    return score


def randomizeGatesetList(gatesetList, randomizationStrength, numCopies,
                         seed=None):
    if len(gatesetList) > 1 and numCopies is not None:
        raise ValueError("Input multiple gate sets XOR request multiple "
                         "copies only!")
    newgatesetList = []
    if len(gatesetList) > 1:
        for gatesetnum, gateset in enumerate(gatesetList):
            newgatesetList.append(gateset.randomize_with_unitary(
                randomizationStrength,
                seed=None if seed is None else seed+gatesetnum))
    else:
        for gatesetnum in range(numCopies if numCopies is not None else 1):
            newgatesetList.append(gatesetList[0].randomize_with_unitary(
                randomizationStrength,
                seed=None if seed is None else seed+gatesetnum))
    return newgatesetList


def checkGermsListCompleteness(gatesetList, germsList, scoreFunc, threshold):
    """Check to see if the germsList is amplificationally complete (AC)

    Checks for AC with respect to all the GateSets in `gatesetList`, returning
    the index of the first GateSet for which it is not AC or `-1` if it is AC
    for all GateSets.

    """
    for gatesetNum, gateset in enumerate(gatesetList):
        initial_test = test_germ_list_infl(gateset, germsList,
                                           scoreFunc=scoreFunc,
                                           threshold=threshold)
        if not initial_test:
            return gatesetNum

    # If the germsList is complete for all gatesets, return -1
    return -1


def removeSPAMVectors(gateset):
    reducedGateset = gateset.copy()
    for prepLabel in list(reducedGateset.preps.keys()):
        del reducedGateset.preps[prepLabel]
    for effectLabel in list(reducedGateset.effects.keys()):
        del reducedGateset.effects[effectLabel]
    return reducedGateset


def get_neighbors(boolVec):
    for i in range(len(boolVec)):
        v = boolVec.copy()
        v[i] = (v[i] + 1) % 2 # Toggle v[i] btwn 0 and 1
        yield v


def num_non_spam_gauge_params(gateset):
    """Return number of non-gauge, non-SPAM parameters in a GateSet.

    """
    return removeSPAMVectors(gateset).num_gauge_params()


# wrt is gate_dim x gate_dim, so is M, Minv, Proj
# so SOP is gate_dim^2 x gate_dim^2 and acts on vectorized *gates*
# Recall vectorizing identity (when vec(.) concats rows as flatten does):
#     vec( A * X * B ) = A tensor B^T * vec( X )
def _SuperOpForPerfectTwirl(wrt, eps):
    """Return super operator for doing a perfect twirl with respect to wrt.

    """
    assert wrt.shape[0] == wrt.shape[1] # only square matrices allowed
    dim = wrt.shape[0]
    SuperOp = _np.zeros((dim**2, dim**2), 'complex')

    # Get spectrum and eigenvectors of wrt
    wrtEvals, wrtEvecs = _np.linalg.eig(wrt)
    wrtEvecsInv = _np.linalg.inv(wrtEvecs)

    # We want to project  X -> M * (Proj_i * (Minv * X * M) * Proj_i) * Minv,
    # where M = wrtEvecs. So A = B = M * Proj_i * Minv and so
    # superop = A tensor B^T == A tensor A^T
    # NOTE: this == (A^T tensor A)^T while *Maple* germ functions seem to just
    # use A^T tensor A -> ^T difference
    for i in range(dim):
        # Create projector onto i-th eigenspace (spanned by i-th eigenvector
        # and other degenerate eigenvectors)
        Proj_i = _np.diag([(1 if (abs(wrtEvals[i] - wrtEvals[j]) <= eps)
                            else 0) for j in range(dim)])
        A = _np.dot(wrtEvecs, _np.dot(Proj_i, wrtEvecsInv))
        # Need to normalize, because we are overcounting projectors onto
        # subspaces of dimension d > 1, giving us d * Proj_i tensor Proj_i^T.
        # We can fix this with a division by tr(Proj_i) = d.
        SuperOp += _np.kron(A, A.T) / _np.trace(Proj_i)
        # SuperOp += _np.kron(A.T,A) # Mimic Maple version (but I think this is
        # wrong... or it doesn't matter?)
    return SuperOp  # a gate_dim^2 x gate_dim^2 matrix


def sq_sing_vals_from_deriv(deriv, weights=None):
    """Calculate the squared singulare values of the Jacobian of the germ set.

    Parameters
    ----------
    deriv : numpy.array
        Array of shape ``(nGerms, flattened_gate_dim, vac_gateset_dim)``. Each
        sub-array corresponding to an individual germ is the Jacobian of the
        vectorized gate representation of that germ raised to some power with
        respect to the gateset parameters, normalized by dividing by the length
        of each germ after repetition.

    weights : numpy.array
        Array of length ``nGerms``, giving the relative contributions of each
        individual germ's Jacobian to the combined Jacobian (which is calculated
        as a convex combination of the individual Jacobians).

    Returns
    -------
    numpy.array
        The sorted squared singular values of the combined Jacobian of the germ
        set.

    """
    # shape (nGerms, vec_gateset_dim, vec_gateset_dim)
    derivDaggerDeriv = _np.einsum('ijk,ijl->ikl', _np.conjugate(deriv), deriv)

    # Take the average of the D^dagger*D/L^2 matrices associated with each germ
    # with optional weights.
    combinedDDD = _np.average(derivDaggerDeriv, weights=weights, axis=0)
    sortedEigenvals = _np.sort(_np.real(_nla.eigvalsh(combinedDDD)))

    return sortedEigenvals


def twirled_deriv(gateset, gatestring, eps=1e-6):
    """Compute the "Twirled Derivative" of a gatestring.

    The twirled derivative is obtained by acting on the standard derivative of
    a gate string with the twirling superoperator.

    Parameters
    ----------
    gateset : Gateset object
        The GateSet which associates gate labels with operators.

    gatestring : GateString object
        The gate string to take a twirled derivative of.

    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. abs(eval1 - eval2) < eps ? )

    Returns
    -------
    numpy array
      An array of shape (gate_dim^2, num_gateset_params)

    """
    prod = gateset.product(gatestring)

    # flattened_gate_dim x vec_gateset_dim
    dProd = gateset.dproduct(gatestring, flat=True)

    # flattened_gate_dim x flattened_gate_dim
    twirler = _SuperOpForPerfectTwirl(prod, eps)

    # flattened_gate_dim x vec_gateset_dim
    return _np.dot(twirler, dProd)


def bulk_twirled_deriv(gateset, gatestrings, eps=1e-6, check=False):
    """Compute the "Twirled Derivative" of a set of gatestrings.

    The twirled derivative is obtained by acting on the standard derivative of
    a gate string with the twirling superoperator.

    Parameters
    ----------
    gateset : Gateset object
        The GateSet which associates gate labels with operators.

    gatestrings : list of GateString objects
        The gate string to take a twirled derivative of.

    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. abs(eval1 - eval2) < eps ? )

    check : bool, optional
        Whether to perform internal consistency checks, at the expense of
        making the function slower.

    Returns
    -------
    numpy array
        An array of shape (num_gate_strings, gate_dim^2, num_gateset_params)

    """
    evalTree = gateset.bulk_evaltree(gatestrings)
    dProds, prods = gateset.bulk_dproduct(evalTree, flat=True, bReturnProds=True)#, memLimit=None)
    gate_dim = gateset.get_dimension()
    fd = gate_dim**2 # flattened gate dimension

    ret = _np.empty((len(gatestrings), fd, dProds.shape[1]), 'complex')
    for i in range(len(gatestrings)):

        # flattened_gate_dim x flattened_gate_dim
        twirler = _SuperOpForPerfectTwirl(prods[i], eps)

        # flattened_gate_dim x vec_gateset_dim
        ret[i] = _np.dot(twirler, dProds[i*fd:(i+1)*fd])

    if check:
        for i, gatestring in enumerate(gatestrings):
            chk_ret = twirled_deriv(gateset, gatestring, eps)
            if _nla.norm(ret[i] - chk_ret) > 1e-6:
                _warnings.warn("bulk twirled derive norm mismatch = "
                               "%g - %g = %g"
                               % (_nla.norm(ret[i]), _nla.norm(chk_ret),
                                  _nla.norm(ret[i] - chk_ret)))

    return ret # nGateStrings x flattened_gate_dim x vec_gateset_dim



def test_germ_list_finitel(gateset, germsToTest, L, weights=None,
                           returnSpectrum=False, tol=1e-6):
    """Test whether a set of germs is able to amplify all non-gauge parameters.

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
    gateset = removeSPAMVectors(gateset)

    nGerms = len(germsToTest)
    germToPowL = [germ*L for germ in germsToTest]

    gate_dim = gateset.get_dimension()
    evt = gateset.bulk_evaltree(germToPowL)

    # shape (nGerms*flattened_gate_dim, vec_gateset_dim)
    dprods = gateset.bulk_dproduct(evt, flat=True)

    # shape (nGerms, flattened_gate_dim, vec_gateset_dim
    dprods = _np.reshape(dprods, (nGerms, gate_dim**2, dprods.shape[1]))

    germLengths = _np.array([len(germ) for germ in germsToTest], 'd')

    normalizedDeriv = dprods / (L * germLengths[:, None, None])

    sortedEigenvals = sq_sing_vals_from_deriv(normalizedDeriv, weights)

    nGaugeParams = gateset.num_gauge_params()

    observableEigenvals = sortedEigenvals[nGaugeParams:]

    bSuccess = bool(_scoring.list_score(observableEigenvals, 'worst') < 1/tol)

    return (bSuccess, sortedEigenvals) if returnSpectrum else bSuccess


def test_germ_list_infl(gateset, germsToTest, scoreFunc='all', weights=None,
                        returnSpectrum=False, threshold=1e6, check=False):
    """Test whether a set of germs is able to amplify all non-gauge parameters.

    Parameters
    ----------
    gateset : GateSet
        The GateSet (associates gate matrices with gate labels).

    germsToTest : list of GateString
        List of germs gate sequences to test for completeness.

    scoreFunc : string
        Label to indicate how a germ set is scored. See
        :func:`~pygsti.algorithms.scoring.list_score` for details.

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
    # Remove any SPAM vectors from gateset since we only want
    # to consider the set of *gate* parameters for amplification
    # and this makes sure our parameter counting is correct
    gateset = removeSPAMVectors(gateset)


    germLengths = _np.array([len(germ) for germ in germsToTest], 'i')
    twirledDerivDaggerDeriv = calc_twirled_DDD(gateset, germsToTest,
                                               1./threshold, check,
                                               germLengths)
       # result[i] = _np.dot( twirledDeriv[i].H, twirledDeriv[i] ) i.e. matrix
       # product
       # result[i,k,l] = sum_j twirledDerivH[i,k,j] * twirledDeriv(i,j,l)
       # result[i,k,l] = sum_j twirledDeriv_conj[i,j,k] * twirledDeriv(i,j,l)

    if weights is None:
        nGerms = len(germsToTest)
        # weights = _np.array( [1.0/nGerms]*nGerms, 'd')
        weights = _np.array([1.0]*nGerms, 'd')

    combinedTDDD = _np.einsum('i,ijk', weights, twirledDerivDaggerDeriv)
    sortedEigenvals = _np.sort(_np.real(_np.linalg.eigvalsh(combinedTDDD)))

    nGaugeParams = gateset.num_gauge_params()
    observableEigenvals = sortedEigenvals[nGaugeParams:]

    bSuccess = bool(_scoring.list_score(observableEigenvals, scoreFunc)
                    < threshold)

    return (bSuccess, sortedEigenvals) if returnSpectrum else bSuccess


def build_up(gatesetList, germsList, randomize=True,
             randomizationStrength=1e-3, numCopies=None, seed=0, gatePenalty=0,
             scoreFunc='all', tol=1e-6, threshold=1e6, check=False,
             force="singletons", verbosity=0):
    """Greedy algorithm starting with 0 germs.

    Tries to minimize the number of germs needed to achieve amplificational
    completeness (AC). Begins with 0 germs and adds the germ that increases the
    score used to check for AC by the largest amount at each step, stopping when
    the threshold for AC is achieved.

    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    gatesetList = setup_gateset_list(gatesetList, randomize,
                                     randomizationStrength, numCopies, seed)

    (reducedGatesetList,
     numGaugeParams,
     numNonGaugeParams, numGates) = get_gateset_params(gatesetList)

    germLengths = _np.array([len(germ) for germ in germsList], 'i')
    numGerms = len(germsList)

    weights = _np.zeros(numGerms, 'i')
    goodGerms = []
    if force:
        if force == "singletons":
            weights[_np.where(germLengths == 1)] = 1
            goodGerms = [germ for germ
                             in _np.array(germsList)[_np.where(germLengths==1)]]
        else: #force should be a list of GateStrings
            for gs in force:
                weights[germsList.index(gs)] = 1
            goodGerms = force[:]

    undercompleteGatesetNum = checkGermsListCompleteness(gatesetList,
                                                         germsList,
                                                         scoreFunc,
                                                         threshold)
    if undercompleteGatesetNum > -1:
        printer.warning("Complete initial germ set FAILS on gateset "
                        + str(undercompleteGatesetNum) + ".")
        printer.warning("Aborting search.")
        return None

    printer.log("Complete initial germ set succeeds on all input gatesets.", 1)
    printer.log("Now searching for best germ set.", 1)

    printer.log("Starting germ set optimization. Lower score is better.", 1)

    twirledDerivDaggerDerivList = [calc_twirled_DDD(gateset, germsList, tol,
                                                    check, germLengths)
                                   for gateset in gatesetList]

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'scoreFn': lambda x: _scoring.list_score(x, scoreFunc=scoreFunc),
        'thresholdAC': threshold,
        'numGaugeParams': numGaugeParams,
        'gatePenalty': gatePenalty,
        'germLengths': germLengths,
        }

    for gatesetNum, reducedGateset in enumerate(reducedGatesetList):
        derivDaggerDeriv = twirledDerivDaggerDerivList[gatesetNum]
        # Make sure the set of germs you come up with is AC for all
        # gatesets.
        # Remove any SPAM vectors from gateset since we only want
        # to consider the set of *gate* parameters for amplification
        # and this makes sure our parameter counting is correct
        while _np.any(weights == 0):
            # As long as there are some unused germs, see if you need to add
            # another one.
            if test_germ_list_infl(reducedGateset, goodGerms,
                                   scoreFunc=scoreFunc, threshold=threshold):
                # The germs are sufficient for the current gateset
                break
            candidateGerms = _np.where(weights == 0)[0]
            candidateGermScores = []
            for candidateGermIdx in _np.where(weights == 0)[0]:
                # If the germs aren't sufficient, try adding a single germ
                candidateWeights = weights.copy()
                candidateWeights[candidateGermIdx] = 1
                partialDDD = derivDaggerDeriv[
                    _np.where(candidateWeights == 1)[0], :, :]
                candidateGermScore = compute_non_AC_score(
                    partialDerivDaggerDeriv=partialDDD, **nonAC_kwargs)
                candidateGermScores.append(candidateGermScore)
            # Add the germ that give the best score
            bestCandidateGerm = candidateGerms[_np.array(
                candidateGermScores).argmin()]
            weights[bestCandidateGerm] = 1
            goodGerms.append(germsList[bestCandidateGerm])

    return goodGerms


def build_up_breadth(gatesetList, germsList, randomize=True,
                     randomizationStrength=1e-3, numCopies=None, seed=0,
                     gatePenalty=0, scoreFunc='all', tol=1e-6, threshold=1e6,
                     check=False, force="singletons", verbosity=0):
    """Greedy algorithm starting with 0 germs.

    Tries to minimize the number of germs needed to achieve amplificational
    completeness (AC). Begins with 0 germs and adds the germ that increases the
    score used to check for AC by the largest amount (for the gateset that
    currently has the lowest score) at each step, stopping when the threshold
    for AC is achieved. This strategy is something of a "breadth-first"
    approach, in contrast to :func:`build_up`, which only looks at the
    scores for one gateset at a time until that gateset achieves AC, then
    turning it's attention to the remaining gatesets.

    Parameters
    ----------
    germsList : list of GateString
        The list of germs to contruct a germ set from.

    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    gatesetList = setup_gateset_list(gatesetList, randomize,
                                     randomizationStrength, numCopies, seed)

    (reducedGatesetList,
     numGaugeParams,
     numNonGaugeParams, numGates) = get_gateset_params(gatesetList)

    germLengths = _np.array([len(germ) for germ in germsList], 'i')

    numGerms = len(germsList)

    goodGerms = []
    weights = _np.zeros(numGerms, 'i')
    if force:
        if force == "singletons":
            weights[_np.where(germLengths == 1)] = 1
            goodGerms = [germ for germ
                             in _np.array(germsList)[_np.where(germLengths==1)]]
        else: #force should be a list of GateStrings
            for gs in force:
                weights[germsList.index(gs)] = 1
            goodGerms = force[:]

    undercompleteGatesetNum = checkGermsListCompleteness(gatesetList,
                                                         germsList,
                                                         scoreFunc,
                                                         threshold)
    if undercompleteGatesetNum > -1:
        printer.warning("Complete initial germ set FAILS on gateset "
                        + str(undercompleteGatesetNum) + ".")
        printer.warning("Aborting search.")
        return None

    printer.log("Complete initial germ set succeeds on all input gatesets.", 1)
    printer.log("Now searching for best germ set.", 1)

    printer.log("Starting germ set optimization. Lower score is better.", 1)

    twirledDerivDaggerDerivList = [calc_twirled_DDD(gateset, germsList, tol,
                                                    check, germLengths)
                                   for gateset in gatesetList]

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'scoreFn': lambda x: _scoring.list_score(x, scoreFunc=scoreFunc),
        'thresholdAC': threshold,
        'numGaugeParams': numGaugeParams,
        'gatePenalty': gatePenalty,
        'germLengths': germLengths,
        }


    initN = 1
    while _np.any(weights == 0):
        printer.log("Outer iteration: %d of %d amplified, %d germs" % 
                    (initN,numNonGaugeParams,len(goodGerms)), 2)
        # As long as there are some unused germs, see if you need to add
        # another one.
        if initN == numNonGaugeParams:
            break   # We are AC for all gatesets, so we can stop adding germs.

        candidateGerms = _np.where(weights == 0)[0]
        candidateGermScores = []
        candidateGermIndices = _np.where(weights == 0)[0]
        with printer.progress_logging(3):
            for i,candidateGermIdx in enumerate(candidateGermIndices):
                printer.show_progress(i, len(candidateGermIndices), 
                                      prefix="Inner iter over candidate germs",
                                      suffix=str(germsList[candidateGermIdx]))
    
                # If the germs aren't sufficient, try adding a single germ
                candidateWeights = weights.copy()
                candidateWeights[candidateGermIdx] = 1
                germVsGatesetScores = []
                for derivDaggerDeriv in twirledDerivDaggerDerivList:
                    # Loop over all gatesets
                    partialDDD = derivDaggerDeriv[
                        _np.where(candidateWeights == 1)[0], :, :]
                    germVsGatesetScores.append(compute_non_AC_score(
                        partialDerivDaggerDeriv=partialDDD, initN=initN,
                        **nonAC_kwargs))
                # Take the score for the current germ to be it's worst score over
                # all gatesets.
                worstScore = max(germVsGatesetScores)
                printer.log(str(worstScore), 4)
                candidateGermScores.append(worstScore)
        # Add the germ that gives the best worst score
        bestCandidateGerm = candidateGerms[_np.array(
            candidateGermScores).argmin()]
        weights[bestCandidateGerm] = 1
        goodGerms.append(germsList[bestCandidateGerm])
        bestScore = min(candidateGermScores)
        initN = bestScore.N
        printer.log("Added %s to final germs (%s)" % 
                    (str(germsList[bestCandidateGerm]), str(bestScore)), 3)

    return goodGerms


#@profile
def optimize_integer_germs_slack(gatesetList, germsList, randomize=True,
                                 randomizationStrength=1e-3, numCopies=None,
                                 seed=0, l1Penalty=1e-2, gatePenalty=0,
                                 initialWeights=None, scoreFunc='all',
                                 maxIter=100, fixedSlack=False,
                                 slackFrac=False, returnAll=False, tol=1e-6,
                                 check=False, force="singletons",
                                 forceScore=1e100, threshold=1e6,
                                 verbosity=1):
    """Find a locally optimal subset of the germs in germsList.

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
        be made (set by the kwarg `numCopies`), or the user may specify their
        own list of GateSets, each of which in turn may or may not be
        randomized (set by the kwarg `randomize`).

    germsList : list of GateString
        List of all germs gate sequences to consider.

    randomize : Bool, optional
        Whether or not the input GateSet(s) are first subject to unitary
        randomization.  If ``False``, the user should perform the unitary
        randomization themselves.  Note:  If the GateSet(s) are perfect (e.g.
        ``std1Q_XYI.gs_target``), then the germ selection output should not be
        trusted, due to accidental degeneracies in the GateSet.  If the
        GateSet(s) include stochastic (non-unitary) error, then germ selection
        will fail, as we score amplificational completeness in the limit of
        infinite sequence length (so any stochastic noise will completely
        depolarize any sequence in that limit).  Default is ``True``.

    randomizationStrength : float, optional
        The strength of the unitary noise used to randomize input GateSet(s);
        is passed to :func:`~pygsti.objects.GateSet.randomize_with_unitary`.
        Default is ``1e-3``.

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

    scoreFunc : string
        Label to indicate how a germ set is scored. See
        :func:`~pygsti.algorithms.scoring.list_score` for details.

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
        Tolerance used for eigenvector degeneracy testing in twirling
        operation.

    check : bool, optional
        Whether to perform internal consistency checks, at the
        expense of making the function slower.

    force : str or list, optional
        A list of GateStrings which *must* be included in the final germ set.
        If set to the special string "singletons" then all length-1 strings will
        be included.  Seting to None is the same as an empty list.

    forceScore : float, optional (default is 1e100)
        When `force` designates a non-empty set of gate strings, the score to
        assign any germ set that does not contain each and every required germ.

    threshold : float, optional (default is 1e6)
        Specifies a maximum score for the score matrix, above which the germ
        set is rejected as amplificationally incomplete.

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

    gatesetList = setup_gateset_list(gatesetList, randomize,
                                     randomizationStrength, numCopies, seed)

    if (fixedSlack and slackFrac) or (not fixedSlack and not slackFrac):
        raise ValueError("Either fixedSlack *or* slackFrac should be specified")

    if initialWeights is not None:
        if len(germsList) != len(initialWeights):
            raise ValueError("The lengths of germsList (%d) and "
                             "initialWeights (%d) must match."
                             % (len(germsList), len(initialWeights)))
        # Normalize the weights array to be 0s and 1s even if it is provided as
        # bools
        weights = _np.array([1 if x else 0 for x in initialWeights])
    else:
        weights = _np.ones(len(germsList), 'i') # default: start with all germs
#        lessWeightOnly = True # we're starting at the max-weight vector

    undercompleteGatesetNum = checkGermsListCompleteness(gatesetList,
                                                         germsList, scoreFunc,
                                                         threshold)
    if undercompleteGatesetNum > -1:
        printer.log("Complete initial germ set FAILS on gateset "
                    + str(undercompleteGatesetNum) + ".", 1)
        printer.log("Aborting search.", 1)
        return (None, None, None) if returnAll else None

    printer.log("Complete initial germ set succeeds on all input gatesets.", 1)
    printer.log("Now searching for best germ set.", 1)

    num_gatesets = len(gatesetList)

    # Remove any SPAM vectors from gateset since we only want
    # to consider the set of *gate* parameters for amplification
    # and this makes sure our parameter counting is correct
    gateset0 = removeSPAMVectors(gatesetList[0])

    # Initially allow adding to weight. -- maybe make this an argument??
    lessWeightOnly = False

    nGaugeParams = gateset0.num_gauge_params()

    # score dictionary:
    #   keys = (gatesetNum, tuple-ized weight vector of 1's and 0's only)
    #   values = list_score
    scoreD = {}
    germLengths = _np.array([len(germ) for germ in germsList], 'i')

    if force:
        if force == "singletons":
            forceIndices = _np.where(germLengths == 1)
        else: #force should be a list of GateStrings
            forceIndices = _np.array([germsList.index(gs) for gs in force])
    else:
        forceIndices = None

    twirledDerivDaggerDerivList = [calc_twirled_DDD(gateset, germsList, tol,
                                                    check, germLengths)
                                   for gateset in gatesetList]

    # Dict of keyword arguments passed to compute_score that don't change from
    # call to call
    cs_kwargs = {
        'scoreFunc': scoreFunc,
        'derivDaggerDerivList': twirledDerivDaggerDerivList,
        'forceIndices': forceIndices,
        'forceScore': forceScore,
        'nGaugeParams': nGaugeParams,
        'gatePenalty': gatePenalty,
        'germLengths': germLengths,
        'l1Penalty': l1Penalty,
        'scoreDict': scoreD,
        }

    scoreList = [compute_score(weights, gateset_num, **cs_kwargs)
                 for gateset_num in range(num_gatesets)]
    score = _np.max(scoreList)
    L1 = sum(weights) # ~ L1 norm of weights

    printer.log("Starting germ set optimization. Lower score is better.", 1)
    printer.log("Gateset has %d gauge params." % nGaugeParams, 1)

    with printer.progress_logging(1):
        for iIter in range(maxIter):
            printer.show_progress(iIter, maxIter,
                                  suffix="score=%g, nGerms=%d" % (score, L1))

            bFoundBetterNeighbor = False
            for neighbor in get_neighbors(weights):
                neighborScoreList = []
                for gateset_num in range(len(gatesetList)):
                    if (gateset_num, tuple(neighbor)) not in scoreD:
                        neighborL1 = sum(neighbor)
                        neighborScoreList.append(compute_score(neighbor,
                                                               gateset_num,
                                                               **cs_kwargs))
                    else:
                        neighborL1 = sum(neighbor)
                        neighborScoreList.append(scoreD[gateset_num,
                                                        tuple(neighbor)])

                neighborScore = _np.max(neighborScoreList)  # Take worst case.
                # Move if we've found better position; if we've relaxed, we
                # only move when L1 is improved.
                if neighborScore <= score and (neighborL1 < L1 or
                                               not lessWeightOnly):
                    weights, score, L1 = neighbor, neighborScore, neighborL1
                    bFoundBetterNeighbor = True

                    printer.log("Found better neighbor: "
                                "nGerms = %d score = %g" % (L1, score), 2)

            if not bFoundBetterNeighbor: # Time to relax our search.
                # From now on, don't allow increasing weight L1
                lessWeightOnly = True

                if fixedSlack is False:
                    # Note score is positive (for sum of 1/lambda)
                    slack = score*slackFrac
                    # print "slack =", slack
                else:
                    slack = fixedSlack
                assert slack > 0

                printer.log("No better neighbor. Relaxing score w/slack: "
                            + "%g => %g" % (score, score+slack), 2)
                # Artificially increase score and see if any neighbor is better
                # now...
                score += slack

                for neighbor in get_neighbors(weights):
                    scoreList = [scoreD[gateset_num, tuple(neighbor)]
                                 for gateset_num in range(len(gatesetList))]
                    maxScore = _np.max(scoreList)
                    if sum(neighbor) < L1 and maxScore < score:
                        weights, score, L1 = neighbor, maxScore, sum(neighbor)
                        bFoundBetterNeighbor = True
                        printer.log("Found better neighbor: "
                                    "nGerms = %d score = %g" % (L1, score), 2)

                if not bFoundBetterNeighbor: # Relaxing didn't help!
                    printer.log("Stationary point found!", 1)
                    break # end main for loop

            printer.log("Moving to better neighbor", 1)
            # print score
        else:
            printer.log("Hit max. iterations", 1)

    printer.log("score = %s" % score, 1)
    printer.log("weights = %s" % weights, 1)
    printer.log("L1(weights) = %s" % sum(weights), 1)

    goodGerms = []
    for index, val in enumerate(weights):
        if val == 1:
            goodGerms.append(germsList[index])

    if returnAll:
        return goodGerms, weights, scoreD
    else:
        return goodGerms


def germ_breadth_score_fn(germSet, germsList, twirledDerivDaggerDerivList,
                          nonAC_kwargs, initN=1):
    """Score a germ set against a collection of gatesets.

    Calculate the score of the germ set with respect to each member of a
    collection of gatesets and return the worst score among that collection.

    Parameters
    ----------
    germSet : list of GateString
        The set of germs to score.

    germsList : list of GateString
        The list of all germs whose Jacobians are provided in
        `twirledDerivDaggerDerivList`.

    twirledDerivDaggerDerivList : numpy.array
        Jacobians for all the germs in `germsList` stored as a 3-dimensional
        array, where the first index indexes the particular germ.

    nonAC_kwargs : dict
        Dictionary containing further arguments to pass to
        :func:`compute_non_AC_score` for the scoring of the germ set against
        individual gatesets.

    initN : int
        The number of eigenvalues to begin checking for amplificational
        completeness with respect to. Passed as an argument to
        :func:`compute_non_AC_score`.

    Returns
    -------
    CompositeScore
        The worst score over all gatesets of the germ set.

    """
    weights = _np.zeros(len(germsList))
    for germ in germSet:
        weights[germsList.index(germ)] = 1
    germsVsGatesetScores = []
    for derivDaggerDeriv in twirledDerivDaggerDerivList:
        # Loop over all gatesets
        partialDDD = derivDaggerDeriv[_np.where(weights == 1)[0], :, :]
        germsVsGatesetScores.append(compute_non_AC_score(
            partialDerivDaggerDeriv=partialDDD, initN=initN, **nonAC_kwargs))
    # Take the score for the current germ set to be its worst score over all
    # gatesets.
    return max(germsVsGatesetScores)


def grasp_germ_set_optimization(gatesetList, germsList, alpha, randomize=True,
                                randomizationStrength=1e-3, numCopies=None,
                                seed=None, l1Penalty=1e-2, gatePenalty=0.0,
                                scoreFunc='all', tol=1e-6, threshold=1e6,
                                check=False, force="singletons",
                                iterations=5, returnAll=False, shuffle=False,
                                verbosity=0):
    """Use GRASP to find a high-performing germ set.

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

    alpha : float
        A number between 0 and 1 that roughly specifies a score theshold
        relative to the spread of scores that a germ must score better than in
        order to be included in the RCL. A value of 0 for `alpha` corresponds
        to a purely greedy algorithm (only the best-scoring germ set is
        included in the RCL), while a value of 1 for `alpha` will include all
        germs in the RCL.

        See :func:`pygsti.algorithms.scoring.composite_rcl_fn` for more details.

    randomize : Bool, optional
        Whether or not the input GateSet(s) are first subject to unitary
        randomization.  If ``False``, the user should perform the unitary
        randomization themselves.  Note:  If the GateSet(s) are perfect (e.g.
        ``std1Q_XYI.gs_target``), then the germ selection output should not be
        trusted, due to accidental degeneracies in the GateSet.  If the
        GateSet(s) include stochastic (non-unitary) error, then germ selection
        will fail, as we score amplificational completeness in the limit of
        infinite sequence length (so any stochastic noise will completely
        depolarize any sequence in that limit).

    randomizationStrength : float, optional
        The strength of the unitary noise used to randomize input GateSet(s);
        is passed to :func:`~pygsti.objects.GateSet.randomize_with_unitary`.
        Default is ``1e-3``.

    numCopies : int, optional
        The number of GateSet copies to be made of the input GateSet (prior to
        unitary randomization).  If more than one GateSet is passed in,
        `numCopies` should be ``None``.  If only one GateSet is passed in and
        `numCopies` is ``None``, no extra copies are made.

    seed : float, optional
        The starting seed used for unitary randomization.  If multiple GateSets
        are to be randomized, ``gatesetList[i]`` is randomized with ``seed +
        i``.

    l1Penalty : float, optional
        How strong the penalty should be for increasing the germ set list by a
        single germ. Used for choosing between outputs of various GRASP
        iterations.

    gatePenalty : float, optional
        How strong the penalty should be for increasing a germ in the germ set
        list by a single gate.

    scoreFunc : string
        Label to indicate how a germ set is scored. See
        :func:`~pygsti.algorithms.scoring.list_score` for details.

    tol : float, optional
        Tolerance used for eigenvector degeneracy testing in twirling
        operation.

    threshold : float, optional (default is 1e6)
        Specifies a maximum score for the score matrix, above which the germ
        set is rejected as amplificationally incomplete.

    check : bool, optional
        Whether to perform internal consistency checks, at the
        expense of making the function slower.

    force : str or list, optional
        A list of GateStrings which *must* be included in the final germ set.
        If set to the special string "singletons" then all length-1 strings will
        be included.  Seting to None is the same as an empty list.

    iterations : int, optional
        The number of GRASP iterations to perform.

    returnAll : bool, optional
        Flag set to tell the routine if it should return lists of all
        initial constructions and local optimizations in addition to the
        optimal solution (useful for diagnostic purposes or if you're not sure
        what your `finalScoreFn` should really be).

    shuffle : bool, optional
        Whether the neighborhood should be presented to the optimizer in a
        random order (important since currently the local optimizer updates the
        solution to the first better solution it finds in the neighborhood).

    verbosity : int, optional
        Integer >= 0 indicating the amount of detail to print.

    Returns
    -------
    finalGermList : list of GateString
        Sublist of `germsList` specifying the final, optimal set of germs.

    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    gatesetList = setup_gateset_list(gatesetList, randomize,
                                     randomizationStrength, numCopies, seed)

    (reducedGatesetList,
     numGaugeParams,
     numNonGaugeParams, numGates) = get_gateset_params(gatesetList)

    germLengths = _np.array([len(germ) for germ in germsList], 'i')

    numGerms = len(germsList)

    initialWeights = _np.zeros(numGerms, dtype='i')
    if force:
        if force == "singletons":
            initialWeights[_np.where(germLengths == 1)] = 1
        else: #force should be a list of GateStrings
            for gs in force:
                initialWeights[germsList.index(gs)] = 1

    getNeighborsFn = lambda weights: _grasp.get_swap_neighbors(
        weights, forcedWeights=initialWeights, shuffle=shuffle)

    undercompleteGatesetNum = checkGermsListCompleteness(gatesetList,
                                                         germsList,
                                                         scoreFunc,
                                                         threshold)
    if undercompleteGatesetNum > -1:
        printer.warning("Complete initial germ set FAILS on gateset "
                        + str(undercompleteGatesetNum) + ".")
        printer.warning("Aborting search.")
        return None

    printer.log("Complete initial germ set succeeds on all input gatesets.", 1)
    printer.log("Now searching for best germ set.", 1)

    printer.log("Starting germ set optimization. Lower score is better.", 1)

    twirledDerivDaggerDerivList = [calc_twirled_DDD(gateset, germsList, tol,
                                                    check, germLengths)
                                   for gateset in gatesetList]

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'scoreFn': lambda x: _scoring.list_score(x, scoreFunc=scoreFunc),
        'thresholdAC': threshold,
        'numGaugeParams': numGaugeParams,
        'gatePenalty': gatePenalty,
        'germLengths': germLengths,
        }

    final_nonAC_kwargs = nonAC_kwargs.copy()
    final_nonAC_kwargs['l1Penalty'] = l1Penalty

    scoreFn = (lambda germSet:
               germ_breadth_score_fn(germSet, germsList,
                                     twirledDerivDaggerDerivList, nonAC_kwargs,
                                     initN=1))
    finalScoreFn = (lambda germSet:
                    germ_breadth_score_fn(germSet, germsList,
                                          twirledDerivDaggerDerivList,
                                          final_nonAC_kwargs, initN=1))

    feasibleThreshold = _scoring.CompositeScore(threshold, numNonGaugeParams)

    rclFn = lambda x: _scoring.composite_rcl_fn(x, alpha)

    initialSolns = []
    localSolns = []

    for iteration in range(iterations):
        # This loop is parallelizable (each iteration is independent of all
        # other iterations).
        printer.log('Starting iteration {} of {}.'.format(iteration + 1,
                                                          iterations), 1)
        success = False
        failCount = 0
        while not success and failCount < 10:
            try:
                iterSolns = _grasp.do_grasp_iteration(
                    elements=germsList, greedyScoreFn=scoreFn, rclFn=rclFn,
                    localScoreFn=scoreFn,
                    getNeighborsFn=getNeighborsFn,
                    feasibleThreshold=feasibleThreshold,
                    initialElements=initialWeights, seed=seed,
                    verbosity=verbosity)

                initialSolns.append(iterSolns[0])
                localSolns.append(iterSolns[1])

                success = True
                printer.log('Finished iteration {} of {}.'.format(
                    iteration + 1, iterations), 1)
            except Exception as e:
                failCount += 1
                if failCount == 10:
                    raise e
                else:
                    printer.warning(e)

    finalScores = _np.array([finalScoreFn(localSoln)
                             for localSoln in localSolns])
    bestSoln = localSolns[_np.argmin(finalScores)]

    return (bestSoln, initialSolns, localSolns) if returnAll else bestSoln

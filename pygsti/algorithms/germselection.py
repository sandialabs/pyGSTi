""" Functions for selecting a complete set of germs for a GST analysis."""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings

import numpy as _np
import numpy.linalg as _nla

from .. import objects as _objs
from .. import construction as _constr
from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from . import grasp as _grasp
from . import scoring as _scoring

FLOATSIZE = 8  # in bytes: TODO: a better way


def generate_germs(target_model, randomize=True, randomization_strength=1e-2,
                   num_gs_copies=5, seed=None, candidate_germ_counts=None,
                   candidate_seed=None, force="singletons", algorithm='greedy',
                   algorithm_kwargs=None, mem_limit=None, comm=None,
                   profiler=None, verbosity=1):
    """
    Generate a germ set for doing GST with a given target model.

    This function provides a streamlined interface to a variety of germ
    selection algorithms. It's goal is to provide a method that typical users
    can run by simply providing a target model and leaving all other settings
    at their default values, while providing flexibility for users desiring
    more control to fine tune some of the general and algorithm-specific
    details.

    Currently, to break troublesome degeneracies and provide some confidence
    that the chosen germ set is amplificationally complete (AC) for all
    models in a neighborhood of the target model (rather than only the
    target model), an ensemble of models with random unitary perturbations
    to their gates must be provided or generated.

    Parameters
    ----------
    target_model : Model or list of Model
        The model you are aiming to implement, or a list of models that are
        copies of the model you are trying to implement (either with or
        without random unitary perturbations applied to the models).

    randomize : bool, optional
        Whether or not to add random unitary perturbations to the model(s)
        provided.

    randomization_strength : float, optional
        The size of the random unitary perturbations applied to gates in the
        model. See :meth:`~pygsti.objects.Model.randomize_with_unitary`
        for more details.

    num_gs_copies : int, optional
        The number of copies of the original model that should be used.

    seed : int, optional
        Seed for generating random unitary perturbations to models. Also
        passed along to stochastic germ-selection algorithms.

    candidate_germ_counts : dict, optional
        A dictionary of *germ_length* : *count* key-value pairs, specifying
        the germ "candidate list" - a list of potential germs to draw from.
        *count* is either an integer specifying the number of random germs
        considered at the given *germ_length* or the special values `"all upto"`
        that considers all of the of all non-equivalent germs of length up to
        the corresponding *germ_length*.  If None, all germs of up to length
        6 are used, the equivalent of `{6: 'all upto'}`.

    candidate_seed : int, optional
        A seed value used when randomly selecting candidate germs.  For each
        germ length being randomly selected, the germ length is added to
        the value of `candidate_seed` to get the actual seed used.

    force : str or list, optional
        A list of Circuits which *must* be included in the final germ set.
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

    mem_limit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    profiler : Profiler, optional
        A profiler object used for to track timing and memory usage.

    verbosity : int, optional
        The verbosity level of the :class:`~pygsti.objects.VerbosityPrinter`
        used to print log messages.

    Returns
    -------
    list of Circuit
        A list containing the germs making up the germ set.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    modelList = setup_model_list(target_model, randomize,
                                 randomization_strength, num_gs_copies, seed)
    gates = list(target_model.operations.keys())
    availableGermsList = []
    if candidate_germ_counts is None: candidate_germ_counts = {6: 'all upto'}
    for germLength, count in candidate_germ_counts.items():
        if count == "all upto":
            availableGermsList.extend(_constr.list_all_circuits_without_powers_and_cycles(
                gates, max_length=germLength))
        else:
            seed = None if candidate_seed is None else candidate_seed + germLength
            availableGermsList.extend(_constr.list_random_circuits_onelen(
                gates, germLength, count, seed=seed))

    if algorithm_kwargs is None:
        # Avoid danger of using empty dict for default value.
        algorithm_kwargs = {}

    if algorithm == 'greedy':
        printer.log('Using greedy algorithm.', 1)
        # Define defaults for parameters that currently have no default or
        # whose default we want to change.
        default_kwargs = {
            'germs_list': availableGermsList,
            'randomize': False,
            'seed': seed,
            'verbosity': max(0, verbosity - 1),
            'force': force,
            'score_func': 'all',
            'comm': comm,
            'mem_limit': mem_limit,
            'profiler': profiler
        }
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
        germList = build_up_breadth(model_list=modelList,
                                    **algorithm_kwargs)
        if germList is not None:
            germsetScore = calculate_germset_score(
                germList, neighborhood=modelList,
                score_func=algorithm_kwargs['score_func'])
            printer.log('Constructed germ set:', 1)
            printer.log(str([germ.str for germ in germList]), 1)
            printer.log('Score: {}'.format(germsetScore), 1)
    elif algorithm == 'grasp':
        printer.log('Using GRASP algorithm.', 1)
        # Define defaults for parameters that currently have no default or
        # whose default we want to change.
        default_kwargs = {
            'alpha': 0.1,   # No real reason for setting this value of alpha.
            'germs_list': availableGermsList,
            'randomize': False,
            'seed': seed,
            'verbosity': max(0, verbosity - 1),
            'force': force,
            'return_all': False,
            'score_func': 'all',
        }
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
        germList = grasp_germ_set_optimization(model_list=modelList,
                                               **algorithm_kwargs)
        printer.log('Constructed germ set:', 1)

        if algorithm_kwargs['return_all'] and germList[0] is not None:
            germsetScore = calculate_germset_score(
                germList[0], neighborhood=modelList,
                score_func=algorithm_kwargs['score_func'])
            printer.log(str([germ.str for germ in germList[0]]), 1)
            printer.log('Score: {}'.format(germsetScore))
        elif not algorithm_kwargs['return_all'] and germList is not None:
            germsetScore = calculate_germset_score(germList,
                                                   neighborhood=modelList)
            printer.log(str([germ.str for germ in germList]), 1)
            printer.log('Score: {}'.format(germsetScore), 1)
    elif algorithm == 'slack':
        printer.log('Using slack algorithm.', 1)
        # Define defaults for parameters that currently have no default or
        # whose default we want to change.
        default_kwargs = {
            'germs_list': availableGermsList,
            'randomize': False,
            'seed': seed,
            'verbosity': max(0, verbosity - 1),
            'force': force,
            'score_func': 'all',
        }
        if ('slack_frac' not in algorithm_kwargs
                and 'fixed_slack' not in algorithm_kwargs):
            algorithm_kwargs['slack_frac'] = 0.1
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
        germList = optimize_integer_germs_slack(modelList,
                                                **algorithm_kwargs)
        if germList is not None:
            germsetScore = calculate_germset_score(
                germList, neighborhood=modelList,
                score_func=algorithm_kwargs['score_func'])
            printer.log('Constructed germ set:', 1)
            printer.log(str([germ.str for germ in germList]), 1)
            printer.log('Score: {}'.format(germsetScore), 1)
    else:
        raise ValueError("'{}' is not a valid algorithm "
                         "identifier.".format(algorithm))

    return germList


def calculate_germset_score(germs, target_model=None, neighborhood=None,
                            neighborhood_size=5,
                            randomization_strength=1e-2, score_func='all',
                            op_penalty=0.0, l1_penalty=0.0):
    """Calculate the score of a germ set with respect to a model.
    """
    def score_fn(x): return _scoring.list_score(x, score_func=score_func)
    if neighborhood is None:
        neighborhood = [target_model.randomize_with_unitary(randomization_strength)
                        for n in range(neighborhood_size)]
    scores = [compute_composite_germ_score(score_fn, model=model,
                                           partial_germs_list=germs,
                                           op_penalty=op_penalty,
                                           l1_penalty=l1_penalty)
              for model in neighborhood]

    return max(scores)


def get_model_params(model_list):
    """Get the number of gates and gauge parameters of the models in a list.
    Also verify all models have the same number of gates and gauge parameters.
    Parameters
    ----------
    model_list : list of Model
        A list of models for which you want an AC germ set.
    Returns
    -------
    reducedModelList : list of Model
        The original list of models with SPAM removed
    numGaugeParams : int
        The number of non-SPAM gauge parameters for all models.
    numNonGaugeParams : int
        The number of non-SPAM non-gauge parameters for all models.
    numOps : int
        The number of gates for all models.
    Raises
    ------
    ValueError
        If the number of gauge parameters or gates varies among the models.
    """
    # We don't care about SPAM, since it can't be amplified.
    reducedModelList = [remove_spam_vectors(model)
                        for model in model_list]

    # All the models should have the same number of parameters and gates, but
    # let's be paranoid here for the time being and make sure.
    numGaugeParamsList = [reducedModel.num_gauge_params()
                          for reducedModel in reducedModelList]
    numGaugeParams = numGaugeParamsList[0]
    if not all([numGaugeParams == otherNumGaugeParams
                for otherNumGaugeParams in numGaugeParamsList[1:]]):
        raise ValueError("All models must have the same number of gauge "
                         "parameters!")

    numNonGaugeParamsList = [reducedModel.num_nongauge_params()
                             for reducedModel in reducedModelList]
    numNonGaugeParams = numNonGaugeParamsList[0]
    if not all([numNonGaugeParams == otherNumNonGaugeParams
                for otherNumNonGaugeParams in numNonGaugeParamsList[1:]]):
        raise ValueError("All models must have the same number of non-gauge "
                         "parameters!")

    numOpsList = [len(reducedModel.operations)
                  for reducedModel in reducedModelList]
    numOps = numOpsList[0]
    if not all([numOps == otherNumOps
                for otherNumOps in numOpsList[1:]]):
        raise ValueError("All models must have the same number of gates!")

    return reducedModelList, numGaugeParams, numNonGaugeParams, numOps


def setup_model_list(model_list, randomize, randomization_strength,
                     num_copies, seed):
    """
    Sets up a list of randomize models (helper function).
    """
    if not isinstance(model_list, (list, tuple)):
        model_list = [model_list]
    if len(model_list) > 1 and num_copies is not None:
        _warnings.warn("Ignoring num_copies={} since multiple models were "
                       "supplied.".format(num_copies))

    if randomize:
        model_list = randomize_model_list(model_list, randomization_strength,
                                          num_copies, seed)

    return model_list


def compute_composite_germ_score(score_fn, threshold_ac=1e6, init_n=1,
                                 partial_deriv_dagger_deriv=None, model=None,
                                 partial_germs_list=None, eps=None, num_gauge_params=None,
                                 op_penalty=0.0, germ_lengths=None, l1_penalty=0.0):
    """
    Compute the score for a germ set when it is not AC against a model.

    Normally scores computed for germ sets against models for which they are
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
    score_fn : callable
        A function that takes as input a list of sorted eigenvalues and returns
        a score for the partial germ set based on those eigenvalues, with lower
        scores indicating better germ sets. Usually some flavor of
        :func:`~pygsti.algorithms.scoring.list_score`.
    threshold_ac : float, optional
        Value which the score (before penalties are applied) must be lower than
        for the germ set to be considered AC.
    init_n : int
        The number of largest eigenvalues to begin with checking.
    partial_deriv_dagger_deriv : numpy.array, optional
        Array with three axes, where the first axis indexes individual germs
        within the partial germ set and the remaining axes index entries in the
        positive square of the Jacobian of each individual germ's parameters
        with respect to the model parameters.
        If this array is not supplied it will need to be computed from
        `germs_list` and `model`, which will take longer, so it is recommended
        to precompute this array if this routine will be called multiple times.
    model : Model, optional
        The model against which the germ set is to be scored. Not needed if
        `partial_deriv_dagger_deriv` is provided.
    partial_germs_list : list of Circuit, optional
        The list of germs in the partial germ set to be evaluated. Not needed
        if `partial_deriv_dagger_deriv` (and `germ_lengths` when
        ``op_penalty > 0``) are provided.
    eps : float, optional
        Used when calculating `partial_deriv_dagger_deriv` to determine if two
        eigenvalues are equal (see :func:`bulk_twirled_deriv` for details). Not
        used if `partial_deriv_dagger_deriv` is provided.
    num_gauge_params : int
        The number of gauge parameters of the model. Not needed if `model`
        is provided.
    op_penalty : float, optional
        Coefficient for a penalty linear in the sum of the germ lengths.
    germ_lengths : numpy.array, optional
        The length of each germ. Not needed if `op_penalty` is ``0.0`` or
        `partial_germs_list` is provided.
    l1_penalty : float, optional
        Coefficient for a penalty linear in the number of germs.
    Returns
    -------
    CompositeScore
        The score for the germ set indicating how many parameters it amplifies
        and its numerical score restricted to those parameters.
    """
    if partial_deriv_dagger_deriv is None:
        if model is None or partial_germs_list is None:
            raise ValueError("Must provide either partial_deriv_dagger_deriv or "
                             "(model, partial_germs_list)!")
        else:
            pDDD_kwargs = {'model': model, 'germs_list': partial_germs_list}
            if eps is not None:
                pDDD_kwargs['eps'] = eps
            if germ_lengths is not None:
                pDDD_kwargs['germ_lengths'] = germ_lengths
            partial_deriv_dagger_deriv = calc_bulk_twirled_ddd(**pDDD_kwargs)

    if num_gauge_params is None:
        if model is None:
            raise ValueError("Must provide either num_gauge_params or model!")
        else:
            num_gauge_params = remove_spam_vectors(model).num_gauge_params()

    # Calculate penalty scores
    numGerms = partial_deriv_dagger_deriv.shape[0]
    l1Score = l1_penalty * numGerms
    opScore = 0.0
    if op_penalty != 0.0:
        if germ_lengths is None:
            if partial_germs_list is None:
                raise ValueError("Must provide either germ_lengths or "
                                 "partial_germs_list when op_penalty != 0.0!")
            else:
                germ_lengths = _np.array([len(germ)
                                         for germ in partial_germs_list])
        opScore = op_penalty * _np.sum(germ_lengths)

    combinedDDD = _np.sum(partial_deriv_dagger_deriv, axis=0)
    sortedEigenvals = _np.sort(_np.real(_nla.eigvalsh(combinedDDD)))
    observableEigenvals = sortedEigenvals[num_gauge_params:]
    N_AC = 0
    AC_score = _np.inf
    for N in range(init_n, len(observableEigenvals) + 1):
        scoredEigenvals = observableEigenvals[-N:]
        candidate_AC_score = score_fn(scoredEigenvals)
        if candidate_AC_score > threshold_ac:
            break   # We've found a set of parameters for which the germ set
            # is not AC.
        else:
            AC_score = candidate_AC_score
            N_AC = N

    # OLD Apply penalties to the minor score; major part is just #amplified
    #major_score = N_AC
    #minor_score = AC_score + l1Score + opScore

    # Apply penalties to the major score
    major_score = -N_AC + opScore + l1Score
    minor_score = AC_score
    ret = _scoring.CompositeScore(major_score, minor_score, N_AC)
    #DEBUG: ret.extra = {'opScore': opScore,
    #    'sum(germ_lengths)': _np.sum(germ_lengths), 'l1': l1Score}
    return ret


def calc_bulk_twirled_ddd(model, germs_list, eps=1e-6, check=False,
                          germ_lengths=None, comm=None):
    """Calculate the positive squares of the germ Jacobians.
    twirledDerivDaggerDeriv == array J.H*J contributions from each germ
    (J=Jacobian) indexed by (iGerm, iModelParam1, iModelParam2)
    size (nGerms, vec_model_dim, vec_model_dim)
    """
    if germ_lengths is None:
        germ_lengths = _np.array([len(germ) for germ in germs_list])

    twirledDeriv = bulk_twirled_deriv(model, germs_list, eps, check, comm) / germ_lengths[:, None, None]

    #OLD: slow, I think because conjugate *copies* a large tensor, causing a memory bottleneck
    #twirledDerivDaggerDeriv = _np.einsum('ijk,ijl->ikl',
    #                                     _np.conjugate(twirledDeriv),
    #                                     twirledDeriv)

    #NEW: faster, one-germ-at-a-time computation requires less memory.
    nGerms, _, vec_model_dim = twirledDeriv.shape
    twirledDerivDaggerDeriv = _np.empty((nGerms, vec_model_dim, vec_model_dim),
                                        dtype=_np.complex)
    for i in range(nGerms):
        twirledDerivDaggerDeriv[i, :, :] = _np.dot(
            twirledDeriv[i, :, :].conjugate().T, twirledDeriv[i, :, :])

    return twirledDerivDaggerDeriv


def calc_twirled_ddd(model, germ, eps=1e-6):
    """Calculate the positive squares of the germ Jacobian.
    twirledDerivDaggerDeriv == array J.H*J contributions from `germ`
    (J=Jacobian) indexed by (iModelParam1, iModelParam2)
    size (vec_model_dim, vec_model_dim)
    """
    twirledDeriv = twirled_deriv(model, germ, eps) / len(germ)
    #twirledDerivDaggerDeriv = _np.einsum('jk,jl->kl',
    #                                     _np.conjugate(twirledDeriv),
    #                                     twirledDeriv)
    twirledDerivDaggerDeriv = _np.tensordot(_np.conjugate(twirledDeriv),
                                            twirledDeriv, (0, 0))

    return twirledDerivDaggerDeriv


def compute_score(weights, model_num, score_func, deriv_dagger_deriv_list,
                  force_indices, force_score,
                  n_gauge_params, op_penalty, germ_lengths, l1_penalty=1e-2,
                  score_dict=None):
    """Returns a germ set "score" in which smaller is better.  Also returns
    intentionally bad score (`force_score`) if `weights` is zero on any of
    the "forced" germs (i.e. at any index in `forcedIndices`).
    This function is included for use by :func:`optimize_integer_germs_slack`,
    but is not convenient for just computing the score of a germ set. For that,
    use :func:`calculate_germset_score`.
    """
    if force_indices is not None and _np.any(weights[force_indices] <= 0):
        score = force_score
    else:
        #combinedDDD = _np.einsum('i,ijk', weights,
        #                         deriv_dagger_deriv_list[model_num])
        combinedDDD = _np.squeeze(
            _np.tensordot(_np.expand_dims(weights, 1),
                          deriv_dagger_deriv_list[model_num], (0, 0)))
        assert len(combinedDDD.shape) == 2

        sortedEigenvals = _np.sort(_np.real(_nla.eigvalsh(combinedDDD)))
        observableEigenvals = sortedEigenvals[n_gauge_params:]
        score = (_scoring.list_score(observableEigenvals, score_func)
                 + l1_penalty * _np.sum(weights)
                 + op_penalty * _np.dot(germ_lengths, weights))
    if score_dict is not None:
        # Side effect: calling compute_score caches result in score_dict
        score_dict[model_num, tuple(weights)] = score
    return score


def randomize_model_list(model_list, randomization_strength, num_copies,
                         seed=None):
    """
    Applies random unitary perturbations to a models.

    If `model_list` is a length-1 list, then `num_copies` determines how
    many randomizations to create.  If `model_list` containes multiple
    models, then `num_copies` must be `None` and each model is
    randomized once to create the corresponding returned model.

    Parameters
    ----------
    model_list : Model or list
        A list of Model objects.

    randomizationStrengh : float
        The strength (input as the `scale` argument to
        :func:`Model.randomize_with_unitary`) of random unitary
        perturbations.

    num_copies : int
        The number of random perturbations of `model_list[0]` to generate when
        `len(model_list) == 1`.  A value of `None` will result in 1 copy.  If
        `len(model_list) > 1` then `num_copies` must be set to None.

    seed : int, optional
        Starting seed for randomization.  Successive randomizations receive
        successive seeds.  `None` results in random seeds.
    """
    if len(model_list) > 1 and num_copies is not None:
        raise ValueError("Input multiple models XOR request multiple "
                         "copies only!")

    newmodelList = []
    if len(model_list) > 1:
        for modelnum, model in enumerate(model_list):
            newmodelList.append(model.randomize_with_unitary(
                randomization_strength,
                seed=None if seed is None else seed + modelnum))
    else:
        for modelnum in range(num_copies if num_copies is not None else 1):
            newmodelList.append(model_list[0].randomize_with_unitary(
                randomization_strength,
                seed=None if seed is None else seed + modelnum))
    return newmodelList


def check_germs_list_completeness(model_list, germs_list, score_func, threshold):
    """Check to see if the germs_list is amplificationally complete (AC)
    Checks for AC with respect to all the Models in `model_list`, returning
    the index of the first Model for which it is not AC or `-1` if it is AC
    for all Models.
    """
    for modelNum, model in enumerate(model_list):
        initial_test = test_germ_list_infl(model, germs_list,
                                           score_func=score_func,
                                           threshold=threshold)
        if not initial_test:
            return modelNum

    # If the germs_list is complete for all models, return -1
    return -1


def remove_spam_vectors(model):
    """
    Returns a copy of `model` with state preparations and effects removed.

    Parameters
    ----------
    model : Model

    Returns
    -------
    Model
    """
    reducedModel = model.copy()
    for prepLabel in list(reducedModel.preps.keys()):
        del reducedModel.preps[prepLabel]
    for povmLabel in list(reducedModel.povms.keys()):
        del reducedModel.povms[povmLabel]
    return reducedModel


def num_non_spam_gauge_params(model):
    """
    Return the number of non-gauge, non-SPAM parameters in `model`.

    Equivalent to `remove_spam_vectors(model).num_gauge_params()`.

    Parameters
    ---------
    model : Model

    Returns
    -------
    int
    """
    return remove_spam_vectors(model).num_gauge_params()


# wrt is op_dim x op_dim, so is M, Minv, Proj
# so SOP is op_dim^2 x op_dim^2 and acts on vectorized *gates*
# Recall vectorizing identity (when vec(.) concats rows as flatten does):
#     vec( A * X * B ) = A tensor B^T * vec( X )
def _super_op_for_perfect_twirl(wrt, eps):
    """Return super operator for doing a perfect twirl with respect to wrt.
    """
    assert wrt.shape[0] == wrt.shape[1]  # only square matrices allowed
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
        #if _np.linalg.norm(A.imag) > 1e-6:
        #    print("DB: imag = ",_np.linalg.norm(A.imag))
        #assert(_np.linalg.norm(A.imag) < 1e-6)
        #A = _np.real(A)
        # Need to normalize, because we are overcounting projectors onto
        # subspaces of dimension d > 1, giving us d * Proj_i tensor Proj_i^T.
        # We can fix this with a division by tr(Proj_i) = d.
        SuperOp += _np.kron(A, A.T) / _np.trace(Proj_i)
        # SuperOp += _np.kron(A.T,A) # Mimic Maple version (but I think this is
        # wrong... or it doesn't matter?)
    return SuperOp  # a op_dim^2 x op_dim^2 matrix


def sq_sing_vals_from_deriv(deriv, weights=None):
    """Calculate the squared singulare values of the Jacobian of the germ set.
    Parameters
    ----------
    deriv : numpy.array
        Array of shape ``(nGerms, flattened_op_dim, vec_model_dim)``. Each
        sub-array corresponding to an individual germ is the Jacobian of the
        vectorized gate representation of that germ raised to some power with
        respect to the model parameters, normalized by dividing by the length
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
    # shape (nGerms, vec_model_dim, vec_model_dim)
    derivDaggerDeriv = _np.einsum('ijk,ijl->ikl', _np.conjugate(deriv), deriv)
    # awkward to convert to tensordot, so leave as einsum

    # Take the average of the D^dagger*D/L^2 matrices associated with each germ
    # with optional weights.
    combinedDDD = _np.average(derivDaggerDeriv, weights=weights, axis=0)
    sortedEigenvals = _np.sort(_np.real(_nla.eigvalsh(combinedDDD)))

    return sortedEigenvals


def twirled_deriv(model, circuit, eps=1e-6):
    """Compute the "Twirled Derivative" of a circuit.
    The twirled derivative is obtained by acting on the standard derivative of
    a operation sequence with the twirling superoperator.

    Parameters
    ----------
    model : Model object
        The Model which associates operation labels with operators.
    circuit : Circuit object
        The operation sequence to take a twirled derivative of.
    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. abs(eval1 - eval2) < eps ? )

    Returns
    -------
    numpy array
      An array of shape (op_dim^2, num_model_params)
    """
    prod = model.product(circuit)

    # flattened_op_dim x vec_model_dim
    dProd = model.dproduct(circuit, flat=True)

    # flattened_op_dim x flattened_op_dim
    twirler = _super_op_for_perfect_twirl(prod, eps)

    # flattened_op_dim x vec_model_dim
    return _np.dot(twirler, dProd)


def bulk_twirled_deriv(model, circuits, eps=1e-6, check=False, comm=None):
    """
    Compute the "Twirled Derivative" of a set of circuits.

    The twirled derivative is obtained by acting on the standard derivative of
    a operation sequence with the twirling superoperator.

    Parameters
    ----------
    model : Model object
        The Model which associates operation labels with operators.

    circuits : list of Circuit objects
        The operation sequence to take a twirled derivative of.

    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. abs(eval1 - eval2) < eps ? )

    check : bool, optional
        Whether to perform internal consistency checks, at the expense of
        making the function slower.

    comm : mpi4py.MPI.Comm, optional
      When not None, an MPI communicator for distributing the computation
      across multiple processors.


    Returns
    -------
    numpy array
        An array of shape (num_simplified_circuits, op_dim^2, num_model_params)
    """
    if len(model.preps) > 0 or len(model.povms) > 0:
        model = remove_spam_vectors(model)
        # This function assumes model has no spam elements so `lookup` below
        #  gives indexes into products computed by evalTree.

    evalTree, lookup, _ = model.bulk_evaltree(circuits)
    dProds, prods = model.bulk_dproduct(evalTree, flat=True, return_prods=True, comm=comm)
    op_dim = model.get_dimension()
    fd = op_dim**2  # flattened gate dimension

    nOrigStrs = len(circuits)

    ret = _np.empty((nOrigStrs, fd, dProds.shape[1]), 'complex')
    for iOrig in range(nOrigStrs):
        iArray = _slct.as_array(lookup[iOrig])
        assert(iArray.size == 1), ("Simplified lookup table should have length-1"
                                   " element slices!  Maybe you're using a"
                                   " Model without SPAM elements removed?")
        i = iArray[0]  # get evalTree-final index (within dProds or prods)

        # flattened_op_dim x flattened_op_dim
        twirler = _super_op_for_perfect_twirl(prods[i], eps)

        # flattened_op_dim x vec_model_dim
        ret[iOrig] = _np.dot(twirler, dProds[i * fd:(i + 1) * fd])

    if check:
        for i, circuit in enumerate(circuits):
            chk_ret = twirled_deriv(model, circuit, eps)
            if _nla.norm(ret[i] - chk_ret) > 1e-6:
                _warnings.warn("bulk twirled derivative norm mismatch = "
                               "%g - %g = %g"
                               % (_nla.norm(ret[i]), _nla.norm(chk_ret),
                                  _nla.norm(ret[i] - chk_ret)))  # pragma: no cover

    return ret  # nSimplifiedCircuits x flattened_op_dim x vec_model_dim


def test_germ_list_finitel(model, germs_to_test, length, weights=None,
                           return_spectrum=False, tol=1e-6):
    """Test whether a set of germs is able to amplify all non-gauge parameters.

    Parameters
    ----------
    model : Model
        The Model (associates operation matrices with operation labels).
    germs_to_test : list of Circuits
        List of germs operation sequences to test for completeness.
    length : int
        The finite length to use in amplification testing.  Larger
        values take longer to compute but give more robust results.
    weights : numpy array, optional
        A 1-D array of weights with length equal len(germs_to_test),
        which multiply the contribution of each germ to the total
        jacobian matrix determining parameter amplification. If
        None, a uniform weighting of 1.0/len(germs_to_test) is applied.
    return_spectrum : bool, optional
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
        Only returned when `return_spectrum` is ``True``.  Sorted array of
        eigenvalues (from small to large) of the jacobian^T * jacobian
        matrix used to determine parameter amplification.
    """
    # Remove any SPAM vectors from model since we only want
    # to consider the set of *gate* parameters for amplification
    # and this makes sure our parameter counting is correct
    model = remove_spam_vectors(model)

    nGerms = len(germs_to_test)
    germToPowL = [germ * length for germ in germs_to_test]

    op_dim = model.get_dimension()
    evt, lookup, _ = model.bulk_evaltree(germToPowL)

    # shape (nGerms*flattened_op_dim, vec_model_dim)
    dprods = model.bulk_dproduct(evt, flat=True)
    dprods.shape = (evt.num_final_strings(), op_dim**2, dprods.shape[1])
    prod_inds = [_slct.as_array(lookup[i]) for i in range(nGerms)]
    assert(all([len(x) == 1 for x in prod_inds])), \
        ("Simplified lookup table should have length-1"
         " element slices!  Maybe you're using a"
         " Model without SPAM elements removed?")
    dprods = _np.take(dprods, _np.concatenate(prod_inds), axis=0)
    # shape (nGerms, flattened_op_dim, vec_model_dim

    germLengths = _np.array([len(germ) for germ in germs_to_test], 'd')

    normalizedDeriv = dprods / (length * germLengths[:, None, None])

    sortedEigenvals = sq_sing_vals_from_deriv(normalizedDeriv, weights)

    nGaugeParams = model.num_gauge_params()

    observableEigenvals = sortedEigenvals[nGaugeParams:]

    bSuccess = bool(_scoring.list_score(observableEigenvals, 'worst') < 1 / tol)

    return (bSuccess, sortedEigenvals) if return_spectrum else bSuccess


def test_germ_list_infl(model, germs_to_test, score_func='all', weights=None,
                        return_spectrum=False, threshold=1e6, check=False):
    """Test whether a set of germs is able to amplify all non-gauge parameters.

    Parameters
    ----------
    model : Model
        The Model (associates operation matrices with operation labels).
    germs_to_test : list of Circuit
        List of germs operation sequences to test for completeness.
    score_func : string
        Label to indicate how a germ set is scored. See
        :func:`~pygsti.algorithms.scoring.list_score` for details.
    weights : numpy array, optional
        A 1-D array of weights with length equal len(germs_to_test),
        which multiply the contribution of each germ to the total
        jacobian matrix determining parameter amplification. If
        None, a uniform weighting of 1.0/len(germs_to_test) is applied.
    return_spectrum : bool, optional
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
        Only returned when `return_spectrum` is ``True``.  Sorted array of
        eigenvalues (from small to large) of the jacobian^T * jacobian
        matrix used to determine parameter amplification.
    """
    # Remove any SPAM vectors from model since we only want
    # to consider the set of *gate* parameters for amplification
    # and this makes sure our parameter counting is correct
    model = remove_spam_vectors(model)

    germLengths = _np.array([len(germ) for germ in germs_to_test], _np.int64)
    twirledDerivDaggerDeriv = calc_bulk_twirled_ddd(model, germs_to_test,
                                                    1. / threshold, check,
                                                    germLengths)
    # result[i] = _np.dot( twirledDeriv[i].H, twirledDeriv[i] ) i.e. matrix
    # product
    # result[i,k,l] = sum_j twirledDerivH[i,k,j] * twirledDeriv(i,j,l)
    # result[i,k,l] = sum_j twirledDeriv_conj[i,j,k] * twirledDeriv(i,j,l)

    if weights is None:
        nGerms = len(germs_to_test)
        # weights = _np.array( [1.0/nGerms]*nGerms, 'd')
        weights = _np.array([1.0] * nGerms, 'd')

    #combinedTDDD = _np.einsum('i,ijk->jk', weights, twirledDerivDaggerDeriv)
    combinedTDDD = _np.tensordot(weights, twirledDerivDaggerDeriv, (0, 0))
    sortedEigenvals = _np.sort(_np.real(_np.linalg.eigvalsh(combinedTDDD)))

    nGaugeParams = model.num_gauge_params()
    observableEigenvals = sortedEigenvals[nGaugeParams:]

    bSuccess = bool(_scoring.list_score(observableEigenvals, score_func)
                    < threshold)

    return (bSuccess, sortedEigenvals) if return_spectrum else bSuccess


def build_up(model_list, germs_list, randomize=True,
             randomization_strength=1e-3, num_copies=None, seed=0, op_penalty=0,
             score_func='all', tol=1e-6, threshold=1e6, check=False,
             force="singletons", verbosity=0):
    """Greedy algorithm starting with 0 germs.
    Tries to minimize the number of germs needed to achieve amplificational
    completeness (AC). Begins with 0 germs and adds the germ that increases the
    score used to check for AC by the largest amount at each step, stopping when
    the threshold for AC is achieved.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    model_list = setup_model_list(model_list, randomize,
                                  randomization_strength, num_copies, seed)

    (reducedModelList,
     numGaugeParams, _, _) = get_model_params(model_list)

    germLengths = _np.array([len(germ) for germ in germs_list], _np.int64)
    numGerms = len(germs_list)

    weights = _np.zeros(numGerms, _np.int64)
    goodGerms = []
    if force:
        if force == "singletons":
            weights[_np.where(germLengths == 1)] = 1
            goodGerms = [germ for germ
                         in _np.array(germs_list)[_np.where(germLengths == 1)]]
        else:  # force should be a list of Circuits
            for opstr in force:
                weights[germs_list.index(opstr)] = 1
            goodGerms = force[:]

    undercompleteModelNum = check_germs_list_completeness(model_list,
                                                          germs_list,
                                                          score_func,
                                                          threshold)
    if undercompleteModelNum > -1:
        printer.warning("Complete initial germ set FAILS on model "
                        + str(undercompleteModelNum) + ". Aborting search.")
        return None

    printer.log("Complete initial germ set succeeds on all input models.", 1)
    printer.log("Now searching for best germ set.", 1)
    printer.log("Starting germ set optimization. Lower score is better.", 1)

    twirledDerivDaggerDerivList = [calc_bulk_twirled_ddd(model, germs_list, tol,
                                                         check, germLengths)
                                   for model in model_list]

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'score_fn': lambda x: _scoring.list_score(x, score_func=score_func),
        'threshold_ac': threshold,
        'num_gauge_params': numGaugeParams,
        'op_penalty': op_penalty,
        'germ_lengths': germLengths,
    }

    for modelNum, reducedModel in enumerate(reducedModelList):
        derivDaggerDeriv = twirledDerivDaggerDerivList[modelNum]
        # Make sure the set of germs you come up with is AC for all
        # models.
        # Remove any SPAM vectors from model since we only want
        # to consider the set of *gate* parameters for amplification
        # and this makes sure our parameter counting is correct
        while _np.any(weights == 0):

            # As long as there are some unused germs, see if you need to add
            # another one.
            if test_germ_list_infl(reducedModel, goodGerms,
                                   score_func=score_func, threshold=threshold):
                # The germs are sufficient for the current model
                break
            candidateGerms = _np.where(weights == 0)[0]
            candidateGermScores = []
            for candidateGermIdx in _np.where(weights == 0)[0]:
                # If the germs aren't sufficient, try adding a single germ
                candidateWeights = weights.copy()
                candidateWeights[candidateGermIdx] = 1
                partialDDD = derivDaggerDeriv[
                    _np.where(candidateWeights == 1)[0], :, :]
                candidateGermScore = compute_composite_germ_score(
                    partial_deriv_dagger_deriv=partialDDD, **nonAC_kwargs)
                candidateGermScores.append(candidateGermScore)
            # Add the germ that give the best score
            bestCandidateGerm = candidateGerms[_np.array(
                candidateGermScores).argmin()]
            weights[bestCandidateGerm] = 1
            goodGerms.append(germs_list[bestCandidateGerm])

    return goodGerms


def build_up_breadth(model_list, germs_list, randomize=True,
                     randomization_strength=1e-3, num_copies=None, seed=0,
                     op_penalty=0, score_func='all', tol=1e-6, threshold=1e6,
                     check=False, force="singletons", pretest=True, mem_limit=None,
                     comm=None, profiler=None, verbosity=0):
    """
    Greedy algorithm starting with 0 germs.

    Tries to minimize the number of germs needed to achieve amplificational
    completeness (AC). Begins with 0 germs and adds the germ that increases the
    score used to check for AC by the largest amount (for the model that
    currently has the lowest score) at each step, stopping when the threshold
    for AC is achieved. This strategy is something of a "breadth-first"
    approach, in contrast to :func:`build_up`, which only looks at the
    scores for one model at a time until that model achieves AC, then
    turning it's attention to the remaining models.

    Parameters
    ----------
    model_list : Model or list
        The model or list of `Model`s to select germs for.

    germs_list : list of Circuit
        The list of germs to contruct a germ set from.

    randomize : bool, optional
        Whether or not to randomize `model_list` (usually just a single
        `Model`) with small (see `randomizationStrengh`) unitary maps
        in order to avoid "accidental" symmetries which could allow for
        fewer germs but *only* for that particular model.  Setting this
        to `True` will increase the run time by a factor equal to the
        numer of randomized copies (`num_copies`).

    randomizationStrengh : float, optional
        The strength (input as the `scale` argument to
        :func:`Model.randomize_with_unitary`) of random unitary
        perturbations (used only when `randomize == True`).

    num_copies : int, optional
        The number of randomized models to create when only a *single* gate
        set is passed via `model_list`.  Otherwise, `num_copies` must be set
        to `None`.

    seed : int, optional
        Seed for generating random unitary perturbations to models.

    op_penalty : float, optional
        Coefficient for a penalty linear in the sum of the germ lengths.

    score_func : {'all', 'worst'}, optional
        Sets the objective function for scoring the eigenvalues. If 'all',
        score is ``sum(1/eigenvalues)``. If 'worst', score is
        ``1/min(eiganvalues)``.

    tol : float, optional
        Tolerance (`eps` arg) for :func:`calc_bulk_twirled_ddd`, which sets
        the differece between eigenvalues below which they're treated as
        degenerate.

    threshold : float, optional
        Value which the score (before penalties are applied) must be lower than
        for a germ set to be considered AC.

    check : bool, optional
        Whether to perform internal checks (will slow down run time
        substantially).

    force : list of Circuits
        A list of `Circuit` objects which *must* be included in the final
        germ set.  If the special string "singletons" is given, then all of
        the single gates (length-1 sequences) must be included.

    pretest : boolean, optional
        Whether germ list should be initially checked for completeness.

    mem_limit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    profiler : Profiler, optional
        A profiler object used for to track timing and memory usage.

    verbosity : int, optional
        Level of detail printed to stdout.
    """
    if comm is not None and comm.Get_size() > 1:
        from mpi4py import MPI  # not at top so pygsti doesn't require mpi4py

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    model_list = setup_model_list(model_list, randomize,
                                  randomization_strength, num_copies, seed)

    dim = model_list[0].dim
    #Np = model_list[0].num_params() #wrong:? includes spam...
    Np = model_list[0].num_params()
    #print("DB Np = %d, Ng = %d" % (Np,Ng))
    assert(all([(mdl.dim == dim) for mdl in model_list])), \
        "All models must have the same dimension!"
    #assert(all([(mdl.num_params() == Np) for mdl in model_list])), \
    #    "All models must have the same number of parameters!"

    (_, numGaugeParams,
     numNonGaugeParams, _) = get_model_params(model_list)
    germLengths = _np.array([len(germ) for germ in germs_list], _np.int64)

    numGerms = len(germs_list)

    goodGerms = []
    weights = _np.zeros(numGerms, _np.int64)
    if force:
        if force == "singletons":
            weights[_np.where(germLengths == 1)] = 1
            goodGerms = [germ for germ
                         in _np.array(germs_list)[_np.where(germLengths == 1)]]
        else:  # force should be a list of Circuits
            for opstr in force:
                weights[germs_list.index(opstr)] = 1
            goodGerms = force[:]

    if pretest:
        undercompleteModelNum = check_germs_list_completeness(model_list,
                                                              germs_list,
                                                              score_func,
                                                              threshold)
        if undercompleteModelNum > -1:
            printer.warning("Complete initial germ set FAILS on model "
                            + str(undercompleteModelNum) + ".")
            printer.warning("Aborting search.")
            return None

        printer.log("Complete initial germ set succeeds on all input models.", 1)
        printer.log("Now searching for best germ set.", 1)

    printer.log("Starting germ set optimization. Lower score is better.", 1)

    mode = "all-Jac"  # compute a all the possible germ's jacobians at once up
    # front and store them separately (requires lots of mem)

    if mem_limit is not None:
        memEstimate = FLOATSIZE * len(model_list) * len(germs_list) * Np**2
        # for calc_bulk_twirled_ddd
        memEstimate += FLOATSIZE * len(model_list) * len(germs_list) * dim**2 * Np
        # for bulk_twirled_deriv sub-call
        printer.log("Memory estimate of %.1f GB (%.1f GB limit) for all-Jac mode." %
                    (memEstimate / 1024.0**3, mem_limit / 1024.0**3), 1)

        if memEstimate > mem_limit:
            mode = "single-Jac"  # compute a single germ's jacobian at a time
            # and store the needed J-sum over chosen germs.
            memEstimate = FLOATSIZE * 3 * len(model_list) * Np**2 + \
                FLOATSIZE * 3 * len(model_list) * dim**2 * Np
            #Factor of 3 accounts for currentDDDs, testDDDs, and bestDDDs
            printer.log("Memory estimate of %.1f GB (%.1f GB limit) for single-Jac mode." %
                        (memEstimate / 1024.0**3, mem_limit / 1024.0**3), 1)

            if memEstimate > mem_limit:
                raise MemoryError("Too little memory, even for single-Jac mode!")

    twirledDerivDaggerDerivList = None

    if mode == "all-Jac":
        twirledDerivDaggerDerivList = \
            [calc_bulk_twirled_ddd(model, germs_list, tol,
                                   check, germLengths, comm)
             for model in model_list]

        currentDDDList = []
        for i, derivDaggerDeriv in enumerate(twirledDerivDaggerDerivList):
            currentDDDList.append(_np.sum(derivDaggerDeriv[_np.where(weights == 1)[0], :, :], axis=0))

    elif mode == "single-Jac":
        currentDDDList = [_np.zeros((Np, Np), 'complex') for mdl in model_list]

        loc_Indices, _, _ = _mpit.distribute_indices(
            list(range(len(goodGerms))), comm, False)

        with printer.progress_logging(3):
            for i, goodGermIdx in enumerate(loc_Indices):
                printer.show_progress(i, len(loc_Indices),
                                      prefix="Initial germ set computation",
                                      suffix=germs_list[goodGermIdx].str)
                #print("DB: Rank%d computing initial index %d" % (comm.Get_rank(),goodGermIdx))

                for k, model in enumerate(model_list):
                    currentDDDList[k] += calc_twirled_ddd(
                        model, germs_list[goodGermIdx], tol)

        #aggregate each currendDDDList across all procs
        if comm is not None and comm.Get_size() > 1:
            for k, model in enumerate(model_list):
                result = _np.empty((Np, Np), 'complex')
                comm.Allreduce(currentDDDList[k], result, op=MPI.SUM)
                currentDDDList[k][:, :] = result[:, :]
                result = None  # free mem

    else:  # should be unreachable since we set 'mode' internally above
        raise ValueError("Invalid mode: %s" % mode)  # pragma: no cover

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'score_fn': lambda x: _scoring.list_score(x, score_func=score_func),
        'threshold_ac': threshold,
        'num_gauge_params': numGaugeParams,
        'op_penalty': op_penalty,
        'germ_lengths': germLengths,
    }

    initN = 1
    while _np.any(weights == 0):
        printer.log("Outer iteration: %d of %d amplified, %d germs" %
                    (initN, numNonGaugeParams, len(goodGerms)), 2)
        # As long as there are some unused germs, see if you need to add
        # another one.
        if initN == numNonGaugeParams:
            break   # We are AC for all models, so we can stop adding germs.

        candidateGermIndices = _np.where(weights == 0)[0]
        loc_candidateIndices, owners, _ = _mpit.distribute_indices(
            candidateGermIndices, comm, False)

        # Since the germs aren't sufficient, add the best single candidate germ
        bestDDDs = None
        bestGermScore = _scoring.CompositeScore(1.0e100, 0, None)  # lower is better
        iBestCandidateGerm = None
        with printer.progress_logging(3):
            for i, candidateGermIdx in enumerate(loc_candidateIndices):
                printer.show_progress(i, len(loc_candidateIndices),
                                      prefix="Inner iter over candidate germs",
                                      suffix=germs_list[candidateGermIdx].str)

                #print("DB: Rank%d computing index %d" % (comm.Get_rank(),candidateGermIdx))
                worstScore = _scoring.CompositeScore(-1.0e100, 0, None)  # worst of all models

                # Loop over all models
                testDDDs = []
                for k, currentDDD in enumerate(currentDDDList):
                    testDDD = currentDDD.copy()

                    if mode == "all-Jac":
                        #just get cached value of deriv-dagger-deriv
                        derivDaggerDeriv = twirledDerivDaggerDerivList[k][candidateGermIdx]
                        testDDD += derivDaggerDeriv

                    elif mode == "single-Jac":
                        #compute value of deriv-dagger-deriv
                        model = model_list[k]
                        testDDD += calc_twirled_ddd(
                            model, germs_list[candidateGermIdx], tol)
                    # (else already checked above)

                    nonAC_kwargs['germ_lengths'] = \
                        _np.array([len(germ) for germ in
                                   (goodGerms + [germs_list[candidateGermIdx]])])
                    worstScore = max(worstScore, compute_composite_germ_score(
                        partial_deriv_dagger_deriv=testDDD[None, :, :], init_n=initN,
                        **nonAC_kwargs))
                    testDDDs.append(testDDD)  # save in case this is a keeper

                # Take the score for the current germ to be its worst score
                # over all the models.
                germScore = worstScore
                printer.log(str(germScore), 4)
                if germScore < bestGermScore:
                    bestGermScore = germScore
                    iBestCandidateGerm = candidateGermIdx
                    bestDDDs = testDDDs
                testDDDs = None

        # Add the germ that gives the best germ score
        if comm is not None and comm.Get_size() > 1:
            #figure out which processor has best germ score and distribute
            # its information to the rest of the procs
            globalMinScore = comm.allreduce(bestGermScore, op=MPI.MIN)
            toSend = comm.Get_rank() if (globalMinScore == bestGermScore) \
                else comm.Get_size() + 1
            winningRank = comm.allreduce(toSend, op=MPI.MIN)
            bestGermScore = globalMinScore
            toCast = iBestCandidateGerm if (comm.Get_rank() == winningRank) else None
            iBestCandidateGerm = comm.bcast(toCast, root=winningRank)
            for k in range(len(model_list)):
                comm.Bcast(bestDDDs[k], root=winningRank)

        #Update variables for next outer iteration
        weights[iBestCandidateGerm] = 1
        initN = bestGermScore.N
        goodGerms.append(germs_list[iBestCandidateGerm])

        for k in range(len(model_list)):
            currentDDDList[k][:, :] = bestDDDs[k][:, :]
            bestDDDs[k] = None

            printer.log("Added %s to final germs (%s)" %
                        (germs_list[iBestCandidateGerm].str, str(bestGermScore)), 3)

    return goodGerms


#@profile
def optimize_integer_germs_slack(model_list, germs_list, randomize=True,
                                 randomization_strength=1e-3, num_copies=None,
                                 seed=0, l1_penalty=1e-2, op_penalty=0,
                                 initial_weights=None, score_func='all',
                                 max_iter=100, fixed_slack=False,
                                 slack_frac=False, return_all=False, tol=1e-6,
                                 check=False, force="singletons",
                                 force_score=1e100, threshold=1e6,
                                 verbosity=1):
    """Find a locally optimal subset of the germs in germs_list.
    Locally optimal here means that no single germ can be excluded
    without making the smallest non-gauge eigenvalue of the
    Jacobian.H*Jacobian matrix smaller, i.e. less amplified,
    by more than a fixed or variable amount of "slack", as
    specified by `fixed_slack` or `slack_frac`.
    Parameters
    ----------
    model_list : Model or list of Model
        The list of Models to be tested.  To ensure that the returned germ
        set is amplficationally complete, it is a good idea to score potential
        germ sets against a collection (~5-10) of similar models.  The user
        may specify a single Model and a number of unitarily close copies to
        be made (set by the kwarg `num_copies`), or the user may specify their
        own list of Models, each of which in turn may or may not be
        randomized (set by the kwarg `randomize`).
    germs_list : list of Circuit
        List of all germs operation sequences to consider.
    randomize : Bool, optional
        Whether or not the input Model(s) are first subject to unitary
        randomization.  If ``False``, the user should perform the unitary
        randomization themselves.  Note:  If the Model(s) are perfect (e.g.
        ``std1Q_XYI.target_model()``), then the germ selection output should not be
        trusted, due to accidental degeneracies in the Model.  If the
        Model(s) include stochastic (non-unitary) error, then germ selection
        will fail, as we score amplificational completeness in the limit of
        infinite sequence length (so any stochastic noise will completely
        depolarize any sequence in that limit).  Default is ``True``.
    randomization_strength : float, optional
        The strength of the unitary noise used to randomize input Model(s);
        is passed to :func:`~pygsti.objects.Model.randomize_with_unitary`.
        Default is ``1e-3``.
    num_copies : int, optional
        The number of Model copies to be made of the input Model (prior to
        unitary randomization).  If more than one Model is passed in,
        `num_copies` should be ``None``.  If only one Model is passed in and
        `num_copies` is ``None``, no extra copies are made.
    seed : float, optional
        The starting seed used for unitary randomization.  If multiple Models
        are to be randomized, ``model_list[i]`` is randomized with ``seed +
        i``.  Default is 0.
    l1_penalty : float, optional
        How strong the penalty should be for increasing the germ set list by a
        single germ.  Default is 1e-2.
    op_penalty : float, optional
        How strong the penalty should be for increasing a germ in the germ set
        list by a single gate.  Default is 0.
    initial_weights : list-like
        List or array of either booleans or (0 or 1) integers
        specifying which germs in `germ_list` comprise the initial
        germ set.  If ``None``, then starting point includes all
        germs.
    score_func : string
        Label to indicate how a germ set is scored. See
        :func:`~pygsti.algorithms.scoring.list_score` for details.
    max_iter : int, optional
        The maximum number of iterations before giving up.
    fixed_slack : float, optional
        If not ``None``, a floating point number which specifies that excluding
        a germ is allowed to increase 1.0/smallest-non-gauge-eigenvalue by
        `fixed_slack`.  You must specify *either* `fixed_slack` or `slack_frac`.
    slack_frac : float, optional
        If not ``None``, a floating point number which specifies that excluding
        a germ is allowed to increase 1.0/smallest-non-gauge-eigenvalue by
        `fixedFrac`*100 percent.  You must specify *either* `fixed_slack` or
        `slack_frac`.
    return_all : bool, optional
        If ``True``, return the final ``weights`` vector and score dictionary
        in addition to the optimal germ list (see below).
    tol : float, optional
        Tolerance used for eigenvector degeneracy testing in twirling
        operation.
    check : bool, optional
        Whether to perform internal consistency checks, at the
        expense of making the function slower.
    force : str or list, optional
        A list of Circuits which *must* be included in the final germ set.
        If set to the special string "singletons" then all length-1 strings will
        be included.  Seting to None is the same as an empty list.
    force_score : float, optional (default is 1e100)
        When `force` designates a non-empty set of operation sequences, the score to
        assign any germ set that does not contain each and every required germ.
    threshold : float, optional (default is 1e6)
        Specifies a maximum score for the score matrix, above which the germ
        set is rejected as amplificationally incomplete.
    verbosity : int, optional
        Integer >= 0 indicating the amount of detail to print.

    Returns
    -------
    finalGermList : list
        Sublist of `germ_list` specifying the final, optimal set of germs.
    weights : array
        Integer array, of length ``len(germ_list)``, containing 0s and 1s to
        indicate which elements of `germ_list` were chosen as `finalGermList`.
        Only returned when `return_all` is ``True``.
    scoreDictionary : dict
        Dictionary with keys which are tuples of 0s and 1s of length
        ``len(germ_list)``, specifying a subset of germs, and values ==
        1.0/smallest-non-gauge-eigenvalue "scores".
    See Also
    --------
    :class:`~pygsti.objects.Model`
    :class:`~pygsti.objects.Circuit`
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    model_list = setup_model_list(model_list, randomize,
                                  randomization_strength, num_copies, seed)

    if (fixed_slack and slack_frac) or (not fixed_slack and not slack_frac):
        raise ValueError("Either fixed_slack *or* slack_frac should be specified")

    if initial_weights is not None:
        if len(germs_list) != len(initial_weights):
            raise ValueError("The lengths of germs_list (%d) and "
                             "initial_weights (%d) must match."
                             % (len(germs_list), len(initial_weights)))
        # Normalize the weights array to be 0s and 1s even if it is provided as
        # bools
        weights = _np.array([1 if x else 0 for x in initial_weights])
    else:
        weights = _np.ones(len(germs_list), _np.int64)  # default: start with all germs
#        lessWeightOnly = True # we're starting at the max-weight vector

    undercompleteModelNum = check_germs_list_completeness(model_list,
                                                          germs_list, score_func,
                                                          threshold)
    if undercompleteModelNum > -1:
        printer.log("Complete initial germ set FAILS on model "
                    + str(undercompleteModelNum) + ".", 1)
        printer.log("Aborting search.", 1)
        return (None, None, None) if return_all else None

    printer.log("Complete initial germ set succeeds on all input models.", 1)
    printer.log("Now searching for best germ set.", 1)

    num_models = len(model_list)

    # Remove any SPAM vectors from model since we only want
    # to consider the set of *gate* parameters for amplification
    # and this makes sure our parameter counting is correct
    model0 = remove_spam_vectors(model_list[0])

    # Initially allow adding to weight. -- maybe make this an argument??
    lessWeightOnly = False

    nGaugeParams = model0.num_gauge_params()

    # score dictionary:
    #   keys = (modelNum, tuple-ized weight vector of 1's and 0's only)
    #   values = list_score
    scoreD = {}
    germLengths = _np.array([len(germ) for germ in germs_list], _np.int64)

    if force:
        if force == "singletons":
            forceIndices = _np.where(germLengths == 1)
        else:  # force should be a list of Circuits
            forceIndices = _np.array([germs_list.index(opstr) for opstr in force])
    else:
        forceIndices = None

    twirledDerivDaggerDerivList = [calc_bulk_twirled_ddd(model, germs_list, tol)
                                   for model in model_list]

    # Dict of keyword arguments passed to compute_score that don't change from
    # call to call
    cs_kwargs = {
        'score_func': score_func,
        'deriv_dagger_deriv_list': twirledDerivDaggerDerivList,
        'force_indices': forceIndices,
        'force_score': force_score,
        'n_gauge_params': nGaugeParams,
        'op_penalty': op_penalty,
        'germ_lengths': germLengths,
        'l1_penalty': l1_penalty,
        'score_dict': scoreD,
    }

    scoreList = [compute_score(weights, model_num, **cs_kwargs)
                 for model_num in range(num_models)]
    score = _np.max(scoreList)
    L1 = sum(weights)  # ~ L1 norm of weights

    printer.log("Starting germ set optimization. Lower score is better.", 1)
    printer.log("Model has %d gauge params." % nGaugeParams, 1)

    def _get_neighbors(bool_vec):
        for i in range(len(bool_vec)):
            v = bool_vec.copy()
            v[i] = (v[i] + 1) % 2  # Toggle v[i] btwn 0 and 1
            yield v

    with printer.progress_logging(1):
        for iIter in range(max_iter):
            printer.show_progress(iIter, max_iter,
                                  suffix="score=%g, nGerms=%d" % (score, L1))

            bFoundBetterNeighbor = False
            for neighbor in _get_neighbors(weights):
                neighborScoreList = []
                for model_num in range(len(model_list)):
                    if (model_num, tuple(neighbor)) not in scoreD:
                        neighborL1 = sum(neighbor)
                        neighborScoreList.append(compute_score(neighbor,
                                                               model_num,
                                                               **cs_kwargs))
                    else:
                        neighborL1 = sum(neighbor)
                        neighborScoreList.append(scoreD[model_num,
                                                        tuple(neighbor)])

                neighborScore = _np.max(neighborScoreList)  # Take worst case.
                # Move if we've found better position; if we've relaxed, we
                # only move when L1 is improved.
                if neighborScore <= score and (neighborL1 < L1 or not lessWeightOnly):
                    weights, score, L1 = neighbor, neighborScore, neighborL1
                    bFoundBetterNeighbor = True

                    printer.log("Found better neighbor: "
                                "nGerms = %d score = %g" % (L1, score), 2)

            if not bFoundBetterNeighbor:  # Time to relax our search.
                # From now on, don't allow increasing weight L1
                lessWeightOnly = True

                if fixed_slack is False:
                    # Note score is positive (for sum of 1/lambda)
                    slack = score * slack_frac
                    # print "slack =", slack
                else:
                    slack = fixed_slack
                assert slack > 0

                printer.log("No better neighbor. Relaxing score w/slack: "
                            + "%g => %g" % (score, score + slack), 2)
                # Artificially increase score and see if any neighbor is better
                # now...
                score += slack

                for neighbor in _get_neighbors(weights):
                    scoreList = [scoreD[model_num, tuple(neighbor)]
                                 for model_num in range(len(model_list))]
                    maxScore = _np.max(scoreList)
                    if sum(neighbor) < L1 and maxScore < score:
                        weights, score, L1 = neighbor, maxScore, sum(neighbor)
                        bFoundBetterNeighbor = True
                        printer.log("Found better neighbor: "
                                    "nGerms = %d score = %g" % (L1, score), 2)

                if not bFoundBetterNeighbor:  # Relaxing didn't help!
                    printer.log("Stationary point found!", 1)
                    break  # end main for loop

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
            goodGerms.append(germs_list[index])

    if return_all:
        return goodGerms, weights, scoreD
    else:
        return goodGerms


def germ_breadth_score_fn(germ_set, germs_list, twirled_deriv_dagger_deriv_list,
                          non_ac_kwargs, init_n=1):
    """Score a germ set against a collection of models.
    Calculate the score of the germ set with respect to each member of a
    collection of models and return the worst score among that collection.
    Parameters
    ----------
    germ_set : list of Circuit
        The set of germs to score.
    germs_list : list of Circuit
        The list of all germs whose Jacobians are provided in
        `twirled_deriv_dagger_deriv_list`.
    twirled_deriv_dagger_deriv_list : numpy.array
        Jacobians for all the germs in `germs_list` stored as a 3-dimensional
        array, where the first index indexes the particular germ.
    non_ac_kwargs : dict
        Dictionary containing further arguments to pass to
        :func:`compute_composite_germ_score` for the scoring of the germ set against
        individual models.
    init_n : int
        The number of eigenvalues to begin checking for amplificational
        completeness with respect to. Passed as an argument to
        :func:`compute_composite_germ_score`.
    Returns
    -------
    CompositeScore
        The worst score over all models of the germ set.
    """
    weights = _np.zeros(len(germs_list))
    for germ in germ_set:
        weights[germs_list.index(germ)] = 1
    germsVsModelScores = []
    for derivDaggerDeriv in twirled_deriv_dagger_deriv_list:
        # Loop over all models
        partialDDD = derivDaggerDeriv[_np.where(weights == 1)[0], :, :]
        germsVsModelScores.append(compute_composite_germ_score(
            partial_deriv_dagger_deriv=partialDDD, init_n=init_n, **non_ac_kwargs))
    # Take the score for the current germ set to be its worst score over all
    # models.
    return max(germsVsModelScores)


def grasp_germ_set_optimization(model_list, germs_list, alpha, randomize=True,
                                randomization_strength=1e-3, num_copies=None,
                                seed=None, l1_penalty=1e-2, op_penalty=0.0,
                                score_func='all', tol=1e-6, threshold=1e6,
                                check=False, force="singletons",
                                iterations=5, return_all=False, shuffle=False,
                                verbosity=0):
    """
    Use GRASP to find a high-performing germ set.

    Parameters
    ----------
    model_list : Model or list of Model
        The list of Models to be tested.  To ensure that the returned germ
        set is amplficationally complete, it is a good idea to score potential
        germ sets against a collection (~5-10) of similar models.  The user
        may specify a single Model and a number of unitarily close copies to
        be made (set by the kwarg `num_copies`, or the user may specify their
        own list of Models, each of which in turn may or may not be
        randomized (set by the kwarg `randomize`).
    germs_list : list of Circuit
        List of all germs operation sequences to consider.
    alpha : float
        A number between 0 and 1 that roughly specifies a score theshold
        relative to the spread of scores that a germ must score better than in
        order to be included in the RCL. A value of 0 for `alpha` corresponds
        to a purely greedy algorithm (only the best-scoring germ set is
        included in the RCL), while a value of 1 for `alpha` will include all
        germs in the RCL.
        See :func:`pygsti.algorithms.scoring.composite_rcl_fn` for more details.
    randomize : Bool, optional
        Whether or not the input Model(s) are first subject to unitary
        randomization.  If ``False``, the user should perform the unitary
        randomization themselves.  Note:  If the Model(s) are perfect (e.g.
        ``std1Q_XYI.target_model()``), then the germ selection output should not be
        trusted, due to accidental degeneracies in the Model.  If the
        Model(s) include stochastic (non-unitary) error, then germ selection
        will fail, as we score amplificational completeness in the limit of
        infinite sequence length (so any stochastic noise will completely
        depolarize any sequence in that limit).
    randomization_strength : float, optional
        The strength of the unitary noise used to randomize input Model(s);
        is passed to :func:`~pygsti.objects.Model.randomize_with_unitary`.
        Default is ``1e-3``.
    num_copies : int, optional
        The number of Model copies to be made of the input Model (prior to
        unitary randomization).  If more than one Model is passed in,
        `num_copies` should be ``None``.  If only one Model is passed in and
        `num_copies` is ``None``, no extra copies are made.
    seed : float, optional
        The starting seed used for unitary randomization.  If multiple Models
        are to be randomized, ``model_list[i]`` is randomized with ``seed +
        i``.
    l1_penalty : float, optional
        How strong the penalty should be for increasing the germ set list by a
        single germ. Used for choosing between outputs of various GRASP
        iterations.
    op_penalty : float, optional
        How strong the penalty should be for increasing a germ in the germ set
        list by a single gate.
    score_func : string
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
        A list of Circuits which *must* be included in the final germ set.
        If set to the special string "singletons" then all length-1 strings will
        be included.  Seting to None is the same as an empty list.
    iterations : int, optional
        The number of GRASP iterations to perform.
    return_all : bool, optional
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
    finalGermList : list of Circuit
        Sublist of `germs_list` specifying the final, optimal set of germs.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    model_list = setup_model_list(model_list, randomize,
                                  randomization_strength, num_copies, seed)

    (_, numGaugeParams,
     numNonGaugeParams, _) = get_model_params(model_list)

    germLengths = _np.array([len(germ) for germ in germs_list], _np.int64)

    numGerms = len(germs_list)

    initialWeights = _np.zeros(numGerms, dtype=_np.int64)
    if force:
        if force == "singletons":
            initialWeights[_np.where(germLengths == 1)] = 1
        else:  # force should be a list of Circuits
            for opstr in force:
                initialWeights[germs_list.index(opstr)] = 1

    def get_neighbors_fn(weights): return _grasp.get_swap_neighbors(
        weights, forced_weights=initialWeights, shuffle=shuffle)

    undercompleteModelNum = check_germs_list_completeness(model_list,
                                                          germs_list,
                                                          score_func,
                                                          threshold)
    if undercompleteModelNum > -1:
        printer.warning("Complete initial germ set FAILS on model "
                        + str(undercompleteModelNum) + ".")
        printer.warning("Aborting search.")
        return (None, None, None) if return_all else None

    printer.log("Complete initial germ set succeeds on all input models.", 1)
    printer.log("Now searching for best germ set.", 1)

    printer.log("Starting germ set optimization. Lower score is better.", 1)

    twirledDerivDaggerDerivList = [calc_bulk_twirled_ddd(model, germs_list, tol,
                                                         check, germLengths)
                                   for model in model_list]

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'score_fn': lambda x: _scoring.list_score(x, score_func=score_func),
        'threshold_ac': threshold,
        'num_gauge_params': numGaugeParams,
        'op_penalty': op_penalty,
        'germ_lengths': germLengths,
    }

    final_nonAC_kwargs = nonAC_kwargs.copy()
    final_nonAC_kwargs['l1_penalty'] = l1_penalty

    scoreFn = (lambda germSet:
               germ_breadth_score_fn(germSet, germs_list,
                                     twirledDerivDaggerDerivList, nonAC_kwargs,
                                     init_n=1))
    finalScoreFn = (lambda germSet:
                    germ_breadth_score_fn(germSet, germs_list,
                                          twirledDerivDaggerDerivList,
                                          final_nonAC_kwargs, init_n=1))

    #OLD: feasibleThreshold = _scoring.CompositeScore(-numNonGaugeParams,threshold,numNonGaugeParams))
    def _feasible_fn(germ_set):  # now that scoring is not ordered entirely by N
        s = germ_breadth_score_fn(germ_set, germs_list,
                                  twirledDerivDaggerDerivList, nonAC_kwargs,
                                  init_n=1)
        return (s.N >= numNonGaugeParams and s.minor < threshold)

    def rcl_fn(x): return _scoring.composite_rcl_fn(x, alpha)

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
                    elements=germs_list, greedy_score_fn=scoreFn, rcl_fn=rcl_fn,
                    local_score_fn=scoreFn,
                    get_neighbors_fn=get_neighbors_fn,
                    feasible_fn=_feasible_fn,
                    initial_elements=initialWeights, seed=seed,
                    verbosity=verbosity)

                initialSolns.append(iterSolns[0])
                localSolns.append(iterSolns[1])

                success = True
                printer.log('Finished iteration {} of {}.'.format(
                    iteration + 1, iterations), 1)
            except Exception as e:
                failCount += 1
                raise e if (failCount == 10) else printer.warning(e)

    finalScores = _np.array([finalScoreFn(localSoln)
                             for localSoln in localSolns])
    bestSoln = localSolns[_np.argmin(finalScores)]

    return (bestSoln, initialSolns, localSolns) if return_all else bestSoln

"""
Functions for selecting a complete set of germs for a GST analysis.
"""
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
import random as _random
import scipy.linalg as _sla
import itertools
from math import floor

from pygsti.algorithms import grasp as _grasp
from pygsti.algorithms import scoring as _scoring
from pygsti import circuits as _circuits
from pygsti import baseobjs as _baseobjs
from pygsti.tools import mpitools as _mpit
from pygsti.baseobjs.statespace import ExplicitStateSpace as _ExplicitStateSpace
from pygsti.baseobjs.statespace import QuditSpace as _QuditSpace
from pygsti.models import ExplicitOpModel as _ExplicitOpModel

FLOATSIZE = 8  # in bytes: TODO: a better way

def find_germs(target_model, randomize=True, randomization_strength=1e-2,
               num_gs_copies=5, seed=None, candidate_germ_counts=None,
               candidate_seed=None, force="singletons", algorithm='greedy',
               algorithm_kwargs=None, mem_limit=None, comm=None,
               profiler=None, verbosity=1, num_nongauge_params=None,
               assume_real=False, float_type=_np.cdouble,
               mode="all-Jac", toss_random_frac=None,
               force_rank_increase=False, save_cevd_cache_filename= None,
               load_cevd_cache_filename=None, file_compression=False):
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
        passed along to stochastic germ-selection algorithms and to the 
        rng for dropping random fraction of germs.

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
        'greedy' : Add germs one-at-a-time until the set is AC, picking the germ that
        improves the germ-set score by the largest amount at each step. See
        :func:`find_germs_breadthfirst` for more details.
        
        'grasp': Use GRASP to generate random greedy germ sets and then locally
        optimize them. See :func:`find_germs_grasp` for more 
        details.
        
        'slack': From a initial set of germs, add or remove a germ at each step in 
        an attempt to improve the germ-set score. Will allow moves that 
        degrade the score in an attempt to escape local optima as long as 
        the degredation is within some specified amount of "slack". See 
        :func:`find_germs_integer_slack` for more details.

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

    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.
        
    float_type : numpy dtype object, optional
        Numpy data type to use for floating point arrays.
    
    toss_random_frac : float, optional
        If specified this is a number between 0 and 1 that indicates the random fraction of candidate
        germs to drop randomly following the deduping procedure.
        
    mode : {'allJac', 'singleJac', 'compactEVD'}, optional (default 'allJac')
        A flag to indicate the caching scheme used for storing the Jacobians for the candidate
        germs. Default value of 'allJac' caches all of the Jacobians and requires the most memory.
        'singleJac' doesn't cache anything and instead generates these Jacobians on the fly. The
        final option, 'compactEVD', is currently only configured to work with the greedy search 
        algorithm. When selected the compact eigenvalue decomposition/compact SVD of each of
        the Jacobians is constructed and is cached. This uses an intermediate amount of memory 
        between singleJac and allJac. When compactEVD mode is selected perform the greedy
        search iterations using an alternative method based on low-rank updates to the 
        psuedoinverse. This alternative approach means that this mode also only works with the
        score function option set to 'all'.
    
    force_rank_increase : bool, optional (default False) 
        Optional flag that can be used in conjunction with the greedy search algorithm
        in compactEVD mode. When set we require that each subsequant addition to the germ
        set must increase the rank of the experiment design's composite Jacobian. Can potentially
        speed up the search when set to True.
        
    save_cevd_cache_filename : str, optional (default None)
        When set and using the greedy search algorithm in 'compactEVD' mode this writes
        the compact EVD cache to disk using the specified filename.
               
    load_cevd_cache_filename : str, optional (default None)
        A filename/path to load an existing compact EVD cache from. Useful for warmstarting
        a germ set search with various cost function parameters, or for restarting a search
        that failed/crashed/ran out of memory. Note that there are no safety checks to confirm 
        that the compact EVD cache indeed corresponds to that for of currently specified
        candidate circuit list, so care must be take to confirm that the candidate
        germ lists are consistent across runs.
        
    file_compression : bool, optional (default False)
        When True and a filename is given for the save_cevd_cache_filename the corresponding
        numpy arrays are stored in a compressed format using numpy's savez_compressed.
        Can significantly decrease the storage requirements on disk at the expense of
        some additional computational cost writing and loading the files.

    Returns
    -------
    list of Circuit
        A list containing the germs making up the germ set.
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)
    modelList = _setup_model_list(target_model, randomize,
                                  randomization_strength, num_gs_copies, seed)
    gates = list(target_model.operations.keys())
    availableGermsList = []
    if candidate_germ_counts is None: candidate_germ_counts = {6: 'all upto'}
    for germLength, count in candidate_germ_counts.items():
        if count == "all upto":
            availableGermsList.extend(_circuits.list_all_circuits_without_powers_and_cycles(
                gates, max_length=germLength))
        else:
            if (candidate_seed is None) and (seed is not None):
                candidate_seed=seed
            availableGermsList.extend(_circuits.list_random_circuits_onelen(
                gates, germLength, count, seed=candidate_seed))

    printer.log('Initial Length Available Germ List: '+ str(len(availableGermsList)), 1)

    #Let's try deduping the available germ list too:
    #build a ckt cache
    ckt_cache= create_circuit_cache(target_model, availableGermsList)
    #Then dedupe this cache:
    #The second value returned is an updated ckt cache which we don't need right now
    availableGermsList, _ = clean_germ_list(target_model, ckt_cache, eq_thresh= 1e-6)
    
    printer.log('Length Available Germ List After Deduping: '+ str(len(availableGermsList)), 1)
    
    #If specified, drop a random fraction of the remaining candidate germs. 
    if toss_random_frac is not None:
        availableGermsList = drop_random_germs(availableGermsList, toss_random_frac, target_model, keep_bare=True, seed=seed)
    
    printer.log('Length Available Germ List After Dropping Random Fraction: '+ str(len(availableGermsList)), 1)
    
    #If we have specified a user specified germs to force inclusion of then there is a chance
    #they got removed by the deduping and removal of random circuits above. The right way to fix this
    #would be to add some logic to those subroutines that prevent this, but for now I am going to just 
    #manually add them back in (this will result almost suredly in a couple duplicate circuits, but oh well).
    if force is not None:
        #iterate through the list of forced germs to check for inclusion and
        #if missing append to the list of available germs.
        if isinstance(force, list):
            for forced_germ in force:
                if not forced_germ in availableGermsList:
                    availableGermsList.append(forced_germ)
        printer.log('Length Available Germ List After Adding Back In Forced Germs: '+ str(len(availableGermsList)), 1)
    
    #Add some checks related to the new option to switch up data types:
    if not assume_real:
        if not (float_type is _np.cdouble or float_type is _np.csingle):
            printer.log('Selected numpy type: '+ str(float_type.dtype), 1)
            raise ValueError('Unless working with (known) real-valued quantities only, please select an appropriate complex numpy dtype (either cdouble or csingle).')
    else:
        if not (float_type is _np.double or float_type is _np.single):
            printer.log('Selected numpy type: '+ str(float_type.dtype), 1)
            raise ValueError('When assuming real-valued quantities, please select a real-values numpy dtype (either double or single).')
        
    #How many bytes per float?
    FLOATSIZE= float_type(0).itemsize
    
    dim = target_model.dim
    #Np = model_list[0].num_params #wrong:? includes spam...
    Np = target_model.num_params
    if randomize==False:
        num_gs_copies=1
    memEstimatealljac = FLOATSIZE * num_gs_copies * len(availableGermsList) * Np**2
    # for _compute_bulk_twirled_ddd
    memEstimatealljac += FLOATSIZE * num_gs_copies * len(availableGermsList) * dim**2 * Np
    # for _bulk_twirled_deriv sub-call
    printer.log("Memory estimate of %.1f GB for all-Jac mode." %
                (memEstimatealljac / 1024.0**3), 1)
                
    memEstimatesinglejac = FLOATSIZE * 3 * len(modelList) * Np**2 + \
            FLOATSIZE * 3 * len(modelList) * dim**2 * Np
    #Factor of 3 accounts for currentDDDs, testDDDs, and bestDDDs
    printer.log("Memory estimate of %.1f GB for single-Jac mode." %
                    (memEstimatesinglejac / 1024.0**3), 1)
                    
    if mem_limit is not None:
        if memEstimatealljac > mem_limit:
            printer.log("Not enough memory for all-Jac mode, mem_limit is %.1f GB." %
                    (mem_limit / 1024.0**3), 1)
            if memEstimatesinglejac > mem_limit:
                raise MemoryError("Too little memory, even for single-Jac mode!")

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
            'op_penalty': 0.0,
            'score_func': 'all',
            'comm': comm,
            'mem_limit': mem_limit,
            'profiler': profiler,
            'num_nongauge_params': num_nongauge_params,
            'float_type': float_type,
            'mode' : mode,
            'force_rank_increase': force_rank_increase,
            'save_cevd_cache_filename': save_cevd_cache_filename,
            'load_cevd_cache_filename': load_cevd_cache_filename,
            'file_compression': file_compression,
            'evd_tol': 1e-10,
            'initial_germ_set_test': True
        }
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
        germList = find_germs_breadthfirst_greedy(model_list=modelList,
                                           **algorithm_kwargs)
        if germList is not None:
            #TODO: We should already know the value of this from
            #the final output of our optimization loop, so we ought
            #to be able to avoid this function call and related overhead.
            germsetScore = compute_germ_set_score(
                germList, neighborhood=modelList,
                score_func=algorithm_kwargs['score_func'],
                op_penalty=algorithm_kwargs['op_penalty'],
                num_nongauge_params=num_nongauge_params,
                float_type=float_type)
            printer.log('Constructed germ set:', 1)
            printer.log(str([germ.str for germ in germList]), 1)
            printer.log(germsetScore, 1)
    elif algorithm == 'grasp':
        printer.log('Using GRASP algorithm.', 1)
        # Define defaults for parameters that currently have no default or
        # whose default we want to change.
        default_kwargs = {
            'alpha': 0.1,   # No real reason for setting this value of alpha.
            'germs_list': availableGermsList,
            'randomize': False,
            'seed': seed,
            'l1_penalty': 0.0,
            'op_penalty': 0.0,
            'verbosity': max(0, verbosity - 1),
            'force': force,
            'return_all': False,
            'score_func': 'all',
            'num_nongauge_params': num_nongauge_params,
            'float_type': float_type
        }
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
        germList = find_germs_grasp(model_list=modelList,
                                    **algorithm_kwargs)
        printer.log('Constructed germ set:', 1)

        if algorithm_kwargs['return_all'] and germList[0] is not None:
            germsetScore = compute_germ_set_score(
                germList[0], neighborhood=modelList,
                score_func=algorithm_kwargs['score_func'],
                op_penalty=algorithm_kwargs['op_penalty'],
                l1_penalty=algorithm_kwargs['l1_penalty'],
                num_nongauge_params=num_nongauge_params,
                float_type=float_type)
            printer.log(str([germ.str for germ in germList[0]]), 1)
            printer.log(germsetScore)
        elif not algorithm_kwargs['return_all'] and germList is not None:
            germsetScore = compute_germ_set_score(
                germList, neighborhood=modelList,
                score_func=algorithm_kwargs['score_func'],
                op_penalty=algorithm_kwargs['op_penalty'],
                l1_penalty=algorithm_kwargs['l1_penalty'],
                num_nongauge_params=num_nongauge_params,
                float_type=float_type)
            printer.log(str([germ.str for germ in germList]), 1)
            printer.log(germsetScore, 1)
    elif algorithm == 'slack':
        printer.log('Using slack algorithm.', 1)
        # Define defaults for parameters that currently have no default or
        # whose default we want to change.
        default_kwargs = {
            'germs_list': availableGermsList,
            'randomize': False,
            'seed': seed,
            'verbosity': max(0, verbosity - 1),
            'l1_penalty': 0.0,
            'op_penalty': 0.0,
            'force': force,
            'score_func': 'all',
            'float_type': float_type
        }
        if ('slack_frac' not in algorithm_kwargs
                and 'fixed_slack' not in algorithm_kwargs):
            algorithm_kwargs['slack_frac'] = 0.1
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
        germList = find_germs_integer_slack(modelList,
                                            **algorithm_kwargs)
        if germList is not None:
            germsetScore = compute_germ_set_score(
                germList, neighborhood=modelList,
                score_func=algorithm_kwargs['score_func'],
                op_penalty=algorithm_kwargs['op_penalty'],
                l1_penalty=algorithm_kwargs['l1_penalty'],
                num_nongauge_params=num_nongauge_params,
                float_type=float_type)
            printer.log('Constructed germ set:', 1)
            printer.log(str([germ.str for germ in germList]), 1)
            printer.log(germsetScore, 1)
    else:
        raise ValueError("'{}' is not a valid algorithm "
                         "identifier.".format(algorithm))

    return germList


def compute_germ_set_score(germs, target_model=None, neighborhood=None,
                           neighborhood_size=5,
                           randomization_strength=1e-2, score_func='all',
                           op_penalty=0.0, l1_penalty=0.0, num_nongauge_params=None,
                           float_type=_np.cdouble):
    """
    Calculate the score of a germ set with respect to a model.

    More precisely, this function computes the maximum score (roughly equal
    to the number of amplified parameters) for a cloud of models.
    If `target_model` is given, it serves as the center of the cloud,
    otherwise the cloud must be supplied directly via `neighborhood`.


    Parameters
    ----------
    germs : list
        The germ set

    target_model : Model, optional
        The target model, used to generate a neighborhood of randomized models.

    neighborhood : list of Models, optional
        The "cloud" of models for which scores are computed.  If not None, this
        overrides `target_model`, `neighborhood_size`, and `randomization_strength`.

    neighborhood_size : int, optional
        Number of randomized models to construct around `target_model`.

    randomization_strength : float, optional
        Strength of unitary randomizations, as passed to :meth:`target_model.randomize_with_unitary`.

    score_func : {'all', 'worst'}
        Sets the objective function for scoring the eigenvalues. If 'all',
        score is ``sum(1/input_array)``. If 'worst', score is ``1/min(input_array)``.

    op_penalty : float, optional
        Coefficient for a penalty linear in the sum of the germ lengths.

    l1_penalty : float, optional
        Coefficient for a penalty linear in the number of germs.

    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.
        
    float_type : numpy dtype object, optional
        Numpy data type to use for floating point arrays.

    Returns
    -------
    CompositeScore
        The maximum score for `germs`, indicating how many parameters it amplifies.
    """
    def score_fn(x): return _scoring.list_score(x, score_func=score_func)
    if neighborhood is None:
        neighborhood = [target_model.randomize_with_unitary(randomization_strength)
                        for n in range(neighborhood_size)]
    scores = [compute_composite_germ_set_score(score_fn, model=model,
                                               partial_germs_list=germs,
                                               op_penalty=op_penalty,
                                               l1_penalty=l1_penalty,
                                               num_nongauge_params=num_nongauge_params,
                                               float_type=float_type)
              for model in neighborhood]

    return max(scores)


def _get_model_params(model_list, printer=None):
    """
    Get the number of gates and gauge parameters of the models in a list.

    Also verifies all models have the same number of gates and gauge parameters.

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
    
    if printer is not None:
        printer.log('Calculating number of gauge and non-gauge parameters', 1)
    
    # We don't care about SPAM, since it can't be amplified.
    reducedModelList = [_remove_spam_vectors(model)
                        for model in model_list]

    # All the models should have the same number of parameters and gates, but
    # let's be paranoid here for the time being and make sure.
    numGaugeParamsList = [reducedModel.num_gauge_params
                          for reducedModel in reducedModelList]
    numGaugeParams = numGaugeParamsList[0]
    if not all([numGaugeParams == otherNumGaugeParams
                for otherNumGaugeParams in numGaugeParamsList[1:]]):
        raise ValueError("All models must have the same number of gauge "
                         "parameters!")

    numNonGaugeParamsList = [reducedModel.num_nongauge_params
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


def _setup_model_list(model_list, randomize, randomization_strength,
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


def compute_composite_germ_set_score(score_fn, threshold_ac=1e6, init_n=1,
                                     partial_deriv_dagger_deriv=None, model=None,
                                     partial_germs_list=None, eps=None, germ_lengths=None,
                                     op_penalty=0.0, l1_penalty=0.0, num_nongauge_params=None,
                                     float_type=_np.cdouble):
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
        eigenvalues are equal (see :func:`~pygsti.algorithms.germselection._bulk_twirled_deriv` for details). Not
        used if `partial_deriv_dagger_deriv` is provided.

    op_penalty : float, optional
        Coefficient for a penalty linear in the sum of the germ lengths.

    germ_lengths : numpy.array, optional
        The length of each germ. Not needed if `op_penalty` is ``0.0`` or
        `partial_germs_list` is provided.

    l1_penalty : float, optional
        Coefficient for a penalty linear in the number of germs.

    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.

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
            Np= model.num_params
            combinedDDD=_np.zeros((Np, Np), dtype=float_type)
            pDDD_kwargs = {'float_type':float_type}
            if eps is not None:
                pDDD_kwargs['eps'] = eps
            #use a more memory efficient calculation when generating
            #twirled ddd from scratch again.
            for germ in partial_germs_list:
                combinedDDD += _compute_twirled_ddd(model, germ, **pDDD_kwargs)

    if num_nongauge_params is None:
        if model is None:
            raise ValueError("Must provide either num_nongauge_params or model!")
        else:
            reduced_model = _remove_spam_vectors(model)
            num_nongauge_params = reduced_model.num_params - reduced_model.num_gauge_params

    # Calculate penalty scores
    if partial_deriv_dagger_deriv is None:
        numGerms= len(partial_germs_list)
    else:
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
    
    if partial_deriv_dagger_deriv is not None:
        combinedDDD = _np.sum(partial_deriv_dagger_deriv, axis=0)
    sortedEigenvals = _np.sort(_np.real(_nla.eigvalsh(combinedDDD)))
    observableEigenvals = sortedEigenvals[-num_nongauge_params:]
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


def _compute_bulk_twirled_ddd(model, germs_list, eps=1e-6, check=False,
                              germ_lengths=None, comm=None, float_type=_np.cdouble):
    """
    Calculate the positive squares of the germ Jacobians.

    twirledDerivDaggerDeriv == array J.H*J contributions from each germ
    (J=Jacobian) indexed by (iGerm, iModelParam1, iModelParam2)
    size (nGerms, vec_model_dim, vec_model_dim)

    Parameters
    ----------
    model : Model
        The model defining the parameters to differentiate with respect to.

    germs_list : list
        The germ set

    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. abs(eval1 - eval2) < eps ? )

    check : bool, optional
        Whether to perform internal consistency checks, at the expense of
        making the function slower.

    germ_lengths : numpy.ndarray, optional
        A pre-computed array of the length (depth) of each germ.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.
        
    float_type : numpy dtype object, optional
        Numpy data type to use in floating point arrays.

    Returns
    -------
    twirledDerivDaggerDeriv : numpy.ndarray
        A complex array of shape `(len(germs), model.num_params, model.num_params)`.
    """
    if germ_lengths is None:
        germ_lengths = _np.array([len(germ) for germ in germs_list])

    twirledDeriv = _bulk_twirled_deriv(model, germs_list, eps, check, comm, float_type=float_type) / germ_lengths[:, None, None]

    #OLD: slow, I think because conjugate *copies* a large tensor, causing a memory bottleneck
    #twirledDerivDaggerDeriv = _np.einsum('ijk,ijl->ikl',
    #                                     _np.conjugate(twirledDeriv),
    #                                     twirledDeriv)

    #NEW: faster, one-germ-at-a-time computation requires less memory.
    nGerms, _, vec_model_dim = twirledDeriv.shape
    twirledDerivDaggerDeriv = _np.empty((nGerms, vec_model_dim, vec_model_dim),
                                        dtype=float_type)
    for i in range(nGerms):
        twirledDerivDaggerDeriv[i, :, :] = _np.dot(
            twirledDeriv[i, :, :].conjugate().T, twirledDeriv[i, :, :])

    return twirledDerivDaggerDeriv


def _compute_twirled_ddd(model, germ, eps=1e-6, float_type=_np.cdouble):
    """
    Calculate the positive squares of the germ Jacobian.

    twirledDerivDaggerDeriv == array J.H*J contributions from `germ`
    (J=Jacobian) indexed by (iModelParam1, iModelParam2)
    size (vec_model_dim, vec_model_dim)

    Parameters
    ----------
    model : Model
        The model defining the parameters to differentiate with respect to.

    germ : Circuit
        The (single) germ circuit to consider.  `J` above is the twirled
        derivative of this circuit's action (process matrix).

    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. abs(eval1 - eval2) < eps ? )

    Returns
    -------
    numpy.ndarray
    """
    twirledDeriv = _twirled_deriv(model, germ, eps, float_type) / len(germ)
    #twirledDerivDaggerDeriv = _np.einsum('jk,jl->kl',
    #                                     _np.conjugate(twirledDeriv),
    #                                     twirledDeriv)
    twirledDerivDaggerDeriv = _np.tensordot(_np.conjugate(twirledDeriv),
                                            twirledDeriv, (0, 0))

    return twirledDerivDaggerDeriv


def _germ_set_score_slack(weights, model_num, score_func, deriv_dagger_deriv_list,
                          force_indices, force_score,
                          n_gauge_params, op_penalty, germ_lengths, l1_penalty=1e-2,
                          score_dict=None):
    """
    Returns a germ set "score" in which smaller is better.

    Also returns intentionally bad score (`force_score`) if `weights` is zero on any of
    the "forced" germs (i.e. at any index in `forcedIndices`).
    This function is included for use by :func:`find_germs_integer_slack`,
    but is not convenient for just computing the score of a germ set. For that,
    use :func:`compute_germ_set_score`.

    Parameters
    ----------
    weights : list
        The per-germ "selection weight", indicating whether the germ
        is present in the selected germ set or not.

    model_num : int
        index into `deriv_dagger_deriv_list` indicating which model (typically in
        a neighborhood) we're computing scores for.

    score_func : {'all', 'worst'}
        Sets the objective function for scoring the eigenvalues. If 'all',
        score is ``sum(1/input_array)``. If 'worst', score is ``1/min(input_array)``.

    deriv_dagger_deriv_list : numpy.ndarray
        Array of J.T * J contributions for each model.

    force_indices : list of ints
        Indices marking the germs that *must* be in the final set (or else `force_score`
        will be returned).

    force_score : float
        The score that is returned when any of the germs indexed by `force_indices` are
        not present (i.e. their weights are <= 0).

    n_gauge_params : int
        The number of gauge (not amplifiable) parameters in the model.

    op_penalty : float
        Coefficient for a penalty linear in the sum of the germ lengths.

    germ_lengths : numpy.ndarray
        A pre-computed array of the length (depth) of each germ.

    l1_penalty : float
        Coefficient for a penalty linear in the number of germs.

    score_dict : dict, optional
        A dictionary to cache the score valies for the given `model_num` and
        `weights`, i.e. `score_dict[model_num, tuple(weights)]` is set to the
        returned value.


    Returns
    -------
    float
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
        # Side effect: calling _germ_set_score_slack caches result in score_dict
        score_dict[model_num, tuple(weights)] = score
    return score


def randomize_model_list(model_list, randomization_strength, num_copies,
                         seed=None):
    """
    Applies random unitary perturbations to a model or list of models.

    If `model_list` is a length-1 list, then `num_copies` determines how
    many randomizations to create.  If `model_list` containes multiple
    models, then `num_copies` must be `None` and each model is
    randomized once to create the corresponding returned model.

    Parameters
    ----------
    model_list : Model or list
        A list of Model objects.

    randomization_strength : float, optional
        Strength of unitary randomizations, as passed to :meth:`Model.randomize_with_unitary`.

    num_copies : int
        The number of random perturbations of `model_list[0]` to generate when
        `len(model_list) == 1`.  A value of `None` will result in 1 copy.  If
        `len(model_list) > 1` then `num_copies` must be set to None.

    seed : int, optional
        Starting seed for randomization.  Successive randomizations receive
        successive seeds.  `None` results in random seeds.

    Returns
    -------
    list
        A list of the randomized Models.
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


def test_germs_list_completeness(model_list, germs_list, score_func, threshold, float_type=_np.cdouble, comm=None, num_gauge_params = None):
    """
    Check to see if the germs_list is amplificationally complete (AC).

    Checks for AC with respect to all the Models in `model_list`, returning
    the index of the first Model for which it is not AC or `-1` if it is AC
    for all Models.

    Parameters
    ----------
    model_list : list
        A list of models to test.  Often this list is a neighborhood ("cloud") of
        models around a model of interest.

    germs_list : list
        A list of the germ :class:`Circuit` objects (the "germ set") to test for completeness.

    score_func : {'all', 'worst'}
        Sets the objective function for scoring the eigenvalues. If 'all',
        score is ``sum(1/eigval_array)``. If 'worst', score is ``1/min(eigval_array)``.

    threshold : float, optional
        An eigenvalue of `jacobian^T*jacobian` is considered zero and thus a
        parameter un-amplified when its reciprocal is greater than threshold.
        Also used for eigenvector degeneracy testing in twirling operation.
        
    float_type : numpy dtype object, optional
        Numpy data type to use for floating point arrays.
    
    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.
        
    num_gauge_params : int, optional (default None)
        A optional kwarg for specifying the number of gauge
        parameters. Specifying this if already precomputed can
        save on computation.

    Returns
    -------
    int
        The index of the first model in `model_list` to fail the amplficational
        completeness test. Returns -1 if germ set is AC for all tested models.
    """
    for modelNum, model in enumerate(model_list):
        initial_test = test_germ_set_infl(model, germs_list,
                                          score_func=score_func,
                                          threshold=threshold, float_type=float_type,
                                          comm=comm)
        if not initial_test:
            return modelNum

    # If the germs_list is complete for all models, return -1
    return -1


def _remove_spam_vectors(model):
    """
    Returns a copy of `model` with state preparations and effects removed.

    Parameters
    ----------
    model : Model
        The model to act on.

    Returns
    -------
    Model
    """
    reducedModel = model.copy()
    try:
        for prepLabel in list(reducedModel.preps.keys()):
            del reducedModel.preps[prepLabel]
        for povmLabel in list(reducedModel.povms.keys()):
            del reducedModel.povms[povmLabel]
    except AttributeError:
        # Implicit model instead
        for prepLabel in list(reducedModel.prep_blks.keys()):
            del reducedModel.prep_blks[prepLabel]
        for povmLabel in list(reducedModel.povm_blks.keys()):
            del reducedModel.povm_blks[povmLabel]

    reducedModel._mark_for_rebuild()
    return reducedModel


def _num_non_spam_gauge_params(model):
    """
    Return the number of non-gauge, non-SPAM parameters in `model`.

    Equivalent to `_remove_spam_vectors(model).num_gauge_params`.

    Parameters
    ---------
    model : Model

    Parameters
    ----------
    model : Model
        The model to act on.

    Returns
    -------
    int
    """
    return _remove_spam_vectors(model).num_gauge_params


# wrt is op_dim x op_dim, so is M, Minv, Proj
# so SOP is op_dim^2 x op_dim^2 and acts on vectorized *gates*
# Recall vectorizing identity (when vec(.) concats rows as flatten does):
#     vec( A * X * B ) = A tensor B^T * vec( X )
def _super_op_for_perfect_twirl(wrt, eps, float_type=_np.cdouble):
    """Return super operator for doing a perfect twirl with respect to wrt.
    """
    assert wrt.shape[0] == wrt.shape[1]  # only square matrices allowed
    dim = wrt.shape[0]
    
    #The eigenvalues and eigenvectors of wrt can be complex valued, even for
    #real-valued transfer matrices. Need to be careful here to start off using a
    #complex data type. The actual projector onto the germs commutant appears to be strictly real valued though
    #(that makes sense because otherwise the projected derivative would become complex
    #So we should be able to cast it back to the specified float_type just before returning it.
    SuperOp = _np.zeros((dim**2, dim**2), dtype=_np.cdouble)

    # Get spectrum and eigenvectors of wrt
    wrtEvals, wrtEvecs = _np.linalg.eig(wrt)
    wrtEvecsInv = _np.linalg.inv(wrtEvecs)
    
    #calculate the dimensions of the eigenspaces:
    subspace_idx_list=[]
    subspace_eval_list=[]
    
    unseen_indices= list(range(len(wrtEvals)))
    while unseen_indices:
        current_idx= unseen_indices.pop()
        current_eval= wrtEvals[current_idx]
        
        existing_subspace_found=False
        
        for i,sublist in enumerate(subspace_eval_list):
            for evals in sublist:
                if abs(current_eval-evals)<=eps:
                    existing_subspace_found=True
                    subspace_eval_list[i].append(current_eval)
                    subspace_idx_list[i].append(current_idx)
                    break
                    
        if not existing_subspace_found:
            subspace_eval_list.append([current_eval])
            subspace_idx_list.append([current_idx])
    
    #Now use these to construct the projectors onto each of the subspaces
    for idx_list in subspace_idx_list:
        #Instead of actually constructing the projector, this is equivalent
        #to simply picking out the requisite rows from wrtEvecsInv and
        #columns from wrtEvecs.
        idx_array= _np.asarray(idx_list)
        A = _np.dot(wrtEvecs[:,idx_array], wrtEvecsInv[idx_array, :])
        SuperOp += fast_kron(A, A.T)
    
    #---------Old Implementation-------------------#

    # We want to project  X -> M * (Proj_i * (Minv * X * M) * Proj_i) * Minv,
    # where M = wrtEvecs. So A = B = M * Proj_i * Minv and so
    # superop = A tensor B^T == A tensor A^T
    # NOTE: this == (A^T tensor A)^T while *Maple* germ functions seem to just
    # use A^T tensor A -> ^T difference
#    for i in range(dim):
#        # Create projector onto i-th eigenspace (spanned by i-th eigenvector
#        # and other degenerate eigenvectors)
#        Proj_i = _np.diag([(1 if (abs(wrtEvals[i] - wrtEvals[j]) <= eps)
#                            else 0) for j in range(dim)])
#        A = _np.dot(wrtEvecs, _np.dot(Proj_i, wrtEvecsInv))
#        
#        #testing:
#        
#        #if _np.linalg.norm(A.imag) > 1e-6:
#        #    print("DB: imag = ",_np.linalg.norm(A.imag))
#        #assert(_np.linalg.norm(A.imag) < 1e-6)
#        #A = _np.real(A)
#        # Need to normalize, because we are overcounting projectors onto
#        # subspaces of dimension d > 1, giving us d * Proj_i tensor Proj_i^T.
#        # We can fix this with a division by tr(Proj_i) = d.
#        #SuperOp += _np.kron(A, A.T) / _np.trace(Proj_i)
#        SuperOp += fast_kron(A, A.T) / _np.trace(Proj_i)
#        # SuperOp += _np.kron(A.T,A) # Mimic Maple version (but I think this is
#        # wrong... or it doesn't matter?)
    
    #Cast the twirling SuperOp back to the specified float type.
    #If the float_type is a real-valued one though we should probably do a quick
    #sanity check to confirm everything we're casting is actually real!
    if (float_type is _np.double) or (float_type is _np.single):
        #might as well use eps as the threshold here too.
        if _np.any(_np.abs(_np.imag(SuperOp))>eps):
            print(f'eps {eps}')
            print(f'{_np.imag(SuperOp)[_np.abs(_np.imag(SuperOp))>eps]}')
            print(f'wrtEvals {wrtEvals}')
            print(f'wrtEvecs {wrtEvecs}')
            print(f'wrtEvecsInv {wrtEvecsInv}')
            
            #print(f'_np.imag(SuperOp)>eps: {_np.imag(SuperOp)}', flush = True)
            raise ValueError("Attempting to cast a twirling superoperator with non-trivial imaginary component to a real-valued data type.")
        #cast just the real part to specified float type.
        SuperOp=SuperOp.real.astype(float_type)
    else:
        SuperOp=SuperOp.astype(float_type)
    
    return SuperOp  # a op_dim^2 x op_dim^2 matrix


def _sq_sing_vals_from_deriv(deriv, weights=None):
    """
    Calculate the squared singular values of the Jacobian of the germ set.

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


def _twirled_deriv(model, circuit, eps=1e-6, float_type=_np.cdouble):
    """
    Compute the "Twirled Derivative" of a circuit.

    The twirled derivative is obtained by acting on the standard derivative of
    a circuit with the twirling superoperator.

    Parameters
    ----------
    model : Model object
        The Model which associates operation labels with operators.

    circuit : Circuit object
        A twirled derivative of this circuit's action (process matrix) is taken.

    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. `abs(eval1 - eval2) < eps` ? )
        
    float_type : numpy dtype object, optional
        Numpy data type to use for floating point arrays.
        
    Returns
    -------
    numpy array
        An array of shape (op_dim^2, num_model_params)
    """
    prod = model.sim.product(circuit)

    # flattened_op_dim x vec_model_dim
    dProd = model.sim.dproduct(circuit, flat=True)

    # flattened_op_dim x flattened_op_dim
    twirler = _super_op_for_perfect_twirl(prod, eps, float_type=float_type)

    # flattened_op_dim x vec_model_dim
    return _np.dot(twirler, dProd)


def _bulk_twirled_deriv(model, circuits, eps=1e-6, check=False, comm=None, float_type=_np.cdouble):
    """
    Compute the "Twirled Derivative" of a set of circuits.

    The twirled derivative is obtained by acting on the standard derivative of
    a circuit with the twirling superoperator.

    Parameters
    ----------
    model : Model object
        The Model which associates operation labels with operators.

    circuits : list of Circuit objects
        A twirled derivative of this circuit's action (process matrix) is taken.

    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. `abs(eval1 - eval2) < eps` ? )

    check : bool, optional
        Whether to perform internal consistency checks, at the expense of
        making the function slower.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.
        
    float_type : numpy dtype object, optional
        Numpy data type to use for floating point arrays.

    Returns
    -------
    numpy array
        An array of shape (num_simplified_circuits, op_dim^2, num_model_params)
    """
    if len(model.preps) > 0 or len(model.povms) > 0:
        model = _remove_spam_vectors(model)
        # This function assumes model has no spam elements so `lookup` below
        #  gives indexes into products computed by evalTree.

    resource_alloc = _baseobjs.ResourceAllocation(comm=comm)
    dProds, prods = model.sim.bulk_dproduct(circuits, flat=True, return_prods=True, resource_alloc=resource_alloc)
    op_dim = model.dim
    fd = op_dim**2  # flattened gate dimension
    nCircuits = len(circuits)

    ret = _np.empty((nCircuits, fd, dProds.shape[1]), dtype=float_type)
    
    for i in range(nCircuits):
        # flattened_op_dim x flattened_op_dim
        twirler = _super_op_for_perfect_twirl(prods[i], eps, float_type=float_type)

        # flattened_op_dim x vec_model_dim
        ret[i] = _np.dot(twirler, dProds[i * fd:(i + 1) * fd])

    if check:
        for i, circuit in enumerate(circuits):
            chk_ret = _twirled_deriv(model, circuit, eps, float_type=float_type)
            if _nla.norm(ret[i] - chk_ret) > 1e-6:
                _warnings.warn("bulk twirled derivative norm mismatch = "
                               "%g - %g = %g"
                               % (_nla.norm(ret[i]), _nla.norm(chk_ret),
                                  _nla.norm(ret[i] - chk_ret)))  # pragma: no cover

    return ret  # nSimplifiedCircuits x flattened_op_dim x vec_model_dim


def test_germ_set_finitel(model, germs_to_test, length, weights=None,
                          return_spectrum=False, tol=1e-6):
    """
    Test whether a set of germs is able to amplify all non-gauge parameters.

    Parameters
    ----------
    model : Model
        The Model (associates operation matrices with operation labels).

    germs_to_test : list of Circuits
        List of germ circuits to test for completeness.

    length : int
        The finite length to use in amplification testing.  Larger
        values take longer to compute but give more robust results.

    weights : numpy array, optional
        A 1-D array of weights with length equal len(germs_to_test),
        which multiply the contribution of each germ to the total
        jacobian matrix determining parameter amplification. If
        None, a uniform weighting of `1.0/len(germs_to_test)` is applied.

    return_spectrum : bool, optional
        If True, return the `jacobian^T*jacobian` spectrum in addition
        to the success flag.

    tol : float, optional
        Tolerance: an eigenvalue of `jacobian^T*jacobian` is considered
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
    model = _remove_spam_vectors(model)

    nGerms = len(germs_to_test)
    germToPowL = [germ * length for germ in germs_to_test]

    op_dim = model.dim
    dprods = model.sim.bulk_dproduct(germToPowL, flat=True)  # shape (nGerms*flattened_op_dim, vec_model_dim)
    dprods.shape = (nGerms, op_dim**2, dprods.shape[1])

    germLengths = _np.array([len(germ) for germ in germs_to_test], 'd')

    normalizedDeriv = dprods / (length * germLengths[:, None, None])

    sortedEigenvals = _sq_sing_vals_from_deriv(normalizedDeriv, weights)

    nGaugeParams = model.num_gauge_params

    observableEigenvals = sortedEigenvals[nGaugeParams:]

    bSuccess = bool(_scoring.list_score(observableEigenvals, 'worst') < 1 / tol)

    return (bSuccess, sortedEigenvals) if return_spectrum else bSuccess


def test_germ_set_infl(model, germs_to_test, score_func='all', weights=None,
                       return_spectrum=False, threshold=1e6, check=False,
                       float_type=_np.cdouble, comm=None, nGaugeParams = None):
    """
    Test whether a set of germs is able to amplify all non-gauge parameters.

    Parameters
    ----------
    model : Model
        The Model (associates operation matrices with operation labels).

    germs_to_test : list of Circuit
        List of germ circuits to test for completeness.

    score_func : string
        Label to indicate how a germ set is scored. See
        :func:`~pygsti.algorithms.scoring.list_score` for details.

    weights : numpy array, optional
        A 1-D array of weights with length equal len(germs_to_test),
        which multiply the contribution of each germ to the total
        jacobian matrix determining parameter amplification. If
        None, a uniform weighting of `1.0/len(germs_to_test)` is applied.

    return_spectrum : bool, optional
        If ``True``, return the `jacobian^T*jacobian` spectrum in addition
        to the success flag.

    threshold : float, optional
        An eigenvalue of `jacobian^T*jacobian` is considered zero and thus a
        parameter un-amplified when its reciprocal is greater than threshold.
        Also used for eigenvector degeneracy testing in twirling operation.

    check : bool, optional
        Whether to perform internal consistency checks, at the
        expense of making the function slower.
        
    float_type: numpy dtype object, optional
        Optional numpy data type to use for internal numpy array calculations.
        
    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.
        
    nGaugeParams : int, optional (default None)
        A optional kwarg for specifying the number of gauge
        parameters. Specifying this if already precomputed can
        save on computation.

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
    model = _remove_spam_vectors(model)
    Np= model.num_params
    
    germLengths = _np.array([len(germ) for germ in germs_to_test], _np.int64)
    #twirledDerivDaggerDeriv = _compute_bulk_twirled_ddd(model, germs_to_test,
    #                                                    1. / threshold, check,
    #                                                    germLengths, 
    #                                                    float_type=float_type)
    
    #Use a more memory efficient implementation that builds up the J^T@J matrix
    #borrowing the code used for single-Jac mode in the germ search routine below:
    twirledDerivDaggerDeriv = _np.zeros((Np, Np), dtype=float_type)
    
    if weights is None:
        nGerms = len(germs_to_test)
        # weights = _np.array( [1.0/nGerms]*nGerms, 'd')
        weights = _np.array([1.0] * nGerms, 'd')

    loc_Indices, _, _ = _mpit.distribute_indices(
        list(range(len(germs_to_test))), comm, False)

    for i, GermIdx in enumerate(loc_Indices):
        twirledDerivDaggerDeriv += weights[GermIdx]*_compute_twirled_ddd(
            model, germs_to_test[GermIdx], 1. / threshold, float_type=float_type)

    #aggregate each twirledDerivDaggerDeriv across all procs
    if comm is not None and comm.Get_size() > 1:
        from mpi4py import MPI  # not at top so pygsti doesn't require mpi4py
        result = _np.empty((Np, Np), dtype=float_type)
        comm.Allreduce(twirledDerivDaggerDeriv, result, op=MPI.SUM)
        twirledDerivDaggerDeriv[:, :] = result[:, :]
        result = None  # free mem
                                                       
    sortedEigenvals = _np.sort(_np.real(_np.linalg.eigvalsh(twirledDerivDaggerDeriv)))

    if nGaugeParams is None:
        nGaugeParams = model.num_gauge_params
    observableEigenvals = sortedEigenvals[nGaugeParams:]

    bSuccess = bool(_scoring.list_score(observableEigenvals, score_func)
                    < threshold)

    return (bSuccess, sortedEigenvals) if return_spectrum else bSuccess


def find_germs_depthfirst(model_list, germs_list, randomize=True,
                          randomization_strength=1e-3, num_copies=None, seed=0, op_penalty=0,
                          score_func='all', tol=1e-6, threshold=1e6, check=False,
                          force="singletons", verbosity=0, float_type=_np.cdouble):
    """
    Greedy germ selection algorithm starting with 0 germs.

    Tries to minimize the number of germs needed to achieve amplificational
    completeness (AC). Begins with 0 germs and adds the germ that increases the
    score used to check for AC by the largest amount at each step, stopping when
    the threshold for AC is achieved.

    Parameters
    ----------
    model_list : Model or list
        The model or list of Models to select germs for.

    germs_list : list of Circuit
        The list of germs to contruct a germ set from.

    randomize : bool, optional
        Whether or not to randomize `model_list` (usually just a single
        `Model`) with small (see `randomizationStrengh`) unitary maps
        in order to avoid "accidental" symmetries which could allow for
        fewer germs but *only* for that particular model.  Setting this
        to `True` will increase the run time by a factor equal to the
        numer of randomized copies (`num_copies`).

    randomization_strength : float, optional
        The strength of the unitary noise used to randomize input Model(s);
        is passed to :func:`~pygsti.objects.Model.randomize_with_unitary`.

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
        Tolerance (`eps` arg) for :func:`_compute_bulk_twirled_ddd`, which sets
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

    verbosity : int, optional
        Level of detail printed to stdout.

    Returns
    -------
    list
        A list of the built-up germ set (a list of :class:`Circuit` objects).
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    model_list = _setup_model_list(model_list, randomize,
                                   randomization_strength, num_copies, seed)

    (reducedModelList,
     numGaugeParams, numNonGaugeParams, _) = _get_model_params(model_list)

    germLengths = _np.array([len(germ) for germ in germs_list], _np.int64)
    numGerms = len(germs_list)

    weights = _np.zeros(numGerms, _np.int64)
    goodGerms = []
    if force:
        if force == "singletons":
            weights[_np.where(germLengths == 1)] = 1
            goodGerms = [germ for i, germ in enumerate(germs_list) if germLengths[i] == 1]
        else:  # force should be a list of Circuits
            for opstr in force:
                weights[germs_list.index(opstr)] = 1
            goodGerms = force[:]

    undercompleteModelNum = test_germs_list_completeness(model_list,
                                                         germs_list,
                                                         score_func,
                                                         threshold,
                                                         float_type=float_type)
    if undercompleteModelNum > -1:
        printer.warning("Complete initial germ set FAILS on model "
                        + str(undercompleteModelNum) + ". Aborting search.")
        return None

    printer.log("Complete initial germ set succeeds on all input models.", 1)
    printer.log("Now searching for best germ set.", 1)
    printer.log("Starting germ set optimization. Lower score is better.", 1)

    twirledDerivDaggerDerivList = [_compute_bulk_twirled_ddd(model, germs_list, tol,
                                                             check, germLengths, float_type=float_type)
                                   for model in model_list]

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'score_fn': lambda x: _scoring.list_score(x, score_func=score_func),
        'threshold_ac': threshold,
        'num_nongauge_params': numNonGaugeParams,
        'op_penalty': op_penalty,
        'germ_lengths': germLengths,
        'float_type': float_type
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
            if test_germ_set_infl(reducedModel, goodGerms,
                                  score_func=score_func, threshold=threshold, float_type=float_type):
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
                candidateGermScore = compute_composite_germ_set_score(
                    partial_deriv_dagger_deriv=partialDDD, **nonAC_kwargs)
                candidateGermScores.append(candidateGermScore)
            # Add the germ that give the best score
            bestCandidateGerm = candidateGerms[_np.array(
                candidateGermScores).argmin()]
            weights[bestCandidateGerm] = 1
            goodGerms.append(germs_list[bestCandidateGerm])

    return goodGerms

def find_germs_breadthfirst(model_list, germs_list, randomize=True,
                            randomization_strength=1e-3, num_copies=None, seed=0,
                            op_penalty=0, score_func='all', tol=1e-6, threshold=1e6,
                            check=False, force="singletons", pretest=True, mem_limit=None,
                            comm=None, profiler=None, verbosity=0, num_nongauge_params=None, 
                            float_type= _np.cdouble, mode="all-Jac"):
    """
    Greedy algorithm starting with 0 germs.

    Tries to minimize the number of germs needed to achieve amplificational
    completeness (AC). Begins with 0 germs and adds the germ that increases the
    score used to check for AC by the largest amount (for the model that
    currently has the lowest score) at each step, stopping when the threshold
    for AC is achieved. This strategy is something of a "breadth-first"
    approach, in contrast to :func:`find_germs_depthfirst`, which only looks at the
    scores for one model at a time until that model achieves AC, then
    turning it's attention to the remaining models.

    Parameters
    ----------
    model_list : Model or list
        The model or list of `Model` objects to select germs for.

    germs_list : list of Circuit
        The list of germs to contruct a germ set from.

    randomize : bool, optional
        Whether or not to randomize `model_list` (usually just a single
        `Model`) with small (see `randomizationStrengh`) unitary maps
        in order to avoid "accidental" symmetries which could allow for
        fewer germs but *only* for that particular model.  Setting this
        to `True` will increase the run time by a factor equal to the
        numer of randomized copies (`num_copies`).

    randomization_strength : float, optional
        The strength of the unitary noise used to randomize input Model(s);
        is passed to :func:`~pygsti.objects.Model.randomize_with_unitary`.

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
        Tolerance (`eps` arg) for :func:`_compute_bulk_twirled_ddd`, which sets
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

    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.
        
    float_type : numpy dtype object, optional
        Use an alternative data type for the values of the numpy arrays generated.

    Returns
    -------
    list
        A list of the built-up germ set (a list of :class:`Circuit` objects).
    """
    if comm is not None and comm.Get_size() > 1:
        from mpi4py import MPI  # not at top so pygsti doesn't require mpi4py

    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)

    model_list = _setup_model_list(model_list, randomize,
                                   randomization_strength, num_copies, seed)

    dim = model_list[0].dim
    Np = model_list[0].num_params
    assert(all([(mdl.dim == dim) for mdl in model_list])), \
        "All models must have the same dimension!"
    #assert(all([(mdl.num_params == Np) for mdl in model_list])), \
    #    "All models must have the same number of parameters!"

    (_, numGaugeParams,
     numNonGaugeParams, _) = _get_model_params(model_list)
    if num_nongauge_params is not None:
        numGaugeParams = numGaugeParams + numNonGaugeParams - num_nongauge_params
        numNonGaugeParams = num_nongauge_params

    germLengths = _np.array([len(germ) for germ in germs_list], _np.int64)

    numGerms = len(germs_list)

    goodGerms = []
    weights = _np.zeros(numGerms, _np.int64)
    if force:
        if force == "singletons":
            weights[_np.where(germLengths == 1)] = 1
            goodGerms = [germ for i, germ in enumerate(germs_list) if germLengths[i] == 1]
        else:  # force should be a list of Circuits
            for opstr in force:
                weights[germs_list.index(opstr)] = 1
            goodGerms = force[:]
            
    #We should do the memory estimates before the pretest:
    FLOATSIZE= float_type(0).itemsize

    memEstimatealljac = FLOATSIZE * len(model_list) * len(germs_list) * Np**2
    # for _compute_bulk_twirled_ddd
    memEstimatealljac += FLOATSIZE * len(model_list) * len(germs_list) * dim**2 * Np
    # for _bulk_twirled_deriv sub-call
    printer.log("Memory estimate of %.1f GB for all-Jac mode." %
                (memEstimatealljac / 1024.0**3), 1)            

    memEstimatesinglejac = FLOATSIZE * 3 * len(model_list) * Np**2 + \
        FLOATSIZE * 3 * len(model_list) * dim**2 * Np
    #Factor of 3 accounts for currentDDDs, testDDDs, and bestDDDs
    printer.log("Memory estimate of %.1f GB for single-Jac mode." %
                (memEstimatesinglejac / 1024.0**3), 1)            

    if mem_limit is not None:
        
        printer.log("Memory limit of %.1f GB specified." %
            (mem_limit / 1024.0**3), 1)
    
        if memEstimatesinglejac > mem_limit:
                raise MemoryError("Too little memory, even for single-Jac mode!")
    
        if mode=="all-Jac" and (memEstimatealljac > mem_limit):
            #fall back to single-Jac mode
            
            printer.log("Not enough memory for all-Jac mode, falling back to single-Jac mode.", 1)
            
            mode = "single-Jac"  # compute a single germ's jacobian at a time    

    if pretest:
        undercompleteModelNum = test_germs_list_completeness(model_list,
                                                             germs_list,
                                                             score_func,
                                                             threshold,
                                                             float_type=float_type,
                                                             comm=comm)
        if undercompleteModelNum > -1:
            printer.warning("Complete initial germ set FAILS on model "
                            + str(undercompleteModelNum) + ".")
            printer.warning("Aborting search.")
            return None

        printer.log("Complete initial germ set succeeds on all input models.", 1)
        printer.log("Now searching for best germ set.", 1)

    printer.log("Starting germ set optimization. Lower score is better.", 1) 

    twirledDerivDaggerDerivList = None

    if mode == "all-Jac":
        twirledDerivDaggerDerivList = \
            [_compute_bulk_twirled_ddd(model, germs_list, tol,
                                       check, germLengths, comm, float_type=float_type)
             for model in model_list]
        printer.log(f'Numpy Array Data Type: {twirledDerivDaggerDerivList[0].dtype}', 2)
        printer.log("Numpy array data type for twirled derivatives is: "+ str(twirledDerivDaggerDerivList[0].dtype)+
                    " If this isn't what you specified then something went wrong.", 2) 
                    
        currentDDDList = []
        for i, derivDaggerDeriv in enumerate(twirledDerivDaggerDerivList):
            currentDDDList.append(_np.sum(derivDaggerDeriv[_np.where(weights == 1)[0], :, :], axis=0))

    elif mode == "single-Jac":
        currentDDDList = [_np.zeros((Np, Np), dtype=float_type) for mdl in model_list]

        loc_Indices, _, _ = _mpit.distribute_indices(
            list(range(len(goodGerms))), comm, False)

        with printer.progress_logging(3):
            for i, goodGermIdx in enumerate(loc_Indices):
                printer.show_progress(i, len(loc_Indices),
                                      prefix="Initial germ set computation",
                                      suffix=germs_list[goodGermIdx].str)

                for k, model in enumerate(model_list):
                    currentDDDList[k] += _compute_twirled_ddd(
                        model, germs_list[goodGermIdx], tol, float_type=float_type)

        #aggregate each currendDDDList across all procs
        if comm is not None and comm.Get_size() > 1:
            for k, model in enumerate(model_list):
                result = _np.empty((Np, Np), dtype=float_type)
                comm.Allreduce(currentDDDList[k], result, op=MPI.SUM)
                currentDDDList[k][:, :] = result[:, :]
                result = None  # free mem
                
    elif mode== "compactEVD":
        #implement a new caching scheme which takes advantage of the fact that the J^T J matrices are typically
        #rather sparse. Instead of caching the J^T J matrices for each germ we'll cache the compact SVD of these
        #and multiply the compact SVD components through each time we need one.
        twirledDerivDaggerDerivList = \
            [_compute_bulk_twirled_ddd_compact(model, germs_list, tol,
                                              evd_tol=1e-10, float_type=float_type, printer=printer)
             for model in model_list]
             
             #_compute_bulk_twirled_ddd_compact returns a tuple with three lists
             #corresponding to the u, sigma and vh matrices for each germ's J^T J matrix's_list
             #compact svd.
        currentDDDList = []
        nonzero_weight_indices= _np.nonzero(weights)
        nonzero_weight_indices= nonzero_weight_indices[0]
        for i, derivDaggerDeriv in enumerate(twirledDerivDaggerDerivList):
            #reconstruct the needed J^T J matrices
            for j, idx in enumerate(nonzero_weight_indices):
                if j==0:
                    temp_DDD = derivDaggerDeriv[0][idx] @ derivDaggerDeriv[2][idx]
                else:
                    temp_DDD += derivDaggerDeriv[0][idx] @ derivDaggerDeriv[2][idx]

            currentDDDList.append(temp_DDD)

    else:
        raise ValueError("Invalid mode: %s" % mode)  # pragma: no cover

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'score_fn': lambda x: _scoring.list_score(x, score_func=score_func),
        'threshold_ac': threshold,
        'num_nongauge_params': numNonGaugeParams,
        'op_penalty': op_penalty,
        'germ_lengths': germLengths,
        'float_type': float_type
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
                        testDDD += _compute_twirled_ddd(
                            model, germs_list[candidateGermIdx], tol, float_type=float_type)
                    
                    elif mode == "compactEVD":
                        #reconstruct the J^T J matrix from it's compact SVD
                        testDDD += twirledDerivDaggerDerivList[k][0][candidateGermIdx] @ \
                                   _np.diag(twirledDerivDaggerDerivList[k][1][candidateGermIdx]) @\
                                   twirledDerivDaggerDerivList[k][2][candidateGermIdx]
                    # (else already checked above)
                    
                    nonAC_kwargs['germ_lengths'] = \
                        _np.array([len(germ) for germ in
                                   (goodGerms + [germs_list[candidateGermIdx]])])
                    worstScore = max(worstScore, compute_composite_germ_set_score(
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


def find_germs_integer_slack(model_list, germs_list, randomize=True,
                             randomization_strength=1e-3, num_copies=None,
                             seed=0, l1_penalty=1e-2, op_penalty=0,
                             initial_weights=None, score_func='all',
                             max_iter=100, fixed_slack=False,
                             slack_frac=False, return_all=False, tol=1e-6,
                             check=False, force="singletons",
                             force_score=1e100, threshold=1e6,
                             verbosity=1, float_type=_np.cdouble):
    """
    Find a locally optimal subset of the germs in germs_list.

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
        List of all germ circuits to consider.

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
        When `force` designates a non-empty set of circuits, the score to
        assign any germ set that does not contain each and every required germ.

    threshold : float, optional (default is 1e6)
        Specifies a maximum score for the score matrix, above which the germ
        set is rejected as amplificationally incomplete.

    verbosity : int, optional
        Integer >= 0 indicating the amount of detail to print.

    See Also
    --------
    :class:`~pygsti.objects.Model`
    :class:`~pygsti.objects.Circuit`
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    model_list = _setup_model_list(model_list, randomize,
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

    undercompleteModelNum = test_germs_list_completeness(model_list,
                                                         germs_list, score_func,
                                                         threshold,
                                                         float_type=float_type)
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
    model0 = _remove_spam_vectors(model_list[0])

    # Initially allow adding to weight. -- maybe make this an argument??
    lessWeightOnly = False

    nGaugeParams = model0.num_gauge_params

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

    twirledDerivDaggerDerivList = [_compute_bulk_twirled_ddd(model, germs_list, tol, float_type=float_type)
                                   for model in model_list]

    # Dict of keyword arguments passed to _germ_set_score_slack that don't change from
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

    scoreList = [_germ_set_score_slack(weights, model_num, **cs_kwargs)
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
                        neighborScoreList.append(_germ_set_score_slack(neighbor,
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


def _germ_set_score_grasp(germ_set, germs_list, twirled_deriv_dagger_deriv_list,
                          non_ac_kwargs, init_n=1):
    """
    Score a germ set against a collection of models.

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
        :func:`compute_composite_germ_set_score` for the scoring of the germ set against
        individual models.

    init_n : int
        The number of eigenvalues to begin checking for amplificational
        completeness with respect to. Passed as an argument to
        :func:`compute_composite_germ_set_score`.

    Returns
    -------
    CompositeScore
        The worst score over all models of the germ set.
    """
    weights = _np.zeros(len(germs_list))
    germ_lengths = []
    for germ in germ_set:
        weights[germs_list.index(germ)] = 1
        germ_lengths.append(len(germ))
    germsVsModelScores = []
    for derivDaggerDeriv in twirled_deriv_dagger_deriv_list:
        # Loop over all models
        partialDDD = derivDaggerDeriv[_np.where(weights == 1)[0], :, :]
        kwargs = non_ac_kwargs.copy()
        if 'germ_lengths' in non_ac_kwargs:
            kwargs['germ_lengths'] = germ_lengths
        germsVsModelScores.append(compute_composite_germ_set_score(
            partial_deriv_dagger_deriv=partialDDD, init_n=init_n, **kwargs))
    # Take the score for the current germ set to be its worst score over all
    # models.
    return max(germsVsModelScores)


def find_germs_grasp(model_list, germs_list, alpha, randomize=True,
                     randomization_strength=1e-3, num_copies=None,
                     seed=None, l1_penalty=1e-2, op_penalty=0.0,
                     score_func='all', tol=1e-6, threshold=1e6,
                     check=False, force="singletons",
                     iterations=5, return_all=False, shuffle=False,
                     verbosity=0, num_nongauge_params=None, float_type=_np.cdouble):
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
        List of all germ circuits to consider.

    alpha : float
        A number between 0 and 1 that roughly specifies a score theshold
        relative to the spread of scores that a germ must score better than in
        order to be included in the RCL. A value of 0 for `alpha` corresponds
        to a purely greedy algorithm (only the best-scoring germ set is
        included in the RCL), while a value of 1 for `alpha` will include all
        germs in the RCL.
        See :func:`pygsti.algorithms.scoring.filter_composite_rcl` for more details.

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

    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.
        
    float_type : Numpy dtype object, optional
        Numpy data type to use for floating point arrays

    Returns
    -------
    finalGermList : list of Circuit
        Sublist of `germs_list` specifying the final, optimal set of germs.
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    model_list = _setup_model_list(model_list, randomize,
                                   randomization_strength, num_copies, seed)

    (_, numGaugeParams,
     numNonGaugeParams, _) = _get_model_params(model_list)
    if num_nongauge_params is not None:
        numGaugeParams = numGaugeParams + numNonGaugeParams - num_nongauge_params
        numNonGaugeParams = num_nongauge_params

    germLengths = _np.array([len(germ) for germ in germs_list], _np.int64)

    numGerms = len(germs_list)

    initialWeights = _np.zeros(numGerms, dtype=_np.int64)
    if force:
        if force == "singletons":
            initialWeights[_np.where(germLengths == 1)] = 1
        else:  # force should be a list of Circuits
            for opstr in force:
                initialWeights[germs_list.index(opstr)] = 1

    def _get_neighbors_fn(weights): return _grasp.neighboring_weight_vectors(
        weights, forced_weights=initialWeights, shuffle=shuffle)

    undercompleteModelNum = test_germs_list_completeness(model_list,
                                                         germs_list,
                                                         score_func,
                                                         threshold,
                                                         float_type=float_type)
    if undercompleteModelNum > -1:
        printer.warning("Complete initial germ set FAILS on model "
                        + str(undercompleteModelNum) + ".")
        printer.warning("Aborting search.")
        return (None, None, None) if return_all else None

    printer.log("Complete initial germ set succeeds on all input models.", 1)
    printer.log("Now searching for best germ set.", 1)

    printer.log("Starting germ set optimization. Lower score is better.", 1)

    twirledDerivDaggerDerivList = [_compute_bulk_twirled_ddd(model, germs_list, tol,
                                                             check, germLengths, float_type=float_type)
                                   for model in model_list]

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'score_fn': lambda x: _scoring.list_score(x, score_func=score_func),
        'threshold_ac': threshold,
        'op_penalty': op_penalty,
        'germ_lengths': germLengths,
        'num_nongauge_params': numNonGaugeParams,
        'float_type' : float_type
    }

    final_nonAC_kwargs = nonAC_kwargs.copy()
    final_nonAC_kwargs['l1_penalty'] = l1_penalty

    scoreFn = (lambda germSet:
               _germ_set_score_grasp(germSet, germs_list,
                                     twirledDerivDaggerDerivList, nonAC_kwargs,
                                     init_n=1))
    finalScoreFn = (lambda germSet:
                    _germ_set_score_grasp(germSet, germs_list,
                                          twirledDerivDaggerDerivList,
                                          final_nonAC_kwargs, init_n=1))

    #OLD: feasibleThreshold = _scoring.CompositeScore(-numNonGaugeParams,threshold,numNonGaugeParams))
    def _feasible_fn(germ_set):  # now that scoring is not ordered entirely by N
        s = _germ_set_score_grasp(germ_set, germs_list,
                                  twirledDerivDaggerDerivList, nonAC_kwargs,
                                  init_n=1)
        return (s.N >= numNonGaugeParams and s.minor < threshold)

    def rcl_fn(x): return _scoring.filter_composite_rcl(x, alpha)

    initialSolns = []
    localSolns = []

    for iteration in range(iterations):
        # This loop is parallelizable (each iteration is independent of all
        # other iterations).
        printer.log('Starting iteration {} of {}.'.format(iteration + 1,
                                                          iterations), 1)
        success = False
        failCount = 0
        rng = _random.Random(seed)
        while not success and failCount < 10:
            try:
                iterSolns = _grasp.run_grasp_iteration(
                    elements=germs_list, greedy_score_fn=scoreFn, rcl_fn=rcl_fn,
                    local_score_fn=scoreFn,
                    get_neighbors_fn=_get_neighbors_fn,
                    feasible_fn=_feasible_fn,
                    initial_elements=initialWeights, rng=rng,
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


def clean_germ_list(model, circuit_cache, eq_thresh= 1e-6):
    #initialize an identity matrix of the appropriate dimension
    
    cleaned_circuit_cache= circuit_cache.copy()
                   
    
    #remove circuits with duplicate PTMs
    #The list of available fidcuials is typically
    #generated in such a way to be listed in increasing order
    #of depth, so if we search for dups in that order this should
    #generally favor the shorted of a pair of duplicate PTMs.
    #cleaned_cache_keys= list(cleaned_circuit_cache.keys())
    #cleaned_cache_PTMs= list(cleaned_circuit_cache.values())
    #len_cache= len(cleaned_cache_keys)
    
    #reverse the list so that the longer circuits are at the start and shorter
    #at the end for better pop behavior.
    
    #TODO: add an option to partition the list into smaller chunks to dedupe
    #separately before regrouping and deduping as a whole. Should be a good deal faster. 
    
    unseen_circs  = list(cleaned_circuit_cache.keys())
    unseen_circs.reverse()
    unique_circs  = []
    
    #While unseen_circs is not empty
    while unseen_circs:
        current_ckt = unseen_circs.pop()
        current_ckt_PTM = cleaned_circuit_cache[current_ckt]
        unique_circs.append(current_ckt)            
        #now iterate through the remaining elements of the set of unseen circuits and remove any duplicates.
        is_not_duplicate=[True]*len(unseen_circs)
        for i, ckt in enumerate(unseen_circs):
            #the default tolerance for allclose is probably fine.
            if _np.linalg.norm(cleaned_circuit_cache[ckt]-current_ckt_PTM)<eq_thresh: #use same threshold as defined in the base find_fiducials function
                is_not_duplicate[i]=False
        #reset the set of unseen circuits.
        unseen_circs=list(itertools.compress(unseen_circs, is_not_duplicate))
    
    #rebuild the circuit cache now that it has been de-duped:
    cleaned_circuit_cache_1= {ckt_key: cleaned_circuit_cache[ckt_key] for ckt_key in unique_circs}
        
    #now that we've de-duped the circuit_cache, we can pull out the keys of cleaned_circuit_cache_1 to get the
    #new list of available fiducials.
    
    cleaned_availableGermList= unique_circs
    
        
    return cleaned_availableGermList, cleaned_circuit_cache_1
    

#new function for taking a list of available fiducials and generating a cache of the PTMs
#this will also be useful trimming the list of effective identities and fiducials with
#duplicated effects.

def create_circuit_cache(model, circuit_list):
    """
    Function for generating a cache of PTMs for the available fiducials.
    
    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    ckt_list : list of Circuits
        Full list of all fiducial circuits avalable for constructing an informationally complete state preparation.
    
    Returns
    -------
    dictionary
        A dictionary with keys given by circuits with corresponding
        entries being the PTMs for that circuit.
    
    """
    
    circuit_cache= {}
    for circuit in circuit_list:
        circuit_cache[circuit] = model.sim.product(circuit)
    
    return circuit_cache
    
#new function to drop a random fraction of the available germ list:
def drop_random_germs(candidate_list, rand_frac, target_model, keep_bare=True, seed=None):
    """
    Function for dropping a random fraction of the candidate germ list.
    
    Parameters
    ----------
    
    candidate_list : list of Circuits
        List of candidate germs
    
    target_model : Model
        The model (associates operation matrices with operation labels)
        
    rand_frac : float between 0 and 1
        random fraction of candidate germs to drop
        
    keep_bare : bool
        Whether to always include the bare germs in the returned set.
       
   
    Returns
    -------
    availableGermsList : List
        list of candidate germs with random fraction dropped.
    
    """
    
    #If keep_bare is true we should get a list of the operations
    #from the target model, then construct two lists. One of the bare
    #germs and another of the candidates sans the bare germs.
    
    
    if seed is not None:
        rng= _np.random.default_rng(seed)
    else:
        rng= _np.random.default_rng()
        
    if keep_bare:
        bare_op_labels= target_model.operations.keys()
#        #pull the labels in a different way depending on if this is a qubit or qudit state space
#        if isinstance(target_model.state_space, _ExplicitStateSpace):
#            tpb0_labels = target_model.state_space.labels[0]
#        elif isinstance(target_model.state_space, _QuditSpace):
#            tpb0_labels = target_model.state_space.qudit_labels
#        else:
#            raise ValueError('I only know how to convert the operations to their corresponding circuits for models with ExplicitStateSpace or QuditSpace associated with them')
#        bare_op_ckts= [_circuits.Circuit([op_label],line_labels=tpb0_labels) for op_label in bare_op_labels]
        bare_op_ckts= _circuits.list_all_circuits_onelen(list(bare_op_labels), length=1)
        #drop these bare ops from the candidate_list
        candidate_list= [ckt for ckt in candidate_list if ckt not in bare_op_ckts]
        
        #now sample a random fraction of these to keep:
        indices= _np.arange(len(candidate_list))
        num_to_keep= len(indices)-floor(rand_frac*len(indices))
        indices_to_keep= rng.choice(indices, size=num_to_keep, replace=False)
        
        #Now reconstruct the list of ckts from these sampled indices:
        updated_candidate_list= [candidate_list[i] for i in indices_to_keep]
        
        #add back in the bare germs
        updated_candidate_list= bare_op_ckts + updated_candidate_list
        
       
    #if not keeping the bare germs then we'll got ahead and just drop a random fraction  
    else:
        #now sample a random fraction of these to keep:
        indices= _np.arange(len(candidate_list))
        num_to_keep= len(indices)-floor(rand_frac*len(indices))
        indices_to_keep= rng.choice(indices, size=num_to_keep, replace=False)
        
        #Now reconstruct the list of ckts from these sampled indices:
        updated_candidate_list= [candidate_list[i] for i in indices_to_keep]
        
    return updated_candidate_list
  
##-------------Low-Rank Update Theory--------------##
#What follows is a brief overview of the theory for the use of low-rank updates with greedy search. 
#The purpose of this is to explain the linear algebra, and not the germ selection theory.
#So I'll be sloppy and refer to the existence of various jacobians generically without 
#explaining how these arise or why they are the jacobians we care about.

#The goal of germ selection and other experimental designates problems is to select a set of germs, 
#or fiducial pairs or whatever with a jacobian J that satisfies some criterion.
#We can choose to work in terms of the Jacobian directly, or in terms of one of the gramians of the Jacobian 
#(J^T@J or J@J^T depending on what is most appropriate).

#For germ selection there are 2-main objective functions that are used. Both are based on a connection
#between the gramian of a jacobian and covariance. So both objective functions can be viewed as alternative
#ways to guarantee the covariance in the parameters for an estimate generated from an experiment design
#isn't relatively small. (See this stackexchange thread for more on that connection https://stats.stackexchange.com/questions/231868/relation-between-covariance-matrix-and-jacobian-in-nonlinear-least-squares)
#This first objective function aims to maximize the minimum eigenvalue of the jacobian's gramian, or 
#equivalently minimize the maximum eigenvalue of the inverse. This is called 'worst' in pygsti. The other option,
#which is the default, is to minimize the sum of the reciprocals of the eigenvalues of the gramian of 
#the jacobian. This is called 'all' and corresponds roughly, but not exactly, to minimizing the average
#covariance. Going forward I will refer to this as the psuedoinverse-trace (since that is indeed what it is).

#We haven't fully figured out how to low-rank updatify the 'worst' objective function (there are a few
#half-baked ideas based on used iterative methods we might try to make fully baked if there is a demand
#down the line). So the remaining will focus on speeding up the evaluation of the 'all' objective function.

#What does each step in the greedy search algorithm for this problem look like.
#At the start of each iteration we have some current jacobian for the current set of germs that
#we've chosen to include in our set that we'll denote J. At each iteration we also have some list 
#of candidate germs to select from among for the next germ to add to our. Each of these  
#candidate germs likewise has a jacobian associated with it that we'll denote by A_i.
#I'll be referring the the A_i matrices (and interchangeably their gramians) as 'updates'.

#We want to know how each of these updates would change the psuedoinverse-trace (and also rank)
#of the jacobian for the current set of germs were it to be added and then select the germ
#that improves the psuedoinverse-trace (and rank) the most in a greedy fashion. That is, for each
#of the updates we want to calculate the psuedoinverse-trace of the matrix J'_i:

#pinvtrace(J'_i)= pinvtrace(J^T@J + A_i^T@A_i)

#Here are the critical observations that allow us to significantly speed things up:
#   1. For each iteration of the greedy search algorithm the matrix J is fixed.
#   2. The update matrices are relatively low-rank. For case of germ selection
#      each of the A_i matrices is d**4 X N_p where d is the dimension of the system's_list
#      underlying Hilbert space and N_p is the number of parameters of a gate set 
#      (technically the number of non-SPAM parameters). Subsequantly the maximum rank of
#      A_i is d**4, but it is typically a good deal less in practice. 
#      In traditional settings it is always the case that N_p >= d**4 and typically 
#      N_p >> rank(d**4).
#   3. We're almost always working from some initial fixed set of germs that we'll
#      be constructing our solution from, so the set of matrices {A_i} is essentially
#      fixed (only getting smaller as we add germs to the solution set) and so all of these
#      matrices can be computed ahead of time and cached in some way.

#Before explaining what is actually done, a quick diversion. The most famous example
#of using low-rank updates comes from the 'Woodbury matrix identity' 
#(AKA, matrix inversion lemma or Sherman–Morrison–Woodbury formula).
#This identity says:

#(A + UCV)^-1 = A^-1 - A^-1 U(C^-1 + V A^1 U)^-1 V A^-1

#A, U, C and V are assumed to be conformable (so all of the matrix multiplications work). 
#A is (n,n), C is (k,k), U is (n,k) and V is (k,n).

#How is this useful? This formula gives us a way to update the inverse of 
#some matrix A given some additive update UCV. Suppose we already knew what A^-1 was somehow,
#then we could replace the inversion of an (n,n) matrix with the inversion of 2 (k,k) matrices
#and some matrix multiplication and addition. Moreover, suppose that C was a diagonal matrix,
#then we could do the inversion in linear time which is basically free and effectively only have
#the cost of inverting (C^-1 + V A^1 U) to worry about. If k<<n and it is the case that we already
#knew A^-1 then this is a clear and significant win.

#Obviously this result requires that the matrix we are updating have a well defined inverse and
#so is full-rank (ditto for C). If this were the case then this would solve our problem.
#At the start of each iteration we pre-compute the inverse of J^T@J once and then use the formula
#to calculate the updated inverse after adding A_i^T@A_i. Since J^T@J isn't full-rank, however,
#we need a version of this that works for the psuedoinverse instead (simply replacing the inverses
#with psuedoinverses in the woodbury formula doesn't work except in a couple special cases that we 
#won't satisfy). 

#For this we'll combine results from 2 papers. The first is from Matthew Brand titled:
#Fast Low-Rank Modifications of the Thin Singular Value Decomposition 
#(https://www.merl.com/publications/docs/TR2006-059.pdf)
#and the second is from Nariyasu Minamide titled:
#An Extension of the Matrix Inversion Lemma
#(https://epubs.siam.org/doi/pdf/10.1137/0606038)


#The problem that Brand's paper solves is the following:
#Let X be some matrix. X has an SVD X=USV^T. Let AB^T be some low-rank additive update.
#What is the updated SVD of X+AB^T? The main result we use comes from equation 3 which
#gives a new matrix K which has the same spectrum as X+AB^T:

# K= [[S 0],[0 0]] + [[U^T A],[R_A]] [[V^T B], [R_B]]^T

#Where R_A = P^T(I-UU^T)A. P is an orthogonal basis for the column space of 
#(I-UU^T)A, the component of A that is orthogonal to U. R_B is defined analogously
#but swapping U->V and A->B.

#K won't have the same psuedoinverse as the matrix we care about, but since K has the same spectrum 
#it will also have the same psuedoinverse-trace as the matrix we care about. So, for the next stage
#we can instead proceed with respect to K. Since K has a nice block structure this will simplify
#the application of the result from minamide's paper. 

#This actually simplifies a bit, since all of the matrices we'll be working with are gramians
#they will all be diagonalizable. As such, we can replace the SVD of X above with the eigenvalue
#decomposition X=UEU^T. Moreover, for the update AB^T we'll be using a rank-decomposition
#of the updates and so these with be of the form AA^T. As such, the K matrix will simplify to:

# K= [[E 0],[0 0]] + [[U^T A],[R_A]] [[U^T A], [R_A]]^T

#The relevant result from Minamide is from theorem 2.1 which states:
#let H= S+\Phi\Phi^\dagger where S is an (n,n) hermitian matrix and \Phi is an (n,m) matrix with
#potentially complex entries. If S is non-negative, letting ^+ denote psuedoinversion we then have:

#H^+ = {I-(\Phi^\dagger T)^+ \Phi^\dagger} S^+ {I-\Phi (T \Phi)^+} +
#      (\Phi^\dagger T)^+(T \Phi)^+ -
#      {I-(\Phi^\dagger T)^+ \Phi^\dagger} (S^+ \Phi B D^-1 B \Phi^\dagger S^+)  {I-\Phi (T \Phi)^+}

#Where B= I- (T \Phi)^+(T \Phi), D= I + B \Phi^\dagger S^+ \Phi B.
#T is a projector onto the orthogonal complement of the column space of S. i.e. T= I- S^+S= I-SS^+.

#The K matrix we defined above satisfies the requirements to use the minamide result where mapping between
#the two notations we let:
#S=[[E 0],[0 0]] and \Phi= [[U^T A],[R_A]] (Note that since gramians are PSD S is nonnegative as required).

#The expression above looks pretty monstrous, but the block structure we have from switching to working with K
#simplifies things considerably. Plugging everything into Minamide's threorem and working
#through a few pages of tedious linear algebra gives us the following result:

#H^+ = [[H_00, H_01], [H_10 H_11]]

#H_00= E^+U^T A \alpha A^T U E^+ 
#H_11 = (R_A^T)^+ A^T U E^+ U^T A R_A^+ + (R_A^+)^T R_A^+ - (R_A^T)^+ A^T U E^+ U^T A \alpha A^T U E^+ U^T A R_A^+
#H_01= E^+ U^T A R_A^+ - E^+ U^T A \alpha A^T E^+ U^T A R_A^+
#H_10 = H_01^T (~90% sure of this last line, turns out we won't actually care either way since we don't need the off-diagonals)

#\alpha= B D^-1 B^+ = B D^-1 B =  (I-R_A^+ R_A) [(I + (I-R_A^+ R_A)(A^T U E^+ U^T A)(I-R_A^+ R_A))^-1] (I-R_A^+ R_A)

#This is still ugly, but:

#   1. We only care about the trace of H^+, so we don't ever need to calculate the off-diagonal blocks H_01 and H_10.
#      Just the diagonal blocks H_00 and H_11.
#   2. Within H_00 and H_11 there is a lot of structure. All of the products are symmetric and so we only need to calculate half
#      each term. There are a lot of repeated subexpressions so with some caching of intermediate expressions there are a lot fewer
#      matmuls to do overall.
#   3. Most of these matrices are small with at least one of the dimensions given by the rank of the update
#      so even with a bunch of matmuls to do they run pretty quickly.

#There are a few more tricks to keep in mind that are used to accelerate the calculation even more 
#(and without knowing about makes the implementation details harder to parse).

#   1. The psuedoinverse of a diagonal matrix is simply a diagonal matrix with the non-zero diagonal
#      entries inverted, so no need for a costly SVD (which is how numpy internally implements pinv)
#   2. The since it is diagonal the matrix multiplications involving E^+ can be done in quadratic time since all we need to do is 
#      rescale the rows of the other matrix be the appropriate diagonal element. In numpy this can be done using element-wise multiplication
#      between a 1D vector with the diagonal elements of E^+ and the target matrix. Numpy automatically uses broadcasting for this so it is
#      very fast.
#   3. For the purposes of calculating the trace we don't need all of the elements of H_00 and H_11, just the diagonals.
#      We can use numpy's einsum to evaluate the final (and largest/most expensive) matrix multiplication in such as
#      way that only the diagonal elements are calculated/returned, and which can be done in just quadratic time.
#   4. A bunch of numpy methods can detect when they should expect to have a symmetric output and use a faster code path
#      in this case that avoids calculating known duplicate entries. So, whenever possible we'll be making sure our subexpressions
#      are collected in such a way as to leverage the symmetry. This also applies to einsum, which is significantly when it detects
#      symmetric inputs, so we'll also try to symmetrize the inputs to einsum when possible (when you see Cholesky decompositions appear
#      out of nowhere that is why).


#Future Performance Enhancements:
#Inventory of possible future perofrmance enhancements.
#   1. At the start of each iteration we compute a so-called 'update cache'. This is really just the eigenvalues and eigenvectors
#      for the jacobian of our current candidate set as well as the projector onto the orthogonal complement of U's column space.
#      This is calculated in the standard way using eigh. We could in principle use the second half of the Matthew Brand result
#      and use low-rank update methods to update the eigenvalues and eigenvectors using whatever final update we selected at the previous
#      greedy search iteration. This still requires doing an actual eigenvalue decomposition on the K matrix we construct, though, so
#      what will happen is that this will be very fast for the earlier iterations where both the jacobian for the current solution set and the
#      updates are both small, but will give us diminishing returns as the Jacobian for the current solution set approaches being full-rank.
#      We could always use some heuristic to swap between the two approaches once the rank is past some threshold, though. For 3-qubit+ 
#      systems profiling suggests that constructing the update cache takes nearly the same amount of time as running through hundreds of
#      low-rank updates, so this could be a source of pretty large speedups. Vauge numerical stability concerns (and currently being fast enough)
#      are the main reasons we haven't already done so.
#   2. Better search space pruning. We can do a better job at identifying useless germs and not including them in future iterations.
#      We currently have an option called 'force_rank_increase' that is plumbed in. This is currently used to short-circuit the low-rank
#      update tests early when it detects complete overlap with existing amplified directions in parameter space, but we aren't currently 
#      using that information to then skip that germ in subsequent iterations. Doing so would speed things up when that option was turned on.
#   3. A decent chunk of time for the low-rank updates is doing a rank-revealing QR decomposition. We learned recently from Riley Murray that these
#      are a good deal slower than a standard un-pivoted QR decomposition. There are algorithms for doing this using randomized linear algebra
#      that are accurate to machine precision and nearly as fast as the standard QR decomposition.

##------------Low-Rank Update Related Functions-----------------##

#new function that computes the J^T J matrices but then returns the result in the form of the 
#compact EVD in order to save on memory.    
def _compute_bulk_twirled_ddd_compact(model, germs_list, eps,
                                       comm=None, evd_tol=1e-10,  float_type=_np.cdouble,
                                       printer=None, return_eigs=False):

    """
    Calculate the positive squares of the germ Jacobians.

    twirledDerivDaggerDeriv == array J.H*J contributions from each germ
    (J=Jacobian) indexed by (iGerm, iModelParam1, iModelParam2)
    size (nGerms, vec_model_dim, vec_model_dim)

    Parameters
    ----------
    model : Model
        The model defining the parameters to differentiate with respect to.

    germs_list : list
        The germ set

    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. abs(eval1 - eval2) < eps ? )
        
    evd_tol : float, optional
        Tolerance used for determining if a singular value has zero magnitude when constructing the
        compact SVD.
    
    check : bool, optional
        Whether to perform internal consistency checks, at the expense of
        making the function slower.

    germ_lengths : numpy.ndarray, optional
        A pre-computed array of the length (depth) of each germ.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.
        
    float_type : numpy dtype object, optional
        Numpy data type to use in floating point arrays.
        
    return_eigs : bool, optional, default False
        If True then additionally return a list of the arrays of eigenvalues for each
        germ's twirled derivative gramian.

    Returns
    -------
    sqrteU_list : list of numpy ndarrays
        list of the non-trivial left eigenvectors for the compact EVD for each germ
        where each left eigenvector is multiplied by the sqrt of the corresponding eigenvalue.
    e_list : ndarray
        list of non-zero eigenvalue arrays for each germ.
    """
       
    #TODO: Figure out how to pipe in a comm object to parallelize some of this with MPI.
       
    sqrteU_list=[]
    e_list=[]

    #remove spam parameters from the model before calculating the jacobian.
    #The representations of the germ process matrices are clearly independent 
    #of the spam parameters. (I say that, but I only realized I had forgotten this like
    #6 months later...)
    if len(model.preps) > 0 or len(model.povms) > 0:
        model = _remove_spam_vectors(model)
        # This function assumes model has no spam elements so `lookup` below
    
    if printer is not None:
        printer.log('Generating compact EVD Cache',1)
        
        with printer.progress_logging(1):
    
            for i, germ in enumerate(germs_list):
            
                printer.show_progress(iteration=i, total=len(germs_list), bar_length=25)
                    
                twirledDeriv = _twirled_deriv(model, germ, eps, float_type) / len(germ)
                #twirledDerivDaggerDeriv = _np.tensordot(_np.conjugate(twirledDeriv),
                #                                        twirledDeriv, (0, 0))
                twirledDerivDerivDagger = twirledDeriv@(twirledDeriv.conj().T) 
                                                        
                #now take twirledDerivDerivDagger and construct its compact EVD.
                e, U= compact_EVD(twirledDerivDerivDagger, evd_tol)
                
                #now connect this to the compact EVD of twirledDerivDaggerDeriv
                #using the definition of the left and right singular
                #vectors of a matrix.
                #Multiply U by twirledDeriv.conj().T and rescale the columns
                #by the corresponding singular value, i.e. the sqrt of the
                #eigenvalue. Use some broadcasting for fast rescaling.
                U_remapped= ((twirledDeriv.conj().T)@U)/_np.sqrt(e.reshape((1,len(e))))
                
                #e, U= compact_EVD_via_SVD(twirledDeriv, evd_tol)
                
                e_list.append(e)
                
                #by doing this I am assuming that the matrix is PSD, but since these are all
                #gramians that should be alright.
                
                #I want to use a rank-decomposition, so split the eigenvalues into a pair of diagonal
                #matrices with the square roots of the eigenvalues on the diagonal and fold those into
                #the matrix of eigenvectors by left multiplying.
                
                sqrteU_list.append( U_remapped@_np.diag(_np.sqrt(e)) )       
    else: 
        for i, germ in enumerate(germs_list):
                
            twirledDeriv = _twirled_deriv(model, germ, eps, float_type) / len(germ)
            #twirledDerivDaggerDeriv = _np.tensordot(_np.conjugate(twirledDeriv),
            #                                        twirledDeriv, (0, 0))
            twirledDerivDerivDagger = twirledDeriv@(twirledDeriv.conj().T) 
                                                    
            #now take twirledDerivDerivDagger and construct its compact EVD.
            e, U= compact_EVD(twirledDerivDerivDagger, evd_tol)
            
            #now connect this to the compact EVD of twirledDerivDaggerDeriv
            #using the definition of the left and right singular
            #vectors of a matrix.
            #Multiply U by twirledDeriv.conj().T and rescale the columns
            #by the corresponding singular value, i.e. the sqrt of the
            #eigenvalue. Use some broadcasting for fast rescaling.
            U_remapped= ((twirledDeriv.conj().T)@U)/_np.sqrt(e.reshape((1,len(e))))
            #e, U= compact_EVD_via_SVD(twirledDeriv, evd_tol)
            
            e_list.append(e)
            
            #by doing this I am assuming that the matrix is PSD, but since these are all
            #gramians that should be alright.
            
            #I want to use a rank-decomposition, so split the eigenvalues into a pair of diagonal
            #matrices with the square roots of the eigenvalues on the diagonal and fold those into
            #the matrix of eigenvectors by left multiplying.
            sqrteU_list.append( U_remapped@_np.diag(_np.sqrt(e)) )  
           
    if return_eigs:
        return sqrteU_list, e_list
    else:
        return sqrteU_list
    
#New function for computing the compact eigenvalue decompostion of a matrix.
#Assumes that we are working with a diagonalizable matrix, no safety checks made.

def compact_EVD(mat, threshold= 1e-10):
    """
    Generate the compact eigenvalue decomposition of the input matrix.
    Assumes of course that the user has specified a diagonalizable matrix,
    there are no safety checks for that made a priori.
    
    input:
    
    mat : ndarray
        input matrix we want the compact EVD for. Assumed to be diagonalizable.
        
    threshold : float, optional
        threshold value for deciding if an eigenvalue is zero.
        
    output:
    
    e : ndarray
        1-D numpy array of the non-zero eigenvalues of mat.
    U : ndarray
        Matrix such that U@diag(s)@U.conj().T=mat.
    """
    
    #take the EVD of mat.
    e, U= _np.linalg.eigh(mat)

    #How many non-zero eigenvalues are there and what are their indices
    nonzero_eigenvalue_indices= _np.nonzero(_np.abs(e)>threshold)

    #extract the corresponding columns and values fom U and s:
    #For EVD/eigh We want the columns of U and the rows of Uh:
    nonzero_e_values = e[nonzero_eigenvalue_indices]
    nonzero_U_columns = U[:, nonzero_eigenvalue_indices[0]]
    
    return nonzero_e_values, nonzero_U_columns
    
#Make a rev1 of the compact_EVD function that actually uses a direct SVD on the Jacobian
#instead, but for compatibility returns the same output as the first revision compact_EVD function.
def compact_EVD_via_SVD(mat, threshold= 1e-10):
    """
    Generate the compact eigenvalue decomposition of the input matrix.
    Assumes of course that the user has specified a diagonalizable matrix,
    there are no safety checks for that made a priori.
    
    input:
    
    mat : ndarray
        input matrix we want the compact EVD for. Assumed to be diagonalizable.
        
    threshold : float, optional
        threshold value for deciding if an eigenvalue is zero.
        
    output:
    
    e : ndarray
        1-D numpy array of the non-zero eigenvalues of mat.
    U : ndarray
        Matrix such that U@diag(s)@U.conj().T=mat.
    """
    
    #take the SVD of mat.
    try:
        _, s, Vh = _np.linalg.svd(mat)
    except _np.linalg.LinAlgError:
        _warnings.warn('SVD Calculation Failed to Converge. Falling back to Scipy' \
                        +' SVD with lapack driver gesvd, which is slower but *should* be more stable.')
        _, s, Vh = _sla.svd(mat, lapack_driver='gesvd')

    #How many non-zero eigenvalues are there and what are their indices
    nonzero_eigenvalue_indices= _np.nonzero(_np.abs(s)>threshold)

    #extract the corresponding columns and values fom U and s:
    #For EVD/eigh We want the columns of U and the rows of Uh:
    nonzero_e_values = s[nonzero_eigenvalue_indices]**2
    nonzero_U_columns = Vh.T[:, nonzero_eigenvalue_indices[0]]
    
    return nonzero_e_values, nonzero_U_columns    


#Function for generating an "update cache" of pre-computed matrices which will be
#reused during a sequence of many additive updates to the same base matrix.

def construct_update_cache(mat, evd_tol=1e-10):
    """
    Calculates the parts of the eigenvalue update loop algorithm that we can 
    pre-compute and reuse throughout all of the potential updates.
    
    Input:
    
    mat : ndarray
        The matrix to construct a set of reusable objects for performing the updates.
        mat is assumed to be a symmetric square matrix.
        
    evd_tol : float (optional)
        A threshold value for setting eigenvalues to zero.
        
    Output:
    
    U, e : ndarrays
        The components of the compact eigenvalue decomposition of mat
        such that U@diag(s)@U.conj().T= mat
        e in this case is a 1-D array of the non-zero eigenvalues.
    projU : ndarray
        A projector onto the complement of the column space of U
        Corresponds to (I-U@U.T)
    """
    
    #Start by constructing a compact EVD of the input matrix. 
    e, U = compact_EVD(mat, evd_tol)
    
    #construct the projector
    #I think the conjugation is superfluous when we have real
    #eigenvectors which in principle we should if using eigh
    #for the compact EVD calculation. 
    projU= _np.eye(mat.shape[0]) - U@U.T
    
    #I think that's all we can pre-compute, so return those values:
    
    #I don't actually need the value of U
    #Nope, that's wrong. I do for the construction of K.
    return e, U, projU
    

#Function that wraps up all of the work for performing the updates.
    
def symmetric_low_rank_spectrum_update(update, orig_e, U, proj_U, force_rank_increase=False):
    """
    This function performs a low-rank update to the spectrum of
    a matrix. It takes as input a symmetric update of the form:
    A@A.T, in other words a symmetric rank-decomposition of the update
    matrix. Since the update is symmetric we only pass as input one
    half (i.e. we only need A, since A.T in numpy is treated simply
    as a different view of A). We also pass in the original spectrum
    as well as a projector onto the complement of the column space
    of the original matrix's eigenvector matrix.
    
    input:
    
    update : ndarray
        symmetric low-rank update to perform.
        This is the first half the symmetric rank decomposition s.t.
        update@update.T= the full update matrix.
    
    orig_e : ndarray
        Spectrum of the original matrix. This is a 1-D array.
        
    proj_U : ndarray
        Projector onto the complement of the column space of the
        original matrix's eigenvectors.
        
    force_rank_increase : bool
        A flag to indicate whether we are looking to force a rank increase.
        If so, then after the rrqr calculation we can check the rank of the projection
        of the update onto the complement of the column space of the base matrix and
        abort early if that is zero.
    """
    
    #First we need to for the matrix P, whose column space
    #forms an orthonormal basis for the component of update
    #that is in the complement of U.
    proj_update= proj_U@update
    
    #Next take the RRQR decomposition of this matrix:
    q_update, r_update, _ = _sla.qr(proj_update, mode='economic', pivoting=True)
    
    #Construct P by taking the columns of q_update corresponding to non-zero values of r_A on the diagonal.
    nonzero_indices_update= _np.nonzero(_np.abs(_np.diag(r_update))>1e-10) #HARDCODED
    
    #print the rank of the orthogonal complement if it is zero.
    if len(nonzero_indices_update[0])==0:
        return None, False
    
    P= q_update[: , nonzero_indices_update[0]]
    
    #Now form the matrix R_update which is given by P.T @ proj_update.
    R_update= P.T@proj_update
    
    #R_update gets concatenated with U.T@update to form
    #a block column matrix
    block_column= _np.concatenate([U.T@update, R_update], axis=0)
    
    #We now need to construct the K matrix, which is given by
    #E+ block_column@block_column.T where E is a matrix with eigenvalues
    #on the diagonal with an appropriate number of zeros padded.
    
    #Instead of explicitly constructing the diagonal matrix of eigenvalues
    #I'll use einsum to construct a view of block_column@block_column.T's
    #diagonal and do an in-place sum directly to it.
    K= block_column@block_column.T
    
    #construct a view of the diagonal of K
    K_diag= _np.einsum('ii->i', K)
    
    #Get the dimension of K so we know how many zeros to pad the original eigenvalue
    #list with.
    K_diag+= _np.pad(orig_e, (0, (K.shape[0]-len(orig_e))) )
    
    #Since K_diag was a view of the original matrix K, this should have
    #modified the original K matrix in-place.
    
    #Now we need to get the spectrum of K, i.e. the spectrum of the 
    #updated matrices
    #I don't actually need the eigenvectors, so we don't need to output these
    new_evals= _np.linalg.eigvalsh(K)
    
    #return the new eigenvalues
    return new_evals, True
 
#Note: This function won't work for our purposes because of the assumptions
#about the rank of the update on the nullspace of the matrix we're updating,
#but keeping this here commented for future reference.
#Function for doing fast calculation of the updated inverse trace:
#def riedel_style_inverse_trace(update, orig_e, U, proj_U, force_rank_increase=True):
#    """
#    input:
#    
#    update : ndarray
#        symmetric low-rank update to perform.
#        This is the first half the symmetric rank decomposition s.t.
#        update@update.T= the full update matrix.
#    
#    orig_e : ndarray
#        Spectrum of the original matrix. This is a 1-D array.
#        
#    proj_U : ndarray
#        Projector onto the complement of the column space of the
#        original matrix's eigenvectors.
#        
#    output:
#    
#    trace : float
#        Value of the trace of the updated psuedoinverse matrix.
#    
#    updated_rank : int
#        total rank of the updated matrix.
#        
#    rank_increase_flag : bool
#        a flag that is returned to indicate is a candidate germ failed to amplify additional parameters. 
#        This indicates things short circuited and so the scoring function should skip this germ.
#    """
#    
#    #First we need to for the matrix P, whose column space
#    #forms an orthonormal basis for the component of update
#    #that is in the complement of U.
#    
#    proj_update= proj_U@update
#    
#    #Next take the RRQR decomposition of this matrix:
#    q_update, r_update, _ = _sla.qr(proj_update, mode='economic', pivoting=True)
#    
#    #Construct P by taking the columns of q_update corresponding to non-zero values of r_A on the diagonal.
#    nonzero_indices_update= _np.nonzero(_np.diag(r_update)>1e-10) #HARDCODED (threshold is hardcoded)
#    
#    #if the rank doesn't increase then we can't use the Riedel approach.
#    #Abort early and return a flag to indicate the rank did not increase.
#    if len(nonzero_indices_update[0])==0 and force_rank_increase:
#        return None, None, False
#    
#    P= q_update[: , nonzero_indices_update[0]]
#    
#    updated_rank= len(orig_e)+ len(nonzero_indices_update[0])
#    
#    #Now form the matrix R_update which is given by P.T @ proj_update.
#    R_update= P.T@proj_update
#    
#    #R_update gets concatenated with U.T@update to form
#    #a block column matrixblock_column= np.concatenate([U.T@update, R_update], axis=0)    
#    
#    Uta= U.T@update
#    
#    try:
#        RRRDinv= R_update@_np.linalg.inv(R_update.T@R_update) 
#    except _np.linalg.LinAlgError as err:
#        print('Numpy thinks this matrix is singular, condition number is: ', _np.linalg.cond(R_update.T@R_update))
#        print((R_update.T@R_update).shape)
#        raise err
#    pinv_orig_e_mat= _np.diag(1/orig_e)
#    
#    trace= _np.sum(1/orig_e) + _np.trace( RRRDinv@(_np.eye(Uta.shape[1]) + Uta.T@pinv_orig_e_mat@Uta)@RRRDinv.T )
#    
#    return trace, updated_rank, True
    
def minamide_style_inverse_trace(update, orig_e, U, proj_U, force_rank_increase=False):
    """
    This function performs a low-rank update to the components of
    the psuedo inverse of a matrix relevant to the calculation of that
    matrix's updated trace. It takes as input a symmetric update of the form:
    A@A.T, in other words a symmetric rank-decomposition of the update
    matrix. Since the update is symmetric we only pass as input one
    half (i.e. we only need A, since A.T in numpy is treated simply
    as a different view of A). We also pass in the original spectrum
    as well as a projector onto the complement of the column space
    of the original matrix's eigenvector matrix.
    
    Based on an update formula for psuedoinverses by minamide combined with
    a result on updating compact SVDs by M. Brand.
    
    input:
    
    update : ndarray
        symmetric low-rank update to perform.
        This is the first half the symmetric rank decomposition s.t.
        update@update.T= the full update matrix.
    
    orig_e : ndarray
        Spectrum of the original matrix. This is a 1-D array.
        
    proj_U : ndarray
        Projector onto the complement of the column space of the
        original matrix's eigenvectors.
        
    updated_trace : float
        Value of the trace of the updated psuedoinverse matrix.
    
    updated_rank : int
        total rank of the updated matrix.
        
    rank_increase_flag : bool
        a flag that is returned to indicate is a candidate germ failed to amplify additional parameters. 
        This indicates things short circuited and so the scoring function should skip this germ.
    """

    #First we need to for the matrix P, whose column space
    #forms an orthonormal basis for the component of update
    #that is in the complement of U.
    proj_update= proj_U@update
    
    #Next take the RRQR decomposition of this matrix:
    q_update, r_update, _ = _sla.qr(proj_update, mode='economic', pivoting=True)
    
    #Construct P by taking the columns of q_update corresponding to non-zero values of r_A on the diagonal.
    nonzero_indices_update= _np.nonzero(_np.abs(_np.diag(r_update))>1e-9) #HARDCODED
    
    #if the rank doesn't increase then we can't use the Riedel approach.
    #Abort early and return a flag to indicate the rank did not increase.
    if len(nonzero_indices_update[0])==0 and force_rank_increase:
        return None, None, False
    #We also need to add logic for the case where the projection onto the orthogonal
    #complement is empty, which reduces the psuedoinverse update such that we can
    #actually use the standard woodbury formula result.
    elif (len(nonzero_indices_update[0])==0) and (not force_rank_increase):
        #I have a bunch of intermediate matrices I need to construct. Some of which are used to build up
        #subsequent ones.
        beta= U.T@update
        #column vector of the original eigenvalues.
        orig_e_inv= _np.reshape(1/orig_e, (len(orig_e),1))
        orig_e_inv_sqrt = _np.sqrt(orig_e_inv)
        pinv_E_beta= orig_e_inv*beta
        pinv_sqrt_E_beta = orig_e_inv_sqrt*beta
        
        #Now apply the woodbury formula.
        #Identity matrix in the central matrix we're inverting should have dimension
        #equal to the number of columns in beta (or equivalently the update matrix)
        central_mat= _np.linalg.inv(_np.eye(beta.shape[1]) + pinv_sqrt_E_beta.T@pinv_sqrt_E_beta)
        
        #diagnostic information
        #if this prints something bad happened
        if central_mat.shape == (0,0):
            _warnings.warn(f'central_mat shape: {central_mat.shape}')
            
        #now calculate the diagonal elements of pinv_E_beta@central_mat@pinv_E_beta.T
        
        #Take the cholesky decomposition of the central matrix
        #The purpose of this cholesky decomposition is entirely to accelerate
        #the subsequent einsum call. Using einsum to calculate the diagonal elements
        #of a product of matrices of the form A@A.T is significantly faster than
        #doing so for a product of matrices of the form A@B@A.T
        
        try:
            central_mat_chol= _np.linalg.cholesky(central_mat)
            cholesky_success=True
        except _np.linalg.LinAlgError as err:
            #Cholesky decomposition probably failed.
            #I'm not sure why it failed though so print some diagnostic info:
            cholesky_success=False
            warnmsg = 'Cholesky Decomposition Probably Failed.' \
                      +' This may be due to a poorly conditioned original Jacobian.'\
                      +' Consider increasing the value of evd_tol, if relevant.'\
                      +' Here is some diagnostic info.'\
                      + f'Minimum original eigenvalue: {_np.min(orig_e)}'
            _warnings.warn(warnmsg)
        if cholesky_success:
            pinv_E_beta_central_chol= pinv_E_beta@central_mat_chol
            inv_update_term_diag= _np.einsum('ij,ji->i', pinv_E_beta_central_chol, pinv_E_beta_central_chol.T)
        else:
            inv_update_term_diag= _np.einsum('ij,jk,ki->i', pinv_E_beta, central_mat, pinv_E_beta.T, optimize=True)
        
        updated_trace= _np.sum(_np.reshape(orig_e_inv, (len(orig_e), ))- inv_update_term_diag)
        updated_rank= len(orig_e)
        rank_increased=False
    
    #otherwise apply the full minamide result to update the psuedoinverse.
    else:
        updated_rank= len(orig_e)+ len(nonzero_indices_update[0])
        P= q_update[: , nonzero_indices_update[0]]
        
        #Now form the matrix R_update which is given by P.T @ proj_update.
        R_update= P.T@proj_update
        
        #Get the psuedoinverse of R_update:
        try:
            pinv_R_update= _np.linalg.pinv(R_update, rcond=1e-10) #hardcoded
        except _np.linalg.LinAlgError:
            #This means the SVD did not converge, try to fall back to a more stable
            #SVD implementation using the scipy lapack_driver options.
            _warnings.warn('pinv Calculation Failed to Converge.'\
                           +'Falling back to pinv implementation based on Scipy SVD with lapack driver gesvd,'\
                           +' which is slower but *should* be more stable.')
            pinv_R_update = stable_pinv(R_update)
            
        #I have a bunch of intermediate matrices I need to construct. Some of which are used to build up
        #subsequent ones.
        beta= U.T@update
        gamma = pinv_R_update.T @ beta.T
        
        #column vector of the original eigenvalues.
        orig_e_inv= _np.reshape(1/orig_e, (len(orig_e),1))
        pinv_E_beta= orig_e_inv*beta
        B= _np.eye(pinv_R_update.shape[0]) - pinv_R_update @ R_update
        
        try:
            Dinv_chol= _np.linalg.cholesky(_np.linalg.inv(_np.eye(pinv_R_update.shape[0]) + B@(pinv_E_beta.T@pinv_E_beta)@B))
            cholesky_success=True
        except _np.linalg.LinAlgError as err:
            #Cholesky decomposition probably failed.
            #I'm not sure why it failed though so print some diagnostic info:
            #Is B symmetric or hermitian?
            cholesky_success=False
            #What are the eigenvalues of the Dinv matrix?
            _warnings.warn('Cholesky Decomposition Probably Failed. This may be due to a poorly conditioned original Jacobian.' \
                           +' Here is some diagnostic info.'\
                           +f' Dinv Condition Number: {_np.linalg.cond(_np.linalg.inv(_np.eye(pinv_R_update.shape[0]) + B@(pinv_E_beta.T@pinv_E_beta)@B))}'\
                           +f' Minimum original eigenvalue: {_np.min(orig_e)}'\
                           +' Falling back w/o use of Cholesky.')
            
        if cholesky_success:
            pinv_E_beta_B_Dinv_chol= pinv_E_beta@B@Dinv_chol
            
            #Now construct the two matrices we need:  
            #numpy einsum based approach for the upper left block:
            upper_left_block_diag = _np.einsum('ij,ji->i', pinv_E_beta_B_Dinv_chol, pinv_E_beta_B_Dinv_chol.T) + _np.reshape(orig_e_inv, (len(orig_e), ))

            #The lower right seems fast enough as it is for now, but we can try an einsum style direct diagonal
            #calculation if need be.
            lower_right_block= (gamma@(orig_e_inv*gamma.T))+ pinv_R_update.T@pinv_R_update - gamma@pinv_E_beta_B_Dinv_chol@pinv_E_beta_B_Dinv_chol.T@gamma.T
        
        else:
            #Since the cholesky decomposition failed go ahead and use an alternative calculation pipeline.
            Dinv= _np.linalg.inv(_np.eye(pinv_R_update.shape[0]) + B@(pinv_E_beta.T@pinv_E_beta)@B)
            pinv_E_beta_B= pinv_E_beta@B
            
            upper_left_block_diag = _np.einsum('ij,jk,ki->i', pinv_E_beta_B, Dinv, pinv_E_beta_B.T, optimize=True) + _np.reshape(orig_e_inv, (len(orig_e), ))
            #The lower right seems fast enough as it is for now, but we can try an einsum style direct diagonal
            #calculation if need be.
            lower_right_block= (gamma@(orig_e_inv*gamma.T))+ pinv_R_update.T@pinv_R_update - gamma@pinv_E_beta_B@Dinv@pinv_E_beta_B.T@gamma.T
        
        #the updated trace should just be the trace of these two matrices:
        updated_trace= _np.sum(upper_left_block_diag) + _np.trace(lower_right_block)
        rank_increased=True
        
    return updated_trace, updated_rank, rank_increased

    
#-------Modified Germ Selection Algorithm-------------------%

#This version of the algorithm adds support for using low-rank
#updates to speed the calculation of eigenvalues for additive
#updates.
    
def find_germs_breadthfirst_greedy(model_list, germs_list, randomize=True,
                            randomization_strength=1e-3, num_copies=None, seed=0,
                            op_penalty=0, score_func='all', tol=1e-6, threshold=1e6,
                            check=False, force="singletons", pretest=True, mem_limit=None,
                            comm=None, profiler=None, verbosity=0, num_nongauge_params=None,
                            float_type= _np.cdouble, 
                            mode="all-Jac", force_rank_increase=False,
                            save_cevd_cache_filename=None, load_cevd_cache_filename=None,
                            file_compression=False, evd_tol=1e-10, initial_germ_set_test=True):
    """
    Greedy algorithm starting with 0 germs.

    Tries to minimize the number of germs needed to achieve amplificational
    completeness (AC). Begins with 0 germs and adds the germ that increases the
    score used to check for AC by the largest amount (for the model that
    currently has the lowest score) at each step, stopping when the threshold
    for AC is achieved. This strategy is something of a "breadth-first"
    approach, in contrast to :func:`find_germs_depthfirst`, which only looks at the
    scores for one model at a time until that model achieves AC, then
    turning it's attention to the remaining models.

    Parameters
    ----------
    model_list : Model or list
        The model or list of Models to select germs for.

    germs_list : list of Circuit
        The list of germs to contruct a germ set from.

    randomize : bool, optional
        Whether or not to randomize `model_list` (usually just a single
        `Model`) with small (see `randomizationStrengh`) unitary maps
        in order to avoid "accidental" symmetries which could allow for
        fewer germs but *only* for that particular model.  Setting this
        to `True` will increase the run time by a factor equal to the
        numer of randomized copies (`num_copies`).

    randomization_strength : float, optional
        The strength of the unitary noise used to randomize input Model(s);
        is passed to :func:`~pygsti.objects.Model.randomize_with_unitary`.

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
        Tolerance (`eps` arg) for :func:`_compute_bulk_twirled_ddd`, which sets
        the differece between eigenvalues below which they're treated as
        degenerate.

    threshold : float, optional
        Value which the score (before penalties are applied) must be lower than
        for a germ set to be considered AC.

    check : bool, optional
        Whether to perform internal checks (will slow down run time
        substantially).

    force : str or list of Circuits, optional (default 'singletons')
        A list of `Circuit` objects which *must* be included in the final
        germ set.  If the special string "singletons" is given, then all of
        the single gates (length-1 sequences) must be included. If none then
        not circuits are forcibly included.

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

    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.
        
    float_type : numpy dtype object, optional
        Use an alternative data type for the values of the numpy arrays generated.
        
    force_rank_increase : bool, optional
        Whether to force the greedy iteration to select a new germ that increases the rank
        of the jacobian at each iteration (this may result in choosing a germ that is sub-optimal
        with respect to the chosen score function). Also results in pruning in subsequent
        optimization iterations. Defaults to False.
        
    evd_tol : float, optional
        A threshold value to use when taking eigenvalue decompositions/SVDs such that
        values below this are set to zero.
    
    initial_germ_set_test : bool, optional (default True)
        A flag indicating whether or not to check the initially constructed germ set, which
        is either the list of singleton germs (if force='singletons'), a user specified list of
        circuits is such a list if passed in for the value of force, or a greedily selected 
        initial germ if force=None. This can be skipped to save computational time (the test can
        be expensive) if the user has reason to believe this initial set won't be AC. Most of the time
        this initial set won't be.

    Returns
    -------
    list
        A list of the built-up germ set (a list of :class:`Circuit` objects).
    """
    if comm is not None and comm.Get_size() > 1:
        from mpi4py import MPI  # not at top so pygsti doesn't require mpi4py

    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)

    model_list = _setup_model_list(model_list, randomize,
                                   randomization_strength, num_copies, seed)

    dim = model_list[0].dim
    Np = model_list[0].num_params
    assert(all([(mdl.dim == dim) for mdl in model_list])), \
        "All models must have the same dimension!"
    #assert(all([(mdl.num_params == Np) for mdl in model_list])), \
    #    "All models must have the same number of parameters!"
    
    if (num_nongauge_params is None):
        (_, numGaugeParams,
         numNonGaugeParams, _) = _get_model_params(model_list)
        #TODO: This block doesn't make sense to me anymore. But I also
        #don't think this can be reached, so figure out what the intention here
        #was supposed to be another time.
        if num_nongauge_params is not None:
            numGaugeParams = numGaugeParams + numNonGaugeParams - num_nongauge_params
            numNonGaugeParams = num_nongauge_params
    elif (num_nongauge_params is not None):
        numGaugeParams =  Np - num_nongauge_params
        numNonGaugeParams = num_nongauge_params
    
    printer.log('Number of gauge parameters: ' + str(numGaugeParams), 1) 
    printer.log('Number of non-gauge parameters: ' + str(numNonGaugeParams), 1)

    #Add some logic to support germ selection on gate sets containing static gates without parameters
    #In this case we need to avoid germs consisting solely of the parameterless gates. I could catch this
    #later, but better to clean up the candidate list upfront (smaller search spaces are good).
    #This only works for ExplicitModels so skip otherwise
    if isinstance(model_list[0], _ExplicitOpModel):
        cleaned_germs_list = []
        for germ in germs_list:
            num_params_for_germ = 0
            for lbl in germ:
                if lbl in model_list[0].operations:
                    num_params_for_germ += model_list[0].operations[lbl].num_params
            if num_params_for_germ>0:
                cleaned_germs_list.append(germ)
        germs_list = cleaned_germs_list
        
    germLengths = _np.array([len(germ) for germ in germs_list], _np.int64)

    numGerms = len(germs_list)

    #We should do the memory estimates before the pretest:
    FLOATSIZE= float_type(0).itemsize

    memEstimatealljac = FLOATSIZE * len(model_list) * len(germs_list) * Np**2
    # for _compute_bulk_twirled_ddd
    memEstimatealljac += FLOATSIZE * len(model_list) * len(germs_list) * dim**2 * Np
    # for _bulk_twirled_deriv sub-call
    printer.log("Memory estimate of %.1f GB for all-Jac mode." %
                (memEstimatealljac / 1024.0**3), 1)            

    memEstimatesinglejac = FLOATSIZE * 3 * len(model_list) * Np**2 + \
        FLOATSIZE * 3 * len(model_list) * dim**2 * Np
    #Factor of 3 accounts for currentDDDs, testDDDs, and bestDDDs
    printer.log("Memory estimate of %.1f GB for single-Jac mode." %
                (memEstimatesinglejac / 1024.0**3), 1)            

    if mem_limit is not None:
        
        printer.log("Memory limit of %.1f GB specified." %
            (mem_limit / 1024.0**3), 1)
    
        if memEstimatesinglejac > mem_limit:
                raise MemoryError("Too little memory, even for single-Jac mode!")
    
        if mode=="all-Jac" and (memEstimatealljac > mem_limit):
            #fall back to single-Jac mode
            
            printer.log("Not enough memory for all-Jac mode, falling back to single-Jac mode.", 1)
            
            mode = "single-Jac"  # compute a single germ's jacobian at a time
            
    goodGerms = []
    weights = _np.zeros(numGerms, _np.int64)
    
    #Either add the specified forced germs to the initial list or identify the best initial to start off the search.
    if force == "singletons":
        weights[_np.where(germLengths == 1)] = 1
        goodGerms = [germ for i, germ in enumerate(germs_list) if germLengths[i] == 1]
        printer.log('Adding Singleton Germs By Default: '+ str(goodGerms) ,1)
    #if the value of force is a list then we'll assume it is a list of circuits.
    elif isinstance(force, list):
        for opstr in force:
            weights[germs_list.index(opstr)] = 1
        goodGerms = force[:]
        printer.log('Adding User-Specified Germs By Default: '+ str(goodGerms) ,1)
    elif force is None:
        pass
    else:
        raise ValueError('Unsupported argument. Force must either be a string, list or None.')    

    if pretest:
        printer.log("Performing pretest on complete candidate germ list to verify amplficational completeness", 1)
        undercompleteModelNum = test_germs_list_completeness(model_list,
                                                             germs_list,
                                                             score_func,
                                                             threshold,
                                                             float_type=float_type,
                                                             comm=comm,
                                                             num_gauge_params = numGaugeParams)
        if undercompleteModelNum > -1:
            printer.warning("Complete initial candidate germ set FAILS on model "
                            + str(undercompleteModelNum) + ".")
            printer.warning("Aborting search.")
            return None

        printer.log("Complete initial candidate germ set succeeds on all input models.", 1)
        printer.log("Now searching for best germ set.", 1)

    printer.log("Starting germ set optimization. Lower score is better.", 1)
    
    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    nonAC_kwargs = {
        'score_fn': lambda x: _scoring.list_score(x, score_func=score_func),
        'threshold_ac': threshold,
        'num_nongauge_params': numNonGaugeParams,
        'op_penalty': op_penalty,
        'float_type': float_type,
    }

    #For all-Jac and compactEVD build out the requisite caches (plus all-zeros placeholders for single-Jac):
    if mode == 'all-Jac':
        twirledDerivDaggerDerivList = \
            [_compute_bulk_twirled_ddd(model, germs_list, tol,
                                       check, germLengths, comm, float_type=float_type)
             for model in model_list]

        printer.log(f'Numpy Array Data Type: {twirledDerivDaggerDerivList[0].dtype}', 3)
        printer.log("Numpy array data type for twirled derivatives is: "+ str(twirledDerivDaggerDerivList[0].dtype)+
                    " If this isn't what you specified then something went wrong.",  3) 
                   
    elif mode == 'compactEVD':
        #implement a new caching scheme which takes advantage of the fact that the J^T J matrices are typically
        #rather low-rank. Instead of caching the J^T J matrices for each germ we'll cache the compact EVD of these
        #and multiply the compact EVD components through each time we need one.
        
        if load_cevd_cache_filename is not None:
            printer.log('Loading Compact EVD Cache From Disk',1)
            with _np.load(load_cevd_cache_filename) as cevd_cache:
                twirledDerivDaggerDerivList=[list(cevd_cache.values())]
            
        else:
            twirledDerivDaggerDerivList = \
                [_compute_bulk_twirled_ddd_compact(model, germs_list, tol,
                                                   evd_tol=evd_tol, float_type=float_type, printer=printer)
             for model in model_list]
             
            if save_cevd_cache_filename is not None:
                if len(twirledDerivDaggerDerivList)>1:
                    raise ValueError('Currently not configured to save compactEVD caches to disk when there is more than one model in the model list. i.e. this is not currently compatible with model randomization to get the non-lite germs.')
                #otherwise conver the first entry of twirledDerivDaggerDerivList,
                #which itself a list of a half of the symmetric rank decompositions
                #and save it to disk using _np.savez or _np.savez_compressed
                printer.log('Saving Compact EVD Cache to Disk', 1)
                if file_compression:
                    _np.savez_compressed(save_cevd_cache_filename,*twirledDerivDaggerDerivList[0])
                else:
                    _np.savez(save_cevd_cache_filename,*twirledDerivDaggerDerivList[0])
             #_compute_bulk_twirled_ddd_compact returns a tuple with two lists
             #corresponding to the U@diag(sqrt(2)), e  matrices for each germ's J^T J matrix's_list
             #compact evd.
    
    #if force is None then we need some logic to seeding the initial iteration of the search.
    if force is None:
        printer.log('Initializing greedy search. `force` is None so generating the germ set from scratch.',2)
        #I want to loop through all of the models and then for each model through all of the germs
        #I'll then pick the germ with the best performance, where best is defined as the least-bad across
        #all models. 
        initial_scores= [[_scoring.CompositeScore(-1.0e100, 0, None)]*len(model_list) for _ in range(len(germs_list))]
        for i in range(len(germs_list)):
            for j in range(len(model_list)):
                if mode=='all-Jac':                
                    #standard slicing squeezes the array losing the first index, which compute_composite_germ_set_score
                    #is expecting, so use a trick with integer array slicing to preserve this
                    derivDaggerDeriv = twirledDerivDaggerDerivList[j][[i],:,:]
                    initial_scores[i][j] = compute_composite_germ_set_score(
                                partial_deriv_dagger_deriv=derivDaggerDeriv, init_n=1, germ_lengths= [germLengths[i]],
                                **nonAC_kwargs)
                elif mode=='single-Jac':
                    derivDaggerDeriv = _compute_twirled_ddd(model_list[j], germs_list[i], tol, float_type=float_type)
                    initial_scores[i][j] = compute_composite_germ_set_score(
                                partial_deriv_dagger_deriv=derivDaggerDeriv[None,:,:], init_n=1, germ_lengths= [germLengths[i]],
                                **nonAC_kwargs)
                elif mode=='compactEVD':
                    initial_scores[i][j] = compute_composite_germ_set_score(
                            partial_deriv_dagger_deriv=(twirledDerivDaggerDerivList[j][i] @ twirledDerivDaggerDerivList[j][i].T)[None,:,:], 
                            init_n=1, germ_lengths= [germLengths[i]],
                            **nonAC_kwargs)
        #We should now have the composite scores for every germ and for every model. Now, for every germ we'll assign it's "best score"
        #to be the worst score (maximum) for that germ among the models.
        best_initial_scores= [_scoring.CompositeScore(-1.0e100, 0, None) for _ in range(len(germs_list))]
        for i, initial_score_list in enumerate(initial_scores):
            best_initial_scores[i]= max(initial_score_list)
        
        #finally find the minimum of these to select the best germ to initialize the list
        best_initial_germ_index= best_initial_scores.index(min(best_initial_scores))
        printer.log('Best initial germ found: ' + str(germs_list[best_initial_germ_index]), 2)
        
        #Set the weight of this germ to 1
        #and add it to the good germs list
        weights[best_initial_germ_index]=1
        goodGerms.append(germs_list[best_initial_germ_index])
    
    
    #Now that the initial germ list has been initialized (if necessary) use the initialized germ list to 
    #create the initial currentDDDList.
    if mode == "all-Jac":
        currentDDDList = []
        for i, derivDaggerDeriv in enumerate(twirledDerivDaggerDerivList):
            currentDDDList.append(_np.sum(derivDaggerDeriv[_np.where(weights == 1)[0], :, :], axis=0))

    elif mode == "single-Jac":
        currentDDDList = [_np.zeros((Np, Np), dtype=float_type) for mdl in model_list] 
        loc_Indices, _, _ = _mpit.distribute_indices(
            list(range(len(goodGerms))), comm, False)

        with printer.progress_logging(3):
            for i, goodGermIdx in enumerate(loc_Indices):
                printer.show_progress(i, len(loc_Indices),
                                      prefix="Initial germ set computation",
                                      suffix=germs_list[goodGermIdx].str)

                for k, model in enumerate(model_list):
                    currentDDDList[k] += _compute_twirled_ddd(
                        model, germs_list[goodGermIdx], tol, float_type=float_type)

        #aggregate each currendDDDList across all procs
        if comm is not None and comm.Get_size() > 1:
            for k, model in enumerate(model_list):
                result = _np.empty((Np, Np), dtype=float_type)
                comm.Allreduce(currentDDDList[k], result, op=MPI.SUM)
                currentDDDList[k][:, :] = result[:, :]
                result = None  # free mem
                
    elif mode== "compactEVD":
        currentDDDList = []
        nonzero_weight_indices= _np.nonzero(weights)
        nonzero_weight_indices= nonzero_weight_indices[0]
        for i, derivDaggerDeriv in enumerate(twirledDerivDaggerDerivList):
            #reconstruct the needed J^T J matrices
            for j, idx in enumerate(nonzero_weight_indices):
                if j==0:
                    temp_DDD = derivDaggerDeriv[idx] @ derivDaggerDeriv[idx].T
                else:
                    temp_DDD += derivDaggerDeriv[idx] @ derivDaggerDeriv[idx].T
            currentDDDList.append(temp_DDD)

    else:  # should be unreachable since we set 'mode' internally above
        raise ValueError("Invalid mode: %s" % mode)  # pragma: no cover

    #Add in a check for the initial germ list to see if we are already AC.
    #Do this by properly initializing initN to the current number of non-zero
    #eigenvalues, which will have the effect of short circuiting the while loop
    #below if we are AC.
    
    #Use test_germs_list_completeness to check if the initial germ list is already AC,
    #if so set initN equal to numNonGaugeParams in order to short circuit the greedy search
    #loop below and simply return the initial goodGerms list.
    if initial_germ_set_test:
        printer.log('Testing initial germ list for AC.', 2)
        initial_germ_set_completeness = test_germs_list_completeness(model_list, goodGerms,
                                                                     score_func, threshold,
                                                                     float_type=float_type,
                                                                     comm=comm)
        if initial_germ_set_completeness == -1:
            initN= numNonGaugeParams
            first_outer_iter_log= False
            printer.log('Initial germ list is AC, concluding search: ' + str(goodGerms), 2)
        else:
            #less than ideal, but it looks like initN needs to be initialized to 1 instead of None
            #as the placeholder if I don't want to dive into the score calculation code again to
            #fix things at a high-effort to low-reward ratio (I don't).
            initN=1
            first_outer_iter_log= True
            printer.log('Initial germ list is not AC, beginning greedy search loop.', 2)
    else:
        initN=None
        first_outer_iter_log= True
    
    while _np.any(weights == 0):
        if first_outer_iter_log:
            printer.log("Outer iteration: %d germs" %
                        (len(goodGerms)), 2)
            first_outer_iter=False
        else:
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
        
        if mode=="compactEVD":
            #calculate the update cache for each element of currentDDDList 
            printer.log('Creating update cache.')
            #TODO: I think I ought to be able to speed up the construction of the update
            #cache by adding some logic to leverage the same trick I use now in the
            #construction of the EVD cache, but that is a problem for another day.
            currentDDDList_update_cache = [construct_update_cache(currentDDD, evd_tol=evd_tol) for currentDDD in currentDDDList]
            #the return value of the update cache is a tuple with the elements
            #(e, U, projU)    
        with printer.progress_logging(2):
            for i, candidateGermIdx in enumerate(loc_candidateIndices):
                printer.show_progress(i, len(loc_candidateIndices),
                                      prefix="Inner iter over candidate germs",
                                      suffix=germs_list[candidateGermIdx].str)

                worstScore = _scoring.CompositeScore(-1.0e100, 0, None)  # worst of all models

                # Loop over all models
                testDDDs = []
                
                if mode == "all-Jac":
                    # Loop over all models
                    for k, currentDDD in enumerate(currentDDDList):
                        testDDD = currentDDD.copy()
                    
                        #just get cached value of deriv-dagger-deriv
                        derivDaggerDeriv = twirledDerivDaggerDerivList[k][candidateGermIdx]
                        testDDD += derivDaggerDeriv
                        
                        nonAC_kwargs['germ_lengths'] = \
                        _np.array([len(germ) for germ in
                                   (goodGerms + [germs_list[candidateGermIdx]])])
                        worstScore = max(worstScore, compute_composite_germ_set_score(
                                    partial_deriv_dagger_deriv=testDDD[None, :, :], init_n=initN,
                                    **nonAC_kwargs))
                        testDDDs.append(testDDD)  # save in case this is a keeper
                
                elif mode == "single-Jac":
                    # Loop over all models
                    for k, currentDDD in enumerate(currentDDDList):
                        testDDD = currentDDD.copy()
                        
                        #compute value of deriv-dagger-deriv
                        model = model_list[k]
                        testDDD += _compute_twirled_ddd(
                            model, germs_list[candidateGermIdx], tol, float_type=float_type)
                            
                        nonAC_kwargs['germ_lengths'] = \
                        _np.array([len(germ) for germ in
                                   (goodGerms + [germs_list[candidateGermIdx]])])
                        worstScore = max(worstScore, compute_composite_germ_set_score(
                                    partial_deriv_dagger_deriv=testDDD[None, :, :], init_n=initN,
                                    **nonAC_kwargs))
                        testDDDs.append(testDDD)  # save in case this is a keeper
                    
                elif mode == "compactEVD":
                    # Loop over all models
                    for k, update_cache in enumerate(currentDDDList_update_cache):
                    
                        nonAC_kwargs['germ_lengths'] = \
                            _np.array([len(germ) for germ in
                                       (goodGerms + [germs_list[candidateGermIdx]])])
                        nonAC_kwargs['num_params']=Np
                        nonAC_kwargs['force_rank_increase']= force_rank_increase
                        
                        
                        if score_func=="worst":
                            worstScore = max(worstScore, compute_composite_germ_set_score_compactevd(
                                                current_update_cache= update_cache,
                                                germ_update=twirledDerivDaggerDerivList[k][candidateGermIdx], 
                                                init_n=initN, **nonAC_kwargs))
                        elif score_func=="all":
                            worstScore = max(worstScore, compute_composite_germ_set_score_low_rank_trace(
                                                current_update_cache= update_cache,
                                                germ_update=twirledDerivDaggerDerivList[k][candidateGermIdx], 
                                                init_n=initN, **nonAC_kwargs))
                            
                # Take the score for the current germ to be its worst score
                # over all the models.
                germScore = worstScore
                printer.log(str(germScore), 4)
                if germScore < bestGermScore:
                    bestGermScore = germScore
                    iBestCandidateGerm = candidateGermIdx
                    
                    #If we are using the modes "all-Jac" or "single-Jac" then we will
                    #have been appending to testDDD throughout the process and can just set
                    #bestDDDs to testDDDs
                    if mode == "all-Jac" or mode == "single-Jac":
                        bestDDDs = testDDDs
                    
                    elif mode == "compactEVD":
                        #if compact EVD mode then we'll avoid reconstructing the J^T J matrix
                        #unless the germ is the current best.
                        bestDDDs= [currentDDD.copy() + \
                            twirledDerivDaggerDerivList[k][candidateGermIdx]@\
                            twirledDerivDaggerDerivList[k][candidateGermIdx].T\
                            for k, currentDDD in enumerate(currentDDDList)]
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
                        (germs_list[iBestCandidateGerm].str, str(bestGermScore)), 2)

    return goodGerms
    
def compute_composite_germ_set_score_compactevd(current_update_cache, germ_update, 
                                                score_fn="all", threshold_ac=1e6, init_n=1, model=None,
                                                 partial_germs_list=None, eps=None, num_germs=None,
                                                 op_penalty=0.0, l1_penalty=0.0, num_nongauge_params=None,
                                                 num_params=None, force_rank_increase=False,
                                                 germ_lengths=None, float_type=_np.cdouble):
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
    
    current_update_cache : tuple
        A tuple whose elements are the components of the current update cache
        for performing a low-rank update. Elements are (e, U , projU).
        
    germ_update : ndarray
        A numpy array corresponding to one half of the low-rank symmetric update to
        to perform.
    
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

    model : Model, optional
        The model against which the germ set is to be scored. Not needed if
        `partial_deriv_dagger_deriv` is provided.

    partial_germs_list : list of Circuit, optional
        The list of germs in the partial germ set to be evaluated. Not needed
        if `partial_deriv_dagger_deriv` (and `germ_lengths` when
        ``op_penalty > 0``) are provided.

    eps : float, optional
        Used when calculating `partial_deriv_dagger_deriv` to determine if two
        eigenvalues are equal (see :func:`_bulk_twirled_deriv` for details). Not
        used if `partial_deriv_dagger_deriv` is provided.

    op_penalty : float, optional
        Coefficient for a penalty linear in the sum of the germ lengths.

    germ_lengths : numpy.array, optional
        The length of each germ. Not needed if `op_penalty` is ``0.0`` or
        `partial_germs_list` is provided.

    l1_penalty : float, optional
        Coefficient for a penalty linear in the number of germs.

    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.
    
    num_params : int
        Total number of model parameters.
    
    force_rank_increase : bool, optional
        Whether to force the greedy iteration to select a new germ that increases the rank
        of the jacobian at each iteration (this may result in choosing a germ that is sub-optimal
        with respect to the chosen score function). Also results in pruning in subsequent
        optimization iterations. Defaults to False.
    
    
    Returns
    -------
    CompositeScore
        The score for the germ set indicating how many parameters it amplifies
        and its numerical score restricted to those parameters.
    """
    
    if germ_lengths is None:
        raise ValueError("Must provide either germ_lengths or "
                                 "partial_germs_list when op_penalty != 0.0!")
   
    if num_nongauge_params is None:
        if model is None:
            raise ValueError("Must provide either num_nongauge_params or model!")
        else:
            reduced_model = _remove_spam_vectors(model)
            num_nongauge_params = reduced_model.num_params - reduced_model.num_gauge_params

    # Calculate penalty scores
    if num_germs is not None:
        numGerms = num_germs
    else:
        numGerms= len(germ_lengths)
    l1Score = l1_penalty * numGerms
    opScore = 0.0
    if op_penalty != 0.0:
        opScore = op_penalty * _np.sum(germ_lengths)
    
    #calculate the updated eigenvalues
    updated_eigenvalues, rank_increase_flag = symmetric_low_rank_spectrum_update(germ_update, current_update_cache[0], current_update_cache[1], current_update_cache[2], force_rank_increase)
    
    N_AC = 0
    AC_score = _np.inf
    
    #check if the rank_increase_flag is set to False, if so then we failed
    #to increase the rank and so couldn't use the inverse trace update.
    if not rank_increase_flag:
        AC_score = -_np.inf
        N_AC = -_np.inf
    else:
        #I want compatibility eith the lines below that pick off just the non_gauge eigenvalues. Rather than
        #do some index gymnastics I'll just pad this eigenvalue list (which is compact) and make it the expected
        #length (num params). Pad on the left because the code below assumes eigenvalues in ascending order.
        padded_updated_eigenvalues= _np.pad(updated_eigenvalues, (num_params-len(updated_eigenvalues),0))

        #now pull out just the top num_nongauge_params eigenvalues
        observableEigenvals = padded_updated_eigenvalues[-num_nongauge_params:]
    
        for N in range(init_n, len(observableEigenvals) + 1):
            scoredEigenvals = observableEigenvals[-N:]
            candidate_AC_score = score_fn(scoredEigenvals)
            if candidate_AC_score > threshold_ac:
                break   # We've found a set of parameters for which the germ set
                # is not AC.
            else:
                AC_score = candidate_AC_score
                N_AC = N

    # Apply penalties to the major score
    major_score = -N_AC + opScore + l1Score
    minor_score = AC_score
    ret = _scoring.CompositeScore(major_score, minor_score, N_AC)

    return ret

def compute_composite_germ_set_score_low_rank_trace(current_update_cache, germ_update, 
                                                score_fn="all", threshold_ac=1e6, init_n=1, model=None,
                                                 partial_germs_list=None, eps=None, num_germs=None,
                                                 op_penalty=0.0, l1_penalty=0.0, num_nongauge_params=None,
                                                 num_params=None, force_rank_increase=False,
                                                 germ_lengths=None, float_type=_np.cdouble):
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
    
    current_update_cache : tuple
        A tuple whose elements are the components of the current update cache
        for performing a low-rank update. Elements are (e, U , projU).
        
    germ_update : ndarray
        A numpy array corresponding to one half of the low-rank symmetric update to
        to perform.
    
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

    model : Model, optional
        The model against which the germ set is to be scored. Not needed if
        `partial_deriv_dagger_deriv` is provided.

    partial_germs_list : list of Circuit, optional
        The list of germs in the partial germ set to be evaluated. Not needed
        if `partial_deriv_dagger_deriv` (and `germ_lengths` when
        ``op_penalty > 0``) are provided.

    eps : float, optional
        Used when calculating `partial_deriv_dagger_deriv` to determine if two
        eigenvalues are equal (see :func:`_bulk_twirled_deriv` for details). Not
        used if `partial_deriv_dagger_deriv` is provided.

    op_penalty : float, optional
        Coefficient for a penalty linear in the sum of the germ lengths.

    germ_lengths : numpy.array, optional
        The length of each germ. Not needed if `op_penalty` is ``0.0`` or
        `partial_germs_list` is provided.

    l1_penalty : float, optional
        Coefficient for a penalty linear in the number of germs.

    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.
    
    num_params : int
        Total number of model parameters.
    
    force_rank_increase : bool, optional
        Whether to force the greedy iteration to select a new germ that increases the rank
        of the jacobian at each iteration (this may result in choosing a germ that is sub-optimal
        with respect to the chosen score function). Also results in pruning in subsequent
        optimization iterations. Defaults to False.
    
    
    Returns
    -------
    CompositeScore
        The score for the germ set indicating how many parameters it amplifies
        and its numerical score restricted to those parameters.
    
    rank_increase_flag : bool
        A flag that indicates whether the candidate update germ increases the rank
        of the overall Jacobian.
    """
    
    if germ_lengths is None:
        raise ValueError("Must provide either germ_lengths or "
                                 "partial_germs_list when op_penalty != 0.0!")
   
    if num_nongauge_params is None:
        if model is None:
            raise ValueError("Must provide either num_gauge_params or model!")
        else:
            reduced_model = _remove_spam_vectors(model)
            num_nongauge_params = reduced_model.num_params - reduced_model.num_gauge_params

    # Calculate penalty scores
    if num_germs is not None:
        numGerms = num_germs
    else:
        numGerms= len(germ_lengths)
    l1Score = l1_penalty * numGerms
    opScore = 0.0
    if op_penalty != 0.0:
        opScore = op_penalty * _np.sum(germ_lengths)
    
    #calculate the updated eigenvalues
    inverse_trace, updated_rank, rank_increase_flag = minamide_style_inverse_trace(germ_update, current_update_cache[0], current_update_cache[1], current_update_cache[2], force_rank_increase)
    
    #check if the rank_increase_flag is set to False, if so then we failed
    #to increase the rank and so couldn't use the inverse trace update.
    if not rank_increase_flag:
        AC_score = -_np.inf
        N_AC = -_np.inf
    else:
        AC_score = inverse_trace
        N_AC = updated_rank
        
    # Apply penalties to the major score
    major_score = -N_AC + opScore + l1Score
    minor_score = AC_score
    ret = _scoring.CompositeScore(major_score, minor_score, N_AC)
    
    #TODO revisit what to do with the rank increase flag so that we can use
    #it to remove unneeded germs from the list of candidates.
    
    return ret#, rank_increase_flag

#Function for even faster kronecker products courtesy of stackexchange:
def fast_kron(a,b):
    #Don't really understand the numpy tricks going on here,
    #But this does appear to work correctly in testing and
    #it is indeed a decent amount faster, fwiw.
    return (a[:, None, :, None]*b[None, :, None, :]).reshape(a.shape[0]*b.shape[0],a.shape[1]*b.shape[1])
   
   
#Stabler implementation of the psuedoinverse using the alternative lapack driver for SVD:
def stable_pinv(mat):
    U, s, Vh = _sla.svd(mat, lapack_driver='gesvd', full_matrices=False)
    pinv_s= _np.zeros((len(s),1))
    for i, sval in enumerate(s):
        if sval>1e-10: #HARDCODED
            pinv_s[i]= 1/sval
    
    #new form the psuedoinverse:
    pinv= Vh.T@(pinv_s*U.T)
    return pinv


#---------- Minimal Germ Spanning Vectors------------#
#Not a great name for this section of code. Idea here is to
#take a user-inputed AC germ-set (presumably from a prior germ
#selection run and then identify a minimal set of amplified directions
#that spans model space collectively accross all germs.
#this can then be used as input to a version of FPR that is
#globally aware of the overlap in amplified directions
#of parameter space.

def germ_set_spanning_vectors(target_model, germ_list, assume_real=False, float_type=_np.cdouble, 
                              num_nongauge_params=None, tol = 1e-6, pretest=False, evd_tol = 1e-10,
                              verbosity=1, threshold = 1e6, mode = 'greedy', update_cache_low_rank = False,
                              final_test = True, comm=None): 
    """
    Parameters
    ----------
    target_model : Model or list of Model
        The model you are aiming to implement, or a list of models that are
        copies of the model you are trying to implement (either with or
        without random unitary perturbations applied to the models).
        
    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.
        
    float_type : numpy dtype object, optional
        Numpy data type to use for floating point arrays.
    
    tol : float, optional
        Tolerance (`eps` arg) for :func:`_compute_bulk_twirled_ddd`, which sets
        the differece between eigenvalues below which they're treated as
        degenerate.
    
    pretest : boolean, optional
        Whether germ list should be initially checked for completeness.
    
    evd_tol : float, optional
        A threshold value to use when taking eigenvalue decompositions/SVDs such that
        values below this are set to zero.
        
    verbosity : int, optional
        Level of detail printed to stdout.
        
    threshold : float, optional
        Value which the score (before penalties are applied) must be lower than
        for a germ set to be considered AC
        
    mode : string, optional (default 'greedy)'
        An optional mode string for specifying the search heuristic used for
        constructing the germ vector set. If 'greedy' we use a greedy search
        algorithm based on low-rank updates. If 'RRQR' we use a heuristic for
        the subset selection problem due to Golub, Klema and Stewart (1976),
        detailed in chapter 12 of the book "Matrix Computations" by Golub and
        Van Loan. 
        
    update_cache_low_rank : bool, optional (default = False)
        A flag indicating whether the psuedoinverses in the update cache used 
        as part of the low-rank update routine should themselves be updated
        between iterations using low-rank updates. Setting this to True
        gives a notable performance boost, but I have found that this can
        also give rise to numerical stability issues, so caveat emptor.
        Only set to True if you're sure of what you're doing.
        
    final_test : bool, optional (default True)
        A flag indicating whether a final test should be performed to validate
        the final score of the germ-vector set found using either the greedy
        or RRQR based search heuristics. Can be useful in conjunction with
        the use of update_cache_low_rank, as this can help detect numerical
        stability problems in the use of low-rank updates for cache construction.
        
    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.
        
    Returns
    -------
    
    germ_vec_dict : dict
        A dictionary whose keys are germs and whose elements are numpy arrays
        whose columns correspond to the amplified directions in parameter space
        identified for that germ such that when combined with the amplified directions
        selected for each other germ in the set we maintain amplificational completeness.

    currentDDD : ndarray
        The J^T@J matrix for subset of twirled derivative columns selected accross
        all of the germs. The spectrum of this matrix provides information about the
        amplificational properties of the reduced vector set. 
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)
    
    #Add some checks related to the option to switch up data types:
    if not assume_real:
        if not (float_type is _np.cdouble or float_type is _np.csingle):
            printer.log('Selected numpy type: '+ str(float_type.dtype), 1)
            raise ValueError('Unless working with (known) real-valued quantities only, please select an appropriate complex numpy dtype (either cdouble or csingle).')
    else:
        if not (float_type is _np.double or float_type is _np.single):
            printer.log('Selected numpy type: '+ str(float_type.dtype), 1)
            raise ValueError('When assuming real-valued quantities, please select a real-values numpy dtype (either double or single).')
    
    Np = target_model.num_params
    
    if (num_nongauge_params is None):
        (_, numGaugeParams,
         numNonGaugeParams, _) = _get_model_params([target_model])
        if num_nongauge_params is not None:
            numGaugeParams = numGaugeParams + numNonGaugeParams - num_nongauge_params
            numNonGaugeParams = num_nongauge_params
    elif (num_nongauge_params is not None):
        numGaugeParams =  Np - num_nongauge_params
        numNonGaugeParams = num_nongauge_params
    
    printer.log('Number of gauge parameters: ' + str(numGaugeParams), 1) 
    printer.log('Number of non-gauge parameters: ' + str(numNonGaugeParams), 1)
    
    
    if pretest:
        undercompleteModelNum = test_germs_list_completeness([target_model],
                                                             germ_list, 
                                                             'all',
                                                             threshold,
                                                             float_type=float_type,
                                                             comm=comm)
        if undercompleteModelNum > -1:
            printer.warning("Complete initial germ set FAILS on model "
                            + str(undercompleteModelNum) + ".")
            printer.warning("Aborting search.")
            return None

        printer.log("Complete initial germ set succeeds on all input models.", 1)
        
    twirledDerivDaggerDerivList, germ_eigval_list = _compute_bulk_twirled_ddd_compact(target_model, germ_list, tol, evd_tol=evd_tol, float_type=float_type, printer=printer, return_eigs=True)
    
    #_compute_bulk_twirled_ddd returns a list of matrices of the form U@np.diag(np.sqrt(e)) where U@diag(e)@U^H is the compact eigenvalue decomposition of a matrix. It is done this way to ensure we have symmetric updates.
    
    #Let's take the list of compactified U@np.diag(np.sqrt(e)) matrices, which are num_params x rank of twirled derivative in dimensions and concatenate them into a single array
    printer.log('Concatenating compact EVDs', 2)
    composite_twirled_deriv_array = _np.concatenate(twirledDerivDaggerDerivList, axis =1)
    
    printer.log('Complete germ set (overcomplete) number of amplified parameters: %d'%(composite_twirled_deriv_array.shape[1]), 1)
    
    #also do this for the eigenvalues:
    composite_eigenvalue_array = _np.concatenate(germ_eigval_list, axis=None)
    
    #I am going to need to easily map back from indexes in the above 2 composite arrays and 
    #the corresponding germ, internal germ index pair.
    idx_to_germ_idx = [(germ, internal_idx) for germ_idx, germ in enumerate(germ_list) 
                                            for internal_idx in range(len(germ_eigval_list[germ_idx]))]
    
    #Initialize a dictionary tracking for each germs the directions in model parameter
    #space being kept for that germ.
    germ_vec_dict = {germ : [] for germ in germ_list}
    
    if mode == 'greedy':
    
        num_candidate_vecs= composite_twirled_deriv_array.shape[1]

        #Named weights for historical reasons, this is a binary vector of length equal to the total number
        #of candidate eigenvectors for including in the J^T@J matrix.  
        weights = _np.zeros(num_candidate_vecs, _np.int64)
        
        #Ok, now let's do the initial iteration of the greedy search and then use this to build off of using low-rank
        #rank-update magic.
        
        #for the initial iteration since we're just adding a single vector, independent of the score function used
        #we basically just want to choose the vector associated with the largest eigenvalue (as this will minimize both
        #the psuedoinverse-trace and the minimum psuedoinverse eigenvalue conditions).
        best_initial_vec_index= _np.argmax(composite_eigenvalue_array)
        printer.log('Best initial vector found: ' + str(best_initial_vec_index), 2)
        
        #Set the weight of this vector to 1
        weights[best_initial_vec_index]=1
        
        #and add it to a dictionary tracking for each germs the directions in model parameter
        #space being kept for that germ.
        germ_vec_dict[idx_to_germ_idx[best_initial_vec_index][0]].append(composite_twirled_deriv_array[:, [best_initial_vec_index]]/_np.sqrt(composite_eigenvalue_array[best_initial_vec_index]))
        #Note: I want the elements of the germ vector dict to correspond to normalized vectors in model
        #parameter space, but the vectors corresponding to the columns of composite_twirled_deriv_array
        #have a factor of sqrt(e) folded into them, so I am dividing that back out here.
        
        #initial value of the current twirled derivative gramian.
        currentDDD = composite_twirled_deriv_array[:, [best_initial_vec_index]]@ composite_twirled_deriv_array[:, [best_initial_vec_index]].T
        
        #Now start the greedy search. The initial number of amplified parameters is 1.
        initN=1
        #initialize a variable for the previous iteration's update cache to None
        prev_update_cache = None
        
        while _np.any(weights == 0):
            printer.log("Outer iteration: %d of %d amplified" %
                            (initN, numNonGaugeParams), 2)
                            
            if initN == numNonGaugeParams:
                break   # We are AC, so we can stop adding model parameter directions.
                    
            #precompute things that can be reused for the rank-one update
            printer.log('Creating update cache.', 3)
            if prev_update_cache is None:
                current_update_cache = construct_update_cache_rank_one(currentDDD, evd_tol=evd_tol)
            else:
                if update_cache_low_rank == True:
                    #do a rank one psuedoinverse update wrt the best vector from the prior round
                    current_update_cache = construct_update_cache_rank_one(currentDDD, evd_tol=evd_tol, 
                                                                           prev_update_cache = prev_update_cache,
                                                                           rank_one_update=composite_twirled_deriv_array[:, [idx_best_candidate_vec]]) # noqa: F821
                else:
                    #otherwise rebuild the update cache from scratch using a fresh psuedoinverse. Could be useful if worried about stability.
                    current_update_cache = construct_update_cache_rank_one(currentDDD, evd_tol=evd_tol)
            
            candidate_vec_indices = _np.where(weights == 0)[0]
            
            best_vec_score = _scoring.CompositeScore(1.0e100, 0, None)  # lower is better
            idx_best_candidate_vec = None
            
            for i,idx in enumerate(candidate_vec_indices):
                printer.log('Inner iter over candidate vectors, %d of %d'%(i, len(candidate_vec_indices)), 3)
                                      
                current_vec_score = compute_composite_vector_set_score(
                                                current_update_cache= current_update_cache,
                                                vector_update= composite_twirled_deriv_array[:, [idx]],
                                                num_nongauge_params= numNonGaugeParams,
                                                float_type= float_type)

                if current_vec_score < best_vec_score:
                    best_vec_score = current_vec_score
                    idx_best_candidate_vec = idx
                    
            #update the weight vector:
            weights[idx_best_candidate_vec]=1
            
            #update initN for the next round, this should just be the N value for the best_vec_score
            initN= best_vec_score.N
            
            #logging for the best found score at this iteration:
            printer.log('Best score this iteration: ' + str(best_vec_score), 2)
            
            #update currentDDD
            currentDDD= composite_twirled_deriv_array[:, _np.where(weights == 1)[0]]@ composite_twirled_deriv_array[:, _np.where(weights == 1)[0]].T
        
            #Add this vector to the germ vector dictionary
            germ_vec_dict[idx_to_germ_idx[idx_best_candidate_vec][0]].append(composite_twirled_deriv_array[:, [idx_best_candidate_vec]]/_np.sqrt(composite_eigenvalue_array[idx_best_candidate_vec]))
            
            #set the previous update cache to the current one in preparation for the next round through the loop.
            prev_update_cache = current_update_cache
            
        printer.log('Returning best found vector set. Final Score: ' + str(best_vec_score),1)
        
        #if true, perform a final test and verify that the final score 
        if final_test:
            #restricted version of the psuedoinverse trace that only looks at the non-gauge parameters.
            #Only care about gramians, so assume hermitian
            def restricted_pinv_trace(mat, num_nongauge):
                #flip so in descending order
                evals = _np.flip(_np.linalg.eigvalsh(mat), axis=0)
                restricted_evals = evals[0:num_nongauge]
                pinv_evals = 1/restricted_evals
                rank = _np.count_nonzero(restricted_evals > 1e-7) #HARDCODED
                
                return _np.sum(pinv_evals), rank
            
            #also validate the rank:    
            final_test_pinv, final_test_rank = restricted_pinv_trace(currentDDD, numNonGaugeParams)
            
            #compare these to the results of the greedy search and raise and error if they are significantly different.
            #Hardcoded tolerance is primarily meant to detect catastrophic failures, hence it being pretty large.
            if (abs(final_test_pinv - best_vec_score.minor) > 1) or (final_test_rank != best_vec_score.N): #HARDCODED
                raise ValueError(f'Final test failed. Either the psuedoinverse traces are different or the final ranks are different. \n'
                                 + f'The final psuedoinverse-trace from the test is: {final_test_pinv} and the final rank from the test is: {final_test_rank} \n'
                                 + f'The final psuedoinverse-trace from the greedy search is: {best_vec_score.minor} and the final rank from the greedy search is: {best_vec_score.N}')
            
    #add in a new code path for using the SVD/RRQR based heuristic.
    elif mode == 'RRQR':
        #Start by calculating the singular value decomposition of the composite_twirled_deriv_array
        #Only need the V matrix.
        _, _, Vh = _np.linalg.svd(composite_twirled_deriv_array, full_matrices=False, compute_uv=True, hermitian=False)
       
        #target rank = numNonGaugeParams
        #Take the rank revealing QR decomposition of the submatrix of Vh corresponding
        #to the first numNonGaugeParams rows.
        #I actually only need the column permutation as well.
        _, P = _sla.qr(Vh[0:numNonGaugeParams, :], mode='r', pivoting=True)
        
        #We should be able to use the integer array returned by the RRQR to
        #index directly into the composite_twirled_deriv_array
        permuted_composite_twirled_deriv_array = composite_twirled_deriv_array[:, P]
        #As for the subset to take, we want the first numNonGaugeParams columns
        #of the permuted array
        selected_vector_subset = permuted_composite_twirled_deriv_array[:, 0:numNonGaugeParams]
        
        #Add the vectors to the germ vector dictionary
        for vec_idx in P[0:numNonGaugeParams]:
            germ_vec_dict[idx_to_germ_idx[vec_idx][0]].append(composite_twirled_deriv_array[:, [vec_idx]]/_np.sqrt(composite_eigenvalue_array[vec_idx]))
        
        #temporarily copy, fix the return behavior later to avoid this.
        currentDDD= selected_vector_subset @ selected_vector_subset.T
        #Need to map back the selected vectors indices (which we should be able to pull
        #directly by taking the first numNonGaugeParams of P) and use them to map back into
        #the germ set for germ_vec_dict construction purposes.
        
    else:
        raise NotImplementedError('The specified mode string ' + mode + ' is not currently implemented. Please use either greedy or RRQR.')    

    return germ_vec_dict, currentDDD

#Updated composite score calculating function specialized to 
#handle the case where we're assembling a composite set of
#model vectors rather than full germs.
def compute_composite_vector_set_score(current_update_cache, vector_update, 
                                       model=None, num_nongauge_params=None, 
                                       force_rank_increase=False, 
                                       float_type=_np.cdouble):
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
    
    current_update_cache : tuple
        A tuple whose elements are the components of the current update cache
        for performing a low-rank update. Elements are (pinv(A), proj_A).
        
    vector_update : ndarray
        A numpy array corresponding to one half of the low-rank symmetric update to
        to perform.
    
    model : Model, optional
        The model against which the germ set is to be scored. Not needed if
        `partial_deriv_dagger_deriv` is provided.

    num_nongauge_params : int, optional
        Force the number of nongauge parameters rather than rely on automated gauge optimization.
    
    force_rank_increase : bool, optional
        Whether to force the greedy iteration to select a new germ that increases the rank
        of the jacobian at each iteration (this may result in choosing a germ that is sub-optimal
        with respect to the chosen score function). Also results in pruning in subsequent
        optimization iterations. Defaults to False.
    
    
    Returns
    -------
    CompositeScore
        The score for the germ set indicating how many parameters it amplifies
        and its numerical score restricted to those parameters.
    
    rank_increase_flag : bool
        A flag that indicates whether the candidate update germ increases the rank
        of the overall Jacobian.
    """
    
    if num_nongauge_params is None:
        if model is None:
            raise ValueError("Must provide either num_gauge_params or model!")
        else:
            reduced_model = _remove_spam_vectors(model)
            num_nongauge_params = reduced_model.num_params - reduced_model.num_gauge_params
    
    #calculate the updated eigenvalues
    inverse_trace, rank_increase_flag = rank_one_inverse_trace_update(vector_update, current_update_cache[0], current_update_cache[1], current_update_cache[2], force_rank_increase)
    
    #check if the rank_increase_flag is set to False, if so then we failed
    #to increase the rank and so couldn't use the inverse trace update.
    if not rank_increase_flag and force_rank_increase:
        AC_score = -_np.inf
        N_AC = -_np.inf
    else:
        AC_score = inverse_trace
        #current_update_cache[3] is the current rank of A, so if the rank increase flag is set we increment this by 1.
        if rank_increase_flag:
            N_AC = current_update_cache[3]+1
        else:
            N_AC= current_update_cache[3]
        
    # Apply penalties to the major score
    major_score = -N_AC
    minor_score = AC_score
    ret = _scoring.CompositeScore(major_score, minor_score, N_AC)
    
    #TODO revisit what to do with the rank increase flag so that we can use
    #it to remove unneeded germs from the list of candidates.
    
    return ret#, rank_increase_flag
    
#version specialized for rank one updates
def construct_update_cache_rank_one(mat, evd_tol=1e-10, prev_update_cache=None, rank_one_update=None):
    """
    Calculates the parts of the psuedoinverse update loop algorithm that we can 
    pre-compute and reuse throughout all of the potential updates.
    
    This is based on a result from Carl Meyer in Generalized Inversion of 
    Modified Matrices, and summarized in 3.2.7 of the matrix cookbook.
    
    quantities we can pre-compute are (for initial matrix A):
    pinv(A)
    I-A@pinv(A)
    
    Input:
    
    mat : ndarray
        The matrix to construct a set of reusable objects for performing the updates.
        mat is assumed to be a symmetric square matrix.
        
    evd_tol : float (optional)
        A threshold value for setting eigenvalues to zero.
        
    Output:
    
    pinv_A : ndarray
        The psuedoinverse of the input matrix
    
    proj_A : ndarray
        A projectors onto the orthogonal complement of the column space of the input matrix.
    
    pinv_A_trace : float
        The trace of pinv_A.
        
    rank : int
        The current rank of A/pinv_A
        
    """
    
    #Start by constructing the psuedoinverse of the input matrix.
    
    #if these are both specified then use a rank-one update for the psuedoinversion.
    if ((prev_update_cache is not None) and (rank_one_update is  not None)):
        pinv_A, rank_increase_flag = rank_one_psuedoinverse_update(rank_one_update, prev_update_cache[0], 
                                                                   prev_update_cache[1], prev_update_cache[2])
        if rank_increase_flag:
            rank= prev_update_cache[3]+1
        else:
            rank=prev_update_cache[3]
        
    #Else construct the psuedoinverse from scratch.
    #Use the scipy implementation since I can get the rank out easily
    else:
        try:
            pinv_A, rank= _sla.pinvh(mat, return_rank=True) #hardcoded
        except _np.linalg.LinAlgError:
            #This means the SVD did not converge, try to fall back to a more stable
            #SVD implementation using the scipy lapack_driver options.
            _warnings.warn('pinv Calculation Failed to Converge.'\
                           +'Falling back to pinv implementation based on Scipy SVD with lapack driver gesvd,'\
                           +' which is slower but *should* be more stable.')
            pinv_A = stable_pinv(mat)
    
    #construct the projector
    proj_A= _np.eye(mat.shape[0]) - mat@pinv_A
    
    #I think that's all we can pre-compute, so return those values:
    return pinv_A, proj_A, _np.trace(pinv_A), rank
    
#function for doing rank-1 psuedoinverse-trace update:
def rank_one_inverse_trace_update(vector_update, pinv_A, proj_A, pinv_A_trace, force_rank_increase=False):
    """
        Helper function for calculating rank-one updates to the trace of the psuedoinverse.
        Takes as input a rank-one update, the psuedo-inverse of the matrix
        we're updating, the projector onto the column space for the matrix whose
        psuedoinverse we are updating and a flag for specifying if we're requiring
        the rank to increase.
    """
    #calculate some quantities we need. Following notation from matrix cookbook.
    v = pinv_A@vector_update
    beta = 1 + _np.sum(vector_update*v)
    w = proj_A@vector_update
    
    #the conditions are based on beta and the 2-norm of w.
    norm_w = _np.linalg.norm(w)
    
    #Note: we only actually need to calculate the diagonal elements of the G matrix. 
    if norm_w > 1e-10: #HARDCODED, need some wiggle room for numerical precision reasons.
        #print('Case 1')
        #the diagonal of an outer-product of 2 vectors is just a vector of the element-wise
        #products of corresponding elements.
        vw_term_diag = (-2/norm_w**2)*(v*w)
        ww_term_diag = (beta/norm_w**4)*(w**2)
        
        G_diag = vw_term_diag + ww_term_diag
        
        #if the norm of w is non-zero this means our update has non-trivial
        #support on the orthogonal complement to the column space of A,
        #so our rank must increase by 1.
        rank_increase_flag = True
    
    elif (norm_w<1e-10) and (beta>1e-10):
        #print('Case 3/5')
        G_diag = (-beta/_np.abs(beta)**2)*(v**2)
        
        rank_increase_flag = False
        
    elif (norm_w<1e-10) and (beta<1e-10):
        #The only circumstance I can think of where we'll hit this is if
        #the update is a -1 eigenstate of pinv_A. I have no intuition for
        #whether we'll encounter this in practice.
        _warnings.warn('Encountered Case 6 of the psuedoinverse update from the matrix cookbook,'\
                       +' which probably should not be possible (AKA look into this).')
        gamma = pinv_A@v
        norm_v= _np.linalg.norm(v)
        
        gamma_v_term_diag = (-2/norm_v**2)*(gamma*v)
        vv_term_diag = (_np.sum(v*gamma)/norm_v**4)*(v**2)
        
        G_diag = gamma_v_term_diag + vv_term_diag
        
        rank_increase_flag = False
    
    else:
        raise ValueError('Some weird shite went down here, none of the cases match.')

    #Ok, now we just take the sum of this vector and we have the trace.
    updated_trace = _np.sum(G_diag) + pinv_A_trace
    
    return updated_trace, rank_increase_flag
    
    
#function for doing rank-1 psuedoinverse update:
def rank_one_psuedoinverse_update(vector_update, pinv_A, proj_A, force_rank_increase=False):
    """
    Helper function for calculating rank-one psuedoinverse updates.
    Takes as input a rank-one update, the psuedo-inverse of the matrix
    we're updating, the projector onto the column space for the matrix whose
    psuedoinverse we are updating and a flag for specifying if we're requiring
    the rank to increase.
    """
    
    #calculate some quantities we need. Following notation from matrix cookbook.
    beta = 1 + vector_update.T@pinv_A@vector_update
    v= pinv_A@vector_update
    w = proj_A@vector_update
    
    #the conditions are based on beta and the 2-norm of w.
    norm_w = _np.linalg.norm(w)
    
    #Note: we only actually need to calculate the diagonal elements of the G matrix. 
    if norm_w > 1e-10: #HARDCODED, need some wiggle room for numerical precision reasons.
        #print('Case 1')
        #the diagonal of an outer-product of 2 vectors is just a vector of the element-wise
        #products of corresponding elements.
        vw = v@w.T
        vw_term = (-1/norm_w**2)*(vw + vw.T)
        ww_term = (beta/norm_w**4)*(w@w.T)
        
        G = vw_term + ww_term
        
        #if the norm of w is non-zero this means our update has non-trivial
        #support on the orthogonal complement to the column space of A,
        #so our rank must increase by 1.
        rank_increase_flag = True
    
    elif (norm_w<1e-10) and (beta>1e-10):
        #print('Case 3/5')
        G = (-beta/_np.abs(beta)**2)*(v@v.T)
        
        rank_increase_flag = False
        
    elif (norm_w<1e-10) and (beta<1e-10):
        #The only circumstance I can think of where we'll hit this is if
        #the update is a -1 eigenstate of pinv_A. I have no intuition for
        #whether we'll encounter this in practice.
        _warnings.warn('Encountered Case 6 of the psuedoinverse update from the matrix cookbook,'\
                       +' which probably should not be possible (AKA look into this).')
        gamma = pinv_A@v
        norm_v= _np.linalg.norm(v)
        
        gamma_v= gamma@v.T
        gamma_v_term = (-1/norm_v**2)*(gamma_v+gamma_v.T)
        vv_term = (_np.sum(v*gamma)/norm_v**4)*(v@v.T)
        
        G = gamma_v_term + vv_term
        
        rank_increase_flag = False
    
    else:
        raise ValueError('Some weird shite went down here, none of the cases match.')

    #Ok, now we just take the sum of this vector and we have the trace.
    updated_pinv = G + pinv_A
    
    return updated_pinv, rank_increase_flag
    

                                                   
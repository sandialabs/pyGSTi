"""
Functions for generating bootstrapped error bars
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
#import matplotlib as _mpl #REMOVED
from . import longsequence as _longseq
from .. import objects as _obj
from .. import algorithms as _alg
from .. import tools as _tools


def create_bootstrap_dataset(input_data_set, generation_method, input_model=None,
                           seed=None, outcome_labels=None, verbosity=1):
    """
    Creates a DataSet used for generating bootstrapped error bars.

    Parameters
    ----------
    input_data_set : DataSet
        The data set to use for generating the "bootstrapped" data set.

    generation_method : { 'nonparametric', 'parametric' }
        The type of dataset to generate.  'parametric' generates a DataSet
        with the same circuits and sample counts as input_data_set but
        using the probabilities in input_model (which must be provided).
        'nonparametric' generates a DataSet with the same circuits
        and sample counts as input_data_set using the count frequencies of
        input_data_set as probabilities.

    input_model : Model, optional
        The model used to compute the probabilities for circuits when
        generation_method is set to 'parametric'.  If 'nonparametric' is selected,
        this argument must be set to None (the default).

    seed : int, optional
        A seed value for numpy's random number generator.

    outcome_labels : list, optional
        The list of outcome labels to include in the output dataset.  If None
        are specified, defaults to the spam labels of input_data_set.

    verbosity : int, optional
        How verbose the function output is.  If 0, then printing is suppressed.
        If 1 (or greater), then printing is not suppressed.

    Returns
    -------
    DataSet
    """
    if generation_method not in ['nonparametric', 'parametric']:
        raise ValueError("generation_method must be 'parametric' or 'nonparametric'!")
    if outcome_labels is None:
        outcome_labels = input_data_set.outcome_labels()

    rndm = seed if isinstance(seed, _np.random.RandomState) \
        else _np.random.RandomState(seed)

    if input_model is None:
        if generation_method == 'nonparametric':
            print("Generating non-parametric dataset.")
        elif generation_method == 'parametric':
            raise ValueError("For 'parmametric', must specify input_model")
    else:
        if generation_method == 'parametric':
            print("Generating parametric dataset.")
        elif generation_method == 'nonparametric':
            raise ValueError("For 'nonparametric', input_model must be None")
        firstPOVMLbl = list(input_model.povms.keys())[0]
        # TODO: allow outcomes from multiple POVMS? (now just consider *first* POVM)
        possibleOutcomeLabels = [(eLbl,) for eLbl in input_model.povms[firstPOVMLbl].keys()]
        assert(all([ol in possibleOutcomeLabels for ol in outcome_labels]))

    possibleOutcomeLabels = input_data_set.outcome_labels()
    assert(all([ol in possibleOutcomeLabels for ol in outcome_labels]))

    #create new dataset
    simDS = _obj.DataSet(outcome_labels=outcome_labels,
                         collision_action=input_data_set.collisionAction)
    circuit_list = list(input_data_set.keys())
    for s in circuit_list:
        nSamples = input_data_set[s].total
        if generation_method == 'parametric':
            ps = input_model.probabilites(s)
        elif generation_method == 'nonparametric':
            ps = {ol: input_data_set[s].fraction(ol) for ol in outcome_labels}
        pList = _np.array([_np.clip(ps[outcomeLabel], 0, 1) for outcomeLabel in outcome_labels])
        #Truncate before normalization; bad extremal values shouldn't
        # screw up not-bad values, yes?
        pList = pList / sum(pList)
        countsArray = rndm.multinomial(nSamples, pList, 1)
        counts = {ol: countsArray[0, i] for i, ol in enumerate(outcome_labels)}
        simDS.add_count_dict(s, counts)
    simDS.done_adding_data()
    return simDS


def create_bootstrap_models(num_models, input_data_set, generation_method,
                          fiducial_prep, fiducial_measure, germs, max_lengths,
                          input_model=None, target_model=None, start_seed=0,
                          outcome_labels=None, lsgst_lists=None,
                          return_data=False, verbosity=2):
    """
    Creates a series of "bootstrapped" Models.

    Models are created from a single DataSet (and possibly Model) and are
    typically used for generating bootstrapped error bars.  The resulting Models
    are obtained by performing MLGST on datasets generated by repeatedly calling
    :function:`create_bootstrap_dataset` with consecutive integer seed values.

    Parameters
    ----------
    num_models : int
        The number of models to create.

    input_data_set : DataSet
        The data set to use for generating the "bootstrapped" data set.

    generation_method : { 'nonparametric', 'parametric' }
        The type of datasets to generate.  'parametric' generates DataSets
        with the same circuits and sample counts as input_data_set but
        using the probabilities in input_model (which must be provided).
        'nonparametric' generates DataSets with the same circuits
        and sample counts as input_data_set using the count frequencies of
        input_data_set as probabilities.

    fiducial_prep : list of Circuits
        The state preparation fiducial circuits used by MLGST.

    fiducial_measure : list of Circuits
        The measurement fiducial circuits used by MLGST.

    germs : list of Circuits
        The germ circuits used by MLGST.

    max_lengths : list of ints
        List of integers, one per MLGST iteration, which set truncation lengths
        for repeated germ strings.  The list of circuits for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    input_model : Model, optional
        The model used to compute the probabilities for circuits when
        generation_method is set to 'parametric'.  If 'nonparametric' is selected,
        this argument must be set to None (the default).

    target_model : Model, optional
        Mandatory model to use for as the target model for MLGST when
        generation_method is set to 'nonparametric'.  When 'parametric'
        is selected, input_model is used as the target.

    start_seed : int, optional
        The initial seed value for numpy's random number generator when
        generating data sets.  For each succesive dataset (and model)
        that are generated, the seed is incremented by one.

    outcome_labels : list, optional
        The list of Outcome labels to include in the output dataset.  If None
        are specified, defaults to the effect labels of `input_data_set`.

    lsgst_lists : list of circuit lists, optional
        Provides explicit list of circuit lists to be used in analysis;
        to be given if the dataset uses "incomplete" or "reduced" sets of
        circuit.  Default is None.

    return_data : bool
        Whether generated data sets should be returned in addition to
        models.

    verbosity : int
        Level of detail printed to stdout.

    Returns
    -------
    models : list
        The list of generated Model objects.
    datasets : list
        The list of generated DataSet objects, only returned when
        return_data == True.
    """

    if max_lengths is None:
        print("No max_lengths value specified; using [0,1,24,...,1024]")
        max_lengths = [0] + [2**k for k in range(10)]

    if (input_model is None and target_model is None):
        raise ValueError("Must supply either input_model or target_model!")
    if (input_model is not None and target_model is not None):
        raise ValueError("Cannot supply both input_model and target_model!")

    if generation_method == 'parametric':
        target_model = input_model

    datasetList = []
    print("Creating DataSets: ")
    for run in range(num_models):
        print("%d " % run, end='')
        datasetList.append(
            create_bootstrap_dataset(input_data_set, generation_method,
                                   input_model, start_seed + run,
                                   outcome_labels)
        )

    modelList = []
    print("Creating Models: ")
    for run in range(num_models):
        print("Running MLGST Iteration %d " % run)
        if lsgst_lists is not None:
            results = _longseq.run_long_sequence_gst_base(
                datasetList[run], target_model, lsgst_lists, verbosity=verbosity)
        else:
            results = _longseq.run_long_sequence_gst(
                datasetList[run], target_model,
                fiducial_prep, fiducial_measure, germs, max_lengths,
                verbosity=verbosity)
        modelList.append(results.estimates.get('default', next(iter(results.estimates.values()))).models['go0'])

    if not return_data:
        return modelList
    else:
        return modelList, datasetList


def gauge_optimize_models(gs_list, target_model,
                              gate_metric='frobenius', spam_metric='frobenius',
                              plot=True):
    """
    Optimizes the "spam weight" parameter used when gauge optimizing a set of models.

    This function gauge optimizes multiple times using a range of spam weights
    and takes the one the minimizes the average spam error multiplied by the
    average gate error (with respect to a target model).

    Parameters
    ----------
    gs_list : list
        The list of Model objects to gauge optimize (simultaneously).

    target_model : Model
        The model to compare the gauge-optimized gates with, and also
        to gauge-optimize them to.

    gate_metric : { "frobenius", "fidelity", "tracedist" }, optional
        The metric used within the gauge optimization to determing error
        in the gates.

    spam_metric : { "frobenius", "fidelity", "tracedist" }, optional
        The metric used within the gauge optimization to determing error
        in the state preparation and measurement.

    plot : bool, optional
        Whether to create a plot of the model-target discrepancy
        as a function of spam weight (figure displayed interactively).

    Returns
    -------
    list
        The list of Models gauge-optimized using the best spamWeight.
    """

    listOfBootStrapEstsNoOpt = list(gs_list)
    numResamples = len(listOfBootStrapEstsNoOpt)
    ddof = 1
    SPAMMin = []
    SPAMMax = []
    SPAMMean = []

    gateMin = []
    gateMax = []
    gateMean = []
    for spWind, spW in enumerate(_np.logspace(-4, 0, 13)):  # try spam weights
        print("Spam weight %s" % spWind)
        listOfBootStrapEstsNoOptG0toTargetVarSpam = []
        for mdl in listOfBootStrapEstsNoOpt:
            listOfBootStrapEstsNoOptG0toTargetVarSpam.append(
                _alg.gaugeopt_to_target(mdl, target_model,
                                        item_weights={'spam': spW},
                                        gates_metric=gate_metric,
                                        spam_metric=spam_metric))

        ModelGOtoTargetVarSpamVecArray = _np.zeros([numResamples],
                                                   dtype='object')
        for i in range(numResamples):
            ModelGOtoTargetVarSpamVecArray[i] = \
                listOfBootStrapEstsNoOptG0toTargetVarSpam[i].to_vector()

        mdlStdevVec = _np.std(ModelGOtoTargetVarSpamVecArray, ddof=ddof)
        gsStdevVecSPAM = mdlStdevVec[:8]
        mdlStdevVecOps = mdlStdevVec[8:]

        SPAMMin.append(_np.min(gsStdevVecSPAM))
        SPAMMax.append(_np.max(gsStdevVecSPAM))
        SPAMMean.append(_np.mean(gsStdevVecSPAM))

        gateMin.append(_np.min(mdlStdevVecOps))
        gateMax.append(_np.max(mdlStdevVecOps))
        gateMean.append(_np.mean(mdlStdevVecOps))

    if plot:
        raise NotImplementedError("plot removed b/c matplotlib support dropped")
        #_mpl.pyplot.loglog(_np.logspace(-4,0,13),SPAMMean,'b-o')
        #_mpl.pyplot.loglog(_np.logspace(-4,0,13),SPAMMin,'b--+')
        #_mpl.pyplot.loglog(_np.logspace(-4,0,13),SPAMMax,'b--x')
        #
        #_mpl.pyplot.loglog(_np.logspace(-4,0,13),gateMean,'r-o')
        #_mpl.pyplot.loglog(_np.logspace(-4,0,13),gateMin,'r--+')
        #_mpl.pyplot.loglog(_np.logspace(-4,0,13),gateMax,'r--x')
        #
        #_mpl.pyplot.xlabel('SPAM weight in gauge optimization')
        #_mpl.pyplot.ylabel('Per element error bar size')
        #_mpl.pyplot.title('Per element error bar size vs. ${\\tt spamWeight}$')
        #_mpl.pyplot.xlim(1e-4,1)
        #_mpl.pyplot.legend(['SPAM-mean','SPAM-min','SPAM-max',
        #                    'gates-mean','gates-min','gates-max'],
        #                   bbox_to_anchor=(1.4, 1.))

    # gateTimesSPAMMean = _np.array(SPAMMean) * _np.array(gateMean)

    bestSPAMWeight = _np.logspace(-4, 0, 13)[_np.argmin(
        _np.array(SPAMMean) * _np.array(gateMean))]
    print("Best SPAM weight is %s" % bestSPAMWeight)

    listOfBootStrapEstsG0toTargetSmallSpam = []
    for mdl in listOfBootStrapEstsNoOpt:
        listOfBootStrapEstsG0toTargetSmallSpam.append(
            _alg.gaugeopt_to_target(mdl, target_model,
                                    item_weights={'spam': bestSPAMWeight},
                                    gates_metric=gate_metric,
                                    spam_metric=spam_metric))

    return listOfBootStrapEstsG0toTargetSmallSpam


################################################################################
# Utility functions (perhaps relocate?)
################################################################################

#For metrics that evaluate model with single scalar:
def _model_stdev(gs_func, gs_ensemble, ddof=1, axis=None, **kwargs):
    """
    Standard deviation of `gs_func` over an ensemble of models.

    Parameters
    ----------
    gs_func : function
        A function that takes a :class:`Model` as its first argument, and
        whose additional arguments may be given by keyword arguments.

    gs_ensemble : list
        A list of `Model` objects.

    ddof : int, optional
        As in numpy.std

    axis : int or None, optional
        As in numpy.std

    Returns
    -------
    numpy.ndarray
        The output of numpy.std
    """
    return _np.std([gs_func(mdl, **kwargs) for mdl in gs_ensemble], axis=axis, ddof=ddof)


def _model_mean(gs_func, gs_ensemble, axis=None, **kwargs):
    """
    Mean of `gs_func` over an ensemble of models.

    Parameters
    ----------
    gs_func : function
        A function that takes a :class:`Model` as its first argument, and
        whose additional arguments may be given by keyword arguments.

    gs_ensemble : list
        A list of `Model` objects.

    axis : int or None, optional
        As in numpy.mean

    Returns
    -------
    numpy.ndarray
        The output of numpy.mean
    """
    return _np.mean([gs_func(mdl, **kwargs) for mdl in gs_ensemble], axis=axis)

#Note: for metrics that evaluate model with scalar for each gate, use axis=0
# argument to above functions


def _to_mean_model(gs_list, target_gs):
    """
    Take the per-gate-element mean of a set of models.

    Return the :class:`Model` constructed from the mean parameter
    vector of the models in `gs_list`, that is, the mean of the
    parameter vectors of each model in `gs_list`.

    Parameters
    ----------
    gs_list : list
        A list of :class:`Model` objects.

    target_gs : Model
        A template model used to specify the parameterization
        of the returned `Model`.

    Returns
    -------
    Model
    """
    numResamples = len(gs_list)
    gsVecArray = _np.zeros([numResamples], dtype='object')
    for i in range(numResamples):
        gsVecArray[i] = gs_list[i].to_vector()
    output_gs = target_gs.copy()
    output_gs.from_vector(_np.mean(gsVecArray))
    return output_gs


def _to_std_model(gs_list, target_gs, ddof=1):
    """
    Take the per-gate-element standard deviation of a list of models.

    Return the :class:`Model` constructed from the standard-deviation
    parameter vector of the models in `gs_list`, that is, the standard-
    devaiation of the parameter vectors of each model in `gs_list`.

    Parameters
    ----------
    gs_list : list
        A list of :class:`Model` objects.

    target_gs : Model
        A template model used to specify the parameterization
        of the returned `Model`.

    ddof : int, optional
        As in numpy.std

    Returns
    -------
    Model
    """
    numResamples = len(gs_list)
    gsVecArray = _np.zeros([numResamples], dtype='object')
    for i in range(numResamples):
        gsVecArray[i] = gs_list[i].to_vector()
    output_gs = target_gs.copy()
    output_gs.from_vector(_np.std(gsVecArray, ddof=ddof))
    return output_gs


def _to_rms_model(gs_list, target_gs):
    """
    Take the per-gate-element RMS of a set of models.

    Return the :class:`Model` constructed from the root-mean-squared
    parameter vector of the models in `gs_list`, that is, the RMS
    of the parameter vectors of each model in `gs_list`.

    Parameters
    ----------
    gs_list : list
        A list of :class:`Model` objects.

    target_gs : Model
        A template model used to specify the parameterization
        of the returned `Model`.

    Returns
    -------
    Model
    """
    numResamples = len(gs_list)
    gsVecArray = _np.zeros([numResamples], dtype='object')
    for i in range(numResamples):
        gsVecArray[i] = _np.sqrt(gs_list[i].to_vector()**2)
    output_gs = target_gs.copy()
    output_gs.from_vector(_np.mean(gsVecArray))
    return output_gs

#Unused?
#def gateset_jtracedist(mdl,target_model,mx_basis="gm"):
#    output = _np.zeros(3,dtype=float)
#    for i, gate in enumerate(target_model.operations.keys()):
#        output[i] = _tools.jtracedist(mdl.operations[gate],target_model.operations[gate],mx_basis=mx_basis)
##    print output
#    return output
#
#def gateset_entanglement_fidelity(mdl,target_model):
#    output = _np.zeros(3,dtype=float)
#    for i, gate in enumerate(target_model.operations.keys()):
#        output[i] = _tools.entanglement_fidelity(mdl.operations[gate],target_model.operations[gate])
#    return output
#
#def gateset_decomp_angle(mdl):
#    output = _np.zeros(3,dtype=float)
#    for i, gate in enumerate(mdl.operations.keys()):
#        output[i] = _tools.decompose_gate_matrix(mdl.operations[gate]).get('pi rotations',0)
#    return output
#
#def gateset_decomp_decay_diag(mdl):
#    output = _np.zeros(3,dtype=float)
#    for i, gate in enumerate(mdl.operations.keys()):
#        output[i] = _tools.decompose_gate_matrix(mdl.operations[gate]).get('decay of diagonal rotation terms',0)
#    return output
#
#def gateset_decomp_decay_offdiag(mdl):
#    output = _np.zeros(3,dtype=float)
#    for i, gate in enumerate(mdl.operations.keys()):
#        output[i] = _tools.decompose_gate_matrix(mdl.operations[gate]).get('decay of off diagonal rotation terms',0)
#    return output
#
##def gateset_fidelity(mdl,target_model,mx_basis="gm"):
##    output = _np.zeros(3,dtype=float)
##    for i, gate in enumerate(target_model.operations.keys()):
##        output[i] = _tools.fidelity(mdl.operations[gate],target_model.operations[gate])
##    return output
#
#def gateset_diamonddist(mdl,target_model,mx_basis="gm"):
#    output = _np.zeros(3,dtype=float)
#    for i, gate in enumerate(target_model.operations.keys()):
#        output[i] = _tools.diamonddist(mdl.operations[gate],target_model.operations[gate],mx_basis=mx_basis)
#    return output
#
#def spamrameter(mdl):
#    firstRho = list(mdl.preps.keys())[0]
#    firstE = list(mdl.effects.keys())[0]
#    return _np.dot(mdl.preps[firstRho].T,mdl.effects[firstE])[0,0]

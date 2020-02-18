""" Functions for generating bootstrapped error bars """
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


def make_bootstrap_dataset(inputDataSet, generationMethod, inputModel=None,
                           seed=None, outcomeLabels=None, verbosity=1):
    """
    Creates a DataSet used for generating bootstrapped error bars.

    Parameters
    ----------
    inputDataSet : DataSet
       The data set to use for generating the "bootstrapped" data set.

    generationMethod : { 'nonparametric', 'parametric' }
      The type of dataset to generate.  'parametric' generates a DataSet
      with the same operation sequences and sample counts as inputDataSet but
      using the probabilities in inputModel (which must be provided).
      'nonparametric' generates a DataSet with the same operation sequences
      and sample counts as inputDataSet using the count frequencies of
      inputDataSet as probabilities.

    inputModel : Model, optional
       The model used to compute the probabilities for operation sequences when
       generationMethod is set to 'parametric'.  If 'nonparametric' is selected,
       this argument must be set to None (the default).

    seed : int, optional
       A seed value for numpy's random number generator.

    outcomeLabels : list, optional
       The list of outcome labels to include in the output dataset.  If None
       are specified, defaults to the spam labels of inputDataSet.

    verbosity : int, optional
       How verbose the function output is.  If 0, then printing is suppressed.
       If 1 (or greater), then printing is not suppressed.

    Returns
    -------
    DataSet
    """
    if generationMethod not in ['nonparametric', 'parametric']:
        raise ValueError("generationMethod must be 'parametric' or 'nonparametric'!")
    if outcomeLabels is None:
        outcomeLabels = inputDataSet.get_outcome_labels()

    rndm = seed if isinstance(seed, _np.random.RandomState) \
        else _np.random.RandomState(seed)

    if inputModel is None:
        if generationMethod == 'nonparametric':
            print("Generating non-parametric dataset.")
        elif generationMethod == 'parametric':
            raise ValueError("For 'parmametric', must specify inputModel")
    else:
        if generationMethod == 'parametric':
            print("Generating parametric dataset.")
        elif generationMethod == 'nonparametric':
            raise ValueError("For 'nonparametric', inputModel must be None")
        firstPOVMLbl = list(inputModel.povms.keys())[0]
        # TODO: allow outcomes from multiple POVMS? (now just consider *first* POVM)
        possibleOutcomeLabels = [(eLbl,) for eLbl in inputModel.povms[firstPOVMLbl].keys()]
        assert(all([ol in possibleOutcomeLabels for ol in outcomeLabels]))

    possibleOutcomeLabels = inputDataSet.get_outcome_labels()
    assert(all([ol in possibleOutcomeLabels for ol in outcomeLabels]))

    #create new dataset
    simDS = _obj.DataSet(outcomeLabels=outcomeLabels,
                         collisionAction=inputDataSet.collisionAction)
    circuit_list = list(inputDataSet.keys())
    for s in circuit_list:
        nSamples = inputDataSet[s].total
        if generationMethod == 'parametric':
            ps = inputModel.probs(s)
        elif generationMethod == 'nonparametric':
            ps = {ol: inputDataSet[s].fraction(ol) for ol in outcomeLabels}
        pList = _np.array([_np.clip(ps[outcomeLabel], 0, 1) for outcomeLabel in outcomeLabels])
        #Truncate before normalization; bad extremal values shouldn't
        # screw up not-bad values, yes?
        pList = pList / sum(pList)
        countsArray = rndm.multinomial(nSamples, pList, 1)
        counts = {ol: countsArray[0, i] for i, ol in enumerate(outcomeLabels)}
        simDS.add_count_dict(s, counts)
    simDS.done_adding_data()
    return simDS


def make_bootstrap_models(numModels, inputDataSet, generationMethod,
                          fiducialPrep, fiducialMeasure, germs, maxLengths,
                          inputModel=None, targetModel=None, startSeed=0,
                          outcomeLabels=None, lsgstLists=None,
                          returnData=False, verbosity=2):
    """
    Creates a series of "bootstrapped" Models form a single DataSet (and
    possibly Model) used for generating bootstrapped error bars.  The
    resulting Models are obtained by performing MLGST on datasets generated
    by repeatedly calling make_bootstrap_dataset with consecutive integer seed
    values.

    Parameters
    ----------
    numModels : int
       The number of models to create.

    inputDataSet : DataSet
       The data set to use for generating the "bootstrapped" data set.

    generationMethod : { 'nonparametric', 'parametric' }
      The type of datasets to generate.  'parametric' generates DataSets
      with the same operation sequences and sample counts as inputDataSet but
      using the probabilities in inputModel (which must be provided).
      'nonparametric' generates DataSets with the same operation sequences
      and sample counts as inputDataSet using the count frequencies of
      inputDataSet as probabilities.

    fiducialPrep : list of Circuits
        The state preparation fiducial operation sequences used by MLGST.

    fiducialMeasure : list of Circuits
        The measurement fiducial operation sequences used by MLGST.

    germs : list of Circuits
        The germ operation sequences used by MLGST.

    maxLengths : list of ints
        List of integers, one per MLGST iteration, which set truncation lengths
        for repeated germ strings.  The list of operation sequences for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    inputModel : Model, optional
       The model used to compute the probabilities for operation sequences when
       generationMethod is set to 'parametric'.  If 'nonparametric' is selected,
       this argument must be set to None (the default).

    targetModel : Model, optional
       Mandatory model to use for as the target model for MLGST when
       generationMethod is set to 'nonparametric'.  When 'parametric'
       is selected, inputModel is used as the target.

    startSeed : int, optional
       The initial seed value for numpy's random number generator when
       generating data sets.  For each succesive dataset (and model)
       that are generated, the seed is incremented by one.

    outcomeLabels : list, optional
       The list of Outcome labels to include in the output dataset.  If None
       are specified, defaults to the effect labels of `inputDataSet`.

    lsgstLists : list of operation sequence lists, optional
        Provides explicit list of operation sequence lists to be used in analysis;
        to be given if the dataset uses "incomplete" or "reduced" sets of
        operation sequence.  Default is None.

    returnData : bool
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
       returnData == True.
    """

    if maxLengths is None:
        print("No maxLengths value specified; using [0,1,24,...,1024]")
        maxLengths = [0] + [2**k for k in range(10)]

    if (inputModel is None and targetModel is None):
        raise ValueError("Must supply either inputModel or targetModel!")
    if (inputModel is not None and targetModel is not None):
        raise ValueError("Cannot supply both inputModel and targetModel!")

    if generationMethod == 'parametric':
        targetModel = inputModel

    datasetList = []
    print("Creating DataSets: ")
    for run in range(numModels):
        print("%d " % run, end='')
        datasetList.append(
            make_bootstrap_dataset(inputDataSet, generationMethod,
                                   inputModel, startSeed + run,
                                   outcomeLabels)
        )

    modelList = []
    print("Creating Models: ")
    for run in range(numModels):
        print("Running MLGST Iteration %d " % run)
        if lsgstLists is not None:
            results = _longseq.do_long_sequence_gst_base(
                datasetList[run], targetModel, lsgstLists, verbosity=verbosity)
        else:
            results = _longseq.do_long_sequence_gst(
                datasetList[run], targetModel,
                fiducialPrep, fiducialMeasure, germs, maxLengths,
                verbosity=verbosity)
        modelList.append(results.estimates['default'].models['go0'])

    if not returnData:
        return modelList
    else:
        return modelList, datasetList


def gauge_optimize_model_list(gsList, targetModel,
                              gateMetric='frobenius', spamMetric='frobenius',
                              plot=True):
    """
    Optimizes the "spam weight" parameter used in gauge optimization by
    attempting spam a range of spam weights and taking the one the minimizes
    the average spam error multiplied by the average gate error (with respect
    to a target model).

    Parameters
    ----------
    gsList : list
       The list of Model objects to gauge optimize (simultaneously).

    targetModel : Model
       The model to compare the gauge-optimized gates with, and also
       to gauge-optimize them to.

    gateMetric : { "frobenius", "fidelity", "tracedist" }, optional
       The metric used within the gauge optimization to determing error
       in the gates.

    spamMetric : { "frobenius", "fidelity", "tracedist" }, optional
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

    listOfBootStrapEstsNoOpt = list(gsList)
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
                _alg.gaugeopt_to_target(mdl, targetModel,
                                        itemWeights={'spam': spW},
                                        gatesMetric=gateMetric,
                                        spamMetric=spamMetric))

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
            _alg.gaugeopt_to_target(mdl, targetModel,
                                    itemWeights={'spam': bestSPAMWeight},
                                    gatesMetric=gateMetric,
                                    spamMetric=spamMetric))

    return listOfBootStrapEstsG0toTargetSmallSpam


################################################################################
# Utility functions (perhaps relocate?)
################################################################################

#For metrics that evaluate model with single scalar:
def mdl_stdev(gsFunc, gsEnsemble, ddof=1, axis=None, **kwargs):
    """
    Standard deviation of `gsFunc` over an ensemble of models.

    Parameters
    ----------
    gsFunc : function
        A function that takes a :class:`Model` as its first argument, and
        whose additional arguments may be given by keyword arguments.

    gsEnsemble : list
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
    return _np.std([gsFunc(mdl, **kwargs) for mdl in gsEnsemble], axis=axis, ddof=ddof)


def mdl_mean(gsFunc, gsEnsemble, axis=None, **kwargs):
    """
    Mean of `gsFunc` over an ensemble of models.

    Parameters
    ----------
    gsFunc : function
        A function that takes a :class:`Model` as its first argument, and
        whose additional arguments may be given by keyword arguments.

    gsEnsemble : list
        A list of `Model` objects.

    axis : int or None, optional
       As in numpy.mean

    Returns
    -------
    numpy.ndarray
        The output of numpy.mean
    """
    return _np.mean([gsFunc(mdl, **kwargs) for mdl in gsEnsemble], axis=axis)

#Note: for metrics that evaluate model with scalar for each gate, use axis=0
# argument to above functions


def to_mean_model(gsList, target_gs):
    """
    Return the :class:`Model` constructed from the mean parameter
    vector of the models in `gsList`, that is, the mean of the
    parameter vectors of each model in `gsList`.

    Parameters
    ----------
    gsList : list
        A list of :class:`Model` objects.

    target_gs : Model
        A template model used to specify the parameterization
        of the returned `Model`.

    Returns
    -------
    Model
    """
    numResamples = len(gsList)
    gsVecArray = _np.zeros([numResamples], dtype='object')
    for i in range(numResamples):
        gsVecArray[i] = gsList[i].to_vector()
    output_gs = target_gs.copy()
    output_gs.from_vector(_np.mean(gsVecArray))
    return output_gs


def to_std_model(gsList, target_gs, ddof=1):
    """
    Return the :class:`Model` constructed from the standard-deviation
    parameter vector of the models in `gsList`, that is, the standard-
    devaiation of the parameter vectors of each model in `gsList`.

    Parameters
    ----------
    gsList : list
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
    numResamples = len(gsList)
    gsVecArray = _np.zeros([numResamples], dtype='object')
    for i in range(numResamples):
        gsVecArray[i] = gsList[i].to_vector()
    output_gs = target_gs.copy()
    output_gs.from_vector(_np.std(gsVecArray, ddof=ddof))
    return output_gs


def to_rms_model(gsList, target_gs):
    """
    Return the :class:`Model` constructed from the root-mean-squared
    parameter vector of the models in `gsList`, that is, the RMS
    of the parameter vectors of each model in `gsList`.

    Parameters
    ----------
    gsList : list
        A list of :class:`Model` objects.

    target_gs : Model
        A template model used to specify the parameterization
        of the returned `Model`.

    Returns
    -------
    Model
    """
    numResamples = len(gsList)
    gsVecArray = _np.zeros([numResamples], dtype='object')
    for i in range(numResamples):
        gsVecArray[i] = _np.sqrt(gsList[i].to_vector()**2)
    output_gs = target_gs.copy()
    output_gs.from_vector(_np.mean(gsVecArray))
    return output_gs

#Unused?
#def gateset_jtracedist(mdl,target_model,mxBasis="gm"):
#    output = _np.zeros(3,dtype=float)
#    for i, gate in enumerate(target_model.operations.keys()):
#        output[i] = _tools.jtracedist(mdl.operations[gate],target_model.operations[gate],mxBasis=mxBasis)
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
##def gateset_fidelity(mdl,target_model,mxBasis="gm"):
##    output = _np.zeros(3,dtype=float)
##    for i, gate in enumerate(target_model.operations.keys()):
##        output[i] = _tools.fidelity(mdl.operations[gate],target_model.operations[gate])
##    return output
#
#def gateset_diamonddist(mdl,target_model,mxBasis="gm"):
#    output = _np.zeros(3,dtype=float)
#    for i, gate in enumerate(target_model.operations.keys()):
#        output[i] = _tools.diamonddist(mdl.operations[gate],target_model.operations[gate],mxBasis=mxBasis)
#    return output
#
#def spamrameter(mdl):
#    firstRho = list(mdl.preps.keys())[0]
#    firstE = list(mdl.effects.keys())[0]
#    return _np.dot(mdl.preps[firstRho].T,mdl.effects[firstE])[0,0]

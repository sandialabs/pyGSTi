from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for generating bootstrapped error bars """
import numpy as _np
import matplotlib as _mpl
from .longsequence import do_long_sequence_gst as _do_long_sequence_gst

from .. import objects as _obj
from .. import algorithms as _alg
from .. import tools as _tools

def make_bootstrap_dataset(inputDataSet,generationMethod,inputGateSet=None,
                           seed=None,spamLabels=None):
    """
    Creates a DataSet used for generating bootstrapped error bars.

    Parameters
    ----------
    inputDataSet : DataSet
       The data set to use for generating the "bootstrapped" data set.

    generationMethod : { 'nonparametric', 'parametric' }
      The type of dataset to generate.  'parametric' generates a DataSet
      with the same gate strings and sample counts as inputDataSet but
      using the probabilities in inputGateSet (which must be provided).
      'nonparametric' generates a DataSet with the same gate strings
      and sample counts as inputDataSet using the count frequencies of
      inputDataSet as probabilities.

    inputGateSet : GateSet, optional
       The gate set used to compute the probabilities for gate strings when
       generationMethod is set to 'parametric'.  If 'nonparametric' is selected,
       this argument must be set to None (the default).

    seed : int, optional
       A seed value for numpy's random number generator.

    spamLabels : list, optional
       The list of SPAM labels to include in the output dataset.  If None
       are specified, defaults to the spam labels of inputDataSet.

    Returns
    -------
    DataSet
    """
    if generationMethod not in ['nonparametric', 'parametric']:
        raise ValueError("generationMethod must be 'parametric' or 'nonparametric'!")
    if spamLabels is None:
        spamLabels = inputDataSet.get_spam_labels()

    rndm = _np.random.RandomState(seed)
    if inputGateSet is None:
        if generationMethod == 'nonparametric':
            print("Generating non-parametric dataset.")
        elif generationMethod == 'parametric':
            raise ValueError("For 'parmametric', must specify inputGateSet")
    else:
        if generationMethod == 'parametric':
            print("Generating parametric dataset.")
        elif generationMethod == 'nonparametric':
            raise ValueError("For 'nonparametric', inputGateSet must be None")
        possibleSpamLabels = inputGateSet.get_spam_labels()
        assert( all([sl in possibleSpamLabels for sl in spamLabels]) )

    possibleSpamLabels = inputDataSet.get_spam_labels()
    assert( all([sl in possibleSpamLabels for sl in spamLabels]) )

    #create new dataset
    simDS = _obj.DataSet(spamLabels=spamLabels, 
                         collisionAction=inputDataSet.collisionAction)
    gatestring_list = list(inputDataSet.keys())
    for s in gatestring_list:
        nSamples = inputDataSet[s].total()
        if generationMethod == 'parametric':
            ps = inputGateSet.probs(s)
        elif generationMethod == 'nonparametric':
            ps = { sl: inputDataSet[s].fraction(sl) for sl in spamLabels }
        pList = _np.array([_np.clip(ps[spamLabel],0,1) for spamLabel in spamLabels])
          #Truncate before normalization; bad extremal values shouldn't
          # screw up not-bad values, yes?
        pList = pList / sum(pList)
        countsArray = rndm.multinomial(nSamples, pList, 1)
        counts = { sl: countsArray[0,i] for i,sl in enumerate(spamLabels) }
        simDS.add_count_dict(s, counts)
    simDS.done_adding_data()
    return simDS

def make_bootstrap_gatesets(numGateSets, inputDataSet, generationMethod,
                            fiducialPrep, fiducialMeasure, germs, maxLengths,
                            inputGateSet=None, targetGateSet=None, startSeed=0,
                            spamLabels=None, lsgstLists=None,
                            returnData=False, verbosity=2):
    """
    Creates a series of "bootstrapped" GateSets form a single DataSet (and
    possibly GateSet) used for generating bootstrapped error bars.  The
    resulting GateSets are obtained by performing MLGST on datasets generated
    by repeatedly calling make_bootstrap_dataset with consecutive integer seed
    values.

    Parameters
    ----------
    numGateSets : int
       The number of gate sets to create.

    inputDataSet : DataSet
       The data set to use for generating the "bootstrapped" data set.

    generationMethod : { 'nonparametric', 'parametric' }
      The type of datasets to generate.  'parametric' generates DataSets
      with the same gate strings and sample counts as inputDataSet but
      using the probabilities in inputGateSet (which must be provided).
      'nonparametric' generates DataSets with the same gate strings
      and sample counts as inputDataSet using the count frequencies of
      inputDataSet as probabilities.

    fiducialPrep : list of GateStrings
        The state preparation fiducial gate strings used by MLGST.

    fiducialMeasure : list of GateStrings
        The measurement fiducial gate strings used by MLGST.

    germs : list of GateStrings
        The germ gate strings used by MLGST.

    maxLengths : list of ints
        List of integers, one per MLGST iteration, which set truncation lengths
        for repeated germ strings.  The list of gate strings for the i-th LSGST
        iteration includes the repeated germs truncated to the L-values *up to*
        and including the i-th one.

    inputGateSet : GateSet, optional
       The gate set used to compute the probabilities for gate strings when
       generationMethod is set to 'parametric'.  If 'nonparametric' is selected,
       this argument must be set to None (the default).

    targetGateSet : GateSet, optional
       Mandatory gate set to use for as the target gate set for MLGST when
       generationMethod is set to 'nonparametric'.  When 'parametric'
       is selected, inputGateSet is used as the target.

    startSeed : int, optional
       The initial seed value for numpy's random number generator when
       generating data sets.  For each succesive dataset (and gateset)
       that are generated, the seed is incremented by one.

    spamLabels : list, optional
       The list of SPAM labels to include in the output dataset.  If None
       are specified, defaults to the spam labels of inputDataSet.

    lsgstLists : list of gate string lists, optional
        Provides explicit list of gate string lists to be used in analysis;
        to be given if the dataset uses "incomplete" or "reduced" sets of
        gate string.  Default is None.

    returnData : bool
        Whether generated data sets should be returned in addition to
        gate sets.

    verbosity : int
        Level of detail printed to stdout.

    Returns
    -------
    gatesets : list
       The list of generated GateSet objects.

    datasets : list
       The list of generated DataSet objects, only returned when
       returnData == True.
    """

    if maxLengths == None:
        print("No maxLengths value specified; using [0,1,24,...,1024]")
        maxLengths = [0]+[2**k for k in range(10)]

    if (inputGateSet is None and targetGateSet is None):
        raise ValueError("Must supply either inputGateSet or targetGateSet!")
    if (inputGateSet is not None and targetGateSet is not None):
        raise ValueError("Cannot supply both inputGateSet and targetGateSet!")

    if generationMethod == 'parametric':
        targetGateSet = inputGateSet

    datasetList = []
    print("Creating DataSets: ")
    for run in range(numGateSets):
        print("%d " % run, end='')
        datasetList.append(
            make_bootstrap_dataset(inputDataSet,generationMethod,
                                   inputGateSet, startSeed+run,
                                   spamLabels)
            )

    gatesetList = []
    print("Creating GateSets: ")
    for run in range(numGateSets):
        print("Running MLGST Iteration %d " % run)
        results = _do_long_sequence_gst(
            datasetList[run], targetGateSet, fiducialPrep, fiducialMeasure,
            germs, maxLengths, lsgstLists=lsgstLists, verbosity=verbosity)
        gatesetList.append(results.gatesets['final estimate'])

    if not returnData:
        return gatesetList
    else:
        return gatesetList, datasetList


def gauge_optimize_gs_list(gsList, targetGateset,
                           gateMetric = 'frobenius', spamMetric = 'frobenius',
                           plot=True):
    """
    Optimizes the "spam weight" parameter used in gauge optimization by
    attempting spam a range of spam weights and taking the one the minimizes
    the average spam error multiplied by the average gate error (with respect
    to a target gate set).

    Parameters
    ----------
    gsList : list
       The list of GateSet objects to gauge optimize (simultaneously).

    targetGateset : GateSet
       The gateset to compare the gauge-optimized gates with, and also
       to gauge-optimize them to.

    gateMetric : { "frobenius", "fidelity", "tracedist" }, optional
       The metric used within the gauge optimization to determing error
       in the gates.

    spamMetric : { "frobenius", "fidelity", "tracedist" }, optional
       The metric used within the gauge optimization to determing error
       in the state preparation and measurement.

    plot : bool, optional
       Whether to create a plot of the gateset-target discrepancy
       as a function of spam weight (figure displayed interactively).

    Returns
    -------
    list
       The list of GateSets gauge-optimized using the best spamWeight.
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
    for spWind, spW in enumerate(_np.logspace(-4,0,13)): #try spam weights
        print("Spam weight %s" % spWind)
        listOfBootStrapEstsNoOptG0toTargetVarSpam = []
        for gs in listOfBootStrapEstsNoOpt:
            listOfBootStrapEstsNoOptG0toTargetVarSpam.append(
                _alg.gaugeopt_to_target(gs,targetGateset, 
                                        itemWeights={'spam': spW },
                                        gatesMetric=gateMetric,
                                        spamMetric=spamMetric))

        GateSetGOtoTargetVarSpamVecArray = _np.zeros([numResamples],
                                                     dtype='object')
        for i in range(numResamples):
            GateSetGOtoTargetVarSpamVecArray[i] = \
                listOfBootStrapEstsNoOptG0toTargetVarSpam[i].to_vector()

        gsStdevVec = _np.std(GateSetGOtoTargetVarSpamVecArray,ddof=ddof)
        gsStdevVecSPAM = gsStdevVec[:8]
        gsStdevVecGates = gsStdevVec[8:]

        SPAMMin.append(_np.min(gsStdevVecSPAM))
        SPAMMax.append(_np.max(gsStdevVecSPAM))
        SPAMMean.append(_np.mean(gsStdevVecSPAM))

        gateMin.append(_np.min(gsStdevVecGates))
        gateMax.append(_np.max(gsStdevVecGates))
        gateMean.append(_np.mean(gsStdevVecGates))

    if plot:
        _mpl.pyplot.loglog(_np.logspace(-4,0,13),SPAMMean,'b-o')
        _mpl.pyplot.loglog(_np.logspace(-4,0,13),SPAMMin,'b--+')
        _mpl.pyplot.loglog(_np.logspace(-4,0,13),SPAMMax,'b--x')

        _mpl.pyplot.loglog(_np.logspace(-4,0,13),gateMean,'r-o')
        _mpl.pyplot.loglog(_np.logspace(-4,0,13),gateMin,'r--+')
        _mpl.pyplot.loglog(_np.logspace(-4,0,13),gateMax,'r--x')

        _mpl.pyplot.xlabel('SPAM weight in gauge optimization')
        _mpl.pyplot.ylabel('Per element error bar size')
        _mpl.pyplot.title('Per element error bar size vs. ${\\tt spamWeight}$')
        _mpl.pyplot.xlim(1e-4,1)
        _mpl.pyplot.legend(['SPAM-mean','SPAM-min','SPAM-max',
                            'gates-mean','gates-min','gates-max'],
                           bbox_to_anchor=(1.4, 1.))

    # gateTimesSPAMMean = _np.array(SPAMMean) * _np.array(gateMean)

    bestSPAMWeight = _np.logspace(-4,0,13)[ _np.argmin(
            _np.array(SPAMMean)*_np.array(gateMean)) ]
    print("Best SPAM weight is %s" % bestSPAMWeight)

    listOfBootStrapEstsG0toTargetSmallSpam = []
    for gs in listOfBootStrapEstsNoOpt:
        listOfBootStrapEstsG0toTargetSmallSpam.append(
            _alg.gaugeopt_to_target(gs,targetGateset,
                                    itemWeights={'spam': bestSPAMWeight},
                                    gatesMetric=gateMetric,
                                    spamMetric=spamMetric))

    return listOfBootStrapEstsG0toTargetSmallSpam




################################################################################
#TODO: need to add docstrings and perhaps relocate the utility functions below #
################################################################################

#For metrics that evaluate gateset with single scalar:
def gs_stdev(gsFunc, gsEnsemble, ddof=1, **kwargs):
    return _np.std([gsFunc(gs, **kwargs) for gs in gsEnsemble],ddof=ddof)

def gs_mean(gsFunc, gsEnsemble, axis = 0,**kwargs):
    return _np.mean([gsFunc(gs, **kwargs) for gs in gsEnsemble])

#For metrics that evaluate gateset with scalar for each gate
def gs_stdev1(gsFunc, gsEnsemble, ddof=1,axis=0, **kwargs):
    return _np.std([gsFunc(gs, **kwargs) for gs in gsEnsemble],axis=axis,ddof=ddof)

def gs_mean1(gsFunc, gsEnsemble, axis = 0,**kwargs):
    return _np.mean([gsFunc(gs,**kwargs) for gs in gsEnsemble],axis=axis)

def to_vector(gs):
    return gs.to_vector()

def to_mean_gateset(gsList,target_gs):
    numResamples = len(gsList)
    gsVecArray = _np.zeros([numResamples],dtype='object')
    for i in range(numResamples):
        gsVecArray[i] = gsList[i].to_vector()
    output_gs = target_gs.copy()
    output_gs.from_vector(_np.mean(gsVecArray))
    return output_gs

def to_std_gateset(gsList,target_gs,ddof=1):
    numResamples = len(gsList)
    gsVecArray = _np.zeros([numResamples],dtype='object')
    for i in range(numResamples):
        gsVecArray[i] = gsList[i].to_vector()
    output_gs = target_gs.copy()
    output_gs.from_vector(_np.std(gsVecArray,ddof=ddof))
    return output_gs

def to_rms_gateset(gsList,target_gs):
    numResamples = len(gsList)
    gsVecArray = _np.zeros([numResamples],dtype='object')
    for i in range(numResamples):
        gsVecArray[i] = _np.sqrt(gsList[i].to_vector()**2)
    output_gs = target_gs.copy()
    output_gs.from_vector(_np.mean(gsVecArray))
    return output_gs

def gateset_jtracedist(gs,gs_target,mxBasis="gm"):
    output = _np.zeros(3,dtype=float)
    for i, gate in enumerate(gs_target.gates.keys()):
        output[i] = _tools.jtracedist(gs.gates[gate],gs_target.gates[gate],mxBasis=mxBasis)
#    print output
    return output

def gateset_process_fidelity(gs,gs_target):
    output = _np.zeros(3,dtype=float)
    for i, gate in enumerate(gs_target.gates.keys()):
        output[i] = _tools.process_fidelity(gs.gates[gate],gs_target.gates[gate])
    return output

def gateset_decomp_angle(gs):
    output = _np.zeros(3,dtype=float)
    for i, gate in enumerate(gs.gates.keys()):
        output[i] = _tools.decompose_gate_matrix(gs.gates[gate]).get('pi rotations',0)
    return output

def gateset_decomp_decay_diag(gs):
    output = _np.zeros(3,dtype=float)
    for i, gate in enumerate(gs.gates.keys()):
        output[i] = _tools.decompose_gate_matrix(gs.gates[gate]).get('decay of diagonal rotation terms',0)
    return output

def gateset_decomp_decay_offdiag(gs):
    output = _np.zeros(3,dtype=float)
    for i, gate in enumerate(gs.gates.keys()):
        output[i] = _tools.decompose_gate_matrix(gs.gates[gate]).get('decay of off diagonal rotation terms',0)
    return output

#def gateset_fidelity(gs,gs_target,mxBasis="gm"):
#    output = _np.zeros(3,dtype=float)
#    for i, gate in enumerate(gs_target.gates.keys()):
#        output[i] = _tools.fidelity(gs.gates[gate],gs_target.gates[gate])
#    return output

def gateset_diamonddist(gs,gs_target,mxBasis="gm"):
    output = _np.zeros(3,dtype=float)
    for i, gate in enumerate(gs_target.gates.keys()):
        output[i] = _tools.diamonddist(gs.gates[gate],gs_target.gates[gate],mxBasis=mxBasis)
    return output

def spamrameter(gs):
    firstRho = list(gs.preps.keys())[0]
    firstE = list(gs.effects.keys())[0]
    return _np.dot(gs.preps[firstRho].T,gs.effects[firstE])[0,0]

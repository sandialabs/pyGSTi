""" Functions for creating RPE Models and Circuit lists """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from . import rpetools as _rpetools
from ... import construction as _cnst
from ... import objects as _objs
from ... import tools as _tools


def make_parameterized_rpe_gate_set(alphaTrue, epsilonTrue, auxRot, SPAMdepol,
                                    gateDepol=None, withId=True, rpeconfig_inst=None):
    """
    Make a model for simulating RPE, paramaterized by rotation angles.  Note
    that the output model also has thetaTrue, alphaTrue, and epsilonTrue
    added attributes.

    Parameters
    ----------
    alphaTrue : float
       Angle of rotation about "fixed axis"

    epsilonTrue : float
       Angle of rotation about "loose axis"

    auxRot : float
       Angle of rotation about the axis perpendicular to fixed and loose axes,
       that, by similarity transformation, changes loose axis.

    SPAMdepol : float
       Amount to depolarize SPAM by.

    gateDepol : float, optional
       Amount to depolarize gates by (defaults to None).

    withId : bool, optional
       Do we include (perfect) identity or no identity? (Defaults to False;
       should be False for RPE, True for GST)

    rpeconfig_inst : rpeconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    Model
        The desired model for RPE; model also has attributes thetaTrue,
        alphaTrue, and epsilonTrue, automatically extracted.
    """

#    if rpeconfig_inst not in rpeInstanceDict.keys():
#        raise Exception('Need valid rpeconfig_inst!')

#    rpeconfig_inst = rpeInstanceDict[rpeconfig_inst]

    loose_axis_gate_label = rpeconfig_inst.loose_axis_gate_label
    loose_axis_label = rpeconfig_inst.loose_axis_label
    fixed_axis_gate_label = rpeconfig_inst.fixed_axis_gate_label
    fixed_axis_label = rpeconfig_inst.fixed_axis_label

    auxiliary_axis_gate_label = rpeconfig_inst.auxiliary_axis_gate_label
    auxiliary_axis_label = rpeconfig_inst.auxiliary_axis_label

    rhoExpressions = rpeconfig_inst.rhoExpressions
    EExpressions = rpeconfig_inst.EExpressions
    ELabels = rpeconfig_inst.ELabels

    if withId:
        outputModel = _cnst.build_explicit_model(
            [('Q0',)], ['Gi', loose_axis_gate_label, fixed_axis_gate_label],
            ["I(Q0)", loose_axis_label + "(%s,Q0)" % epsilonTrue, fixed_axis_label + "(%s,Q0)" % alphaTrue],
            prepLabels=["rho0"], prepExpressions=rhoExpressions,
            effectLabels=ELabels, effectExpressions=EExpressions)
    else:
        outputModel = _cnst.build_explicit_model(
            [('Q0',)], [loose_axis_gate_label, fixed_axis_gate_label],
            [loose_axis_label + "(%s,Q0)" % epsilonTrue, fixed_axis_label + "(%s,Q0)" % alphaTrue],
            prepLabels=["rho0"], prepExpressions=rhoExpressions,
            effectLabels=ELabels, effectExpressions=EExpressions)

    if auxRot != 0:
        modelAux1 = _cnst.build_explicit_model(
            [('Q0',)], ['Gi', auxiliary_axis_gate_label, fixed_axis_gate_label],
            ["I(Q0)", auxiliary_axis_label + "(%s,Q0)" % auxRot, fixed_axis_label + "(pi/2,Q0)"],
            prepLabels=["rho0"], prepExpressions=rhoExpressions,
            effectLabels=ELabels, effectExpressions=EExpressions)

        outputModel.operations[loose_axis_gate_label] = \
            _np.dot(_np.dot(_np.linalg.inv(modelAux1.operations[auxiliary_axis_gate_label]),
                            outputModel.operations[loose_axis_gate_label]),
                    modelAux1.operations[auxiliary_axis_gate_label])

    outputModel = outputModel.depolarize(op_noise=gateDepol,
                                         spam_noise=SPAMdepol)

    thetaTrue = _rpetools.extract_theta(outputModel, rpeconfig_inst)
    outputModel.thetaTrue = thetaTrue

    outputModel.alphaTrue = _rpetools.extract_alpha(outputModel, rpeconfig_inst)
    outputModel.alphaTrue = alphaTrue

    outputModel.epsilonTrue = _rpetools.extract_epsilon(outputModel, rpeconfig_inst)
    outputModel.epsilonTrue = epsilonTrue

    return outputModel

#def make_rpe_alpha_str_lists(kList,angleStr,rpeconfig_inst):


def make_rpe_angle_str_lists(kList, angleName, rpeconfig_inst):
    """
    Make cosine and sine circuit lists.  These operation sequences are used to estimate the angle specified
    by angleName ('alpha', 'epsilon', or 'theta')

    Parameters
    ----------
    kList : list of ints
        The list of "germ powers" to be used.  Typically successive powers of
        two; e.g. [1,2,4,8,16].

    angleName : string
        The angle to be deduced from these operation sequences.
        (Choices are 'alpha', 'epsilon', or 'theta')

    rpeconfig_inst : rpeconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    cosStrList : list of Circuits
        The list of "cosine strings" to be used for alpha estimation.
    sinStrList : list of Circuits
        The list of "sine strings" to be used for alpha estimation.
    """

#    rpeconfig_inst = rpeInstanceDict[rpeconfig_inst]

    if angleName == 'alpha':
        cos_prep_tuple = rpeconfig_inst.alpha_cos_prep_tuple
        cos_prep_str = rpeconfig_inst.alpha_cos_prep_str
        cos_germ_tuple = rpeconfig_inst.alpha_cos_germ_tuple
        cos_germ_str = rpeconfig_inst.alpha_cos_germ_str
        cos_meas_tuple = rpeconfig_inst.alpha_cos_meas_tuple
        cos_meas_str = rpeconfig_inst.alpha_cos_meas_str
        sin_prep_tuple = rpeconfig_inst.alpha_sin_prep_tuple
        sin_prep_str = rpeconfig_inst.alpha_sin_prep_str
        sin_germ_tuple = rpeconfig_inst.alpha_sin_germ_tuple
        sin_germ_str = rpeconfig_inst.alpha_sin_germ_str
        sin_meas_tuple = rpeconfig_inst.alpha_sin_meas_tuple
        sin_meas_str = rpeconfig_inst.alpha_sin_meas_str

    elif angleName == 'epsilon':
        cos_prep_tuple = rpeconfig_inst.epsilon_cos_prep_tuple
        cos_prep_str = rpeconfig_inst.epsilon_cos_prep_str
        cos_germ_tuple = rpeconfig_inst.epsilon_cos_germ_tuple
        cos_germ_str = rpeconfig_inst.epsilon_cos_germ_str
        cos_meas_tuple = rpeconfig_inst.epsilon_cos_meas_tuple
        cos_meas_str = rpeconfig_inst.epsilon_cos_meas_str
        sin_prep_tuple = rpeconfig_inst.epsilon_sin_prep_tuple
        sin_prep_str = rpeconfig_inst.epsilon_sin_prep_str
        sin_germ_tuple = rpeconfig_inst.epsilon_sin_germ_tuple
        sin_germ_str = rpeconfig_inst.epsilon_sin_germ_str
        sin_meas_tuple = rpeconfig_inst.epsilon_sin_meas_tuple
        sin_meas_str = rpeconfig_inst.epsilon_sin_meas_str

    elif angleName == 'theta':
        cos_prep_tuple = rpeconfig_inst.theta_cos_prep_tuple
        cos_prep_str = rpeconfig_inst.theta_cos_prep_str
        cos_germ_tuple = rpeconfig_inst.theta_cos_germ_tuple
        cos_germ_str = rpeconfig_inst.theta_cos_germ_str
        cos_meas_tuple = rpeconfig_inst.theta_cos_meas_tuple
        cos_meas_str = rpeconfig_inst.theta_cos_meas_str
        sin_prep_tuple = rpeconfig_inst.theta_sin_prep_tuple
        sin_prep_str = rpeconfig_inst.theta_sin_prep_str
        sin_germ_tuple = rpeconfig_inst.theta_sin_germ_tuple
        sin_germ_str = rpeconfig_inst.theta_sin_germ_str
        sin_meas_tuple = rpeconfig_inst.theta_sin_meas_tuple
        sin_meas_str = rpeconfig_inst.theta_sin_meas_str

    else:
        raise Exception("Need valid angle!")

    cosStrList = []
    sinStrList = []
    for k in kList:
        cosStrList += [_objs.Circuit(cos_prep_tuple + cos_germ_tuple * k + cos_meas_tuple,
                                     stringrep=cos_prep_str + '(' + cos_germ_str + ')^' + str(k) + cos_meas_str)]
        sinStrList += [_objs.Circuit(sin_prep_tuple + sin_germ_tuple * k + sin_meas_tuple,
                                     stringrep=sin_prep_str + '(' + sin_germ_str + ')^' + str(k) + sin_meas_str)]
    return cosStrList, sinStrList


def make_rpe_angle_string_list_dict(log2kMaxOrkList, rpeconfig_inst):
    """
    Generates a dictionary that contains operation sequences for all RPE cosine and
    sine experiments for all three angles.

    Parameters
    ----------
    log2kMaxOrkList : int or list
        int - log2(Maximum number of times to repeat an RPE germ)
        list - List of maximum number of times to repeat an RPE germ

    rpeconfig_inst : rpeconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    totalStrListD : dict
        A dictionary containing all operation sequences for all sine and cosine
        experiments for alpha, epsilon, and theta.
        The keys of the returned dictionary are:

        - 'alpha','cos' : List of operation sequences for cosine experiments used
          to determine alpha.
        - 'alpha','sin' : List of operation sequences for sine experiments used to
          determine alpha.
        - 'epsilon','cos' : List of operation sequences for cosine experiments used to
           determine epsilon.
        - 'epsilon','sin' : List of operation sequences for sine experiments used to
          determine epsilon.
        - 'theta','cos' : List of operation sequences for cosine experiments used to
          determine theta.
        - 'theta','sin' : List of operation sequences for sine experiments used to
          determine theta.
        - 'totalStrList' : All above operation sequences combined into one list;
          duplicates removed.
    """
    if isinstance(log2kMaxOrkList, int):
        kList = [2**k for k in range(log2kMaxOrkList + 1)]
    else:
        kList = log2kMaxOrkList
    alphaCosStrList, alphaSinStrList = make_rpe_angle_str_lists(kList, 'alpha', rpeconfig_inst)
    epsilonCosStrList, epsilonSinStrList = make_rpe_angle_str_lists(kList, 'epsilon', rpeconfig_inst)
    thetaCosStrList, thetaSinStrList = make_rpe_angle_str_lists(kList, 'theta', rpeconfig_inst)
    totalStrList = alphaCosStrList + alphaSinStrList \
        + epsilonCosStrList + epsilonSinStrList \
        + thetaCosStrList + thetaSinStrList
    totalStrList = _tools.remove_duplicates(totalStrList)  # probably superfluous

    stringListD = {}
    stringListD['alpha', 'cos'] = alphaCosStrList
    stringListD['alpha', 'sin'] = alphaSinStrList
    stringListD['epsilon', 'cos'] = epsilonCosStrList
    stringListD['epsilon', 'sin'] = epsilonSinStrList
    stringListD['theta', 'cos'] = thetaCosStrList
    stringListD['theta', 'sin'] = thetaSinStrList
    stringListD['totalStrList'] = totalStrList
    return stringListD


def make_rpe_data_set(modelOrDataset, stringListD, nSamples, sampleError='binomial', seed=None):
    """
    Generate a fake RPE DataSet using the probabilities obtained from a model.
    Is a thin wrapper for pygsti.construction.generate_fake_data, changing
    default behavior of sampleError, and taking a dictionary of operation sequences
    as input.

    Parameters
    ----------
    modelOrDataset : Model or DataSet object
        If a Model, the model whose probabilities generate the data.
        If a DataSet, the data set whose frequencies generate the data.

    stringListD : Dictionary of list of (tuples or Circuits)
        Each tuple or Circuit contains operation labels and
        specifies a gate sequence whose counts are included
        in the returned DataSet.  The dictionary must have the key
        'totalStrList'; easiest if this dictionary is generated by
        make_rpe_string_list_d.

    nSamples : int or list of ints or None
        The simulated number of samples for each operation sequence.  This only
        has effect when  sampleError == "binomial" or "multinomial".  If
        an integer, all operation sequences have this number of total samples. If
        a list, integer elements specify the number of samples for the
        corresponding operation sequence.  If None, then modelOrDataset must be
        a DataSet, and total counts are taken from it (on a per-circuit
        basis).

    sampleError : string, optional
        What type of sample error is included in the counts.  Can be:

        - "none"  - no sample error:
          counts are floating point numbers such that the exact
          probabilty can be found by the ratio of count / total.
        - "round" - same as "none", except counts are rounded to the nearest
          integer.
        - "binomial" - the number of counts is taken from a binomial
          distribution. Distribution has parameters p = probability of the
          operation sequence and n = number of samples.  This can only be used when
          there are exactly two outcome labels in modelOrDataset.
        - "multinomial" - counts are taken from a multinomial distribution.
          Distribution has parameters p_k = probability of the operation sequence
          using the k-th outcome label and n = number of samples.  This should not
          be used for RPE.

    seed : int, optional
        If not None, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    Returns
    -------
    DataSet
       A static data set filled with counts for the specified operation sequences.
    """
    return _cnst.generate_fake_data(modelOrDataset,
                                    stringListD['totalStrList'],
                                    nSamples, sampleError=sampleError, seed=seed)

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

from pygsti.extras.rpe import rpetools as _rpetools
from pygsti import models as _models
from pygsti import data as _data
from pygsti import tools as _tools
from pygsti.circuits.circuit import Circuit as _Circuit


def create_parameterized_rpe_model(alpha_true, epsilon_true, aux_rot, spam_depol,
                                   gate_depol=None, with_id=True, rpeconfig_inst=None):
    """
    Make a model for simulating RPE, paramaterized by rotation angles.  Note
    that the output model also has thetaTrue, alpha_true, and epsilon_true
    added attributes.

    Parameters
    ----------
    alpha_true : float
       Angle of rotation about "fixed axis"

    epsilon_true : float
       Angle of rotation about "loose axis"

    aux_rot : float
       Angle of rotation about the axis perpendicular to fixed and loose axes,
       that, by similarity transformation, changes loose axis.

    spam_depol : float
       Amount to depolarize SPAM by.

    gate_depol : float, optional
       Amount to depolarize gates by (defaults to None).

    with_id : bool, optional
       Do we include (perfect) identity or no identity? (Defaults to False;
       should be False for RPE, True for GST)

    rpeconfig_inst : RPEconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    Model
        The desired model for RPE; model also has attributes thetaTrue,
        alpha_true, and epsilon_true, automatically extracted.
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

    if with_id:
        outputModel = _models.create_explicit_model(
            [('Q0',)], ['Gi', loose_axis_gate_label, fixed_axis_gate_label],
            ["I(Q0)", loose_axis_label + "(%s,Q0)" % epsilon_true, fixed_axis_label + "(%s,Q0)" % alpha_true],
            prep_labels=["rho0"], prep_expressions=rhoExpressions,
            effect_labels=ELabels, effect_expressions=EExpressions)
    else:
        outputModel = _models.create_explicit_model(
            [('Q0',)], [loose_axis_gate_label, fixed_axis_gate_label],
            [loose_axis_label + "(%s,Q0)" % epsilon_true, fixed_axis_label + "(%s,Q0)" % alpha_true],
            prep_labels=["rho0"], prep_expressions=rhoExpressions,
            effect_labels=ELabels, effect_expressions=EExpressions)

    if aux_rot != 0:
        modelAux1 = _models.create_explicit_model(
            [('Q0',)], ['Gi', auxiliary_axis_gate_label, fixed_axis_gate_label],
            ["I(Q0)", auxiliary_axis_label + "(%s,Q0)" % aux_rot, fixed_axis_label + "(pi/2,Q0)"],
            prep_labels=["rho0"], prep_expressions=rhoExpressions,
            effect_labels=ELabels, effect_expressions=EExpressions)

        outputModel.operations[loose_axis_gate_label] = \
            _np.dot(_np.dot(_np.linalg.inv(modelAux1.operations[auxiliary_axis_gate_label]),
                            outputModel.operations[loose_axis_gate_label]),
                    modelAux1.operations[auxiliary_axis_gate_label])

    outputModel = outputModel.depolarize(op_noise=gate_depol,
                                         spam_noise=spam_depol)

    thetaTrue = _rpetools.extract_theta(outputModel, rpeconfig_inst)
    outputModel.thetaTrue = thetaTrue

    outputModel.alphaTrue = _rpetools.extract_alpha(outputModel, rpeconfig_inst)
    outputModel.alphaTrue = alpha_true

    outputModel.epsilonTrue = _rpetools.extract_epsilon(outputModel, rpeconfig_inst)
    outputModel.epsilonTrue = epsilon_true

    return outputModel

#def make_rpe_alpha_str_lists(k_list,angleStr,rpeconfig_inst):


def create_rpe_angle_circuit_lists(k_list, angle_name, rpeconfig_inst):
    """
    Make cosine and sine circuit lists.  These operation sequences are used to estimate the angle specified
    by angle_name ('alpha', 'epsilon', or 'theta')

    Parameters
    ----------
    k_list : list of ints
        The list of "germ powers" to be used.  Typically successive powers of
        two; e.g. [1,2,4,8,16].

    angle_name : string
        The angle to be deduced from these operation sequences.
        (Choices are 'alpha', 'epsilon', or 'theta')

    rpeconfig_inst : RPEconfig object
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

    if angle_name == 'alpha':
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

    elif angle_name == 'epsilon':
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

    elif angle_name == 'theta':
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
    for k in k_list:
        cosStrList += [_Circuit(cos_prep_tuple + cos_germ_tuple * k + cos_meas_tuple,
                                stringrep=cos_prep_str + '(' + cos_germ_str + ')^' + str(k) + cos_meas_str)]
        sinStrList += [_Circuit(sin_prep_tuple + sin_germ_tuple * k + sin_meas_tuple,
                                stringrep=sin_prep_str + '(' + sin_germ_str + ')^' + str(k) + sin_meas_str)]
    return cosStrList, sinStrList


def create_rpe_angle_circuits_dict(log2k_max_or_k_list, rpeconfig_inst):
    """
    Generates a dictionary that contains operation sequences for all RPE cosine and
    sine experiments for all three angles.

    Parameters
    ----------
    log2k_max_or_k_list : int or list
        int - log2(Maximum number of times to repeat an RPE germ)
        list - List of maximum number of times to repeat an RPE germ

    rpeconfig_inst : RPEconfig object
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
    if isinstance(log2k_max_or_k_list, int):
        kList = [2**k for k in range(log2k_max_or_k_list + 1)]
    else:
        kList = log2k_max_or_k_list
    alphaCosStrList, alphaSinStrList = create_rpe_angle_circuit_lists(kList, 'alpha', rpeconfig_inst)
    epsilonCosStrList, epsilonSinStrList = create_rpe_angle_circuit_lists(kList, 'epsilon', rpeconfig_inst)
    thetaCosStrList, thetaSinStrList = create_rpe_angle_circuit_lists(kList, 'theta', rpeconfig_inst)
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


def create_rpe_dataset(model_or_dataset, string_list_d, n_samples, sample_error='binomial', seed=None):
    """
    Generate a fake RPE DataSet using the probabilities obtained from a model.
    Is a thin wrapper for pygsti.data.simulate_data, changing
    default behavior of sample_error, and taking a dictionary of operation sequences
    as input.

    Parameters
    ----------
    model_or_dataset : Model or DataSet object
        If a Model, the model whose probabilities generate the data.
        If a DataSet, the data set whose frequencies generate the data.

    string_list_d : Dictionary of list of (tuples or Circuits)
        Each tuple or Circuit contains operation labels and
        specifies a gate sequence whose counts are included
        in the returned DataSet.  The dictionary must have the key
        'totalStrList'; easiest if this dictionary is generated by
        make_rpe_string_list_d.

    n_samples : int or list of ints or None
        The simulated number of samples for each operation sequence.  This only
        has effect when  sample_error == "binomial" or "multinomial".  If
        an integer, all operation sequences have this number of total samples. If
        a list, integer elements specify the number of samples for the
        corresponding operation sequence.  If None, then model_or_dataset must be
        a DataSet, and total counts are taken from it (on a per-circuit
        basis).

    sample_error : string, optional
        What type of sample error is included in the counts.  Can be:

        - "none"  - no sample error:
          counts are floating point numbers such that the exact
          probabilty can be found by the ratio of count / total.
        - "round" - same as "none", except counts are rounded to the nearest
          integer.
        - "binomial" - the number of counts is taken from a binomial
          distribution. Distribution has parameters p = probability of the
          operation sequence and n = number of samples.  This can only be used when
          there are exactly two outcome labels in model_or_dataset.
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
    return _data.simulate_data(model_or_dataset,
                               string_list_d['totalStrList'],
                               n_samples, sample_error=sample_error, seed=seed)

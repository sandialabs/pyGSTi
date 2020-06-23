""" Utility functions for RPE """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from scipy import optimize as _opt
from ...tools import decompose_gate_matrix as _decompose_gate_matrix

#from rpe_models import rpeInstanceDict


def extract_rotation_hat(xhat, yhat, k, nx, ny, angle_name="epsilon",
                         previous_angle=None, rpeconfig_inst=None):
    """
    For a single germ generation (k value), estimate the angle of rotation
    for either alpha, epsilon, or Phi.  (Warning:  Do not use for theta
    estimate without further processing!)

    Parameters
    ----------
    xhat : float
       The number of 0 counts for the sin string being used.

    yhat : float
       The number of 0 counts for the cos string being used.

    k : float
       The generation of experiments that xhat and yhat come from.

    nx : float
       The number of sin string clicks.

    ny : float
       The number cos string clicks.

    angle_name : { "alpha", "epsilon", "Phi" }, optional
      The angle to be extracted

    previous_angle : float, optional
       Angle estimate from previous generation; used to refine this
       generation's estimate.  Default is None (for estimation with no
       previous genereation's data)

    rpeconfig_inst : Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    alpha_j : float
        The current angle estimate.
    """

    if angle_name == 'alpha':
        arctan2Val = rpeconfig_inst.alpha_hat_func(xhat, yhat, nx, ny)
#            _np.arctan2((xhat-nx/2.)/nx,-(yhat-ny/2.)/ny)
    elif angle_name == 'epsilon':
        arctan2Val = rpeconfig_inst.epsilon_hat_func(xhat, yhat, nx, ny)
    elif angle_name == 'Phi':
        arctan2Val = rpeconfig_inst.Phi_hat_func(xhat, yhat, nx, ny)
#            arctan2Val = _np.arctan2((xhat-nx/2.)/nx,-(yhat-ny/2.)/ny)
    else:
        raise Exception('Need valid angle name!')

    if k != 1 and previous_angle is None:
        raise Exception('Need previous_angle!')
    if k == 1:
        #        return _np.arctan2((xhat-nx/2.)/nx,(yhat-ny/2.)/ny)
        return arctan2Val

    elif k > 1:
        #        angle_j = 1./k * _np.arctan2((xhat-nx/2.)/nx,(yhat-ny/2.)/ny)
        angle_j = 1. / k * arctan2Val
        while not (angle_j >= previous_angle - _np.pi / k
                   and angle_j <= previous_angle + _np.pi / k):
            if angle_j <= previous_angle - _np.pi / k:
                angle_j += 2 * _np.pi / k
            elif angle_j > previous_angle + _np.pi / k:
                angle_j -= 2 * _np.pi / k
            else:
                raise Exception('What?!')
        return angle_j


def estimate_angles(dataset, angle_sin_strs, angle_cos_strs, angle_name="epsilon",
                   length_list=None, rpeconfig_inst=None):
    """
    For a dataset containing sin and cos strings to estimate either alpha,
    epsilon, or Phi return a list of alpha, epsilon, or Phi estimates (one for
    each generation).  Note: this assumes the dataset contains '0' and
    '1' SPAM labels.

    Parameters
    ----------
    dataset : DataSet
       The dataset from which the angle estimates will be extracted.

    angle_sin_strs : list of Circuits
       The list of sin strs that the estimator will use.

    angle_cos_strs : list of Circuits
       The list of cos strs that the estimator will use.

    angle_name : { "alpha", "epsilon", "Phi" }, optional
      The angle to be extracted

    length_list : The list of sequence lengths.  Default is None;
        If None is specified, then length_list becomes [1,2,4,...,2**(len(angle_sin_strs)-1)]

    rpeconfig_inst : RPEconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    angleHatList : list of floats
        A list of angle estimates, ordered by generation (k).
    """
    angleTemp1 = None
    angleHatList = []
    genNum = len(angle_sin_strs)

    up_labels = rpeconfig_inst.up_labels
    dn_labels = rpeconfig_inst.dn_labels

    if length_list is None:
        length_list = [2**k for k in range(genNum)]
    for i, length in enumerate(length_list):
        xhatTemp = _np.sum(dataset[angle_sin_strs[i]][up_label] for up_label in up_labels)
        yhatTemp = _np.sum(dataset[angle_cos_strs[i]][up_label] for up_label in up_labels)
        Nx = xhatTemp + _np.sum(dataset[angle_sin_strs[i]][dn_label] for dn_label in dn_labels)
        Ny = yhatTemp + _np.sum(dataset[angle_cos_strs[i]][dn_label] for dn_label in dn_labels)
#        xhatTemp = dataset[angle_sin_strs[i]]['0']
#        yhatTemp = dataset[angle_cos_strs[i]]['0']
#        Nx = xhatTemp + dataset[angle_sin_strs[i]]['1']
#        Ny = yhatTemp + dataset[angle_cos_strs[i]]['1']
        angleTemp1 = extract_rotation_hat(xhatTemp, yhatTemp, length,
                                          Nx, Ny, angle_name, angleTemp1, rpeconfig_inst)
        angleHatList.append(angleTemp1)
    return angleHatList


def _sin_phi2(theta, phi, epsilon, rpeconfig_inst=None):
    """
    Returns the function whose zero, for fixed phi and epsilon, occurs at the
    desired value of theta. (This function exists to be passed to a minimizer
    to obtain theta.)

    Parameters
    ----------
    theta : float
       Angle between estimated "loose axis" and target "loose axis".

    phi : float
       The auxiliary angle phi; necessary to calculate theta.

    epsilon : float
       Angle of rotation about "loose axis".

    Returns
    -------
    sinPhi2FuncVal
        The value of _sin_phi2 for given inputs.  (Must be 0 to achieve "true" theta.)
    """

    newEpsilon = rpeconfig_inst.new_epsilon_func(epsilon)

    sinPhi2FuncVal = _np.abs(2 * _np.sin(theta) * _np.cos(_np.pi * newEpsilon / 2)
                             * _np.sqrt(1 - _np.sin(theta)**2
                                        * _np.cos(_np.pi * newEpsilon / 2)**2)
                             - _np.sin(phi / 2))
    return sinPhi2FuncVal


def estimate_thetas(dataset, angle_sin_strs, angle_cos_strs, epsilon_list,
                   return_phi_fun_list=False, rpeconfig_inst=None):
    """
    For a dataset containing sin and cos strings to estimate theta,
    along with already-made estimates of epsilon, return a list of theta
    (one for each generation).

    Parameters
    ----------
    dataset : DataSet
       The dataset from which the theta estimates will be extracted.

    angle_sin_strs : list of Circuits
       The list of sin strs that the estimator will use.

    angle_cos_strs : list of Circuits
       The list of cos strs that the estimator will use.

    epsilon_list : list of floats
       List of epsilon estimates.

    return_phi_fun_list : bool, optional
       Set to True to obtain measure of how well Eq. III.7 is satisfied.
       Default is False.

    rpeconfig_inst : RPEconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    thetaHatList : list of floats
        A list of theta estimates, ordered by generation (k).

    PhiFunList : list of floats
        A list of _sin_phi2 vals at optimal theta values.  If not close to
        0, constraints unsatisfiable.  Only returned if return_phi_fun_list is set
        to True.
    """

    PhiList = estimate_angles(dataset, angle_sin_strs, angle_cos_strs, 'Phi', rpeconfig_inst=rpeconfig_inst)
    thetaList = []
    PhiFunList = []
    for index, Phi in enumerate(PhiList):
        epsilon = epsilon_list[index]
        soln = _opt.minimize(lambda x: _sin_phi2(x, Phi, epsilon, rpeconfig_inst), 0)
        thetaList.append(soln['x'][0])
        PhiFunList.append(soln['fun'])
#        if soln['fun'] > 1e-2:
#            print Phi, epsilon
    if return_phi_fun_list:
        return thetaList, PhiFunList
    else:
        return thetaList


def extract_alpha(model, rpeconfig_inst):
    """
    For a given model, obtain the angle of rotation about the "fixed axis"

    WARNING:  This is a gauge-covariant parameter!  Gauge must be fixed prior
    to estimating.

    Parameters
    ----------
    model : Model
       The model whose angle of rotation about the fixed axis is to be calculated.

    rpeconfig_inst : RPEconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    alphaVal : float
        The value of alpha for the input model.
    """
    op_label = rpeconfig_inst.fixed_axis_gate_label
    decomp = _decompose_gate_matrix(model.operations[op_label])
    alphaVal = decomp['pi rotations'] * _np.pi
    return alphaVal


def extract_epsilon(model, rpeconfig_inst):
    """
    For a given model, obtain the angle of rotation about the "loose axis"

    WARNING:  This is a gauge-covariant parameter!  Gauge must be fixed prior
    to estimating.

    Parameters
    ----------
    model : Model
       The model whose angle of rotation about the "loose axis" is to be calculated.

    rpeconfig_inst : RPEconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    epsilonVal : float
        The value of epsilon for the input model.
    """
    op_label = rpeconfig_inst.loose_axis_gate_label
    decomp = _decompose_gate_matrix(model.operations[op_label])

    epsilonVal = decomp['pi rotations'] * _np.pi
    return epsilonVal


def extract_theta(model, rpeconfig_inst):
    """
    For a given model, obtain the angle between the estimated "loose axis" and
    the target "loose axis".

    WARNING:  This is a gauge-covariant parameter!  (I think!)  Gauge must be
    fixed prior to estimating.

    Parameters
    ----------
    model : Model
        The model whose loose axis misalignment is to be calculated.

    rpeconfig_inst : RPEconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    thetaVal : float
        The value of theta for the input model.
    """
    op_label = rpeconfig_inst.loose_axis_gate_label
    decomp = _decompose_gate_matrix(model.operations[op_label])
    target_axis = rpeconfig_inst.loose_axis_target

    decomp = _decompose_gate_matrix(model.operations[op_label])
    thetaVal = _np.real_if_close([_np.arccos(
        _np.dot(decomp['axis of rotation'], target_axis))])[0]
    if thetaVal > _np.pi / 2:
        thetaVal = _np.pi - thetaVal
    elif thetaVal < -_np.pi / 2:
        thetaVal = _np.pi + thetaVal
    return thetaVal


def consistency_check(angle_k, angle_final, k):
    """ Check internal consistency """
    wedge_size = _np.pi / (2 * k)
    angle_k += _np.pi
    angle_k = angle_k % (2 * _np.pi)
    angle_k -= _np.pi

    angle_final += _np.pi
    angle_final = angle_final % (2 * _np.pi)
    angle_final -= _np.pi

    if _np.abs(angle_k - angle_final) <= wedge_size:
        return 1.0
    elif _np.abs(angle_k - (angle_final + 2 * _np.pi)) <= wedge_size:
        return 1.0
    elif _np.abs(angle_k - (angle_final - 2 * _np.pi)) <= wedge_size:
        return 1.0
    else:
        return 0.0


def analyze_rpe_data(input_dataset, true_or_target_model, string_list_d, rpeconfig_inst, do_consistency_check=False,
                     k_list=None):
    """
    Compute angle estimates and compare to true or target values for alpha, epsilon,
    and theta.  ("True" will typically be used for simulated data, when the
    true angle values are known a priori; "target" will typically be used for
    experimental data, where we do not know the true angle values, and can
    only compare to our desired angles.)

    Parameters
    ----------
    input_dataset : DataSet
        The dataset containing the RPE experiments.

    true_or_target_model : Model
        The model used to generate the RPE data OR the target model.

    string_list_d : dict
       The dictionary of operation sequence lists used for the RPE experiments.
       This should be generated via make_rpe_string_list_d.

    rpeconfig_inst : RPEconfig object
        Declares which model configuration RPE should be trying to fit;
        determines particular functions and values to be used.

    Returns
    -------
    resultsD : dict
        A dictionary of the results
        The keys of the dictionary are:

        -'alphaHatList' : List (ordered by k) of alpha estimates.
        -'epsilonHatList' : List (ordered by k) of epsilon estimates.
        -'thetaHatList' : List (ordered by k) of theta estimates.
        -'alphaErrorList' : List (ordered by k) of difference between true
          alpha and RPE estimate of alpha.
        -'epsilonErrorList' : List (ordered by k) of difference between true
          epsilon and RPE estimate of epsilon.
        -'thetaErrorList' : List (ordered by k) of difference between true
          theta and RPE estimate of theta.
        -'PhiFunErrorList' : List (ordered by k) of _sin_phi2 values.

    """
    alphaCosStrList = string_list_d['alpha', 'cos']
    alphaSinStrList = string_list_d['alpha', 'sin']
    epsilonCosStrList = string_list_d['epsilon', 'cos']
    epsilonSinStrList = string_list_d['epsilon', 'sin']
    thetaCosStrList = string_list_d['theta', 'cos']
    thetaSinStrList = string_list_d['theta', 'sin']
    try:
        alphaTrue = true_or_target_model.alphaTrue
    except:
        alphaTrue = extract_alpha(true_or_target_model, rpeconfig_inst)
    try:
        epsilonTrue = true_or_target_model.epsilonTrue
    except:
        epsilonTrue = extract_epsilon(true_or_target_model, rpeconfig_inst)
    try:
        thetaTrue = true_or_target_model.thetaTrue
    except:
        thetaTrue = extract_theta(true_or_target_model, rpeconfig_inst)
    alphaErrorList = []
    epsilonErrorList = []
    thetaErrorList = []
#    PhiFunErrorList = []
    alphaHatList = estimate_angles(input_dataset,
                                  alphaSinStrList,
                                  alphaCosStrList, 'alpha', rpeconfig_inst=rpeconfig_inst)
    epsilonHatList = estimate_angles(input_dataset,
                                    epsilonSinStrList,
                                    epsilonCosStrList, 'epsilon', rpeconfig_inst=rpeconfig_inst)
    thetaHatList, PhiFunErrorList = estimate_thetas(input_dataset,
                                                   thetaSinStrList,
                                                   thetaCosStrList,
                                                   epsilonHatList, rpeconfig_inst=rpeconfig_inst,
                                                   return_phi_fun_list=True)
    for alphaTemp1 in alphaHatList:
        alphaErrorList.append(abs(alphaTrue - alphaTemp1))
    for epsilonTemp1 in epsilonHatList:
        epsilonErrorList.append(abs(epsilonTrue - epsilonTemp1))
#        print abs(_np.pi/2-abs(alphaTemp1))
    for thetaTemp1 in thetaHatList:
        thetaErrorList.append(abs(thetaTrue - thetaTemp1))
#    for PhiFunTemp1 in PhiFunList:
#        PhiFunErrorList.append(PhiFunTemp1)

    resultsD = {}

    if do_consistency_check:
        if k_list is None:
            raise ValueError("Consistency check requested, but no k List given!")
        else:
            num_ks = len(k_list)
            resultsD['alphaCheckMat'] = _np.zeros([num_ks, num_ks], float)
            resultsD['epsilonCheckMat'] = _np.zeros([num_ks, num_ks], float)
            resultsD['thetaCheckMat'] = _np.zeros([num_ks, num_ks], float)
            for k_final_ind, _ in enumerate(k_list):
                alpha_final_k = alphaHatList[k_final_ind]
                epsilon_final_k = epsilonHatList[k_final_ind]
                theta_final_k = thetaHatList[k_final_ind]
                k_list_temp = list(k_list[:k_final_ind + 1])
                for k_small_ind, k_small_val in enumerate(k_list_temp):
                    #                    print k_small_ind, k_small_val, k_final_ind, k_final_val, k_list_temp
                    alpha_small_k = alphaHatList[k_small_ind]
                    epsilon_small_k = epsilonHatList[k_small_ind]
                    theta_small_k = thetaHatList[k_small_ind]
                    resultsD['alphaCheckMat'][k_small_ind, k_final_ind] = consistency_check(
                        alpha_small_k, alpha_final_k, k_small_val)
                    resultsD['epsilonCheckMat'][k_small_ind, k_final_ind] = consistency_check(
                        epsilon_small_k, epsilon_final_k, k_small_val)
                    resultsD['thetaCheckMat'][k_small_ind, k_final_ind] = consistency_check(
                        theta_small_k, theta_final_k, k_small_val)

    resultsD['alphaHatList'] = alphaHatList
    resultsD['epsilonHatList'] = epsilonHatList
    resultsD['thetaHatList'] = thetaHatList
    resultsD['alphaErrorList'] = alphaErrorList
    resultsD['epsilonErrorList'] = epsilonErrorList
    resultsD['thetaErrorList'] = thetaErrorList
    resultsD['PhiFunErrorList'] = PhiFunErrorList
    return resultsD

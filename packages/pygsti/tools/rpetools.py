from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Utility functions for RPE """
import numpy as _np
from scipy import optimize as _opt
from . import gatetools as _gt


def extract_rotation_hat(xhat,yhat,k,Nx,Ny,angleName="epsilon",
                         previousAngle=None):
    """
    For a single germ generation (k value), estimate the angle of rotation
    for either alpha, epsilon, or Phi.  (Warning:  Do not use for theta
    estimate without further processing!)

    Parameters
    ----------
    xhat : float
       The number of plus counts for the sin string being used.

    yhat : float
       The number of plus counts for the cos string being used.

    k : float
       The generation of experiments that xhat and yhat come from.

    Nx : float
       The number of sin string clicks.

    Ny : float
       The number cos string clicks.

    angleName : { "alpha", "epsilon", "Phi" }, optional
      The angle to be extracted

    previousAngle : float, optional
       Angle estimate from previous generation; used to refine this
       generation's estimate.  Default is None (for estimation with no
       previous genereation's data)

    Returns
    -------
    alpha_j : float
        The current angle estimate.
    """

    if angleName == 'alpha':
        arctan2Val = _np.arctan2(-(xhat-Nx/2.)/Nx,(yhat-Ny/2.)/Ny)
    elif angleName == 'epsilon' or angleName == 'Phi':
        arctan2Val = _np.arctan2((xhat-Nx/2.)/Nx,-(yhat-Ny/2.)/Ny)
    else:
        raise Exception('Need valid angle name!')


    if k!=1 and previousAngle == None:
        raise Exception('Need previousAngle!')
    if k == 1:
#        return _np.arctan2((xhat-Nx/2.)/Nx,(yhat-Ny/2.)/Ny)
        return arctan2Val

    elif k>1:
#        angle_j = 1./k * _np.arctan2((xhat-Nx/2.)/Nx,(yhat-Ny/2.)/Ny)
        angle_j = 1./k * arctan2Val
        while not (angle_j > previousAngle - _np.pi/k and \
                   angle_j <= previousAngle + _np.pi/k):
            if angle_j <= previousAngle - _np.pi/k:
                angle_j += 2 * _np.pi/k
            elif angle_j > previousAngle + _np.pi/k:
                angle_j -= 2 * _np.pi/k
            else:
                raise Exception('What?!')
        return angle_j

def est_angle_list(DS,angleSinStrs,angleCosStrs,angleName="epsilon"):
    """
    For a dataset containing sin and cos strings to estimate either alpha,
    epsilon, or Phi return a list of alpha, epsilon, or Phi estimates (one for
    each generation).  Note: this assumes the dataset contains 'plus' and
    'minus' SPAM labels.

    WARNING:  At present, kList must be of form [1,2,4,...,2**log2kMax].

    Parameters
    ----------
    DS : DataSet
       The dataset from which the angle estimates will be extracted.

    angleSinStrs : list of GateStrings
       The list of sin strs that the estimator will use.

    angleCosStrs : list of GateStrings
       The list of cos strs that the estimator will use.

    angleName : { "alpha", "epsilon", "Phi" }, optional
      The angle to be extracted

    Returns
    -------
    angleHatList : list of floats
        A list of angle estimates, ordered by generation (k).
    """
    angleTemp1 = None
    angleHatList = []
    genNum = len(angleSinStrs)
    for i in range(genNum):
        xhatTemp = DS[angleSinStrs[i]]['plus']
        yhatTemp = DS[angleCosStrs[i]]['plus']
        Nx = xhatTemp + DS[angleSinStrs[i]]['minus']
        Ny = yhatTemp + DS[angleCosStrs[i]]['minus']
        angleTemp1 = extract_rotation_hat(xhatTemp,yhatTemp,2**i,
                                          Nx,Ny,angleName,angleTemp1)
        angleHatList.append(angleTemp1)
    return angleHatList

def sin_phi2_func(theta,Phi,epsilon):
    """
    Returns the function whose zero, for fixed Phi and epsilon, occurs at the
    desired value of theta. (This function exists to be passed to a minimizer
    to obtain theta.)

    WARNING:  epsilon gets rescaled to newEpsilon, by dividing by pi/4; will
    have to change for epsilons far from pi/4.

    Parameters
    ----------
    theta : float
       Angle between X and Z axes.

    Phi : float
       The auxiliary angle Phi; necessary to calculate theta.

    epsilon : float
       Angle of X rotation.

    Returns
    -------
    sinPhi2FuncVal
        The value of sin_phi2_func for given inputs.  (Must be 0 to achieve "true" theta.)
    """
    newEpsilon = (epsilon / (_np.pi/4)) - 1
    sinPhi2FuncVal = _np.abs(2*_np.sin(theta) * _np.cos(_np.pi*newEpsilon/2)*
                            _np.sqrt(1-_np.sin(theta)**2*
                                    _np.cos(_np.pi*newEpsilon/2)**2)
                            - _np.sin(Phi/2))
    return sinPhi2FuncVal

def est_theta_list(DS,angleSinStrs,angleCosStrs,epsilonList,returnPhiFunList = False):
    """
    For a dataset containing sin and cos strings to estimate theta,
    along with already-made estimates of epsilon, return a list of theta
    (one for each generation).

    Parameters
    ----------
    DS : DataSet
       The dataset from which the theta estimates will be extracted.

    angleSinStrs : list of GateStrings
       The list of sin strs that the estimator will use.

    angleCosStrs : list of GateStrings
       The list of cos strs that the estimator will use.

    epsilonList : list of floats
       List of epsilon estimates.

    returnPhiFunList : bool, optional
       Set to True to obtain measure of how well Eq. III.7 is satisfied.
       Default is False.

    Returns
    -------
    thetaHatList : list of floats
        A list of theta estimates, ordered by generation (k).

    PhiFunList : list of floats
        A list of sin_phi2_func vals at optimal theta values.  If not close to
        0, constraints unsatisfiable.  Only returned if returnPhiFunList is set
        to True.
    """

    PhiList = est_angle_list(DS,angleSinStrs,angleCosStrs,'Phi')
    thetaList = []
    PhiFunList = []
    for index, Phi in enumerate(PhiList):
        epsilon = epsilonList[index]
        soln = _opt.minimize(lambda x: sin_phi2_func(x,Phi,epsilon),0)
        thetaList.append(soln['x'][0])
        PhiFunList.append(soln['fun'])
#        if soln['fun'] > 1e-2:
#            print Phi, epsilon
    if returnPhiFunList:
        return thetaList, PhiFunList
    else:
        return thetaList


def extract_alpha(gateset):
    """
    For a given gateset, obtain the angle of rotation about Z axis
    (for gate "Gz").

    WARNING:  This is a gauge-covariant parameter!  Gauge must be fixed prior
    to estimating.

    Parameters
    ----------
    gateset : GateSet
       The gateset whose "Gz" angle of rotation is to be calculated.

    Returns
    -------
    alphaVal : float
        The value of alpha for the input gateset.
    """
    decomp = _gt.decompose_gate_matrix( gateset.gates['Gz'] )
    alphaVal = decomp['pi rotations'] * _np.pi
    return alphaVal

def extract_epsilon(gateset):
    """
    For a given gateset, obtain the angle of rotation about X axis
    (for gate "Gx").

    WARNING:  This is a gauge-covariant parameter!  Gauge must be fixed prior
    to estimating.

    Parameters
    ----------
    gateset : GateSet
       The gateset whose "Gx" angle of rotation is to be calculated.

    Returns
    -------
    epsilonVal : float
        The value of epsilon for the input gateset.
    """
    decomp = _gt.decompose_gate_matrix( gateset.gates['Gx'] )
    epsilonVal = decomp['pi rotations'] * _np.pi
    return epsilonVal

def extract_theta(gateset):
    """
    For a given gateset, obtain the angle between the "X axis of rotation" and
    the "true" X axis (perpendicular to Z). (Angle of misalignment between "Gx"
    axis of rotation and X axis as defined by "Gz".)

    WARNING:  This is a gauge-covariant parameter!  (I think!)  Gauge must be
    fixed prior to estimating.

    Parameters
    ----------
    gateset : GateSet
       The gateset whose X axis misaligment is to be calculated.

    Returns
    -------
    thetaVal : float
        The value of theta for the input gateset.
    """
    decomp = _gt.decompose_gate_matrix( gateset.gates['Gx'] )
    thetaVal =  _np.real_if_close( [ _np.arccos(
                _np.dot(decomp['axis of rotation'], [0,1,0,0]))])[0]
    if thetaVal > _np.pi/2:
        thetaVal = _np.pi - thetaVal
    elif thetaVal < -_np.pi/2:
        thetaVal = _np.pi + thetaVal
    return thetaVal


def analyze_simulated_rpe_experiment(inputDataset,trueGateset,stringListD):
    """
    Compute angle estimates and compare to true estimates for alpha, epsilon,
    and theta.

    Parameters
    ----------
    inputDataset : DataSet
       The dataset containing the RPE experiments.

    trueGateset : GateSet
       The gateset used to generate the RPE data.

    stringListD : dict
       The dictionary of gate string lists used for the RPE experiments.
       This should be generated via make_rpe_string_list_d.

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
        -'PhiFunErrorList' : List (ordered by k) of sin_phi2_func values.

    """
    alphaCosStrList = stringListD['alpha','cos']
    alphaSinStrList = stringListD['alpha','sin']
    epsilonCosStrList = stringListD['epsilon','cos']
    epsilonSinStrList = stringListD['epsilon','sin']
    thetaCosStrList = stringListD['theta','cos']
    thetaSinStrList = stringListD['theta','sin']
    try:
        alphaTrue = trueGateset.alphaTrue
    except:
        alphaTrue = extract_alpha(trueGateset)
    try:
        epsilonTrue = trueGateset.epsilonTrue
    except:
        epsilonTrue = extract_epsilon(trueGateset)
    try:
        thetaTrue = trueGateset.thetaTrue
    except:
        thetaTrue = extract_theta(trueGateset)
    alphaErrorList = []
    epsilonErrorList = []
    thetaErrorList = []
#    PhiFunErrorList = []
    alphaHatList = est_angle_list(inputDataset,
                                  alphaSinStrList,
                                  alphaCosStrList,'alpha')
    epsilonHatList = est_angle_list(inputDataset,
                                    epsilonSinStrList,
                                    epsilonCosStrList, 'epsilon')
    thetaHatList,PhiFunErrorList = est_theta_list(inputDataset,
                                                  thetaSinStrList,
                                                  thetaCosStrList,
                                                  epsilonHatList,
                                                  returnPhiFunList=True)
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
    resultsD['alphaHatList'] = alphaHatList
    resultsD['epsilonHatList'] = epsilonHatList
    resultsD['thetaHatList'] = thetaHatList
    resultsD['alphaErrorList'] = alphaErrorList
    resultsD['epsilonErrorList'] = epsilonErrorList
    resultsD['thetaErrorList'] = thetaErrorList
    resultsD['PhiFunErrorList'] = PhiFunErrorList
    return resultsD

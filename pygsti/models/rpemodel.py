#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Functions for creating RPE Models
"""
import numpy as _np

from pygsti.models import modelconstruction as _setc
from pygsti import tools as _tools
from pygsti.modelmembers import operations as _op


def make_rpe_model(alpha_true, epsilon_true, y_rot, spam_depol, gate_depol=None, with_id=True):
    """
    Make a model for simulating RPE, paramaterized by rotation angles.

    Note that the output model also has thetaTrue, alpha_true, and epsilon_true
    added attributes.

    Parameters
    ----------
    alpha_true : float
        Angle of Z rotation (canonical RPE requires alpha_true to be close to
        pi/2).

    epsilon_true : float
        Angle of X rotation (canonical RPE requires epsilon_true to be close to
        pi/4).

    y_rot : float
        Angle of rotation about Y axis that, by similarity transformation,
        rotates X rotation.

    spam_depol : float
        Amount to depolarize SPAM by.

    gate_depol : float, optional
        Amount to depolarize gates by (defaults to None).

    with_id : bool, optional
        Do we include (perfect) identity or no identity? (Defaults to False;
        should be False for RPE, True for GST)

    Returns
    -------
    Model
        The desired model for RPE; model also has attributes thetaTrue,
        alpha_true, and epsilon_true, automatically extracted.
    """

    if with_id:
        outputModel = _setc.create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gx', 'Gz'],
            ["I(Q0)", "X(%s,Q0)" % epsilon_true, "Z(%s,Q0)" % alpha_true],
            prep_labels=["rho0"], prep_expressions=["0"],
            effect_labels=["E0", "Ec"], effect_expressions=["0", "complement"],
            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})
    else:
        outputModel = _setc.create_explicit_model_from_expressions(
            [('Q0',)], ['Gx', 'Gz'],
            ["X(%s,Q0)" % epsilon_true, "Z(%s,Q0)" % alpha_true],
            prep_labels=["rho0"], prep_expressions=["0"],
            effect_labels=["E0", "Ec"], effect_expressions=["0", "complement"],
            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

    if y_rot != 0:
        modelAux1 = _setc.create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gy', 'Gz'],
            ["I(Q0)", "Y(%s,Q0)" % y_rot, "Z(pi/2,Q0)"],
            prep_labels=["rho0"], prep_expressions=["0"],
            effect_labels=["E0", "Ec"], effect_expressions=["0", "complement"],
            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

        outputModel.operations['Gx'] = _op.FullArbitraryOp(_np.dot(_np.dot(_np.linalg.inv(modelAux1.operations['Gy']),
                                                                           outputModel.operations['Gx']),
                                                                   modelAux1.operations['Gy']))

    outputModel = outputModel.depolarize(op_noise=gate_depol,
                                         spam_noise=spam_depol)

    thetaTrue = _tools.rpe.extract_theta(outputModel)
    outputModel.thetaTrue = thetaTrue

    outputModel.alphaTrue = _tools.rpe.extract_alpha(outputModel)
    outputModel.alphaTrue = alpha_true

    outputModel.epsilonTrue = _tools.rpe.extract_epsilon(outputModel)
    outputModel.epsilonTrue = epsilon_true

    return outputModel


def rpe_ensemble_test(alpha_true, epsilon_true, y_rot, spam_depol, log2k_max, n, runs):
    """
    Experimental test function
    """
    from pygsti.circuits.rpecircuits import make_rpe_alpha_str_lists_gx_gz, make_rpe_theta_str_lists_gx_gz, \
        make_rpe_epsilon_str_lists_gx_gz
    import pygsti.data as _dsc
    kList = [2**k for k in range(log2k_max + 1)]

    alphaCosStrList, alphaSinStrList = make_rpe_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_rpe_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_rpe_theta_str_lists_gx_gz(kList)

    #percentAlphaError = 100*_np.abs((_np.pi/2-alpha_true)/alpha_true)
    #percentEpsilonError = 100*_np.abs((_np.pi/4 - epsilon_true)/epsilon_true)

    simModel = _setc.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gz'],
                                                            ["I(Q0)", "X(" + str(epsilon_true) + ",Q0)",
                                                             "Z(" + str(alpha_true) + ",Q0)"],
                                                            prep_labels=["rho0"], prep_expressions=["0"],
                                                            effect_labels=["E0", "Ec"],
                                                            effect_expressions=["0", "complement"],
                                                            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

    modelAux1 = _setc.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gy', 'Gz'],
                                                             ["I(Q0)", "Y(" + str(y_rot) + ",Q0)", "Z(pi/2,Q0)"],
                                                             prep_labels=["rho0"], prep_expressions=["0"],
                                                             effect_labels=["E0", "Ec"],
                                                             effect_expressions=["0", "complement"],
                                                             spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

    simModel.operations['Gx'] = _op.FullArbitraryOp(
        _np.dot(_np.dot(_np.linalg.inv(modelAux1.operations['Gy']), simModel.operations['Gx']),
                modelAux1.operations['Gy']))

    simModel = simModel.depolarize(spam_noise=spam_depol)

    thetaTrue = _tools.rpe.extract_theta(simModel)

    #SPAMerror = _np.dot(simModel.effects['E0'].T,simModel.preps['rho0'])[0,0]

    jMax = runs

    alphaHatListArray = _np.zeros([jMax, log2k_max + 1], dtype='object')
    epsilonHatListArray = _np.zeros([jMax, log2k_max + 1], dtype='object')
    thetaHatListArray = _np.zeros([jMax, log2k_max + 1], dtype='object')

    alphaErrorArray = _np.zeros([jMax, log2k_max + 1], dtype='object')
    epsilonErrorArray = _np.zeros([jMax, log2k_max + 1], dtype='object')
    thetaErrorArray = _np.zeros([jMax, log2k_max + 1], dtype='object')
    PhiFunErrorArray = _np.zeros([jMax, log2k_max + 1], dtype='object')

    for j in range(jMax):
        simDS = _dsc.simulate_data(
            simModel, alphaCosStrList + alphaSinStrList + epsilonCosStrList
            + epsilonSinStrList + thetaCosStrList + thetaSinStrList,
            n, sample_error='binomial', seed=j)
        alphaErrorList = []
        epsilonErrorList = []
        thetaErrorList = []
        PhiFunErrorList = []
        alphaHatList = _tools.rpe.estimate_angles(simDS, alphaSinStrList,
                                                  alphaCosStrList, 'alpha')
        epsilonHatList = _tools.rpe.estimate_angles(simDS, epsilonSinStrList,
                                                    epsilonCosStrList, 'epsilon')
        thetaHatList, PhiFunList = _tools.rpe.estimate_thetas(simDS, thetaSinStrList,
                                                              thetaCosStrList, epsilonHatList,
                                                              return_phi_fun_list=True)
        for alphaTemp1 in alphaHatList:
            alphaErrorList.append(abs(alpha_true - alphaTemp1))
        for epsilonTemp1 in epsilonHatList:
            epsilonErrorList.append(abs(epsilon_true - epsilonTemp1))
    #        print abs(_np.pi/2-abs(alphaTemp1))
        for thetaTemp1 in thetaHatList:
            thetaErrorList.append(abs(thetaTrue - thetaTemp1))
        for PhiFunTemp1 in PhiFunList:
            PhiFunErrorList.append(PhiFunTemp1)

        alphaErrorArray[j, :] = _np.array(alphaErrorList)
        epsilonErrorArray[j, :] = _np.array(epsilonErrorList)
        thetaErrorArray[j, :] = _np.array(thetaErrorList)
        PhiFunErrorArray[j, :] = _np.array(PhiFunErrorList)

        alphaHatListArray[j, :] = _np.array(alphaHatList)
        epsilonHatListArray[j, :] = _np.array(epsilonHatList)
        thetaHatListArray[j, :] = _np.array(thetaHatList)

    #print "True alpha:",alpha_true
    #print "True alpha:",alpha_true
    #print "True alpha:",alpha_true
    #print "True alpha:",alpha_true
    #print "% true alpha deviation from target:", percentAlphaError

    outputDict = {}
#    outputDict['alphaArray'] = alphaHatListArray
#    outputDict['alphaErrorArray'] = alphaErrorArray
#    outputDict['epsilonArray'] = epsilonHatListArray
#    outputDict['epsilonErrorArray'] = epsilonErrorArray
#    outputDict['thetaArray'] = thetaHatListArray
#    outputDict['thetaErrorArray'] = thetaErrorArray
#    outputDict['PhiFunErrorArray'] = PhiFunErrorArray
#    outputDict['alpha'] = alpha_true
#    outputDict['epsilon_true'] = epsilon_true
#    outputDict['thetaTrue'] = thetaTrue
#    outputDict['y_rot'] = y_rot
#    outputDict['spam_depol'] = spam_depol#Input value to depolarize SPAM by
#    outputDict['SPAMerror'] = SPAMerror#<<E|rho>>
#    outputDict['mdl'] = simModel
#    outputDict['n'] = n

    return outputDict

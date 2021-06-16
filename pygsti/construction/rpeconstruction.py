#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Functions for creating RPE Models and Circuit lists
"""
import numpy as _np

from pygsti.construction import datasetconstruction as _dsc
from pygsti.construction import modelconstruction as _setc
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti import tools as _tools
from pygsti.modelmembers import operations as _op


def make_parameterized_rpe_gate_set(alpha_true, epsilon_true, y_rot, spam_depol,
                                    gate_depol=None, with_id=True):
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
        outputModel = _setc.create_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gz'],
            ["I(Q0)", "X(%s,Q0)" % epsilon_true, "Z(%s,Q0)" % alpha_true],
            prep_labels=["rho0"], prep_expressions=["0"],
            effect_labels=["E0", "Ec"], effect_expressions=["0", "complement"],
            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})
    else:
        outputModel = _setc.create_explicit_model(
            [('Q0',)], ['Gx', 'Gz'],
            ["X(%s,Q0)" % epsilon_true, "Z(%s,Q0)" % alpha_true],
            prep_labels=["rho0"], prep_expressions=["0"],
            effect_labels=["E0", "Ec"], effect_expressions=["0", "complement"],
            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

    if y_rot != 0:
        modelAux1 = _setc.create_explicit_model(
            [('Q0',)], ['Gi', 'Gy', 'Gz'],
            ["I(Q0)", "Y(%s,Q0)" % y_rot, "Z(pi/2,Q0)"],
            prep_labels=["rho0"], prep_expressions=["0"],
            effect_labels=["E0", "Ec"], effect_expressions=["0", "complement"],
            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

        outputModel.operations['Gx'] = _op.FullArbitraryOp(
            _np.dot(_np.dot(_np.linalg.inv(modelAux1.operations['Gy']),
                            outputModel.operations['Gx']), modelAux1.operations['Gy']))

    outputModel = outputModel.depolarize(op_noise=gate_depol,
                                         spam_noise=spam_depol)

    thetaTrue = _tools.rpe.extract_theta(outputModel)
    outputModel.thetaTrue = thetaTrue

    outputModel.alphaTrue = _tools.rpe.extract_alpha(outputModel)
    outputModel.alphaTrue = alpha_true

    outputModel.epsilonTrue = _tools.rpe.extract_epsilon(outputModel)
    outputModel.epsilonTrue = epsilon_true

    return outputModel


def make_rpe_alpha_str_lists_gx_gz(k_list):
    """
    Make alpha cosine and sine circuit lists for (approx) X pi/4 and Z pi/2 gates.

    These circuits are used to estimate alpha (Z rotation angle).

    Parameters
    ----------
    k_list : list of ints
        The list of "germ powers" to be used.  Typically successive powers of
        two; e.g. [1,2,4,8,16].

    Returns
    -------
    cosStrList : list of Circuits
        The list of "cosine strings" to be used for alpha estimation.
    sinStrList : list of Circuits
        The list of "sine strings" to be used for alpha estimation.
    """
    cosStrList = []
    sinStrList = []
    for k in k_list:
        cosStrList += [_Circuit(('Gi', 'Gx', 'Gx', 'Gz')
                                + ('Gz',) * k
                                + ('Gz', 'Gz', 'Gz', 'Gx', 'Gx'),
                                'GiGxGxGzGz^' + str(k) + 'GzGzGzGxGx')]

        sinStrList += [_Circuit(('Gx', 'Gx', 'Gz', 'Gz')
                                + ('Gz',) * k
                                + ('Gz', 'Gz', 'Gz', 'Gx', 'Gx'),
                                'GxGxGzGzGz^' + str(k) + 'GzGzGzGxGx')]

        #From RPEToolsNewNew.py
        ##cosStrList += [_Circuit(('Gi','Gx','Gx')+
        ##                                ('Gz',)*k +
        ##                                ('Gx','Gx'),
        ##                                'GiGxGxGz^'+str(k)+'GxGx')]
        #
        #
        #cosStrList += [_Circuit(('Gx','Gx')+
        #                                ('Gz',)*k +
        #                                ('Gx','Gx'),
        #                                'GxGxGz^'+str(k)+'GxGx')]
        #
        #
        #sinStrList += [_Circuit(('Gx','Gx')+
        #                                ('Gz',)*k +
        #                                ('Gz','Gx','Gx'),
        #                                'GxGxGz^'+str(k)+'GzGxGx')]

    return cosStrList, sinStrList


def make_rpe_epsilon_str_lists_gx_gz(k_list):
    """
    Make epsilon cosine and sine circuit lists for (approx) X pi/4 and Z pi/2 gates.

    These circuits are used to estimate epsilon (X rotation angle).

    Parameters
    ----------
    k_list : list of ints
        The list of "germ powers" to be used.  Typically successive powers of
        two; e.g. [1,2,4,8,16].

    Returns
    -------
    epsilonCosStrList : list of Circuits
        The list of "cosine strings" to be used for epsilon estimation.
    epsilonSinStrList : list of Circuits
        The list of "sine strings" to be used for epsilon estimation.
    """
    epsilonCosStrList = []
    epsilonSinStrList = []

    for k in k_list:
        epsilonCosStrList += [_Circuit(('Gx',) * k
                                       + ('Gx',) * 4,
                                       'Gx^' + str(k) + 'GxGxGxGx')]

        epsilonSinStrList += [_Circuit(('Gx', 'Gx', 'Gz', 'Gz')
                                       + ('Gx',) * k
                                       + ('Gx',) * 4,
                                       'GxGxGzGzGx^' + str(k) + 'GxGxGxGx')]

        #From RPEToolsNewNew.py
        #epsilonCosStrList += [_Circuit(('Gx',)*k,
        #                                       'Gx^'+str(k))]
        #
        #epsilonSinStrList += [_Circuit(('Gx','Gx')+('Gx',)*k,
        #                                       'GxGxGx^'+str(k))]

    return epsilonCosStrList, epsilonSinStrList


def make_rpe_theta_str_lists_gx_gz(k_list):
    """
    Make theta cosine and sine circuit lists for (approx) X pi/4 and Z pi/2 gates.

    These circuits are used to estimate theta (X-Z axes angle).

    Parameters
    ----------
    k_list : list of ints
        The list of "germ powers" to be used.  Typically successive powers of
        two; e.g. [1,2,4,8,16].

    Returns
    -------
    thetaCosStrList : list of Circuits
        The list of "cosine strings" to be used for theta estimation.
    thetaSinStrList : list of Circuits
        The list of "sine strings" to be used for theta estimation.
    """
    thetaCosStrList = []
    thetaSinStrList = []

    for k in k_list:
        thetaCosStrList += [_Circuit(
            ('Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz', 'Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz') * k
            + ('Gx',) * 4, '(GzGxGxGxGxGzGzGxGxGxGxGz)^' + str(k) + 'GxGxGxGx')]

        thetaSinStrList += [_Circuit(
            ('Gx', 'Gx', 'Gz', 'Gz')
            + ('Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz', 'Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz') * k
            + ('Gx',) * 4,
            '(GxGxGzGz)(GzGxGxGxGxGzGzGxGxGxGxGz)^' + str(k) + 'GxGxGxGx')]

        #From RPEToolsNewNew.py
        #thetaCosStrList += [_Circuit(
        #       ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k,
        #       '(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k))]
        #
        #thetaSinStrList += [_Circuit(
        #       ('Gx','Gx')+
        #       ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k,
        #       'GxGx(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k))]

    return thetaCosStrList, thetaSinStrList


def make_rpe_string_list_d(log2k_max):
    """
    Creates a dictionary containing all the circuits needed for RPE.

    This includes circuits for all RPE cosine and sine experiments for all three angles.

    Parameters
    ----------
    log2k_max : int
        Maximum number of times to repeat an RPE "germ"

    Returns
    -------
    totalStrListD : dict
        A dictionary containing all circuits for all sine and cosine
        experiments for alpha, epsilon, and theta.
        The keys of the returned dictionary are:

        - 'alpha','cos' : List of circuits for cosine experiments used
          to determine alpha.
        - 'alpha','sin' : List of circuits for sine experiments used to
          determine alpha.
        - 'epsilon','cos' : List of circuits for cosine experiments used to
           determine epsilon.
        - 'epsilon','sin' : List of circuits for sine experiments used to
          determine epsilon.
        - 'theta','cos' : List of circuits for cosine experiments used to
          determine theta.
        - 'theta','sin' : List of circuits for sine experiments used to
          determine theta.
        - 'totalStrList' : All above circuits combined into one list;
          duplicates removed.
    """
    kList = [2**k for k in range(log2k_max + 1)]
    alphaCosStrList, alphaSinStrList = make_rpe_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_rpe_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_rpe_theta_str_lists_gx_gz(kList)
    totalStrList = alphaCosStrList + alphaSinStrList + \
        epsilonCosStrList + epsilonSinStrList + \
        thetaCosStrList + thetaSinStrList
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


def make_rpe_data_set(model_or_dataset, string_list_d, num_samples, sample_error='binomial', seed=None):
    """
    Generate a fake RPE DataSet using the probabilities obtained from a model.

    Is a thin wrapper for pygsti.construction.simulate_data, changing
    default behavior of sample_error, and taking a dictionary of circuits
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

    num_samples : int or list of ints or None
        The simulated number of samples for each circuit.  This only
        has effect when  sample_error == "binomial" or "multinomial".  If
        an integer, all circuits have this number of total samples. If
        a list, integer elements specify the number of samples for the
        corresponding circuit.  If None, then model_or_dataset must be
        a DataSet, and total counts are taken from it (on a per-circuit
        basis).

    sample_error : string, optional
        What type of sample error is included in the counts.  Can be:

        - "none"  - no sampl error:
          counts are floating point numbers such that the exact
          probabilty can be found by the ratio of count / total.
        - "round" - same as "none", except counts are rounded to the nearest
          integer.
        - "binomial" - the number of counts is taken from a binomial
          distribution. Distribution has parameters p = probability of the
          circuit and n = number of samples.  This can only be used when
          there are exactly two SPAM labels in model_or_dataset.
        - "multinomial" - counts are taken from a multinomial distribution.
          Distribution has parameters p_k = probability of the circuit
          using the k-th SPAM label and n = number of samples.  This should not
          be used for RPE.

    seed : int, optional
        If not None, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    Returns
    -------
    DataSet
        A static data set filled with counts for the specified circuits.
    """
    return _dsc.simulate_data(model_or_dataset,
                              string_list_d['totalStrList'],
                              num_samples, sample_error=sample_error, seed=seed)


def rpe_ensemble_test(alpha_true, epsilon_true, y_rot, spam_depol, log2k_max, n, runs):
    """
    Experimental test function
    """
    kList = [2**k for k in range(log2k_max + 1)]

    alphaCosStrList, alphaSinStrList = make_rpe_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_rpe_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_rpe_theta_str_lists_gx_gz(kList)

    #percentAlphaError = 100*_np.abs((_np.pi/2-alpha_true)/alpha_true)
    #percentEpsilonError = 100*_np.abs((_np.pi/4 - epsilon_true)/epsilon_true)

    simModel = _setc.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gz'],
                                           ["I(Q0)", "X(" + str(epsilon_true) + ",Q0)",
                                            "Z(" + str(alpha_true) + ",Q0)"],
                                           prep_labels=["rho0"], prep_expressions=["0"],
                                           effect_labels=["E0", "Ec"], effect_expressions=["0", "complement"],
                                           spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

    modelAux1 = _setc.create_explicit_model([('Q0',)], ['Gi', 'Gy', 'Gz'],
                                            ["I(Q0)", "Y(" + str(y_rot) + ",Q0)", "Z(pi/2,Q0)"],
                                            prep_labels=["rho0"], prep_expressions=["0"],
                                            effect_labels=["E0", "Ec"], effect_expressions=["0", "complement"],
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

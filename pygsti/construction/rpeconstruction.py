#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
""" Functions for creating RPE Models and Circuit lists """
import numpy as _np
from . import modelconstruction as _setc
from . import datasetconstruction as _dsc

from .. import objects as _objs
from .. import tools as _tools


def make_parameterized_rpe_gate_set(alphaTrue, epsilonTrue, Yrot, SPAMdepol,
                                    gateDepol=None, withId=True):
    """
    Make a model for simulating RPE, paramaterized by rotation angles.  Note
    that the output model also has thetaTrue, alphaTrue, and epsilonTrue
    added attributes.

    Parameters
    ----------
    alphaTrue : float
       Angle of Z rotation (canonical RPE requires alphaTrue to be close to
       pi/2).

    epsilonTrue : float
       Angle of X rotation (canonical RPE requires epsilonTrue to be close to
       pi/4).

    Yrot : float
       Angle of rotation about Y axis that, by similarity transformation,
       rotates X rotation.

    SPAMdepol : float
       Amount to depolarize SPAM by.

    gateDepol : float, optional
       Amount to depolarize gates by (defaults to None).

    withId : bool, optional
       Do we include (perfect) identity or no identity? (Defaults to False;
       should be False for RPE, True for GST)

    Returns
    -------
    Model
        The desired model for RPE; model also has attributes thetaTrue,
        alphaTrue, and epsilonTrue, automatically extracted.
    """

    if withId:
        outputModel = _setc.build_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gz'],
            ["I(Q0)", "X(%s,Q0)" % epsilonTrue, "Z(%s,Q0)" % alphaTrue],
            prepLabels=["rho0"], prepExpressions=["0"],
            effectLabels=["E0", "Ec"], effectExpressions=["0", "complement"],
            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})
    else:
        outputModel = _setc.build_explicit_model(
            [('Q0',)], ['Gx', 'Gz'],
            ["X(%s,Q0)" % epsilonTrue, "Z(%s,Q0)" % alphaTrue],
            prepLabels=["rho0"], prepExpressions=["0"],
            effectLabels=["E0", "Ec"], effectExpressions=["0", "complement"],
            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

    if Yrot != 0:
        modelAux1 = _setc.build_explicit_model(
            [('Q0',)], ['Gi', 'Gy', 'Gz'],
            ["I(Q0)", "Y(%s,Q0)" % Yrot, "Z(pi/2,Q0)"],
            prepLabels=["rho0"], prepExpressions=["0"],
            effectLabels=["E0", "Ec"], effectExpressions=["0", "complement"],
            spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

        outputModel.operations['Gx'] = _objs.FullDenseOp(
            _np.dot(_np.dot(_np.linalg.inv(modelAux1.operations['Gy']),
                            outputModel.operations['Gx']), modelAux1.operations['Gy']))

    outputModel = outputModel.depolarize(op_noise=gateDepol,
                                         spam_noise=SPAMdepol)

    thetaTrue = _tools.rpe.extract_theta(outputModel)
    outputModel.thetaTrue = thetaTrue

    outputModel.alphaTrue = _tools.rpe.extract_alpha(outputModel)
    outputModel.alphaTrue = alphaTrue

    outputModel.epsilonTrue = _tools.rpe.extract_epsilon(outputModel)
    outputModel.epsilonTrue = epsilonTrue

    return outputModel


def make_rpe_alpha_str_lists_gx_gz(kList):
    """
    Make alpha cosine and sine circuit lists for (approx) X pi/4 and Z pi/2
    gates. These operation sequences are used to estimate alpha (Z rotation angle).

    Parameters
    ----------
    kList : list of ints
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
    for k in kList:
        cosStrList += [_objs.Circuit(('Gi', 'Gx', 'Gx', 'Gz')
                                     + ('Gz',) * k
                                     + ('Gz', 'Gz', 'Gz', 'Gx', 'Gx'),
                                     'GiGxGxGzGz^' + str(k) + 'GzGzGzGxGx')]

        sinStrList += [_objs.Circuit(('Gx', 'Gx', 'Gz', 'Gz')
                                     + ('Gz',) * k
                                     + ('Gz', 'Gz', 'Gz', 'Gx', 'Gx'),
                                     'GxGxGzGzGz^' + str(k) + 'GzGzGzGxGx')]

        #From RPEToolsNewNew.py
        ##cosStrList += [_objs.Circuit(('Gi','Gx','Gx')+
        ##                                ('Gz',)*k +
        ##                                ('Gx','Gx'),
        ##                                'GiGxGxGz^'+str(k)+'GxGx')]
        #
        #
        #cosStrList += [_objs.Circuit(('Gx','Gx')+
        #                                ('Gz',)*k +
        #                                ('Gx','Gx'),
        #                                'GxGxGz^'+str(k)+'GxGx')]
        #
        #
        #sinStrList += [_objs.Circuit(('Gx','Gx')+
        #                                ('Gz',)*k +
        #                                ('Gz','Gx','Gx'),
        #                                'GxGxGz^'+str(k)+'GzGxGx')]

    return cosStrList, sinStrList


def make_rpe_epsilon_str_lists_gx_gz(kList):
    """
    Make epsilon cosine and sine circuit lists for (approx) X pi/4 and
    Z pi/2 gates. These operation sequences are used to estimate epsilon (X rotation
    angle).

    Parameters
    ----------
    kList : list of ints
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

    for k in kList:
        epsilonCosStrList += [_objs.Circuit(('Gx',) * k
                                            + ('Gx',) * 4,
                                            'Gx^' + str(k) + 'GxGxGxGx')]

        epsilonSinStrList += [_objs.Circuit(('Gx', 'Gx', 'Gz', 'Gz')
                                            + ('Gx',) * k
                                            + ('Gx',) * 4,
                                            'GxGxGzGzGx^' + str(k) + 'GxGxGxGx')]

        #From RPEToolsNewNew.py
        #epsilonCosStrList += [_objs.Circuit(('Gx',)*k,
        #                                       'Gx^'+str(k))]
        #
        #epsilonSinStrList += [_objs.Circuit(('Gx','Gx')+('Gx',)*k,
        #                                       'GxGxGx^'+str(k))]

    return epsilonCosStrList, epsilonSinStrList


def make_rpe_theta_str_lists_gx_gz(kList):
    """
    Make theta cosine and sine circuit lists for (approx) X pi/4 and Z pi/2
    gates. These operation sequences are used to estimate theta (X-Z axes angle).

    Parameters
    ----------
    kList : list of ints
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

    for k in kList:
        thetaCosStrList += [_objs.Circuit(
            ('Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz', 'Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz') * k
            + ('Gx',) * 4, '(GzGxGxGxGxGzGzGxGxGxGxGz)^' + str(k) + 'GxGxGxGx')]

        thetaSinStrList += [_objs.Circuit(
            ('Gx', 'Gx', 'Gz', 'Gz')
            + ('Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz', 'Gz', 'Gx', 'Gx', 'Gx', 'Gx', 'Gz') * k
            + ('Gx',) * 4,
            '(GxGxGzGz)(GzGxGxGxGxGzGzGxGxGxGxGz)^' + str(k) + 'GxGxGxGx')]

        #From RPEToolsNewNew.py
        #thetaCosStrList += [_objs.Circuit(
        #       ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k,
        #       '(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k))]
        #
        #thetaSinStrList += [_objs.Circuit(
        #       ('Gx','Gx')+
        #       ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k,
        #       'GxGx(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k))]

    return thetaCosStrList, thetaSinStrList


def make_rpe_string_list_d(log2kMax):
    """
    Generates a dictionary that contains operation sequences for all RPE cosine and
    sine experiments for all three angles.

    Parameters
    ----------
    log2kMax : int
       Maximum number of times to repeat an RPE "germ"

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
    kList = [2**k for k in range(log2kMax + 1)]
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

        - "none"  - no sampl error:
          counts are floating point numbers such that the exact
          probabilty can be found by the ratio of count / total.
        - "round" - same as "none", except counts are rounded to the nearest
          integer.
        - "binomial" - the number of counts is taken from a binomial
          distribution. Distribution has parameters p = probability of the
          operation sequence and n = number of samples.  This can only be used when
          there are exactly two SPAM labels in modelOrDataset.
        - "multinomial" - counts are taken from a multinomial distribution.
          Distribution has parameters p_k = probability of the operation sequence
          using the k-th SPAM label and n = number of samples.  This should not
          be used for RPE.

    seed : int, optional
        If not None, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    Returns
    -------
    DataSet
       A static data set filled with counts for the specified operation sequences.
    """
    return _dsc.generate_fake_data(modelOrDataset,
                                   stringListD['totalStrList'],
                                   nSamples, sampleError=sampleError, seed=seed)


#TODO savePlot arg is never used?
def rpe_ensemble_test(alphaTrue, epsilonTrue, Yrot, SPAMdepol, log2kMax, N, runs):
    #                  plot=False):
    """ Experimental test function """
    kList = [2**k for k in range(log2kMax + 1)]

    alphaCosStrList, alphaSinStrList = make_rpe_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_rpe_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_rpe_theta_str_lists_gx_gz(kList)

    #percentAlphaError = 100*_np.abs((_np.pi/2-alphaTrue)/alphaTrue)
    #percentEpsilonError = 100*_np.abs((_np.pi/4 - epsilonTrue)/epsilonTrue)

    simModel = _setc.build_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gz'],
                                          ["I(Q0)", "X(" + str(epsilonTrue) + ",Q0)", "Z(" + str(alphaTrue) + ",Q0)"],
                                          prepLabels=["rho0"], prepExpressions=["0"],
                                          effectLabels=["E0", "Ec"], effectExpressions=["0", "complement"],
                                          spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

    modelAux1 = _setc.build_explicit_model([('Q0',)], ['Gi', 'Gy', 'Gz'],
                                           ["I(Q0)", "Y(" + str(Yrot) + ",Q0)", "Z(pi/2,Q0)"],
                                           prepLabels=["rho0"], prepExpressions=["0"],
                                           effectLabels=["E0", "Ec"], effectExpressions=["0", "complement"],
                                           spamdefs={'0': ('rho0', 'E0'), '1': ('rho0', 'Ec')})

    simModel.operations['Gx'] = _objs.FullDenseOp(
        _np.dot(_np.dot(_np.linalg.inv(modelAux1.operations['Gy']), simModel.operations['Gx']),
                modelAux1.operations['Gy']))

    simModel = simModel.depolarize(spam_noise=SPAMdepol)

    thetaTrue = _tools.rpe.extract_theta(simModel)

    #SPAMerror = _np.dot(simModel.effects['E0'].T,simModel.preps['rho0'])[0,0]

    jMax = runs

    alphaHatListArray = _np.zeros([jMax, log2kMax + 1], dtype='object')
    epsilonHatListArray = _np.zeros([jMax, log2kMax + 1], dtype='object')
    thetaHatListArray = _np.zeros([jMax, log2kMax + 1], dtype='object')

    alphaErrorArray = _np.zeros([jMax, log2kMax + 1], dtype='object')
    epsilonErrorArray = _np.zeros([jMax, log2kMax + 1], dtype='object')
    thetaErrorArray = _np.zeros([jMax, log2kMax + 1], dtype='object')
    PhiFunErrorArray = _np.zeros([jMax, log2kMax + 1], dtype='object')

    for j in range(jMax):
        simDS = _dsc.generate_fake_data(
            simModel, alphaCosStrList + alphaSinStrList + epsilonCosStrList
            + epsilonSinStrList + thetaCosStrList + thetaSinStrList,
            N, sampleError='binomial', seed=j)
        alphaErrorList = []
        epsilonErrorList = []
        thetaErrorList = []
        PhiFunErrorList = []
        alphaHatList = _tools.rpe.est_angle_list(simDS, alphaSinStrList,
                                                 alphaCosStrList, 'alpha')
        epsilonHatList = _tools.rpe.est_angle_list(simDS, epsilonSinStrList,
                                                   epsilonCosStrList, 'epsilon')
        thetaHatList, PhiFunList = _tools.rpe.est_theta_list(simDS, thetaSinStrList,
                                                             thetaCosStrList, epsilonHatList,
                                                             returnPhiFunList=True)
        for alphaTemp1 in alphaHatList:
            alphaErrorList.append(abs(alphaTrue - alphaTemp1))
        for epsilonTemp1 in epsilonHatList:
            epsilonErrorList.append(abs(epsilonTrue - epsilonTemp1))
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

    #print "True alpha:",alphaTrue
    #print "True alpha:",alphaTrue
    #print "True alpha:",alphaTrue
    #print "True alpha:",alphaTrue
    #print "% true alpha deviation from target:", percentAlphaError

    outputDict = {}
#    outputDict['alphaArray'] = alphaHatListArray
#    outputDict['alphaErrorArray'] = alphaErrorArray
#    outputDict['epsilonArray'] = epsilonHatListArray
#    outputDict['epsilonErrorArray'] = epsilonErrorArray
#    outputDict['thetaArray'] = thetaHatListArray
#    outputDict['thetaErrorArray'] = thetaErrorArray
#    outputDict['PhiFunErrorArray'] = PhiFunErrorArray
#    outputDict['alpha'] = alphaTrue
#    outputDict['epsilonTrue'] = epsilonTrue
#    outputDict['thetaTrue'] = thetaTrue
#    outputDict['Yrot'] = Yrot
#    outputDict['SPAMdepol'] = SPAMdepol#Input value to depolarize SPAM by
#    outputDict['SPAMerror'] = SPAMerror#<<E|rho>>
#    outputDict['mdl'] = simModel
#    outputDict['N'] = N

    return outputDict

from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for creating RPE GateSets and GateString lists """
import numpy as _np
from . import gatesetconstruction as _setc
from . import datasetconstruction as _dsc

from .. import objects as _objs
from .. import tools as _tools


def make_parameterized_rpe_gate_set(alphaTrue, epsilonTrue, Yrot, SPAMdepol,
                                   gateDepol=None, withId=True):
    """
    Make a gateset for simulating RPE, paramaterized by rotation angles.  Note
    that the output gateset also has thetaTrue, alphaTrue, and epsilonTrue
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
    GateSet
        The desired gateset for RPE; gateset also has attributes thetaTrue,
        alphaTrue, and epsilonTrue, automatically extracted.
    """

    if withId:
        outputGateset = _setc.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gz'],
            [ "I(Q0)", "X(%s,Q0)" % epsilonTrue, "Z(%s,Q0)" % alphaTrue],
            prepLabels=["rho0"], prepExpressions=["0"],
            effectLabels=["E0"], effectExpressions=["1"],
            spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )
    else:
        outputGateset = _setc.build_gateset(
            [2], [('Q0',)],['Gx','Gz'],
            [ "X(%s,Q0)" % epsilonTrue, "Z(%s,Q0)" % alphaTrue],
            prepLabels=["rho0"], prepExpressions=["0"],
            effectLabels=["E0"], effectExpressions=["1"],
            spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )

    if Yrot != 0:
        gatesetAux1 = _setc.build_gateset(
            [2], [('Q0',)],['Gi','Gy','Gz'],
            [ "I(Q0)", "Y(%s,Q0)" % Yrot, "Z(pi/2,Q0)"],
            prepLabels=["rho0"], prepExpressions=["0"],
            effectLabels=["E0"], effectExpressions=["1"],
            spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )

        outputGateset.gates['Gx'] = _objs.FullyParameterizedGate(
            _np.dot( _np.dot(_np.linalg.inv(gatesetAux1.gates['Gy']),
                             outputGateset.gates['Gx']),gatesetAux1.gates['Gy']))

    outputGateset = outputGateset.depolarize(gate_noise=gateDepol,
                                             spam_noise=SPAMdepol)

    thetaTrue = _tools.rpe.extract_theta(outputGateset)
    outputGateset.thetaTrue = thetaTrue

    outputGateset.alphaTrue = _tools.rpe.extract_alpha(outputGateset)
    outputGateset.alphaTrue = alphaTrue

    outputGateset.epsilonTrue = _tools.rpe.extract_epsilon(outputGateset)
    outputGateset.epsilonTrue = epsilonTrue

    return outputGateset

def make_rpe_alpha_str_lists_gx_gz(kList):
    """
    Make alpha cosine and sine gatestring lists for (approx) X pi/4 and Z pi/2
    gates. These gate strings are used to estimate alpha (Z rotation angle).

    Parameters
    ----------
    kList : list of ints
       The list of "germ powers" to be used.  Typically successive powers of
       two; e.g. [1,2,4,8,16].

    Returns
    -------
    cosStrList : list of GateStrings
        The list of "cosine strings" to be used for alpha estimation.
    sinStrList : list of GateStrings
        The list of "sine strings" to be used for alpha estimation.
    """
    cosStrList = []
    sinStrList = []
    for k in kList:
        cosStrList += [ _objs.GateString(('Gi','Gx','Gx','Gz')+
                                         ('Gz',)*k +
                                         ('Gz','Gz','Gz','Gx','Gx'),
                                         'GiGxGxGzGz^'+str(k)+'GzGzGzGxGx')]

        sinStrList += [ _objs.GateString(('Gx','Gx','Gz','Gz')+
                                         ('Gz',)*k +
                                         ('Gz','Gz','Gz','Gx','Gx'),
                                         'GxGxGzGzGz^'+str(k)+'GzGzGzGxGx')]

        #From RPEToolsNewNew.py
        ##cosStrList += [_objs.GateString(('Gi','Gx','Gx')+
        ##                                ('Gz',)*k +
        ##                                ('Gx','Gx'),
        ##                                'GiGxGxGz^'+str(k)+'GxGx')]
        #
        #
        #cosStrList += [_objs.GateString(('Gx','Gx')+
        #                                ('Gz',)*k +
        #                                ('Gx','Gx'),
        #                                'GxGxGz^'+str(k)+'GxGx')]
        #
        #
        #sinStrList += [_objs.GateString(('Gx','Gx')+
        #                                ('Gz',)*k +
        #                                ('Gz','Gx','Gx'),
        #                                'GxGxGz^'+str(k)+'GzGxGx')]

    return cosStrList, sinStrList

def make_rpe_epsilon_str_lists_gx_gz(kList):
    """
    Make epsilon cosine and sine gatestring lists for (approx) X pi/4 and
    Z pi/2 gates. These gate strings are used to estimate epsilon (X rotation
    angle).

    Parameters
    ----------
    kList : list of ints
       The list of "germ powers" to be used.  Typically successive powers of
       two; e.g. [1,2,4,8,16].

    Returns
    -------
    epsilonCosStrList : list of GateStrings
        The list of "cosine strings" to be used for epsilon estimation.
    epsilonSinStrList : list of GateStrings
        The list of "sine strings" to be used for epsilon estimation.
    """
    epsilonCosStrList = []
    epsilonSinStrList = []

    for k in kList:
        epsilonCosStrList += [_objs.GateString(('Gx',)*k+
                                               ('Gx',)*4,
                                               'Gx^'+str(k)+'GxGxGxGx')]

        epsilonSinStrList += [_objs.GateString(('Gx','Gx','Gz','Gz')+
                                               ('Gx',)*k+
                                               ('Gx',)*4,
                                               'GxGxGzGzGx^'+str(k)+'GxGxGxGx')]

        #From RPEToolsNewNew.py
        #epsilonCosStrList += [_objs.GateString(('Gx',)*k,
        #                                       'Gx^'+str(k))]
        #
        #epsilonSinStrList += [_objs.GateString(('Gx','Gx')+('Gx',)*k,
        #                                       'GxGxGx^'+str(k))]

    return epsilonCosStrList, epsilonSinStrList

def make_rpe_theta_str_lists_gx_gz(kList):
    """
    Make theta cosine and sine gatestring lists for (approx) X pi/4 and Z pi/2
    gates. These gate strings are used to estimate theta (X-Z axes angle).

    Parameters
    ----------
    kList : list of ints
       The list of "germ powers" to be used.  Typically successive powers of
       two; e.g. [1,2,4,8,16].

    Returns
    -------
    thetaCosStrList : list of GateStrings
        The list of "cosine strings" to be used for theta estimation.
    thetaSinStrList : list of GateStrings
        The list of "sine strings" to be used for theta estimation.
    """
    thetaCosStrList = []
    thetaSinStrList = []

    for k in kList:
        thetaCosStrList += [_objs.GateString(
                ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k+
                ('Gx',)*4, '(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k)+'GxGxGxGx')]

        thetaSinStrList += [_objs.GateString(
                ('Gx','Gx','Gz','Gz')+
                ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k+
                ('Gx',)*4,
                '(GxGxGzGz)(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k)+'GxGxGxGx')]

        #From RPEToolsNewNew.py
        #thetaCosStrList += [_objs.GateString(
        #       ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k,
        #       '(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k))]
        #
        #thetaSinStrList += [_objs.GateString(
        #       ('Gx','Gx')+
        #       ('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k,
        #       'GxGx(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k))]

    return thetaCosStrList, thetaSinStrList

def make_rpe_string_list_d(log2kMax):
    """
    Generates a dictionary that contains gate strings for all RPE cosine and
    sine experiments for all three angles.

    Parameters
    ----------
    log2kMax : int
       Maximum number of times to repeat an RPE "germ"

    Returns
    -------
    totalStrListD : dict
        A dictionary containing all gate strings for all sine and cosine
        experiments for alpha, epsilon, and theta.
        The keys of the returned dictionary are:

        - 'alpha','cos' : List of gate strings for cosine experiments used
          to determine alpha.
        - 'alpha','sin' : List of gate strings for sine experiments used to
          determine alpha.
        - 'epsilon','cos' : List of gate strings for cosine experiments used to
           determine epsilon.
        - 'epsilon','sin' : List of gate strings for sine experiments used to
          determine epsilon.
        - 'theta','cos' : List of gate strings for cosine experiments used to
          determine theta.
        - 'theta','sin' : List of gate strings for sine experiments used to
          determine theta.
        - 'totalStrList' : All above gate strings combined into one list;
          duplicates removed.
    """
    kList = [2**k for k in range(log2kMax+1)]
    alphaCosStrList, alphaSinStrList = make_rpe_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_rpe_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_rpe_theta_str_lists_gx_gz(kList)
    totalStrList = alphaCosStrList + alphaSinStrList + epsilonCosStrList + epsilonSinStrList + thetaCosStrList + thetaSinStrList
    totalStrList = _tools.remove_duplicates(totalStrList) #probably superfluous

    stringListD = {}
    stringListD['alpha','cos'] = alphaCosStrList
    stringListD['alpha','sin'] = alphaSinStrList
    stringListD['epsilon','cos'] = epsilonCosStrList
    stringListD['epsilon','sin'] = epsilonSinStrList
    stringListD['theta','cos'] = thetaCosStrList
    stringListD['theta','sin'] = thetaSinStrList
    stringListD['totalStrList'] = totalStrList
    return stringListD

def make_rpe_data_set(gatesetOrDataset,stringListD,nSamples,sampleError='binomial',seed=None):
    """
    Generate a fake RPE DataSet using the probabilities obtained from a gateset.
    Is a thin wrapper for pygsti.construction.generate_fake_data, changing
    default behavior of sampleError, and taking a dictionary of gate strings
    as input.

    Parameters
    ----------
    gatesetOrDataset : GateSet or DataSet object
        If a GateSet, the gate set whose probabilities generate the data.
        If a DataSet, the data set whose frequencies generate the data.

    stringListD : Dictionary of list of (tuples or GateStrings)
        Each tuple or GateString contains gate labels and
        specifies a gate sequence whose counts are included
        in the returned DataSet.  The dictionary must have the key
        'totalStrList'; easiest if this dictionary is generated by
        make_rpe_string_list_d.

    nSamples : int or list of ints or None
        The simulated number of samples for each gate string.  This only
        has effect when  sampleError == "binomial" or "multinomial".  If
        an integer, all gate strings have this number of total samples. If
        a list, integer elements specify the number of samples for the
        corresponding gate string.  If None, then gatesetOrDataset must be
        a DataSet, and total counts are taken from it (on a per-gatestring
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
          gate string and n = number of samples.  This can only be used when
          there are exactly two SPAM labels in gatesetOrDataset.
        - "multinomial" - counts are taken from a multinomial distribution.
          Distribution has parameters p_k = probability of the gate string
          using the k-th SPAM label and n = number of samples.  This should not
          be used for RPE.

    seed : int, optional
        If not None, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    Returns
    -------
    DataSet
       A static data set filled with counts for the specified gate strings.
    """
    return _dsc.generate_fake_data(gatesetOrDataset,
                                   stringListD['totalStrList'],
                                   nSamples,sampleError=sampleError,seed=seed)



#TODO savePlot arg is never used?
def rpe_ensemble_test(alphaTrue, epsilonTrue, Yrot, SPAMdepol, log2kMax, N, runs,
                  plot=False, savePlot=False):

    """ Experimental test function """
    kList = [2**k for k in range(log2kMax+1)]

    alphaCosStrList, alphaSinStrList = make_rpe_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_rpe_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_rpe_theta_str_lists_gx_gz(kList)

    percentAlphaError = 100*_np.abs((_np.pi/2-alphaTrue)/alphaTrue)
    percentEpsilonError = 100*_np.abs((_np.pi/4 - epsilonTrue)/epsilonTrue)

    simGateset = _setc.build_gateset( [2], [('Q0',)],['Gi','Gx','Gz'],
                                      [ "I(Q0)", "X("+str(epsilonTrue)+",Q0)", "Z("+str(alphaTrue)+",Q0)"],
                                      prepLabels=["rho0"], prepExpressions=["0"],
                                      effectLabels=["E0"], effectExpressions=["1"],
                                      spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )


    gatesetAux1 = _setc.build_gateset( [2], [('Q0',)],['Gi','Gy','Gz'],
                                       [ "I(Q0)", "Y("+str(Yrot)+",Q0)", "Z(pi/2,Q0)"],
                                       prepLabels=["rho0"], prepExpressions=["0"],
                                       effectLabels=["E0"], effectExpressions=["1"],
                                       spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )

    simGateset.gates['Gx'] =  _objs.FullyParameterizedGate(
        _np.dot(_np.dot(_np.linalg.inv(gatesetAux1.gates['Gy']),simGateset.gates['Gx']),
                gatesetAux1.gates['Gy']))

    simGateset = simGateset.depolarize(spam_noise=SPAMdepol)

    thetaTrue = _tools.rpe.extract_theta(simGateset)

    SPAMerror = _np.dot(simGateset.effects['E0'].T,simGateset.preps['rho0'])[0,0]

    jMax = runs

    alphaHatListArray = _np.zeros([jMax,log2kMax+1],dtype='object')
    epsilonHatListArray = _np.zeros([jMax,log2kMax+1],dtype='object')
    thetaHatListArray = _np.zeros([jMax,log2kMax+1],dtype='object')

    alphaErrorArray = _np.zeros([jMax,log2kMax+1],dtype='object')
    epsilonErrorArray = _np.zeros([jMax,log2kMax+1],dtype='object')
    thetaErrorArray = _np.zeros([jMax,log2kMax+1],dtype='object')
    PhiFunErrorArray = _np.zeros([jMax,log2kMax+1],dtype='object')

    for j in range(jMax):
    #    simDS = _dsc.generate_fake_data(gateset3,alphaCosStrList+alphaSinStrList+epsilonCosStrList+epsilonSinStrList+thetaCosStrList+epsilonSinStrList,
        simDS = _dsc.generate_fake_data(
            simGateset, alphaCosStrList+alphaSinStrList+epsilonCosStrList+
            epsilonSinStrList+thetaCosStrList+thetaSinStrList,
            N,sampleError='binomial',seed=j)
        alphaErrorList = []
        epsilonErrorList = []
        thetaErrorList = []
        PhiFunErrorList = []
        alphaHatList = _tools.rpe.est_angle_list(simDS,alphaSinStrList,
                                                 alphaCosStrList,'alpha')
        epsilonHatList = _tools.rpe.est_angle_list(simDS,epsilonSinStrList,
                                                   epsilonCosStrList,'epsilon')
        thetaHatList,PhiFunList = _tools.rpe.est_theta_list(simDS,thetaSinStrList,
                                                 thetaCosStrList,epsilonHatList,
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

        alphaErrorArray[j,:] = _np.array(alphaErrorList)
        epsilonErrorArray[j,:] = _np.array(epsilonErrorList)
        thetaErrorArray[j,:] = _np.array(thetaErrorList)
        PhiFunErrorArray[j,:] = _np.array(PhiFunErrorList)

        alphaHatListArray[j,:] = _np.array(alphaHatList)
        epsilonHatListArray[j,:] = _np.array(epsilonHatList)
        thetaHatListArray[j,:] = _np.array(thetaHatList)

    #print "True alpha:",alphaTrue
    #print "True alpha:",alphaTrue
    #print "True alpha:",alphaTrue
    #print "True alpha:",alphaTrue
    #print "% true alpha deviation from target:", percentAlphaError

    if plot:
        import matplotlib as _mpl
        _mpl.pyplot.loglog(kList,_np.median(alphaErrorArray,axis=0),label='N='+str(N))

        _mpl.pyplot.loglog(kList,_np.array(kList)**-1.,'-o',label='1/k')
        _mpl.pyplot.xlabel('k')
        _mpl.pyplot.ylabel(r'$\alpha_z - \widehat{\alpha_z}$')
        _mpl.pyplot.title('RPE error in Z angle\n% error in Z angle '+str(percentAlphaError)+'%, % error in X angle '+str(percentEpsilonError)+'%\n% error in SPAM, '+str(100*SPAMerror)+'%, X-Z axis error '+str(Yrot)+'\nMedian of '+str(jMax)+' Trials')
        _mpl.pyplot.legend()

        _mpl.pyplot.show()

        _mpl.pyplot.loglog(kList,_np.median(epsilonErrorArray,axis=0),label='N='+str(N))

        _mpl.pyplot.loglog(kList,_np.array(kList)**-1.,'-o',label='1/k')
        _mpl.pyplot.xlabel('k')
        _mpl.pyplot.ylabel(r'$\epsilon_x - \widehat{\epsilon_x}$')
        _mpl.pyplot.title('RPE error in X angle\n% error in Z angle '+str(percentAlphaError)+'%, % error in X angle '+str(percentEpsilonError)+'%\n% error in SPAM, '+str(100*SPAMerror)+'%, X-Z axis error '+str(Yrot)+'\nMedian of '+str(jMax)+' Trials')
        _mpl.pyplot.legend()

        _mpl.pyplot.show()

        _mpl.pyplot.loglog(kList,_np.median(thetaErrorArray,axis=0),label='N='+str(N))

        _mpl.pyplot.loglog(kList,_np.array(kList)**-1.,'-o',label='1/k')
        _mpl.pyplot.xlabel('k')
        _mpl.pyplot.ylabel(r'$\theta_{xz} - \widehat{\theta_{xz}}$')
        _mpl.pyplot.title('RPE error in X axis angle\n% error in Z angle '+str(percentAlphaError)+'%, % error in X angle '+str(percentEpsilonError)+'%\n% error in SPAM, '+str(100*SPAMerror)+'%, X-Z axis error '+str(Yrot)+'\nMedian of '+str(jMax)+' Trials')
        _mpl.pyplot.legend()

        _mpl.pyplot.show()

        _mpl.pyplot.loglog(kList,_np.median(PhiFunErrorArray,axis=0),label='N='+str(N))

#        _mpl.pyplot.loglog(kList,_np.array(kList)**-1.,'-o',label='1/k')
        _mpl.pyplot.xlabel('k')
        _mpl.pyplot.ylabel(r'$\Phi func.$')
        _mpl.pyplot.title('RPE error in Phi func.\n% error in Z angle '+str(percentAlphaError)+'%, % error in X angle '+str(percentEpsilonError)+'%\n% error in SPAM, '+str(100*SPAMerror)+'%, X-Z axis error '+str(Yrot)+'\nMedian of '+str(jMax)+' Trials')
        _mpl.pyplot.legend()

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
#    outputDict['gs'] = simGateset
#    outputDict['N'] = N

    return outputDict



#def make_rpe_data_set(inputGateset, log2kMax, N, seed = None, returnStringListDict = False):
#    """
#    Generate a fake RPE dataset.  At present, only works for kList of form [1,2,4,...,2**log2kMax]
#
#    Parameters
#    ----------
#    inputGateset : The gateset used to generate the data.
#    log2kMax : Maximum number of times to repeat an RPE "germ"
#    N : Number of clicks per experiment.
#    seed : Used to seed numpy's random number generator.  Default is None.
#    returnStringListDict : Do we want a dictionary of the sin and cos experiments for the various angles?  Default is False.
#
#    Returns
#    -------
#    simDS
#        The simulated dataset containing the RPE experiments.
#    stringListD
#        Dictionary of gate string lists for sin and cos experiments; is not returned by default.
#    """
#    kList = [2**k for k in range(log2kMax+1)]
#    alphaCosStrList, alphaSinStrList = make_alpha_str_lists_gx_gz(kList)
#    epsilonCosStrList, epsilonSinStrList = make_epsilon_str_lists_gx_gz(kList)
#    thetaCosStrList, thetaSinStrList = make_theta_str_lists_gx_gz(kList)
#    totalStrList = alphaCosStrList + alphaSinStrList + epsilonCosStrList + epsilonSinStrList + thetaCosStrList + thetaSinStrList
#    totalStrList = _tools.remove_duplicates(totalStrList)#This step is probably superfluous.
#    simDS = _dsc.generate_fake_data(inputGateset,totalStrList,N,sampleError='binomial',seed=seed)
#    if returnStringListDict:
#        stringListD = {}
#        stringListD['alpha','cos'] = alphaCosStrList
#        stringListD['alpha','sin'] = alphaSinStrList
#        stringListD['epsilon','cos'] = epsilonCosStrList
#        stringListD['epsilon','sin'] = epsilonSinStrList
#        stringListD['theta','cos'] = thetaCosStrList
#        stringListD['theta','sin'] = thetaSinStrList
#        return simDS, stringListD
#    else:
#        return simDS, None

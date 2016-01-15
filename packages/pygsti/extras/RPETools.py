import sys
import GST
import numpy.random as random
import numpy as np
from GSTCommons import MakeLists_WholeGermPowers
import matplotlib
import GST.ComputeReportables
from scipy import optimize
#%matplotlib inline

class myGate():
    def __init__(self,inputArray):
        if inputArray.shape[0] != inputArray.shape[1]:
            raise ValueError("Gate matrix must be square!")
        self.dim = inputArray.shape[0]
        self.matrix = inputArray
    def copy(self):
        return myGate(self.matrix)

def make_paramterized_rpe_gate_set(alphaTrue,epsilonTrue,Yrot,SPAMdepol,gateDepol=None,withId = True):
    """
    Make a gateset for simulating RPE, paramaterized by rotation angles.  Note that output gateset also has thetaTrue, alphaTrue, and epsilonTrue attributes.

    Parameters
    ----------
    alphaTrue : Angle of Z rotation (canonical RPE requires alphaTrue to be close to pi/2).
    epsilonTrue : Angle of X rotation (canonical RPE requires epsilonTrue to be close to pi/4).
    Yrot : Angle of rotation about Y axis that, by similarity transformation, rotates X rotation.
    SPAMdepol : Amount to depolarize SPAM by.
    gateDepol : Amount to depolarize gates by (defaults to None).
    withId : Do we include (perfect) identity or no identity? (Defaults to False; should be False for RPE, True for GST)
    Returns
    -------
    outputGateset
        The desired gateset for RPE; gateset also has attributes thetaTrue, alphaTrue, and epsilonTrue, automatically extracted.
    """

    if withId:
        outputGateset = GST.build_gateset( [2], [('Q0',)],['Gi','Gx','Gz'], 
                                     [ "I(Q0)", "X("+str(epsilonTrue)+",Q0)", "Z("+str(alphaTrue)+",Q0)"],
                                     rhoExpressions=["0"], EExpressions=["1"], 
                                     spamLabelDict={'plus': (0,0), 'minus': (0,-1) })
    else:
        outputGateset = GST.build_gateset( [2], [('Q0',)],['Gx','Gz'], 
                                     [ "X("+str(epsilonTrue)+",Q0)", "Z("+str(alphaTrue)+",Q0)"],
                                     rhoExpressions=["0"], EExpressions=["1"], 
                                     spamLabelDict={'plus': (0,0), 'minus': (0,-1) })


    if Yrot != 0:
        gatesetAux1 = GST.build_gateset( [2], [('Q0',)],['Gi','Gy','Gz'], 
                                     [ "I(Q0)", "Y("+str(Yrot)+",Q0)", "Z(pi/2,Q0)"],
                                     rhoExpressions=["0"], EExpressions=["1"], 
                                     spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

        outputGateset.set_gate('Gx',GST.Gate.FullyParameterizedGate(np.dot(np.dot(np.linalg.inv(gatesetAux1['Gy']),outputGateset['Gx']),gatesetAux1['Gy'])))       
#        myGate(np.dot(np.dot(np.linalg.inv(gatesetAux1['Gy']),outputGateset['Gx']),gatesetAux1['Gy'])))

    outputGateset = GST.GateSetTools.depolarize_spam(outputGateset,noise=SPAMdepol)
    
    if gateDepol:
        outputGateset = GST.GateSetTools.depolarize_gateset(outputGateset,noise=gateDepol)
    
    thetaTrue = extract_theta(outputGateset)
    outputGateset.thetaTrue = thetaTrue
    
    outputGateset.alphaTrue = extract_alpha(outputGateset)
    outputGateset.alphaTrue = alphaTrue
    
    outputGateset.epsilonTrue = extract_epsilon(outputGateset)
    outputGateset.epsilonTrue = epsilonTrue
    
    return outputGateset

def make_alpha_str_lists_gx_gz(kList):
    """
    Make alpha cosine and sine gatestring lists for (approx) X pi/4 and Z pi/2 gates.
    These gate strings are used to estimate alpha (Z rotation angle).

    Parameters
    ----------
    kList : The list of "germ powers" to be used.  Typically successive powers of two;
    e.g. [1,2,4,8,16].

    Returns
    -------
    cosStrList
        The list of "cosine strings" to be used for alpha estimation.
    sinStrList
        The list of "sine strings" to be used for alpha estimation.
    """
    cosStrList = []
    sinStrList = []
    for k in kList:
        cosStrList += [GST.gatestring.GateString(('Gi','Gx','Gx','Gz')+('Gz',)*k + ('Gz','Gz','Gz','Gx','Gx'),
                                                 'GiGxGxGzGz^'+str(k)+'GzGzGzGxGx')]

        sinStrList += [GST.gatestring.GateString(('Gx','Gx','Gz','Gz')+('Gz',)*k + ('Gz','Gz','Gz','Gx','Gx'),
                                                 'GxGxGzGzGz^'+str(k)+'GzGzGzGxGx')]
    return cosStrList, sinStrList
    
def make_epsilon_str_lists_gx_gz(kList):
    """
    Make epsilon cosine and sine gatestring lists for (approx) X pi/4 and Z pi/2 gates.
    These gate strings are used to estimate epsilon (X rotation angle).

    Parameters
    ----------
    kList : The list of "germ powers" to be used.  Typically successive powers of two;
    e.g. [1,2,4,8,16].

    Returns
    -------
    epsilonCosStrList
        The list of "cosine strings" to be used for epsilon estimation.
    epsilonSinStrList
        The list of "sine strings" to be used for epsilon estimation.
    """
    epsilonCosStrList = []
    epsilonSinStrList = []

    for k in kList:
        epsilonCosStrList += [GST.gatestring.GateString(('Gx',)*k+('Gx',)*4,
                             'Gx^'+str(k)+'GxGxGxGx')]
    
        epsilonSinStrList += [GST.gatestring.GateString(('Gx','Gx','Gz','Gz')+('Gx',)*k+('Gx',)*4,
                             'GxGxGzGzGx^'+str(k)+'GxGxGxGx')]
    return epsilonCosStrList, epsilonSinStrList

def make_theta_str_lists_gx_gz(kList):
    """
    Make theta cosine and sine gatestring lists for (approx) X pi/4 and Z pi/2 gates.
    These gate strings are used to estimate theta (X-Z axes angle).

    Parameters
    ----------
    kList : The list of "germ powers" to be used.  Typically successive powers of two;
    e.g. [1,2,4,8,16].

    Returns
    -------
    thetaCosStrList
        The list of "cosine strings" to be used for theta estimation.
    thetaSinStrList
        The list of "sine strings" to be used for theta estimation.
    """
    thetaCosStrList = []
    thetaSinStrList = []

    for k in kList:
        thetaCosStrList += [GST.gatestring.GateString(('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k+('Gx',)*4,
                             'GzGxGxGxGxGzGzGxGxGxGxGz^'+str(k)+'GxGxGxGx')]
    
        thetaSinStrList += [GST.gatestring.GateString(('Gx','Gx','Gz','Gz')+('Gz','Gx','Gx','Gx','Gx','Gz','Gz','Gx','Gx','Gx','Gx','Gz')*k+('Gx',)*4,
                             '(GzGxGxGxGxGzGzGxGxGxGxGz)^'+str(k)+'GxGxGxGx')]
    return thetaCosStrList, thetaSinStrList

def make_rpe_string_list_d(log2kMax):
    """
    Generates a dictionary that contains gate strings for all RPE cosine and sine experiments for all three angles.
    
    Parameters
    ----------
    log2kMax : Maximum number of times to repeat an RPE "germ"
    
    Returns
    -------
    totalStrListD
        A dictionary containing all gate strings for all sine and cosine experiments for alpha, epsilon, and theta.
        The keys of the returned dictionary are:
        
        - 'alpha','cos' : List of gate strings for cosine experiments used to determine alpha.
        - 'alpha','sin' : List of gate strings for sine experiments used to determine alpha.
        - 'epsilon','cos' : List of gate strings for cosine experiments used to determine epsilon.
        - 'epsilon','sin' : List of gate strings for sine experiments used to determine epsilon.
        - 'theta','cos' : List of gate strings for cosine experiments used to determine theta.
        - 'theta','sin' : List of gate strings for sine experiments used to determine theta.
        - 'totalStrList' : All above gate strings combined into one list; duplicates removed.
    """
    kList = [2**k for k in range(log2kMax+1)]
    alphaCosStrList, alphaSinStrList = make_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_theta_str_lists_gx_gz(kList)
    totalStrList = alphaCosStrList + alphaSinStrList + epsilonCosStrList + epsilonSinStrList + thetaCosStrList + thetaSinStrList
    totalStrList = GST.ListTools.remove_duplicates(totalStrList)#This step is probably superfluous.
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
    Generate a fake RPE DataSet using the probabilities obtained from a gateset..  Is a thin wrapper for GST.generate_fake_data, changing default behavior of sampleError,
    and taking a dictionary of gate strings as input.

    Parameters
    ----------
    gatesetOrDataset : GateSet or DataSet object
        If a GateSet, the gate set whose probabilities generate the data.
        If a DataSet, the data set whose frequencies generate the data.

    stringListD : Dictionary of list of (tuples or GateStrings)
        Each tuple or GateString contains gate labels and 
        specifies a gate sequence whose counts are included 
        in the returned DataSet.  The dictionary must have the key 'totalStrList';
        easiest if this dictionary is generated by make_rpe_string_list_d.

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
                  counts are floating point numbers such that the exact probabilty
                  can be found by the ratio of count / total.
        - "round" - same as "none", except counts are rounded to the nearest integer.
        - "binomial" - the number of counts is taken from a binomial distribution.
                     Distribution has parameters p = probability of the gate string
                     and n = number of samples.  This can only be used when there
                     are exactly two SPAM labels in gatesetOrDataset.
        - "multinomial" - counts are taken from a multinomial distribution.
                        Distribution has parameters p_k = probability of the 
                        gate string using the k-th SPAM label and n = number
                        of samples.  This should not be used for RPE.

    seed : int, optional
        If not None, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    Returns
    -------
    DataSet
       A static data set filled with counts for the specified gate strings.
    """
    simDS = GST.generate_fake_data(gatesetOrDataset,stringListD['totalStrList'],nSamples,sampleError=sampleError,seed=seed)
    return simDS

def extract_rotation_hat(xhat,yhat,k,Nx,Ny,previousAngle=None):
    """
    For a single germ generation (k value), estimate the angle of rotation for either
    alpha, epsilon, or Phi.  (Warning:  Do not use for theta estimate without further processing!)
    
    Parameters
    ----------
    xhat : The number of plus counts for the sin string being used.
    yhat : The number of plus counts for the cos string being used.
    k : The generation of experiments that xhat and yhat come from.
    Nx : The number of sin string clicks.
    Ny : The number cos string clicks.
    previousAngle : Angle estimate from previous generation; used to refine this generation's estimate.  Default is None (for estimation with no previous genereation's data)

    Returns
    -------
    alpha_j
        The current angle estimate.
    """

    if k!=1 and previousAngle == None:
        raise Exception('Need previousAngle!')
    if k == 1:
        return np.arctan2((xhat-Nx/2.)/Nx,(yhat-Ny/2.)/Ny)
    elif k>1:
        angle_j = 1./k * np.arctan2((xhat-Nx/2.)/Nx,(yhat-Ny/2.)/Ny)
        while not (angle_j > previousAngle - np.pi/k and angle_j <= previousAngle + np.pi/k):
            if angle_j <= previousAngle - np.pi/k:
                angle_j += 2 * np.pi/k
            elif angle_j > previousAngle + np.pi/k:
                angle_j -= 2 * np.pi/k
            else:
                raise Exception('What?!')
        return angle_j

def est_angle_list(DS,angleSinStrs,angleCosStrs):
    """
    For a dataset containing sin and cos strings to estimate either alpha, epsilon, or Phi
    return a list of alpha, epsilon, or Phi estimates (one for each generation).

    WARNING:  At present, kList must be of form [1,2,4,...,2**log2kMax].
    
    Parameters
    ----------
    DS : The dataset from which the angle estimates will be extracted.
    angleSinStrs : The list of sin strs that the estimator will use.
    angleCosStrs : The list of cos strs that the estimator will use.
    
    Returns
    -------
    angleHatList
        A list of angle estimates, ordered by generation (k).
    """
    angleTemp1 = None
    angleHatList = []
    genNum = len(angleSinStrs)
    for i in xrange(genNum):
        xhatTemp = DS[angleSinStrs[i]]['plus']
        yhatTemp = DS[angleCosStrs[i]]['plus']
        Nx = xhatTemp + DS[angleSinStrs[i]]['minus']
        Ny = yhatTemp + DS[angleCosStrs[i]]['minus']
        angleTemp1 = extract_rotation_hat(xhatTemp,yhatTemp,2**i,Nx,Ny,angleTemp1)
        angleHatList.append(angleTemp1)
    return angleHatList
    
def sin_phi2_func(theta,Phi,epsilon):
    """
    Returns the function whose zero, for fixed Phi and epsilon, occurs at the desired value of theta.
    (This function exists to be passed to a minimizer to obtain theta.)
    
    WARNING:  epsilon gets rescaled to newEpsilon, by dividing by pi/4; will have to change for epsilons far from pi/4.
    
    Parameters
    ----------
    theta : Angle between X and Z axes.
    Phi : The auxiliary angle Phi; necessary to calculate theta.
    epsilon : Angle of X rotation.
    
    Returns
    -------
    sinPhi2FuncVal
        The value of sin_phi2_func for given inputs.  (Must be 0 to achieve "true" theta.)
    """
    newEpsilon = (epsilon / (np.pi/4)) - 1
    sinPhi2FuncVal = np.abs(2*np.sin(theta)*np.cos(np.pi*newEpsilon/2)*np.sqrt(1-np.sin(theta)**2*np.cos(np.pi*newEpsilon/2)**2)-np.sin(Phi/2))
    return sinPhi2FuncVal

def est_theta_list(DS,angleSinStrs,angleCosStrs,epsilonList,returnPhiFunList = False):
    """
    For a dataset containing sin and cos strings to estimate theta,
    along with already-made estimates of epsilon, return a list of theta 
    (one for each generation).

    Parameters
    ----------
    DS : The dataset from which the theta estimates will be extracted.
    angleSinStrs : The list of sin strs that the estimator will use.
    angleCosStrs : The list of cos strs that the estimator will use.
    epsilonList : List of epsilon estimates.
    returnPhiFunList : Set to True to obtain measure of how well Eq. III.7 is satisfied.  Default is False.

    Returns
    -------
    thetaHatList
        A list of theta estimates, ordered by generation (k).
    PhiFunList
        A list of sin_phi2_func vals at optimal theta values.  If not close to 0, constraints unsatisfiable.  Only returned if returnPhiFunList is set to True.
    """

    PhiList = est_angle_list(DS,angleSinStrs,angleCosStrs)
    thetaList = []
    PhiFunList = []
    for index, Phi in enumerate(PhiList):
        epsilon = epsilonList[index]
        soln = optimize.minimize(lambda x: sin_phi2_func(x,Phi,epsilon),0)
        thetaList.append(soln['x'][0])
        PhiFunList.append(soln['fun'])
#        if soln['fun'] > 1e-2:
#            print Phi, epsilon
    if returnPhiFunList:
        return thetaList, PhiFunList
    else:
        return thetaList

#def gs2qtys(gateset):
#    """
#    For a given gateset, extract 
#    """

def extract_alpha(gateset):
    """
    For a given gateset, obtain the angle of rotation about Z axis (for gate "Gz").
    
    WARNING:  This is a gauge-covariant parameter!  Gauge must be fixed prior to estimating.
    
    Parameters
    ----------
    gateset : The gateset whose "Gz" angle of rotation is to be calculated.
    
    Returns
    -------
    alphaVal
        The value of alpha for the input gateset.
    """
    gateLabels = gateset.keys()  # gate labels
    qtys_to_compute = [ ('%s decomposition' % gl) for gl in gateLabels ]
    qtys = GST.ComputeReportables.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo=None)
    alphaVal = qtys['Gz decomposition'].value['pi rotations'] * np.pi
    return alphaVal

def extract_epsilon(gateset):
    """
    For a given gateset, obtain the angle of rotation about X axis (for gate "Gx").
    
    WARNING:  This is a gauge-covariant parameter!  Gauge must be fixed prior to estimating.
    
    Parameters
    ----------
    gateset : The gateset whose "Gx" angle of rotation is to be calculated.
    
    Returns
    -------
    epsilonVal
        The value of epsilon for the input gateset.
    """
    gateLabels = gateset.keys()  # gate labels
    qtys_to_compute = [ ('%s decomposition' % gl) for gl in gateLabels ]
    qtys = GST.ComputeReportables.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo=None)
    epsilonVal = qtys['Gx decomposition'].value['pi rotations'] * np.pi
    return epsilonVal

def extract_theta(gateset):
    """
    For a given gateset, obtain the angle between the "X axis of rotation" and the "true" X axis (perpendicular to Z).
    (Angle of misalignment between "Gx" axis of rotation and X axis as defined by "Gz".)
    
    WARNING:  This is a gauge-covariant parameter!  (I think!)  Gauge must be fixed prior to estimating.
    
    Parameters
    ----------
    gateset : The gateset whose X axis misaligment is to be calculated.
    
    Returns
    -------
    thetaVal
        The value of theta for the input gateset.
    """
    gateLabels = gateset.keys()  # gate labels
    qtys_to_compute = [ ('%s decomposition' % gl) for gl in gateLabels ]
    qtys = GST.ComputeReportables.compute_gateset_qtys(qtys_to_compute, gateset, confidenceRegionInfo=None)
    thetaVal = np.real_if_close([np.arccos(np.dot(
            qtys['Gx decomposition'].get_value()['axis of rotation'],
            [0,1,0,0]))])[0]
    if thetaVal > np.pi/2:
        thetaVal = np.pi - thetaVal
    elif thetaVal < -np.pi/2:
        thetaVal = np.pi + thetaVal
    return thetaVal

def analyze_simulated_rpe_experiment(inputDataset,trueGateset,stringListD):
    """
    Compute angle estimates and compare to true estimates for alpha, epsilon, and theta.
    
    Parameters
    ----------
    inputDataset : The dataset containing the RPE experiments.
    trueGateset : The gateset used to generate the RPE data.
    stringListD : The dictionary of gate string lists used for the RPE experiments.  This should be generated via make_rpe_string_list_d.
    
    Returns
    -------
    resultsD
        A dictionary of the results
        The keys of the dictionary are:
        
        -'alphaHatList' : List (ordered by k) of alpha estimates.
        -'epsilonHatList' : List (ordered by k) of epsilon estimates.
        -'thetaHatList' : List (ordered by k) of theta estimates.
        -'alphaErrorList' : List (ordered by k) of difference between true alpha and RPE estimate of alpha.
        -'epsilonErrorList' : List (ordered by k) of difference between true epsilon and RPE estimate of epsilon.
        -'thetaErrorList' : List (ordered by k) of difference between true theta and RPE estimate of theta.
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
        thetaTrue = trueGateset(trueGateset)
    alphaErrorList = []
    epsilonErrorList = []
    thetaErrorList = []
#    PhiFunErrorList = []
    alphaHatList = est_angle_list(inputDataset,alphaSinStrList,alphaCosStrList)
    epsilonHatList = est_angle_list(inputDataset,epsilonSinStrList,epsilonCosStrList)
    thetaHatList,PhiFunErrorList = est_theta_list(inputDataset,thetaSinStrList,thetaCosStrList,epsilonHatList,returnPhiFunList=True)
    for alphaTemp1 in alphaHatList:
        alphaErrorList.append(abs(alphaTrue+alphaTemp1))
    for epsilonTemp1 in epsilonHatList:
        epsilonErrorList.append(abs(epsilonTrue-epsilonTemp1))
#        print abs(np.pi/2-abs(alphaTemp1))
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



def ensemble_test(alphaTrue, epsilonTrue, Yrot, SPAMdepol, log2kMax, N, runs, plot=False, savePlot=False):

    kList = [2**k for k in range(log2kMax+1)]

    alphaCosStrList, alphaSinStrList = make_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_theta_str_lists_gx_gz(kList)

    percentAlphaError = 100*np.abs((np.pi/2-alphaTrue)/alphaTrue)
    percentEpsilonError = 100*np.abs((np.pi/4 - epsilonTrue)/epsilonTrue)

    simGateset = GST.build_gateset( [2], [('Q0',)],['Gi','Gx','Gz'], 
                                 [ "I(Q0)", "X("+str(epsilonTrue)+",Q0)", "Z("+str(alphaTrue)+",Q0)"],
                                 rhoExpressions=["0"], EExpressions=["1"], 
                                 spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

    gatesetAux1 = GST.build_gateset( [2], [('Q0',)],['Gi','Gy','Gz'], 
                                 [ "I(Q0)", "Y("+str(Yrot)+",Q0)", "Z(pi/2,Q0)"],
                                 rhoExpressions=["0"], EExpressions=["1"], 
                                 spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

    simGateset.set_gate('Gx',myGate(np.dot(np.dot(np.linalg.inv(gatesetAux1['Gy']),simGateset['Gx']),gatesetAux1['Gy'])))

    simGateset = GST.GateSetTools.depolarize_spam(simGateset,noise=SPAMdepol)

    #gateset3 = GST.GateSetTools.depolarize_spam(gateset3,noise=0.01)

    thetaTrue = extract_theta(simGateset)

    SPAMerror = np.dot(simGateset.EVecs[0].T,simGateset.rhoVecs[0])[0,0]

    jMax = runs
    
    alphaHatListArray = np.zeros([jMax,log2kMax+1],dtype='object')
    epsilonHatListArray = np.zeros([jMax,log2kMax+1],dtype='object')
    thetaHatListArray = np.zeros([jMax,log2kMax+1],dtype='object')
    
    alphaErrorArray = np.zeros([jMax,log2kMax+1],dtype='object')
    epsilonErrorArray = np.zeros([jMax,log2kMax+1],dtype='object')
    thetaErrorArray = np.zeros([jMax,log2kMax+1],dtype='object')
    PhiFunErrorArray = np.zeros([jMax,log2kMax+1],dtype='object')

    for j in xrange(jMax):
    #    simDS = GST.generate_fake_data(gateset3,alphaCosStrList+alphaSinStrList+epsilonCosStrList+epsilonSinStrList+thetaCosStrList+epsilonSinStrList,
        simDS = GST.generate_fake_data(simGateset,alphaCosStrList+alphaSinStrList+epsilonCosStrList+epsilonSinStrList+thetaCosStrList+thetaSinStrList,
                                   N,sampleError='binomial',seed=j)
        alphaErrorList = []
        epsilonErrorList = []
        thetaErrorList = []
        PhiFunErrorList = []
        alphaHatList = est_angle_list(simDS,alphaSinStrList,alphaCosStrList)
        epsilonHatList = est_angle_list(simDS,epsilonSinStrList,epsilonCosStrList)
        thetaHatList,PhiFunList = est_theta_list(simDS,thetaSinStrList,thetaCosStrList,epsilonHatList,returnPhiFunList=True)
        for alphaTemp1 in alphaHatList:
            alphaErrorList.append(abs(alphaTrue+alphaTemp1))
        for epsilonTemp1 in epsilonHatList:
            epsilonErrorList.append(abs(epsilonTrue-epsilonTemp1))
    #        print abs(np.pi/2-abs(alphaTemp1))
        for thetaTemp1 in thetaHatList:
            thetaErrorList.append(abs(thetaTrue - thetaTemp1))
        for PhiFunTemp1 in PhiFunList:
            PhiFunErrorList.append(PhiFunTemp1)
 
        alphaErrorArray[j,:] = np.array(alphaErrorList)
        epsilonErrorArray[j,:] = np.array(epsilonErrorList)
        thetaErrorArray[j,:] = np.array(thetaErrorList)
        PhiFunErrorArray[j,:] = np.array(PhiFunErrorList)
 
        alphaHatListArray[j,:] = np.array(alphaHatList)
        epsilonHatListArray[j,:] = np.array(epsilonHatList)
        thetaHatListArray[j,:] = np.array(thetaHatList)

    #print "True alpha:",alphaTrue
    #print "True alpha:",alphaTrue
    #print "True alpha:",alphaTrue
    #print "True alpha:",alphaTrue
    #print "% true alpha deviation from target:", percentAlphaError

    if plot:
        matplotlib.pyplot.loglog(kList,np.median(alphaErrorArray,axis=0),label='N='+str(N))

        matplotlib.pyplot.loglog(kList,np.array(kList)**-1.,'-o',label='1/k')
        matplotlib.pyplot.xlabel('k')
        matplotlib.pyplot.ylabel(r'$\alpha_z - \widehat{\alpha_z}$')
        matplotlib.pyplot.title('RPE error in Z angle\n% error in Z angle '+str(percentAlphaError)+'%, % error in X angle '+str(percentEpsilonError)+'%\n% error in SPAM, '+str(100*SPAMerror)+'%, X-Z axis error '+str(Yrot)+'\nMedian of '+str(jMax)+' Trials')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.show()

        matplotlib.pyplot.loglog(kList,np.median(epsilonErrorArray,axis=0),label='N='+str(N))

        matplotlib.pyplot.loglog(kList,np.array(kList)**-1.,'-o',label='1/k')
        matplotlib.pyplot.xlabel('k')
        matplotlib.pyplot.ylabel(r'$\epsilon_x - \widehat{\epsilon_x}$')
        matplotlib.pyplot.title('RPE error in X angle\n% error in Z angle '+str(percentAlphaError)+'%, % error in X angle '+str(percentEpsilonError)+'%\n% error in SPAM, '+str(100*SPAMerror)+'%, X-Z axis error '+str(Yrot)+'\nMedian of '+str(jMax)+' Trials')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.show()

        matplotlib.pyplot.loglog(kList,np.median(thetaErrorArray,axis=0),label='N='+str(N))

        matplotlib.pyplot.loglog(kList,np.array(kList)**-1.,'-o',label='1/k')
        matplotlib.pyplot.xlabel('k')
        matplotlib.pyplot.ylabel(r'$\theta_{xz} - \widehat{\theta_{xz}}$')
        matplotlib.pyplot.title('RPE error in X axis angle\n% error in Z angle '+str(percentAlphaError)+'%, % error in X angle '+str(percentEpsilonError)+'%\n% error in SPAM, '+str(100*SPAMerror)+'%, X-Z axis error '+str(Yrot)+'\nMedian of '+str(jMax)+' Trials')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.show()

        matplotlib.pyplot.loglog(kList,np.median(PhiFunErrorArray,axis=0),label='N='+str(N))

#        matplotlib.pyplot.loglog(kList,np.array(kList)**-1.,'-o',label='1/k')
        matplotlib.pyplot.xlabel('k')
        matplotlib.pyplot.ylabel(r'$\Phi func.$')
        matplotlib.pyplot.title('RPE error in Phi func.\n% error in Z angle '+str(percentAlphaError)+'%, % error in X angle '+str(percentEpsilonError)+'%\n% error in SPAM, '+str(100*SPAMerror)+'%, X-Z axis error '+str(Yrot)+'\nMedian of '+str(jMax)+' Trials')
        matplotlib.pyplot.legend()
    
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


'''
def make_rpe_data_set(inputGateset, log2kMax, N, seed = None, returnStringListDict = False):
    """
    Generate a fake RPE dataset.  At present, only works for kList of form [1,2,4,...,2**log2kMax]

    Parameters
    ----------
    inputGateset : The gateset used to generate the data.
    log2kMax : Maximum number of times to repeat an RPE "germ"
    N : Number of clicks per experiment.
    seed : Used to seed numpy's random number generator.  Default is None.
    returnStringListDict : Do we want a dictionary of the sin and cos experiments for the various angles?  Default is False.

    Returns
    -------
    simDS
        The simulated dataset containing the RPE experiments.
    stringListD
        Dictionary of gate string lists for sin and cos experiments; is not returned by default.
    """
    kList = [2**k for k in range(log2kMax+1)]
    alphaCosStrList, alphaSinStrList = make_alpha_str_lists_gx_gz(kList)
    epsilonCosStrList, epsilonSinStrList = make_epsilon_str_lists_gx_gz(kList)
    thetaCosStrList, thetaSinStrList = make_theta_str_lists_gx_gz(kList)
    totalStrList = alphaCosStrList + alphaSinStrList + epsilonCosStrList + epsilonSinStrList + thetaCosStrList + thetaSinStrList
    totalStrList = GST.ListTools.remove_duplicates(totalStrList)#This step is probably superfluous.
    simDS = GST.generate_fake_data(inputGateset,totalStrList,N,sampleError='binomial',seed=seed)
    if returnStringListDict:
        stringListD = {}
        stringListD['alpha','cos'] = alphaCosStrList
        stringListD['alpha','sin'] = alphaSinStrList
        stringListD['epsilon','cos'] = epsilonCosStrList
        stringListD['epsilon','sin'] = epsilonSinStrList
        stringListD['theta','cos'] = thetaCosStrList
        stringListD['theta','sin'] = thetaSinStrList
        return simDS, stringListD
    else:
        return simDS, None
'''
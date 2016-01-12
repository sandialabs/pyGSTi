""" Functions for generating plots and computing chi^2 """
from __future__ import division
import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib as _matplotlib
import Core as _Core
import GateOps as _GateOps
import gatestring as _gatestring
import os as _os
import pickle as _pickle
from matplotlib.ticker import AutoMinorLocator as _AutoMinorLocator
from matplotlib.ticker import FixedLocator as _FixedLocator

class GSTFigure(object):
    def __init__(self, axes, extraInfo=None):
        self.pickledAxes = _pickle.dumps(axes)
        self.extraInfo = extraInfo

    def saveTo(self, filename):
        if filename is not None and len(filename) > 0:
            try:
                axes = _pickle.loads(self.pickledAxes) #this creates a new (current) figure in matplotlib
            except:
                raise ValueError("GSTFigure unpickling error!  This could be caused by using matplotlib or pylab" +
                                 " magic functions ('%pylab inline' or '%matplotlib inline') within an iPython" +
                                 " notebook, so if you used either of these please remove it and all should be well.")
            _plt.savefig(filename, bbox_extra_artists=(axes,), bbox_inches='tight') #need extra artists otherwise axis labels get clipped
            _plt.close(_plt.gcf()) # gcf == "get current figure"; closes the figure created by unpickling

    def setExtraInfo(self, extraInfo):
        self.extraInfo = extraInfo

    def getExtraInfo(self):
        return self.extraInfo
    
    def check(self):
        axes = _pickle.loads(self.pickledAxes) #this creates a new (current) figure in matplotlib
        _plt.close(_plt.gcf()) # gcf == "get current figure"; closes the figure created by unpickling
        

def ChiSqFunc_2outcome( N, p, f, minProbClipForWeighting=1e-4 ):
    """ 
    Computes chi^2 for a 2-outcome measurement.

    The chi-squared function for a 2-outcome measurement using 
    a clipped probability for the statistical weighting.  
    
    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClipForWeighting : float, optional
        Defines clipping interval (see return value).
        
    Returns
    -------
    float or numpy array
        N(p-f)^2 / (cp(1-cp)), 
        where cp is the value of p clipped to the interval
        (minProbClipForWeighting, 1-minProbClipForWeighting)        
    """
    cp = _np.clip(p,minProbClipForWeighting,1-minProbClipForWeighting)
    return N*(p-f)**2/(cp*(1-cp))


def ChiSqFunc_2outcome_wFreqs( N, p, f, minProbClipForWeighting=1e-4 ):
    """ 
    Computes chi^2 for a 2-outcome measurement using frequency-weighting.

    The chi-squared function for a 2-outcome measurement using 
    the observed frequency in the statistical weight.
    
    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClipForWeighting : float, optional
        unused but present to keep the same function 
        signature as ChiSqFunc_2outcome.
        
    Returns
    -------
    float or numpy array
        N(p-f)^2 / (f*(1-f*)), 
        where f* = (f*N+1)/N+2 is the frequency value used in the
        statistical weighting (prevents divide by zero errors)
    """
    f1 = (f*N+1)/(N+2)
    return N*(p-f)**2/(f1*(1-f1))


def ChiSqFunc( N, p, f, minProbClipForWeighting=1e-4 ):
    """ 
    Computes the chi^2 term corresponding to a single outcome.

    The chi-squared term for a single outcome of a multi-outcome
    measurement using a clipped probability for the statistical
    weighting.  
    
    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClipForWeighting : float, optional
        Defines clipping interval (see return value).
        
    Returns
    -------
    float or numpy array
        N(p-f)^2 / cp , 
        where cp is the value of p clipped to the interval
        (minProbClipForWeighting, 1-minProbClipForWeighting)        
    """
    #cp = _np.clip(p,minProbClipForWeighting,1-minProbClipForWeighting)
    cp = _np.clip(p,minProbClipForWeighting,1e10) #effectively no upper bound
    return N*(p-f)**2/cp


def ChiSqFunc_wFreqs( N, p, f, minProbClipForWeighting=1e-4 ):
    """ 
    Computes the frequency-weighed chi^2 term corresponding to a single outcome.

    The chi-squared term for a single outcome of a multi-outcome
    measurement using the observed frequency in the statistical weight.
    
    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClipForWeighting : float, optional
        unused but present to keep the same function 
        signature as ChiSqFunc.
        
    Returns
    -------
    float or numpy array
        N(p-f)^2 / f*, 
        where f* = (f*N+1)/N+2 is the frequency value used in the
        statistical weighting (prevents divide by zero errors)
    """
    f1 = (f*N+1)/(N+2)
    return N*(p-f)**2/f1

def TwoDeltaLogLFunc(N, p, f, minProbClip=1e-6, poissonPicture=True): 
    """
    Term of the 2*[log(L)-upper-bound - log(L)] sum corresponding
     to a single gate string and spam label.

    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClip : float, optional
        Minimum probability clip point to avoid evaluating
        log(number <= zero)

    poissonPicture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logL value.
        
    Returns
    -------
    float or numpy array
    """
    cp = _np.clip(p,minProbClip,1e10) #effectively no upper bound
    zf = _np.where(f < 1e-10, 0.0, f) #set zero-freqs to zero
    nzf = _np.where(f < 1e-10, 1.0, f) #set zero-freqs to one -- together w/above line makes 0 * log(0) == 0
    if poissonPicture:
        return 2 * (N * zf * _np.log(nzf/cp) - N * (f-cp))
    else:
        return 2 * N * zf * _np.log(nzf/cp)


def GateStringChiSq( gatestring, dataset, gateset, useFreqWeightedChiSq=False, 
                     minProbClipForWeighting=1e-4):
    """
    Computes the chi-squared term for a single gate string.

    The sum of the chi-squared terms for each of the possible
    outcomes (the SPAM labels in gateset).

    Parameters
    ----------
    gatestring : GateString or tuple
        The gate string to compute chi-squared for.

    dataset : Dataset
        The object from which frequencies are extracted.

    gateset : GateSet
        The object from which probabilities and SPAM labels are extracted.

    useFreqWeightedChiSq : bool, optional
        Whether or not frequencies (instead of probabilities) should be used 
        in statistical weight factors.

    minProbClipForWeighting : float, optional
        If useFreqWeightedChiSq == False, defines the clipping interval
        for the statistical weight (see ChiSqFunc).

    Returns
    -------
    float
        The sum of chi^2 terms for corresponding to gatestring
        (one term per SPAM label).
    """
    
    chiSqFn  = ChiSqFunc_wFreqs if useFreqWeightedChiSq else ChiSqFunc
    rowData  = dataset[gatestring]
    clip = minProbClipForWeighting

    N = rowData.total()
    p = gateset.Probs(gatestring)
    return sum( [ chiSqFn(N, p[sl], rowData[sl]/N, clip) for sl in gateset.get_SPAM_labels() ] )


def TotalCountMatrix( gateString, dataset, strs, rhoEPairs=None):
    """
    Computes the total count matrix for a base gatestring.

    Parameters
    ----------
    gateString : tuple of gate labels
        The gate sequence that is sandwiched between each EStr and rhoStr

    dataset : DataSet
        The data used to specify the counts

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the matrix.  Other values are set to NaN.

    Returns
    -------
    numpy array of shape (M,N)
        total count values (sum of count values for each SPAM label)
        corresponding to gate sequences where gateString is sandwiched 
        between the specified set of N rhoSpec and M ESpec gate strings.
    """
    rhoStrs, EStrs = strs # LEXICOGRAPHICAL VS MATRIX ORDER
    if rhoEPairs is None:
        mxlst = []
        for EStr in EStrs:
            rowLst = []
            for rhoStr in rhoStrs:
                gstr = rhoStr + gateString + EStr
                if gstr in dataset: rowLst.append( dataset[gstr].total() )
                else: rowLst.append( _np.nan )
            mxlst.append(rowLst)
        return _np.array( mxlst )
    else:
        ret = _np.nan * _np.ones( (len(EStrs),len(rhoStrs)), 'd')
        for i,j in rhoEPairs:
            gstr = rhoStrs[i] + gateString + EStrs[j]
            if gstr in dataset: ret[j,i] = dataset[ gstr ].total()
        return ret


def CountMatrix( gateString, dataset, spamlabel, strs, rhoEPairs=None ):
    """
    Computes spamLabel's count matrix for a base gatestring.

    Parameters
    ----------
    gateString : tuple of gate labels
        The gate sequence that is sandwiched between each EStr and rhoStr

    dataset : DataSet
        The data used to specify the counts

    spamLabel : string
        The spam label to extract counts for, e.g. 'plus'

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the matrix.  Other values are set to NaN.

    Returns
    -------
    numpy array of shape ( len(EStrs), len(rhoStrs) )
        count values corresponding to spamLabel and gate sequences 
        where gateString is sandwiched between the each (EStr,rhoStr) pair.
    """
    rhoStrs, EStrs = strs # LEXICOGRAPHICAL VS MATRIX ORDER
    if rhoEPairs is None:
        mxlst = []
        for EStr in EStrs:
            rowLst = []
            for rhoStr in rhoStrs:
                gstr = rhoStr + gateString + EStr
                if gstr in dataset: rowLst.append( dataset[gstr][spamlabel] )
                else: rowLst.append( _np.nan )
            mxlst.append(rowLst)
        return _np.array( mxlst )
    else:
        ret = _np.nan * _np.ones( (len(EStrs),len(rhoStrs)), 'd')
        for i,j in rhoEPairs:
            gstr = rhoStrs[i] + gateString + EStrs[j]
            if gstr in dataset: ret[j,i] = dataset[ gstr ][spamlabel]
        return ret




def FrequencyMatrix( gateString, dataset, spamlabel, strs, rhoEPairs=None):
    """
    Computes spamLabel's frequency matrix for a base gatestring.

    Parameters
    ----------
    gateString : tuple of gate labels
        The gate sequence that is sandwiched between each EStr and rhoStr

    dataset : DataSet
        The data used to specify the frequencies

    spamLabel : string
        The spam label to extract frequencies for, e.g. 'plus'

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the matrix.  Other values are set to NaN.

    Returns
    -------
    numpy array of shape ( len(EStrs), len(rhoStrs) )
        frequency values corresponding to spamLabel and gate sequences 
        where gateString is sandwiched between the each (EStr,rhoStr) pair.
    """
    return CountMatrix( gateString, dataset, spamlabel, strs, rhoEPairs) / TotalCountMatrix( gateString, dataset, strs, rhoEPairs)


def ProbabilityMatrix( gateString, gateset, spamlabel, strs, rhoEPairs=None):
    """
    Computes spamLabel's probability matrix for a base gatestring.

    Parameters
    ----------
    gateString : tuple of gate labels
        The gate sequence that is sandwiched between each EStr and rhoStr

    gateset : GateSet
        The gate set used to specify the probabilities

    spamLabel : string
        The spam label to extract probabilities for, e.g. 'plus'

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the matrix.  Other values are set to NaN.

    Returns
    -------
    numpy array of shape ( len(EStrs), len(rhoStrs) )
        probability values corresponding to spamLabel and gate sequences 
        where gateString is sandwiched between the each (EStr,rhoStr) pair.
    """
    rhoStrs, EStrs = strs # LEXICOGRAPHICAL VS MATRIX ORDER
    if rhoEPairs is None:
        return _np.array( [ [ gateset.Pr( spamlabel, rhoStr + gateString + EStr ) for rhoStr in rhoStrs ] for EStr in EStrs ] )
    else:
        ret = _np.nan * _np.ones( (len(EStrs),len(rhoStrs)), 'd')
        for i,j in rhoEPairs:
            ret[j,i] = gateset.Pr( spamlabel, rhoStrs[i] + gateString + EStrs[j] )
        return ret




def ChiSqMatrix( gateString, dataset, gateset, strs, minProbClipForWeighting=1e-4, rhoEPairs=None):
    """
    Computes the chi^2 matrix for a base gatestring.

    Parameters
    ----------
    gateString : tuple of gate labels
        The gate sequence that is sandwiched between each EStr and rhoStr

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight (see ChiSqFunc).

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the matrix.  Other values are set to NaN.

    Returns
    -------
    numpy array of shape ( len(EStrs), len(rhoStrs) )
        chi^2 values corresponding to gate sequences where 
        gateString is sandwiched between the each (EStr,rhoStr) pair.
        (i.e. element_ij = GateStringChiSq( rhoStrs[j] + gateString + EStrs[i])
    """
    rhoStrs, EStrs = strs
    chiSqMx = _np.zeros( (len(EStrs),len(rhoStrs)), 'd')
    if gateString is None: return _np.nan*chiSqMx
    cntMx  = TotalCountMatrix(  gateString, dataset, strs, rhoEPairs)
    for sl in gateset.get_SPAM_labels():
        probMx = ProbabilityMatrix( gateString, gateset, sl, strs, rhoEPairs)
        freqMx = FrequencyMatrix(   gateString, dataset, sl, strs, rhoEPairs)
        chiSqMx += ChiSqFunc( cntMx, probMx, freqMx, minProbClipForWeighting)
    return chiSqMx


def LogLMatrix( gateString, dataset, gateset, strs, minProbClip=1e-6, rhoEPairs=None):
    """
    Computes the log-likelihood matrix of 2*( log(L)_upperbound - log(L) ) 
    values for a base gatestring.

    Parameters
    ----------
    gateString : tuple of gate labels
        The gate sequence that is sandwiched between each EStr and rhoStr

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    minProbClip : float, optional
        defines the minimum probability "patch-point" of the log-likelihood function.

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the matrix.  Other values are set to NaN.

    Returns
    -------
    numpy array of shape ( len(EStrs), len(rhoStrs) )
        logL values corresponding to gate sequences where 
        gateString is sandwiched between the each (EStr,rhoStr) pair.
    """
    rhoStrs, EStrs = strs
    logLMx = _np.zeros( (len(EStrs),len(rhoStrs)), 'd')
    if gateString is None: return _np.nan*logLMx
    cntMx  = TotalCountMatrix(  gateString, dataset, strs, rhoEPairs)
    for sl in gateset.get_SPAM_labels():
        probMx = ProbabilityMatrix( gateString, gateset, sl, strs, rhoEPairs)
        freqMx = FrequencyMatrix(   gateString, dataset, sl, strs, rhoEPairs)
        logLMx += TwoDeltaLogLFunc( cntMx, probMx, freqMx, minProbClip)
    return logLMx




def TotalChiSquared(dataset, gateset, gateStrings=None,
                    returnGradient=False, returnHessian=False, 
                    G0=True, SP0=True, SPAM=True, gates=True,
                    minProbClipForWeighting=1e-4, clipTo=None,
                    useFreqWeightedChiSq=False, check=False):
    """ 
    Computes the total chi^2 for a set of gate strings.

    The chi^2 test statistic obtained by summing up the
    contributions of a given set of gate strings or all
    the strings available in a dataset.  Optionally,
    the gradient and/or Hessian of chi^2 can be returned too.

    Parameters
    ----------
    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    gateStrings : list of GateStrings or tuples, optional
        List of gate strings whose terms will be included in chi^2 sum.
        Default value (None) means "all strings in dataset".

    returnGradient, returnHessian : bool
        Whether to compute and return the gradient and/or Hessian of chi^2.

    G0, SP0, SPAM, gates : bool
        How to parameterize the gateset:
          G0 == whether first row of gate matrices are parameterized
          SP0 == whether first element of each rho vector is parameterized
          SPAM == whether SPAM (rho and E) vectors are parameterized
          gates == whether gate matrices are parameterized

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight (see ChiSqFunc).

    clipTo : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.Bulk_fillProbs)

    useFreqWeightedChiSq : bool, optional
        Whether or not frequencies (instead of probabilities) should be used 
        in statistical weight factors.

    check : bool, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    Returns
    -------
    chi2 : float
        chi^2 value, equal to the sum of chi^2 terms from all specified gate strings
    dchi2 : numpy array
        Only returned if returnGradient == True. The gradient vector of 
        length nGatesetParams, the number of gateset parameters. 
        (nGatesetParams depends on values of G0, SP0, SPAM, and gates).
    d2chi2 : numpy array
        Only returned if returnHessian == True. The Hessian matrix of 
        shape (nGatesetParams, nGatesetParams).
    """

    # Scratch work:
    # chi^2 = sum_i N_i*(p_i-f_i)^2 / p_i  (i over gatestrings & spam labels)
    # d(chi^2)/dx = sum_i N_i * [ 2(p_i-f_i)*dp_i/dx / p_i - (p_i-f_i)^2 / p_i^2 * dp_i/dx ]
    #             = sum_i N_i * (p_i-f_i) / p_i * [2 - (p_i-f_i)/p_i   ] * dp_i/dx
    #             = sum_i N_i * t_i * [2 - t_i ] * dp_i/dx     where t_i = (p_i-f_i) / p_i
    # d2(chi^2)/dydx = sum_i N_i * [ dt_i/dy * [2 - t_i ] * dp_i/dx - t_i * dt_i/dy * dp_i/dx + t_i * [2 - t_i] * d2p_i/dydx ]
    #                          where dt_i/dy = [ 1/p_i - (p_i-f_i) / p_i^2 ] * dp_i/dy
    if useFreqWeightedChiSq:
      raise ValueError("frequency weighted chi2 is not implemented yet.")

    spamLabels = gateset.get_SPAM_labels() #this list fixes the ordering of the spam labels
    spam_lbl_rows = { sl:i for (i,sl) in enumerate(spamLabels) }
    vec_gs_len = gateset.getNumParams(gates, G0, SPAM, SP0)

    if gateStrings is None:
      gateStrings = dataset.keys()

    nSpamLabels = len(spamLabels)
    nGateStrings = len(gateStrings)
    
    N      = _np.empty( nGateStrings )
    f      = _np.empty( (nSpamLabels, nGateStrings) )
    probs  = _np.empty( (nSpamLabels, nGateStrings) )

    if returnGradient or returnHessian:
      dprobs = _np.empty( (nSpamLabels, nGateStrings,vec_gs_len) )
    if returnHessian:
      hprobs = _np.empty( (nSpamLabels, nGateStrings,vec_gs_len,vec_gs_len) )

    for (i,gateStr) in enumerate(gateStrings):
        N[i] = float(dataset[gateStr].total())
        for k,sl in enumerate(spamLabels):
            f[k,i] = dataset[gateStr].fraction(sl)

    evTree = gateset.Bulk_evalTree(gateStrings)
    
    if returnHessian:
      gateset.Bulk_fillhProbs(hprobs, spam_lbl_rows, evTree, 
                              gates, G0, SPAM, SP0, probs, dprobs, clipTo,
                              check)
    elif returnGradient:
      gateset.Bulk_filldProbs(dprobs, spam_lbl_rows, evTree, 
                              gates, G0, SPAM, SP0, probs, clipTo, check)
    else:
      gateset.Bulk_fillProbs(probs, spam_lbl_rows, evTree, clipTo, check)


    #cprobs = _np.clip(probs,minProbClipForWeighting,1-minProbClipForWeighting) #clipped probabilities (also clip derivs to 0?)
    cprobs = _np.clip(probs,minProbClipForWeighting,1e10) #effectively no upper bound
    chi2 = _np.sum( N[None,:] * ((probs - f)**2/cprobs), axis=(0,1) ) # Note (0,1) are all axes in this case
    #TODO: try to replace final N[...] multiplication with dot or einsum, or do summing sooner to reduce memory

    if returnGradient:
      t = ((probs - f)/cprobs)[:,:,None] # (iSpamLabel, iGateString, 0) = (K,M,1)
      dchi2 = N[None,:,None] * t * (2 - t) * dprobs  # (1,M,1) * (K,M,1) * (K,M,N)  (K=#spam, M=#strings, N=#vec_gs)
      dchi2 = _np.sum( dchi2, axis=(0,1) ) # sum over gate strings and spam labels => (N)
      
    if returnHessian:
      dprobs_p = dprobs[:,:,None,:] # (K,M,1,N)
      t = ((probs - f)/cprobs)[:,:,None,None] # (iSpamLabel, iGateString, 0,0) = (K,M,1,1)
      dt = ((1.0/cprobs - (probs-f)/cprobs**2)[:,:,None] * dprobs)[:,:,:,None] # (K,M,1) * (K,M,N) = (K,M,N) => (K,M,N,1)
      d2chi2 = N[None,:,None,None] * (dt * (2 - t) * dprobs_p - t * dt * dprobs_p + t * (2 - t) * hprobs)
      d2chi2 = _np.sum( d2chi2, axis=(0,1) ) # sum over gate strings and spam labels => (N1,N2)
        # (1,M,1,1) * ( (K,M,N1,1) * (K,M,1,1) * (K,M,1,N2) + (K,M,1,1) * (K,M,N1,1) * (K,M,1,N2) + (K,M,1,1) * (K,M,1,1) * (K,M,N1,N2) )
      
    if returnGradient:
      return (chi2, dchi2, d2chi2) if returnHessian else (chi2, dchi2)
    else:
      return (chi2, d2chi2) if returnHessian else chi2
    


def _oldTotalChiSquared( dataset, gateset, gateStrings=None, useFreqWeightedChiSq=False,
                     minProbClipForWeighting=1e-4):
    """ 
    Return summed chi^2 values for given gateStrings or all gatestrings
    in the given dataset (if gateStrings is None).
    If useFreqWeightedChiSq is True, then use the dataset frequencies instead
    of the (correct) gateset probabilities in the expression for "Chi^2".
    """

    if gateStrings is None:
        return sum( [ GateStringChiSq( gatestring, dataset, gateset, useFreqWeightedChiSq, 
                                       minProbClipForWeighting) for gatestring in dataset] )
        #return sum( [ ChiSqFunc_2outcome( dsRow.total(), 
        #                     gateset.Pr('plus', gatestring), 
        #                     dsRow.fraction('plus'),
        #                     minProbClipForWeighting ) for gatestring,dsRow in dataset.iteritems() ] )
    else:
        return sum( [ GateStringChiSq( gatestring, dataset, gateset, useFreqWeightedChiSq, 
                                       minProbClipForWeighting) for gatestring in gateStrings] )



def SmallEigvalErrRate(sigma, dataset, directGSTgatesets):
    """
    Compute per-gate error rate.

    The per-gate error rate, extrapolated from the smallest eigvalue 
    of the Direct GST estimate of the given gate string sigma.

    Parameters
    ----------
    sigma : GateString or tuple of gate labels
        The gate sequence that is used to estimate the error rate

    dataset : DataSet
        The dataset used obtain gate string frequencies

    directGSTgatesets : dictionary of GateSets
        A dictionary with keys = gate strings and
        values = GateSets.

    Returns
    -------
    float
        the approximate per-gate error rate.
    """
    if sigma is None: return _np.nan # in plot processing, "None" gatestrings = no plot output = nan values
    gs_direct = directGSTgatesets[sigma]
    minEigval = min(abs(_np.linalg.eigvals( gs_direct["sigmaLbl"] )))
    return 1.0 - minEigval**(1.0/max(len(sigma),1)) # (approximate) per-gate error rate; max averts divide by zero error


def besttxtcolor( x, xmin, xmax ):
    """ 
    Determinining function for whether text should be white or black

    Parameters
    ----------
    x, xmin, xmax : float
        Values of the cell in question, the minimum cell value, and the maximum cell value

    Returns
    -------
    {"white","black"}
    """
    if ((x-xmin)/xmax<0.37) or ((x-xmin)/xmax>0.77):
        return "white"
    else:
        return "black"
    
def ColorBoxPlot(plt_data, title=None, xlabels=None, ylabels=None, xtics=None, ytics=None,
                 vmin=None, vmax=None, colorbar=True, fig=None, axes=None, size=None, prec=0, boxLabels=True,
                 xlabel=None, ylabel=None, saveTo=None, ticSize=14, grid=False):
    """
    Create a color box plot.

    Creates a figure composed of colored boxes and possibly labels.

    Parameters
    ----------
    plt_data : numpy array
        A 2D array containing the values to be plotted.

    title : string, optional
        Plot title (latex can be used)

    xlabels, ylabels : list of strings, optional
        Tic labels for x and y axes.  If both are None, then tics are not drawn.

    xtics, ytics : list or array of floats, optional
        Values of x and y axis tics.  If None, then half-integers from 0.5 to 
        0.5 + (nCols-1) or 0.5 + (nRows-1) are used, respectively.

    vmin, vmax : float, optional
        Min and max values of the color scale.

    colorbar : bool, optional
        Whether to display a colorbar or not.

    fig, axes : matplotlib figure and axes, optional
        If non-None, use these figure and axes objects instead of creating new ones
        via fig,axes = pyplot.supblots()

    size : 2-tuple, optional
        The width and heigh of the final figure in inches.

    prec : int or {'compact','compacthp'}, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    xlabel, ylabel : str, optional
        X and Y axis labels

    saveTo : str, optional
        save figure as this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    grid : bool, optional
        Whether or not grid lines should be displayed.
    
    Returns
    -------
    GSTFigure
        The encapsulated matplotlib figure that was generated
    """
    if axes is None: fig,axes = _plt.subplots()  # create a new figure if no axes are given

    finite_plt_data_flat = _np.take(plt_data.flat, _np.where(_np.isfinite(plt_data.flat)))[0]
    if vmin is None: vmin = min( finite_plt_data_flat )
    if vmax is None: vmax = max( finite_plt_data_flat )

    cmap = _matplotlib.cm.jet
    cmap.set_bad('w',1)
    masked_data = _np.ma.array (plt_data, mask=_np.isnan(plt_data))
    #heatmap = ax.pcolor( plt_data, vmin=vmin, vmax=vmax)
    heatmap = axes.pcolormesh( masked_data, vmin=vmin, vmax=vmax, cmap=cmap)

    if size is not None and fig is not None:
        fig.set_size_inches(size[0],size[1]) # was 12,8 for "super" color plot

    axes.set_xlim(0,plt_data.shape[1])
    axes.set_ylim(0,plt_data.shape[0])

    if xlabels is not None:
        if xtics is None:
            xtics = _np.arange(plt_data.shape[1])+0.5
        axes.set_xticks(xtics, minor=False)
        axes.set_xticklabels( xlabels,rotation=0, fontsize=ticSize )
    if ylabels is not None:
        if ytics is None: 
            ytics = _np.arange(plt_data.shape[0])+0.5
        axes.set_yticks(ytics, minor=False)
        axes.set_yticklabels( ylabels, fontsize=ticSize )

    if grid:
        def getMinorTics(t):
            return [ (t[i]+t[i+1])/2.0 for i in range(len(t)-1) ]
        axes.set_xticks(getMinorTics(xtics), minor=True)
        axes.set_yticks(getMinorTics(ytics), minor=True)
        axes.grid(which='minor', axis='both', linestyle='-', linewidth=2)


    if xlabels is None and ylabels is None:
        axes.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off') #white tics
    else:
        axes.tick_params(top='off', bottom='off', left='off', right='off')

    if title is not None:
        axes.set_title( title, fontsize=(ticSize+4) )

    if xlabel is not None:
        axes.set_xlabel( xlabel, fontsize=(ticSize+4) )

    if ylabel is not None:
        axes.set_ylabel( ylabel, fontsize=(ticSize+4) )

    def eformat(f, prec):  #NOTE: prec doesn't actually do anything anymore
        if prec == 'compact' or prec == 'compacthp':
            if f < 0: 
                return "-" + eformat(-f,prec)

            if prec == 'compacthp':
                if f < 0.005: #can't fit in 2 digits; would just be .00, so just print "0"
                    return "0"
                if f < 1:
                    z = "%.2f" % f # print first two decimal places
                    if z.startswith("0."): return z[1:]  # fails for '1.00'; then thunk down to next f<10 case
                if f < 10:
                    return "%.1f" % f # print whole number and tenths

            if f < 100: 
                return "%.0f" % f # print nearest whole number if only 1 or 2 digits
            
            #if f >= 100, minimal scientific notation, such as "4e7", not "4e+07"
            s = "%.0e" % f
            try:
                mantissa, exp = s.split('e')
                exp = int(exp)
                if exp >= 100: return "B" #if number is too big to print
                if exp >= 10: return "*%d" % exp
                return "%se%d" % (mantissa, exp)
            except:
                return str(s)[0:3]

        elif type(prec) == int:
            if prec >= 0:
                return "%.*f" % (prec,f)
            else: 
                return "%.*g" % (-prec,f)
        else:
            return "%g" % f #fallback to general format

    if boxLabels:
        # Write values on colored squares
        for y in range(plt_data.shape[0]):
            for x in range(plt_data.shape[1]):
                if _np.isnan(plt_data[y, x]): continue
                axes.text(x + 0.5, y + 0.5, eformat(plt_data[y, x], prec),
                        horizontalalignment='center',
                        verticalalignment='center', color=besttxtcolor( plt_data[y,x], vmin, vmax) )

    if colorbar:
        _plt.colorbar(heatmap)

    gstFig = GSTFigure(axes)

    if saveTo is not None:
        if len(saveTo) > 0: #So you can pass saveTo="" and figure will be closed but not saved to a file
            _plt.savefig(saveTo, bbox_extra_artists=(axes,), bbox_inches='tight') #need extra artists otherwise axis labels get clipped
        if fig is not None: _plt.close(fig) #close the figure if we're saving it to a file

    return gstFig



def NestedColorBoxPlot(plt_data_list_of_lists, title=None, xlabels=None, ylabels=None, xtics=None, ytics=None,
                       vmin=None, vmax=None, colorbar=True, fig=None, axes=None, size=None, prec=0, 
                       boxLabels=True, xlabel=None, ylabel=None, saveTo=None, ticSize=14, grid=False):
    """
    Create a color box plot.

    Creates a figure composed of colored boxes and possibly labels.

    Parameters
    ----------
    plt_data_list_of_lists : list of lists of numpy arrays
        A complete square 2D list of lists, such that each element is a
        2D numpy array of the same size.

    title : string, optional
        Plot title (latex can be used)

    xlabels, ylabels : list of strings, optional
        Tic labels for x and y axes.  If both are None, then tics are not drawn.

    xtics, ytics : list or array of floats, optional
        Values of x and y axis tics.  If None, then half-integers from 0.5 to 
        0.5 + (nCols-1) or 0.5 + (nRows-1) are used, respectively.

    vmin, vmax : float, optional
        Min and max values of the color scale.

    colorbar : bool, optional
        Whether to display a colorbar or not.

    fig, axes : matplotlib figure and axes, optional
        If non-None, use these figure and axes objects instead of creating new ones
        via fig,axes = pyplot.supblots()

    size : 2-tuple, optional
        The width and heigh of the final figure in inches.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    xlabel, ylabel : str, optional
        X and Y axis labels

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    grid : bool, optional
        Whether or not grid lines should be displayed.
    
    Returns
    -------
    GSTFigure
        The encapsulated matplotlib figure that was generated
    """

    #Assume a complete 2D rectangular list of lists, and that each element is a numpy array of the same size
    if len(plt_data_list_of_lists) == 0 or len(plt_data_list_of_lists[0]) == 0: return
    elRows,elCols = plt_data_list_of_lists[0][0].shape #nE,nr
    nRows = len(plt_data_list_of_lists)
    nCols = len(plt_data_list_of_lists[0])

    data = _np.zeros( ( elRows*nRows + (nRows-1), elCols*nCols + (nCols-1)) )
    for i in range(1,nRows):
        data[(elRows+1)*i-1:(elRows+1)*i,:] = _np.nan
    for j in range(1,nCols):
        data[:, (elCols+1)*j-1:(elCols+1)*j] = _np.nan

    for i in range(nRows):
        for j in range(nCols):
            data[(elRows+1)*i:(elRows+1)*(i+1)-1, (elCols+1)*j:(elCols+1)*(j+1)-1] = plt_data_list_of_lists[i][j]

    xtics = []; ytics = []
    for i in range(nRows):   ytics.append( float((elRows+1)*(i+0.5)) )
    for j in range(nCols):   xtics.append( float((elCols+1)*(j+0.5)) )

    return ColorBoxPlot(data,title, xlabels, ylabels, _np.array(xtics), _np.array(ytics),
                        vmin, vmax, colorbar, fig, axes, size, prec, boxLabels, xlabel, ylabel,
                        saveTo, ticSize, grid)

def _computeSubMxs(xvals, yvals, xyGateStringDict, subMxCreationFn):
    subMxs = [ [ subMxCreationFn( xyGateStringDict[(x,y)] ) for x in xvals ] for y in yvals]
    return subMxs #Note: subMxs[y-index][x-index] is proper usage

def generateBoxPlot( xvals, yvals, xyGateStringDict, subMxCreationFn, xlabel="", ylabel="", m=None, M=None, scale=1.0, prec=0, 
                     title='sub-mx', sumUp=False, interactive=False, boxLabels=True, histogram=False, histBins=50, saveTo=None,
                     ticSize=20, invert=False, inner_x_labels=None, inner_y_labels=None, inner_x_label=None, inner_y_label=None,
                     grid=False):
    """
    Creates a view of nested box plot data (i.e. a matrix for each (x,y) pair).

    Given lists of x and y values, a dictionary to convert (x,y) pairs into gate strings,
    and a function to convert a "base" gate string into a matrix of floating point values,
    this function computes (x,y) => matrix data and displays it in one of two ways:

    1. As a full nested color box plot, showing all the matrix values individually
    2. As a color box plot containing the sum of the elements in the (x,y) matrix as
       the (x,y) box.

    A histogram of the values can also be computed and displayed.  Setting
    interactive == True allows the user to interactively change the color scale's min and
    max values.
    
    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xyGateStringDict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.
    
    subMxCreationFn : function
        A function that takes a singe gate string parameter and returns a matrix of values to 
        display.  If the function is passed None instead of a gate string, the function 
        should return an appropriately sized matrix of NaNs to indicate these elements should
        not be displayed.

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    inner_x_labels, inner_y_labels : list, optional
        Similar to xvals, yvals but labels for the columns and rows of the (x,y) matrices
        computed by subMxCreationFn.  Used when invert == True.

    grid : bool, optional
        Whether or not grid lines should be displayed.

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Note that 
        figure extra info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """

    init_min_clip = m
    init_max_clip = M

    used_xvals = [ x for x in xvals if any([ (xyGateStringDict[(x,y)] is not None) for y in yvals]) ]
    used_yvals = [ y for y in yvals if any([ (xyGateStringDict[(x,y)] is not None) for x in xvals]) ]

    nXs,nYs = len(used_xvals),len(used_yvals)

    def valFilter(vals):  #filter to latex-ify gate strings.  Later add filter as a possible parameter
        formatted_vals = []
        for val in vals:
            if type(val) == tuple and all([type(el) == str for el in val]):
                if len(val) == 0:
                    formatted_vals.append(r"$\{\}$")
                else:
                    formatted_vals.append( "$" + "\\cdot".join([("\\mathrm{%s}" % el) for el in val]) + "$" )
            else:
                formatted_vals.append(val)
        return formatted_vals
                

    if interactive:
        from IPython.html import widgets
        from IPython.html.widgets import interact, fixed

    #Compute sub-matrices (which are either displayed as nested sub-boxes of plot or are summed)
    subMxs = _computeSubMxs(used_xvals, used_yvals, xyGateStringDict, subMxCreationFn)

    def strToFloat(s):
        if s is None or s == "None" or len(str(s)) == 0: return None
        else: return float(s)

    def sumUpMx(mx):
        flat_mx = mx.flatten()
        if any([_np.isnan(x) for x in flat_mx]):
            if all([_np.isnan(x) for x in flat_mx]): 
                return _np.nan
            return sum(_np.nan_to_num(flat_mx)) #replace NaNs with zeros for purpose of summing (when there's at least one non-NaN)
        else:
            return sum(flat_mx)

    #Setup and create plotting functions
    if sumUp:
        subMxSums = _np.array( [ [ sumUpMx(subMxs[iy][ix]) for ix in range(nXs) ] for iy in range(nYs) ], 'd' )
        if invert: print "Warning: cannot invert a summed-up plot.  Ignoring invert=True."

        def makeplot(min_clip, max_clip):
            minclip = strToFloat( min_clip )
            maxclip = strToFloat( max_clip )
            fig,ax = _plt.subplots( 1, 1, figsize=(nXs*scale, nYs*scale))
            gstFig = ColorBoxPlot( subMxSums, fig=fig, axes=ax, title=title,
                                   xlabels=valFilter(used_xvals), ylabels=valFilter(used_yvals),
                                   vmin=minclip, vmax=maxclip, colorbar=False, prec=prec, xlabel=xlabel, ylabel=ylabel,
                                   ticSize=ticSize, grid=grid)
            gstFig.saveTo(saveTo)

            if histogram:
                fig = _plt.figure()
                histdata = subMxSums.flatten()
                histdata_finite = _np.take(histdata, _np.where(_np.isfinite(histdata)))[0] #take gives back (1,N) shaped array (why?)
                histMin = min( histdata_finite ) if minclip is None else minclip
                histMax = max( histdata_finite ) if maxclip is None else maxclip
                _plt.hist(_np.clip(histdata_finite,histMin,histMax), histBins,
                          range=[histMin, histMax], facecolor='gray', align='mid')
                if saveTo is not None:
                    if len(saveTo) > 0:
                        _plt.savefig( _makeHistFilename(saveTo) )
                    _plt.close(fig)                    
            return gstFig


        if interactive:
            interact(makeplot, 
                     min_clip=str(init_min_clip) if init_min_clip is not None else str(0),
                     max_clip=str(init_max_clip) if init_max_clip is not None else str(10) )
        else:
            gstFig = makeplot(init_min_clip, init_max_clip)

    else: #not summing up

        nIYs = nIXs = 0
        for ix in range(nXs):
            for iy in range(nYs):
                if subMxs[iy][ix] is not None:
                    nIYs,nIXs = subMxs[iy][ix].shape
                    break
        
        if invert:
            invertedSubMxs = []  #will be indexed as invertedSubMxs[inner-y][inner-x]
            for iny in range(nIYs):
                invertedSubMxs.append( [] )
                for inx in range(nIXs):
                    mx = _np.array( [[ subMxs[iy][ix][iny,inx] for ix in range(nXs) ] for iy in range(nYs)],  'd' )
                    invertedSubMxs[-1].append( mx )

            #Replace usual params with ones corresponding to "inverted" plot
            subMxs = invertedSubMxs
            used_xvals = inner_x_labels if inner_x_labels else [""]*nIXs
            used_yvals = inner_y_labels if inner_y_labels else [""]*nIYs
            xlabel = inner_x_label if inner_x_label else ""
            ylabel = inner_y_label if inner_y_label else ""
            nXs, nYs, nIXs, nIYs = nIXs, nIYs, nXs, nYs #swap nXs <=> nIXs b/c of inversion

        def makeplot(min_clip, max_clip, labels):
            minclip = strToFloat( min_clip )
            maxclip = strToFloat( max_clip )
            #print "data = ",subMxs
            fig,ax = _plt.subplots( 1, 1, figsize=(nXs*nIXs*scale*0.4, nYs*nIYs*scale*0.4))
            gstFig = NestedColorBoxPlot(subMxs, fig=fig, axes=ax, title=title,vmin=minclip, vmax=maxclip, prec=prec, 
                                        ylabels=valFilter(used_yvals), xlabels=valFilter(used_xvals), boxLabels=labels,
                                        colorbar=False, ylabel=ylabel, xlabel=xlabel, ticSize=ticSize, grid=grid)
            gstFig.saveTo(saveTo)

            if histogram:
                fig = _plt.figure()
                histdata = _np.concatenate( [ subMxs[iy][ix].flatten() for ix in range(nXs) for iy in range(nYs)] )
                histdata_finite = _np.take(histdata, _np.where(_np.isfinite(histdata)))[0] #take gives back (1,N) shaped array (why?)
                histMin = min( histdata_finite ) if minclip is None else minclip
                histMax = max( histdata_finite ) if maxclip is None else maxclip
                _plt.hist(_np.clip(histdata_finite,histMin,histMax), histBins,
                          range=[histMin, histMax], facecolor='gray', align='mid')
                if saveTo is not None:
                    if len(saveTo) > 0:
                        _plt.savefig( _makeHistFilename(saveTo) )
                    _plt.close(fig)
            return gstFig


        if interactive:
            interact(makeplot,
                     min_clip=str(init_min_clip) if init_min_clip is not None else str(0),
                     max_clip=str(init_max_clip) if init_max_clip is not None else str(10),
                     labels=boxLabels)
        else:
            gstFig = makeplot(init_min_clip, init_max_clip, boxLabels)

    gstFig.setExtraInfo( { 'nUsedXs': len(used_xvals),
                           'nUsedYs': len(used_yvals) } )                     
    # gstFig.check() #DEBUG - test that figure can unpickle correctly -- if not, probably used magic matplotlib (don't do that)
    return gstFig


def generateZoomedBoxPlot(xvals, yvals, xyGateStringDict, subMxCreationFn, strs, 
                          xlabel="", ylabel="", m=None, M=None, scale=1.0, prec=0, title='sub-mx',
                          saveTo=None, ticSize=14):
    """
    Creates an interactive view of one (x,y) matrix of nested box plot data.

    Given lists of x and y values, a dictionary to convert (x,y) pairs into gate strings,
    and a function to convert a "base" gate string into a matrix of floating point values,
    this function computes (x,y) => matrix data and interactively displays a single (x,y)
    matrix as a color box plot.  The user can change x and y interactively to display 
    the color box plots corresponding to different (x,y) pairs.
    
    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xyGateStringDict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.
    
    subMxCreationFn : function
        A function that takes a singe gate string parameter and returns a matrix of values to 
        display.  If the function is passed None instead of a gate string, the function 
        should return an appropriately sized matrix of NaNs to indicate these elements should
        not be displayed.

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    Returns
    -------
    None
    """

    rhoStrs, EStrs = strs
    init_min_clip = m
    init_max_clip = M
    nXs, nYs = len(xvals), len(yvals)

    from IPython.html import widgets
    from IPython.html.widgets import interact, fixed

    def valFilter(vals):  #filter to latex-ify gate strings.  Later add filter as a possible parameter
        formatted_vals = []
        for val in vals:
            if len(val) == 1 and val[0] == 0:
                formatted_vals.append(r"$\{\}$")
            elif type(val) == tuple and all([type(el) == str for el in val[1:]]) and val[0] == 0:
                formatted_vals.append( "$" + "\\cdot".join([("\\mathrm{%s}" % el) for el in val[1:]]) + "$" )
            elif type(val) == tuple and all([type(el) == str for el in val[:-1]]) and val[-1] == 0:
                formatted_vals.append( "$" + "\\cdot".join([("\\mathrm{%s}" % el) for el in val[:-1]]) + "$" )
            else:
                formatted_vals.append(val)
        return formatted_vals


    #Compute sub-matrices
    subMxs = _computeSubMxs(xvals, yvals, xyGateStringDict, subMxCreationFn)

    def strToFloat(s):
        if s is None or s == "None" or len(str(s)) == 0: return None
        else: return float(s)

    cb = False; prefix = "Chi squareds for "

    def makeplot(min_clip, max_clip, x,y):
        minclip = strToFloat(min_clip)
        maxclip = strToFloat(max_clip)
        zoomToX,zoomToY = x,y

        fig, ax = _plt.subplots( 1, 1, figsize=(len(rhoStrs)*scale, len(EStrs)*scale))
        ix,iy = xvals.index(zoomToX), yvals.index(zoomToY)
        ColorBoxPlot( subMxs[iy][ix], fig=fig, axes=ax, 
                      title=title + " for %s=%s, %s=%s" % (xlabel,str(zoomToX),ylabel,str(zoomToY)),
                      xlabels=valFilter(rhoStrs), ylabels=valFilter(EStrs), vmin=minclip, vmax=maxclip, colorbar=False, 
                      prec=prec, saveTo=saveTo, ticSize=ticSize)

    interact(makeplot,
             min_clip=str(init_min_clip) if init_min_clip is not None else str(0),
             max_clip=str(init_max_clip) if init_max_clip is not None else str(10),
             y=dict([(str(y),y) for y in yvals]),
             x=dict([(str(x),x) for x in xvals]) )



def ChiSqBoxPlot( xvals, yvals, xy_gatestring_dict, dataset, gateset, strs,
                  xlabel="", ylabel="", m=None, M=None, scale=1.0, prec='compact', title='$\\chi^2$', sumUp=False,
                  interactive=False, boxLabels=True, histogram=False, histBins=50, minProbClipForWeighting=1e-4,
                  saveTo=None, ticSize=20, invert=False, rhoEPairs=None):
    """
    Create a color box plot of chi^2 values.

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the plot.

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """

    rhoStrs, EStrs = strs
    def mxFn(gateStr):
        return ChiSqMatrix( gateStr, dataset, gateset, strs, minProbClipForWeighting, rhoEPairs)
    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel,ylabel, m,M,
                            scale,prec,title,sumUp,interactive,boxLabels,histogram,histBins,saveTo,ticSize,
                            invert, rhoStrs, EStrs, r"$\rho_i$", r"$E_i$")

def LogLBoxPlot( xvals, yvals, xy_gatestring_dict, dataset, gateset, strs,
                  xlabel="", ylabel="", m=None, M=None, scale=1.0, prec='compact', title='$\\log(L)$', sumUp=False,
                  interactive=False, boxLabels=True, histogram=False, histBins=50, minProbClipForWeighting=1e-4,
                  saveTo=None, ticSize=20, invert=False, rhoEPairs=None):
    """
    Create a color box plot of log-likelihood values.

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the logL function.

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the plot.

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """

    rhoStrs, EStrs = strs
    def mxFn(gateStr):
        return LogLMatrix( gateStr, dataset, gateset, strs, minProbClipForWeighting, rhoEPairs)
    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel,ylabel, m,M,
                            scale,prec,title,sumUp,interactive,boxLabels,histogram,histBins,saveTo,ticSize,
                            invert, rhoStrs, EStrs, r"$\rho_i$", r"$E_i$")



def BlankBoxPlot( xvals, yvals, xy_gatestring_dict, strs, xlabel="", ylabel="",
                  scale=1.0, title='', sumUp=False, saveTo=None, ticSize=20, invert=False):
    """
    Create only the outline of a color box plot.

    This function has been useful for creating presentations
    containing box plots to introduce the viewer to these
    types of plots.

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """
    rhoStrs, EStrs = strs
    def mxFn(gateStr):
        return _np.nan * _np.zeros( (len(strs[1]),len(strs[0])), 'd')
    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel,ylabel, 0,1,
                            scale,'compact',title,sumUp,False,False,False,0,saveTo,ticSize,
                            invert, rhoStrs, EStrs, r"$\rho_i$", r"$E_i$",True)



def ZoomedChiSqBoxPlot(xvals, yvals, xy_gatestring_dict, dataset, gateset, strs, 
                       xlabel="", ylabel="", m=None, M=None, scale=1.0, prec='compact', title='$\\chi^2$',
                       minProbClipForWeighting=1e-4, saveTo=None, ticSize=14):
    """
    Create an interactive zoomed color box plot of chi^2 values (within an iPython notebook)

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used to specify the probabilities and SPAM labels

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    Returns
    -------
    None
    """
    def mxFn(gateStr):
        return ChiSqMatrix( gateStr, dataset, gateset, strs, minProbClipForWeighting)
    generateZoomedBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, strs, xlabel, ylabel, m,M,scale,prec,title,saveTo,ticSize)


def SmallEigvalErrRateBoxPlot( xvals, yvals, xy_gatestring_dict, dataset, directGSTgatesets,
                               xlabel="", ylabel="", m=None, M=None, scale=1.0, prec=-1, 
                               title='Error rate, extrap. from small eigenvalue of Direct GST estimate',
                               interactive=False, boxLabels=True, histogram=False, histBins=50,
                               saveTo=None, ticSize=14):
    """
    Create a color box plot of per-gate error rates.

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    directGSTgatesets : dict
        Dictionary with keys == gate strings and values == GateSets linking a gate
        string to correponding direct-GST gate set.

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """

    def mxFn(gateStr): #error rate as 1x1 matrix which we have plotting function sum up
        return _np.array( [[ SmallEigvalErrRate(gateStr, dataset,  directGSTgatesets) ]] )
    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel,ylabel, m,M,
                            scale,prec,title, True,interactive,boxLabels,histogram,histBins,saveTo,ticSize)


            
def GatesetWithLGSTGatestringEstimates( gateStringsToEstimate, dataset, specs,
                                        targetGateset=None, includeTargetGates=True,
                                        spamDict=None, guessGatesetForGauge=None,
                                        gateStringLabels=None, svdTruncateTo=0, verbosity=0 ):
    """
    Constructs a gateset that contains LGST estimates for gateStringsToEstimate.

    For each gate string s in gateStringsToEstimate, the constructed gateset
    contains the LGST estimate for s as separate gate, labeled either by 
    the corresponding element of gateStringLabels or by the tuple of s itself.

    Parameters
    ----------
    gateStringsToEstimate : list of GateStrings or tuples
        The gate strings to estimate using LGST

    dataset : DataSet
        The data to use for LGST
        
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...)

    targetGateset : GateSet, optional
        The target gate set used by LGST to extract gate labels and an initial gauge

    includeTargetGates : bool, optional
        If True, the gate labels in targetGateset will be included in the
        returned gate set.

    spamDict : dict, optional
        Dictionary mapping (rhoVec_index,EVec_index) integer tuples to string spam labels.
        Defaults to the spam dictionary of targetGateset

    guessGatesetForGauge : GateSet, optional
        A gateset used to compute a gauge transformation that is applied to
        the LGST estimates.  This gauge transformation is computed such that
        if the estimated gates matched the gateset given, then the gate 
        matrices would match, i.e. the gauge would be the same as
        the gateset supplied. Defaults to the targetGateset.

    gateStringLabels : list of strings, optional
        A list of labels in one-to-one correspondence with the 
        gate string in gateStringsToEstimate.  These labels are
        the keys to access the gate matrices in the returned
        GateSet, i.e. gate_matrix = returned_gateset[gate_label]

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Defaults to no truncation.

    verbosity : int, optional
        Verbosity value to send to doLGST(...) call.

    Returns
    -------
    Gateset
        A gateset containing LGST estimates for all the requested 
        gate strings and possibly the gates in targetGateset.
    """
    gateLabels = [] #list of gate labels for LGST to estimate

    #Add gate strings to estimate as aliases
    aliases = { }
    if gateStringLabels is not None: 
        assert(len(gateStringLabels) == len(gateStringsToEstimate))
        for gateLabel,gateStr in zip(gateStringLabels,gateStringsToEstimate):
            aliases[gateLabel] = tuple(gateStr)
            gateLabels.append(gateLabel)
    else:
        for gateStr in gateStringsToEstimate:
            aliases[tuple(gateStr)] = tuple(gateStr) #use gatestring tuple as label
            gateLabels.append(tuple(gateStr))
            
    #Add target gateset labels (not aliased) if requested
    if includeTargetGates and targetGateset is not None:
        for targetGateLabel in targetGateset:
            if targetGateLabel not in gateLabels: #very unlikely that this is false
                gateLabels.append(targetGateLabel)
        
    return _Core.doLGST( dataset, specs, targetGateset, gateLabels, aliases, spamDict,
                         guessGatesetForGauge, svdTruncateTo, None, verbosity )

def DirectLGSTGateset( gateStringToEstimate, gateStringLabel, dataset, 
                       specs, targetGateset, svdTruncateTo=0, verbosity=0 ):
    """
    Constructs a gateset of LGST estimates for target gates and gateStringToEstimate.

    Parameters
    ----------
    gateStringToEstimate : GateString or tuple
        The single gate string to estimate using LGST

    gateStringLabel : string
        The label for the estimate of gateStringToEstimate.
        i.e. gate_matrix = returned_gateset[gate_label]

    dataset : DataSet
        The data to use for LGST
        
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...)

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Defaults to no truncation.

    verbosity : int, optional
        Verbosity value to send to doLGST(...) call.

    Returns
    -------
    Gateset
        A gateset containing LGST estimates of gateStringToEstimate
        and the gates of targetGateset.
    """
    return GatesetWithLGSTGatestringEstimates( [gateStringToEstimate], dataset, specs, targetGateset,
                                               True, None, None, [gateStringLabel], svdTruncateTo, verbosity )

def DirectLGSTGatesets(gateStrings, dataset, specs, targetGateset, svdTruncateTo=0, verbosity=0):
    """
    Constructs a dictionary with keys == gate strings and values == Direct-LGST GateSets.

    Parameters
    ----------
    gateStrings : list of GateString or tuple objects
        The gate strings to estimate using LGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST estimates.
        
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...)

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Defaults to no truncation.

    verbosity : int, optional
        Verbosity value to send to doLGST(...) call.

    Returns
    -------
    dict
        A dictionary that relates each gate string of gateStrings to a
        GateSet containing the LGST estimate of that gate string stored under 
        the gate label "sigmaLbl", along with LGST estimates of the gates in
        targetGateset.
    """    
    directLGSTgatesets = {}
    if verbosity > 0: print "--- Direct LGST precomputation ---"
    for i,sigma in enumerate(gateStrings):
        if verbosity > 0: print "--- Computing gateset for string %d of %d ---" % (i,len(gateStrings))
        directLGSTgatesets[sigma] = DirectLGSTGateset( sigma, "sigmaLbl", dataset, specs, targetGateset,
                                                        svdTruncateTo, verbosity)
    return directLGSTgatesets



def DirectLSGSTGateset( gateStringToEstimate, gateStringLabel, dataset, specs, targetGateset, svdTruncateTo=0,
                        minProbClipForWeighting=1e-4, probClipInterval=None, verbosity=0 ):
    """
    Constructs a gateset of LSGST estimates for target gates and gateStringToEstimate.

    Starting with a Direct-LGST estimate for gateStringToEstimate, runs LSGST
    using the same strings that LGST would have used to estimate gateStringToEstimate
    and each of the target gates.  That is, LSGST is run with strings of the form: 

    1. rhoStr
    2. EStr
    3. rhoStr + EStr
    4. rhoStr + singleGate + EStr
    5. rhoStr + gateStringToEstimate + EStr

    and the resulting Gateset estimate is returned.

    Parameters
    ----------
    gateStringToEstimate : GateString or tuple
        The single gate string to estimate using LSGST

    gateStringLabel : string
        The label for the estimate of gateStringToEstimate.
        i.e. gate_matrix = returned_gateset[gate_label]

    dataset : DataSet
        The data to use for LGST
        
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...)

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Defaults to no truncation.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.Bulk_fillProbs)

    verbosity : int, optional
        Verbosity value to send to doLGST(...) and doLSGST(...) calls.

    Returns
    -------
    Gateset
        A gateset containing LSGST estimates of gateStringToEstimate
        and the gates of targetGateset.
    """
    direct_lgst = GatesetWithLGSTGatestringEstimates( [gateStringToEstimate], dataset, specs, targetGateset,
                                                      True, None, None, [gateStringLabel], svdTruncateTo, verbosity )

    rhoStrs, EStrs = _Core.getRhoAndEStrs(specs)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    gatestrings = rhoStrs + EStrs + [ rhoStr + EStr for rhoStr in rhoStrs for EStr in EStrs ]
    for gateLabel in direct_lgst:
        gatestrings.extend( [ rhoStr + _gatestring.GateString( (gateLabel,) ) + EStr for rhoStr in rhoStrs for EStr in EStrs ] )

    errvec, direct_lsgst = _Core.doLSGST(dataset, direct_lgst, gatestrings, 
                                         minProbClipForWeighting=minProbClipForWeighting,
                                         probClipInterval=probClipInterval, verbosity=verbosity,
                                         gateLabelAliases={gateStringLabel: gateStringToEstimate} )
                                         #opt_gates=[gateStringLabel])
    return direct_lsgst
    
    
def DirectLSGSTGatesets(gateStrings, dataset, specs, targetGateset, svdTruncateTo=0,
                        minProbClipForWeighting=1e-4, probClipInterval=None, verbosity=0):
    """
    Constructs a dictionary with keys == gate strings and values == Direct-LSGST GateSets.

    Parameters
    ----------
    gateStrings : list of GateString or tuple objects
        The gate strings to estimate using LSGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.
        
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...)

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Defaults to no truncation.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.Bulk_fillProbs)

    verbosity : int, optional
        Verbosity value to send to doLGST(...) and doLSGST(...) calls.

    Returns
    -------
    dict
        A dictionary that relates each gate string of gateStrings to a
        GateSet containing the LSGST estimate of that gate string stored under 
        the gate label "sigmaLbl", along with LSGST estimates of the gates in
        targetGateset.
    """    
    directLSGSTgatesets = {}
    if verbosity > 0: print "--- Direct LSGST precomputation ---"
    for i,sigma in enumerate(gateStrings):
        if verbosity > 0: print "--- Computing gateset for string %d of %d ---" % (i,len(gateStrings))
        directLSGSTgatesets[sigma] = DirectLSGSTGateset( sigma, "sigmaLbl", dataset, specs, targetGateset,
                                                        svdTruncateTo, minProbClipForWeighting,
                                                        probClipInterval, verbosity)
    return directLSGSTgatesets


def DirectMLEGSTGateset( gateStringToEstimate, gateStringLabel, dataset, specs, targetGateset, svdTruncateTo=0,
                        minProbClip=1e-6, probClipInterval=None, verbosity=0 ):
    """
    Constructs a gateset of MLEGST estimates for target gates and gateStringToEstimate.

    Starting with a Direct-LGST estimate for gateStringToEstimate, runs MLEGST
    using the same strings that LGST would have used to estimate gateStringToEstimate
    and each of the target gates.  That is, MLEGST is run with strings of the form: 

    1. rhoStr
    2. EStr
    3. rhoStr + EStr
    4. rhoStr + singleGate + EStr
    5. rhoStr + gateStringToEstimate + EStr

    and the resulting Gateset estimate is returned.

    Parameters
    ----------
    gateStringToEstimate : GateString or tuple
        The single gate string to estimate using LSGST

    gateStringLabel : string
        The label for the estimate of gateStringToEstimate.
        i.e. gate_matrix = returned_gateset[gate_label]

    dataset : DataSet
        The data to use for LGST
        
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...)

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Defaults to no truncation.

    minProbClip : float, optional
        defines the minimum probability "patch point" used
        within the logL function.

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.Bulk_fillProbs)

    verbosity : int, optional
        Verbosity value to send to doLGST(...) and doMLEGST(...) calls.

    Returns
    -------
    Gateset
        A gateset containing MLEGST estimates of gateStringToEstimate
        and the gates of targetGateset.
    """
    direct_lgst = GatesetWithLGSTGatestringEstimates( [gateStringToEstimate], dataset, specs, targetGateset,
                                                      True, None, None, [gateStringLabel], svdTruncateTo, verbosity )

    rhoStrs, EStrs = _Core.getRhoAndEStrs(specs)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    gatestrings = rhoStrs + EStrs + [ rhoStr + EStr for rhoStr in rhoStrs for EStr in EStrs ]
    for gateLabel in direct_lgst:
        gatestrings.extend( [ rhoStr + _gatestring.GateString( (gateLabel,) ) + EStr for rhoStr in rhoStrs for EStr in EStrs ] )

    maxLogL, direct_mlegst = _Core.doMLEGST(dataset, direct_lgst, gatestrings, 
                                            minProbClip=minProbClip,
                                            probClipInterval=probClipInterval, verbosity=verbosity,
                                            gateLabelAliases={gateStringLabel: gateStringToEstimate} )
    return direct_mlegst


def DirectMLEGSTGatesets(gateStrings, dataset, specs, targetGateset, svdTruncateTo=0,
                        minProbClip=1e-6, probClipInterval=None, verbosity=0):
    """
    Constructs a dictionary with keys == gate strings and values == Direct-MLEGST GateSets.

    Parameters
    ----------
    gateStrings : list of GateString or tuple objects
        The gate strings to estimate using MLEGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.
        
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...)

    targetGateset : GateSet
        The target gate set used by LGST to extract gate labels and an initial gauge

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the gate matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Defaults to no truncation.

    minProbClip : float, optional
        defines the minimum probability "patch point" used
        within the logL function.

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.Bulk_fillProbs)

    verbosity : int, optional
        Verbosity value to send to doLGST(...) and doMLEGST(...) calls.

    Returns
    -------
    dict
        A dictionary that relates each gate string of gateStrings to a
        GateSet containing the MLEGST estimate of that gate string stored under 
        the gate label "sigmaLbl", along with MLEGST estimates of the gates in
        targetGateset.
    """    
    directMLEGSTgatesets = {}
    if verbosity > 0: print "--- Direct MLEGST precomputation ---"
    for i,sigma in enumerate(gateStrings):
        if verbosity > 0: print "--- Computing gateset for string %d of %d ---" % (i,len(gateStrings))
        directMLEGSTgatesets[sigma] = DirectMLEGSTGateset( sigma, "sigmaLbl", dataset, specs, targetGateset,
                                                        svdTruncateTo, minProbClip, probClipInterval, verbosity)
    return directMLEGSTgatesets


def FocusedLSGSTGateset( gateStringToEstimate, gateStringLabel, dataset, specs, startGateset,
                         minProbClipForWeighting=1e-4, probClipInterval=None, verbosity=0 ):
    """
    Constructs a gateset containing a single LSGST estimate of gateStringToEstimate.

    Starting with startGateset, run LSGST with the same gate strings that LGST 
    would use to estimate gateStringToEstimate.  That is, LSGST is run with
    strings of the form:  rhoStr + gateStringToEstimate + EStr
    and return the resulting Gateset.

    Parameters
    ----------
    gateStringToEstimate : GateString or tuple
        The single gate string to estimate using LSGST

    gateStringLabel : string
        The label for the estimate of gateStringToEstimate.
        i.e. gate_matrix = returned_gateset[gate_label]

    dataset : DataSet
        The data to use for LGST
        
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...)

    startGateset : GateSet
        The gate set to seed LSGST with. Often times obtained via LGST.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.Bulk_fillProbs)

    verbosity : int, optional
        Verbosity value to send doLSGST(...) call.

    Returns
    -------
    Gateset
        A gateset containing LSGST estimate of gateStringToEstimate.
    """
    rhoStrs, EStrs = _Core.getRhoAndEStrs(specs) # LEXICOGRAPHICAL VS MATRIX ORDER
    gatestrings = [ rhoStr + gateStringToEstimate + EStr for rhoStr in rhoStrs for EStr in EStrs ]

    errvec, focused_lsgst = _Core.doLSGST(dataset, startGateset, gatestrings, 
                                          minProbClipForWeighting=minProbClipForWeighting,
                                          probClipInterval=probClipInterval, verbosity=verbosity)
    focused_lsgst[gateStringLabel] = focused_lsgst.product(gateStringToEstimate) #add desired string as a separate labelled gate
    return focused_lsgst


def FocusedLSGSTGatesets(gateStrings, dataset, specs, startGateset,
                         minProbClipForWeighting=1e-4, probClipInterval=None, verbosity=0):
    """
    Constructs a dictionary with keys == gate strings and values == Focused-LSGST GateSets.

    Parameters
    ----------
    gateStrings : list of GateString or tuple objects
        The gate strings to estimate using LSGST.  The elements of this list
        are the keys of the returned dictionary.

    dataset : DataSet
        The data to use for all LGST and LSGST estimates.
        
    specs : 2-tuple
        A (rhoSpecs,ESpecs) tuple usually generated by calling getRhoAndESpecs(...)

    startGateset : GateSet
        The gate set to seed LSGST with. Often times obtained via LGST.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    probClipInterval : 2-tuple, optional
        (min,max) to clip probabilities to within GateSet probability
        computation routines (see GateSet.Bulk_fillProbs)

    verbosity : int, optional
        Verbosity value to send to doLSGST(...) call.

    Returns
    -------
    dict
        A dictionary that relates each gate string of gateStrings to a
        GateSet containing the LSGST estimate of that gate string stored under 
        the gate label "sigmaLbl".
    """    
    focusedLSGSTgatesets = {}
    if verbosity > 0: print "--- Focused LSGST precomputation ---"
    for i,sigma in enumerate(gateStrings):
        if verbosity > 0: print "--- Computing gateset for string %d of %d ---" % (i,len(gateStrings))
        focusedLSGSTgatesets[sigma] = FocusedLSGSTGateset( sigma, "sigmaLbl", dataset, specs, startGateset,
                                                           minProbClipForWeighting, probClipInterval, verbosity)
    return focusedLSGSTgatesets



def DirectChiSqMatrix( sigma, dataset, directGateset, strs,
                       minProbClipForWeighting=1e-4, rhoEPairs=None):
    """
    Computes the Direct-X chi^2 matrix for a base gatestring sigma.

    Similar to ChiSqMatrix, except the probabilities used to compute
    chi^2 values come from using the "composite gate" of directGatesets[sigma],
    a GateSet assumed to contain some estimate of sigma stored under the
    gate label "sigmaLbl".

    Parameters
    ----------
    sigma : GateString or tuple of gate labels
        The gate sequence that is sandwiched between each rhoStr and EStr

    dataset : DataSet
        The data used to specify frequencies and counts

    directGateset : GateSet
        GateSet which contains an estimate of sigma stored
        under the gate label "sigmaLbl".

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight (see ChiSqFunc).

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the matrix.  Other elements are set to NaN.

    Returns
    -------
    numpy array of shape ( len(EStrs), len(rhoStrs) )
        Direct-X chi^2 values corresponding to gate sequences where 
        gateString is sandwiched between the each (EStr,rhoStr) pair.
    """
    chiSqMx = _np.zeros( (len(strs[1]),len(strs[0])), 'd')
    if sigma is None: return _np.nan*chiSqMx
    cntMx  = TotalCountMatrix(  sigma, dataset, strs, rhoEPairs)
    gs_direct = directGateset
    for sl in gs_direct.get_SPAM_labels():
        probMx = ProbabilityMatrix( _gatestring.GateString( ("sigmaLbl",) ), gs_direct, sl, strs, rhoEPairs)
        freqMx = FrequencyMatrix( sigma, dataset, sl, strs, rhoEPairs)
        chiSqMx += ChiSqFunc( cntMx, probMx, freqMx, minProbClipForWeighting)
    return chiSqMx

def DirectChiSqBoxPlot( xvals, yvals, xy_gatestring_dict, dataset, directGatesets, strs, xlabel="", ylabel="",
                        m=None, M=None, scale=1.0, prec='compact', title="Direct Chi^2", sumUp=False,
                        interactive=False, boxLabels=True, histogram=False, histBins=50, minProbClipForWeighting=1e-4,
                        saveTo=None, ticSize=20, invert=False, rhoEPairs=None):
    """
    Create a color box plot of Direct-X chi^2 values.

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    directGatesets : dict
        Dictionary with keys == gate strings and values == GateSets.  
        directGatesets[sigma] must be a GateSet which contains an estimate
        of sigma stored under the gate label "sigmaLbl".

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the plot.

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """
    rhoStrs, EStrs = strs
    def mxFn(gateStr):
        return DirectChiSqMatrix( gateStr, dataset, directGatesets.get(gateStr,None), strs, minProbClipForWeighting, rhoEPairs)
    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel, ylabel, m,M,
                            scale,prec,title,sumUp,interactive,boxLabels,histogram,histBins,saveTo,ticSize,
                            invert, rhoStrs, EStrs, r"$\rho_i$", r"$E_i$",  )


def ZoomedDirectChiSqBoxPlot(xvals, yvals, xy_gatestring_dict, dataset, directGatesets, strs, xlabel="", ylabel="",
                                  m=None, M=None, scale=1.0, prec='compact', title="Direct Chi^2",
                                  minProbClipForWeighting=1e-4, saveTo=None,ticSize=14):
    """
    Create an interactive zoomed color box plot of Direct-X chi^2 values (within an iPython notebook)

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    directGatesets : dict
        Dictionary with keys == gate strings and values == GateSets.  
        directGatesets[sigma] must be a GateSet which contains an estimate
        of sigma stored under the gate label "sigmaLbl".

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    Returns
    -------
    None
    """
    def mxFn(gateStr):
        return DirectChiSqMatrix( gateStr, dataset, directGatesets.get(gateStr,None), strs, minProbClipForWeighting)
    generateZoomedBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, strs, xlabel, ylabel, m,M,scale,prec,title,saveTo,ticSize )


def DirectLogLMatrix( sigma, dataset, directGateset, strs,
                       minProbClip=1e-6, rhoEPairs=None):
    """
    Computes the Direct-X log-likelihood matrix, containing the values
     of 2*( log(L)_upperbound - log(L) ) for a base gatestring sigma.

    Similar to LogLMatrix, except the probabilities used to compute
    LogL values come from using the "composite gate" of directGatesets[sigma],
    a GateSet assumed to contain some estimate of sigma stored under the
    gate label "sigmaLbl".

    Parameters
    ----------
    sigma : GateString or tuple of gate labels
        The gate sequence that is sandwiched between each rhoStr and EStr

    dataset : DataSet
        The data used to specify frequencies and counts

    directGateset : GateSet
        GateSet which contains an estimate of sigma stored
        under the gate label "sigmaLbl".

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    minProbClip : float, optional
        defines the minimum probability clipping.

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the matrix.  Other elements are set to NaN.

    Returns
    -------
    numpy array of shape ( len(EStrs), len(rhoStrs) )
        Direct-X chi^2 values corresponding to gate sequences where 
        gateString is sandwiched between the each (EStr,rhoStr) pair.
    """
    logLMx = _np.zeros( (len(strs[1]),len(strs[0])), 'd')
    if sigma is None: return _np.nan*logLMx
    cntMx  = TotalCountMatrix(  sigma, dataset, strs, rhoEPairs)
    gs_direct = directGateset
    for sl in gs_direct.get_SPAM_labels():
        probMx = ProbabilityMatrix( _gatestring.GateString( ("sigmaLbl",) ), gs_direct, sl, strs, rhoEPairs)
        freqMx = FrequencyMatrix( sigma, dataset, sl, strs, rhoEPairs)
        logLMx += TwoDeltaLogLFunc( cntMx, probMx, freqMx, minProbClip)
    return logLMx


def DirectLogLBoxPlot( xvals, yvals, xy_gatestring_dict, dataset, directGatesets, strs, xlabel="", ylabel="",
                        m=None, M=None, scale=1.0, prec='compact', title="Direct Log(L)", sumUp=False,
                        interactive=False, boxLabels=True, histogram=False, histBins=50, minProbClipForWeighting=1e-4,
                        saveTo=None, ticSize=20, invert=False, rhoEPairs=None):
    """
    Create a color box plot of Direct-X log-likelihood values.

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    directGatesets : dict
        Dictionary with keys == gate strings and values == GateSets.  
        directGatesets[sigma] must be a GateSet which contains an estimate
        of sigma stored under the gate label "sigmaLbl".

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the logL function.

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    rhoEPairs : list, optional
        A list of (iRhoStr,iEStr) tuples specifying a subset of all the rhoStr,EStr
        pairs to include in the plot.

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """
    rhoStrs, EStrs = strs
    def mxFn(gateStr):
        return DirectLogLMatrix( gateStr, dataset, directGatesets.get(gateStr,None), strs, minProbClipForWeighting, rhoEPairs)
    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel, ylabel, m,M,
                            scale,prec,title,sumUp,interactive,boxLabels,histogram,histBins,saveTo,ticSize,
                            invert, rhoStrs, EStrs, r"$\rho_i$", r"$E_i$",  )


def Direct2xCompBoxPlot( xvals, yvals, xy_gatestring_dict, dataset, directGatesets, strs, xlabel="", ylabel="",
                             m=None, M=None, scale=1.0, prec='compact', title="Direct 2x Chi^2 Comparison", sumUp=False,
                             interactive=False, boxLabels=True, histogram=False, histBins=50, minProbClipForWeighting=1e-4,
                             saveTo=None, ticSize=20, invert=False):
    """
    Create a box plot indicating how well the Direct-X estimates of string s 
    predict the data for 2s (the string repeated)

    Creates a color box plot whose boxes (or box, if sumUp == True) at
    position (x,y) display the chi^2 for the (x,y) base gate string 
    **repeated twice** (if this data is available), where the probabilities
    used in the chi^2 calculation are obtained using the Direct-X gateset
    for the un-repeated (x,y) base gate string.  That is, the box(es) at
    coordinates x,y show how well the Direct-X estimates of xy_gatestring_dict[(x,y)]
    reproduce the observed frequencies of 2 * xy_gatestring_dict[(x,y)] (the
    gate string repeated twice).

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    directGatesets : dict
        Dictionary with keys == gate strings and values == GateSets.  
        directGatesets[sigma] must be a GateSet which contains an estimate
        of sigma stored under the gate label "sigmaLbl".

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.   Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """
    rhoStrs, EStrs = strs
    def mxFn(gateStr):
        chiSqMx = _np.zeros( (len(strs[1]),len(strs[0])), 'd')
        if gateStr is None: return _np.nan*chiSqMx
        gs_direct = directGatesets[ gateStr ] #contains "sigmaLbl" gate <=> gateStr
        try:
        #if gateStr*2 in directGatesets: 
            cntMx  = TotalCountMatrix(  gateStr*2, dataset, strs)
            for sl in gs_direct.get_SPAM_labels():
                probMx = ProbabilityMatrix( _gatestring.GateString( ("sigmaLbl","sigmaLbl") ), gs_direct, sl, strs)
                freqMx = FrequencyMatrix( gateStr*2, dataset, sl, strs)
                chiSqMx += ChiSqFunc( cntMx, probMx, freqMx, minProbClipForWeighting)
        #else:
        #    print "Warning: didn't find len-%d str: " % len(gateStr*2), (gateStr*2)[0:20]
        except:
            return _np.nan*chiSqMx #if something fails, just punt
        return chiSqMx

    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel, ylabel, m,M,
                            scale,prec,title,sumUp,interactive,boxLabels,histogram,histBins,
                            saveTo,ticSize,invert, rhoStrs, EStrs, r"$\rho_i$", r"$E_i$" )


def DirectDeviationBoxPlot( xvals, yvals, xy_gatestring_dict, dataset, gateset, directGatesets,
                                 xlabel="", ylabel="", m=None, M=None, scale=1.0, prec='compact', title="Direct Deviation",
                                 interactive=False, boxLabels=True, histogram=False, histBins=50, saveTo=None, ticSize=20):
    """
    Create a box plot showing the difference in max-fidelity-with-unitary
    between gateset's estimate for each base gate string and the Direct-X estimate.

    Creates a color box plot whose box at position (x,y) shows the 
    the difference between:

    1. the upper bound of the fidelity between the map corresponding to 
       this base gate string using the Direct-X estimate of this map
       (i.e. by using only data relevant to this particular string) and
       a unitary map.

    2. the upper bound of the fidelity between the map corresponding to 
       this base gate string using gateset (i.e. by multiplying together
       single gate estimates) and a unitary map.

    The plotted quantity indicates how much more "unitary", i.e. how 
    much less "depolarized", the map corresponding to each base gate
    sequence is when considering only the data immediately relevant
    to predicting that map.  If 2. is larger than 1., a zero is displayed
    so that all results are non-negative.  Large values indicate that
    the data used for fitting other gate sequences has made the estimate
    for the subject gate sequence more depolarized (~worse) than the
    data for the sequence alone would suggest.

    Parameters
    ----------
    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    directGatesets : dict
        Dictionary with keys == gate strings and values == GateSets.  
        directGatesets[sigma] must be a GateSet which contains an estimate
        of sigma stored under the gate label "sigmaLbl".

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """
    def mxFn(gateStr):
        if gateStr is None: return _np.nan * _np.zeros( (1,1), 'd')
        gate = gateset.product( gateStr )
        gate_direct = directGatesets[ gateStr ][ "sigmaLbl" ]
        #evals = _np.linalg.eigvals(gate)
        #evals_direct = _np.linalg.eigvals(gate_direct)
        ubF, ubGateMx = _GateOps.getFidelityUpperBound(gate)
        ubF_direct, ubGateMx = _GateOps.getFidelityUpperBound(gate_direct)
        return _np.array( [[ max(ubF_direct - ubF,0.0) ]], 'd' ) 

    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel, ylabel, m,M,
                            scale,prec,title,True,interactive,boxLabels,histogram,histBins,saveTo,ticSize)



def WhackAChiSqMoleBoxPlot( gatestringToWhack, allGatestringsUsedInChi2Opt, 
                           xvals, yvals, xy_gatestring_dict, dataset, gateset, strs, xlabel="", ylabel="",
                           m=None, M=None, scale=1.0, prec='compact', title="Whack a Chi^2 Mole", sumUp=False,
                           interactive=False, boxLabels=True, histogram=False, histBins=50, minProbClipForWeighting=1e-4,
                           saveTo=None, ticSize=20, whackWith=10.0, invert=False, rhoEPairs=None):
    """
    Create a box plot indicating how the chi^2 would change if the chi^2 of one
      base gate string blocks were forced to be smaller ("whacked").

    Creates a color box plot which displays the change in chi^2 caused by
      changing the gate set parameters such that the chi^2 of gatestringToWhack's
      (x,y) block decreases by whackWith.  This changes the gate set along
      the direction of parameter space given by the gradient of chi^2 restricted
      to only those gatestrings in gatestringToWhack's block, and the the
      displayed difference in chi^2 values are based on the linear interpolation
      of the full gradient of chi^2 after this change.

    Parameters
    ----------
    gatestringToWhack : GateString or tuple
        The **base** gate sequence for which chi^2 will be decreased.
        
    allGatestringsUsedInChi2Opt : list of GateStrings or tuples
        List of all the gate strings used to form the total chi^2 that is being decreased.

    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used for computing probabilities

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    minProbClipForWeighting : float, optional
        defines the clipping interval for the statistical weight used
        within the chi^2 function (see ChiSqFunc).

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    whackWith : float, optional
        the total amount to decrease chi^2 by.  This number just sets the
        overall scale of the numbers displayed, since the extrapolation is
        linear.

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """



    #We want the derivative of chi^2 = sum_i N_i*(p_i-f_i)^2 / p_i  (i over gatestrings & spam labels)
    # and the ability to separate the chi^2 of just "gatestringToWhack" (sandwiched with strs, optionally)
    # This latter derivative (w.r.t gateset params) gives the direction in gateset space to move to reduce 
    #   the "whacked" string(s) chi2.  Applying this direction to the full chi^2 (or to other base strings
    #   sandwiched with strs) will give the relative change in the chi^2 for these strings if the whacked 
    #   string(s) was in fact whacked.
    # D(chi^2) = sum_i N_i * [ 2(p_i-f_i)*dp_i / p_i - (p_i-f_i)^2 / p_i^2 * dp_i ]
    #          = sum_i N_i * (p_i-f_i) / p_i * [2 - (p_i-f_i)/p_i   ] * dp_i
    rhoStrs, EStrs = strs
    spamLabels = gateset.get_SPAM_labels() #this list fixes the ordering of the spam labels
    spam_lbl_rows = { sl:i for (i,sl) in enumerate(spamLabels) }
    vec_gs_len = gateset.getNumParams(G0=True, SP0=True, SPAM=True, gates=True)

    N      = _np.empty( len(allGatestringsUsedInChi2Opt) )
    f      = _np.empty( (len(spamLabels),len(allGatestringsUsedInChi2Opt)) )
    probs  = _np.empty( (len(spamLabels),len(allGatestringsUsedInChi2Opt)) )
    dprobs = _np.empty( (len(spamLabels),len(allGatestringsUsedInChi2Opt),vec_gs_len) )

    for (i,gateStr) in enumerate(allGatestringsUsedInChi2Opt):
        N[i] = float(dataset[gateStr].total())
        for k,sl in enumerate(spamLabels):
            f[k,i] = dataset[gateStr].fraction(sl)

    evTree = gateset.Bulk_evalTree(allGatestringsUsedInChi2Opt)
    gateset.Bulk_filldProbs(dprobs, spam_lbl_rows, evTree, 
                             G0=True, SP0=True, SPAM=True, gates=True, prMxToFill=probs)

    t = ((probs - f)/probs)[:,:,None]
    Dchi2 = N[None,:,None] * t * (2 - t) * dprobs  # (1,M,1) * (K,M,1) * (K,M,N)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    if rhoEPairs is None:
        gatestringsToWhack = [ (rhoStr + gatestringToWhack + EStr) for rhoStr in rhoStrs for EStr in EStrs ]
    else:
        gatestringsToWhack = [ (rhoStrs[i] + gatestringToWhack + EStrs[j]) for (i,j) in rhoEPairs ]

    whacked_indices = [ allGatestringsUsedInChi2Opt.index(s) for s in gatestringsToWhack ]
    whacked_Dchi2 = _np.take(Dchi2,whacked_indices,axis=1) # (K,m,N) where m == len(whacked_indices)

    grad = -1.0 * _np.sum(whacked_Dchi2,axis=(0,1)) # (N) after summing over gate strings and spam labels
    dx = whackWith * grad / _np.dot(grad,grad) # direction in gateset space of direction to move to *decrease* the chi2
                                               #  of the desired base string by whackWith
    delta = _np.sum( _np.dot(Dchi2,dx), axis=0 ) # sum{(K,M), axis=1} ==> (M); the change in chi2 for each gateString
                                                 #   as a result of a unit decrease in the chi2 of the base string

    def mxFn(gateStr):
        # LEXICOGRAPHICAL VS MATRIX ORDER
        if gateStr is None: return _np.nan * _np.zeros( (len(EStrs),len(rhoStrs)), 'd')

        if rhoEPairs is None:
            return _np.array( [ [ delta[ allGatestringsUsedInChi2Opt.index(rhoStr + gateStr + EStr) ] for rhoStr in rhoStrs ] for EStr in EStrs ] )
        else:
            ret = _np.nan * _np.ones( (len(EStrs),len(rhoStrs)), 'd')
            for i,j in rhoEPairs:
                ret[j,i] = delta[ allGatestringsUsedInChi2Opt.index(rhoStrs[i] + gateStr + EStrs[j]) ]
            return ret

    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel, ylabel, m,M,
                            scale,prec,title,sumUp,interactive,boxLabels,histogram,histBins,saveTo,ticSize,
                            invert, rhoStrs, EStrs, r"$\rho_i$", r"$E_i$" )



def WhackALogLMoleBoxPlot( gatestringToWhack, allGatestringsUsedInLogLOpt, 
                           xvals, yvals, xy_gatestring_dict, dataset, gateset, strs, xlabel="", ylabel="",
                           m=None, M=None, scale=1.0, prec='compact', title="Whack a log(L) Mole", sumUp=False,
                           interactive=False, boxLabels=True, histogram=False, histBins=50, minProbClipForWeighting=1e-4,
                           saveTo=None, ticSize=20, whackWith=10.0, invert=False, rhoEPairs=None):
    """
    Create a box plot indicating how the log-likelihood would change if the log(L)
      of one base gate string blocks were forced to be smaller ("whacked").

    Creates a color box plot which displays the change in log(L) caused by
      changing the gate set parameters such that the log(L) of gatestringToWhack's
      (x,y) block decreases by whackWith.  This changes the gate set along
      the direction of parameter space given by the gradient of log(L) restricted
      to only those gatestrings in gatestringToWhack's block, and the the
      displayed difference in log(L) values are based on the linear interpolation
      of the full gradient of log(L) after this change.

    Parameters
    ----------
    gatestringToWhack : GateString or tuple
        The **base** gate sequence for which log(L) will be decreased.
        
    allGatestringsUsedInLogLOpt : list of GateStrings or tuples
        List of all the gate strings used to form the total log(L) that is being decreased.

    xvals, yvals : list
        List of x and y values. Elements can be any hashable quantity, and will be converted
        into x and y tic labels.  Tuples of strings are converted specially for nice latex
        rendering of gate strings.

    xy_gatestring_dict : dict
        Dictionary with keys == (x_value,y_value) tuples and values == gate strings, where
        a gate string can either be a GateString object or a tuple of gate labels.  Provides
        the mapping between x,y pairs and gate strings.  None values are allowed, and 
        indicate that there is not data for that x,y pair and nothing should be plotted.

    dataset : DataSet
        The data used to specify frequencies and counts

    gateset : GateSet
        The gate set used for computing probabilities

    strs : 2-tuple
        A (rhoStrs,EStrs) tuple usually generated by calling getRhoAndEStrs(...)

    xlabel, ylabel : str, optional
        X and Y axis labels

    m, M : float, optional
        Min and max values of the color scale.

    scale : float, optional
        Scaling factor to adjust the size of the final figure.

    prec : int, optional
        Precision for box labels.  Allowed values are:
          'compact' = round to nearest whole number using at most 3 characters
          'compacthp' = show as much precision as possible using at most 3 characters
          int >= 0 = fixed precision given by int
          int <  0 = number of significant figures given by -int

    title : string, optional
        Plot title (latex can be used)

    sumUp : bool, optional
        False displays each matrix element as it's own color box
        True sums the elements of each (x,y) matrix and displays
        a single color box for the sum.

    interactive : bool, optional
        If true and wihin an iPython notebook, widgets are used
        to create an interactive plot whereby the user can adjust
        the min and max of the colorscale.

    boxLabels : bool, optional
        Whether box labels are displayed.  It takes much longer to 
        generate the figure when this is set to True.

    histogram : bool, optional
        Whether a histogram of the matrix element values or summed matrix
        values (depending on sumUp) should also be computed and displayed.

    histBins : int, optional
        The number of bins to use in the histogram.

    minProbClipForWeighting : float, optional
        defines the minimum probability clipping for the log(L) function.

    saveTo : str, optional
        save figure to this filename (usually ending in .pdf)

    ticSize : int, optional
        size of tic marks

    whackWith : float, optional
        the total amount to decrease chi^2 by.  This number just sets the
        overall scale of the numbers displayed, since the extrapolation is
        linear.

    invert : bool, optional
        If True, invert the nesting order of the nested color box plot (applicable
        only when sumUp == False).  Use inner_x_labels and inner_y_labels to label
        the x and y axes.

    Returns
    -------
    gstFig : GSTFigure
        The encapsulated matplotlib figure that was generated.  Extra figure
        info is a dict with keys:

        nUsedXs : int
            The number of used X-values, proportional to the overall final figure width
        nUsedYs : int
            The number of used Y-values, proportional to the overall final figure height
    """

    #We want the derivative of 2*Delta_LogL = 2 * sum_i N_i*(f_i*log(f_i/p_i) + (p_i-f_i))  (i over gatestrings & spam labels)
    # and the ability to separate the 2*Delta_LogL of just "gatestringToWhack" (sandwiched with strs, optionally)
    # This latter derivative (w.r.t gateset params) gives the direction in gateset space to move to reduce 
    #   the "whacked" string(s) 2*Delta_LogL.  Applying this direction to the full 2*Delta_LogL (or to other base strings
    #   sandwiched with strs) will give the relative change in the 2*Delta_LogL for these strings if the whacked 
    #   string(s) was in fact whacked.
    # D(2*Delta_LogL) = sum_i 2* N_i * [ -f_i/p_i + 1.0 ] * dp_i

    rhoStrs, EStrs = strs
    spamLabels = gateset.get_SPAM_labels() #this list fixes the ordering of the spam labels
    spam_lbl_rows = { sl:i for (i,sl) in enumerate(spamLabels) }
    vec_gs_len = gateset.getNumParams(G0=True, SP0=True, SPAM=True, gates=True) 
      #Note: assumes *all* gateset params vary, which may not be what we always want (e.g. for TP-constrained analyses)

    N      = _np.empty( len(allGatestringsUsedInLogLOpt) )
    f      = _np.empty( (len(spamLabels),len(allGatestringsUsedInLogLOpt)) )
    probs  = _np.empty( (len(spamLabels),len(allGatestringsUsedInLogLOpt)) )
    dprobs = _np.empty( (len(spamLabels),len(allGatestringsUsedInLogLOpt),vec_gs_len) )

    for (i,gateStr) in enumerate(allGatestringsUsedInLogLOpt):
        N[i] = float(dataset[gateStr].total())
        for k,sl in enumerate(spamLabels):
            f[k,i] = dataset[gateStr].fraction(sl)

    evTree = gateset.Bulk_evalTree(allGatestringsUsedInLogLOpt)
    gateset.Bulk_filldProbs(dprobs, spam_lbl_rows, evTree, 
                             G0=True, SP0=True, SPAM=True, gates=True, prMxToFill=probs) # spamlabel, gatestring, gsParam

    pos_probs = _np.maximum(probs, minProbClipForWeighting) #make sure all probs are positive? TODO: make this fn handle minProbClip like doMLEGST does...
    DlogL = 2 * (N[None,:] * (1.0 - f/pos_probs))[:,:,None] * dprobs # (K,M,1) * (K,M,N)

    # LEXICOGRAPHICAL VS MATRIX ORDER
    if rhoEPairs is None:
        gatestringsToWhack = [ (rhoStr + gatestringToWhack + EStr) for rhoStr in rhoStrs for EStr in EStrs ]
    else:
        gatestringsToWhack = [ (rhoStrs[i] + gatestringToWhack + EStrs[j]) for (i,j) in rhoEPairs ]

    whacked_indices = [ allGatestringsUsedInLogLOpt.index(s) for s in gatestringsToWhack ]
    whacked_DlogL = _np.take(DlogL,whacked_indices,axis=1) # (K,m,N) where m == len(whacked_indices)

    grad = -1.0 * _np.sum(whacked_DlogL,axis=(0,1)) # (N) after summing over gate strings and spam labels
    dx = whackWith * grad / _np.dot(grad,grad) # direction in gateset space of direction to move to *decrease* the 2*Delta_LogL
                                               #  of the desired base string by whackWith
    delta = _np.sum( _np.dot(DlogL,dx), axis=0 ) # sum{(K,M), axis=1} ==> (M); the change in 2*Delta_LogL for each gateString
                                                 #   as a result of a unit decrease in the 2*Delta_LogL of the base string

    def mxFn(gateStr):
        # LEXICOGRAPHICAL VS MATRIX ORDER
        if gateStr is None: return _np.nan * _np.zeros( (len(EStrs),len(rhoStrs)), 'd')

        if rhoEPairs is None:
            return _np.array( [ [ delta[ allGatestringsUsedInLogLOpt.index(rhoStr + gateStr + EStr) ] for rhoStr in rhoStrs ] for EStr in EStrs ] )
        else:
            ret = _np.nan * _np.ones( (len(EStrs),len(rhoStrs)), 'd')
            for i,j in rhoEPairs:
                ret[j,i] = delta[ allGatestringsUsedInLogLOpt.index(rhoStrs[i] + gateStr + EStrs[j]) ]
            return ret

    return generateBoxPlot( xvals, yvals, xy_gatestring_dict, mxFn, xlabel, ylabel, m,M,
                            scale,prec,title,sumUp,interactive,boxLabels,histogram,histBins,saveTo,ticSize,
                            invert, rhoStrs, EStrs, r"$\rho_i$", r"$E_i$" )



def _makeHistFilename(mainFilename):
    #Insert "_hist" before extension, e.g. /one/two.txt ==> /one/two_hist.txt
    if len(mainFilename) > 0:
        return "_hist".join(_os.path.splitext(mainFilename))    
    else: return "" #keep empty string empty, as this signals not actually saving any files

""" Functions for analyzing RB data"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from scipy.optimize import curve_fit as _curve_fit
from . import results as _results
from ...tools import compattools as _compat
  
def p_to_r(p, d, rtype='EI'):
    """
    Converts an RB decay constant (p) to the RB error rate (r), where
    p is (normally) obtained from fitting data to A + Bp^m. There are
    two 'types' of RB error rate corresponding to different rescalings
    of 1 - p. These are the entanglement infidelity (EI) type r and
    the average gate infidelity (AGI) type r. The EI-type r is given by:
    
    r =  (d^2 - 1)(1 - p)/d^2,

    where d is the dimension of the system (i.e., 2^n for n qubits).
    The AGI-type r is given by

    r =  (d - 1)(1 - p)/d.

    For RB on gates whereby every gate is followed by an n-qubit
    uniform depolarizing channel (the most idealized RB scenario)
    then the EI-type (AGI-type) r corresponds to the EI (AGI) of
    the depolarizing channel to the identity channel.

    The default (EI) is the convention used in direct RB, and is perhaps 
    the most well-motivated as then r corresponds to the error probablity
    of the gates (in the idealized pauli-errors scenario). AGI is
    the convention used throughout Clifford RB theory.
    
    Parameters
    ----------
    p : float
        Fit parameter p from P_m = A + B*p**m.
    
    d : int
        Number of dimensions of the Hilbert space

    rtype : {'EI','AGI'}, optional
        The RB error rate rescaling convention.
     
    Returns
    -------
    r : float
        The RB error rate         
    """
    if rtype == 'AGI': r = (1 - p) * (d - 1) / d
    elif rtype == 'EI': r = (d**2 - 1) * (1 - p)/d**2
    else:
        raise ValueError("rtype must be `EI` (for entanglement infidelity) or `AGI` (for average gate infidelity)")
    
    return r

def r_to_p(r, d, rtype='EI'):
    """
    Inverse of the p_to_r function. 

    Parameters
    ----------
    r : float
        The RB error rate
    
    d : int
        Number of dimensions of the Hilbert space 

    rtype : {'EI','AGI'}, optional
        The RB error rate rescaling convention.   
     
    Returns
    -------
    p : float
        The RB decay constant  
    """
    if rtype == 'AGI': p = 1 - d * r / (d - 1)       
    elif rtype == 'EI': p = 1 - d**2 * r / (d**2 - 1)
    else:
        raise ValueError("rtype must be `EI` (for entanglement infidelity) or `AGI` (for average gate infidelity)")
        
    return p

def marginalize(results, keepqubits, allqubits):
    # Future -- docstring, once this function is used.
    mresults = []
    mask = _np.zeros(len(allqubits),bool)

    for q in keepqubits: mask[allqubits.index(q)] = True

    for i in range(len(results)): mresults.append(tuple(_np.array(results[i])[mask]))

    return mresults

def magesan_first_order_fit_function(m, A, B, C, p):
    """
    The 'first order' fitting function P_m = A + (B + m * C) * p^m, from
    "Scalable and Robust Randomized Benchmarking of Quantum Processes" , 
    Magesan et al. PRL 106 180504 (2011). This is a simplified verion of 
    the 'first order' in that paper (see Eq. 3), as the model therein has 
    one too many parameters for fitting. The conversion is
    
    A = B_1
    B = A_1 - C_1(q/p^(-2) - 1)
    C = C_1(q/p^(-2) - 1)

    where the LHS (RHS) quantites in this equation are those of our (Magesan 
    et al.'s) fitting function.

    Parameters
    ----------
    m : integer
        The RB length of the random RB sequence.

    A,B,C,p : float

    Returns
    -------
    float
        A + (B + m * C) * p^m.
    """
    return A+(B+C*m)*p**m

def rescaling_factor(lengths, quantity, offset=2):
    """
    Finds a rescaling value alpha that can be used to map the Clifford RB decay constant
    p to p_(rescaled) = p^(1/alpha) for finding e.g., a "CRB r per CNOT" or a "CRB r per 
    compiled Clifford depth".

    Parameters
    ----------
    lengths : list
        A list of the RB lengths, which each value in 'quantity' will be rescaled by.

    quantity : list
        A list, of the same length as `lengths`, that contains lists of values of the quantity
        that the rescaling factor is extracted from.

    offset : int, optional 
        A constant offset to add to lengths.

    Returns
        mean over i of [mean(quantity[i])/(lengths[i]+offset)]
    """
    assert(len(lengths)==len(quantity)), "Data format incorrect!"
    rescaling_factor = []
    
    for i in range(len(lengths)):   
        rescaling_factor.append(_np.mean(_np.array(quantity[i])/(lengths[i]+offset)))
    
    rescaling_factor = _np.mean(_np.array(rescaling_factor))

    return rescaling_factor

def std_practice_analysis(RBSdataset, seed=[0.8,0.95], bootstrap_samples=200,  asymptote='std', rtype='EI'):
    """
    Implements a "standard practice" analysis of RB data. Fits the average success probabilities to the exponential 
    decay A + Bp^m, using least-squares fitting, with (1) A fixed (as standard, to 1/2^n where n is the number of 
    qubits the data is for), and (2) A, B and p all allowed to varying. Confidence intervals are also estimated using
    a standard non-parameteric boostrap.

    Parameters
    ----------
    RBSdataset : RBSummaryDataset
        An RBSUmmaryDataset containing the data to analyze

    seed : list, optional   
        Seeds for the fit of B and p (A is seeded to the asymptote defined by `asympote`).

    bootstrap_samples : int, optional
        The number of samples in the bootstrap.

    asymptote : str or float, optional
        The A value for the fitting to A + Bp^m with A fixed. If a string must be 'std', in
        in which case A is fixed to 1/2^n.

    rtype : {'EI','AGI'}, optional
        The RB error rate rescaling convention. 'EI' results in RB error rates that are associated
        with the entanglement infidelity, which is the error probability with stochastic errors (and
        is equal to the diamond distance). 'AGI' results in RB error rates that are associated with
        average gate infidelity.

    Returns
    -------
    RBResults
        An object encapsulating the RB results (and data).
    """  
    lengths = RBSdataset.lengths
    ASPs = RBSdataset.ASPs
    success_counts = RBSdataset.success_counts
    total_counts = RBSdataset.total_counts
    n = RBSdataset.number_of_qubits

    if _compat.isstr(asymptote):
        assert(asymptote == 'std'), "If `asympotote` is a string it must be 'std'!"
        asymptote = 1/2**n

    FF_results, FAF_results = std_least_squares_data_fitting(lengths, ASPs, n, seed=seed, asymptote=asymptote, ftype='full+FA',
                                                             rtype=rtype)

    if bootstrap_samples > 0:
        
        A_bootstraps_FF = []
        B_bootstraps_FF = []
        p_bootstraps_FF = []
        r_bootstraps_FF = []

        A_bootstraps_FAF = []
        B_bootstraps_FAF = []
        p_bootstraps_FAF = []
        r_bootstraps_FAF = []

        bs_failcount_FF = 0
        bs_failcount_FAF = 0

        RBSdataset.add_bootstrapped_datasets(samples=bootstrap_samples)
        for i in range(bootstrap_samples):

            BS_ASPs  = RBSdataset.bootstraps[i].ASPs
            BS_FF_results, BS_FAF_results = std_least_squares_data_fitting(lengths, BS_ASPs, n, seed=seed, asymptote=asymptote, 
                                                                           ftype='full+FA', rtype=rtype)

            if BS_FF_results['success']:
                A_bootstraps_FF.append(BS_FF_results['estimates']['A'])
                B_bootstraps_FF.append(BS_FF_results['estimates']['B'])
                p_bootstraps_FF.append(BS_FF_results['estimates']['p'])
                r_bootstraps_FF.append(BS_FF_results['estimates']['r'])
            else:
                bs_failcount_FF += 1
            if BS_FAF_results['success']:
                A_bootstraps_FAF.append(BS_FAF_results['estimates']['A'])
                B_bootstraps_FAF.append(BS_FAF_results['estimates']['B'])
                p_bootstraps_FAF.append(BS_FAF_results['estimates']['p'])
                r_bootstraps_FAF.append(BS_FAF_results['estimates']['r'])
            else:
                bs_failcount_FAF += 1
    
        bootstraps_FF = {}
        bootstraps_FF['A'] = A_bootstraps_FF
        bootstraps_FF['B'] = B_bootstraps_FF
        bootstraps_FF['p'] = p_bootstraps_FF
        bootstraps_FF['r'] = r_bootstraps_FF
        bootstraps_failrate_FF = bs_failcount_FF/bootstrap_samples

        std_FF = {}
        std_FF['A'] = _np.std(_np.array(A_bootstraps_FF))
        std_FF['B'] = _np.std(_np.array(B_bootstraps_FF))
        std_FF['p'] = _np.std(_np.array(p_bootstraps_FF))
        std_FF['r'] = _np.std(_np.array(r_bootstraps_FF))

        bootstraps_FAF = {}
        bootstraps_FAF['A'] = A_bootstraps_FAF
        bootstraps_FAF['B'] = B_bootstraps_FAF
        bootstraps_FAF['p'] = p_bootstraps_FAF
        bootstraps_FAF['r'] = r_bootstraps_FAF
        bootstraps_failrate_FAF = bs_failcount_FAF/bootstrap_samples

        std_FAF = {}
        std_FAF['A'] = _np.std(_np.array(A_bootstraps_FAF))
        std_FAF['B'] = _np.std(_np.array(B_bootstraps_FAF))
        std_FAF['p'] = _np.std(_np.array(p_bootstraps_FAF))
        std_FAF['r'] = _np.std(_np.array(r_bootstraps_FAF))

    else:
        bootstraps_FF = None
        std_FF = None
        bootstraps_failrate_FF = None
        bootstraps_FAF = None
        std_FAF = None
        bootstraps_failrate_FAF = None

    fits = {}
    fits['full'] =  _results.FitResults('LS', FF_results['seed'], rtype, FF_results['success'], FF_results['estimates'], 
                                        FF_results['variable'], stds=std_FF,  bootstraps=bootstraps_FF, 
                                        bootstraps_failrate=bootstraps_failrate_FF)

    fits['A-fixed'] =  _results.FitResults('LS', FAF_results['seed'], rtype, FAF_results['success'], FAF_results['estimates'], 
                                          FAF_results['variable'], stds=std_FAF, bootstraps=bootstraps_FAF, 
                                          bootstraps_failrate=bootstraps_failrate_FAF)

    results = _results.RBResults(RBSdataset,rtype,fits)

    return results
    
def std_least_squares_data_fitting(lengths, ASPs, n, seed=None, asymptote=None, ftype='full', rtype='EI'):
    """
    Implements a "standard" least-squares fit of RB data. Fits the average success probabilities to 
    the exponential decay A + Bp^m, using least-squares fitting.

    Parameters
    ----------
    lengths : list
        The RB lengths to fit to (the 'm' values in A + Bp^m).
 
    ASPs : list
        The average survival probabilities to fit (the observed P_m values to fit 
        to P_m = A + Bp^m).

    seed : list, optional
        Seeds for the fit of B and p (A, if a variable, is seeded to the asymptote defined by `asympote`).

    asymptote : float, optional
        If not None, the A value for the fitting to A + Bp^m with A fixed. Defaults to 1/2^n. 
        Note that this value is used even when fitting A; in that case B and p are estimated 
        with A fixed to this value, and then this A and the estimated B and p are seed for the
        full fit.

    ftype : {'full','FA','full+FA'}, optional
        The fit type to implement. 'full' corresponds to fitting all of A, B and p. 'FA' corresponds
        to fixing 'A' to the value specified by `asymptote`. 'full+FA' returns the results of both
        fits.

    rtype : {'EI','AGI'}, optional
        The RB error rate rescaling convention. 'EI' results in RB error rates that are associated
        with the entanglement infidelity, which is the error probability with stochastic errors (and
        is equal to the diamond distance). 'AGI' results in RB error rates that are associated with
        average gate infidelity.

    Returns
    -------
    Dict or Dicts 
        If `ftype` = 'full' or `ftype`  = 'FA' then a dict containing the results of the relevant fit.
        If `ftype` = 'full+FA' then two dicts are returned. The first dict corresponds to the full fit
        and the second to the fixed-asymptote fit.
    """  
    if asymptote is not None: A = asymptote
    else: A = 1/2**n 
    # First perform a fit with a fixed asymptotic value
    FAF_results = custom_least_squares_data_fitting(lengths, ASPs, n, A=A, seed=seed)
    # Full fit is seeded by the fixed asymptote fit.
    seed_full = [FAF_results['estimates']['A'], FAF_results['estimates']['B'], FAF_results['estimates']['p']]        
    FF_results =  custom_least_squares_data_fitting(lengths, ASPs, n, seed=seed_full)
    # Returns the requested fit type.    
    if ftype == 'full': return FF_results
    elif ftype == 'FA': return FAF_results
    elif ftype == 'full+FA': return FF_results, FAF_results
    else: raise ValueError("The `ftype` value is invalid!") 

def custom_least_squares_data_fitting(lengths, ASPs, n, A=None, B=None, seed=None, rtype='EI'):
    """
    Fits RB average success probabilities to the exponential decay A + Bp^m using least-squares fitting.

    Parameters
    ----------
    lengths : list
        The RB lengths to fit to (the 'm' values in A + Bp^m).
 
    ASPs : list
        The average survival probabilities to fit (the observed P_m values to fit 
        to P_m = A + Bp^m).

    n : int
        The number of qubits the data is on..

    A : float, optional
        If not None, a value to fix A to.

    B : float, optional
        If not None, a value to fix B to.

    seed : list, optional
        Seeds for variables in the fit, in the order [A,B,p] (with A and/or B dropped if it is set
        to a fixed value).

    rtype : {'EI','AGI'}, optional
        The RB error rate rescaling convention. 'EI' results in RB error rates that are associated
        with the entanglement infidelity, which is the error probability with stochastic errors (and
        is equal to the diamond distance). 'AGI' results in RB error rates that are associated with
        average gate infidelity.

    Returns
    -------
    Dict 
        The fit results. If item with the key 'success' is False, the fit has failed.
    """
    seed_dict = {}
    variable = {}
    variable['A'] = True
    variable['B'] = True
    variable['p'] = True

    # The fit to do if a fixed value for A is given    
    if A is not None:
        
        variable['A'] = False
        
        if B is not None:
            
            variable['B'] = False

            def curve_to_fit(m,p):
                return A + B*p**m
            
            if seed is None:
                seed = 0.9
                seed_dict['A'] = None
                seed_dict['B'] = None
                seed_dict['p'] = seed
            
            try:    
                fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.],[1.]))
                p = fitout
                success = True
            except:
                success = False
         
        else:
            
            def curve_to_fit(m,B,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1.-A,0.9]
                seed_dict['A'] = None
                seed_dict['B'] = 1.-A
                seed_dict['p'] = 0.9
            try:    
                fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([-_np.inf,0.],[+_np.inf,1.]))
                B = fitout[0]
                p = fitout[1]
                success = True
            except:
                success = False

    # The fit to do if a fixed value for A is not given       
    else:
        
        if B is not None:
            
            variable['B'] = False
            
            def curve_to_fit(m,A,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1/2**n,0.9]
                seed_dict['A'] = 1/2**n
                seed_dict['B'] = None
                seed_dict['p'] = 0.9
            
            try:    
                fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.,0.],[1.,1.]))
                A = fitout[0]
                p = fitout[1]
                success = True
            except:
                success = False

        else:
            
            def curve_to_fit(m,A,B,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1/2**n,1-1/2**n,0.9]
                seed_dict['A'] = 1/2**n
                seed_dict['B'] = 1-1/2**n
                seed_dict['p'] = 0.9
                    
            try:
                fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.,-_np.inf,0.],[1.,+_np.inf,1.]))
                A = fitout[0]
                B = fitout[1]
                p = fitout[2]
                success = True
            except:
                success = False

    estimates = {}
    if success:
        estimates['A'] = A
        estimates['B'] = B
        estimates['p'] = p
        estimates['r'] = p_to_r(p,2**n)

    results = {}
    results['estimates'] = estimates
    results['variable'] = variable
    results['seed'] = seed_dict
    # Todo : fix this.
    results['success'] = success
    
    return results

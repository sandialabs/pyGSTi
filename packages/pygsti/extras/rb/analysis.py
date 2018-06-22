""" Functions for analyzing RB data"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from scipy.optimize import curve_fit as _curve_fit
  

def p_to_r(p, d, rtype='EI'):
    # Todo : docstring sort.
    """
    Converts an RB decay rate (p) to the RB error rate (r).
    
    The entanglement infidelity (EI) type r is given by:
    
    ,
    
    This converts the depolarizing constant ... in a u
    The average gate infidelity (AGI) type r is given by:
    
    
    r = (1 - p) * (d - 1) / d, given in, e.g., Magesan et al PRA 85
    042311 2012, or arXiv:1702.01853.
    
    Parameters
    ----------
    p : float
        Fit parameter p from P_m = A + B*p**m.
    
    d : int, optional
        Number of dimensions of the Hilbert space (default is 2,
        corresponding to a single qubit).     
     
    Returns
    -------
    r : float
        The RB number      
    
    """
    if rtype == 'AGI':
        r = (1 - p) * (d - 1) / d
    elif rtype == 'EI':
        # Todo : check this is correct
        r = (d**2 - 1) * (1 - p)/d**2
    else:
        raise ValueError("rtype must be `EI` (for entanglement infidelity) or `AGI` (for average gate infidelity)")
    
    return r

def r_to_p(r, d, rtype='EI'):
    """
    Inverse of p_to_r function. 
    
    """
    if rtype == 'AGI':
        p = 1 - d * r / (d - 1)       
    elif rtype == 'EI':       
        p = 1 - d**2 * r / (d**2 - 1)
    else:
        raise ValueError("rtype must be `EI` (for entanglement infidelity) or `AGI` (for average gate infidelity)")
        
    return p



#
# FUNCTIONS FROM OLD CODE
# FUNCTIONS FROM OLD CODE
# FUNCTIONS FROM OLD CODE
#
# ---- Fitting functions and related ----#
def standard_fit_function(m,A,B,p):
    """
    The standard RB decay fitting function P_m = A + B * p^m. This is 
    used in standard RB, and also variants on this (e.g., interleaved RB).
    
    Parameters
    ----------
    m : integer
        Length of random RB sequence (not including the inversion gate).
    
    A,B,p : float

    Returns
    -------
    float
    """
    return A+B*p**m

def first_order_fit_function(m,A,B,C,p):
    """
    The 'first order' fitting function P_m = A + (B + m * C) * p^m, from
    "Scalable and Robust Randomized Benchmarking of Quantum Processes" 
    (http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504).
    This is a simplified verion of the 'first order' in that paper (see Eq. 3),
    as the model therein has 1 too many parameters for fitting. The conversion is
    A = B_1
    B = A_1 - C_1(q/p^(-2) - 1)
    C = C_1(q/p^(-2) - 1)
    where the LHS (RHS) quantites in this equation are those of our (Magesan 
    et al.'s) fitting function.

    Parameters
    ----------
    m : integer
        Length of random RB sequence (not including the inversion gate).
    
    A,B,C,p : float

    Returns
    -------
    float
    """
    return A+(B+C*m)*p**m
#
# END OF FUNCTIONS FROM OLD CODE
# END OF FUNCTIONS FROM OLD CODE
# END OF FUNCTIONS FROM OLD CODE
#

def crb_rescaling_factor(lengths,quantity):
    
    rescaling_factor = []
    
    for i in range(len(lengths)):
        
        rescaling_factor.append(quantity[i]/(lengths[i]+1))
        
    rescaling_factor = _np.mean(_np.array(rescaling_factor))
    
    return rescaling_factor 

def custom_fit_data(lengths, ASPs, n, fixed_A=False, fixed_B=False, seed=None):
    
    # The fit to do if a fixed value for A is given    
    if fixed_A is not False:
        
        A = fixed_A
        
        if fixed_B is not False:
            
            B = fixed_B

            def curve_to_fit(m,p):
                return A + B*p**m
            
            if seed is None:
                seed = 0.9
                
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.],[1.]))
            p = fitout 
            
        else:
            
            def curve_to_fit(m,B,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1.-A,0.9]
                
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([-_np.inf,0.],[+_np.inf,1.]))
            B = fitout[0]
            p = fitout[1]
    
    # The fit to do if a fixed value for A is not given       
    else:
        
        if fixed_B is not False:
            
            B = fixed_B
            
            def curve_to_fit(m,A,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1/2**n,0.9]
                
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.,0.],[1.,1.]))
            A = fitout[0]
            p = fitout[1]
        
        else:
            
            def curve_to_fit(m,A,B,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1/2**n,1-1/2**n,0.9]
                    
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.,-_np.inf,0.],[1.,+_np.inf,1.]))
            A = fitout[0]
            B = fitout[1]
            p = fitout[2]
   
    results = {}
    results['A'] = A
    results['B'] = B
    results['p'] = p
    results['r'] = p_to_r(p,n)
    
    return results


def std_practice_analysis(RBSdataset, seed=[0.8,0.95], bootstrap_samples=1000, 
                          asymptote='std', finite_sample_error=True):
    
    lengths = RBSdataset.lengths
    ASPs = RBSdataset.ASPs
    successcounts = RBSdataset.successcounts
    counts = RBSdataset.counts
    n = RBSdataset.number_of_qubits

    if asymptote == 'std':
        asymptote = 1/2**n
    
    RBResults = _results.RBResults(RBSdataset)
    
    create_bootstraped_datasets(RBSdataset,finite_sample_error=True)    
    fit_full
    full_fit, fixed_asymptote_fit = std_fit_data(RBResults,seed=seed,asymptote=asymptote)
    
    #
    # Todo -- replace with a bootstrap data creation?
    #
    full_fit['r_bootstraps'] = bootstrap(RBSdataset, seed=seed, samples=bootstrap_samples, 
              fixed_asymptote=False,  asymptote=None, finite_sample_error=finite_sample_error)
    fixed_asymptote_fit['r_bootstraps'] = bootstrap(lengths, SPs, n, counts, seed=seed, samples=bootstrap_samples, 
              fixed_asymptote=True,  asymptote=asymptote, finite_sample_error=finite_sample_error)
    
    full_fit['r_std'] = _np.std(_np.array(full_fit['r_bootstraps']))
    fixed_asymptote_fit['r_std'] = _np.std(_np.array(fixed_asymptote_fit['r_bootstraps']))

    return full_fit, fixed_asymptote_fit
    
    
def std_fit_data(lengths, ASPs, n, seed=None, asymptote=None):
    
    lengths = RBSdataset.lengths
    ASPs = RBSdataset.ASPs
    n = RBSdataset.number_of_qubits
    
    if asymptote is not None:
        A = asymptote
    else:
        A = 1/2**n
    
    fixed_asymptote_fit = custom_fit_data(lengths, ASPs, n, fixed_A=A, fixed_B=False, seed=seed)
    seed_full = [fixed_asymptote_fit['A'], fixed_asymptote_fit['B'], fixed_asymptote_fit['p']]        
    full_fit =  custom_fit_data(lengths, ASPs, n, fixed_A=False, fixed_B=False, seed=seed_full)
    
    return full_fit, fixed_asymptote_fit

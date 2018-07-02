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



# #
# # FUNCTIONS FROM OLD CODE
# # FUNCTIONS FROM OLD CODE
# # FUNCTIONS FROM OLD CODE
# #
# # ---- Fitting functions and related ----#
# def standard_fit_function(m,A,B,p):
#     """
#     The standard RB decay fitting function P_m = A + B * p^m. This is 
#     used in standard RB, and also variants on this (e.g., interleaved RB).
    
#     Parameters
#     ----------
#     m : integer
#         Length of random RB sequence (not including the inversion gate).
    
#     A,B,p : float

#     Returns
#     -------
#     float
#     """
#     return A+B*p**m

# def first_order_fit_function(m,A,B,C,p):
#     """
#     The 'first order' fitting function P_m = A + (B + m * C) * p^m, from
#     "Scalable and Robust Randomized Benchmarking of Quantum Processes" 
#     (http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504).
#     This is a simplified verion of the 'first order' in that paper (see Eq. 3),
#     as the model therein has 1 too many parameters for fitting. The conversion is
#     A = B_1
#     B = A_1 - C_1(q/p^(-2) - 1)
#     C = C_1(q/p^(-2) - 1)
#     where the LHS (RHS) quantites in this equation are those of our (Magesan 
#     et al.'s) fitting function.

#     Parameters
#     ----------
#     m : integer
#         Length of random RB sequence (not including the inversion gate).
    
#     A,B,C,p : float

#     Returns
#     -------
#     float
#     """
#     return A+(B+C*m)*p**m
# #
# # END OF FUNCTIONS FROM OLD CODE
# # END OF FUNCTIONS FROM OLD CODE
# # END OF FUNCTIONS FROM OLD CODE
# #

def rescaling_factor(lengths, quantity, offset=2):
    
    rescaling_factor = []
    
    for i in range(len(lengths)):
        
        rescaling_factor.append(_np.array(quantity[i])/(_np.array(lengths[i])+offset))
        
    rescaling_factor = _np.mean(_np.array(rescaling_factor))
    
    return rescaling_factor 


def std_practice_analysis(RBSdataset, seed=[0.8,0.95], bootstrap_samples=200,  asymptote='std', rtype='EI'):
    """
    Todo : docstring.
    """  
    lengths = RBSdataset.lengths
    ASPs = RBSdataset.ASPs
    successcounts = RBSdataset.successcounts
    totalcounts = RBSdataset.totalcounts
    n = RBSdataset.number_of_qubits

    if asymptote == 'std':
        asymptote = 1/2**n

    FF_results, FAF_results = std_least_squares_data_fitting(lengths, ASPs, n, seed=seed, asymptote=asymptote, ftype='full+FA')

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
            BS_FF_results, BS_FAF_results = std_least_squares_data_fitting(lengths, BS_ASPs, n, seed=seed, asymptote=asymptote, ftype='full+FA')

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
    fits['full'] =  _results.FitResults('LS', FF_results['seed'], rtype, FF_results['success'], FF_results['estimates'], FF_results['variable'], stds=std_FF,  
               bootstraps=bootstraps_FF, bootstraps_failrate=bootstraps_failrate_FF)
    fits['A-fixed'] =  _results.FitResults('LS', FAF_results['seed'], rtype, FAF_results['success'], FAF_results['estimates'], FAF_results['variable'], stds=std_FAF,  
               bootstraps=bootstraps_FAF, bootstraps_failrate=bootstraps_failrate_FAF)

    results = _results.RBResults(RBSdataset,rtype,fits)

    return results
    
def std_least_squares_data_fitting(lengths, ASPs, n, seed=None, asymptote=None, ftype='full'):
    """
    ftype options 'full', 'full+FA', 'FA'

    """
   
    if asymptote is not None:
        A = asymptote
    else:
        A = 1/2**n
    
    # First perform a fit with a fixed asymptotic value
    FAF_results = custom_least_squares_data_fitting(lengths, ASPs, n, A=A, seed=seed)
    # Full fit is seeded by the fixed asymptote fit.
    seed_full = [FAF_results['estimates']['A'], FAF_results['estimates']['B'], FAF_results['estimates']['p']]        
    FF_results =  custom_least_squares_data_fitting(lengths, ASPs, n, seed=seed_full)
    
    if ftype == 'full':
        return FF_results
    if ftype == 'FA':
        return FAF_results
    if ftype == 'full+FA':
        return FF_results, FAF_results

def custom_least_squares_data_fitting(lengths, ASPs, n, A=None, B=None, seed=None, rtype='EI'):
    
    #todo : fix this
    success = True

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
            
            #try:    
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.],[1.]))
            p = fitout
            #except:
            #    # todo : fix this. 
            #    success = False

            
        else:
            
            def curve_to_fit(m,B,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1.-A,0.9]
                seed_dict['A'] = None
                seed_dict['B'] = 1.-A
                seed_dict['p'] = 0.9
                
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([-_np.inf,0.],[+_np.inf,1.]))
            B = fitout[0]
            p = fitout[1]
    
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
                
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.,0.],[1.,1.]))
            A = fitout[0]
            p = fitout[1]
        
        else:
            
            def curve_to_fit(m,A,B,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1/2**n,1-1/2**n,0.9]
                seed_dict['A'] = 1/2**n
                seed_dict['B'] = 1-1/2**n
                seed_dict['p'] = 0.9
                    
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.,-_np.inf,0.],[1.,+_np.inf,1.]))
            A = fitout[0]
            B = fitout[1]
            p = fitout[2]
   
    estimates = {}
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

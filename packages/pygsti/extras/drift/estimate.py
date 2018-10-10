from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

from . import signal as _sig

import numpy as _np
import time as _tm
from scipy.optimize import minimize as _minimize

def xlogp_rectified(x, p, min_p=1e-4, max_p=1-1e-6):
    """
    Todo
    """
    if x == 0: return 0
    # Fix pos_p to be no smaller than min_p and no larger than max_p
    pos_p = max(min_p,p)
    pos_p = min(max_p,pos_p)
    # The f(x_0) term in a Taylor expansion of xlog(y) around pos_p
    xlogp_rectified = x * _np.log(pos_p)
    # If p is less than this value, add in a quadratic term.
    if p < min_p:
        # The derivative of xlog(y) evaluated at min_p    
        S = x / min_p
        # The 2nd derivative of xlog(y) evaluated at min_p                                                                                              
        S2 = -0.5 * x / (min_p**2)
        # Adjusts v to be the Taylor expansion, to second order, of xlog(y) around min_p evaluated at p.                                                                                   
        xlogp_rectified += S*(p - min_p) + S2*(p - min_p)**2
    elif p > max_p:
        # The derivative of xlog(y) evaluated at max_p  
        S = x / max_p               
        # The 2nd derivative of xlog(y)evaluated at min_p                                                                                   
        S2 = -0.5 * x / (max_p**2)
        # Adds a fairly arbitrary drop-off term, to smooth out the hard boundary that should be imposed at p=1.                                                                                           
        xlogp_rectified += S*(p - max_p) + S2*(1 + p - max_p)**100

    return xlogp_rectified

def negLogLikelihood(alphas, omegas, data, min_p=1e-4, max_p=1-1e-6):
    """
    Todo
    """
    T = data.shape[0]
    obj = 0

    for t in _np.arange(T):
        pTemp = _sig.probability_from_DCT_amplitudes(alphas, omegas, T, t)
        xTemp = data[t]
        obj += xlogp_rectified(xTemp, pTemp, min_p, max_p) + xlogp_rectified(1-xTemp, 1-pTemp, min_p, max_p)

    return -obj

def do_maximum_likelihood_estimation_of_time_resolved_probability(data, omegas, alphas_seed=None, min_p=1e-4, max_p=1-1e-6,
                                                               method='Nelder-Mead', verbosity=1, return_aux=False):
    """
    Todo
    """
    assert(0 in omegas), "The zero mode (a constant probability) should always be included!"

    if alphas_seed is None:
        alphas_seed = _np.zeros(len(omegas))
        aplahs_seed[omegas.index(0)] = _np.mean(data)

    assert(len(omegas) == len(alphas_seed)), "The seed for the amplitudes must be the same length as the number of frequencies in the model!"
    T = len(data)
    start = _tm.time()
    if verbosity > 1:
        options = {'disp':True}
    else:
        options = {'disp':False}
    opt_results = _minimize(negLogLikelihood, alphas_seed, args=(omegas,data,min_p,max_p),
                            method=method, options=options)
    alphas = opt_results.x
    end = _tm.time()

    if verbosity > 0:
        print("Time taken: {} seconds".format(end-start)),
        NLLseed_adj = negLogLikelihood(alphas_seed, omegas, data, min_p, max_p)
        NLLseed = negLogLikelihood(alphas_seed, omegas, data, 0., 1.)
        NLLrsesult_adj = negLogLikelihood(alphas, omegas, data, min_p, max_p)
        NLLrsesult = negLogLikelihood(alphas, omegas, data, 0., 1.)
        print("The boundary-adjusted and actual negative log-likelihoods of the seed are {} and {}".format(NLLseed_adj, NLLseed))
        print("The boundary-adjusted and actual negative log-likelihoods of the result are {} and {}".format(NLLrsesult_adj, NLLrsesult))


    if return_aux:
        return alphas, opt_results
    else:
        return alphas
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

from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct
from scipy.fftpack import fft as _fft
from scipy.fftpack import ifft as _ifft

try:
    from astropy.stats import LombScargle as _LombScargle
except:
    pass

def xlogp_rectified(x, p, min_p=1e-4, max_p=1-1e-6):
    """
    Returns x*log(p) where p is bound within (0,1], with
    adjustments at the boundarys that are useful in minimization
    algorithms.

    If x == 0, returns 0.
    Otherwse:
        If p > min_p and p < max_p returns x*log(p)
        If ...
        If ...
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
        # The 2nd derivative of xlog(y)evaluated at max_p                                                                                   
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

def estimate_probability_trajectory(hyperparameters, timeseries, timestamps=None, transform='DCT', 
                                    estimator='FFLogistic', seed=None, modes=None, estimatorSettings=[]):
    """
    
    """
    # outcomes = list(timeseries.keys())

    # # There is a single hyper-parameter, it is assumed to corresponding to allowing fitting of the
    # # time-averaged probability for each outcome. 
    # if len(hyperparameters) == 1:
    #     parameters = {o: _np.mean(timeseries[o]) for o in outcomes}
    #     reconstructions = {o:[parameters[o] for t in range(len(timeseries[o]))] for o in outcomes}
    #     # Todo: update this 
    #     uncertainties = None 
    #     #auxDict = {'success':True, 'optimizerOut':None}
    #     return parameters, reconstruction, uncertainty, {}

    # if estimator == 'FFRaw':
    #     parameters = amplitudes_from_fourier_filter(freqInds, timeseries)

    # elif estimator == 'FFUniReduce':
    #     paramaters = iterative_amplitude_reduction(parameters, omegas, T, epsilon=epsilon, step_size=0.001, verbosity=verbosity)

    # elif estimator == 'FFLogistic':

    # elif estimator == 'MLE':

    # #elif estimator == 'LS':
    # #
    # #

    # if transform == 'DCT':

    return parameters, reconstruction, uncertainty, auxDict 

def amplitudes_from_fourier_filter(freqInds, timeseries):
    """

    """
    amplitudes = {}
    for o in list(timeseries.keys())[:-1]:
        modes = _dct(timeseries[o])[freInds]
        #keptmodes = _np.zeros(len(timeseries[o]),float)
        #keptmodes[freqInds] = modes[freqInds]
        parameters[o] = list(modes[freqInds])
        #reconstruction[o] = _idct(keptmodes)

    return parameters #, reconstruction, None


def estimate_probability_trace(sequence, outcome=0, entity=0, method='MLE', epsilon=0.001, minp=1e-6,
                               maxp=1-1e-6, verbosity=1, model_selection='local'):
    """        
    method :  'FFRaw', 'FFSharp', 'FFLogistic', 'FFUniReduce' 'MLE',
    """

    if not isinstance(outcome,int):
        assert(self.outcomes is not None)
        assert(outcome in self.outcomes)
        outcomeind = self.outcomes.index(outcome)
    else:
        outcomeind = outcome

    if not isinstance(sequence,int):
        assert(self.sequences_to_indices is not None)
        sequenceind = self.sequences_to_indices[sequence]
    else:
        sequenceind = sequence

    T = self.number_of_timesteps
    
    data = self.data[sequenceind,outcomeind,entity,:]
    modes = self.pspepo_modes[sequenceind,outcomeind,entity,:]
    mean = _np.mean(data)
    # This normalizer undoes the normalization done before converting to a power spectrum, and
    # the DCT normalization, and multiples by 1/N to make this a probability estimate rather than
    # a counts trace estimate.
    normalizer = _np.sqrt(2/T)*_np.sqrt(mean*(self.number_of_counts-mean)/self.number_of_counts)/self.number_of_counts_

    if model_selection == 'local':
        threshold = self.pspepo_significance_threshold           
        omegas = _np.arange(T)
        omegas = omegas[modes**2 >= threshold]
        omegas = list(omegas)
        omegas.insert(0,0)
        rawalphas = list(normalizer*modes[modes**2 >= threshold])
        rawalphas = list(rawalphas)
        rawalphas.insert(0,mean/self.number_of_counts)

    if model_selection == 'global':
        omegas = self.global_drift_frequencies
        omegas = list(omegas)
        rawalphas = list(normalizer*modes[omegas])
        rawalphas = list(rawalphas)
        omegas.insert(0,0)       
        rawalphas.insert(0,mean/self.number_of_counts)

    assert(method in ('FFRaw','FFSharp','FFLogistic','FFUniReduce','MLE')), "Method choice is not valid!"

    if method == 'FFRaw':
        def pt(t):
            return _sig.probability_from_DCT_amplitudes(rawalphas, omegas, T, t)
        return pt, omegas, rawalphas

    if method == 'FFSharp':
        def pt(t):
            raw = _sig.probability_from_DCT_amplitudes(rawalphas, omegas, T, t)
            if raw > 1:
                return 1
            elif raw < 0:
                return 0
            else:
                return raw
        return pt, omegas, rawalphas

    if method == 'FFLogistic':
        def pt(t):
            return _sig.logistic_transform(_sig.probability_from_DCT_amplitudes(rawalphas, omegas, T, t),mean)
        return pt, None, None

    reducedalphas = _sig.reduce_DCT_amplitudes_until_probability_is_physical(rawalphas, omegas, T, epsilon=epsilon, step_size=0.001, verbosity=verbosity)

    if method == 'FFUniReduce':
        def pt(t):
            return _sig.probability_from_DCT_amplitudes(reducedalphas, omegas, T, t)
        return pt, omegas, reducedalphas

    if method == 'MLE':
        mle_alphas = _est.do_maximum_likelihood_estimation_of_time_resolved_probability(data, omegas, alphas_seed=reducedalphas, min_p=minp, max_p=maxp,
                                                           method='Nelder-Mead', verbosity=verbosity, return_aux=False)
        def pt(t):
            return _sig.probability_from_DCT_amplitudes(mle_alphas, omegas, T, t)
        return pt, omegas, mle_alphas

    return
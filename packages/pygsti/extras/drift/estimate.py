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

# def negLogLikelihood(alphas, omegas, data, min_p=1e-4, max_p=1-1e-6):
#     """
#     Todo
#     """
#     T = data.shape[0]
#     obj = 0

#     for t in _np.arange(T):
#         pTemp = _sig.probability_from_DCT_amplitudes(alphas, omegas, T, t)
#         xTemp = data[t]
#         obj += xlogp_rectified(xTemp, pTemp, min_p, max_p) + xlogp_rectified(1-xTemp, 1-pTemp, min_p, max_p)

#     return -obj

def trajectoryNegLogLikelihood(p, x, min_p=1e-4, max_p=1-1e-6):
    """
    The negative log-likelihood of a probabilities trajectory.

    Parameters
    ----------
    p : dict
        The probabilities for each outcome, ....
    """
    logl = 0
    for outcome in x.keys():
        logl += _np.sum([xlogp_rectified(xot, pot, min_p, max_p) for xot,pot in zip(x[outcome],p[outcome])])
        #print(logl)

    # T = data.shape[0]
    # obj = 0

    # for t in _np.arange(T):
    #     pTemp = _sig.probability_from_DCT_amplitudes(alphas, omegas, T, t)
    #     xTemp = data[t]
    #     obj += xlogp_rectified(xTemp, pTemp, min_p, max_p) + xlogp_rectified(1-xTemp, 1-pTemp, min_p, max_p)

    return -logl

def loglikelihood_of_model(model, data, times, min_p=0., max_p=1.):
    """
    The log-likelihood of a time-resolved probabilities trajectory model.

    Parameters
    ----------
    model : ProbabilityTrajectoryModel
        The model to find the log-likelihood of.

    data : dict
        The data, consisting of a counts time-series for each measurement outcome. This is a dictionary
        whereby the keys are the outcome labels and the values are list (or arrays) giving the number
        of times that measurement outcome was observed at the corresponding time in the `times` list.

    times : list or array
        The times associated with the data. The probabilities are extracted from the model at these
        times, using the model.get_probabilites method
        .
    min_p : float, optional
        A positive value close to zero. The value of `p` below which x*log(p) is approximated using
        a Taylor expansion (used to smooth out the parameter boundaries and obtain better fitting
        performance). The default value of 0. give the true log-likelihood.

    max_p : float, optional
        A positive value close to and <= 1. The value of `p` above which x*log(p) the boundary on p
        being <= 1 is enforced using a smooth, quickly growing function. The default value of 1. 
        gives the true log-likelihood.

    Returns
    -------
    float
        The log-likehood of the model given the time-series data.

    """
    p = model.get_probabilities(times)
    return -trajectoryNegLogLikelihood(p, data, min_p, max_p)

def maximum_likelihood_model(model, data, times, min_p=1e-4, max_p=1-1e-6, method='Nelder-Mead', 
                             verbosity=1, returnOptout=False):                    
    """
    Implements maximum likelihood estimation over a model for a time-resolved probabilities trajectory,
    and returns the maximum likelihood model.

    Parameters
    ----------
    model : ProbabilityTrajectoryModel
        The model for which to maximize the likelihood of the parameters. The value of the parameters
        in the input model is used as the seed.

    data : dict
        The data, consisting of a counts time-series for each measurement outcome. This is a dictionary
        whereby the keys are the outcome labels and the values are list (or arrays) giving the number
        of times that measurement outcome was observed at the corresponding time in the `times` list.

    times : list or array
        The times associated with the data. The probabilities are extracted from the model at these
        times (see the model.get_probabilites method), to implement the model parameters optimization.

    min_p : float, optional
        A positive value close to zero. The value of `p` below which x*log(p) is approximated using
        a Taylor expansion (used to smooth out the parameter boundaries and obtain better fitting
        performance). The default value should be fine. 

    max_p : float, optional
        A positive value close to and <= 1. The value of `p` above which x*log(p) the boundary on p
        being <= 1 is enforced using a smooth, quickly growing function. The default value should be 
        fine. 

    method : str, optional
        Any value allowed for the method parameter in scipy.optimize.minimize(). 

    verbosity : int, optional
        The amount of print to screen.

    returnOptout : bool, optional
        Whether or not to return the output of the optimizer.

    Returns
    -------
    ProbabilityTrajectoryModel
        The maximum likelihood model returned by the optimizer.

    if returnOptout:
        optout
            The output of the optimizer.
    """
    mlemodel = model.copy()

    def objfunc(parameterslist):

        mlemodel.set_parameters_from_list(parameterslist)
        p = mlemodel.get_probabilities(times)

        return trajectoryNegLogLikelihood(p, data, min_p, max_p)
    
    options = {'disp':False}
    numparams = len(model.hyperparameters)*len(model.independent_outcomes)
    if verbosity > 0:
        print("      - Performing MLE over {} parameters...".format(numparams),end='')
    if verbosity > 1:
        print("")
        options = {'disp':True}

    start = _tm.time()
    seed = model.get_parameters_as_list()
    optout = _minimize(objfunc, seed, method=method, options=options)
    mleparameters = optout.x
    end = _tm.time()

    mlemodel.set_parameters_from_list(mleparameters)

    if verbosity == 1:
        print("complete.")
    if verbosity > 1:
        print("      - Complete!")
        print("      - Time taken: {} seconds".format(end-start)),
        ll_seed_adj = -loglikelihood_of_model(model, data, times, min_p, max_p)
        ll_seed = -loglikelihood_of_model(model, data, times)
        ll_result_adj = -loglikelihood_of_model(mlemodel, data, times, min_p, max_p)
        ll_result = -loglikelihood_of_model(mlemodel, data, times)
        print("      - The -loglikelihood of the seed = {} (with boundard adjustment = {})".format(ll_seed,ll_seed_adj))
        print("      - The -loglikelihood of the ouput = {} (with boundard adjustment = {})".format(ll_result,ll_result_adj))

    if returnOptout:
        return mlemodel, optout
    else:
        return mlemodel

def uniform_amplitude_compression(model, times, epsilon=0.001, stepsize=0.005, verbosity=1):
    """
    Todo. This only works given that parameter 0 is the DC-mode and the probablities sum to
    1 at each time.

    Returns
    -------

    """
    newmodel = model.copy()
    if len(newmodel.hyperparameters) <= 1:
        return model, False

    pt = newmodel.get_probabilities(times)

    maxpt = max([max(pt[o]) for o in model.parameters.keys()])
    minpt = min([min(pt[o]) for o in model.parameters.keys()])

    iteration = 1
    modelchanged = False
    while maxpt >= 1-epsilon or minpt <= epsilon:
        
        modelchanged = True
        newparameters = model.parameters.copy()
        for i in model.parameters.keys():
            newparameters[i][1:] = [_sig.decrease_magnitude(p,stepsize) for p in newparameters[i][1:]]
        
        # Input the new parameters to the model
        newmodel.set_parameters(newparameters)
        # Get the new probabilities trajectory
        pt = model.get_probabilities(times)
        # Find out it's max and min value
        maxpt = max([max(pt[o]) for o in model.parameters.keys()])
        minpt = min([min(pt[o]) for o in model.parameters.keys()])

        if iteration >= 10000:
            _warnings.warning("10,000 iterations implemented trying to make model physical! Quiting and returning unphysical model.")
            return model 

        iteration += 1

    return newmodel, modelchanged

def likelihood_of_general_model(probTrajectoriesFunction, parameters, times, data): 
    """
    Todo.
    """      
    negll = 0.
    for opstr in data.keys():
        p = probTrajectoriesFunction(parameters, opstr, times[mdl])
        negll += trajectoryNegLogLikelihood(p, data[opstr], min_p, max_p)
    
    return negll


def maximum_likelihood_over_general_model(probTrajectoriesFunction, times, data, seed, min_p=1e-4, max_p=1-1e-6, 
                                          verbosity=1, bounds=None, returnOptout=False): 
    """
    Todo.
    """
    def objfunc(parameters):
        
        negll = 0.
        for mdl in data.keys():
            p = probTrajectoriesFunction(parameters, mdl, times[mdl])
            negll += trajectoryNegLogLikelihood(p, data[mdl], min_p, max_p)
        
        return negll

    options = {'disp':False}
    if verbosity > 0:
        print("- Performing MLE over {} parameters...".format(len(seed)),end='')
    if verbosity > 1:
        print("")
        options = {'disp':True}
    
    start = _tm.time()
    optout = _minimize(objfunc, seed, options=options, bounds=bounds)
    mleparameters = optout.x
    end = _tm.time()

    if verbosity == 1:
        print("complete.")
    if verbosity > 0:
        print("- Time taken: {} seconds".format(end-start)),

    if returnOptout:
        return mleparameters, optout 
    else:
        return mleparameters

# def estimate_probability_trajectory(hyperparameters, timeseries, timestamps=None, transform='DCT', 
#                                     estimator='FFLogistic', seed=None, modes=None, estimatorSettings=[]):
#     """
    
#     """
#     # outcomes = list(timeseries.keys())

#     # # There is a single hyper-parameter, it is assumed to corresponding to allowing fitting of the
#     # # time-averaged probability for each outcome. 
#     # if len(hyperparameters) == 1:
#     #     parameters = {o: _np.mean(timeseries[o]) for o in outcomes}
#     #     reconstructions = {o:[parameters[o] for t in range(len(timeseries[o]))] for o in outcomes}
#     #     # Todo: update this 
#     #     uncertainties = None 
#     #     #auxDict = {'success':True, 'optimizerOut':None}
#     #     return parameters, reconstruction, uncertainty, {}

#     # if estimator == 'FFRaw':
#     #     parameters = amplitudes_from_fourier_filter(freqInds, timeseries)

#     # elif estimator == 'FFUniReduce':
#     #     paramaters = iterative_amplitude_reduction(parameters, omegas, T, epsilon=epsilon, step_size=0.001, verbosity=verbosity)

#     # elif estimator == 'FFLogistic':

#     # elif estimator == 'MLE':

#     # #elif estimator == 'LS':
#     # #
#     # #

#     # if transform == 'DCT':

#     return parameters, reconstruction, uncertainty, auxDict 

# def amplitudes_from_fourier_filter(freqInds, timeseries):
#     """

#     """
#     amplitudes = {}
#     for o in list(timeseries.keys())[:-1]:
#         modes = _dct(timeseries[o])[freInds]
#         #keptmodes = _np.zeros(len(timeseries[o]),float)
#         #keptmodes[freqInds] = modes[freqInds]
#         parameters[o] = list(modes[freqInds])
#         #reconstruction[o] = _idct(keptmodes)

#     return parameters #, reconstruction, None

# def estimate_probability_trace(sequence, outcome=0, entity=0, method='MLE', epsilon=0.001, minp=1e-6,
#                                maxp=1-1e-6, verbosity=1, model_selection='local'):
#     """        
#     method :  'FFRaw', 'FFSharp', 'FFLogistic', 'FFUniReduce' 'MLE',
#     """

#     if not isinstance(outcome,int):
#         assert(self.outcomes is not None)
#         assert(outcome in self.outcomes)
#         outcomeind = self.outcomes.index(outcome)
#     else:
#         outcomeind = outcome

#     if not isinstance(sequence,int):
#         assert(self.sequences_to_indices is not None)
#         sequenceind = self.sequences_to_indices[sequence]
#     else:
#         sequenceind = sequence

#     T = self.number_of_timesteps
    
#     data = self.data[sequenceind,outcomeind,entity,:]
#     modes = self.pspepo_modes[sequenceind,outcomeind,entity,:]
#     mean = _np.mean(data)
#     # This normalizer undoes the normalization done before converting to a power spectrum, and
#     # the DCT normalization, and multiples by 1/N to make this a probability estimate rather than
#     # a counts trace estimate.
#     normalizer = _np.sqrt(2/T)*_np.sqrt(mean*(self.number_of_counts-mean)/self.number_of_counts)/self.number_of_counts_

#     if model_selection == 'local':
#         threshold = self.pspepo_significance_threshold           
#         omegas = _np.arange(T)
#         omegas = omegas[modes**2 >= threshold]
#         omegas = list(omegas)
#         omegas.insert(0,0)
#         rawalphas = list(normalizer*modes[modes**2 >= threshold])
#         rawalphas = list(rawalphas)
#         rawalphas.insert(0,mean/self.number_of_counts)

#     if model_selection == 'global':
#         omegas = self.global_drift_frequencies
#         omegas = list(omegas)
#         rawalphas = list(normalizer*modes[omegas])
#         rawalphas = list(rawalphas)
#         omegas.insert(0,0)       
#         rawalphas.insert(0,mean/self.number_of_counts)

#     assert(method in ('FFRaw','FFSharp','FFLogistic','FFUniReduce','MLE')), "Method choice is not valid!"

#     if method == 'FFRaw':
#         def pt(t):
#             return _sig.probability_from_DCT_amplitudes(rawalphas, omegas, T, t)
#         return pt, omegas, rawalphas

#     if method == 'FFSharp':
#         def pt(t):
#             raw = _sig.probability_from_DCT_amplitudes(rawalphas, omegas, T, t)
#             if raw > 1:
#                 return 1
#             elif raw < 0:
#                 return 0
#             else:
#                 return raw
#         return pt, omegas, rawalphas

#     if method == 'FFLogistic':
#         def pt(t):
#             return _sig.logistic_transform(_sig.probability_from_DCT_amplitudes(rawalphas, omegas, T, t),mean)
#         return pt, None, None

#     reducedalphas = _sig.reduce_DCT_amplitudes_until_probability_is_physical(rawalphas, omegas, T, epsilon=epsilon, step_size=0.001, verbosity=verbosity)

#     if method == 'FFUniReduce':
#         def pt(t):
#             return _sig.probability_from_DCT_amplitudes(reducedalphas, omegas, T, t)
#         return pt, omegas, reducedalphas

#     if method == 'MLE':
#         mle_alphas = _est.do_maximum_likelihood_estimation_of_time_resolved_probability(data, omegas, alphas_seed=reducedalphas, min_p=minp, max_p=maxp,
#                                                            method='Nelder-Mead', verbosity=verbosity, return_aux=False)
#         def pt(t):
#             return _sig.probability_from_DCT_amplitudes(mle_alphas, omegas, T, t)
#         return pt, omegas, mle_alphas

#     return

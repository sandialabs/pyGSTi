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
import warnings as _warnings
import copy as _copy
from scipy.optimize import minimize as _minimize


from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct
from scipy.fftpack import fft as _fft
from scipy.fftpack import ifft as _ifft

try:
    from astropy.stats import LombScargle as _LombScargle
except:
    pass


class ProbTrajectory(object):
    """
    todo

    """
    def __init__(self, outcomes, hyperparameters, parameters):
        """
        todo:

        Parameters
        ----------

        hyperparameters :

        parameters :
            A dictonary where the keys are all but the last element of `outcomes`, and the elements
            are lists of the same lengths as `hyperparameters`.

        Returns
        -------
        None
        """
        self.outcomes = outcomes
        self.numoutcomes = len(outcomes)
        self.set_hyperparameters(hyperparameters, parameters)

    def copy(self):

        return _copy.deepcopy(self)

    def basisfunction(self, i, times):
        """
        todo.

        """
        raise NotImplementedError("This should be defined in derived classes!")

    def set_hyperparameters(self, hyperparameters, parameters):
        """
        todo

        """
        hyperparameters = list(hyperparameters)
        self.hyperparameters = hyperparameters
        self.set_parameters(parameters)

    def set_parameters(self, parameters):
        """
        todo

        """
        self.parameters = parameters

    def set_parameters_from_list(self, parameterslist):
        """
        todo:

        """
        self.parameters = {o: parameterslist[oind * len(self.hyperparameters):(1 + oind) * len(self.hyperparameters)]
                           for oind, o in enumerate(self.outcomes[:-1])}

    def get_parameters_as_list(self):
        """
        todo:

        """
        parameterslist = []
        for o in self.outcomes[:-1]:
            parameterslist += self.parameters[o]

        return parameterslist

    def get_parameters(self):
        """
        todo:

        """
        return self.parameters

    def get_probabilities(self, times):
        """
        Returns the probability distribution for each time in `times`.

        Parameters
        ----------
        times : list
            A list of times, to return the probability distributions for.

        Returns
        -------
        dict
            A dictionary where the keys are the possible outcomes, and the
            value for a key is a list (of the same lengths as `times`) containing
            the probability for that outcome at the times in `times`.

        """
        totalprob = 0
        probs = {}
        # Loop through all but the last outcome, and find their probability trajectories.
        for o in self.outcomes[:-1]:
            p = _np.sum(_np.array([self.parameters[o][ind] * self.basisfunction(i, times)
                                   for ind, i in enumerate(self.hyperparameters)]), axis=0)
            probs[o] = p
            totalprob += p  # Keeps track of the total probability accounted for.

        # Add in the probability trace for the final outcome, which is specified indirectly.
        probs[self.outcomes[-1]] = 1 - totalprob

        return probs


class ConstantProbTrajectory(ProbTrajectory):

    def __init__(self, outcomes, probabilities):
        """

        """
        super().__init__(outcomes, [0, ], probabilities)

    def basisfunction(self, i, times):
        """

        """
        return _np.ones(len(times), float)


class CosineProbTrajectory(ProbTrajectory):

    def __init__(self, outcomes, hyperparameters, parameters, starttime, timestep, numtimes):
        """
        todo

        """

        self.starttime = starttime
        self.timestep = timestep
        self.numtimes = numtimes

        #    self.fullmodelsize = self.numsteps

        super().__init__(outcomes, hyperparameters, parameters)

    def basisfunction(self, i, times):
        """
        Todo
        """
        return _np.array([_np.cos(i * _np.pi * ((t - self.starttime) / self.timestep + 0.5) / self.numtimes)
                          for t in times])


def _xlogp_rectified(x, p, minp=0.0001, maxp=0.999999):
    """
    Returns x*log(p) where p is bound within (0,1], with
    adjustments at the boundaries that are useful in minimization
    algorithms.

    """
    if x == 0: return 0

    # Fix pos_p to be no smaller than minp and no larger than maxp
    pos_p = max(minp, p)
    pos_p = min(maxp, pos_p)
    _xlogp_rectified = x * _np.log(pos_p)  # The f(x_0) term in a Taylor expansion of xlog(y) around pos_p

    # Adjusts to the Taylor expansion, to second order, of xlog(y) around minp evaluated at p.
    if p < minp: 

        S = x / minp  # The derivative of xlog(y) evaluated at minp
        S2 = -0.5 * x / (minp**2)  # The 2nd derivative of xlog(y) evaluated at minp
        _xlogp_rectified += S * (p - minp) + S2 * (p - minp)**2

    # Adds a fairly arbitrary smooth drop off to the hard boundary at p=1.
    elif p > maxp:  

        S = x / maxp  # The derivative of xlog(y) evaluated at maxp
        S2 = -0.5 * x / (maxp**2)  # The 2nd derivative of xlog(y)evaluated at maxp
        _xlogp_rectified += S * (p - maxp) + S2 * (1 + p - maxp)**100

    return _xlogp_rectified


def negloglikelihood(probtrajectory, clickstreams, times, minp=0., maxp=1.):
    """
    The log-likelihood of a time-resolved probabilities trajectory model.

    Parameters
    ----------
    model : ProbTrajectoryModel
        The model to find the log-likelihood of.

    data : dict
        The data, consisting of a counts time-series for each measurement outcome. This is a dictionary
        whereby the keys are the outcome labels and the values are list (or arrays) giving the number
        of times that measurement outcome was observed at the corresponding time in the `times` list.

    times : list or array
        The times associated with the data. The probabilities are extracted from the model at these
        times, using the model.get_probabilites method
        .
    minp : float, optional
        A positive value close to zero. The value of `p` below which x*log(p) is approximated using
        a Taylor expansion (used to smooth out the parameter boundaries and obtain better fitting
        performance). The default value of 0. give the true log-likelihood.

    maxp : float, optional
        A positive value close to and <= 1. The value of `p` above which x*log(p) the boundary on p
        being <= 1 is enforced using a smooth, quickly growing function. If set to 1. it gives the 
        true log-likelihood.

    Returns
    -------
    float
        The log-likehood of the model given the time-series data.
    """
    p = probtrajectory.get_probabilities(times)
    logl = 0
    for outcome in clickstreams.keys():
        logl += _np.sum([_xlogp_rectified(xot, pot, minp, maxp) for xot, pot in zip(clickstreams[outcome],
                                                                                    p[outcome])])
    return -logl


def maxlikelihood(probtrajectory, clickstreams, times, minp=0.0001, maxp=0.999999, method='Nelder-Mead',
                  returnOptout=False, options={}, verbosity=1):
    """
    Implements maximum likelihood estimation over a model for a time-resolved probabilities trajectory,
    and returns the maximum likelihood model.

    Parameters
    ----------
    model : ProbTrajectoryModel
        The model for which to maximize the likelihood of the parameters. The value of the parameters
        in the input model is used as the seed.

    clickstreams : dict
        The data, consisting of a counts time-series for each measurement outcome. This is a dictionary
        whereby the keys are the outcome labels and the values are list (or arrays) giving the number
        of times that measurement outcome was observed at the corresponding time in the `times` list.

    times : list or array
        The times associated with the data. The probabilities are extracted from the model at these
        times (see the model.get_probabilites method), to implement the model parameters optimization.

    minp : float, optional
        A positive value close to zero. The value of `p` below which x*log(p) is approximated using
        a Taylor expansion (used to smooth out the parameter boundaries and obtain better fitting
        performance). The default value should be fine. 

    maxp : float, optional
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
    ProbTrajectoryModel
        The maximum likelihood model returned by the optimizer.

    if returnOptout:
        optout
            The output of the optimizer.
    """
    maxlprobtrajectory = probtrajectory.copy()

    def objfunc(parameterslist):

        maxlprobtrajectory.set_parameters_from_list(parameterslist)

        return negloglikelihood(maxlprobtrajectory, clickstreams, times, minp, maxp)

    numparams = len(probtrajectory.hyperparameters) * (len(probtrajectory.outcomes) - 1)

    if verbosity > 0:
        print("      - Performing MLE over {} parameters...".format(numparams), end='')
    if verbosity > 1:
        print("")
        options['disp'] = True 

    start = _tm.time()
    seed = probtrajectory.get_parameters_as_list()
    optout = _minimize(objfunc, seed, method=method, options=options)
    mleparameters = optout.x
    end = _tm.time()

    maxlprobtrajectory.set_parameters_from_list(mleparameters)

    if verbosity == 1:
        print("complete.")
    if verbosity > 1:
        print("      - Complete!")
        print("      - Time taken: {} seconds".format(end - start)),
        nll_seed_adj = negloglikelihood(probtrajectory, clickstreams, times, minp, maxp)
        nll_seed = negloglikelihood(probtrajectory, clickstreams, times, 0, 1)
        nll_result_adj = negloglikelihood(maxlprobtrajectory, clickstreams, times, minp, maxp)
        nll_result = negloglikelihood(maxlprobtrajectory, clickstreams, times, 0, 1)
        print("      - The negloglikelihood of the seed = {} (with boundard adjustment = {})".format(nll_seed, nll_seed_adj))
        print("      - The negloglikelihood of the ouput = {} (with boundard adjustment = {})".format(nll_result, nll_result_adj))

    if returnOptout:
        return maxlprobtrajectory, optout
    else:
        return maxlprobtrajectory


def amplitude_compression(probtrajectory, epsilon=0., verbosity=1):
    """
    todo

    Returns
    -------

    """
    assert(isinstance(probtrajectory, CosineProbTrajectory)), "Input must be a CosineProbTrajectory!"

    def update_multiplier(multipler, a, b, epsilon):
        """
        Updates `multiplier` so that the following two equations
        are satisfied:

        a - (b * multipler) >= epsilon
        a + (b * multipler) =< 1 - epsilon

        where `a`, `b` and `epsilon` are positive. If the equations
        are already satisfied, `multipler` is unchanged. Otherwise it
        this function returns the maximum `multipler` such that this holds,
        i.e., the `multipler` that satisfies at least one of these conditions
        as an equality.

        """
        if a - (b * multipler) < epsilon:
            multipler = (a - epsilon) / b
        if a + (b * multipler) > 1 - epsilon:
            multipler = (1 - epsilon - a) / b

        return multipler

    multiplier = 1

    alphassum = _np.zeros(len(probtrajectory.hyperparameters), float)

    for o in probtrajectory.outcomes[:-1]:
        alphas = probtrajectory.parameters[o]
        alpha0 = alphas[0]
        alphai_abssum = _np.sum(abs(_np.array(alphas[1:])))
        multiplier = update_multiplier(multiplier, alpha0, alphai_abssum, epsilon)

        alphassum += _np.array(alphas)

    alpha0 = 1 - alphassum[0]
    alphai_abssum = _np.sum(abs(-_np.array(alphassum[1:])))
    multiplier = update_multiplier(multiplier, alpha0, alphai_abssum, epsilon)

    if multiplier < 1:
        shape = (len(probtrajectory.outcomes) - 1, len(probtrajectory.hyperparameters))
        parameters = _np.reshape(probtrajectory.get_parameters_as_list(), shape)
        compparameters = multiplier * parameters
        compparameters[:, 0] = parameters[:, 0]
        compparameters = compparameters.flatten()
        comppt = probtrajectory.copy()
        comppt.set_parameters_from_list(compparameters)

        return comppt, True
    else:
        return probtrajectory, False

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
        self.hyperparameters = _copy.deepcopy(hyperparameters)
        self.set_parameters(parameters)

    def set_parameters(self, parameters):
        """
        todo

        """
        self.parameters = _copy.deepcopy(parameters)

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

        return _copy.copy(parameterslist)

    def get_parameters(self):
        """
        todo:

        """
        return _copy.deepcopy(self.parameters)

    def get_probabilities(self, times, trim=True):
        """
        Returns the probability distribution for each time in `times`.

        Parameters
        ----------
        times : list
            A list of times, to return the probability distributions for.

        time : bool, optional
            Whether or not to set probability > 1 to 1 and probabilities
            < 0 to 0. If set to True then there is no guarantee that the
            probabilities will sum to 1 at all times anymore.

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
            # This trimming means it's possible to have probabilities that don't sum exactly to 1.
            if trim:
                p[p > 1] = 1
                p[p < 0] = 0
            probs[o] = p
            totalprob += p  # Keeps track of the total probability accounted for.

        # Calculate and add in the probability trace for the final outcome, which is specified indirectly.
        p = 1 - totalprob
        # This trimming means it's possible to have probabilities that don't sum exactly to 1.
        if trim:
            p[p > 1] = 1
            p[p < 0] = 0
        probs[self.outcomes[-1]] = p
        return probs


class ConstantProbTrajectory(ProbTrajectory):

    def __init__(self, outcomes, probabilities):
        """

        """
        super(ConstantProbTrajectory, self).__init__(outcomes, [0, ], probabilities)

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

        super(CosineProbTrajectory, self).__init__(outcomes, hyperparameters, parameters)

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
    The negative log-likelihood of a time-resolved probabilities trajectory model.

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
    probs = probtrajectory.get_probabilities(times)
    return probsdict_negloglikelihood(probs, clickstreams, minp, maxp)


def probsdict_negloglikelihood(probs, clickstreams, minp=0., maxp=1.):
    """
    The negative log-likelihood of varying probabilities `probs`, evaluated for the data streams
    in `clickstreams`.
    """
    logl = 0
    for outcome in clickstreams.keys():
        logl += _np.sum([_xlogp_rectified(xot, pot, minp, maxp) for xot, pot in zip(clickstreams[outcome],
                                                                                    probs[outcome])])

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
        print("      - The negloglikelihood of the seed = {} (with boundard adjustment = {})".format(
            nll_seed, nll_seed_adj))
        print("      - The negloglikelihood of the ouput = {} (with boundard adjustment = {})".format(
            nll_result, nll_result_adj))

    if returnOptout:
        return maxlprobtrajectory, optout
    else:
        return maxlprobtrajectory


def amplitude_compression(probtrajectory, times, epsilon=0., verbosity=1):
    """
    Reduces the amplitudes in a CosineProbTrajectory model until the
    model is valid, i.e., all probabilities are within [0,1].

    Parameters
    ----------
    probtrajectory: CosineProbTrajectory
        The model on which to perform the amplitude reduction

    times: list
        The times at which to enforce the validity of the model (this algorithm
        does *not* guarantee that the probabilities will be within [0,1] at all
        times)

    epsilon: float, optional
        The amplitudes are compressed so that all the probabilities are
        within [0+epsilon,1-epsilon] at all times. Setting this to be
        larger than 0 can be useful as it guarantees that the resultant
        probability trajectory has a non-zero likelihood.

    Returns
    -------
    CosineProbTrajectory
        The new model, that may have had the amplitudes reduced

    Bool
        Whether or not the function did anything non-trivial, i.e, whether any
        compression was required.

    """
    assert(isinstance(probtrajectory, CosineProbTrajectory)), "Input must be a CosineProbTrajectory!"

    def satisfies_hard_constraint(a, b, epsilon):
        """
        Returns True if  a - b >= epsilon and  a + b =< 1 - epsilon. Otherwise
        Returns False.

        """
        return (a - b >= epsilon) and (a + b >= 1 - epsilon)

    def rectify_alpha0(alpha0):
        """
        Fixes a probability that's strayed slightly out of [0,1]
        """
        if alpha0 > 1:
            assert(abs(alpha0 - 1) < 1e-6), "The mean of each trajectory must be within [0,1]!"
            return 1

        elif alpha0 < 0:
            assert(abs(alpha0 - 0) < 1e-6), "The mean of each trajectory must be within [0,1]!"
            return 0

        else:
            return alpha0

    multiplier = 1

    alpha0s = {}
    probs = probtrajectory.get_probabilities(times, trim=False)
    params = probtrajectory.get_parameters()
    # Add in the outcome that's parameters are fully defined by the others.
    params[probtrajectory.outcomes[-1]] = _np.zeros(len(params[list(params.keys())[0]]))
    # Populate those parameters.
    for o in probtrajectory.outcomes[:-1]: params[probtrajectory.outcomes[-1]] += params[o]
    # The constant prob is 1 - the sum of the other constant probs.
    params[probtrajectory.outcomes[-1]][0] = 1 - params[probtrajectory.outcomes[-1]][0]

    for o in probtrajectory.outcomes:

        alpha0s[o] = rectify_alpha0(params[o][0])
        minprob = _np.min(probs[o])
        maxprob = _np.max(probs[o])
        # If it's a constant probability trajectory, we skip this.
        if abs(minprob - maxprob) < 1e-7:
            pass
        else:
            if minprob < epsilon:
                # Find the multipler such that alpha0 + min(probs-alpha0) * multipler = epsilon
                if minprob - alpha0s[o] < 0:
                    newmultiplier = (epsilon - alpha0s[o]) / (minprob - alpha0s[o])
                    multiplier = min(multiplier, newmultiplier)

            if maxprob > 1 - epsilon:
                # Find the multipler such that alpha0 + max(probs-alpha0) * multipler = 1 - epsilon
                if maxprob - alpha0s[o] > 0:
                    newmultiplier = (1 - epsilon - alpha0s[o]) / (maxprob - alpha0s[o])
                    multiplier = min(multiplier, newmultiplier)

    if multiplier < 1:
        shape = (len(probtrajectory.outcomes) - 1, len(probtrajectory.hyperparameters))
        parameters = _np.reshape(probtrajectory.get_parameters_as_list(), shape)
        compparameters = multiplier * parameters
        # set the alpha0s as unchanged, except if rectified:
        for ind, o in enumerate(probtrajectory.outcomes[:-1]):
            compparameters[ind, 0] = alpha0s[o]
        compparameters = compparameters.flatten()
        comppt = probtrajectory.copy()
        comppt.set_parameters_from_list(list(compparameters))

        return comppt, True

    else:
        return probtrajectory, False

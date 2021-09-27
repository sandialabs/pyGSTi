#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

import copy as _copy
import time as _tm

import numpy as _np
from scipy.optimize import minimize as _minimize

try:
    from astropy.stats import LombScargle as _LombScargle
except:
    pass


class ProbTrajectory(object):
    """
    Encapulates a time-dependent probability distribution, as a sum of time-dependent basis functions.

    """

    def __init__(self, outcomes, hyperparameters, parameters):
        """
        Initializes a ProbTrajectory object.

        Parameters
        ----------
        outcomes : list
            What the probability distribution is over. Typically, all possible outcomes for the circuit
            that this is a probability trajectory for.

        hyperparameters : list
            Each derived ProbTrajectory object is intended to encompass a family of parameterized models
            for a time-varying probability, and this specifies the specific parameterized model in the class.
            For example, the CosineProbTrajectory object is a derived class whereby each probability is the
            summation of some number of cosines, and this specifies the number and frequencies of those
            cosines. The probability trajectory for *each* outcome is parameterized by a value for each
            hyperparameter.

        parameters :
            A dictonary where the keys are all but the last element of `outcomes`, and the elements
            are lists of the same lengths as `hyperparameters`. These are the parameters of the parameterized
            model defined by (the derived class) and `hyperparameters`. The parameter values for the final
            outcome are assumed to be entirely fixed by the necessity for the probability trajectories
            to sum to 1 at all times.

        Returns
        -------
        A new ProbTrajectory object.

        """
        self.outcomes = outcomes
        self.numoutcomes = len(outcomes)
        self.set_hyperparameters(hyperparameters, parameters)

    def copy(self):

        return _copy.deepcopy(self)

    def basisfunction(self, i, times):
        """
        The ith basis function of the model, evaluated at the times in `times.

        *** Defined in a derived class ***

        Parameters
        ----------
        i : Type specified by derived class.
            The basis function specified by the hyperparameter `i`. This method should
            expect all possible hyperparameter values (often ints in some range, or a float)
            as this input

        times : list
            The times to evaluate the basis function at.

        Returns
        -------
        list
            The values of the basis function at the specified times.

        """
        raise NotImplementedError("This should be defined in derived classes!")

    def set_hyperparameters(self, hyperparameters, parameters):
        """
        Sets the hyperparameters -- i.e., defines a new parameterized model -- and
        the parameters (see init for details).

        """
        hyperparameters = list(hyperparameters)
        self.hyperparameters = _copy.deepcopy(hyperparameters)
        self.set_parameters(parameters)

    def set_parameters(self, parameters):
        """
        Sets the parameters of the model (see init for details).

        """
        self.parameters = _copy.deepcopy(parameters)

    def set_parameters_from_list(self, parameterslist):
        """
        Sets the parameters of the model from a list

        Parameters
        ----------
        parametersaslist : list
            The new parameter values as a list, where the first len(self.hyperparameters) values
            are the parameter values for the first outcome (the first element of self.outcomes),
            the second set of len(self.hyperparameters) values are for the second outcome in this
            list, and so on, up to the second last value of self.outcomes.

        Returns
        -------
        None
        """
        self.parameters = {o: parameterslist[oind * len(self.hyperparameters):(1 + oind) * len(self.hyperparameters)]
                           for oind, o in enumerate(self.outcomes[:-1])}

    def parameters_as_list(self):
        """
        Returns the parameters as a list, in the same format as when input to `set_parameters_from_list`.
        See the docstring of that method for more info.

        """
        parameterslist = []
        for o in self.outcomes[:-1]:
            parameterslist += self.parameters[o]

        return _copy.copy(parameterslist)

    def parameters_copy(self):
        """
        Returns the values of the parameters, in the dictionary form in which it is internally stored.

        """
        return _copy.deepcopy(self.parameters)

    def probabilities(self, times, trim=True):
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
            # This trimming means it is possible to have probabilities that don't sum exactly to 1,
            # even if they did before this trimming was implemented.
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
    """
    Encapulates a time-dependent probability distribution that is actually a constant.
    Useful when wanting to encode a constant probability distribution in a way that
    can be used consistently with any other ProbTrajectory object.

    """

    def __init__(self, outcomes, probabilities):
        """
        Initializes a ConstantProbTrajectory object.

        Parameters
        ----------
        outcomes : list
            What the probability distribution is over. Typically, all possible outcomes for the circuit
            that this is a probability trajectory for.

        probabilities : dict
           The static probability to obtained all but the last outcome (which is set by the other
           probabilities).

        Returns
        -------
        A new ConstantProbTrajectory object.

        """
        super(ConstantProbTrajectory, self).__init__(outcomes, [0, ], probabilities)

    def basisfunction(self, i, times):
        """

        """
        return _np.ones(len(times), float)


class CosineProbTrajectory(ProbTrajectory):
    """
    Encapulates a time-dependent probability distribution that is parameterized as a
    sum of cosines. Specifically, it is parameterized as the sum of the Type-II DCT
    basis functions.

    """

    def __init__(self, outcomes, hyperparameters, parameters, starttime, timestep, numtimes):
        """
        Initializes a CosineProbTrajectory object.

        Parameters
        ----------
        outcomes : list
            What the probability distribution is over. Typically, all possible outcomes for the circuit
            that this is a probability trajectory for.

        hyperparameters : list
            A set of integers, that specify the indices of the the DCT basis functions to include.
            This *must* include 0 as the first element, corresponding to the constant component of
            the probability trajectories.

        parameters : dict
            A dictonary where the keys are all but the last element of `outcomes`, and the elements
            are lists of the same lengths as `hyperparameters`. These are amplitudes for the DCT basis
            functions, for each outcome. The first element of each list is the constant component of
            the that probability trajectory.

        starttime : float
            The start time of the time period over which the DCT basis functions are being defined.
            This is typically set to the first data collection time of the circuit that this probability
            trajectory is being defined for.

        timestep : float
            The size of the time step used to define the DCT basis functions. This is typically set to
            the time step between the data collection times of the circuit that this probability trajectory
            is being defined for

        numtimes : int
            The number of data collection times defining the DCT basis functions (defines the total number
            of DCT basis functions: the hyperparameters list is then a subset of this [0,1,2,...,numtimes-1]).
            This is typically set to the number of data collection times for the circuit that this probability
            trajectory is being defined for.

        Returns
        -------
        A new CosineProbTrajectory object.

        """
        self.starttime = starttime
        self.timestep = timestep
        self.numtimes = numtimes

        super(CosineProbTrajectory, self).__init__(outcomes, hyperparameters, parameters)

    def basisfunction(self, i, times):
        """
        The ith Type-II DCT basis function, evaluated at the specified times, where the DCT basis functions
        under consideration are defined by the time parameters set in the initialization.

        The normalization of the functions is such that the max/min of each function is +1/-1.

        Parameters
        ----------
        i : int
            The frequency index of the DCT basis function.

        times : list
            The times to evaluate the basis function at

        Returns
        -------
        array
            The value of the basis function at the specified times.
        """
        return _np.array([_np.cos(i * _np.pi * ((t - self.starttime) / self.timestep + 0.5) / self.numtimes)
                          for t in times])


def _xlogp_rectified(x, p, minp=0.0001, maxp=0.999999):
    """
    Returns x * log(p) where p is bound within (0,1], with adjustments at the boundaries that are useful in
    minimization algorithms.

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
    The negative log-likelihood of a ProbTrajectory, modelling a time-dependent probability distribution.

    Parameters
    ----------
    model : ProbTrajectory
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
    probs = probtrajectory.probabilities(times)
    return probsdict_negloglikelihood(probs, clickstreams, minp, maxp)


def probsdict_negloglikelihood(probs, clickstreams, minp=0., maxp=1.):
    """
    The negative log-likelihood of varying probabilities `probs`, evaluated for the data streams
    in `clickstreams`.

    Parameters
    ----------
    probs : dict
        A dictionary where the keys are the outcome strings in the clickstream (its keys) and the
        value for an outcome is the time-dependent probability list for that outcome, at the times
        associated with the data in the clickstreams.

    clickstreams : dict
        A dictionary where the keys are the different measurement outcomes, and the values are lists
        that give counts for that measurement outcome.

    Returns
    -------
    float
        The negative logi-likelihood of the probability trajectories given the clickstream data.
    """
    logl = 0
    for outcome in clickstreams.keys():
        logl += _np.sum([_xlogp_rectified(xot, pot, minp, maxp) for xot, pot in zip(clickstreams[outcome],
                                                                                    probs[outcome])])

    return -logl


def maxlikelihood(probtrajectory, clickstreams, times, minp=0.0001, maxp=0.999999, method='Nelder-Mead',
                  return_opt_output=False, options={}, verbosity=1):
    """
    Implements maximum likelihood estimation over a model for a time-resolved probabilities trajectory,
    and returns the maximum likelihood model.

    Parameters
    ----------
    model : ProbTrajectory
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

    return_opt_output : bool, optional
        Whether or not to return the output of the optimizer.

    Returns
    -------
    ProbTrajectory
        The maximum likelihood model returned by the optimizer.

    if return_opt_output:
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
    seed = probtrajectory.parameters_as_list()
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

    if return_opt_output:
        return maxlprobtrajectory, optout
    else:
        return maxlprobtrajectory


def amplitude_compression(probtrajectory, times, epsilon=0., verbosity=1):
    """
    Reduces the amplitudes in a CosineProbTrajectory model until the model is valid, i.e., all probabilities
    are within [0, 1]. Also rectifies any of the constant components of the probability trajectories that
    are slightly outside [0, 1].

    Parameters
    ----------
    probtrajectory: CosineProbTrajectory
        The model on which to perform the amplitude reduction

    times: list
        The times at which to enforce the validity of the model (this algorithm does *not* guarantee that the
        probabilities will be within [0, 1] at *all* times in the reals).

    epsilon: float, optional
        The amplitudes are compressed so that all the probabilities are within [0+epsilon,1-epsilon] at all
        times. Setting this to be larger than 0 can be useful as it guarantees that the resultant probability
        trajectory has a non-zero likelihood.

    Returns
    -------
    CosineProbTrajectory
        The new model, that may have had the amplitudes reduced

    Bool
        Whether or not the function did anything non-trivial, i.e, whether any compression was required.

    """
    assert(isinstance(probtrajectory, CosineProbTrajectory)), "Input must be a CosineProbTrajectory!"

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
    probs = probtrajectory.probabilities(times, trim=False)
    params = probtrajectory.parameters_copy()
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
        parameters = _np.reshape(probtrajectory.parameters_as_list(), shape)
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

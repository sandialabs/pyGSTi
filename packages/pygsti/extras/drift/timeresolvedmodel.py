from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

from . import signal as _sig
from . import probabilitytrajectoryestimation as _ptest

import numpy as _np
import time as _tm
import copy as _copy

from scipy.optimize import minimize as _minimize


class TimeResolvedModel(object):
    """
    todo
    """
    def __init__(self, hyperparameters, parameters):
        """
        todo:

        Returns
        -------
        None
        """

        self.hyperparameters = hyperparameters
        self.parameters = parameters

        return None

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def set_parameters(self, parameters):
        self.parameters = parameters

    def get_parameters(self):
        return self.parameters

    def get_probabilities(self, circuit, times):
        """
        todo
        """
        raise NotImplementedError("Derived classes need to implement this!")

    def copy(self):
        return _copy.deepcopy(self)


class TimeResolvedRamseyModel(TimeResolvedModel):
    """
    todo.
    """
    def __init__(self, hyperparameters, parameters):
        """
        todo.

        """
        super().__init(hyperparameters, parameters)
        return

    def get_probabilities(self, circuit, times):
        """
        todo

        """ 
        assert(len(self.parameters) == len(self.hyperparameters) - 4)

        delta = self.parameters[0]
        gamma = self.parameters[1]
        Lambda = self.parameters[2]
        alphas = np.array(self.parameters[3:])

        L = len(gs) - 2
        T = len(times)
        theta = np.zeros(T)
        for omgInd, omg in enumerate(omegas):
            if omg != 0:
                theta += alphas[omgInd] * np.array([np.cos(omg*np.pi*(t-starttime+0.5)/timedif) for t in times])
            else:
                theta += alphas[omgInd]
        p = OutcomeLabelDict() 
        p[('1',)] = delta*np.ones(T) + 0.5  + 0.5*gamma*(Lambda**L)*np.sin(L*theta)
        p[('0',)] = 1. - p[('1',)]
        return p


def negloglikelihood(timeresolvedmodel, times, data, min_p, max_p): 
    """
    The negative loglikelihood for a TimeResolvedModel.

    """
    negll = 0.
    for circuit in data.keys():
        p = timeresolvedmodel.get_probabilities(circuit, times[circuit])
        negll += _ptest.probabilitytrajectoryNegLogLikelihood(p, data[circuit], min_p, max_p)

    return negll


def maxlikelihood_model(timeresolvedmodel, times, data, seed, min_p=1e-4, max_p=1-1e-6, bounds=None, returnOptout=False,
                        verbosity=1):
    """
    Todo.

    """
    maxltimeresolvedmodel = timeresolvedmodel.copy()
    def objfunc(parameters):

        negll = 0.
        for circuit in data.keys():
            maxltimeresolvedmodel.set_parameters(parameters)
            p = maxltimeresolvedmodel.get_probabilities(circuit, times[mdl], parameters)
            negll += _ptest.probabilitytrajectoryNegLogLikelihood(p, data[circuit], min_p, max_p)

        return negll

    options = {'disp':False}
    if verbosity > 0:
        print("- Performing MLE over {} parameters...".format(len(seed)), end='')
    if verbosity > 1:
        print("")
        options = {'disp':True}

    seed = maxltimeresolvedmodel.get_parameter()
    start = _tm.time()
    optout = _minimize(objfunc, seed, options=options, bounds=bounds)
    maxlparameters = optout.x
    end = _tm.time()

    if verbosity == 1:
        print("complete.")
    if verbosity > 0:
        print("- Time taken: {} seconds".format(end - start)),

    if returnOptout:
        return maxltimeresolvedmodel, maxlparameters, optout
    else:
        return maxltimeresolvedmodel

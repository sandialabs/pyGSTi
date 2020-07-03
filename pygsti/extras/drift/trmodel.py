#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

from . import signal as _sig
from . import probtrajectory as _ptraj

import numpy as _np
import time as _tm
import copy as _copy

from scipy.optimize import minimize as _minimize


class TimeResolvedModel(object):
    """
    Encapsulates a basic form of time-resolved model, for implementing simple types of time-resolved characterization,
    e.g., time-resolved Ramsey spectroscopy. This object is a container for specifying a particular time-resolved
    model, which is achieved by defining the method `get_probabilities`. See the docstring of that method for further
    details.

    This object is *not* intended to be used to encapsulate a time-resolved model that requires any intensive
    computations, e.g., a time-resolved process matrix model for full time-resolved GST. Instead, it is intend to be
    used for easy DIY time-resolved tomography on very simple models.

    """

    def __init__(self, hyperparameters, parameters):
        """
        Initializes a TimResolvedModel object.

        Parameters
        ----------
        hyperparameters: list
            A set of meta-parameters, that define the model. For example, these could be frequencies to include in a
            Fourier decomposition.

        parameters: list
           The values for the parameters of the model. For example, these could be the amplitudes for each frequency
           in a Fourier decomposition.

        Returns
        -------
        TimeResolvedModel

        """
        self.hyperparameters = hyperparameters
        self.parameters = parameters

        return None

    def set_parameters(self, parameters):
        """
        Sets the parameters of the model.
        """
        self.parameters = _copy.deepcopy(parameters)

    def get_parameters(self):
        """
        Returns the parameters of the model.
        """
        return _copy.deepcopy(self.parameters)

    def get_probabilities(self, circuit, times):
        """
        *** Specified in each derive class ***

        Specifying this method is the core to building a time-resolved model. This method should return the
        probabiilties for each outcome, for the input circuit at the specified times.

        Parameters
        ----------
        circuit : Circuit
            The circuit to return the probability trajectories for.

        times : list
            The times to calculate the probabilities for.

        Returns
        -------
        dict
            A dictionary where the keys are the possible outcomes of the circuit, and the value
            for an outcome is a list of the probabilities to obtain that outcomes at the specified
            times (so this list is the same length as `times`).

        """
        raise NotImplementedError("Derived classes need to implement this!")

    def copy(self):
        return _copy.deepcopy(self)


def negloglikelihood(trmodel, ds, minp=0, maxp=1):
    """
    The negative loglikelihood for a TimeResolvedModel given the time-series data.

    Parameters
    ----------
    timeresolvedmodel: TimeResolvedModel
        The TimeResolvedModel to calculate the likelihood of.

    ds: DataSet
        A DataSet, containing time-series data.

    minp, maxp: float, optional
        Value used to smooth the 0 and 1 probability boundaries for the likelihood function.
        To get the extact nll, leave as 0 and 1.

    Returns
    -------
    float
        The negative loglikelihood of the model.

    """
    negll = 0.
    for circuit in ds.keys():
        times, clickstreams = ds[circuit].timeseries_for_outcomes
        probs = trmodel.get_probabilities(circuit, times)
        negll += _ptraj.probsdict_negloglikelihood(probs, clickstreams, minp, maxp)

    return negll


def maxlikelihood(trmodel, ds, minp=1e-4, maxp=1 - 1e-6, bounds=None, returnoptout=False,
                  optoptions={}, verbosity=1):
    """
    Finds the maximum likelihood TimeResolvedModel given the data.

    Parameters
    ----------
    timeresolvedmodel: TimeResolvedModel
        The TimeResolvedModel that is used as the seed, and which defines the class of parameterized models to optimize
        over.

    ds: DataSet
        A DataSet, containing time-series data.

    minp, maxp: float, optional
        Value used to smooth the 0 and 1 probability boundaries for the likelihood function.

    bounds: list or None, optional
        Bounds on the parameters, as specified in scipy.optimize.minimize

    optout: bool, optional
        Wether to return the output of scipy.optimize.minimize

    optoptions: dict, optional
        Optional arguments for scipy.optimize.minimize.

    Returns
    -------
    float
        The maximum loglikelihood model

    """
    maxlmodel = trmodel.copy()

    def objfunc(parameters):
        maxlmodel.set_parameters(parameters)
        return negloglikelihood(maxlmodel, ds, minp, maxp)

    if verbosity > 0:
        print("- Performing MLE over {} parameters...".format(len(maxlmodel.get_parameters())), end='')
    if verbosity > 1:
        print("")

    seed = maxlmodel.get_parameters()
    start = _tm.time()
    optout = _minimize(objfunc, seed, options=optoptions, bounds=bounds)
    maxlparameters = optout.x
    maxlmodel.set_parameters(maxlparameters)
    end = _tm.time()

    if verbosity == 1:
        print("complete.")
    if verbosity > 0:
        print("- Time taken: {} seconds".format(end - start)),

    if returnoptout:
        return maxlmodel, optout
    else:
        return maxlmodel

"""
ModelTest Protocol objects
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import time as _time
import os as _os
import numpy as _np
import numpy.random as _rndm
import pickle as _pickle
import collections as _collections
import warnings as _warnings
import scipy.optimize as _spo
from scipy.stats import chi2 as _chi2

from . import protocol as _proto
from .. import objects as _objs
from .. import algorithms as _alg
from .. import construction as _construction
from .. import io as _io
from .. import tools as _tools

from pygsti.protocols.estimate import Estimate as _Estimate
from ..objects import wildcardbudget as _wild
from ..objects.profiler import DummyProfiler as _DummyProfiler
from ..objects import objectivefns as _objfns
from ..objects.circuitlist import CircuitList as _CircuitList
from ..objects.resourceallocation import ResourceAllocation as _ResourceAllocation
from ..objects.objectivefns import ModelDatasetCircuitsStore as _ModelDatasetCircuitStore
from ..construction import datasetconstruction as _dcnst


class StateFunctionDataSimulator(_proto.DataSimulator):
    """
    A data simulator that also computes user-defined functions of the final states.

    Parameters
    ----------
    model : Model
        The model to simulate.

    target_model : Model
        Another model, often containing the perfect target gates, that is simulated
        alongside `model` and gives each element of `state_functions` (see below) a second
        target-state argument.  If `None`, then no target-state is simulated or supplied
        to the state functions.

    state_functions : dict
        A dictionary of callable objects (functions or callable objects) with
        keys that name each quantity.  These objects are called with two or three
        arguments: 1) the circuit that was simulated, 2) a :class:`SPAMVector`
        representing the quantum state from simulating `model`, and 3) the similar
        quantum state from simulating `target_model` (only supplied if `target_model`
        is not `None`.  Each function may return a floating point number, a NumPy array,
        or a dictionary of such values.

    num_samples : int or list of ints or None, optional
        The simulated number of samples for each circuit.  This only has
        effect when  ``sample_error == "binomial"`` or ``"multinomial"``.  If an
        integer, all circuits have this number of total samples. If a list,
        integer elements specify the number of samples for the corresponding
        circuit.  If ``None``, then `model_or_dataset` must be a
        :class:`~pygsti.objects.DataSet`, and total counts are taken from it
        (on a per-circuit basis).

    sample_error : string, optional
        What type of sample error is included in the counts.  Can be:

        - "none"  - no sample error: counts are floating point numbers such
          that the exact probabilty can be found by the ratio of count / total.
        - "clip" - no sample error, but clip probabilities to [0,1] so, e.g.,
          counts are always positive.
        - "round" - same as "clip", except counts are rounded to the nearest
          integer.
        - "binomial" - the number of counts is taken from a binomial
          distribution.  Distribution has parameters p = (clipped) probability
          of the circuit and n = number of samples.  This can only be used
          when there are exactly two SPAM labels in model_or_dataset.
        - "multinomial" - counts are taken from a multinomial distribution.
          Distribution has parameters p_k = (clipped) probability of the gate
          string using the k-th SPAM label and n = number of samples.

    seed : int, optional
        If not ``None``, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    rand_state : numpy.random.RandomState
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.

    alias_dict : dict, optional
        A dictionary mapping single operation labels into tuples of one or more
        other operation labels which translate the given circuits before values
        are computed using `model_or_dataset`.  The resulting Dataset, however,
        contains the *un-translated* circuits as keys.

    collision_action : {"aggregate", "keepseparate"}
        Determines how duplicate circuits are handled by the resulting
        `DataSet`.  Please see the constructor documentation for `DataSet`.

    record_zero_counts : bool, optional
        Whether zero-counts are actually recorded (stored) in the returned
        DataSet.  If False, then zero counts are ignored, except for
        potentially registering new outcome labels.

    times : iterable, optional
        When not None, a list of time-stamps at which data should be sampled.
        `num_samples` samples will be simulated at each time value, meaning that
        each circuit in `circuit_list` will be evaluated with the given time
        value as its *start time*.
    """

    def __init__(self, model, target_model, state_functions, num_samples=1000, sample_error='multinomial',
                 seed=None, rand_state=None, alias_dict=None, collision_action="aggregate",
                 record_zero_counts=True):
        times = None  # assume time independent so logic is simpler below
        super().__init__(model, num_samples, sample_error, seed, rand_state, alias_dict, collision_action,
                         record_zero_counts, times)
        self.state_functions = state_functions
        self.target_model = target_model

    def run(self, edesign, memlimit=None, comm=None):
        """
        Run this data simulator on an experiment design.

        Parameters
        ----------
        edesign : ExperimentDesign
            The input experiment design.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this data
            simulator in parallel.

        Returns
        -------
        ProtocolData
        """
        #This code parallels the simulate_data function somewhat, but we propagate states explicitly
        circuit_list = edesign.all_circuits_needing_data
        if self.alias_dict is not None:
            trans_circuit_list = [_construction.translate_circuit(s, self.alias_dict) for s in circuit_list]
        else:
            trans_circuit_list = circuit_list

        if self.sample_error in ("binomial", "multinomial"):
            if self.rand_state is None:
                rndm = _rndm.RandomState(self.seed)  # ok if seed is None
            else:
                rndm = self.rand_state

        count_lists = _collections.OrderedDict()
        aux_data = {}
        for k, (s, trans_s) in enumerate(zip(circuit_list, trans_circuit_list)):
            ps = self.model.probabilities(trans_s)
            complete_circuit = self.model.complete_circuit(trans_s).layertup

            rho = self.model.circuit_layer_operator(complete_circuit[0], 'prep')
            for layer in complete_circuit[1:-1]:
                layerop = self.model.circuit_layer_operator(layer, 'op')
                rho = layerop.acton(rho)
            args = [s, rho]  # circuit and final state (right before POVM)

            if self.target_model is not None:
                rho = self.target_model.circuit_layer_operator(complete_circuit[0], 'prep')
                for layer in complete_circuit[1:-1]:
                    layerop = self.target_model.circuit_layer_operator(layer, 'op')
                    rho = layerop.acton(rho)
                args.append(rho)  # target state

            aux_data = {}
            for nm, statefn in self.state_functions.items():
                out = statefn(*args)
                if isinstance(out, dict):
                    aux_data.update({"%s.%s" % (nm, out_nm): val for out_nm, val in out.items()})
                else:
                    aux_data[nm] = out

            try:
                N = self.num_samples[k]  # try to treat num_samples as a list
            except:
                N = self.num_samples  # if not indexable, num_samples should be a single number

            counts = _dcnst._sample_distribution(ps, self.sample_error, N, rndm)
            if s not in count_lists: count_lists[s] = []
            count_lists[s].append((counts, aux_data))

        dataset = _objs.DataSet(collision_action=self.collision_action)
        for s, counts_list in count_lists.items():
            for counts_dict, aux_data in counts_list:
                dataset.add_count_dict(s, counts_dict, record_zero_counts=self.record_zero_counts,
                                       update_ol=False, aux=aux_data)
        dataset.update_ol()  # because we set update_ol=False above, we need to do this
        dataset.done_adding_data()
        return _proto.ProtocolData(edesign, dataset)


#class StateFunctions(_proto.Protocol):
#    """
#    A protocol that computes a given set of user-defined functions of the states.
#
#    Parameters
#    ----------
#    model : Model
#        The model to simulate circuits with when :method:`run` is called.
#
#    state_functions : dict
#        A dictionary of callable objects (functions or callable objects) with
#        keys that name each quantity.  These objects are called with a single
#        argument - :class:`SPAMVector` that represents a quantum state - and
#        they may return a floating point number, a NumPy array, or a dictionary
#        of such values.
#
#    verbosity : int, optional
#        Level of detail printed to stdout.
#
#    name : str, optional
#        The name of this protocol, also used to (by default) name the
#        results produced by this protocol.  If None, the class name will
#        be used.
#    """
#
#    def __init__(self, model, state_functions, verbosity=2, name=None):
#
#        super().__init__(name)
#        self.model = model
#        self.state_functions = state_functions
#        self.verbosity = verbosity
#
#        self.auxfile_types['model'] = 'pickle'
#        self.auxfile_types['state_functions'] = 'pickle'
#
#    def run(self, data, memlimit=None, comm=None):
#        """
#        Run this protocol on `data`.
#
#        Parameters
#        ----------
#        data : ProtocolData
#            The input data.
#
#        memlimit : int, optional
#            A rough per-processor memory limit in bytes.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not ``None``, an MPI communicator used to run this protocol
#            in parallel.
#
#        Returns
#        -------
#        ModelEstimateResults
#        """        
#
#        
#        the_model = self.model

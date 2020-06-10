"""
Defines the ForwardSimulator calculator class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import numpy.linalg as _nla
import collections as _collections
import itertools as _itertools
import warnings as _warnings

from ..tools import slicetools as _slct
from ..tools import basistools as _bt
from ..tools import matrixtools as _mt
from ..tools import mpitools as _mpit
from . import spamvec as _sv
from . import operation as _op
from . import labeldicts as _ld
from .resourceallocation import ResourceAllocation as _ResourceAllocation
from .copalayout import CircuitOutcomeProbabilityArrayLayout as _CircuitOutcomeProbabilityArrayLayout
from .circuit import Circuit as _Circuit

# SCRATCH NOTES TODO REMOVE
# ForwardSimulator
#  - user implements _outcome_probs(circuit, outcomes)
#  -> base class extends to probs (are these the same?) -> finite-diff gives derivs, standard COPAlayout gives bulk_fill_probs
#  - if use "simplified circuits" (as all do now), do model -> simplified_model[_helper] (simplifies all instruments and povms, but keeps? paramvec),
#    then pass this to fwdsim?  or can fwdsim somehow query model for simplified ops?


class ForwardSimulator(object):
    """
    A calculator of circuit outcome probability calculations and their derivatives w.r.t. model parameters.

    Some forward simulators may also be used to perform operation-product calculations.

    This functionality exists in a class separate from Model to allow for additional
    model classes (e.g. ones which use entirely different -- non-gate-local
    -- parameterizations of operation matrices and SPAM vectors) access to these
    fundamental operations.  It also allows for the easier addition of new forward simulators.

    Parameters
    ----------
    dim : int
        The model-dimension.  All operations act on a `dim`-dimensional Hilbert-Schmidt space.

    layer_op_server : LayerLizard
        An object that can be queried for circuit-layer operations.

    paramvec : numpy.ndarray
        The current parameter vector of the Model.
    """

    def __init__(self, model=None):
        """
        TODO: docstring
        Construct a new ForwardSimulator object.

        Parameters
        ----------
        dim : int
            The model-dimension.  All operations act on a `dim`-dimensional Hilbert-Schmidt space.

        layer_op_server : LayerLizard
            An object that can be queried for circuit-layer operations.

        paramvec : numpy.ndarray
            The current parameter vector of the Model.
        """
        #self.dim = model.dim
        self._model = model

        #self.paramvec = paramvec
        #self.Np = len(paramvec)
        #self.evotype = layer_op_server.evotype()

        #Conversion of labels -> integers for speed & C-compatibility
        #self.operation_lookup = { lbl:i for i,lbl in enumerate(gates.keys()) }
        #self.prep_lookup = { lbl:i for i,lbl in enumerate(preps.keys()) }
        #self.effect_lookup = { lbl:i for i,lbl in enumerate(effects.keys()) }
        #
        #self.operationreps = { i:self.operations[lbl].torep() for lbl,i in self.operation_lookup.items() }
        #self.prepreps = { lbl:p.torep('prep') for lbl,p in preps.items() }
        #self.effectreps = { lbl:e.torep('effect') for lbl,e in effects.items() }

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    #def to_vector(self):
    #    """
    #    Returns the parameter vector of the associated Model.
    #
    #    Returns
    #    -------
    #    numpy array
    #        The vectorized model parameters.
    #    """
    #    return self.paramvec
    #
    #def from_vector(self, v, close=False, nodirty=False):
    #    """
    #    The inverse of to_vector.
    #
    #    Initializes the Model-like members of this
    #    calculator based on `v`. Used for computing finite-difference derivatives.
    #
    #    Parameters
    #    ----------
    #    v : numpy.ndarray
    #        The parameter vector.
    #
    #    close : bool, optional
    #        Set to `True` if `v` is close to the current parameter vector.
    #        This can make some operations more efficient.
    #
    #    nodirty : bool, optional
    #        If True, the framework for marking and detecting when operations
    #        have changed and a Model's parameter-vector needs to be updated
    #        is disabled.  Disabling this will increases the speed of the call.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    #Note: this *will* initialize the parent Model's objects too,
    #    # since only references to preps, effects, and gates are held
    #    # by the calculator class.  ORDER is important, as elements of
    #    # POVMs and Instruments rely on a fixed from_vector ordering
    #    # of their simplified effects/gates.
    #    self.paramvec = v.copy()  # now self.paramvec is *not* the same as the Model's paramvec
    #    self.sos.from_vector(v, close, nodirty)  # so don't always want ", nodirty=True)" - we
    #    # need to set dirty flags so *parent* will re-init it's paramvec...
    #
    #    #Re-init reps for computation
    #    #self.operationreps = { i:self.operations[lbl].torep() for lbl,i in self.operation_lookup.items() }
    #    #self.operationreps = { lbl:g.torep() for lbl,g in gates.items() }
    #    #self.prepreps = { lbl:p.torep('prep') for lbl,p in preps.items() }
    #    #self.effectreps = { lbl:e.torep('effect') for lbl,e in effects.items() }

    #UNUSED - REMOVE?
    #def propagate(self, state, simplified_circuit, time=None):
    #    """
    #    Propagate a state given a set of operations.
    #    """
    #    raise NotImplementedError()  # TODO - create an interface for running circuits

    def _compute_circuit_outcome_probabilities(self, array_to_fill, circuit, outcomes, resource_alloc, time=None):
        raise NotImplementedError("Derived classes should implement this!")

    def probs(self, circuit, outcomes=None, time=None):
        """
        Construct a dictionary containing the outcome probabilities of `simplified_circuit`
        #TODO: docstrings: simplified_circuit => circuit in routines **below**, similar to this one.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        outcomes : list or tuple
            A sequence of outcomes, which can themselves be either tuples
            (to include intermediate measurements) or simple strings, e.g. `'010'`.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        probs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to probabilities.
        """
        copa_layout = self.create_layout([circuit])
        probs_array = _np.empty(copa_layout.size, 'd')
        if time is None:
            self.bulk_fill_probs(probs_array, copa_layout)
        else:
            self._bulk_fill_probs_at_times(probs_array, copa_layout, [time])

        probs = _ld.OutcomeLabelDict()
        elindices, outcomes = copa_layout.indices_and_outcomes_for_index(0)
        for element_index, outcome in zip(_slct.indices(elindices), outcomes):
            probs[outcome] = probs_array[element_index]
        return probs

    def dprobs(self, circuit, clip_to=None):
        copa_layout = self.create_layout([circuit])
        dprobs_array = _np.empty((copa_layout.size, self.model.num_params()), 'd')
        self.bulk_fill_dprobs(dprobs_array, copa_layout, clip_to)

        dprobs = _ld.OutcomeLabelDict()
        elindices, outcomes = copa_layout.indices_and_outcomes_for_index(0)
        for element_index, outcome in zip(_slct.indices(elindices), outcomes):
            dprobs[outcome] = dprobs_array[element_index]
        return dprobs

    def hprobs(self, circuit, clip_to=None):
        copa_layout = self.create_layout([circuit])
        hprobs_array = _np.empty((copa_layout.size, self.model.num_params(), self.model.num_params()), 'd')
        self.bulk_fill_dprobs(hprobs_array, copa_layout, clip_to)

        hprobs = _ld.OutcomeLabelDict()
        elindices, outcomes = copa_layout.indices_and_outcomes_for_index(0)
        for element_index, outcome in zip(_slct.indices(elindices), outcomes):
            hprobs[outcome] = hprobs_array[element_index]
        return hprobs

    # ---------------------------------------------------------------------------
    # BULK operations -----------------------------------------------------------
    # ---------------------------------------------------------------------------

    def create_layout(self, circuits, dataset=None, resource_alloc=None,
                      array_types=(), derivative_dimensions=None, verbosity=0):
        """
        Constructs an circuit-outcome-probability-array (COPA) layout for `circuits` and `dataset`.

        Parameters
        ----------
        circuits : list
            The circuits whose outcome probabilities should be computed.

        dataset : DataSet
            The source of data counts that will be compared to the circuit outcome
            probabilities.  The computed outcome probabilities are limited to those
            with counts present in `dataset`.

        resource_alloc : ResourceAllocation
            A available resources and allocation information.  These factors influence how
            the layout (evaluation strategy) is constructed.

        Returns
        -------
        CircuitOutcomeProbabilityArrayLayout
        """
        #Note: resource_alloc not even used -- make a slightly more complex "default" strategy?
        return _CircuitOutcomeProbabilityArrayLayout.create_from(circuits, self.model, dataset)

    #TODO UPDATE
    #def bulk_prep_probs(self, eval_tree, comm=None, mem_limit=None):
    #    """
    #    Performs initial computation needed for bulk_fill_probs and related calls.
    #
    #    For example, as computing probability polynomials. This is usually coupled with
    #    the creation of an evaluation tree, but is separated from it because this
    #    "preparation" may use `comm` to distribute a computationally intensive task.
    #
    #    Parameters
    #    ----------
    #    eval_tree : EvalTree
    #        The evaluation tree used to define a list of circuits and hold (cache)
    #        any computed quantities.
    #
    #    comm : mpi4py.MPI.Comm, optional
    #        When not None, an MPI communicator for distributing the computation
    #        across multiple processors.  Distribution is performed over
    #        subtrees of `eval_tree` (if it is split).
    #
    #    mem_limit : int
    #        Rough memory limit in bytes.
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    pass  # default is to have no pre-computed quantities (but not an error to call this fn)

    def bulk_probs(self, circuits, clip_to=None, resource_alloc=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[circuit]` is an ordered dictionary of
            outcome probabilities whose keys are outcome labels.
        """
        if isinstance(circuits, _CircuitOutcomeProbabilityArrayLayout):
            copa_layout = circuits
        else:
            circuits = [c if isinstance(c, _Circuit) else _Circuit(c) for c in circuits]  # cast to Circuits (needed?)
            copa_layout = self.create_layout(circuits, resource_alloc=resource_alloc)

        vp = _np.empty(copa_layout.size, 'd')
        if smartc:
            smartc.cached_compute(self.bulk_fill_probs, vp, copa_layout,
                                  resource_alloc, _filledarrays=(0,))
        else:
            self.bulk_fill_probs(vp, copa_layout, resource_alloc)

        if clip_to is not None:
            vp = _np.clip(vp, clip_to[0], clip_to[1])

        ret = _collections.OrderedDict()
        for elInds, c, outcomes in copa_layout.iter_circuits():
            if isinstance(elInds, slice): elInds = _slct.indices(elInds)
            ret[c] = _ld.OutcomeLabelDict([(outLbl, vp[ei]) for ei, outLbl in zip(elInds, outcomes)])
        return ret

    def bulk_dprobs(self, circuits, clip_to=None, resource_alloc=None, smartc=None,
                    wrt_filter=None, wrt_block_size=None):
        """
        Construct a dictionary containing the probability derivatives for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which parameters
            to include in the derivative dimension. This argument is used
            internally for distributing calculations across multiple
            processors and to control memory usage.  Cannot be specified
            in conjuction with wrt_block_size.

        wrt_block_size : int or float, optional
            The maximum average number of derivative columns to compute values
            for simultaneously.  None means compute all requested columns
            at once.  The  minimum of wrt_block_size and the size that makes
            maximal use of available processors is used as the final block size.
            This argument must be None if wrt_filter is not None.  Set this to
            non-None to reduce amount of intermediate memory required.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that `dprobs[circuit]` is an ordered dictionary of
            derivative arrays (one element per differentiated parameter) whose
            keys are outcome labels
        """
        if isinstance(circuits, _CircuitOutcomeProbabilityArrayLayout):
            copa_layout = circuits
        else:
            copa_layout = self.create_layout(circuits, resource_alloc=resource_alloc)

        #Note: don't use smartc for now.
        vdp = _np.empty((copa_layout.size, self.model.num_params()), 'd')
        self.bulk_fill_dprobs(vdp, copa_layout, clip_to, resource_alloc, wrt_filter, wrt_block_size)

        ret = _collections.OrderedDict()
        for elInds, c, outcomes in copa_layout.iter_circuits():
            if isinstance(elInds, slice): elInds = _slct.indices(elInds)
            ret[c] = _ld.OutcomeLabelDict([(outLbl, vdp[ei]) for ei, outLbl in zip(elInds, outcomes)])
        return ret

    def bulk_hprobs(self, circuits, clip_to=None, resource_alloc=None, smartc=None,
                    wrt_filter1=None, wrt_filter2=None, wrt_block_size1=None, wrt_block_size2=None):
        """
        Construct a dictionary containing the probability Hessians for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        wrt_filter1 : list of ints, optional
            If not None, a list of integers specifying which model parameters
            to differentiate with respect to in the first (row) derivative operations.

        wrt_filter2 : list of ints, optional
            If not None, a list of integers specifying which model parameters
            to differentiate with respect to in the second (col) derivative operations.

        wrt_block_size1: int or float, optional
            The maximum number of 1st (row) derivatives to compute
            values for simultaneously.  None means compute all requested
            rows or columns at once.  The minimum of `wrt_block_size1` and the size
            that makes maximal use of available processors is used as the final
            block size.  This argument must be None if the `wrt_filter1` is not None.
            Set this to non-None to reduce amount of intermediate memory required.

        wrt_block_size2 : int or float, optional
            The maximum number of 2nd (col) derivatives to compute
            values for simultaneously.  None means compute all requested
            rows or columns at once.  The minimum of `wrt_block_size2` and the size
            that makes maximal use of available processors is used as the final
            block size.  This argument must be None if the `wrt_filter2` is not None.
            Set this to non-None to reduce amount of intermediate memory required.

        Returns
        -------
        hprobs : dictionary
            A dictionary such that `hprobs[circuit]` is an ordered dictionary of
            Hessian arrays (a square matrix with one row/column per differentiated
            parameter) whose keys are outcome labels
        """
        if isinstance(circuits, _CircuitOutcomeProbabilityArrayLayout):
            copa_layout = circuits
        else:
            copa_layout = self.create_layout(circuits, resource_alloc=resource_alloc)

        #Note: don't use smartc for now.
        vhp = _np.empty((copa_layout.size, self.model.num_params()), 'd')
        self.bulk_fill_hprobs(vhp, copa_layout, clip_to, resource_alloc,
                              wrt_filter1, wrt_filter1, wrt_block_size1, wrt_block_size2)

        ret = _collections.OrderedDict()
        for elInds, c, outcomes in copa_layout.iter_circuits():
            if isinstance(elInds, slice): elInds = _slct.indices(elInds)
            ret[c] = _ld.OutcomeLabelDict([(outLbl, vhp[ei]) for ei, outLbl in zip(elInds, outcomes)])
        return ret

    def bulk_fill_probs(self, array_to_fill, layout, resource_alloc=None):
        """
        Compute the outcome probabilities for a list circuits.

        This routine fills a 1D array, `array_to_fill` with circuit outcome probabilities
        as dictated by a :class:`CircuitOutcomeProbabilityArrayLayout` ("COPA layout")
        object, which is usually specifically tailored for efficiency.

        The `array_to_fill` array must have length equal to the number of elements in
        `layout`, and the meanings of each element are given by `layout`.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated 1D numpy array of length equal to the
            total number of computed elements (i.e. `len(layout)`).

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        return self._bulk_fill_probs(array_to_fill, layout, resource_alloc)

    def _bulk_fill_probs(self, array_to_fill, layout, resource_alloc):
        return self._bulk_fill_probs_block(array_to_fill, layout, resource_alloc)

    def _bulk_fill_probs_block(self, array_to_fill, layout, resource_alloc):
        for element_indices, circuit, outcomes in layout.iter_circuits():
            self._compute_circuit_outcome_probabilities(array_to_fill[element_indices], circuit,
                                                        outcomes, resource_alloc, time=None)

    def _bulk_fill_probs_at_times(self, array_to_fill, layout, times, resource_alloc):
        # A separate function because computation with time-dependence is often approached differently
        return self._bulk_fill_probs_block_at_time(array_to_fill, layout, times, resource_alloc)

    def _bulk_fill_probs_block_at_times(self, array_to_fill, layout, times, resource_alloc):
        for (element_indices, circuit, outcomes), time in zip(layout.iter_circuits(), times):
            self._compute_circuit_outcome_probabilities(array_to_fill[element_indices], circuit,
                                                        outcomes, resource_alloc, time)

    def bulk_fill_dprobs(self, array_to_fill, layout,
                         pr_array_to_fill=None, resource_alloc=None, wrt_filter=None):
        """
        Compute the outcome probability-derivatives for an entire tree of circuits.

        This routine fills a 2D array, `array_to_fill` with circuit outcome probabilities
        as dictated by a :class:`CircuitOutcomeProbabilityArrayLayout` ("COPA layout")
        object, which is usually specifically tailored for efficiency.

        The `array_to_fill` array must have length equal to the number of elements in
        `layout`, and the meanings of each element are given by `layout`.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated 2D numpy array of shape `(len(layout), Np)`, where
            `Np` is the number of model parameters being differentiated with respect to.

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

        pr_mx_to_fill : numpy array, optional
            when not None, an already-allocated length-`len(layout)` numpy array that is
            filled with probabilities, just as in :method:`bulk_fill_probs`.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which parameters
            to include in the derivative dimension. This argument is used
            internally for distributing calculations across multiple
            processors and to control memory usage.  Cannot be specified
            in conjuction with wrt_block_size.

        Returns
        -------
        None
        """
        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        return self._bulk_fill_dprobs(array_to_fill, layout, pr_array_to_fill, resource_alloc, wrt_filter)

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill, resource_alloc, wrt_filter):
        if wrt_filter is not None:
            wrtSlice = _slct.list_to_slice(wrt_filter)  # for now, require the filter specify a slice
        if pr_array_to_fill is not None:
            self._bulk_fill_probs_block(pr_array_to_fill, layout, resource_alloc)
        return self._bulk_fill_dprobs_block(array_to_fill, None, layout, wrtSlice, resource_alloc)

    def _bulk_fill_dprobs_block(self, array_to_fill, dest_param_slice, layout, param_slice, resource_alloc):
        eps = 1e-7  # hardcoded?
        if param_slice is None:
            param_slice = slice(0, self.model.num_params())
        param_indices = _slct.to_array(param_slice)

        if dest_param_slice is None:
            dest_param_slice = slice(0, len(param_indices))
        dest_param_indices = _slct.to_array(dest_param_slice)

        iParamToFinal = {i: dest_param_indices[ii] for ii, i in enumerate(param_indices)}

        probs = _np.empty(len(layout), 'd')
        self._bulk_fill_probs_block(probs, layout, resource_alloc)

        probs2 = _np.empty(len(layout), 'd')
        orig_vec = self.model.to_vector().copy()
        for i in range(self.model.num_params()):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.model.from_vector(vec, close=True)
                self._bulk_fill_probs_block(probs2, layout, resource_alloc)
                array_to_fill[:, iFinal] = (probs2 - probs) / eps
        self.model.from_vector(orig_vec, close=True)

    def bulk_fill_hprobs(self, array_to_fill, layout,
                         pr_array_to_fill=None, deriv1_array_to_fill=None, deriv2_array_to_fill=None,
                         resource_alloc=None, wrt_filter1=None, wrt_filter2=None):
        """
        Compute the outcome probability-Hessians for an entire list of circuits.

        Similar to `bulk_fill_probs(...)`, but fills a 3D array with
        the Hessians for each circuit outcome probability.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated numpy array of shape `(len(layout),M1,M2)` where
            `M1` and `M2` are the number of selected model parameters (by `wrt_filter1`
            and `wrt_filter2`).

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

        pr_mx_to_fill : numpy array, optional
            when not None, an already-allocated length-`len(layout)` numpy array that is
            filled with probabilities, just as in :method:`bulk_fill_probs`.

        deriv1_array_to_fill : numpy array, optional
            when not None, an already-allocated numpy array of shape `(len(layout),M1)`
            that is filled with probability derivatives, similar to
            :method:`bulk_fill_dprobs` (see `array_to_fill` for a definition of `M1`).

        deriv2_array_to_fill : numpy array, optional
            when not None, an already-allocated numpy array of shape `(len(layout),M2)`
            that is filled with probability derivatives, similar to
            :method:`bulk_fill_dprobs` (see `array_to_fill` for a definition of `M2`).

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        wrt_filter1 : list of ints, optional
            If not None, a list of integers specifying which model parameters
            to differentiate with respect to in the first (row) derivative operations.

        wrt_filter2 : list of ints, optional
            If not None, a list of integers specifying which model parameters
            to differentiate with respect to in the second (col) derivative operations.

        Returns
        -------
        None
        """
        return self._bulk_fill_hprobs(array_to_fill, layout, pr_array_to_fill,
                                      deriv1_array_to_fill, deriv2_array_to_fill,
                                      resource_alloc, wrt_filter1, wrt_filter2)

    def _bulk_fill_hprobs(self, array_to_fill, layout,
                          pr_array_to_fill, deriv1_array_to_fill, deriv2_array_to_fill,
                          resource_alloc, wrt_filter1, wrt_filter2):
        if wrt_filter1 is not None:
            wrtSlice1 = _slct.list_to_slice(wrt_filter1)  # for now, require the filter specify a slice
        if wrt_filter2 is not None:
            wrtSlice2 = _slct.list_to_slice(wrt_filter2)  # for now, require the filter specify a slice

        if pr_array_to_fill is not None:
            self._bulk_fill_probs_block(pr_array_to_fill, layout, resource_alloc)
        if deriv1_array_to_fill is not None:
            self._bulk_fill_dprobs_block(deriv1_array_to_fill, None, layout, wrtSlice1, resource_alloc)
        if deriv2_array_to_fill is not None:
            if wrtSlice1 == wrtSlice2:
                deriv2_array_to_fill[:, :] = deriv1_array_to_fill[:, :]
            else:
                self._bulk_fill_dprobs_block(deriv2_array_to_fill, None, layout, wrtSlice2, resource_alloc)

        return self._bulk_fill_hprobs_block(array_to_fill, None, None, layout, wrtSlice1, wrtSlice2, resource_alloc)

    def _bulk_fill_hprobs_block(self, array_to_fill, dest_param_slice1, dest_param_slice2, layout,
                                param_slice1, param_slice2, resource_alloc):
        eps = 1e-4  # hardcoded?
        if param_slice1 is None: param_slice1 = slice(0, self.model.num_params())
        if param_slice2 is None: param_slice2 = slice(0, self.model.num_params())
        param_indices1 = _slct.to_array(param_slice1)
        param_indices2 = _slct.to_array(param_slice2)

        if dest_param_slice1 is None:
            dest_param_slice1 = slice(0, len(param_indices1))
        if dest_param_slice2 is None:
            dest_param_slice2 = slice(0, len(param_indices2))
        dest_param_indices1 = _slct.to_array(dest_param_slice1)
        #dest_param_indices2 = _slct.to_array(dest_param_slice2)  # unused

        iParamToFinal = {i: dest_param_indices1[ii] for ii, i in enumerate(param_indices1)}

        nP2 = len(param_indices2)
        dprobs = _np.empty((len(layout), nP2), 'd')
        self._bulk_fill_dprobs_block(dprobs, None, layout, param_slice2, resource_alloc)

        dprobs2 = _np.empty((len(layout), nP2), 'd')
        orig_vec = self.model.to_vector().copy()
        for i in range(self.model.num_params()):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.model.from_vector(vec, close=True)
                self._bulk_fill_dprobs_block(dprobs2, None, layout, param_slice2, resource_alloc)
                array_to_fill[:, iFinal, dest_param_slice2] = (dprobs2 - dprobs) / eps
        self.model.from_vector(orig_vec, close=True)

    def bulk_hprobs_by_block(self, layout, wrt_slices_list,
                             return_dprobs_12=False, resource_alloc=None):
        """
        An iterator that computes 2nd derivatives of the `eval_tree`'s circuit probabilities column-by-column.

        This routine can be useful when memory constraints make constructing
        the entire Hessian at once impractical, and one is able to compute
        reduce results from a single column of the Hessian at a time.  For
        example, the Hessian of a function of many gate sequence probabilities
        can often be computed column-by-column from the using the columns of
        the circuits.

        Parameters
        ----------
        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for generated arrays, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

        wrt_slices_list : list
            A list of `(rowSlice,colSlice)` 2-tuples, each of which specify
            a "block" of the Hessian to compute.  Iterating over the output
            of this function iterates over these computed blocks, in the order
            given by `wrt_slices_list`.  `rowSlice` and `colSlice` must by Python
            `slice` objects.

        return_dprobs_12 : boolean, optional
            If true, the generator computes a 2-tuple: (hessian_col, d12_col),
            where d12_col is a column of the matrix d12 defined by:
            d12[iSpamLabel,iOpStr,p1,p2] = dP/d(p1)*dP/d(p2) where P is is
            the probability generated by the sequence and spam label indexed
            by iOpStr and iSpamLabel.  d12 has the same dimensions as the
            Hessian, and turns out to be useful when computing the Hessian
            of functions of the probabilities.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        TODO: docstring - this is outdated!
        block_generator
            A generator which, when iterated, yields the 3-tuple
            `(rowSlice, colSlice, hprobs)` or `(rowSlice, colSlice, dprobs12)`
            (the latter if `return_dprobs_12 == True`).  `rowSlice` and `colSlice`
            are slices directly from `wrt_slices_list`. `hprobs` and `dprobs12` are
            arrays of shape K x S x B x B', where:

            - K is the length of spam_label_rows,
            - S is the number of circuits (i.e. eval_tree.num_final_circuits()),
            - B is the number of parameter rows (the length of rowSlice)
            - B' is the number of parameter columns (the length of colSlice)

            If `mx`, `dp1`, and `dp2` are the outputs of :func:`bulk_fill_hprobs`
            (i.e. args `mx_to_fill`, `deriv1_mx_to_fill`, and `deriv2_mx_to_fill`), then:

            - `hprobs == mx[:,:,rowSlice,colSlice]`
            - `dprobs12 == dp1[:,:,rowSlice,None] * dp2[:,:,None,colSlice]`
        """
        yield from self._bulk_hprobs_by_block(layout, wrt_slices_list, return_dprobs_12, resource_alloc)

    def _bulk_hprobs_by_block(self, layout, wrt_slices_list, return_dprobs_12, resource_alloc):

        nElements = len(layout)

        #NOTE: don't override this method in DistributableForwardSimulator
        # by a method that distributes wrt_slices_list across comm procs,
        # as we assume the user has already done any such distribution
        # and has given each processor a list appropriate for it.
        # Use comm only for speeding up the calcs of the given
        # wrt_slices_list

        for wrtSlice1, wrtSlice2 in wrt_slices_list:

            if return_dprobs_12:
                dprobs1 = _np.zeros((nElements, _slct.length(wrtSlice1)), 'd')
                dprobs2 = _np.zeros((nElements, _slct.length(wrtSlice2)), 'd')
            else:
                dprobs1 = dprobs2 = None
            hprobs = _np.zeros((nElements, _slct.length(wrtSlice1),
                                _slct.length(wrtSlice2)), 'd')

            self.bulk_fill_hprobs(hprobs, layout, None, dprobs1, dprobs2, resource_alloc,
                                  wrt_filter1=_slct.indices(wrtSlice1),
                                  wrt_filter2=_slct.indices(wrtSlice2))

            if return_dprobs_12:
                dprobs12 = dprobs1[:, :, None] * dprobs2[:, None, :]  # (KM,N,1) * (KM,1,N') = (KM,N,N')
                yield wrtSlice1, wrtSlice2, hprobs, dprobs12
            else:
                yield wrtSlice1, wrtSlice2, hprobs

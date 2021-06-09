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
from ..tools import sharedmemtools as _smt
from ..models import labeldicts as _ld
from ..objects.resourceallocation import ResourceAllocation as _ResourceAllocation
from ..layouts.copalayout import CircuitOutcomeProbabilityArrayLayout as _CircuitOutcomeProbabilityArrayLayout
from ..layouts.distlayout import DistributableCOPALayout as _DistributableCOPALayout
from ..layouts.cachedlayout import CachedCOPALayout as _CachedCOPALayout
from ..objects.circuit import Circuit as _Circuit


class ForwardSimulator(object):
    """
    A calculator of circuit outcome probability calculations and their derivatives w.r.t. model parameters.

    Some forward simulators may also be used to perform operation-product calculations.

    This functionality exists in a class separate from Model to allow for additional
    model classes (e.g. ones which use entirely different -- non-gate-local
    -- parameterizations of operation matrices and SPAM vectors) access to these
    fundamental operations.  It also allows for the easier addition of new forward simulators.

    Note: a model holds or "contains" a forward simulator instance to perform its computations,
    and a forward simulator holds a reference to its parent model, so we need to make sure the
    forward simulator doesn't serialize the model or we have a circular reference.

    Parameters
    ----------
    model : Model, optional
        The model this forward simulator will use to compute circuit outcome probabilities.
    """

    @classmethod
    def _array_types_for_method(cls, method_name):
        # The array types of *intermediate* or *returned* values within various class methods (for memory estimates)
        if method_name == 'bulk_probs': return ('E',) + cls._array_types_for_method('bulk_fill_probs')
        if method_name == 'bulk_dprobs': return ('EP',) + cls._array_types_for_method('bulk_fill_dprobs')
        if method_name == 'bulk_hprobs': return ('EPP',) + cls._array_types_for_method('bulk_fill_hprobs')
        if method_name == 'iter_hprobs_by_rectangle': return cls._array_types_for_method('_iter_hprobs_by_rectangle')
        if method_name == '_iter_hprobs_by_rectangle': return ('epp',) + cls._array_types_for_method('bulk_fill_hprobs')
        if method_name == 'bulk_fill_probs': return cls._array_types_for_method('_bulk_fill_probs_block')
        if method_name == 'bulk_fill_dprobs': return cls._array_types_for_method('_bulk_fill_dprobs_block')
        if method_name == 'bulk_fill_hprobs': return cls._array_types_for_method('_bulk_fill_hprobs_block')
        if method_name == '_bulk_fill_probs_block': return ()
        if method_name == '_bulk_fill_dprobs_block':
            return ('e',) + cls._array_types_for_method('_bulk_fill_probs_block')
        if method_name == '_bulk_fill_hprobs_block':
            return ('ep', 'ep') + cls._array_types_for_method('_bulk_fill_dprobs_block')
        return ()

    def __init__(self, model=None):
        #self.dim = model.dim
        self.model = model

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

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        state_dict['_model'] = None  # don't serialize parent model (will cause recursion)
        return state_dict

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val
        evotype = None if val is None else self._model.evotype
        self._set_evotype(evotype)  # alert the class of the evotype (allows loading evotype-specific calc functions)

    def _set_evotype(self, evotype):
        """ Called when the evotype being used (defined by the parent model) changes.
            `evotype` will be `None` when the current model is None"""
        pass

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

    def _compute_sparse_circuit_outcome_probabilities(self, circuit, resource_alloc, time=None):
        raise NotImplementedError("Derived classes should implement this to provide sparse (non-zero) probabilites!")

    def _compute_circuit_outcome_probability_derivatives(self, array_to_fill, circuit, outcomes, param_slice,
                                                         resource_alloc):
        # array to fill has shape (num_outcomes, len(param_slice)) and should be filled with the "w.r.t. param_slice"
        # derivatives of each specified circuit outcome probability.
        raise NotImplementedError("Derived classes can implement this to speed up derivative computation")

    def probs(self, circuit, outcomes=None, time=None, resource_alloc=None):
        """
        Construct a dictionary containing the outcome probabilities for a single circuit.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        outcomes : list or tuple
            A sequence of outcomes, which can themselves be either tuples
            (to include intermediate measurements) or simple strings, e.g. `'010'`.
            If None, only non-zero outcome probabilities will be reported.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        resource_alloc : ResourceAllocation, optional
            The resources available for computing circuit outcome probabilities.

        Returns
        -------
        probs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to probabilities. If no target outcomes provided,
            only non-zero probabilities will be reported.
        """
        if outcomes is None:
            try:
                return self._compute_sparse_circuit_outcome_probabilities(circuit, resource_alloc, time)
            except NotImplementedError:
                pass  # continue on to create full layout and calcualte all outcomes

        copa_layout = self.create_layout([circuit], array_types=('e',), resource_alloc=resource_alloc)
        probs_array = _np.empty(copa_layout.num_elements, 'd')
        if time is None:
            self.bulk_fill_probs(probs_array, copa_layout)
        else:
            self._bulk_fill_probs_at_times(probs_array, copa_layout, [time])

        if _np.any(_np.isnan(probs_array)):
            to_print = str(circuit) if len(circuit) < 10 else str(circuit[0:10]) + " ... (len %d)" % len(circuit)
            _warnings.warn("pr(%s) == nan" % to_print)

        probs = _ld.OutcomeLabelDict()
        elindices, outcomes = copa_layout.indices_and_outcomes_for_index(0)
        for element_index, outcome in zip(_slct.indices(elindices), outcomes):
            probs[outcome] = probs_array[element_index]
        return probs

    def dprobs(self, circuit, resource_alloc=None):
        """
        Construct a dictionary containing outcome probability derivatives for a single circuit.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        resource_alloc : ResourceAllocation, optional
            The resources available for computing circuit outcome probability derivatives.

        Returns
        -------
        dprobs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to an array containing the (partial) derivatives
            of the outcome probability with respect to all model parameters.
        """
        copa_layout = self.create_layout([circuit], array_types=('ep',), resource_alloc=resource_alloc)
        dprobs_array = _np.empty((copa_layout.num_elements, self.model.num_params), 'd')
        self.bulk_fill_dprobs(dprobs_array, copa_layout)

        dprobs = _ld.OutcomeLabelDict()
        elindices, outcomes = copa_layout.indices_and_outcomes_for_index(0)
        for element_index, outcome in zip(_slct.indices(elindices), outcomes):
            dprobs[outcome] = dprobs_array[element_index]
        return dprobs

    def hprobs(self, circuit, resource_alloc=None):
        """
        Construct a dictionary containing outcome probability Hessians for a single circuit.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        resource_alloc : ResourceAllocation, optional
            The resources available for computing circuit outcome probability Hessians.

        Returns
        -------
        hprobs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to a 2D array that is the Hessian matrix for
            the corresponding outcome probability (with respect to all model parameters).
        """
        copa_layout = self.create_layout([circuit], array_types=('epp',), resource_alloc=None)
        hprobs_array = _np.empty((copa_layout.num_elements, self.model.num_params, self.model.num_params), 'd')
        self.bulk_fill_hprobs(hprobs_array, copa_layout)

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

        array_types : tuple, optional
            A tuple of string-valued array types, as given by
            :method:`CircuitOutcomeProbabilityArrayLayout.allocate_local_array`.  These types determine
            what types of arrays we anticipate computing using this layout (and forward simulator).  These
            are used to check available memory against the limit (if it exists) within `resource_alloc`.
            The array types also determine the number of derivatives that this layout is able to compute.
            So, for example, if you ever want to compute derivatives or Hessians of element arrays then
            `array_types` must contain at least one `'ep'` or `'epp'` type, respectively or the layout
            will not allocate needed intermediate storage for derivative-containing types.  If you don't
            care about accurate memory limits, use `('e',)` when you only ever compute probabilities and
            never their derivatives, and `('e','ep')` or `('e','ep','epp')` if you need to compute
            Jacobians or Hessians too.

        derivative_dimensions : tuple, optional
            A tuple containing, optionally, the parameter-space dimension used when taking first
            and second derivatives with respect to the cirucit outcome probabilities.  This must be
            have minimally 1 or 2 elements when `array_types` contains `'ep'` or `'epp'` types,
            respectively.

        verbosity : int or VerbosityPrinter
            Determines how much output to send to stdout.  0 means no output, higher
            integers mean more output.

        Returns
        -------
        CircuitOutcomeProbabilityArrayLayout
        """
        return _CircuitOutcomeProbabilityArrayLayout.create_from(circuits, self.model, dataset, derivative_dimensions,
                                                                 resource_alloc=resource_alloc)

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
            copa_layout = self.create_layout(circuits, array_types=('e',), resource_alloc=resource_alloc)
        global_layout = copa_layout.global_layout

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        with resource_alloc.temporarily_track_memory(global_layout.num_elements):  # 'E' (vp)
            local_vp = copa_layout.allocate_local_array('e', 'd')
            if smartc:
                smartc.cached_compute(self.bulk_fill_probs, local_vp, copa_layout, _filledarrays=(0,))
            else:
                self.bulk_fill_probs(local_vp, copa_layout)
            vp = copa_layout.gather_local_array('e', local_vp)  # gather data onto rank-0 processor
            copa_layout.free_local_array(local_vp)

        if resource_alloc.comm is None or resource_alloc.comm.rank == 0:
            ret = _collections.OrderedDict()
            for elInds, c, outcomes in global_layout.iter_unique_circuits():
                if isinstance(elInds, slice): elInds = _slct.indices(elInds)
                ret[c] = _ld.OutcomeLabelDict([(outLbl, vp[ei]) for ei, outLbl in zip(elInds, outcomes)])
            return ret
        else:
            return None  # on non-root ranks

    def bulk_dprobs(self, circuits, resource_alloc=None, smartc=None):
        """
        Construct a dictionary containing the probability derivatives for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

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
            copa_layout = self.create_layout(circuits, array_types=('ep',), resource_alloc=resource_alloc)
        global_layout = copa_layout.global_layout

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        with resource_alloc.temporarily_track_memory(global_layout.num_elements * self.model.num_params):  # 'EP' (vdp)
            #Note: don't use smartc for now.
            local_vdp = copa_layout.allocate_local_array('ep', 'd')
            self.bulk_fill_dprobs(local_vdp, copa_layout, None)
            vdp = copa_layout.gather_local_array('ep', local_vdp)  # gather data onto rank-0 processor
            copa_layout.free_local_array(local_vdp)

        if resource_alloc.comm_rank == 0:
            ret = _collections.OrderedDict()
            for elInds, c, outcomes in global_layout.iter_unique_circuits():
                if isinstance(elInds, slice): elInds = _slct.indices(elInds)
                ret[c] = _ld.OutcomeLabelDict([(outLbl, vdp[ei]) for ei, outLbl in zip(elInds, outcomes)])
            return ret
        else:
            return None  # on non-root ranks

    def bulk_hprobs(self, circuits, resource_alloc=None, smartc=None):
        """
        Construct a dictionary containing the probability Hessians for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

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
            copa_layout = self.create_layout(circuits, array_types=('epp',), resource_alloc=resource_alloc)
        global_layout = copa_layout.global_layout

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        with resource_alloc.temporarily_track_memory(global_layout.num_elements * self.model.num_params**2):  # 'EPP'
            #Note: don't use smartc for now.
            local_vhp = copa_layout.allocate_local_array('epp', 'd')
            self.bulk_fill_hprobs(local_vhp, copa_layout, None, None, None)
            vhp = copa_layout.gather_local_array('epp', local_vhp)  # gather data onto rank-0 processor
            copa_layout.free_local_array(local_vhp)

        if resource_alloc.comm_rank == 0:
            ret = _collections.OrderedDict()
            for elInds, c, outcomes in global_layout.iter_unique_circuits():
                if isinstance(elInds, slice): elInds = _slct.indices(elInds)
                ret[c] = _ld.OutcomeLabelDict([(outLbl, vhp[ei]) for ei, outLbl in zip(elInds, outcomes)])
            return ret
        else:
            return None  # on non-root ranks

    def bulk_fill_probs(self, array_to_fill, layout):
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

        Returns
        -------
        None
        """
        return self._bulk_fill_probs(array_to_fill, layout)

    def _bulk_fill_probs(self, array_to_fill, layout):
        return self._bulk_fill_probs_block(array_to_fill, layout)

    def _bulk_fill_probs_block(self, array_to_fill, layout):
        for element_indices, circuit, outcomes in layout.iter_unique_circuits():
            self._compute_circuit_outcome_probabilities(array_to_fill[element_indices], circuit,
                                                        outcomes, layout.resource_alloc(), time=None)

    def _bulk_fill_probs_at_times(self, array_to_fill, layout, times):
        # A separate function because computation with time-dependence is often approached differently
        return self._bulk_fill_probs_block_at_times(array_to_fill, layout, times)

    def _bulk_fill_probs_block_at_times(self, array_to_fill, layout, times):
        for (element_indices, circuit, outcomes), time in zip(layout.iter_unique_circuits(), times):
            self._compute_circuit_outcome_probabilities(array_to_fill[element_indices], circuit,
                                                        outcomes, layout.resource_alloc(), time)

    def bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill=None):
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

        Returns
        -------
        None
        """
        return self._bulk_fill_dprobs(array_to_fill, layout, pr_array_to_fill)

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill):
        if pr_array_to_fill is not None:
            self._bulk_fill_probs_block(pr_array_to_fill, layout)
        return self._bulk_fill_dprobs_block(array_to_fill, None, layout, None)

    def _bulk_fill_dprobs_block(self, array_to_fill, dest_param_slice, layout, param_slice):

        #If _compute_circuit_outcome_probability_derivatives is implemented, use it!
        resource_alloc = layout.resource_alloc()
        try:
            for element_indices, circuit, outcomes in layout.iter_unique_circuits():
                self._compute_circuit_outcome_probability_derivatives(
                    array_to_fill[element_indices, dest_param_slice], circuit, outcomes, param_slice, resource_alloc)
            return
        except NotImplementedError:
            pass  # otherwise, proceed to compute derivatives via finite difference.

        eps = 1e-7  # hardcoded?
        if param_slice is None:
            param_slice = slice(0, self.model.num_params)
        param_indices = _slct.to_array(param_slice)

        if dest_param_slice is None:
            dest_param_slice = slice(0, len(param_indices))
        dest_param_indices = _slct.to_array(dest_param_slice)

        iParamToFinal = {i: dest_param_indices[ii] for ii, i in enumerate(param_indices)}

        probs = _np.empty(len(layout), 'd')
        self._bulk_fill_probs_block(probs, layout)

        probs2 = _np.empty(len(layout), 'd')
        orig_vec = self.model.to_vector().copy()
        for i in range(self.model.num_params):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.model.from_vector(vec, close=True)
                self._bulk_fill_probs_block(probs2, layout, resource_alloc)
                array_to_fill[:, iFinal] = (probs2 - probs) / eps
        self.model.from_vector(orig_vec, close=True)

    def bulk_fill_hprobs(self, array_to_fill, layout,
                         pr_array_to_fill=None, deriv1_array_to_fill=None, deriv2_array_to_fill=None):
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

        Returns
        -------
        None
        """
        return self._bulk_fill_hprobs(array_to_fill, layout, pr_array_to_fill,
                                      deriv1_array_to_fill, deriv2_array_to_fill)

    def _bulk_fill_hprobs(self, array_to_fill, layout,
                          pr_array_to_fill, deriv1_array_to_fill, deriv2_array_to_fill):
        if pr_array_to_fill is not None:
            self._bulk_fill_probs_block(pr_array_to_fill, layout)
        if deriv1_array_to_fill is not None:
            self._bulk_fill_dprobs_block(deriv1_array_to_fill, None, layout, None)
        if deriv2_array_to_fill is not None:
            #if wrtSlice1 == wrtSlice2:
            deriv2_array_to_fill[:, :] = deriv1_array_to_fill[:, :]
            #else:
            #    self._bulk_fill_dprobs_block(deriv2_array_to_fill, None, layout, None)

        return self._bulk_fill_hprobs_block(array_to_fill, None, None, layout, None, None)

    def _bulk_fill_hprobs_block(self, array_to_fill, dest_param_slice1, dest_param_slice2, layout,
                                param_slice1, param_slice2):
        eps = 1e-4  # hardcoded?
        if param_slice1 is None: param_slice1 = slice(0, self.model.num_params)
        if param_slice2 is None: param_slice2 = slice(0, self.model.num_params)
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
        self._bulk_fill_dprobs_block(dprobs, None, layout, param_slice2)

        dprobs2 = _np.empty((len(layout), nP2), 'd')
        orig_vec = self.model.to_vector().copy()
        for i in range(self.model.num_params):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.model.from_vector(vec, close=True)
                self._bulk_fill_dprobs_block(dprobs2, None, layout, param_slice2)
                array_to_fill[:, iFinal, dest_param_slice2] = (dprobs2 - dprobs) / eps
        self.model.from_vector(orig_vec, close=True)

    def iter_hprobs_by_rectangle(self, layout, wrt_slices_list,
                                 return_dprobs_12=False):
        """
        Iterates over the 2nd derivatives of a layout's circuit probabilities one rectangle at a time.

        This routine can be useful when memory constraints make constructing
        the entire Hessian at once impractical, and as it only computes a subset of
        the Hessian's rows and colums (a "rectangle") at once.  For example, the
        Hessian of a function of many circuit probabilities can often be computed
        rectangle-by-rectangle and without the need to ever store the entire Hessian at once.

        Parameters
        ----------
        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for generated arrays, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

        wrt_slices_list : list
            A list of `(rowSlice,colSlice)` 2-tuples, each of which specify
            a "rectangle" of the Hessian to compute.  Iterating over the output
            of this function iterates over these computed rectangles, in the order
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

        Returns
        -------
        rectangle_generator
            A generator which, when iterated, yields the 3-tuple
            `(rowSlice, colSlice, hprobs)` or `(rowSlice, colSlice, hprobs, dprobs12)`
            (the latter if `return_dprobs_12 == True`).  `rowSlice` and `colSlice`
            are slices directly from `wrt_slices_list`. `hprobs` and `dprobs12` are
            arrays of shape E x B x B', where:

            - E is the length of layout elements
            - B is the number of parameter rows (the length of rowSlice)
            - B' is the number of parameter columns (the length of colSlice)

            If `mx`, `dp1`, and `dp2` are the outputs of :func:`bulk_fill_hprobs`
            (i.e. args `mx_to_fill`, `deriv1_mx_to_fill`, and `deriv2_mx_to_fill`), then:

            - `hprobs == mx[:,rowSlice,colSlice]`
            - `dprobs12 == dp1[:,rowSlice,None] * dp2[:,None,colSlice]`
        """
        yield from self._iter_hprobs_by_rectangle(layout, wrt_slices_list, return_dprobs_12)

    def _iter_hprobs_by_rectangle(self, layout, wrt_slices_list, return_dprobs_12):
        # under distributed layout each proc already has a local set of parameter slices, and
        # this routine could just compute parts of that piecemeal so we never compute an entire
        # proc's hprobs (may be too large) - so I think this function signature may still be fine,
        # but need to construct wrt_slices_list from global slices assigned to it by the layout.
        # (note the values in the wrt_slices_list must be global param indices - just not all of them)

        #REMOVE  - a distributed layout should be using a DistributablsForwardSimulator that overrides this.
        #if isinstance(layout, _DistributableCOPALayout):  # gather data onto rank-0 processor
        #    nElements = layout.host_num_elements
        #else:

        nElements = len(layout)  # (global number of elements, though "global" isn't really defined)

        #NOTE: don't override this method in DistributableForwardSimulator
        # by a method that distributes wrt_slices_list across comm procs,
        # as we assume the user has already done any such distribution
        # and has given each processor a list appropriate for it.
        # Use comm only for speeding up the calcs of the given
        # wrt_slices_list

        for wrtSlice1, wrtSlice2 in wrt_slices_list:

            if return_dprobs_12:
                dprobs1 = _np.zeros((nElements, _slct.length(wrtSlice1)), 'd')
                self._bulk_fill_dprobs_block(dprobs1, None, layout, wrtSlice1)

                if wrtSlice1 == wrtSlice2:
                    dprobs2 = dprobs1
                else:
                    dprobs2 = _np.zeros((nElements, _slct.length(wrtSlice2)), 'd')
                    self._bulk_fill_dprobs_block(dprobs2, None, layout, wrtSlice2)
            else:
                dprobs1 = dprobs2 = None

            hprobs = _np.zeros((nElements, _slct.length(wrtSlice1), _slct.length(wrtSlice2)), 'd')
            self._bulk_fill_hprobs_block(hprobs, None, None, layout, wrtSlice1, wrtSlice2)

            if return_dprobs_12:
                dprobs12 = dprobs1[:, :, None] * dprobs2[:, None, :]  # (KM,N,1) * (KM,1,N') = (KM,N,N')
                yield wrtSlice1, wrtSlice2, hprobs, dprobs12
            else:
                yield wrtSlice1, wrtSlice2, hprobs


class CacheForwardSimulator(ForwardSimulator):
    """
    A forward simulator that works with non-distributed :class:`CachedCOPALayout` layouts.

    This is just a small addition to :class:`ForwardSimulator`, adding a
    persistent cache passed to new derived-class-overridable compute routines.
    """

    def create_layout(self, circuits, dataset=None, resource_alloc=None,
                      array_types=(), derivative_dimensions=None, verbosity=0):
        """
        Constructs an circuit-outcome-probability-array (COPA) layout for a list of circuits.

        Parameters
        ----------
        circuits : list
            The circuits whose outcome probabilities should be included in the layout.

        dataset : DataSet
            The source of data counts that will be compared to the circuit outcome
            probabilities.  The computed outcome probabilities are limited to those
            with counts present in `dataset`.

        resource_alloc : ResourceAllocation
            A available resources and allocation information.  These factors influence how
            the layout (evaluation strategy) is constructed.

        array_types : tuple, optional
            A tuple of string-valued array types.  See :method:`ForwardSimulator.create_layout`.

        derivative_dimensions : tuple, optional
            A tuple containing, optionally, the parameter-space dimension used when taking first
            and second derivatives with respect to the cirucit outcome probabilities.  This must be
            have minimally 1 or 2 elements when `array_types` contains `'ep'` or `'epp'` types,
            respectively.

        verbosity : int or VerbosityPrinter
            Determines how much output to send to stdout.  0 means no output, higher
            integers mean more output.

        Returns
        -------
        CachedCOPALayout
        """
        #Note: resource_alloc not even used -- make a slightly more complex "default" strategy?
        cache = None  # Derived classes should override this function and create a cache here.
        # A dictionary whose keys are the elements of `circuits` and values can be
        #    whatever the user wants.  These values are provided when calling
        #    :method:`iter_unique_circuits_with_cache`.
        return _CachedCOPALayout.create_from(circuits, self.model, dataset, derivative_dimensions, cache)

    # Override these two functions to plumb `cache` down to _compute* methods
    def _bulk_fill_probs_block(self, array_to_fill, layout):
        for element_indices, circuit, outcomes, cache in layout.iter_unique_circuits_with_cache():
            self._compute_circuit_outcome_probabilities_with_cache(array_to_fill[element_indices], circuit,
                                                                   outcomes, layout.resource_alloc(), cache, time=None)

    def _bulk_fill_dprobs_block(self, array_to_fill, dest_param_slice, layout, param_slice):
        for element_indices, circuit, outcomes, cache in layout.iter_unique_circuits_with_cache():
            self._compute_circuit_outcome_probability_derivatives_with_cache(
                array_to_fill[element_indices, dest_param_slice], circuit, outcomes, param_slice,
                layout.resource_alloc(), cache)

    def _compute_circuit_outcome_probabilities_with_cache(self, array_to_fill, circuit, outcomes, resource_alloc,
                                                          cache, time=None):
        raise NotImplementedError("Derived classes should implement this!")

    def _compute_circuit_outcome_probability_derivatives_with_cache(self, array_to_fill, circuit, outcomes, param_slice,
                                                                    resource_alloc, cache):
        # array to fill has shape (num_outcomes, len(param_slice)) and should be filled with the "w.r.t. param_slice"
        # derivatives of each specified circuit outcome probability.
        raise NotImplementedError("Derived classes can implement this to speed up derivative computation")


def _bytes_for_array_type(array_type, global_elements, max_local_elements, max_atom_size,
                          total_circuits, max_local_circuits,
                          global_num_params, max_local_num_params, max_param_block_size,
                          max_per_processor_cachesize, dim, dtype='d'):
    bytes_per_item = _np.dtype(dtype).itemsize

    size = 1; cur_deriv_dim = 0
    for letter in array_type:
        if letter == 'E': size *= global_elements
        if letter == 'e': size *= max_local_elements
        if letter == 'a': size *= max_atom_size
        if letter == 'C': size *= total_circuits
        if letter == 'c': size *= max_local_circuits
        if letter == 'P':
            size *= global_num_params[cur_deriv_dim]; cur_deriv_dim += 1
        if letter == 'p':
            size *= max_local_num_params[cur_deriv_dim]; cur_deriv_dim += 1
        if letter == 'b':
            size *= max_param_block_size[cur_deriv_dim]; cur_deriv_dim += 1
        if letter == 'z': size *= max_per_processor_cachesize
        if letter == 'd': size *= dim
    return size * bytes_per_item


def _bytes_for_array_types(array_types, global_elements, max_local_elements, max_atom_size,
                           total_circuits, max_local_circuits,
                           global_num_params, max_local_num_params, max_param_block_size,
                           max_per_processor_cachesize, dim, dtype='d'):  # cache is only local to processors
    return sum([_bytes_for_array_type(array_type, global_elements, max_local_elements, max_atom_size,
                                      total_circuits, max_local_circuits,
                                      global_num_params, max_local_num_params, max_param_block_size,
                                      max_per_processor_cachesize, dim, dtype) for array_type in array_types])

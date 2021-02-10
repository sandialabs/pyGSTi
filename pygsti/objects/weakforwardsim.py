"""
Defines the WeakForwardSimulator calculator class
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


class WeakForwardSimulator(object):
    """
    A calculator of circuit outcome calculations (not probabilities, but as frequencies).

    Unlike ForwardSimulator, gradients and Hessians are not immediately accessible.
    However, WeakForwardSimulators can be wrapped as ForwardSimulators (with a finite sample size)
    and gradients and Hessians can be determined by finite difference in that context.
    """

    @classmethod
    def _array_types_for_method(cls, method_name):
        # The array types of *intermediate* or *returned* values within various class methods (for memory estimates)
        if method_name == 'bulk_freqs': return ('E',) + cls._array_types_for_method('bulk_fill_freqs')
        if method_name == 'bulk_fill_freqs': return cls._array_types_for_method('_bulk_fill_freqs_block')
        if method_name == '_bulk_fill_freqs_block': return ()
        return ()

    def __init__(self, model=None):
        """
        Construct a new WeakForwardSimulator object.

        Parameters
        ----------
        model : Model
            Optional parent Model to be stored with the Simulator
        """
        self._model = model

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

    def _compute_circuit_outcome_frequencies(self, array_to_fill, circuit, outcomes, shots, resource_alloc, time=None):
        raise NotImplementedError("Derived classes should implement this!")

    def freqs(self, circuit, shots, outcomes=None, time=None):
        """
        Construct a dictionary containing the outcome frequencies of `circuit`
        #TODO: docstrings: simplified_circuit => circuit in routines **below**, similar to this one.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        shots: int
            Number of times to run each circuit to obtain frequencies

        outcomes : list or tuple
            A sequence of outcomes, which can themselves be either tuples
            (to include intermediate measurements) or simple strings, e.g. `'010'`.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        freqs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to frequencies.
        """
        copa_layout = self.create_layout([circuit])
        freqs_array = _np.empty(copa_layout.num_elements, 'd')
        if time is None:
            self.bulk_fill_freqs(freqs_array, copa_layout, shots)
        else:
            resource_alloc = _ResourceAllocation.cast(None)
            self._bulk_fill_freqs_at_times(freqs_array, copa_layout, [time], resource_alloc, shots)

        if _np.any(_np.isnan(freqs_array)):
            to_print = str(circuit) if len(circuit) < 10 else str(circuit[0:10]) + " ... (len %d)" % len(circuit)
            _warnings.warn("pr(%s) == nan" % to_print)

        freqs = _ld.OutcomeLabelDict()
        elindices, outcomes = copa_layout.indices_and_outcomes_for_index(0)
        for element_index, outcome in zip(_slct.indices(elindices), outcomes):
            freqs[outcome] = freqs_array[element_index]
        return freqs

    # ---------------------------------------------------------------------------
    # BULK operations -----------------------------------------------------------
    # ---------------------------------------------------------------------------

    def create_layout(self, circuits, dataset=None, resource_alloc=None, verbosity=0):
        """
        Constructs an circuit-outcome-probability-array (COPA) layout for `circuits` and `dataset`.

        Parameters
        ----------
        circuits : list
            The circuits whose outcome frequencies should be computed.

        dataset : DataSet
            The source of data counts that will be compared to the circuit outcome
            frequencies.  The computed outcome frequencies are limited to those
            with counts present in `dataset`.

        resource_alloc : ResourceAllocation
            A available resources and allocation information.  These factors influence how
            the layout (evaluation strategy) is constructed.

        Returns
        -------
        CircuitOutcomeProbabilityArrayLayout
        """
        #Note: resource_alloc not even used -- make a slightly more complex "default" strategy?
        ## SS: For WeakForwardSimulator, hard-code None into derivative dimensions?
        ## Also, array_types would only be ('E') if it were used
        ## TODO: Does it make sense to have a COFA Layout analogous to COPA but with shot information?
        ## Maybe it does (i.e. shots can also be parallelized over) - in this case,
        ## shots arg can be collapsed in from function prototypes into the COFALayout (just as circuits are)
        return _CircuitOutcomeProbabilityArrayLayout.create_from(circuits, self.model, dataset, None)

    def bulk_freqs(self, circuits, shots, clip_to=None, resource_alloc=None, smartc=None):
        """
        Construct a dictionary containing the frequencies for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        shots: int
            Number of times to run each circuit to obtain frequencies

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        shots: int
            Number of times to run each circuit to obtain frequencies

        Returns
        -------
        freqs : dictionary
            A dictionary such that `freqs[circuit]` is an ordered dictionary of
            outcome frequencies whose keys are outcome labels.
        """
        if isinstance(circuits, _CircuitOutcomeProbabilityArrayLayout):
            copa_layout = circuits
        else:
            circuits = [c if isinstance(c, _Circuit) else _Circuit(c) for c in circuits]  # cast to Circuits (needed?)
            copa_layout = self.create_layout(circuits, resource_alloc=resource_alloc)

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        with resource_alloc.temporarily_track_memory(copa_layout.num_elements):  # 'E' (vp)
            vp = _np.empty(copa_layout.num_elements, 'd')
            if smartc:
                smartc.cached_compute(self.bulk_fill_freqs, vp, copa_layout, shots,
                                      resource_alloc, _filledarrays=(0,))
            else:
                self.bulk_fill_freqs(vp, copa_layout, shots, resource_alloc)

            if clip_to is not None:
                vp = _np.clip(vp, clip_to[0], clip_to[1])

        ret = _collections.OrderedDict()
        for elInds, c, outcomes in copa_layout.iter_unique_circuits():
            if isinstance(elInds, slice): elInds = _slct.indices(elInds)
            ret[c] = _ld.OutcomeLabelDict([(outLbl, vp[ei]) for ei, outLbl in zip(elInds, outcomes)])
        return ret

    def bulk_fill_freqs(self, array_to_fill, layout, shots, resource_alloc=None):
        """
        Compute the outcome frequencies for a list circuits.

        This routine fills a 1D array, `array_to_fill` with circuit outcome frequencies
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

        shots: int
            Number of times to run each circuit to obtain frequencies

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        return self._bulk_fill_freqs(array_to_fill, layout, shots, resource_alloc)

    def _bulk_fill_freqs(self, array_to_fill, layout, shots, resource_alloc):
        return self._bulk_fill_freqs_block(array_to_fill, layout, shots, resource_alloc)

    def _bulk_fill_freqs_block(self, array_to_fill, layout, shots, resource_alloc):
        for element_indices, circuit, outcomes in layout.iter_unique_circuits():
            self._compute_circuit_outcome_frequencies(array_to_fill[element_indices], circuit,
                                                      outcomes, shots, resource_alloc, time=None)

    def _bulk_fill_freqs_at_times(self, array_to_fill, layout, shots, times, resource_alloc):
        # A separate function because computation with time-dependence is often approached differently
        return self._bulk_fill_freqs_block_at_times(array_to_fill, layout, shots, times, resource_alloc)

    def _bulk_fill_freqs_block_at_times(self, array_to_fill, layout, shots, times, resource_alloc):
        for (element_indices, circuit, outcomes), time in zip(layout.iter_unique_circuits(), times):
            self._compute_circuit_outcome_frequencies(array_to_fill[element_indices], circuit,
                                                      outcomes, shots, resource_alloc, time)

"""
A object representing the indexing into a (flat) array of circuit outcome probabilities.
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
import collections as _collections
import itertools as _it
import copy as _copy
from functools import reduce as _reduce
from operator import add as _add

from .circuitlist import CircuitList as _CircuitList
from .label import Label as _Label
from .circuit import Circuit as _Circuit
from .resourceallocation import ResourceAllocation as _ResourceAllocation
from ..tools import listtools as _lt
from ..tools import slicetools as _slct


class CircuitOutcomeProbabilityArrayLayout(object):
    """
        raw_elabels_dict : collections.OrderedDict
            A dictionary whose keys are simplified circuits (containing just
            "simplified" gates, i.e. not instruments) that include preparation
            labels but no measurement (POVM). Values are lists of simplified
            effect labels, each label corresponds to a single "final element" of
            the computation, e.g. a probability.  The ordering is important - and
            is why this needs to be an ordered dictionary - when the lists of tuples
            are concatenated (by key) the resulting tuple orderings corresponds to
            the final-element axis of an output array that is being filled (computed).
        elIndices : collections.OrderedDict
            A dictionary whose keys are integer indices into `circuits` and
            whose values are slices and/or integer-arrays into the space/axis of
            final elements.  Thus, to get the final elements corresponding to
            `circuits[i]`, use `filledArray[ elIndices[i] ]`.
        outcomes : collections.OrderedDict
            A dictionary whose keys are integer indices into `circuits` and
            whose values are lists of outcome labels (an outcome label is a tuple
            of POVM-effect and/or instrument-element labels).  Thus, to obtain
            what outcomes the i-th operation sequences's final elements
            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.
        nTotElements : int
            The total number of "final elements" - this is how big of an array
            is need to hold all of the probabilities `circuits` generates.
    """

    @classmethod
    def _compute_unique_circuits(cls, circuits):
        first_copy = _collections.OrderedDict(); to_unique = {}
        nUnique = 0
        for i, c in enumerate(circuits):
            if not isinstance(c, _Circuit): c = _Circuit(c)  # ensure all returned circuits are Circuits
            if c not in first_copy:
                first_copy[c] = to_unique[i] = nUnique
                nUnique += 1
            else:
                to_unique[i] = first_copy[c]
        unique_circuits = list(first_copy.keys())  # unique_circuits is in same order as in `circuits`
        return unique_circuits, to_unique

    @classmethod
    def create_from(cls, circuits, model, dataset=None, param_dimensions=()):
        """
        TODO: docstring
        Simplifies a list of :class:`Circuit`s.

        Circuits must be "simplified" before probabilities can be computed for
        them. Each string corresponds to some number of "outcomes", indexed by an
        "outcome label" that is a tuple of POVM-effect or instrument-element
        labels like "0".  Compiling creates maps between operation sequences and their
        outcomes and the structures used in probability computation (see return
        values below).

        Parameters
        ----------
        circuits : list of Circuits
            The list to simplify.

        dataset : DataSet, optional
            If not None, restrict what is simplified to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.
        """
        circuits = circuits if isinstance(circuits, _CircuitList) else _CircuitList(circuits)
        unique_circuits, to_unique = cls._compute_unique_circuits(circuits)
        unique_complete_circuits = [model.complete_circuit(c) for c in unique_circuits]
        ds_circuits = _lt.apply_aliases_to_circuits(unique_circuits, circuits.op_label_aliases)

        # Create a dict of the "present outcomes" of each circuit, defined as those outcomes
        #  for which `dataset` contains data (if `dataset is None` treat *all* outcomes as present).
        #  Note: `circuits` may have duplicates; this is ok: `dataset` doesn't have duplicates so outcomes are the same.
        #  Note2: dict keys are integer unique-circuit indices rather than complete circuits for hashing speed.

        # If we don't need to expand the instruments and POVMs, then just use the outcomes
        # given in the dataset or by the op container.
        if dataset is not None:
            present_outcomes = {i: dataset[ds_c].outcomes for i, ds_c in enumerate(ds_circuits)}
        else:
            present_outcomes = {i: model.circuit_outcomes(c) for i, c in enumerate(unique_circuits)}

        # Step3: create a dictionary of element indices by concatenating the present outcomes of all
        #  the circuits in order.
        elindex_outcome_tuples = _collections.OrderedDict(); k = 0
        for i, c in enumerate(circuits):
            num_outcomes = len(present_outcomes[i])
            elindex_outcome_tuples[i] = tuple([(k + j, outcome) for j, outcome in enumerate(present_outcomes[i])])
            k += num_outcomes

        return cls(circuits, unique_circuits, to_unique, elindex_outcome_tuples, unique_complete_circuits,
                   param_dimensions)

    def __init__(self, circuits, unique_circuits, to_unique, elindex_outcome_tuples,
                 unique_complete_circuits=None, param_dimensions=(), resource_alloc=None):
        # to_unique : dict maping indices of `circuits` to indices of `unique_circuits`
        # elindex_outcome_tuples : dict w/keys == indices into `unique_circuits` (which is why `unique_circuits`
        #                          is needed) and values == lists of (element_index, outcome) pairs.

        self.circuits = circuits if isinstance(circuits, _CircuitList) else _CircuitList(circuits)
        self._unique_circuits = unique_circuits
        self._unique_circuit_index = _collections.OrderedDict(
            [(c, i) for i, c in enumerate(self._unique_circuits)])  # original circuits => unique circuit indices
        self._to_unique = to_unique  # original indices => unique circuit indices
        self._unique_complete_circuits = unique_complete_circuits  # Note: can be None
        self._param_dimensions = param_dimensions
        self._resource_alloc = _ResourceAllocation.cast(resource_alloc)

        max_element_index = max(_it.chain(*[[ei for ei, _ in pairs] for pairs in elindex_outcome_tuples.values()])) \
            if len(elindex_outcome_tuples) > 0 else -1  # -1 makes _size = 0 below
        indices = set(i for tuples in elindex_outcome_tuples.values() for i, o in tuples)
        self._size = max_element_index + 1
        assert(len(indices) == self._size), \
            "Inconsistency: %d distinct indices but max index + 1 is %d!" % (len(indices), self._size)

        self._outcomes = _collections.OrderedDict()
        self._element_indices = _collections.OrderedDict()
        for i_unique, tuples in elindex_outcome_tuples.items():
            sorted_tuples = sorted(tuples, key=lambda x: x[0])  # sort by element index
            elindices, outcomes = zip(*sorted_tuples)  # sorted by elindex so we make slices whenever possible
            self._outcomes[i_unique] = outcomes
            self._element_indices[i_unique] = _slct.list_to_slice(elindices, array_ok=True)

    def __len__(self):
        return self._size  # the number of computed *elements* (!= number of circuits)

    @property
    def num_elements(self):
        return self._size

    @property
    def num_circuits(self):
        return len(self.circuits)

    @property
    def global_layout(self):
        """ The global layout that this layout is or is a part of.  Cannot be comm-dependent. """
        return self  # default is that this object *is* a global layout

    def allocate_local_array(self, array_type, dtype, zero_out=False, memory_tracker=None,
                             extra_elements=0):
        """
        Allocate an array that is distributed according to this layout.

        TODO: docstring - returns the *local* memory and shared mem handle
        """
        # type can be "p", "dp" or "hp"
        nelements = self._size + extra_elements
        alloc_fn = _np.zeros if zero_out else _np.empty
        if array_type == 'e': shape = (nelements,)
        elif array_type == 'ep': shape = (nelements, self._param_dimensions[0])
        elif array_type == 'ep2': shape = (nelements, self._param_dimensions[1])
        elif array_type == 'epp':
            shape = (nelements, self._param_dimensions[0], self._param_dimensions[1])
        elif array_type == 'p': shape = (self._param_dimensions[0],)
        elif array_type == 'jtj': shape = (self._param_dimensions[0], self._param_dimensions[0])
        elif array_type == 'jtf': shape = (self._param_dimensions[0],)
        elif array_type == 'c': shape = (self.num_circuits,)
        else:
            raise ValueError("Invalid `array_type`: %s" % array_type)

        ret = alloc_fn(shape, dtype=dtype)

        if memory_tracker: memory_tracker.add_tracked_memory(ret.size)
        return ret  # local_array

    def free_local_array(self, local_array):
        pass

    def gather_local_array_base(self, array_portion, extra_elements=0, all_gather=False, return_shared=False):
        """
        Gathers an array onto the root processor.
        TODO: docstring (update)

        Gathers the portions of an array that was distributed using this
        layout (i.e. according to the host_element_slice, etc. slices in
        this layout).  Arrays can be 1, 2, or 3-dimensional.  The dimensions
        are understood to be along the "element", "parameter", and
        "2nd parameter" directions in that order.

        Parameters
        ----------
        array_portion : numpy.ndarray
            The portion of the final array that is local to the calling
            processor.  This should be a shared memory array when a
            `resource_alloc` with shared memory enabled was used to construct
            this layout.

        resource_alloc : ResourceAllocation, optional
            The resource allocation object that was used to construt this
            layout, specifying the number and organization of processors
            to distribute arrays among.

        Returns
        -------
        numpy.ndarray or None
            The full (global) output array on the root (rank=0) processor.
            `None` on all other processors.
        """
        if return_shared:
            return array_portion, None  # no shared memory handle
        else:
            return array_portion  # no gathering is performed by this layout class

    def gather_local_array(self, array_type, array_portion, extra_elements=0, return_shared=False):
        """ TODO: docstring """
        return self.gather_local_array_base(array_type, array_portion, extra_elements, False, return_shared)

    def allgather_local_array(self, array_type, array_portion, extra_elements=0, return_shared=False):
        """ TODO: docstring """
        return self.gather_local_array_base(array_type, array_portion, extra_elements, True, return_shared)

    def allsum_local_quantity(self, typ, value, use_shared_mem="auto"):
        return value

    def fill_jtf(self, j, f, jtf):
        """ TODO: docstring """
        jtf[:] = _np.dot(j.T, f)

    def fill_jtj(self, j, jtj):
        """ TODO: docstring """
        jtj[:] = _np.dot(j.T, j)

    def memory_estimate(self, array_type, dtype='d'):
        """
        Memory required to allocate an array (an estimate in bytes).
        """
        bytes_per_element = _np.dtype(dtype).itemsize
        if array_type == "p": return self._size * bytes_per_element
        if array_type == "dp": return self._size * self._param_dimensions[0] * bytes_per_element
        if array_type == "hp": return self._size * self._param_dimensions[0] * \
           self._param_dimensions[1] * bytes_per_element
        raise ValueError("Invalid `array_type`: %s" % array_type)

    def indices(self, circuit):
        return self._element_indices[self._unique_circuit_index[circuit]]

    def outcomes(self, circuit):
        return self._outcomes[self._unique_circuit_index[circuit]]

    def indices_and_outcomes(self, circuit):
        unique_circuit_index = self._unique_circuit_index[circuit]
        return self._element_indices[unique_circuit_index], self._outcomes[unique_circuit_index]

    def indices_for_index(self, index):
        return self._element_indices[self._to_unique[index]]

    def outcomes_for_index(self, index):
        return self._outcomes[self._to_unique[index]]

    def indices_and_outcomes_for_index(self, index):
        unique_circuit_index = self._to_unique[index]
        return self._element_indices[unique_circuit_index], self._outcomes[unique_circuit_index]

    def __iter__(self):
        for circuit, i in self._unique_circuit_index.items():
            for element_index, outcome in zip(self._element_indices[i], self._outcomes[i]):
                yield element_index, circuit, outcome

    def iter_unique_circuits(self):
        for circuit, i in self._unique_circuit_index.items():
            yield self._element_indices[i], circuit, self._outcomes[i]

    def copy(self):
        """
        Create a copy of this layout.

        Returns
        -------
        MatrixCOPALayout
        """
        return _copy.deepcopy(self)  # in the future maybe we need to do something more complicated?

    def resource_alloc(self, sub_alloc_name=None, empty_if_missing=True):
        """
        Retrieves the resource-allocation objectfor this layout.

        Sub-resource-allocations can also be obtained by passing a non-None
        `sub_alloc_name`.

        Parameters
        ----------
        sub_alloc_name : str
            The name to retrieve

        empty_if_missing : bool
            When `True`, an empty resource allocation object is returned when
            `sub_alloc_name` doesn't exist for this layout.  Otherwise a
            `KeyError` is raised when this occurs.

        Returns
        -------
        ResourceAllocation
        """
        if sub_alloc_name is None:
            return self._resource_alloc
        if empty_if_missing:
            return _ResourceAllocation(None)
        raise KeyError("COPA layout has no '%s' resource alloc" % str(sub_alloc_name))

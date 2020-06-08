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
from functools import reduce as _reduce
from operator import add as _add

from .bulkcircuitlist import BulkCircuitList as _BulkCircuitList
from .label import Label as _Label
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
        for i, c in enumerate(circuits):
            if c not in first_copy:
                first_copy[c] = to_unique[i] = i
            else:
                to_unique[i] = first_copy[c]
        unique_circuits = list(first_copy.keys())  # unique_circuits is in same order as in `circuits`
        return unique_circuits, to_unique

    @classmethod
    def create_from(cls, circuits, model_shlp, dataset=None, additional_dimensions=()):
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
        circuits = circuits if isinstance(circuits, _BulkCircuitList) else _BulkCircuitList(circuits)
        unique_circuits, to_unique = cls._compute_unique_circuits(circuits)
        unique_complete_circuits = [model_shlp.complete_circuit(c) for c in unique_circuits]
        ds_circuits = _lt.apply_aliases_to_circuit_list(unique_circuits, circuits.op_label_aliases)

        # Create a dict of the "present outcomes" of each circuit, defined as those outcomes
        #  for which `dataset` contains data (if `dataset is None` treat *all* outcomes as present).
        #  Note: `circuits` may have duplicates; this is ok: `dataset` doesn't have duplicates so outcomes are the same.
        #  Note2: dict keys are integer unique-circuit indices rather than complete circuits for hashing speed.

        #if expand:
        #    # if we're going to expand the instruments and POVMs of the circuits, then we'll store the outcomes
        #    # for each circuit in the order given by this expansion, rather than using the ordering of the dataset.
        #    # (This is more likely to produce "cache-hit" speedups later on.)
        #    if dataset is not None:
        #        excircuit_outcomes_by_indx = {i: c.expand_instruments_and_povms(model_shlp, dataset[ds_c].outcomes)
        #                                      for i, (c, ds_c) in enumerate(zip(unique_circuits, ds_circuits))}
        #    else:
        #        excircuit_outcomes_by_indx = {i: c.expand_instruments_and_povms(model_shlp, None)
        #                                      for i, c in enumerate(unique_circuits)}
        #
        #    expanded_present_outcomes = _collections.OrderedDict()  # keys = expanded circuits, vals = list of outcomes
        #    for i, expanded_dict in excircuit_outcomes_by_indx.items():
        #        expanded_present_outcomes.update(expanded_dict)
        #
        #    #NEEDED?
        #    #expanded_element_indices = _collections.OrderedDict(); k = 0
        #    #for expanded_circuit, outcomes in expanded_present_outcomes.items():
        #    #    num_outcomes = len(expanded_present_outcomes[i])
        #    #    expanded_element_indices[expanded_circuit] = slice(k, k + num_outcomes)
        #    #    k += num_outcomes
        #
        #    present_outcomes = {i: _reduce(_add, exdict.values()) for i, exdict in excircuit_outcomes_by_indx.items()}
        #else:

        # If we don't need to expand the instruments and POVMs, then just use the outcomes
        # given in the dataset or by the op container.
        if dataset is not None:
            present_outcomes = {i: dataset[ds_c].outcomes for i, ds_c in enumerate(ds_circuits)}
        else:
            present_outcomes = {i: model_shlp.circuit_outcomes(c) for i, c in enumerate(unique_circuits)}

        # Step3: create a dictionary of element indices by concatenating the present outcomes of all
        #  the circuits in order.
        elindex_outcome_tuples = _collections.OrderedDict(); k = 0
        for i, c in enumerate(circuits):
            num_outcomes = len(present_outcomes[i])
            elindex_outcome_tuples[i] = tuple([(k + j, outcome) for j, outcome in enumerate(present_outcomes[i])])
            k += num_outcomes

        return cls(circuits, unique_circuits, to_unique, elindex_outcome_tuples, unique_complete_circuits,
                   additional_dimensions)

    def __init__(self, circuits, unique_circuits, to_unique, elindex_outcome_tuples,
                 unique_complete_circuits=None, additional_dimensions=()):
        # to_unique : dict maping indices of `circuits` to indices of `unique_circuits`
        # elindex_outcome_tuples : dict w/keys == indices into `unique_circuits` (which is why `unique_circuits`
        #                          is needed) and values == lists of (element_index, outcome) pairs.

        self.circuits = circuits if isinstance(circuits, _BulkCircuitList) else _BulkCircuitList(circuits)
        self._unique_circuits = unique_circuits
        self._unique_circuit_index = _collections.OrderedDict(
            [(c, i) for i, c in enumerate(self._unique_circuits)])  # original circuits => unique circuit indices
        self._to_unique = to_unique  # original indices => unique circuit indices
        self._size = sum(map(len, elindex_outcome_tuples.values()))  # total number of elements
        self._unique_complete_circuits = unique_complete_circuits  # Note: can be None
        self._additional_dimensions = additional_dimensions

        max_element_index = max(_it.chain(*[tup[0] for tup in elindex_outcome_tuples.values()]))
        assert(self._size == max_element_index + 1), \
            f"Inconsistency: {self._size} elements but max index is {max_element_index}!"

        self._outcomes = _collections.OrderedDict()
        self._element_indices = _collections.OrderedDict()
        for i_unique, tuples in elindex_outcome_tuples.items():
            sorted_tuples = sorted(tuples, key=lambda x: x[0])  # sort by element index
            elindices, outcomes = zip(*sorted_tuples)  # sorted by elindex so we make slices whenever possible
            self._outcomes[i_unique] = outcomes
            self._element_indices[i_unique] = _slct.list_to_slice(elindices, array_ok=True)

    def __len__(self):
        return self._size

    @property
    def size(self):
        return self._size

    @property
    def num_circuits(self):
        return len(self.circuits)

    def allocate_array(self, array_type, zero_out=False, dtype='d'):
        # type can be "p", "dp" or "hp"
        alloc_fn = _np.zeros if zero_out else _np.empty
        if array_type == "p": return alloc_fn((self._size,), dtype=dtype)
        if array_type == "dp": return alloc_fn((self._size, self._additional_dimensions[0]), dtype=dtype)
        if array_type == "hp": return alloc_fn((self._size, self._additional_dimensions[0],
                                                self._additional_dimensions[1]), dtype=dtype)
        raise ValueError(f"Invalid `array_type`: {array_type}")

    def memory_estimate(self, array_type, dtype='d'):
        """
        Memory required to allocate an array (an estimate in bytes).
        """
        bytes_per_element = _np.dtype(dtype).itemsize
        if array_type == "p": return self._size * bytes_per_element
        if array_type == "dp": return self._size * self._additional_dimensions[0] * bytes_per_element
        if array_type == "hp": return self._size * self._additional_dimensions[0] * \
           self._additional_dimensions[1] * bytes_per_element
        raise ValueError(f"Invalid `array_type`: {array_type}")

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

    def iter_circuits(self):
        for circuit, i in self._unique_circuit_index.items():
            yield self._element_indices[i], circuit, self._outcomes[i]

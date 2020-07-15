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

from .copalayout import CircuitOutcomeProbabilityArrayLayout as _CircuitOutcomeProbabilityArrayLayout


#TODO: make this a distributable layout in the FUTURE?
class CachedCOPALayout(_CircuitOutcomeProbabilityArrayLayout):
    """
    A COPA Layout that includes a generic per-circuit cache dictionary.

    A commonly useful addition to the base COPA layout is to add a persistent
    "cache" dictionary that tags along with each circuit, so that parts of
    computations that depend *only* on the circuit (and its outcomes) can be
    computed just once.
    """

    @classmethod
    def create_from(cls, circuits, model, dataset=None, additional_dimensions=(), cache=None):
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

        cache : dict
            A dictionary whose keys are the elements of `circuits` and values can be
            whatever the user wants.  These values are provided when calling
            :method:`iter_unique_circuits_with_cache`.
        """
        if cache is None: cache = {}
        ret = super().create_from(circuits, model, dataset, additional_dimensions)
        ret._cache = {i: cache.get(c, None) for c, i in ret._unique_circuit_index.items()}
        return ret

    def __init__(self, circuits, unique_circuits, to_unique, elindex_outcome_tuples,
                 unique_complete_circuits=None, additional_dimensions=(), cache=None):
        super().__init__(circuits, unique_circuits, to_unique, elindex_outcome_tuples,
                         unique_complete_circuits, additional_dimensions)
        if cache is None: cache = {}
        self._cache = {i: cache.get(c, None) for c, i in self._unique_circuit_index.items()}

    def iter_unique_circuits_with_cache(self):
        """Includes a persistent per-circuit cache dictionary to hold metadata """
        for circuit, i in self._unique_circuit_index.items():
            yield self._element_indices[i], circuit, self._outcomes[i], self._cache[i]

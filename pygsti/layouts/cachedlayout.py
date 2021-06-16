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

from pygsti.layouts.copalayout import CircuitOutcomeProbabilityArrayLayout as _CircuitOutcomeProbabilityArrayLayout


#TODO: make this a distributable layout in the FUTURE?
class CachedCOPALayout(_CircuitOutcomeProbabilityArrayLayout):
    """
    A COPA Layout that includes a generic per-circuit cache dictionary.

    A commonly useful addition to the base COPA layout is to add a persistent
    "cache" dictionary that tags along with each circuit, so that parts of
    computations that depend *only* on the circuit (and its outcomes) can be
    computed just once.

    Parameters
    ----------
    circuits : list of Circuits
        The circuits whose outcome probabilities are to be computed.  This list may
        contain duplicates.

    unique_circuits : list of Circuits
        The same as `circuits`, except duplicates are removed.  Often this value is obtained
        by a derived class calling the class method :method:`_compute_unique_circuits`.

    to_unique : dict
        A mapping that translates an index into `circuits` to one into `unique_circuits`.
        Keys are the integers 0 to `len(circuits)` and values are indices into `unique_circuits`.

    elindex_outcome_tuples : collections.OrderedDict
        A dictionary whose keys are integer indices into `unique_circuits` and
        whose values are lists of `(element_index, outcome_label)` tuples that
        give the element index within the 1D array of the probability (or other quantity)
        corresponding to the given circuit and outcome label.  Note that outcome labels
        themselves are tuples of instrument/POVM member labels.

    unique_complete_circuits : list, optional
        A list, parallel to `unique_circuits`, that contains the "complete" version of these
        circuits.  This information is currently unused, and is included for potential future
        expansion and flexibility.

    param_dimensions : tuple, optional
        A tuple containing, optionally, the parameter-space dimension used when taking first
        and second derivatives with respect to the circuit outcome probabilities.  This is
        meta-data bundled along with the main layout information, and is needed for allocating
        arrays with derivative dimensions.

    resource_alloc : ResourceAllocation, optional
        The resources available for computing circuit outcome probabilities.

    cache : dict
        The cache dictionary for this layout.  Its keys are the elements of `circuits` and
        its values can be whatever the user wants.  These values are provided when calling
        :method:`iter_unique_circuits_with_cache`, so that a forward simulator using this
        layout can cache arbitrary precomputed information within the layout.
    """

    @classmethod
    def create_from(cls, circuits, model, dataset=None, param_dimensions=(), resource_alloc=None, cache=None):
        """
        Creates a :class:`CachedCOPALayout` from a list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The circuits to include in the layout.  Note that the produced layout may not
            retain the ordering of these circuits internally, but that it's `.global_layout`
            does.

        model : Model, optional
            A model used to "complete" the circuits (add implied prep and/or POVM layers).
            Usually this is a/the model that will be used to compute outcomes probabilities
            using this layout.  If `None`, then each element of `circuits` is assumed to
            be a complete circuit, i.e., to begin with a state preparation layer and end
            with a POVM layer.

        dataset : DataSet, optional
            If not None, restrict what is simplified to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.

        param_dimensions : tuple, optional
            A tuple containing, optionally, the parameter-space dimension used when taking first
            and second derivatives with respect to the circuit outcome probabilities.

        resource_alloc : ResourceAllocation, optional
            The resources available for computing circuit outcome probabilities.

        cache : dict
            A dictionary whose keys are the elements of `circuits` and values can be
            whatever the user wants.  These values are provided when calling
            :method:`iter_unique_circuits_with_cache`.
        """
        if cache is None: cache = {}
        ret = super().create_from(circuits, model, dataset, param_dimensions)
        ret._cache = {i: cache.get(c, None) for c, i in ret._unique_circuit_index.items()}
        return ret

    def __init__(self, circuits, unique_circuits, to_unique, elindex_outcome_tuples,
                 unique_complete_circuits=None, param_dimensions=(), resource_alloc=None, cache=None):
        super().__init__(circuits, unique_circuits, to_unique, elindex_outcome_tuples,
                         unique_complete_circuits, param_dimensions, resource_alloc)
        if cache is None: cache = {}
        self._cache = {i: cache.get(c, None) for c, i in self._unique_circuit_index.items()}

    def iter_unique_circuits_with_cache(self):
        """
        Iterate over the element-indices, circuit, outcomes, and cache of each unique circuit in this layout.

        A generator used to iterate over a `(element_indices, circuit, outcomes, cache)` tuple
        for each *unique* circuit held by this layout, where `element_indices` and `outcomes`
        are the values that would be retrieved by the :method:`indices` and :method:`outcomes`
        methods, `circuit` is the unique circuit itself, and `cache` is the user-defined value
        of the cache-dictionary entry for this circuit..

        Returns
        -------
        generator
        """
        for circuit, i in self._unique_circuit_index.items():
            yield self._element_indices[i], circuit, self._outcomes[i], self._cache[i]

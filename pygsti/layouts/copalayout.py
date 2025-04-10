"""
A object representing the indexing into a (flat) array of circuit outcome probabilities.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import copy as _copy
import itertools as _it

import numpy as _np

from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.circuits.circuitlist import CircuitList as _CircuitList
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.tools import listtools as _lt
from pygsti.tools import slicetools as _slct


class CircuitOutcomeProbabilityArrayLayout(_NicelySerializable):
    """
    The arrangement of circuit outcome probabilities into a 1D array.

    This class describes how the outcome probabilities for a list of circuits
    map to the elements of a one-dimensional array.  Computation, e.g., of an
    objective function such as the log-likelihood, is performed using the 1D
    array, and the layout is then used to assign meaning to each of the array
    elements, i.e., to lookup which element corresponds to a given circuit and
    outcome.

    This could be a simple concatenation of all the possible outcomes for each
    circuit in turn.  However, when not all outcomes are observed it is unnecessary
    to compute the corresponding probabilities, and so this layout can be "sparse"
    in this sense.

    When there are multiple processors, a layout may assign different outcome
    probabilities (and their derivatives) to different processors.  Thus, a layout
    can be dependent on the available processors and holds (and "owns") a
    :class:`ResourceAllocation` object.  This class creates a non-distributed
    layout that is simply duplicated across all the available processors.

    Parameters
    ----------
    circuits : list of Circuits
        The circuits whose outcome probabilities are to be computed.  This list may
        contain duplicates.

    unique_circuits : list of Circuits
        The same as `circuits`, except duplicates are removed.  Often this value is obtained
        by a derived class calling the class method :meth:`_compute_unique_circuits`.

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

    Attributes
    ----------
    num_elements : int
        The total number of elements in this layout.  In a multi-processor context,
        the number of elements locally owned by the current processor.

    num_elements : int
        The total number of circuits in this layout.  In a multi-processor context,
        the number of circuits locally owned by the current processor.

    global_layout : CircuitOutcomeProbabilityArrayLayout
        A layout containing all the circuits in their original order, that is the
        same on all processors and doesn't depend on a specific resource allocation.
        This is either the layout itself or a larger layout that this layout is a part of.
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
    def create_from(cls, circuits, model=None, dataset=None, param_dimensions=(), resource_alloc=None):
        """
        Creates a simple layout from a list of circuits.

        Optionally, a model can be used to "complete" (add implied prep or POVM layers)
        circuits, and a dataset to restrict the layout's elements to the observed outcomes.

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

        Returns
        -------
        CircuitOutcomeProbabilityArrayLayout
        """
        circuits = circuits if isinstance(circuits, _CircuitList) else _CircuitList(circuits)
        unique_circuits, to_unique = cls._compute_unique_circuits(circuits)
        unique_complete_circuits = [model.complete_circuit(c) for c in unique_circuits] \
            if (model is not None) else unique_circuits[:]
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
                   param_dimensions, resource_alloc)

    def __init__(self, circuits, unique_circuits, to_unique, elindex_outcome_tuples,
                 unique_complete_circuits=None, param_dimensions=(), resource_alloc=None):
        # See class definitiion for docstring -- usually this is only called by derived classes; see `create_from` above
        # to_unique : dict maping indices of `circuits` to indices of `unique_circuits`
        # elindex_outcome_tuples : dict w/keys == indices into `unique_circuits` (which is why `unique_circuits`
        #                          is needed) and values == lists of (element_index, outcome) pairs.

        super().__init__()
        self.circuits = circuits if isinstance(circuits, _CircuitList) else _CircuitList(circuits)
        if unique_circuits is None and to_unique is None:
            unique_circuits, to_unique = self._compute_unique_circuits(circuits)
        self._unique_circuits = unique_circuits
        self._unique_circuit_index = {c:i for i, c in enumerate(self._unique_circuits)}  # original circuits => unique circuit indices
        self._to_unique = to_unique  # original indices => unique circuit indices
        self._unique_complete_circuits = unique_complete_circuits  # Note: can be None
        self._param_dimensions = param_dimensions
        self._resource_alloc = _ResourceAllocation.cast(resource_alloc)

        indices = [i for tuples in elindex_outcome_tuples.values() for i, _ in tuples]
        max_element_index = max(indices) if len(elindex_outcome_tuples) > 0 else -1  # -1 makes _size = 0 below
        indices = set(indices)
        
        
        self._size = max_element_index + 1
        assert(len(indices) == self._size), \
            "Inconsistency: %d distinct indices but max index + 1 is %d!" % (len(indices), self._size)

        self._outcomes = dict()
        self._element_indices = dict()
        sort_idx_func = lambda x: x[0]
        for i_unique, tuples in elindex_outcome_tuples.items():
            sorted_tuples = sorted(tuples, key=sort_idx_func)  # sort by element index
            elindices, outcomes = zip(*sorted_tuples)  # sorted by elindex so we make slices whenever possible
            self._outcomes[i_unique] = tuple(outcomes)
            self._element_indices[i_unique] = _slct.list_to_slice(elindices, array_ok=True)

#    def hotswap_circuits(self, circuits, unique_complete_circuits=None):
#        self.circuits = circuits if isinstance(circuits, _CircuitList) else _CircuitList(circuits)
#        unique_circuits_dict = {}
#        for orig_i, unique_i in self._to_unique.items():
#            unique_circuits_dict[unique_i] = self.circuits[orig_i]
#        self._unique_circuits = [unique_circuits_dict[i] for i in range(len(unique_circuits_dict))]
#        self._unique_circuit_index = _collections.OrderedDict(
#            [(c, i) for i, c in enumerate(self._unique_circuits)])  # original circuits => unique circuit indices
#        self._unique_complete_circuits = unique_complete_circuits  # Note: can be None

    def _to_nice_serialization(self):
        elindex_outcome_tuples = []
        for i_unique, outcomes in self._outcomes.items():
            elindices = _slct.to_array(self._element_indices[i_unique])
            assert(len(outcomes) == len(elindices))
            elindex_outcome_tuples.append((i_unique, list(zip(map(int, elindices), outcomes))))
            # Note: map to int above to avoid int64 integers which aren't JSON-able

        state = super()._to_nice_serialization()
        state.update({'circuits': self.circuits.to_nice_serialization(),  # a CircuitList
                      'unique_circuits': [c.str for c in self._unique_circuits],
                      'to_unique': [(k, v) for k, v in self._to_unique.items()],  # just a dict mapping ints -> ints
                      'elindex_outcome_tuples': elindex_outcome_tuples,
                      'parameter_dimensions': self._param_dimensions,
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        from pygsti.io import stdinput as _stdinput
        std = _stdinput.StdInputParser()

        circuits = _CircuitList.from_nice_serialization(state['circuits'])
        unique_circuits = [std.parse_circuit(s, create_subcircuits=_Circuit.default_expand_subcircuits)
                           for s in state['unique_circuits']]
        to_unique = {k: v for k, v in state['to_unique']}
        elindex_outcome_tuples = _collections.OrderedDict(state['elindex_outcome_tuples'])

        return cls(circuits, unique_circuits, to_unique, elindex_outcome_tuples,
                   unique_complete_circuits=None, param_dimensions=state['parameter_dimensions'],
                   resource_alloc=None)

    def __len__(self):
        return self._size  # the number of computed *elements* (!= number of circuits)

    @property
    def num_elements(self):
        """
        The total number of elements in this layout.  In a multi-processor context,
        the number of elements locally owned by the current processor.
        """
        return self._size

    @property
    def num_circuits(self):
        """
        The total number of circuits in this layout.  In a multi-processor context,
        the number of circuits locally owned by the current processor.
        """
        return len(self.circuits)

    @property
    def global_layout(self):
        """
        A layout containing all the circuits in their original order, that is the
        same on all processors and doesn't depend on a specific resource allocation.
        This is either the layout itself or a larger layout that this layout is a part of.
        """
        return self  # default is that this object *is* a global layout

    def allocate_local_array(self, array_type, dtype, zero_out=False, memory_tracker=None,
                             extra_elements=0):
        """
        Allocate an array that is distributed according to this layout.

        Creates an array for holding elements and/or derivatives with respect
        to model parameters, possibly distributed among multiple processors
        as dictated by this layout.

        Parameters
        ----------
        array_type : {"e", "ep", "ep2", "epp", "p", "jtj", "jtf", "c", "cp", "cp2", "cpp"}
            The type of array to allocate, often corresponding to the array shape.  Let
            `nE` be the layout's number of elements, `nP1` and `nP2` be the number of
            parameters we differentiate with respect to (for first and second derivatives),
            and `nC` be the number of circuits.  Then the array types designate the
            following array shapes:
            - `"e"`: (nE,)
            - `"ep"`: (nE, nP1)
            - `"ep2"`: (nE, nP2)
            - `"epp"`: (nE, nP1, nP2)
            - `"p"`: (nP1,)
            - `"jtj"`: (nP1, nP2)
            - `"jtf"`: (nP1,)
            - `"c"`: (nC,)
            - `"cp"`: (nC, nP1)
            - `"cp2"`: (nC, nP2)
            - `"cpp"`: (nC, nP1, nP2)
            Note that, even though the `"p"` and `"jtf"` types are the same shape
            they are used for different purposes and are distributed differently
            when there are multiple processors.  The `"p"` type is for use with
            other element-dimentions-containing arrays, whereas the `"jtf"` type
            assumes that the element dimension has already been summed over.

        dtype : numpy.dtype
            The NumPy data type for the array.

        zero_out : bool, optional
            Whether the array should be zeroed out initially.

        memory_tracker : ResourceAllocation, optional
            If not None, the amount of memory being allocated is added, using
            :meth:`add_tracked_memory` to this resource allocation object.

        extra_elements : int, optional
            The number of additional "extra" elements to append to the element
            dimension, beyond those called for by this layout.  Such additional
            elements are used to store penalty terms that are treated by the
            objective function just like usual outcome-probability-type terms.

        Returns
        -------
        numpy.ndarray
        """
        # type can be "p", "dp" or "hp"
        nelements = self._size + extra_elements
        ncircuits = self.num_circuits + extra_elements
        alloc_fn = _np.zeros if zero_out else _np.empty
        if array_type == 'e': shape = (nelements,)
        elif array_type == 'ep': shape = (nelements, self._param_dimensions[0])
        elif array_type == 'ep2': shape = (nelements, self._param_dimensions[1])
        elif array_type == 'epp':
            shape = (nelements, self._param_dimensions[0], self._param_dimensions[1])
        elif array_type == 'p': shape = (self._param_dimensions[0],)
        elif array_type == 'jtj': shape = (self._param_dimensions[0], self._param_dimensions[0])
        elif array_type == 'jtf': shape = (self._param_dimensions[0],)
        elif array_type == 'c': shape = (ncircuits,)
        elif array_type == 'cp': shape = (ncircuits, self._param_dimensions[0])
        elif array_type == 'cp2': shape = (ncircuits, self._param_dimensions[1])
        elif array_type == 'cpp':
            shape = (ncircuits, self._param_dimensions[0], self._param_dimensions[1])
        else:
            raise ValueError("Invalid `array_type`: %s" % array_type)

        ret = alloc_fn(shape, dtype=dtype)

        if memory_tracker: memory_tracker.add_tracked_memory(ret.size)
        return ret  # local_array

    def free_local_array(self, local_array):
        """
        Frees an array allocated by :meth:`allocate_local_array`.

        This method should always be paired with a call to
        :meth:`allocate_local_array`, since the allocated array
        may utilize shared memory, which must be explicitly de-allocated.

        Parameters
        ----------
        local_array : numpy.ndarray or LocalNumpyArray
            The array to free, as returned from `allocate_local_array`.

        Returns
        -------
        None
        """
        pass

    def gather_local_array_base(self, array_type, array_portion, extra_elements=0,
                                all_gather=False, return_shared=False):
        """
        Gathers an array onto the root processor or all the processors.

        Gathers the portions of an array that was distributed using this
        layout (i.e. according to the host_element_slice, etc. slices in
        this layout).  This could be an array allocated by :meth:`allocate_local_array`
        but need not be, as this routine does not require that `array_portion` be
        shared.  Arrays can be 1, 2, or 3-dimensional.  The dimensions
        are understood to be along the "element", "parameter", and
        "2nd parameter" directions in that order.

        Parameters
        ----------
        array_type : ("e", "ep", "ep2", "epp", "p", "jtj", "jtf", "c", "cp", "cp2", "cpp")
            The type of array to allocate, often corresponding to the array shape.  See
            :meth:`allocate_local_array` for a more detailed description.

        array_portion : numpy.ndarray
            The portion of the final array that is local to the calling
            processor.  This could be a shared memory array, but just needs
            to be of the correct size.

        extra_elements : int, optional
            The number of additional "extra" elements to append to the element
            dimension, beyond those called for by this layout.  Should match
            usage in :meth:`allocate_local_array`.

        all_gather : bool, optional
            Whether the result should be returned on all the processors (when `all_gather=True`)
            or just the rank-0 processor (when `all_gather=False`).

        return_shared : bool, optional
            Whether the returned array is allowed to be a shared-memory array, which results
            in a small performance gain because the array used internally to gather the results
            can be returned directly. When `True` a shared memory handle is also returned, and
            the caller assumes responsibilty for freeing the memory via
            :func:`pygsti.tools.sharedmemtools.cleanup_shared_ndarray`.

        Returns
        -------
        gathered_array : numpy.ndarray or None
            The full (global) output array on the root (rank=0) processor and
            `None` on all other processors, unless `all_gather == True`, in which
            case the array is returned on all the processors.
        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            Returned only when `return_shared == True`.  The shared memory handle
            associated with `gathered_array`, which is needed to free the memory.
        """
        if return_shared:
            return array_portion, None  # no shared memory handle
        else:
            return array_portion  # no gathering is performed by this layout class

    def gather_local_array(self, array_type, array_portion, extra_elements=0, return_shared=False):
        """
        Gathers an array onto the root processor.

        Gathers the portions of an array that was distributed using this
        layout (i.e. according to the host_element_slice, etc. slices in
        this layout).  This could be an array allocated by :meth:`allocate_local_array`
        but need not be, as this routine does not require that `array_portion` be
        shared.  Arrays can be 1, 2, or 3-dimensional.  The dimensions
        are understood to be along the "element", "parameter", and
        "2nd parameter" directions in that order.

        Parameters
        ----------
        array_portion : numpy.ndarray
            The portion of the final array that is local to the calling
            processor.  This could be a shared memory array, but just needs
            to be of the correct size.

        extra_elements : int, optional
            The number of additional "extra" elements to append to the element
            dimension, beyond those called for by this layout.  Should match
            usage in :meth:`allocate_local_array`.

        return_shared : bool, optional
            If `True` then, when shared memory is being used, the shared array used
            to accumulate the gathered results is returned directly along with its
            shared-memory handle (`None` if shared memory isn't used).  This results
            in a small performance gain.

        Returns
        -------
        result : numpy.ndarray or None
            The full (global) output array on the root (rank=0) processor and
            `None` on all other processors.

        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            Returned only when `return_shared == True`.  The shared memory handle
            associated with `result`, which is needed to free the memory.
        """
        return self.gather_local_array_base(array_type, array_portion, extra_elements, False, return_shared)

    def allgather_local_array(self, array_type, array_portion, extra_elements=0, return_shared=False):
        """
        Gathers an array onto all the processors.

        Gathers the portions of an array that was distributed using this
        layout (i.e. according to the host_element_slice, etc. slices in
        this layout).  This could be an array allocated by :meth:`allocate_local_array`
        but need not be, as this routine does not require that `array_portion` be
        shared.  Arrays can be 1, 2, or 3-dimensional.  The dimensions
        are understood to be along the "element", "parameter", and
        "2nd parameter" directions in that order.

        Parameters
        ----------
        array_portion : numpy.ndarray
            The portion of the final array that is local to the calling
            processor.  This could be a shared memory array, but just needs
            to be of the correct size.

        extra_elements : int, optional
            The number of additional "extra" elements to append to the element
            dimension, beyond those called for by this layout.  Should match
            usage in :meth:`allocate_local_array`.

        return_shared : bool, optional
            If `True` then, when shared memory is being used, the shared array used
            to accumulate the gathered results is returned directly along with its
            shared-memory handle (`None` if shared memory isn't used).  This results
            in a small performance gain.

        Returns
        -------
        result : numpy.ndarray or None
            The full (global) output array.

        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            Returned only when `return_shared == True`.  The shared memory handle
            associated with `result`, which is needed to free the memory.
        """
        return self.gather_local_array_base(array_type, array_portion, extra_elements, True, return_shared)

    def allsum_local_quantity(self, typ, value, use_shared_mem="auto"):
        """
        Sum a local array (or scalar) distributed using this layout.

        Sums per-element or per-circuit values across the processors used by this layout.
        Each array must be the same size, but need not be allocated in any particular way.

        Parameters
        ----------
        typ : {"e", "c"}
            Whether the array is an element or circuit array.

        value : float or numpy.ndarray
            The value to sum.  Must be the same size on all processors.

        use_shared_mem : bool, optional
            If `True` then, a tempoary shared memory array is allocated and used
            for the sum (when shared memory is available).  For large arrays, using
            shared memory is faster than MPI communication, but for small arrays the
            overhead of creating the shared memory negates these gains.  This argument
            must be `False` when `value` is just a float.

        Returns
        -------
        numpy.ndarray or float
            The summed value, returned on all the processors.
        """
        return value

    def fill_jtf(self, j, f, jtf):
        """
        Calculate the matrix-vector product `j.T @ f`.

        Here `j` is often a jacobian matrix, and `f` a vector of objective function term
        values.  `j` and `f` must be local arrays, created with :meth:`allocate_local_array`.
        This function performs any necessary MPI/shared-memory communication when the
        arrays are distributed over multiple processors.

        Parameters
        ----------
        j : LocalNumpyArray
            A local 2D array (matrix) allocated using `allocate_local_array` with the `"ep"`
            (jacobian) type.

        f : LocalNumpyArray
            A local array allocated using `allocate_local_array` with the `"e"` (element array)
            type.

        jtf : LocalNumpyArray
            The result.  This must be a pre-allocated local array of type `"jtf"`.

        Returns
        -------
        None
        """
        jtf[:] = _np.dot(j.T, f)

    def fill_jtj(self, j, jtj):
        """
        Calculate the matrix-matrix product `j.T @ j`.

        Here `j` is often a jacobian matrix, so the result is an approximate Hessian.
        This function performs any necessary MPI/shared-memory communication when the
        arrays are distributed over multiple processors.

        Parameters
        ----------
        j : LocalNumpyArray
            A local 2D array (matrix) allocated using `allocate_local_array` with the `"ep"`
            (jacobian) type.

        jtj : LocalNumpyArray
            The result.  This must be a pre-allocated local array of type `"jtj"`.

        Returns
        -------
        None
        """
        jtj[:] = _np.dot(j.T, j)

    #Not needed
    #def allocate_jtj_shared_mem_buf(self):
    #    return _np.empty((self._param_dimensions[0], self._param_dimensions[0]), 'd'), None

    def memory_estimate(self, array_type, dtype='d'):
        """
        Memory required to allocate an array of a given type (in bytes).

        Parameters
        ----------
        array_type : {'e', 'ep', 'epp'}
            The type of array.  This string specifies the shape of the array,
            with `'e'` indicating dimension holding the layout's elements and
            `'p'` indicating parameter dimensions.

        dtype : numpy.dtype
            The NumPy data type for the array.

        Returns
        -------
        int
            The memory that would be required, in bytes.
        """
        bytes_per_element = _np.dtype(dtype).itemsize
        if array_type == "e": return self._size * bytes_per_element
        if array_type == "ep": return self._size * self._param_dimensions[0] * bytes_per_element
        if array_type == "epp": return self._size * self._param_dimensions[0] * \
           self._param_dimensions[1] * bytes_per_element
        raise ValueError("Invalid `array_type`: %s" % array_type)

    def indices(self, circuit):
        """
        The element indices corresponding to a circuit in this layout.

        This is a slice into the element-dimension of arrays allocated using this layout,
        e.g. an `'e'`-type array allocated by :meth:`allocate_local_array`.  The
        entries of such an array correspond to different outcomes of this circuit, which
        are separately given by :meth:`outcomes` or alongside the indices in
        :meth:`indices_and_outcomes`.

        Parameters
        ----------
        circuit : Circuit
            The circuit to lookup element indices of.

        Returns
        -------
        slice
        """
        return self._element_indices[self._unique_circuit_index[circuit]]

    def outcomes(self, circuit):
        """
        The outcome labels of a circuit in this layout.

        Parameters
        ----------
        circuit : Circuit
            The circuit to lookup outcome labels of.

        Returns
        -------
        tuple
        """
        return self._outcomes[self._unique_circuit_index[circuit]]

    def indices_and_outcomes(self, circuit):
        """
        The element indices and outcomes corresponding to a circuit in this layout.

        Returns both the element indices and outcome labels corresponding
        to a circuit in this layout.  These quantities can be separately obtained
        using the :meth:`indices` and :meth:`outcomes` methods, respectively.

        Parameters
        ----------
        circuit : Circuit
            The circuit to lookup element indices and outcomes of.

        Returns
        -------
        element_indices : slice
        outcome_labels : tuple
        """
        unique_circuit_index = self._unique_circuit_index[circuit]
        return self._element_indices[unique_circuit_index], self._outcomes[unique_circuit_index]

    def indices_for_index(self, index):
        """
        Lookup the element indices corresponding to a given circuit by the circuit's index.

        Similar to :meth:`indices` but uses a circuit's index within this layout directly,
        thus avoiding having to hash a :class:`Circuit` object and gaining a modicum of
        performance.

        Parameters
        ----------
        index : int
            The index of a circuit within this layout, i.e., within `self.circuits`.

        Returns
        -------
        slice
        """
        return self._element_indices[self._to_unique[index]]

    def outcomes_for_index(self, index):
        """
        Lookup the outcomes of a given circuit by the circuit's index.

        Similar to :meth:`outcomes` but uses a circuit's index within this layout directly,
        thus avoiding having to hash a :class:`Circuit` object and gaining a modicum of
        performance.

        Parameters
        ----------
        index : int
            The index of a circuit within this layout, i.e., within `self.circuits`.

        Returns
        -------
        tuple
        """
        return self._outcomes[self._to_unique[index]]

    def indices_and_outcomes_for_index(self, index):
        """
        Lookup the element indices and outcomes corresponding to a given circuit by the circuit's index.

        Similar to :meth:`indices_and_outcomes` but uses a circuit's index within this layout
        directly, thus avoiding having to hash a :class:`Circuit` object and gaining a modicum of
        performance.

        Parameters
        ----------
        index : int
            The index of a circuit within this layout, i.e., within `self.circuits`.

        Returns
        -------
        element_indices : slice
        outcome_labels : tuple
        """
        unique_circuit_index = self._to_unique[index]
        return self._element_indices[unique_circuit_index], self._outcomes[unique_circuit_index]

    def __iter__(self):
        for circuit, i in self._unique_circuit_index.items():
            for element_index, outcome in zip(self._element_indices[i], self._outcomes[i]):
                yield element_index, circuit, outcome

    def iter_unique_circuits(self):
        """
        Iterate over the element-indices, circuit, and outcomes of each unique circuit in this layout.

        A generator used to iterate over a `(element_indices, circuit, outcomes)` tuple
        for each *unique* circuit held by this layout, where `element_indices` and `outcomes`
        are the values that would be retrieved by the :meth:`indices` and :meth:`outcomes`
        methods, and `circuit` is the unique circuit itself.

        Returns
        -------
        generator
        """
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

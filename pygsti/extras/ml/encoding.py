"""Methods for encoding circuits into arrays and creating other circuit-derived tensors for QPANNs.

This module provides:
  * Circuit encoders that map `pygsti.circuits.Circuit` objects to fixed-shape numpy arrays.
  * Utilities for batching circuits into tensors.
  * Utilities for computing error-propagation tensors and first-order probability correction
    coefficients (alpha-like quantities) used by QPANN models.

The core idea is to represent a circuit as a depth-by-features array, then compute additional
tensors that describe how error generators propagate through the circuit and how they affect
measurement outcome probabilities in a first-order approximation.
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
import warnings as _warnings
import itertools as _itertools
import stim as _stim
from pygsti.errorgenpropagation import errorpropagator as _ep
from pygsti.errorgenpropagation import localstimerrorgen as _lseg
from pygsti.extras.ml import errgentools as _tools
from pygsti.tools import errgenproptools as _egptools
from pygsti.circuits import Circuit as _Circuit
import tqdm
from tqdm import trange as _trange
from multiprocessing import Pool
from itertools import starmap
from typing import TYPE_CHECKING, cast, Any

if TYPE_CHECKING:
    from pygsti.processors import ProcessorSpec
    from pygsti.data import DataSet

class CircuitEncoder(object):
    """
   Base class for encoders that map `pygsti.circuits.Circuit` objects to numpy arrays.

    A `CircuitEncoder` defines:
      * A per-layer encoding (`layer_encoding`)
      * Optional encodings for implicit initialization/measurement layers
      * A fixed per-layer feature length (`self.length`)

    Derived classes implement `layer_encoding` and usually `indices_for_qubits`.
    """
    def __init__(self, pspec: "ProcessorSpec") -> None:
        """
        Initialize a CircuitEncoder object.

        Parameters
        ----------
        pspec : ProcessorSpec
            Describes the qubits and gates in the circuits that will be encoded.

        Returns
        -------
        CircuitEncoder

        Notes
        -----
        Derived classes should set `self.length` to the feature length of each encoded layer.
        """
        from pygsti.processors import QubitProcessorSpec
        assert isinstance(pspec, QubitProcessorSpec)
        self.pspec = pspec
        self.length: int = 0 # This should be specified in a derived class

    def __call__(self, circuit: _Circuit, padded_depth: int | None = None) -> _np.ndarray:
        """
        Encode a circuit as a 2D numpy array.

        The returned encoding is constructed as:

            initialization_encoding(circuit)
            + [layer_encoding(layer_i) for each layer i]
            + padding layers (if padded_depth > circuit.depth)
            + measurement_encoding(circuit)

        Parameters
        ----------
        circuit : pygsti.circuits.Circuit
            Circuit to encode.
        padded_depth : int or None, optional
            If not None, pad/truncate the layer portion to this depth (padding uses
            `layer_encoding(None)`). If None, uses `circuit.depth`.

        Returns
        -------
        numpy.ndarray
            Array of shape `(encoding_depth, self.length)` where `encoding_depth`
            depends on `padded_depth` and the init/measurement encoding depths.
        """
        if padded_depth is None: padded_depth = circuit.depth
        circuit_array: list[Any] = []
        circuit_array += self.initialization_encoding(circuit)
        circuit_array += [self.layer_encoding(circuit.layer(i)) for i in range(circuit.depth)]
        circuit_array += [self.layer_encoding(None) for l in range(circuit.depth, padded_depth)]
        circuit_array += self.measurement_encoding(circuit)

        return _np.array(circuit_array)

    def initialization_encoding(self, circuit: _Circuit) -> list[list[float]] | list[list[int]]:
        """
        This method defines the encoding of the implicit initialization layer at the end of a circuit.
        In this base class, we define this to be a trivial encoding: the implicit initialization layer 
        is not represented. In general, this method should return a list of lists. If this measurement 
        is not represented in the encoding, as here, this is an empty list. Otherwise each element in the 
        list is contains the encoding values (typically 0 or 1)

        Parameters
        ----------
        circuit : pygsti.circuits.Circuit
            Circuit whose initialization is to be encoded.

        Returns
        -------
        list[list[float]] or list[list[int]]
            A list of per-layer feature vectors. Empty if initialization is not represented.
        """
        return []

    def initialization_encoding_depth(self) -> int:
        """
        Returns the length of the initalization encoding list, i.e., the length of lists returned by
        `initialization_encoding`.

        Returns
        -------
        int
            The length of the list returned by `initialization_encoding`.
        """
        return 0

    def measurement_encoding(self, circuit: _Circuit) -> list[list[float]] | list[list[int]]:
        """
        This method defines the encoding of the implicit measurement layer at the end of a circuit.
        In this base class, we define this to be a trivial encoding: the implicit measurement layer 
        is not represented. In general, this method should return a list of lists. If this measurement 
        is not represented in the encoding, as here, this is an empty list. Otherwise each element in the 
        list is contains the encoding values (typically 0 or 1)

        Parameters
        ----------
        circuit : pygsti.circuits.Circuit
            Circuit whose measurement is to be encoded.

        Returns
        -------
        list[list[float]] or list[list[int]]
            A list of per-layer feature vectors. Empty if measurement is not represented.
        """
        return []

    def measurement_encoding_depth(self) -> int: 
        """
        Returns the length of the measurement encoding list, i.e., the length of lists returned by
        `measurement_encoding`.

        Returns
        -------
        int
            The length of the list returned by `measurement_encoding`.
        """
        return 0

    def layer_encoding(self, layer: tuple | list | None) -> list:
        """
        This method defines the encoding of a circuit layer, return a list. The list must be of the
        same length for all layers, and it typically containing 0s and 1s, but it can contain any
        objects that can be elements of a numpy.array. This method is implemented in derived
        classes.

        Parameters
        ----------
        layer : tuple or list of pygsti.baseobjs.Label, or None
            The layer to turn into a list of numerical values. `None` is used for padding layers.

        Returns
        -------
        list
            Encoded layer as a list of length `self.length`.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError("Specified in a derived class!")

    def depth(self, max_circuit_depth: int) -> int:
        """
        The depth of the encoding, meaning the second dimension in the array returned by
        call the CircuitEncoder.

        Parameters
        ----------
        max_circuit_depth : int
            The maximum depth of the circuits that will be encoded using the encoder (excluding implicit init/measurement).

        Returns
        -------
        int
            The encoding depth, which is `max_circuit_depth` + the depths for initialization
            and measurment encoding. (i.e. `max_circuit_depth + initialization_encoding_depth() + measurement_encoding_depth()`)
        """
        return max_circuit_depth + self.initialization_encoding_depth() + self.measurement_encoding_depth()

    def indices_for_qubits(self, qubits: list | set | tuple) -> list[int]:
        """
        Return encoding indices that correspond to operations touching the specified qubits.

        Derived encoders should implement this to support "snippers" and locality-based
        feature selection.

        Parameters
        ----------
        qubits : iterable
            Qubit identifiers/labels as used by `self.pspec` and the encoder.

        Returns
        -------
        list[int]
            Indices into the per-layer encoding vector.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError("Specified in a derived class!")

class StandardCircuitEncoder(CircuitEncoder):
    """
    "Standard" circuit encoder used by QPANN workflows.

    Each layer is encoded as a one-hot (or multi-hot) vector over all gate labels
    available in the `ProcessorSpec`. If multiple gates occur in a layer, multiple
    entries can be 1.

    Notes
    -----
    `self.gate_indexing` is a flat list of all available gate labels across all gate names
    and all allowed qubit placements; the per-layer encoding length is `len(self.gate_indexing)`.
    """

    def __init__(self, pspec: "ProcessorSpec") -> None:
        """
        Construct a StandardCircuitEncoder.

        Parameters
        ----------
        pspec : ProcessorSpec
            Processor specification that defines:
              * `gate_names`
              * `qubit_labels`
              * `available_gatelabels(...)`
        """
        from pygsti.processors import QubitProcessorSpec
        assert isinstance(pspec, QubitProcessorSpec)
        self.pspec = pspec
        all_gates = []
        for gate in pspec.gate_names:
            all_gates += list(pspec.available_gatelabels(gate, pspec.qubit_labels))
        self.gate_indexing = all_gates
        self.length = len(all_gates)

    def layer_encoding(self, layer: tuple | list | None) -> list[float]:
        """
        Encode a layer as a multi-hot vector over `self.gate_indexing`.

        Parameters
        ----------
        layer : tuple or list of pygsti.baseobjs.Label objects, or None
            The layer to encode. If None, returns an all-zeros vector.

        Returns
        -------
        list[float]
            A list of length `self.length` containing 0/1 floats.
        """
        
        if layer is not None:
            assert(isinstance(layer, (tuple, list))), "The layer must be a list or tuple of label objects, or None!"
        if layer is not None:
            encoded_layer = _np.zeros(self.length, float)
            for gate in layer:
                encoded_layer[self.gate_indexing.index(gate)] = 1
            return list(encoded_layer)
        else:
            return list(_np.zeros(self.length, float))

    def indices_for_qubits(self, qubits: list | set | tuple) -> list[int]:
        """
        Return encoding indices for gate labels that touch any of the specified qubits.

        Parameters
        ----------
        qubits : iterable
            Qubit labels.

        Returns
        -------
        list[int]
            Indices into `self.gate_indexing` (and hence into the per-layer encoding).
        """
        return [i for i, gate in enumerate(self.gate_indexing) if len(set(qubits).intersection(gate.qubits)) > 0]


def circuits_to_tensor(circuits: list[_Circuit], encoder: CircuitEncoder, encoding_depth: int | None = None) -> _np.ndarray:
    """
    Encode a list of circuits into a 3D tensor.

    Parameters
    ----------
    circuits : list[pygsti.circuits.Circuit]
        Circuits to encode.
    encoder : CircuitEncoder
        Encoder used to convert each circuit to a 2D array.
    encoding_depth : int or None, optional
        If None, uses the maximum circuit depth in `circuits` and computes
        the corresponding encoder depth via `encoder.depth(max_depth)`.

    Returns
    -------
    numpy.ndarray
        Tensor of shape `(len(circuits), encoding_depth, encoder.length)`.
    """

    if encoding_depth is None: 
        max_depth = _np.max([c.depth for c in circuits])
        encoding_depth = encoder.depth(max_depth)
    
    circuits_tensor = _np.zeros((len(circuits), encoding_depth, encoder.length), float)
    for i, circuit in enumerate(circuits):
        circuits_tensor[i,:,:] = encoder(circuit, padded_depth=encoding_depth)

    return circuits_tensor

def dense_dataset_encoding(ds: "DataSet", n: int, circuits: list[_Circuit] | None = None) -> _np.ndarray:
    """
    Convert a pyGSTi dataset into a dense frequency array over all (2^n) bitstrings.

    Parameters
    ----------
    ds : pygsti.data.DataSet
        Dataset keyed by circuits, with outcome counts.
    n : int
        Number of qubits (determines (2^n) outcomes).
    circs : list[pygsti.circuits.Circuit] or None, optional
        Subset of circuits to include. If None, uses `list(ds.keys())`.

    Returns
    -------
    numpy.ndarray
        Array of shape `(num_circuits, 2**n)` with frequencies for each bitstring.
    """

    if circuits is None: circuits = list(ds.keys())
    nbit_strings: list[str] = [] # TO DO
    freqs_array = _np.zeros((len(circuits), 2**n), float)
    for i in _trange(len(circuits)):
        dsrow: Any = ds[circuits[i]]
        freqs_array[i,:] =  _np.array([dsrow.counts.get((bs,), 0.) / dsrow.total for bs in nbit_strings])

    return freqs_array

def error_generator_tensors(circuits: list[_Circuit], error_generators: list, pspec: "ProcessorSpec", alpha_representation: str = 'concise',  measurements: str= 'probabilities', 
                            measurement_paulis: list | None = None, process_num: int = 5) -> dict:
    """
    Compute the tensors needed by QPANN probability-approximation layers.

    This is a convenience wrapper that combines:
      * error-propagation tensors (`indices`, `signs`)
      * first-order outcome tensors (`probabilities`, and either dense alphas or concise coefficients)

    Parameters
    ----------
    circuits : list[pygsti.circuits.Circuit]
        Circuits of interest.
    error_generators : list
        Elementary error generators to consider (see `circuit_error_propagation_matrices`).
    pspec : ProcessorSpec
        Processor specification (used mainly for num_qubits).
    alpha_representation : {'concise', 'matrix'}, default 'concise'
        Which alpha/first-order representation to compute.

    Returns
    -------
    dict
        Dictionary with keys:
          * 'indices' : numpy.ndarray
          * 'signs' : numpy.ndarray
          * 'probabilities' : numpy.ndarray
          * 'alphas' : numpy.ndarray
        where the meaning/shape of 'alphas' depends on `alpha_representation`.
    """
    from pygsti.processors import QubitProcessorSpec
    assert isinstance(pspec, QubitProcessorSpec)
    indices, signs = error_propagation_tensors(circuits, error_generators, pspec)
    if alpha_representation == 'matrix':
        probabilities, alphas = first_order_outcome_probabilities_tensors(circuits, error_generators, pspec, indices=indices)
    elif alpha_representation == 'concise':
        probabilities, alphas = first_order_outcome_probabilities_tensors_concise(circuits, pspec, indices, signs, measurements = measurements, 
                                                                                  measurement_paulis = measurement_paulis, process_num=process_num)        
    else:
        raise NotImplementedError('No other representations have been implemented yet!')
    return {'indices':indices, 'signs':signs, 'probabilities':probabilities, 'alphas':alphas}

def circuit_error_propagation_matrices(circuit: _Circuit, error_generators: list) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Computers the `indices` and `signs` matrices used in quantum physics aware neural networks,
    for the input circuit and the error generators specified in the `error_generators` list.

    Conceptually, this function:
      1. Constructs Stim layers for the circuit.
      2. Propagates a fixed set of elementary error generators through each layer.
      3. Records (a) which end-of-layer error generator each one maps to (an index),
         and (b) the sign acquired under propagation.

    Parameters
    ----------
    circuit : pygsti.circuits.Circuit
        Circuit to compute propagation data for.
    error_generators : list
        List of error generator labels. Each element should be a tuple:
            (errorgen_type, paulis_tuple)
        where `errorgen_type` is the error generator type ('H'', 'S', 'C', or 'A') and `paulis_tuple` is a tuple of Pauli strings.
        For 'H' and 'S', this is a single Pauli string. The Pauli strings should be length-n strings
        where n is the number of qubits on which `circuit` acts.        


        
    Returns
    -------
    indices : numpy.ndarray
        Integer array of shape `(circuit.depth, len(error_generators))` giving, for each
        layer and each generator inserted after that layer, the index of the propagated
        (end-of-circuit) elementary error generator.
    signs : numpy.ndarray
        Integer array of the same shape giving (+-1) sign factors for each propagation.

    """
    error_propagator = _ep.ErrorGeneratorPropagator(None)
    stim_layers = error_propagator.construct_stim_layers(circuit, drop_first_layer=True)
    propagation_layers = error_propagator.construct_propagation_layers(stim_layers)
    # TODO : It's the list of error gens in every layer of the circuit.
    errorgen_layers = [{_lseg.LocalStimErrorgenLabel.cast(lbl):1 for lbl in error_generators}] * circuit.depth

    propagated_errorgen_layers = error_propagator._propagate_errorgen_layers(errorgen_layers, propagation_layers, include_spam=False) # list of dicts of error generators

    indices = _np.array([[_tools.error_generator_index(err.errorgen_type, err.bel_to_strings()) for err in propagated_errorgen_layers[l]] for  l in range(circuit.depth)])
    signs = _np.array([[_np.sign(val) for val in propagated_errorgen_layers[l].values()] for l in range(circuit.depth)])

    return indices, signs

def error_propagation_tensors(circuits: list[_Circuit], error_generators: list, pspec: "ProcessorSpec", prior_error_generators: list | None = None, 
                              prior_indices: _np.ndarray | None = None, prior_signs: _np.ndarray | None = None) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Compute batched error-propagation tensors (`indices`, `signs`) for multiple circuits.

    Parameters
    ----------
    circuits : list[pygsti.circuits.Circuit]
        Circuits to process.
    error_generators : list
        New error generators to add.
    pspec : ProcessorSpec
        Processor spec; used for `num_qubits` and to determine max circuit depth.
    prior_error_generators : list or None, optional
        Previously computed generators. If provided, this function appends new generators
        after the prior ones (and checks that there is no overlap).
    prior_indices : numpy.ndarray or None, optional
        Previously computed indices tensor to copy into the left block.
    prior_signs : numpy.ndarray or None, optional
        Previously computed signs tensor to copy into the left block.

    Returns
    -------
    indices : numpy.ndarray
        Array of shape `(num_circuits, max_depth, num_error_generators_total)` with propagated indices.
    signs : numpy.ndarray
        Array of the same shape with (+-1) sign factors.

    Notes
    -----
    For a circuit with depth < max_depth, the trailing layers remain zero-filled.
    For more information on what a circuit's `indices` and `signs` matrix encode, see the docstring for
    the function `circuit_error_propagation_matrices`, which this function calls to compute these matrices
    for each circuit.
    """
    from pygsti.processors import QubitProcessorSpec
    assert isinstance(pspec, QubitProcessorSpec)

    if prior_error_generators is not None:
        assert(len(set(error_generators).intersection(set(prior_error_generators))) == 0), "Can only add new error generators!"
        num_pregs = len(prior_error_generators)
    else:
        num_pregs = 0

    num_qubits = pspec.num_qubits
    max_depth = _np.max([circuit.depth for circuit in circuits])

    indices = _np.zeros((len(circuits), max_depth, len(error_generators) + num_pregs), int)
    signs = _np.zeros((len(circuits), max_depth, len(error_generators) + num_pregs), int)

    if prior_error_generators is not None:
        assert prior_indices is not None and prior_signs is not None
        indices[:, 0:num_pregs, :] = prior_indices.copy()
        signs[:, 0:num_pregs, :] = prior_signs.copy()
       
    for i, circuit in enumerate(circuits):

        # compute the sign and indices matrices for this circuit
        indices_for_circuit, signs_for_circuit = circuit_error_propagation_matrices(circuit, error_generators)
        indices[i, 0:circuit.depth, num_pregs:] = indices_for_circuit.copy()
        signs[i, 0:circuit.depth, num_pregs:] = signs_for_circuit.copy()
    
    return indices, signs

def alpha_coefficient(i: int, num_qubits: int, tableau: _stim.Tableau, bs: str) -> float:
    """
    Computes the alpha coefficient for the ith error generator, with the circuit defined by the
    input tableau and for the given bit string.

    Parameters
    ----------
    i : int
        The index of the error generator, as specified in the ordering of ml.tools.index_to_error_gen

    num_qubits : int
        The number of qubits (n)

    tableau : stim.Tableau
        The stabilizer tableau of the circuit for which we are calculating the alpha coefficient

    bs : str
        The bit string (of length n)for which the alpha coefficient is to be computed.

    Returns
    -------
    float
        The alpha coefficient
    """
    lbl = cast(_lseg.LocalStimErrorgenLabel, _tools.index_to_error_gen(i, num_qubits, as_label=True))
    return _np.float64(_egptools.alpha(lbl, tableau, bs).real)

def alpha_coefficient_pauli(i: int, num_qubits: int, tableau: _stim.Tableau, pauli: str) -> float:
    """
    Computes the alpha coefficient for the ith error generator, with the circuit defined by the
    input tableau and for the given bit string.

    Parameters
    ----------
    i : int
        The index of the error generator, as specified in the ordering of ml.tools.index_to_error_gen

    num_qubits : int
        The number of qubits

    tableau : stim.Tableau
        The tableau of the circuit for which we are calculating the alpha coefficient

    bs : str
        The bit string for which the alpha coefficient is to be computed.

    Returns
    -------
    float
        The alpha coefficient
    """
    lbl = cast(_lseg.LocalStimErrorgenLabel, _tools.index_to_error_gen(i, num_qubits, as_label=True))
    return _np.float64(_egptools.alpha_pauli(lbl, tableau, cast(Any, pauli)))

def _get_tableau(circuit_or_tableau: _Circuit | _stim.Tableau) -> _stim.Tableau:
    """
    Helper function that returns a tableau given a Circuit or a Stim.Tableau object

    Parameters
    ----------
    circuit_or_tableau : pygsti.circuits.Circuit or stim.Tableau
        Object to convert/validate.

    Returns
    -------
    stim.Tableau
        Tableau representation of the circuit.

    Raises
    ------
    ValueError
        If input is neither a `Circuit` nor a `stim.Tableau`.
    """
    if isinstance(circuit_or_tableau, _Circuit):
        return circuit_or_tableau.convert_to_stim_tableau()
    elif isinstance(circuit_or_tableau, _stim.Tableau):
        return circuit_or_tableau
    else:
        raise ValueError('Input must be a Circuit or a Stim.Tableau!')

def dense_alpha_matrix(circuit: _Circuit | _stim.Tableau, num_qubits: int, populate_for_error_generators: list[int] | None = None, existing_alpha_matrix: _np.ndarray | None = None) -> _np.ndarray:
    """
    Creates the alpha matrix, a 2**n by 2 * 4**n matrix, that is used to compute the 
    first-order impact of each end-of-circuit error generator on each n-bit string. Can
    be used to update an existing partially-populated alpha matrix.

    Parameters
    ----------
    circuit: Circuit or Stim.Tableau
        The circuit for which we are computing the alpha tensor.

    num_qubits: int
        The number of qubits in the circuit.

    populate_for_error_generators : list[int] or None, optional
        If not None, the indices of the error generators for which to compute the alpha tensor for.
        If None, the entire alpha tensor is computed. If a list of integers, this function creates
        an alpha tensor that is zero except for the alphas corresponding to the end-of-circuit 
        error generators in this list.

    existing_alpha_matrix : None or numpy.ndarray of shape (2**n, 2 * 4**n), optional
        An existing partially-populated alpha matrix. Note that providing an existing alpha matrix
        has no purpose if `populuate_for_error_generators` is None, as then the complete alpha
        matrix is computed. 

    Returns
    -------
    The alpha matrix, as a dense array. This is an array that is of shape 2**n by 2 * 4**n, with
    element [i,j] corresponding to the alpha value for the ith n-bit string and the jth end-of-circuit
    H or S type error generator, using the indexing defined in ml.tools. If an existing alpha matrix
    was provided, only elements corresponding to `populate_for_error_generators` have been changed.

    """
    num_nq_errgens = 2 * 4 ** num_qubits # We include the S_{III} and H_{III..} in our indexing
    nbit_strings = [''.join(p) for p in _itertools.product('01', repeat=num_qubits)]
    if populate_for_error_generators is None: populate_for_error_generators = list(range(num_nq_errgens))

    tableau = _get_tableau(circuit)

    if existing_alpha_matrix is not None:
        alpha_matrix = existing_alpha_matrix.copy()
    else:
        alpha_matrix = _np.zeros((2 ** num_qubits, num_nq_errgens), float)

    scale = 1 / 2 ** cast(Any, _egptools.random_support(tableau)) #TODO: This might overflow
    for i in populate_for_error_generators:
        alpha_matrix[:, i] = scale * _np.array([alpha_coefficient(i, num_qubits, tableau, bs) for bs in nbit_strings])

    return alpha_matrix

### Code that Tim wrote but never tested as he decided that to write `first_order_outcome_probabilities_tensors_concise`
### which creates an alternative representation that is probably more useful in practice. This has been left in in case
### it becomes useful, but it shouldn't be assumed that it works correctly as it was never tested.
# def sparse_alpha_matrix(circuit, num_qubits, error_generators, bitstrings):
#     """
#     Creates a sparse reprentation of the alpha matrix, a 2**n by 2 * 4**n matrix, that is used to compute the 
#     first-order impact of each end-of-circuit error generator on each n-bit string.

#     Parameters
#     ----------
#     circuit: Circuit or Stim.Tableau
#         The circuit for which we are computing the alpha tensor.

#     num_qubits: int
#         The number of qubits in the circuit.

#     error_generators : list of integers.
#         Tthe **indices** of the error generators for which to compute the alpha matrix elements for.

#     bitstrings : list of strings 
#         The n-bit strings for which to compute the alppha coefficients for.

#     Returns
#     -------
#     list of lists
#         A list of lists in which each element of the list ... TODO

#     """
#     tableau = _get_tableau(circuit)

#     contributing_error_generators = [[] for bs in bitstrings]
#     alphas = [[] for bs in bitstrings]
#     for i, bs in enumerate(bitstrings):
#         for j in error_generators:
#             alpha = alpha_coefficient(j, num_qubits, tableau, bs)
#             if not _np.isclose(alpha, 0.):
#                 contributing_error_generators[i].append(j)
#                 alphas[i].append(alpha)

#     return contributing_error_generators, alphas
def first_order_outcome_probabilities_tensors(circuits: list[_Circuit], error_generators: list, pspec: "ProcessorSpec", indices: _np.ndarray | None = None, prior_error_generators: list | None = None, prior_alphas: _np.ndarray | None = None) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Compute ideal outcome probabilities and dense alpha matrices for a batch of circuits.

    Parameters
    ----------
    circuits : list[pygsti.circuits.Circuit]
        Circuits to process.
    error_generators : list
        Currently unused directly (kept for API symmetry); alphas are computed in the
        canonical indexing over all H/S end-of-circuit generators, optionally restricted
        by `indices`.
    pspec : ProcessorSpec
        Processor spec; used for `num_qubits`.
    indices : numpy.ndarray or None, optional
        If provided, only compute alpha columns for end-of-circuit generator indices that
        appear in `indices[i,:,:]` for each circuit i.
    prior_error_generators : list or None, optional
        If provided, indicates we are updating existing alpha matrices (no overlap allowed).
    prior_alphas : numpy.ndarray or None, optional
        Existing alpha tensor to partially reuse.

    Returns
    -------
    probabilities : numpy.ndarray
        Array of shape `(num_circuits, 2**n)` with ideal outcome probabilities.
    alphas : numpy.ndarray
        Array of shape `(num_circuits, 2**n, 2*4**n)` with dense alpha matrices.
    """
    from pygsti.processors import QubitProcessorSpec
    assert isinstance(pspec, QubitProcessorSpec)

    if prior_error_generators is not None:
        assert(len(set(error_generators).intersection(set(prior_error_generators))) == 0), "Can only add new error generators!"

    num_qubits = pspec.num_qubits
    nbit_strings = [''.join(p) for p in _itertools.product('01', repeat=num_qubits)]

    alphas = _np.zeros((len(circuits), 2 ** num_qubits, 2 * 4 ** num_qubits), float)
    probabilities = _np.zeros((len(circuits), 2 ** num_qubits), float)
      
    for i, circuit in enumerate(circuits):

        # Compute the error-free probabilities for each bit string.
        tableau = circuit.convert_to_stim_tableau()
        probabilities[i, :] = _np.array([_egptools.stabilizer_probability(tableau, bs) for bs in nbit_strings]).T

        # Compute the alpha matrix for the circuit, filling in only those 
        if prior_error_generators is not None:
            assert prior_alphas is not None
            prior_alpha_matrix = prior_alphas[i, :, :]
        else:
            prior_alpha_matrix = None

        if indices is not None:
            unique_end_of_circuit_error_generators = list(set(indices[i,:,:].flatten()))
        else:
            # We compute the entire alpha matrix
            unique_end_of_circuit_error_generators = None

        alphas[i, :, :] = dense_alpha_matrix(tableau, num_qubits, populate_for_error_generators=unique_end_of_circuit_error_generators,
                                             existing_alpha_matrix=prior_alpha_matrix)
    
    return probabilities, alphas


def first_order_outcome_probabilities_tensors_concise(circuits: list[_Circuit], pspec: "ProcessorSpec", indices: _np.ndarray, signs: _np.ndarray, measurements: str = 'probabilities', 
                                                      measurement_paulis: list | None = None, process_num: int = 5) -> tuple[_np.ndarray, _np.ndarray]:
    """
    TODO

    Parameters
    ----------
    circuits: list of circuits we are calculating the probabilities tensors for

    pspec: the processor spec of the quantum computer

    indices:

    signs:

    measurements: measurement to calculate the sensitivity tensor for, currently supports 'probabilties' (bitstring probabilties)
    and 'paulis' (pauli outcomes)

    pauli_params: describes the measurements to use in the calculation, recieves a two component list, the first argument in the max weight
    of any pauli observables, the second element is the list of allowed paulis

    process_num: number of processes to use for parallel computation

    """
    from pygsti.processors import QubitProcessorSpec
    assert isinstance(pspec, QubitProcessorSpec)
    num_qubits = pspec.num_qubits
    nbit_strings = [''.join(p) for p in _itertools.product('01', repeat=num_qubits)]

    measurements_array: _np.ndarray
    first_order_coefficients: _np.ndarray

    if measurements == 'probabilities':
        shape = (indices.shape[0], 2 ** num_qubits, indices.shape[1], indices.shape[2])
        first_order_coefficients = _np.zeros(shape, float)
        measurements_array = _np.zeros((len(circuits), 2 ** num_qubits), float)
        circ_indices_tuples=[]
        for idx,circ in enumerate(circuits):
            circ_indices_tuples.append((circ,indices[idx],nbit_strings,num_qubits))  
        with Pool(process_num) as p:
            output_list = p.starmap(_circuit_loop_probs, tqdm.tqdm(circ_indices_tuples))
        for idx, tup in enumerate(output_list):
            measurements_array[idx]=tup[0]
            first_order_coefficients[idx]=tup[1]

        for l, bs in enumerate(nbit_strings):
            first_order_coefficients[:, l, :, :] = first_order_coefficients[:, l, :, :] * signs
        
    elif measurements == 'paulis':
        
        assert(measurement_paulis is not None), "Must provided the measurement Pauli operators!"
        shape = (indices.shape[0], len(measurement_paulis), indices.shape[1], indices.shape[2])
        first_order_coefficients = _np.zeros(shape, float)
        measurements_array = _np.zeros((len(circuits),len(measurement_paulis)))
        circ_indices_tuples=[]

        # TIM COPIED OUT WHILE BUG HUNTING
        for idx, circ in enumerate(circuits):
            circ_indices_tuples.append((circ, indices[idx], num_qubits, measurement_paulis)) 

        with Pool(process_num) as p:
            output_list = p.starmap(_circuit_loop_paulis, tqdm.tqdm(circ_indices_tuples))

        for idx, tup in enumerate(output_list):
            measurements_array[idx, :] = tup[0]
            first_order_coefficients[idx, :, :, :] = tup[1]

        for l in range(len(measurement_paulis)):
            first_order_coefficients[:, l, :, :] = first_order_coefficients[:, l, :, :] * signs

    else:
        raise ValueError(f"Unknown measurements type: {measurements}")

    return measurements_array, first_order_coefficients

def _circuit_loop_probs(circuit: _Circuit, indices: _np.ndarray, nbit_strings: list[str], num_qubits: int) -> tuple[_np.ndarray, _np.ndarray]:

    unique_indices = set(indices.flatten())
    
    tableau = _get_tableau(circuit)
    shape = ( 2 ** num_qubits, indices.shape[0], indices.shape[1])
    first_order_coefficients = _np.zeros(shape, float)
    scale = 1 / 2 ** cast(Any, _egptools.random_support(tableau)) #TODO: This might overflow
    probabilities = _np.array([_egptools.stabilizer_probability(tableau, bs) for bs in nbit_strings]).T
    alphas_dict = {}
    for l, bs in enumerate(nbit_strings):
        for error_generator_index in unique_indices:
            egtype = cast(Any, _tools.index_to_error_gen(error_generator_index, num_qubits))[0]

            if egtype == 'H' and (_np.isclose(probabilities[l], 0.) or _np.isclose(probabilities[l], 1.)):
                alpha = 0

            alpha = scale * alpha_coefficient(error_generator_index, num_qubits, tableau, bs)
            alphas_dict[l, error_generator_index] = alpha


    for l in range(len(nbit_strings)):
        for j in range(indices.shape[0]):
            for k in range(indices.shape[1]):
                first_order_coefficients[l, j, k] = alphas_dict[l, indices[j,k]]

    return (probabilities, first_order_coefficients)

def _circuit_loop_paulis(circuit: _Circuit, indices: _np.ndarray, num_qubits: int, paulis: list[str]) -> tuple[_np.ndarray, _np.ndarray]:

    unique_indices = set(indices.flatten())
    
    tableau = _get_tableau(circuit)
    shape = (len(paulis), indices.shape[0], indices.shape[1])
    first_order_coefficients = _np.zeros(shape, float)
    #scale = 1 / 2 ** _egptools.random_support(tableau) #TODO: This might overflow
    measurements = _np.array([_egptools.stabilizer_pauli_expectation(tableau, p) for p in paulis]).T
    alphas_dict = {}
    for l, p in enumerate(paulis):
        for error_generator_index in unique_indices:
            egtype = cast(Any, _tools.index_to_error_gen(error_generator_index, num_qubits))[0]
            #print(egtype)
            # TIM THINKS THIS IS CORRECT BUT COMMENTING OUT WHILE BUG FIXING
            #if egtype == 'H' and (_np.isclose(measurements[l], -1.) or _np.isclose(measurements[l], 1.)):
            #    alpha = 0

            alphas_dict[l, error_generator_index] = alpha_coefficient_pauli(error_generator_index, num_qubits, tableau, p)


    for l in range(len(paulis)):
        for j in range(indices.shape[0]):
            for k in range(indices.shape[1]):
                first_order_coefficients[l, j, k] = alphas_dict[l, indices[j,k]]

    return (measurements, first_order_coefficients)


def make_paulis(num_qubits: int, maximum_weight: int) -> list[Any]:
    """
    """
    paulis = []
    for w in range(1, maximum_weight+1):
        paulis += make_paulis_of_weight(num_qubits, w)
    return paulis

def make_paulis_of_weight(num_qubits: int, weight: int) -> list[Any]:
    """
    num_qubits : number of qubits
    weight : the weight of the Pauli operators
    """
    
    # Generate all combinations of positions for 'Z'
    positions = _itertools.combinations(range(num_qubits), weight)
    
    result = []
    for pos in positions:
        # Start with all 'I's
        chars = ['I'] * num_qubits
        # Place 'Z' at the chosen positions
        for p in pos:
            chars[p] = 'Z'
        result.append(''.join(chars))

    return [_stim.PauliString(pauli) for pauli in result]

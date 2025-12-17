""" Methods for encoding circuits into array and creating other information about circuits needed for QPANNs """
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
from tqdm import trange as _trange

class CircuitEncoder(object):
    """
    An object that converts Circuit objects into numpy arrays. This is a base class that
    contains some general-purpose routines but not a specific encoding method.
    """
    def __init__(self, pspec):
        """
        Initialize a CircuitEncoder object.

        Parameters
        ----------
        pspec : ProcessorSpec
            The ProcessorSpec describing the qubits and gates in the circuits that
            will be encoded.

        Returns
        -------
        CircuitEncoder
        """
        self.pspec = pspec

    def __call__(self, circuit, padded_depth=None):
        """
        Turns the input circuit circuit, a Circuit object, into a numpy array.
        """
        if padded_depth is None: padded_depth = circuit.depth
        circuit_array = []
        circuit_array += self.initialization_encoding(circuit)
        circuit_array += [self.layer_encoding(circuit.layer(i)) for i in range(circuit.depth)]
        circuit_array += [self.layer_encoding(None) for l in range(circuit.depth, padded_depth)]
        circuit_array += self.measurement_encoding(circuit)
        circuit_array = _np.array(circuit_array)

        return circuit_array

    def initialization_encoding(self, circuit):
        """
        This method defines the encoding of the implicit initiliziation layer at the start of a circuit.
        In this base class, we define this to be a trivial encoding: the implicit initialization layer 
        is not represented.

        Parameters
        ----------
        circuit

        Returns
        -------
        List of lists.
        A list containing lists that encode the initiliation layer. If this inititialization is not
        represented in the encoding, as here, this is an empty list.
        """
        return []

    def initialization_encoding_depth(self):
        """
        Returns the length of the initalization encoding list, i.e., the length of lists returned by
        `initialization_encoding`.
        """
        return 0

    def measurement_encoding(self, circuit):
        """
        The default encoding of the implicit measuremnt layer at the end of a circuit,
        which is to not include it in the encoding.
        """
        return []

    def measurement_encoding_depth(self): 
        """
        TODO
        """
        return 0

    def layer_encoding(self, layer):
        """
        TODO
        """
        raise NotImplementedError("Specified in a derived class!")

    def depth(self, max_circuit_depth):
        """
        
        """
        return max_circuit_depth + self.initialization_encoding_depth() + self.measurement_encoding_depth()

    def indices_for_qubits(self, qubits):
        """
        TODO
        """
        raise NotImplementedError("Specified in a derived class!")

class StandardCircuitEncoder(CircuitEncoder):
    """
    TODO
    """

    def __init__(self, pspec):
        """
        TODO
        """
        self.pspec = pspec
        all_gates = []
        for gate in pspec.gate_names:
            all_gates += list(pspec.available_gatelabels(gate, pspec.qubit_labels))
        self.gate_indexing = all_gates
        self.length = len(all_gates)

    def layer_encoding(self, layer):
        """
        TODO

        EXPECTS A CIRCUIT LAYER AS  A TUPLE OR A LIST OF LABEL OBJECTS.
        """
        assert(isinstance(layer, tuple) or isinstance(layer, list)), "The layer must be a list or tuple of label objects!"
        if layer is not None:
            encoded_layer = _np.zeros(self.length, float)
            for gate in layer:
                encoded_layer[self.gate_indexing.index(gate)] = 1
            return list(encoded_layer)
        else:
            return list(_np.zeros(self.length, float))

    def indices_for_qubits(self, qubits):
        """
        TODO
        """
        return [i for i, gate in enumerate(self.gate_indexing) if len(set(qubits).intersection(gate.qubits)) > 0]


def circuits_to_tensor(circuits, encoder, encoding_depth=None):
    """
    TODO
    """
    if encoding_depth is None: 
        max_depth = _np.max([c.depth for c in circuits])
        encoding_depth = encoder.depth(max_depth)
    
    circuits_tensor = _np.zeros((len(circuits), encoding_depth, encoder.length), float)
    for i, circuit in enumerate(circuits):
        circuits_tensor[i,:,:] = encoder(circuit, padded_depth=encoding_depth)

    return circuits_tensor

def dense_dataset_encoding(ds, n, circs=None):
    """
    TODO
    """
    if circuits is None: circuits = list(ds.keys())
    nbit_strings = [] # TO DO
    freqs_array = _np.zeros((len(circuits), 2**n), float)
    for i in _trange(len(circuits)):
        dsrow = ds[circuits[i]]
        freqs_array[i,:] =  _np.array([dsrow.counts.get((bs,), 0.) / dsrow.total for bs in nbit_strings])

    return freqs_array

def error_generator_tensors(circuits, error_generators, pspec, alpha_representation='concise'):
    """
    TODO
    """
    indices, signs = error_propagation_tensors(circuits, error_generators, pspec)
    if alpha_representation == 'matrix':
        probabilities, alphas = first_order_outcome_probabilities_tensors(circuits, error_generators, pspec, indices=indices)
    elif alpha_representation == 'concise':
        probabilities, alphas = first_order_outcome_probabilities_tensors_concise(circuits, pspec, indices, signs)        
    else:
        _warnings.NotImplementedError('No other representations have been implemented yet!')
    return {'indices':indices, 'signs':signs, 'probabilities':probabilities, 'alphas':alphas}

def circuit_error_propagation_matrices(circuit, error_generators):
    """
    Computers the `indices` and `signs` matrices used in quantum physics aware neural networks,
    for the input circuit and the error generators specified in the `error_generators` list.
    TODO : More details here.
    ----------
    circuit : Circuit
        The circuit to compute the error progagation matrices for.

    error_generators : list
        A list of error generators for which to compute the error propagation matrices.
        Each element of this list is a tuple, where the first element is the error generator
        type ('H'', 'S', 'C', or 'A') and second element of the list is a tuple of Pauli strings.
        For 'H' and 'S', this is a single Pauli string. The Pauli strings should be length-n strings
        where n is the number of qubits on which `circuit` acts.
        
    Returns
    -------
    TODO
        These error generators occur after each layer in the circuit, and the output of 
        this function specifies how each of those error generators comp
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

def error_propagation_tensors(circuits, error_generators, pspec, prior_error_generators=None, 
                              prior_indices=None, prior_signs=None):
    """
    TODO

    For more information on what a circuit's `indices` and `signs` matrix encode, see the docstring for
    the function `circuit_error_propagation_matrices`, which this function calls to compute these matrices
    for each circuit.


    """
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
        indices[:, 0:num_pregs, :] = prior_indices.copy()
        signs[:, 0:num_pregs, :] = prior_signs.copy()
       
    for i, circuit in enumerate(circuits):

        # compute the sign and indices matrices for this circuit
        indices_for_circuit, signs_for_circuit = circuit_error_propagation_matrices(circuit, error_generators)
        indices[i, 0:circuit.depth, num_pregs:] = indices_for_circuit.copy()
        signs[i, 0:circuit.depth, num_pregs:] = signs_for_circuit.copy()
    
    return indices, signs

def alpha_coefficient(i, num_qubits, tableau, bs):
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
    return _np.float64(_egptools.alpha(_tools.index_to_error_gen(i, num_qubits, as_label=True), tableau, bs).real)

def _get_tableau(circuit_or_tableau):
    """
    Helper function that returns a tableau given a Circuit or a Stim.Tableau object
    """
    if isinstance(circuit_or_tableau, _Circuit):
        return circuit_or_tableau.convert_to_stim_tableau()
    elif isinstance(circuit_or_tableau, _stim.Tableau):
        return circuit_or_tableau
    else:
        raise ValueError('Input must be a Circuit or a Stim.Tableau!')

def dense_alpha_matrix(circuit, num_qubits, populate_for_error_generators=None, existing_alpha_matrix=None):
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

    populate_for_error_generators : None or list of integers.
        If not None, the indices of the error generators for which to compute the alpha tensor for.
        If None, the entire alpha tensor is computed. If a list of integers, this function creates
        an alpha tensor that is zero except for the alphas corresponding to the end-of-circuit 
        error generators in this list.

    existing_alpha_matrix : None or numpy array of shape (2**n, 2 * 4**n)
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

    scale = 1 / 2 ** _egptools.random_support(tableau) #TODO: This might overflow
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

def first_order_outcome_probabilities_tensors(circuits, error_generators, pspec, indices=None, prior_error_generators=None, prior_alphas=None):
    """
    TODO
    """
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

def first_order_outcome_probabilities_tensors_concise(circuits, pspec, indices, signs):
    """
    TODO
    """
    num_qubits = pspec.num_qubits
    nbit_strings = [''.join(p) for p in _itertools.product('01', repeat=num_qubits)]

    shape = (indices.shape[0], 2 ** num_qubits, indices.shape[1], indices.shape[2])
    first_order_coefficients = _np.zeros(shape, float)
    probabilities = _np.zeros((len(circuits), 2 ** num_qubits), float)
      
    for i in _trange(len(circuits)):
        circuit = circuits[i]
        # Compute the error-free probabilities for each bit string.
        tableau = _get_tableau(circuit)
        scale = 1 / 2 ** _egptools.random_support(tableau) #TODO: This might overflow
        probabilities[i, :] = _np.array([_egptools.stabilizer_probability(tableau, bs) for bs in nbit_strings]).T
        unique_indices = set(indices[i,:,:].flatten())
        alphas_dict = {}
        for l, bs in enumerate(nbit_strings):
            for error_generator_index in unique_indices:
                egtype =  _tools.index_to_error_gen(error_generator_index, num_qubits)[0]
                #print(egtype)
                if egtype == 'H' and (_np.isclose(probabilities[i, l], 0.) or _np.isclose(probabilities[i, l], 1.)):
                    alpha = 0
                elif egtype == 'S' and not (_np.isclose(probabilities[i, l], 0.) or _np.isclose(probabilities[i, l], 1.)):
                    alpha = 0
                else:
                    alpha = scale * alpha_coefficient(error_generator_index, num_qubits, tableau, bs)
                alphas_dict[l, error_generator_index] = alpha


        for l, bs in enumerate(nbit_strings):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    first_order_coefficients[i, l, j, k] = alphas_dict[l, indices[i,j,k]]

    for l, bs in enumerate(nbit_strings):
        first_order_coefficients[:, l, :, :] = first_order_coefficients[:, l, :, :] * signs
    
    return probabilities, first_order_coefficients

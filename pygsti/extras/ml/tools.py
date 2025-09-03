
import numpy as np
import itertools as _itertools
import copy as _copy
import warnings as _warnings

# from ..errorgenpropagation import propagatableerrorgen as _peg
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator

# from ..errorgenpropagation import errorpropagator as _ep

from tensorflow import unique
import tensorflow as _tf

def layer_snipper_from_qubit_graph(error_gen, num_qubits, num_channels, qubit_graph_laplacian, num_hops):
    """

    """

    laplace_power = np.linalg.matrix_power(qubit_graph_laplacian, num_hops)
    nodes_within_hops = []
    for i in range(num_qubits):
        print(i)
        nodes_within_hops.append(np.arange(num_qubits)[abs(laplace_power[i, :]) > 0])

    # These next few lines assumes only 'H' and 'S' errors because it only looks at the *first* Pauli that labels 
    # the error generator, but there are two Paulis for 'A' and 'C'
    
    # The Pauli that labels the error gen, as a string of length num_qubits containing 'I', 'X', 'Y', and 'Z'.
    pauli_string = error_gen[1][0]
    print(pauli_string)
    pauli_string = pauli_string[::-1] # for reverse indexing
    # The indices of `pauli` that are not equal to 'I'.
    qubits_acted_on_by_error = np.where(np.array(list(pauli_string)) != 'I')[0]
    qubits_acted_on_by_error = list(qubits_acted_on_by_error)

    # All the qubits that are within `hops` steps, of the qubits acted on by the error, on the connectivity
    # graph of the qubits
    relevant_qubits = np.unique(np.concatenate([nodes_within_hops[i] for i in qubits_acted_on_by_error]))
    indices_for_error = np.concatenate([[num_channels * q + i for i in range(num_channels)] for q in relevant_qubits])

    return indices_for_error

        
# @_keras.utils.register_keras_serializable()
def layer_snipper_from_qubit_graph_with_lookback(error_gen, num_qubits, num_channels, qubit_graph_laplacian, 
                                                 num_hops, lookback=-1):
    """

    """
    encoding_indices_for_error = layer_snipper_from_qubit_graph(error_gen=error_gen, num_qubits=num_qubits, 
                                                                num_channels=num_channels, 
                                                                qubit_graph_laplacian= qubit_graph_laplacian,
                                                                num_hops=num_hops)

    indices_for_error = []
    for relative_layer_index in range(lookback, 1):
        indices_for_error += [[relative_layer_index, i] for i in encoding_indices_for_error]

    return np.array(indices_for_error)


# @_keras.utils.register_keras_serializable()
def layer_snipper_from_qubit_graph_with_simplified_lookback(error_gen, num_qubits, num_channels, qubit_graph_laplacian, 
                                                 num_hops, lookback=-1):
    """

    """
    encoding_indices_for_error = layer_snipper_from_qubit_graph(error_gen=error_gen, num_qubits=num_qubits, 
                                                                num_channels=num_channels, 
                                                                qubit_graph_laplacian= qubit_graph_laplacian,
                                                                num_hops=num_hops)

    indices_for_error = []
    for i in range(0, np.abs(lookback)):
        indices_for_error += list(encoding_indices_for_error + ((num_qubits * num_channels) * i))

    return np.array(indices_for_error)
    

def laplace_from_qubit_graph(adj_matrix: np.array) -> np.array: 
    """
    Returns the graph laplacian for the graph defined by a given adjacency matrix.
    """
    deg_matrix = np.diag(np.sum(adj_matrix, axis = 1))
    return deg_matrix - adj_matrix

def numberToBase(n, b):
    """
    Returns the (base-10) integer n in base b, expressed as a list (of values between 0 and b).
    """
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def padded_numberToBase4(n, length):
    """
    Returns the (base-10) integer in base 4, as a length `length` list values between 0 and 3, i.e., it 
    pads the list with 0s at the start if n can be expressed with less than `length` string in base 4.
    """
    a = numberToBase(n, 4)
    if length < len(a):
        raise ValueError('The input in base 4 is longer than the specified padding length!')
    return [0] * (length - len(a)) + a


def index_to_paulistring(i, num_qubits):
    """
    Implements the inverse of `paulitstring_to_index`.
    """
    i_to_p = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    assert (i < 4**num_qubits), "The input integer is too large!"
    return ''.join([i_to_p[i] for i in padded_numberToBase4(i, num_qubits)])
 

def paulistring_to_index(ps, num_qubits):
    """
    Maps an n-qubit Pauli operator (represented as a string, list or tuple of elements from
    {'I', 'X', 'Y', 'Z'}) to an integer.  It uses the most conventional mapping, whereby, e.g.,
    if `num_qubits` is 2, then 'II' -> 0, and 'IX' -> 1, and 'ZZ' -> 15.

    ps: str, list, or tuple. 

    num_qubits: int

    Returns
    int
    """
    idx = 0
    p_to_i = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    for i in range(num_qubits):
        idx += p_to_i[ps[num_qubits - 1 - i]] * 4**i
    return idx


# This function has not been properly vetted, and it is definitely not the best way to do this.
# It's very slow.
def up_to_weight_k_paulis(k, n):
    """
    Returns the string representation of all n-qubit Pauli operators that
    are weight 1 up to weight k (i.e., all Paulis contain at leat one and at most 
    k non-identity Paulis).
    """
    assert (k <= 2), "Only implemented up to k = 2!"
    paulis = []

    # weight 1
    for i in range(n):
        for p in ['X', 'Y', 'Z']:
            nq_pauli = n * ['I']
            nq_pauli[n - 1 - i] = p # reversed index
            paulis.append(''.join(nq_pauli))

    # weight 2
    if k > 1:
        for i in range(n):
            for j in range(i + 1, n):
                for p in ['X', 'Y', 'Z']:
                    for q in ['X', 'Y', 'Z']:
                        nq_pauli = n * ['I']
                        nq_pauli[n - 1 - i] = p # reversed index
                        nq_pauli[n - 1 - j] = q # reversed index
                        paulis.append(''.join(nq_pauli))

    return paulis

def up_to_weight_k_paulis_from_qubit_graph(k: int, n: int, qubit_graph_laplacian: np.array, num_hops: int) -> list:
    """
    Returns the string representation of all n-qubit Pauli operators that 
    are weight 1 up to weight k (i.e., all Paulis contain at least one and
    at most k non-identity Paulis) with support on qubits connected by m 
    hops in the qubit connectivity graph.

    Assumes that the device's connectivity graph is connected!!!!!

    Paulis are in reverse order: [qubit n, qubit n-1, ..., qubit 0]
    """
    assert (k <= 2), "Only implemented up to k = 2!"
    paulis = []

    # weight 1
    for i in range(n):
        for p in ['X', 'Y', 'Z']:
            nq_pauli = n * ['I']
            nq_pauli[n - 1 - i] = p # reverse indexing
            paulis.append(''.join(nq_pauli))
    
    # weight 2
    if k > 1:
        qubit_graph_laplacian = _copy.deepcopy(qubit_graph_laplacian) # Don't delete! Otherwise this function modifies the laplacian globally for some reason?
        laplace_power = np.linalg.matrix_power(qubit_graph_laplacian, num_hops)
        for i in np.arange(n):
            laplace_power[i, i] = 0
        # assert (laplace_power == 0).all(axis=1).any() == False, 'Graph must be connected'
    
        nodes_within_hops = []
        for i in range(n):
            nodes_within_hops.append(np.arange(n)[abs(laplace_power[i, :]) > 0])
    
        for i , qubit_list in enumerate(nodes_within_hops):
            unseen_qubits = qubit_list[np.where(qubit_list > i)[0]]
            for j in unseen_qubits:
                for p in ['X', 'Y', 'Z']:
                        for q in ['X', 'Y', 'Z']:
                            nq_pauli = n * ['I']
                            nq_pauli[n - 1 - i] = p # reverse indexing
                            nq_pauli[n - 1 - j] = q # reverse indexing
                            paulis.append(''.join(nq_pauli))
    
    return paulis

def up_to_weight_k_error_gens_from_qubit_graph(k: int, n: int, qubit_graph_laplacian: np.array, num_hops: int, egtypes=['H', 'S']) -> list:
    """
    Returns a list of all n-qubit error generators up to weight k, of types given in
    egtypes and based on the qubit connectivity graph, in a tuple-of-strings format.

    k: int

    n: int, the number of qubits.

    Returns
    -------
    List of error generators, represented as a tuple where the first element is 
    the error generators type (e.g., 'H' or 'S'') and the second element is a 
    tuple specifying the Pauli(s) that index that error generator.
    """
    if n is None: n = qubit_graph_laplacian.shape[0]
    relevant_paulis = up_to_weight_k_paulis_from_qubit_graph(k, n, qubit_graph_laplacian, num_hops)
    error_generators = []
    for egtype in egtypes:
        error_generators += [(egtype, (p,)) for p in relevant_paulis]
    return error_generators
    
def up_to_weight_k_error_gens(k, n, egtypes=['H', 'S']):
    """
    Returns a list of all n-qubit error generators up to weight k, of types given in
    egtypes, in a tuple-of-strings format.

    k: int

    n: int, the number of qubits.

    Returns
    -------
    List of error generators, represented as a tuple where the first element is 
    the error generators type (e.g., 'H' or 'S'') and the second element is a 
    tuple specifying the Pauli(s) that index that error generator.
    """
    relevant_paulis = up_to_weight_k_paulis(k, n)
    error_generators = []
    for egtype in egtypes:
        error_generators += [(egtype, (p,)) for p in relevant_paulis]
    return error_generators


def error_gen_to_index(typ, paulis):
    """
    A function that *defines* an indexing of the primitive error generators. Currently
    specifies indexing for all 'H' and 'S' errors. In future, will add 'C' and 'A' 
    error generators, but will maintain current indexing for 'H' and 'S'.
    
    typ: 'H' or 'S', specifying the tuype of primitive error generator
    
    paulis: tuple, single element tuple, containing a string specifying the Pauli
        the labels the 'H' or 'S' error. The string's length implicitly 
        defines the number of qubit that the error gen acts on
    """
    assert isinstance(paulis, tuple)
    p1 = paulis[0]
    n = len(p1)
    if typ == 'H':
        base = 0
    elif typ == 'S':
        base = 4**n  
    else:
        raise ValueError('Invalid error generator specification! Note "C" and "A" errors are not implemented yet.') 
    # Future to do: C and A errors
    return base + paulistring_to_index(p1, n)

def index_to_error_gen(i, n):
    """
    Maps from the index to the 'label' representation of an elementary
    error generator.
    """
    if i < 4**n:
        typ = 'H'
        paulis = [index_to_paulistring(i, n)]
    elif i < 2 * 4**n:
        typ = 'S'
        paulis = [index_to_paulistring(i - 4**n, n)]
    # Future to do: implement C and A error generators
    else:
        raise ValueError('Invalid index!')
        
    return typ, paulis






def create_error_propagation_matrix(c, error_gens, stim_dict = None):
    """
    Computes how the errors in error_gens propogate through the circuit c.
    
    c:
    
    error_gens: a list specifying the error generators to insert after each circuit
        layer and propagate through the circuit. These primative error generators 
    """
    # error_gen_objs = [_peg.propagatableerrorgen(egen[0], egen[1], 1) for egen in error_gens]

    error_propagator = ErrorGeneratorPropagator(None)
    stim_layers = error_propagator.construct_stim_layers(c, drop_first_layer=True)
    propagation_layers = error_propagator.construct_propagation_layers(stim_layers)
    error_gen_objs = {LocalStimErrorgenLabel.cast(lbl):1 for lbl in error_gens} # dict to iterate over

    errorgen_layers = [error_gen_objs] * (len(c) - 0) # could be 1, tbd

    propagated_errorgen_layers = error_propagator._propagate_errorgen_layers(errorgen_layers, propagation_layers, include_spam=False) # list of dicts of error generators

    # indices references error generators
    # create matrix same shape and indices and signs and populate with alpha calculations for each index. this is bitstring dependent. for small qubits, can go ahead and precompute 2**n bitstrings
    # with signs and alphas, can convert to signed_alphas
    # use signed alphas as elementwise product in second half of NN
    indices, signs = [], [] # in separate function go and compute 
    for l in range(c.depth):
        indices.append([error_gen_to_index(err.errorgen_type, err.bel_to_strings()) 
                        for err in propagated_errorgen_layers[l]])
        signs.append([np.sign(val) for val in propagated_errorgen_layers[l].values()])

    indices = np.array(indices)
    signs = np.array(signs)

    return indices, signs

# def remap_indices(c_indices):
#     # Takes in a permutation matrix for a circuit and
#     # remaps the entries.
#     # e.g., P = [[1,256], [1, 0]]   ---> [[0, 1], [0, 3]]
#     flat_indices = c_indices.flatten()
#     unique_values, idx = np.unique(flat_indices, return_inverse = True)
#     return idx.reshape(c_indices.shape)

def clockwise_cnot(g):
    return (g.qubits[0] - g.qubits[1]) % num_qubits == num_qubits - 1

def gate_to_index(g, q):
    assert(q in g.qubits)
    if g.name == 'Gxpi2':
        return 0
    elif g.name == 'Gypi2':
        return 1
    elif g.name == 'Gcnot':
        qs = g.qubits
        if q == g.qubits[0] and clockwise_cnot(g):
            return 2
        if q == g.qubits[1] and clockwise_cnot(g):
            return 3
        if q == g.qubits[0] and not clockwise_cnot(g):
            return 4
        if q == g.qubits[1] and not clockwise_cnot(g):
            return 5
    else:
        raise ValueError('Invalid gate name for this encoding!')
        
# def layer_to_matrix(layer):
#     mat = np.zeros((num_qubits, num_channels), float)
#     for g in layer:
#         for q in g.qubits:
#             mat[q, gate_to_index(g, q)] = 1
#     return mat

# def circuit_to_tensor(circ, depth=None):
    
#     if depth is None: depth = circ.depth
#     ctensor = np.zeros((num_qubits, depth, num_channels), float)
#     for i in range(circ.depth):
#         ctensor[:, i, :] = layer_to_matrix(circ.layer(i))
#     return ctensor

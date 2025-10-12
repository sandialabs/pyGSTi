
import numpy as _np
import itertools as _itertools
import copy as _copy
import warnings as _warnings

from . import tools as _tools

import warnings as _warning

# from ..errorgenpropagation import propagatableerrorgen as _peg
# from ..errorgenpropagation import errorpropagator as _ep

from tensorflow import unique
import tensorflow as _tf


def ring_adj_matrix(num_qubits: int):
    adj_matrix = _np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        adj_matrix[i, (i-1) % num_qubits] = 1
        adj_matrix[i, (i+1) % num_qubits] = 1
    return adj_matrix

def melbourne_adj_matrix(num_qubits = 14):
    assert(num_qubits == 14), "We only support 5 qubits"
    adj_matrix = _np.zeros((num_qubits, num_qubits))
    adj_matrix[0,1] = 1
    adj_matrix[1,2], adj_matrix[1,13] = 1,1
    adj_matrix[2,3], adj_matrix[2,12] = 1,1
    adj_matrix[3,4], adj_matrix[3,11] = 1,1
    adj_matrix[4,5], adj_matrix[4,10] = 1,1
    adj_matrix[5,6], adj_matrix[5,9] = 1,1
    adj_matrix[6,8] = 1
    adj_matrix[7,8] = 1
    adj_matrix[8,9] = 1
    adj_matrix[9,10] = 1
    adj_matrix[10,11] = 1
    adj_matrix[11,12] = 1
    adj_matrix[12,13] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix

def bowtie_adj_matrix(num_qubits = 5):
    '''
    Builds the adjacency matrix for a five-qubit bowtie graph:

    0 - 1
     \ /
      2
     / \
    3 - 4 
    '''
    assert(num_qubits == 5), "We only support 5 qubits"
    adj_matrix = _np.zeros((num_qubits, num_qubits))
    adj_matrix[0, 1], adj_matrix[0, 2] = 1, 1
    adj_matrix[1, 2] = 1
    adj_matrix[2, 3], adj_matrix[2, 4] = 1, 1
    adj_matrix[3, 4] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix

def t_bar_adj_matrix(num_qubits = 5):
    '''
    Builds the adjacency matrix for a five-qubit T-bar graph:

    0 - 1 - 2
        |
        3
        |
        4
    '''
    assert(num_qubits == 5), "We only support 5 qubits"
    adj_matrix = _np.zeros((num_qubits, num_qubits))
    adj_matrix[0, 1] = 1
    adj_matrix[1, 2], adj_matrix[1, 3] = 1, 1
    adj_matrix[3, 4] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix

def algiers_t_bar_adj_matrix():
    '''
    Builds the adjacency matrix for a five-qubit T-bar graph:

    0 - 1 - 4
        |
        2
        |
        3
    '''
    adj_matrix = _np.zeros((5,5))
    adj_matrix[0, 1] = 1
    adj_matrix[1, 2], adj_matrix[1, 4] = 1, 1
    adj_matrix[2, 3] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return adj_matrix

def laplace_from_qubit_graph(adj_matrix: _np.array) -> _np.array: 
    """
    Returns the graph laplacian for the graph defined by a given adjacency matrix.
    """
    deg_matrix = _np.diag(_np.sum(adj_matrix, axis = 1))
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

def up_to_weight_k_paulis_from_qubit_graph(k: int, n: int, qubit_graph_laplacian: _np.array, num_hops: int) -> list:
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
        laplace_power = _np.linalg.matrix_power(qubit_graph_laplacian, num_hops)
        for i in _np.arange(n):
            laplace_power[i, i] = 0
        # assert (laplace_power == 0).all(axis=1).any() == False, 'Graph must be connected'
    
        nodes_within_hops = []
        for i in range(n):
            nodes_within_hops.append(_np.arange(n)[abs(laplace_power[i, :]) > 0])
    
        for i , qubit_list in enumerate(nodes_within_hops):
            unseen_qubits = qubit_list[_np.where(qubit_list > i)[0]]
            for j in unseen_qubits:
                for p in ['X', 'Y', 'Z']:
                        for q in ['X', 'Y', 'Z']:
                            nq_pauli = n * ['I']
                            nq_pauli[n - 1 - i] = p # reverse indexing
                            nq_pauli[n - 1 - j] = q # reverse indexing
                            paulis.append(''.join(nq_pauli))
    
    return paulis

def up_to_weight_k_error_gens_from_qubit_graph(k: int, n: int, qubit_graph_laplacian: _np.array, num_hops: int, egtypes=['H', 'S']) -> list:
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
    error_gen_objs = [_peg.propagatableerrorgen(egen[0], egen[1], 1) for egen in error_gens]
    propagated_errors = _ep.ErrorPropagator(c, error_gen_objs, NonMarkovian=True, ErrorLayerDef=True, stim_dict = stim_dict)
    
    indices, signs = [], []
    for l in range(c.depth):
        indices.append([error_gen_to_index(err.errorgen_type, err.basis_element_labels) 
                        for err in propagated_errors[l][0]])
        signs.append([_np.sign(err.error_rate.real) for err in propagated_errors[l][0]])

    indices = _np.array(indices)
    signs = _np.array(signs)

    return indices, signs

def remap_indices(c_indices):
    # Takes in a permutation matrix for a circuit and
    # remaps the entries.
    # e.g., P = [[1,256], [1, 0]]   ---> [[0, 1], [0, 3]]
    flat_indices = c_indices.flatten()
    unique_values, idx = _np.unique(flat_indices, return_inverse = True)
    return idx.reshape(c_indices.shape)

### DEPRECATED: Use create_input_data in encoding.py ###
    
def create_input_data(circs, fidelities, tracked_error_gens: list, num_channels: int,
    _qubits: int, max_depth=None, return_separate = False, stimDict = None):
    _warning.warn('newtools create_input_data is deprecated. Use the version in encoding.py.')
    if max_depth is None: max_depth = _np.max([c.depth for c in circs])
    print(max_depth)
    
    numchannels = num_channels
    numqubits = num_qubits
    numcircs = len(circs)
    num_error_gens = len(tracked_error_gens)

    x_circs = _np.zeros((numcircs, numqubits, max_depth, numchannels), float)
    x_signs = _np.zeros((numcircs, num_error_gens, max_depth), int)
    x_indices = _np.zeros((numcircs, num_error_gens, max_depth), int)
    y = _np.array(fidelities)
                    
    for i, c in enumerate(circs):
        if i % 200 == 0:
            print(i, end=',')
        x_circs[i, :, :, :] = _tools.circuit_to_tensor(c, max_depth)              
        c_indices, c_signs = create_error_propagation_matrix(c, tracked_error_gens, stimDict = stimDict)
        # c_indices = remap_indices(c_indices)
        x_indices[i, :, 0:c.depth] = c_indices.T # deprecated: np.rint(c_indices)
        x_signs[i, :, 0:c.depth] = c_signs.T # deprecated: np.rint(c_signs)
        
    if return_separate:
        return x_circs, x_signs, x_indices, y

    else:
        len_gate_encoding = numqubits * numchannels
        xc_reshaped = _np.zeros((x_circs.shape[0], x_circs.shape[1] * x_circs.shape[3], x_circs.shape[2]), float) 
        for qi in range(4):
            for ci in range(6):
                xc_reshaped[:, qi * num_channels + ci, :] = x_circs[:, qi, :, ci].copy()

        xi2 = _np.transpose(x_indices, (0, 2, 1))
        xc2 = _np.transpose(xc_reshaped, (0, 2, 1))
        xs2 = _np.transpose(x_signs, (0, 2, 1))

        xt = _np.zeros((xi2.shape[0], xi2.shape[1], 2 * num_error_gens + len_gate_encoding), float)
        xt[:, :, 0:len_gate_encoding] = xc2[:, :, :]
        xt[:, :, len_gate_encoding:num_error_gens + len_gate_encoding] = xi2[:, :, :]
        xt[:, :, num_error_gens + len_gate_encoding:2 * num_error_gens + len_gate_encoding] = xs2[:, :, :]

        return xt, y

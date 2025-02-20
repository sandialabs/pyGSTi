
import numpy as _np
import itertools as _itertools
import copy as _copy
import warnings as _warnings

# from ..errorgenpropagation import propagatableerrorgen as _peg
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator

# from ..errorgenpropagation import errorpropagator as _ep

from tensorflow import unique
import tensorflow as _tf

def grid_adj_matrix(grid_width: int):
    num_qubits = grid_width**2
    adj_matrix = _np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        if i % grid_width == grid_width - 1 and i != grid_width*grid_width - 1:
            # far right column, not the bottom left corner
            # print(i)
            # print('first')
            adj_matrix[i, i+grid_width] = 1
        elif i // grid_width == grid_width - 1 and i != grid_width*grid_width - 1:
            # bottom row, not the bottom left corner
            adj_matrix[i, i+1] = 1
        elif i != num_qubits - 1:
            # print(i)
            # print('third')
            # not the bottom right corner
            adj_matrix[i, i+grid_width] = 1
            adj_matrix[i, i+1] = 1
    adj_matrix = adj_matrix + adj_matrix.T
    return  adj_matrix

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
    # error_gen_objs = [_peg.propagatableerrorgen(egen[0], egen[1], 1) for egen in error_gens]

    error_propagator = ErrorGeneratorPropagator(None)
    stim_layers = error_propagator.construct_stim_layers(c, None, drop_first_layer=True)
    propagation_layers = error_propagator.construct_propagation_layers(stim_layers)
    error_gen_objs = {LocalStimErrorgenLabel.cast(lbl):1 for lbl in error_gens} # dict to iterate over

    errorgen_layers = [error_gen_objs] * (len(c) - 0) # could be 1, tbd

    propagated_errorgen_layers = error_propagator._propagate_errorgen_layers(errorgen_layers, propagation_layers, include_spam=False) # list of dicts of error generators

    # propagated_errors = _ep.ErrorPropagator(c, error_gen_objs, NonMarkovian=True, ErrorLayerDef=True, stim_dict = stim_dict)

    # print(propagated_errorgen_layers)

    # indices references error generators
    # create matrix same shape and indices and signs and populate with alpha calculations for each index. this is bitstring dependent. for small qubits, can go ahead and precompute 2**n bitstrings
    # with signs and alphas, can convert to signed_alphas
    # use signed alphas as elementwise product in second half of NN
    indices, signs = [], [] # in separate function go and compute 
    for l in range(c.depth):
        indices.append([error_gen_to_index(err.errorgen_type, err.bel_to_strings()) 
                        for err in propagated_errorgen_layers[l]])
        signs.append([_np.sign(val) for val in propagated_errorgen_layers[l].values()])

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
        
def layer_to_matrix(layer):
    mat = np.zeros((num_qubits, num_channels), float)
    for g in layer:
        for q in g.qubits:
            mat[q, gate_to_index(g, q)] = 1
    return mat

def circuit_to_tensor(circ, depth=None):
    
    if depth is None: depth = circ.depth
    ctensor = np.zeros((num_qubits, depth, num_channels), float)
    for i in range(circ.depth):
        ctensor[:, i, :] = layer_to_matrix(circ.layer(i))
    return ctensor

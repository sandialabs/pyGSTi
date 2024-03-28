
import numpy as _np
import itertools as _itertools

from . import tools as _tools

from ..errorgenpropagation import propagatableerrorgen as _peg
from ..errorgenpropagation import errorpropagator as _ep


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
            nq_pauli[n - 1 - i] = p
            paulis.append(''.join(nq_pauli))

    # weight 2
    if k > 1:
        for i in range(n):
            for j in range(i + 1, n):
                for p in ['X', 'Y', 'Z']:
                    for q in ['X', 'Y', 'Z']:
                        nq_pauli = n * ['I']
                        nq_pauli[n - 1 - i] = p
                        nq_pauli[n - 1 - j] = q
                        paulis.append(''.join(nq_pauli))

    return paulis


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
        raise ValueError('Invali error generator specification! Note "C" and "A" errors are not implemented yet.') 
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


def create_error_propagation_matrix(c, error_gens):
    """
    Computes how the errors in error_gens propogate through the circuit c.
    
    c:
    
    error_gens: a list specifying the error generators to insert after each circuit
        layer and propagate through the circuit. These primative error generators 
    """
    error_gen_objs = [_peg.propagatableerrorgen(egen[0], egen[1], 1) for egen in error_gens]
    propagated_errors = _ep.ErrorPropagator(c, error_gen_objs, NonMarkovian=True, ErrorLayerDef=True)
    
    indices, signs = [], []
    for l in range(c.depth):
        indices.append([error_gen_to_index(err.errorgen_type, err.basis_element_labels) 
                        for err in propagated_errors[l][0]])
        signs.append([_np.sign(err.error_rate.real) for err in propagated_errors[l][0]])

    indices = _np.array(indices)
    signs = _np.array(signs)

    return indices, signs

def create_input_data(circs, fidelities, tracked_error_gens: list, num_channels: int, num_qubits: int, max_depth=None):
    
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
        if i % 25 == 0:
            print(i,end=',')
        x_circs[i, :, :, :] = _tools.circuit_to_tensor(c, max_depth)              
        c_indices, c_signs = create_error_propagation_matrix(c, tracked_error_gens)
        x_indices[i, :, 0:c.depth] = c_indices.T # deprecated: np.rint(c_indices)
        x_signs[i, :, 0:c.depth] = c_signs.T # deprecated: np.rint(c_signs)
        
    return x_circs, x_signs, x_indices, y

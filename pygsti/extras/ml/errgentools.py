""" Error generator manipulation tools """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as np
import itertools as _itertools
import copy as _copy
import warnings as _warnings

from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.errorgenpropagation import localstimerrorgen as _lseg


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
    Implements the inverse of `paulistring_to_index`.
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


# This is definitely not the best way to do this.
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


def error_generator_index(typ, paulis):
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

def index_to_error_gen(i, n, as_label=False):
    """
    Maps from the index to the 'label' representation of an elementary
    error generator.
    """
    if i < 4**n:
        typ = 'H'
        paulis = (index_to_paulistring(i, n),)
    elif i < 2 * 4**n:
        typ = 'S'
        paulis = (index_to_paulistring(i - 4**n, n),)
    # Future to do: implement C and A error generators
    else:
        raise ValueError('Invalid index!')

    if not as_label:
        return typ, paulis
    else:
        return _lseg.LocalStimErrorgenLabel(typ, paulis)

def num_error_generators(num_qubits):
    """
    The number of H and S type error generators
    """
    return 4 ** num_qubits - 2

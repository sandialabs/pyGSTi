"""Error generator manipulation tools.

This module defines utility functions for:
  * Converting between Pauli strings and integer indices.
  * Enumerating Pauli operators / error generators up to a given weight, optionally
    restricted by a qubit connectivity graph.
  * Defining a stable indexing scheme for elementary error generators (currently H and S).
"""
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

    Parameters
    ----------
    n : int
        Base-10 integer to convert.
    b : int
        Target base.

    Returns
    -------
    list[int]
        Digits of `n` in base `b`, most-significant digit first. Returns `[0]` when `n == 0`.
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

    Parameters
    ----------
    n : int
        Base-10 integer to convert.
    length : int
        Desired output length.

    Returns
    -------
    list[int]
        Length-`length` list of digits in `{0,1,2,3}`.

    Raises
    ------
    ValueError
        If `n` requires more than `length` base-4 digits.
    """
    a = numberToBase(n, 4)
    if length < len(a):
        raise ValueError('The input in base 4 is longer than the specified padding length!')
    return [0] * (length - len(a)) + a


def index_to_paulistring(i, num_qubits):
    """
    Implements the inverse of `paulistring_to_index` i.e. mapping an integer index to an n-qubit Pauli string.

    Parameters
    ----------
    i : int
        Integer in `[0, 4**num_qubits)`.
    num_qubits : int
        Number of qubits \(n\).

    Returns
    -------
    str
        Pauli string of length `num_qubits` over alphabet {'I','X','Y','Z'}.

    Notes
    -----
    This implements the inverse of `paulistring_to_index` under the same ordering convention.
    """
    i_to_p = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    assert (i < 4**num_qubits), "The input integer is too large! Please input an integer in `[0, 4**num_qubits)`"
    return ''.join([i_to_p[i] for i in padded_numberToBase4(i, num_qubits)])
 

def paulistring_to_index(ps, num_qubits):
    """
    Maps an n-qubit Pauli operator (represented as a string, list or tuple of elements from
    {'I', 'X', 'Y', 'Z'}) to an integer.  It uses the most conventional mapping, whereby, e.g.,
    if `num_qubits` is 2, then 'II' -> 0, and 'IX' -> 1, and 'ZZ' -> 15.

    Parameters
    ----------
    ps : str or list or tuple
        Pauli operator representation over {'I','X','Y','Z'} of length `num_qubits`.
    num_qubits : int
        Number of qubits \(n\).

    Returns
    -------
    int
        Integer index in `[0, 4**num_qubits)`.

    Notes
    -----
    Uses the common base-4 ordering where the *rightmost* character is the least-significant digit.
    For example, when `num_qubits == 2`:
      * 'II' -> 0
      * 'IX' -> 1
      * 'ZZ' -> 15
    """
    idx = 0
    p_to_i = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    for i in range(num_qubits):
        idx += p_to_i[ps[num_qubits - 1 - i]] * 4**i
    return idx


# # This is definitely not the best way to do this.
# # It's very slow.
# def up_to_weight_k_paulis(k, n):
#     """
#     Returns the string representation of all n-qubit Pauli operators that
#     are weight 1 up to weight k (i.e., all Paulis contain at leat one and at most 
#     k non-identity Paulis).
#     """
#     assert (k <= 2), "Only implemented up to k = 2!"
#     paulis = []

#     # weight 1
#     for i in range(n):
#         for p in ['X', 'Y', 'Z']:
#             nq_pauli = n * ['I']
#             nq_pauli[n - 1 - i] = p # reversed index
#             paulis.append(''.join(nq_pauli))

#     # weight 2
#     if k > 1:
#         for i in range(n):
#             for j in range(i + 1, n):
#                 for p in ['X', 'Y', 'Z']:
#                     for q in ['X', 'Y', 'Z']:
#                         nq_pauli = n * ['I']
#                         nq_pauli[n - 1 - i] = p # reversed index
#                         nq_pauli[n - 1 - j] = q # reversed index
#                         paulis.append(''.join(nq_pauli))

#     return paulis


##### Noah's Updated implementation 5/8/2026 #####
def up_to_weight_k_paulis(k: int, n: int):
    """
    Return all n-qubit Pauli strings with weight 1..k (non-identity count). If k=0, returns the all-identity string. If k>n, automatically sets k=n.

    A Pauli string is a length-n string over {'I','X','Y','Z'}.
    The *weight* is the number of non-'I' characters.

    This implementation is general for any k (clipped to n). It follows the same
    reverse-index convention as the original codebase/documentation:
    qubit i corresponds to string position (n - 1 - i). In other words, qubit 0
    is the *rightmost* character of the string.

    Parameters
    ----------
    k : int
        Positive integer. Maximum Pauli weight. Values > n are treated as n.
    n : int
        Number of qubits (string length).

    Returns
    -------
    list[str]
        All Pauli strings of weight 1..k.
    """

    if not isinstance(k, int) or k < 1:
        raise TypeError("Pauli weight must be an integer > 1.")
    
    if not isinstance(n, int) or n < 0:
        raise TypeError("Number of qubits must be a non-negative integer.")

    # Use a mutable "template" list of characters for efficient updates.
    # We will copy this list and then overwrite selected positions with X/Y/Z.
    base = list("I" * n)

    # Weight cannot exceed the number of qubits.
    if k > n:
        print("Pauli weight cannot exceed the number of qubits. Automatically setting k = min(k,n)")

    k = min(k, n)

    # These are indices into the Pauli string (0..n-1).
    #
    # Important convention note:
    #   If you interpret "qubit i" as mapping to string position (n-1-i),
    #   then the *rightmost* character corresponds to qubit 0.
    #
    # Here we simply generate strings by directly setting string indices.
    # Whether index j corresponds to qubit j or qubit (n-1-j) depends on
    # how the rest of your code interprets the string; this function just
    # produces strings with characters at indices 0..n-1.
    positions = list(range(n))

    # Collect all generated Pauli strings.
    paulis = []

    # Enumerate all weights w = 1..k
    for w in range(1, k + 1):

        # Choose which w positions (string indices) will be non-identity.
        # "support" is a tuple of length w with strictly increasing indices.
        for support in _itertools.combinations(positions, w):

            # For those w positions, choose an assignment of X/Y/Z at each position.
            # There are 3^w such assignments.
            for letters in _itertools.product("XYZ", repeat=w):

                # Copy the all-identity template, then place X/Y/Z on the support.
                s = base[:]  # shallow copy of list of characters
                for idx, P in zip(support, letters):
                    s[idx] = P

                # Convert list of characters back to a string and store.
                paulis.append("".join(s))

    return paulis

def up_to_weight_k_paulis_from_qubit_graph(
    k: int, n: int, qubit_graph_laplacian: np.ndarray, num_hops: int
) -> list[str]:
    """
    Enumerate Pauli strings of weight 1..k whose *support set* of non-identity qubits
    is connected under a "within num_hops" connectivity rule derived from the qubit graph.

    Convention used throughout this module:
        qubit i  <->  string position (n - 1 - i)
    so qubit 0 is the rightmost character in the Pauli string.

    Parameters
    ----------
    k : int
        Maximum Pauli weight.
    n : int
        Number of qubits (string length).
    qubit_graph_laplacian : numpy.ndarray
        Laplacian matrix of the qubit connectivity graph.
    num_hops : int
        Hop distance defining which qubit pairs are considered "close enough."

    Returns
    -------
    list[str]
        All Pauli strings of weight 1..k with support on qubits connected within num_hops.

    Notes
    -----
    This is a generalization of the legacy k<=2 implementation:
      - For w=1, every single-qubit Pauli is allowed.
      - For w=2, a pair (i,j) is allowed iff i and j are within num_hops hops.
      - For w>2, a support S is allowed iff the induced subgraph on S is connected
        using the "within num_hops" adjacency.
    """
    # ---- Basic input checks ----
    if not isinstance(k, int) or k < 1:
        raise TypeError("Pauli weight, k, must be an integer > 1.")
    if not isinstance(n, int) or n < 0:
        raise TypeError("n must be a non-negative integer.")

    if k > n:
        print("Pauli weight cannot exceed the number of qubits. Automatically setting k = min(k,n)")

    # Clamp k so we never ask for weight > n.
    k = min(k, n)

    # Make a private copy to avoid mutating the caller's matrix accidentally.
    L = np.array(qubit_graph_laplacian, copy=True)

    # Sanity-check dimensions: Laplacian must be n x n.
    if L.shape != (n, n):
        raise ValueError("qubit_graph_laplacian must have shape (n, n).")

    # ---- Build the "within num_hops" connectivity relation ----
    #
    # Legacy code used:
    #   laplace_power = L**num_hops
    #   nodes are considered "within hops" if laplace_power[i,j] != 0
    #
    M = np.linalg.matrix_power(L, num_hops)

    # Remove diagonal entries; we don't want to treat i as "connected to itself"
    # for the purpose of defining edges between *distinct* qubits.
    np.fill_diagonal(M, 0)

    # Convert to a boolean adjacency: close[i,j] == True means "i and j are within num_hops"
    # according to the matrix-power criterion above.
    close = (np.abs(M) > 0)

    # If the underlying graph is undirected, "close" should be symmetric.
    # This line enforces symmetry just in case numerical issues or input oddities break it.
    close = np.logical_or(close, close.T)

    # ---- Helper: test whether a proposed support set is connected ----
    def support_is_connected(support_qubits: tuple[int, ...]) -> bool:
        """
        Return True iff the induced subgraph on `support_qubits` is connected,
        where edges are given by `close`.
        """
        # Empty support shouldn't appear here (we generate weights starting at 1),
        # and a single vertex is trivially connected.
        if len(support_qubits) <= 1:
            return True

        # We'll do a simple DFS/BFS over the support set using the `close` adjacency.
        support = list(support_qubits)
        support_set = set(support)

        # Start from the first qubit in the support.
        seen = {support[0]}
        stack = [support[0]]

        # Standard depth-first traversal restricted to nodes in the support.
        while stack:
            u = stack.pop()

            # Consider only neighbors v that are:
            #   (1) in the support set
            #   (2) adjacent to u under "close"
            #   (3) not yet visited
            for v in support:
                if v not in seen and close[u, v]:
                    seen.add(v)
                    stack.append(v)

        # Connected iff we reached every vertex in the support.
        return len(seen) == len(support_set)

    # ---- Enumerate all valid Pauli strings ----
    base = ['I'] * n            # template for fast construction
    paulis = []                 # output list of Pauli strings
    qubits = range(n)           # qubit labels 0..n-1

    # Enumerate all weights w = 1..k
    for w in range(1, k + 1):

        # Enumerate all possible supports (sets of qubits) of size w.
        for support_qubits in _itertools.combinations(qubits, w):

            # Skip supports that are not connected under the within-num_hops rule.
            if not support_is_connected(support_qubits):
                continue

            # For each support, assign an X/Y/Z to each qubit in the support.
            # There are 3^w assignments.
            for letters in _itertools.product("XYZ", repeat=w):

                # Start from all-identity string, then overwrite the support positions.
                s = base[:]

                # Place each chosen letter at the appropriate *string* index.
                # Remember: qubit q corresponds to string position (n-1-q).
                for q, P in zip(support_qubits, letters):
                    s[n - 1 - q] = P

                # Convert list-of-chars to a string and store it.
                paulis.append("".join(s))

    return paulis

def up_to_weight_k_error_gens_from_qubit_graph(k: int, n: int, qubit_graph_laplacian: np.ndarray, num_hops: int, egtypes=['H', 'S']) -> list:
    """Returns a list of all n-qubit error generators up to weight k of the specified
    types (e.g., 'H' and 'S') whose supports are connected subgraphs of the qubit connectivity
    graph (where connectivity is defined by hop distance on the qubit graph).

    This function first identifies the subset of multi-qubit Pauli operators (up to weight k)
    whose active qubit supports form a connected subgraph. The notion of connectivity is
    defined by a hop distance: two qubits are considered connected if they are at most
    `num_hops` apart in the graph represented by `qubit_graph_laplacian`.

    Once the relevant Pauli operators are found, the function constructs primitive error
    generators for each Pauli operator and for each of the specified error generator types
    in `egtypes`. Typically, these include Hamiltonian-like ('H') and stochastic-like ('S')
    error generators.

    Parameters
    ----------
    k : int
        Maximum Pauli weight. That is, the maximum number of qubits on which the 
        constituent Pauli operator acts non-trivially (non-identity).
    n : int or None
        Number of qubits. If None, it is automatically inferred from the size of 
        `qubit_graph_laplacian` (specifically, `qubit_graph_laplacian.shape[0]`).
    qubit_graph_laplacian : numpy.ndarray
        The Laplacian matrix of the qubit connectivity graph. A square matrix of 
        shape (n, n) representing the graph structure.
    num_hops : int
        The maximum graph hop distance defining allowable connectivity between 
        individual qubits. Supports with a hop distance larger than `num_hops` 
        between nodes are treated as disconnected.
    egtypes : list[str], default ['H', 'S']
        A list of error generator types to generate. Standard choices are:
          * 'H' : Hamiltonian-like error generators
          * 'S' : Stochastic-like error generators

    Returns
    -------
    list[tuple]
        A list of error generator descriptors. Each descriptor is a tuple of the 
        form `(egtype, (pauli_string,))`, where:
          * `egtype` (str) is the type of the error generator (e.g., 'H' or 'S').
          * `pauli_string` (str) is the Pauli representation (e.g. 'IX', 'XY') 
            which indexes the error generator on the connected support.
    """

    # 1. Infer the number of qubits from the Laplacian shape if n is not provided (None).
    if n is None:
        n = qubit_graph_laplacian.shape[0]

    # 2. Retrieve all Pauli strings up to weight k whose supports are connected
    #    subgraphs of the qubit graph (with adjacency defined by <= num_hops).
    relevant_paulis = up_to_weight_k_paulis_from_qubit_graph(k, n, qubit_graph_laplacian, num_hops)

    # 3. Construct error generator descriptors for each requested error type and Pauli string.
    error_generators = []
    for egtype in egtypes:
        # Each error generator is represented as a tuple: (type, (pauli_string,))
        error_generators += [(egtype, (p,)) for p in relevant_paulis]

    return error_generators
    
def up_to_weight_k_error_gens(k, n, egtypes=['H', 'S']):
    """Returns a list of all n-qubit error generators up to weight k, of types given in
    egtypes, in a tuple-of-strings format.

    Parameters
    ----------
    k : int
        Maximum Pauli weight.
    n : int
        Number of qubits.
    egtypes : list[str], default ['H','S']
        Error generator types to include.

    Returns
    -------
    list[tuple]
        List of error generators in the tuple form `(egtype, (pauli_string,))`. The first element is 
    the error generators type (e.g., 'H' or 'S'') and the second element is a 
    tuple specifying the Pauli(s) that index that error generator.
    """

    relevant_paulis = up_to_weight_k_paulis(k, n)
    error_generators = []
    for egtype in egtypes:
        error_generators += [(egtype, (p,)) for p in relevant_paulis]
    return error_generators


def error_generator_index(typ, paulis):
    """A function that *defines* an indexing of the primitive error generators.Currently
    specifies indexing for all 'H' and 'S' errors. In future, will add 'C' and 'A' 
    error generators, but will maintain current indexing for 'H' and 'S'.

    Parameters
    ----------
    typ : str
        Error generator type. Currently supports:
          * 'H' : Hamiltonian-like error generators
          * 'S' : Stochastic-like error generators
    paulis : tuple
        single element tuple, containing a string specifying the Pauli
        the labels the 'H' or 'S' error. The string's length implicitly 
        defines the number of qubit that the error gen acts on

    Returns
    -------
    int
        Integer index defining the canonical ordering:
          * 'H' generators map to `[0, 4**n)`
          * 'S' generators map to `[4**n, 2*4**n)`

    Raises
    ------
    ValueError
        If `typ` is not supported.
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
    error generator. Inverse of `error_generator_index` for H/S generators.

    Parameters
    ----------
    i : int
        Error generator index.
    n : int
        Number of qubits.
    as_label : bool, default False
        If True, return a `LocalStimErrorgenLabel` instead of tuple form.

    Returns
    -------
    tuple or LocalStimErrorgenLabel
        If `as_label` is False, returns `(typ, (pauli_string,))`.
        If `as_label` is True, returns `LocalStimErrorgenLabel(typ, paulis)`.

    Raises
    ------
    ValueError
        If `i` is outside the supported range `[0, 2*4**n)`.
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
    """Return the number of indexed H/S error generators for `num_qubits`.

    Parameters
    ----------
    num_qubits : int
        Number of qubits \(n\).

    Returns
    -------
    int
        Number of generators. (As implemented here: `4**n - 2`.)
    """
    return 4 ** num_qubits - 2

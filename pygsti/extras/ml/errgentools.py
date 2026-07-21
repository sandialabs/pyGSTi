"""Error generator manipulation tools.

This module defines utility functions for:
  * Converting between Pauli strings and integer indices.
  * Enumerating Pauli operators / error generators up to a given weight, optionally
    restricted by a qubit connectivity graph.
  * Defining a stable indexing scheme for elementary error generators of all four types in
    the "Taxonomy of Small Errors" (Blume-Kohout et al.) classification: Hamiltonian ('H'),
    Stochastic-Pauli ('S'), Stochastic Pauli-Correlation ('C'), and Active ('A'). 'H' and 'S'
    generators are indexed by a single non-identity Pauli operator; 'C' and 'A' generators are
    indexed by an unordered pair of two DISTINCT non-identity Pauli operators (see
    `error_generator_index`/`index_to_error_gen` for the exact indexing scheme, and
    `canonical_pauli_pair` for the canonical-ordering convention used for 'C'/'A' pairs).
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


def numberToBase(n: int, b: int) -> list[int]:
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


def padded_numberToBase4(n: int, length: int) -> list[int]:
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


def index_to_paulistring(i: int, num_qubits: int) -> str:
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
 

def paulistring_to_index(ps: str | list | tuple, num_qubits: int) -> int:
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


##### Utilities for indexing PAIRS of Paulis, used by 'C' and 'A' type error generators #####
# 'C' (Pauli-correlation) and 'A' (active) elementary error generators are indexed by an
# *unordered* pair of two DISTINCT, non-identity Paulis (P, Q); see "A Taxonomy of Small
# Errors" (Blume-Kohout et al., PRX Quantum 3, 020335 (2022)), Sec. V.C-V.D, Eqs. 15-16. This
# is in contrast to 'H' and 'S', which are each indexed by a single Pauli. The functions below
# provide a canonical ordering convention for such pairs and a bijection between canonically-
# ordered pairs and a contiguous range of integers (mirroring `paulistring_to_index`/
# `index_to_paulistring`'s role for single Paulis).
#
# IMPORTANT ASYMMETRY: C_{P,Q} = C_{Q,P} (order doesn't matter -- see Eq. 15, which is manifestly
# symmetric under swapping P and Q), but A_{P,Q} = -A_{Q,P} (order matters up to an overall sign
# -- see Eq. 16, which flips sign under swapping P and Q, since both `PρQ - QρP` and the
# commutator `[P,Q]` are antisymmetric under the swap). This means that when *canonicalizing* an
# already-existing 'A'-type label into this module's canonical pair order (e.g. after an 'A'
# generator has been propagated through a circuit -- see `encoding.circuit_error_propagation_matrices`),
# callers MUST also track and apply a compensating sign flip whenever a swap occurs; see
# `canonical_pauli_pair`'s return value and `error_generator_canonicalization_sign` below. This
# subtlety does not apply to 'C' (no sign correction is ever needed for 'C').


def canonical_pauli_pair(p1: str, p2: str) -> tuple[str, str, bool]:
    """
    Returns the canonical (lexicographically sorted) ordering of the unordered pair `{p1, p2}`,
    which is the ordering convention used by `pauli_pair_to_index`/`error_generator_index` for
    'C' and 'A' type error generators.

    Parameters
    ----------
    p1, p2 : str
        Two DISTINCT n-qubit Pauli strings (over {'I','X','Y','Z'}) indexing a 'C' or 'A' type
        error generator. Identity-checking (i.e. that neither is the all-identity string) is
        *not* performed here; that is the caller's responsibility (see `error_generator_index`).

    Returns
    -------
    P, Q : str
        `p1` and `p2`, reordered (if necessary) so that `P < Q` as Python strings (equivalently,
        ordinary lexicographic order over {'I','X','Y','Z'}, treating 'I' < 'X' < 'Y' < 'Z').
    was_swapped : bool
        True if and only if `(p1, p2) != (P, Q)`, i.e. a swap was needed to reach canonical
        order. Needed by callers that must track the associated sign for 'A'-type generators.

    Raises
    ------
    ValueError
        If `p1 == p2` (a 'C' or 'A' type error generator must be indexed by two *distinct*
        Paulis; see "A Taxonomy of Small Errors" Sec. V.C-V.D).
    """
    if p1 == p2:
        raise ValueError(
            "'C' and 'A' type error generators must be indexed by two DISTINCT Paulis! "
            f"Got the same Pauli twice: {p1!r}."
        )
    if p1 < p2:
        return p1, p2, False
    else:
        return p2, p1, True


def error_generator_canonicalization_sign(typ: str, paulis: tuple[str, ...]) -> int:
    """
    Returns the sign correction (+1 or -1) that must be applied when canonicalizing the given
    error generator label's Pauli(s) into the canonical ordering used internally by
    `error_generator_index` (see `canonical_pauli_pair`).

    This is only ever nontrivial for 'A'-type (active) generators: since `A_{P,Q} = -A_{Q,P}`
    (antisymmetric under swapping its two indexing Paulis -- see module-level notes above),
    reindexing an 'A' generator whose Paulis happen to be given out of canonical order requires
    also flipping the sign of its rate/coefficient to compensate. 'C'-type generators are
    symmetric under swapping (`C_{P,Q} = C_{Q,P}`), so no sign correction is ever needed for
    them, and 'H'/'S' are each indexed by only a single Pauli (no ordering ambiguity at all).

    Parameters
    ----------
    typ : str
        Error generator type: 'H', 'S', 'C', or 'A'.
    paulis : tuple
        The Pauli(s) indexing the error generator, as actually given/observed (e.g. as returned
        by `LocalStimErrorgenLabel.bel_to_strings()` for a propagated error generator) -- NOT
        necessarily already in canonical order. For 'H'/'S' this is a 1-tuple; for 'C'/'A' this
        is a 2-tuple `(p1, p2)`.

    Returns
    -------
    int
        +1, unless `typ == 'A'` and `(paulis[0], paulis[1])` is not already in canonical
        (lexicographically sorted) order, in which case -1.
    """
    if typ == 'A':
        _, _, was_swapped = canonical_pauli_pair(paulis[0], paulis[1])
        return -1 if was_swapped else 1
    return 1


def num_nonidentity_paulis(n: int) -> int:
    """
    Returns the number of non-identity n-qubit Pauli strings, i.e. `4**n - 1`.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    int
    """
    return 4**n - 1


def num_pauli_pairs(n: int) -> int:
    """
    Returns the number of unordered pairs `{P, Q}` of two DISTINCT, non-identity n-qubit Pauli
    strings, i.e. the number of 'C' (or, separately, 'A') type error generators for `n` qubits:
    `(4**n - 1)*(4**n - 2)/2`. (Matches "A Taxonomy of Small Errors" Eq. 8's count of
    `(d**2-1)*(d**2-2)/2` with `d = 2**n`; e.g. for `n=2`, this gives 105, matching the paper's
    own statement in Sec. V.G that "there are 105 linearly independent two-qubit Pauli-
    correlation generators".)

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    int
    """
    k = num_nonidentity_paulis(n)
    return k * (k - 1) // 2


def pauli_pair_to_index(p1: str, p2: str, n: int) -> int:
    """
    Maps an unordered pair `{p1, p2}` of two DISTINCT, non-identity n-qubit Pauli strings to a
    unique integer index in `[0, num_pauli_pairs(n))`. This is the pair-indexing analog of
    `paulistring_to_index` (used for the single-Pauli-indexed 'H'/'S' types), and is used
    internally by `error_generator_index` for 'C'/'A' types.

    Parameters
    ----------
    p1, p2 : str
        Two DISTINCT, non-identity n-qubit Pauli strings. Order does not matter (the pair is
        first canonicalized via `canonical_pauli_pair`).
    n : int
        Number of qubits.

    Returns
    -------
    int
        Integer index in `[0, num_pauli_pairs(n))`.

    Notes
    -----
    Implementation: each non-identity Pauli string is mapped via `paulistring_to_index` to
    `[1, 4**n)` and then shifted down by 1 to `[0, K)` where `K = num_nonidentity_paulis(n)`.
    The canonically-ordered pair of shifted indices `(a, b)`, `a < b`, is then given the
    standard lexicographic "rank of an unordered pair" index:
    `rank(a,b) = a*(K-1) - a*(a-1)//2 + (b-a-1)`, which ranges over `[0, K*(K-1)/2)` as
    `(a,b)` ranges over all pairs with `0 <= a < b < K`.
    """
    P, Q, _ = canonical_pauli_pair(p1, p2)
    identity = 'I' * n
    if P == identity or Q == identity:
        raise ValueError("'C'/'A' error generators must be indexed by two NON-IDENTITY Paulis!")
    a = paulistring_to_index(P, n) - 1
    b = paulistring_to_index(Q, n) - 1
    k = num_nonidentity_paulis(n)
    return a * (k - 1) - a * (a - 1) // 2 + (b - a - 1)


def index_to_pauli_pair(idx: int, n: int) -> tuple[str, str]:
    """
    Inverse of `pauli_pair_to_index`. Maps an integer index in `[0, num_pauli_pairs(n))` to the
    canonically-ordered pair `(P, Q)`, `P < Q`, of non-identity n-qubit Pauli strings it
    represents.

    Parameters
    ----------
    idx : int
        Integer in `[0, num_pauli_pairs(n))`.
    n : int
        Number of qubits.

    Returns
    -------
    P, Q : str
        The canonically-ordered (`P < Q`) pair of Pauli strings corresponding to `idx`.

    Notes
    -----
    Uses an integer-only binary search (rather than the closed-form quadratic-formula inverse of
    the triangular-number-based rank, which risks floating-point precision loss for large `n`)
    to invert the `rank(a,b)` formula documented in `pauli_pair_to_index`.
    """
    k = num_nonidentity_paulis(n)
    total = k * (k - 1) // 2
    if not (0 <= idx < total):
        raise ValueError(f"idx must be in [0, {total}), got {idx}.")

    def offset(a: int) -> int:
        # Number of pairs (a', b') with a' < a, for a' ranging over [0, k-1).
        return a * (k - 1) - a * (a - 1) // 2

    lo, hi = 0, k - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offset(mid) <= idx:
            lo = mid
        else:
            hi = mid - 1
    a = lo
    b = a + 1 + (idx - offset(a))
    P = index_to_paulistring(a + 1, n)
    Q = index_to_paulistring(b + 1, n)
    return P, Q


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
# All 15 valid (non-'both-identity') local (single-qubit) `(P_letter, Q_letter)` choices used
# when enumerating pairs of Paulis for 'C'/'A' error generators (see `_pauli_pairs_for_support`
# below): 3 where only P is non-trivial at this qubit, 3 where only Q is, and 9 where both are
# (letters need not differ locally -- only the *overall* n-qubit strings P and Q must differ).
_PAULI_PAIR_LOCAL_CHOICES: list[tuple[str, str]] = (
    [(p, 'I') for p in 'XYZ'] + [('I', q) for q in 'XYZ'] + [(p, q) for p in 'XYZ' for q in 'XYZ']
)


def _pauli_pairs_for_support(support: tuple[int, ...], n: int, reverse_index: bool) -> list[tuple[str, str]]:
    """
    Enumerate all canonically-ordered pairs `(P, Q)`, `P < Q` lexicographically, of DISTINCT,
    non-identity n-qubit Pauli strings such that `support(P) union support(Q)` equals `support`
    EXACTLY (not merely a subset) -- i.e. every qubit in `support` is acted on non-trivially by
    at least one of P, Q, and every qubit outside `support` is 'I' in both.

    This is the combinatorial core shared by `up_to_weight_k_pauli_pairs` and
    `up_to_weight_k_pauli_pairs_from_qubit_graph`, which each call this once per candidate
    support set found by their own (differently-indexed/oriented) outer loops.

    Parameters
    ----------
    support : tuple[int, ...]
        Qubit indices (positions in `[0, n)`) on which the pair's union support must lie,
        exactly.
    n : int
        Number of qubits (string length).
    reverse_index : bool
        If True, qubit `q` maps to string position `n - 1 - q` (matching the convention used by
        `up_to_weight_k_paulis_from_qubit_graph`); if False, qubit `q` maps directly to string
        position `q` (matching the convention used by `up_to_weight_k_paulis`). This mirrors an
        existing (pre-existing, not introduced here) inconsistency between those two single-
        Pauli functions; each of this module's two pair-enumeration functions below passes
        whichever value keeps it consistent with its own single-Pauli counterpart.

    Returns
    -------
    list[tuple[str, str]]
        All valid canonically-ordered `(P, Q)` pairs with union-support exactly `support`.

    Notes
    -----
    For a fixed support of size `w`, there are `15**w` raw per-qubit-choice combinations. Of
    these, the ones to exclude are: all-P-only (which would make Q the identity, disallowed),
    all-Q-only (which would make P the identity, disallowed), and all-'both'-with-matching-
    letters (which would make P == Q, disallowed) -- these three cases are disjoint for `w >= 1`
    (each forces the *other* two Pauli(s) to be manifestly non-identity/distinct), so by
    inclusion-exclusion there are `15**w - 3*3**w = 15**w - 3**(w+1)` valid *ordered* pairs, and
    half that many *unordered* (canonically-ordered) pairs, since swapping the roles of P and Q
    in the per-qubit choices always produces another valid combination representing the same
    unordered pair. This was cross-checked (during development) against
    `num_pauli_pairs(n)` (the *unrestricted* total, obtained here by summing over all supports
    of all sizes `1..n`) for small `n`, and against the paper's own stated total of 105 for
    `n=2`; see the corresponding unit tests.
    """
    w = len(support)
    pairs = []
    base = ['I'] * n
    for combo in _itertools.product(_PAULI_PAIR_LOCAL_CHOICES, repeat=w):
        p_letters = [c[0] for c in combo]
        q_letters = [c[1] for c in combo]
        if all(letter == 'I' for letter in p_letters):
            continue  # P would be the (disallowed) global identity
        if all(letter == 'I' for letter in q_letters):
            continue  # Q would be the (disallowed) global identity
        p_chars = base[:]
        q_chars = base[:]
        for pos, p_letter, q_letter in zip(support, p_letters, q_letters):
            string_idx = (n - 1 - pos) if reverse_index else pos
            p_chars[string_idx] = p_letter
            q_chars[string_idx] = q_letter
        P = ''.join(p_chars)
        Q = ''.join(q_chars)
        if P == Q:
            continue  # disallowed: 'C'/'A' generators require two DISTINCT Paulis
        if P < Q:
            # Only emit when already in canonical order; the "mirror" combo (with P's and Q's
            # per-qubit roles swapped) will independently satisfy P < Q for the other member of
            # this unordered pair, so each unordered pair is emitted exactly once overall.
            pairs.append((P, Q))
    return pairs


def up_to_weight_k_paulis(k: int, n: int) -> list[str]:
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


def up_to_weight_k_pauli_pairs(k: int, n: int) -> list[tuple[str, str]]:
    """
    Return all unordered pairs `(P, Q)` (canonically ordered, `P < Q`) of DISTINCT, non-identity
    n-qubit Pauli strings such that the *union* of P's and Q's individual qubit supports has
    weight (size) 1..k. This is the pair-indexed analog of `up_to_weight_k_paulis`, used to
    enumerate the Pauli-pairs indexing 'C' (Pauli-correlation) and 'A' (active) type error
    generators (see "A Taxonomy of Small Errors", Sec. V.C-V.D, and this module's docstring).

    Follows the same indexing convention (no reverse-indexing; qubit `i` maps directly to string
    position `i`) as `up_to_weight_k_paulis`.

    Parameters
    ----------
    k : int
        Positive integer. Maximum weight (size of `support(P) union support(Q)`). Values > n
        are treated as n.
    n : int
        Number of qubits (string length).

    Returns
    -------
    list[tuple[str, str]]
        All valid canonically-ordered `(P, Q)` pairs with union-weight 1..k.
    """
    if not isinstance(k, int) or k < 1:
        raise TypeError("Pauli weight must be an integer > 1.")

    if not isinstance(n, int) or n < 0:
        raise TypeError("Number of qubits must be a non-negative integer.")

    if k > n:
        print("Pauli weight cannot exceed the number of qubits. Automatically setting k = min(k,n)")

    k = min(k, n)

    positions = list(range(n))
    pairs: list[tuple[str, str]] = []

    # Enumerate all union-weights w = 1..k
    for w in range(1, k + 1):
        # Choose which w positions form the union support, exactly (see
        # `_pauli_pairs_for_support`'s docstring for why "exactly" avoids double-counting across
        # different values of w).
        for support in _itertools.combinations(positions, w):
            pairs.extend(_pauli_pairs_for_support(support, n, reverse_index=False))

    return pairs


def _qubit_graph_close_matrix(qubit_graph_laplacian: np.ndarray, n: int, num_hops: int) -> np.ndarray:
    """
    Computes the boolean "within num_hops" adjacency matrix for a qubit connectivity graph,
    given its Laplacian. Shared helper used by both `up_to_weight_k_paulis_from_qubit_graph` and
    `up_to_weight_k_pauli_pairs_from_qubit_graph` to determine which candidate multi-qubit
    supports are "connected" (and hence allowed) under the graph-locality restriction.

    Parameters
    ----------
    qubit_graph_laplacian : numpy.ndarray
        Laplacian matrix of the qubit connectivity graph, of shape `(n, n)`.
    n : int
        Number of qubits.
    num_hops : int
        Hop distance defining which qubit pairs are considered "close enough."

    Returns
    -------
    numpy.ndarray
        Boolean `(n, n)` matrix; `close[i, j]` is True iff qubits `i` and `j` (`i != j`) are
        within `num_hops` hops of each other.
    """
    # Make a private copy to avoid mutating the caller's matrix accidentally.
    L = np.array(qubit_graph_laplacian, copy=True)

    # Sanity-check dimensions: Laplacian must be n x n.
    if L.shape != (n, n):
        raise ValueError("qubit_graph_laplacian must have shape (n, n).")

    # Legacy code used:
    #   laplace_power = L**num_hops
    #   nodes are considered "within hops" if laplace_power[i,j] != 0
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
    return close


def _support_is_connected(support_qubits: tuple[int, ...], close: np.ndarray) -> bool:
    """
    Returns True iff the induced subgraph on `support_qubits` is connected, where edges are
    given by the boolean adjacency matrix `close` (see `_qubit_graph_close_matrix`).

    Parameters
    ----------
    support_qubits : tuple[int, ...]
        Qubit indices forming the candidate support set.
    close : numpy.ndarray
        Boolean `(n, n)` "within num_hops" adjacency matrix, as returned by
        `_qubit_graph_close_matrix`.

    Returns
    -------
    bool
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

    # ---- Build the "within num_hops" connectivity relation ----
    close = _qubit_graph_close_matrix(qubit_graph_laplacian, n, num_hops)

    # ---- Enumerate all valid Pauli strings ----
    base = ['I'] * n            # template for fast construction
    paulis = []                 # output list of Pauli strings
    qubits = range(n)           # qubit labels 0..n-1

    # Enumerate all weights w = 1..k
    for w in range(1, k + 1):

        # Enumerate all possible supports (sets of qubits) of size w.
        for support_qubits in _itertools.combinations(qubits, w):

            # Skip supports that are not connected under the within-num_hops rule.
            if not _support_is_connected(support_qubits, close):
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


def up_to_weight_k_pauli_pairs_from_qubit_graph(
    k: int, n: int, qubit_graph_laplacian: np.ndarray, num_hops: int
) -> list[tuple[str, str]]:
    """
    Enumerate unordered pairs `(P, Q)` (canonically ordered, `P < Q`) of DISTINCT, non-identity
    n-qubit Pauli strings whose *union* support (`support(P) union support(Q)`) is connected
    under a "within num_hops" connectivity rule derived from the qubit graph, and has weight
    (size) 1..k. This is the pair-indexed analog of `up_to_weight_k_paulis_from_qubit_graph`,
    used to enumerate the Pauli-pairs indexing locality-restricted 'C'/'A' type error generators.

    The union support (rather than each of P, Q's individual supports) is what must form a
    connected "blob" -- i.e. a 'C'/'A' pair is treated the same way a single weight-2+ 'H'/'S'
    Pauli already is: as one multi-qubit error whose combined support must be a connected
    subgraph. (An alternative convention -- requiring each of P, Q to be individually local,
    without necessarily being close to *each other* -- was considered but not implemented here;
    it would require different combinatorics and is left as a possible future extension.)

    Convention used throughout this module (matching `up_to_weight_k_paulis_from_qubit_graph`):
        qubit i  <->  string position (n - 1 - i)
    so qubit 0 is the rightmost character in each Pauli string.

    Parameters
    ----------
    k : int
        Maximum weight (size of `support(P) union support(Q)`).
    n : int
        Number of qubits (string length).
    qubit_graph_laplacian : numpy.ndarray
        Laplacian matrix of the qubit connectivity graph.
    num_hops : int
        Hop distance defining which qubit pairs are considered "close enough."

    Returns
    -------
    list[tuple[str, str]]
        All valid canonically-ordered `(P, Q)` pairs with union-weight 1..k and connected union
        support.
    """
    # ---- Basic input checks ----
    if not isinstance(k, int) or k < 1:
        raise TypeError("Pauli weight, k, must be an integer > 1.")
    if not isinstance(n, int) or n < 0:
        raise TypeError("n must be a non-negative integer.")

    if k > n:
        print("Pauli weight cannot exceed the number of qubits. Automatically setting k = min(k,n)")

    k = min(k, n)

    # ---- Build the "within num_hops" connectivity relation ----
    close = _qubit_graph_close_matrix(qubit_graph_laplacian, n, num_hops)

    # ---- Enumerate all valid Pauli pairs ----
    pairs: list[tuple[str, str]] = []
    qubits = range(n)

    # Enumerate all union-weights w = 1..k
    for w in range(1, k + 1):

        # Enumerate all possible (candidate union) supports (sets of qubits) of size w.
        for support_qubits in _itertools.combinations(qubits, w):

            # Skip supports that are not connected under the within-num_hops rule.
            if not _support_is_connected(support_qubits, close):
                continue

            pairs.extend(_pauli_pairs_for_support(support_qubits, n, reverse_index=True))

    return pairs


def _split_and_validate_egtypes(egtypes: list[str]) -> tuple[list[str], list[str]]:
    """
    Splits `egtypes` into its single-Pauli-indexed ('H'/'S') and pair-indexed ('C'/'A')
    members, raising `ValueError` if any entry is not one of the four supported types.

    Parameters
    ----------
    egtypes : list[str]
        Requested error generator types.

    Returns
    -------
    single_pauli_types, pair_types : list[str]
        The 'H'/'S' and 'C'/'A' entries of `egtypes`, respectively (each in their original
        relative order).
    """
    single_pauli_types = [t for t in egtypes if t in ('H', 'S')]
    pair_types = [t for t in egtypes if t in ('C', 'A')]
    unknown_types = [t for t in egtypes if t not in ('H', 'S', 'C', 'A')]
    if unknown_types:
        raise ValueError(
            f"Unknown error generator type(s) {unknown_types}! Supported types are 'H', 'S', 'C', 'A'."
        )
    return single_pauli_types, pair_types


def up_to_weight_k_error_gens_from_qubit_graph(k: int, n: int | None, qubit_graph_laplacian: np.ndarray, num_hops: int, egtypes: list[str] = ['H', 'S']) -> list[tuple[str, tuple[str, ...]]]:
    """Returns a list of all n-qubit error generators up to weight k of the specified
    types (e.g., 'H', 'S', 'C', and/or 'A') whose supports are connected subgraphs of the
    qubit connectivity graph (where connectivity is defined by hop distance on the qubit graph).

    For 'H'/'S' (each indexed by a single Pauli), this function first identifies the subset of
    multi-qubit Pauli operators (up to weight k) whose active qubit supports form a connected
    subgraph. For 'C'/'A' (each indexed by an unordered pair of two DISTINCT, non-identity
    Paulis; see "A Taxonomy of Small Errors", Sec. V.C-V.D), the analogous restriction is applied
    to the *union* of the two Paulis' supports (i.e. a 'C'/'A' pair is treated as a single
    multi-qubit error whose combined support must be a connected subgraph, exactly like a
    weight-2+ 'H'/'S' Pauli already is). The notion of connectivity is defined by a hop distance:
    two qubits are considered connected if they are at most `num_hops` apart in the graph
    represented by `qubit_graph_laplacian`.

    Once the relevant Pauli operators/pairs are found, the function constructs primitive error
    generators for each one and for each of the specified error generator types in `egtypes`.

    Parameters
    ----------
    k : int
        Maximum Pauli weight (for 'H'/'S') or maximum union-support weight (for 'C'/'A'). That
        is, the maximum number of qubits on which the constituent Pauli operator(s) act
        non-trivially (non-identity).
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
        A list of error generator types to generate. Supported values are:
          * 'H' : Hamiltonian error generators
          * 'S' : Stochastic-Pauli error generators
          * 'C' : Stochastic Pauli-Correlation error generators
          * 'A' : Active error generators

    Returns
    -------
    list[tuple]
        A list of error generator descriptors. Each descriptor is a tuple of the 
        form `(egtype, (pauli_string,))` for 'H'/'S', or `(egtype, (pauli_string_1,
        pauli_string_2))` for 'C'/'A', where:
          * `egtype` (str) is the type of the error generator ('H', 'S', 'C', or 'A').
          * each `pauli_string` (str) is a Pauli representation (e.g. 'IX', 'XY') 
            which (jointly, for 'C'/'A') indexes the error generator on the connected support.
    """

    # 1. Infer the number of qubits from the Laplacian shape if n is not provided (None).
    if n is None:
        n = qubit_graph_laplacian.shape[0]
    assert n is not None

    single_pauli_types, pair_types = _split_and_validate_egtypes(egtypes)

    error_generators: list[tuple[str, tuple[str, ...]]] = []

    if single_pauli_types:
        # 2a. Retrieve all Pauli strings up to weight k whose supports are connected
        #     subgraphs of the qubit graph (with adjacency defined by <= num_hops).
        relevant_paulis = up_to_weight_k_paulis_from_qubit_graph(k, n, qubit_graph_laplacian, num_hops)
        for egtype in single_pauli_types:
            # Each error generator is represented as a tuple: (type, (pauli_string,))
            error_generators += [(egtype, (p,)) for p in relevant_paulis]

    if pair_types:
        # 2b. Retrieve all Pauli pairs up to union-weight k whose union support is a connected
        #     subgraph of the qubit graph (with adjacency defined by <= num_hops).
        relevant_pairs = up_to_weight_k_pauli_pairs_from_qubit_graph(k, n, qubit_graph_laplacian, num_hops)
        for egtype in pair_types:
            # Each error generator is represented as a tuple: (type, (pauli_string_1, pauli_string_2))
            error_generators += [(egtype, (p1, p2)) for (p1, p2) in relevant_pairs]

    return error_generators


def up_to_weight_k_error_gens(k: int, n: int, egtypes: list[str] = ['H', 'S']) -> list[tuple[str, tuple[str, ...]]]:
    """Returns a list of all n-qubit error generators up to weight k, of types given in
    egtypes, in a tuple-of-strings format.

    Parameters
    ----------
    k : int
        Maximum Pauli weight (for 'H'/'S') or maximum union-support weight (for 'C'/'A').
    n : int
        Number of qubits.
    egtypes : list[str], default ['H','S']
        Error generator types to include. Supported values are 'H', 'S', 'C', and 'A'.

    Returns
    -------
    list[tuple]
        List of error generators in the tuple form `(egtype, (pauli_string,))` for 'H'/'S', or
        `(egtype, (pauli_string_1, pauli_string_2))` for 'C'/'A'. The first element is the error
        generator's type and the second element is a tuple specifying the Pauli(s) that index
        that error generator.
    """
    single_pauli_types, pair_types = _split_and_validate_egtypes(egtypes)

    error_generators: list[tuple[str, tuple[str, ...]]] = []

    if single_pauli_types:
        relevant_paulis = up_to_weight_k_paulis(k, n)
        for egtype in single_pauli_types:
            error_generators += [(egtype, (p,)) for p in relevant_paulis]

    if pair_types:
        relevant_pairs = up_to_weight_k_pauli_pairs(k, n)
        for egtype in pair_types:
            error_generators += [(egtype, (p1, p2)) for (p1, p2) in relevant_pairs]

    return error_generators


def error_generator_index(typ: str, paulis: tuple[str, ...]) -> int:
    """A function that *defines* an indexing of the primitive error generators, covering all
    four types in the "Taxonomy of Small Errors" (Blume-Kohout et al.) classification: 'H'
    (Hamiltonian), 'S' (Stochastic-Pauli), 'C' (Stochastic Pauli-Correlation), and 'A' (Active).

    Parameters
    ----------
    typ : str
        Error generator type. Supports:
          * 'H' : Hamiltonian error generators
          * 'S' : Stochastic-Pauli error generators
          * 'C' : Stochastic Pauli-Correlation error generators
          * 'A' : Active error generators
    paulis : tuple
        For 'H'/'S': a single-element tuple containing a string specifying the Pauli that
        labels the error. For 'C'/'A': a two-element tuple `(p1, p2)` of two DISTINCT,
        non-identity Pauli strings (order does not matter -- see `canonical_pauli_pair`; the
        pair is canonicalized internally before indexing). In all cases, each Pauli string's
        length implicitly defines the number of qubits the error gen acts on.

    Returns
    -------
    int
        Integer index defining the canonical ordering:
          * 'H' generators map to `[0, 4**n)`
          * 'S' generators map to `[4**n, 2*4**n)`
          * 'C' generators map to `[2*4**n, 2*4**n + M)`, where `M = num_pauli_pairs(n)`
          * 'A' generators map to `[2*4**n + M, 2*4**n + 2*M)`

    Raises
    ------
    ValueError
        If `typ` is not one of 'H', 'S', 'C', 'A'; or, for 'C'/'A', if `paulis` does not
        contain two distinct, non-identity Paulis.

    Notes
    -----
    For 'A'-type generators specifically, note that `A_{P,Q} = -A_{Q,P}` (see this module's
    docstring): if `paulis` is not already in canonical (sorted) order, this function still
    returns the correct *index* (since it canonicalizes internally), but does NOT tell you
    whether a compensating sign flip is needed for any associated *rate*/coefficient -- use
    `error_generator_canonicalization_sign` for that.
    """
    assert isinstance(paulis, tuple)
    p1 = paulis[0]
    n = len(p1)
    if typ == 'H':
        return paulistring_to_index(p1, n)
    elif typ == 'S':
        return 4**n + paulistring_to_index(p1, n)
    elif typ in ('C', 'A'):
        if len(paulis) != 2:
            raise ValueError(
                f"'{typ}' error generators must be indexed by a 2-tuple of two distinct Paulis; "
                f"got a {len(paulis)}-tuple."
            )
        p2 = paulis[1]
        if len(p2) != n:
            raise ValueError("Both Paulis in a 'C'/'A' pair must act on the same number of qubits!")
        pair_idx = pauli_pair_to_index(p1, p2, n)  # validates p1 != p2 and both non-identity
        M = num_pauli_pairs(n)
        base = 2 * 4**n if typ == 'C' else 2 * 4**n + M
        return base + pair_idx
    else:
        raise ValueError(f"Invalid error generator specification: {typ!r}! Supported types are 'H', 'S', 'C', 'A'.")

def index_to_error_gen(i: int, n: int, as_label: bool = False) -> tuple[str, tuple[str, ...]] | LocalStimErrorgenLabel:
    """
    Maps from the index to the 'label' representation of an elementary
    error generator. Inverse of `error_generator_index`.

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
        If `as_label` is False, returns `(typ, (pauli_string,))` for 'H'/'S', or
        `(typ, (pauli_string_1, pauli_string_2))` (in canonical order -- see
        `canonical_pauli_pair`) for 'C'/'A'.
        If `as_label` is True, returns `LocalStimErrorgenLabel(typ, paulis)`.

    Raises
    ------
    ValueError
        If `i` is outside the supported range `[0, 2*4**n + 2*num_pauli_pairs(n))`.
    """
    NP = 4**n
    M = num_pauli_pairs(n)
    if i < NP:
        typ = 'H'
        paulis: tuple[str, ...] = (index_to_paulistring(i, n),)
    elif i < 2 * NP:
        typ = 'S'
        paulis = (index_to_paulistring(i - NP, n),)
    elif i < 2 * NP + M:
        typ = 'C'
        paulis = index_to_pauli_pair(i - 2 * NP, n)
    elif i < 2 * NP + 2 * M:
        typ = 'A'
        paulis = index_to_pauli_pair(i - 2 * NP - M, n)
    else:
        raise ValueError('Invalid index!')

    if not as_label:
        return typ, paulis
    else:
        return _lseg.LocalStimErrorgenLabel(typ, paulis)

def num_error_generators(num_qubits: int) -> int:
    """Return the total number of indexed H/S/C/A error generators for `num_qubits`, i.e. the
    size of the full index range `[0, num_error_generators(num_qubits))` used by
    `error_generator_index`/`index_to_error_gen`.

    Parameters
    ----------
    num_qubits : int
        Number of qubits \(n\).

    Returns
    -------
    int
        `2*4**n + 2*num_pauli_pairs(n)` -- the combined size of the 'H', 'S', 'C', and 'A'
        index ranges.
    """
    return 2 * 4 ** num_qubits + 2 * num_pauli_pairs(num_qubits)

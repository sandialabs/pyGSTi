"""
Symplectic representation utility functions
"""
import numpy as _np

from . import matrixmod2 as _mtx
# ***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************
from ..baseobjs.label import Label as _Label
from ..baseobjs.smartcache import smart_cached

try:
    from . import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None


def symplectic_form(n, convention='standard'):
    """
    Creates the symplectic form for the number of qubits specified.

    There are two variants, of the sympletic form over the finite field of the
    integers modulo 2, used in pyGSTi. These corresponding to the 'standard' and
    'directsum' conventions.  In the case of 'standard', the symplectic form is the
    2n x 2n matrix of ((0,1),(1,0)), where '1' and '0' are the identity and all-zeros
    matrices of size n x n. The 'standard' symplectic form is probably the most
    commonly used, and it is the definition used throughout most of the code,
    including the Clifford compilers. In the case of 'directsum', the symplectic form
    is the direct sum of n 2x2 bit-flip matrices.  This is only used in pyGSTi for
    sampling from the symplectic group.

    Parameters
    ----------
    n : int
        The number of qubits the symplectic form should be constructed for. That
        is, the function creates a 2n x 2n matrix that is a sympletic form

    convention : str, optional
        Can be either 'standard' or 'directsum', which correspond to two different
        definitions for the symplectic form.

    Returns
    -------
    numpy array
        The specified symplectic form.
    """
    nn = 2 * n
    sym_form = _np.zeros((nn, nn), int)

    assert(convention == 'standard' or convention == 'directsum')

    if convention == 'standard':
        sym_form[n:nn, 0:n] = _np.identity(n, int)
        sym_form[0:n, n:nn] = _np.identity(n, int)

    if convention == 'directsum':
        # This current construction method is pretty stupid.
        for j in range(0, n):
            sym_form[2 * j, 2 * j + 1] = 1
            sym_form[2 * j + 1, 2 * j] = 1

    return sym_form


def change_symplectic_form_convention(s, outconvention='standard'):
    """
    Maps the input symplectic matrix between the 'standard' and 'directsum' symplectic form conventions.

    That is, if the input is a symplectic matrix with respect to the 'directsum'
    convention and outconvention ='standard' the output of this function is the
    equivalent symplectic matrix in the 'standard' symplectic form
    convention. Similarily, if the input is a symplectic matrix with respect to the
    'standard' convention and outconvention = 'directsum' the output of this function
    is the equivalent symplectic matrix in the 'directsum' symplectic form
    convention.

    Parameters
    ----------
    s : numpy.ndarray
        The input symplectic matrix.

    outconvention : str, optional
        Can be either 'standard' or 'directsum', which correspond to two different
        definitions for the symplectic form. This is the convention the input is
        being converted to (and so the input should be a symplectic matrix in the
        other convention).

    Returns
    -------
    numpy array
        The matrix `s` converted to `outconvention`.
    """
    n = _np.shape(s)[0] // 2

    if n == 1:
        return _np.copy(s)

    permutation_matrix = _np.zeros((2 * n, 2 * n), int)
    for i in range(0, n):
        permutation_matrix[2 * i, i] = 1
        permutation_matrix[2 * i + 1, n + i] = 1

    if outconvention == 'standard':
        sout = _np.dot(_np.dot(permutation_matrix.T, s), permutation_matrix)

    if outconvention == 'directsum':
        sout = _np.dot(_np.dot(permutation_matrix, s), permutation_matrix.T)

    return sout


def check_symplectic(m, convention='standard'):
    """
    Checks whether a matrix is symplectic.

    Parameters
    ----------
    m : numpy array
        The matrix to check.

    convention : str, optional
        Can be either 'standard' or 'directsum', Specifies the convention of
        the symplectic form with respect to which the matrix should be
        sympletic.

    Returns
    -------
    bool
        A bool specifying whether the matrix is symplectic
    """
    n = _np.shape(m)[0] // 2
    s_form = symplectic_form(n, convention=convention)
    conj = _mtx.dot_mod2(_np.dot(m, s_form), _np.transpose(m))

    return _np.array_equal(conj, s_form)


def inverse_symplectic(s):
    """
    Returns the inverse of a symplectic matrix over the integers mod 2.

    Parameters
    ----------
    s : numpy array
        The matrix to invert

    Returns
    -------
    numpy array
        The inverse of s, over the field of the integers mod 2.
    """
    assert(check_symplectic(s)), "The input matrix is not symplectic!"

    n = _np.shape(s)[0] // 2
    s_form = symplectic_form(n)
    s_inverse = _mtx.dot_mod2(_np.dot(s_form, _np.transpose(s)), s_form)

    assert(check_symplectic(s_inverse)), "The inverse is not symplectic. Function has failed"
    assert(_np.array_equal(_mtx.dot_mod2(s_inverse, s), _np.identity(2 * n, int))
           ), "The found matrix is not the inverse of the input. Function has failed"

    return s_inverse


def inverse_clifford(s, p):
    """
    Returns the inverse of a Clifford gate in the symplectic representation.

    This uses the formualas derived in Hostens and De Moor PRA 71, 042315 (2005).

    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford

    p : numpy array
        The 'phase vector' over the integers mod 4 representing the Clifford

    Returns
    -------
    sinverse : numpy array
        The symplectic matrix representing the inverse of the input Clifford.
    pinverse : numpy array
        The 'phase vector' representing the inverse of the input Clifford.
    """
    assert(check_valid_clifford(s, p)), \
        "The input symplectic matrix - phase vector pair does not define a valid Clifford!"

    sinverse = inverse_symplectic(s)

    n = _np.shape(s)[0] // 2

    # The formula used below for the inverse p vector comes from Hostens
    # and De Moor PRA 71, 042315 (2005).
    u = _np.zeros((2 * n, 2 * n), int)
    u[n:2 * n, 0:n] = _np.identity(n, int)

    vec1 = -1 * _np.dot(_np.transpose(sinverse), p)
    inner = _np.dot(_np.dot(_np.transpose(sinverse), u), sinverse)
    temp = 2 * _mtx.strictly_upper_triangle(inner) + _mtx.diagonal_as_matrix(inner)
    temp = _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s), temp), s))
    vec2 = -1 * _np.dot(_np.transpose(sinverse), temp)
    vec3 = _mtx.diagonal_as_vec(inner)

    pinverse = vec1 + vec2 + vec3
    pinverse = pinverse % 4

    assert(check_valid_clifford(sinverse, pinverse)), "The output does not define a valid Clifford. Function has failed"

    s_check, p_check = compose_cliffords(s, p, sinverse, pinverse)
    assert(_np.array_equal(s_check, _np.identity(2 * n, int))
           ), "The output is not the inverse of the input. Function has failed"
    assert(_np.array_equal(p_check, _np.zeros(2 * n, int))
           ), "The output is not the inverse of the input. Function has failed"

    s_check, p_check = compose_cliffords(sinverse, pinverse, s, p)
    assert(_np.array_equal(s_check, _np.identity(2 * n, int))
           ), "The output is not the inverse of the input. Function has failed"
    assert(_np.array_equal(p_check, _np.zeros(2 * n, int))
           ), "The output is not the inverse of the input. Function has failed"

    return sinverse, pinverse


def check_valid_clifford(s, p):
    """
    Checks if a symplectic matrix - phase vector pair (s,p) is the symplectic representation of a Clifford.

    This uses the formualas derived in Hostens and De Moor PRA 71, 042315 (2005).

    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford

    p : numpy array
        The 'phase vector' over the integers mod 4 representing the Clifford

    Returns
    -------
    bool
        True if (s,p) is the symplectic representation of some Clifford.
    """
    # Checks if the matrix s is symplectic, which is the only constraint on s.
    is_symplectic_matrix = check_symplectic(s)

    # Check whether the phase vector is valid. This currently does *not* check
    # that p is a vector over [0,1,2,3]. Perhaps it should. The constraint
    # that we check is satisfied comes from Hostens and De Moor PRA 71, 042315 (2005).
    n = _np.shape(s)[0] // 2
    u = _np.zeros((2 * n, 2 * n), int)
    u[n:2 * n, 0:n] = _np.identity(n, int)
    vec = p + _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s), u), s))
    vec = vec % 2

    is_valid_phase_vector = _np.array_equal(vec, _np.zeros(len(p), int))
    assert(is_valid_phase_vector)

    return (is_symplectic_matrix and is_valid_phase_vector)


def construct_valid_phase_vector(s, pseed):
    """
    Constructs a phase vector that, when paired with the provided symplectic matrix, defines a Clifford gate.

    If the seed phase vector, when paired with `s`, represents some Clifford this
    seed is returned. Otherwise 1 mod 4 is added to the required elements of the
    `pseed` in order to make it at valid phase vector (which is one of many possible
    phase vectors that, together with s, define a valid Clifford).

    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford

    pseed : numpy array
        The seed 'phase vector' over the integers mod 4.

    Returns
    -------
    numpy array
        Some p such that (s,p) is the symplectic representation of some Clifford.
    """
    pout = pseed.copy()
    n = _np.shape(s)[0] // 2

    assert(check_symplectic(s)), "The input matrix is not symplectic!"

    u = _np.zeros((2 * n, 2 * n), int)
    u[n:2 * n, 0:n] = _np.identity(n, int)

    # Each element of this vector should be 0 (mod 2) if this is a valid phase vector.
    # This comes from the formulas in Hostens and De Moor PRA 71, 042315 (2005).
    vec = pout + _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s), u), s))
    vec = vec % 2

    # Adds 1 mod 4 to all the elements of the vector where the required constraint is
    # not satisfied. This is then always a valid phase vector.
    pout[vec != 0] += 1
    pout = pout % 4

    assert(check_valid_clifford(s, pout)), "The output does not define a valid Clifford. Function has failed"

    return pout


def find_postmultipled_pauli(s, p_implemented, p_target, qubit_labels=None):
    """
    Finds the Pauli layer that should be appended to a circuit to implement a given Clifford.

    If some circuit implements the clifford described by the symplectic matrix s and
    the vector p_implemented, this function returns the Pauli layer that should be
    appended to this circuit to implement the clifford described by s and the vector
    p_target.

    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford
        implemented by the circuit

    p_implemented : numpy array
        The 'phase vector' over the integers mod 4 representing the Clifford
        implemented by the circuit

    p_target : numpy array
        The 'phase vector' over the integers mod 4 that, together with `s` represents
        the Clifford that you want to implement. Together with `s`, this vector must
        define a valid Clifford.

    qubit_labels : list, optional
        A list of qubit labels, that are strings or ints. The length of this list should
        be equal to the number of qubits the Clifford acts on. The ith element of the
        list is the label corresponding to the qubit at the ith index of `s` and the two
        phase vectors. If None, defaults to the integers from 0 to number of qubits - 1.

    Returns
    -------
    list
        A list that defines a Pauli layer, with the ith element containig one of the
        4 tuples (P,qubit_labels[i]) with P = 'I', 'Z', 'Y' and 'Z'
    """
    n = _np.shape(s)[0] // 2
    s_form = symplectic_form(n)
    vec = _mtx.dot_mod2(s, _np.dot(s_form, (p_target - p_implemented) // 2))

    if qubit_labels is None:
        qubit_labels = list(range(n))

    pauli_layer = []
    for q in range(0, n):
        if vec[q] == 0 and vec[q + n] == 0:
            pauli_layer.append(('I', qubit_labels[q]))
        elif vec[q] == 0 and vec[q + n] == 1:
            pauli_layer.append(('Z', qubit_labels[q]))
        elif vec[q] == 1 and vec[q + n] == 0:
            pauli_layer.append(('X', qubit_labels[q]))
        elif vec[q] == 1 and vec[q + n] == 1:
            pauli_layer.append(('Y', qubit_labels[q]))

    return pauli_layer


def find_premultipled_pauli(s, p_implemented, p_target, qubit_labels=None):
    """
    Finds the Pauli layer that should be prepended to a circuit to implement a given Clifford.

    If some circuit implements the clifford described by the symplectic matrix s and
    the vector p_implemented, this function returns the Pauli layer that should be
    prefixed to this circuit to implement the clifford described by s and the vector
    p_target.

    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford
        implemented by the circuit

    p_implemented : numpy array
        The 'phase vector' over the integers mod 4 representing the Clifford
        implemented by the circuit

    p_target : numpy array
        The 'phase vector' over the integers mod 4 that, together with `s` represents
        the Clifford that you want to implement. Together with `s`, this vector must
        define a valid Clifford.

    qubit_labels : list, optional
        A list of qubit labels, that are strings or ints. The length of this list should
        be equal to the number of qubits the Clifford acts on. The ith element of the
        list is the label corresponding to the qubit at the ith index of `s` and the two
        phase vectors. If None, defaults to the integers from 0 to number of qubits - 1.

    Returns
    -------
    list
        A list that defines a Pauli layer, with the ith element containig one of the
        4 tuples ('I',i), ('X',i), ('Y',i), ('Z',i).
    """
    n = _np.shape(s)[0] // 2
    s_form = symplectic_form(n)
    vec = _mtx.dot_mod2(s_form, (p_target - p_implemented) // 2)

    if qubit_labels is None:
        qubit_labels = list(range(n))

    pauli_layer = []
    for q in range(n):
        if vec[q] == 0 and vec[q + n] == 0:
            pauli_layer.append(('I', qubit_labels[q]))
        elif vec[q] == 0 and vec[q + n] == 1:
            pauli_layer.append(('Z', qubit_labels[q]))
        elif vec[q] == 1 and vec[q + n] == 0:
            pauli_layer.append(('X', qubit_labels[q]))
        elif vec[q] == 1 and vec[q + n] == 1:
            pauli_layer.append(('Y', qubit_labels[q]))

    return pauli_layer


def compose_cliffords(s1, p1, s2, p2, do_checks=True):
    """
    Multiplies two cliffords in the symplectic representation.

    The output corresponds to the symplectic representation of C2 times C1 (i.e., C1
    acts first) where s1 (s2) and p1 (p2) are the symplectic matrix and phase vector,
    respectively, for Clifford C1 (C2). This uses the formualas derived in Hostens
    and De Moor PRA 71, 042315 (2005).

    Parameters
    ----------
    s1 : numpy array
        The symplectic matrix over the integers mod 2 representing the first Clifford

    p1 : numpy array
        The 'phase vector' over the integers mod 4 representing the first Clifford

    s2 : numpy array
        The symplectic matrix over the integers mod 2 representing the second Clifford

    p2 : numpy array
        The 'phase vector' over the integers mod 4 representing the second Clifford

    do_checks : bool
        If True (default), check inputs and output are valid cliffords.
        If False, these checks are skipped (for speed)

    Returns
    -------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the composite Clifford
    p : numpy array
        The 'phase vector' over the integers mod 4 representing the compsite Clifford
    """
    assert(_np.shape(s1) == _np.shape(s2)), "Input must be Cliffords acting on the same number of qubits!"
    if do_checks:
        assert(check_valid_clifford(s1, p1)), "The first matrix-vector pair is not a valid Clifford!"
        assert(check_valid_clifford(s2, p2)), "The second matrix-vector pair is not a valid Clifford!"

    n = _np.shape(s1)[0] // 2

    # Below we calculate the s and p for the composite Clifford using the formulas from
    # Hostens and De Moor PRA 71, 042315 (2005).
    s = _mtx.dot_mod2(s2, s1)

    u = _np.zeros((2 * n, 2 * n), int)
    u[n:2 * n, 0:n] = _np.identity(n, int)

    vec1 = _np.dot(_np.transpose(s1), p2)
    inner = _np.dot(_np.dot(_np.transpose(s2), u), s2)
    matrix = 2 * _mtx.strictly_upper_triangle(inner) + _mtx.diagonal_as_matrix(inner)
    vec2 = _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s1), matrix), s1))
    vec3 = _np.dot(_np.transpose(s1), _mtx.diagonal_as_vec(inner))

    p = p1 + vec1 + vec2 - vec3
    p = p % 4

    if do_checks:
        assert(check_valid_clifford(s, p)), "The output is not a valid Clifford! Function has failed."

    return s, p


def symplectic_kronecker(sp_factors):
    """
    Takes a kronecker product of symplectic representations.

    Construct a single `(s,p)` symplectic (or stabilizer) representation that
    corresponds to the tensor (kronecker) product of the objects represented
    by each `(s,p)` element of `sp_factors`.

    This is performed by inserting each factor's `s` and `p` elements into the
    appropriate places of the final (large) `s` and `p` arrays.  This operation
    works for combining Clifford operations AND also stabilizer states.

    Parameters
    ----------
    sp_factors : iterable
        A list of `(s,p)` symplectic (or stabilizer) representation factors.

    Returns
    -------
    s : numpy.ndarray
        An array of shape (2n,2n) where n is the *total* number of qubits (the
        sum of the number of qubits in each `sp_factors` element).
    p : numpy.ndarray
        A 1D array of length 2n.
    """
    nlist = [len(p) // 2 for s, p in sp_factors]  # number of qubits per factor
    n = sum(nlist)  # total number of qubits

    sout = _np.zeros((2 * n, 2 * n), int)
    pout = _np.zeros(2 * n, int)
    k = 0  # current qubit index
    for (s, p), nq in zip(sp_factors, nlist):
        assert(s.shape == (2 * nq, 2 * nq))
        sout[k:k + nq, k:k + nq] = s[0:nq, 0:nq]
        sout[k:k + nq, n + k:n + k + nq] = s[0:nq, nq:2 * nq]
        sout[n + k:n + k + nq, k:k + nq] = s[nq:2 * nq, 0:nq]
        sout[n + k:n + k + nq, n + k:n + k + nq] = s[nq:2 * nq, nq:2 * nq]
        pout[k:k + nq] = p[0:nq]
        pout[n + k:n + k + nq] = p[nq:2 * nq]
        k += nq

    return sout, pout


def prep_stabilizer_state(nqubits, zvals=None):
    """
    Contruct the `(s,p)` stabilizer representation for a computational basis state given by `zvals`.

    Parameters
    ----------
    nqubits : int
        Number of qubits

    zvals : iterable, optional
        An iterable over anything that can be cast as True/False
        to indicate the 0/1 value of each qubit in the Z basis.
        If None, the all-zeros state is created.  If None, then
        all zeros is assumed.

    Returns
    -------
    s,p : numpy.ndarray
        The stabilizer "matrix" and phase vector corresponding to the desired
        state.  `s` has shape (2n,2n) (it includes antistabilizers) and `p`
        has shape 2n, where n equals `nqubits`.
    """
    n = nqubits
    s = _np.fliplr(_np.identity(2 * n, int))  # flip b/c stab cols are *first*
    p = _np.zeros(2 * n, int)
    if zvals:
        for i, z in enumerate(zvals):
            p[i] = p[i + n] = 2 if bool(z) else 0  # EGN TODO: check this is right -- (how to update the destabilizers?)
    return s, p


def apply_clifford_to_stabilizer_state(s, p, state_s, state_p):
    """
    Applies a clifford in the symplectic representation to a stabilizer state in the standard stabilizer representation.

    The output corresponds to the stabilizer representation of the output state.

    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford

    p : numpy array
        The 'phase vector' over the integers mod 4 representing the Clifford

    state_s : numpy array
        The matrix over the integers mod 2 representing the stabilizer state

    state_p : numpy array
        The 'phase vector' over the integers mod 4 representing the stabilizer state

    Returns
    -------
    out_s : numpy array
        The symplectic matrix over the integers mod 2 representing the output state
    out_p : numpy array
        The 'phase vector' over the integers mod 4 representing the output state
    """
    two_n = _np.shape(s)[0]; n = two_n // 2
    assert(_np.shape(state_s) == (two_n, two_n)), "Clifford and state must be for the same number of qubits!"
    assert(_np.shape(state_p) == (two_n,)), "Invalid stabilizer state representation"
    assert(check_valid_clifford(s, p)), "The `s`,`p` matrix-vector pair is not a valid Clifford!"
    #EGN TODO: check valid stabilizer state?

    # Below we calculate the s and p for the output state using the formulas from
    # Hostens and De Moor PRA 71, 042315 (2005).
    out_s = _mtx.dot_mod2(s, state_s)

    u = _np.zeros((2 * n, 2 * n), int)
    u[n:2 * n, 0:n] = _np.identity(n, int)

    inner = _np.dot(_np.dot(_np.transpose(s), u), s)
    vec1 = _np.dot(_np.transpose(state_s), p - _mtx.diagonal_as_vec(inner))
    matrix = 2 * _mtx.strictly_upper_triangle(inner) + _mtx.diagonal_as_matrix(inner)
    vec2 = _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(state_s), matrix), state_s))

    out_p = state_p + vec1 + vec2
    out_p = out_p % 4

    ##More explicitly operates on stabilizer and antistabilizer separately, but same as above
    #out_p = _np.zeros(2*n,int)
    #for slc in (slice(0,n),slice(n,2*n)):
    #    ss = state_s[:,slc]
    #    inner = _np.dot(_np.dot(_np.transpose(s),u),s)
    #    vec1 = _np.dot(_np.transpose(ss),p - _mtx.diagonal_as_vec(inner))
    #    matrix = 2*_mtx.strictly_upper_triangle(inner)+_mtx.diagonal_as_matrix(inner)
    #    vec2 = _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(ss),matrix),ss))
    #    out_p[slc] = state_p[slc] + vec1 + vec2
    #    out_p[slc] = out_p[slc] % 4

    #EGN TODO: check for valid stabilizer state

    return out_s, out_p


def pauli_z_measurement(state_s, state_p, qubit_index):
    """
    Computes the probabilities of 0/1 (+/-) outcomes from measuring a Pauli operator on a stabilizer state.

    Parameters
    ----------
    state_s : numpy array
        The matrix over the integers mod 2 representing the stabilizer state

    state_p : numpy array
        The 'phase vector' over the integers mod 4 representing the stabilizer state

    qubit_index : int
        The index of the qubit being measured

    Returns
    -------
    p0, p1 : float
        Probabilities of 0 (+ eigenvalue) and 1 (- eigenvalue) outcomes.
    state_s_0, state_s_1 : numpy array
        Matrix over the integers mod 2 representing the output stabilizer
        states.
    state_p_0, state_p_1 : numpy array
        Phase vectors over the integers mod 4 representing the output
        stabilizer states.
    """
    two_n = len(state_p); n = two_n // 2
    assert(_np.shape(state_s) == (two_n, two_n)), "Inconsistent stabilizier representation!"

    #This algorithm follows that of PRA 70, 052328 (2004),
    # except note:
    # 0) columns & rows are reversed
    # 1) that states carry arount full "mod 4"
    # phase vectors instead of the minmal "mod 2" ones
    # comprised of the higher-order-bit of each "mod 4" vector.
    # 2) that the stabilizer is held in the *first* n
    # columns of state_s rather than the last n.

    a = qubit_index
    #print("DB: pauli Z_%d meas on state:" % a) #DEBUG
    #print(state_s); print(state_p) #DEBUG

    # Let A be the Pauli op being measured
    #Step1: determine if all stabilizer elements commute with Z_a
    # which amounts to checking whether there are any 1-bits in the a-th
    # row of state_s for the first n columns (any stabilizers that have X's
    # or Y's for qubit a, as 00=I, 10=X, 11=Y, 01=Z)
    for col in range(n):
        if state_s[a, col] == 1:
            p = col
            #print("Column %d anticommutes with Z_%d" % (p,a)) #DEBUG

            # p is first stabilizer that anticommutes w/Z_a. Outcome is random,
            # and we just need to update the state (if requested).
            s_out = state_s.copy(); p_out = state_p.copy()
            for i in range(two_n):
                if i != p and state_s[a, i] == 1:
                    #print("COLSUM ",i,p) #DEBUG
                    colsum(i, p, s_out, p_out, n)
                    #print(s_out); print(p_out) #DEBUG
                    #print("-----") #DEBUG
            s_out[:, p + n] = s_out[:, p]; p_out[p + n] = p_out[p]  # set p-th col -> (p+n)-th col
            s_out[:, p] = 0; s_out[a + n, p] = 1  # p-th col = I*...Z_a*...I stabilizer

            icount = sum([3 if (s_out[i, p] == s_out[i + n, p] == 1) else 0 for i in range(n)])  # 11 = -iY convention
            p_out0 = p_out.copy(); p_out0[p] = (4 - (icount % 4)) % 4  # so overall phase is 0
            p_out1 = p_out.copy(); p_out1[p] = (p_out0[p] + 2) % 4    # so overall phase is -1
            return 0.5, 0.5, s_out, s_out, p_out0, p_out1  # Note: s is same for 0 and 1 outcomes

    #print("Nothing anticommutes!") #DEBUG

    # no break ==> all commute, so outcome is deterministic, so no
    # state update; just determine whether Z_a or -Z_a is in the stabilizer,
    # which we do using the "anti-stabilizer" cleverness of PRA 70, 052328
    acc_s = _np.zeros(two_n, int); acc_p = _np.zeros(1, int)
    for i in range(n, two_n):  # loop over anti-stabilizer
        if state_s[a, i] == 1:  # for elements that anti-commute w/Z_a
            colsum_acc(acc_s, acc_p, i - n, state_s, state_p, n)  # act w/corresponding *stabilizer* el
    # now the high bit of acc_p holds the outcome
    icount = acc_p[0] + sum([3 if (acc_s[i] == acc_s[i + n] == 1) else 0 for i in range(n)])  # 11 = -iY convention
    icount = icount % 4
    if icount == 0:  # outcome is always zero
        #print("Always 0!") #DEBUG
        return (1.0, 0.0, state_s, state_s, state_p, state_p)
    else:  # outcome is always 1
        #print("Always 1!") #DEBUG
        assert(icount == 2)  # should never get 1 or 3 (low bit should always be 0)
        return (0.0, 1.0, state_s, state_s, state_p, state_p)


def colsum(i, j, s, p, n):
    """
    A helper routine used for manipulating stabilizer state representations.

    Updates the `i`-th stabilizer generator (column of `s` and element of `p`)
    with the group-action product of the `j`-th and the `i`-th generators, i.e.

    generator[i] -> generator[j] + generator[i]

    Parameters
    ----------
    i : int
        Destination generator index.

    j : int
        Sournce generator index.

    s : numpy array
        The matrix over the integers mod 2 representing the stabilizer state

    p : numpy array
        The 'phase vector' over the integers mod 4 representing the stabilizer state

    n : int
        The number of qubits.  `s` must be shape (2n,2n) and `p` must be
        length 2n.

    Returns
    -------
    None
    """
    #OLD: according to Aaronson convention (not what we use)
    ##Note: need to divide p-vals by 2 to access high-bit
    #test = 2*(p[i]//2 + p[j]//2) + sum(
    #    [colsum_g(s[k,j], s[k+n,j], s[k,i], s[k+n,i]) for k in range(n)] )
    #test = test % 4
    #assert(test in (0,2)) # test should never be congruent to 1 or 3 (mod 4)
    #p[i] = 0 if (test == 0) else 2 # ( = 10 = 1 in high bit)

    u = _np.zeros((2 * n, 2 * n), int)  # CACHE!
    u[n:2 * n, 0:n] = _np.identity(n, int)

    p[i] += p[j] + 2 * float(_np.dot(s[:, i].T, _np.dot(u, s[:, j])))
    for k in range(n):
        s[k, i] = s[k, j] ^ s[k, i]
        s[k + n, i] = s[k + n, j] ^ s[k + n, i]
        #EGN TODO: use _np.bitwise_xor or logical_xor here? -- keep it obvious (&slow) for now...
    return


def colsum_acc(acc_s, acc_p, j, s, p, n):
    """
    A helper routine used for manipulating stabilizer state representations.

    Similar to :function:`colsum` except a separate "accumulator" column is
    used instead of the `i`-th column of `s` and element of `p`. I.e., this
    performs:

    acc[0] -> generator[j] + acc[0]

    Parameters
    ----------
    acc_s : numpy array
        The matrix over the integers mod 2 representing the "accumulator" stabilizer state

    acc_p : numpy array
        The 'phase vector' over the integers mod 4 representing the "accumulator" stabilizer state

    j : int
        Index of the stabilizer generator being accumulated (see above).

    s : numpy array
        The matrix over the integers mod 2 representing the stabilizer state

    p : numpy array
        The 'phase vector' over the integers mod 4 representing the stabilizer state

    n : int
        The number of qubits.  `s` must be shape (2n,2n) and `p` must be
        length 2n.

    Returns
    -------
    None
    """

    ##Note: need to divide p-vals by 2 to access high-bit
    #test = 2*(acc_p[0]//2 + p[j]//2) + sum(
    #    [colsum_g(s[k,j], s[k+n,j], acc_s[k], acc_s[k+n]) for k in range(n)] )
    #test = test % 4
    #assert(test in (0,2)) # test should never be congruent to 1 or 3 (mod 4)
    #acc_p[0] = 0 if (test == 0) else 2 # ( = 10 = 1 in high bit)

    u = _np.zeros((2 * n, 2 * n), int)  # CACHE!
    u[n:2 * n, 0:n] = _np.identity(n, int)

    acc_p[0] += p[j] + 2 * float(_np.dot(acc_s.T, _np.dot(u, s[:, j])))

    for k in range(n):
        acc_s[k] = s[k, j] ^ acc_s[k]
        acc_s[k + n] = s[k + n, j] ^ acc_s[k + n]
        #EGN TODO: use _np.bitwise_xor or logical_xor here? -- keep it obvious (&slow) for now...
    return


def stabilizer_measurement_prob(state_sp_tuple, moutcomes, qubit_filter=None,
                                return_state=False):
    """
    Compute the probability of a given outcome when measuring some or all of the qubits in a stabilizer state.

    Returns this probability, optionally along with the updated (post-measurement) stabilizer state.

    Parameters
    ----------
    state_sp_tuple : tuple
        A `(s,p)` tuple giving the stabilizer state to measure.

    moutcomes : array-like
        The z-values identifying which measurement outcome (a computational
        basis state) to compute the probability for.

    qubit_filter : iterable, optional
        If not None, a list of qubit indices which are measured.
        `len(qubit_filter)` should always equal `len(moutcomes)`. If None, then
        assume *all* qubits are measured (`len(moutcomes)` == num_qubits).

    return_state : bool, optional
        Whether the post-measurement (w/outcome `moutcomes`) state is also
        returned.

    Returns
    -------
    p : float
        The probability of the given measurement outcome.
    state_s,state_p : numpy.ndarray
        Only returned when `return_state=True`.  The post-measurement stabilizer
        state representation (an updated version of `state_sp_tuple`).
    """
    state_s, state_p = state_sp_tuple  # should be a StabilizerState.to_dense() "object"

    p = 1
    if qubit_filter is None:  # len(moutcomes) == nQubits
        qubit_filter = range(len(moutcomes))

    for i, outcm in zip(qubit_filter, moutcomes):
        p0, p1, ss0, ss1, sp0, sp1 = pauli_z_measurement(state_s, state_p, i)
        # could cache these results in a FUTURE _stabilizer_measurement_probs function?
        if outcm == 0:
            p *= p0; state_s, state_p = ss0, sp0
        else:
            p *= p1; state_s, state_p = ss1, sp1

    return (p, state_s, state_p) if return_state else p


def embed_clifford(s, p, qubit_inds, n):
    """
    Embeds the `(s,p)` Clifford symplectic representation into a larger symplectic representation.

    The action of `(s,p)` takes place on the qubit indices specified by `qubit_inds`.

    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford

    p : numpy array
        The 'phase vector' over the integers mod 4 representing the Clifford

    qubit_inds : list
        A list or array of integers specifying which qubits `s` and `p` act on.

    n : int
        The total number of qubits

    Returns
    -------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the embedded Clifford
    p : numpy array
        The 'phase vector' over the integers mod 4 representing the embedded Clifford
    """
    ne = len(qubit_inds)  # nQubits for embedded_op
    s_out = _np.identity(2 * n, int)
    p_out = _np.zeros(2 * n, int)

    for i, di in enumerate(qubit_inds):  # di = "destination index"
        p_out[di] = p[i]
        p_out[di + n] = p[i + ne]

        for j, dj in enumerate(qubit_inds):
            s_out[di, dj] = s[i, j]
            s_out[di + n, dj + n] = s[i + ne, j + ne]
            s_out[di, dj + n] = s[i, j + ne]
            s_out[di + n, dj] = s[i + ne, j]

    return s_out, p_out


def compute_internal_gate_symplectic_representations(gllist=None):
    """
    Creates a dictionary of the symplectic representations of 'standard' Clifford gates.

    Returns a dictionary containing the symplectic matrices and phase vectors that represent
    the specified 'standard' Clifford gates, or the representations of *all* the standard gates
    if no list of operation labels is supplied. These 'standard' Clifford gates are those gates that
    are already known to the code gates (e.g., the label 'CNOT' has a specfic meaning in the
    code), and are recorded as unitaries in "internalgates.py".

    Parameters
    ----------
    gllist : list, optional
        If not None, a list of strings corresponding to operation labels for any of the standard
        gates that have fixed meaning for the code (e.g., 'CNOT' corresponds to
        the CNOT gate with the first qubit the target). For example, this list could be
        gllist = ['CNOT','H','P','I','X'].

    Returns
    -------
    srep_dict : dict
        dictionary of `(smatrix,svector)` tuples, where `smatrix` and `svector`
        are numpy arrays containing the symplectic matrix and phase vector
        representing the operation label given by the key.
    """
    # Full dictionaries, containing the symplectic representations of *all* gates
    # that are hard-coded, and which have a specific meaning to the code.
    complete_s_dict = {}
    complete_p_dict = {}

    # The Pauli gates
    complete_s_dict['I'] = _np.array([[1, 0], [0, 1]], int)
    complete_s_dict['X'] = _np.array([[1, 0], [0, 1]], int)
    complete_s_dict['Y'] = _np.array([[1, 0], [0, 1]], int)
    complete_s_dict['Z'] = _np.array([[1, 0], [0, 1]], int)

    complete_p_dict['I'] = _np.array([0, 0], int)
    complete_p_dict['X'] = _np.array([0, 2], int)
    complete_p_dict['Y'] = _np.array([2, 2], int)
    complete_p_dict['Z'] = _np.array([2, 0], int)

    # Five single qubit gates that each represent one of five classes of Cliffords
    # that equivalent up to Pauli gates and are not equivalent to idle (that class
    # is covered by any one of the Pauli gates above).
    complete_s_dict['H'] = _np.array([[0, 1], [1, 0]], int)
    complete_s_dict['P'] = _np.array([[1, 0], [1, 1]], int)
    complete_s_dict['PH'] = _np.array([[0, 1], [1, 1]], int)
    complete_s_dict['HP'] = _np.array([[1, 1], [1, 0]], int)
    complete_s_dict['HPH'] = _np.array([[1, 1], [0, 1]], int)
    complete_p_dict['H'] = _np.array([0, 0], int)
    complete_p_dict['P'] = _np.array([1, 0], int)
    complete_p_dict['PH'] = _np.array([0, 1], int)
    complete_p_dict['HP'] = _np.array([3, 0], int)
    complete_p_dict['HPH'] = _np.array([0, 3], int)
    # The full 1-qubit Cliffor group, using the same labelling as in extras.rb.group
    complete_s_dict['C0'] = _np.array([[1, 0], [0, 1]], int)
    complete_p_dict['C0'] = _np.array([0, 0], int)
    complete_s_dict['C1'] = _np.array([[1, 1], [1, 0]], int)
    complete_p_dict['C1'] = _np.array([1, 0], int)
    complete_s_dict['C2'] = _np.array([[0, 1], [1, 1]], int)
    complete_p_dict['C2'] = _np.array([0, 1], int)
    complete_s_dict['C3'] = _np.array([[1, 0], [0, 1]], int)
    complete_p_dict['C3'] = _np.array([0, 2], int)
    complete_s_dict['C4'] = _np.array([[1, 1], [1, 0]], int)
    complete_p_dict['C4'] = _np.array([1, 2], int)
    complete_s_dict['C5'] = _np.array([[0, 1], [1, 1]], int)
    complete_p_dict['C5'] = _np.array([0, 3], int)
    complete_s_dict['C6'] = _np.array([[1, 0], [0, 1]], int)
    complete_p_dict['C6'] = _np.array([2, 2], int)
    complete_s_dict['C7'] = _np.array([[1, 1], [1, 0]], int)
    complete_p_dict['C7'] = _np.array([3, 2], int)
    complete_s_dict['C8'] = _np.array([[0, 1], [1, 1]], int)
    complete_p_dict['C8'] = _np.array([2, 3], int)
    complete_s_dict['C9'] = _np.array([[1, 0], [0, 1]], int)
    complete_p_dict['C9'] = _np.array([2, 0], int)
    complete_s_dict['C10'] = _np.array([[1, 1], [1, 0]], int)
    complete_p_dict['C10'] = _np.array([3, 0], int)
    complete_s_dict['C11'] = _np.array([[0, 1], [1, 1]], int)
    complete_p_dict['C11'] = _np.array([2, 1], int)
    complete_s_dict['C12'] = _np.array([[0, 1], [1, 0]], int)
    complete_p_dict['C12'] = _np.array([0, 0], int)
    complete_s_dict['C13'] = _np.array([[1, 1], [0, 1]], int)
    complete_p_dict['C13'] = _np.array([0, 1], int)
    complete_s_dict['C14'] = _np.array([[1, 0], [1, 1]], int)
    complete_p_dict['C14'] = _np.array([1, 0], int)
    complete_s_dict['C15'] = _np.array([[0, 1], [1, 0]], int)
    complete_p_dict['C15'] = _np.array([0, 2], int)
    complete_s_dict['C16'] = _np.array([[1, 1], [0, 1]], int)
    complete_p_dict['C16'] = _np.array([0, 3], int)
    complete_s_dict['C17'] = _np.array([[1, 0], [1, 1]], int)
    complete_p_dict['C17'] = _np.array([1, 2], int)
    complete_s_dict['C18'] = _np.array([[0, 1], [1, 0]], int)
    complete_p_dict['C18'] = _np.array([2, 2], int)
    complete_s_dict['C19'] = _np.array([[1, 1], [0, 1]], int)
    complete_p_dict['C19'] = _np.array([2, 3], int)
    complete_s_dict['C20'] = _np.array([[1, 0], [1, 1]], int)
    complete_p_dict['C20'] = _np.array([3, 2], int)
    complete_s_dict['C21'] = _np.array([[0, 1], [1, 0]], int)
    complete_p_dict['C21'] = _np.array([2, 0], int)
    complete_s_dict['C22'] = _np.array([[1, 1], [0, 1]], int)
    complete_p_dict['C22'] = _np.array([2, 1], int)
    complete_s_dict['C23'] = _np.array([[1, 0], [1, 1]], int)
    complete_p_dict['C23'] = _np.array([3, 0], int)
    # The CNOT gate, CPHASE gate, and SWAP gate.
    complete_s_dict['CNOT'] = _np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], int)
    complete_s_dict['CPHASE'] = _np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
    complete_s_dict['SWAP'] = _np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    complete_p_dict['CNOT'] = _np.array([0, 0, 0, 0], int)
    complete_p_dict['CPHASE'] = _np.array([0, 0, 0, 0], int)
    complete_p_dict['SWAP'] = _np.array([0, 0, 0, 0], int)

    if gllist is None:
        keys = list(complete_s_dict.keys())
        assert(set(keys) == set(complete_p_dict.keys()))
    else:
        keys = gllist

    srep_dict = {k: (complete_s_dict[k], complete_p_dict[k]) for k in keys}
    return srep_dict


def symplectic_rep_of_clifford_circuit(circuit, srep_dict=None, pspec=None):
    """
    Returns the symplectic representation of the composite Clifford implemented by the specified Clifford circuit.

    This uses the formualas derived in Hostens and De Moor PRA 71, 042315 (2005).

    Parameters
    ----------
    circuit : Circuit
        The Clifford circuit to calculate the global action of, input as a
        Circuit object.

    srep_dict : dict, optional
        If not None, a dictionary providing the (symplectic matrix, phase vector)
        tuples associated with each operation label. If the circuit layer contains only
        'standard' gates which have a hard-coded symplectic representation this
        may be None. Alternatively, if `pspec` is specifed and it contains the
        gates in `circuit` in a Clifford model, it also does not need to be
        specified (and it is ignored if it is specified). Otherwise it must be
        specified.

    pspec : ProcessorSpec, optional
        A ProcessorSpec that contains a Clifford model that defines the symplectic
        action of all of the gates in `circuit`. If this is not None it over-rides
        `srep_dict`. Both `pspec` and `srep_dict` can only be None if the circuit
        contains only gates with names that are hard-coded into pyGSTi.

    Returns
    -------
    s : numpy array
        The symplectic matrix representing the Clifford implement by the input circuit
    p : dictionary of numpy arrays
        The phase vector representing the Clifford implement by the input circuit
    """
    n = circuit.num_lines
    depth = circuit.depth

    if srep_dict is None:
        srep_dict = {}
    srep_dict.update(compute_internal_gate_symplectic_representations())
    if pspec is not None:
        srep_dict.update(pspec.models['clifford'].compute_clifford_symplectic_reps())

    # The initial action of the circuit before any layers are applied.
    s = _np.identity(2 * n, int)
    p = _np.zeros(2 * n, int)

    for i in range(0, depth):
        # This relies on the circuit having a valid self.identity identifier -- as those gates are
        # not returned in the layer. Note that the layer contains each gate only once.
        layer = circuit.layer_label(i)
        # future : update so that we don't use this function, because it slower than necessary (possibly much slower).
        layer_s, layer_p = symplectic_rep_of_clifford_layer(layer, n, circuit.line_labels, srep_dict,
                                                            add_internal_sreps=False)
        #s, p = compose_cliffords(s, p, layer_s, layer_p, do_checks=False)
        if _fastcalc is not None:
            s, p = _fastcalc.fast_compose_cliffords(s, p, layer_s, layer_p)
        else:
            s, p = compose_cliffords(s, p, layer_s, layer_p, do_checks=False)

    return s, p


def symplectic_rep_of_clifford_layer(layer, n=None, q_labels=None, srep_dict=None, add_internal_sreps=True):
    """
    Constructs the symplectic representation of the n-qubit Clifford implemented by a single quantum circuit layer.

    (Gates in a "single layer" must act on disjoint sets of qubits, but not all qubits
    need to be acted upon in the layer.)

    Parameters
    ----------
    layer : Label
        A layer label, often a compound label with components. Specifies
        The Clifford gate(s) to calculate the global action of.

    n : int, optional
        The total number of qubits. Must be specified if `q_labels` is None.

    q_labels : list, optional
        A list of all the qubit labels. If the layer is over qubits that are not
        labelled by integers 0 to n-1 then it is necessary to specify this list.
        Note that this should contain *all* the qubit labels for the circuit that
        this is a layer from, and they should be ordered as in that circuit, otherwise
        the symplectic rep returned might not be of the correct dimension or of the
        correct order.

    srep_dict : dict, optional
        If not None, a dictionary providing the (symplectic matrix, phase vector)
        tuples associated with each operation label. If the circuit layer contains only
        'standard' gates which have a hard-coded symplectic representation this
        may be None. Otherwise it must be specified. If the layer contains some
        standard gates it is not necesary to specify the symplectic represenation
        for those gates.

    add_internal_sreps : bool, optional
        If True, the symplectic reps for internal gates are calculated and added to srep_dict.
        For speed, calculate these reps once, store them in srep_dict, and set this to False.

    Returns
    -------
    s : numpy array
        The symplectic matrix representing the Clifford implement by specified
        circuit layer
    p : numpy array
        The phase vector representing the Clifford implement by specified
        circuit layer
    """
    # This method uses a brute-force matrix construction. Future: perhaps this should be updated.
    if srep_dict is None:
        srep_dict = {}
    if add_internal_sreps is True or len(srep_dict) == 0:
        srep_dict.update(compute_internal_gate_symplectic_representations())

    if q_labels is None:
        assert(n is not None), "The number of qubits must be specified if `q_labels` is None!"
        q_labels = list(range(n))
    elif n is None:
        assert(q_labels is not None), "Cannot have both `n` and `q_labels` as None!"
        n = len(q_labels)
    else:
        assert(len(q_labels) == n), "`n` and `q_labels` are inconsistent!"

    s = _np.identity(2 * n, int)
    p = _np.zeros(2 * n, int)

    if not isinstance(layer, _Label):
        layer = _Label(layer)

    for sub_lbl in layer.components:
        matrix, phase = srep_dict[sub_lbl.name]
        nforgate = sub_lbl.number_of_qubits
        sub_lbl_qubits = sub_lbl.qubits if (sub_lbl.qubits is not None) else q_labels
        for ind1, qlabel1 in enumerate(sub_lbl_qubits):
            qindex1 = q_labels.index(qlabel1)
            for ind2, qlabel2 in enumerate(sub_lbl_qubits):
                qindex2 = q_labels.index(qlabel2)
                # Put in the symp matrix elements
                s[qindex1, qindex2] = matrix[ind1, ind2]
                s[qindex1, qindex2 + n] = matrix[ind1, ind2 + nforgate]
                s[qindex1 + n, qindex2] = matrix[ind1 + nforgate, ind2]
                s[qindex1 + n, qindex2 + n] = matrix[ind1 + nforgate, ind2 + nforgate]

            # Put in the phase elements
            p[qindex1] = phase[ind1]
            p[qindex1 + n] = phase[ind1 + nforgate]

    return s, p


def one_q_clifford_symplectic_group_relations():
    """
    Gives the group relationship between the 'I', 'H', 'P' 'HP', 'PH', and 'HPH' up-to-Paulis operators.

    The returned dictionary contains keys (A,B) for all A and B in
    the above list. The value for key (A,B) is C if BA = C x some
    Pauli operator. E,g, ('P','P') = 'I'.

    This dictionary is important for Compiling multi-qubit Clifford
    gates without unneccessary 1-qubit gate over-heads. But note that
    this dictionary should not be used for compressing circuits containing
    these gates when the exact action of the circuit is of importance (not
    only the up-to-Paulis action of the circuit).

    Returns
    -------
    dict
    """
    group_relations = {}

    group_relations['I', 'I'] = 'I'
    group_relations['I', 'H'] = 'H'
    group_relations['I', 'P'] = 'P'
    group_relations['I', 'HP'] = 'HP'
    group_relations['I', 'PH'] = 'PH'
    group_relations['I', 'HPH'] = 'HPH'

    group_relations['H', 'I'] = 'H'
    group_relations['H', 'H'] = 'I'
    group_relations['H', 'P'] = 'PH'
    group_relations['H', 'HP'] = 'HPH'
    group_relations['H', 'PH'] = 'P'
    group_relations['H', 'HPH'] = 'HP'

    group_relations['P', 'I'] = 'P'
    group_relations['P', 'H'] = 'HP'
    group_relations['P', 'P'] = 'I'
    group_relations['P', 'HP'] = 'H'
    group_relations['P', 'PH'] = 'HPH'
    group_relations['P', 'HPH'] = 'PH'

    group_relations['HP', 'I'] = 'HP'
    group_relations['HP', 'H'] = 'P'
    group_relations['HP', 'P'] = 'HPH'
    group_relations['HP', 'HP'] = 'PH'
    group_relations['HP', 'PH'] = 'I'
    group_relations['HP', 'HPH'] = 'H'

    group_relations['PH', 'I'] = 'PH'
    group_relations['PH', 'H'] = 'HPH'
    group_relations['PH', 'P'] = 'H'
    group_relations['PH', 'HP'] = 'I'
    group_relations['PH', 'PH'] = 'HP'
    group_relations['PH', 'HPH'] = 'P'

    group_relations['HPH', 'I'] = 'HPH'
    group_relations['HPH', 'H'] = 'PH'
    group_relations['HPH', 'P'] = 'HP'
    group_relations['HPH', 'HP'] = 'P'
    group_relations['HPH', 'PH'] = 'H'
    group_relations['HPH', 'HPH'] = 'I'

    return group_relations


def unitary_is_clifford(unitary):
    """
    Returns True if the unitary is a Clifford gate (w.r.t the standard basis), and False otherwise.

    Parameters
    ----------
    unitary : numpy.ndarray
        A unitary matrix to test.

    Returns
    -------
    bool
    """
    s, p = unitary_to_symplectic(unitary, flagnonclifford=False)
    if s is None: return False
    else: return True


def _unitary_to_symplectic_1q(u, flagnonclifford=True):
    """
    Returns the symplectic representation of a single qubit Clifford unitary,
    input as a complex matrix in the standard computational basis.

    Parameters
    ----------
    u : numpy array
        The unitary matrix to construct the symplectic representation for. This
        must be a single-qubit gate (so, it is a 2 x 2 matrix), and it must be
        in the standard computational basis. E.g., the unitary for the Z gate
        is matrix ((1.,0.),(0.,-1.)). It also must be a Clifford gate in the
        standard sense.

    flagnonclifford : bool, opt
        If True, a ValueError is raised when the input unitary is not a Clifford gate.
        If False, when the unitary is not a Clifford the returned s and p are
        None.

    Returns
    -------
    s : numpy array or None
        The symplectic matrix representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False

    p : numpy array or None
        The phase vector representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False

    """
    assert(_np.shape(u) == (2, 2)), "Input is not a single qubit unitary!"

    x = _np.array([[0, 1.], [1., 0]])
    z = _np.array([[1., 0], [0, -1.]])
    fund_paulis = [x, z]

    s = _np.zeros((2, 2), int)
    p = _np.zeros(2, int)

    for pauli_label in range(0, 2):

        # Calculate the matrix that the input unitary transforms the current Pauli group
        # generator to (X or Z).
        conj = _np.dot(_np.dot(u, fund_paulis[pauli_label]), _np.linalg.inv(u))

        # Find which element of the Pauli group this is, and fill out the relevant
        # bits of the symplectic matrix and phase vector as this implies.
        for i in range(0, 2):
            for j in range(0, 2):
                for k in range(0, 4):

                    pauli_ijk = (1j**(k)) * _np.dot(_np.linalg.matrix_power(x, i), _np.linalg.matrix_power(z, j))

                    if _np.allclose(conj, pauli_ijk):
                        s[:, pauli_label] = _np.array([i, j])
                        p[pauli_label] = k

    valid_clifford = check_valid_clifford(s, p)

    if flagnonclifford and not valid_clifford:
        raise ValueError("Input unitary is not a Clifford with respect to the standard basis!")

    else:
        if not valid_clifford:
            s = None
            p = None

    return s, p


def _unitary_to_symplectic_2q(u, flagnonclifford=True):
    """
    Returns the symplectic representation of a two-qubit Clifford unitary,
    input as a complex matrix in the standard computational basis.

    Parameters
    ----------
    u : numpy array
        The unitary matrix to construct the symplectic representation for. This
        must be a two-qubit gate (so, it is a 4 x 4 matrix), and it must be
        in the standard computational basis. It also must be a Clifford gate in the
        standard sense.

    flagnonclifford : bool, opt
        If True, n ValueError is raised when the input unitary is not a Clifford gate.
        If False, when the unitary is not a Clifford the returned s and p are
        None.

    Returns
    -------
    s : numpy array or None
        The symplectic matrix representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False

    p : numpy array or None
        The phase vector representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False

    """
    assert(_np.shape(u) == (4, 4)), "Input is not a two-qubit unitary!"

    x = _np.array([[0, 1.], [1., 0]])
    z = _np.array([[1., 0], [0, -1.]])
    i = _np.array([[1., 0.], [0., 1.]])

    xi = _np.kron(x, i)
    ix = _np.kron(i, x)
    zi = _np.kron(z, i)
    iz = _np.kron(i, z)

    fund_paulis = [xi, ix, zi, iz]

    s = _np.zeros((4, 4), int)
    p = _np.zeros(4, int)

    for pauli_label in range(0, 4):

        # Calculate the matrix that the input unitary transforms the current Pauli group
        # generator to (xi, ix, ...).
        conj = _np.dot(_np.dot(u, fund_paulis[pauli_label]), _np.linalg.inv(u))

        # Find which element of the two-qubit Pauli group this is, and fill out the relevant
        # bits of the symplectic matrix and phase vector as this implies.
        for xi_l in range(0, 2):
            for ix_l in range(0, 2):
                for zi_l in range(0, 2):
                    for iz_l in range(0, 2):
                        for phase_l in range(0, 4):

                            tempx = _np.dot(_np.linalg.matrix_power(xi, xi_l), _np.linalg.matrix_power(ix, ix_l))
                            tempz = _np.dot(_np.linalg.matrix_power(zi, zi_l), _np.linalg.matrix_power(iz, iz_l))
                            pauli = (1j**(phase_l)) * _np.dot(tempx, tempz)

                            if _np.allclose(conj, pauli):
                                s[:, pauli_label] = _np.array([xi_l, ix_l, zi_l, iz_l])
                                p[pauli_label] = phase_l

    valid_clifford = check_valid_clifford(s, p)

    if flagnonclifford and not valid_clifford:
        raise ValueError("Input unitary is not a Clifford with respect to the standard basis!")

    else:
        if not valid_clifford:
            s = None
            p = None

    return s, p


@smart_cached
def unitary_to_symplectic(u, flagnonclifford=True):
    """
    Returns the symplectic representation of a one-qubit or two-qubit Clifford unitary.

    The Clifford is input as a complex matrix in the standard computational basis.

    Parameters
    ----------
    u : numpy array
        The unitary matrix to construct the symplectic representation for. This
        must be a one-qubit or two-qubit gate (so, it is a 2 x 2 or 4 x 4 matrix), and
        it must be provided in the standard computational basis. It also must be a
        Clifford gate in the standard sense.

    flagnonclifford : bool, opt
        If True, a ValueError is raised when the input unitary is not a Clifford gate.
        If False, when the unitary is not a Clifford the returned s and p are
        None.

    Returns
    -------
    s : numpy array or None
        The symplectic matrix representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False
    p : numpy array or None
        The phase vector representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False
    """
    assert(_np.shape(u) == (2, 2) or _np.shape(u) == (4, 4)), "Input must be a one or two qubit unitary!"

    if _np.shape(u) == (2, 2):
        s, p = _unitary_to_symplectic_1q(u, flagnonclifford)
    if _np.shape(u) == (4, 4):
        s, p = _unitary_to_symplectic_2q(u, flagnonclifford)

    return s, p


def random_symplectic_matrix(n, convention='standard'):
    """
    Returns a symplectic matrix of dimensions 2n x 2n sampled uniformly at random from the symplectic group S(n).

    This uses the method of Robert Koenig and John A. Smolin, presented in "How to
    efficiently select an arbitrary Clifford group element".

    Parameters
    ----------
    n : int
        The size of the symplectic group to sample from.

    convention : str, optional
        Can be either 'standard' or 'directsum', which correspond to two different
        definitions for the symplectic form. In the case of 'standard', the symplectic
        form is the 2n x 2n matrix of ((0,1),(1,0)), where '1' and '0' are the identity
        and all-zeros matrices of size n x n. The 'standard' symplectic form is the
        convention used throughout most of the code. In the case of 'directsum', the
        symplectic form is the direct sum of n 2x2 bit-flip matrices.

    Returns
    -------
    s : numpy array
        A uniformly sampled random symplectic matrix.
    """
    index = random_symplectic_index(n)
    s = compute_symplectic_matrix(index, n)

    if convention == 'standard':
        s = change_symplectic_form_convention(s)

    return s


def random_clifford(n):
    """
    Returns a Clifford, in the symplectic representation, sampled uniformly at random from the n-qubit Clifford group.

    The core of this function uses the method of Robert Koenig and John A. Smolin,
    presented in "How to efficiently select an arbitrary Clifford group element", for
    sampling a uniformly random symplectic matrix.

    Parameters
    ----------
    n : int
        The number of qubits the Clifford group is over.

    Returns
    -------
    s : numpy array
        The symplectic matrix representating the uniformly sampled random Clifford.
    p : numpy array
        The phase vector representating the uniformly sampled random Clifford.
    """
    s = random_symplectic_matrix(n, convention='standard')
    p = random_phase_vector(s, n)

    return s, p


def random_phase_vector(s, n):
    """
    Generates a uniformly random phase vector for a n-qubit Clifford.

    (This vector, together with the provided symplectic matrix, define a valid
    Clifford operation.)  In combination with a uniformly random `s` the returned `p`
    defines a uniformly random Clifford gate.

    Parameters
    ----------
    s : numpy array
        The symplectic matrix to construct a random phase vector

    n : int
        The number of qubits the Clifford group is over.

    Returns
    -------
    p : numpy array
        A phase vector sampled uniformly at random from all those phase
        vectors that, as a pair with `s`, define a valid n-qubit Clifford.
    """
    p = _np.zeros(2 * n, int)

    # A matrix to hold all possible phase vectors -- half of which do not, when
    # combined with the sampled symplectic matrix -- represent Cliffords.
    all_values = _np.zeros((2 * n, 4), int)
    for i in range(0, 2 * n):
        all_values[i, :] = _np.array([0, 1, 2, 3])

    # We now work out which of these are valid choices for the phase vector.
    possible = _np.zeros((2 * n, 4), bool)

    u = _np.zeros((2 * n, 2 * n), int)
    u[n:2 * n, 0:n] = _np.identity(n, int)
    v = _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s), u), s))
    v_matrix = _np.zeros((2 * n, 4), int)

    for i in range(0, 4):
        v_matrix[:, i] = v

    summed = all_values + v_matrix
    possible[summed % 2 == 0] = True

    # The valid choices for the phase vector, to sample uniformly from.
    allowed_values = _np.reshape(all_values[possible], (2 * n, 2))

    # Sample a uniformly random valid phase vector.
    index = _np.random.randint(2, size=2 * n)
    for i in range(0, 2 * n):
        p[i] = allowed_values[i, index[i]]

    assert(check_valid_clifford(s, p))

    return p


def bitstring_for_pauli(p):
    """
    Get the bitstring corresponding to a Pauli.

    The state, represented by a bitstring, that the Pauli operator represented by
    the phase-vector p creates when acting on the standard input state.

    Parameters
    ----------
    p : numpy.ndarray
        Phase vector of a symplectic representation, encoding a Pauli operation.

    Returns
    -------
    list
       A list of 0 or 1 elements.
    """
    n = len(p) // 2
    bitstring = p[n:]
    bitstring[bitstring > 0] = 1
    return list(bitstring)


def apply_internal_gate_to_symplectic(s, gate_name, qindex_list, optype='row'):
    """
    Applies a Clifford gate to the n-qubit Clifford gate specified by the 2n x 2n symplectic matrix.

    The Clifford gate is specified by the internally hard-coded name `gate_name`.
    This gate is applied to the qubits with *indices* in `qindex_list`, where these
    indices are w.r.t to indeices of `s`. This gate is applied from the left (right)
    of `s` if `optype` is 'row' ('column'), and has a row-action (column-action) on
    `s`. E.g., the Hadmard ('H') on qubit with index i swaps the ith row (or column)
    with the (i+n)th row (or column) of `s`; CNOT adds rows, etc.

    Note that this function *updates* `s`, and returns None.

    Parameters
    ----------
    s : np.array
        A even-dimension square array over [0,1] that is the symplectic representation
        of some (normally multi-qubit) Clifford gate.

    gate_name : str
        The gate name. Should be one of the gate-names of the hard-coded gates
        used internally in pyGSTi that is also a Clifford gate. Currently not
        all of those gates are supported, and `gate_name` must be one of:
        'H', 'P', 'CNOT', 'SWAP'.

    qindex_list : list or tuple
        The qubit indices that `gate_name` acts on (can be either length
        1 or 2 depending on whether the gate acts on 1 or 2 qubits).

    optype : {'row', 'column'}, optional
        Whether the symplectic operator type uses rows or columns:
        TODO: docstring - better explanation.

    Returns
    -------
    None
    """
    n = _np.shape(s)[0] // 2

    if gate_name == 'H':
        i = qindex_list[0]
        if optype == 'row': s[[i + n, i], :] = s[[i, i + n], :]
        elif optype == 'column': s[:, [i + n, i]] = s[:, [i, i + n]]
        else: raise ValueError("optype must be 'row' or 'column'!")
    elif gate_name == 'P':
        i = qindex_list[0]
        if optype == 'row': s[i + n, :] = s[i, :] ^ s[i + n, :]
        elif optype == 'column': s[:, i] = s[:, i] ^ s[:, i + n]
        else: raise ValueError("optype must be 'row' or 'column'!")
    elif gate_name == 'CNOT':
        control = qindex_list[0]
        target = qindex_list[1]
        if optype == 'row':
            s[target, :] = s[target, :] ^ s[control, :]
            s[control + n, :] = s[target + n, :] ^ s[control + n, :]
        elif optype == 'column':
            s[:, control] = s[:, control] ^ s[:, target]
            s[:, target + n] = s[:, target + n] ^ s[:, control + n]
        else: raise ValueError("optype must be 'row' or 'column'!")
    elif gate_name == 'SWAP':
        i = qindex_list[0]
        j = qindex_list[1]
        if optype == 'row': s[[i, j, i + n, j + n], :] = s[[j, i, j + n, i + n], :]
        if optype == 'column': s[:, [i, j, i + n, j + n]] = s[:, [j, i, j + n, i + n]]
    else:
        raise ValueError("This gate name is incorrect or not currently supported!")

# The code below is taken from the appendix of "How to efficiently select an arbitrary Clifford
# group element", by Robert Koenig and John A. Smolin. It is almost exactly the same as that code,
# and has only had minor edits to make it work properly. A couple of the basic utility routines
# from that code have been moved into the matrixtools file.


def compute_num_cliffords(n):
    """
    The number of Clifford gates in the n-qubit Clifford group.

    Code from "How to efficiently select an arbitrary Clifford
    group element" by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    n : int
        The number of qubits the Clifford group is over.

    Returns
    -------
    long integer
        The cardinality of the n-qubit Clifford group.
    """
    return (4**int(n)) * compute_num_symplectics(n)


def compute_num_symplectics(n):
    """
    The number of elements in the symplectic group S(n) over the 2-element finite field.

    Code from "How to efficiently select an arbitrary Clifford group element"
    by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    n : int
        S(n) group parameter.

    Returns
    -------
    int
    """
    x = 1
    for j in range(1, n + 1):
        x = x * compute_num_cosets(j)

    return x


def compute_num_cosets(n):
    """
    Returns the number of different cosets for the symplectic group S(n) over the 2-element finite field.

    Code from "How to efficiently select an arbitrary Clifford group element"
    by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    n : int
        S(n) group parameter.

    Returns
    -------
    int
    """
    x = 2**int(2 * n - 1) * ((2**int(2 * n)) - 1)
    return x


def symplectic_innerproduct(v, w):
    """
    Returns the symplectic inner product of two vectors in F_2^(2n).

    Here F_2 is the finite field containing 0 and 1, and 2n is the length of
    the vectors. Code from "How to efficiently select an arbitrary Clifford
    group element" by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    v : numpy.ndarray
        A length-2n vector.

    w : numpy.ndarray
        A length-2n vector.

    Returns
    -------
    int
    """
    t = 0
    for i in range(0, _np.size(v) >> 1):
        t += v[2 * i] * w[2 * i + 1]
        t += w[2 * i] * v[2 * i + 1]
    return t % 2


def symplectic_transvection(k, v):
    """
    Applies transvection Z k to v.

    Code from "How to efficiently select an arbitrary Clifford group element
    by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    k : numpy.ndarray
        A length-2n vector.

    v : numpy.ndarray
        A length-2n vector.

    Returns
    -------
    numpy.ndarray
    """
    return (v + symplectic_innerproduct(k, v) * k) % 2


def int_to_bitstring(i, n):
    """
    Converts integer i to an length n array of bits.

    Code from "How to efficiently select an arbitrary Clifford group element
    by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    i : int
        Any integer.

    n : int
        Number of bits

    Returns
    -------
    numpy.ndarray
        Integer array of 0s and 1s.
    """
    output = _np.zeros(n, dtype='int8')
    for j in range(0, n):
        output[j] = i & 1
        i >>= 1

    return output


def bitstring_to_int(b, n):
    """
    Converts an `n`-bit string `b` to an integer between 0 and 2^`n` - 1.

    Code from "How to efficiently select an arbitrary Clifford group element"
    by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    b : list, tuple, or array
        Sequence of bits (a bitstring).

    n : int
        Number of bits.

    Returns
    -------
    int
    """
    output = 0
    tmp = 1

    for j in range(0, n):
        if b[j] == 1:
            output = output + tmp
        tmp = tmp * 2

    return output


def find_symplectic_transvection(x, y):
    """
    A utility function for selecting a random Clifford element.

    Code from "How to efficiently select an arbitrary Clifford group element"
    by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    x : numpy.ndarray
        A length-2n vector.

    y : numpy.ndarray
        A length-2n vector.

    Returns
    -------
    numpy.ndarray
    """
    # finds h1,h2 such that y = Z h1 Z h2 x
    # Lemma 2 in the text
    # Note that if only one transvection is required output [1] will be
    # zero and applying the all-zero transvection does nothing.

    output = _np.zeros((2, _np.size(x)), dtype='int8')
    if _np.array_equal(x, y):
        return output
    if symplectic_innerproduct(x, y) == 1:
        output[0] = (x + y) % 2
        return output

    # Try to find a pair where they are both not 00
    z = _np.zeros(_np.size(x))
    for i in range(0, _np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) != 0) and ((y[ii] + y[ii + 1]) != 0):  # found the pair
            z[ii] = (x[ii] + y[ii]) % 2
            z[ii + 1] = (x[ii + 1] + y[ii + 1]) % 2
            if (z[ii] + z[ii + 1]) == 0:  # they were the same so they added to 00
                z[ii + 1] = 1
                if x[ii] != x[ii + 1]:
                    z[ii] = 1
            output[0] = (x + z) % 2
            output[1] = (y + z) % 2
            return output

    #Failed to find any such pair, so look for two places where x has 00 and y does not,
    #and vice versa. First try y==00 and x does not.
    for i in range(0, _np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) != 0) and ((y[ii] + y[ii + 1]) == 0):  # found the pair
            if x[ii] == x[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = x[ii]
                z[ii] = x[ii + 1]
            break

    # finally try x==00 and y does not
    for i in range(0, _np.size(x) >> 1):
        ii = 2 * i
        if ((x[ii] + x[ii + 1]) == 0) and ((y[ii] + y[ii + 1]) != 0):  # found the pair
            if y[ii] == y[ii + 1]:
                z[ii + 1] = 1
            else:
                z[ii + 1] = y[ii]
                z[ii] = y[ii + 1]
            break

    output[0] = (x + z) % 2
    output[1] = (y + z) % 2

    return output


def compute_symplectic_matrix(i, n):
    """
    Returns the 2n x 2n symplectic matrix, over the finite field containing 0 and 1, with the "canonical" index `i`.

    Code from "How to efficiently select an arbitrary Clifford group element"
    by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    i : int
        Canonical index.

    n : int
        Number of qubits.

    Returns
    -------
    numpy.ndarray
    """
    # output symplectic canonical matrix i of size 2nX2n
    #Note, compared to the text the transpose of the symplectic matrix is returned.
    #This is not particularly important since Transpose(g in Sp(2n)) is in Sp(2n)
    #but it means the program doesnt quite agree with the algorithm in the text.
    #In python, row ordering of matrices is convenient , so it is used internally ,
    #but for column ordering is used in the text so that matrix multiplication of
    #symplectics will correspond to conjugation by unitaries as conventionally defined Eq. (2).
    #We cant just return the transpose every time as this would alternate doing the incorrect
    #thing as the algorithm recurses.

    nn = 2 * n

    # step 1
    s = ((1 << nn) - 1)
    k = (i % s) + 1
    i //= s

    # step 2
    f1 = int_to_bitstring(k, nn)

    # step 3
    e1 = _np.zeros(nn, dtype='int8')  # define first basis vectors
    e1[0] = 1
    T = find_symplectic_transvection(e1, f1)  # use Lemma 2 to compute T

    # step 4
    # b[0]=b in the text, b[1]...b[2n-2] are b_3...b_2n in the text
    bits = int_to_bitstring(i % (1 << (nn - 1)), nn - 1)

    # step 5
    eprime = _np.copy(e1)
    for j in range(2, nn):
        eprime[j] = bits[j - 1]

    h0 = symplectic_transvection(T[0], eprime)
    h0 = symplectic_transvection(T[1], h0)

    # step 6
    if bits[0] == 1:
        f1 *= 0

    #T' from the text will be Z_f1 Z_h0. If f1 has been set to zero it doesnt do anything.
    #We could now compute f2 as said in the text but step 7 is slightly
    # changed and will recompute f1, f2 for us anyway

    # step 7
    id2 = _np.identity(2, dtype='int8')

    if n != 1:
        g = _mtx.matrix_directsum(id2, compute_symplectic_matrix(i >> (nn - 1), n - 1))
    else:
        g = id2

    for j in range(0, nn):
        g[j] = symplectic_transvection(T[0], g[j])
        g[j] = symplectic_transvection(T[1], g[j])
        g[j] = symplectic_transvection(h0, g[j])
        g[j] = symplectic_transvection(f1, g[j])

    return g


def compute_symplectic_label(gn, n=None):
    """
    Returns the "canonical" index of 2n x 2n symplectic matrix `gn` over the finite field containing 0 and 1.

    Code from "How to efficiently select an arbitrary Clifford group element"
    by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    gn : numpy.ndarray
        symplectic matrix

    n : int, optional
        Number of qubits (if `None`, use `gn.shape[0] // 2`).

    Returns
    -------
    int
        The canonical index of `gn`.
    """
    # produce an index associated with group element gn

    if n is None:
        n = _np.shape(gn)[0] // 2

    nn = 2 * n

    # step 1
    v = gn[0]
    w = gn[1]

    # step 2
    e1 = _np.zeros(nn, dtype='int8')  # define first basis vectors
    e1[0] = 1
    T = find_symplectic_transvection(v, e1)  # use Lemma 2 to compute T

    # step 3
    tw = _np.copy(w)
    tw = symplectic_transvection(T[0], tw)
    tw = symplectic_transvection(T[1], tw)
    b = tw[0]
    h0 = _np.zeros(nn, dtype='int8')
    h0[0] = 1
    h0[1] = 0
    for j in range(2, nn):
        h0[j] = tw[j]

    # step 4
    bb = _np.zeros(nn - 1, dtype='int8')
    bb[0] = b
    for j in range(2, nn):
        bb[j - 1] = tw[j]
    zv = bitstring_to_int(v, nn) - 1
    zw = bitstring_to_int(bb, nn - 1)
    cvw = zw * ((2**int(2 * n)) - 1) + zv

    #step 5
    if n == 1:
        return cvw

    #step 6
    gprime = _np.copy(gn)
    if b == 0:
        for j in range(0, nn):
            gprime[j] = symplectic_transvection(T[1], symplectic_transvection(T[0], gn[j]))
            gprime[j] = symplectic_transvection(h0, gprime[j])
            gprime[j] = symplectic_transvection(e1, gprime[j])
    else:
        for j in range(0, nn):
            gprime[j] = symplectic_transvection(T[1], symplectic_transvection(T[0], gn[j]))
            gprime[j] = symplectic_transvection(h0, gprime[j])

    # step 7
    gnew = gprime[2:nn, 2:nn]  # take submatrix
    gnidx = compute_symplectic_label(gnew, n - 1) * compute_num_cosets(n) + cvw
    return gnidx


def random_symplectic_index(n):
    """
    The index of a uniformly random 2n x 2n symplectic matrix over the finite field containing 0 and 1.

    Code from "How to efficiently select an arbitrary Clifford group element"
    by Robert Koenig and John A. Smolin.

    Parameters
    ----------
    n : int
        Number of qubits (half dimension of symplectic matrix).

    Returns
    -------
    numpy.ndarray
    """
    cardinality = compute_num_symplectics(n)
    max_integer = 9223372036854775808  # The maximum integer of int64 type

    def zeros_string(k):
        zeros_str = ''
        for j in range(0, k):
            zeros_str += '0'
        return zeros_str

    if cardinality <= max_integer:
        index = _np.random.randint(cardinality)

    else:
        digits1 = len(str(cardinality))
        digits2 = len(str(max_integer)) - 1
        n = digits1 // digits2
        m = digits1 - n * digits2

        index = cardinality
        while index >= cardinality:

            temp = 0
            for i in range(0, n):
                add = zeros_string(m)
                sample = _np.random.randint(10**digits2, dtype=_np.int64)
                for j in range(0, i):
                    add += zeros_string(digits2)
                add += str(sample)
                for j in range(i + 1, n):
                    add += zeros_string(digits2)
                temp += int(add)

            add = str(_np.random.randint(10**m, dtype=_np.int64)) + zeros_string(n * digits2)
            index = int(add) + temp

    return index

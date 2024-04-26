"""
General matrix utilities. Some, but not all, are specific to matrices over the ints modulo 2.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

# Contains general matrix utilities. Some, but not all, of these tools are specific to

import numpy as _np


def dot_mod2(m1, m2):
    """
    Returns the product over the integers modulo 2 of two matrices.

    Parameters
    ----------
    m1 : numpy.ndarray
        First matrix

    m2 : numpy.ndarray
        Second matrix

    Returns
    -------
    numpy.ndarray
    """
    return m1 @ m2 % 2


def multidot_mod2(mlist):
    """
    Returns the product over the integers modulo 2 of a list of matrices.

    Parameters
    ----------
    mlist : list
        A list of matrices.

    Returns
    -------
    numpy.ndarray
    """
    return _np.linalg.multi_dot(mlist) % 2


def det_mod2(m):
    """
    Returns the determinant of a matrix over the integers modulo 2 (GL(n,2)).

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to take determinant of.

    Returns
    -------
    numpy.ndarray
    """
    return _np.round(_np.linalg.det(m)) % 2

# A utility function used by the random symplectic matrix sampler.


def matrix_directsum(m1, m2):
    """
    Returns the direct sum of two square matrices of integers.

    Parameters
    ----------
    m1 : numpy.ndarray
        First matrix

    m2 : numpy.ndarray
        Second matrix

    Returns
    -------
    numpy.ndarray
    """
    n1 = len(m1[0, :])
    n2 = len(m2[0, :])
    output = _np.zeros((n1 + n2, n1 + n2), dtype='int8')
    output[0:n1, 0:n1] = m1
    output[n1:n1 + n2, n1:n1 + n2] = m2

    return output


def inv_mod2(m):
    """
    Finds the inverse of a matrix over GL(n,2)

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to take inverse of.

    Returns
    -------
    numpy.ndarray
    """
    t = len(m)
    c = _np.append(m, _np.eye(t), 1)
    return _np.array(gaussian_elimination_mod2(c)[:, t:])


def Axb_mod2(A, b):  # noqa N803
    """
    Solves Ax = b over GF(2)

    Parameters
    ----------
    A : numpy.ndarray
        Matrix to operate on.

    b : numpy.ndarray
        Vector to operate on.

    Returns
    -------
    numpy.ndarray
    """
    b = _np.array([b]).T
    C = _np.append(A, b, 1)
    return _np.array([gaussian_elimination_mod2(C)[:, -1]]).T


def gaussian_elimination_mod2(a):
    """
    Gaussian elimination mod2 of a.

    Parameters
    ----------
    a : numpy.ndarray
        Matrix to operate on.

    Returns
    -------
    numpy.ndarray
    """

    a = _np.array(a, dtype='int')
    m, n = a.shape
    i, j = 0, 0

    while (i < m) and (j < n):
        k = a[i:m, j].argmax() + i
        a[_np.array([i, k]), :] = a[_np.array([k, i]), :]
        aijn = _np.array([a[i, j:]])
        col = _np.array([a[:, j]]).T
        col[i] = 0
        flip = col @ aijn
        a[:, j:] = _np.bitwise_xor(a[:, j:], flip)
        i += 1
        j += 1
    return a


def diagonal_as_vec(m):
    """
    Returns a 1D array containing the diagonal of the input square 2D array m.

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to operate on.

    Returns
    -------
    numpy.ndarray
    """
    l = _np.shape(m)[0]
    vec = _np.zeros(l, _np.int64)
    for i in range(0, l):
        vec[i] = m[i, i]
    return vec


def strictly_upper_triangle(m):
    """
    Returns a matrix containing the strictly upper triangle of m and zeros elsewhere.

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to operate on.

    Returns
    -------
    numpy.ndarray
    """
    l = _np.shape(m)[0]
    out = m.copy()

    for i in range(0, l):
        for j in range(0, i + 1):
            out[i, j] = 0

    return out


def diagonal_as_matrix(m):
    """
    Returns a diagonal matrix containing the diagonal of m.

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to operate on.

    Returns
    -------
    numpy.ndarray
    """
    l = _np.shape(m)[0]
    out = _np.zeros((l, l), _np.int64)

    for i in range(0, l):
        out[i, i] = m[i, i]

    return out

# Code for factorizing a symmetric matrix invertable matrix A over GL(n,2) into
# the form A = F F.T. The algorithm mostly follows the proof in *Orthogonal Matrices
# Over Finite Fields* by Jessie MacWilliams in The American Mathematical Monthly,
# Vol. 76, No. 2 (Feb., 1969), pp. 152-164


def albert_factor(d, failcount=0, rand_state=None):
    """
    Returns a matrix M such that d = M M.T for symmetric d, where d and M are matrices over [0,1] mod 2.

    The algorithm mostly follows the proof in "Orthogonal Matrices Over Finite
    Fields" by Jessie MacWilliams in The American Mathematical Monthly, Vol. 76,
    No. 2 (Feb., 1969), pp. 152-164

    There is generally not a unique albert factorization, and this algorthm is
    randomized. It will general return a different factorizations from multiple
    calls.

    Parameters
    ----------
    d : array-like
        Symmetric matrix mod 2.

    failcount : int, optional
        UNUSED.

    rand_state : np.random.RandomState, optional
        Random number generator to allow for determinism.

    Returns
    -------
    numpy.ndarray
    """
    d = _np.array(d, dtype='int')
    if rand_state is None: rand_state = _np.random.RandomState()

    proper = False
    while not proper:
        N = onesify(d, rand_state=rand_state)
        aa = multidot_mod2([N, d, N.T])
        P = proper_permutation(aa)
        A = multidot_mod2([P, aa, P.T])
        proper = _check_proper_permutation(A)

    t = len(A)

    # Start in lower right
    L = _np.array([[1]])

    for ind in range(t - 2, -1, -1):
        block = A[ind:, ind:].copy()
        z = block[0, 1:]
        B = block[1:, 1:]
        n = Axb_mod2(B, z).T
        x = _np.array(n @ L, dtype='int')
        zer = _np.zeros([t - ind - 1, 1])
        L = _np.array(_np.bmat([[_np.eye(1), x], [zer, L]]), dtype='int')

    Qinv = inv_mod2(dot_mod2(P, N))
    L = dot_mod2(_np.array(Qinv), L)

    return L


def random_bitstring(n, p, failcount=0, rand_state=None):
    """
    Constructs a random bitstring of length n with parity p

    Parameters
    ----------
    n : int
        Number of bits.

    p : int
        Parity.

    failcount : int, optional
        Internal use only.

    rand_state : np.random.RandomState, optional
        Random number generator to allow for determinism.

    Returns
    -------
    numpy.ndarray
    """
    if rand_state is None: rand_state = _np.random.RandomState()
    bitstring = rand_state.randint(0, 2, size=n)
    if _np.mod(sum(bitstring), 2) == p:
        return bitstring
    elif failcount < 100:
        return _np.array(random_bitstring(n, p, failcount + 1, rand_state), dtype='int')


def random_invertable_matrix(n, failcount=0, rand_state=None):
    """
    Finds a random invertable matrix M over GL(n,2)

    Parameters
    ----------
    n : int
        matrix dimension

    failcount : int, optional
        Internal use only.

    rand_state : np.random.RandomState, optional
        Random number generator to allow for determinism.

    Returns
    -------
    numpy.ndarray
    """
    if rand_state is None: rand_state = _np.random.RandomState()
    M = _np.array([random_bitstring(n, rand_state.randint(0, 2), rand_state=rand_state) for x in range(n)])
    if det_mod2(M) == 0:
        if failcount < 100:
            return random_invertable_matrix(n, failcount + 1, rand_state)
    else:
        return M


def random_symmetric_invertable_matrix(n, failcount=0, rand_state=None):
    """
    Creates a random, symmetric, invertible matrix from GL(n,2)

    Parameters
    ----------
    n : int
        Matrix dimension.

    failcount : int, optional
        Internal use only.

    rand_state : np.random.RandomState, optional
        Random number generator to allow for determinism.

    Returns
    -------
    numpy.ndarray
    """
    M = random_invertable_matrix(n, failcount, rand_state)
    return dot_mod2(M, M.T)


def onesify(a, failcount=0, maxfailcount=100, rand_state=None):
    """
    Returns M such that `M a M.T` has ones along the main diagonal

    Parameters
    ----------
    a : numpy.ndarray
        The matrix.

    failcount : int, optional
        Internal use only.

    maxfailcount : int, optional
        Maximum number of tries before giving up.

    rand_state : np.random.RandomState, optional
        Random number generator to allow for determinism.

    Returns
    -------
    numpy.ndarray
    """
    assert(failcount < maxfailcount), "The function has failed too many times! Perhaps the input is invalid."
    if rand_state is None: rand_state = _np.random.RandomState()

    # This is probably the slowest function since it just tries things
    t = len(a)
    count = 0
    test_string = _np.diag(a)

    M = []
    while (len(M) < t) and (count < 40):
        bitstr = random_bitstring(t, rand_state.randint(0, 2), rand_state=rand_state)
        if dot_mod2(bitstr, test_string) == 1:
            if not _np.any([_np.array_equal(bitstr, m) for m in M]):
                M += [bitstr]
            else:
                count += 1

    if len(M) < t:
        return onesify(a, failcount + 1, rand_state=rand_state)

    M = _np.array(M, dtype='int')

    if _np.array_equal(dot_mod2(M, inv_mod2(M)), _np.identity(t, _np.int64)):
        return _np.array(M)
    else:
        return onesify(a, failcount + 1, maxfailcount=maxfailcount, rand_state=rand_state)


def permute_top(a, i):
    """
    Permutes the first row & col with the i'th row & col

    Parameters
    ----------
    a : numpy.ndarray
        The matrix to act on.

    i : int
        index to permute with first row/col.

    Returns
    -------
    numpy.ndarray
    """
    t = len(a)
    P = _np.eye(t)
    P[0, 0] = 0
    P[i, i] = 0
    P[0, i] = 1
    P[i, 0] = 1
    return multidot_mod2([P, a, P]), P


def fix_top(a):
    """
    Computes the permutation matrix `P` such that the [1:t,1:t] submatrix of `P a P` is invertible.

    Parameters
    ----------
    a : numpy.ndarray
        A symmetric binary matrix with ones along the diagonal.

    Returns
    -------
    numpy.ndarray
    """
    if a.shape == (1, 1):
        return _np.eye(1, dtype='int')

    t = len(a)

    found_B = False
    for ind in range(t):
        aa, P = permute_top(a, ind)
        B = _np.round(aa[1:, 1:])

        if det_mod2(B) == 0:
            continue
        else:
            found_B = True
            break

    # Todo : put a more meaningful fail message here #
    assert(found_B), "Algorithm failed!"

    return P


def proper_permutation(a):
    """
    Computes the permutation matrix `P` such that all [n:t,n:t] submatrices of `P a P` are invertible.

    Parameters
    ----------
    a : numpy.ndarray
        A symmetric binary matrix with ones along the diagonal.

    Returns
    -------
    numpy.ndarray
    """
    t = len(a)
    Ps = []  # permutation matrices
    for ind in range(t):
        perm = fix_top(a[ind:, ind:])
        zer = _np.zeros([ind, t - ind])
        full_perm = _np.array(_np.bmat([[_np.eye(ind), zer], [zer.T, perm]]))
        a = multidot_mod2([full_perm, a, full_perm.T])
        Ps += [full_perm]
#     return Ps
    return multidot_mod2(list(reversed(Ps)))
    #return _np.linalg.multi_dot(list(reversed(Ps))) # Should this not be multidot_mod2 ?


def _check_proper_permutation(a):
    """
    Check to see if the matrix has been properly permuted.

    This should be redundent to what is already built into 'fix_top'.

    Parameters
    ----------
    a : numpy.ndarray
        A matrix.

    Returns
    -------
    bool
    """
    t = len(a)
    for ind in range(0, t):
        b = a[ind:, ind:]
        if det_mod2(b) == 0:
            return False
    return True

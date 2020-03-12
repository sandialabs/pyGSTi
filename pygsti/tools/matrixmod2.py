#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

# Contains general matrix utilities. Some, but not all, of these tools are specific to
# matrices over the ints modulo 2.

import numpy as _np


def dotmod2(m1, m2):
    """
    Returns the product over the itegers modulo 2 of
    two matrices.
    """
    return _np.dot(m1, m2) % 2


def multidotmod2(mlist):
    """
    Returns the product over the itegers modulo 2 of
    a list of matrices.
    """
    return _np.linalg.multi_dot(mlist) % 2


def detmod2(m):
    """
    Returns the determinant of a matrix over the itegers
    modulo 2 (GL(n,2)).
    """
    return _np.round(_np.linalg.det(m)) % 2

# A utility function used by the random symplectic matrix sampler.


def matrix_directsum(m1, m2):
    """
    Returns the direct sum of two square matrices of integers.
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
    """
    t = len(m)
    c = _np.append(m, _np.eye(t), 1)
    return _np.array(gaussian_elimination_mod2(c)[:, t:])


def Axb_mod2(A, b):  # noqa N803
    """
    Solves Ax = b over GF(2)

    """
    b = _np.array([b]).T
    C = _np.append(A, b, 1)
    return _np.array([gaussian_elimination_mod2(C)[:, -1]]).T


def gaussian_elimination_mod2(a):
    """
    Gaussian elimination mod2 of a.

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
        flip = _np.dot(col, aijn)
        a[:, j:] = _np.bitwise_xor(a[:, j:], flip)
        i += 1
        j += 1
    return a


def diagonal_as_vec(m):
    """
    Returns a 1D array containing the diagonal of the input square 2D array m.

    """
    l = _np.shape(m)[0]
    vec = _np.zeros(l, int)
    for i in range(0, l):
        vec[i] = m[i, i]
    return vec


def strictly_upper_triangle(m):
    """
    Returns a matrix containing the strictly upper triangle of m and zeros elsewhere.

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

    """
    l = _np.shape(m)[0]
    out = _np.zeros((l, l), int)

    for i in range(0, l):
        out[i, i] = m[i, i]

    return out

# Code for factorizing a symmetric matrix invertable matrix A over GL(n,2) into
# the form A = F F.T. The algorithm mostly follows the proof in *Orthogonal Matrices
# Over Finite Fields* by Jessie MacWilliams in The American Mathematical Monthly,
# Vol. 76, No. 2 (Feb., 1969), pp. 152-164


def albert_factor(d, failcount=0):
    """
    Returns a matrix M such that d = M M.T for symmetric d, where d and M are
    matrices over [0,1] mod 2. The algorithm mostly follows the proof in "Orthogonal Matrices
    Over Finite Fields" by Jessie MacWilliams in The American Mathematical Monthly, Vol. 76, No. 2
    (Feb., 1969), pp. 152-164

    There is generally not a unique albert factorization, and this algorthm is randomized. It will
    general return a different factorizations from multiple calls.
    """
    d = _np.array(d, dtype='int')

    proper = False
    while not proper:
        N = onesify(d)
        aa = multidotmod2([N, d, N.T])
        P = proper_permutation(aa)
        A = multidotmod2([P, aa, P.T])
        proper = check_proper_permutation(A)

    t = len(A)

    # Start in lower right
    L = _np.array([[1]])

    for ind in range(t - 2, -1, -1):
        block = A[ind:, ind:].copy()
        z = block[0, 1:]
        B = block[1:, 1:]
        n = Axb_mod2(B, z).T
        x = _np.array(_np.dot(n, L), dtype='int')
        zer = _np.zeros([t - ind - 1, 1])
        L = _np.array(_np.bmat([[_np.eye(1), x], [zer, L]]), dtype='int')

    Qinv = inv_mod2(dotmod2(P, N))
    L = dotmod2(_np.array(Qinv), L)

    return L


def random_bitstring(n, p, failcount=0):
    """
    Constructs a random bitstring of length n with parity p
    """
    bitstring = _np.random.randint(0, 2, size=n)
    if _np.mod(sum(bitstring), 2) == p:
        return bitstring
    elif failcount < 100:
        return _np.array(random_bitstring(n, p, failcount + 1), dtype='int')


def random_invertable_matrix(n, failcount=0):
    """
    Finds a random invertable matrix M over GL(n,2)
    """
    M = _np.array([random_bitstring(n, _np.random.randint(0, 2)) for x in range(n)])
    if detmod2(M) == 0:
        if failcount < 100:
            return random_invertable_matrix(n, failcount + 1)
    else:
        return M


def random_symmetric_invertable_matrix(n):
    """
    Creates a random, symmetric, invertible matrix from GL(n,2)
    """
    M = random_invertable_matrix(n)
    return dotmod2(M, M.T)


def onesify(a, failcount=0, maxfailcount=100):
    """
    Returns M such that M a M.T has ones along the main diagonal
    """
    assert(failcount < maxfailcount), "The function has failed too many times! Perhaps the input is invalid."

    # This is probably the slowest function since it just tries things
    t = len(a)
    count = 0
    test_string = _np.diag(a)

    M = []
    while (len(M) < t) and (count < 40):
        bitstr = random_bitstring(t, _np.random.randint(0, 2))
        if dotmod2(bitstr, test_string) == 1:
            if not _np.any([_np.array_equal(bitstr, m) for m in M]):
                M += [bitstr]
            else:
                count += 1

    if len(M) < t:
        return onesify(a, failcount + 1)

    M = _np.array(M, dtype='int')

    if _np.array_equal(dotmod2(M, inv_mod2(M)), _np.identity(t, int)):
        return _np.array(M)
    else:
        return onesify(a, failcount + 1, maxfailcount=maxfailcount)


def permute_top(a, i):
    """
    Permutes the first row & col with the i'th row & col

    """
    t = len(a)
    P = _np.eye(t)
    P[0, 0] = 0
    P[i, i] = 0
    P[0, i] = 1
    P[i, 0] = 1
    return multidotmod2([P, a, P]), P


def fix_top(a):
    """
    Takes a symmetric binary matrix with ones along the diagonal
    and returns the permutation matrix P such that the [1:t,1:t]
    submatrix of P a P is invertible

    """
    if a.shape == (1, 1):
        return _np.eye(1, dtype='int')

    t = len(a)

    found_B = False
    for ind in range(t):
        aa, P = permute_top(a, ind)
        B = _np.round_(aa[1:, 1:])

        if detmod2(B) == 0:
            continue
        else:
            found_B = True
            break

    # Todo : put a more meaningful fail message here #
    assert(found_B), "Algorithm failed!"

    return P


def proper_permutation(a):
    """
    Takes a symmetric binary matrix with ones along the diagonal
    and returns the permutation matrix P such that all [n:t,n:t]
    submatrices of P a P are invertible.

    """
    t = len(a)
    Ps = []  # permutation matrices
    for ind in range(t):
        perm = fix_top(a[ind:, ind:])
        zer = _np.zeros([ind, t - ind])
        full_perm = _np.array(_np.bmat([[_np.eye(ind), zer], [zer.T, perm]]))
        a = multidotmod2([full_perm, a, full_perm.T])
        Ps += [full_perm]
#     return Ps
    return multidotmod2(list(reversed(Ps)))
    #return _np.linalg.multi_dot(list(reversed(Ps))) # Should this not be multidot_mod2 ?


def check_proper_permutation(a):
    """
    Check to see if the matrix has been properly permuted
    This should be redundent to what is already built into
    'fix_top'.

    """
    t = len(a)
    for ind in range(0, t):
        b = a[ind:, ind:]
        if detmod2(b) == 0:
            return False
    return True

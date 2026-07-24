"""
General matrix utilities. Some, but not all, are specific to matrices over the ints modulo 2.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

# Contains general matrix utilities. Some, but not all, of these tools are specific to
from typing import Optional, Sequence, Tuple, Union

import numpy as _np
import scipy.sparse as _sps


def dot_mod2(m1: _np.ndarray, m2: _np.ndarray) -> _np.ndarray:
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
    return _np.dot(m1, m2) % 2


def multidot_mod2(mlist: Sequence[_np.ndarray]) -> _np.ndarray:
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


def det_mod2(m: _np.ndarray) -> Union[_np.floating, _np.ndarray]:
    """
    Returns the determinant of a matrix over the integers modulo 2 (GL(n,2)).

    This is computed exactly, via Gaussian elimination mod 2, rather than by
    rounding the (floating-point) result of `numpy.linalg.det`. The latter
    approach is unreliable for even moderately-sized matrices: the true
    integer determinant of an n x n binary matrix can be astronomically
    large, and floating-point LU decomposition accumulates enough rounding
    error that the rounded, mod-2-reduced result becomes meaningless well
    before n reaches 50.

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to take determinant of.

    Returns
    -------
    float
        1.0 if m is invertible over GF(2), otherwise 0.0.
    """
    m = _np.array(m, dtype='int')
    n = m.shape[0]
    reduced = gaussian_elimination_mod2(m)
    return 1.0 if _np.array_equal(reduced, _np.eye(n, dtype='int')) else 0.0

# A utility function used by the random symplectic matrix sampler.


def matrix_directsum(m1: _np.ndarray, m2: _np.ndarray) -> _np.ndarray:
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


def inv_mod2(m: _np.ndarray) -> _np.ndarray:
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


def Axb_mod2(A: _np.ndarray, b: _np.ndarray) -> _np.ndarray:  # noqa N803
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


def _eliminate_with_pivot_mod2(a, i, j):
    """
    Zeros out column j in every row except row i, mod 2.

    Given that `a[i, j]` is a nonzero pivot, this XORs row i (restricted to
    columns j onward) into every other row that has a 1 in column j. This is
    the shared elimination step used by both `gaussian_elimination_mod2`
    (which searches for and swaps in a pivot row before calling this) and
    `_check_proper_permutation` (which uses only the natural diagonal
    pivot, without any row searching/swapping).

    Operates in-place on `a[:, j:]`.

    Parameters
    ----------
    a : numpy.ndarray
        Matrix to operate on (modified in-place). Must satisfy `a[i, j] == 1`.

    i : int
        Pivot row index.

    j : int
        Pivot column index.

    Returns
    -------
    None
    """
    pivot_row = a[i:i + 1, j:]  # view, not a copy: only read below, before `a` is mutated
    col = a[:, j].copy()  # must be a copy: mutated in-place on the next line
    col[i] = 0
    a[:, j:] = _np.bitwise_xor(a[:, j:], _np.dot(col.reshape(-1, 1), pivot_row))


def _eliminate_below_pivot_mod2(a, k):
    """
    Zeros out column k in every row below row k, mod 2 (one-sided elimination).

    Given that `a[k, k]` is a nonzero pivot, this XORs row k (restricted to
    columns k onward) into every row below row k that has a 1 in column k.
    Unlike `_eliminate_with_pivot_mod2` (which zeros out column j in *every*
    other row, both above and below, as needed for full Gauss-Jordan
    reduction), this only eliminates *below* the pivot. That is what is
    needed to build a genuine LU decomposition (as opposed to a fully
    row-reduced form): the entries eliminated here become 0 in `U`, while
    the eliminated column (before elimination) is recorded by the caller as
    the corresponding column of `L`.

    Operates in-place on `a[k + 1:, k:]`.

    Parameters
    ----------
    a : numpy.ndarray
        Matrix to operate on (modified in-place). Must satisfy `a[k, k] == 1`.

    k : int
        Pivot row/column index.

    Returns
    -------
    None
    """
    pivot_row = a[k:k + 1, k:]  # view, not a copy: only read below, before `a` is mutated
    col = a[k + 1:, k].copy()  # must be a copy: mutated in-place on the next line
    a[k + 1:, k:] = _np.bitwise_xor(a[k + 1:, k:], _np.dot(col.reshape(-1, 1), pivot_row))


def _lu_no_pivot_mod2(a: _np.ndarray) -> Tuple[_np.ndarray, _np.ndarray]:
    """
    LU decomposition mod 2 of `a`, without pivoting.

    Returns `L` (unit lower triangular) and `U` (upper triangular) such
    that `dot_mod2(L, U) == a`. This only works if `a` does not require row
    pivoting to eliminate, i.e., all of its leading principal minors are
    nonzero mod 2 (equivalently, invertible over GF(2)) -- callers are
    responsible for ensuring this; if a zero pivot is encountered anyway,
    this raises `ValueError` rather than silently returning an incorrect
    factorization.

    Because no pivot search/row-swapping is needed, this costs O(t^3) for a
    `t x t` matrix, same as a single `gaussian_elimination_mod2` call. Its
    purpose is to let callers who need to solve several *nested* linear
    systems against leading (or, after reversing, trailing) principal
    submatrices of the same fixed matrix do so with a single O(t^3)
    factorization followed by O(m^2) triangular solves per submatrix,
    rather than paying a fresh O(m^3) Gaussian elimination for each one.

    Parameters
    ----------
    a : numpy.ndarray
        Square matrix to decompose. All leading principal minors must be
        nonzero mod 2.

    Returns
    -------
    L : numpy.ndarray
        Unit lower triangular matrix mod 2.

    U : numpy.ndarray
        Upper triangular matrix mod 2.
    """
    u = _np.array(a, dtype='int').copy()
    t = u.shape[0]
    l_mat = _np.eye(t, dtype='int')

    for k in range(t):
        if u[k, k] == 0:
            raise ValueError(
                "_lu_no_pivot_mod2 encountered a zero pivot at index %d; "
                "the input matrix does not have all leading principal "
                "minors invertible mod 2, so a pivot-free LU decomposition "
                "does not exist." % k
            )
        if k < t - 1:
            l_mat[k + 1:, k] = u[k + 1:, k]
            _eliminate_below_pivot_mod2(u, k)

    return l_mat, u


def _forward_substitution_mod2(l_mat: _np.ndarray, b: _np.ndarray) -> _np.ndarray:
    """
    Solves `L y = b` mod 2 for `y`, where `L` is unit lower triangular.

    Parameters
    ----------
    l_mat : numpy.ndarray
        Unit lower triangular matrix mod 2.

    b : numpy.ndarray
        1D right-hand-side vector mod 2.

    Returns
    -------
    numpy.ndarray
        1D solution vector `y` mod 2.
    """
    m = l_mat.shape[0]
    y = _np.zeros(m, dtype='int')
    for i in range(m):
        y[i] = (b[i] ^ (_np.dot(l_mat[i, :i], y[:i]) % 2)) % 2
    return y


def _back_substitution_mod2(u_mat: _np.ndarray, y: _np.ndarray) -> _np.ndarray:
    """
    Solves `U x = y` mod 2 for `x`, where `U` is upper triangular with a
    nonzero (i.e. 1) diagonal.

    Parameters
    ----------
    u_mat : numpy.ndarray
        Upper triangular matrix mod 2, with nonzero diagonal entries.

    y : numpy.ndarray
        1D right-hand-side vector mod 2.

    Returns
    -------
    numpy.ndarray
        1D solution vector `x` mod 2.
    """
    m = u_mat.shape[0]
    x = _np.zeros(m, dtype='int')
    for i in range(m - 1, -1, -1):
        x[i] = (y[i] ^ (_np.dot(u_mat[i, i + 1:], x[i + 1:]) % 2)) % 2
    return x


def gaussian_elimination_mod2(a: _np.ndarray) -> _np.ndarray:
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
        _eliminate_with_pivot_mod2(a, i, j)
        i += 1
        j += 1
    return a


def diagonal_as_vec(m: _np.ndarray) -> _np.ndarray:
    """
    Returns a 1D array containing the diagonal of the input square 2D array m.

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to operate on. Must be a dense array (sparse matrices are
        not supported).

    Returns
    -------
    numpy.ndarray
    """
    if _sps.issparse(m):
        raise TypeError("diagonal_as_vec does not support sparse matrices; pass a dense numpy.ndarray.")
    return _np.diag(m).astype(_np.int64)


def strictly_upper_triangle(m: _np.ndarray) -> _np.ndarray:
    """
    Returns a matrix containing the strictly upper triangle of m and zeros elsewhere.

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to operate on. Must be a dense array (sparse matrices are
        not supported).

    Returns
    -------
    numpy.ndarray
    """
    if _sps.issparse(m):
        raise TypeError("strictly_upper_triangle does not support sparse matrices; pass a dense numpy.ndarray.")
    return _np.triu(m, k=1)


def diagonal_as_matrix(m: _np.ndarray) -> _np.ndarray:
    """
    Returns a diagonal matrix containing the diagonal of m.

    Parameters
    ----------
    m : numpy.ndarray
        Matrix to operate on. Must be a dense array (sparse matrices are
        not supported).

    Returns
    -------
    numpy.ndarray
    """
    if _sps.issparse(m):
        raise TypeError("diagonal_as_matrix does not support sparse matrices; pass a dense numpy.ndarray.")
    return _np.diag(_np.diag(m)).astype(_np.int64)

# Code for factorizing a symmetric matrix invertable matrix A over GL(n,2) into
# the form A = F F.T. The algorithm mostly follows the proof in *Orthogonal Matrices
# Over Finite Fields* by Jessie MacWilliams in The American Mathematical Monthly,
# Vol. 76, No. 2 (Feb., 1969), pp. 152-164


def albert_factor(d: _np.ndarray, failcount: int = 0,
                   rand_state: Optional[_np.random.RandomState] = None) -> _np.ndarray:
    """
    Returns a matrix M such that d = M M.T for symmetric d, where d and M are matrices over [0,1] mod 2.

    The algorithm mostly follows the proof in "Orthogonal Matrices Over Finite
    Fields" by Jessie MacWilliams in The American Mathematical Monthly, Vol. 76,
    No. 2 (Feb., 1969), pp. 152-164

    There is generally not a unique albert factorization, and this algorithm is
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

    if t > 1:
        # The submatrices B_ind = A[ind+1:, ind+1:], for ind = t-2, ..., 0,
        # are nested trailing principal submatrices of the fixed matrix
        # A[1:, 1:] (each one is the previous one plus one more row and
        # column). Rather than solving a fresh Ax=b system (a full O(m^3)
        # Gaussian elimination via `Axb_mod2`) for each of the t-1 nested
        # submatrices -- which costs O(t^4) overall, since it sums O(m^3)
        # over m = 1, ..., t-1 -- we factor A[1:, 1:] into LU form *once*
        # (O(t^3)) and reuse truncated blocks of L and U to solve each
        # nested system via O(m^2) triangular substitutions instead, for a
        # total of O(t^3).
        #
        # Reversing row/column order turns "trailing submatrices of
        # A[1:, 1:]" into "leading submatrices of the reversed matrix"
        # (the same trick used in `_check_proper_permutation`), which is
        # exactly the structure needed for LU-block reuse: the leading
        # m x m blocks of the LU factors of a matrix are themselves the LU
        # factors of its leading m x m principal submatrix (valid here
        # because `A` was constructed so that all of its trailing/leading
        # principal submatrices are invertible, so no pivoting is needed).
        r = A[1:, 1:][::-1, ::-1]
        l_full, u_full = _lu_no_pivot_mod2(r)

        for ind in range(t - 2, -1, -1):
            m = t - 1 - ind  # size of B_ind = A[ind+1:, ind+1:]
            z = A[ind, ind + 1:]

            # B_ind is the row/column reversal of the leading m x m
            # submatrix of `r`. Conjugating by the reversal permutation
            # turns `B_ind @ x = z` into `r[:m, :m] @ reverse(x) = reverse(z)`.
            z_rev = z[::-1]
            y = _forward_substitution_mod2(l_full[:m, :m], z_rev)
            x_rev = _back_substitution_mod2(u_full[:m, :m], y)
            x_vec = x_rev[::-1]

            n = x_vec.reshape(1, -1)
            x = _np.array(dot_mod2(n, L), dtype='int')
            zer = _np.zeros([m, 1])
            L = _np.block([[_np.eye(1), x], [zer, L]]).astype('int')

    Qinv = inv_mod2(dot_mod2(P, N))
    L = dot_mod2(_np.array(Qinv), L)

    return L


def random_bitstring(n: int, p: int, failcount: int = 0,
                      rand_state: Optional[_np.random.RandomState] = None) -> Optional[_np.ndarray]:
    """
    Constructs a random bitstring of length n with parity p

    Parameters
    ----------
    n : int
        Number of bits.

    p : int
        Parity.

    failcount : int, optional
        UNUSED.

    rand_state : np.random.RandomState, optional
        Random number generator to allow for determinism.

    Returns
    -------
    numpy.ndarray
    """
    if rand_state is None: rand_state = _np.random.RandomState()
    if n == 0:
        return _np.array([], dtype='int')
    # Sample the first n-1 bits uniformly at random, then fix the last bit so
    # that the overall parity is p. Every valid bitstring is in bijection
    # with its own length-(n-1) prefix (the last bit is uniquely determined
    # by the parity constraint), so this produces a uniform distribution
    # over all bitstrings with parity p, without any retries.
    bits = rand_state.randint(0, 2, size=n - 1)
    last_bit = (p - _np.sum(bits)) % 2
    return _np.append(bits, last_bit).astype('int')


def random_invertable_matrix(n: int, failcount: int = 0,
                              rand_state: Optional[_np.random.RandomState] = None) -> Optional[_np.ndarray]:
    """
    Finds a random invertable matrix M over GL(n,2)

    Parameters
    ----------
    n : int
        matrix dimension

    failcount : int, optional
        UNUSED.

    rand_state : np.random.RandomState, optional
        Random number generator to allow for determinism.

    Returns
    -------
    numpy.ndarray
    """
    if rand_state is None: rand_state = _np.random.RandomState()
    for _ in range(100):
        M = _np.array([random_bitstring(n, rand_state.randint(0, 2), rand_state=rand_state) for x in range(n)])
        if det_mod2(M) == 1.0:
            return M
    raise RuntimeError("Failed to generate a random invertible matrix over GL(n,2) after 100 attempts.")


def random_symmetric_invertable_matrix(n: int, failcount: int = 0,
                                        rand_state: Optional[_np.random.RandomState] = None) -> Optional[_np.ndarray]:
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


def onesify(a: _np.ndarray, failcount: int = 0, maxfailcount: int = 100,
            rand_state: Optional[_np.random.RandomState] = None) -> _np.ndarray:
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


def permute_top(a: _np.ndarray, i: int) -> Tuple[_np.ndarray, _np.ndarray]:
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
    # Conjugating by the transposition matrix P (below) only swaps row 0
    # with row i and column 0 with column i; everything else is left
    # unchanged. So we apply that directly instead of via two chained
    # matrix-matrix multiplications (which would cost O(t^3) instead of
    # O(t)).
    aa = a.copy()
    aa[[0, i]] = aa[[i, 0]]
    aa[:, [0, i]] = aa[:, [i, 0]]

    P = _np.eye(t, dtype='int')
    P[0, 0] = 0
    P[i, i] = 0
    P[0, i] = 1
    P[i, 0] = 1
    return aa, P


def fix_top(a: _np.ndarray) -> _np.ndarray:
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


def proper_permutation(a: _np.ndarray) -> _np.ndarray:
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
    a = _np.array(a, dtype='int').copy()

    # Track the accumulated permutation as an index array instead of
    # building a full t x t permutation matrix and chaining
    # `multidot_mod2` calls at each of the t iterations (and again at the
    # end, over all t of them). `perm_indices[k]` is the original row/column
    # index of `a` that currently sits at position `k`. Composing an
    # additional transposition into this array is O(1); doing the
    # equivalent with full permutation matrices is O(t^3) per step, i.e.
    # O(t^4) overall.
    perm_indices = _np.arange(t)

    for ind in range(t):
        # fix_top always returns a single-transposition permutation matrix
        # (or the identity, if no swap is needed): its first row has a
        # single 1, at the position it swaps with local index 0. Extract
        # that swap target directly, and apply it to `a` (and to the
        # tracked permutation) via a row/column swap rather than by
        # embedding it into a full t x t block matrix and multiplying it
        # in.
        perm = fix_top(a[ind:, ind:])
        local_swap = int(_np.argmax(perm[0, :]))
        j = ind + local_swap

        if j != ind:
            a[[ind, j], :] = a[[j, ind], :]
            a[:, [ind, j]] = a[:, [j, ind]]
            perm_indices[[ind, j]] = perm_indices[[j, ind]]

    # Materialize the accumulated permutation as a dense t x t matrix only
    # once, from the index array, to preserve this function's documented
    # return type (an actual permutation matrix).
    P = _np.zeros((t, t), dtype='int')
    P[_np.arange(t), perm_indices] = 1
    return P


def _check_proper_permutation(a: _np.ndarray) -> bool:
    """
    Check to see if the matrix has been properly permuted.

    This should be redundant to what is already built into 'fix_top'.

    This checks that every trailing principal submatrix `a[ind:, ind:]` is
    invertible over GF(2), for `ind = 0, ..., t-1`. Rather than calling
    `det_mod2` independently on each of the t submatrices (which costs
    O(t^4) overall, since submatrix k costs O(k^3)), this performs a single
    O(t^3) pass: reversing the row and column order of `a` turns the
    "trailing submatrices of a" question into a "leading submatrices of the
    reversed matrix" question (reversal is a permutation congruence, which
    preserves determinants mod 2), and a matrix has an LU decomposition
    without row pivoting (i.e. natural-order elimination never needs to
    swap in a different pivot row) if and only if all of its leading
    principal minors are nonzero. So a single natural-order elimination
    pass over the reversed matrix, checking that each diagonal pivot is
    nonzero as we go, answers the same question as the original per-submatrix
    loop.

    Parameters
    ----------
    a : numpy.ndarray
        A matrix.

    Returns
    -------
    bool
    """
    t = len(a)
    r = _np.array(a, dtype='int')[::-1, ::-1].copy()
    for k in range(t):
        if r[k, k] == 0:
            return False
        _eliminate_with_pivot_mod2(r, k, k)
    return True

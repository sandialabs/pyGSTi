"""
Matrix related utility functions
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import functools as _functools
import itertools as _itertools
import warnings as _warnings

from typing import Protocol, Any, runtime_checkable, TypeVar

import numpy as _np
import scipy.linalg as _spl
import scipy.optimize as _spo
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl

from pygsti.tools.basistools import change_basis

try:
    from . import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None

#EXPM_DEFAULT_TOL = 1e-7
EXPM_DEFAULT_TOL = 2**-53  # Scipy default

BLAS_FUNCS = {
    'herk': {
        's' : _spl.blas.ssyrk,
        'd' : _spl.blas.dsyrk,
        'c' : _spl.blas.cherk,
        'z': _spl.blas.zherk
    }
}

def gram_matrix(m, adjoint=False):
    """
    If adjoint=False, then return m.T.conj() @ m, computed in a more efficient way.

    If adjoint=True, return m @ m.T.conj(), likewise computed in a more efficient way.
    """
    assert isinstance(m, _np.ndarray)
    prefix_char, _, _ = _spl.blas.find_best_blas_type(dtype=m.dtype)
    herk = BLAS_FUNCS["herk"][prefix_char]

    if adjoint:
        trans = 0
    elif _np.iscomplexobj(m):
        trans = 2
    else:
        trans = 1
    out = herk(1.0, m, trans=trans)
    i_lower = _np.tril_indices(out.shape[0], -1)
    upper_values = out.T[i_lower]
    out[i_lower] = upper_values.real
    if trans > 0:
        out[i_lower] += upper_values.imag
    return out


def is_hermitian(mx, tol=1e-9):
    """
    Test whether mx is a hermitian matrix.

    Parameters
    ----------
    mx : numpy array
        Matrix to test.

    tol : float, optional
        Tolerance on absolute magitude of elements.

    Returns
    -------
    bool
        True if mx is hermitian, otherwise False.
    """
    (m, n) = mx.shape
    if m != n:
        return False
    else:
        return _np.all(_np.abs(mx - mx.T.conj()) <= tol)


def assert_hermitian(mat : _np.ndarray, tol: _np.floating) -> None:
    hermiticity_error = _np.abs(mat - mat.T.conj())
    if _np.any(hermiticity_error > tol):
        message = f"""
            Input matrix 'mat' is not Hermitian, up to tolerance {tol}.
            The absolute values of entries in (mat - mat^H) are \n{hermiticity_error}. 
        """
        raise ValueError(message)


def is_pos_def(mx, tol=1e-9, attempt_cholesky=False):
    """
    Test whether mx is a positive-definite matrix.

    Parameters
    ----------
    mx : numpy array
        Matrix to test.

    tol : float, optional
        Tolerance on absolute magitude of elements.

    Returns
    -------
    bool
        True if mx is positive-semidefinite, otherwise False.
    """
    if not is_hermitian(mx, tol):
        return False
    if attempt_cholesky:
        try:
            _ = _spl.cholesky(mx)
            return True # Cholesky succeeded
        except _spl.LinAlgError:
            pass # we fall back on eigenvalue decomposition
    evals = _np.linalg.eigvalsh(mx)
    return all([ev > -tol for ev in evals])


def is_valid_density_mx(mx, tol=1e-9):
    """
    Test whether mx is a valid density matrix (hermitian, positive-definite, and unit trace).

    Parameters
    ----------
    mx : numpy array
        Matrix to test.

    tol : float, optional
        Tolerance on absolute magitude of elements.

    Returns
    -------
    bool
        True if mx is a valid density matrix, otherwise False.
    """
    # is_pos_def includes a check that the matrix is Hermitian.
    return abs(_np.trace(mx) - 1.0) < tol and is_pos_def(mx, tol)


def nullspace(m, tol=1e-7):
    """
    Compute the nullspace of a matrix.

    Parameters
    ----------
    m : numpy array
        An matrix of shape (M,N) whose nullspace to compute.

    tol : float , optional
        Nullspace tolerance, used when comparing singular values with zero.

    Returns
    -------
    An matrix of shape (M,K) whose columns contain nullspace basis vectors.
    """
    _, s, vh = _np.linalg.svd(m)
    rank = (s > tol).sum()
    return vh[rank:].T.conjugate()


def nullspace_qr(m, tol=1e-7):
    """
    Compute the nullspace of a matrix using the QR decomposition.

    The QR decomposition is faster but less accurate than the SVD
    used by :func:`nullspace`.

    Parameters
    ----------
    m : numpy array
        An matrix of shape (M,N) whose nullspace to compute.

    tol : float , optional
        Nullspace tolerance, used when comparing diagonal values of R with zero.

    Returns
    -------
    An matrix of shape (M,K) whose columns contain nullspace basis vectors.
    """
    #if M,N = m.shape, and q,r,p = _spl.qr(...)
    # q.shape == (N,N), r.shape = (N,M), p.shape = (M,)
    q, r, _ = _spl.qr(m.T, mode='full', pivoting=True)
    rank = (_np.abs(_np.diagonal(r)) > tol).sum()

    #DEBUG: requires q,r,p = _sql.qr(...) above
    #assert( _np.linalg.norm(_np.dot(q,r) - m.T[:,p]) < 1e-8) #check QR decomp
    #print("Rank QR = ",rank)
    #print('\n'.join(map(str,_np.abs(_np.diagonal(r)))))
    #print("Ret = ", q[:,rank:].shape, " Q = ",q.shape, " R = ",r.shape)

    return q[:, rank:]


def nice_nullspace(m, tol=1e-7, orthogonalize=False):
    """
    Computes the nullspace of a matrix, and tries to return a "nice" basis for it.

    Columns of the returned value (a basis for the nullspace) each have a maximum
    absolute value of 1.0.

    Parameters
    ----------
    m : numpy array
        An matrix of shape (M,N) whose nullspace to compute.

    tol : float , optional
        Nullspace tolerance, used when comparing diagonal values of R with zero.

    orthogonalize : bool, optional
        If `True`, the nullspace vectors are additionally orthogonalized.

    Returns
    -------
    An matrix of shape (M,K) whose columns contain nullspace basis vectors.
    """
    nullsp = nullspace(m, tol)
    dim_ker = nullsp.shape[1]
    if dim_ker == 0:
        return nullsp  # empty 0-by-N array
    _, _, p = _spl.qr(nullsp.T.conj(), mode='raw', pivoting=True)
    ret = nullsp @ (nullsp.T[:, p[:dim_ker]]).conj()
    # ^ That's equivalent to, but faster than:
    #     nullsp_projector = nullsp @ nullsp.T.conj()
    #     _, _, p = _spl.qr(nullsp_projector mode='raw', pivoting=True)
    #     ret = nullsp_projector[:, p[:dim_ker]]

    if orthogonalize:
        ret, _ = _spl.qr(ret, mode='economic')
    for j in range(ret.shape[1]):  # normalize columns so largest element is +1.0
        imax = _np.argmax(_np.abs(ret[:, j]))
        if abs(ret[imax, j]) > 1e-6:
            ret[:, j] /= ret[imax, j]

    return ret


def normalize_columns(m, return_norms=False, ord=None):
    """
    Normalizes the columns of a matrix.

    Parameters
    ----------
    m : numpy.ndarray or scipy sparse matrix
        The matrix.

    return_norms : bool, optional
        If `True`, also return a 1D array containing the norms
        of the columns (before they were normalized).

    ord : int or list of ints, optional
        The order of the norm.  See :func:`numpy.linalg.norm`.  An
        array of orders can be given to specify the norm on a per-column
        basis.

    Returns
    -------
    normalized_m : numpy.ndarray
        The matrix after columns are normalized

    column_norms : numpy.ndarray
        Only returned when `return_norms=True`, a 1-dimensional array
        of the pre-normalization norm of each column.
    """
    norms = column_norms(m, ord)
    norms[norms == 0.0] = 1.0  # avoid division of zero-column by zero
    normalized_m = scale_columns(m, 1 / norms)
    return (normalized_m, norms) if return_norms else normalized_m


def column_norms(m, ord=None):
    """
    Compute the norms of the columns of a matrix.

    Parameters
    ----------
    m : numpy.ndarray or scipy sparse matrix
        The matrix.

    ord : int or list of ints, optional
        The order of the norm.  See :func:`numpy.linalg.norm`.  An
        array of orders can be given to specify the norm on a per-column
        basis.

    Returns
    -------
    numpy.ndarray
        A 1-dimensional array of the column norms (length is number of columns of `m`).
    """
    if _sps.issparse(m):
        ord_list = ord if isinstance(ord, (list, _np.ndarray)) else [ord] * m.shape[1]
        assert(len(ord_list) == m.shape[1])
        norms = _np.array([_np.linalg.norm(m[:, j].toarray(), ord=o) for j, o in enumerate(ord_list)])
    elif isinstance(ord, (list, _np.ndarray)):
        assert(len(ord) == m.shape[1])
        norms = _np.array([_np.linalg.norm(m[:, j], ord=o) for j, o in enumerate(ord)])
    else:
        norms = _np.linalg.norm(m, axis=0, ord=ord)

    return norms


def scale_columns(m, scale_values):
    """
    Scale each column of a matrix by a given value.

    Usually used for normalization purposes, when the
    matrix columns represent vectors.

    Parameters
    ----------
    m : numpy.ndarray or scipy sparse matrix
        The matrix.

    scale_values : numpy.ndarray
        A 1-dimensional array of scale values, one per
        column of `m`.

    Returns
    -------
    numpy.ndarray or scipy sparse matrix
        A copy of `m` with scaled columns, possibly with different sparsity structure.
    """
    if _sps.issparse(m):
        assert(len(scale_values) == m.shape[1])

        m_csc = _sps.csc_matrix(m)
        for j, scale in enumerate(scale_values):
            m_csc.data[m_csc.indptr[j]:m_csc.indptr[j + 1]] *= scale
        return m_csc
    else:
        return m * scale_values[None, :]


def columns_are_orthogonal(m, tol=1e-7):
    """
    Checks whether a matrix contains orthogonal columns.

    The columns do not need to be normalized.  In the
    complex case, two vectors v and w are considered orthogonal
    if `dot(v.conj(), w) == 0`.

    Parameters
    ----------
    m : numpy.ndarray
        The matrix to check.

    tol : float, optional
        Tolerance for checking whether dot products are zero.

    Returns
    -------
    bool
    """
    if m.size == 0:
        return True  # boundary case
    check = gram_matrix(m)
    check[_np.diag_indices_from(check)] = 0.0
    return bool(_np.linalg.norm(check) / check.size < tol)


def columns_are_orthonormal(m, tol=1e-7):
    """
    Checks whether a matrix contains orthogonal columns.

    The columns do not need to be normalized.  In the
    complex case, two vectors v and w are considered orthogonal
    if `dot(v.conj(), w) == 0`.

    Parameters
    ----------
    m : numpy.ndarray
        The matrix to check.

    tol : float, optional
        Tolerance for checking whether dot products are zero.

    Returns
    -------
    bool
    """
    if m.size == 0:
        return True  # boundary case
    check = gram_matrix(m)
    check[_np.diag_indices_from(check)] -= 1.0
    return bool(_np.linalg.norm(check) / check.size < tol)


def independent_columns(m, initial_independent_cols=None, tol=1e-7):
    """
    Computes the indices of the linearly-independent columns in a matrix.

    Optionally starts with a "base" matrix of independent columns, so that
    the returned indices indicate the columns of `m` that are independent
    of all the base columns and the other independent columns of `m`.

    Parameters
    ----------
    m : numpy.ndarray or scipy sparse matrix
        The matrix.

    initial_independent_cols : numpy.ndarray or scipy sparse matrix, optional
        If not `None`, a matrix of known-to-be independent columns so to test the
        columns of `m` with respect to (in addition to the already chosen independent
        columns of `m`.

    tol : float, optional
        Tolerance threshold used to decide whether a singular value is nonzero
        (it is if it's is greater than `tol`).

    Returns
    -------
    list
        A list of the independent-column indices of `m`.
    """
    if not _sps.issparse(m):
        if initial_independent_cols is None:
            proj_m = m.copy()
        else:
            # We assume initial_independent_cols is full column-rank.
            # This lets us use unpivoted QR instead of pivoted QR or SVD.
            assert initial_independent_cols.shape[0] == m.shape[0]
            q = _spl.qr(initial_independent_cols, mode='economic')[0]
            # proj_m = (I - qq')m
            temp1 = q.T.conj() @ m
            temp2 = q @ temp1
            proj_m = m - temp2

        rank = _np.linalg.matrix_rank(proj_m, tol=tol)
        pivots = _spl.qr(proj_m, overwrite_a=True, mode='raw', pivoting=True)[2]
        indep_cols = pivots[:rank].tolist()

    else:
        # TODO: re-implement to avoid unreliable calls to ARPACK's svds.
        indep_cols = []
        from scipy.sparse.linalg import ArpackNoConvergence as _ArpackNoConvergence
        from scipy.sparse.linalg import ArpackError as _ArpackError
        running_indep_cols = initial_independent_cols.copy() \
            if (initial_independent_cols is not None) else _sps.csc_matrix((m.shape[0], 0), dtype=m.dtype)

        for j in range(m.shape[1]):
            trial = _sps.hstack((running_indep_cols, m[:, j]))

            try:
                lowest_sval = _spsl.svds(trial, k=1, which="SM", return_singular_vectors=False)
            except (_ArpackNoConvergence, _ArpackError):
                lowest_sval = 0  # assume lack of convergence means smallest singular value was too small (?)

            if lowest_sval > tol:  # trial fogi dirs still linearly independent (full rank)
                running_indep_cols = trial
                indep_cols.append(j)
            # else trial column made fogi dirs linearly dependent and so don't tally indep column

    return indep_cols


def pinv_of_matrix_with_orthogonal_columns(m):
    """
    Return the matrix "pinv_m" so m @ pinvm and pinv_m @ m are orthogonal projectors
    onto subspaces of dimension rank(m).

    Parameters
    ----------
    m : numpy.ndarray

    Returns
    ----------
    pinv_m : numpy.ndarray
    """
    col_scaling = _np.linalg.norm(m, axis=0)**2
    m_with_scaled_cols = m.conj() * col_scaling[None, :]
    return m_with_scaled_cols.T


def matrix_sign(m):
    """
    Compute the matrix s = sign(m). The eigenvectors of s are the same as those of m.
    The eigenvalues of s are +/- 1, corresponding to the signs of m's eigenvalues.

    It's straightforward to compute s when m is a normal operator. If m is not normal,
    then the definition of s can be given in terms of m's Jordan form, and s
    can be computed by (suitably post-processing) the Schur decomposition of m.

    See https://nhigham.com/2020/12/15/what-is-the-matrix-sign-function/ for background.

    Parameters
    ----------
    m : numpy.ndarray
        the matrix.

    Returns
    -------
    numpy.ndarray
    """
    N = m.shape[0]
    assert(m.shape == (N, N)), "m must be square!"

    if is_hermitian(m):
        eigvals, eigvecs = _spl.eigh(m)
        sign = (eigvecs * _np.sign(eigvals)[None, :]) @ eigvecs.T.conj()
        return sign

    T, Z = _spl.schur(m, 'complex')    # m = Z T Z^H where Z is unitary and T is upper-triangular
    U = _np.zeros(T.shape, 'complex') 
    U[_np.diag_indices_from(U)] = _np.sign(_np.diagonal(T))
    # If T is diagonal, then we're basically done. If T isn't diagonal, then we have work to do.

    if not _np.all(_np.isclose(T[_np.triu_indices(N, 1)], 0.0)):
        # Use the extremely numerically stable (but expensive) method from
        # N. Higham's book, Functions of Matrices : Theory and Practice, Chapter 5.

        #Off diagonals: use U^2 = I or TU = UT
        # Note: Tij = Uij = 0 when i > j and i==j easy so just consider i<j case
        # 0 = sum_k Uik Ukj =  (i!=j b/c off-diag)
        # FUTURE: speed this up by using np.dot instead of sums below
        for j in range(1, N):
            for i in range(j - 1, -1, -1):
                S = U[i, i] + U[j, j]
                if _np.isclose(S, 0):  # then use TU = UT
                    if _np.isclose(T[i, i] - T[j, j], 0):  # then just set to zero
                        U[i, j] = 0.0  # TODO: check correctness of this case
                    else:
                        U[i, j] = T[i, j] * (U[i, i] - U[j, j]) / (T[i, i] - T[j, j]) + \
                            sum([U[i, k] * T[k, j] - T[i, k] * U[k, j] for k in range(i + 1, j)]) \
                            / (T[i, i] - T[j, j])
                else:  # use U^2 = I
                    U[i, j] = - sum([U[i, k] * U[k, j] for k in range(i + 1, j)]) / S
    else:
        pass

    sign = Z @ (U @ Z.T.conj())
    return sign



def print_mx(mx, width=9, prec=4, withbrackets=False):
    """
    Print matrix in pretty format.

    Will print real or complex matrices with a desired precision and
    "cell" width.

    Parameters
    ----------
    mx : numpy array
        the matrix (2-D array) to print.

    width : int, opitonal
        the width (in characters) of each printed element

    prec : int optional
        the precision (in characters) of each printed element

    withbrackets : bool, optional
        whether to print brackets and commas to make the result
        something that Python can read back in.

    Returns
    -------
    None
    """
    print(mx_to_string(mx, width, prec, withbrackets))


def mx_to_string(m, width=9, prec=4, withbrackets=False):
    """
    Generate a "pretty-format" string for a matrix.

    Will generate strings for real or complex matrices with a desired
    precision and "cell" width.

    Parameters
    ----------
    m : numpy.ndarray
        array to print.

    width : int, opitonal
        the width (in characters) of each converted element

    prec : int optional
        the precision (in characters) of each converted element

    withbrackets : bool, optional
        whether to print brackets and commas to make the result
        something that Python can read back in.

    Returns
    -------
    string
        matrix m as a pretty formated string.
    """
    if m.size == 0: return ""
    s = ""; tol = 10**(-prec)
    if _np.max(abs(_np.imag(m))) > tol:
        return mx_to_string_complex(m, width, width, prec)

    if len(m.shape) == 1: m = _np.array(m)[None, :]  # so it works w/vectors too
    if withbrackets: s += "["
    for i in range(m.shape[0]):
        if withbrackets: s += " [" if i > 0 else "["
        for j in range(m.shape[1]):
            if abs(m[i, j]) < tol: s += '{0: {w}.0f}'.format(0, w=width)
            else: s += '{0: {w}.{p}f}'.format(m[i, j].real, w=width, p=prec)
            if withbrackets and j + 1 < m.shape[1]: s += ","
        if withbrackets: s += "]," if i + 1 < m.shape[0] else "]]"
        s += "\n"
    return s


def mx_to_string_complex(m, real_width=9, im_width=9, prec=4):
    """
    Generate a "pretty-format" string for a complex-valued matrix.

    Parameters
    ----------
    m : numpy array
        array to format.

    real_width : int, opitonal
        the width (in characters) of the real part of each element.

    im_width : int, opitonal
        the width (in characters) of the imaginary part of each element.

    prec : int optional
        the precision (in characters) of each element's real and imaginary parts.

    Returns
    -------
    string
        matrix m as a pretty formated string.
    """
    if len(m.shape) == 1: m = m[None, :]  # so it works w/vectors too
    s = ""; tol = 10**(-prec)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if abs(m[i, j].real) < tol: s += "{0: {w}.0f}".format(0, w=real_width)
            else: s += "{0: {w}.{p}f}".format(m[i, j].real, w=real_width, p=prec)
            if abs(m[i, j].imag) < tol: s += "{0: >+{w}.0f}j".format(0, w=im_width)
            else: s += "{0: >+{w}.{p}f}j".format(m[i, j].imag, w=im_width, p=prec)
        s += "\n"
    return s


def unitary_superoperator_matrix_log(m, mx_basis):
    """
    Construct the logarithm of superoperator matrix `m`.

    This function assumes that `m` acts as a unitary on density-matrix space,
    (`m: rho -> U rho Udagger`) so that log(m) can be written as the action
    by Hamiltonian `H`:

    `log(m): rho -> -i[H,rho]`.

    Parameters
    ----------
    m : numpy array
        The superoperator matrix whose logarithm is taken

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        A matrix `logM`, of the same shape as `m`, such that `m = exp(logM)`
        and `logM` can be written as the action `rho -> -i[H,rho]`.
    """
    from . import lindbladtools as _lt  # (would create circular imports if at top)
    from . import optools as _ot  # (would create circular imports if at top)

    M_std = change_basis(m, mx_basis, "std")
    evals = _np.linalg.eigvals(M_std)
    assert(_np.allclose(_np.abs(evals), 1.0))  # simple but technically incomplete check for a unitary superop
    # (e.g. could be anti-unitary: diag(1, -1, -1, -1))

    U = _ot.std_process_mx_to_unitary(M_std)
    H = _spl.logm(U) / -1j  # U = exp(-iH)
    logM_std = _lt.create_elementary_errorgen('H', H)  # rho --> -i[H, rho]
    logM = change_basis(logM_std, "std", mx_basis)
    assert(_np.linalg.norm(_spl.expm(logM) - m) < 1e-8)  # expensive b/c of expm - could comment for performance
    return logM


def near_identity_matrix_log(m, tol=1e-8):
    """
    Construct the logarithm of superoperator matrix `m` that is near the identity.

    If `m` is real, the resulting logarithm will be real.

    Parameters
    ----------
    m : numpy array
        The superoperator matrix whose logarithm is taken

    tol : float, optional
        The tolerance used when testing for zero imaginary parts.

    Returns
    -------
    numpy array
        An matrix `logM`, of the same shape as `m`, such that `m = exp(logM)`
        and `logM` is real when `m` is real.
    """
    # A near-identity matrix should have a unique logarithm, and it should be
    # real if the original matrix is real
    M_is_real = bool(_np.linalg.norm(m.imag) < tol)
    logM = _spl.logm(m)
    if M_is_real:
        assert(_np.linalg.norm(logM.imag) < tol), \
            "Failed to construct a real logarithm! " \
            + "This is probably because m is not near the identity.\n" \
            + "Its eigenvalues are: " + str(_np.linalg.eigvals(m))
        logM = logM.real
    return logM


def approximate_matrix_log(m, target_logm, target_weight=10.0, tol=1e-6):
    """
    Construct an approximate logarithm of superoperator matrix `m` that is real and near the `target_logm`.


    The equation `m = exp( logM )` is allowed to become inexact in order to make
    `logM` close to `target_logm`.  In particular, the objective function that is
    minimized is (where `||` indicates the 2-norm):

    `|exp(logM) - m|_1 + target_weight * ||logM - target_logm||^2`

    Parameters
    ----------
    m : numpy array
        The superoperator matrix whose logarithm is taken

    target_logm : numpy array
        The target logarithm

    target_weight : float
        A weighting factor used to blance the exactness-of-log term
        with the closeness-to-target term in the optimized objective
        function.  This value multiplies the latter term.

    tol : float, optional
        Optimzer tolerance.

    Returns
    -------
    logM : numpy array
        An matrix of the same shape as `m`.
    """

    assert(_np.linalg.norm(m.imag) < 1e-8), "Argument `m` must be a *real* matrix!"
    mx_shape = m.shape

    def _objective(flat_logm):
        logM = flat_logm.reshape(mx_shape)
        testM = _spl.expm(logM)
        ret = target_weight * _np.linalg.norm(logM - target_logm)**2 + \
            _np.linalg.norm(testM.ravel() - m.ravel(), 1)
        #print("DEBUG: ",ret)
        return ret

        #Alt objective1: puts L1 on target term
        #return _np.linalg.norm(testM-m)**2 + target_weight*_np.linalg.norm(
        #                      logM.flatten() - target_logm.flatten(), 1)

        #Alt objective2: all L2 terms (ridge regression)
        #return target_weight*_np.linalg.norm(logM-target_logm)**2 + \
        #        _np.linalg.norm(testM - m)**2

    #from .. import optimize as _opt
    #print_obj_func = _opt.create_obj_func_printer(_objective) #only ever prints to stdout!
    print_obj_func = None

    logM = _np.real(real_matrix_log(m, action_if_imaginary="ignore"))  # just drop any imaginary part
    initial_flat_logM = logM.ravel()  # + 0.1*target_logm.flatten()
    # Note: adding some of target_logm doesn't seem to help; and hurts in easy cases

    if _objective(initial_flat_logM) > 1e-16:  # otherwise initial logM is fine!

        #print("Initial objective fn val = ",_objective(initial_flat_logM))
        #print("Initial inexactness = ",_np.linalg.norm(_spl.expm(logM)-m),
        #      _np.linalg.norm(_spl.expm(logM).flatten()-m.flatten(), 1),
        #      _np.linalg.norm(logM-target_logm)**2)

        solution = _spo.minimize(_objective, initial_flat_logM, options={'maxiter': 1000},
                                 method='L-BFGS-B', callback=print_obj_func, tol=tol)
        logM = solution.x.reshape(mx_shape)
        #print("Final objective fn val = ",_objective(solution.x))
        #print("Final inexactness = ",_np.linalg.norm(_spl.expm(logM)-m),
        #      _np.linalg.norm(_spl.expm(logM).flatten()-m.flatten(), 1),
        #      _np.linalg.norm(logM-target_logm)**2)

    return logM


def real_matrix_log(m, action_if_imaginary="raise", tol=1e-8):
    """
    Construct a *real* logarithm of real matrix `m`.

    This is possible when negative eigenvalues of `m` come in pairs, so
    that they can be viewed as complex conjugate pairs.

    Parameters
    ----------
    m : numpy array
        The matrix to take the logarithm of

    action_if_imaginary : {"raise","warn","ignore"}, optional
        What action should be taken if a real-valued logarithm cannot be found.
        "raise" raises a ValueError, "warn" issues a warning, and "ignore"
        ignores the condition and simply returns the complex-valued result.

    tol : float, optional
        An internal tolerance used when testing for equivalence and zero
        imaginary parts (real-ness).

    Returns
    -------
    logM : numpy array
        An matrix `logM`, of the same shape as `m`, such that `m = exp(logM)`
    """
    assert(_np.linalg.norm(_np.imag(m)) < tol), "real_matrix_log must be passed a *real* matrix!"
    evals, U = _np.linalg.eig(m)
    U = U.astype("complex")

    used_indices = set()
    neg_real_pairs_real_evecs = []
    neg_real_pairs_conj_evecs = []
    unpaired_indices = []
    for i, ev in enumerate(evals):
        if i in used_indices: continue
        used_indices.add(i)
        if abs(_np.imag(ev)) < tol and _np.real(ev) < 0:
            evec1 = U[:, i]
            if _np.linalg.norm(_np.imag(evec1)) < tol:
                # evec1 is real, so look for ev2 corresponding to another real evec
                for j, ev2 in enumerate(evals[i + 1:], start=i + 1):
                    if abs(ev - ev2) < tol and _np.linalg.norm(_np.imag(U[:, j])) < tol:
                        used_indices.add(j)
                        neg_real_pairs_real_evecs.append((i, j)); break
                else: unpaired_indices.append(i)
            else:
                # evec1 is complex, so look for ev2 corresponding to the conjugate of evec1
                evec1C = evec1.conjugate()
                for j, ev2 in enumerate(evals[i + 1:], start=i + 1):
                    if abs(ev - ev2) < tol and _np.linalg.norm(evec1C - U[:, j]) < tol:
                        used_indices.add(j)
                        neg_real_pairs_conj_evecs.append((i, j)); break
                else: unpaired_indices.append(i)

    log_evals = _np.log(evals.astype("complex"))
    # astype guards against case all evals are real but some are negative

    #DEBUG
    #print("DB: evals = ",evals)
    #print("DB: log_evals:",log_evals)
    #for i,ev in enumerate(log_evals):
    #    print(i,": ",ev, ",".join([str(j) for j in range(U.shape[0]) if abs(U[j,i]) > 0.05]))
    #print("DB: neg_real_pairs_real_evecs = ",neg_real_pairs_real_evecs)
    #print("DB: neg_real_pairs_conj_evecs = ",neg_real_pairs_conj_evecs)
    #print("DB: evec[5] = ",mx_to_string(U[:,5]))
    #print("DB: evec[6] = ",mx_to_string(U[:,6]))

    for (i, j) in neg_real_pairs_real_evecs:  # need to adjust evecs as well
        log_evals[i] = _np.log(-evals[i]) + 1j * _np.pi
        log_evals[j] = log_evals[i].conjugate()
        U[:, i] = (U[:, i] + 1j * U[:, j]) / _np.sqrt(2)
        U[:, j] = U[:, i].conjugate()

    for (i, j) in neg_real_pairs_conj_evecs:  # evecs already conjugates of each other
        log_evals[i] = _np.log(-evals[i].real) + 1j * _np.pi
        log_evals[j] = log_evals[i].conjugate()
        #Note: if *don't* conjugate j-th, then this picks *consistent* branch cut (what scipy would do), which
        # results, in general, in a complex logarithm BUT one which seems more intuitive (?) - at least permits
        # expected angle extraction, etc.

    logM = _np.dot(U, _np.dot(_np.diag(log_evals), _np.linalg.inv(U)))

    #if there are unpaired negative real eigenvalues, the logarithm might be imaginary
    mayBeImaginary = bool(len(unpaired_indices) > 0)
    imMag = _np.linalg.norm(_np.imag(logM))

    if mayBeImaginary and imMag > tol:
        if action_if_imaginary == "raise":
            raise ValueError("Cannot construct a real log: unpaired negative"
                             + " real eigenvalues: %s" % [evals[i] for i in unpaired_indices])
            #+ "\nDEBUG m = \n%s" % m + "\nDEBUG evals = %s" % evals)
        elif action_if_imaginary == "warn":
            _warnings.warn("Cannot construct a real log: unpaired negative"
                           + " real eigenvalues: %s" % [evals[i] for i in unpaired_indices])
        elif action_if_imaginary == "ignore":
            pass
        else:
            assert(False), "Invalid 'action_if_imaginary' argument: %s" % action_if_imaginary
    else:
        assert(imMag <= tol), "real_matrix_log failed to construct a real logarithm!"
        logM = _np.real(logM)

    return logM


## ------------------------ Erik : Matrix tools that Tim has moved here -----------

def column_basis_vector(i, dim):
    """
    Returns the ith standard basis vector in dimension dim.

    Parameters
    ----------
    i : int
        Basis vector index.

    dim : int
        Vector dimension.

    Returns
    -------
    numpy.ndarray
        An array of shape `(dim, 1)` that is all zeros except for
        its `i`-th element, which equals 1.
    """
    output = _np.zeros([dim, 1], float)
    output[i] = 1.
    return output


def vec(matrix_in):
    """
    Stacks the columns of a matrix to return a vector

    Parameters
    ----------
    matrix_in : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    return [b for a in _np.transpose(matrix_in) for b in a]


def norm1(m):
    """
    Returns the Schatten 1-norm of a matrix

    Parameters
    ----------
    m : numpy.ndarray
        The matrix.

    Returns
    -------
    numpy.ndarray
    """
    s = _spl.svdvals(m)
    nrm = _np.sum(s)
    return nrm


# Riley note: I'd like to rewrite this, but I don't want to mess with reproducibility
# issues. For now I've just made it a teeny bit more efficient.
def random_hermitian(dim):
    """
    Generates a random Hermitian matrix

    Parameters
    ----------
    dim : int
        the matrix dimensinon.

    Returns
    -------
    numpy.ndarray
    """
    my_norm = 0.
    while my_norm < 0.5:
        dim = int(dim)
        a = _np.random.random(size=[dim, dim])
        b = _np.random.random(size=[dim, dim])
        c = a + 1.j * b
        c += c.conj().T
        my_norm = norm1(c)
    return c / my_norm


def norm1to1(operator, num_samples=10000, mx_basis="gm"):
    """
    The Hermitian 1-to-1 norm of a superoperator represented in the standard basis.

    This is calculated via Monte-Carlo sampling.  The definition of Hermitian 1-to-1
    norm can be found in arxiv:1109.6887.

    Parameters
    ----------
    operator : numpy.ndarray
        The operator matrix to take the norm of.

    num_samples : int, optional
        Number of Monte-Carlo samples.

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis
        The basis of `operator`.

    Returns
    -------
    float or list
        Depends on the value of `return_list`.
    """
    std_operator = change_basis(operator, mx_basis, 'std')
    dim = int(_np.sqrt(len(std_operator)))
    max_val = 0.0
    for _ in range(num_samples):
        invec = random_hermitian(dim).ravel(order='F')
        outvec = std_operator @ invec
        val = norm1(outvec.reshape((dim,dim), order='F'))
        max_val = max(val, max_val)
    return max_val

## ------------------------ General utility fns -----------------------------------


def complex_compare(a, b):
    """
    Comparison function for complex numbers that compares real part, then imaginary part.

    Parameters
    ----------
    a : complex

    b : complex

    Returns
    -------
    -1 if a < b
    
    0 if a == b
    
    +1 if a > b
    """
    if a.real < b.real: return -1
    elif a.real > b.real: return 1
    elif a.imag < b.imag: return -1
    elif a.imag > b.imag: return 1
    else: return 0


def prime_factors(n):
    """
    GCD algorithm to produce prime factors of `n`

    Parameters
    ----------
    n : int
        The number to factorize.

    Returns
    -------
    list
        The prime factors of `n`.
    """
    i = 2; factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def minweight_match(a, b, metricfn=None, return_pairs=True,
                    pass_indices_to_metricfn=False):
    """
    Matches the elements of two vectors, `a` and `b` by minimizing the weight between them.

    The weight is defined as the sum of `metricfn(x,y)` over all `(x,y)` pairs
    (`x` in `a` and `y` in `b`).

    Parameters
    ----------
    a : list or numpy.ndarray
        First 1D array to match elements between.

    b : list or numpy.ndarray
        Second 1D array to match elements between.

    metricfn : function, optional
        A function of two float parameters, `x` and `y`,which defines the cost
        associated with matching `x` with `y`.  If None, `abs(x-y)` is used.

    return_pairs : bool, optional
        If True, the matching is also returned.

    pass_indices_to_metricfn : bool, optional
        If True, the metric function is passed two *indices* into the `a` and
        `b` arrays, respectively, instead of the values.

    Returns
    -------
    weight_array : numpy.ndarray
        The array of weights corresponding to the min-weight matching. The sum
        of this array's elements is the minimized total weight.
    pairs : list
        Only returned when `return_pairs == True`, a list of 2-tuple pairs of
        indices `(ix,iy)` giving the indices into `a` and `b` respectively of
        each matched pair.  The first (ix) indices will be in continuous
        ascending order starting at zero.
    """
    assert(len(a) == len(b))
    if metricfn is None:
        def metricfn(x, y): return abs(x - y)

    D = len(a)
    weightMx = _np.empty((D, D), 'd')

    if pass_indices_to_metricfn:
        for i, x in enumerate(a):
            weightMx[i, :] = _np.ravel(_np.array([metricfn(i, j) for j, y in enumerate(b)]))
    else:
        for i, x in enumerate(a):
            weightMx[i, :] = _np.ravel(_np.array([metricfn(x, y) for j, y in enumerate(b)]))

    a_inds, b_inds = _spo.linear_sum_assignment(weightMx)
    assert(_np.allclose(a_inds, range(D))), "linear_sum_assignment returned unexpected row indices!"

    matched_pairs = list(zip(a_inds, b_inds))
    min_weights = weightMx[a_inds, b_inds]

    if return_pairs:
        return min_weights, matched_pairs
    else:
        return min_weights


def minweight_match_realmxeigs(a, b, metricfn=None,
                               pass_indices_to_metricfn=False, eps=1e-9):
    """
    Matches the elements of `a` and `b`, whose elements are assumed to either real or one-half of a conjugate pair.

    Matching is performed by minimizing the weight between elements,
    defined as the sum of `metricfn(x,y)` over all `(x,y)` pairs
    (`x` in `a` and `y` in `b`).  If straightforward matching fails
    to preserve eigenvalue conjugacy relations, then real and conjugate-
    pair eigenvalues are matched *separately* to ensure relations are
    preserved (but this can result in a sub-optimal matching).  A
    ValueError is raised when the elements of `a` and `b` have incompatible
    conjugacy structures (#'s of conjugate vs. real pairs).

    Parameters
    ----------
    a : numpy.ndarray
        First 1D array to match.

    b : numpy.ndarray
        Second 1D array to match.

    metricfn : function, optional
        A function of two float parameters, `x` and `y`,which defines the cost
        associated with matching `x` with `y`.  If None, `abs(x-y)` is used.

    pass_indices_to_metricfn : bool, optional
        If True, the metric function is passed two *indices* into the `a` and
        `b` arrays, respectively, instead of the values.

    eps : float, optional
        Tolerance when checking if eigenvalues are equal to each other.

    Returns
    -------
    pairs : list
        A list of 2-tuple pairs of indices `(ix,iy)` giving the indices into
        `a` and `b` respectively of each matched pair.
    """

    def check(pairs):
        for i, (p0, p1) in enumerate(pairs):
            for q0, q1 in pairs[i + 1:]:
                a_conj = _np.isclose(a[p0], _np.conjugate(a[q0]))
                b_conj = _np.isclose(b[p1], _np.conjugate(b[q1]))
                if (abs(a[p0].imag) > 1e-6 and a_conj and not b_conj) or \
                   (abs(b[p1].imag) > 1e-6 and b_conj and not a_conj):
                    #print("DB: FALSE at: ",(p0,p1),(q0,q1),(a[p0],b[p1]),(a[q0],b[q1]),a_conj,b_conj)
                    return False
        return True

    #First attempt:
    # See if matching everything at once satisfies conjugacy relations
    # (if this works, this is the best, since it considers everything)
    _, pairs = minweight_match(a, b, metricfn, True,
                               pass_indices_to_metricfn)

    if check(pairs):
        return pairs  # we're done! that was easy

    #Otherwise we fall back to considering real values and conj pairs separately

    #identify real values and conjugate pairs
    def split_real_conj(ar):
        real_inds = []; conj_inds = []
        for i, v in enumerate(ar):
            if abs(v.imag) < eps: real_inds.append(i)
            else:
                for pair in conj_inds:
                    if i in pair: break  # ok, we've already found v's pair
                else:
                    for j, v2 in enumerate(ar[i + 1:], start=i + 1):
                        if _np.isclose(_np.conj(v), v2) and all([(j not in cpair) for cpair in conj_inds]):
                            conj_inds.append((i, j)); break
                    else:
                        raise ValueError("No conjugate pair found for %s" % str(v))

        # choose 'a+ib' to be representative of pair
        conj_rep_inds = [p0 if (ar[p0].imag > ar[p1].imag) else p1
                         for (p0, p1) in conj_inds]

        return real_inds, conj_inds, conj_rep_inds

    def add_conjpair(ar, conj_inds, conj_rep_inds, real_inds):
        for ii, i in enumerate(real_inds):
            for jj, j in enumerate(real_inds[ii + 1:], start=ii + 1):
                if _np.isclose(ar[i], ar[j]):
                    conj_inds.append((i, j))
                    conj_rep_inds.append(i)
                    del real_inds[jj]; del real_inds[ii]  # note: we know jj > ii
                    return True
        return False

    a_real, a_conj, a_reps = split_real_conj(a)  # hold indices to a & b arrays
    b_real, b_conj, b_reps = split_real_conj(b)  # hold indices to a & b arrays

    while len(a_conj) > len(b_conj):  # try to add real-pair(s) to b_conj
        if not add_conjpair(b, b_conj, b_reps, b_real):
            raise ValueError(("Vectors `a` and `b` don't have the same conjugate-pair structure, "
                              " and so they cannot be matched in a way the preserves this structure."))
    while len(b_conj) > len(a_conj):  # try to add real-pair(s) to a_conj
        if not add_conjpair(a, a_conj, a_reps, a_real):
            raise ValueError(("Vectors `a` and `b` don't have the same conjugate-pair structure, "
                              " and so they cannot be matched in a way the preserves this structure."))
    #Note: problem with this approach is that we might convert a
    # real-pair -> conj-pair sub-optimally (i.e. there might be muliple
    # such conversions and we just choose one at random).

    _, pairs1 = minweight_match(a[a_real], b[b_real], metricfn, True,
                                pass_indices_to_metricfn)
    _, pairs2 = minweight_match(a[a_reps], b[b_reps], metricfn, True,
                                pass_indices_to_metricfn)

    #pair1 gives matching of real values, pairs2 gives that of conj pairs.
    # Now just need to assemble a master pairs list to return.
    pairs = []
    for p0, p1 in pairs1:  # p0 & p1 are indices into a_real & b_real
        pairs.append((a_real[p0], b_real[p1]))
    for p0, p1 in pairs2:  # p0 & p1 are indices into a_reps & b_reps
        pairs.append((a_reps[p0], b_reps[p1]))
        a_other = a_conj[p0][0] if (a_conj[p0][0] != a_reps[p0]) else a_conj[p0][1]
        b_other = b_conj[p1][0] if (b_conj[p1][0] != b_reps[p1]) else b_conj[p1][1]
        pairs.append((a_other, b_other))

    return sorted(pairs, key=lambda x: x[0])  # sort by a's index


def _fas(a, inds, rhs, add=False):
    """
    Fancy Assignment, equivalent to `a[*inds] = rhs` but with
    the elements of inds (allowed to be integers, slices, or
    integer arrays) always specifing a generalize-slice along
    the given dimension.  This avoids some weird numpy indexing
    rules that make using square brackets a pain.
    """
    inds = tuple([slice(None) if (i is None) else i for i in inds])

    #Mixes of ints and tuples are fine, and a single
    # index-list index is fine too.  The case we need to
    # deal with is indexing a multi-dimensional array with
    # one or more index-lists
    if all([isinstance(i, (int, slice)) for i in inds]) or len(inds) == 1:
        if add:
            a[inds] += rhs  # all integers or slices behave nicely
        else:
            a[inds] = rhs  # all integers or slices behave nicely
    else:
        #convert each dimension's index to a list, take a product of
        # these lists, and flatten the right hand side to get the
        # proper assignment:
        b = []
        single_int_inds = []  # for Cython, a and rhs must have the same
        # number of dims.  This keeps track of single-ints
        for ii, i in enumerate(inds):
            if isinstance(i, (int, _np.int64)):
                b.append(_np.array([i], _np.int64))
                single_int_inds.append(ii)
            elif isinstance(i, slice):
                b.append(_np.array(list(range(*i.indices(a.shape[ii]))), _np.int64))
            else:
                b.append(_np.array(i, _np.int64))

        nDims = len(b)
        if nDims > 0 and all([len(x) > 0 for x in b]):  # b/c a[()] just returns the entire array!

            if _fastcalc is not None and not add:
                #Note: we rarely/never use add=True, so don't bother implementing in Cython yet...

                if len(single_int_inds) > 0:
                    remove_single_int_dims = [b[i][0] if (i in single_int_inds) else slice(None)
                                              for i in range(nDims)]  # e.g. [:,2,:] if index 1 is a single int
                    for ii in reversed(single_int_inds): del b[ii]  # remove single-int els of b
                    av = a[tuple(remove_single_int_dims)]  # a view into a
                    nDims -= len(single_int_inds)  # for cython routines below
                else:
                    av = a

                #Note: we do not require these arrays to be contiguous
                if nDims == 1:
                    _fastcalc.fast_fas_helper_1d(av, rhs, b[0])
                elif nDims == 2:
                    _fastcalc.fast_fas_helper_2d(av, rhs, b[0], b[1])
                elif nDims == 3:
                    _fastcalc.fast_fas_helper_3d(av, rhs, b[0], b[1], b[2])
                else:
                    raise NotImplementedError("No fas helper for nDims=%d" % nDims)
            else:
                indx_tups = list(_itertools.product(*b))
                inds = tuple(zip(*indx_tups))  # un-zips to one list per dim
                if add:
                    a[inds] += rhs.ravel()
                else:
                    a[inds] = rhs.ravel()

            #OLD DEBUG: just a reference for building the C-implementation (this is very slow in python!)
            ##Alt: C-able impl
            #indsPerDim = b # list of indices per dimension
            #nDims = len(inds)
            #b = [0]*nDims
            #a_strides = []; stride = 1
            #for s in reversed(a.shape):
            #    a_strides.insert(0,stride)
            #    stride *= s
            #rhs_dims = rhs.shape
            #
            #a_indx = 0
            #for i in range(nDims):
            #    a_indx += indsPerDim[i][0] * a_strides[i]
            #rhs_indx = 0
            #
            #while(True):
            #
            #    #a.flat[a_indx] = rhs.flat[rhs_indx]
            #    assert(_np.isclose(a.flat[a_indx],rhs.flat[rhs_indx]))
            #    rhs_indx += 1 # always increments by 1
            #
            #    #increment b ~ itertools.product & update vec_index_noop = _np.dot(self.multipliers, b)
            #    for i in range(nDims-1,-1,-1):
            #        if b[i]+1 < rhs_dims[i]:
            #            a_indx -= indsPerDim[i][b[i]] * a_strides[i]
            #            b[i] += 1
            #            a_indx += indsPerDim[i][b[i]] * a_strides[i]
            #            break
            #        else:
            #            a_indx -= indsPerDim[i][b[i]] * a_strides[i]
            #            b[i] = 0
            #            a_indx += indsPerDim[i][b[i]] * a_strides[i]
            #    else:
            #        break # can't increment anything - break while(True) loop

    return a


def _findx_shape(a, inds):
    """ Returns the shape of a fancy-indexed array (`a[*inds].shape`) """
    shape = []
    for ii, N in enumerate(a.shape):
        indx = inds[ii] if ii < len(inds) else None
        if indx is None: shape.append(N)
        elif isinstance(indx, slice):
            shape.append(len(range(*indx.indices(N))))
        else:  # assume indx is an index list or array
            shape.append(len(indx))
    return shape


def _findx(a, inds, always_copy=False):
    """
    Fancy Indexing, equivalent to `a[*inds].copy()` but with
    the elements of inds (allowed to be integers, slices, or
    integer arrays) always specifing a generalize-slice along
    the given dimension.  This avoids some weird numpy indexing
    rules that make using square brackets a pain.
    """
    inds = tuple([slice(None) if (i is None) else i for i in inds])

    #Mixes of ints and tuples are fine, and a single
    # index-list index is fine too.  The case we need to
    # deal with is indexing a multi-dimensional array with
    # one or more index-lists
    if all([isinstance(i, (int, slice)) for i in inds]) or len(inds) == 1:
        return a[inds].copy() if always_copy else a[inds]  # all integers or slices behave nicely

    else:
        #Need to copy to a new array
        b = []; squeeze = []
        for ii, i in enumerate(inds):
            if isinstance(i, int):
                b.append([i]); squeeze.append(ii)  # squeeze ii-th dimension at end
            elif isinstance(i, slice):
                b.append(list(range(*i.indices(a.shape[ii]))))
            else:
                b.append(list(i))

        a_inds_shape = [len(x) for x in b]
        indx_tups = list(_itertools.product(*b))
        if len(indx_tups) > 0:  # b/c a[()] just returns the entire array!
            inds = tuple(zip(*indx_tups))  # un-zips to one list per dim
            a_inds = a[inds].copy()  # a 1D array of flattened "fancy" a[inds]
            a_inds.shape = a_inds_shape  # reshape
        else:
            a_inds = _np.zeros(a_inds_shape, a.dtype)  # has zero elements
            assert(a_inds.size == 0)

        a_inds = a_inds.squeeze(axis=tuple(squeeze))
        return a_inds


def safe_norm(a, part=None):
    """
    Get the frobenius norm of a matrix or vector, `a`, when it is either a dense array or a sparse matrix.

    Parameters
    ----------
    a : ndarray or scipy.sparse matrix
        The matrix or vector to take the norm of.

    part : {None,'real','imag'}
        If not None, return the norm of the real or imaginary
        part of `a`.

    Returns
    -------
    float
    """
    if part == 'real': takepart = _np.real
    elif part == 'imag': takepart = _np.imag
    else: takepart = lambda x: x
    if _sps.issparse(a):
        assert(_sps.isspmatrix_csr(a)), "Non-CSR sparse formats not implemented"
        return _np.linalg.norm(takepart(a.data))
    else:
        return _np.linalg.norm(takepart(a))
    # could also use _spsl.norm(A)


def safe_onenorm(a):
    """
    Computes the 1-norm of the dense or sparse matrix `a`.

    Parameters
    ----------
    a : ndarray or sparse matrix
        The matrix or vector to take the norm of.

    Returns
    -------
    float
    """
    if _sps.isspmatrix(a):
        return sparse_onenorm(a)
    else:
        return _np.linalg.norm(a, 1)


def csr_sum_indices(csr_matrices):
    """
    Precomputes the indices needed to sum a set of CSR sparse matrices.

    Computes the index-arrays needed for use in :meth:`csr_sum`,
    along with the index pointer and column-indices arrays for constructing
    a "template" CSR matrix to be the destination of `csr_sum`.

    Parameters
    ----------
    csr_matrices : list
        The SciPy CSR matrices to be summed.

    Returns
    -------
    ind_arrays : list
        A list of numpy arrays giving the destination data-array indices
        of each element of `csr_matrices`.
    indptr, indices : numpy.ndarray
        The row-pointer and column-indices arrays specifying the sparsity
        structure of a the destination CSR matrix.
    N : int
        The dimension of the destination matrix (and of each member of
        `csr_matrices`)
    """
    if len(csr_matrices) == 0: return [], _np.empty(0, _np.int64), _np.empty(0, _np.int64), 0

    N = csr_matrices[0].shape[0]
    for mx in csr_matrices:
        assert(mx.shape == (N, N)), "Matrices must have the same square shape!"

    indptr = [0]
    indices = []
    csr_sum_array = [list() for mx in csr_matrices]

    #FUTURE sort column indices

    for iRow in range(N):
        dataInds = {}  # keys = column indices, values = data indices (for data in current row)

        for iMx, mx in enumerate(csr_matrices):
            for i in range(mx.indptr[iRow], mx.indptr[iRow + 1]):
                iCol = mx.indices[i]
                if iCol not in dataInds:  # add a new element to final mx
                    indices.append(iCol)
                    dataInds[iCol] = len(indices) - 1  # marks the final data index for this column
                csr_sum_array[iMx].append(dataInds[iCol])
        indptr.append(len(indices))

    #convert lists -> arrays
    csr_sum_array = [_np.array(lst, _np.int64) for lst in csr_sum_array]
    indptr = _np.array(indptr)
    indices = _np.array(indices)

    return csr_sum_array, indptr, indices, N


def csr_sum(data, coeffs, csr_mxs, csr_sum_indices):
    """
    Accelerated summation of several CSR-format sparse matrices.

    :meth:`csr_sum_indices` precomputes the necessary indices for
    summing directly into the data-array of a destination CSR sparse matrix.
    If `data` is the data-array of matrix `D` (for "destination"), then this
    method performs:

    `D += sum_i( coeff[i] * csr_mxs[i] )`

    Note that `D` is not returned; the sum is done internally into D's
    data-array.

    Parameters
    ----------
    data : numpy.ndarray
        The data-array of the destination CSR-matrix.

    coeffs : iterable
        The weight coefficients which multiply each summed matrix.

    csr_mxs : iterable
        A list of CSR matrix objects whose data-array is given by
        `obj.data` (e.g. a SciPy CSR sparse matrix).

    csr_sum_indices : list
        A list of precomputed index arrays as returned by
        :meth:`csr_sum_indices`.

    Returns
    -------
    None
    """
    for coeff, mx, inds in zip(coeffs, csr_mxs, csr_sum_indices):
        data[inds] += coeff * mx.data


def csr_sum_flat_indices(csr_matrices):
    """
    Precomputes quantities allowing fast computation of linear combinations of CSR sparse matrices.

    The returned quantities can later be used to quickly compute a linear
    combination of the CSR sparse matrices `csr_matrices`.

    Computes the index and data arrays needed for use in :meth:`csr_sum_flat`,
    along with the index pointer and column-indices arrays for constructing
    a "template" CSR matrix to be the destination of `csr_sum_flat`.

    Parameters
    ----------
    csr_matrices : list
        The SciPy CSR matrices to be summed.

    Returns
    -------
    flat_dest_index_array : numpy array
        A 1D array of one element per nonzero element in any of
        `csr_matrices`, giving the destination-index of that element.
    flat_csr_mx_data : numpy array
        A 1D array of the same length as `flat_dest_index_array`, which
        simply concatenates the data arrays of `csr_matrices`.
    mx_nnz_indptr : numpy array
        A 1D array of length `len(csr_matrices)+1` such that the data
        for the i-th element of `csr_matrices` lie in the index-range of
        mx_nnz_indptr[i] to mx_nnz_indptr[i+1]-1 of the flat arrays.
    indptr, indices : numpy.ndarray
        The row-pointer and column-indices arrays specifying the sparsity
        structure of a the destination CSR matrix.
    N : int
        The dimension of the destination matrix (and of each member of
        `csr_matrices`)
    """
    csr_sum_array, indptr, indices, N = csr_sum_indices(csr_matrices)
    if len(csr_sum_array) == 0:
        return (_np.empty(0, _np.int64), _np.empty(0, 'd'), _np.zeros(1, _np.int64), indptr, indices, N)

    flat_dest_index_array = _np.ascontiguousarray(_np.concatenate(csr_sum_array, axis=0), dtype=_np.int64)
    flat_csr_mx_data = _np.ascontiguousarray(_np.concatenate([mx.data for mx in csr_matrices], axis=0), dtype=complex)
    mx_nnz_indptr = _np.cumsum([0] + [mx.nnz for mx in csr_matrices], dtype=_np.int64)

    return flat_dest_index_array, flat_csr_mx_data, mx_nnz_indptr, indptr, indices, N


if _fastcalc is None:
    def csr_sum_flat(data, coeffs, flat_dest_index_array, flat_csr_mx_data, mx_nnz_indptr):
        """
        Computation of the summation of several CSR-format sparse matrices.

        :meth:`csr_sum_flat_indices` precomputes the necessary indices for
        summing directly into the data-array of a destination CSR sparse matrix.
        If `data` is the data-array of matrix `D` (for "destination"), then this
        method performs:

        `D += sum_i( coeff[i] * csr_mxs[i] )`

        Note that `D` is not returned; the sum is done internally into D's
        data-array.

        Parameters
        ----------
        data : numpy.ndarray
            The data-array of the destination CSR-matrix.

        coeffs : ndarray
            The weight coefficients which multiply each summed matrix.

        flat_dest_index_array : ndarray
            The index array generated by :func:`csr_sum_flat_indices`.

        flat_csr_mx_data : ndarray
            The data array generated by :func:`csr_sum_flat_indices`.

        mx_nnz_indptr : ndarray
            The number-of-nonzero-elements pointer array generated by
            :func:`csr_sum_flat_indices`.

        Returns
        -------
        None
        """
        Nmxs = len(mx_nnz_indptr) - 1  # the number of CSR matrices
        for iMx in range(Nmxs):
            coeff = coeffs[iMx]
            for i in range(mx_nnz_indptr[iMx], mx_nnz_indptr[iMx + 1]):
                data[flat_dest_index_array[i]] += coeff * flat_csr_mx_data[i]
else:
    def csr_sum_flat(data, coeffs, flat_dest_index_array, flat_csr_mx_data, mx_nnz_indptr):
        """
        Computes the summation of several CSR-format sparse matrices.

        :meth:`csr_sum_flat_indices` precomputes the necessary indices for
        summing directly into the data-array of a destination CSR sparse matrix.
        If `data` is the data-array of matrix `D` (for "destination"), then this
        method performs:

        `D += sum_i( coeff[i] * csr_mxs[i] )`

        Note that `D` is not returned; the sum is done internally into D's
        data-array.

        Parameters
        ----------
        data : numpy.ndarray
            The data-array of the destination CSR-matrix.

        coeffs : ndarray
            The weight coefficients which multiply each summed matrix.

        flat_dest_index_array : ndarray
            The index array generated by :func:`csr_sum_flat_indices`.

        flat_csr_mx_data : ndarray
            The data array generated by :func:`csr_sum_flat_indices`.

        mx_nnz_indptr : ndarray
            The number-of-nonzero-elements pointer array generated by
            :func:`csr_sum_flat_indices`.
        """
        coeffs_complex = _np.ascontiguousarray(coeffs, dtype=complex)
        return _fastcalc.fast_csr_sum_flat(data, coeffs_complex, flat_dest_index_array, flat_csr_mx_data, mx_nnz_indptr)


def expm_multiply_prep(a, tol=EXPM_DEFAULT_TOL):
    """
    Computes "prepared" meta-info about matrix `a`, to be used in `expm_multiply_fast`.

    This includes a shifted version of `a`.

    Parameters
    ----------
    a : numpy.ndarray
        the matrix that will be later exponentiated.

    tol : float, optional
        Tolerance used to within matrix exponentiation routines.

    Returns
    -------
    tuple
        A tuple of values to pass to `expm_multiply_fast`.
    """
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('expected a to be like a square matrix')
    assert(_sps.isspmatrix_csr(a))  # assuming this allows faster computations

    n = a.shape[0]
    n0 = 1  # always act exp(a) on *single* vectors
    mu = _spsl._expm_multiply._trace(a) / float(n)

    #ident = _spsl._expm_multiply._ident_like(a) #general case

    if _fastcalc is None:
        ident = _sps.identity(a.shape[0], dtype=a.dtype, format='csr')  # CSR specific
        a = a - mu * ident  # SLOW!
    else:
        indptr = _np.empty(n + 1, _np.int64)
        indices = _np.empty(a.data.shape[0] + n, _np.int64)  # pessimistic (assume no diags exist)
        data = _np.empty(a.data.shape[0] + n, a.dtype)  # pessimistic (assume no diags exist)
        nxt = _fastcalc.csr_subtract_identity(a.data,
                                              _np.ascontiguousarray(a.indptr, _np.int64),
                                              _np.ascontiguousarray(a.indices, _np.int64),
                                              data, indptr, indices, -mu, n)
        a = _sps.csr_matrix((data[0:nxt], indices[0:nxt], indptr), shape=(n, n))
    #DB: CHECK: assert(_spsl.norm(A1 - A2) < 1e-6); a = A1

    #exact_1_norm specific for CSR
    A_1_norm = max(_np.sum(_np.abs(a.data[_np.where(a.indices == iCol)])) for iCol in range(n))
    #A_1_norm = _spsl._expm_multiply._exact_1_norm(a) # general case

    t = 1.0  # always
    if t * A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = _spsl._expm_multiply.LazyOperatorNormInfo(t * a, A_1_norm=t * A_1_norm, ell=ell)
        m_star, s = _spsl._expm_multiply._fragment_3_1(norm_info, n0, tol, ell=ell)

    eta = _np.exp(t * mu / float(s))
    assert(_sps.isspmatrix_csr(a))
    return a, mu, m_star, s, eta


if _fastcalc is None:
    def expm_multiply_fast(prep_a, v, tol=EXPM_DEFAULT_TOL):
        """
        Multiplies `v` by an exponentiated matrix.

        Parameters
        ----------
        prep_a : tuple
            A tuple of values from :func:`expm_multiply_prep` that
            defines the matrix to be exponentiated and holds other pre-computed
            quantities.

        v : numpy.ndarray
            Vector to multiply (take dot product with).

        tol : float, optional
            Tolerance used to within matrix exponentiation routines.

        Returns
        -------
        numpy.ndarray
        """
        A, mu, m_star, s, eta = prep_a
        return _custom_expm_multiply_simple_core(
            A, v, mu, m_star, s, tol, eta)  # t == 1.0 always, `balance` not implemented so removed

else:
    def expm_multiply_fast(prep_a, v, tol=EXPM_DEFAULT_TOL):
        """
        Multiplies `v` by an exponentiated matrix.

        Parameters
        ----------
        prep_a : tuple
            A tuple of values from :func:`expm_multiply_prep` that
            defines the matrix to be exponentiated and holds other pre-computed
            quantities.

        v : numpy.ndarray
            Vector to multiply (take dot product with).

        tol : float, optional
            Tolerance used to within matrix exponentiation routines.

        Returns
        -------
        numpy.ndarray
        """
        #Note: copy v for now since it's modified by simple_core fn
        A, mu, m_star, s, eta = prep_a

        indices = _np.array(A.indices, dtype=_np.int64)  # convert to 64-bit ints if needed
        indptr = _np.array(A.indptr, dtype=_np.int64)
        return _fastcalc.custom_expm_multiply_simple_core(A.data, indptr, indices,
                                                          v.copy(), mu, m_star, s, tol, eta)


def _custom_expm_multiply_simple_core(a, b, mu, m_star, s, tol, eta):  # t == 1.0 replaced below
    """
    a helper function.  Note that this (python) version works when a is a LinearOperator
    as well as a SciPy CSR sparse matrix.
    """
    #if balance:
    #    raise NotImplementedError
    F = b
    for i in range(s):
        #if m_star > 0: #added
        #    c1 = _np.linalg.norm(b, _np.inf) #_exact_inf_norm(b)
        for j in range(m_star):
            coeff = 1.0 / float(s * (j + 1))  # t == 1.0
            b = coeff * a.dot(b)
            F = F + b
        #    if j % 3 == 0: #every == 3 #TODO: work on this
        #        c2 = _np.linalg.norm(b, _np.inf) #_exact_inf_norm(b)
        #        if c1 + c2 <= tol * _np.linalg.norm(F, _np.inf): #_exact_inf_norm(F)
        #            break
        #        c1 = c2
        F = eta * F
        b = F
    return F


def expop_multiply_prep(op, a_1_norm=None, tol=EXPM_DEFAULT_TOL):
    """
    Returns "prepared" meta-info about operation op, which is assumed to be traceless (so no shift is needed).

    Used as input for use with `_custom_expm_multiply_simple_core` or fast C-reps.

    Parameters
    ----------
    op : scipy.sparse.linalg.LinearOperator
        The operator to exponentiate.

    a_1_norm : float, optional
        The 1-norm (if computed separately) of `op`.

    tol : float, optional
        Tolerance used to within matrix exponentiation routines.

    Returns
    -------
    tuple
        A tuple of values to pass to `expm_multiply_fast`.
    """
    assert(isinstance(op, _spsl.LinearOperator))
    if len(op.shape) != 2 or op.shape[0] != op.shape[1]:
        raise ValueError('expected op to have equal input and output dimensions')

    # n = op.shape[0]
    n0 = 1  # always act exp(op) on *single* vectors
    mu = 0  # _spsl._expm_multiply._trace(A) / float(n)
    #ASSUME op is *traceless*

    #FUTURE: get exact_1_norm specific for our ops - now just use approximate
    if a_1_norm is None:
        a_1_norm = _spsl.onenormest(op)

    #t = 1.0 # always, so t*<X> => just <X> below
    if a_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = _spsl._expm_multiply.LazyOperatorNormInfo(op, A_1_norm=a_1_norm, ell=ell)
        m_star, s = _spsl._expm_multiply._fragment_3_1(norm_info, n0, tol, ell=ell)

    eta = 1.0  # _np.exp(t*mu / float(s)) # b/c mu always == 0 (traceless assumption)
    return mu, m_star, s, eta


def sparse_equal(a, b, atol=1e-8):
    """
    Checks whether two Scipy sparse matrices are (almost) equal.

    Parameters
    ----------
    a : scipy.sparse matrix
        First matrix.

    b : scipy.sparse matrix
        Second matrix.

    atol : float, optional
        The tolerance to use, passed to `numpy.allclose`, when comparing
        the elements of `a` and `b`.

    Returns
    -------
    bool
    """
    if _np.array_equal(a.shape, b.shape) == 0:
        return False

    r1, c1 = a.nonzero()
    r2, c2 = b.nonzero()

    lidx1 = _np.ravel_multi_index((r1, c1), a.shape)
    lidx2 = _np.ravel_multi_index((r2, c2), b.shape)
    sidx1 = lidx1.argsort()
    sidx2 = lidx2.argsort()

    index_match = _np.array_equal(lidx1[sidx1], lidx2[sidx2])
    if index_match == 0:
        return False
    else:
        v1 = a.data
        v2 = b.data
        V1 = v1[sidx1]
        V2 = v2[sidx2]
    return _np.allclose(V1, V2, atol=atol)


def sparse_onenorm(a):
    """
    Computes the 1-norm of the scipy sparse matrix `a`.

    Parameters
    ----------
    a : scipy sparse matrix
        The matrix or vector to take the norm of.

    Returns
    -------
    float
    """
    return max(abs(a).sum(axis=0).flat)
    # also == return _spsl.norm(a, ord=1) (comparable speed)


def ndarray_base(a, verbosity=0):
    """
    Get the base memory object for numpy array `a`.

    This is found by following `.base` until it comes up None.

    Parameters
    ----------
    a : numpy.ndarray
        Array to get base of.

    verbosity : int, optional
        Print additional debugging information if this is > 0.

    Returns
    -------
    numpy.ndarray
    """
    if verbosity: print("ndarray_base debug:")
    while a.base is not None:
        if verbosity: print(" -> base = ", id(a.base))
        a = a.base
    if verbosity: print(" ==> ", id(a))
    return a


def to_unitary(scaled_unitary):
    """
    Compute the scaling factor required to turn a scalar multiple of a unitary matrix to a unitary matrix.

    Parameters
    ----------
    scaled_unitary : ndarray
        A scaled unitary matrix

    Returns
    -------
    scale : float
    
    unitary : ndarray
        Such that `scale * unitary == scaled_unitary`.
    """
    scaled_identity = gram_matrix(scaled_unitary, adjoint=True)
    scale = _np.sqrt(scaled_identity[0, 0])
    assert(_np.allclose(scaled_identity / (scale**2), _np.identity(scaled_identity.shape[0], 'd'))), \
        "Given `scaled_unitary` does not appear to be a scaled unitary matrix!"
    return scale, (scaled_unitary / scale)


def sorted_eig(mx):
    """
    Similar to `numpy.eig`, but returns sorted output.

    In particular, the eigenvalues and vectors sorted by eigenvalue,
    where sorting is done according to (real_part, imaginary_part) tuple.

    Parameters
    ----------
    mx : numpy.ndarray
        Matrix to act on.

    Returns
    -------
    eigenvalues : numpy.ndarray
    
    eigenvectors : numpy.ndarray
    """
    ev, U = _np.linalg.eig(mx)
    sorted_evals = sorted(list(enumerate(ev)), key=lambda x: (x[1].real, x[1].imag))
    sorted_ev = ev.copy()
    sorted_U = U.copy()
    for idest, (isrc, _) in enumerate(sorted_evals):
        sorted_ev[idest] = ev[isrc]
        sorted_U[:, idest] = U[:, isrc]
    return sorted_ev, sorted_U


def compute_kite(eigenvalues):
    """
    Computes the "kite" corresponding to a list of eigenvalues.

    The kite is defined as a list of integers, each indicating that
    there is a degnenerate block of that many eigenvalues within
    `eigenvalues`.  Thus the sum of the list values equals `len(eigenvalues)`.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        A *sorted* array of eigenvalues.

    Returns
    -------
    list
        A list giving the multiplicity structure of `evals`.
    """
    kite = []
    blk = 0; last_ev = eigenvalues[0]
    for ev in eigenvalues:
        if _np.isclose(ev, last_ev):
            blk += 1
        else:
            kite.append(blk)
            blk = 1; last_ev = ev
    kite.append(blk)
    return kite


def find_zero_communtant_connection(u, u_inv, u0, u0_inv, kite):
    """
    Find a matrix `R` such that u_inv R u0 is diagonal AND log(R) has no projection onto the commutant of G0.

    More specifically, find a matrix `R` such that u_inv R u0 is diagonal
    (so G = R G0 Rinv if G and G0 share the same eigenvalues and have eigenvectors u
    and u0 respectively) AND log(R) has no (zero) projection onto the commutant of
    G0 = u0 diag(evals) u0_inv.

    Parameters
    ----------
    u : numpy.ndarray
        Usually the eigenvector matrix of a gate (G).

    u_inv : numpy.ndarray
        Inverse of `u`.

    u0 : numpy.ndarray
        Usually the eigenvector matrix of the corresponding target gate (G0).

    u0_inv : numpy.ndarray
        Inverse of `u0`.

    kite : list
        The kite structure of `u0`.

    Returns
    -------
    numpy.ndarray
    """

    #0.  Let R be a matrix that maps G0 -> Gp, where Gp has evecs of G and evals of G0.
    #1.  Does R vanish on the commutant of G0?  If so, we’re done.
    #2.  Define x = PROJ_COMMUTANT[ log(R) ], and X = exp(-x).
    #3.  Redefine R = X.R.
    #4.  GOTO 1.

    # G0 = u0 * diag * u0_inv, G = u * diag * u_inv
    D = project_onto_kite(_np.dot(u_inv, u0), kite)
    R = _np.dot(u, _np.dot(D, u0_inv))  # Include D so R is as close to identity as possible
    assert(_np.linalg.norm(R.imag) < 1e-8)

    def project_onto_commutant(x):
        a = _np.dot(u0_inv, _np.dot(x, u0))
        a = project_onto_kite(a, kite)
        return _np.dot(u0, _np.dot(a, u0_inv))

    iter = 0; lastR = R
    while iter < 100:
        #Starting condition = u_inv * R * u0 is diagonal, so
        # G' = R G0 Rinv where G' has the same spectrum as G0 but different eigenvecs (u vs u0)
        assert(_np.linalg.norm(R.imag) < 1e-8)
        test = _np.dot(u_inv, _np.dot(R, u0))
        assert(_np.linalg.norm(project_onto_antikite(test, kite)) < 1e-8)

        r = real_matrix_log(R)
        assert(_np.linalg.norm(r.imag) < 1e-8), "log of real matrix should be real!"
        r_on_comm = project_onto_commutant(r)
        assert(_np.linalg.norm(r_on_comm.imag) < 1e-8), "projection to commutant should not make complex!"

        oncomm_norm = _np.linalg.norm(r_on_comm)
        #print("Iter %d: onkite-norm = %g  lastdiff = %g" % (iter, oncomm_norm, _np.linalg.norm(R-lastR)))
        # if r has desired form or we didn't really update R
        if oncomm_norm < 1e-12 or (iter > 0 and _np.linalg.norm(R - lastR) < 1e-8):
            break  # STOP - converged!

        X = _spl.expm(-r_on_comm)
        assert(_np.linalg.norm(X.imag) < 1e-8)

        lastR = R
        R = _np.dot(R, X)
        iter += 1

    assert(_np.linalg.norm(R.imag) < 1e-8), "R should always be real!"
    return R.real


def project_onto_kite(mx, kite):
    """
    Project `mx` onto `kite`, so `mx` is zero everywhere except on the kite.

    Parameters
    ----------
    mx : numpy.ndarray
       Matrix to project.

    kite : list
        A kite structure.

    Returns
    -------
    numpy.ndarray
    """
    #Kite is a list of block sizes, such that sum(kite) == dimension of `mx`
    mx = mx.copy()
    dim = mx.shape[0]
    assert(dim == mx.shape[1]), "`mx` must be square!"
    k0 = 0
    for k in kite:
        mx[k0:k0 + k, k0 + k:] = 0
        mx[k0 + k:, k0:k0 + k] = 0
        k0 += k
    assert(k0 == dim), "Invalid kite %d-dimensional matrix: %s" % (dim, str(kite))
    return mx


def project_onto_antikite(mx, kite):
    """
    Project `mx` onto the complement of `kite`, so `mx` is zero everywhere *on* the kite.

    Parameters
    ----------
    mx : numpy.ndarray
       Matrix to project.

    kite : list
        A kite structure.

    Returns
    -------
    numpy.ndarray
    """
    #Kite is a list of block sizes, such that sum(kite) == dimension of `mx`
    mx = mx.copy()
    dim = mx.shape[0]
    assert(dim == mx.shape[1]), "`mx` must be square!"
    k0 = 0
    for k in kite:
        mx[k0:k0 + k, k0:k0 + k] = 0
        k0 += k
    assert(k0 == dim), "Invalid kite %d-dimensional matrix: %s" % (dim, str(kite))
    return mx


def intersection_space(space1, space2, tol=1e-7, use_nice_nullspace=False):
    """
    TODO: docstring
    """
    VW = _np.concatenate((space1, -space2), axis=1)
    nullsp = nice_nullspace(VW, tol) if use_nice_nullspace else nullspace(VW, tol)
    #nullsp = _spl.null_space(VW, rcond=1e-3)  # alternative
    return _np.dot(space1, nullsp[0:space1.shape[1], :])


def union_space(space1, space2, tol=1e-7):
    """
    TODO: docstring
    """
    VW = _np.concatenate((space1, space2), axis=1)
    indep_cols = independent_columns(VW, None, tol)
    return VW[:, indep_cols]


def jamiolkowski_angle(hamiltonian_mx):
    """
    TODO: docstring
    """
    Hmx = hamiltonian_mx
    d = Hmx.shape[0]
    I = _np.identity(d)
    errmap = _np.kron(I, _spl.expm(1j * Hmx))
    psi = _np.zeros(d**2)  # will be a maximally entangled state
    for i in range(d):
        x = _np.zeros(d); x[i] = 1.0
        xx = _np.kron(x, x)
        psi += xx / _np.sqrt(d)
    assert(_np.isclose(_np.dot(psi, psi), 1.0))
    cos_theta = abs(_np.dot(psi.conj(), _np.dot(errmap, psi)))
    return _np.real_if_close(_np.arccos(cos_theta))
    #cos_squared_theta = entanglement_infidelity(expm(1j * Hmx), identity)
    #return _np.arccos(_np.sqrt(cos_squared_theta))


def zvals_to_dense(self, zvals, superket=True):
    """
    Construct the dense operator or superoperator representation of a computational basis state.

    Parameters
    ----------
    zvals : list or numpy.ndarray
        The z-values, each 0 or 1, defining the computational basis state.

    superket : bool, optional
        If `True`, the super-ket representation of the state is returned.  If `False`,
        then the complex ket representation is returned.

    Returns
    -------
    numpy.ndarray
    """
    if superket:
        factor_dim = 4
        v0 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, 1), 'd')  # '0' qubit state as Pauli dmvec
        v1 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, -1), 'd')  # '1' qubit state as Pauli dmvec
    else:
        factor_dim = 2
        v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
        v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
    v = (v0, v1)

    if _fastcalc is None:  # do it the slow way using numpy
        return _functools.reduce(_np.kron, [v[i] for i in zvals])
    else:
        fast_kron_array = _np.ascontiguousarray(
            _np.empty((len(self._zvals), factor_dim), v0.dtype))
        fast_kron_factordims = _np.ascontiguousarray(_np.array([factor_dim] * len(self._zvals), _np.int64))
        for i, zi in enumerate(self._zvals):
            fast_kron_array[i, :] = v[zi]
        ret = _np.ascontiguousarray(_np.empty(factor_dim**len(self._zvals), v0.dtype))
        if superket:
            _fastcalc.fast_kron(ret, fast_kron_array, fast_kron_factordims)
        else:
            _fastcalc.fast_kron_complex(ret, fast_kron_array, fast_kron_factordims)
        return ret


def int64_parity(x):
    """
    Compute the partity of x.

    Recursively divide a (64-bit) integer (x) into two equal
    halves and take their XOR until only 1 bit is left.

    Parameters
    ----------
    x : int64

    Returns
    -------
    int64
    """
    x = (x & 0x00000000FFFFFFFF) ^ (x >> 32)
    x = (x & 0x000000000000FFFF) ^ (x >> 16)
    x = (x & 0x00000000000000FF) ^ (x >> 8)
    x = (x & 0x000000000000000F) ^ (x >> 4)
    x = (x & 0x0000000000000003) ^ (x >> 2)
    x = (x & 0x0000000000000001) ^ (x >> 1)
    return x & 1  # return the last bit (0 or 1)


def zvals_int64_to_dense(zvals_int, nqubits, outvec=None, trust_outvec_sparsity=False, abs_elval=None):
    """
    Fills a dense array with the super-ket representation of a computational basis state.

    Parameters
    ----------
    zvals_int : int64
        The array of (up to 64) z-values, encoded as the 0s and 1s in the binary representation
        of this integer.

    nqubits : int
        The number of z-values (up to 64)

    outvec : numpy.ndarray, optional
        The output array, which must be a 1D array of length 4**nqubits or `None`, in
        which case a new array is allocated.

    trust_outvec_sparsity : bool, optional
        When `True`, it is assumed that the provided `outvec` starts as all zeros
        and so only non-zero elements of outvec need to be set.

    abs_elval : float
        the value `1 / (sqrt(2)**nqubits)`, which can be passed here so that
        it doesn't need to be recomputed on every call to this function.  If
        `None`, then we just compute the value.

    Returns
    -------
    numpy.ndarray
    """

    if outvec is None:
        outvec = _np.zeros(4**nqubits, 'd')
    if abs_elval is None:
        abs_elval = 1 / (_np.sqrt(2)**nqubits)

    # when trust_outvec_sparsity is True, assume we only need to fill in the
    # non-zero elements of outvec (i.e. that outvec is already zero wherever
    # this vector is zero).
    if not trust_outvec_sparsity:
        outvec[:] = 0  # reset everything to zero

    N = nqubits

    # there are nQubits factors
    # each factor (4-element, 1Q dmvec) has 2 zero elements and 2 nonzero ones
    # loop is over all non-zero elements of the final outvec by looping over
    #  all the sets of *entirely* nonzero elements from the factors.

    # Let the two possible nonzero elements of the k-th factor be represented
    # by the k-th bit of `finds` below, which ranges from 0 to 2^nFactors-1
    for finds in range(2**N):

        #Create the final index (within outvec) corresponding to finds
        # assume, like tensorprod, that factor ordering == kron ordering
        # so outvec = kron( factor[0], factor[1], ... factor[N-1] ).
        # Let factorDim[k] == 4**(N-1-k) be the stride associated with the k-th index
        # Whenever finds[bit k] == 0 => finalIndx += 0*factorDim[k]
        #          finds[bit k] == 1 => finalIndx += 3*factorDim[k] (3 b/c factor's 2nd nonzero el is at index 3)
        finalIndx = sum([3 * (4**(N - 1 - k)) for k in range(N) if bool(finds & (1 << k))])

        #Determine the sign of this element (the element is either +/- (1/sqrt(2))^N )
        # A minus sign is picked up whenever finds[bit k] == 1 (which means we're looking
        # at the index=3 element of the factor vec) AND zvals_int[bit k] == 1
        # (which means it's a [1 0 0 -1] state rather than a [1 0 0 1] state).
        # Since we only care whether the number of minus signs is even or odd, we can
        # BITWISE-AND finds with zvals_int (giving an integer whose binary-expansion's
        # number of 1's == the number of minus signs) and compute the parity of this.
        minus_sign = int64_parity(finds & zvals_int)

        outvec[finalIndx] = -abs_elval if minus_sign else abs_elval

    return outvec


def sign_fix_qr(q, r, tol=1e-6):
    """
    Change the signs of the columns of Q and rows of R to follow a convention.

    Flips the signs of Q-columns and R-rows from a QR decomposition so that the
    largest absolute element in each Q-column is positive.  This is an arbitrary
    but consisten convention that resolves sign-ambiguity in the output of a QR
    decomposition.

    Parameters
    ----------
    q, r : numpy.ndarray
        Input Q and R matrices from QR decomposition.

    tol : float, optional
        Tolerance for computing the maximum element, i.e., anything
        within `tol` of the true max is counted as a maximal element,
        the *first* of which is set positive by the convention.

    Returns
    -------
    qq, rr : numpy.ndarray
        Updated versions of `q` and `r`.
    """
    qq = q.copy()
    rr = r.copy()
    for i in range(q.shape[1]):
        max_abs = max(_np.abs(q[:, i]))
        k = _np.argmax(_np.abs(q[:, i]) > (max_abs - tol))
        if q[k, i] < 0.0:
            qq[:, i] = -q[:, i]
            rr[i, :] = -r[i, :]
    return qq, rr


# a TypeVar to allow methods to return "self"
_T = TypeVar("_T", bound="OperatorLike")


@runtime_checkable
class OperatorLike(Protocol[_T]): # type: ignore

    @property
    def T(self) -> _T:
        ...

    def __matmul__(self, other: Any) -> Any:
        ...

    def __rmatmul__(self, other: Any) -> Any:
        ...

    def conj(self) -> _T:
        ...


class IdentityOperator:
    """
    A representation of the identity operator on any and all vector spaces.
    """

    __array_priority__ = 101
    # ^ The __array_priority__ static class variable determines how infix
    #   operators (most notably, @) are dispatched.
    #
    #   If Python initially dispatches the ndarray implementation of an infix
    #   operator, then the ndarray implementation will first check if the other
    #   operand has __array_priority__ greater than zero. If it does, then a
    #   function call like ndarray.__matmul__(array, other) will return the
    #   result of other.__rmatmul__(array).
    #   
    #   For our purposes, it should suffice to set __array_priority__ to 1.
    #   We set it to 101 in case someone does something weird and passes in 
    #   a numpy matrix (which behaves like an ndarray in many respects, and
    #   has __array_priority__ of 10). 
    #

    def __matmul__(self, other: Any) -> Any:
        return other
    
    def __rmatmul__(self, other: Any) -> Any:
        return other
    
    @property
    def T(self):
        return self
    
    def conj(self):
        return self


def to_operatorlike(obj):
    if obj is None:
        return IdentityOperator()
    elif isinstance(obj, OperatorLike):
        return obj
    else:
        raise ValueError()

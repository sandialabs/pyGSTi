"""
Private numerical helpers for sampling positive-semidefinite matrices with sparse support.

The support is imposed on matrix entries. Cholesky fill and component restrictions can require
masking followed by conservative PSD repair; the sampling procedure is not a parameterization
or completion solver for the whole constrained PSD cone.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

from typing import Literal

import numpy as _np

from . import sparsechol as _sparsechol

__all__ = []


def _subpattern(pattern, structure):
    """Normalize an off-diagonal component mask to a dense Boolean array.

    Nonzero entries define undirected edges, and diagonal entries are ignored. The mask must
    have the structure's shape and allow no edges outside its requested pattern. Real and
    imaginary components may use different subsets of those edges.
    """
    if _np.ndim(pattern) != 2:
        raise ValueError("A component pattern must be a square matrix.")
    sub = _sparsechol._adjacency(pattern)
    if sub.shape != structure.pattern.shape:
        raise ValueError("Component patterns must have the same shape as the overall pattern.")
    sub = sub.toarray()
    if _np.any(sub & ~structure.pattern.toarray()):
        raise ValueError("Component patterns must be contained in the overall pattern.")
    return sub


def _psd_shrink_factor(B: _np.ndarray, O: _np.ndarray) -> float:
    """
    Choose a scale in [0, 1] for which the Hermitian matrix ``B + t*O`` remains PSD.

    The sampler uses B for the diagonal and any fixed entries, and O for free off-diagonal
    entries. Scaling O preserves fixed entries and zeros. This helper implements that repair
    policy; it requires a PSD anchor B and raises ValueError if B is numerically indefinite.

    For positive-definite B, write ``M = B^(-1/2) O B^(-1/2)``. The feasible upper bound is one
    when ``lambda_min(M) >= -1``, and ``-1/lambda_min(M)`` otherwise. A relative margin keeps
    the latter result inside the feasible interval. For numerically singular B, return zero
    conservatively, even when positive steps might be feasible. This is not a completion test.
    """
    evals, evecs = _np.linalg.eigh(B)
    tol = 1e-12 * max(1.0, _np.max(_np.abs(evals)))
    if evals[0] < -tol:
        raise ValueError("The fixed part of the matrix is not positive semidefinite "
                         "(minimum eigenvalue %g)." % evals[0])
    if evals[0] <= tol:
        return 0.0
    B_inv_sqrt = (evecs / _np.sqrt(evals)) @ evecs.conj().T
    lam_min = _np.linalg.eigvalsh(B_inv_sqrt @ O @ B_inv_sqrt)[0]
    return 1.0 if lam_min >= -1.0 else (1.0 - 1e-9) / -lam_min


def _sample_psd(
        pattern,
        rng        : _np.random.Generator,
        offdiag    : Literal['complex', 'real', 'imag'] = 'complex',
        scale      : float = 1.0,
        delta      : float = 1.0,
        *, real_pattern=None, imag_pattern=None
    ) -> _np.ndarray:
    """
    Draw a Hermitian PSD matrix with prescribed off-diagonal component support.

    Draw a lower-triangular factor on the filled Cholesky pattern, form its Gram matrix, and
    normalize the diagonal by a congruence. Mask disallowed components and, when needed, shrink
    the remaining off-diagonal entries toward the diagonal matrix to restore positivity.

    A graph is chordal when it has an elimination ordering without Cholesky fill. Such an
    ordering repeatedly removes a vertex whose remaining neighbors form a clique. With that
    ordering, the factor draw needs no support repair unless additional real or imaginary
    component restrictions are imposed. Other patterns can introduce fill that must be masked.

    Parameters
    ----------
    pattern : scipy.sparse matrix, numpy.ndarray, or CholeskyStructure
        A square off-diagonal support pattern, or cached symbolic analysis of that pattern.
        Nonzero entries define undirected edges; diagonal values are ignored.
    rng : numpy.random.Generator
        Source of randomness, advanced in place.
    offdiag : {'complex', 'real', 'imag'}, optional
        Allowed off-diagonal components. The real mode uses a real factor; the other modes
        use a complex factor. Imaginary-only off-diagonals require masking and repair.
    scale : float, optional
        Scale of the factor entries. The expected matrix diagonal is ``scale**2``.
    delta : float, optional
        Positive degree parameter. Real squared factor diagonals are chi-squared with
        ``delta + column_counts`` degrees of freedom; complex squared diagonals have twice
        that many degrees of freedom and are divided by two. Factor off-diagonals are
        independent standard real or complex normal draws before scaling.
    real_pattern, imag_pattern : scipy.sparse matrix or numpy.ndarray, optional
        Further restrictions on the corresponding components, with the same shape as
        ``pattern`` and support contained in it. Defaults to the full requested pattern.
        These restrictions intersect the components selected by ``offdiag``.

    Returns
    -------
    numpy.ndarray
        Dense Hermitian PSD matrix in the original index order. The draw and any repair
        define its distribution; this routine does not solve a PSD completion problem.
    """
    st = pattern if isinstance(pattern, _sparsechol.CholeskyStructure) else _sparsechol.CholeskyStructure(pattern)
    allowed = st.pattern.toarray()
    real_allowed = allowed if real_pattern is None else _subpattern(real_pattern, st)
    imag_allowed = allowed if imag_pattern is None else _subpattern(imag_pattern, st)
    n, cplx, m = st.n, offdiag != 'real', len(st.rows)

    # Bartlett draw in the permuted order: first all diagonal entries, then the off-diagonal ones.
    if cplx:
        diag = _np.sqrt(rng.chisquare(2 * (delta + st.column_counts)) / 2)
        off = (rng.normal(size=m) + 1j * rng.normal(size=m)) / _np.sqrt(2)
    else:
        diag = _np.sqrt(rng.chisquare(delta + st.column_counts))
        off = rng.normal(size=m)
    L = _np.zeros((n, n), dtype=complex if cplx else float)
    L[_np.arange(n), _np.arange(n)] = diag
    L[st.rows, st.cols] = off
    L *= scale
    K = (L @ L.conj().T)[_np.ix_(st.inv, st.inv)]

    # E[K_ii] = scale**2 (delta + deg_i), with deg_i the degree in the filled (chordal) graph.
    d = 1 / _np.sqrt(delta + st.filled_degrees)
    K = d[:, None] * K * d[None, :]

    # Impose the support and the requested off-diagonal parts, then restore positivity if needed.
    # (The diagonal of a complex L L^dag can carry rounding-level imaginary parts; drop them.)
    D = _np.diag(_np.real(_np.diag(K)))
    offd = K - _np.diag(_np.diag(K))
    O = _np.where(real_allowed & (offdiag != 'imag'), offd.real, 0)
    if cplx:
        O = O + 1j * _np.where(imag_allowed, offd.imag, 0)
    if _np.any(O != offd):  # entries outside the filled pattern are exact zeros
        return D + _psd_shrink_factor(D, O) * O
    return D + O


def _impose_diagonal(K: _np.ndarray, indices, values) -> _np.ndarray:
    """
    Rescale K by a diagonal congruence so that K[i, i] = v for each i, v in zip(indices, values)
    (exactly: the diagonal is then overwritten, a rounding-level change).

    The congruence preserves positive semidefiniteness and the sparsity pattern. Rows whose
    diagonal is zero cannot be rescaled and raise ValueError unless their target is zero.
    """
    d = _np.ones(K.shape[0])
    for i, v in zip(indices, values):
        current = _np.real(K[i, i])
        if current <= 0:
            if v != 0:
                raise ValueError("Cannot rescale a zero diagonal entry to %g." % v)
            d[i] = 0.0
        else:
            d[i] = _np.sqrt(v / current)
    K = d[:, None] * K * d[None, :]
    K[list(indices), list(indices)] = values  # exactly, rather than up to rounding
    return K


def _impose_offdiagonals(K: _np.ndarray, real: dict | None = None, imag: dict | None = None) -> _np.ndarray:
    """
    Set Re K[i, j] = v for each (i, j): v in `real` and Im K[i, j] = v for each (i, j): v in `imag`
    (and K[j, i] to the conjugate), keeping K positive semidefinite by shrinking only the free
    off-diagonal parts. A part of an entry that is not fixed is free.

    Raises ValueError if the diagonal together with the fixed entries is not positive
    semidefinite. This conservative policy requires a PSD zero-filled anchor; rejection does
    not establish that the fixed entries admit no PSD completion. None is searched for.
    """
    n = K.shape[0]
    offd = K - _np.diag(_np.diag(K))
    re, im = _np.real(offd).copy(), _np.imag(offd).copy()
    F = _np.zeros((n, n), dtype=complex)
    for part, fixed, free, sign in ((1, real or {}, re, 1), (1j, imag or {}, im, -1)):
        for (i, j), v in fixed.items():
            if i == j:
                raise ValueError("Use _impose_diagonal for diagonal entries.")
            F[i, j] += part * v
            F[j, i] += sign * part * v  # the conjugate
            free[i, j] = free[j, i] = 0
    D = _np.diag(_np.real(_np.diag(K)))
    B = D + F
    O_free = re + 1j * im
    try:
        t = _psd_shrink_factor(B, O_free)
    except ValueError as e:
        raise ValueError("The diagonal and fixed off-diagonal entries do not form a positive-semidefinite anchor: "
                         "%s A PSD completion using free entries may still exist, but none was searched for." % e) from None
    return B + t * O_free

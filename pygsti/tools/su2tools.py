"""
Representation-theoretic tools for the special unitary group SU(2), generalized to
arbitrary spin j.

This module provides a single `SpinJ` class plus a set of module-level,
representation-independent SU(2) group utilities (Haar sampling of Euler angles,
Euler-angle <-> 2x2-unitary conversions, composition/inversion of Euler-angle
sequences, and the irrep character functions), for arbitrary spin j.

Conventions (see the paper "Randomized Benchmarking with Synthetic Quantum Circuits"
for the equations cited below):

- `d = 2j + 1` is the dimension of the spin-j representation; irreps of the "square"
  representation (the one relevant to superoperators, i.e. j (x) j) are labeled by
  integers `k = 0, ..., 2j` (always integer, even when j is half-integer), with irrep
  k having dimension `2k + 1`.
- `SpinJ.synthetic_spam_matrix` (M) and `SpinJ.decay_recoupling_matrix` (F) are the
  SPAM-synthesis and decay/rate recoupling matrices from the paper.
- `SpinJ.clebsch_gordan_cob` (C) and `SpinJ.superop_stdmx_cob` are change-of-basis
  matrices that block-diagonalize the "square" (superoperator) representation of
  SU(2) into its irreducible pieces.

Euler angles throughout use the ZXZ convention: a triple `(alpha, beta, gamma)`
represents `exp(i*alpha*Jz) @ exp(i*beta*Jx) @ exp(i*gamma*Jz)` in whatever
representation is relevant (the fundamental spin-1/2 representation for the
representation-independent group utilities; the instance's own representation for
`SpinJ` methods) -- i.e. `gamma` is applied first (rightmost factor) and `alpha`
last (leftmost factor).
"""
# ***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import functools as _functools
import math as _math
from fractions import Fraction as _Fraction
from typing import List, Optional, Tuple, Union

import numpy as _np
import scipy.linalg as _spl
from scipy.special import eval_legendre as _eval_legendre

from pygsti.baseobjs.label import Label as _Label
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.tools.matrixtools import eigendecomposition as _eigendecomposition
from pygsti.tools.optools import unitary_to_std_process_mx as _unitary_to_std_process_mx
from pygsti.tools.wignersymbols import clebsch_gordan as _clebsch_gordan, wigner_6j as _wigner_6j

__all__ = [
    'batch_normal_expm_1jscales',
    'distance_mod_phase',
    'random_euler_angles',
    'angles_from_2x2_unitaries',
    'angles_from_2x2_unitary',
    'axis_rotation_angle_from_2x2_unitaries',
    'axis_rotation_angle_from_euler_angles',
    'composition_asmatrix',
    'composition_inverse',
    'charactercores_from_euler_angles',
    'SpinJ',
    'GATE_NAME'
]


# ***************************************************************************************************
# Internal helpers
# ***************************************************************************************************

def _validate_spin(j):
    """
    Validate that `j` is a non-negative integer or half-integer.

    Returns
    -------
    j_float : float
        `j`, as a float (exact for any valid input, since halving an integer is
        exact in binary floating point).

    two_j : int
        `round(2 * j)`.
    """
    if isinstance(j, bool):
        raise TypeError("j must be an int, float, or Fraction, not bool")
    if isinstance(j, _Fraction):
        two_j_frac = 2 * j
        if two_j_frac.denominator != 1:
            raise ValueError(f"j = {j} is not an integer or half-integer (2*j must be an integer)")
        two_j = two_j_frac.numerator
    elif isinstance(j, (int, float)):
        if isinstance(j, float) and not _math.isfinite(j):
            raise ValueError(f"j must be finite (got {j!r})")
        two_j_val = 2 * j
        two_j = round(two_j_val)
        if two_j_val != two_j:
            # Exact comparison, not a tolerance-based one: 2*j is exactly representable
            # in binary floating point for every valid half-integer j (halving is exact),
            # so there is no legitimate rounding error to tolerate here. Matches the
            # exactness convention used by pygsti.tools.wignersymbols._as_half_integer.
            raise ValueError(f"j = {j!r} is not an integer or half-integer (2*j must be an integer)")
    else:
        raise TypeError(f"j must be an int, float, or Fraction, not {type(j).__name__}")
    if two_j < 0:
        raise ValueError(f"j must be non-negative (got {j})")
    j_float = float(two_j) / 2.0
    return j_float, two_j


def _spin_generators(j, dim, spins):
    """
    Construct the (2j+1)-dimensional angular-momentum generators Jx, Jy, Jz for spin j,
    in the |j, m> basis with m descending (matches `spins`).
    """
    Jz = _np.diag(spins)
    if dim == 1:
        return _np.zeros((1, 1)), _np.zeros((1, 1), dtype=complex), Jz
    off = _np.empty(dim - 1)
    for i in range(dim - 1):
        m = spins[i]
        off[i] = 0.5 * _math.sqrt(j * (j + 1) - m * (m - 1))
        # ^ Standard ladder-operator matrix element: <m-1|J-|m> = sqrt(j(j+1) - m(m-1)).
        #   Jx = (J+ + J-)/2 and Jy = (J+ - J-)/(2i) then have these (real) off-diagonals.
    Jx = _np.diag(off, 1) + _np.diag(off, -1)
    Jy = _np.diag(-1j * off, 1) + _np.diag(1j * off, -1)
    return Jx, Jy, Jz


def _check_su2_generators(Jx, Jy, Jz, tol=1e-10):
    """
    Assert that Jx, Jy, Jz satisfy su(2) commutation relations up to an overall scale
    (the scale is ambiguous given only the convention that Jz is diagonal with the
    "spins" on its diagonal).
    """
    def bracket(a, b):
        return a @ b - b @ a

    def is_zero(arg):
        return _np.all(_np.abs(arg) <= tol)

    abxy = _np.abs(bracket(Jx, Jy))
    aJz = _np.abs(Jz)
    mask = aJz > 0
    if _np.any(mask):
        ratios = abxy[mask] / aJz[mask]
        scale = ratios[0]
        assert _np.allclose(ratios, scale), 'Inconsistent implied scale in su(2) commutation relations.'
    else:
        scale = 1.0

    diff = bracket(Jx, Jy) - 1j * scale * Jz
    assert is_zero(diff), f'[Jx, Jy] != i*scale*Jz up to tolerance {tol}'
    diff = bracket(Jx, Jz) + 1j * scale * Jy
    assert is_zero(diff), f'[Jx, Jz] != -i*scale*Jy up to tolerance {tol}'
    diff = bracket(Jy, Jz) - 1j * scale * Jx
    assert is_zero(diff), f'[Jy, Jz] != i*scale*Jx up to tolerance {tol}'


def _build_clebsch_gordan_cob(j, dim, spins):
    """
    C[(J,M),(m1,m2)] = <j m1; j m2 | J M>, rows ordered with J ascending 0..2j and M
    descending J..-J within each J; columns (m1, m2) both descending j..-j, m1-major.
    """
    two_j = dim - 1
    rows = [(J, M) for J in range(two_j + 1) for M in range(J, -J - 1, -1)]
    C = _np.zeros((dim * dim, dim * dim))
    for r, (J, M) in enumerate(rows):
        for a, m1 in enumerate(spins):
            for b, m2 in enumerate(spins):
                if m1 + m2 == M:
                    C[r, a * dim + b] = _clebsch_gordan(j, m1, j, m2, J, M)
    return C


def _build_synthetic_spam_matrix(j, dim, spins):
    """M[k, idx(ell)] = sqrt((2k+1)/(2j+1)) * <j ell; k 0 | j ell>."""
    M = _np.zeros((dim, dim))
    for k in range(dim):
        for i, ell in enumerate(spins):
            M[k, i] = _math.sqrt((2 * k + 1) / (2 * j + 1)) * _clebsch_gordan(j, ell, k, 0, j, ell)
    return M


def _build_decay_recoupling_matrix(j, dim):
    """F[k,k'] = (2j+1) * (-1)^(2j+k+k') * {k j j; k' j j}_6j."""
    two_j = dim - 1
    F = _np.zeros((dim, dim))
    for k in range(dim):
        for kp in range(k, dim):
            val = _wigner_6j(k, j, j, kp, j, j) * ((-1) ** (two_j + k + kp))
            F[k, kp] = F[kp, k] = (2 * j + 1) * val
    return F


def _irrep_projectors(block_sizes, cob):
    """
    Return the list of orthogonal projectors (in the *original* -- i.e. pre-change-
    of-basis -- coordinates) onto each isotypic component, given a unitary `cob` for
    which `cob @ V @ cob.conj().T` is block diagonal (with the stated block sizes, in
    order) for every V in the representation being block-diagonalized.

    Since `cob` acts on (vectorized) operators as `new = cob @ old`, the rows of
    `cob` -- not its columns -- are the block's basis vectors expressed in the
    original coordinates; the projector for a block is therefore built from
    `cob[start:stop, :].conj().T`, whose columns span that block's preimage. A
    projector built from `cob`'s columns instead has the right rank/trace but does
    not commute with the representation matrices, so it is not a valid isotypic
    projector.
    """
    start = 0
    projectors = []
    for bk_sz in block_sizes:
        stop = start + bk_sz
        subspace = cob[start:stop, :].conj().T
        P = subspace @ subspace.conj().T
        projectors.append(P)
        start = stop
    return projectors


# ***************************************************************************************************
# Module-level, representation-independent SU(2) group utilities
# ***************************************************************************************************

def batch_normal_expm_1jscales(
        eigvecs: _np.ndarray, eigvals: _np.ndarray, scales: Union[float, _np.ndarray]) -> _np.ndarray:
    """
    Efficiently compute a batch of matrix exponentials of a scaled normal operator.

    Given the eigendecomposition `A = eigvecs @ diag(eigvals) @ eigvecs.conj().T` of
    a normal operator `A`, return an array equivalent to
    `[eigvecs @ diag(1j*s*eigvals) @ eigvecs.conj().T for s in scales]`, but computed
    without repeated matrix-matrix products.

    Parameters
    ----------
    eigvecs : numpy array
        A square, unitary matrix of order n (the eigenvectors of the normal operator,
        as columns).

    eigvals : numpy array
        A length-n vector of eigenvalues corresponding to the columns of `eigvecs`.

    scales : float or array-like
        The scale(s) `s` at which to evaluate `expm(1j * s * A)`.

    Returns
    -------
    numpy array
        Shape `(len(scales), n, n)`.
    """
    assert eigvals.ndim == 1
    n = eigvals.size
    assert eigvecs.shape == (n, n)
    scales = _np.atleast_1d(scales)
    eigvecs_inv = eigvecs.T.conj()
    batch_out = _np.array([
        (eigvecs * _np.exp(1j * s * eigvals)[_np.newaxis, :]) @ eigvecs_inv for s in scales
    ])
    return batch_out


def distance_mod_phase(U1: _np.ndarray, U2: _np.ndarray) -> float:
    """
    Return `min{ ||U1 - z*U2|| : |z| = 1 }`, the Frobenius-norm distance between
    `U1` and `U2` up to an overall (physically irrelevant) global phase on `U2`.
    """
    scale = _np.vdot(U2, U1)
    scale /= abs(scale)
    delta = U1 - scale * U2
    distance = _spl.norm(delta)
    return distance


def random_euler_angles(
        size: Union[int, Tuple[int, ...]] = 1,
        rng: Optional[_np.random.Generator] = None) -> Tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
    Sample ZXZ Euler angles for Haar-random elements of SU(2).

    Parameters
    ----------
    size : int or tuple of ints, optional
        The number/shape of samples to draw.

    rng : numpy.random.Generator, optional
        If given, angles are drawn from `rng` instead of the global numpy random
        state.

    Returns
    -------
    alpha, beta, gamma : numpy arrays
        Arrays of shape `size`.  Note the asymmetry between `alpha` (range
        `[0, 2*pi)`) and `gamma` (range `[0, 4*pi)`) -- this is intentional, and
        required for Haar-uniformity given how `beta` is sampled.
    """
    uniform = _np.random.uniform if rng is None else rng.uniform
    alpha = uniform(low=0, high=2 * _np.pi, size=size)
    beta = _np.arccos(uniform(low=-1, high=1, size=size))
    gamma = uniform(low=0, high=4 * _np.pi, size=size)
    return alpha, beta, gamma


def angles_from_2x2_unitaries(R: _np.ndarray) -> Tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
    Recover ZXZ Euler angles from a batch of 2x2 (special) unitary matrices.

    Parameters
    ----------
    R : numpy array
        Shape `(2, 2)` or `(N, 2, 2)`.

    Returns
    -------
    alpha, beta, gamma : numpy arrays
        Shape `(N,)` (or `(1,)` if `R` was 2-dimensional).
    """
    if R.ndim == 2:
        R = R[_np.newaxis, :, :]

    beta = 2 * _np.arccos(_np.real(_np.sqrt(R[:, 0, 0] * R[:, 1, 1])))
    alpha = _np.zeros_like(beta)
    gamma = _np.zeros_like(beta)

    den = _np.sin(beta / 2) * _np.cos(beta / 2)
    s = den != 0
    alpha[s] = _np.angle(-1.j * R[s, 0, 0] * R[s, 0, 1] / den[s])
    alpha[alpha < 0] += 2 * _np.pi

    gamma[s] = _np.angle(-1.j * R[s, 0, 0] * R[s, 1, 0] / den[s])
    gamma[gamma < 0] += 2 * _np.pi

    s1 = R[:, 0, 0] != 0
    if _np.any(s1):
        s2 = _np.zeros_like(s1)
        s2[s1] = _np.isclose(_np.exp(1.j * (alpha[s1] + gamma[s1]) / 2) * _np.cos(beta[s1] / 2) / R[s1, 0, 0], -1)
        gamma[s2] += 2 * _np.pi

    return alpha, beta, gamma


def angles_from_2x2_unitary(R: _np.ndarray) -> Tuple[float, float, float]:
    """Single-matrix convenience wrapper around `angles_from_2x2_unitaries`."""
    a, b, g = angles_from_2x2_unitaries(R)
    angles = a.item(), b.item(), g.item()
    return angles


def axis_rotation_angle_from_2x2_unitaries(U: _np.ndarray) -> _np.ndarray:
    """
    Return the axis-rotation angle(s) theta in `[0, pi]` implied by a batch of 2x2
    (special) unitary matrices (i.e. `U = exp(-i*theta/2 * n.J)` for some axis `n`).
    """
    if U.ndim == 2:
        U = U[_np.newaxis, :, :]
    eigs = _np.linalg.eigvals(U)  # eigs = exp(+-i theta/2)
    theta = _np.real(2 * _np.log(eigs[:, 0]) / 1j)
    tr = _np.trace(U, axis1=1, axis2=2)
    check = tr - 2 * _np.cos(theta / 2)
    assert _np.all(abs(check) < 1e-10)
    theta[theta < 0] += 2 * _np.pi
    theta[theta > _np.pi] = 2 * _np.pi - theta[theta > _np.pi]
    return theta


def axis_rotation_angle_from_euler_angles(abg: _np.ndarray) -> _np.ndarray:
    """
    Return the axis-rotation angle(s) theta in `[0, pi]` implied by ZXZ Euler
    angles `abg = (alpha, beta, gamma)` (each a scalar, or `abg` shape `(3, N)`).
    """
    theta_by_2 = _np.atleast_1d(_np.arccos(_np.cos(abg[1] / 2) * _np.cos((abg[0] + abg[2]) / 2)))
    theta = 2 * theta_by_2
    theta[theta > _np.pi] = 2 * _np.pi - theta[theta > _np.pi]
    return theta


def composition_asmatrix(angles: _np.ndarray) -> _np.ndarray:
    """
    Return the 2x2 unitary representing the composition of a sequence of ZXZ
    Euler-angle gates.

    Parameters
    ----------
    angles : numpy array
        Shape `(N, 3)`.  The circuit defined by `angles[0]`, then `angles[1]`, ...,
        then `angles[N-1]` is represented by the unitary
        `R[N-1] @ R[N-2] @ ... @ R[0]`.

    Returns
    -------
    numpy array
        Shape `(2, 2)`.
    """
    assert angles.shape[1] == 3
    alphas, betas, gammas = angles.T
    assert alphas.ndim == betas.ndim == gammas.ndim == 1
    R_composed = _np.eye(2)
    if alphas.size > 0:
        Rs = _fundamental_spinj().unitaries_from_angles(alphas, betas, gammas)
        assert Rs.shape == (alphas.size, 2, 2)
        for R in Rs:
            R_composed = R @ R_composed
    return R_composed


def composition_inverse(alphas: _np.ndarray, betas: _np.ndarray, gammas: _np.ndarray) -> Tuple[float, float, float]:
    """
    Return the ZXZ Euler angles of the inverse of the composed gate sequence
    `(alphas, betas, gammas)` (see `composition_asmatrix`).
    """
    assert alphas.ndim == betas.ndim == gammas.ndim == 1
    R_composed = composition_asmatrix(_np.column_stack([alphas, betas, gammas]))
    invR = R_composed.T.conj()
    inverse_angles = angles_from_2x2_unitary(invR)
    return inverse_angles


def charactercores_from_euler_angles(irrep_labels: _np.ndarray, angles: _np.ndarray) -> _np.ndarray:
    """
    Evaluate the Legendre-polynomial "character cores" `P_k(cos(beta))` used by the rank-1
    synthetic-SPAM RB variant, for ZXZ Euler angles `angles = (alpha, beta, gamma)`.

    angles : numpy array
        Shape `(N, 3)` for a batch of N

    Returns
    -------
    numpy array
        Shape `(N, len(irrep_labels))`
    """
    angles = _np.asarray(angles)
    assert angles.ndim == 2
    assert angles.shape[1] == 3 
    beta = angles[:, 1]
    irrep_labels = _np.asarray(irrep_labels)
    out = _eval_legendre(irrep_labels[_np.newaxis, :], _np.cos(beta[:, _np.newaxis]))
    return out


# ***************************************************************************************************
# Circuit objects
# ***************************************************************************************************

GATE_NAME = 'Gu'
"""The name of the single arg-carrying gate label used by SU(2) RB circuits."""


def circuit_from_euler_angles(angles: _np.ndarray, qudit_label: str = 'Q0') -> _Circuit:
    """
    Build a `Circuit` of `Gu` gate layers from a sequence of ZXZ Euler angles.

    Parameters
    ----------
    angles : numpy array
        Shape `(m+1, 3)`. Row `i` gives the ZXZ Euler angles `(alpha, beta, gamma)`
        of the `i`-th layer's `Gu` gate.

    qudit_label : str, optional
        The line label of the single qudit this circuit acts on.

    Returns
    -------
    Circuit
    """
    angles = _np.asarray(angles, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(f"`angles` must have shape (m+1, 3); got {angles.shape}")
    layers = [_Label(GATE_NAME, qudit_label, args=tuple(float(x) for x in row)) for row in angles]
    circuit = _Circuit(layers, line_labels=(qudit_label,), editable=True)
    return circuit


def euler_angles_from_circuit(circuit: _Circuit) -> _np.ndarray:
    """
    Recover the `(m+1, 3)` array of ZXZ Euler angles from a `Circuit`'s `Gu` layers.

    Any non-`Gu` layers (e.g. explicit `rho{ell}` prep or `Mdefault` POVM layers) are
    ignored. This makes it the inverse of `circuit_from_euler_angles`.

    Parameters
    ----------
    circuit : Circuit
        A circuit from `circuit_from_euler_angles` (or one with the same `Gu`-layer convention).

    Returns
    -------
    numpy array
        Shape `(m+1, 3)`, or `(0, 3)` if `circuit` has no `Gu` layers.
    """
    rows = []
    for layer in circuit:
        if layer.name == GATE_NAME:
            args = layer.args
            if len(args) != 3:
                raise ValueError(f"Gate layer {layer} does not carry exactly 3 args")
            rows.append([float(a) for a in args])
    angles = _np.array(rows, dtype=float).reshape(-1, 3)
    return angles



# ***************************************************************************************************
# SpinJ
# ***************************************************************************************************

class SpinJ:
    """
    Representation-theoretic tools for the spin-j irreducible representation of
    SU(2), for arbitrary (non-negative integer or half-integer) j.

    Parameters
    ----------
    j : int, float, or Fraction
        The spin. `2*j` must be an integer.

    Attributes
    ----------
    j : float
        The spin.

    dim : int
        The dimension `2*j + 1` of the representation.

    spins : numpy array
        The magnetic quantum numbers `j, j-1, ..., -j`, in that (descending) order --
        this fixes the basis ordering used throughout (row/column index `i`
        corresponds to `spins[i]`).

    Jx, Jy, Jz : numpy array
        The angular-momentum generators in the spin-j representation, in the
        `spins`-ordered basis (`Jz` is diagonal with `spins` on the diagonal).

    eigJx, VJx, eigJy, VJy : numpy array
        Eigendecompositions of `Jx` and `Jy` (`Jx = VJx @ diag(eigJx) @ VJx.conj().T`,
        and likewise for `Jy`), computed via `matrixtools.eigendecomposition` with
        `assume_normal=True`.

    irrep_labels : numpy array
        The integers `k = 0, ..., 2j` labeling the irreps appearing in `j (x) j`.

    irrep_block_sizes : numpy array
        The dimensions `2k + 1` of those irreps, in the same order as `irrep_labels`.
    """

    def __init__(self, j: Union[int, float, _Fraction]) -> None:
        j_float, two_j = _validate_spin(j)
        dim = two_j + 1
        spins = _np.array([j_float - i for i in range(dim)])

        Jx, Jy, Jz = _spin_generators(j_float, dim, spins)
        _check_su2_generators(Jx, Jy, Jz)

        VJx, eigJx, _ = _eigendecomposition(Jx, assume_normal=True)
        VJy, eigJy, _ = _eigendecomposition(Jy, assume_normal=True)

        self.j = j_float
        self.dim = dim
        self.spins = spins
        self.Jx, self.Jy, self.Jz = Jx, Jy, Jz
        self.eigJx, self.VJx = eigJx, VJx
        self.eigJy, self.VJy = eigJy, VJy
        self.irrep_labels = _np.arange(dim)
        self.irrep_block_sizes = 2 * self.irrep_labels + 1

    # -----------------------------------------------------------------
    # Cached, more expensive derived quantities.
    # -----------------------------------------------------------------

    @_functools.cached_property
    def clebsch_gordan_cob(self) -> _np.ndarray:
        """
        The change-of-basis matrix C, with `C[(J,M),(m1,m2)] = <j m1; j m2 | J M>`.
        Rows are ordered with J ascending `0..2j` and M descending `J..-J` within
        each J; columns `(m1, m2)` are both descending `j..-j`, m1-major.  See the
        paper's Mathematica-code comment for the historical spin-7/2 version of this
        convention.
        """
        cob = _build_clebsch_gordan_cob(self.j, self.dim, self.spins)
        return cob

    @_functools.cached_property
    def superop_stdmx_cob(self) -> _np.ndarray:
        """
        The full block-diagonalizer of the standard-basis superoperator
        representation: `C @ kron(expm(i*pi*Jy), I)`.  For a unitary U in this
        representation, `superop_stdmx_cob @ kron(conj(U), U) @ superop_stdmx_cob.conj().T`
        is block diagonal with blocks of size `irrep_block_sizes` (equivalently,
        using `pygsti.tools.unitary_to_std_process_mx(U) = kron(U, conj(U))`, since
        the two tensor-factor orderings give the same block structure here).
        """
        expm_i_pi_Jy = self.expm_iJy(_np.pi)[0]
        cob = self.clebsch_gordan_cob @ _np.kron(expm_i_pi_Jy, _np.eye(self.dim))
        return cob

    @_functools.cached_property
    def irrep_stdmx_projectors(self) -> List[_np.ndarray]:
        """
        The list of orthogonal projectors (one per irrep, ordered as in
        `irrep_labels`) onto each isotypic component of the standard-basis
        superoperator representation, expressed in the original (pre-change-of-
        basis) coordinates.
        """
        projectors = _irrep_projectors(self.irrep_block_sizes, self.superop_stdmx_cob)
        return projectors

    @_functools.cached_property
    def synthetic_spam_matrix(self) -> _np.ndarray:
        """
        The SPAM-synthesis matrix M (paper eq. "Mmaintext"):
        `M[k, idx(ell)] = sqrt((2k+1)/(2j+1)) * <j ell; k 0 | j ell>`, with rows
        `k = 0..2j` and columns `ell` ranging over `spins` (i.e. `j` down to `-j`).
        M is orthogonal (`M @ M.T = I`) for every j.
        """
        M = _build_synthetic_spam_matrix(self.j, self.dim, self.spins)
        return M

    @_functools.cached_property
    def decay_recoupling_matrix(self) -> _np.ndarray:
        """
        The decay/rate recoupling matrix F (paper eq. "recoupling", normalized-rates
        convention): `F[k,k'] = (2j+1) * (-1)^(2j+k+k') * {k j j; k' j j}_6j`. F is
        symmetric with an all-ones row/column 0. Per-irrep decay rates `p` are
        recovered from per-irrep decay parameters `f` via `p = solve(F, f)`.
        """
        F = _build_decay_recoupling_matrix(self.j, self.dim)
        return F

    # -----------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------

    def expm_iJx(self, thetas: Union[float, _np.ndarray]) -> _np.ndarray:
        """Batch-evaluate `expm(1j * theta * Jx)` for each `theta` in `thetas`."""
        thetas = _np.atleast_1d(thetas)
        result = batch_normal_expm_1jscales(self.VJx, self.eigJx, thetas)
        return result

    def expm_iJy(self, thetas: Union[float, _np.ndarray]) -> _np.ndarray:
        """Batch-evaluate `expm(1j * theta * Jy)` for each `theta` in `thetas`."""
        thetas = _np.atleast_1d(thetas)
        result = batch_normal_expm_1jscales(self.VJy, self.eigJy, thetas)
        return result

    def unitaries_from_angles(
            self, alpha: Union[float, _np.ndarray], beta: Union[float, _np.ndarray],
            gamma: Union[float, _np.ndarray]) -> _np.ndarray:
        """
        Construct spin-j-representation unitaries `exp(i*alpha*Jz) @ exp(i*beta*Jx)
        @ exp(i*gamma*Jz)` from batches of ZXZ Euler angles.

        Parameters
        ----------
        alpha, beta, gamma : float or numpy array
            Equal-length (or all-scalar) Euler angles.

        Returns
        -------
        numpy array
            Shape `(N, dim, dim)` (or `(1, dim, dim)` for scalar input).
        """
        array_on_input = isinstance(alpha, _np.ndarray)
        alpha = _np.atleast_1d(alpha)
        beta = _np.atleast_1d(beta)
        gamma = _np.atleast_1d(gamma)
        if not array_on_input:
            assert alpha.size == beta.size == gamma.size == 1

        dJz = _np.diag(self.Jz)
        right = (_np.exp(1j * alpha[:, _np.newaxis] * dJz[_np.newaxis, :]))[:, :, _np.newaxis]
        center = self.expm_iJx(beta)
        left = (_np.exp(1j * gamma[:, _np.newaxis] * dJz[_np.newaxis, :]))[:, _np.newaxis, :]
        unitaries = left * center * right
        return unitaries

    def stdmx_twirl(self, A: _np.ndarray) -> _np.ndarray:
        """
        Twirl the (Hermitian, standard-basis) superoperator `A` over SU(2), i.e.
        project it onto the space of operators that are scalar on each isotypic
        component.
        """
        projectors = self.irrep_stdmx_projectors
        coeffs = _np.array([_np.vdot(A, P) for P in projectors])
        coeffs = coeffs / self.irrep_block_sizes
        tA = sum(coeffs[i] * P for i, P in enumerate(projectors))
        return tA

    def all_characters_from_unitary(self, U: _np.ndarray) -> _np.ndarray:
        """
        Return the irrep characters `chi_k` (k = 0..2j, i.e. one per entry of
        `irrep_labels`) implied by the spin-j-representation unitary `U`, computed
        by block-diagonalizing `U`'s standard-basis superoperator and summing the
        diagonal of each block.
        """
        A = _unitary_to_std_process_mx(U)
        diag = _np.diag(self.superop_stdmx_cob @ A @ self.superop_stdmx_cob.conj().T).real
        out = []
        idx = 0
        for b_sz in self.irrep_block_sizes:
            out.append(_np.sum(diag[idx:idx + b_sz]))
            idx += b_sz
        characters = _np.array(out).reshape((1, -1))
        return characters


@_functools.lru_cache(maxsize=1)
def _fundamental_spinj():
    """The spin-1/2 (fundamental) `SpinJ` instance used by the group-level Euler-angle
    composition utilities (`composition_asmatrix`, `composition_inverse`)."""
    fundamental = SpinJ(_Fraction(1, 2))
    return fundamental

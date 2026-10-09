r"""
Post-measurement unitaries for instrument members.

An instrument member `I_k` with effect `E_k = I_k^dagger(I)` factors as

    I_k = Ad_U o RootConj(E_k),   Ad_U(rho) = U rho U^dagger,   RootConj(E)(rho) = E^{1/2} rho E^{1/2},

for some unitary `U` exactly when `I_k` has a single Kraus operator (Choi rank one).  Lindblad
instrument charts freeze `U` as the static base of the member's post-measurement gate, so
:meth:`Instrument.from_effects` expects the caller to supply it.  When only the members are known,
:func:`infer_unitary_factors` chooses one `U` per member.

The choice keeps `E_k` and takes `U` as the unitary polar factor of `A E_k^{1/2}`, where `A` is the
Kraus direction of the leading Choi eigenvector.  This is exact for a one-Kraus member, and for
any member it is within a factor of three, in Frobenius norm, of the best map of the form
`Ad_U o RootConj(E_k)`.  Where the member does not determine `U` -- inputs outside the support of
`A E_k^{1/2}`, a repeated leading Choi eigenvalue, the phase of `A` -- the choice is biased
toward a reference unitary, the identity unless the caller supplies another.  Directions the
reference leaves tied are completed so that `U` swaps them with the supported directions, which
gives the bit flip `X` for the second member of a measure-and-reset instrument.  A zero member
gets the reference itself.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from __future__ import annotations

import collections as _collections
import dataclasses as _dataclasses

import numpy as _np

from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs import BasisLike as _BasisLike
from pygsti.tools import basistools as _bt
from pygsti.tools.jamiolkowski import jamiolkowski_iso as _jamiolkowski_iso

#: Leading Choi eigenvalues within this relative distance of the largest are treated as tied.
TIE_RTOL = 1e-9

#: Singular values of `A E^{1/2}` below this fraction of the largest are treated as zero.
RANK_RTOL = 1e-9

#: Overlaps with a reference below this are treated as zero, leaving the choice to the next rule.
OVERLAP_TOL = 1e-9


@_dataclasses.dataclass(frozen=True)
class UnitaryFactor:
    """How :func:`infer_unitary_factors` chose one member's post-measurement unitary."""

    residual: float
    """Frobenius norm of `I_k - Ad_U o RootConj(E_k)` as a standard-basis superoperator."""

    relative_residual: float
    """`residual` divided by the Frobenius norm of `I_k` (zero for a zero member)."""

    choi_eigenvalues: _np.ndarray
    """Eigenvalues of the member's unnormalized Choi matrix, in decreasing order."""

    leading_multiplicity: int
    """How many Choi eigenvalues tie with the largest (zero for a zero member)."""

    completion_dim: int
    """Input dimensions on which the member does not determine `U`, so the reference did."""

    unresolved_dim: int
    """Completion dimensions the tie-breaking rules leave tied; `U` pairs them arbitrarily there."""


def _kraus_superop_std(K: _np.ndarray) -> _np.ndarray:
    """Standard-basis superoperator of `rho -> K rho K^dagger` (row-major vectorization)."""
    return _np.kron(K, K.conj())


def _closest_completion(V0: _np.ndarray, W0: _np.ndarray,
                        references: list[_np.ndarray]) -> tuple[_np.ndarray, int]:
    """
    The unitary `Z` that maximizes `Re Tr(R^dagger V0 Z W0^dagger)` for the first reference `R`,
    then for each later reference on the directions earlier ones leave tied.  Returns `Z` and
    the number of directions no reference resolves.
    """
    k = V0.shape[1]
    if k == 0:
        return _np.zeros((0, 0), complex), 0
    if not references:
        return _np.eye(k, dtype=complex), k
    P, s, Qh = _np.linalg.svd(V0.conj().T @ references[0] @ W0)
    m = int(_np.count_nonzero(s > OVERLAP_TOL))
    Z = P[:, :m] @ Qh[:m]
    if m == k:
        return Z, 0
    Pn, Qn = P[:, m:], Qh[m:].conj().T
    Zn, unresolved = _closest_completion(V0 @ Pn, W0 @ Qn, references[1:])
    return Z + Pn @ Zn @ Qn.conj().T, unresolved


def _unitary_factor(member: _np.ndarray, basis: _Basis, reference: _np.ndarray | None = None,
                    error_tol: float = 1e-6, trunc_tol: float = 1e-7
                    ) -> tuple[_np.ndarray, _np.ndarray, UnitaryFactor]:
    """
    Choose the post-measurement unitary of one dense member superop (in `basis`).

    Returns the effect superket `E` (in `basis`), the unitary `U`, and a :class:`UnitaryFactor`.
    Raises ValueError if the member is not completely positive within `error_tol`.  A member
    whose Choi eigenvalues are all at most `trunc_tol` is treated as zero.
    """
    member = _np.asarray(member)
    dim = basis.dim
    udim = round(dim ** 0.5)
    if member.shape != (dim, dim):
        raise ValueError(f"Instrument member has shape {member.shape}; expected ({dim}, {dim}).")
    R = _np.eye(udim, dtype=complex) if reference is None else _np.asarray(reference, dtype=complex)
    if R.shape != (udim, udim) or not _np.allclose(R.conj().T @ R, _np.eye(udim), atol=1e-8):
        raise ValueError(f"The reference must be a ({udim}, {udim}) unitary.")

    I_superket = _bt.stdmx_to_vec(_np.eye(udim), basis).ravel().real
    E_superket = (member.T @ I_superket).real
    E_mat = _bt.vec_to_stdmx(E_superket, basis, keep_complex=True)
    E_mat = (E_mat + E_mat.conj().T) / 2
    e_vals, e_vecs = _np.linalg.eigh(E_mat)
    sqrtE = (e_vecs * _np.sqrt(e_vals.clip(0, None))) @ e_vecs.conj().T

    # Unnormalized Choi matrix, with the Kraus convention of optools.minimal_kraus_decomposition:
    # an eigenvector v with eigenvalue lam is the Kraus operator sqrt(lam) * v.reshape(d, d).
    choi = _jamiolkowski_iso(member, basis, 'std', normalized=True) * udim
    choi = (choi + choi.conj().T) / 2
    c_vals, c_vecs = _np.linalg.eigh(choi)
    c_vals, c_vecs = c_vals[::-1], c_vecs[:, ::-1]
    if c_vals[-1] < -error_tol:
        raise ValueError(f"The member is not completely positive: its Choi matrix has eigenvalue "
                         f"{c_vals[-1]:.3e}.")

    if c_vals[0] <= trunc_tol:
        U = R.copy()
        multiplicity, completion_dim, unresolved = 0, udim, 0
    else:
        multiplicity = int(_np.count_nonzero(c_vals >= c_vals[0] * (1 - TIE_RTOL)))
        v = c_vecs[:, 0]
        if multiplicity > 1:
            # Among the tied leading Kraus directions, take the one closest to the reference's.
            target = (R @ sqrtE).reshape(-1)
            Vs = c_vecs[:, :multiplicity]
            w = Vs @ (Vs.conj().T @ target)
            if _np.linalg.norm(w) > OVERLAP_TOL * max(_np.linalg.norm(target), 1.0):
                v = w / _np.linalg.norm(w)
        v = v * _np.exp(-1j * _np.angle(v[_np.argmax(_np.abs(v))]))  # deterministic phase
        H = v.reshape(udim, udim) @ sqrtE

        L, s, Rh = _np.linalg.svd(H)
        r = int(_np.count_nonzero(s > RANK_RTOL * s[0]))
        U_supp = L[:, :r] @ Rh[:r]
        overlap = _np.trace(R.conj().T @ U_supp)
        if abs(overlap) > OVERLAP_TOL:
            U_supp = U_supp * _np.exp(-1j * _np.angle(overlap))
        V0, W0 = L[:, r:], Rh[r:].conj().T
        Z, unresolved = _closest_completion(V0, W0, [R, U_supp.conj().T])
        U = U_supp + V0 @ Z @ W0.conj().T
        completion_dim = udim - r

    member_std = _bt.change_basis(member, basis, 'std')
    residual = float(_np.linalg.norm(member_std - _kraus_superop_std(U @ sqrtE)))
    scale = float(_np.linalg.norm(member_std))
    factor = UnitaryFactor(residual=residual, relative_residual=residual / scale if scale > 0 else 0.0,
                           choi_eigenvalues=c_vals, leading_multiplicity=multiplicity,
                           completion_dim=completion_dim, unresolved_dim=unresolved)
    return E_superket, U, factor


def _as_dense_members(instrument_or_members) -> _collections.OrderedDict:
    if isinstance(instrument_or_members, dict):
        return _collections.OrderedDict((lbl, _np.asarray(m.to_dense('HilbertSchmidt') if hasattr(m, 'to_dense')
                                                          else m)) for lbl, m in instrument_or_members.items())
    if hasattr(instrument_or_members, 'items'):  # Instrument, TPInstrument
        return _collections.OrderedDict((lbl, _np.asarray(m.to_dense('HilbertSchmidt')))
                                        for lbl, m in instrument_or_members.items())
    raise ValueError(f"Cannot interpret {type(instrument_or_members).__name__} as an instrument; pass an "
                     "Instrument, a TPInstrument, or a {label: superop} dict.")


def infer_unitary_factors(instrument_or_members, basis: _BasisLike, *,
                          reference: _np.ndarray | dict[str, _np.ndarray] | None = None,
                          error_tol: float = 1e-6, trunc_tol: float = 1e-7
                          ) -> tuple[dict[str, tuple[_np.ndarray, _np.ndarray]], dict[str, UnitaryFactor]]:
    r"""
    Choose a post-measurement unitary for each member of a completely positive instrument.

    Each member `I_k` keeps its effect `E_k = I_k^dagger(I)` and gets the unitary described in
    this module's docstring, so the returned pairs rebuild an instrument with the same effects,
    hence the same outcome probabilities for every input state::

        members, factors = infer_unitary_factors(instrument, basis)
        seed = Instrument.from_effects(members, basis)

    The rebuilt member is `Ad_U o RootConj(E_k)`.  It equals `I_k` when `I_k` has one Kraus
    operator; otherwise it is an approximation whose error `factors[k].residual` reports.  A
    multi-Kraus member's conditional output states, and so downstream circuit probabilities,
    change.

    Parameters
    ----------
    instrument_or_members : Instrument, TPInstrument, or dict
        The instrument, or a `{label: superop}` dict of dense `d**2 x d**2` members in `basis`.
        Members must be completely positive; project a TP fit's members first, e.g. with
        :func:`~pygsti.modelmembers.instruments.seedrepair.project_instrument_to_cptp`.

    basis : BasisLike
        The basis of the dense members.

    reference : numpy.ndarray or dict, optional
        The unitary toward which ties are broken: one `d x d` unitary for every member, or a
        `{label: unitary}` dict.  Members missing from a dict, and every member by default, use
        the identity.

    error_tol : float, optional
        Negative Choi eigenvalues below `-error_tol` raise ValueError.

    trunc_tol : float, optional
        A member whose Choi eigenvalues are all at most `trunc_tol` is treated as zero and gets
        the reference unitary.

    Returns
    -------
    members : dict
        `{label: (effect superket, unitary)}`, the input of :meth:`Instrument.from_effects`.
    factors : dict
        `{label: UnitaryFactor}`: residuals, Choi spectra, and how much of each unitary the
        tie-breaking rules chose.
    """
    dense = _as_dense_members(instrument_or_members)
    basis = _Basis.cast(basis, next(iter(dense.values())).shape[0])
    members, factors = _collections.OrderedDict(), _collections.OrderedDict()
    for lbl, member in dense.items():
        ref = reference.get(lbl) if isinstance(reference, dict) else reference
        try:
            E, U, factor = _unitary_factor(member, basis, ref, error_tol, trunc_tol)
        except ValueError as e:
            raise ValueError(f"Instrument member {lbl!r}: {e}") from e
        members[lbl], factors[lbl] = (E, U), factor
    return members, factors

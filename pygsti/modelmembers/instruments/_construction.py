#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

r"""
A quantum instrument member `I_k` factors as a measurement effect `E_k`
followed by a post-measurement CPTP gate `G_k`::

    I_k(rho) = G_k( E_k^{1/2} rho E_k^{1/2} ),   E_k = I_k^dagger(I).

This module builds a parameterized :class:`Instrument` from that decomposition.
A single :class:`ComposedPOVM` carries the `n` measurement effects `{E_k}`:
its (shared, CP-constrained) error map keeps each `E_k` positive and makes the
effects sum to the identity, which is exactly the statement that the whole
instrument is trace-preserving.  One parameterized gate `G_k` per outcome then
carries the post-measurement back-action.  Each member is the single
:class:`ComposedOp`::

    ComposedOp([ RootConjOperator(E_k), G_k ]).

This needs only `n` effects and `n` gates.  Complete positivity of a member is
opt-in through `gate_parameterization` (a CP-constrained Lindblad type such as
`'CPTPLND'` / `'H+S'`); trace preservation holds for any TP gate parameterization.

This module deliberately never imports :class:`Instrument` (to avoid an import
cycle): the builders return a `dict` mapping outcome label -> :class:`ComposedOp`,
which the classmethods wrap with `cls(...)`.
"""
from __future__ import annotations

import numpy as _np

from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs import BasisLike as _BasisLike
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _pv
from pygsti.modelmembers.povms.basepovm import _BasePOVM
from pygsti.tools import optools as _ot
from pygsti.tools import basistools as _bt
from pygsti.tools import matrixtools as _mt

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pygsti.modelmembers.instruments.instrument import EffectSpec, GateSpec, InstMemberSpec


def _conjugation_superop(herm_mat: _np.ndarray, basis: _BasisLike) -> _np.ndarray:
    """
    Dense superop (in `basis`) of the conjugation map `sigma -> M sigma M` for a
    Hermitian matrix `M = herm_mat`.  Mirrors the core of
    :func:`~pygsti.tools.optools.rootconj_superop` (which builds `kron(M, M.T)` in
    the standard basis), but takes `M` directly rather than `M = E^{1/2}`.
    """
    mx_std = _np.kron(herm_mat, herm_mat.T)
    return _bt.change_basis(mx_std, 'std', basis, expect_real=True)


def _is_lindblad_type(spec: str) -> bool:
    """Whether `spec` names a Lindblad error-generator parameterization."""
    return isinstance(spec, str) and _ot.is_valid_lindblad_paramtype(spec)


def _validate_gate_parameterization(gate_parameterization: str) -> None:
    """
    Reject post-measurement gate parameterizations that are not trace-preserving.

    A non-TP gate would break the instrument's joint trace preservation, so we
    only allow a Lindblad type (always TP), `'full TP'`, or `'static'`.  The
    unconstrained `'full'` parameterization is rejected explicitly.
    """
    if _is_lindblad_type(gate_parameterization):
        return
    if gate_parameterization in ('full TP', 'static'):
        return
    if gate_parameterization == 'full':
        raise ValueError(
            "gate_parameterization='full' would let the post-measurement gates drift off "
            "the trace-preserving manifold and break the instrument's joint trace "
            "preservation. Use a TP parameterization: a Lindblad type "
            "('CPTPLND', 'GLND', 'H+S', 'H+s', ...), 'full TP', or 'static'.")
    raise ValueError(
        f"Unrecognized gate_parameterization {gate_parameterization!r}. Expected a Lindblad "
        f"type ('CPTPLND', 'GLND', 'H+S', 'H+s', ...), 'full TP', or 'static'.")


def _normalize_effects_and_gates(members: dict[str, InstMemberSpec], basis: _BasisLike,
                                 atol: float = 1e-6) -> tuple[_Basis, dict[str, _np.ndarray], dict[str, _np.ndarray]]:
    """
    Normalize a `{label: (effect, gate)}` (or `{label: effect}`) mapping into
    separate dicts of effect superkets and post-measurement gate superops.

    Validates that each gate is trace-preserving and that the effects satisfy the
    POVM completeness relation `sum_k E_k == I` (i.e. the instrument is TP).
    The effect `E_k` and gate `G_k` are kept *separate* (not pre-composed into
    a single `G_k @ rootconj(E_k)` member superop) so each can be parameterized
    independently downstream.

    Parameters
    ----------
    members : dict
        Maps each outcome label to either an `(effect, gate)` pair or a bare
        `effect` (gate defaults to the identity).  `effect` may be a length-
        `d**2` superket in `basis`, a Hermitian `d x d` matrix, or a
        :class:`POVMEffect`.  `gate` may be a `d**2 x d**2` superop, a
        `d x d` unitary, `None`, or a :class:`LinearOperator`.

    basis : Basis or str
        The basis in which dense arrays are represented.  If a string, the
        dimension is inferred from the first effect.

    atol : float, optional
        Absolute tolerance for the per-gate TP check and the completeness check.

    Returns
    -------
    basis : Basis
    effect_superkets : dict[label, numpy.ndarray]
    gate_superops : dict[label, numpy.ndarray]
    """
    def _effect_of(val: InstMemberSpec) -> EffectSpec:
        return val[0] if isinstance(val, tuple) else val

    def _gate_of(val: InstMemberSpec) -> GateSpec:
        return val[1] if isinstance(val, tuple) and len(val) > 1 else None

    if isinstance(basis, str):
        # Infer the Hilbert-Schmidt dimension from the first effect.
        first = _effect_of(next(iter(members.values())))
        arr = _np.asarray(first.to_dense() if hasattr(first, 'to_dense') else first)
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            dim = arr.shape[0]
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            dim = arr.shape[0] ** 2   # a Hermitian (udim x udim) matrix
        else:
            raise ValueError("Could not infer the dimension from the effects; "
                             "pass `basis` as a Basis object instead of a string.")
        basis = _Basis.cast(basis, dim)
    else:
        basis = _Basis.cast(basis)

    dim = basis.dim
    udim = round(dim ** 0.5)
    I_hilbert = _np.eye(udim)
    I_superket = _bt.stdmx_to_vec(I_hilbert, basis).reshape(-1)  # <<I| extracts the trace

    def as_superket(effect: EffectSpec) -> _np.ndarray:
        arr = _np.asarray(effect.to_dense() if hasattr(effect, 'to_dense') else effect)
        if arr.ndim == 2 and arr.shape == (udim, udim):
            arr = _bt.stdmx_to_vec(arr, basis)   # Hermitian d x d matrix -> superket
        return arr.reshape(-1)

    def as_superop(gate: GateSpec) -> _np.ndarray:
        if gate is None:
            return _np.eye(dim)
        if hasattr(gate, 'to_dense'):
            return _np.asarray(gate.to_dense('HilbertSchmidt'))
        arr = _np.asarray(gate)
        if arr.shape == (udim, udim):
            return _ot.unitary_to_superop(arr, basis)   # unitary -> superop
        if arr.shape == (dim, dim):
            return arr
        raise ValueError(f"Gate has shape {arr.shape}; expected a ({dim}, {dim}) "
                         f"superoperator or a ({udim}, {udim}) unitary.")

    superkets = dict()
    superops = dict()
    effect_sum = _np.zeros((udim, udim), dtype=complex)
    for label, val in members.items():
        E_superket = as_superket(_effect_of(val))
        G_superop  = as_superop(_gate_of(val))
        if not _np.allclose(I_superket @ G_superop, I_superket, atol=atol):
            raise ValueError(f"The post-measurement gate for outcome {label!r} is not TP.")
        # rootconj_superop validates 0 <= E_k <= I and raises/warns otherwise.
        _ot.rootconj_superop(E_superket, basis)
        superkets[label] = E_superket
        superops[label]  = G_superop
        effect_sum += _bt.vec_to_stdmx(E_superket, basis, keep_complex=True)

    if not _np.allclose(effect_sum, I_hilbert, atol=atol):
        raise ValueError("The provided effects do not sum to the identity; an "
                         "instrument's effects must satisfy sum_k E_k == I.")

    return basis, superkets, superops


def _decompose_cptr(superop: _np.ndarray, basis: _BasisLike,
                    error_tol: float = 1e-6, trunc_tol: float = 1e-7) -> tuple[_np.ndarray, _np.ndarray]:
    r"""
    Decompose a dense CPTR member superop `I_k` (in `basis`) into its
    measurement effect and a canonical CPTP post-measurement gate.

    The effect is the Heisenberg-dual applied to the identity,
    `E_k = I_k^dagger(I)` -- computed directly from the superop, with no Kraus
    decomposition.  The gate is `G_k = Q_k + P_k`, where
    `Q_k = I_k . pinv(rootconj(E_k))` is the on-support part and `P_k` (a
    conjugation by the projector onto `ker(E_k)`) completes `G_k` so it is
    trace-preserving.  When `E_k` is full rank `P_k = 0` and `G_k = Q_k`.

    Parameters
    ----------
    superop : numpy.ndarray
        A `d**2 x d**2` CPTR superoperator in `basis`.

    basis : Basis or str
        The basis in which `superop` is represented.

    error_tol, trunc_tol : float, optional
        Tolerances for the complete-positivity check (negative Choi eigenvalues
        below `-error_tol` raise; those in `[-error_tol, trunc_tol)` are
        truncated) and for the `ker(E_k)` cutoff.

    Returns
    -------
    E_superket : numpy.ndarray
        The length-`d**2` effect superket `E_k` in `basis`.
    G_superop : numpy.ndarray
        The `d**2 x d**2` CPTP post-measurement gate superop `G_k` in `basis`.
    """
    basis = _Basis.cast(basis)
    dim = basis.dim
    udim = round(dim ** 0.5)
    I_k = _np.asarray(superop)
    assert I_k.shape == (dim, dim), \
        f"CPTR superop has shape {I_k.shape}; expected ({dim}, {dim})."

    # Validate complete positivity: this raises if I_k has a negative Choi
    # spectrum beyond error_tol (and silently clips tiny negatives).
    _ot.minimal_kraus_decomposition(I_k, basis, error_tol, trunc_tol)

    I_hilbert = _np.eye(udim)
    I_superket = _bt.stdmx_to_vec(I_hilbert, basis).ravel().real

    # 1. Effect E_k = I_k^dagger(I): the Heisenberg-dual superop is I_k.T (real,
    #    Hermitian-preserving map in a Hermitian basis), applied to the trace functional.
    E_superket = I_k.T @ I_superket

    # Eigendecompose the effect once and split support / kernel at a SINGLE threshold
    # (trunc_tol).  Using a consistent threshold for both the on-support inverse (in
    # Q_k) and the kernel completion (P_k) is essential: R_k's eigenvalues are products
    # of E_k^{1/2}'s eigenvalues, so a pseudo-inverse of R_k cannot separate support
    # from kernel cleanly, and a tiny effect eigenvalue could be both inverted by Q_k
    # and completed by P_k -- double-counting it and breaking trace preservation.
    E_mat = _bt.vec_to_stdmx(E_superket, basis, keep_complex=True)
    evecs, evals, _ = _mt.eigendecomposition(E_mat, assume_hermitian=True)
    support = evals > trunc_tol
    kernel = ~support

    # 2. On-support gate Q_k = I_k . (rho -> E^{-1/2} rho E^{-1/2}), the inverse taken
    #    only over supp(E_k).
    inv_sqrt = _np.zeros_like(evals)
    inv_sqrt[support] = evals[support] ** -0.5
    E_inv_sqrt = (evecs * inv_sqrt[_np.newaxis, :]) @ evecs.conj().T
    Q_k = I_k @ _conjugation_superop(E_inv_sqrt, basis)

    # 3-4. Complete to a CPTP gate G_k = Q_k + (rho -> P_k rho P_k), where P_k is the
    #      orthogonal projector onto ker(E_k).  When E_k is full rank P_k = 0.
    if _np.any(kernel):
        P_mat = evecs[:, kernel] @ evecs[:, kernel].conj().T
        G_k = (Q_k + _conjugation_superop(P_mat, basis)).real
    else:
        G_k = Q_k.real

    # Sanity checks: G_k is exactly trace-preserving, and G_k . rootconj(E_k)
    # reproduces I_k up to the discarded (sub-trunc_tol) effect directions.
    R_k = _ot.rootconj_superop(E_superket, basis)          # rho -> E^{1/2} rho E^{1/2}
    assert _np.allclose(I_superket @ G_k, I_superket, atol=1e-7), \
        "Internal error: the recovered post-measurement gate is not trace-preserving."
    recon_atol = 1e-6 if not _np.any(kernel) else max(1e-6, 10.0 * trunc_tol ** 0.5)
    assert _np.allclose(G_k @ R_k, I_k, atol=recon_atol), \
        "Internal error: G_k . rootconj(E_k) failed to reproduce the CPTR superop."

    return E_superket, G_k


def _check_effects_complete(effect_superkets: dict[str, _np.ndarray], basis: _BasisLike,
                            atol: float = 1e-6) -> None:
    """Raise unless the effects sum to the identity (i.e. the instrument is TP)."""
    basis = _Basis.cast(basis)
    udim = round(basis.dim ** 0.5)
    I_hilbert = _np.eye(udim)
    effect_sum = _np.zeros((udim, udim), dtype=complex)
    for E in effect_superkets.values():
        effect_sum += _bt.vec_to_stdmx(E, basis, keep_complex=True)
    if not _np.allclose(effect_sum, I_hilbert, atol=atol):
        msg = "The CPTR member superops do not sum to a TP channel: their " \
            "measurement effects E_k = I_k^dagger(I) must satisfy sum_k E_k == I."
        raise ValueError(msg)


def _parameterize_gate(G_superop: _np.ndarray, basis: _BasisLike,
                       gate_parameterization: str) -> _op.LinearOperator:
    """
    Wrap the dense post-measurement gate superop `G_k` in the requested
    TP parameterization, seeded so that it reproduces `G_k` exactly
    at zero error.

    For Lindblad types we compose a fresh, zero-initialized CP/TP error generator
    onto a static base `G_k`.  Crucially, the error generator is built from the
    *identity* (not from `G_k`), so this works even when `G_k` is singular --
    e.g. the complete-dephasing gate produced by an ideal projective effect.
    """
    basis = _Basis.cast(basis)
    if gate_parameterization == 'static':
        return _op.StaticArbitraryOp(G_superop, basis)
    if gate_parameterization == 'full TP':
        return _op.FullTPOp(G_superop, basis)

    # Lindblad type: ComposedOp([static G_k, fresh zero error map]).
    G_static = _op.StaticArbitraryOp(G_superop, basis)
    udim = round(basis.dim ** 0.5)
    I_static = _op.StaticUnitaryOp(_np.eye(udim, dtype='complex'), basis)
    error_map = _op.convert(I_static, gate_parameterization, basis).factorops[1]
    return _op.ComposedOp([G_static, error_map])


def _parameterized_instrument(basis: _BasisLike,
                              effect_superkets: dict[str, _np.ndarray],
                              gate_superops: dict[str, _np.ndarray],
                              gate_parameterization: str = 'CPTPLND',
                              povm_errormap: _op.LinearOperator | str = 'CPTPLND') -> dict[str, _op.ComposedOp]:
    """
    Assemble the `{label: ComposedOp([RootConjOperator(E_k), G_k])}` member dict.

    A single :class:`ComposedPOVM` (a CP-constrained error map `povm_errormap`
    over the static effects `{E_k}`) supplies the measurement effects; each gate
    `G_k` is parameterized per `gate_parameterization`.  Returns the member
    dict (the classmethod wraps it with `Instrument(...)`).

    Parameters
    ----------
    basis : Basis
        The operator basis.

    effect_superkets : dict[label, numpy.ndarray]
        Effect superkets `E_k` in `basis`.

    gate_superops : dict[label, numpy.ndarray]
        Post-measurement gate superops `G_k` in `basis` (same keys as
        `effect_superkets`).

    gate_parameterization : str, optional
        TP parameterization for the gates `{G_k}`.  A CP-constrained
        Lindblad type (`'CPTPLND'` / `'H+S'`) makes each member CP; a
        non-CP-constrained one (`'GLND'` / `'H+s'` / `'full TP'`) keeps the
        instrument TP but allows non-CP members.  `'static'` freezes the gates.

    povm_errormap : LinearOperator or str
        A CP-by-construction :class:`LinearOperator`, or a string spec for one,
        used as the shared error map of the effects' :class:`ComposedPOVM`.

    Returns
    -------
    dict[label, ComposedOp]
    """
    _validate_gate_parameterization(gate_parameterization)
    basis = _Basis.cast(basis)
    udim = round(basis.dim ** 0.5)
    I_hilbert = _np.eye(udim, dtype='complex')

    # Build the shared, CP-constrained POVM over the static effects {E_k}.
    if not isinstance(povm_errormap, _op.LinearOperator):
        assert isinstance(povm_errormap, str)
        I_static = _op.StaticUnitaryOp(I_hilbert, basis)
        I_param = _op.convert(I_static, povm_errormap, basis)
        povm_errormap = I_param.factorops[1]

    static_effects = {lbl: _pv.StaticPOVMEffect(E_superket)
                      for lbl, E_superket in effect_superkets.items()}
    base_povm = _BasePOVM(static_effects)
    composed_povm = _pv.ComposedPOVM(povm_errormap, base_povm)

    member_ops = dict()
    for lbl, G_superop in gate_superops.items():
        root = _op.RootConjOperator(composed_povm[lbl], basis)
        gate = _parameterize_gate(G_superop, basis, gate_parameterization)
        member_ops[lbl] = _op.ComposedOp([root, gate])

    return member_ops

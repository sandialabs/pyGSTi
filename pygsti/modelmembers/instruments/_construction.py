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

A Lindblad gate parameterization composes an error map onto a frozen static gate,
and the constructors make that static gate a unitary: for all practical purposes the
ideal post-measurement conditional gate is unitary, the ideal instrument is then the
chart's zero-error point, and every noise process lives in the Lindblad parameters.
A noisy or singular frozen channel would instead bound what a fit can reach (see
:mod:`pygsti.modelmembers.instruments.diagnostics`).  :func:`_parameterized_instrument`
still accepts superoperator gates, for anyone who assembles such a chart by hand.

This module deliberately never imports :class:`Instrument` (to avoid an import
cycle): the builders return a `dict` mapping outcome label -> :class:`ComposedOp`,
which the classmethods wrap with `cls(...)`.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as _np

from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs import BasisLike as _BasisLike
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _pv
from pygsti.modelmembers.povms.basepovm import _BasePOVM
from pygsti.modelmembers.instruments.unitaryfactors import _unitary_approximation
from pygsti.tools import optools as _ot
from pygsti.tools import basistools as _bt

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pygsti.modelmembers.instruments.instrument import EffectSpec, GateSpec, MemberSpec


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


def _normalize_effects_and_gates(members: dict[str, MemberSpec], basis: _BasisLike, atol: float = 1e-6
                                 ) -> tuple[_Basis, dict[str, _np.ndarray], dict[str, _np.ndarray]]:
    """
    Normalize a `{label: (effect, gate)}` (or `{label: effect}`) mapping into
    separate dicts of effect superkets and post-measurement unitaries.

    Each gate must be unitary: a unitary matrix, or a superoperator or
    :class:`LinearOperator` that is a unitary channel.

    Validates that the effects satisfy the
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
        `d x d` unitary, `None`, or a :class:`LinearOperator`, and must be unitary.

    basis : Basis or str
        The basis in which dense arrays are represented.  If a string, the
        dimension is inferred from the first effect.

    atol : float, optional
        Absolute tolerance for the per-gate unitarity check and the completeness check.

    Returns
    -------
    basis : Basis
    effect_superkets : dict[label, numpy.ndarray]
    unitaries : dict[label, numpy.ndarray]
        The `d x d` post-measurement unitaries.
    """
    def _effect_of(val: MemberSpec) -> EffectSpec:
        return val[0] if isinstance(val, tuple) else val

    def _gate_of(val: MemberSpec) -> GateSpec:
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

    def as_unitary(label, gate: GateSpec) -> _np.ndarray:
        if gate is None:
            return _np.eye(udim, dtype=complex)
        arr = None if hasattr(gate, 'to_dense') else _np.asarray(gate)
        if arr is not None and arr.shape == (udim, udim):
            if not _np.allclose(arr.conj().T @ arr, _np.eye(udim), atol=atol):
                raise ValueError(_nonunitary_gate_message(label, "it is not a unitary matrix"))
            return arr.astype(complex)
        G_superop = as_superop(gate)
        if not _np.allclose(I_superket @ G_superop, I_superket, atol=atol):
            raise ValueError(f"The post-measurement gate for outcome {label!r} is not TP.")
        try:
            _, U, report = _unitary_approximation(G_superop, basis)
        except ValueError as e:
            raise ValueError(_nonunitary_gate_message(label, str(e))) from e
        if report.residual > atol:
            raise ValueError(_nonunitary_gate_message(
                label, f"its superoperator is {report.residual:.2e} (Frobenius) from the nearest "
                f"unitary channel, and it has {report.kraus_rank} Kraus operators"))
        return U

    superkets = dict()
    gates = dict()
    effect_sum = _np.zeros((udim, udim), dtype=complex)
    for label, val in members.items():
        E_superket = as_superket(_effect_of(val))
        gate = as_unitary(label, _gate_of(val))
        # rootconj_superop validates 0 <= E_k <= I and raises/warns otherwise.
        _ot.rootconj_superop(E_superket, basis)
        superkets[label] = E_superket
        gates[label] = gate
        effect_sum += _bt.vec_to_stdmx(E_superket, basis, keep_complex=True)

    if not _np.allclose(effect_sum, I_hilbert, atol=atol):
        raise ValueError("The provided effects do not sum to the identity; an "
                         "instrument's effects must satisfy sum_k E_k == I.")

    return basis, superkets, gates


_HAND_BUILT_NOTE = ("To freeze a general channel G as the static gate anyway, assemble the members "
                    "yourself, e.g. ComposedOp([RootConjOperator(effect, basis), ComposedOp([StaticArbitraryOp(G), "
                    "error_map])]), and pass them to Instrument(...).")


def _nonunitary_gate_message(label, reason: str) -> str:
    return (f"The post-measurement gate for outcome {label!r} is not unitary: {reason}. An instrument "
            "member's static post-measurement gate is unitary; noise in the post-measurement dynamics "
            "belongs in the gate's parameters. " + _HAND_BUILT_NOTE)


def _unitary_members_from_cptr(op_arrays: dict[str, _np.ndarray], basis: _Basis, error_tol: float = 1e-6,
                               trunc_tol: float = 1e-7) -> tuple[dict[str, _np.ndarray], dict[str, _np.ndarray]]:
    """
    Factor each dense CPTR member exactly as `Ad_U o RootConj(E)` with a unitary `U`, or raise.

    `U` is chosen by :func:`~pygsti.modelmembers.instruments.unitaryfactors.infer_unitary_factors`
    (ties broken toward the identity).  A member that no unitary reproduces within `error_tol`
    (Frobenius norm of the standard-basis superoperator difference) raises ValueError.
    """
    superkets, unitaries = dict(), dict()
    inexact = []
    for lbl, I_k in op_arrays.items():
        try:
            E_k, U_k, report = _unitary_approximation(I_k, basis, None, error_tol, trunc_tol)
        except ValueError as e:
            raise ValueError(f"Instrument member {lbl!r}: {e}") from e
        if report.residual > error_tol:
            inexact.append(f"{lbl!r} (Kraus rank {report.kraus_rank}, residual {report.residual:.2e})")
        superkets[lbl], unitaries[lbl] = E_k, U_k
    if inexact:
        raise ValueError(
            "No unitary static base reproduces instrument member(s) " + ", ".join(inexact) + ": a member "
            "factors as Ad_U o RootConj(E) with U unitary only if it has a single Kraus operator. "
            "To start from the effect-preserving unitary approximation, use "
            "pygsti.modelmembers.instruments.infer_unitary_factors with Instrument.from_effects; the "
            "Lindblad parameters can then fit the remaining noise. " + _HAND_BUILT_NOTE)
    return superkets, unitaries


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


def _parameterize_gate(gate: _np.ndarray, basis: _BasisLike,
                       gate_parameterization: str) -> _op.LinearOperator:
    """
    Wrap the post-measurement gate `G_k` -- a `d x d` unitary or a `d**2 x d**2`
    superop -- in the requested TP parameterization, seeded so that it reproduces
    `G_k` exactly at zero error.

    For Lindblad types we compose a fresh, zero-initialized CP/TP error generator
    onto a static base `G_k` (a :class:`StaticUnitaryOp` for a unitary).  The error
    generator is built from the *identity* (not from `G_k`), so this works even when
    a superop `G_k` is singular.
    """
    basis = _Basis.cast(basis)
    udim = round(basis.dim ** 0.5)
    is_unitary = gate.shape == (udim, udim)
    if gate_parameterization == 'static':
        return _op.StaticUnitaryOp(gate, basis) if is_unitary else _op.StaticArbitraryOp(gate, basis)
    if gate_parameterization == 'full TP':
        return _op.FullTPOp(_ot.unitary_to_superop(gate, basis) if is_unitary else gate, basis)

    # Lindblad type: ComposedOp([static G_k, fresh zero error map]).
    G_static = _op.StaticUnitaryOp(gate, basis) if is_unitary else _op.StaticArbitraryOp(gate, basis)
    I_static = _op.StaticUnitaryOp(_np.eye(udim, dtype='complex'), basis)
    error_map = _op.convert(I_static, gate_parameterization, basis).factorops[1]
    return _op.ComposedOp([G_static, error_map])


def _parameterized_instrument(basis: _BasisLike,
                              effect_superkets: dict[str, _np.ndarray],
                              gates: dict[str, _np.ndarray],
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

    gates : dict[label, numpy.ndarray]
        Post-measurement gates `G_k`, each a `d x d` unitary or a `d**2 x d**2`
        superop in `basis` (same keys as `effect_superkets`).

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
    for lbl, gate in gates.items():
        if _is_lindblad_type(gate_parameterization) and gate.shape == (basis.dim, basis.dim):
            svals = _np.linalg.svd(gate, compute_uv=False)
            if svals[-1] < 1e-7 * svals[0]:
                rank = int(_np.sum(svals > 1e-7 * svals[0]))
                _warnings.warn(
                    f"The post-measurement gate for outcome {lbl!r} is singular "
                    f"(superoperator rank {rank} < {basis.dim}).  A Lindblad gate "
                    "parameterization freezes this gate as a static base and only "
                    "composes an invertible exp(L) onto it, so no optimized member "
                    "can ever exceed rank "
                    f"{rank}; fitting data from a device whose instrument members "
                    "are full rank will fail.  If this rank cap is intended, ignore "
                    "this warning.  Otherwise use a unitary static base, as "
                    "Instrument.from_effects does (even a measure-and-reset "
                    "instrument has one) -- see "
                    "pygsti.modelmembers.instruments.infer_unitary_factors "
                    "and .diagnostics.diagnose_instrument.")
        root = _op.RootConjOperator(composed_povm[lbl], basis)
        member_ops[lbl] = _op.ComposedOp([root, _parameterize_gate(gate, basis, gate_parameterization)])

    return member_ops

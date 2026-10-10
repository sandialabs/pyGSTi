r"""
Repairs ("patchers") for the static base of parameterized instrument charts.

Companion to :mod:`pygsti.modelmembers.instruments.diagnostics`: each function
here is the named repair for one of that module's flags.

============================  =========================================
flag                          repair
============================  =========================================
RANK-CAP                      :func:`full_rank_base_blend` (c ~ 0.9)
SPAN-DEFICIENT / ZERO-EXTENT  :func:`displace_base_effects` (delta ~ 0.1-0.2)
CP-BOUNDARY                   :func:`interior_effect_offset` (eps ~ 0.01)
============================  =========================================

:func:`patch_instrument_seed` runs the diagnostics and applies exactly the
repairs whose checks fail, returning a ready-to-fit parameterized
:class:`Instrument`; :func:`patch_model_instrument_seeds` does the same for
every instrument in a model copy.  :func:`project_instrument_to_cptp` is the
entry point for projecting (possibly non-CP) dense members onto the
CPTP-instrument set, with or without cvxpy.

All repairs operate on the dense ``(effect, gate)`` base *before* the
parameterized chart is built -- mutating a built chart's static operations in
place is unsupported (read-only member dicts, cached representations).

The default repair magnitudes (c = 0.9, eps = 0.01, delta = 0.2) were
calibrated on a single 1-qubit, 196-circuit dataset with 2-outcome
instruments; they are sensible starting points, not universal constants.  In
particular `delta` bounds the certified reach of the repaired effects (the
extent check reports the resulting ceiling), so a `delta` smaller than the
true deviation of your device saturates silently -- prefer too large over too
small, within positivity.
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
import warnings as _warnings

import numpy as _np

from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.modelmembers.instruments.instrument import Instrument as _Instrument
from pygsti.modelmembers.instruments.tpinstrument import TPInstrument as _TPInstrument
from pygsti.modelmembers.instruments._construction import (
    _decompose_cptr, _parameterized_instrument
)
from pygsti.modelmembers.instruments.diagnostics import (
    _ranks_from_base, _span_from_base, _extent_from_base
)
from pygsti.tools import basistools as _bt
from pygsti.tools import optools as _ot
from pygsti.tools.jamiolkowski import jamiolkowski_iso as _jam, \
    jamiolkowski_iso_inv as _jam_inv


def _as_dense_members(instrument_or_members):
    """Normalize to an OrderedDict of dense member superops."""
    if isinstance(instrument_or_members, (_Instrument, _TPInstrument)):
        return _collections.OrderedDict(
            (lbl, _np.asarray(m.to_dense('HilbertSchmidt')))
            for lbl, m in instrument_or_members.items())
    if isinstance(instrument_or_members, dict):
        return _collections.OrderedDict(
            (lbl, _np.asarray(m)) for lbl, m in instrument_or_members.items())
    raise ValueError(f"Cannot interpret {type(instrument_or_members).__name__} as "
                     "an instrument; pass an Instrument, TPInstrument, or "
                     "{label: superop} dict.")


def conj_superop(M, basis):
    """
    Dense superop (in `basis`) of the conjugation map ``rho -> M rho M^dagger``
    for a general (possibly non-Hermitian) matrix `M`.

    Note :func:`.._construction._conjugation_superop` builds ``kron(M, M.T)``,
    which is only correct for Hermitian `M`; the polar unitaries used by
    :func:`full_rank_base_blend` are generally not Hermitian.
    """
    return _np.real(_bt.change_basis(_np.kron(M, M.conj()), 'std', basis))


# --------------------------------------------------------------------------- #
# Repair 1: rank cap -> blend the static base toward a unitary completion
# --------------------------------------------------------------------------- #

def full_rank_base_blend(member_superop, base_gate_superop, basis, c=0.9,
                         rank_tol=1e-9):
    r"""
    Replace a rank-deficient static base gate ``G_k`` with the full-rank convex
    blend

        ``G_c = (1 - c) * G_k + c * conj(U)``,

    where ``U`` is the polar unitary of the member's Kraus operator.  ``G_c``
    is CPTP by convexity (both endpoints are), full rank for any ``c > 0``, and
    -- for a single-Kraus member -- *exactly seed-preserving*:
    ``conj(U) . rootconj(E_k) == member`` holds identically, so the seeded
    member is untouched while the frozen base's rank cap is lifted.

    ``c`` must be near 1, not merely nonzero: the blend's conditioning is what
    the optimizer actually feels, and measured fit quality degrades steeply as
    ``c`` falls (c = 0.9 fits ~2x better than c = 0.5 on the calibration
    dataset even though both pass every structural check).

    For a member that genuinely needs more than one Kraus operator there is no
    canonical unitary completion; this function blends toward the polar unitary
    of the *largest-weight* Kraus operator and warns, since seed preservation
    is then only approximate.  (Measured on random completions, the fit is
    insensitive to which completion is chosen.)  If the blend fails its
    post-hoc CP/TP/rank verification the original ``G_k`` is returned with a
    warning instead.

    Parameters
    ----------
    member_superop : numpy.ndarray
        The dense CPTR member ``I_k`` (in `basis`) whose base gate is being
        repaired.

    base_gate_superop : numpy.ndarray
        The static base gate ``G_k`` (e.g. from ``_decompose_cptr``).

    basis : BasisLike
        The basis of both superops.

    c : float, optional
        The blend weight toward the unitary completion.

    rank_tol : float, optional
        Singular-value cutoff for the post-hoc full-rank verification.

    Returns
    -------
    numpy.ndarray
        The repaired (or, on verification failure, original) base gate.
    """
    basis = _Basis.cast(basis, member_superop.shape[0])
    dim = basis.dim
    udim = round(dim ** 0.5)
    G_k = _np.asarray(base_gate_superop)
    if not (0.0 <= c <= 1.0):
        raise ValueError(f"c must be in [0, 1], not {c}.")
    if c == 0.0:
        return G_k

    kraus = _ot.minimal_kraus_decomposition(_np.asarray(member_superop), basis)
    if len(kraus) > 1:
        weights = [_np.linalg.norm(K) for K in kraus]
        K_top = kraus[int(_np.argmax(weights))]
        _warnings.warn(
            f"full_rank_base_blend: the member has Kraus rank {len(kraus)}, so there "
            "is no canonical unitary completion; blending toward the polar unitary of "
            "the largest-weight Kraus operator (weight fraction "
            f"{max(weights) / _np.linalg.norm(weights):.2f}).  The blended base is "
            "CPTP and full rank but the seeded member is only approximately "
            "preserved.")
    else:
        K_top = kraus[0]
    U_svd, _, Wh_svd = _np.linalg.svd(K_top)
    K_superop = conj_superop(U_svd @ Wh_svd, basis)

    G_c = (1.0 - c) * G_k + c * K_superop

    # Post-hoc verification: CP, TP, full rank.  On failure, warn and keep G_k.
    I_superket = _bt.stdmx_to_vec(_np.eye(udim), basis).ravel().real
    min_choi = _np.linalg.eigvalsh(_jam(G_c, basis, 'std', normalized=True)).min()
    tp_ok = _np.allclose(I_superket @ G_c, I_superket, atol=1e-9)
    svals = _np.linalg.svd(G_c, compute_uv=False)
    full_rank = bool(_np.sum(svals > rank_tol * svals[0]) == dim)
    if min_choi < -1e-9 or not tp_ok or not full_rank:
        _warnings.warn(
            f"full_rank_base_blend: the blended base failed verification "
            f"(min Choi eig {min_choi:.2e}, TP {tp_ok}, full rank {full_rank}); "
            "returning the original base gate unrepaired.")
        return G_k
    return G_c


# --------------------------------------------------------------------------- #
# Repair 2: CP boundary -> pull the effect spectra strictly inside (0, 1)
# --------------------------------------------------------------------------- #

def interior_effect_offset(effect_superkets, basis, eps=0.01):
    r"""
    Mix each base effect with the identity,

        ``E_k -> (1 - eps) * E_k + (eps / n) * I``,

    which preserves the completeness sum for any number of outcomes ``n`` and
    moves every effect eigenvalue strictly inside ``(0, 1)`` (new spectrum in
    ``[eps/n, 1 - eps + eps/n]``).

    Why: an effect eigenvalue exactly at 0 or 1 puts the member map at a
    non-differentiable point (the sqrt(E) cusp) *and* -- because the shared
    error map's unital dual compresses spectra -- lets the fit trajectory
    reach the CP boundary and stick there, a measured mid-fit trap.  A
    positive ``eps`` buys a certified spectral floor for the entire fit at a
    statistically invisible price (the offset is free at the optimum).

    Parameters
    ----------
    effect_superkets : dict[label, numpy.ndarray]
        The base effect superkets in `basis`.

    basis : BasisLike
        The basis of the superkets.

    eps : float, optional
        The identity admixture.

    Returns
    -------
    dict[label, numpy.ndarray]
    """
    first = next(iter(effect_superkets.values()))
    basis = _Basis.cast(basis, _np.asarray(first).size)
    udim = round(basis.dim ** 0.5)
    I_sk = _bt.stdmx_to_vec(_np.eye(udim), basis).ravel().real
    n = len(effect_superkets)
    return _collections.OrderedDict(
        (lbl, (1.0 - eps) * _np.asarray(E).ravel() + (eps / n) * I_sk)
        for lbl, E in effect_superkets.items())


# --------------------------------------------------------------------------- #
# Repair 3: span deficiency -> displace the base effects until m = n - 1
# --------------------------------------------------------------------------- #

def _spectral_normalized_traceless_directions(dim):
    """Traceless Hermitian direction candidates with unit spectral norm."""
    pp = _Basis.cast('pp', dim)
    out = []
    for el in pp.elements[1:]:            # elements after I are traceless Hermitian
        el = _np.asarray(el)
        out.append(el / max(_np.abs(_np.linalg.eigvalsh(el)).max(), 1e-300))
    return out


def displace_base_effects(effect_superkets, basis, delta=0.2, directions=None):
    r"""
    Displace span-deficient base effects until they span ``n`` dimensions
    (``m = n - 1``), in completeness-preserving pairs:

        ``E_i -> E_i + (delta/2) * T,   E_j -> E_j - (delta/2) * T``

    for traceless Hermitian directions ``T`` (unit spectral norm) chosen
    orthogonal to the current span.  For the canonical 2-outcome
    proportional-to-identity case this is exactly ``(I +/- delta*sigma) / 2``
    for a Pauli ``sigma`` -- and the direction is immaterial: the dominant
    needed deviation is typically a trace component, reachable from any sigma
    through damping-type duals.

    This is a *displaced seed*: the effects move and the chart's error map
    stays at L = 0, so the seeded instrument is deliberately not the original
    one.  Do NOT try to preserve the seed by initializing a compensating error
    generator instead -- an exactly-compensating L needs unbounded rates and
    starts the fit in a gradient desert (measured).

    ``delta`` also bounds what the repaired chart can reach: the base spectral
    interval it creates is a certified box for the whole fit (extent
    ``~ delta/sqrt(2)`` per effect at d = 2), so an under-sized ``delta``
    saturates silently.  If a displacement would push an effect's spectrum out
    of (0, 1) it is automatically shrunk for that pair (with a warning).

    Parameters
    ----------
    effect_superkets : dict[label, numpy.ndarray]
        The base effect superkets in `basis`.

    basis : BasisLike
        The basis of the superkets.

    delta : float, optional
        The displacement magnitude (peak spectral shift of a displaced pair).

    directions : list of numpy.ndarray, optional
        Traceless Hermitian matrices to try (in order) as displacement
        directions before the built-in Pauli-product candidates.

    Returns
    -------
    dict[label, numpy.ndarray]
    """
    first = next(iter(effect_superkets.values()))
    basis = _Basis.cast(basis, _np.asarray(first).size)
    dim = basis.dim
    labels = list(effect_superkets.keys())
    n = len(labels)
    E_sk = {lbl: _np.asarray(E).ravel().astype(complex)
            for lbl, E in effect_superkets.items()}

    candidates = list(directions) if directions is not None else []
    candidates += _spectral_normalized_traceless_directions(dim)

    def _span_rank():
        rows = _np.array([E_sk[lbl] for lbl in labels])
        return int(_np.linalg.matrix_rank(rows, tol=1e-9))

    def _orth_component(T):
        v = _bt.stdmx_to_vec(T, basis).ravel()
        rows = _np.array([E_sk[lbl] for lbl in labels])
        Q = _np.linalg.svd(rows, full_matrices=False)[2]
        r = int(_np.sum(_np.linalg.svd(rows, compute_uv=False) > 1e-9))
        Q = Q[:r]
        return _np.linalg.norm(v - Q.conj().T @ (Q @ v))

    def _spec_ok(E_vec, margin=1e-6):
        spec = _np.linalg.eigvalsh(_bt.vec_to_stdmx(E_vec, basis, keep_complex=True))
        return spec.min() > margin and spec.max() < 1.0 - margin

    pair = 0
    max_rounds = dim * n  # generous upper bound; each round should add a dimension
    for _ in range(max_rounds):
        if _span_rank() - 1 >= n - 1:
            break
        # the direction most orthogonal to the current span
        best = max(candidates, key=_orth_component)
        if _orth_component(best) < 1e-9:
            raise ValueError(
                "displace_base_effects: no traceless direction outside the current "
                "effect span was found; the span cannot be increased further "
                "(is n larger than d**2?).")
        T_sk = _bt.stdmx_to_vec(best, basis).ravel()

        i, j = labels[pair % n], labels[(pair + 1) % n]
        pair += 1
        a = delta / 2.0
        for _ in range(6):
            if _spec_ok(E_sk[i] + a * T_sk) and _spec_ok(E_sk[j] - a * T_sk):
                break
            a /= 2.0
        else:
            continue   # this pair cannot absorb the displacement; try the next pair
        if a < delta / 2.0:
            _warnings.warn(
                f"displace_base_effects: displacement for effects ({i!r}, {j!r}) was "
                f"shrunk from {delta / 2.0:.3g} to {a:.3g} to keep their spectra "
                "inside (0, 1); the repaired chart's certified reach shrinks "
                "accordingly.")
        E_sk[i] = E_sk[i] + a * T_sk
        E_sk[j] = E_sk[j] - a * T_sk

    if _span_rank() - 1 < n - 1:
        _warnings.warn(
            f"displace_base_effects: the effect span is still deficient "
            f"(m = {_span_rank() - 1} < {n - 1}) after displacement; consider a "
            "smaller delta (positivity kept shrinking the steps) or explicit "
            "`directions`.")
    return _collections.OrderedDict((lbl, _np.real_if_close(E_sk[lbl]).real)
                                    for lbl in labels)


# --------------------------------------------------------------------------- #
# Projection onto the CPTP-instrument set
# --------------------------------------------------------------------------- #

def _proj_cp_choi_clip(M, basis):
    """Nearest CP map in Frobenius norm: clip the Choi spectrum at zero."""
    choi = _jam(M, basis, 'std', normalized=True)
    evals, evecs = _np.linalg.eigh(choi)
    if evals.min() >= 0:
        return M
    choi_psd = (evecs * _np.clip(evals, 0.0, None)) @ evecs.conj().T
    return _np.real(_jam_inv(choi_psd, 'std', basis, normalized=True))


def _proj_tp_sum(Ms, basis):
    """
    Nearest member tuple (Frobenius) whose sum is TP.

    TP of the sum reads ``(sum_k M_k)^T @ vecI == vecI`` in any basis; the
    minimum-norm correction distributes the defect over the members along the
    constraint normals.  (In a first-element-identity basis this reduces to
    spreading the row-0 defect evenly.)
    """
    vecI = _bt.stdmx_to_vec(_np.eye(round(basis.dim ** 0.5)), basis).ravel().real
    c2 = float(vecI @ vecI)
    K = len(Ms)
    defect = sum(M.T @ vecI for M in Ms) - vecI
    corr = _np.outer(vecI, defect) / (K * c2)
    return [M - corr for M in Ms]


def _project_cptp_dykstra(members, basis, iters=2000, tol=1e-14):
    """
    Dykstra's algorithm onto {each member CP} n {sum TP}: the same
    Frobenius-norm projection the SDP computes, with no cvxpy dependency.
    (Plain alternating projections would land somewhere in the intersection;
    Dykstra converges to the *nearest* point.)
    """
    X = [_np.array(M, dtype=float) for M in members]
    p = [_np.zeros_like(M) for M in X]
    q = [_np.zeros_like(M) for M in X]
    for _ in range(iters):
        Y = [_proj_cp_choi_clip(X[k] + p[k], basis) for k in range(len(X))]
        p = [X[k] + p[k] - Y[k] for k in range(len(X))]
        Xn = _proj_tp_sum([Y[k] + q[k] for k in range(len(X))], basis)
        q = [Y[k] + q[k] - Xn[k] for k in range(len(X))]
        shift = max(_np.linalg.norm(Xn[k] - X[k]) for k in range(len(X)))
        X = Xn
        if shift < tol:
            break
    return X


def project_instrument_to_cptp(instrument_or_members, basis, norm='frobenius',
                               method='auto', dykstra_iters=2000,
                               dykstra_tol=1e-14, **solve_kwargs):
    """
    Project an instrument's members onto the CPTP-instrument set (each member
    CP, members summing to a TP map).

    This is the instrument-level convenience for
    :func:`pygsti.tools.sdptools.project_instrument_members`, with a
    dependency-free fallback: when cvxpy is unavailable (or ``method='dykstra'``)
    the Frobenius projection is computed by Dykstra's alternating-projection
    algorithm instead of an SDP.  The two agree to solver tolerance (the
    feasible set is the same intersection of convex sets).

    Parameters
    ----------
    instrument_or_members : Instrument or TPInstrument or dict
        The instrument (or ``{label: dense superop}`` members) to project.
        Members may be non-CP -- e.g. instruments from a full-TP GST fit.

    basis : BasisLike
        The basis of the dense representations.

    norm : {'frobenius', 'diamond', 'spectral'}, optional
        The projection norm.  Only ``'frobenius'`` (the unique-minimizer
        Euclidean projection) is available with the Dykstra method.

    method : {'auto', 'sdp', 'dykstra'}, optional
        ``'auto'`` uses the SDP when cvxpy is available and otherwise falls
        back to Dykstra with a warning.

    dykstra_iters, dykstra_tol : optional
        Iteration cap and convergence tolerance for the Dykstra method.

    solve_kwargs : optional
        Forwarded to the SDP solver (see :func:`~pygsti.tools.sdptools.solve_sdp`).

    Returns
    -------
    dict[label, numpy.ndarray]
        The projected member superops, keyed like the input.
    """
    from pygsti.tools import sdptools as _sdps
    members = _as_dense_members(instrument_or_members)
    basis = _Basis.cast(basis, next(iter(members.values())).shape[0])

    if method not in ('auto', 'sdp', 'dykstra'):
        raise ValueError(f"method must be 'auto', 'sdp' or 'dykstra', not {method!r}.")
    use_sdp = (method == 'sdp') or (method == 'auto' and _sdps.CVXPY_ENABLED)
    if method == 'auto' and not _sdps.CVXPY_ENABLED:
        _warnings.warn("project_instrument_to_cptp: cvxpy is not available; using "
                       "the Dykstra alternating-projection fallback (Frobenius "
                       "norm only).")
    if not use_sdp and norm != 'frobenius':
        raise ValueError(f"The Dykstra method computes the Frobenius projection "
                         f"only; norm={norm!r} requires cvxpy (method='sdp').")

    labels = list(members.keys())
    if use_sdp:
        projected, _ = _sdps.project_instrument_members(
            [members[lbl] for lbl in labels], basis, norm=norm, **solve_kwargs)
        if any(P is None for P in projected):
            if method == 'sdp':
                raise RuntimeError(
                    "project_instrument_to_cptp: every available SDP solver failed; "
                    "try method='dykstra' (Frobenius) or different solver options.")
            _warnings.warn("project_instrument_to_cptp: the SDP solvers failed; "
                           "falling back to the Dykstra method.")
            projected = _project_cptp_dykstra([members[lbl] for lbl in labels],
                                              basis, dykstra_iters, dykstra_tol)
    else:
        projected = _project_cptp_dykstra([members[lbl] for lbl in labels],
                                          basis, dykstra_iters, dykstra_tol)
    return _collections.OrderedDict(zip(labels, (_np.asarray(P) for P in projected)))


# --------------------------------------------------------------------------- #
# One-call patcher
# --------------------------------------------------------------------------- #

def patch_instrument_seed(instrument_or_members, basis, mode='auto', c=0.9,
                          eps=0.01, delta=0.2, gate_parameterization='CPTPLND',
                          povm_errormap='CPTPLND', return_report=False):
    r"""
    Diagnose an instrument seed's static base, apply the standard repairs, and
    return a ready-to-fit parameterized :class:`Instrument`.

    The pipeline: decompose each dense member into ``(E_k, G_k)`` (the base
    that a Lindblad chart would freeze), then

    1. blend any rank-deficient ``G_k`` toward its unitary completion
       (:func:`full_rank_base_blend`, weight `c`) -- lifts the member rank cap;
    2. displace the effects if their span is deficient
       (:func:`displace_base_effects`, magnitude `delta`) -- unfreezes
       proportional-to-identity and n >= 3 commuting effect tuples;
    3. offset the effects into the open interval (0, 1) if any spectrum
       touches the boundary after step 2 (:func:`interior_effect_offset`,
       admixture `eps`) -- removes the CP-boundary trap.

    With ``mode='auto'`` (default) each repair is applied only when its check
    fails; ``mode='all'`` applies all three unconditionally.  The repaired
    base is re-checked and a warning names anything still flagged (nothing
    here ever raises on a flag).

    Note the repaired chart is a *deliberately displaced seed* wherever step 2
    fires (and approximately displaced for multi-Kraus members in step 1): the
    fit is expected to recover the difference, and measured end-to-end this
    recipe took a target-seeded CPTPLND instrument fit from 2*Delta(logL)
    3838 to ~411 on the calibration dataset (TP floor 390).

    Parameters
    ----------
    instrument_or_members : Instrument or TPInstrument or dict
        The seed instrument (or its dense ``{label: superop}`` members).
        Members must be CP -- project first (:func:`project_instrument_to_cptp`)
        if they are not.

    basis : BasisLike
        The basis of the dense representations.

    mode : {'auto', 'all'}, optional
        Apply repairs only on check failure, or unconditionally.

    c, eps, delta : float, optional
        Repair magnitudes (see the module docstring for calibration caveats).

    gate_parameterization, povm_errormap : str, optional
        The chart to build, as in :meth:`Instrument.from_effects`.

    return_report : bool, optional
        Also return a dict recording which repairs fired and the residual
        diagnostics of the repaired base.

    Returns
    -------
    Instrument
        The parameterized instrument, seeded on the repaired base.
    report : dict
        Only when `return_report` is True.
    """
    if mode not in ('auto', 'all'):
        raise ValueError(f"mode must be 'auto' or 'all', not {mode!r}.")
    members = _as_dense_members(instrument_or_members)
    basis = _Basis.cast(basis, next(iter(members.values())).shape[0])
    dim = basis.dim

    effects, gates = _collections.OrderedDict(), _collections.OrderedDict()
    for lbl, I_k in members.items():
        try:
            E_k, G_k = _decompose_cptr(I_k, basis)
        except (ValueError, AssertionError) as e:
            raise ValueError(
                f"Could not decompose instrument member {lbl!r}: {e}\nIf the members "
                "are not completely positive (e.g. they come from a TP fit), project "
                "them first with project_instrument_to_cptp and patch the result.") from e
        effects[lbl], gates[lbl] = E_k, G_k

    applied = []

    # 1. rank cap -> blend
    for lbl, G_k in gates.items():
        rank = int(_np.linalg.matrix_rank(G_k, tol=1e-9))
        if mode == 'all' or rank < dim:
            G_c = full_rank_base_blend(members[lbl], G_k, basis, c=c)
            if G_c is not G_k:
                gates[lbl] = G_c
                applied.append(('full_rank_base_blend', lbl))

    # 2. span deficiency -> displace
    span = _span_from_base(basis, effects)
    if mode == 'all' or span['deficient']:
        effects = displace_base_effects(effects, basis, delta=delta)
        applied.append(('displace_base_effects', None))

    # 3. boundary contact (evaluated after any displacement) -> interior offset
    extent = _extent_from_base(basis, effects)
    on_boundary = any(row['boundary_margin'] <= 1e-9 for row in extent.values())
    if mode == 'all' or on_boundary:
        effects = interior_effect_offset(effects, basis, eps=eps)
        applied.append(('interior_effect_offset', None))

    # Re-check the repaired base; warn (never raise) if anything is still flagged.
    info = {'gate_frozen': {lbl: True for lbl in gates}}
    residual = []
    for lbl, row in _ranks_from_base(basis, gates, info).items():
        if row['rank'] < row['full']:
            residual.append(f"RANK-CAP({row['rank']}/{row['full']}) on {lbl!r}")
    span_after = _span_from_base(basis, effects)
    if span_after['deficient']:
        residual.append(f"SPAN-DEFICIENT({span_after['m']}/{span_after['target']})")
    extent_after = _extent_from_base(basis, effects)
    if any(row['boundary_margin'] <= 1e-9 for row in extent_after.values()):
        residual.append("CP-BOUNDARY")
    if residual:
        _warnings.warn("patch_instrument_seed: the repaired base still flags " +
                       ", ".join(residual) + ".  If the flagged structure is "
                       "intended (e.g. genuinely rank-deficient members), this "
                       "is informational.")

    with _warnings.catch_warnings():
        # any surviving rank deficiency was already reported as a residual flag
        _warnings.filterwarnings('ignore', message='.*post-measurement gate.*singular.*')
        inst = _Instrument(_parameterized_instrument(basis, effects, gates,
                                                     gate_parameterization, povm_errormap))
    if return_report:
        max_shift = max(
            _np.linalg.norm(_np.asarray(m.to_dense('HilbertSchmidt')) - members[lbl])
            for lbl, m in inst.items())
        report = {'applied': applied, 'residual_flags': residual,
                  'effect_span': span_after, 'effect_extent': extent_after,
                  'max_member_shift': float(max_shift)}
        return inst, report
    return inst


def patch_model_instrument_seeds(model, inst_labels=None, **patch_kwargs):
    """
    Return a copy of `model` in which every (selected) instrument has been
    rebuilt by :func:`patch_instrument_seed` from its current dense members.

    The returned model is intended as a GST *initial model*: its instruments
    carry the parameterization requested via ``gate_parameterization`` /
    ``povm_errormap`` (default CPTPLND) regardless of what they were before.
    The rest of the model is untouched.

    Parameters
    ----------
    model : ExplicitOpModel
        The model whose instrument seeds to repair.  Not modified.

    inst_labels : iterable, optional
        Which instruments (default: all).

    patch_kwargs : optional
        Forwarded to :func:`patch_instrument_seed`.

    Returns
    -------
    ExplicitOpModel
    """
    mdl = model.copy()
    basis = mdl.basis
    if inst_labels is None:
        inst_labels = list(mdl.instruments.keys())
    for lbl in inst_labels:
        patched = patch_instrument_seed(mdl.instruments[lbl], basis, **patch_kwargs)
        mdl[lbl] = patched
    return mdl

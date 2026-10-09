r"""
Repairs ("patchers") for the static base of parameterized instrument charts.

Companion to :mod:`pygsti.modelmembers.instruments.diagnostics`: each function
here is the named repair for one of that module's flags.

================================  ==========================================
flag                              repair
================================  ==========================================
RANK-CAP / ILL-CONDITIONED        unitary static gates (:func:`patch_instrument_seed`,
                                  or :func:`~pygsti.modelmembers.instruments.unitaryfactors.infer_unitary_factors`
                                  with :meth:`Instrument.from_effects`)
SPAN-DEFICIENT / ZERO-EXTENT      :func:`displace_base_effects` (delta ~ 0.1-0.2)
================================  ==========================================

The first pair only arises for static gates that are not unitary, which the
:class:`Instrument` constructors never build.

The CP-BOUNDARY flag is not a defect of the base: base effects at 0 or 1 (as
in an ideal measurement) keep the ideal instrument reachable.  Moving them
into the interior would shrink the reachable set, since the shared error map's
unital dual preserves spectral bounds.  What the flag affects is the fit's
starting point, and :func:`pygsti.models.prepare_lindblad_seed` handles that on
the model by giving the error maps small nonzero rates.

:func:`patch_instrument_seed` builds the chart on unitary static gates and
displaces the effects when their span check fails, returning a ready-to-fit
parameterized :class:`Instrument`; :func:`patch_model_instrument_seeds` does the same for
every instrument in a model copy.  :func:`project_instrument_to_cptp` is the
entry point for projecting (possibly non-CP) dense members onto the
CPTP-instrument set, with or without cvxpy.

All repairs operate on the dense ``(effect, gate)`` base *before* the
parameterized chart is built -- mutating a built chart's static operations in
place is unsupported (read-only member dicts, cached representations).

The default displacement (delta = 0.2) was
calibrated on a single 1-qubit, 196-circuit dataset with 2-outcome
instruments; it is a sensible starting point, not a universal constant.  In
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
from pygsti.modelmembers.instruments._construction import _parameterized_instrument
from pygsti.modelmembers.instruments.diagnostics import _span_from_base, _extent_from_base
from pygsti.modelmembers.instruments.unitaryfactors import _as_dense_members, infer_unitary_factors
from pygsti.tools import basistools as _bt
from pygsti.tools.jamiolkowski import jamiolkowski_iso as _jam, \
    jamiolkowski_iso_inv as _jam_inv


# --------------------------------------------------------------------------- #
# Span deficiency -> displace the base effects until m = n - 1
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

def patch_instrument_seed(instrument_or_members, basis, mode='auto', delta=0.2,
                          gate_parameterization='CPTPLND', povm_errormap='CPTPLND',
                          reference=None, return_report=False):
    r"""
    Build a ready-to-fit parameterized :class:`Instrument` from dense members,
    repairing the static base that a Lindblad chart would freeze.

    The pipeline:

    1. factor each member as its effect ``E_k`` followed by a post-measurement
       unitary ``U_k``
       (:func:`~pygsti.modelmembers.instruments.unitaryfactors.infer_unitary_factors`,
       ties broken toward `reference`).  The effects are kept exactly.  A member
       with more than one Kraus operator, such as a member of a fitted
       instrument, is replaced by its effect-preserving one-Kraus approximation;
       the Lindblad parameters are left to fit the difference;
    2. displace the effects if their span is deficient
       (:func:`displace_base_effects`, magnitude `delta`) -- unfreezes
       proportional-to-identity and n >= 3 commuting effect tuples.

    Effects touching 0 or 1 are left in place, so the ideal measurement stays
    reachable; run :func:`pygsti.models.prepare_lindblad_seed` on the model
    before fitting to start from interior effects.

    With ``mode='auto'`` (default) step 2 is applied only when its check fails;
    ``mode='all'`` applies it unconditionally.  The repaired base is re-checked
    and a warning names anything still flagged (nothing here ever raises on a
    flag).

    The repaired chart is a *deliberately displaced seed* wherever step 2 fires
    or step 1 approximates a member: the fit is expected to recover the
    difference.

    Parameters
    ----------
    instrument_or_members : Instrument or TPInstrument or dict
        The seed instrument (or its dense ``{label: superop}`` members).
        Members must be CP -- project first (:func:`project_instrument_to_cptp`)
        if they are not.

    basis : BasisLike
        The basis of the dense representations.

    mode : {'auto', 'all'}, optional
        Displace the effects only on a span-check failure, or unconditionally.

    delta : float, optional
        Displacement magnitude (see the module docstring for calibration caveats).

    gate_parameterization, povm_errormap : str, optional
        The chart to build, as in :meth:`Instrument.from_effects`.

    reference : numpy.ndarray or dict, optional
        Unitary (or `{label: unitary}`) toward which step 1 breaks ties; the
        identity by default.  See ``infer_unitary_factors``.

    return_report : bool, optional
        Also return a dict recording which repairs fired, each member's
        unitary factorization, and the residual diagnostics of the repaired base.

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

    # 1. effects and post-measurement unitaries
    try:
        pairs, factors = infer_unitary_factors(members, basis, reference=reference)
    except ValueError as e:
        raise ValueError(f"{e}\nIf the members are not completely positive (e.g. they come from a TP "
                         "fit), project them first with project_instrument_to_cptp and patch the "
                         "result.") from e
    effects = _collections.OrderedDict((lbl, E) for lbl, (E, _) in pairs.items())
    unitaries = _collections.OrderedDict((lbl, U) for lbl, (_, U) in pairs.items())
    applied = [('infer_unitary_factors', None)]

    # 2. span deficiency -> displace
    span = _span_from_base(basis, effects)
    if mode == 'all' or span['deficient']:
        effects = displace_base_effects(effects, basis, delta=delta)
        applied.append(('displace_base_effects', None))

    # Re-check the repaired base; warn (never raise) if anything is still flagged.
    residual = []
    span_after = _span_from_base(basis, effects)
    if span_after['deficient']:
        residual.append(f"SPAN-DEFICIENT({span_after['m']}/{span_after['target']})")
    extent_after = _extent_from_base(basis, effects)
    if residual:
        _warnings.warn("patch_instrument_seed: the repaired base still flags " +
                       ", ".join(residual) + ".  If the flagged structure is "
                       "intended, this is informational.")

    inst = _Instrument(_parameterized_instrument(basis, effects, unitaries,
                                                 gate_parameterization, povm_errormap))
    if return_report:
        max_shift = max(
            _np.linalg.norm(_np.asarray(m.to_dense('HilbertSchmidt')) - members[lbl])
            for lbl, m in inst.items())
        report = {'applied': applied, 'residual_flags': residual, 'unitary_factors': factors,
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

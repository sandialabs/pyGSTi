r"""
Data-free diagnostics for the static base of parameterized instrument charts.

The question these functions answer, precisely:

    Given an :class:`Instrument` built as ``member_k(theta) = <parameterized
    family> around a STATIC base (G_static,k, E_base,k)`` -- the structure
    produced by :meth:`Instrument.from_effects` / :meth:`from_cptr_superops`
    for Lindblad gate parameterizations -- can the chart reach every physical
    instrument near the seed, or has the base commitment frozen directions that
    no seeding, scheduling or restarting inside the chart can recover?

Three O(1), data-free structural checks decide this in advance:

* **rank** (:func:`static_base_ranks`): a static base gate ``G_static,k`` with
  superop rank ``r < d**2`` caps every reachable member at rank ``r``, because
  the parameterized factor ``exp(L)`` is invertible.

* **span** (:func:`base_effect_span`): all ``n`` effects are images of ONE
  shared error map's unital dual, ``E_k = M^dag(E_base,k)``, and every unital
  dual annihilates the identity component.  The reachable effect-tuple set
  therefore has dimension exactly ``d**2 * m`` with
  ``m = dim span{E_base,k} - 1``, against a target of ``d**2 * (n-1)``.
  ``m = 0`` is the proportional-to-identity cap (any ``c*I`` base effect is a
  fixed point of every unital dual); ``m < n-1`` also bites whenever ``n >= 3``
  base effects commute -- an *ideal* 3-outcome measurement is structurally
  deficient in this chart.

* **extent** (:func:`base_effect_extent`): the unital dual compresses spectra,
  so ``spec(E_k(theta))`` lies inside ``[lmin(E_base,k), lmax(E_base,k)]`` for
  every ``theta``.  The base's spectral interval is a certified containing box
  for the entire fit trajectory: an interval touching {0, 1} is a CP-boundary
  trap, and one narrower than the true deviation saturates silently.

A fourth, sampled check -- the Jacobian certificate, run by default from
:func:`diagnose_instrument` / :func:`diagnose_model_instruments` -- compares
the chart Jacobian's rank at generic parameter points against the
TP-instrument tangent dimension ``n*d**4 - d**2``.  It is the net for failure
species the structural checks do not know about, and it names its blind
directions physically (frozen effect vs. frozen member direction).  Its rank
decision self-reports as ambiguous when the singular-value gap is small
(observed at d = 4); the structural checks above stay decisive there.

Diagnostics **warn with a named repair and never raise**: a flag is not always
a bug.  A measure-and-reset instrument, for example, legitimately reports a
rank cap -- its members really are rank-1 superoperators, and if the user
believes that structure the flag is informational.  Repairs live in
:mod:`pygsti.modelmembers.instruments.seeding`.
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
from pygsti.baseobjs import BasisLike as _BasisLike
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _pv
from pygsti.modelmembers.instruments.instrument import Instrument as _Instrument
from pygsti.modelmembers.instruments.tpinstrument import TPInstrument as _TPInstrument
from pygsti.modelmembers.instruments._construction import (
    _decompose_cptr, _parameterized_instrument, _is_lindblad_type
)
from pygsti.tools import basistools as _bt

#: Relative singular-value cutoff for Jacobian ranks.  Sits above pyGSTi's
#: analytic-derivative noise floor (~1e-8 relative) and far below the smallest
#: genuine singular value seen in practice (~1e-3 relative).
JACOBIAN_RANK_RTOL = 1e-6

#: A rank decision whose singular-value gap is below this is reported AMBIGUOUS.
AMBIGUOUS_GAP = 1e2

#: Reachable-conditioning threshold for the ILL-CONDITIONED warning.  This is a
#: CALIBRATED HEURISTIC, not a certificate: it was set from a single measured
#: c-grid (reachable conditioning 0.010 -> 2*Delta(logL) 2900, 0.050 -> 1199,
#: 0.090 -> 621, on a 1-qubit, 196-circuit dataset) and should not be trusted
#: quantitatively elsewhere.
ILL_CONDITIONED_TOL = 2e-2


class InstrumentSeedFlag(_collections.namedtuple('InstrumentSeedFlag',
                                                 ['name', 'detail', 'repair'])):
    """
    One diagnostic finding: a short flag `name`, a human-readable `detail`,
    and the named `repair` (or ``None`` for purely informational flags).
    """
    def __str__(self):
        s = f"{self.name}: {self.detail}"
        if self.repair:
            s += f"  [repair: {self.repair}]"
        return s


class InstrumentSeedReport(object):
    """
    The result of diagnosing one instrument's static base.

    Attributes
    ----------
    label : object
        The instrument label (or None when diagnosing a bare instrument).

    flags : list of InstrumentSeedFlag
        The findings, in the order a user should act on them.  Empty means
        every check passed.

    details : dict
        The raw check outputs: keys ``'base_ranks'``, ``'effect_span'``,
        ``'effect_extent'`` always; plus the Jacobian-certificate keys
        (``'member_rank'``, ``'member_target'``, ``'effect_rank'``,
        ``'effect_target'``, ``'blind'``, ...) when the deep check ran.

    deep : bool
        Whether the Jacobian certificate ran.
    """

    def __init__(self, label, flags, details, deep):
        self.label = label
        self.flags = flags
        self.details = details
        self.deep = deep

    @property
    def ok(self):
        """True when no check raised a flag."""
        return len(self.flags) == 0

    @property
    def verdict(self):
        """A one-line summary: 'pass', or the space-joined flag names."""
        return ' '.join(f.name for f in self.flags) if self.flags else 'pass'

    def __str__(self):
        lbl = f" {self.label}" if self.label is not None else ""
        lines = [f"InstrumentSeedReport{lbl}: {self.verdict}"]
        for f in self.flags:
            lines.append(f"  - {f}")
        span = self.details['effect_span']
        lines.append(f"  effect span m = {span['m']} (target {span['target']}); "
                     f"reachable effect-tuple dim {span['reachable_effect_dim']} "
                     f"of {span['full_effect_dim']}")
        for k, row in self.details['effect_extent'].items():
            lo, hi = row['spec_interval']
            lines.append(f"  effect {k!r}: base spectrum in [{lo:.4f}, {hi:.4f}], "
                         f"certified extent {row['extent']:.4f}, "
                         f"boundary margin {row['boundary_margin']:.4f}")
        for k, row in self.details['base_ranks'].items():
            frozen = "frozen" if row['frozen'] else "NOT frozen (additively parameterized)"
            lines.append(f"  gate {k!r}: static-base rank {row['rank']}/{row['full']} ({frozen})")
        if self.deep:
            d = self.details
            lines.append(f"  jacobian: member rank {d['member_rank']}/{d['member_target']} "
                         f"(gap {d['member_gap']:.1e}), effect rank "
                         f"{d['effect_rank']}/{d['effect_target']} (gap {d['effect_gap']:.1e})")
        return '\n'.join(lines)

    def __repr__(self):
        return f"<InstrumentSeedReport {self.label!r}: {self.verdict}>"


# --------------------------------------------------------------------------- #
# Static-base extraction
# --------------------------------------------------------------------------- #

def _static_parts_of_member(member):
    """
    Read the frozen base ``(E_base superket, G_static superop, gate_frozen)``
    off a member built by ``_parameterized_instrument``, or return None if the
    member does not have that structure.
    """
    if not (isinstance(member, _op.ComposedOp) and len(member.factorops) == 2
            and isinstance(member.factorops[0], _op.RootConjOperator)):
        return None
    root, gate = member.factorops
    effect = root._effect
    if isinstance(effect, _pv.ComposedPOVMEffect):
        E = _np.asarray(effect.effect_vec.to_dense()).ravel()
    else:
        E = _np.asarray(effect.to_dense()).ravel()

    if isinstance(gate, _op.ComposedOp) and len(gate.factorops) >= 1 \
       and isinstance(gate.factorops[0], _op.StaticArbitraryOp):
        return E, _np.asarray(gate.factorops[0].to_dense()), True
    if isinstance(gate, _op.StaticArbitraryOp):
        return E, _np.asarray(gate.to_dense()), True
    if isinstance(gate, _op.FullTPOp):
        return E, _np.asarray(gate.to_dense()), False
    return None


def instrument_static_base(instrument, basis):
    """
    The static base ``(E_base,k, G_static,k)`` an instrument chart is (or would
    be) committed to.

    For an instrument whose members have the canonical
    ``ComposedOp([RootConjOperator(E_k), gate])`` structure the base is read
    off exactly.  For anything else -- a plain dense :class:`Instrument`, a
    :class:`TPInstrument`, or a ``{label: superop}`` dict -- the base is
    *predicted* by running the same decomposition
    (:func:`~pygsti.modelmembers.instruments._construction._decompose_cptr`)
    that :meth:`Instrument.from_cptr_superops` would use, i.e. the base that a
    conversion to a Lindblad parameterization would freeze.

    Parameters
    ----------
    instrument : Instrument or TPInstrument or dict
        The instrument, or a ``{label: dense superop}`` mapping of its members.

    basis : BasisLike
        The basis in which dense arrays are expressed.

    Returns
    -------
    basis : Basis
    effect_superkets : dict[label, numpy.ndarray]
    gate_superops : dict[label, numpy.ndarray]
    info : dict
        ``info['exact']`` is True when the base was read off an existing chart
        rather than predicted; ``info['gate_frozen'][label]`` says whether the
        chart holds (or would hold) that gate as a frozen static op.
    """
    if isinstance(instrument, (_Instrument, _TPInstrument)):
        members = _collections.OrderedDict(instrument.items())
        dim = next(iter(members.values())).dim
        basis = _Basis.cast(basis, dim)

        parts = {lbl: _static_parts_of_member(m) for lbl, m in members.items()}
        if all(p is not None for p in parts.values()):
            effects = {lbl: p[0] for lbl, p in parts.items()}
            gates = {lbl: p[1] for lbl, p in parts.items()}
            frozen = {lbl: p[2] for lbl, p in parts.items()}
            return basis, effects, gates, {'exact': True, 'gate_frozen': frozen}

        dense = _collections.OrderedDict(
            (lbl, _np.asarray(m.to_dense('HilbertSchmidt'))) for lbl, m in members.items())
    elif isinstance(instrument, dict):
        dense = _collections.OrderedDict(
            (lbl, _np.asarray(m)) for lbl, m in instrument.items())
        dim = next(iter(dense.values())).shape[0]
        basis = _Basis.cast(basis, dim)
    else:
        raise ValueError(f"Cannot interpret {type(instrument).__name__} as an instrument; "
                         "pass an Instrument, TPInstrument, or {label: superop} dict.")

    effects, gates, frozen = {}, {}, {}
    for lbl, I_k in dense.items():
        try:
            E_k, G_k = _decompose_cptr(I_k, basis)
        except (ValueError, AssertionError) as e:
            raise ValueError(
                f"Could not decompose instrument member {lbl!r} into effect and "
                f"post-measurement gate: {e}\nIf the members are not completely "
                "positive (e.g. they come from a TP fit), project them onto the "
                "CPTP-instrument set first -- see "
                "pygsti.modelmembers.instruments.seeding.project_instrument_to_cptp "
                "or pygsti.tools.sdptools.project_instrument_members.") from e
        effects[lbl], gates[lbl] = E_k, G_k
        frozen[lbl] = True   # a Lindblad conversion would freeze this gate
    return basis, effects, gates, {'exact': False, 'gate_frozen': frozen}


# --------------------------------------------------------------------------- #
# The three O(1) structural checks
# --------------------------------------------------------------------------- #

def _ranks_from_base(basis, gates, info, tol=1e-9):
    out = {}
    for lbl, G in gates.items():
        s = _np.linalg.svd(G, compute_uv=False)
        out[lbl] = {'rank': int(_np.sum(s > tol)), 'full': basis.dim,
                    'frozen': info['gate_frozen'][lbl], 'svals': s}
    return out


def static_base_ranks(instrument, basis, tol=1e-9):
    """
    Per-outcome superoperator rank of the static base gate ``G_static,k``.

    A frozen base gate with rank ``r < d**2`` is a hard cap: every member the
    chart can reach has rank at most ``r``, because the Lindblad factor
    ``exp(L)`` composed onto it is invertible.  This is exactly the failure
    mode of an ideal projective instrument, whose canonical decomposition
    produces a singular (complete-dephasing or worse) post-measurement gate.

    Returns
    -------
    dict[label, dict]
        Keys ``'rank'``, ``'full'`` (= d**2), ``'frozen'`` (whether the chart
        actually freezes this gate -- a `'full TP'`-parameterized gate is not
        rank-capped), and ``'svals'`` (the singular values, for inspecting
        near-deficiency).
    """
    basis, _, gates, info = instrument_static_base(instrument, basis)
    return _ranks_from_base(basis, gates, info, tol)


def base_effect_span(instrument, basis, tol=1e-9):
    """
    The span law: the reachable effect-tuple set has dimension exactly
    ``d**2 * m``, where ``m = dim span{E_base,k : k = 1..n} - 1``, against a
    target of ``d**2 * (n-1)``.

    The "-1" is exact rather than a convention: a TP instrument's effects sum
    to the identity, so ``I`` is always in their span, and the traceless parts
    span a complement of it.  (Do not misread the traceless-part form: a rank-1
    projector like ``|0><0|`` is NOT traceless -- its traceless part is
    ``Z/2`` -- so an ideal two-outcome projective measurement is healthy at
    ``m = 1 = n-1``.  It is proportional-to-identity effects, and ``n >= 3``
    commuting ones, that are deficient.)

    Returns
    -------
    dict
        Keys ``'m'``, ``'target'`` (= n-1), ``'n_members'``,
        ``'reachable_effect_dim'`` (= d**2 * m), ``'full_effect_dim'``
        (= d**2 * (n-1)) and ``'deficient'`` (= m < n-1).
    """
    basis, effects, _, _ = instrument_static_base(instrument, basis)
    return _span_from_base(basis, effects, tol)


def _span_from_base(basis, effects, tol=1e-9):
    rows = _np.array([_np.asarray(E).ravel() for E in effects.values()])
    m = int(_np.linalg.matrix_rank(rows, tol=tol)) - 1
    n = len(effects)
    dim = basis.dim
    return {'m': m, 'target': n - 1, 'n_members': n,
            'reachable_effect_dim': dim * m, 'full_effect_dim': dim * (n - 1),
            'deficient': m < n - 1}


def base_effect_extent(instrument, basis):
    """
    The certified spectral box: for a shared-``ComposedPOVM`` chart the effect
    map's unital dual compresses spectra, so every reachable effect satisfies
    ``spec(E_k(theta)) subset [lmin(E_base,k), lmax(E_base,k)]`` -- exactly,
    for all theta.

    Consequences, all read off the base interval:

    * interval degenerate to a point -> the effect can never move at all
      (the span law's m = 0, per effect);
    * interval touching 0 or 1 -> the fit can (and, measured, does) hit the CP
      boundary mid-flight and stick there;
    * interval narrower than the deviation the data wants -> the fit saturates
      the box and quietly stops short.  ``'extent'`` is the certified maximum
      Frobenius distance any reachable effect can sit from the interval
      center: ``sqrt(d) * (lmax - lmin) / 2``.

    Returns
    -------
    dict[label, dict]
        Keys ``'spec_interval'``, ``'extent'``, ``'boundary_margin'``
        (= min(lmin, 1 - lmax)).
    """
    basis, effects, _, _ = instrument_static_base(instrument, basis)
    return _extent_from_base(basis, effects)


def _extent_from_base(basis, effects):
    udim = round(basis.dim ** 0.5)
    out = {}
    for lbl, E in effects.items():
        spec = _np.linalg.eigvalsh(_bt.vec_to_stdmx(_np.asarray(E).ravel(), basis,
                                                    keep_complex=True))
        lo, hi = float(spec.min()), float(spec.max())
        out[lbl] = {'spec_interval': (lo, hi),
                    'extent': float(_np.sqrt(udim) * (hi - lo) / 2.0),
                    'boundary_margin': float(min(lo, 1.0 - hi))}
    return out


# --------------------------------------------------------------------------- #
# The Jacobian certificate (sampled, run by default from diagnose_*)
# --------------------------------------------------------------------------- #

def _adaptive_scale(scale, dim):
    """
    Per-parameter sampling sigma that keeps ||L|| comparable across dimensions.

    A Lindblad generator on a dim-dimensional superoperator space has
    ~dim**2 - dim parameters, so drawing each at a fixed sigma makes ||L|| grow
    with dimension -- at d = 4 a nominal 0.25 puts the sample so far from the
    identity that exp(L) is strongly contracting and every rank measurement
    degrades for reasons unrelated to the base.  Normalized to the d = 2 case
    (12 parameters), leaving all d = 2 behavior unchanged.
    """
    return scale * _np.sqrt(12.0 / max(dim * dim - dim, 1))


def _chart_jacobian(inst):
    """
    d(stacked member superops) / d(instrument parameters) at the instrument's
    current parameter values, from the members' analytic ``deriv_wrt_params``.
    The shared POVM error map appears in every member's column block, which is
    what makes the parameter sharing show up correctly in the rank.
    """
    members = list(inst.items())
    dim = members[0][1].dim
    Np = inst.num_params
    own = _np.asarray(inst.gpindices_as_array())
    where = {g: i for i, g in enumerate(own)}

    J = _np.zeros((len(members) * dim * dim, Np))
    for i, (_, m) in enumerate(members):
        d = m.deriv_wrt_params()
        cols = [where[g] for g in _np.asarray(m.gpindices_as_array())]
        J[i * dim * dim:(i + 1) * dim * dim, cols] = d
    return J


def _effect_jacobian(J, dim, n_members, basis):
    """d(stacked effects E_k = member_k^dag(I)) / d(theta), linear in J."""
    udim = round(dim ** 0.5)
    I_sk = _bt.stdmx_to_vec(_np.eye(udim), basis).ravel().real
    Np = J.shape[1]
    out = _np.zeros((n_members * dim, Np))
    for i in range(n_members):
        blk = J[i * dim * dim:(i + 1) * dim * dim, :].reshape(dim, dim, Np)
        out[i * dim:(i + 1) * dim, :] = _np.einsum('baj,b->aj', blk, I_sk)
    return out


def _rank_report(A, rel_tol=JACOBIAN_RANK_RTOL):
    """Numerical rank with its evidence: (rank, singular values, decision gap)."""
    s = _np.linalg.svd(A, compute_uv=False) if A.size else _np.zeros(0)
    smax = s[0] if s.size else 0.0
    r = int(_np.sum(s > rel_tol * smax))
    gap = (s[r - 1] / s[r]) if (0 < r < s.size and s[r] > 0) else _np.inf
    return r, s, gap


def _tp_tangent_constraints(n_members, dim):
    """
    The TP-instrument tangent space is {(Delta_k) : sum_k Delta_k[0, :] = 0}
    (first row of the summed superop pinned, in a first-element-identity
    basis), of dimension n*dim**2 - dim.  Returns the constraint matrix.
    """
    C = _np.zeros((dim, n_members * dim * dim))
    for k in range(n_members):
        for c in range(dim):
            C[c, k * dim * dim + 0 * dim + c] = 1.0
    return C


def _blind_directions(J, n_members, dim, rel_tol=JACOBIAN_RANK_RTOL):
    """
    Orthonormal basis for the TP-tangent directions the chart cannot move in,
    to first order.  Computed within the TP tangent space so constraint
    directions are not miscounted as blind.
    """
    U, s, _ = _np.linalg.svd(J, full_matrices=False)
    r = int(_np.sum(s > rel_tol * (s[0] if s.size else 1.0)))
    U = U[:, :r]
    C = _tp_tangent_constraints(n_members, dim)
    A = _np.vstack([U.T, C])
    _, sv, Vt = _np.linalg.svd(A)
    nullmask = _np.zeros(A.shape[1], dtype=bool)
    nullmask[len(sv):] = True
    if sv.size:
        nullmask[:len(sv)] = sv <= rel_tol * sv[0]
    return Vt[nullmask].T


def _effect_map(n_members, dim, basis):
    """The linear map T: (Delta_k) -> (delta E_k), dense (n*dim, n*dim**2)."""
    udim = round(dim ** 0.5)
    I_sk = _bt.stdmx_to_vec(_np.eye(udim), basis).ravel().real
    T = _np.zeros((n_members * dim, n_members * dim * dim))
    for k in range(n_members):
        for a in range(dim):
            for b in range(dim):
                T[k * dim + a, k * dim * dim + b * dim + a] = I_sk[b]
    return T


def _describe_blind(V, n_members, dim, basis):
    """
    Name each blind direction physically: its effect fraction (the share of its
    norm lying in perturbations that move an effect; 1.0 = purely a frozen
    effect, 0.0 = a frozen member direction, i.e. a rank cap) and its
    distribution over members.
    """
    T = _effect_map(n_members, dim, basis)
    Q = _np.linalg.svd(T, full_matrices=False)[2]
    out = []
    for j in range(V.shape[1]):
        v = V[:, j]
        per_member = [float(_np.linalg.norm(v[k * dim * dim:(k + 1) * dim * dim]))
                      for k in range(n_members)]
        frac = _np.linalg.norm(Q @ v) / max(_np.linalg.norm(v), 1e-300)
        out.append({'effect_fraction': float(frac),
                    'per_member_norm': _np.array(per_member)})
    return out


def _effects_at(inst, basis):
    """The current (dense) effects E_k = member_k^dag(I) of an instrument."""
    dim = next(iter(inst.values())).dim
    udim = round(dim ** 0.5)
    I_sk = _bt.stdmx_to_vec(_np.eye(udim), basis).ravel().real
    return {lbl: _np.asarray(m.to_dense()).T @ I_sk for lbl, m in inst.items()}


def _deep_certificate(mdl, inst_label, basis, base_intervals, n_samples=4,
                      scale=0.25, seed=0, reference_rank=None,
                      rel_tol=JACOBIAN_RANK_RTOL):
    """
    Sample the chart at generic parameter points and measure everything that
    needs a Jacobian or a generic point: chart/effect ranks, blind directions,
    dead parameters, generic member ranks and conditioning, and machine
    verification of the spectral-compression bound.

    Two implementation facts here are load-bearing and NOT optional:

    * Sample at GENERIC theta, never at the seed.  CPTPLND's stochastic block
      uses 'cholesky' parameter mode (rates quadratic in the parameters), so at
      theta = 0 every stochastic derivative is exactly zero -- an at-seed test
      flags every CPTPLND model and no GLND model, for reasons unrelated to
      any pathology.  Rank is lower semicontinuous, so a generic sample attains
      the true image dimension with probability 1.

    * Take the MAX rank over samples, never the pooled union.  Tangent spaces
      at different points of a curved manifold span strictly more than the
      manifold: the measure-and-reset chart is genuinely 12-dimensional and
      pools to 31 of a possible 32.  Extra samples only guard against an
      unlucky draw.
    """
    inst = mdl.instruments[inst_label]
    n = len(inst)
    dim = next(iter(inst.values())).dim
    idx = _np.asarray(inst.gpindices_as_array())
    v0 = mdl.to_vector().copy()
    sig = _adaptive_scale(scale, dim)
    rng = _np.random.default_rng(seed)

    from pygsti.tools.jamiolkowski import jamiolkowski_iso as _jam

    Js = []
    violated = 0.0
    ranks, choi_ranks, conds = {}, {}, {}
    try:
        for _ in range(n_samples):
            v = v0.copy()
            v[idx] = rng.normal(scale=sig, size=idx.size)
            mdl.from_vector(v)
            inst = mdl.instruments[inst_label]
            Js.append(_chart_jacobian(inst))
            for ek, E in _effects_at(inst, basis).items():
                sp = _np.linalg.eigvalsh(_bt.vec_to_stdmx(E, basis, keep_complex=True))
                lo, hi = base_intervals[ek]['spec_interval']
                violated = max(violated, lo - sp.min(), sp.max() - hi)
            for ek, m in inst.items():
                d = _np.asarray(m.to_dense())
                ranks[ek] = max(ranks.get(ek, 0), int(_np.linalg.matrix_rank(d, tol=1e-9)))
                sv = _np.linalg.svd(d, compute_uv=False)
                conds[ek] = max(conds.get(ek, 0.0), sv.min() / max(sv.max(), 1e-300))
                choi = _jam(d, basis, 'std', normalized=True)
                choi_ranks[ek] = max(choi_ranks.get(ek, 0),
                                     int(_np.sum(_np.linalg.eigvalsh(choi) > 1e-9)))
    finally:
        mdl.from_vector(v0)

    mem = [_rank_report(J, rel_tol) for J in Js]
    per_sample = [m[0] for m in mem]
    best = int(_np.argmax(per_sample))
    r_mem, svals_mem, gap_mem = mem[best]

    eff = [_rank_report(_effect_jacobian(J, dim, n, basis), rel_tol) for J in Js]
    r_eff = max(e[0] for e in eff)
    gap_eff = eff[int(_np.argmax([e[0] for e in eff]))][2]

    colnorm = _np.max([_np.linalg.norm(J, axis=0) for J in Js], axis=0)
    dead = int(_np.sum(colnorm <= 1e-10 * max(colnorm.max(), 1e-300)))

    tp_dim = n * dim * dim - dim
    ref = tp_dim if reference_rank is None else reference_rank
    V = _blind_directions(Js[best], n, dim, rel_tol)

    return {
        'member_rank': r_mem, 'member_target': ref, 'member_tp_dim': tp_dim,
        'member_gap': gap_mem, 'member_svals': svals_mem, 'per_sample_rank': per_sample,
        'effect_rank': r_eff, 'effect_target': (n - 1) * dim, 'effect_gap': gap_eff,
        'dead_params': dead, 'n_blind': V.shape[1],
        'blind': _describe_blind(V, n, dim, basis),
        'generic_member_ranks': ranks, 'generic_choi_ranks': choi_ranks,
        'reachable_cond': conds, 'compression_violation': float(violated),
    }


def _scratch_host(effect_superkets, gate_superops, basis,
                  gate_parameterization, povm_errormap):
    """
    A throwaway ExplicitOpModel hosting one chart built on the given static
    base, so the Jacobian certificate can run on instruments that do not live
    in a model.  The instrument label must name every state-space label, else
    the model would try to auto-embed the instrument (unsupported).
    """
    from pygsti.models.explicitmodel import ExplicitOpModel as _ExplicitOpModel
    from pygsti.baseobjs import statespace as _statespace
    basis = _Basis.cast(basis)
    ss = _statespace.default_space_for_dim(basis.dim)
    lbl = ('Idiagnose',) + tuple(ss.sole_tensor_product_block_labels)
    mdl = _ExplicitOpModel(ss, basis)
    with _warnings.catch_warnings():
        # deliberately hosting a possibly-deficient base to measure it: the
        # construction-time singular-base warning would duplicate our report
        _warnings.filterwarnings('ignore', message='.*post-measurement gate.*singular.*')
        mdl[lbl] = _Instrument(_parameterized_instrument(
            basis, effect_superkets, gate_superops, gate_parameterization, povm_errormap))
    mdl.to_vector()   # force parameter allocation before anything reads gpindices
    return mdl, lbl


# --------------------------------------------------------------------------- #
# Flag assembly and the top-level entry points
# --------------------------------------------------------------------------- #

_SEEDING_MOD = "pygsti.modelmembers.instruments.seeding"


def _assemble_flags(base_ranks, span, extent, deep_res, dim,
                    boundary_tol=1e-9, extent_tol=1e-9):
    """The flags, in the order a user should act on them."""
    flags = []

    capped = {k: r for k, r in base_ranks.items() if r['frozen'] and r['rank'] < r['full']}
    if capped:
        worst = min(r['rank'] for r in capped.values())
        full = next(iter(capped.values()))['full']
        flags.append(InstrumentSeedFlag(
            f"RANK-CAP({worst}/{full})",
            f"static base gate(s) {sorted(map(str, capped))} are rank deficient; every "
            "reachable member is capped at the base's rank because exp(L) is invertible. "
            "If your instrument's members really are rank-deficient (e.g. measure-and-"
            "reset), this flag is informational.",
            f"{_SEEDING_MOD}.full_rank_base_blend (c ~ 0.9) or patch_instrument_seed"))

    if span['deficient']:
        flags.append(InstrumentSeedFlag(
            f"SPAN-DEFICIENT({span['m']}/{span['target']})",
            f"base effects span m = {span['m']} dimensions beyond the identity but the "
            f"chart needs m = {span['target']}; the effect tuple is confined to "
            f"{span['reachable_effect_dim']} of {span['full_effect_dim']} dimensions. "
            "Proportional-to-identity base effects (m = 0) and n >= 3 commuting base "
            "effects are the two known causes.",
            f"{_SEEDING_MOD}.displace_base_effects (delta ~ 0.1-0.2)"))

    frozen_effects = [k for k, row in extent.items() if row['extent'] <= extent_tol]
    if frozen_effects:
        flags.append(InstrumentSeedFlag(
            "ZERO-EXTENT",
            f"effect(s) {sorted(map(str, frozen_effects))} have a degenerate base "
            "spectrum, so the shared error map's unital dual can never move them at all.",
            f"{_SEEDING_MOD}.displace_base_effects"))

    on_boundary = [k for k, row in extent.items() if row['boundary_margin'] <= boundary_tol]
    if on_boundary:
        flags.append(InstrumentSeedFlag(
            "CP-BOUNDARY",
            f"effect(s) {sorted(map(str, on_boundary))} have base eigenvalues exactly at "
            "0 or 1; the fit trajectory can reach the CP boundary and stick there "
            "(a measured mid-fit trap), and the member map is non-differentiable "
            "at that point.",
            f"{_SEEDING_MOD}.interior_effect_offset (eps ~ 0.01)"))

    if deep_res is not None:
        ref, r_mem = deep_res['member_target'], deep_res['member_rank']
        if r_mem < ref:
            fracs = [b['effect_fraction'] for b in deep_res['blind']]
            frac_note = (f" (blind-direction effect fractions "
                         f"{min(fracs):.2f}-{max(fracs):.2f}; 1.0 = frozen effect, "
                         f"0.0 = frozen member direction)") if fracs else ""
            flags.append(InstrumentSeedFlag(
                f"BLIND({ref - r_mem})",
                f"the chart Jacobian's generic rank is {r_mem} against a target of "
                f"{ref}: the chart's image is measure-zero in the physical instruments "
                f"and no seeding or restarting inside it can reach a generic "
                f"target{frac_note}.", None))
        r_eff, eff_dim = deep_res['effect_rank'], deep_res['effect_target']
        if r_eff < eff_dim:
            flags.append(InstrumentSeedFlag(
                f"EFFECTS-FROZEN({eff_dim - r_eff}/{eff_dim})",
                "the reachable effect tuple is rank deficient at generic parameters "
                "(the sampled counterpart of the span law).",
                f"{_SEEDING_MOD}.displace_base_effects"))
        if not capped:
            cond = deep_res['reachable_cond']
            ill = {k: c for k, c in cond.items() if c < ILL_CONDITIONED_TOL}
            if ill:
                worst = min(ill.values())
                flags.append(InstrumentSeedFlag(
                    f"ILL-CONDITIONED({worst:.1e})",
                    "the base is full rank but so poorly conditioned that fits are "
                    "expected to stall (a calibrated warning, not a certificate: "
                    "there is no principled threshold).",
                    f"{_SEEDING_MOD}.full_rank_base_blend with c near 1"))
        if deep_res['member_gap'] < AMBIGUOUS_GAP or deep_res['effect_gap'] < AMBIGUOUS_GAP:
            flags.append(InstrumentSeedFlag(
                "AMBIGUOUS-RANK",
                f"the Jacobian rank decision's singular-value gap is small (member "
                f"{deep_res['member_gap']:.1e}, effect {deep_res['effect_gap']:.1e}); "
                "the sampled rank test is not decisive here (observed at d = 4) -- "
                "rely on the O(1) structural checks, which stay exact.", None))
        if deep_res['compression_violation'] > 1e-9:
            flags.append(InstrumentSeedFlag(
                "EXTENT-UNCERTIFIED",
                f"the spectral-compression bound was violated by "
                f"{deep_res['compression_violation']:.2e} at sampled parameters; this "
                "chart is not of the shared-ComposedPOVM form the extent certificate "
                "assumes, so the reported spectral boxes are not certified.", None))
    return flags


def _emit_warning(report):
    lbl = f" {report.label!r}" if report.label is not None else ""
    _warnings.warn(
        f"Instrument{lbl} static-base diagnostics raised "
        f"{len(report.flags)} flag(s): {report.verdict}\n" +
        "\n".join(f"  - {f}" for f in report.flags) +
        "\n(A flag is not always a bug: if the flagged structure is intended, "
        "ignore this warning.  Repairs are in "
        "pygsti.modelmembers.instruments.seeding.)")


def diagnose_instrument(instrument, basis, deep=True, warn=True,
                        gate_parameterization='CPTPLND', povm_errormap='CPTPLND',
                        n_samples=4, scale=0.25, seed=0, reference_rank=None,
                        label=None):
    """
    Run every static-base check on one instrument and return an
    :class:`InstrumentSeedReport`.

    All checks are data-free.  The three O(1) structural checks
    (:func:`static_base_ranks`, :func:`base_effect_span`,
    :func:`base_effect_extent`) always run and are exact at any dimension.
    When ``deep`` is True (the default) the sampled Jacobian certificate also
    runs, comparing the chart's generic rank against the TP-instrument tangent
    dimension -- it is the net for failure species the structural checks do
    not know about, and reports itself as AMBIGUOUS-RANK when its decision is
    not clean.

    For the deep check, an instrument that does not already carry a
    parameterized chart (a dense :class:`Instrument`, a :class:`TPInstrument`,
    or a ``{label: superop}`` dict) is hosted in a scratch chart built with
    ``gate_parameterization`` / ``povm_errormap`` -- i.e. the chart that
    :meth:`Instrument.from_cptr_superops` would build.  If you built your
    chart with a different parameterization, pass the same strings here.
    For an instrument already inside a model, prefer
    :func:`diagnose_model_instruments`, which certifies the actual in-model
    chart.

    Parameters
    ----------
    instrument : Instrument or TPInstrument or dict
        The instrument (or ``{label: dense superop}`` members) to diagnose.

    basis : BasisLike
        The basis for dense representations.

    deep : bool, optional
        Whether to run the sampled Jacobian certificate (default True).

    warn : bool, optional
        Emit a warning (with named repairs) when any flag is raised.
        Diagnostics never raise on a flag.

    gate_parameterization, povm_errormap : str, optional
        The chart to certify when one has to be built (see above).

    n_samples, scale, seed : optional
        Sampling controls for the deep check.  ``scale`` is normalized by
        parameter count internally so d = 2 and d = 4 sample comparably.

    reference_rank : int, optional
        Generic member-chart rank of the same parameterization on a known-good
        base, if you have one; the default target is the TP tangent dimension
        ``n*d**4 - d**2``, which is right for any generator class that spans
        all of TP (CPTPLND, GLND) and over-strict otherwise (e.g. H+S).

    label : object, optional
        A label for the report (cosmetic).

    Returns
    -------
    InstrumentSeedReport
    """
    basis, effects, gates, info = instrument_static_base(instrument, basis)
    base_ranks = _ranks_from_base(basis, gates, info)
    span = _span_from_base(basis, effects)
    extent = _extent_from_base(basis, effects)
    details = {'base_ranks': base_ranks, 'effect_span': span,
               'effect_extent': extent, 'static_base_exact': info['exact']}

    deep_res = None
    if deep:
        if not _is_lindblad_type(gate_parameterization):
            raise ValueError(
                f"The Jacobian certificate needs a Lindblad gate_parameterization to "
                f"build the chart under test, not {gate_parameterization!r}. Pass "
                "deep=False to run only the structural checks.")
        mdl, host_lbl = _scratch_host(effects, gates, basis,
                                      gate_parameterization, povm_errormap)
        deep_res = _deep_certificate(mdl, host_lbl, basis, extent, n_samples,
                                     scale, seed, reference_rank)
        details.update(deep_res)

    dim = basis.dim
    flags = _assemble_flags(base_ranks, span, extent, deep_res, dim)
    report = InstrumentSeedReport(label, flags, details, deep_res is not None)
    if warn and flags:
        _emit_warning(report)
    return report


def diagnose_model_instruments(model, inst_labels=None, deep=True, warn=True,
                               n_samples=4, scale=0.25, seed=0,
                               reference_rank=None):
    """
    Run :func:`diagnose_instrument` on (some of) a model's instruments,
    certifying the *actual in-model charts*: the deep check samples the
    model's own parameter vector, so whatever parameterization the model's
    instruments really carry is what gets certified.

    Parameters
    ----------
    model : ExplicitOpModel
        The model whose instruments to diagnose.

    inst_labels : iterable, optional
        Which instruments (default: all of ``model.instruments``).

    deep, warn, n_samples, scale, seed, reference_rank : optional
        As in :func:`diagnose_instrument`.

    Returns
    -------
    dict[label, InstrumentSeedReport]
    """
    basis = model.basis
    if inst_labels is None:
        inst_labels = list(model.instruments.keys())
    model.to_vector()   # force parameter allocation before reading gpindices

    reports = {}
    for lbl in inst_labels:
        inst = model.instruments[lbl]
        bs, effects, gates, info = instrument_static_base(inst, basis)
        base_ranks = _ranks_from_base(bs, gates, info)
        span = _span_from_base(bs, effects)
        extent = _extent_from_base(bs, effects)
        details = {'base_ranks': base_ranks, 'effect_span': span,
                   'effect_extent': extent, 'static_base_exact': info['exact']}

        deep_res = None
        if deep:
            deep_res = _deep_certificate(model, lbl, basis, extent, n_samples,
                                         scale, seed, reference_rank)
            details.update(deep_res)

        flags = _assemble_flags(base_ranks, span, extent, deep_res, basis.dim)
        report = InstrumentSeedReport(lbl, flags, details, deep_res is not None)
        if warn and flags:
            _emit_warning(report)
        reports[lbl] = report
    return reports

"""
Interior initialization of the Lindblad error generators in a model.

At zero noise, CP-constrained Lindblad coefficient blocks sit on the boundary of their
parameterizations: squared rates and Cholesky factors are zero, so the dissipative coefficients
have zero derivative and an exact-Jacobian optimizer cannot move them. Instrument effects built
by :meth:`~pygsti.modelmembers.instruments.Instrument.from_effects` also sit on the spectral
boundary, where E½ has no derivative. :func:`prepare_lindblad_seed` moves every such block a
bounded distance into its interior, changing parameter values only: the static parts of the
model, and the ability to return to zero noise, are unchanged.
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

import dataclasses as _dataclasses

import numpy as _np

from pygsti.modelmembers.operations.cptrop import RootConjOperator as _RootConjOperator
from pygsti.modelmembers.operations.lindbladerrorgen import LindbladErrorgen as _LindbladErrorgen
from pygsti.tools import basistools as _bt
from pygsti.tools import lindbladtools as _lt

#: Dissipative block parameterizations whose coefficients have zero derivative at zero noise.
_DEGENERATE_PARAM_MODES = ('cholesky', 'depol')


@_dataclasses.dataclass(frozen=True)
class PerturbedErrorgen:
    """How :func:`prepare_lindblad_seed` changed one Lindblad error generator."""

    paths: tuple[str, ...]
    """Every path from a top-level model member to this error generator."""

    block_modes: tuple[tuple[str, str], ...]
    """`(block_type, param_mode)` of each coefficient block."""

    diamond_bound: float
    """Triangle-inequality bound on the diamond norm of the added generator."""

    min_stochastic_rate: float
    """The smallest S coefficient after preparation."""

    min_dissipative_eigenvalue_before: float
    """The smallest eigenvalue of the dissipative coefficient matrix before preparation."""

    min_dissipative_eigenvalue_after: float
    """The smallest eigenvalue of the dissipative coefficient matrix after preparation."""


@_dataclasses.dataclass(frozen=True)
class SkippedErrorgen:
    """A Lindblad error generator that :func:`prepare_lindblad_seed` left unchanged, and why."""

    paths: tuple[str, ...]
    reason: str


@_dataclasses.dataclass(frozen=True)
class LindbladSeedReport:
    """What :func:`prepare_lindblad_seed` did. Pass `seed` back to reproduce the same vector."""

    diamond_budget: float
    hs_split: tuple[float, float]
    floor_fraction: float
    seed: int
    perturbed: tuple[PerturbedErrorgen, ...]
    skipped: tuple[SkippedErrorgen, ...]
    effect_min_eigenvalues: dict[str, float]
    """For each nonzero effect inside a :class:`RootConjOperator`, keyed by path, its smallest eigenvalue
    after preparation."""


def prepare_lindblad_seed(model, diamond_budget: float = 1e-3, *, hs_split: tuple[float, float] = (0.5, 0.5),
                          floor_fraction: float = 0.5, seed: int | None = None
                          ) -> tuple[_np.ndarray, LindbladSeedReport]:
    """
    Parameters that move a model's CP-constrained Lindblad coefficient blocks into their interiors.

    Every distinct :class:`LindbladErrorgen` reachable from the model's members is considered once,
    so an error map shared by several members (such as the effect error map of an instrument built
    by :meth:`Instrument.from_effects`) is perturbed once. Error generators with a dissipative block
    in a parameterization whose coefficients have zero derivative at zero noise ('cholesky' or
    'depol', as in CPTPLND, H+S and S) are perturbed; others, such as GLND, have no such degeneracy
    and are skipped.

    Each perturbed error generator gains Hamiltonian rates h_a and stochastic rates s_a > 0, added
    to its existing coefficients. With F_a the basis element of rate a, they are scaled so that

        sum_a 2 |h_a| ‖F_a‖ + sum_a 2 s_a ‖F_a‖² = diamond_budget

    (operator norms). The left side bounds the diamond norm of the added generator, and so, by
    Duhamel's formula, the diamond-norm change in the error generator's exponential.
    `hs_split` divides this budget between H and S; an error generator without a Hamiltonian block
    gives all of it to S. Of the S share, `floor_fraction` is spread evenly over every S direction,
    guaranteeing each a positive minimum, and the rest is drawn by
    :func:`~pygsti.tools.lindbladtools.random_cptp_errorgen_rates` (as are the H rates). A 'depol'
    block, having a single rate, gets its S share as one common rate.

    The floor also bounds instrument effects away from the spectral boundary. In the unnormalized
    Pauli basis, the effects of a shared error map with stochastic rates s_a satisfy
    λ_min(E) ≥ exp(-Σ s_a) · min(s_a) · d · Tr(E_static) on a d-dimensional Hilbert space.

    Parameters
    ----------
    model : OpModel
        The model to prepare. Not modified.

    diamond_budget : float, optional
        The diamond-norm bound, per perturbed error generator, on the added generator. In
        one-qubit instrument fits the final fit was insensitive to it between 1e-3 and 1e-2,
        except that larger perturbations occasionally led to a worse local optimum, and slightly
        worse at 5e-2.

    hs_split : tuple of float, optional
        The fractions of `diamond_budget` given to H and S rates. They must sum to 1, and the S
        fraction must be positive. A nonzero H share also breaks symmetries that stochastic rates
        alone leave in place: with S rates only, a Z rotation of a Z-diagonal state preparation,
        for example, still has zero first derivative.

    floor_fraction : float, optional
        The fraction of the S share spread evenly over every S direction, in [0, 1]. It had no
        visible effect on one-qubit fits; it guarantees the effect-eigenvalue bound above.

    seed : int, optional
        Seed for the random rates. None draws fresh entropy, which is recorded in the report.

    Returns
    -------
    vector : numpy.ndarray
        The prepared parameter vector; apply it with `model.from_vector(vector)`.

    report : LindbladSeedReport
    """
    if not (_np.isfinite(diamond_budget) and diamond_budget > 0):
        raise ValueError("`diamond_budget` must be positive and finite, not %s." % diamond_budget)
    hs_split = tuple(float(x) for x in hs_split)
    if len(hs_split) != 2 or min(hs_split) < 0 or abs(sum(hs_split) - 1) > 1e-12:
        raise ValueError("`hs_split` must be two nonnegative fractions summing to 1, not %s." % (hs_split,))
    if hs_split[1] == 0:
        raise ValueError("`hs_split` must give S a positive share; S rates are what move a seed into the interior.")
    if not 0 <= floor_fraction <= 1:
        raise ValueError("`floor_fraction` must be in [0, 1], not %s." % floor_fraction)
    if getattr(model, 'param_interposer', None) is not None:
        raise NotImplementedError("prepare_lindblad_seed does not support models with a parameter interposer.")

    if seed is None:
        seed = int(_np.random.SeedSequence().entropy)
    rng = _np.random.default_rng(seed)

    work = model.copy()
    work._clean_paramvec()  # so that every member's gpindices index the model's parameter vector
    errorgens, rootconjs = _collect_members(work)

    candidates, skipped = [], []
    for errgen, paths in errorgens:
        reason = _skip_reason(errgen)
        if reason is None:
            candidates.append((errgen, paths))
        else:
            skipped.append(SkippedErrorgen(tuple(paths), reason))
    _check_disjoint_parameters(candidates)

    perturbed = []
    for errgen, paths in candidates:
        before = _min_dissipative_eigenvalue(errgen)
        added, bound = _interior_rates(errgen, diamond_budget, hs_split, floor_fraction, rng)
        errgen.set_coefficients(added, action='add', truncate=False)
        perturbed.append((errgen, paths, before, bound))

    vector = work.to_vector().copy()  # collects the edited coefficients and refreshes every member

    perturbed_report = []
    for errgen, paths, before, bound in perturbed:
        coeffs = errgen.coefficients(label_type='local')
        s_rates = [v.real for k, v in coeffs.items() if k.errorgen_type == 'S']
        perturbed_report.append(PerturbedErrorgen(
            tuple(paths), tuple((b._block_type, b._param_mode) for b in errgen.coefficient_blocks), bound,
            min(s_rates), before, _min_dissipative_eigenvalue(errgen)))

    effect_min_eigenvalues = {}
    for rcop, paths in rootconjs:
        effect = rcop.submembers()[0]
        evals = _np.linalg.eigvalsh(_bt.vec_to_stdmx(effect.to_dense(), rcop._basis, keep_complex=True))
        if _np.max(_np.abs(evals)) > rcop.DERIV_EIGTOL:
            effect_min_eigenvalues[paths[0]] = float(evals.min())

    report = LindbladSeedReport(float(diamond_budget), hs_split, float(floor_fraction), seed,
                                tuple(perturbed_report), tuple(skipped), effect_min_eigenvalues)
    return vector, report


def _collect_members(model):
    """Distinct LindbladErrorgens and RootConjOperators reachable from `model`, each with every path to it."""
    found = {}  # id -> (obj, paths), in first-visit order

    def walk(obj, path, on_path):
        if isinstance(obj, (_LindbladErrorgen, _RootConjOperator)):
            found.setdefault(id(obj), (obj, []))[1].append(' / '.join(path))
        if id(obj) in on_path:
            return
        names = {id(v): str(k) for k, v in obj.items()} if isinstance(obj, dict) else {}
        for sub in obj.submembers():
            walk(sub, path + [names.get(id(sub), type(sub).__name__)], on_path | {id(obj)})

    for lbl, obj in model._iter_parameterized_objs():
        walk(obj, [str(lbl)], frozenset())
    errorgens = [(o, p) for o, p in found.values() if isinstance(o, _LindbladErrorgen)]
    rootconjs = [(o, p) for o, p in found.values() if isinstance(o, _RootConjOperator)]
    return errorgens, rootconjs


def _skip_reason(errgen):
    if errgen.num_params == 0:
        return "it has no parameters"
    if not any(_is_degenerate(b) for b in errgen.coefficient_blocks):
        modes = ', '.join("%s/%s" % (b._block_type, b._param_mode) for b in errgen.coefficient_blocks)
        return "no dissipative block has zero derivative at zero noise (blocks: %s)" % modes
    return None


def _is_degenerate(block):
    return block._block_type in ('other', 'other_diagonal') and block._param_mode in _DEGENERATE_PARAM_MODES


def _check_disjoint_parameters(candidates):
    owner = {}
    for errgen, paths in candidates:
        for i in errgen.gpindices_as_array():
            if i in owner and owner[i] is not errgen:
                other_paths = next(p for e, p in candidates if e is owner[i])
                raise ValueError("Distinct Lindblad error generators at %s and %s share model parameter %d, so "
                                 "perturbing them independently is ill-defined." % (other_paths[0], paths[0], i))
            owner[i] = errgen


def _interior_rates(errgen, diamond_budget, hs_split, floor_fraction, rng):
    """The H and S rates to add to `errgen`, and the diamond-norm bound they achieve."""
    ham_blocks = [b for b in errgen.coefficient_blocks if b._block_type == 'ham' and b._param_mode != 'static']
    diss_blocks = [b for b in errgen.coefficient_blocks if _is_degenerate(b)]
    H_lbls = [k for b in ham_blocks for k in b.elementary_errorgens]
    S_lbls = [k for b in diss_blocks for k in b.elementary_errorgens if k.errorgen_type == 'S']
    H_share, S_share = (hs_split[0] * diamond_budget, hs_split[1] * diamond_budget) if H_lbls \
        else (0.0, diamond_budget)

    def op_norms(blocks, lbls):
        block_of = {k: b for b in blocks for k in b.elementary_errorgens}
        norms = []
        for k in lbls:
            basis = block_of[k]._basis
            el = basis.elements[list(basis.labels).index(k.basis_element_labels[0])]
            norms.append(_np.linalg.norm(el, 2))
        return _np.array(norms)

    H_weights = 2 * op_norms(ham_blocks, H_lbls)
    S_weights = 2 * op_norms(diss_blocks, S_lbls)**2

    # Random directions from the existing sampler, on the error generator's own basis and labels.
    sample_H = bool(H_lbls) and H_share > 0
    basis = diss_blocks[0]._basis
    sampled = _lt.random_cptp_errorgen_rates(errgen.state_space, errorgen_types=('H', 'S') if sample_H else ('S',),
                                             elementary_errorgen_basis=basis, label_type='local', seed=rng)
    missing = [k for k in (H_lbls if sample_H else []) + S_lbls if k not in sampled]
    if missing:
        raise ValueError("random_cptp_errorgen_rates on basis %s did not produce rates for %s; the error generator's "
                         "blocks must share one identity-first basis." % (basis.name, missing))
    added = {}
    if sample_H:
        h = _np.array([sampled[k] for k in H_lbls])
        h *= H_share / _np.dot(H_weights, _np.abs(h))
        added.update(zip(H_lbls, h))

    if any(b._param_mode == 'depol' for b in diss_blocks):
        s = _np.full(len(S_lbls), S_share / S_weights.sum())
    else:
        floor = floor_fraction * S_share / len(S_lbls) / S_weights
        s_rand = _np.array([sampled[k] for k in S_lbls])
        s = floor + s_rand * ((1 - floor_fraction) * S_share / _np.dot(S_weights, s_rand))
    added.update(zip(S_lbls, s))

    bound = float(sum(H_weights[i] * abs(added[k]) for i, k in enumerate(H_lbls) if k in added)
                  + _np.dot(S_weights, s))
    return added, bound


def _min_dissipative_eigenvalue(errgen):
    mins = []
    for b in errgen.coefficient_blocks:
        if b._block_type == 'other':
            mins.append(_np.linalg.eigvalsh(b.block_data).min())
        elif b._block_type == 'other_diagonal':
            mins.append(_np.real(b.block_data).min())
    return float(min(mins))

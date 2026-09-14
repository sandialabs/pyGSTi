"""
First-order linear inversion of character gate set tomography (cGST) decay data
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings

import numpy as _np
from scipy.optimize import least_squares as _least_squares

from pygsti.tools import chartools as _ct

# cGST's germ-decay fits produce, for each (germ, irrep) pair, one or two real
# numbers: the log of the fitted eigenvalue's magnitude (a stochastic decay
# rate per germ repetition), the fitted phase (a coherent angle error per germ
# repetition) and -- for the trivial irrep -- an "active" (amplitude-damping
# type) error read off of the decay's asymptote.  Each of these is a smooth,
# gauge-invariant function of the underlying gate set, and to first order in
# the error rates it is a LINEAR function of the gates' elementary error
# generator coefficients.  This module
#
#   1. enumerates those observables for a given design (`observable_labels`),
#   2. reads them off of fitted data (`observables_from_results`) or predicts
#      them for a given model (`simulate_observables`),
#   3. enumerates the elementary error generator coefficients of the target
#      model's gates (`errorgen_parameter_labels`) and can build a model from
#      a coefficient vector (`model_from_errorgen_coefficients`),
#   4. builds the Jacobian of (1) with respect to (3) by finite differences
#      around the ideal model (`first_order_design_matrix`), and
#   5. solves the resulting linear system (`invert_first_order`).
#
# Everything the observables can see is gauge invariant, so the design matrix
# is necessarily rank deficient: its null space contains the gauge directions
# plus any error generator the germ set does not amplify.  The minimum-norm
# least-squares solution returned by `invert_first_order` therefore lives in a
# particular gauge (the one minimizing the total error generator norm); gauge
# fixing is a downstream step (see :mod:`pygsti.algorithms.cgstgauge`).
#
# ---------------------------------------------------------------------------
# NOTE on the trivial-irrep asymptote.  The raw fitted asymptote `C` is NOT a
# differentiable function of the model at the ideal point: for a two-dimensional
# trivial block the signal is `z(k) = C + (B - C) lam**k` with
# `C = 1/2 - a/(2(1 - lam))`, where both the active error `a` and the decay
# deficit `1 - lam` are first order in the error rates.  `C` is therefore
# homogeneous of degree ZERO in the perturbation -- it has a direction-dependent
# limit at the ideal point and its finite-difference derivative diverges like
# 1/step.  What IS linear is the product
#
#     active := (1 - lam) * (B - C)   [ = (1 - lam)/2 + a/2 for the standard-gauge
#                                       S-gate channel, where B = 1 ]
#
# so that is the quantity this module uses (quantity name `'active'`).  It
# carries exactly the same first-order information as `C` does given `lam`,
# and it degrades gracefully to zero when the trivial decay has no amplitude.
#
# NOTE on trivial-irrep decay fits.  Near the ideal point a trivial block's
# decay rate `lam` is within `O(error rate)` of 1, so the signal is nearly
# affine in the depth and the `(lam, C)` pair is nearly unidentifiable.  Two
# failure modes follow, and both are handled here rather than in
# `pygsti.algorithms.cgstfit` (whose seeds and tolerances are tuned for
# experimental data): an *exactly* affine signal, where the answer is
# "no decay" (see `DEGENERATE_LINEAR_TOL`), and a merely nearly-affine one,
# where the fit must be re-run globally and to tight tolerances to pick the
# right branch and the right rate (see `_refit_trivial_decay`).
# ---------------------------------------------------------------------------


def _child_quantities(design, target_model, name, warn=True):
    """
    The observable quantity names contributed by one cGST germ-decay child design.

    Returns a (possibly empty) tuple of strings drawn from `'log_magnitude'`,
    `'phase'` and `'active'`; see :func:`observable_labels`.
    """
    child = design[name]
    germ_superop = target_model.sim.product(child.germ)
    mults = _ct.germ_irrep_multiplicities(germ_superop, child.group_order)
    mult = mults.get(child.irrep_index, 0)

    if child.irrep_index != 0:
        if mult == 1:
            return ('log_magnitude', 'phase')
        if warn:
            _warnings.warn(("cGST inversion is skipping germ-decay experiment '%s': its "
                            "nontrivial irrep %d has multiplicity %d (only multiplicity-one "
                            "nontrivial irreps are supported).")
                           % (name, child.irrep_index, mult))
        return ()

    # trivial irrep
    if mult == 2:
        return ('log_magnitude', 'active')
    if mult == 1:
        return ()  # only the trace-preserving fixed point lives here: no information
    if warn:
        _warnings.warn(("cGST inversion is skipping germ-decay experiment '%s': the trivial "
                        "irrep has multiplicity %d (only multiplicity 1 or 2 is supported).")
                       % (name, mult))
    return ()


def observable_labels(design, target_model):
    """
    The ordered list of first-order cGST observables a design provides.

    Each germ-decay sub-experiment of `design` contributes zero, one or two
    real observables depending on the multiplicity, in the *ideal* germ
    superoperator, of the irrep that sub-experiment targets:

    * nontrivial irrep with multiplicity 1: `'log_magnitude'` and `'phase'`
      (the log-magnitude and phase of the fitted germ eigenvalue deviation);
    * trivial irrep with multiplicity 2: `'log_magnitude'` and `'active'`
      (the fitted decay rate and the linearizable asymptote combination
      `(1 - lam) * (B - C)`; see this module's source comments for why the raw
      asymptote `C` cannot be used);
    * trivial irrep with multiplicity 1: nothing (that block is only the
      trace-preserving fixed point, which carries no information);
    * anything else: nothing, plus a :class:`UserWarning`.

    Parameters
    ----------
    design : CharacterGSTDesign
        A cGST design whose children are
        :class:`~pygsti.protocols.cgst.CharacterGSTGermDesign` objects.

    target_model : Model
        The ideal model, used to compute the germs' irrep multiplicities.  It
        must support `model.sim.product` (e.g. a matrix forward simulator).

    Returns
    -------
    list
        A list of `(child_name, quantity)` tuples of strings.
    """
    labels = []
    for name in design.keys():
        for quantity in _child_quantities(design, target_model, name):
            labels.append((name, quantity))
    return labels


def _fit_stderrs(fit):
    """The best available standard errors of a cGST decay fit (bootstrap if present)."""
    boot = fit.get('bootstrap_stderrs')
    return boot if boot else fit.get('stderrs', {})


#: Relative tolerance for declaring a trivial-irrep decay curve to be an exact
#: straight line in the depth.  A trivial-block signal is
#: `z(k) = C + (B - C) lam**k`; when it is *affine* in `k` the pair `(lam, C)`
#: is unidentifiable (only the slope `(B - C)(1 - lam)` is determined) and the
#: fitted `lam` is arbitrary.  This is not an edge case for finite differencing:
#: perturbing a gate by a purely Hamiltonian or purely *active* ('A'-type)
#: elementary error generator leaves every trivial decay rate exactly 1 while
#: shifting the block's fixed point, which makes the signal exactly affine.  In
#: that situation `'log_magnitude'` is reported as exactly zero ("no measurable
#: decay"), which is the correct first-order answer.  The default is set well
#: above the ~1e-8 relative noise floor of exact (double precision) simulation
#: but far below the curvature of any decay that a real experiment could
#: resolve, so it never fires on shot-noisy data.
DEGENERATE_LINEAR_TOL = 1e-6

#: Absolute size below which a trivial-irrep decay curve carries no signal at all.
#: The straight-line test below is purely *relative*, so a curve that is nothing
#: but round-off (`|z| ~ 1e-16`) looks strongly curved to it and the re-fit then
#: returns an arbitrary decay rate.  That is not hypothetical: whenever a
#: fiducial pair's ideal decaying amplitude cancels the trivial block's constant
#: background, the whole signal vanishes identically for every perturbation that
#: leaves the block's eigenvalue at 1, which is exactly what the columns of
#: :func:`first_order_design_matrix` do.  Character-weighted signals are
#: probability-scale, so anything this small is numerically zero.
DEGENERATE_SIGNAL_TOL = 1e-9

#: How much better (in RMS residual) the full `C + (B - C) lam**k` model must fit
#: a trivial-irrep decay curve than a straight line does before the curvature --
#: and hence the decay rate -- is considered resolved.  The straight line is the
#: `lam -> 1` limit of the full model, so the full model's residual is never
#: larger; when it is not *substantially* smaller, whatever misfit the straight
#: line leaves is noise the exponential cannot explain either (in
#: `'reduced'`-mode designs the Monte-Carlo synthetic projector leaves an
#: `O(error)` ripple, periodic in the depth modulo the group order, that is not
#: of exponential form), and the fitted `1 - lam` is then determined by that
#: noise rather than by a decay.  Such a curve is treated exactly like an exact
#: straight line: "no measurable decay".
DEGENERATE_RESIDUAL_RATIO = 2.0

#: Smallest trivial-block decay rate the re-fit of :func:`_refit_trivial_decay`
#: will accept (it is also the smallest value on :data:`_TRIVIAL_LAMBDA_GRID`).
#: Below it the fitted curve is a "spike" -- the first data point is matched by
#: `B` and every other one by the asymptote `C` -- which is not a decay at all
#: but the fitter's favourite way of absorbing a depth-dependent ripple in
#: nearly-affine data.
TRIVIAL_LAMBDA_FLOOR = 0.05


def _straight_line_rms(depths, z):
    """RMS residual of the least-squares straight line through `(depths, z)`."""
    design_mx = _np.column_stack([_np.ones(len(depths)), depths])
    coeffs = _np.linalg.lstsq(design_mx, z, rcond=None)[0]
    residual = z - design_mx @ coeffs
    return float(_np.sqrt(_np.mean(residual ** 2)))


def _trivial_decay_is_degenerate(res, tol=DEGENERATE_LINEAR_TOL,
                                 signal_tol=DEGENERATE_SIGNAL_TOL,
                                 residual_ratio=DEGENERATE_RESIDUAL_RATIO, refit_cache=None,
                                 cache_key=None):
    """
    Whether a trivial-irrep decay curve carries no resolvable decay rate.

    True when the curve is (numerically) an exact straight line in the depth,
    when it carries no signal at all, or when the full exponential model fits it
    no better than :data:`DEGENERATE_RESIDUAL_RATIO` times a straight line's RMS
    residual -- see the notes on those module constants.
    """
    depths = _np.asarray(res.depths, dtype='d')
    z = _np.real(res.signal)
    if len(depths) < 3:
        _warnings.warn("A trivial-irrep cGST decay with fewer than 3 depths cannot resolve a "
                       "decay rate from its asymptote; treating it as 'no measurable decay'.")
        return True
    peak = float(_np.max(_np.abs(z)))
    if peak <= signal_tol:
        return True  # no signal whatsoever: "no measurable decay"
    rms_line = _straight_line_rms(depths, z)
    if rms_line <= tol * max(peak, 1e-12):
        return True  # an exact straight line
    # scale-aware part: is the curvature resolved above whatever the exponential
    # model itself cannot fit?
    refit = _refit_trivial_decay(res, refit_cache, cache_key)
    if refit is None:
        return False
    lam, b, c = refit
    with _np.errstate(over='ignore'):
        rms_full = float(_np.sqrt(_np.mean(((b - c) * lam ** depths + c - z) ** 2)))
    return bool(rms_line <= residual_ratio * max(rms_full, 0.0))


#: Grid of candidate trivial-block decay rates for :func:`_refit_trivial_decay`,
#: covering both the decaying (`lam < 1`) and the growing (`lam > 1`) branches
#: densely near the ideal value `lam = 1`.
_TRIVIAL_LAMBDA_GRID = _np.concatenate([
    1.0 - _np.logspace(-10, _np.log10(1.0 - TRIVIAL_LAMBDA_FLOOR), 100),
    1.0 + _np.logspace(-10, -1, 50),
    [1.0]])


def _profiled_trivial_costs(lambdas, depths, zs):
    """
    Least-squares cost of `z(k) = C + (B - C) lam**k` at each `lam`, with `(B, C)` profiled out.

    For a fixed `lam` the model is linear in the amplitude and the asymptote, so
    the optimal pair is available in closed form; this returns the resulting
    residual sum of squares for every `lam` at once.
    """
    with _np.errstate(over='ignore', invalid='ignore'):
        basis = _np.asarray(lambdas)[:, None] ** depths[None, :]   # (nlam, ndepths)
    s11 = _np.sum(basis * basis, axis=1)
    s12 = _np.sum(basis, axis=1)
    s22 = float(len(depths))
    t1 = basis @ zs
    t2 = float(_np.sum(zs))
    det = s11 * s22 - s12 ** 2
    with _np.errstate(divide='ignore', invalid='ignore'):
        amp = (t1 * s22 - s12 * t2) / det
        asym = (s11 * t2 - s12 * t1) / det
    # Evaluate the residual explicitly rather than via `z.z - amp*t1 - asym*t2`:
    # near `lam = 1` the two terms agree to ~15 digits and the shortcut loses
    # every significant figure of the cost (it can even come out negative).
    with _np.errstate(over='ignore', invalid='ignore'):
        residual = asym[:, None] + amp[:, None] * basis - zs[None, :]
        cost = _np.sum(residual * residual, axis=1)
    bad = ~(_np.isfinite(cost) & _np.isfinite(amp) & _np.isfinite(asym))
    cost = _np.where(bad, _np.inf, cost)
    return cost, amp, asym


def _refit_trivial_decay(res, refit_cache=None, cache_key=None):
    """
    A globally-seeded, tightly-converged re-fit of a trivial-irrep decay curve.

    Two things go wrong with :func:`~pygsti.algorithms.cgstfit.fit_real_decay`
    when the trivial block's eigenvalue `lam` is very close to 1, which is
    exactly the regime of weak errors and of the finite differences taken by
    :func:`first_order_design_matrix`:

    1. `C + (B - C) lam**k` is then nearly *affine* in `k`, so the cost surface
       has a long flat valley along the `(lam, C)` trade-off and scipy's default
       tolerances (`1e-8`) stop the optimizer with an `O(10%)` error in
       `1 - lam`.
    2. A slowly *growing* signal (`lam > 1`) is nearly degenerate with a slow
       decay toward an asymptote *above* the initial value; only the sign of the
       (tiny) curvature distinguishes them, and a local optimizer seeded on the
       wrong branch stays there -- reporting a decay where the germ actually has
       an amplifying trivial eigenvalue.

    Both are cured by profiling `(B, C)` out analytically, scanning `lam` over
    :data:`_TRIVIAL_LAMBDA_GRID` (which straddles `lam = 1`), and polishing the
    best grid point with tight tolerances.  Unlike `fit_real_decay` the
    amplitude and asymptote are left unbounded here, since near `lam = 1` the
    (poorly determined) asymptote legitimately runs far outside `[-2, 2]` while
    the decay rate itself stays well determined.

    The decay rate, on the other hand, is bounded *below* by
    :data:`TRIVIAL_LAMBDA_FLOOR`, and a solution that ends up pinned there -- or
    one that has decayed away before the second depth, `lam**d_1 < 1e-3` for
    the smallest positive depth `d_1` -- is not a decay at all but a "spike"
    absorbing a ripple in the data (see the module constants), so it is
    treated as a failed re-fit.

    Returns `(lam, B, C)`, or None if the re-fit failed (in which case callers
    fall back on `fit_real_decay`'s own estimate).  Results are memoized in
    `refit_cache` (a dict, if given) under `cache_key`; the results object is
    never modified.
    """
    if refit_cache is not None and cache_key in refit_cache:
        return refit_cache[cache_key]

    est = res.fit['estimates']
    depths = _np.asarray(res.depths, dtype='d')
    zs = _np.real(res.signal)
    positive = depths[depths > 0]
    d1 = float(positive.min()) if len(positive) else 1.0

    def _resid(params):
        lam, b, c = params
        with _np.errstate(over='ignore'):
            return (b - c) * lam ** depths + c - zs

    def _cost(params):
        r = _resid(params)
        return _np.inf if not _np.all(_np.isfinite(r)) else 0.5 * float(r @ r)

    def _plausible(lam):
        return lam > TRIVIAL_LAMBDA_FLOOR * (1 + 1e-6) and lam ** d1 >= 1e-3

    seed_params = (est['lam'], est['B'], est['C'])
    best, best_cost = None, _cost(seed_params)

    grid_cost, amp, asym = _profiled_trivial_costs(_TRIVIAL_LAMBDA_GRID, depths, zs)
    i = int(_np.argmin(grid_cost))
    start = [float(_TRIVIAL_LAMBDA_GRID[i]), float(amp[i] + asym[i]), float(asym[i])]
    seeds = [start]
    if _plausible(seed_params[0]):
        seeds.append(list(seed_params))
    for seed in seeds:
        try:
            # a perfectly affine signal makes the (lam, C) trade-off exactly singular, so
            # scipy's trust-region solve divides by a zero singular value; the step it
            # takes is still fine, so just silence the numpy warnings it raises
            with _np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                result = _least_squares(_resid, seed,
                                        bounds=([TRIVIAL_LAMBDA_FLOOR, -_np.inf, -_np.inf],
                                                [1.1, _np.inf, _np.inf]),
                                        ftol=1e-15, xtol=1e-15, gtol=1e-15)
        except Exception:  # pragma: no cover - defensive; fall back on the original fit
            continue
        if result.cost < best_cost and _plausible(float(result.x[0])):
            best, best_cost = (float(result.x[0]), float(result.x[1]), float(result.x[2])), result.cost

    if refit_cache is not None:
        refit_cache[cache_key] = best
    return best


def _trivial_estimates(res, refit_cache=None, cache_key=None):
    """`(lam, B, C)` for a trivial-irrep decay, refined when possible."""
    est = res.fit['estimates']
    refit = _refit_trivial_decay(res, refit_cache, cache_key)
    return refit if (refit is not None) else (est['lam'], est['B'], est['C'])


def _quantity_value(res, quantity, refit_cache=None, cache_key=None):
    """The value of one observable quantity from a :class:`CharacterDecayResults`."""
    est = res.fit['estimates']
    if quantity == 'log_magnitude':
        mag = res.germ_eigenvalue_magnitude
        if res.irrep_index == 0:
            if _trivial_decay_is_degenerate(res, refit_cache=refit_cache, cache_key=cache_key):
                return 0.0
            lam = _trivial_estimates(res, refit_cache, cache_key)[0]
            if lam > 0 and est['lam'] > 0 and mag > 0:
                # `mag` may have been passed through the full-mode projector
                # inversion; shift it by the refit's (small) correction to the
                # raw decay rate rather than re-deriving it.
                return float(_np.log(mag) + _np.log(lam) - _np.log(est['lam']))
        return _np.log(mag) if mag > 0 else -_np.inf
    if quantity == 'phase':
        return res.germ_eigenvalue_phase
    if quantity == 'active':
        lam, b, c = _trivial_estimates(res, refit_cache, cache_key) if res.irrep_index == 0 \
            else (est['lam'], est['B'], est['C'])
        return float((1.0 - lam) * (b - c))
    raise ValueError("Unknown cGST observable quantity '%s'" % quantity)


def _quantity_stderr(res, quantity, refit_cache=None, cache_key=None):
    """A delta-method standard error for one observable quantity (nan if unavailable)."""
    est = res.fit['estimates']
    stderrs = _fit_stderrs(res.fit)

    def _s(key):
        v = stderrs.get(key)
        return _np.nan if v is None else abs(float(v))

    if quantity == 'log_magnitude':
        mag = res.germ_eigenvalue_magnitude
        return _s('lam') / mag if mag > 0 else _np.nan
    if quantity == 'phase':
        return _s('theta')
    if quantity == 'active':
        # linearize about the same point estimate `_quantity_value` uses
        lam, b, c = _trivial_estimates(res, refit_cache, cache_key) if res.irrep_index == 0 \
            else (est['lam'], est['B'], est['C'])
        amp, decay = b - c, 1.0 - lam
        return _np.sqrt((amp * _s('lam')) ** 2 + (decay * _s('B')) ** 2
                        + (decay * _s('C')) ** 2)
    raise ValueError("Unknown cGST observable quantity '%s'" % quantity)


def observables_from_results(decay_results_by_name, design, target_model):
    """
    Read the first-order cGST observables (and their uncertainties) off of fitted decays.

    Parameters
    ----------
    decay_results_by_name : dict
        Maps each sub-experiment name of `design` to its
        :class:`~pygsti.protocols.cgst.CharacterDecayResults`.

    design : CharacterGSTDesign
        The cGST design the results came from.

    target_model : Model
        The ideal model (see :func:`observable_labels`).

    Returns
    -------
    y : numpy.ndarray
        The observable values, in :func:`observable_labels` order.

    stderr : numpy.ndarray
        The corresponding standard errors, propagated by the delta method
        (`sigma_lambda / lambda` for the log-magnitude, and the usual
        uncorrelated combination of the `lam`, `B` and `C` errors for
        `'active'`).  Entries are
        `numpy.nan` where the fit did not provide an uncertainty.  When the
        data was taken in `'full'` sampling mode the uncertainties are those
        of the *raw* fit, i.e. the (mild) nonlinearity of the Fourier-operator
        eigenvalue inversion is not propagated.
    """
    values, errors, refits = [], [], {}
    for name, quantity in observable_labels(design, target_model):
        res = decay_results_by_name[name]
        values.append(_quantity_value(res, quantity, refits, name))
        errors.append(_quantity_stderr(res, quantity, refits, name))
    return _np.array(values, dtype='d'), _np.array(errors, dtype='d')


class _ExactDataRow(object):
    """A single circuit's exact outcome probabilities, quacking like a `DataSet` row."""
    __slots__ = ('counts', 'total')

    def __init__(self, probabilities):
        self.counts = probabilities
        self.total = float(sum(probabilities.values()))


class _ExactProbabilityData(object):
    """
    A read-only stand-in for a :class:`DataSet` holding a model's exact probabilities.

    pyGSTi's :class:`DataSet` cannot be used to carry noiseless cGST data for
    finite differencing, for two reasons: its repetition counts are stored as
    `float32` (so probabilities are only good to ~1e-7 relative), and any
    negative count is silently zeroed when data is added (so a *non-completely-
    positive* model -- which is exactly what a negative stochastic error
    generator coefficient produces, and which occurs on one side of every
    central difference -- has its out-of-range probabilities clipped).  This
    tiny class supports the only two things
    :meth:`~pygsti.protocols.cgst.CharacterDecay.run` asks of a dataset,
    `ds[circuit].counts` and `ds[circuit].total`, in exact double precision.
    """

    def __init__(self, model, circuits):
        self._probs = model.bulk_probabilities(list(circuits))

    def __getitem__(self, circuit):
        return _ExactDataRow(self._probs[circuit])

    def __contains__(self, circuit):
        return circuit in self._probs

    def __len__(self):
        return len(self._probs)

    def keys(self):
        """The circuits this object holds probabilities for."""
        return self._probs.keys()


def simulate_observables(model, design, target_model, invert_full_mode=True,
                         obs_labels=None):
    """
    Predict the first-order cGST observables of a model, without sampling error.

    The exact outcome probabilities of every circuit of `design` are computed
    with `model`, the :class:`~pygsti.protocols.cgst.CharacterDecay` protocol
    is run on each germ-decay sub-experiment, and the resulting observables are
    returned in :func:`observable_labels` order.  This is the noiseless
    (infinite-sample) limit of simulating data with
    :func:`pygsti.data.simulate_data` and running `CharacterDecay` on it; see
    :class:`_ExactProbabilityData` for why the probabilities are *not* routed
    through a :class:`DataSet`.

    Parameters
    ----------
    model : Model
        The model to simulate.  Should use a matrix forward simulator.

    design : CharacterGSTDesign
        The cGST design.

    target_model : Model
        The ideal model (see :func:`observable_labels`).

    invert_full_mode : bool, optional
        Whether `'full'`-mode fits are inverted into bare germ eigenvalues
        (passed through to :class:`~pygsti.protocols.cgst.CharacterDecay`).

    obs_labels : list, optional
        A precomputed :func:`observable_labels` list (an optimization; it must
        be consistent with `design` and `target_model`).

    Returns
    -------
    numpy.ndarray
        The observable values.
    """
    from pygsti.protocols.cgst import CharacterDecay as _CharacterDecay
    from pygsti.protocols.protocol import ProtocolData as _ProtocolData

    if obs_labels is None:
        obs_labels = observable_labels(design, target_model)
    needed = set(name for name, _ in obs_labels)

    ds = _ExactProbabilityData(model, design.all_circuits_needing_data)
    proto = _CharacterDecay(bootstrap_samples=0, invert_full_mode=invert_full_mode)
    results = {name: proto.run(_ProtocolData(design[name], ds))
               for name in design.keys() if name in needed}

    refits = {}
    return _np.array([_quantity_value(results[name], quantity, refits, name)
                      for name, quantity in obs_labels], dtype='d')


def _glnd_model(target_model):
    """
    A copy of `target_model` with every gate written as `exp(L) * ideal`.

    The gates are :class:`~pygsti.modelmembers.operations.ComposedOp` objects
    pairing the target's (static) gate with an
    :class:`~pygsti.modelmembers.operations.ExpErrorgenOp` holding an
    unconstrained ('GLND') :class:`LindbladErrorgen` -- i.e. all H, S, C and A
    elementary error generators with arbitrary real coefficients.
    """
    mdl = target_model.copy()
    mdl.set_all_parameterizations('GLND')
    mdl.sim = 'matrix'
    return mdl


def errorgen_parameter_labels(target_model):
    """
    Every elementary error generator coefficient of a target model's gates.

    The coefficients are pyGSTi's standard 'GLND' (unconstrained
    Hamiltonian + stochastic + correlation + active) elementary error
    generator coefficients: for each gate `G` the noisy gate is
    `exp(sum_i c_i L_i) G`, where `L_i` is the elementary error generator of
    the label `i` built from *unnormalized* Pauli matrices (see
    :func:`pygsti.tools.lindbladtools.create_elementary_errorgen`).

    Note that pyGSTi's 'S' (stochastic) coefficients must be non-negative for
    the gate to be completely positive, but the linear inversion of
    :func:`invert_first_order` is unconstrained and may return negative
    values; this is expected and is not corrected here.

    Parameters
    ----------
    target_model : ExplicitOpModel
        The ideal model.  Must support `set_all_parameterizations('GLND')`
        (e.g. a `'full TP'`-parameterized explicit model).

    Returns
    -------
    list
        A list of `(gate_label, errorgen_label)` tuples, where `gate_label` is
        a :class:`~pygsti.baseobjs.Label` naming one of the model's operations
        and `errorgen_label` is a
        :class:`~pygsti.baseobjs.errorgenlabel.GlobalElementaryErrorgenLabel`.
        Gates appear in the model's own order.
    """
    mdl = _glnd_model(target_model)
    labels = []
    for gate_label in target_model.operations.keys():
        for egl in mdl.operations[gate_label].errorgen_coefficient_labels():
            labels.append((gate_label, egl))
    return labels


def model_from_errorgen_coefficients(target_model, coefficients):
    """
    Build a noisy model from elementary error generator coefficients.

    Parameters
    ----------
    target_model : ExplicitOpModel
        The ideal model.  Its state preparations and POVMs are carried over
        unchanged (cGST does not characterize SPAM).

    coefficients : dict
        Maps a gate label to a dict mapping
        :class:`~pygsti.baseobjs.errorgenlabel.GlobalElementaryErrorgenLabel`
        (or any label accepted by `set_errorgen_coefficients`) to a real
        coefficient.  Gates that appear are *reset*: labels omitted from an
        inner dict get coefficient zero.  Gates absent from `coefficients`
        keep the target's own (typically zero) error generator.

    Returns
    -------
    ExplicitOpModel
        A copy of `target_model` with `'GLND'` parameterized gates and a
        matrix forward simulator, whose gate `G` has dense matrix
        `expm(L) @ G_ideal`.
    """
    mdl = _glnd_model(target_model)
    for gate_label, coeffs in coefficients.items():
        mdl.operations[gate_label].set_errorgen_coefficients(dict(coeffs), action='reset')
    return mdl


def check_design_modes(design):
    """
    Check that a design's sampling modes are supported by the linear inversion.

    Raises a :class:`ValueError` if any sub-experiment uses `'full'` mode (its
    Fourier-operator eigenvalue inversion is not supported by the first-order
    linear inversion) and emits a :class:`UserWarning` if any uses `'reduced'`
    mode (whose Monte-Carlo synthetic projector leaves an `O(error)` ripple in
    the decay curves that degrades the finite-differenced design matrix; see
    :func:`first_order_design_matrix`).

    Parameters
    ----------
    design : CharacterGSTDesign

    Returns
    -------
    None
    """
    modes = {}
    for name in design.keys():
        modes.setdefault(getattr(design[name], 'mode', None), []).append(name)
    if 'full' in modes:
        raise ValueError("The cGST linear inversion does not support 'full'-mode germ-decay "
                         "experiments (the Fourier-operator eigenvalue inversion they require is "
                         "not first-order linearizable and produces spurious amplified "
                         "directions).  %d sub-experiment(s) use 'full' mode, e.g. '%s'; build the "
                         "design with mode='exact'." % (len(modes['full']), modes['full'][0]))
    if 'reduced' in modes:
        _warnings.warn("%d cGST sub-experiment(s) use 'reduced' sampling mode.  Its Monte-Carlo "
                       "synthetic projector leaves an O(error) ripple in the trivial-irrep decay "
                       "curves that the finite-differenced design matrix cannot distinguish from "
                       "a decay, so the linear inversion is less accurate than with mode='exact' "
                       "(which has zero projector sampling error and is the recommended mode)."
                       % len(modes['reduced']))


def first_order_design_matrix(target_model, design, step=1e-4, labels=None,
                              invert_full_mode=True, obs_labels=None):
    """
    The Jacobian of the cGST observables with respect to error generator coefficients.

    `J[i, j]` is the central finite difference of observable `i` (see
    :func:`observable_labels`) with respect to elementary error generator
    coefficient `j` (see :func:`errorgen_parameter_labels`), evaluated at the
    ideal (`target_model`) point.

    Parameters
    ----------
    target_model : ExplicitOpModel
        The ideal model.

    design : CharacterGSTDesign
        The cGST design.

    step : float, optional
        The finite-difference step size.  It must be small enough that the
        response is linear but large enough that the decay fits resolve it: a
        trivial-irrep decay only bends away from a straight line by
        `~((1 - lam) * max(depths))**2 / 2` in relative terms, so
        `step * max(design depths)` should be at least ~1e-2.  Anywhere in
        `1e-4` to `1e-3` works for the standard designs; note that `1e-3` is
        about five times *slower* than `1e-4` (the decay fits converge from
        further away) with no gain in accuracy.

    labels : list, optional
        The error generator parameters to differentiate with respect to (see
        :func:`errorgen_parameter_labels`), which is also the column order.
        Defaults to all of them.

    invert_full_mode : bool, optional
        Passed through to :func:`simulate_observables`.

    obs_labels : list, optional
        A precomputed :func:`observable_labels` list.

    Returns
    -------
    numpy.ndarray
        A `(num_observables, num_parameters)` array.

    Raises
    ------
    ValueError
        If any sub-experiment of `design` uses the `'full'` sampling mode.  The
        Fourier-operator eigenvalue inversion those decays go through is not a
        first-order-linearizable map (it produces spurious amplified
        directions in the design matrix), so the linear inversion does not
        support it; use `'exact'` (or `'reduced'`) mode designs.

    Notes
    -----
    The returned matrix is rank deficient by construction (gauge freedom plus
    unamplified error generators).  Because it is built by finite differences
    of *fitted* decay rates, its numerically-zero singular values sit at
    roughly `1e-5` relative for designs containing long germs (and near machine
    precision for short ones) rather than at `1e-16`; pass a correspondingly
    generous `rcond` (`1e-4` is a safe default) to :func:`invert_first_order`
    when determining the rank.
    """
    check_design_modes(design)
    if labels is None:
        labels = errorgen_parameter_labels(target_model)
    if obs_labels is None:
        obs_labels = observable_labels(design, target_model)

    mdl = _glnd_model(target_model)
    # cGST designs contain deep circuits (a length-7 germ at depth 128 is ~900
    # layers), and pyGSTi's matrix forward simulator builds an evaluation tree
    # over them that is often far larger than the circuit list itself.  Only
    # outcome probabilities are needed here, so use the map simulator instead:
    # it agrees to machine precision and is one to two orders of magnitude
    # faster on these circuits.  (`_glnd_model`'s own simulator stays 'matrix'
    # because the estimated model it produces is later asked for `sim.product`.)
    try:
        mdl.sim = 'map'
    except Exception:  # pragma: no cover - defensive: keep the matrix simulator
        pass
    base = {gate_label: dict(mdl.operations[gate_label].errorgen_coefficients(
        return_basis=False, logscale_nonham=False))
        for gate_label in target_model.operations.keys()}

    jacobian = _np.zeros((len(obs_labels), len(labels)), 'd')
    for j, (gate_label, egl) in enumerate(labels):
        op = mdl.operations[gate_label]
        plus = dict(base[gate_label]); plus[egl] = plus.get(egl, 0.0) + step
        op.set_errorgen_coefficients(plus, action='reset')
        y_plus = simulate_observables(mdl, design, target_model,
                                      invert_full_mode=invert_full_mode, obs_labels=obs_labels)
        minus = dict(base[gate_label]); minus[egl] = minus.get(egl, 0.0) - step
        op.set_errorgen_coefficients(minus, action='reset')
        y_minus = simulate_observables(mdl, design, target_model,
                                       invert_full_mode=invert_full_mode, obs_labels=obs_labels)
        op.set_errorgen_coefficients(dict(base[gate_label]), action='reset')
        jacobian[:, j] = (y_plus - y_minus) / (2 * step)
    return jacobian


def invert_first_order(y, y_ideal, jacobian, y_stderr=None, rcond=1e-8):
    """
    Solve the first-order cGST linear system for error generator coefficients.

    Finds the minimum-norm least-squares solution `x` of `J x = y - y_ideal`,
    optionally weighted by `1 / y_stderr`.

    Because every cGST observable is gauge invariant, `J` is rank deficient:
    its null space contains both the gauge directions and any error generator
    the germ set fails to amplify.  The minimum-norm solution returned here is
    therefore *a* representative of the estimated gauge orbit -- specifically
    the one of smallest total error generator (Euclidean) norm.  Converting it
    to a physically meaningful gauge (e.g. the cGST standard gauge) is
    a separate, downstream step.

    Parameters
    ----------
    y : array-like
        The measured observables (see :func:`observables_from_results`).

    y_ideal : array-like
        The observables predicted for the ideal model (see
        :func:`simulate_observables`).

    jacobian : numpy.ndarray
        The design matrix from :func:`first_order_design_matrix`.

    y_stderr : array-like, optional
        Standard errors of `y`.  Non-finite or non-positive entries are
        replaced by the median of the finite, positive ones (or by 1 if there
        are none), so that a partially-populated uncertainty vector still
        gives a sensible weighting.

    rcond : float, optional
        Relative cutoff on the singular values, used both for the
        least-squares solve and for identifying the numerical null space.
        With a finite-difference `jacobian` the numerically-zero singular
        values are set by the differencing error rather than by machine
        precision -- see the notes on :func:`first_order_design_matrix` -- so
        `1e-4` is often a better choice than the default here.

    Returns
    -------
    dict
        With keys

        - `'coefficients'` : the minimum-norm solution `x` (ndarray).
        - `'covariance'` : the parameter covariance `pinv(Jw) @ pinv(Jw)^T`
          implied by unit-variance whitened residuals, or None if `y_stderr`
          was not given.
        - `'rank'` : the numerical rank of the (weighted) design matrix.
        - `'singular_values'` : the singular values of the weighted matrix.
        - `'residual'` : `(y - y_ideal) - J x`, in the original (unweighted)
          observable units.
        - `'unamplified_directions'` : a `(num_params, k)` orthonormal basis
          of the design matrix's null space.
    """
    y = _np.asarray(y, dtype='d')
    y_ideal = _np.asarray(y_ideal, dtype='d')
    jacobian = _np.asarray(jacobian, dtype='d')
    delta = y - y_ideal

    if y_stderr is not None:
        sig = _np.array(y_stderr, dtype='d').copy()
        good = _np.isfinite(sig) & (sig > 0)
        fill = _np.median(sig[good]) if _np.any(good) else 1.0
        sig[~good] = fill
        weights = 1.0 / sig
    else:
        weights = _np.ones(len(delta))

    jw = jacobian * weights[:, None]
    dw = delta * weights

    x, _, rank, sv = _np.linalg.lstsq(jw, dw, rcond=rcond)
    rank = int(rank)

    _, s, vt = _np.linalg.svd(jw, full_matrices=True)
    cutoff = rcond * (s[0] if len(s) > 0 else 0.0)
    num_nonzero = int(_np.count_nonzero(s > cutoff))
    null_basis = vt[num_nonzero:].T.copy() if vt.shape[0] > num_nonzero \
        else _np.zeros((jacobian.shape[1], 0))

    covariance = None
    if y_stderr is not None:
        pinv = _np.linalg.pinv(jw, rcond=rcond)
        covariance = pinv @ pinv.T

    return {'coefficients': x, 'covariance': covariance, 'rank': rank,
            'singular_values': sv if len(sv) else s, 'residual': delta - jacobian @ x,
            'unamplified_directions': null_basis}


def coefficients_to_dict(x, labels):
    """
    Group a flat coefficient vector into a per-gate dictionary.

    Parameters
    ----------
    x : array-like
        A coefficient vector ordered as `labels`.

    labels : list
        `(gate_label, errorgen_label)` tuples from
        :func:`errorgen_parameter_labels`.

    Returns
    -------
    dict
        Maps gate label to a dict mapping error generator label to float.
    """
    out = {}
    for value, (gate_label, egl) in zip(_np.asarray(x, dtype='d'), labels):
        out.setdefault(gate_label, {})[egl] = float(value)
    return out

"""
Decay-curve fitting for character gate set tomography (cGST) data
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from scipy.optimize import least_squares as _least_squares

# cGST filters a germ's decay onto single irreps of the group the germ generates,
# so -- unlike unfiltered GST data -- each character-weighted signal is fit by a
# SINGLE exponential:
#
#   trivial irrep (T1-type):     z(k) = (B - C) * lam**k + C          (real)
#   nontrivial irrep (Ramsey):   z(k) = A * lam**k * exp(i(theta*k + phi))   (complex)
#
# These parameterizations differ from the RB conventions in
# pygsti.algorithms.rbfit (A + B*p**m with RB error-rate rescalings), which is
# why cGST gets its own fitting module.


def _stderrs_from_least_squares(result, num_points):
    """Parameter standard errors from a scipy least_squares result's Jacobian."""
    num_params = result.x.size
    dof = max(num_points - num_params, 1)
    s_sq = 2 * result.cost / dof
    try:
        cov = _np.linalg.inv(result.jac.T @ result.jac) * s_sq
        return _np.sqrt(_np.abs(_np.diag(cov)))
    except _np.linalg.LinAlgError:
        return _np.full(num_params, _np.nan)


def fit_real_decay(depths, zs, bootstrap_samples=0, rand_state=None):
    """
    Fit a real single-exponential decay to an asymptote: `z(k) = (B - C) * lam**k + C`.

    This is the fit model for trivial-irrep ("T1-type") character-weighted cGST
    data.  Initial values are seeded from the data (asymptote from the tail,
    rate from a log-linear regression) and refined with nonlinear least squares.

    Parameters
    ----------
    depths : array-like of int
        The germ depths `k`.

    zs : array-like of float
        The character-weighted probabilities `z(k)` (real part is taken).

    bootstrap_samples : int, optional
        If > 0, also compute residual-bootstrap standard errors from this many
        resamples (usually tighter-tailed than the Jacobian estimate).

    rand_state : numpy.random.RandomState, optional
        Source of randomness for the bootstrap.

    Returns
    -------
    dict
        With keys `'estimates'` (dict of `'lam'`, `'B'`, `'C'`), `'stderrs'`
        (same keys), `'success'` (bool), and `'bootstrap_stderrs'`
        (same keys, or None if no bootstrap was run).
    """
    depths = _np.asarray(depths, dtype=float)
    zs = _np.real(_np.asarray(zs, dtype=complex))

    # -- seed values
    c0 = _np.mean(zs[depths >= _np.median(depths)])  # asymptote ~ tail mean
    b0 = zs[_np.argmin(depths)]
    resid0 = zs - c0
    mask = _np.abs(resid0) > 1e-12
    if _np.count_nonzero(mask) >= 2 and abs(b0 - c0) > 1e-12:
        slope = _np.polyfit(depths[mask], _np.log(_np.abs(resid0[mask])), 1)[0]
        lam0 = float(_np.clip(_np.exp(slope), 1e-6, 1.0))
    else:
        lam0 = 1.0

    def residuals(params):
        lam, b, c = params
        return (b - c) * lam ** depths + c - zs

    # A slowly-growing signal (lam slightly > 1, e.g. from correlated errors
    # boosting a compound germ's trivial branch) is nearly degenerate, over a
    # finite depth range, with a slow decay toward a large asymptote.  Fit
    # from both seed hypotheses and keep the lower-cost solution.
    span = depths.max() - depths.min() if len(depths) > 1 else 1.0
    ratio = zs[_np.argmax(depths)] / zs[_np.argmin(depths)] if abs(zs[_np.argmin(depths)]) > 1e-12 else 1.0
    lam0_growth = float(_np.clip(abs(ratio) ** (1.0 / span), 1.0, 1.09)) if ratio > 0 else 1.0

    def do_fit(target_zs):
        def resid(params):
            lam, b, c = params
            return (b - c) * lam ** depths + c - target_zs
        fits = [_least_squares(resid, seed, bounds=([0., -2., -2.], [1.1, 2., 2.]))
                for seed in ([lam0, b0, c0], [lam0_growth, b0, 0.])]
        return min(fits, key=lambda f: f.cost)

    result = do_fit(zs)
    lam, b, c = result.x
    stderrs = _stderrs_from_least_squares(result, len(zs))

    boot = None
    if bootstrap_samples > 0:
        rand_state = rand_state if (rand_state is not None) else _np.random.RandomState()
        fit_resids = residuals(result.x)
        samples = []
        for _ in range(bootstrap_samples):
            fake = (b - c) * lam ** depths + c \
                + rand_state.choice(fit_resids, size=len(zs), replace=True)
            samples.append(do_fit(fake).x)
        boot_std = _np.std(_np.array(samples), axis=0)
        boot = {'lam': boot_std[0], 'B': boot_std[1], 'C': boot_std[2]}

    return {'estimates': {'lam': lam, 'B': b, 'C': c},
            'stderrs': {'lam': stderrs[0], 'B': stderrs[1], 'C': stderrs[2]},
            'success': bool(result.success),
            'bootstrap_stderrs': boot}


def fit_complex_decay(depths, zs, bootstrap_samples=0, rand_state=None):
    """
    Fit a complex single-exponential: `z(k) = A * lam**k * exp(1j*(theta*k + phi))`.

    This is the fit model for nontrivial-irrep ("Ramsey-type") character-weighted
    cGST data: `lam` is the decay eigenvalue magnitude and `theta` the
    phase-winding rate per germ application (the coherent angle deviation).
    The magnitude and unwrapped phase are first fit by separate linear
    regressions, then all four parameters are refined jointly on the stacked
    real and imaginary residuals.

    Parameters
    ----------
    depths : array-like of int
        The germ depths `k`.

    zs : array-like of complex
        The character-weighted (complex) signal `z(k)`.

    bootstrap_samples : int, optional
        If > 0, also compute residual-bootstrap standard errors.

    rand_state : numpy.random.RandomState, optional
        Source of randomness for the bootstrap.

    Returns
    -------
    dict
        With keys `'estimates'` (dict of `'A'`, `'lam'`, `'theta'`, `'phi'`),
        `'stderrs'`, `'success'`, and `'bootstrap_stderrs'`.
    """
    depths = _np.asarray(depths, dtype=float)
    zs = _np.asarray(zs, dtype=complex)

    # -- seed values: magnitude via log-linear fit, phase via unwrapped-angle fit.
    mags = _np.abs(zs)
    noise_floor = max(1e-12, 0.02 * mags.max())
    mask = mags > noise_floor
    order = _np.argsort(depths[mask])
    d_m, z_m = depths[mask][order], zs[mask][order]
    log_slope, log_icpt = _np.polyfit(d_m, _np.log(_np.abs(z_m)), 1)
    lam0 = float(_np.clip(_np.exp(log_slope), 1e-6, 1.0))
    a0 = float(_np.exp(log_icpt))
    theta0, phi0 = _np.polyfit(d_m, _np.unwrap(_np.angle(z_m)), 1)

    def model(params, k):
        a, lam, theta, phi = params
        return a * lam ** k * _np.exp(1j * (theta * k + phi))

    def do_fit(target_zs):
        def resid(params):
            diff = model(params, depths) - target_zs
            return _np.concatenate([diff.real, diff.imag])
        return _least_squares(resid, [a0, lam0, theta0, phi0],
                              bounds=([0., 0., theta0 - _np.pi, phi0 - _np.pi],
                                      [2., 1.1, theta0 + _np.pi, phi0 + _np.pi]))

    result = do_fit(zs)
    a, lam, theta, phi = result.x
    stderrs = _stderrs_from_least_squares(result, 2 * len(zs))

    boot = None
    if bootstrap_samples > 0:
        rand_state = rand_state if (rand_state is not None) else _np.random.RandomState()
        fit_resids = model(result.x, depths) - zs
        samples = []
        for _ in range(bootstrap_samples):
            fake = model(result.x, depths) \
                + rand_state.choice(fit_resids, size=len(zs), replace=True)
            samples.append(do_fit(fake).x)
        boot_std = _np.std(_np.array(samples), axis=0)
        boot = {'A': boot_std[0], 'lam': boot_std[1], 'theta': boot_std[2], 'phi': boot_std[3]}

    return {'estimates': {'A': a, 'lam': lam, 'theta': theta, 'phi': phi},
            'stderrs': {'A': stderrs[0], 'lam': stderrs[1], 'theta': stderrs[2], 'phi': stderrs[3]},
            'success': bool(result.success),
            'bootstrap_stderrs': boot}


def invert_projector_eigenvalue(z, order, tol=1e-12, max_iter=100):
    """
    Solve `projector_eigenvalue_map(y, order) == z` for the germ eigenvalue deviation `y`.

    In full-random-sampling cGST the fitted per-round decay is an eigenvalue of
    the (noisy) group Fourier operator, `f(y) = (1 - y**order)/(order*(1 - y))`,
    rather than the bare germ's eigenvalue deviation `y` itself.  This numerically
    inverts `f` (complex Newton iteration) on the branch near `y = 1`, converting
    full-mode fit results into bare germ eigenvalues comparable to reduced-mode fits.

    Parameters
    ----------
    z : complex
        Fitted Fourier-operator eigenvalue (e.g. `lam * exp(1j*theta)` from
        :func:`fit_complex_decay` on full-mode data).

    order : int
        The order of the cyclic group generated by the germ.

    tol : float, optional
        Convergence tolerance on `|f(y) - z|`.

    max_iter : int, optional
        Maximum Newton iterations.

    Returns
    -------
    complex
        The germ eigenvalue deviation `y` with `f(y) = z`.
    """
    if order < 2:
        raise ValueError("Inversion requires a group of order >= 2")
    z = complex(z)
    # Small-deviation seed: f(y) ~= y**((order-1)/2) near y = 1.
    y = z ** (2.0 / (order - 1))
    for _ in range(max_iter):
        # f(y) = (1/order) * sum_m y^m ;  f'(y) = (1/order) * sum_m m*y^(m-1)
        powers = _np.array([y ** m for m in range(order)])
        f = powers.sum() / order
        fprime = sum(m * y ** (m - 1) for m in range(1, order)) / order
        if abs(f - z) < tol:
            return y
        if fprime == 0:
            break
        y = y - (f - z) / fprime
    if abs(f - z) > 1e-6:
        raise RuntimeError("Newton iteration failed to invert the projector eigenvalue map "
                           "(final error %g)" % abs(f - z))
    return y

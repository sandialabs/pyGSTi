"""
Character utilities for finite abelian groups, used by character gate set tomography (cGST)
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

# A finite abelian group is (isomorphic to) a direct product of cyclic groups
# Z_{N_1} x ... x Z_{N_f}.  Throughout this module such a group is specified by
# `orders = (N_1, ..., N_f)`, elements and irreps are specified by integer tuples
# of the same length (plain integers are accepted when the group is a single
# cyclic factor), and the irreducible characters -- all one-dimensional -- are
#
#     chi_j(m) = exp(2 pi i * sum_f j_f m_f / N_f).


def _as_tuple(x, num_factors):
    """Normalize an int or tuple group-element/irrep label to a tuple of ints."""
    if isinstance(x, (int, _np.integer)):
        if num_factors != 1:
            raise ValueError("Integer labels are only allowed for single-factor (cyclic) groups")
        return (int(x),)
    x = tuple(int(v) for v in x)
    if len(x) != num_factors:
        raise ValueError("Label %s has %d entries but the group has %d cyclic factors"
                         % (str(x), len(x), num_factors))
    return x


def abelian_character(orders, irrep, element):
    """
    Evaluate one irreducible character of a finite abelian group.

    The group is `Z_{orders[0]} x ... x Z_{orders[-1]}` and the character is
    `chi_irrep(element) = exp(2 pi i * sum_f irrep[f] * element[f] / orders[f])`.

    Parameters
    ----------
    orders : int or tuple of ints
        The orders of the cyclic factors of the group.

    irrep : int or tuple of ints
        The irrep label; entry `f` is interpreted modulo `orders[f]`.

    element : int or tuple of ints
        The group element; entry `f` is interpreted modulo `orders[f]`.

    Returns
    -------
    complex
    """
    if isinstance(orders, (int, _np.integer)): orders = (int(orders),)
    irrep = _as_tuple(irrep, len(orders))
    element = _as_tuple(element, len(orders))
    phase = sum(j * m / N for j, m, N in zip(irrep, element, orders))
    return _np.exp(2j * _np.pi * phase)


def abelian_characters(orders):
    """
    The full character table of a finite abelian group.

    Parameters
    ----------
    orders : int or tuple of ints
        The orders of the cyclic factors of the group.

    Returns
    -------
    numpy.ndarray
        A complex array of shape `(N, N)` where `N = prod(orders)`.  Row `i`
        is the character of the irrep with mixed-radix label `i`, and column
        `k` corresponds to the element with mixed-radix label `k` (labels are
        flattened with the *last* factor varying fastest, as in
        :func:`numpy.unravel_index` with C ordering).
    """
    if isinstance(orders, (int, _np.integer)): orders = (int(orders),)
    n = int(_np.prod(orders))
    table = _np.empty((n, n), dtype=complex)
    for i in range(n):
        irrep = _np.unravel_index(i, orders)
        for k in range(n):
            element = _np.unravel_index(k, orders)
            table[i, k] = abelian_character(orders, irrep, element)
    return table


def character_weights(orders, irrep, elements):
    """
    Conjugated character values used to weight measurement outcomes in cGST.

    Character-weighted probabilities are formed by averaging circuit outcomes
    multiplied by `conj(chi_irrep(element))`, which isolates the part of the
    signal transforming in the given irrep.

    Parameters
    ----------
    orders : int or tuple of ints
        The orders of the cyclic factors of the group.

    irrep : int or tuple of ints
        The irrep label to project onto.

    elements : array-like
        A sequence of group elements: ints for a cyclic group, or int-tuples
        (equivalently a 2D array with one row per element) in general.

    Returns
    -------
    numpy.ndarray
        Complex array with one weight, `conj(chi_irrep(element))`, per element.
    """
    return _np.conj(_np.array([abelian_character(orders, irrep, el) for el in elements]))


def uniform_power_distribution(order, rounds):
    """
    The distribution of the total of `rounds` i.i.d. uniform draws from `{0, ..., order-1}`.

    This is the distribution of the total random germ power accumulated by
    `rounds` rounds of uniform cyclic-group sampling in cGST; it supplies the
    quadrature weights with which the group Fourier (character projection)
    operator can be evaluated exactly for cyclic germs.

    Parameters
    ----------
    order : int
        The order of the cyclic group.

    rounds : int
        The number of i.i.d. uniform draws.

    Returns
    -------
    numpy.ndarray
        Probabilities of totals `0, 1, ..., rounds*(order-1)`.
    """
    single = _np.full(order, 1.0 / order)
    dist = _np.array([1.0])
    for _ in range(rounds):
        dist = _np.convolve(dist, single)
    return dist


def germ_group_order(germ_superop, max_order=24, tol=1e-8):
    """
    The order of the cyclic group generated by an ideal germ superoperator.

    Parameters
    ----------
    germ_superop : numpy.ndarray
        The (dense) superoperator matrix, e.g. Pauli-transfer matrix, of the
        ideal germ.

    max_order : int, optional
        Give up (raise a ValueError) if no power up to this one equals the identity.

    tol : float, optional
        Tolerance on the max-abs-difference from the identity.

    Returns
    -------
    int
        The smallest `n >= 1` with `germ_superop**n == identity`.
    """
    dim = germ_superop.shape[0]
    ident = _np.identity(dim)
    current = _np.identity(dim)
    for n in range(1, max_order + 1):
        current = current @ germ_superop
        if _np.max(_np.abs(current - ident)) < tol:
            return n
    raise ValueError("Germ does not generate a cyclic group of order <= %d" % max_order)


def fourier_operator(germ_superop, order, irrep):
    """
    The group Fourier transform (character projection) operator of a cyclic germ.

    For a germ `G` generating `Z_order`, this is
    `Pi_irrep = (1/order) * sum_m conj(chi_irrep(m)) G^m`.  When `G` is ideal
    this is the projector onto the eigenspace of `G` with eigenvalue
    `chi_irrep(1)`; for a noisy germ it is the "synthetic germ" whose powers
    cGST estimates via character weighting.

    Parameters
    ----------
    germ_superop : numpy.ndarray
        The (dense) superoperator matrix of the germ (ideal or noisy).

    order : int
        The order of the cyclic group generated by the ideal germ.

    irrep : int
        The irrep index in `[0, order)`; 0 is the trivial irrep.

    Returns
    -------
    numpy.ndarray
    """
    dim = germ_superop.shape[0]
    op = _np.zeros((dim, dim), dtype=complex)
    current = _np.identity(dim)
    for m in range(order):
        op += _np.conj(abelian_character(order, irrep, m)) * current
        current = current @ germ_superop
    return op / order


def projector_eigenvalue_map(y, order):
    """
    Eigenvalue of a noisy Fourier operator in terms of the germ's eigenvalue deviation.

    If a noisy germ has eigenvalue `chi_irrep(1) * y` on some eigenvector --
    i.e. `y` is the *deviation* of the eigenvalue from its ideal phase -- then
    (when the noise commutes with the germ) the Fourier operator
    :func:`fourier_operator` has eigenvalue `f(y) = (1 - y**order) /
    (order * (1 - y))` on that eigenvector.  For a pure phase deviation
    `y = exp(i*theta)` with small `theta`, `arg(f(y)) ~= (order-1)/2 * theta`:
    full-random-sampling cGST amplifies phase deviations by `(order-1)/2`
    relative to a single application of the germ.

    Parameters
    ----------
    y : complex or numpy.ndarray
        Germ eigenvalue deviation(s) from the ideal irrep phase.

    order : int
        The order of the cyclic group generated by the ideal germ.

    Returns
    -------
    complex or numpy.ndarray
    """
    y = _np.asarray(y, dtype=complex)
    scalar = (y.ndim == 0)
    y = _np.atleast_1d(y)
    out = _np.empty(y.shape, dtype=complex)
    near_one = _np.abs(y - 1.0) < 1e-12
    # geometric-series mean (1/N) * sum_{m<N} y^m; equals 1 at y == 1
    out[near_one] = _np.mean([y[near_one] ** m for m in range(order)], axis=0) if order > 1 else 1.0
    yn = y[~near_one]
    out[~near_one] = (1.0 - yn ** order) / (order * (1.0 - yn))
    return out[0] if scalar else out

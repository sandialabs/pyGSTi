"""
Exact Clebsch-Gordan coefficients and Wigner 6-j symbols.

This module evaluates Racah's closed-form formulas for the Clebsch-Gordan
(CG) coefficients and Wigner 6-j symbols using exact rational arithmetic
(`fractions.Fraction` and `math.factorial`).  Internally, the *square* of
each symbol is accumulated as an exact `Fraction` (a rational prefactor
times the square of a rational sum), the overall sign is tracked
separately, and the final float is produced via `math.sqrt(float(...))`.
Since `Fraction.__float__` is correctly rounded in CPython, this yields
results accurate to about 1 ulp -- effectively exact for double precision
use.

Both symbols use the Condon-Shortley phase convention.  No third-party
packages are imported by this module; `sympy` is used only as an oracle in
the associated unit tests (a testing-only dependency).
"""
# ***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import math as _math
from fractions import Fraction as _Fraction

__all__ = ['clebsch_gordan', 'wigner_6j']


def _as_half_integer(value, name):
    """
    Coerce `value` to an exact `Fraction` and validate that it is an
    integer or half-integer (i.e. that `2 * value` is an integer).

    Parameters
    ----------
    value : int, float, or Fraction
        The value to validate/convert.

    name : str
        The argument name, used only to build an informative error message.

    Returns
    -------
    Fraction
    """
    if isinstance(value, _Fraction):
        frac = value
    elif isinstance(value, bool):
        # bool is a subclass of int; exclude it explicitly to avoid surprises.
        raise TypeError("%s must be an int, float, or Fraction, not bool" % name)
    elif isinstance(value, int):
        frac = _Fraction(value)
    elif isinstance(value, float):
        if not _math.isfinite(value):
            raise ValueError("%s must be finite (got %r)" % (name, value))
        two_value = 2 * value
        if two_value != round(two_value):
            raise ValueError(
                "%s = %r is not an integer or half-integer (2*%s must be an integer)"
                % (name, value, name))
        frac = _Fraction(round(two_value), 2)
    else:
        raise TypeError("%s must be an int, float, or Fraction, not %s" % (name, type(value).__name__))

    if (2 * frac).denominator != 1:
        raise ValueError(
            "%s = %s is not an integer or half-integer (2*%s must be an integer)"
            % (name, frac, name))
    return frac


def _as_nonneg_int(fraction_value):
    """
    Return `int(fraction_value)` if `fraction_value` is a non-negative
    integer-valued `Fraction`, else `None`.
    """
    if fraction_value.denominator != 1 or fraction_value.numerator < 0:
        return None
    return fraction_value.numerator


def _int_val(fraction_value):
    """
    Return `int(fraction_value)`, asserting that `fraction_value` is exactly
    integer-valued.  Used internally where integrality is already guaranteed
    by construction.
    """
    assert fraction_value.denominator == 1, \
        "internal error: expected an integer-valued Fraction, got %s" % fraction_value
    return fraction_value.numerator


def _triangle_delta(j1, j2, j3):
    """
    Return the Fraction (j1+j2-j3)!(j1-j2+j3)!(-j1+j2+j3)! / (j1+j2+j3+1)!
    (the squared triangle-coefficient `Delta(j1,j2,j3)^2`), or `None` if the
    triangle inequality / integrality conditions on (j1, j2, j3) are not
    satisfied.
    """
    perimeter_term = _as_nonneg_int(j1 + j2 + j3 + 1)
    if perimeter_term is None:
        return None
    a = _as_nonneg_int(j1 + j2 - j3)
    b = _as_nonneg_int(j1 - j2 + j3)
    c = _as_nonneg_int(-j1 + j2 + j3)
    if a is None or b is None or c is None:
        return None
    return _Fraction(_math.factorial(a) * _math.factorial(b) * _math.factorial(c),
                      _math.factorial(perimeter_term))


def clebsch_gordan(j1, m1, j2, m2, j, m):
    """
    The Clebsch-Gordan coefficient <j1 m1; j2 m2 | j m>, evaluated exactly
    via Racah's formula (Condon-Shortley phase convention).

    Parameters
    ----------
    j1, j2, j : int, float, or Fraction
        Angular-momentum quantum numbers.  Each may be any non-negative
        integer or half-integer (i.e. `2*j1`, `2*j2`, `2*j` must be
        integers).

    m1, m2, m : int, float, or Fraction
        Magnetic quantum numbers associated with j1, j2, and j respectively.
        Each must be an integer or half-integer.

    Returns
    -------
    float
        The value of <j1 m1; j2 m2 | j m>.  Returns exactly `0.0` when the
        triangle inequality on (j1, j2, j), the selection rule
        `m1 + m2 == m`, or the `|m| <= j` constraints are violated (rather
        than raising).

    Raises
    ------
    ValueError
        If any argument is not (representable as) an integer or
        half-integer.
    """
    j1 = _as_half_integer(j1, 'j1')
    m1 = _as_half_integer(m1, 'm1')
    j2 = _as_half_integer(j2, 'j2')
    m2 = _as_half_integer(m2, 'm2')
    j = _as_half_integer(j, 'j')
    m = _as_half_integer(m, 'm')

    if m1 + m2 != m:
        return 0.0

    triangle_sq = _triangle_delta(j1, j2, j)
    if triangle_sq is None:
        return 0.0

    two_j_plus_1 = _as_nonneg_int(2 * j + 1)

    base_terms = (j + m, j - m, j1 - m1, j1 + m1, j2 - m2, j2 + m2)
    base_ints = []
    for term in base_terms:
        n = _as_nonneg_int(term)
        if n is None:
            return 0.0
        base_ints.append(n)
    j_pm, j_mm, j1_mm1, j1_pm1, j2_mm2, j2_pm2 = base_ints

    prefactor = triangle_sq * two_j_plus_1 * _Fraction(
        _math.factorial(j_pm) * _math.factorial(j_mm)
        * _math.factorial(j1_mm1) * _math.factorial(j1_pm1)
        * _math.factorial(j2_mm2) * _math.factorial(j2_pm2))

    k_min = max(0, _int_val(j2 - j - m1), _int_val(j1 + m2 - j))
    k_max = min(_int_val(j1 + j2 - j), j1_mm1, j2_pm2)

    total = _Fraction(0)
    for k in range(k_min, k_max + 1):
        denom = (_math.factorial(k)
                  * _math.factorial(_int_val(j1 + j2 - j - k))
                  * _math.factorial(_int_val(j1 - m1 - k))
                  * _math.factorial(_int_val(j2 + m2 - k))
                  * _math.factorial(_int_val(j - j2 + m1 + k))
                  * _math.factorial(_int_val(j - j1 - m2 + k)))
        total += _Fraction((-1) ** k, denom)

    squared_value = prefactor * total * total
    if squared_value == 0:
        return 0.0
    sign = 1 if total > 0 else -1
    return sign * _math.sqrt(float(squared_value))


def wigner_6j(j1, j2, j3, j4, j5, j6):
    """
    The Wigner 6-j symbol {j1 j2 j3; j4 j5 j6}, evaluated exactly via
    Racah's formula.

    Parameters
    ----------
    j1, j2, j3, j4, j5, j6 : int, float, or Fraction
        Angular-momentum quantum numbers.  Each may be any non-negative
        integer or half-integer (i.e. `2*j1`, ..., `2*j6` must be
        integers).

    Returns
    -------
    float
        The value of the 6-j symbol.  Returns exactly `0.0` when any of the
        four triangle conditions -- (j1,j2,j3), (j1,j5,j6), (j4,j2,j6),
        (j4,j5,j3) -- is violated (rather than raising).

    Raises
    ------
    ValueError
        If any argument is not (representable as) an integer or
        half-integer.
    """
    j1 = _as_half_integer(j1, 'j1')
    j2 = _as_half_integer(j2, 'j2')
    j3 = _as_half_integer(j3, 'j3')
    j4 = _as_half_integer(j4, 'j4')
    j5 = _as_half_integer(j5, 'j5')
    j6 = _as_half_integer(j6, 'j6')

    triads = ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3))
    deltas = [_triangle_delta(*triad) for triad in triads]
    if any(delta is None for delta in deltas):
        return 0.0
    prefactor = deltas[0] * deltas[1] * deltas[2] * deltas[3]

    t_lower_bounds = [j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3]
    t_upper_bounds = [j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j3 + j1 + j6 + j4]
    t_min = max(_int_val(bound) for bound in t_lower_bounds)
    t_max = min(_int_val(bound) for bound in t_upper_bounds)

    total = _Fraction(0)
    for t in range(t_min, t_max + 1):
        denom = 1
        for bound in t_lower_bounds:
            denom *= _math.factorial(_int_val(t - bound))
        for bound in t_upper_bounds:
            denom *= _math.factorial(_int_val(bound - t))
        total += _Fraction((-1) ** t * _math.factorial(t + 1), denom)

    squared_value = prefactor * total * total
    if squared_value == 0:
        return 0.0
    sign = 1 if total > 0 else -1
    return sign * _math.sqrt(float(squared_value))

from fractions import Fraction

import pytest

from pygsti.tools import wignersymbols as ws
from ..util import BaseCase


def _half_integer_range(lo, hi):
    """All of lo, lo+1, ..., hi (as Fractions), assuming hi - lo is a non-negative integer."""
    vals = []
    x = Fraction(lo)
    hi = Fraction(hi)
    while x <= hi:
        vals.append(x)
        x += 1
    return vals


class ClebschGordanOracleTester(BaseCase):
    """Cross-checks against sympy's Clebsch-Gordan implementation."""

    def test_against_sympy_j_up_to_4(self):
        sympy = pytest.importorskip('sympy')
        from sympy.physics.quantum.cg import CG as SympyCG

        js = [Fraction(k, 2) for k in range(0, 9)]  # 0, 1/2, 1, ..., 4
        n_checked = 0
        for j1 in js:
            for j2 in js:
                jvals = _half_integer_range(abs(j1 - j2), j1 + j2)
                m1vals = _half_integer_range(-j1, j1)
                m2vals = _half_integer_range(-j2, j2)
                for j in jvals:
                    for m1 in m1vals:
                        for m2 in m2vals:
                            m = m1 + m2
                            if abs(m) > j:
                                continue
                            got = ws.clebsch_gordan(j1, m1, j2, m2, j, m)
                            want = float(SympyCG(
                                sympy.Rational(j1.numerator, j1.denominator),
                                sympy.Rational(m1.numerator, m1.denominator),
                                sympy.Rational(j2.numerator, j2.denominator),
                                sympy.Rational(m2.numerator, m2.denominator),
                                sympy.Rational(j.numerator, j.denominator),
                                sympy.Rational(m.numerator, m.denominator),
                            ).doit())
                            self.assertAlmostEqual(got, want, delta=1e-14)
                            n_checked += 1
        # Sanity check that the sweep actually exercised a meaningful number of cases.
        self.assertGreater(n_checked, 1000)


class Wigner6jOracleTester(BaseCase):
    """Cross-checks against sympy's Wigner 6-j implementation."""

    def test_against_sympy_kjj_kpjj_j_up_to_8(self):
        sympy = pytest.importorskip('sympy')
        from sympy.physics.wigner import wigner_6j as sympy_wigner_6j

        n_checked = 0
        for two_j in range(1, 17):  # j = 1/2, 1, ..., 8
            j = Fraction(two_j, 2)
            for k in range(0, two_j + 1):
                for kp in range(0, two_j + 1):
                    got = ws.wigner_6j(k, j, j, kp, j, j)
                    want = float(sympy_wigner_6j(
                        k, sympy.Rational(j.numerator, j.denominator), sympy.Rational(j.numerator, j.denominator),
                        kp, sympy.Rational(j.numerator, j.denominator), sympy.Rational(j.numerator, j.denominator)))
                    self.assertAlmostEqual(got, want, delta=1e-14)
                    n_checked += 1
        self.assertGreater(n_checked, 500)


class ClebschGordanIdentityTester(BaseCase):
    """Oracle-free identity/consistency tests for clebsch_gordan."""

    def test_orthogonality_relation_1(self):
        # sum_{m1,m2} CG(j1 m1 j2 m2 | j3 m3) CG(j1 m1 j2 m2 | j3' m3') = delta_{j3,j3'} delta_{m3,m3'}
        j1, j2 = Fraction(3, 2), Fraction(1)
        for j3 in _half_integer_range(abs(j1 - j2), j1 + j2):
            for j3p in _half_integer_range(abs(j1 - j2), j1 + j2):
                for m3 in _half_integer_range(-j3, j3):
                    for m3p in _half_integer_range(-j3p, j3p):
                        total = 0.0
                        for m1 in _half_integer_range(-j1, j1):
                            m2 = m3 - m1
                            if abs(m2) > j2:
                                continue
                            m2p = m3p - m1
                            if abs(m2p) > j2:
                                continue
                            if m2 != m2p:
                                continue
                            total += (ws.clebsch_gordan(j1, m1, j2, m2, j3, m3)
                                      * ws.clebsch_gordan(j1, m1, j2, m2, j3p, m3p))
                        expected = 1.0 if (j3 == j3p and m3 == m3p) else 0.0
                        self.assertAlmostEqual(total, expected, delta=1e-12)

    def test_orthogonality_relation_2(self):
        # sum_{j,m} CG(j1 m1 j2 m2 | j m) CG(j1 m1' j2 m2' | j m) = delta_{m1,m1'} delta_{m2,m2'}
        j1, j2 = Fraction(1, 2), Fraction(3, 2)
        m1vals = _half_integer_range(-j1, j1)
        m2vals = _half_integer_range(-j2, j2)
        for m1 in m1vals:
            for m2 in m2vals:
                for m1p in m1vals:
                    for m2p in m2vals:
                        m = m1 + m2
                        mp = m1p + m2p
                        if m != mp:
                            continue
                        total = 0.0
                        for j in _half_integer_range(abs(j1 - j2), j1 + j2):
                            if abs(m) > j:
                                continue
                            total += (ws.clebsch_gordan(j1, m1, j2, m2, j, m)
                                      * ws.clebsch_gordan(j1, m1p, j2, m2p, j, m))
                        expected = 1.0 if (m1 == m1p and m2 == m2p) else 0.0
                        self.assertAlmostEqual(total, expected, delta=1e-12)

    def test_selection_rule_zeros(self):
        # m1 + m2 != m
        self.assertEqual(ws.clebsch_gordan(1, 1, 1, 1, 2, 1), 0.0)
        # triangle inequality violated
        self.assertEqual(ws.clebsch_gordan(1, 0, 1, 0, 3, 0), 0.0)
        # |m| > j
        self.assertEqual(ws.clebsch_gordan(0.5, 0.5, 0.5, 0.5, 0, 0), 0.0)
        # m out of range for j1 (j1 - m1 not an integer/valid)
        self.assertEqual(ws.clebsch_gordan(0.5, 1, 0.5, 0, 1, 1), 0.0)

    def test_input_validation_errors(self):
        with self.assertRaises(ValueError):
            ws.clebsch_gordan(0.3, 0, 0.3, 0, 0, 0)
        with self.assertRaises(ValueError):
            ws.clebsch_gordan(1, 0.3, 1, 0, 1, 0.3)
        with self.assertRaises(TypeError):
            ws.clebsch_gordan('1', 0, 1, 0, 1, 0)

        # Fraction inputs that are not half-integers should also raise.
        with self.assertRaises(ValueError):
            ws.clebsch_gordan(Fraction(1, 3), 0, 1, 0, 1, 0)

        # A float that is merely *close* to a half-integer (but not exactly
        # one) must raise rather than being silently coerced.
        with self.assertRaises(ValueError):
            ws.clebsch_gordan(0.5000000001, 0.5, 0.5, 0.5, 1, 1)

        # Valid Fraction inputs should work fine (no exception).
        val = ws.clebsch_gordan(Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), 1, 1)
        self.assertAlmostEqual(val, 1.0, delta=1e-14)


class Wigner6jIdentityTester(BaseCase):
    """Oracle-free identity/consistency tests for wigner_6j."""

    def test_symmetry_under_column_permutation(self):
        j1, j2, j3, j4, j5, j6 = 1, Fraction(3, 2), Fraction(3, 2), Fraction(3, 2), 1, Fraction(3, 2)
        base = ws.wigner_6j(j1, j2, j3, j4, j5, j6)

        # Any permutation of the three columns leaves the symbol invariant.
        swapped_12 = ws.wigner_6j(j2, j1, j3, j5, j4, j6)
        swapped_23 = ws.wigner_6j(j1, j3, j2, j4, j6, j5)
        cyclic = ws.wigner_6j(j2, j3, j1, j5, j6, j4)
        self.assertAlmostEqual(base, swapped_12, delta=1e-13)
        self.assertAlmostEqual(base, swapped_23, delta=1e-13)
        self.assertAlmostEqual(base, cyclic, delta=1e-13)

    def test_symmetry_under_upper_lower_swap(self):
        j1, j2, j3, j4, j5, j6 = 1, Fraction(3, 2), Fraction(3, 2), Fraction(3, 2), 1, Fraction(3, 2)
        base = ws.wigner_6j(j1, j2, j3, j4, j5, j6)
        # Swap the upper and lower arguments of any two columns.
        swapped = ws.wigner_6j(j4, j5, j3, j1, j2, j6)
        self.assertAlmostEqual(base, swapped, delta=1e-13)

    def test_selection_rule_zeros(self):
        # Triangle inequality violated on the first triad.
        self.assertEqual(ws.wigner_6j(5, 1, 1, 1, 1, 1), 0.0)
        # Non-integer perimeter on a triad (j1+j2+j3 not integer).
        self.assertEqual(ws.wigner_6j(0.5, 1, 1, 1, 1, 1), 0.0)

    def test_input_validation_errors(self):
        with self.assertRaises(ValueError):
            ws.wigner_6j(0.3, 1, 1, 1, 1, 1)
        with self.assertRaises(TypeError):
            ws.wigner_6j(None, 1, 1, 1, 1, 1)

    def test_known_value(self):
        # {1 1 1; 1 1 1} = 1/6 (Condon-Shortley convention; matches sympy).
        self.assertAlmostEqual(ws.wigner_6j(1, 1, 1, 1, 1, 1), 1.0 / 6.0, delta=1e-14)

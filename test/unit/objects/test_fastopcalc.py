import numpy as np
import unittest

from ..util import BaseCase

from pygsti.objects.polynomial import Polynomial

try:
    from pygsti.objects import fastopcalc
    _FASTOPCALC_LOADED = True
except ImportError:
    _FASTOPCALC_LOADED = False


@unittest.skipUnless(_FASTOPCALC_LOADED, "`pygsti.objects.fastopcalc` not built")
class FastOpCalcTester(BaseCase):
    def test_fast_compact_deriv(self):
        # p = Polynomial({(): 1.0, (1, 2): 2.0, (1, 1, 2): 3.0})
        # v, c = p.compact()
        q = Polynomial({(): 4.0, (1, 1): 5.0, (2, 2, 3): 6.0})
        v, c = q.compact()

        d_v, d_c = fastopcalc.fast_compact_deriv(
            v,
            np.ascontiguousarray(c, complex),
            np.array((1, 2, 3), int)
        )
        self.assertArraysAlmostEqual(d_v, np.array([1, 1, 1, 1, 2, 2, 3, 1, 2, 2, 2]))
        self.assertArraysAlmostEqual(d_c, np.array([10, 12, 6], dtype='complex'))

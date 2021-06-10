import unittest

import numpy as np
from pygsti.baseobjs.polynomial import Polynomial

from pygsti.baseobjs.opcalc import slowopcalc
from ..util import BaseCase

try:
    from pygsti.baseobjs.opcalc import fastopcalc
    _FASTOPCALC_LOADED = True
except ImportError:
    _FASTOPCALC_LOADED = False


class OpCalcBase:
    def test_fast_compact_deriv(self):
        q = Polynomial({(): 4.0, (1, 1): 5.0, (2, 2, 3): 6.0})
        v, c = q.compact()

        d_v, d_c = self.opcalc.compact_deriv(
            v,
            np.ascontiguousarray(c, complex),
            np.array((1, 2, 3), int)
        )
        self.assertArraysAlmostEqual(d_v, np.array([1, 1, 1, 1, 2, 2, 3, 1, 2, 2, 2]))
        self.assertArraysAlmostEqual(d_c, np.array([10, 12, 6], dtype='complex'))


class SlowOpCalcTester(OpCalcBase, BaseCase):
    opcalc = slowopcalc


@unittest.skipUnless(_FASTOPCALC_LOADED, "`pygsti.objects.fastopcalc` not built")
class FastOpCalcTester(OpCalcBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        # bind opcalc during test setup
        # class should still be defined without fastopcalc, so it can be shown as skipped
        cls.opcalc = fastopcalc

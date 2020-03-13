import numpy as np
import unittest
from ..util import BaseCase

from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.objects.replib import slowreplib

try:
    from pygsti.objects.replib import fastreplib
    _FASTREPLIB_LOADED = True
except ImportError:
    _FASTREPLIB_LOADED = False


class ReplibBase(object):
    def test_DMStateRep(self):
        x = np.zeros(4, 'd')
        staterep = self.replib.DMStateRep(x)  # state rep
        # TODO assert correctness

    def test_DMEffectRepDense(self):
        x = np.zeros(4, 'd')
        staterep = self.replib.DMStateRep(x)
        erep = self.replib.DMEffectRepDense(x)
        self.assertAlmostEqual(erep.probability(staterep), 0.0)

    def test_DMOpRepDense(self):
        x = np.zeros(4, 'd')
        staterep = self.replib.DMStateRep(x)
        g = np.zeros((4, 4), 'd')
        grep = self.replib.DMOpRepDense(g)
        staterep2 = grep.acton(staterep)
        self.assertEqual(type(staterep2), self.replib.DMStateRep)


class SlowReplibTester(ReplibBase, BaseCase):
    replib = slowreplib


@unittest.skipUnless(_FASTREPLIB_LOADED, "`pygsti.objects.replib.fastreplib` not built")
class FastReplibTester(ReplibBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        # bind replib during test setup
        # class should still be defined without fastreplib, so it can be shown as skipped
        cls.replib = fastreplib

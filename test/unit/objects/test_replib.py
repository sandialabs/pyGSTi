import numpy as np
from ..util import BaseCase

from pygsti.construction import std1Q_XYI as std


class ReplibBase:
    def test_DMStateRep(self):
        x = np.zeros(4, 'd')
        staterep = self.replib.DMStateRep(x)  # state rep
        # TODO assert correctness

    def test_DMEffectRep_Dense(self):
        x = np.zeros(4, 'd')
        staterep = self.replib.DMStateRep(x)
        erep = self.replib.DMEffectRep_Dense(x)
        self.assertAlmostEqual(erep.probability(staterep), 0.0)

    def test_DMOpRep_Dense(self):
        x = np.zeros(4, 'd')
        staterep = self.replib.DMStateRep(x)
        g = np.zeros((4, 4), 'd')
        grep = self.replib.DMOpRep_Dense(g)
        staterep2 = grep.acton(staterep)
        self.assertEqual(type(staterep2), self.replib.DMStateRep)


class SlowReplibTester(ReplibBase, BaseCase):
    from pygsti.objects import slowreplib as replib


class FastReplibTester(ReplibBase, BaseCase):
    def setUp(self):
        # tests requiring fastreplib will fail if the module is absent
        from pygsti.objects import fastreplib
        self.replib = fastreplib

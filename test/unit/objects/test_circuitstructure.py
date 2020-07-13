from ..util import BaseCase

import pygsti.objects.circuitstructure as cs
from pygsti.objects import Circuit


class LSGermsStructureTester(BaseCase):
    def setUp(self):
        plaquettes = {}
        self.xvals = ['x1', 'x2']
        self.yvals = ['y1', 'y2']
        self.circuit = Circuit("GxGy")
        for x in self.xvals:
            for y in self.yvals:
                plaquettes[(x,y)] = cs.CircuitPlaquette({(minor_x, minor_y): self.circuit
                                                         for minor_x in [0,1] for minor_y in [0,1]})
        self.gss = cs.PlaquetteGridCircuitStructure(plaquettes, self.xvals, self.yvals, 'xlabel', 'ylabel')

    def test_truncate(self):
        self.gss.truncate(xs_to_keep=['x1'])
        self.gss.truncate(ys_to_keep=['y1'])
        # TODO assert correctness

    def test_xvals(self):
        self.assertEqual(self.gss.xs, self.xvals)

    def test_yvals(self):
        self.assertEqual(self.gss.ys, self.yvals)

    def test_get_plaquette(self):
        plaq = self.gss.plaquette('x1', 'y1')
        self.assertTrue(plaq is not None)
        self.assertEqual(len(plaq), 4)

        plaq = self.gss.plaquette('x10', 'y10', empty_if_missing=True)
        self.assertEqual(len(plaq), 0)

    def test_plaquette_iteration(self):
        cnt = 0
        for (x,y), plaq in self.gss.iter_plaquettes():
            cnt += 1
        self.assertEqual(cnt, len(self.xvals) * len(self.yvals))

import numpy as np

from ..util import BaseCase

import pygsti
import pygsti.construction.modelconstruction as mc


class ModelConstructionTester(BaseCase):
    def setUp(self):
        #OK for these tests, since we test user interface?
        #Set Model objects to "strict" mode for testing
        pygsti.objects.ExplicitOpModel._strict = False

    def test_build_basis_gateset(self):
        modelA = pygsti.construction.build_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )
        modelB = pygsti.construction.basis_build_explicit_model(
            [('Q0',)], pygsti.Basis.cast('gm', 4),
            ['Gi', 'Gx', 'Gy'], ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"]
        )
        self.assertAlmostEqual(modelA.frobeniusdist(modelB), 0)
        # TODO assert correctness

    def test_raises_on_bad_parameterization(self):
        with self.assertRaises(ValueError):
            mc.build_operation([(4, 4)], [('Q0', 'Q1')], "X(pi,Q0)", "gm", parameterization="FooBar")

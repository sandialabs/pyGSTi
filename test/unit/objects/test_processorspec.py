import scipy
import numpy as np
import unittest

from ..util import BaseCase

from pygsti.objects import Circuit
from pygsti.objects.processorspec import ProcessorSpec


class ProcessorSpecTester(BaseCase):
    @unittest.skip("REMOVEME")
    def test_construct_with_nonstd_gate_unitary_factory(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            a, = args
            sigmaZ = np.array([[1, 0], [0, -1]], 'd')
            return scipy.linalg.expm(1j * float(a) * sigmaZ)

        ps = ProcessorSpec(nQubits, ('Gx','Gy','Gcnot','Ga'), nonstd_gate_unitaries={'Ga': fn},
                           construct_models=('target','clifford'))

        c = Circuit("Gx:1Ga;0.3:1Gx:1@(0,1)")
        p = ps.models['target'].probabilities(c)

        self.assertAlmostEqual(p['00'], 0.08733219254516078)
        self.assertAlmostEqual(p['01'], 0.9126678074548386)

        c2 = Circuit("Gx:1Ga;0.78539816:1Gx:1@(0,1)")  # a clifford: 0.78539816 = pi/4
        p2 = ps.models['clifford'].probabilities(c2)
        self.assertAlmostEqual(p2['00'], 0.5)
        self.assertAlmostEqual(p2['01'], 0.5)

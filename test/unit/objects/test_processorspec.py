import unittest

import numpy as np
import scipy

from pygsti.processors import QubitProcessorSpec
from pygsti.models import modelconstruction as mc
from pygsti.circuits import Circuit
from ..util import BaseCase


class ProcessorSpecTester(BaseCase):
    @unittest.skip("REMOVEME")
    def test_construct_with_nonstd_gate_unitary_factory(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            a, = args
            sigmaZ = np.array([[1, 0], [0, -1]], 'd')
            return scipy.linalg.expm(1j * float(a) * sigmaZ)

        ps = QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Ga'), nonstd_gate_unitaries={'Ga': fn})
        mdl = mc.create_crosstalk_free_model(ps)

        c = Circuit("Gx:1Ga;0.3:1Gx:1@(0,1)")
        p = mdl.probabilities(c)

        self.assertAlmostEqual(p['00'], 0.08733219254516078)
        self.assertAlmostEqual(p['01'], 0.9126678074548386)

        c2 = Circuit("Gx:1Ga;0.78539816:1Gx:1@(0,1)")  # a clifford: 0.78539816 = pi/4
        p2 = mdl.probabilities(c2)
        self.assertAlmostEqual(p2['00'], 0.5)
        self.assertAlmostEqual(p2['01'], 0.5)

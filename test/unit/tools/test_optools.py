import numpy as np
import scipy

from ..util import BaseCase

import pygsti.tools.optools as ot
import pygsti.tools.basistools as bt


class OpToolsTests(BaseCase):
    def test_unitary_to_pauligate(self):
        theta = np.pi
        ex = 1j * theta * bt.sigmax / 2
        U = scipy.linalg.expm(ex)
        # U is 2x2 unitary matrix operating on single qubit in [0,1] basis (X(pi) rotation)

        op = ot.unitary_to_pauligate(U)
        op_ans = np.array([[ 1.,  0.,  0.,  0.],
                           [ 0.,  1.,  0.,  0.],
                           [ 0.,  0., -1.,  0.],
                           [ 0.,  0.,  0., -1.]], 'd')
        self.assertArraysAlmostEqual(op, op_ans)

        U_2Q = np.identity(4, 'complex'); U_2Q[2:, 2:] = U
        # U_2Q is 4x4 unitary matrix operating on isolated two-qubit space (CX(pi) rotation)

        op_2Q = ot.unitary_to_pauligate(U_2Q)
        # TODO assert correctness

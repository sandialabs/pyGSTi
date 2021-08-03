import numpy as np

import pygsti.tools.gatetools as gc
from ..util import BaseCase


class GateConstructionTester(BaseCase):
    def test_single_qubit_gate_matrix(self):
        expected = np.array([[1.00000000e+00, 2.77555756e-16, -2.28983499e-16, 0.00000000e+00],
                             [-3.53885261e-16, -8.09667193e-01, 5.22395269e-01, -2.67473774e-01],
                             [-3.92523115e-17, 5.22395269e-01, 8.49200550e-01, 7.72114534e-02],
                             [1.66533454e-16, 2.67473774e-01, -7.72114534e-02, -9.60466643e-01]]
                            )
        mx = gc.single_qubit_gate(24.0, 83.140134, 0.0000)
        self.assertArraysAlmostEqual(expected, mx)

    def test_two_qubit_gate_mx(self):
        gate = gc.two_qubit_gate()
        expected = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ]])
        self.assertArraysAlmostEqual(gate, expected)

    def test_two_qubit_gate(self):
        # For independent operations, can check directly against tensored paulis
        for op1, op2 in ['ii', 'xi', 'yi', 'zi', 'ix', 'iy', 'iz']:
            coeffs = {'hx': 0, 'hy': 0, 'hz': 0}
            if op1 != 'i':
                coeffs[f'h{op1}'] = 1
            gate1 = gc.single_qubit_gate(**coeffs)

            coeffs = {'hx': 0, 'hy': 0, 'hz': 0}
            if op2 != 'i':
                coeffs[f'h{op2}'] = 1
            gate2 = gc.single_qubit_gate(**coeffs)

            composed_gate = np.kron(gate1, gate2)

            arg = ''.join([op1, op2])
            coeffs = {arg: 1}
            full_gate = gc.two_qubit_gate(**coeffs)

            self.assertArraysAlmostEqual(composed_gate, full_gate)

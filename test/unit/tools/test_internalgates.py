import numpy as np

# from pygsti.extras import rb
from pygsti.tools import internalgates, optools as ot, basistools as bt
from ..util import BaseCase


class InternalGatesTester(BaseCase):

    def test_internalgate_definitions(self):
        # TODO is this test needed?
        self.skipTest("RB analysis is known to be broken.  Skip tests until it gets fixed.")
        std_unitaries = internalgates.standard_gatename_unitaries()
        std_quil = internalgates.standard_gatenames_quil_conversions()
        std_quil = internalgates.standard_gatenames_openqasm_conversions()

        # Checks the standard Clifford gate unitaries agree with the Clifford group unitaries.
        group = rb.group.construct_1q_clifford_group()
        for key in group.labels:
            self.assertLess(np.sum(abs(np.array(group.matrix(key))
                                       - ot.unitary_to_pauligate(std_unitaries[key]))), 10**-10)

    def test_u3_unitary_generator(self):
        # Checks the u3 unitary generator runs
        u = internalgates.qasm_u3(0., 0., 0., output='unitary')
        sup = internalgates.qasm_u3(0., 0., 0., output='superoperator')
        sup_u = ot.std_process_mx_to_unitary(bt.change_basis(sup, 'pp', 'std')) # Backtransform to unitary
        self.assertArraysAlmostEqual(u, sup_u)

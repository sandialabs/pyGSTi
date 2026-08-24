import numpy as np

from pygsti.tools import internalgates, optools as ot, basistools as bt, group
from ..util import BaseCase


class InternalGatesTester(BaseCase):

    def test_internalgate_definitions(self):
        # Checks the standard Clifford gate unitaries agree with the Clifford group unitaries.
        std_unitaries = internalgates.standard_gatename_unitaries()
        g = group.construct_1q_clifford_group()
        assert g.labels is not None
        for key in g.labels:
            self.assertLess(np.sum(abs(np.array(g.matrix(key))
                                       - ot.unitary_to_pauligate(std_unitaries[str(key)]))), 10**-10)

    def test_u3_unitary_generator(self):
        # Checks the u3 unitary generator runs
        u = internalgates.qasm_u3(0., 0., 0., output='unitary')
        sup = internalgates.qasm_u3(0., 0., 0., output='superoperator')
        sup_u = ot.std_process_mx_to_unitary(bt.change_basis(sup, 'pp', 'std')) # Backtransform to unitary
        self.assertArraysAlmostEqual(u, sup_u)

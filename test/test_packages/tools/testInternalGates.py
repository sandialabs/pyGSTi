from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np

import pygsti
from pygsti import unitary_to_pauligate
from pygsti.extras import rb
from pygsti.tools import internalgates
#Move to RB?

class InternalGatesTestCase(BaseTestCase):

    def test_internalgates(self):
    
        # Checks we can get the dicts this file generates.
        std_unitaries = internalgates.get_standard_gatename_unitaries()
        std_quil = internalgates.get_standard_gatenames_quil_conversions()
        std_quil = internalgates.get_standard_gatenames_openqasm_conversions()
        
        # Checks the standard Clifford gate unitaries agree with the Clifford group unitaries.
        group = rb.group.construct_1Q_Clifford_group()
        for key in group.labels:
            self.assertLess(np.sum(abs(np.array(group.get_matrix(key))-unitary_to_pauligate(std_unitaries[key]))), 10**-10)
            
        # Checks the u3 unitary generator runs
        u = internalgates.qasm_u3(0., 0., 0., output='unitary')
        sup = internalgates.qasm_u3(0., 0., 0., output='superoperator')

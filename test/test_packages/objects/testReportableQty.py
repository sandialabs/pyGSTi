import unittest
import pygsti
import numpy as np
import warnings
import pickle
import os
import copy

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some gatesets that get used in this file and in testGateSets2.py
class ReportableQtyTestCase(BaseTestCase):

    def setUp(self):
        super(ReportableQtyTestCase, self).setUp()

    def test_reportable_qty_methods(self):
        val_errbar_tups = [ (0.0, 0.01),
                            (np.ones(4,'d'), 0.1*np.ones(4,'d')),
                            (np.identity(4,'d'), 0.1*np.ones((4,4),'d')) ]
        for val,eb in val_errbar_tups:
            r = pygsti.objects.reportableqty.ReportableQty(val)
            r2 = pygsti.objects.reportableqty.ReportableQty(val, eb)

            s = str(r)
            s = repr(r)

            r = r + 1.0
            r2 = r2 + 1.0

            r = r*2.0
            r2 = r2*2.0

            r = r/2.0
            r2 = r2/2.0

            r_cpy = copy.copy(r)
            r_dcpy = copy.deepcopy(r)

            r.log()
            r2.log()
            r.scale(2.0) #like *=
            r2.scale(2.0)

            r.real()
            r2.real()
            r.imag()
            r2.imag()

            r.absdiff(1.0, separate_re_im=True)
            r.absdiff(1.0, separate_re_im=False)
            r2.absdiff(1.0, separate_re_im=True)
            r2.absdiff(1.0, separate_re_im=False)

            r.infidelity_diff(1.0)
            r2.infidelity_diff(1.0)

            r.mod(1.0)
            r2.mod(1.0)

            if hasattr(val,'shape') and len(val.shape) == 2:
                r.hermitian_to_real()
                r2.hermitian_to_real()

            self.assertFalse(r.has_eb())
            self.assertTrue(r2.has_eb())

            r.get_value()
            r2.get_value()
            r.get_err_bar()
            r2.get_err_bar()
            r.get_value_and_err_bar()
            r2.get_value_and_err_bar()

    def test_non_hermitian_fail(self):
        non_herm_mx = np.array([[1,1j],
                                [0,1]], 'complex')
        r = pygsti.objects.reportableqty.ReportableQty(non_herm_mx)
        with self.assertRaises(ValueError):
            r.hermitian_to_real()


if __name__ == '__main__':
    unittest.main(verbosity=2)

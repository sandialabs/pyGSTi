from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.construction import std1Q_XYI as std
import pygsti
import unittest


class Chi2LogLTestCase(BaseTestCase):

    def test_chi2_terms(self):
        gs = pygsti.io.load_gateset(compare_files + '/analysis.gateset')
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + '/analysis.dataset')
        terms = pygsti.chi2_terms(ds, gs)

    def test_chi2_fn(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")
        chi2, grad = pygsti.chi2(ds, std.gs_target, returnGradient=True)
        pygsti.chi2(ds, std.gs_target, returnHessian=True)

        pygsti.gate_string_chi2( ('Gx',), ds, std.gs_target)
        pygsti.chi2fn_2outcome( N=100, p=0.5, f=0.6)
        pygsti.chi2fn_2outcome_wfreqs( N=100, p=0.5, f=0.6)
        pygsti.chi2fn( N=100, p=0.5, f=0.6)
        pygsti.chi2fn_wfreqs( N=100, p=0.5, f=0.6)

        with self.assertRaises(ValueError):
            pygsti.chi2(ds, std.gs_target, useFreqWeightedChiSq=True) #no impl yet

        # Memory tests

        with self.assertRaises(MemoryError):
            pygsti.chi2(ds, std.gs_target, memLimit=0) # No memory for you
        pygsti.chi2(ds, std.gs_target, memLimit=100000)


if __name__ == '__main__':
    unittest.main(verbosity=2)

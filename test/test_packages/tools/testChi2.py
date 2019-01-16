from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.construction import std1Q_XYI as std
import pygsti
import unittest


class Chi2LogLTestCase(BaseTestCase):

    def test_chi2_terms(self):
        mdl = pygsti.io.load_model(compare_files + '/analysis.model')
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + '/analysis.dataset%s' % self.versionsuffix)
        terms = pygsti.chi2_terms(mdl, ds)

    def test_chi2_fn(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset%s" % self.versionsuffix)
        chi2, grad = pygsti.chi2(std.target_model(), ds, returnGradient=True)
        pygsti.chi2(std.target_model(), ds, returnHessian=True)

        #pygsti.gate_string_chi2( ('Gx',), ds, std.target_model())
        pygsti.chi2fn_2outcome( N=100, p=0.5, f=0.6)
        pygsti.chi2fn_2outcome_wfreqs( N=100, p=0.5, f=0.6)
        pygsti.chi2fn( N=100, p=0.5, f=0.6)
        pygsti.chi2fn_wfreqs( N=100, p=0.5, f=0.6)

        with self.assertRaises(ValueError):
            pygsti.chi2(std.target_model(), ds, useFreqWeightedChiSq=True) #no impl yet

        # Memory tests

        with self.assertRaises(MemoryError):
            pygsti.chi2(std.target_model(), ds, memLimit=0) # No memory for you
        pygsti.chi2(std.target_model(), ds, memLimit=100000)


if __name__ == '__main__':
    unittest.main(verbosity=2)

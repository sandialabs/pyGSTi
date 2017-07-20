import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std
import os

from ..testutils import BaseTestCase, compare_files, temp_files

class TestMetrics(BaseTestCase):

    def setUp(self):
        super(TestMetrics, self).setUp()

        self.gateset = std.gs_target
        self.gateset_dep = self.gateset.depolarize(gate_noise=0.05, spam_noise=0)
        self.gateset_rdm = self.gateset_dep.kick( absmag=0.1 )
        self.gatestrings = [ (), ('Gx',), ('Gx','Gy') ]

        dataset_txt = \
"""## Columns = plus count, count total
{} 0 100
Gx 10 90
GxGy 40 60
Gx^4 20 90
"""
        with open(temp_files + "/MetricsDataset.txt","w") as output:
            output.write(dataset_txt)
        self.ds = pygsti.io.load_dataset(temp_files + "/MetricsDataset.txt")

        chi2, chi2Hessian = pygsti.chi2(self.ds, self.gateset,
                                        returnHessian=True)
        self.ci = pygsti.obj.ConfidenceRegion(self.gateset, chi2Hessian, 95.0,
                                             hessianProjection="std")

        chi2, chi2Hessian = pygsti.chi2(self.ds, self.gateset_dep,
                                        returnHessian=True)
        self.ci_dep = pygsti.obj.ConfidenceRegion(self.gateset_dep, chi2Hessian, 95.0,
                                             hessianProjection="std")

        chi2, chi2Hessian = pygsti.chi2(self.ds, self.gateset_rdm,
                                        returnHessian=True)
        self.ci_rdm = pygsti.obj.ConfidenceRegion(self.gateset_rdm, chi2Hessian, 95.0,
                                             hessianProjection="std")


    def test_dataset_qtys(self):
        names = ('gate string', 'gate string index', 'gate string length', 'count total',
                 'Exp prob(plus)', 'Exp count(plus)', 'max logl', 'number of gate strings')
        qtys = pygsti.report.compute_dataset_qtys(names, self.ds, self.gatestrings)

        possible_names = pygsti.report.compute_dataset_qty(None, self.ds, self.gatestrings)
        qtys = pygsti.report.compute_dataset_qtys(possible_names, self.ds, self.gatestrings)
        qty = pygsti.report.compute_dataset_qty('max logl', self.ds, self.gatestrings)
        #TODO: test quantities

    def test_gateset_qtys(self):
        names = pygsti.report.compute_gateset_qty(None, self.gateset)
        qtys = pygsti.report.compute_gateset_qtys(names, self.gateset, self.ci)
        qty = pygsti.report.compute_gateset_qty("Gx eigenvalues", self.gateset, self.ci)

        qtys_dep = pygsti.report.compute_gateset_qtys(names, self.gateset_dep, self.ci_dep)
        qtys_rdm = pygsti.report.compute_gateset_qtys(names, self.gateset_rdm, self.ci_rdm)
        #TODO: test quantities

    def test_gateset_dataset_qtys(self):
        names = pygsti.report.compute_gateset_dataset_qty(None, self.gateset, self.ds, self.gatestrings)
        qtys  = pygsti.report.compute_gateset_dataset_qtys(names, self.gateset, self.ds, self.gatestrings)
        qty  = pygsti.report.compute_gateset_dataset_qty("chi2", self.gateset, self.ds, self.gatestrings)

        qtys_dep  = pygsti.report.compute_gateset_dataset_qtys(names, self.gateset_dep, self.ds, self.gatestrings)
        qtys_rdm  = pygsti.report.compute_gateset_dataset_qtys(names, self.gateset_rdm, self.ds, self.gatestrings)
        #TODO: test quantities

    def test_gateset_gateset_qtys(self):
        names = pygsti.report.compute_gateset_gateset_qty(None, self.gateset, self.gateset_dep)
        qtys = pygsti.report.compute_gateset_gateset_qtys(names, self.gateset, self.gateset_dep, self.ci)
        qty = pygsti.report.compute_gateset_gateset_qty("Gx fidelity", self.gateset, self.gateset_dep, self.ci)

        qtys2 = pygsti.report.compute_gateset_gateset_qtys(names, self.gateset, self.gateset_rdm, self.ci)
        qtys3 = pygsti.report.compute_gateset_gateset_qtys(names, self.gateset_dep, self.gateset_rdm, self.ci_dep)
        #TODO: test quantities

if __name__ == "__main__":
    unittest.main(verbosity=2)

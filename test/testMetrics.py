import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std
import numpy as np


class MetricsTestCase(unittest.TestCase):

    def setUp(self):
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = True

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

        open("temp_test_files/MetricsDataset.txt","w").write(dataset_txt)
        self.ds = pygsti.io.load_dataset("temp_test_files/MetricsDataset.txt")


class TestMetrics(MetricsTestCase):
    
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
        qtys = pygsti.report.compute_gateset_qtys(names, self.gateset)
        qty = pygsti.report.compute_gateset_qty("Gx eigenvalues", self.gateset)

        qtys_dep = pygsti.report.compute_gateset_qtys(names, self.gateset_dep)
        qtys_rdm = pygsti.report.compute_gateset_qtys(names, self.gateset_rdm)
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
        qtys = pygsti.report.compute_gateset_gateset_qtys(names, self.gateset, self.gateset_dep)
        qty = pygsti.report.compute_gateset_gateset_qty("Gx fidelity", self.gateset, self.gateset_dep)

        qtys2 = pygsti.report.compute_gateset_gateset_qtys(names, self.gateset, self.gateset_rdm)
        qtys3 = pygsti.report.compute_gateset_gateset_qtys(names, self.gateset_dep, self.gateset_rdm)
        #TODO: test quantities

if __name__ == "__main__":
    unittest.main(verbosity=2)

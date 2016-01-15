import unittest
import GST
from GSTCommons import Std1Q_XYI as Std
import numpy as np

class MetricsTestCase(unittest.TestCase):

    def setUp(self):
        self.gateset = Std.gs_target
        self.gateset_dep = GST.GateSetTools.depolarize_gateset(self.gateset, noise=0.05)
        self.gateset_rdm = GST.GateSetTools.kick_gateset(self.gateset_dep, absmag=0.1)
        self.gatestrings = [ (), ('Gx',), ('Gx','Gy') ]

        dataset_txt = \
"""## Columns = plus count, count total
{} 0 100
Gx 10 90
GxGy 40 60
Gx^4 20 90
"""

        open("temp_test_files/MetricsDataset.txt","w").write(dataset_txt)
        self.ds = GST.load_dataset("temp_test_files/MetricsDataset.txt")


class TestMetrics(MetricsTestCase):
    
    def test_dataset_qtys(self):
        names = ('gate string', 'gate string index', 'gate string length', 'count total',
                 'Exp prob(plus)', 'Exp count(plus)', 'max logl', 'number of gate strings')
        qtys = GST.ComputeReportables.compute_dataset_qtys(names, self.ds, self.gatestrings)

        possible_names = GST.ComputeReportables.compute_dataset_qty(None, self.ds, self.gatestrings)
        qtys = GST.ComputeReportables.compute_dataset_qtys(possible_names, self.ds, self.gatestrings)
        qty = GST.ComputeReportables.compute_dataset_qty('max logl', self.ds, self.gatestrings)
        #TODO: test quantities



    def test_gateset_qtys(self):
        names = GST.ComputeReportables.compute_gateset_qty(None, self.gateset)
        qtys = GST.ComputeReportables.compute_gateset_qtys(names, self.gateset)
        qty = GST.ComputeReportables.compute_gateset_qty("Gx eigenvalues", self.gateset)

        qtys_dep = GST.ComputeReportables.compute_gateset_qtys(names, self.gateset_dep)
        qtys_rdm = GST.ComputeReportables.compute_gateset_qtys(names, self.gateset_rdm)
        #TODO: test quantities

    def test_gateset_dataset_qtys(self):
        names = GST.ComputeReportables.compute_gateset_dataset_qty(None, self.gateset, self.ds, self.gatestrings)
        qtys  = GST.ComputeReportables.compute_gateset_dataset_qtys(names, self.gateset, self.ds, self.gatestrings)
        qty  = GST.ComputeReportables.compute_gateset_dataset_qty("chi2", self.gateset, self.ds, self.gatestrings)

        qtys_dep  = GST.ComputeReportables.compute_gateset_dataset_qtys(names, self.gateset_dep, self.ds, self.gatestrings)
        qtys_rdm  = GST.ComputeReportables.compute_gateset_dataset_qtys(names, self.gateset_rdm, self.ds, self.gatestrings)
        #TODO: test quantities

    def test_gateset_gateset_qtys(self):
        names = GST.ComputeReportables.compute_gateset_gateset_qty(None, self.gateset, self.gateset_dep)
        qtys = GST.ComputeReportables.compute_gateset_gateset_qtys(names, self.gateset, self.gateset_dep)
        qty = GST.ComputeReportables.compute_gateset_gateset_qty("Gx fidelity", self.gateset, self.gateset_dep)

        qtys2 = GST.ComputeReportables.compute_gateset_gateset_qtys(names, self.gateset, self.gateset_rdm)
        qtys3 = GST.ComputeReportables.compute_gateset_gateset_qtys(names, self.gateset_dep, self.gateset_rdm)
        #TODO: test quantities

if __name__ == "__main__":
    unittest.main(verbosity=2)

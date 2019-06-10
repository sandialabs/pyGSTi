from ..util import BaseCase

from pygsti.objects import DataSet
from pygsti.construction import std1Q_XYI, std2Q_XXYYII, std2Q_XYICNOT
import pygsti.construction.nqnoiseconstruction as nc


class KCoverageTester(BaseCase):
    def test_kcoverage(self):
        # TODO optimize
        n = 10  # nqubits
        k = 4  # number of "labels" needing distribution
        rows = nc.get_kcoverage_template(n, k, verbosity=2)
        nc.check_kcoverage_template(rows, n, k, verbosity=1)


class StdModuleBase:
    def test_upgrade_to_multiq_module(self):
        newmod = nc.stdmodule_to_smqmodule(self.std)
        opLabels = list(newmod.target_model().operations.keys())
        germStrs = newmod.germs

        for gl in opLabels:
            if gl != "Gi" and gl != ():
                self.assertGreater(len(gl.sslbls), 0)

        for str in germStrs:
            for gl in str:
                if gl != "Gi" and gl != ():
                    self.assertGreater(len(gl.sslbls), 0)


class Std1Q_XYITester(StdModuleBase, BaseCase):
    std = std1Q_XYI


class Std2Q_XXYYIITester(StdModuleBase, BaseCase):
    std = std2Q_XXYYII


class Std2Q_XYICNOTTester(StdModuleBase, BaseCase):
    std = std2Q_XYICNOT

    def test_upgrade_dataset(self):
        #Test upgrade of 2Q dataset
        ds = DataSet(outcomeLabels=('00', '01', '10', '11'))
        ds.get_outcome_labels()
        ds.add_count_dict(('Gix',), {'00': 90, '10': 10})
        ds.add_count_dict(('Giy',), {'00': 80, '10': 20})
        ds.add_count_dict(('Gxi',), {'00': 55, '10': 45})
        ds.add_count_dict(('Gyi',), {'00': 40, '10': 60})

        ds2 = ds.copy()
        newmod = nc.stdmodule_to_smqmodule(self.std)
        newmod.upgrade_dataset(ds2)
        self.assertEqual(ds2[(('Gx', 0),)].counts, {('00',): 55, ('10',): 45})
        self.assertEqual(ds2[(('Gy', 0),)].counts, {('00',): 40, ('10',): 60})
        self.assertEqual(ds2[(('Gx', 1),)].counts, {('00',): 90, ('10',): 10})
        self.assertEqual(ds2[(('Gy', 1),)].counts, {('00',): 80, ('10',): 20})

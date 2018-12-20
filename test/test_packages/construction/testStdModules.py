import unittest
import pygsti
import numpy as np

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some models that get used in this file and in testGateSets2.py
class StdModuleTestCase(BaseTestCase):

    def setUp(self):
        super(StdModuleTestCase, self).setUp()

    def test_upgrade_to_multiq_module(self):
        from pygsti.construction import std1Q_XYI
        from pygsti.construction import std2Q_XXYYII
        from pygsti.construction import std2Q_XYICNOT

        for std in (std1Q_XYI, std2Q_XXYYII, std2Q_XYICNOT):
            newmod = pygsti.construction.stdmodule_to_smqmodule(std)
            opLabels = list(newmod.target_model().operations.keys())
            germStrs = newmod.germs

            for gl in opLabels:
                if gl != "Gi": 
                    self.assertGreater(len(gl.sslbls),0)

            for str in germStrs:
                for gl in str:
                    if gl != "Gi": 
                        self.assertGreater(len(gl.sslbls),0)
                    
        #Test upgrade of 2Q dataset
        ds = pygsti.obj.DataSet(outcomeLabels=('00','01','10','11'))
        ds.get_outcome_labels()
        ds.add_count_dict( ('Gix',), {'00': 90,'10': 10} )
        ds.add_count_dict( ('Giy',), {'00': 80,'10': 20} )
        ds.add_count_dict( ('Gxi',), {'00': 55,'10': 45} )
        ds.add_count_dict( ('Gyi',), {'00': 40,'10': 60} )
        
        ds2 = ds.copy()
        newmod.upgrade_dataset(ds2)
        self.assertEqual( ds2[(('Gx',0),)].counts, {('00',):55, ('10',):45} )
        self.assertEqual( ds2[(('Gy',0),)].counts, {('00',):40, ('10',):60} )
        self.assertEqual( ds2[(('Gx',1),)].counts, {('00',):90, ('10',):10} )
        self.assertEqual( ds2[(('Gy',1),)].counts, {('00',):80, ('10',):20} )

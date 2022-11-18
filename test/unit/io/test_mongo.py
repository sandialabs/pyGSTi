import unittest
from ..util import BaseCase

import pygsti
from pygsti.modelpacks import smq1Q_XYI as std

import collections
collections.Callable = collections.abc.Callable

try:
    import pymongo
    PYMONGO_IMPORTED = True
except ImportError:
    pymongo = None
    PYMONGO_IMPORTED = False


class MongoDBTester(BaseCase):

    def setUp(self):
        if pymongo is not None:
            myclient = pymongo.MongoClient("mongodb://localhost:27017/")
            self.mydb = myclient["pygsti_unittest_database"]
        else:
            self.mydb = None

    @unittest.skipUnless(PYMONGO_IMPORTED, "pymongo not installed")
    def test_mongodb_edesign(self):
        edesign = std.create_gst_experiment_design(1)
        #len(edesign.all_circuits_needing_data)  # 92

        bRemoved = pygsti.protocols.ExperimentDesign.remove_from_mongodb(self.mydb, "my_test_edesign")
        self.assertFalse(bRemoved)

        edesign.write_to_mongodb(self.mydb, "my_test_edesign")
        edesign2 = pygsti.io.read_edesign_from_mongodb(self.mydb, "my_test_edesign")
        self.assertEqual(len(edesign2.all_circuits_needing_data), len(edesign.all_circuits_needing_data))

        bRemoved = pygsti.protocols.ExperimentDesign.remove_from_mongodb(self.mydb, "my_test_edesign")
        self.assertTrue(bRemoved)

    @unittest.skipUnless(PYMONGO_IMPORTED, "pymongo not installed")
    def test_mongodb_data(self):
        edesign = std.create_gst_experiment_design(1)
        datagen_mdl = std.target_model().depolarize(op_noise=0.03, spam_noise=0.05)
        ds = pygsti.data.simulate_data(datagen_mdl, edesign, 1000, seed=123456)
        data = pygsti.protocols.ProtocolData(edesign, ds)

        bRemoved = pygsti.protocols.ProtocolData.remove_from_mongodb(self.mydb, "my_test_data")
        self.assertFalse(bRemoved)

        data.write_to_mongodb(self.mydb, "my_test_data")
        data2 = pygsti.io.read_data_from_mongodb(self.mydb, "my_test_data")
        self.assertTrue(isinstance(data2.edesign._loaded_from[0], pymongo.database.Database))
        self.assertEqual(len(data.dataset), len(data2.dataset))

        bRemoved = pygsti.protocols.ProtocolData.remove_from_mongodb(self.mydb, "my_test_data")
        self.assertTrue(bRemoved)

    @unittest.skipUnless(PYMONGO_IMPORTED, "pymongo not installed")
    def test_mongodb_results(self):
        edesign = std.create_gst_experiment_design(1)
        datagen_mdl = std.target_model().depolarize(op_noise=0.03, spam_noise=0.05)
        ds = pygsti.data.simulate_data(datagen_mdl, edesign, 1000, seed=123456)
        data = pygsti.protocols.ProtocolData(edesign, ds)

        gst = pygsti.protocols.StandardGST('full TP')
        results = gst.run(data)

        bRemoved = pygsti.protocols.ProtocolResultsDir.remove_from_mongodb(self.mydb, "my_test_results")
        self.assertFalse(bRemoved)

        results.write_to_mongodb(self.mydb, "my_test_results")
        resultsdir2 = pygsti.io.read_results_from_mongodb(self.mydb, 'my_test_results')
        results2 = resultsdir2.for_protocol['StandardGST']

        self.assertTrue(isinstance(results2.estimates['full TP'].models['target'], pygsti.models.Model))
        bRemoved = pygsti.protocols.ProtocolResultsDir.remove_from_mongodb(self.mydb, "my_test_results")
        self.assertTrue(bRemoved)

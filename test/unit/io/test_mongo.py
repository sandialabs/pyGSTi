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

        edesign_id = edesign.write_to_mongodb(self.mydb)
        edesign2 = pygsti.io.read_edesign_from_mongodb(self.mydb, edesign_id)
        self.assertEqual(len(edesign2.all_circuits_needing_data), len(edesign.all_circuits_needing_data))

        edesign2.remove_me_from_mongodb(self.mydb)

    @unittest.skipUnless(PYMONGO_IMPORTED, "pymongo not installed")
    def test_mongodb_data(self):
        edesign = std.create_gst_experiment_design(1)
        datagen_mdl = std.target_model().depolarize(op_noise=0.03, spam_noise=0.05)
        ds = pygsti.data.simulate_data(datagen_mdl, edesign, 1000, seed=123456)
        data = pygsti.protocols.ProtocolData(edesign, ds)

        data_id = data.write_to_mongodb(self.mydb)
        data2 = pygsti.io.read_data_from_mongodb(self.mydb, data_id)
        self.assertEqual(len(data.dataset), len(data2.dataset))

        pygsti.protocols.ProtocolData.remove_from_mongodb(self.mydb, data_id)

    @unittest.skipUnless(PYMONGO_IMPORTED, "pymongo not installed")
    def test_mongodb_results(self):
        edesign = std.create_gst_experiment_design(1)
        datagen_mdl = std.target_model().depolarize(op_noise=0.03, spam_noise=0.05)
        ds = pygsti.data.simulate_data(datagen_mdl, edesign, 1000, seed=123456)
        data = pygsti.protocols.ProtocolData(edesign, ds)

        gst = pygsti.protocols.StandardGST('full TP')
        results = gst.run(data)

        results_id = results.write_to_mongodb(self.mydb)
        results2 = pygsti.io.read_results_from_mongodb(self.mydb, results_id)

        self.assertTrue(isinstance(results2.estimates['full TP'].models['target'], pygsti.models.Model))
        results2.remove_me_from_mongodb(self.mydb)

import pickle
from collections import OrderedDict

import numpy as np

from pygsti.circuits import Circuit
from pygsti.data import MultiDataSet, DataSet
from ..util import BaseCase

# module-level test fixtures
gstrInds = OrderedDict([(Circuit(('Gx',)), slice(0, 2)),
                        (Circuit(('Gx', 'Gy')), slice(2, 4)),
                        (Circuit(('Gy',)), slice(4, 6))])
olInds = OrderedDict([('0', 0), ('1', 1)])

ds1_oli = np.array([0, 1] * 3, 'i')  # 3 operation sequences * 2 outcome labels
ds1_time = np.zeros(6, 'd')
ds1_rep = 10 * np.ones(6, 'i')

ds2_oli = np.array([0, 1] * 3, 'i')  # 3 operation sequences * 2 outcome labels
ds2_time = np.zeros(6, 'd')
ds2_rep = 5 * np.ones(6, 'i')

mds_oli = OrderedDict([('ds1', ds1_oli), ('ds2', ds2_oli)])
mds_time = OrderedDict([('ds1', ds1_time), ('ds2', ds2_time)])
mds_rep = OrderedDict([('ds1', ds1_rep), ('ds2', ds2_rep)])



class MultiDataSetTester(BaseCase):
    def test_construct_with_outcome_label_indices(self):
        mds = MultiDataSet(mds_oli, mds_time, mds_rep, circuit_indices=gstrInds,
                           outcome_label_indices=olInds)
        # TODO assert correctness

    def test_construct_with_no_data(self):
        mds4 = MultiDataSet(outcome_labels=['0', '1'])
        mds5 = MultiDataSet()
        # TODO assert correctness


class MultiDataSetMethodBase(object):
    def _assert_datasets_equal(self, a, b):
        for a_row, b_row in zip(a, b):
            for a_element, b_element in zip(a_row, b_row):
                self.assertEqual(a_element, b_element)

    def test_add_dataset(self):
        expected_length = len(self.mds) + 1
        expected_keys = self.mds.keys() + ['newDS']
        ds = DataSet(outcome_labels=['0', '1'])
        ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        ds.add_count_dict(('Gx', 'Gy'), {'0': 20, '1': 80})
        ds.add_count_dict(('Gy',), {'0': 20, '1': 80})
        ds.done_adding_data()
        self.mds['newDS'] = ds
        self.assertTrue('newDS' in self.mds)
        self.assertEqual(len(self.mds), expected_length)
        self.assertEqual(self.mds.keys(), expected_keys)

    def test_indexing(self):
        labels, datasets = tuple(zip(*self.mds.items()))
        self.assertEqual(labels, tuple(self.mds))
        self.assertEqual(labels, tuple(self.mds.keys()))
        for a, b in zip(datasets, self.mds.values()):
            self._assert_datasets_equal(a, b)

    def test_get_outcome_labels(self):
        labels = self.mds.outcome_labels
        # TODO assert correctness

    def test_get_datasets_aggregate(self):
        keyset = self.mds.keys()
        sumDS = self.mds.datasets_aggregate(*keyset)
        # TODO assert correctness

    def test_to_string(self):
        mds_str = str(self.mds)
        # TODO assert correctness

    def test_copy(self):
        mds_copy = self.mds.copy()
        # TODO assert correctness

    def test_pickle(self):
        s = pickle.dumps(self.mds)
        mds_unpickle = pickle.loads(s)
        # TODO assert correctness

    def test_get_datasets_aggregate_raises_on_unknown_name(self):
        with self.assertRaises(ValueError):
            self.mds.datasets_aggregate('ds1', 'foobar')

    def test_add_dataset_raises_on_gate_mismatch(self):
        ds = DataSet(outcome_labels=['0', '1'])  # different operation sequences
        ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        ds.done_adding_data()
        with self.assertRaises(ValueError):
            self.mds['newDS'] = ds

    def test_add_dataset_raises_on_nonstatic_dataset(self):
        ds = DataSet(outcome_labels=['0', '1'])  # different operation sequences
        ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        with self.assertRaises(ValueError):
            self.mds['newDS'] = ds


class MultiDataSetInstanceTester(MultiDataSetMethodBase, BaseCase):
    def setUp(self):
        self.mds = MultiDataSet(mds_oli, mds_time, mds_rep, circuit_indices=gstrInds,
                                outcome_labels=['0', '1'])


class MultiDataSetNoRepInstanceTester(MultiDataSetMethodBase, BaseCase):
    def setUp(self):
        self.mds = MultiDataSet(mds_oli, mds_time, None, circuit_indices=gstrInds,
                                outcome_labels=['0', '1'])

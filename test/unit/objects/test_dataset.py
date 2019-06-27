import numpy as np
from collections import OrderedDict
import pickle

from ..util import BaseCase

import pygsti.construction as pc
from pygsti.objects import DataSet, labeldicts as ld, Circuit


class DataSetTester(BaseCase):
    def setUp(self):
        self.gstrs = [('Gx',), ('Gx', 'Gy'), ('Gy',)]
        self.gstrInds = OrderedDict([(('Gx',), 0), (('Gx', 'Gy'), 1), (('Gy',), 2)])
        self.gstrInds_static = OrderedDict([(Circuit(('Gx',)), slice(0, 2)),
                                            (Circuit(('Gx', 'Gy')), slice(2, 4)),
                                            (Circuit(('Gy',)), slice(4, 6))])
        self.olInds = OrderedDict([('0', 0), ('1', 1)])

        oli = np.array([0, 1], 'i')
        self.oli_static = np.array([0, 1] * 3, 'd')  # 3 operation sequences * 2 outcome labels each
        self.time_static = np.zeros((6,), 'd')
        self.reps_static = 10 * np.ones((6,), 'd')

        self.oli_nonstc = [oli, oli, oli]  # each item has num_outcomes elements
        self.time_nonstc = [np.zeros(2, 'd'), np.zeros(2, 'd'), np.zeros(2, 'd')]
        self.reps_nonstc = [10 * np.ones(2, 'i'), 10 * np.ones(2, 'i'), 10 * np.ones(2, 'i')]

    def test_construct_empty_dataset(self):
        dsEmpty = DataSet(outcomeLabels=['0', '1'])
        dsEmpty.done_adding_data()
        # TODO assert correctness

    def test_initialize_by_index(self):
        ds1 = DataSet(outcomeLabels=['0', '1'])
        ds1.add_count_dict(('Gx',), {'0': 10, '1': 90})
        ds2 = DataSet(outcomeLabels=['0', '1'])
        ds2[('Gx',)] = {'0': 10, '1': 90}
        # TODO assert correctness

    def test_construct_from_list(self):
        ds = DataSet(self.oli_nonstc, self.time_nonstc, self.reps_nonstc,
                     circuits=self.gstrs, outcomeLabels=['0', '1'])
        # TODO assert correctness

    def test_construct_from_map(self):
        ds = DataSet(self.oli_nonstc[:], self.time_nonstc[:], self.reps_nonstc[:],
                     circuitIndices=self.gstrInds, outcomeLabelIndices=self.olInds)
        # TODO assert correctness

    def test_construct_static(self):
        ds = DataSet(self.oli_nonstc, self.time_nonstc, self.reps_nonstc,
                     circuitIndices=self.gstrInds_static, outcomeLabels=['0', '1'], bStatic=True)
        with self.assertRaises(ValueError):
            ds.add_counts_from_dataset(ds)  # can't add to static DataSet

    def test_constructor_raises_on_missing_spam_labels(self):
        gstrs = [('Gx',), ('Gx', 'Gy'), ('Gy',)]
        with self.assertRaises(AssertionError):
            DataSet(circuits=gstrs)  # no spam labels specified

    def test_constructor_raises_on_missing_oplabels_when_static(self):
        with self.assertRaises(ValueError):
            DataSet(self.oli_static, self.time_static, self.reps_static,
                    outcomeLabels=['0', '1'], bStatic=True)

    def test_constructor_raises_on_missing_counts_when_static(self):
        with self.assertRaises(ValueError):
            DataSet(circuits=self.gstrs, outcomeLabels=['0', '1'], bStatic=True)


class DataSetInstanceBase:
    def setUp(self):
        super(DataSetInstanceBase, self).setUp()
        self.ds = DataSet(outcomeLabels=['0', '1'])
        self.ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        self.ds.add_count_dict(('Gy', 'Gy'), {'1': 90})
        self.ds.add_count_dict(('Gy', 'Gy'), ld.OutcomeLabelDict([('0', 10), ('1', 90)]),
                               overwriteExisting=False)  # adds counts at next available integer timestep
        for opstr in self.ds:  # XXX is there a direct way of indexing the first row?
            self.dsRow = self.ds[opstr]
            break

    def test_construction(self):
        self.assertEqual(self.ds[('Gx',)]['0'], 10)
        self.assertEqual(self.ds[('Gx',)]['1'], 90)
        self.assertAlmostEqual(self.ds[('Gx',)].fraction('0'), 0.1)

    def test_add_counts_from_dataset(self):
        gstrs = [('Gx',), ('Gx', 'Gy'), ('Gy',)]
        oli = np.array([0, 1], 'i')
        oli_nonstc = [oli, oli, oli]
        time_nonstc = [np.zeros(2, 'd'), np.zeros(2, 'd'), np.zeros(2, 'd')]
        reps_nonstc = [10 * np.ones(2, 'i'), 10 * np.ones(2, 'i'), 10 * np.ones(2, 'i')]
        ds2 = DataSet(oli_nonstc, time_nonstc, reps_nonstc,
                      circuits=gstrs, outcomeLabels=['0', '1'])
        ds2.add_counts_from_dataset(self.ds)
        # TODO assert correctness

    def test_copy(self):
        ds_copy = self.ds.copy()
        # TODO assert correctness

    def test_copy_nonstatic(self):
        writable = self.ds.copy_nonstatic()
        # implicitly assert copy is not readonly
        writable[('Gy',)] = {'0': 20, '1': 80}

    def test_get_degrees_of_freedom(self):
        dof = self.ds.get_degrees_of_freedom()
        # TODO assert correctness

    def test_truncate(self):
        trunc = self.ds.truncate([('Gx',)])
        # TODO assert correctness

    def test_truncate_ignore_on_missing(self):
        with self.assertNoWarns():
            self.ds.truncate([('Gx',), ('Gz',)], missingAction="ignore")

    def test_truncate_warn_on_missing(self):
        with self.assertWarns(Warning):
            self.ds.truncate([('Gx',), ('Gz',)], missingAction="warn")

    def test_truncate_raise_on_missing(self):
        with self.assertRaises(KeyError):
            self.ds.truncate([('Gx',), ('Gz',)], missingAction="raise")

    def test_len(self):
        n = len(self.ds)
        # TODO assert correctness

    def test_to_string(self):
        ds_str = str(self.ds)
        # TODO assert correctness

    def test_indexing(self):
        self.assertFalse(('Gz',) in self.ds)
        for opstr in self.ds:
            self.assertTrue(opstr in self.ds)
            self.assertTrue(Circuit(opstr) in self.ds)

    def test_pickle(self):
        s = pickle.dumps(self.ds)
        ds_pickled = pickle.loads(s)
        self.assertEqual(ds_pickled[('Gx',)]['0'], 10)
        self.assertAlmostEqual(ds_pickled[('Gx',)].fraction('0'), 0.1)

    def test_raise_on_new_outcome_label(self):
        with self.assertRaises(NotImplementedError):
            self.ds[('Gx',)]['new'] = 20  # assignment can't create *new* outcome labels (yet)

    # Row instance tests
    def test_row_indexing(self):
        cnt = 0
        for spamlabel, count in self.dsRow.counts.items():
            cnt += count
        # TODO assert correctness

    def test_row_to_string(self):
        row_str = str(self.dsRow)
        # TODO assert correctness

    def test_row_as_dict(self):
        cntDict = self.dsRow.as_dict()
        # TODO assert correctness


class DataSetNonstaticInstanceTester(DataSetInstanceBase, BaseCase):
    def test_process_circuits(self):
        self.ds.process_circuits(lambda s: pc.manipulate_circuit(s, [(('Gx',), ('Gy',))]))
        test_cntDict = self.ds[('Gy',)].as_dict()
        # TODO assert correctness

    def test_scale(self):
        self.dsRow.scale(2.0)
        self.assertEqual(self.dsRow['0'], 20)
        self.assertEqual(self.dsRow['1'], 180)


class DataSetStaticInstanceTester(DataSetInstanceBase, BaseCase):
    def setUp(self):
        super(DataSetStaticInstanceTester, self).setUp()
        self.ds.done_adding_data()

    def test_is_static(self):
        with self.assertRaises(ValueError):
            self.ds.add_count_dict(('Gx',), {'0': 10, '1': 90})  # done adding data
        with self.assertRaises(ValueError):
            self.ds.add_counts_from_dataset(self.ds)  # done adding data
        with self.assertRaises(ValueError):
            self.ds[('Gy',)] = {'0': 20, '1': 80}

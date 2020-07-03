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
        dsEmpty = DataSet(outcome_labels=['0', '1'])
        dsEmpty.done_adding_data()
        # TODO assert correctness

    def test_initialize_by_index(self):
        ds1 = DataSet(outcome_labels=['0', '1'])
        ds1.add_count_dict(('Gx',), {'0': 10, '1': 90})
        ds2 = DataSet(outcome_labels=['0', '1'])
        ds2[('Gx',)] = {'0': 10, '1': 90}
        # TODO assert correctness

    def test_initialize_from_raw_series_data(self):
        ds = DataSet(outcome_label_indices=self.olInds)
        ds.add_raw_series_data(('Gy',),  # gate sequence
                               ['0', '1'],  # spam labels
                               [0.0, 1.0],  # time stamps
                               [3, 7])  # repeats
        # TODO assert correctness

    def test_initialize_from_series_data(self):
        ds = DataSet(outcome_labels=['0', '1'])
        ds.add_series_data(('Gy', 'Gy'), [{'0': 2, '1': 8}, {'0': 6, '1': 4}, {'1': 10}],
                           [0.0, 1.2, 2.4])
        ds.add_series_data(('Gy', 'Gy', 'Gy'),
                           [OrderedDict([('0', 2), ('1', 8)]),
                            OrderedDict([('0', 6), ('1', 4)]),
                            OrderedDict([('1', 10)])],
                           [0.0, 1.2, 2.4])  # add with ordered dicts
        # TODO assert correctness

    def test_construct_from_list(self):
        ds = DataSet(self.oli_nonstc, self.time_nonstc, self.reps_nonstc,
                     circuits=self.gstrs, outcome_labels=['0', '1'])
        # TODO assert correctness

    def test_construct_from_map(self):
        ds = DataSet(self.oli_nonstc[:], self.time_nonstc[:], self.reps_nonstc[:],
                     circuit_indices=self.gstrInds, outcome_label_indices=self.olInds)
        # TODO assert correctness

    def test_construct_static(self):
        ds = DataSet(self.oli_nonstc, self.time_nonstc, self.reps_nonstc,
                     circuit_indices=self.gstrInds_static, outcome_labels=['0', '1'], static=True)
        with self.assertRaises(ValueError):
            ds.add_counts_from_dataset(ds)  # can't add to static DataSet

    def test_construct_keep_separate_on_collision(self):
        ds = DataSet(outcome_labels=['0', '1'], collision_action="keepseparate")
        ds.add_count_dict(('Gx', 'Gx'), {'0': 10, '1': 90})
        ds.add_count_dict(('Gx', 'Gy'), {'0': 20, '1': 80})
        ds.add_count_dict(('Gx', 'Gx'), {'0': 30, '1': 70})  # a duplicate
        self.assertEqual(ds.keys(), [('Gx', 'Gx'), ('Gx', 'Gy'), ('Gx', 'Gx', '#1')])
        self.assertEqual(ds.keys(strip_occurrence_tags=True), [('Gx', 'Gx'), ('Gx', 'Gy'), ('Gx', 'Gx')])
        # TODO _set_row test separately
        ds._set_row(('Gx', 'Gx'), {'0': 5, '1': 95}, occurrence=1)  # test _set_row with occurrence arg

    def test_constructor_raises_on_missing_spam_labels(self):
        gstrs = [('Gx',), ('Gx', 'Gy'), ('Gy',)]
        with self.assertRaises(AssertionError):
            DataSet(circuits=gstrs)  # no spam labels specified

    def test_static_constructor_raises_on_missing_oplabels(self):
        with self.assertRaises(ValueError):
            DataSet(self.oli_static, self.time_static, self.reps_static,
                    outcome_labels=['0', '1'], static=True)

    def test_static_constructor_raises_on_missing_counts(self):
        with self.assertRaises(ValueError):
            DataSet(circuits=self.gstrs, outcome_labels=['0', '1'], static=True)


class DefaultDataSetInstance(object):
    def setUp(self):
        super(DefaultDataSetInstance, self).setUp()
        self.ds = DataSet(outcome_labels=['0', '1'], collision_action='aggregate') # adds counts at next available integer timestep
        self.ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        self.ds.add_count_dict(('Gy', 'Gy'), {'1': 90})
        self.ds.add_count_dict(('Gy', 'Gy'), ld.OutcomeLabelDict([('0', 10), ('1', 90)]))  

    def test_construction(self):
        self.assertEqual(self.ds[('Gx',)]['0'], 10)
        self.assertEqual(self.ds[('Gx',)]['1'], 90)
        self.assertAlmostEqual(self.ds[('Gx',)].fraction('0'), 0.1)

    def test_raise_on_new_outcome_label(self):
        with self.assertRaises(NotImplementedError):
            self.ds[('Gx',)]['new'] = 20  # assignment can't create *new* outcome labels (yet)


class RawSeriesDataSetInstance(object):
    def setUp(self):
        super(RawSeriesDataSetInstance, self).setUp()
        self.ds = DataSet(outcome_labels=['0', '1'])
        self.ds.add_raw_series_data(('Gx',),
                                    ['0', '0', '1', '0', '1', '0', '1', '1', '1', '0'],
                                    [0.0, 0.2, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3, 1.35, 1.5], None)
        self.ds[('Gy', 'Gy')] = (['0', '1'], [0.0, 1.0])  # add via spam-labels, times
        # TODO test construction


class DataSetMethodBase(object):
    def setUp(self):
        super(DataSetMethodBase, self).setUp()
        for opstr in self.ds:  # XXX is there a direct way of indexing the first row?  EGN: I don't know of one.
            self.dsRow = self.ds[opstr]
            break

    def test_add_counts_from_dataset(self):
        gstrs = [('Gx',), ('Gx', 'Gy'), ('Gy',)]
        oli = np.array([0, 1], 'i')
        oli_nonstc = [oli, oli, oli]
        time_nonstc = [np.zeros(2, 'd'), np.zeros(2, 'd'), np.zeros(2, 'd')]
        reps_nonstc = [10 * np.ones(2, 'i'), 10 * np.ones(2, 'i'), 10 * np.ones(2, 'i')]
        ds2 = DataSet(oli_nonstc, time_nonstc, reps_nonstc,
                      circuits=gstrs, outcome_labels=['0', '1'])
        ds2.add_counts_from_dataset(self.ds)
        # TODO assert correctness

    def test_get_outcome_labels(self):
        outcomes = self.ds.outcome_labels
        self.assertEqual(outcomes, [('0',), ('1',)])

    def test_get_gate_labels(self):
        gates = self.ds.gate_labels()
        self.assertEqual(gates, ['Gx', 'Gy'])

    def test_copy(self):
        ds_copy = self.ds.copy()
        # TODO assert correctness

    def test_copy_nonstatic(self):
        writable = self.ds.copy_nonstatic()
        # implicitly assert copy is not readonly
        writable[('Gy',)] = {'0': 20, '1': 80}

    def test_get_degrees_of_freedom(self):
        dof = self.ds.degrees_of_freedom()
        dof = self.ds.degrees_of_freedom([('Gx',)])
        # TODO assert correctness

    def test_truncate(self):
        trunc = self.ds.truncate([('Gx',)])
        # TODO assert correctness

    def test_truncate_ignore_on_missing(self):
        with self.assertNoWarns():
            self.ds.truncate([('Gx',), ('Gz',)], missing_action="ignore")

    def test_truncate_warn_on_missing(self):
        with self.assertWarns(Warning):
            self.ds.truncate([('Gx',), ('Gz',)], missing_action="warn")

    def test_truncate_raise_on_missing(self):
        with self.assertRaises(KeyError):
            self.ds.truncate([('Gx',), ('Gz',)], missing_action="raise")

    def test_len(self):
        n = len(self.ds)
        # TODO assert correctness

    def test_to_string(self):
        ds_str = str(self.ds)
        # TODO assert correctness

    def test_indexing(self):
        self.assertFalse(('Gz',) in self.ds)
        opstrs, rows = tuple(zip(*self.ds.items()))
        for a, b in zip(opstrs, self.ds):
            self.assertEqual(a, b)
        for opstr in self.ds:
            self.assertTrue(opstr in self.ds)
            self.assertTrue(Circuit(opstr) in self.ds)

    def test_time_slice(self):
        empty_slice = self.ds.time_slice(100.0, 101.0)
        ds_slice = self.ds.time_slice(1.0, 2.0)
        ds_slice = self.ds.time_slice(1.0, 2.0, aggregate_to_time=0.0)
        # TODO assert correctness

    def test_pickle(self):
        s = pickle.dumps(self.ds)
        ds_pickled = pickle.loads(s)
        for expected_row, actual_row in zip(self.ds, ds_pickled):
            for expected, actual in zip(expected_row, actual_row):
                self.assertEqual(expected, actual)

    # Row instance tests
    def test_row_get_expanded_ol(self):
        self.dsRow.expanded_ol
        # TODO assert correctness

    def test_row_get_expanded_oli(self):
        self.dsRow.expanded_oli
        # TODO assert correctness

    def test_row_get_expanded_times(self):
        self.dsRow.expanded_times
        # TODO assert correctness

    def test_row_fraction(self):
        self.dsRow.fraction('0')
        # TODO assert correctness

    def test_row_counts_at_time(self):
        self.dsRow.counts_at_time(0.0)
        # TODO assert correctness

    def test_row_timeseries(self):
        all_times, _ = self.dsRow.timeseries('all')
        self.dsRow.timeseries('0')
        self.dsRow.timeseries('1')
        self.dsRow.timeseries('0', all_times)
        # TODO assert correctness

    def test_row_len(self):
        len(self.dsRow)
        # TODO assert correctness

    def test_row_indexing(self):
        cnt = 0
        for spamlabel, count in self.dsRow.counts.items():
            cnt += count
        # TODO assert correctness

    def test_row_to_string(self):
        row_str = str(self.dsRow)
        # TODO assert correctness

    def test_row_as_dict(self):
        cntDict = self.dsRow.to_dict()
        # TODO assert correctness

    def test_row_outcomes_raise_on_modify(self):
        with self.assertRaises(ValueError):
            self.dsRow.outcomes = ['x', 'x']


class DataSetNonstaticInstanceTester(DataSetMethodBase, DefaultDataSetInstance, BaseCase):
    def test_process_circuits(self):
        self.ds.process_circuits(lambda s: pc.manipulate_circuit(s, [(('Gx',), ('Gy',))]))
        test_cntDict = self.ds[('Gy',)].to_dict()
        # TODO assert correctness

    def test_scale(self):
        self.dsRow.scale_inplace(2.0)
        self.assertEqual(self.dsRow['0'], 20)
        self.assertEqual(self.dsRow['1'], 180)

    def test_warn_on_nonintegral_scaled_row_access(self):
        self.dsRow.scale_inplace(3.141592)
        with self.assertWarns(Warning):
            self.dsRow.expanded_ol
        with self.assertWarns(Warning):
            self.dsRow.expanded_oli
        with self.assertWarns(Warning):
            self.dsRow.expanded_times


class DataSetStaticInstanceTester(DataSetMethodBase, DefaultDataSetInstance, BaseCase):
    def setUp(self):
        super(DataSetStaticInstanceTester, self).setUp()
        self.ds.done_adding_data()

    def test_raise_on_add_count_dict(self):
        with self.assertRaises(ValueError):
            self.ds.add_count_dict(('Gx',), {'0': 10, '1': 90})

    def test_raise_on_add_counts_from_dataset(self):
        with self.assertRaises(ValueError):
            self.ds.add_counts_from_dataset(self.ds)

    def test_raise_on_add_raw_series_data(self):
        with self.assertRaises(ValueError):
            self.ds.add_raw_series_data(('Gy', 'Gx'), ['0', '1'], [0.0, 1.0], [2, 2])

    def test_raise_on_add_series_from_dataset(self):
        with self.assertRaises(ValueError):
            self.ds.add_series_from_dataset(self.ds)

    def test_raise_on_modify_by_index(self):
        with self.assertRaises(ValueError):
            self.ds[('Gy',)] = {'0': 20, '1': 80}

    def test_raise_on_scale(self):
        with self.assertRaises(ValueError):
            self.dsRow.scale_inplace(2.0)


class RawSeriesDataSetInstanceTester(DataSetMethodBase, RawSeriesDataSetInstance, BaseCase):
    def test_build_repetition_counts(self):
        self.ds._add_explicit_repetition_counts()
        # TODO assert correctness

    def test_scale_raises_on_missing_repeat_counts(self):
        with self.assertRaises(ValueError):
            self.ds[('Gx',)].scale_inplace(2.0)


class RawSeriesDatasetStaticInstanceTester(DataSetMethodBase, RawSeriesDataSetInstance, BaseCase):
    def setUp(self):
        super(RawSeriesDatasetStaticInstanceTester, self).setUp()
        self.ds.done_adding_data()

    def test_raise_on_build_repetition_counts(self):
        with self.assertRaises(ValueError):
            self.ds._add_explicit_repetition_counts()

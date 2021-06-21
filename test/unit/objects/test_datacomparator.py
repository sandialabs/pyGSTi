import pygsti.circuits as pc
import pygsti.data as pdata
import pygsti.data.datacomparator as dc
from pygsti.modelpacks.legacy import std1Q_XYI
from pygsti.data import DataSet, MultiDataSet
from ..util import BaseCase


class DataComparatorTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        #Let's make our underlying model have a little bit of random unitary noise.
        cls.mdl_exp_0 = std1Q_XYI.target_model().randomize_with_unitary(.01, seed=0)
        cls.mdl_exp_1 = std1Q_XYI.target_model().randomize_with_unitary(.01, seed=1234)

        germs = std1Q_XYI.germs
        fiducials = std1Q_XYI.fiducials
        max_lengths = [1, 2, 4, 8]
        gate_sequences = pc.create_lsgst_circuits(std1Q_XYI.gates, fiducials, fiducials, germs, max_lengths)
        #Generate the data for the two data, using the same model, with 100 repetitions of each sequence.
        N = 100
        cls.DS_0 = pdata.simulate_data(cls.mdl_exp_0, gate_sequences, N, 'binomial', seed=10)
        cls.DS_1 = pdata.simulate_data(cls.mdl_exp_1, gate_sequences, N, 'binomial', seed=20)

    def setUp(self):
        self.mdl_exp_0 = self.mdl_exp_0.copy()
        self.mdl_exp_1 = self.mdl_exp_1.copy()
        self.DS_0 = self.DS_0.copy()
        self.DS_1 = self.DS_1.copy()

    def test_implement(self):
        comparator = dc.DataComparator([self.DS_0, self.DS_1])
        comparator.run(significance=0.05)
        # TODO assert correctness

    def test_getters(self):
        comparator = dc.DataComparator([self.DS_0, self.DS_1])
        comparator.run(significance=0.05)
        # XXX do these need unit tests?  EGN: maybe not - could ask Kenny
        mdl = list(self.DS_0.keys())[10]
        comparator.jsd(mdl)
        comparator.jsd_pseudothreshold
        comparator.llr(mdl)
        comparator.llr_pseudothreshold
        comparator.ssjsd(mdl)
        comparator.sstvd(mdl)
        comparator.tvd(mdl)
        comparator.aggregate_llr
        comparator.aggregate_llr_threshold
        comparator.aggregate_nsigma
        comparator.aggregate_pvalue
        comparator.aggregate_pvalue_threshold
        comparator.maximum_sstvd
        comparator.pvalue(mdl)
        comparator.pvalue_pseudothreshold
        comparator.worst_circuits(10)
        # TODO assert correctness for all of the above

    def test_implement_exclusive(self):
        comparator = dc.DataComparator([self.DS_0, self.DS_1], op_exclusions=['Gx'], ds_names=['D0', 'D1'])
        comparator.run(significance=0.05)
        # TODO assert correctness

    def test_implement_inclusive(self):
        comparator = dc.DataComparator([self.DS_0, self.DS_1], op_inclusions=['Gi'], ds_names=['D0', 'D1'])
        comparator.run(significance=0.05)
        # TODO assert correctness

    def test_implement_multidataset(self):
        mds = MultiDataSet(outcome_labels=[('0',),('1',)])
        mds.add_dataset('D0', self.DS_0)
        mds.add_dataset('D1', self.DS_1)
        comparator = dc.DataComparator(mds)
        comparator.run(significance=0.05)
        # TODO assert correctness



    def test_construction_raises_on_bad_ds_names(self):
        with self.assertRaises(ValueError):
            dc.DataComparator([self.DS_0, self.DS_1], ds_names=["foobar"])

    def test_construction_raises_on_outcome_label_mismatch(self):
        DS_bad = DataSet(outcome_labels=['1', '0'])  # bad order!
        DS_bad.add_count_dict(('Gx',), {'0': 10, '1': 90})
        DS_bad.done_adding_data()
        with self.assertRaises(ValueError):
            dc.DataComparator([self.DS_0, DS_bad])

    def test_construction_raises_on_op_sequence_mismatch(self):
        DS_bad = DataSet(outcome_labels=['0', '1'])  # order ok...
        DS_bad.add_count_dict(('Gx',), {'0': 10, '1': 90})
        DS_bad.done_adding_data()
        with self.assertRaises(ValueError):
            dc.DataComparator([self.DS_0, DS_bad])

import numpy as np
from collections import OrderedDict

from ..util import BaseCase

from pygsti.objects import MultiDataSet, DataSet, Circuit


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
        mds = MultiDataSet(mds_oli, mds_time, mds_rep, circuitIndices=gstrInds,
                           outcomeLabelIndices=olInds)
        # TODO assert correctness

    def test_construct_with_no_data(self):
        mds4 = MultiDataSet(outcomeLabels=['0', '1'])
        mds5 = MultiDataSet()
        # TODO assert correctness


class MultiDataSetMethodBase:
    def test_add_dataset_raises_on_spam_label_mismatch(self):
        ds2 = DataSet(outcomeLabels=['0', 'foobar'])  # different spam labels than multids
        ds2.add_count_dict((), {'0': 10, 'foobar': 90})
        ds2.add_count_dict(('Gx',), {'0': 10, 'foobar': 90})
        ds2.add_count_dict(('Gx', 'Gy'), {'0': 10, 'foobar': 90})
        ds2.add_count_dict(('Gx', 'Gx', 'Gx', 'Gx'), {'0': 10, 'foobar': 90})
        ds2.done_adding_data()
        with self.assertRaises(ValueError):
            self.mds['newDS'] = ds2

    def test_add_dataset_raises_on_gate_mismatch(self):
        ds3 = DataSet(outcomeLabels=['0','1']) #different operation sequences
        ds3.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        ds3.done_adding_data()
        with self.assertRaises(ValueError):
            self.mds['newDS'] = ds3

    def test_add_dataset_raises_on_nonstatic_dataset(self):
        ds = DataSet(outcomeLabels=['0','1']) #different operation sequences
        ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        with self.assertRaises(ValueError):
            self.mds['newDS'] = ds



class MultiDataSetInstanceTester(MultiDataSetMethodBase, BaseCase):
    def setUp(self):
        self.mds = MultiDataSet(mds_oli, mds_time, mds_rep, circuitIndices=gstrInds,
                                outcomeLabels=['0', '1'])


class MultiDataSetNoRepsInstanceTester(MultiDataSetMethodBase, BaseCase):
    def setUp(self):
        self.mds = MultiDataSet(mds_oli, mds_time, None, circuitIndices=gstrInds,
                                outcomeLabels=['0', '1'])

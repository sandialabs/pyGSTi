import numpy as np

import pygsti.circuits as pc
import pygsti.models as models
import pygsti.data as pdata
from pygsti.tools import listtools as lt
from ..util import BaseCase


class DataSetConstructionTester(BaseCase):
    def setUp(self):
        # TODO optimize
        self.model = models.create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy'], ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])
        self.depolGateset = self.model.depolarize(op_noise=0.1)

        def make_lsgst_lists(opLabels, fiducialList, germList, maxLengthList):
            singleOps = pc.to_circuits([(g,) for g in opLabels])
            lgstStrings = pc.create_lgst_circuits(fiducialList, fiducialList, opLabels)
            lsgst_list = pc.to_circuits([()])  # running list of all strings so far

            if maxLengthList[0] == 0:
                lsgst_listOfLists = [lgstStrings]
                maxLengthList = maxLengthList[1:]
            else: lsgst_listOfLists = []

            for maxLen in maxLengthList:
                lsgst_list += pc.create_circuits("f0+R(germ,N)+f1", f0=fiducialList,
                                                     f1=fiducialList, germ=germList, N=maxLen,
                                                     R=pc.repeat_with_max_length,
                                                     order=('germ', 'f0', 'f1'))
                lsgst_listOfLists.append(lt.remove_duplicates(lgstStrings + lsgst_list))

            print("%d LSGST sets w/lengths" % len(lsgst_listOfLists), map(len, lsgst_listOfLists))
            return lsgst_listOfLists

        gates = ['Gi', 'Gx', 'Gy']
        fiducials = pc.to_circuits([(), ('Gx',), ('Gy',), ('Gx', 'Gx'), ('Gx', 'Gx', 'Gx'),
                                     ('Gy', 'Gy', 'Gy')])  # fiducials for 1Q MUB
        germs = pc.to_circuits([('Gx',), ('Gy',), ('Gi',), ('Gx', 'Gy',),
                                 ('Gx', 'Gy', 'Gi',), ('Gx', 'Gi', 'Gy',), ('Gx', 'Gi', 'Gi',), ('Gy', 'Gi', 'Gi',),
                                 ('Gx', 'Gx', 'Gi', 'Gy',), ('Gx', 'Gy', 'Gy', 'Gi',),
                                 ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy',)])
        maxLengths = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.lsgst_lists = make_lsgst_lists(gates, fiducials, germs, maxLengths)
        self.circuit_list = self.lsgst_lists[-1]
        self.dataset = pdata.simulate_data(self.depolGateset, self.circuit_list, num_samples=1000,
                                             sample_error='binomial', seed=100)

    def test_generate_fake_data(self):
        dataset = pdata.simulate_data(self.dataset, self.circuit_list, num_samples=None,
                                        sample_error='none', seed=100)
        dataset = pdata.simulate_data(self.dataset, self.circuit_list, num_samples=100,
                                        sample_error='round', seed=100)
        dataset = pdata.simulate_data(self.dataset, self.circuit_list, num_samples=100,
                                        sample_error='multinomial', seed=100)

        randState = np.random.RandomState(1234)
        dataset1 = pdata.simulate_data(dataset, self.circuit_list, num_samples=100,
                                        sample_error='binomial', rand_state=randState)
        dataset2 = pdata.simulate_data(dataset, self.circuit_list, num_samples=100,
                                        sample_error='binomial', seed=1234)
        for dr1, dr2 in zip(dataset1.values(), dataset2.values()):
            self.assertEqual(dr1.counts, dr2.counts)

    def test_generate_fake_data_raises_on_bad_sample_error(self):
        with self.assertRaises(ValueError):
            pdata.simulate_data(self.dataset, self.circuit_list, num_samples=None,
                                  sample_error='foobar', seed=100)

    def test_merge_outcomes(self):
        merged_dataset = pdata.aggregate_dataset_outcomes(self.dataset, {'merged_outcome_label': [('0',), ('1',)]})
        for dsRow in merged_dataset.values():
            self.assertEqual(dsRow.total, dsRow['merged_outcome_label'])

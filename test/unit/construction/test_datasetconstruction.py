import numpy as np

from ..util import BaseCase

import pygsti.construction as pc
from pygsti.tools import listtools as lt
import pygsti.construction.datasetconstruction as dc


class DataSetConstructionTester(BaseCase):
    def setUp(self):
        # TODO optimize
        self.model = pc.build_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'], ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])
        self.depolGateset = self.model.depolarize(op_noise=0.1)

        def make_lsgst_lists(opLabels, fiducialList, germList, maxLengthList):
            singleOps = pc.circuit_list([(g,) for g in opLabels])
            lgstStrings = pc.list_lgst_circuits(fiducialList, fiducialList, opLabels)
            lsgst_list = pc.circuit_list([()])  # running list of all strings so far

            if maxLengthList[0] == 0:
                lsgst_listOfLists = [lgstStrings]
                maxLengthList = maxLengthList[1:]
            else: lsgst_listOfLists = []

            for maxLen in maxLengthList:
                lsgst_list += pc.create_circuit_list("f0+R(germ,N)+f1", f0=fiducialList,
                                                     f1=fiducialList, germ=germList, N=maxLen,
                                                     R=pc.repeat_with_max_length,
                                                     order=('germ', 'f0', 'f1'))
                lsgst_listOfLists.append(lt.remove_duplicates(lgstStrings + lsgst_list))

            print("%d LSGST sets w/lengths" % len(lsgst_listOfLists), map(len, lsgst_listOfLists))
            return lsgst_listOfLists

        gates = ['Gi', 'Gx', 'Gy']
        fiducials = pc.circuit_list([(), ('Gx',), ('Gy',), ('Gx', 'Gx'), ('Gx', 'Gx', 'Gx'),
                                     ('Gy', 'Gy', 'Gy')])  # fiducials for 1Q MUB
        germs = pc.circuit_list([('Gx',), ('Gy',), ('Gi',), ('Gx', 'Gy',),
                                 ('Gx', 'Gy', 'Gi',), ('Gx', 'Gi', 'Gy',), ('Gx', 'Gi', 'Gi',), ('Gy', 'Gi', 'Gi',),
                                 ('Gx', 'Gx', 'Gi', 'Gy',), ('Gx', 'Gy', 'Gy', 'Gi',),
                                 ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy',)])
        maxLengths = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.lsgst_lists = make_lsgst_lists(gates, fiducials, germs, maxLengths)
        self.circuit_list = self.lsgst_lists[-1]
        self.dataset = pc.generate_fake_data(self.depolGateset, self.circuit_list, nSamples=1000,
                                             sampleError='binomial', seed=100)

    def test_generate_fake_data(self):
        dataset = pc.generate_fake_data(self.dataset, self.circuit_list, nSamples=None,
                                        sampleError='none', seed=100)
        dataset = pc.generate_fake_data(self.dataset, self.circuit_list, nSamples=100,
                                        sampleError='round', seed=100)
        dataset = pc.generate_fake_data(self.dataset, self.circuit_list, nSamples=100,
                                        sampleError='multinomial', seed=100)

        randState = np.random.RandomState(1234)
        dataset = pc.generate_fake_data(dataset, self.circuit_list, nSamples=100,
                                        sampleError='binomial', randState=randState)
        # TODO assert correctness

    def test_generate_fake_data_raises_on_bad_sample_error(self):
        with self.assertRaises(ValueError):
            pc.generate_fake_data(self.dataset, self.circuit_list, nSamples=None,
                                  sampleError='foobar', seed=100)

    def test_merge_outcomes(self):
        merged_dataset = pc.merge_outcomes(self.dataset, {'merged_outcome_label': [('0',), ('1',)]})
        for dsRow in merged_dataset.values():
            self.assertEqual(dsRow.total, dsRow['merged_outcome_label'])

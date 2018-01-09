from ..testutils import BaseTestCase, compare_files, temp_files

import unittest
import pygsti
import pygsti.construction as pc
import numpy as np

class DataSetConstructionTestCase(BaseTestCase):

    def setUp(self):
        super(DataSetConstructionTestCase, self).setUp()
        self.gateset = pc.build_gateset( [2], [('Q0',)],
                                         ['Gi','Gx','Gy'], [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"])
        self.depolGateset = self.gateset.depolarize(gate_noise=0.1)

        def make_lsgst_lists(gateLabels, fiducialList, germList, maxLengthList):
            singleGates = pc.gatestring_list([(g,) for g in gateLabels])
            lgstStrings = pc.list_lgst_gatestrings(fiducialList, fiducialList, gateLabels)
            lsgst_list  = pc.gatestring_list([ () ]) #running list of all strings so far

            if maxLengthList[0] == 0:
                lsgst_listOfLists = [ lgstStrings ]
                maxLengthList = maxLengthList[1:]
            else: lsgst_listOfLists = [ ]

            for maxLen in maxLengthList:
                lsgst_list += pc.create_gatestring_list("f0+R(germ,N)+f1", f0=fiducialList,
                f1=fiducialList, germ=germList, N=maxLen,
                R=pc.repeat_with_max_length,
                order=('germ','f0','f1'))
                lsgst_listOfLists.append( pygsti.remove_duplicates(lgstStrings + lsgst_list) )

            print("%d LSGST sets w/lengths" % len(lsgst_listOfLists), map(len,lsgst_listOfLists))
            return lsgst_listOfLists

        gates      = ['Gi','Gx','Gy']
        fiducials  = pc.gatestring_list([ (), ('Gx',), ('Gy',), ('Gx','Gx'), ('Gx','Gx','Gx'), ('Gy','Gy','Gy') ]) # fiducials for 1Q MUB
        germs      = pc.gatestring_list( [('Gx',), ('Gy',), ('Gi',), ('Gx', 'Gy',),
                                     ('Gx', 'Gy', 'Gi',), ('Gx', 'Gi', 'Gy',),('Gx', 'Gi', 'Gi',), ('Gy', 'Gi', 'Gi',),
                                     ('Gx', 'Gx', 'Gi', 'Gy',), ('Gx', 'Gy', 'Gy', 'Gi',),
                                     ('Gx', 'Gx', 'Gy', 'Gx', 'Gy', 'Gy',)] )
        maxLengths = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.lsgst_lists     = make_lsgst_lists(gates, fiducials, germs, maxLengths)
        self.gatestring_list = self.lsgst_lists[-1]
        self.dataset = pc.generate_fake_data(self.depolGateset, self.gatestring_list, nSamples=1000,
                                             sampleError='binomial', seed=100)

    def test_generate_fake_data(self):
        dataset = pc.generate_fake_data(self.dataset, self.gatestring_list, nSamples=None, sampleError='multinomial', seed=100)
        dataset = pc.generate_fake_data(dataset, self.gatestring_list, nSamples=1000, sampleError='round', seed=100)

        randState = np.random.RandomState(1234)
        dataset = pc.generate_fake_data(dataset, self.gatestring_list, nSamples=1000, sampleError='binomial', randState=randState)


    def test_merge_outcomes(self):
        merged_dataset = pc.merge_outcomes(self.dataset, {'merged_outcome_label': [('0',), ('1',)]})
        for dsRow in merged_dataset.values():
            self.assertEqual( dsRow.total, dsRow['merged_outcome_label'] )


if __name__ == '__main__':
    unittest.main(verbosity=2)

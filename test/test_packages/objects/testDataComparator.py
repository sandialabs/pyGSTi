import unittest
import pygsti
import numpy as np
import pickle

from pygsti.construction import std1Q_XYI
import pygsti.construction as pc

from ..testutils import BaseTestCase, compare_files, temp_files

class DataComparatorTestCase(BaseTestCase):

    def setUp(self):
        super(DataComparatorTestCase, self).setUp()

    def test_data_comparison(self):
        #Let's make our underlying gate set have a little bit of random unitary noise.
        gs_exp_0 = std1Q_XYI.gs_target.copy()
        gs_exp_0 = gs_exp_0.randomize_with_unitary(.01,seed=0)
        gs_exp_1 = std1Q_XYI.gs_target.copy()
        gs_exp_1 = gs_exp_0.randomize_with_unitary(.01,seed=1234)
        germs = std1Q_XYI.germs
        fiducials = std1Q_XYI.fiducials
        max_lengths = [1,2,4,8,16,32,64]
        gate_sequences = pygsti.construction.make_lsgst_experiment_list(std1Q_XYI.gates,fiducials,fiducials,germs,max_lengths)
        
        #Generate the data for the two datasets, using the same gate set, with 100 repetitions of each sequence.
        N=100
        DS_0 = pygsti.construction.generate_fake_data(gs_exp_0,gate_sequences,N,'binomial',seed=10)
        DS_1 = pygsti.construction.generate_fake_data(gs_exp_1,gate_sequences,N,'binomial',seed=20)
        
        #Let's compare the two datasets.
        comparator_0_1 = pygsti.objects.DataComparator([DS_0,DS_1])

        #In constrast, here are ways to incorrectly create a DataComparatory
        with self.assertRaises(ValueError):
            pygsti.objects.DataComparator([DS_0,DS_1], DS_names=["foobar"]) # length mismatch

        with self.assertRaises(ValueError):
            DS_bad = pygsti.objects.DataSet(outcomeLabels=['1','0']) #bad order!
            DS_bad.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
            DS_bad.done_adding_data()
            pygsti.objects.DataComparator([DS_0,DS_bad]) #outcome label ordering of datasets doesn't match

        with self.assertRaises(ValueError):
            DS_bad = pygsti.objects.DataSet(outcomeLabels=['0','1']) # order ok...
            DS_bad.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
            DS_bad.done_adding_data()
            pygsti.objects.DataComparator([DS_0,DS_bad]) # but gate sequences don't match!

        #Let's get the report from the comparator.
        comparator_0_1.report(confidence_level=0.95)

        gs_exp_1 = std1Q_XYI.gs_target.copy()
        gs_exp_1 = gs_exp_1.randomize_with_unitary(.01,seed=1)
        DS_2 = pygsti.construction.generate_fake_data(gs_exp_1,gate_sequences,N,'binomial',seed=30)
        
        #Let's make the comparator and get the report.
        comparator_1_2 = pygsti.objects.DataComparator([DS_1,DS_2])
        comparator_1_2.report(confidence_level=0.95)

        #get the 10 worst offenders
        worst_strings = comparator_1_2.worst_strings(10)

        #Also test "rectification" (re-scaling to make consistent) here:
        comparator_0_1.rectify_datasets(confidence_level=0.95,
                                        target_score='dof')
        comparator_0_1.rectify_datasets(confidence_level=0.1,
                                        target_score='dof') #also use a low confidence_level to ensure "violator" cases get run

        #Test TVD computation
        comparator_0_1.compute_stat_sig_TVDs()



    def test_inclusion_exclusion(self):
        gs_exp_0 = std1Q_XYI.gs_target.copy()
        gs_exp_0 = gs_exp_0.randomize_with_unitary(.01,seed=0)
        germs = std1Q_XYI.germs
        fiducials = std1Q_XYI.fiducials
        max_lengths = [1,2,4,8]
        gate_sequences = pygsti.construction.make_lsgst_experiment_list(std1Q_XYI.gates,fiducials,fiducials,germs,max_lengths)
        
        #Generate the data for the two datasets, using the same gate set, with 100 repetitions of each sequence.
        N=100
        DS_0 = pygsti.construction.generate_fake_data(gs_exp_0,gate_sequences,N,'binomial',seed=10)
        DS_1 = pygsti.construction.generate_fake_data(gs_exp_0,gate_sequences,N,'binomial',seed=20)
        
        #Let's compare the two datasets.
        comparator_0_1 = pygsti.objects.DataComparator([DS_0,DS_1], gate_exclusions=['Gx'],
                                                       gate_inclusions=['Gi'], DS_names=["D0","D1"])

        #Let's get the report from the comparator.
        comparator_0_1.report(confidence_level=0.95)



    def test_multidataset(self):
        gs_exp_0 = std1Q_XYI.gs_target.copy()
        gs_exp_0 = gs_exp_0.randomize_with_unitary(.01,seed=0)
        germs = std1Q_XYI.germs
        fiducials = std1Q_XYI.fiducials
        max_lengths = [1,2,4,8]
        gate_sequences = pygsti.construction.make_lsgst_experiment_list(std1Q_XYI.gates,fiducials,fiducials,germs,max_lengths)
        
        #Generate the data for the two datasets, using the same gate set, with 100 repetitions of each sequence.
        N=100
        DS_0 = pygsti.construction.generate_fake_data(gs_exp_0,gate_sequences,N,'binomial',seed=10)
        DS_1 = pygsti.construction.generate_fake_data(gs_exp_0,gate_sequences,N,'binomial',seed=20)
        mds = pygsti.objects.MultiDataSet(outcomeLabels=[('0',),('1',)])
        mds.add_dataset('D0', DS_0)
        mds.add_dataset('D1', DS_1)
        
        #Let's compare the two datasets.
        comparator_0_1 = pygsti.objects.DataComparator(mds)

        #Let's get the report from the comparator.
        comparator_0_1.report(confidence_level=0.95)

        #Also test "rectification" (re-scaling to make consistent) here:
        comparator_0_1.rectify_datasets(confidence_level=0.95,
                                        target_score='dof')


    def test_likelihood_fn(self):
        #Otherwise unused, so just make sure this runs
        pList = [0.5,0.5]; Nlist = [10,10]
        L = pygsti.objects.datacomparator.likelihood(pList,Nlist)
        


        

if __name__ == '__main__':
    unittest.main(verbosity=2)

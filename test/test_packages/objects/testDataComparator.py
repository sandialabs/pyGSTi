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
        germs = std1Q_XYI.germs
        fiducials = std1Q_XYI.fiducials
        max_lengths = [1,2,4,8,16,32,64]
        gate_sequences = pygsti.construction.make_lsgst_experiment_list(std1Q_XYI.gates,fiducials,fiducials,germs,max_lengths)
        
        #Generate the data for the two datasets, using the same gate set, with 100 repetitions of each sequence.
        N=100
        DS_0 = pygsti.construction.generate_fake_data(gs_exp_0,gate_sequences,N,'binomial',seed=10)
        DS_1 = pygsti.construction.generate_fake_data(gs_exp_0,gate_sequences,N,'binomial',seed=20)
        
        #Let's compare the two datasets.
        comparator_0_1 = pygsti.objects.DataComparator([DS_0,DS_1])

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


        

if __name__ == '__main__':
    unittest.main(verbosity=2)

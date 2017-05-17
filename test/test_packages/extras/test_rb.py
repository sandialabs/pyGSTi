from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
import os
import scipy
import pygsti

from pygsti.extras import rb
from pygsti.construction import std1Q_XYI

class RBTestCase(BaseTestCase):

    def test_rb_full(self):
        gs_target = std1Q_XYI.gs_target
        clifford_group = rb.std1Q.clifford_group
        clifford_to_canonical = rb.std1Q.clifford_to_canonical

        canonical_to_primitive = {'Gi':['Gi'], 'Gxp2':['Gx'], 'Gxp':['Gx','Gx'],
                                  'Gxmp2':['Gx','Gx','Gx'], 'Gyp2':['Gy'],
                                  'Gyp':['Gy','Gy'],'Gymp2':['Gy','Gy','Gy']}

        clifford_to_primitive = \
            pygsti.construction.compose_alias_dicts(clifford_to_canonical,
                                                    canonical_to_primitive)

        m_min = 1
        m_max = 1000
        delta_m = 100
        K_m_sched = 10

        filename_base = os.path.join(temp_files,'rb_template')
        rb_sequences = rb.write_empty_rb_files(filename_base, m_min, m_max, 
                                               delta_m, clifford_group, K_m_sched,
                                               {'primitive': clifford_to_primitive},
                                               seed=0)
        depol_strength = 1e-3
        gs_experimental = std1Q_XYI.gs_target
        gs_experimental = gs_experimental.depolarize(gate_noise=depol_strength)

        all_rb_sequences = []
        for seqs_for_single_cliff_len in rb_sequences['clifford']:
            all_rb_sequences.extend(seqs_for_single_cliff_len)

        N=100    
        rb_data = pygsti.construction.generate_fake_data(gs_experimental,all_rb_sequences,
                                                         N,'binomial',seed=1,
                                                         aliasDict=clifford_to_primitive,
                                                         collisionAction="keepseparate")

        rb_results = rb.do_randomized_benchmarking(
            rb_data, all_rb_sequences, success_spamlabel='minus', dim=2,
            pre_avg=True, clifford_to_primitive = clifford_to_primitive)

        rb_results_wcanonical = rb.do_randomized_benchmarking(
            rb_data, all_rb_sequences, success_spamlabel='minus', dim=2,
            pre_avg=True, clifford_to_canonical = clifford_to_canonical,
            canonical_to_primitive = canonical_to_primitive)

        #Maybe move this to workspace tests?
        w = pygsti.report.Workspace()
        w.RandomizedBenchmarkingPlot(rb_results,'clifford')
        w.RandomizedBenchmarkingPlot(rb_results,'primitive')
        w.RandomizedBenchmarkingPlot(rb_results,'primitive', xlim=(0,500), ylim=(0,1))
        with self.assertRaises(ValueError):
            w.RandomizedBenchmarkingPlot(rb_results,'foobar')

        rb_results.print_clifford()
        rb_results.print_primitive()

        gs_cliff_experimental = pygsti.construction.build_alias_gateset(
            gs_experimental,clifford_to_primitive)

        print("Experimental RB error rate:", rb_results.dicts['clifford']['r'])

        rb_results.compute_bootstrap_error_bars(('clifford','primitive'),seed=0)
        rb_results.compute_bootstrap_error_bars("all", randState=np.random.RandomState(0)) #same
        rb_results.compute_analytic_error_bars(0.001, 0.001, 1.0)

        rb_results.print_clifford()
        rb_results.print_primitive()
        rb_results.print_detail('foobar') #test displayed message
        
        print(rb_results) #test __str__
        print(rb_results.dicts['clifford']['r_error_BS']) #random test

    def test_rb_strings(self):
        rndm = np.random.RandomState(1234)
        rs = rb.create_random_rb_clifford_string(10, rb.std1Q.clifford_group, seed=0)
        rs = rb.create_random_rb_clifford_string(10, rb.std1Q.clifford_group, randState=rndm)
        
        K_m_sched = rb.create_K_m_sched(1,11,5,0.01,0.01,0.001)
        lst = rb.list_random_rb_clifford_strings(1,11,5, rb.std1Q.clifford_group, K_m_sched,
                                                 {'canonical': rb.std1Q.clifford_to_canonical},
                                                 seed=0)
        lst = rb.list_random_rb_clifford_strings(1,11,5, rb.std1Q.clifford_group,
                                                 10, randState=rndm)

        filename_base = os.path.join(temp_files,'rb_test_empty')
        rb.write_empty_rb_files(filename_base, 1, 11, 5, rb.std1Q.clifford_group, 10,
                                None, seed=0)


    def test_rb_objects(self):
        mxs = [ np.identity(2) ]
        mg = rb.MatrixGroup(mxs)

        M = mg.get_matrix(0)
        Minv = mg.get_matrix_inv(0)
        iMinv = mg.get_inv(0)
        iProd = mg.product( (0,0,0) )
        
        labels = [ 'I' ]
        mg = rb.MatrixGroup(mxs, labels)

        M = mg.get_matrix('I')
        Minv = mg.get_matrix_inv('I')
        iMinv = mg.get_inv('I')
        iProd = mg.product( ('I','I','I') )
        
        with self.assertRaises(AssertionError):
            non_group_mxs = [ np.identity(2), np.diag([0,2]) ]
            bad_mg = rb.MatrixGroup(non_group_mxs)

    def test_rb_utils(self):
        pass

    
    def test_rb_dataset_construction(self):
        gateset = std1Q_XYI.gs_target
        fids,germs = std1Q_XYI.fiducials, std1Q_XYI.germs
        depol_gateset = gateset.depolarize(gate_noise=0.1,spam_noise=0)
        gateStrings = pygsti.construction.create_gatestring_list(
            "f0+T(germ,N)+f1", f0=fids, f1=fids, germ=germs, N=3,
            T=pygsti.construction.repeat_with_max_length,
            order=["germ","f0","f1"])
        ds_binom = pygsti.construction.generate_fake_data(depol_gateset, gateStrings, nSamples=1000,
                                                          sampleError='binomial', seed=100,
                                                          collisionAction="keepseparate")

        rbDS = rb.generate_sim_rb_data(depol_gateset, ds_binom, seed=1234)
        rbDS_perfect = rb.generate_sim_rb_data_perfect(depol_gateset, ds_binom)

            


if __name__ == '__main__':
    unittest.main(verbosity=2)




from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np
from scipy.linalg import expm
import os
import scipy
import pygsti

from pygsti.extras import rb
from pygsti.construction import std1Q_XYI

class RBTestCase(BaseTestCase):

    def test_rb_full(self):
        gs_target = std1Q_XYI.gs_target
        clifford_group = rb.std1Q.clifford_group
        clifford_to_canonical = rb.std1Q.clifford_to_generators

        clifford_to_primitive = std1Q_XYI.clifford_compilation

        m_list = [1,101]# ,201,301,401,501,601,701,801,801,1001]
        K_m = 10

        filename_base = os.path.join(temp_files,'rb_template')
        rb_sequences = rb.write_empty_rb_files(filename_base, m_list, K_m,clifford_group,
                                               {'primitive': clifford_to_primitive},
                                               seed=0)

        depol_strength = 1e-3
        gs_experimental = std1Q_XYI.gs_target.depolarize(gate_noise=depol_strength)

        all_rb_sequences = []
        for seqs_for_single_cliff_len in rb_sequences:
            all_rb_sequences.extend(seqs_for_single_cliff_len)

        N=100    
        rb_data = pygsti.construction.generate_fake_data(gs_experimental,all_rb_sequences,
                                                         N,'binomial',seed=1,
                                                         aliasDict=clifford_to_primitive,
                                                         collisionAction="keepseparate")

        rb_results = rb.do_randomized_benchmarking(rb_data, all_rb_sequences,
                                                   fit='standard',success_outcomelabel=('1',), 
                                                   dim=2)
        #Maybe move this to workspace tests?
        w = pygsti.report.Workspace()
        w.RandomizedBenchmarkingPlot(rb_results)
        w.RandomizedBenchmarkingPlot(rb_results, xlim=(0,500), ylim=(0,1))
        #with self.assertRaises(ValueError):
        #    w.RandomizedBenchmarkingPlot(rb_results,'foobar')

        rb_results.print_results()
        print(rb_results.results['r'])

        rb_results.compute_bootstrap_error_bars(randState=np.random.RandomState(0))

        rb_results.print_results()
        
        print(rb_results) #test __str__
        print(rb_results.results['r_error_BS']) #random test
        
        rb_results = rb.do_randomized_benchmarking(rb_data, all_rb_sequences,
                                                   fit='first order',success_outcomelabel=('1',), 
                                                   dim=2)
        #Maybe move this to workspace tests?
        w = pygsti.report.Workspace()
        w.RandomizedBenchmarkingPlot(rb_results, fit='first order')
        with self.assertRaises(ValueError): # needs gateset
            w.RandomizedBenchmarkingPlot(rb_results, Magesan_zeroth=False, Magesan_zeroth_SEB=True) # sets M_zeroth=True
        with self.assertRaises(ValueError): # needs gateset
            w.RandomizedBenchmarkingPlot(rb_results, Magesan_first=False, Magesan_first_SEB=True) # sets M_first=True
        with self.assertRaises(ValueError): # needs gateset
            w.RandomizedBenchmarkingPlot(rb_results, exact_decay=True)
        with self.assertRaises(ValueError): # needs gateset
            w.RandomizedBenchmarkingPlot(rb_results, L_matrix_decay=True)
        #with self.assertRaises(ValueError):
        #    w.RandomizedBenchmarkingPlot(rb_results,'foobar')
            
        #rb_results.print_results()
        #rb_results.compute_bootstrap_error_bars()
        #rb_results.print_results()

        #Test plotting of results
        clifford_gateset = pygsti.construction.build_alias_gateset(gs_experimental,clifford_to_primitive)
        clifford_targetGateset = pygsti.construction.build_alias_gateset(gs_target,clifford_to_primitive)
        w = pygsti.report.Workspace()
        w.RandomizedBenchmarkingPlot(
            rb_results, xlim=None, ylim=None, 
            fit='standard', Magesan_zeroth=True, Magesan_first=True,
            exact_decay=True,L_matrix_decay=True, Magesan_zeroth_SEB=True, 
            Magesan_first_SEB=True, L_matrix_decay_SEB=True,gs=clifford_gateset,
            gs_target=clifford_targetGateset,group=clifford_group,
            norm='1to1', legend=True)


    def test_rb_strings(self):
        clifford_group = rb.std1Q.clifford_group
        gs_clifford_target = rb.std1Q.gs_target
        rndm = np.random.RandomState(1234)
        m = 5
        rndm = np.random.RandomState(1234)
        gtf = rb.create_random_gatestring(m, clifford_group, inverse = True, interleaved = None, randState=rndm)
        gtt = rb.create_random_gatestring(m, clifford_group, inverse = True, interleaved = 'Gc0', randState=rndm)
        gstf = rb.create_random_gatestring(m, gs_clifford_target, inverse = True, interleaved = None, randState=rndm)
        gstt = rb.create_random_gatestring(m, gs_clifford_target, inverse = True, interleaved = 'Gc0', randState=rndm)
        self.assertEqual(clifford_group.label_indices[clifford_group.product(gtf)],0)
        self.assertEqual(clifford_group.label_indices[clifford_group.product(gtt)],0)
        self.assertEqual(clifford_group.label_indices[clifford_group.product(gstf)],0)
        self.assertEqual(clifford_group.label_indices[clifford_group.product(gstt)],0)
        self.assertEqual(len(gtf),m+1)
        self.assertEqual(len(gtt),2*m+1)
        self.assertEqual(len(gstf),m+1)
        self.assertEqual(len(gstt),2*m+1)

        gff = rb.create_random_gatestring(5, clifford_group, inverse = False, interleaved = None, randState=rndm)
        gft = rb.create_random_gatestring(5, clifford_group, inverse = False, interleaved = 'Gc0', randState=rndm)
        gsff = rb.create_random_gatestring(5, gs_clifford_target, inverse = False, interleaved = None, randState=rndm)
        gsft = rb.create_random_gatestring(5, gs_clifford_target, inverse = False, interleaved = 'Gc0', randState=rndm)
        self.assertEqual(len(gff),m)
        self.assertEqual(len(gft),2*m)
        self.assertEqual(len(gsff),m)
        self.assertEqual(len(gsft),2*m)

        interleaved_gates_gtt = []
        interleaved_gates_gstt = []
        interleaved_gates_gft = []
        interleaved_gates_gsft = []
        interleaved_gate = []
        for i in range(0,m):
            interleaved_gates_gtt.append(gtt[2*i+1])
            interleaved_gates_gstt.append(gtt[2*i+1])
            interleaved_gates_gft.append(gtt[2*i+1])
            interleaved_gates_gsft.append(gtt[2*i+1])
            interleaved_gate.append('Gc0')  

        self.assertEqual(interleaved_gates_gtt,  interleaved_gate)
        self.assertEqual(interleaved_gates_gstt, interleaved_gate) # Changed by LSaldyt: Closest match to syntax error
        self.assertEqual(interleaved_gates_gft,  interleaved_gate)
        self.assertEqual(interleaved_gates_gsft, interleaved_gate) # Changed by LSaldyt: Closest match to syntax error
        
        K_m_sched = rb.create_K_m_sched(1,11,5,0.01,0.01,0.001)
        #Seems this method was removed...
        #lst = rb.list_random_rb_clifford_strings(1,11,5, rb.std1Q.clifford_group, K_m_sched,
        #                                         {'canonical': rb.std1Q.clifford_to_generators},
        #                                         seed=0)
        #lst = rb.list_random_rb_clifford_strings(1,11,5, rb.std1Q.clifford_group,
        #                                         10, randState=rndm)

        #This has also changed apparently...
        #filename_base = os.path.join(temp_files,'rb_test_empty')
        #rb.write_empty_rb_files(filename_base, 1, 11, 5, rb.std1Q.clifford_group, 10,
        #                        None, seed=0)


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
        gs_target = std1Q_XYI.gs_target
        clifford_group = rb.std1Q.clifford_group
        gs_clifford_target = rb.std1Q.gs_target
        clifford_to_primitive = std1Q_XYI.clifford_compilation #??rb.std1Q.clifford_to_XYI
        
        depol_strength = 1e-2
        gs_d = gs_target.copy()
        gs_d_cliff = gs_clifford_target.copy()
        gs_d = gs_d.depolarize(gate_noise=depol_strength)
        gs_d_cliff = gs_d_cliff.depolarize(gate_noise=depol_strength)

        print("AGI = ",rb.average_gate_infidelity(gs_d.gates['Gx'],gs_target.gates['Gx']))
        print("r = ",rb.p_to_r(1-depol_strength))
        self.assertAlmostEqual(rb.average_gate_infidelity(gs_d.gates['Gx'],gs_target.gates['Gx']),
                               rb.p_to_r(1-depol_strength))
        A = pygsti.change_basis(gs_d.gates['Gx'], 'gm', 'std')
        B = pygsti.change_basis(gs_target.gates['Gx'], 'gm', 'std')
        self.assertAlmostEqual(rb.average_gate_infidelity(A,B,mxBasis='std'),
                               rb.p_to_r(1-depol_strength))
        self.assertAlmostEqual(rb.r_to_p(rb.p_to_r(0.79898)),0.79898)
        self.assertAlmostEqual(rb.p_to_r(rb.r_to_p(0.79898)),0.79898)
        self.assertAlmostEqual(rb.r_to_p(rb.average_gate_infidelity(gs_d.gates['Gx'],gs_target.gates['Gx'])),
                               1-depol_strength)
        
        self.assertAlmostEqual(rb.unitarity(gs_d.gates['Gx'],d=2),(1-depol_strength)**2)
        self.assertAlmostEqual(rb.unitarity(gs_target.gates['Gx'],d=2),1.)
        A = pygsti.change_basis(gs_d.gates['Gx'], 'gm', 'std')
        B = pygsti.change_basis(gs_target.gates['Gx'], 'gm', 'std') 
        self.assertAlmostEqual(rb.unitarity(A,mxBasis='std',d=2),(1-depol_strength)**2)
        self.assertAlmostEqual(rb.unitarity(B,mxBasis='std',d=2),1.)
        
        self.assertAlmostEqual(rb.average_gateset_infidelity(gs_d,gs_target,d=2),
                               rb.p_to_r(1-depol_strength))
        
        error_maps = rb.errormaps(gs_d, gs_target)
        correct_error_map = np.array([[1.,0.,0.,0.],[0.,1-depol_strength,0.,0.],[0.,0.,1-
                                            depol_strength,0.],[0.,0.,0.,1-depol_strength]])
        for key in error_maps.gates.keys():
            self.assertAlmostEqual(np.amax(abs(error_maps.gates[key]-correct_error_map)),0.)
            
        self.assertAlmostEqual(rb.gatedependence_of_errormaps(gs_d, gs_target,
                                                              norm='diamond',mxBasis=None, d=2),0.)
        self.assertAlmostEqual(rb.gatedependence_of_errormaps(gs_d, gs_target,
                                                              norm='1to1',mxBasis=None, d=2),0.)
        
        self.assertAlmostEqual(rb.predicted_RB_number(gs_d_cliff,gs_clifford_target),
                               rb.p_to_r(1-depol_strength))
        self.assertAlmostEqual(rb.predicted_RB_decay_parameter(gs_d_cliff,gs_clifford_target),
                               1-depol_strength)
        gs_d_cliff_out=rb.transform_to_RB_gauge(gs_d_cliff,gs_clifford_target)
        self.assertAlmostEqual(rb.predicted_RB_number(gs_d_cliff,gs_clifford_target),
                        rb.average_gateset_infidelity(gs_d_cliff_out,gs_clifford_target,d=2))
        self.assertAlmostEqual(rb.average_gateset_infidelity(gs_d_cliff_out,gs_d_cliff),0)
        
        z = np.array([[1.,0],[0,-1.]])       
        error_unitary_i = expm(-1j * (0.00 / 2) *z)
        error_unitary_x = expm(-1j * (0.05 / 2) *z)
        error_unitary_y = expm(-1j * (0.05 / 2) *z)

        gs_Z = gs_target.copy()

        error_gate_i = pygsti.unitary_to_pauligate(error_unitary_i)
        error_gate_x = pygsti.unitary_to_pauligate(error_unitary_x)
        error_gate_y = pygsti.unitary_to_pauligate(error_unitary_y)
        gs_Z.gates['Gi'] = np.dot(error_gate_i,gs_target.gates['Gi'])
        gs_Z.gates['Gx'] = np.dot(error_gate_x,gs_target.gates['Gx'])
        gs_Z.gates['Gy'] = np.dot(error_gate_y,gs_target.gates['Gy'])

        gs_clifford_Z = pygsti.construction.build_alias_gateset(gs_Z,clifford_to_primitive)
        gs_clifford_Z_out = rb.transform_to_RB_gauge(gs_clifford_Z,gs_clifford_target)
        self.assertAlmostEqual(rb.predicted_RB_number(gs_clifford_Z,gs_clifford_target),
                       rb.predicted_RB_number(gs_clifford_Z_out,gs_clifford_target))
        self.assertAlmostEqual(rb.predicted_RB_number(gs_clifford_Z,gs_clifford_target),
                       rb.average_gateset_infidelity(gs_clifford_Z_out,gs_clifford_target,d=2))

        self.assertEqual(np.shape(rb.R_matrix(gs_clifford_Z,clifford_group,d=2)), (96,96))
        self.assertEqual(np.shape(rb.R_matrix(gs_clifford_Z,clifford_group,d=2,
                                               subset_sampling=['Gc0','Gc16','Gc21'])),(96,96))
        
        full_set = ['Gc0','Gc1','Gc2','Gc3','Gc4','Gc5','Gc6','Gc7','Gc8','Gc9','Gc10','Gc11','Gc12','Gc13',
            'Gc14','Gc15','Gc16','Gc17','Gc18','Gc19','Gc20','Gc21','Gc22','Gc23']
        full_dict = {'Gc0':'Gc0','Gc1':'Gc1','Gc2':'Gc2','Gc3':'Gc3','Gc4':'Gc4','Gc5':'Gc5',
             'Gc6':'Gc6','Gc7':'Gc7','Gc8':'Gc8','Gc9':'Gc9','Gc10':'Gc10','Gc11':'Gc11',
             'Gc12':'Gc12','Gc13':'Gc13','Gc14':'Gc14','Gc15':'Gc15','Gc16':'Gc16','Gc17':'Gc17',
             'Gc18':'Gc18','Gc19':'Gc19','Gc20':'Gc20','Gc21':'Gc21','Gc22':'Gc22','Gc23':'Gc23'}

        # Check for consistency between different ways to construct the full R matrix.
        #self.assertAlmostEqual(np.amax(rb.R_matrix(gs_clifford_Z,clifford_group,d=2,subset_sampling=full_set)-
        #                                rb.R_matrix(gs_clifford_Z,clifford_group,d=2,subset_sampling=None)),0.0)
        #self.assertAlmostEqual(np.amax(rb.R_matrix(gs_clifford_Z,clifford_group,d=2,subset_sampling=full_set,
        #                                    group_to_gateset=full_dict)-rb.R_matrix(gs_clifford_Z,clifford_group,
        #        d=2,subset_sampling=None)),0.0)
        
        # check the predicted R decay parameter agrees with L-matrix method for standard Clifford RB
        self.assertAlmostEqual(rb.p_to_r(rb.R_matrix_predicted_RB_decay_parameter(gs_clifford_Z,clifford_group,d=2)),
                                  rb.predicted_RB_number(gs_clifford_Z,gs_clifford_target))
        self.assertAlmostEqual(rb.p_to_r(rb.R_matrix_predicted_RB_decay_parameter(gs_d_cliff,clifford_group,d=2)),
                                  rb.predicted_RB_number(gs_d_cliff,gs_clifford_target))
        # check the predicted R decay parameter agrees with L-matrix method for "generator" RB
        self.assertAlmostEqual(rb.p_to_r(rb.R_matrix_predicted_RB_decay_parameter(gs_Z,clifford_group,d=2,
                subset_sampling=['Gi','Gx','Gy'],group_to_gateset = {'Gc0':'Gi','Gc16':'Gx','Gc21':'Gy'})),
                rb.predicted_RB_number(gs_Z,gs_target))
        self.assertAlmostEqual(rb.p_to_r(rb.R_matrix_predicted_RB_decay_parameter(gs_d,clifford_group,d=2,
                subset_sampling=['Gi','Gx','Gy'],group_to_gateset={'Gc0':'Gi','Gc16':'Gx','Gc21':'Gy'})),
                rb.predicted_RB_number(gs_d,gs_target))
        # check the "generator RB" prediction is accurate for depolarizing noise.
        self.assertAlmostEqual(rb.predicted_RB_number(gs_d,gs_target),rb.p_to_r(1-depol_strength))
        
        m1, P_m1 = rb.exact_RB_ASPs(gs_d_cliff,clifford_group,m_max=1000,m_min=1,m_step=1,
                   d=2,success_outcomelabel=('0',),subset_sampling=None,group_to_gateset=None,
                   compilation=None) # fixed_length_each_m = False, 
        self.assertAlmostEqual(np.amax(abs(P_m1 - rb.standard_fit_function(m1+1,0.5,0.5,1-depol_strength))),0.0)
        m2, P_m2 = rb.exact_RB_ASPs(gs_d,clifford_group,m_max=100,m_min=1,m_step=1,
                   d=2,success_outcomelabel=('0',), subset_sampling=['Gi','Gx','Gy'],group_to_gateset=
                   {'Gc0':'Gi','Gc16':'Gx','Gc21':'Gy'}, #fixed_length_each_m = False,
                   compilation=clifford_to_primitive)
        self.assertAlmostEqual(P_m2[0],(0.5+0.5*((1.*(1-depol_strength)**1. + 
                                          2.*(1-depol_strength)**3.)/3.)*(1-depol_strength)))
        
        m_L1 , P_m_L1 = rb.L_matrix_ASPs(gs_d_cliff,gs_clifford_target,m_max=1000,m_min=1,m_step=1,
                                         d=2,success_outcomelabel=('0',),error_bounds=False)
        self.assertAlmostEqual(np.amax(abs(P_m_L1 - rb.standard_fit_function(m_L1+1,0.5,0.5,1-depol_strength))),0.0)
        # Once the bounds on the asps are updated, put a test here for them, to check they are
        # consistent with exact ASPs.

        # The following test checks that the function behaves as currently intended, when
        # the full group is not sampled over, but this will need to be changed when the
        # function is updated to be consistent with the exact ASPs with a compilation
        # at the end.
        m_L1 , P_m_L1 = rb.L_matrix_ASPs(gs_d,gs_target,m_max=100,m_min=1,m_step=1,
                                          d=2,success_outcomelabel=('0',),error_bounds=False)
        self.assertTrue(np.amax(abs(P_m_L1- rb.standard_fit_function(m_L1+1,
                                                                     0.5,0.5,1-depol_strength))) < 1e-9)
               
        A = np.array([[1.,2.],[3.4,5.4]])
        Avec = np.array([1.,3.4,2.,5.4])
        self.assertAlmostEqual(np.amax(abs(rb.unvec(rb.vec(A)) - A)),0.0)
        A = np.array([[1.1,2.],[3.4,4.2]])
        B = np.array([[5.2,6.],[7.,8.2]])
        C = np.array([[9.3,10.],[11.1,12.4]])
        self.assertAlmostEqual(np.amax(abs(np.dot(np.dot(A,B),C)-rb.unvec(np.dot(np.kron(C.T,A), rb.vec(B))))),0.0)
        

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




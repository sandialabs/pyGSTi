from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np

import pygsti
from pygsti.extras import rb
from pygsti.baseobjs import Label

class RBTestCase(BaseTestCase):

    def test_rb_group(self):        
        # Tests the key aspects of the group module by creating
        # the 1Q clifford group
        rb.group.construct_1Q_Clifford_group()        
        return

    def test_sample(self):
        
        # -- pspecs to use in all the tests. They cover a variety of possibilities -- #
        
        n_1 = 4
        glist = ['Gi','Gxpi2','Gypi2','Gcnot']
        pspec_1 = pygsti.obj.ProcessorSpec(n_1,glist,verbosity=0,qubit_labels=['Q0','Q1','Q2','Q3'])
    
        n_2 = 3
        glist = ['Gi','Gxpi','Gypi','Gzpi','Gh','Gp','Gcphase']
        availability = {'Gcphase':[(0,1),(1,2)]}
        pspec_2 = pygsti.obj.ProcessorSpec(n_2,glist,availability=availability,verbosity=0)
        
        # Tests Clifford RB samplers
        lengths = [0,2,5]
        circuits_per_length = 2
        subsetQs = ['Q1','Q2','Q3']
        out = rb.sample.clifford_rb_experiment(pspec_1, lengths, circuits_per_length, subsetQs=subsetQs, randomizeout=False, 
                                           citerations=2, compilerargs=[], descriptor='A Clifford RB experiment', verbosity=0)
        for key in list(out['idealout'].keys()):
            self.assertEqual(out['idealout'][key], (0,0,0))
    
        self.assertEqual(len(out['circuits']), circuits_per_length * len(lengths))
    
        out = rb.sample.clifford_rb_experiment(pspec_2, lengths, circuits_per_length, subsetQs=None, randomizeout=True, 
                                           citerations=1, compilerargs=[], descriptor='A Clifford RB experiment',verbosity=0)
        
        # --- Tests of the circuit layer samplers --- #
    
        # Tests for the sampling by pairs function
        layer = rb.sample.circuit_layer_by_pairing_qubits(pspec_1, twoQprob=0.0, oneQgatenames='all', twoQgatenames='all', 
                                          gatesetname = 'clifford')
        self.assertEqual(len(layer), n_1)
        layer = rb.sample.circuit_layer_by_pairing_qubits(pspec_1, twoQprob=1.0, oneQgatenames='all', twoQgatenames='all', 
                                          gatesetname = 'clifford')
        self.assertEqual(len(layer), n_1//2)
        layer = rb.sample.circuit_layer_by_pairing_qubits(pspec_1, twoQprob=0.0, oneQgatenames=['Gx',], twoQgatenames='all', 
                                          gatesetname = 'target')
        self.assertEqual(layer[0].name, 'Gx')
    
        layer = rb.sample.circuit_layer_by_Qelimination(pspec_2, twoQprob=0.0, oneQgates='all', twoQgates='all',
                                                        gatesetname='clifford')
        self.assertEqual(len(layer), n_2)
        layer = rb.sample.circuit_layer_by_Qelimination(pspec_2, twoQprob=1.0, oneQgates='all', twoQgates='all',
                                                        gatesetname='clifford')
        self.assertEqual(len(layer), (n_2 % 2) + n_2//2)
        layer = rb.sample.circuit_layer_by_pairing_qubits(pspec_1, twoQprob=0.0, oneQgatenames=['Gxpi'], twoQgatenames='all', 
                                          gatesetname = 'target')
        self.assertEqual(layer[0].name, 'Gxpi')
    
        # Tests for the sampling by co2Qgates function
        C01 = Label('Gcnot',('Q0','Q1'))
        C23 = Label('Gcnot',('Q2','Q3'))
        co2Qgates = [[],[C01,C23]]
        layer = rb.sample.circuit_layer_by_co2Qgates(pspec_1, None, co2Qgates, co2Qgatesprob='uniform', twoQprob=1.0, 
                                                   oneQgatenames='all', gatesetname='clifford')
        self.assertTrue(len(layer) == n_1 or len(layer) == n_1//2)
        layer = rb.sample.circuit_layer_by_co2Qgates(pspec_1, None, co2Qgates, co2Qgatesprob=[0.,1.], twoQprob=1.0, 
                                                   oneQgatenames='all', gatesetname='clifford')
        self.assertEqual(len(layer), n_1//2)
        layer = rb.sample.circuit_layer_by_co2Qgates(pspec_1, None, co2Qgates, co2Qgatesprob=[1.,0.], twoQprob=1.0, 
                                                   oneQgatenames=['Gx',], gatesetname='clifford')
        self.assertEqual(len(layer), n_1)
        self.assertEqual(layer[0].name, 'Gx')
        
        co2Qgates = [[],[C23,]]
        layer = rb.sample.circuit_layer_by_co2Qgates(pspec_1, ['Q2','Q3'], co2Qgates, co2Qgatesprob=[0.25,0.75], twoQprob=0.5, 
                                                   oneQgatenames='all', gatesetname='clifford')
        co2Qgates = [[C01,]]
        layer = rb.sample.circuit_layer_by_co2Qgates(pspec_1, None, co2Qgates, co2Qgatesprob=[1.], twoQprob=1.0, 
                                                   oneQgatenames='all', gatesetname='clifford')
        self.assertEqual(layer[0].name, 'Gcnot')
        self.assertEqual(len(layer), 3)
        
        # Tests the nested co2Qgates option.
        co2Qgates = [[],[[C01,C23],[C01,]]]
        layer = rb.sample.circuit_layer_by_co2Qgates(pspec_1, None, co2Qgates, co2Qgatesprob='uniform', twoQprob=1.0, 
                                                   oneQgatenames='all', gatesetname='clifford')
        # Tests for the sampling a layer of 1Q gates.
        layer = rb.sample.circuit_layer_of_oneQgates(pspec_1, oneQgatenames='all', pdist='uniform',
                                                    gatesetname='clifford')
        self.assertEqual(len(layer), n_1)
        layer = rb.sample.circuit_layer_of_oneQgates(pspec_1, subsetQs=['Q1','Q2'], oneQgatenames=['Gx','Gy'], pdist=[1.,0.],
                                                    gatesetname='clifford')
        self.assertEqual(len(layer), 2)
        self.assertEqual(layer[0].name, 'Gx')
        layer = rb.sample.circuit_layer_of_oneQgates(pspec_1, subsetQs=['Q2'],oneQgatenames=['Gx'], pdist=[3.,],
                                                    gatesetname='clifford')
        self.assertEqual(layer[0], Label('Gx','Q2'))
        self.assertEqual(len(layer), 1)
        layer = rb.sample.circuit_layer_of_oneQgates(pspec_1, oneQgatenames=['Gx'], pdist='uniform',
                                                    gatesetname='clifford')
        
        # Tests of the random_circuit sampler that is a wrap-around for the circuit-layer samplers
        
        C01 = Label('Gcnot',('Q0','Q1'))
        C23 = Label('Gcnot',('Q2','Q3'))
        co2Qgates = [[],[[C01,C23],[C01,]]]
        circuit = rb.sample.random_circuit(pspec_1, length=100, sampler='Qelimination')
        self.assertEqual(circuit.depth(), 100)
        circuit = rb.sample.random_circuit(pspec_2, length=100, sampler='Qelimination', samplerargs=[0.1,], addlocal = True)
        self.assertEqual(circuit.depth(), 201)
        self.assertLessEqual(len(circuit.get_layer(0)), n_2)
        circuit = rb.sample.random_circuit(pspec_1, length=100, sampler='pairingQs')
        circuit = rb.sample.random_circuit(pspec_1, length=10, sampler='pairingQs', samplerargs=[0.1,['Gx',]])
    
        circuit = rb.sample.random_circuit(pspec_1, length=100, sampler='co2Qgates', samplerargs=[co2Qgates])
        circuit = rb.sample.random_circuit(pspec_1, length=100, sampler='co2Qgates', samplerargs=[co2Qgates,[0.1,0.2],0.1], 
                                    addlocal = True, lsargs=[['Gx',]])
        self.assertEqual(circuit.depth(), 201)
        circuit = rb.sample.random_circuit(pspec_1, length=5, sampler='local')
        self.assertEqual(circuit.depth(), 5)
        circuit = rb.sample.random_circuit(pspec_1, length=5, sampler='local',samplerargs=[['Gx']])
        self.assertEqual(circuit.line_items[0][0].name, 'Gx')
        
        lengths = [0,2,5]
        circuits_per_length = 2
        # Test DRB experiment with all defaults.
        exp = rb.sample.direct_rb_experiment(pspec_2, lengths, circuits_per_length, verbosity=0)
        
        exp = rb.sample.direct_rb_experiment(pspec_2, lengths, circuits_per_length, subsetQs=[0,1], sampler='pairingQs',
                                            cliffordtwirl=False, conditionaltwirl=False, citerations=2, partitioned=True,
                                            verbosity=0)
        
        exp = rb.sample.direct_rb_experiment(pspec_2, lengths, circuits_per_length, subsetQs=[0,1], sampler='co2Qgates',
                                             samplerargs = [[[],[Label('Gcphase',(0,1)),]],[0.,1.]],
                                            cliffordtwirl=False, conditionaltwirl=False, citerations=2, partitioned=True,
                                            verbosity=0)
        
        exp = rb.sample.direct_rb_experiment(pspec_2, lengths, circuits_per_length, subsetQs=[0,1], sampler='local',
                                            cliffordtwirl=False, conditionaltwirl=False, citerations=2, partitioned=True,
                                            verbosity=0)
        
        # Tests of MRB : gateset must have self-inverses in the gate-set.
        n_1 = 4
        glist = ['Gi','Gxpi2','Gxmpi2','Gypi2','Gympi2','Gcnot']
        pspec_inv = pygsti.obj.ProcessorSpec(n_1, glist, verbosity=0, qubit_labels=['Q0','Q1','Q2','Q3'])
        lengths = [0,4,8]
        circuits_per_length = 10
        exp = rb.sample.mirror_rb_experiment(pspec_inv, lengths, circuits_per_length, subsetQs=['Q1','Q2','Q3'],
                                             sampler='Qelimination', samplerargs=[], localclifford=True, 
                                             paulirandomize=True)
        
        exp = rb.sample.mirror_rb_experiment(pspec_inv, lengths, circuits_per_length, subsetQs=['Q1','Q2','Q3'],
                                             sampler='Qelimination', samplerargs=[], localclifford=True, 
                                             paulirandomize=False)
        
        exp = rb.sample.mirror_rb_experiment(pspec_inv, lengths, circuits_per_length, subsetQs=['Q1','Q2','Q3'],
                                             sampler='Qelimination', samplerargs=[], localclifford=False, 
                                             paulirandomize=False)
     
        exp = rb.sample.mirror_rb_experiment(pspec_inv, lengths, circuits_per_length, subsetQs=['Q1','Q2','Q3'],
                                             sampler='Qelimination', samplerargs=[], localclifford=False, 
                                             paulirandomize=True)

    def test_rb_theory(self):
        
        # Tests can create the Cliffords gateset, using std1Q_Cliffords.
        from pygsti.construction import std1Q_Cliffords
        gs_target = std1Q_Cliffords.gs_target
    
        # Tests rb.group. This tests we can successfully construct a MatrixGroup
        clifford_group = rb.group.construct_1Q_Clifford_group()
    
        depol_strength = 1e-3
        gs = gs_target.depolarize(gate_noise=depol_strength)
    
        # Tests AGI function and p_to_r with AGI.
        AGI = pygsti.tools.average_gate_infidelity(gs.gates['Gc0'],gs_target.gates['Gc0'])
        r_AGI = rb.analysis.p_to_r(1-depol_strength,d=2,rtype='AGI')
        self.assertLess(np.abs(AGI-r_AGI), 10**(-10))
    
        # Tests EI function and p_to_r with EI.
        EI = pygsti.tools.entanglement_infidelity(gs.gates['Gc0'],gs_target.gates['Gc0'])
        r_EI = rb.analysis.p_to_r(1-depol_strength,d=2,rtype='EI')
        self.assertLess(np.abs(EI-r_EI), 10**(-10))
    
        # Tests uniform-average AGI function and the r-prediction function with uniform-weighting 
        AGsI = rb.theory.gateset_infidelity(gs,gs_target,itype='AGI')
        r_AGI = rb.analysis.p_to_r(1-depol_strength,d=2,rtype='AGI')
        r_pred_AGI = rb.theory.predicted_RB_number(gs,gs_target,rtype='AGI')
        self.assertLess(np.abs(AGsI-r_AGI), 10**(-10))
        self.assertLess(np.abs(r_pred_AGI-r_AGI), 10**(-10))
    
        # Tests uniform-average EI function and the r-prediction function with uniform-weighting 
        AEI = rb.theory.gateset_infidelity(gs,gs_target,itype='EI')
        r_EI = rb.analysis.p_to_r(1-depol_strength,d=2,rtype='EI')
        r_pred_EI = rb.theory.predicted_RB_number(gs,gs_target,rtype='EI')
        self.assertLess(np.abs(AEI-r_EI), 10**(-10))
        self.assertLess(np.abs(AEI-r_pred_EI), 10**(-10))
    
        # Tests the transform to RB gauge, and RB gauge transformation generating functions.
        gs_in_RB_gauge = rb.theory.transform_to_rb_gauge(gs, gs_target, eigenvector_weighting=0.5)
        AEI = rb.theory.gateset_infidelity(gs,gs_target,itype='EI')
        self.assertLess(np.abs(AEI-r_EI), 10**(-10))
    
    
        from pygsti.construction import std1Q_XY
        gs_target = std1Q_XY.gs_target.copy()
        gs = gs_target.copy()
    
        Zrot_unitary = np.array([[1.,0.],[0.,np.exp(-1j*0.01)]])
        Zrot_channel = pygsti.unitary_to_pauligate(Zrot_unitary)
    
        for key in gs_target.gates.keys():
            gs.gates[key] = np.dot(Zrot_channel,gs_target.gates[key])
    
        gs_in_RB_gauge = rb.theory.transform_to_rb_gauge(gs, gs_target, eigenvector_weighting=0.5)
    
        # A test that the RB gauge transformation behaves as expected -- a gateset that does not
        # have r = infidelity in its initial gauge does have this in the RB gauge. This also
        # tests that the r predictions are working for not-all-the-Cliffords gatesets.
    
        AEI = rb.theory.gateset_infidelity(gs_in_RB_gauge,gs_target,itype='EI')
        r_pred_EI = rb.theory.predicted_RB_number(gs,gs_target,rtype='EI')
        self.assertLess(np.abs(AEI-r_pred_EI), 10**(-10))
    
        # Test that weighted-infidelities + RB error rates functions working.
        gs_target = std1Q_XY.gs_target.copy()
        gs = gs_target.copy()
    
        depol_strength_X = 1e-3
        depol_strength_Y = 3e-3
    
        lx =1.-depol_strength_X
        depmap_X = np.array([[1.,0.,0.,0.],[0.,lx,0.,0.],[0.,0.,lx,0.],[0,0.,0.,lx]])
        ly =1.-depol_strength_Y
        depmap_Y = np.array([[1.,0.,0.,0.],[0.,ly,0.,0.],[0.,0.,ly,0.],[0,0.,0.,ly]])
        gs.gates['Gx'] = np.dot(depmap_X,gs_target.gates['Gx'])
        gs.gates['Gy'] = np.dot(depmap_Y,gs_target.gates['Gy'])
    
        Gx_weight = 1
        Gy_weight = 2
        weights = {'Gx':Gx_weight,'Gy':Gy_weight}
        WAEI = rb.theory.gateset_infidelity(gs,gs_target,weights=weights,itype='EI')
        GxAEI = pygsti.tools.entanglement_infidelity(gs.gates['Gx'],gs_target.gates['Gx'])
        GyAEI = pygsti.tools.entanglement_infidelity(gs.gates['Gy'],gs_target.gates['Gy'])
        manual_WAEI = (Gx_weight*GxAEI + Gy_weight*GyAEI)/(Gx_weight + Gy_weight)
        # Checks that a manual weighted-average agrees with the function
        self.assertLess(abs(manual_WAEI-WAEI), 10**(-10))
    
        gs_in_RB_gauge = rb.theory.transform_to_rb_gauge(gs, gs_target, weights=weights,
                                                         eigenvector_weighting=0.5)
        WAEI = rb.theory.gateset_infidelity(gs_in_RB_gauge,gs_target,weights=weights,itype='EI')
        # Checks the predicted RB number function works with specified weights
        r_pred_EI = rb.theory.predicted_RB_number(gs,gs_target,weights=weights,rtype='EI')
        # Checks predictions agree with weighted-infidelity
        self.assertLess(abs(r_pred_EI-WAEI), 10**(-10))
    
    
        # -------------------------------------- #
        #   Tests for R-matrix related functions
        # -------------------------------------- #
    
        # Test constructing the R matrix in the simplest case
        gs_target = std1Q_Cliffords.gs_target
        clifford_group = rb.group.construct_1Q_Clifford_group()
        R = rb.theory.R_matrix(gs_target, clifford_group, group_to_gateset=None, weights=None)
    
        # Test constructing the R matrix for a group-subset gateset with weights
        from pygsti.construction import std1Q_XYI
        clifford_compilation = std1Q_XYI.clifford_compilation
        gs_target = std1Q_XYI.gs_target.copy()
        group_to_gateset = {'Gc0':'Gi','Gc16':'Gx','Gc21':'Gy'}
        weights = {'Gi':1.,'Gx':1,'Gy':1}
        clifford_group = rb.group.construct_1Q_Clifford_group()
        R = rb.theory.R_matrix(gs_target, clifford_group, group_to_gateset=group_to_gateset, weights=weights)
    
        # Tests the p-prediction function works, and that we get the correct predictions from the R-matrix.
        p = rb.theory.R_matrix_predicted_RB_decay_parameter(gs_target, clifford_group, 
                                                            group_to_gateset=group_to_gateset, 
                                                            weights=weights)
        self.assertLess(abs(p - 1.) ,  10**(-10))
        depol_strength = 1e-3
        gs = gs_target.depolarize(gate_noise=depol_strength)
        p = rb.theory.R_matrix_predicted_RB_decay_parameter(gs, clifford_group, 
                                                            group_to_gateset=group_to_gateset, 
                                                            weights=weights)
        self.assertLess(abs(p - (1.0-depol_strength)) ,  10**(-10))
    
        # Tests the exact RB ASPs function on a Clifford gateset. 
        gs_target = std1Q_Cliffords.gs_target
        gs = std1Q_Cliffords.gs_target.depolarize(depol_strength)
        m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=1000, m_min=0, m_step=100, success_outcomelabel=('0',), 
                                          group_to_gateset=None, weights=None, compilation=None, 
                                          group_twirled=False)
        self.assertLess(abs(ASPs[1]- (0.5 + 0.5*(1.0-depol_strength)**101)) ,  10**(-10))
        m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=1000, m_min=0, m_step=100, success_outcomelabel=('0',), 
                                          group_to_gateset=None, weights=None, compilation=None, 
                                          group_twirled=True)
        self.assertLess(abs(ASPs[1]- (0.5 + 0.5*(1.0-depol_strength)**102)) ,  10**(-10))
    
        # Tests the exact RB ASPs function on a subset-of-Cliffords gateset. 
        clifford_compilation = std1Q_XY.clifford_compilation
        gs_target = std1Q_XY.gs_target.copy()
        group_to_gateset = {'Gc16':'Gx','Gc21':'Gy'}
        weights = {'Gx':5,'Gy':10}
        m, ASPs = rb.theory.exact_RB_ASPs(gs_target, clifford_group, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',), 
                                          group_to_gateset=group_to_gateset, weights=None, compilation=clifford_compilation, 
                                          group_twirled=False)
        self.assertLess(abs(np.sum(ASPs) - len(ASPs)) ,  10**(-10))
    
        # Tests the function behaves reasonably with a depolarized gateset + works with group_twirled + weights.
        depol_strength = 1e-3
        gs = gs_target.depolarize(gate_noise=depol_strength)
        m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',), 
                                          group_to_gateset=group_to_gateset, weights=None, compilation=clifford_compilation, 
                                          group_twirled=False)
        self.assertLess(abs(ASPs[0] - 1) ,  10**(-10))
    
        m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=10, m_min=0, m_step=3, success_outcomelabel=('0',), 
                                          group_to_gateset=group_to_gateset, weights=weights, compilation=clifford_compilation, 
                                          group_twirled=True)
        self.assertTrue((ASPs > 0.99).all())
    
    
        # Check the L-matrix theory predictions work and are consistent with the exact predictions
        m, ASPs = rb.theory.exact_RB_ASPs(gs, clifford_group, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',), 
                                          group_to_gateset=group_to_gateset, weights=weights, compilation=clifford_compilation, 
                                          group_twirled=True)
    
        # Todo : change '1to1' to 'diamond' in 2 of 3 of the following, when diamonddist is working.
        L_m, L_ASPs, L_LASPs, L_UASPs = rb.theory.L_matrix_ASPs(gs, gs_target, m_max=10, m_min=0, m_step=1, 
                                                                success_outcomelabel=('0',), compilation=clifford_compilation,
                                                                group_twirled=True, weights=weights, gauge_optimize=True,
                                                                return_error_bounds=True, norm='1to1')
        self.assertTrue((abs(ASPs-L_ASPs) < 0.001).all())
    
        # Check it works without the twirl, and gives plausable output
        L_m, L_ASPs = rb.theory.L_matrix_ASPs(gs, gs_target, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',),
                          compilation=clifford_compilation, group_twirled=False, weights=None, gauge_optimize=False, 
                          return_error_bounds=False, norm='1to1')
        self.assertTrue((ASPs > 0.98).all())
    
        # Check it works with a Clifford gateset, and gives plausable output
        gs_target = std1Q_Cliffords.gs_target
        gs = std1Q_Cliffords.gs_target.depolarize(depol_strength)
        m, ASPs = rb.theory.L_matrix_ASPs(gs, gs_target, m_max=10, m_min=0, m_step=1, success_outcomelabel=('0',),
                          compilation=None, group_twirled=False, weights=None, gauge_optimize=False, 
                          return_error_bounds=False, norm='1to1')
        self.assertTrue((ASPs > 0.98).all())

    def test_rb_io_results_and_analysis(self):

        # Just checks that we can succesfully import the standard data type.
        data = rb.io.import_rb_summary_data([compare_files + '/rb_io_test.txt',])
        # This is a basic test that the imported dataset makes sense : we can
        # successfully run the analysis on it.
        out = rb.analysis.std_practice_analysis(data,bootstrap_samples=100)
        # Checks plotting works. This requires matplotlib, so should do a try/except

        from pygsti.report import workspace
        w = workspace.Workspace()
        w.init_notebook_mode(connected=False)
        plt = w.RandomizedBenchmarkingPlot(out)

        # TravisCI doesn't install matplotlib
        #plt.saveas(temp_files + "/rbdecay_plot.pdf") 
        #out.plot() # matplotlib version (keep around for now)
        return


    
    def test_rb_simulate(self):
        n = 3
        glist = ['Gi','Gxpi','Gypi','Gzpi','Gh','Gp','Gcphase']
        availability = {'Gcphase':[(0,1),(1,2)]}
        pspec = pygsti.obj.ProcessorSpec(n,glist,availability=availability,verbosity=0)
    
        errormodel = rb.simulate.create_iid_pauli_error_model(pspec, oneQgate_errorrate=0.01, twoQgate_errorrate=0.05, 
                                                              idle_errorrate=0.005, measurement_errorrate=0.05, 
                                                              ptype='uniform')
        errormodel = rb.simulate.create_iid_pauli_error_model(pspec, oneQgate_errorrate=0.001, twoQgate_errorrate=0.01, 
                                                              idle_errorrate=0.005, measurement_errorrate=0.05, 
                                                              ptype='X')
    
        out = rb.simulate.rb_with_pauli_errors(pspec,errormodel,[0,2,4],2,3,filename=temp_files + '/simtest_CRB.txt',rbtype='CRB',
                                        returndata=True, verbosity=0)
    
        errormodel = rb.simulate.create_locally_gate_independent_pauli_error_model(pspec, {0: 0.0, 1: 0.01, 2: 0.02}, 
                                                                                   {0: 0.0, 1: 0.1, 2: 0.01},ptype='uniform')
    
        out = rb.simulate.rb_with_pauli_errors(pspec,errormodel,[0,10,20],2,2,filename=temp_files + '/simtest_DRB.txt',rbtype='DRB',
                                        returndata=True, verbosity=0)


    def test_clifford_compilations(self):
    
        # Tests the Clifford compilations hard-coded into the various std gatesets. Perhaps this can be
        # automated to run over all the std gatesets that contain a Clifford compilation?
        
        from pygsti.construction import std1Q_Cliffords
        gs_target = std1Q_Cliffords.gs_target
        clifford_group = rb.group.construct_1Q_Clifford_group()
    
        from pygsti.construction import std1Q_XY
        gs_target = std1Q_XY.gs_target.copy()
        clifford_compilation = std1Q_XY.clifford_compilation
        compiled_cliffords = pygsti.construction.build_alias_gateset(gs_target,clifford_compilation)
    
        for key in list(compiled_cliffords.gates.keys()):
            self.assertLess(np.sum(abs(compiled_cliffords.gates[key]-clifford_group.get_matrix(key))), 10**(-10))
    
        from pygsti.construction import std1Q_XYI
        gs_target = std1Q_XYI.gs_target.copy()
        clifford_compilation = std1Q_XYI.clifford_compilation
        compiled_cliffords = pygsti.construction.build_alias_gateset(gs_target,clifford_compilation)
    
        for key in list(compiled_cliffords.gates.keys()):
            self.assertLess(np.sum(abs(compiled_cliffords.gates[key]-clifford_group.get_matrix(key))), 10**(-10))
    
        # Future : add the remaining Clifford compilations here.
            


if __name__ == '__main__':
    unittest.main(verbosity=2)




import unittest # noqa: E999
import numpy as np
from pygsti.forwardsims.mapforwardsim import MapForwardSimulator
from numpy.linalg import norm
import pygsti
from pygsti.modelpacks import smq1Q_XY as std
from ..testutils import BaseTestCase, compare_files, temp_files, regenerate_references

class DriversTestCase(BaseTestCase):

    def setUp(self):
        super(DriversTestCase, self).setUp()

        self.model = std.target_model()

        self.germs = std.germs(lite=True)
        self.prep_fiducials = std.prep_fiducials()[0:4]
        self.meas_fiducials = std.meas_fiducials()[0:3]
        self.maxLens = [1,2]
        self.op_labels = list(self.model.operations.keys())

        self.lsgstStrings = pygsti.circuits.create_lsgst_circuit_lists(
            self.op_labels, self.prep_fiducials, self.meas_fiducials, self.germs, self.maxLens)

        datagen_gateset = self.model.copy()
        datagen_gateset = datagen_gateset.depolarize(op_noise=0.05, spam_noise=0.1)
        self.ds = pygsti.data.simulate_data(
            datagen_gateset, self.lsgstStrings[-1],
            num_samples=1000,sample_error='binomial', seed=100)

class TestDriversMethods(DriversTestCase):
    def test_longSequenceGST_fiducialPairReduction(self):
        ds = self.ds
        maxLens = self.maxLens

        #Make list-of-lists of GST operation sequences
        fullStructs = pygsti.circuits.create_lsgst_circuit_lists(
            self.model, self.prep_fiducials, self.meas_fiducials, self.germs, self.maxLens)

        lens = [ len(strct) for strct in fullStructs ]
        self.assertEqual(lens, [19, 33])

        #Global FPR
        fidPairs = pygsti.alg.find_sufficient_fiducial_pairs(
            self.model, self.prep_fiducials, self.meas_fiducials, self.germs,
            search_mode="random", n_random=10, seed=1234,
            verbosity=0, mem_limit=int(2*(1024)**3), minimum_pairs=2,
            test_lengths = (64, 512))

        gfprStructs = pygsti.circuits.create_lsgst_circuit_lists(
            self.model, self.prep_fiducials, self.meas_fiducials, self.germs, maxLens, 
            fid_pairs=fidPairs)

        lens = [ len(strct) for strct in gfprStructs ]
        #self.assertEqual(lens, [92,100,130]) #,163,196,229]
          #can't test reliably b/c "random" above
          # means different answers on different systems

        gfprExperiments = pygsti.circuits.create_lsgst_circuits(
            self.model, self.prep_fiducials, self.prep_fiducials, self.germs, maxLens,
            fid_pairs=fidPairs)

        result = pygsti.run_long_sequence_gst_base(ds, self.model, gfprStructs, verbosity=0,
                                                   disable_checkpointing = True,
                                                   advanced_options= {'max_iterations':3})
        pygsti.report.construct_standard_report(result, title ="GFPR report", verbosity=0).write_html(temp_files + "/full_report_GFPR")

        #Per-germ FPR
        fidPairsDict = pygsti.alg.find_sufficient_fiducial_pairs_per_germ(
            self.model, self.prep_fiducials, self.meas_fiducials, self.germs,
            search_mode="random", constrain_to_tp=True,
            n_random=10, seed=1234, verbosity=0,
            mem_limit=int(2*(1024)**3))

        pfprStructs = pygsti.circuits.create_lsgst_circuit_lists(
            self.model, self.prep_fiducials, self.meas_fiducials, self.germs, maxLens,
            fid_pairs=fidPairsDict) #note: fidPairs arg can be a dict too!

        lens = [ len(strct) for strct in pfprStructs ]
        #self.assertEqual(lens, [92,99,138]) # ,185,233,281]
          #can't test reliably b/c "random" above
          # means different answers on different systems


        pfprExperiments = pygsti.circuits.create_lsgst_circuits(
            self.model, self.prep_fiducials, self.meas_fiducials, self.germs, maxLens,
            fid_pairs=fidPairsDict)

        result = pygsti.run_long_sequence_gst_base(ds, self.model, pfprStructs, verbosity=0,
                                                   disable_checkpointing = True,
                                                   advanced_options= {'max_iterations':3})
        pygsti.report.construct_standard_report(result, title="PFPR report", verbosity=0).write_html(temp_files + "/full_report_PFPR")

    def test_longSequenceGST_randomReduction(self):
        ds = self.ds
        ts = "whole germ powers"
        maxLens = self.maxLens

        #Without fixed initial fiducial pairs
        fidPairs = None
        reducedLists = pygsti.circuits.create_lsgst_circuit_lists(
            self.model.operations.keys(), self.prep_fiducials, self.meas_fiducials, self.germs,
            maxLens, fidPairs, ts, keep_fraction=0.25, keep_seed=1234)
        result = self.runSilent(pygsti.run_long_sequence_gst_base,
                                ds, self.model, reducedLists,
                                advanced_options={'truncScheme': ts},
                                disable_checkpointing=True)

        #create a report...
        pygsti.report.construct_standard_report(result, title="RFPR report", verbosity=0).write_html(temp_files + "/full_report_RFPR")

    def test_longSequenceGST_CPTP(self):
        ds = self.ds

        target_model = self.model
        target_model.set_all_parameterizations("CPTPLND")

        maxLens = self.maxLens
        result = self.runSilent(pygsti.run_long_sequence_gst,
                                ds, target_model, self.prep_fiducials, self.meas_fiducials,
                                self.germs, maxLens, disable_checkpointing=True, 
                                advanced_options= {'max_iterations':3})

        #create a report...
        pygsti.report.construct_standard_report(result, title="CPTP Gates report", verbosity=0).write_html(temp_files + "/full_report_CPTPGates")


    def test_longSequenceGST_Sonly(self):
        ds = self.ds
        
        target_model = self.model.copy()
        target_model.set_all_parameterizations("S")

        maxLens = self.maxLens
        result = self.runSilent(pygsti.run_long_sequence_gst,
                                ds, target_model, self.prep_fiducials, self.meas_fiducials,
                                self.germs, maxLens, disable_checkpointing=True, advanced_options= {'max_iterations':3})

        #create a report...
        pygsti.report.construct_standard_report(result, title="SGates report", verbosity=0).write_html(temp_files + "/full_report_SGates")


    def test_longSequenceGST_GLND(self):
        #General Lindbladian parameterization (allowed to be non-CPTP)
        ds = self.ds
        
        target_model = self.model.copy()

        #No set_all_parameterizations option for this one, since it probably isn't so useful
        for lbl,gate in target_model.operations.items():
            target_model.operations[lbl] = pygsti.modelmembers.operations.convert(gate, "GLND", "gm")
        target_model.default_gauge_group = pygsti.models.gaugegroup.UnitaryGaugeGroup(target_model.state_space, "gm")
          #Lindblad gates only know how to do unitary transforms currently, even though
          # in the non-cptp case it they should be able to transform generally.

        maxLens = self.maxLens
        result = self.runSilent(pygsti.run_long_sequence_gst,
                                ds, target_model, self.prep_fiducials, self.meas_fiducials,
                                self.germs, maxLens, disable_checkpointing=True, advanced_options= {'max_iterations':3})

        #create a report...
        pygsti.report.construct_standard_report(result, title="GLND report", verbosity=0).write_html( temp_files + "/full_report_GLND")


    def test_longSequenceGST_HplusS(self):
        ds = self.ds
        
        target_model = self.model.copy()
        target_model.set_all_parameterizations("H+S")

        maxLens = self.maxLens
        result = self.runSilent(pygsti.run_long_sequence_gst,
                                ds, target_model, self.prep_fiducials, self.meas_fiducials,
                                self.germs, maxLens, disable_checkpointing=True, advanced_options= {'max_iterations':3})

        #create a report...
        pygsti.report.construct_standard_report(result, title= "HpS report", verbosity=0).write_html(temp_files + "/full_report_HplusSGates")

    def test_longSequenceGST_badfit(self):
        ds = self.ds

        #lower bad-fit threshold to zero to trigger bad-fit additional processing
        maxLens = self.maxLens
        result = self.runSilent(pygsti.run_long_sequence_gst,
                                ds, self.model.copy(), self.prep_fiducials, self.meas_fiducials,
                                self.germs, maxLens, advanced_options={'bad_fit_threshold': -100, 'max_iterations':3},
                                disable_checkpointing=True)

        pygsti.report.construct_standard_report(result, title="badfit report", verbosity=0).write_html(temp_files + "/full_report_badfit")

    def test_stdpracticeGST(self):
        ds = self.ds
        mdl_guess = self.model.copy()
        mdl_guess = mdl_guess.depolarize(op_noise=0.01,spam_noise=0.01)

        #lower bad-fit threshold to zero to trigger bad-fit additional processing
        maxLens = self.maxLens
        result = self.runSilent(pygsti.run_stdpractice_gst,
                                ds, self.model.copy(), self.prep_fiducials, self.meas_fiducials,
                                self.germs, maxLens, modes=['CPTPLND','Test','Target'],
                                models_to_test = {"Test": mdl_guess},
                                comm=None, mem_limit=None, verbosity=0,
                                disable_checkpointing=True, advanced_options= {'max_iterations':3})
        pygsti.report.construct_standard_report(result, title= "Std Practice Test Report", verbosity=2).write_html(temp_files + "/full_report_stdpractice")

    def test_bootstrap(self):
        """Test bootstrap model generation"""
        ds = self.ds
        tp_target = self.model.copy()
        tp_target.set_all_parameterizations("full TP")
        mdl = pygsti.run_lgst(ds, self.prep_fiducials, self.meas_fiducials, target_model=tp_target, svd_truncate_to=4, verbosity=0)

        default_maxLens = [2**k for k in range(4)]
        circuits = pygsti.circuits.create_lsgst_circuits(
            self.op_labels, self.prep_fiducials, self.meas_fiducials, self.germs,
            default_maxLens, fid_pairs=None, trunc_scheme="whole germ powers")
        ds_defaultMaxLens = pygsti.data.simulate_data(
            mdl, circuits, num_samples=10000, sample_error='round')

        bootgs_p_defaultMaxLens = \
            pygsti.drivers.create_bootstrap_models(
                2, ds_defaultMaxLens, 'parametric', self.prep_fiducials, self.meas_fiducials,
                self.germs, default_maxLens, input_model=mdl, target_model=tp_target,
                return_data=False) #test when max_lengths == None ?
                
    def test_GST_checkpointing(self):
        ds= self.ds
        maxLens = self.maxLens
        
        target_model = self.model.copy()

        #Make list-of-lists of GST operation sequences
        fullStructs = pygsti.circuits.create_lsgst_circuit_lists(
            target_model, self.prep_fiducials, self.meas_fiducials, self.germs, maxLens)
        
        #Test GateSetTomographyCheckpoint:
        #First run from scratch:
        result_gst = pygsti.run_long_sequence_gst_base(ds, target_model.copy(), fullStructs, verbosity=0, 
                                                   checkpoint_path= temp_files + '/checkpoint_testing/GateSetTomography',
                                                   advanced_options= {'max_iterations':3})
                                                   
        #double check that we can read in this checkpoint object correctly:
        gst_checkpoint = pygsti.protocols.GateSetTomographyCheckpoint.read(temp_files + '/checkpoint_testing/GateSetTomography_iteration_0.json')
        
        #run GST using this checkpoint
        result_gst_warmstart = pygsti.run_long_sequence_gst_base(ds, target_model.copy(), fullStructs, verbosity=0,
                                                                 checkpoint = gst_checkpoint,
                                                                 checkpoint_path= temp_files + '/checkpoint_testing/GateSetTomography',
                                                                 advanced_options= {'max_iterations':3})
        
        diff = norm(result_gst.estimates['GateSetTomography'].models['final iteration estimate'].to_vector()-
                         result_gst_warmstart.estimates['GateSetTomography'].models['final iteration estimate'].to_vector())
        #Assert that this gives the same result as before:
        self.assertTrue(diff<=1e-10)
            

    def test_ModelTest_checkpointing(self):
        ds = self.ds
        maxLens = self.maxLens
            
        target_model = self.model.copy()
        
        #Next test ModelTestCheckpoint
        #First run from scratch:
        result_modeltest = pygsti.run_model_test(target_model.copy(), ds, target_model.create_processor_spec(), 
                                                 self.prep_fiducials, self.meas_fiducials, self.germs,
                                                 maxLens, verbosity=0, 
                                                 checkpoint_path= temp_files + '/checkpoint_testing/ModelTest')
                                                   
        #double check that we can read in this checkpoint object correctly:
        model_test_checkpoint = pygsti.protocols.ModelTestCheckpoint.read(temp_files + '/checkpoint_testing/ModelTest_iteration_0.json')
        
        #run GST using this checkpoint
        result_modeltest_warmstart = pygsti.run_model_test(target_model.copy(), ds,target_model.create_processor_spec(), 
                                                           self.prep_fiducials, self.meas_fiducials, self.germs,
                                                           maxLens, verbosity=0,
                                                           checkpoint = model_test_checkpoint,
                                                           checkpoint_path= temp_files + '/checkpoint_testing/ModelTest')
        
        diff = norm(np.array(result_modeltest.estimates['ModelTest'].parameters['model_test_values'])- 
                         np.array(result_modeltest_warmstart.estimates['ModelTest'].parameters['model_test_values']))
        #Assert that this gives the same result as before:
        self.assertTrue(diff<=1e-10)
                     
                     
    def test_StandardGST_checkpointing(self):
        ds= self.ds
        maxLens = self.maxLens

        #Finally test StandardGSTCheckpoint
        #First run from scratch:
        mdl_guess = self.model.copy()
        mdl_guess = mdl_guess.depolarize(op_noise=0.01,spam_noise=0.01)
                
        result_standardgst = pygsti.run_stdpractice_gst(ds, self.model.copy(), self.prep_fiducials, self.meas_fiducials,
                                                        self.germs, maxLens, modes=['full TP','CPTPLND','Test','Target'],
                                                        models_to_test = {"Test": mdl_guess},
                                                        comm=None, mem_limit=None, verbosity=0,
                                                        checkpoint_path= temp_files + '/checkpoint_testing/StandardGST',
                                                        advanced_options= {'max_iterations':3})
                                                   
        #double check that we can read in this checkpoint object correctly:
        standardgst_checkpoint = pygsti.protocols.StandardGSTCheckpoint.read(temp_files + '/checkpoint_testing/StandardGST_CPTPLND_iteration_1.json')
        
        #run GST using this checkpoint
        result_standardgst_warmstart = pygsti.run_stdpractice_gst(ds, self.model.copy(), self.prep_fiducials, self.meas_fiducials,
                                                                  self.germs, maxLens, modes=['full TP','CPTPLND','Test','Target'],
                                                                  models_to_test = {"Test": mdl_guess},
                                                                  comm=None, mem_limit=None, verbosity=0,
                                                                  checkpoint = standardgst_checkpoint,
                                                                  checkpoint_path= temp_files + '/checkpoint_testing/StandardGST',
                                                                  advanced_options= {'max_iterations':3})

        #Assert that this gives the same result as before:
        diff = norm(result_standardgst.estimates['CPTPLND'].models['final iteration estimate'].to_vector()- 
                         result_standardgst_warmstart.estimates['CPTPLND'].models['final iteration estimate'].to_vector())               
        diff1 = norm(result_standardgst.estimates['full TP'].models['final iteration estimate'].to_vector()- 
                     result_standardgst_warmstart.estimates['full TP'].models['final iteration estimate'].to_vector())
        
        self.assertTrue(abs(diff)<=1e-10)
        self.assertTrue(diff1<=1e-10)

if __name__ == "__main__":
    unittest.main(verbosity=2)

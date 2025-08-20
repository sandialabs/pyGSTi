import time
import numpy as _np
from pygsti.baseobjs import Label
from pygsti.modelpacks import smq2Q_XYICNOT, smq1Q_XYI
from pygsti.tools import edesigntools as et
from pygsti.protocols import CircuitListsDesign, SimultaneousExperimentDesign, CombinedExperimentDesign
from pygsti.circuits import Circuit as C
from pygsti.circuits import create_lsgst_circuit_lists
from pygsti.modelmembers.instruments import TPInstrument

from ..util import BaseCase


class ExperimentDesignTimeEstimationTester(BaseCase):

    def test_time_estimation(self):
        edesign = smq2Q_XYICNOT.create_gst_experiment_design(8)
        
        # Dummy test: No time
        time0 = et.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_1Q=0,
            gate_time_2Q=0,
            measure_reset_time=0,
            interbatch_latency=0,
        )
        self.assertAlmostEqual(time0, 0.0)
    
        # Dummy test: 1 second for each circuit shot
        time0 = et.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_1Q=0,
            gate_time_2Q=0,
            measure_reset_time=1,
            interbatch_latency=0,
            total_shots_per_circuit=1000
        )
        self.assertAlmostEqual(time0, 1000*len(edesign.all_circuits_needing_data))

        # Dummy test: 1 second for each circuit shot, + 10 s for each circuit due to batching
        time0 = et.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_1Q=0,
            gate_time_2Q=0,
            measure_reset_time=1,
            interbatch_latency=10,
            total_shots_per_circuit=1000,
            circuits_per_batch=1
        )
        self.assertAlmostEqual(time0, 1010*len(edesign.all_circuits_needing_data))

        # Dummy test: 1 second for each circuit shot, + (10 s for each circuit due to batching x 10 rounds)
        time0 = et.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_1Q=0,
            gate_time_2Q=0,
            measure_reset_time=1,
            interbatch_latency=10,
            total_shots_per_circuit=1000,
            shots_per_circuit_per_batch=100,
            circuits_per_batch=1
        )
        self.assertAlmostEqual(time0, 1100*len(edesign.all_circuits_needing_data))


        # Try dict version of trapped ion example
        gate_times = {
            'Gxpi2': 10e-6,
            'Gypi2': 10e-6,
            'Gcnot': 100e-6,
            Label(()): 10e-6,
        }
        time1 = et.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_dict=gate_times,
            measure_reset_time=500e-6,
            interbatch_latency=0.1,
            total_shots_per_circuit=1000,
            shots_per_circuit_per_batch=100,
            circuits_per_batch=200
        )

        # Try equivalent gate time version
        time2 = et.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_1Q=10e-6,
            gate_time_2Q=100e-6,
            measure_reset_time=500e-6,
            interbatch_latency=0.1,
            total_shots_per_circuit=1000,
            shots_per_circuit_per_batch=100,
            circuits_per_batch=200
        )
        self.assertAlmostEqual(time1, time2)

        # Qubit-specific overload
        gate_times2 = {
            'Gxpi2': 10e-6,
            ('Gxpi2', 0): 20e-6,
            'Gypi2': 10e-6,
            'Gcnot': 100e-6,
            Label(()): 10e-6,
        }
        time3 = et.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_dict=gate_times2,
            measure_reset_time=500e-6,
            interbatch_latency=0.1,
            total_shots_per_circuit=1000,
            shots_per_circuit_per_batch=100,
            circuits_per_batch=200
        )
        self.assertGreater(time3, time1)
    
class FisherInformationTester(BaseCase):

    def setUp(self):

        self.target_model = smq1Q_XYI.target_model('full TP')
        self.edesign = smq1Q_XYI.create_gst_experiment_design(8)
        self.Ls = [1,2,4,8]
        #create a model with instruments too.
        self.target_model_inst = self.target_model.copy()
        #Create and add the ideal instrument
        #E0 = target_model.effects['0']
        #E1 = target_model.effects['1']
        # Alternate indexing that uses POVM label explicitly
        E0 = self.target_model['Mdefault']['0']  # 'Mdefault' = POVM label, '0' = effect label
        E1 = self.target_model['Mdefault']['1']
        Gmz_plus = _np.dot(E0,E0.T) #note effect vectors are stored as column vectors
        Gmz_minus = _np.dot(E1,E1.T)
        self.target_model_inst[('Iz',0)] = TPInstrument({'p0': Gmz_plus, 'p1': Gmz_minus})

        #create experiment design for instruments
        germs = smq1Q_XYI.germs()
        germs += [C([('Iz', 0)])]  # add the instrument as a germ.

        prep_fiducials = smq1Q_XYI.prep_fiducials()
        meas_fiducials = smq1Q_XYI.meas_fiducials()
        self.lsgst_list_instruments = create_lsgst_circuit_lists(
            self.target_model_inst,prep_fiducials,meas_fiducials,germs,self.Ls)

    def test_calculate_fisher_information_matrix(self):
        
        # Basic usage
        start = time.time()
        fim1 = et.calculate_fisher_information_matrix(self.target_model, self.edesign.all_circuits_needing_data)
        fim1_time = time.time() - start
    
        # Try pre-cached version
        fim2_terms, _ = et.calculate_fisher_information_per_circuit(self.target_model, self.edesign.all_circuits_needing_data)
        start = time.time()
        fim2 = et.calculate_fisher_information_matrix(self.target_model, self.edesign.all_circuits_needing_data, term_cache=fim2_terms)
        fim2_time = time.time() - start
        
        self.assertArraysAlmostEqual(fim1, fim2)
        self.assertLess(10*fim2_time, fim1_time) # Cached version should be very fast compared to uncached

    def test_calculate_fisher_info_by_L(self):

        fim1 = et.calculate_fisher_information_matrix(self.target_model, self.edesign.all_circuits_needing_data)

        # Try by-L version
        fim_by_L = et.calculate_fisher_information_matrices_by_L(self.target_model, self.edesign.circuit_lists, self.Ls)
        self.assertTrue(_np.linalg.norm(fim1-fim_by_L[8])<1e-3)

    #test approximate versions of the fisher information calculation.
    def test_fisher_information_approximate(self):

        #Test approximate fisher information calculations:
        fim_approx = et.calculate_fisher_information_matrix(self.target_model, self.edesign.all_circuits_needing_data, 
                                                            approx=True)

        #test per-circuit
        fim_approx_per_circuit = et.calculate_fisher_information_per_circuit(self.target_model, 
                                                                             self.edesign.all_circuits_needing_data, 
                                                                             approx=True)

        #Test by L:
        fim_approx_by_L = et.calculate_fisher_information_matrices_by_L(self.target_model, self.edesign.circuit_lists, self.Ls,
                                                                        approx=True)
        self.assertTrue(_np.linalg.norm(fim_approx-fim_approx_by_L[8])<1e-3)

    def test_calculate_fisher_information_matrix_with_instrument(self):
        #Test approximate fisher information calculations:
        fim_approx = et.calculate_fisher_information_matrix(self.target_model_inst, self.lsgst_list_instruments[-1], 
                                                            approx=True)

        #test per-circuit
        fim_approx_per_circuit = et.calculate_fisher_information_per_circuit(self.target_model_inst, 
                                                                             self.lsgst_list_instruments[-1], 
                                                                             approx=True)

        #Test by L:
        fim_approx_by_L = et.calculate_fisher_information_matrices_by_L(self.target_model_inst, self.lsgst_list_instruments, self.Ls,
                                                                        approx=True)
        self.assertTrue(_np.linalg.norm(fim_approx-fim_approx_by_L[8])<1e-3)


class EdesignPaddingTester(BaseCase):

    def test_generic_design_padding(self):
        # Create a series of designs with some overlap when they will be padded out
        design_124 = CircuitListsDesign([[
            C.cast('Gx:Q1Gy:Q1@(Q1,Q2,Q4)'), 
            C.cast('Gx:Q2Gy:Q2@(Q1,Q2,Q4)'), # Will be repeat with design_2
            C.cast('Gx:Q4Gy:Q4@(Q1,Q2,Q4)'), # Will be repeat with design_14 (but only on Q4)
            C.cast('Gx:Q1Gy:Q4@(Q1,Q2,Q4)'), # Will be repeat with design_14 (on both Q1 and Q4)
            C.cast('[Gx:Q1Gy:Q2][Gy:Q1Gx:Q2]@(Q1,Q2,Q4)') # Will be repeat with sim_design_12
        ]], qubit_labels=('Q1', 'Q2', 'Q4'))

        design_2 = CircuitListsDesign([[
            C.cast('Gx:Q2Gy:Q2@(Q2)'), # Repeat from design_124 after padding
            C.cast('Gy:Q2@(Q2)')
        ]], qubit_labels=('Q2',))

        design_14 = CircuitListsDesign([[
            C.cast('Gx:Q4Gy:Q4@(Q1,Q4)'), # Repeat from design_124 after padding
            C.cast('Gx:Q1Gy:Q4@(Q1,Q4)'), # Repeat from design_124 after padding
            C.cast('Gx:Q1@(Q1,Q4)')
        ]], qubit_labels=('Q1', 'Q4'))

        sim_design_1 = CircuitListsDesign([[
            C.cast('Gx:Q1Gy:Q1@(Q1)'), # Q1 part of repeat from design_124 after padding
            C.cast('Gx:Q1Gx:Q1')
        ]], qubit_labels=('Q1',))

        sim_design_2 = CircuitListsDesign([[
            C.cast('Gy:Q2Gx:Q2@(Q2)'), # Q2 part of repeat from design_124 after padding
            C.cast('Gx:Q2Gx:Q2')
        ]], qubit_labels=('Q2',))

        sim_design_12 = SimultaneousExperimentDesign([sim_design_1, sim_design_2], qubit_labels=('Q1', 'Q2'))

        # The expected deduplicated experiment
        expected_design_012345 = CircuitListsDesign([[
            C.cast('Gx:Q1Gy:Q1@(Q0,Q1,Q2,Q3,Q4,Q5)'), # design_124
            C.cast('Gx:Q2Gy:Q2@(Q0,Q1,Q2,Q3,Q4,Q5)'), # design_124 and design_2
            C.cast('Gx:Q4Gy:Q4@(Q0,Q1,Q2,Q3,Q4,Q5)'), # design_124 and design_14
            C.cast('Gx:Q1Gy:Q4@(Q0,Q1,Q2,Q3,Q4,Q5)'), # design_124 and design_14
            C.cast('Gy:Q2@(Q0,Q1,Q2,Q3,Q4,Q5)'), # design_2
            C.cast('Gx:Q1@(Q0,Q1,Q2,Q3,Q4,Q5)'), # design_14
            C.cast('[Gx:Q1Gy:Q2][Gy:Q1Gx:Q2]@(Q0,Q1,Q2,Q3,Q4,Q5)'), # design_124 and sim_design_12
            C.cast('[Gx:Q1Gx:Q2][Gx:Q1Gx:Q2]@(Q0,Q1,Q2,Q3,Q4,Q5)') # sim_design_12
        ]], qubit_labels=('Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'))


        # Create nested combined designs and test padding
        nested_design = CombinedExperimentDesign({
            '2': design_2,
            '14': design_14
        })

        full_design = CombinedExperimentDesign({
            '124': design_124,
            '2+14': nested_design,
            'sim_12': sim_design_12
        })

        # Padding should dedup "repeats" and add qubits before/during/after the current lines
        padded_design = et.pad_edesign_with_idle_lines(full_design, ('Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'))

        self.assertTrue(set(padded_design.all_circuits_needing_data) == set(expected_design_012345.all_circuits_needing_data),
            "Padded experiment circuits did not match expected experiment circuits")
    
    def test_gst_design_padding(self):
        # Get GST designs
        gst_1 = smq1Q_XYI.create_gst_experiment_design(8, ('Q1',))
        gst_2 = smq1Q_XYI.create_gst_experiment_design(8, ('Q2',))
        gst_12 = smq2Q_XYICNOT.create_gst_experiment_design(8, ('Q1', 'Q2'))

        # Get nested combined design
        nested_12 = CombinedExperimentDesign({
            '1': gst_1,
            '2': gst_2,
        })

        full_gst = CombinedExperimentDesign({
            '1+2': nested_12,
            '12': gst_12
        })

        # Pad and test
        padded_gst_design = et.pad_edesign_with_idle_lines(full_gst, ('Q1', 'Q2'))

        padded_circs_1 = [circ.insert_idling_lines(None, ('Q2',)) for circ in gst_1.all_circuits_needing_data]
        self.assertTrue(set(padded_gst_design.all_circuits_needing_data).issuperset(set(padded_circs_1)),
            "GST on qubit 1 was not a subset of padded experiment design")

        padded_circs_2 = [circ.insert_idling_lines('Q2', ('Q1',)) for circ in gst_2.all_circuits_needing_data]
        self.assertTrue(set(padded_gst_design.all_circuits_needing_data).issuperset(set(padded_circs_2)),
            "GST on qubit 2 was not a subset of the padded experiment design")

        self.assertTrue(set(padded_gst_design.all_circuits_needing_data).issuperset(set(gst_12.all_circuits_needing_data)),
            "2Q GST was not a subset of the padded experiment design")
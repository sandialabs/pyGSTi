import time

from pygsti.baseobjs import Label
from pygsti.modelpacks import smq2Q_XYICNOT, smq1Q_XYI
from pygsti.tools import edesigntools as et

from ..util import BaseCase


class EdesignToolsTester(BaseCase):

    def test_time_estimation(self):
        edesign = smq2Q_XYICNOT.create_gst_experiment_design(256)
        
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
    
    def test_fisher_information(self):
        target_model = smq1Q_XYI.target_model('TP')
        edesign = smq1Q_XYI.create_gst_experiment_design(8)

        # Basic usage
        start = time.time()
        fim1 = et.calculate_fisher_information_matrix(target_model, edesign.all_circuits_needing_data)
        fim1_time = time.time() - start

        # Try external regularized model version
        regularized_model = target_model.copy().depolarize(spam_noise=1e-3)
        fim2 = et.calculate_fisher_information_matrix(regularized_model, edesign.all_circuits_needing_data,
                                                      regularize_spam=False)
        self.assertArraysAlmostEqual(fim1, fim2)

        # Try pre-cached version
        fim3_terms = et.calculate_fisher_information_per_circuit(regularized_model, edesign.all_circuits_needing_data)
        start = time.time()
        fim3 = et.calculate_fisher_information_matrix(target_model, edesign.all_circuits_needing_data, term_cache=fim3_terms)
        fim3_time = time.time() - start
        
        self.assertArraysAlmostEqual(fim1, fim3)
        self.assertLess(10*fim3_time, fim1_time) # Cached version should be very fast compared to uncached

        # Try by-L version
        fim_by_L = et.calculate_fisher_information_matrices_by_L(target_model, edesign.all_circuits_needing_data)
        self.assertArraysAlmostEqual(fim1, fim_by_L[8])

        # Try pre-cached by-L version
        start = time.time()
        fim_by_L2 = et.calculate_fisher_information_matrices_by_L(target_model, edesign.all_circuits_needing_data, term_cache=fim3_terms)
        fim_by_L2_time = time.time() - start
        for k,v in fim_by_L2.items():
            self.assertArraysAlmostEqual(v, fim_by_L[k])
        self.assertLess(10*fim_by_L2_time, fim1_time) # Cached version should be very fast compared to uncached



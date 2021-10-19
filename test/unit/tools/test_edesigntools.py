from pygsti.baseobjs import Label
from pygsti.modelpacks import smq2Q_XYICNOT
from pygsti.tools import edesigntools

from ..util import BaseCase


class EdesignToolsTester(BaseCase):

    def test_time_estimation(self):
        edesign = smq2Q_XYICNOT.create_gst_experiment_design(256)
        
        # Dummy test: No time
        time0 = edesigntools.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_1Q=0,
            gate_time_2Q=0,
            measure_reset_time=0,
            interbatch_latency=0,
        )
        self.assertAlmostEqual(time0, 0.0)
    
        # Dummy test: 1 second for each circuit shot
        time0 = edesigntools.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_1Q=0,
            gate_time_2Q=0,
            measure_reset_time=1,
            interbatch_latency=0,
            total_shots_per_circuit=1000
        )
        self.assertAlmostEqual(time0, 1000*len(edesign.all_circuits_needing_data))

        # Dummy test: 1 second for each circuit shot, + 10 s for each circuit due to batching
        time0 = edesigntools.calculate_edesign_estimated_runtime(
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
        time0 = edesigntools.calculate_edesign_estimated_runtime(
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
        time1 = edesigntools.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_dict=gate_times,
            measure_reset_time=500e-6,
            interbatch_latency=0.1,
            total_shots_per_circuit=1000,
            shots_per_circuit_per_batch=100,
            circuits_per_batch=200
        )

        # Try equivalent gate time version
        time2 = edesigntools.calculate_edesign_estimated_runtime(
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
        time3 = edesigntools.calculate_edesign_estimated_runtime(
            edesign,
            gate_time_dict=gate_times2,
            measure_reset_time=500e-6,
            interbatch_latency=0.1,
            total_shots_per_circuit=1000,
            shots_per_circuit_per_batch=100,
            circuits_per_batch=200
        )
        self.assertGreater(time3, time1)

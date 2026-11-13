from ..util import BaseCase

import numpy as _np

from pygsti.protocols import scarab


class ScarabBenchmarkTester(BaseCase):
    """
    Smoke tests for the scarab benchmark-creation wrappers. These are thin wrappers
    around the mirror_edesign pipeline (tested in depth in test_mirror_edesign.py),
    so these tests just exercise each wrapper end-to-end on small circuits and check
    the shape of the returned edesigns.
    """

    def _require_qiskit(self):
        try:
            import qiskit
            return qiskit
        except ImportError:
            self.skipTest('Qiskit is required for this operation, and does not appear to be installed.')

    def test_lowlevel_mirror_benchmark(self):
        qiskit = self._require_qiskit()

        num_mcs_per_circ = 2
        num_ref_per_qubit_subset = 3

        qc = qiskit.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qk_circ = qiskit.transpile(qc, basis_gates=['u3', 'cz'], optimization_level=1,
                                   seed_transpiler=0)

        test_edesign, mirror_edesign = scarab.lowlevel_mirror_benchmark(
            [qk_circ],
            mirroring_kwargs_dict={'num_mcs_per_circ': num_mcs_per_circ,
                                   'num_ref_per_qubit_subset': num_ref_per_qubit_subset,
                                   'rand_state': _np.random.RandomState(0)})

        self.assertEqual(len(test_edesign.aux_info), 1)
        self.assertEqual(2 * num_mcs_per_circ + num_ref_per_qubit_subset,
                         len(mirror_edesign.all_circuits_needing_data))
        for key in ('br', 'rr', 'ref'):
            self.assertTrue(len(mirror_edesign[key].all_circuits_needing_data) > 0)

    def test_fullstack_mirror_benchmark(self):
        qiskit = self._require_qiskit()
        from qiskit.providers.fake_provider import GenericBackendV2

        num_mcs_per_circ = 2
        num_ref_per_qubit_subset = 3

        backend = GenericBackendV2(num_qubits=2, seed=0)
        qc = qiskit.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        test_edesign, mirror_edesign = scarab.fullstack_mirror_benchmark(
            [qc],
            qk_backend=backend,
            transpiler_kwargs_dict={'seed_transpiler': 0},
            mirroring_kwargs_dict={'num_mcs_per_circ': num_mcs_per_circ,
                                   'num_ref_per_qubit_subset': num_ref_per_qubit_subset,
                                   'rand_state': _np.random.RandomState(0)})

        self.assertEqual(len(test_edesign.aux_info), 1)
        self.assertEqual(2 * num_mcs_per_circ + num_ref_per_qubit_subset,
                         len(mirror_edesign.all_circuits_needing_data))

        # full-stack benchmarks track the transpiler's layout choices
        for auxlist in test_edesign.aux_info.values():
            for aux in auxlist:
                self.assertIn('routing_permutation', aux)

    def test_subcircuit_mirror_benchmark(self):
        qiskit = self._require_qiskit()
        from qiskit.transpiler import CouplingMap, InstructionDurations

        coupling_map = CouplingMap.from_line(4)
        instruction_durations = InstructionDurations(
            [('u', None, 100), ('cz', None, 100), ('delay', None, 100)])

        qc = qiskit.QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qk_circ = qiskit.transpile(qc, basis_gates=['u3', 'cz'], coupling_map=coupling_map,
                                   optimization_level=1, seed_transpiler=0)

        width_depth_dict = {2: [2, 3]}
        num_samples_per_width_depth = 2

        test_edesign, mirror_edesign = scarab.subcircuit_mirror_benchmark(
            [qk_circ],
            aggregate_subcircs=True,
            width_depth_dict=width_depth_dict,
            coupling_map=coupling_map,
            instruction_durations=instruction_durations,
            subcirc_kwargs_dict={'num_samples_per_width_depth': num_samples_per_width_depth,
                                 'rand_state': _np.random.RandomState(0)},
            mirroring_kwargs_dict={'num_mcs_per_circ': 1,
                                   'num_ref_per_qubit_subset': 1,
                                   'rand_state': _np.random.RandomState(0)})

        # 2 subcircuits per (width, depth) pair were requested
        flat_aux = [aux for auxlist in test_edesign.aux_info.values() for aux in auxlist]
        self.assertEqual(len(flat_aux), num_samples_per_width_depth * 2)
        for aux in flat_aux:
            self.assertEqual(aux['width'], 2)
            self.assertIn(aux['depth'], (2, 3))

        self.assertTrue(len(mirror_edesign.all_circuits_needing_data) > 0)

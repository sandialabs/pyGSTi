from ..util import BaseCase

from pygsti.circuits import subcircuit_selection as _subcircsel, Circuit as C
from pygsti.baseobjs import Label as L
from pygsti.tools.exceptions import MissingDependencyWarning

import numpy as _np
import networkx as _nx
import warnings

from collections import defaultdict


# `sample_subcircuits` emits MissingDependencyWarning on every call when
# qiskit isn't installed at the expected version, regardless of whether
# the test passes a qiskit-typed input. Tests in this file pass strings
# / duck-typed inputs and don't actually need qiskit; suppress the
# resulting noise via a context manager around each call.

class TestSubcircuitSelection(BaseCase):
    def test_random_subgraph_creation(self):

        rand_state = _np.random.RandomState(0)
        
        n = 100
        g = _nx.gnp_random_graph(n, 2 * _np.log(n)/n, seed=0)

        widths = list(range(1,21))

        subgraphs = [_subcircsel.random_connected_subgraph(g, width, rand_state) for width in widths]


        self.assertTrue(all(len(subgraphs[i]) == widths[i] for i in range(len(widths))))
        self.assertTrue(all(_nx.is_connected(g.subgraph(subgraph)) for subgraph in subgraphs))


    def test_random_subgraph_reproducibility(self):

        rand_state = _np.random.RandomState(0)
        
        n = 100
        g = _nx.gnp_random_graph(n, 2 * _np.log(n)/n, seed=0)

        widths = list(range(1,21))

        subgraphs_1 = [_subcircsel.random_connected_subgraph(g, width, rand_state) for width in widths]

        rand_state = _np.random.RandomState(0)
        subgraphs_2 = [_subcircsel.random_connected_subgraph(g, width, rand_state) for width in widths]

        self.assertTrue(len(widths) == len(subgraphs_1))
        self.assertTrue(len(widths) == len(subgraphs_2))

        self.assertTrue(all(subgraphs_1[i] == subgraphs_2[i] for i in range(len(widths))))


    def test_simple_subcirc_selection_linear_connectivity(self):

        rand_state = _np.random.RandomState(0)

        class NoDelayInstructions(object):
            def get(self, *args):
                return 0.0
            
        line_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

        I_args = [0,0,0]
        X_args = [_np.pi,0,_np.pi]
        Y_args = [_np.pi,_np.pi/2,_np.pi/2]
        Z_args = [0,0,_np.pi]

        layers = [
            [L('Gu3', ['Q1'], args=Z_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcnot', ['Q1', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q3'], args=None)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=Z_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=X_args)],
            [L('Gcnot', ['Q2', 'Q3'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q3', 'Q4'], args=None)],
            [L('Gcnot', ['Q3', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=X_args), L('Gu3', ['Q3'], args=Z_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q4', 'Q5'], args=None)]
                  ]
                
        circ = C(layers, line_labels=line_labels)

        width_depths = {2: [2,4,6],
                        3: [3,6,9]}
        
        num_subcircs_per_width_depth = 5
        

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingDependencyWarning)
            design = _subcircsel.sample_subcircuits(circ, width_depths=width_depths,
                                                    instruction_durations=NoDelayInstructions(),
                                                    coupling_map='linear',
                                                    num_samples_per_width_depth=num_subcircs_per_width_depth,
                                                    rand_state=rand_state
                                                    )

        # check that the width and depth metadata agree with the circuit width and depth for each circuit in the design.
        for subcirc, auxlist in design.aux_info.items():
            self.assertTrue(all(subcirc.width == aux['width'] for aux in auxlist))
            self.assertTrue(all(subcirc.depth == aux['depth'] for aux in auxlist))

        # check that the correct number of circuits of each width and depth exist by looking at aux_info
        reshaped_width_depths = []
        for width, depths in width_depths.items():
            for depth in depths:
                reshaped_width_depths.append((width, depth))

        created_width_depths = defaultdict(int)
        for subcirc, auxlist in design.aux_info.items():
            created_width_depths[(subcirc.width, subcirc.depth)] += len(auxlist)

        self.assertListEqual(sorted(reshaped_width_depths), sorted(list(created_width_depths.keys())))
        self.assertTrue(all(created_width_depths[k] == num_subcircs_per_width_depth for k in reshaped_width_depths))



    def test_simple_subcirc_selection_all_to_all_connectivity(self):

        rand_state = _np.random.RandomState(0)

        class NoDelayInstructions(object):
            def get(self, *args):
                return 0.0

        line_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

        I_args = [0,0,0]
        X_args = [_np.pi,0,_np.pi]
        Y_args = [_np.pi,_np.pi/2,_np.pi/2]
        Z_args = [0,0,_np.pi]

        layers = [
            [L('Gu3', ['Q1'], args=Z_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcnot', ['Q1', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q3'], args=None)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=Z_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=X_args)],
            [L('Gcnot', ['Q2', 'Q3'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q3', 'Q4'], args=None)],
            [L('Gcnot', ['Q3', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=X_args), L('Gu3', ['Q3'], args=Z_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q4', 'Q5'], args=None)]
                  ]

        circ = C(layers, line_labels=line_labels)

        width_depths = {2: [2,4,6],
                        3: [3,6,9]}

        num_subcircs_per_width_depth = 20


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingDependencyWarning)
            design = _subcircsel.sample_subcircuits(circ, width_depths=width_depths,
                                                    instruction_durations=NoDelayInstructions(),
                                                    coupling_map='all-to-all',
                                                    num_samples_per_width_depth=num_subcircs_per_width_depth,
                                                    rand_state=rand_state
                                                    )


        # check that the width and depth metadata agree with the circuit width and depth for each circuit in the design.
        for subcirc, auxlist in design.aux_info.items():
            self.assertTrue(all(subcirc.width == aux['width'] for aux in auxlist))
            self.assertTrue(all(subcirc.depth == aux['depth'] for aux in auxlist))

        # check that the correct number of circuits of each width and depth exist by looking at aux_info
        reshaped_width_depths = []
        for width, depths in width_depths.items():
            for depth in depths:
                reshaped_width_depths.append((width, depth))

        created_width_depths = defaultdict(int)
        for subcirc, auxlist in design.aux_info.items():
            created_width_depths[(subcirc.width, subcirc.depth)] += len(auxlist)

        self.assertListEqual(sorted(reshaped_width_depths), sorted(list(created_width_depths.keys())))
        self.assertTrue(all(created_width_depths[k] == num_subcircs_per_width_depth for k in reshaped_width_depths))


    def test_simple_subcirc_selection_partial_connectivity(self):

        rand_state = _np.random.RandomState(0)

        class CustomInstructions(object):
            def get(self, *args):
                if args[0] == 'Gu3':
                    return 5.0
                elif args[0] == 'Gcnot':
                    return 12.0
                elif args[0] == 'Gcphase':
                    return 13.0
                else:
                    raise RuntimeError(f'No duration known for instruction {args[0]}')
            
        line_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

        I_args = [0,0,0]
        X_args = [_np.pi,0,_np.pi]
        Y_args = [_np.pi,_np.pi/2,_np.pi/2]
        Z_args = [0,0,_np.pi]

        layers = [
            [L('Gu3', ['Q1'], args=Z_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcnot', ['Q1', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q3'], args=None)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=Z_args), L('Gu3', ['Q3'], args=X_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=X_args)],
            [L('Gcnot', ['Q2', 'Q3'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q3', 'Q4'], args=None)],
            [L('Gcnot', ['Q3', 'Q2'], args=None), L('Gcnot', ['Q4', 'Q5'], args=None)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=I_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=X_args), L('Gu3', ['Q2'], args=Y_args), L('Gu3', ['Q3'], args=I_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=Y_args)],
            [L('Gu3', ['Q1'], args=Y_args), L('Gu3', ['Q2'], args=X_args), L('Gu3', ['Q3'], args=Z_args), L('Gu3', ['Q4'], args=Y_args), L('Gu3', ['Q5'], args=I_args)],
            [L('Gcphase', ['Q1', 'Q2'], args=None), L('Gcphase', ['Q4', 'Q5'], args=None)]
                  ]
                
        circ = C(layers, line_labels=line_labels)

        width_depths = {2: [2,4,6],
                        4: [3,6,9]}
        
        num_subcircs_per_width_depth = 31

        graph = _nx.barbell_graph(10, 3)

        int_coupling_list = list(graph.edges())
        coupling_list = [(f'Q{i}', f'Q{j}') for i,j in int_coupling_list]
        

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingDependencyWarning)
            design = _subcircsel.sample_subcircuits(circ, width_depths=width_depths,
                                                    instruction_durations=CustomInstructions(),
                                                    coupling_map=coupling_list,
                                                    num_samples_per_width_depth=num_subcircs_per_width_depth,
                                                    rand_state=rand_state
                                                    )


        # check that the width and depth metadata agree with the circuit width and depth for each circuit in the design.
        for subcirc, auxlist in design.aux_info.items():
            self.assertTrue(all(subcirc.width == aux['width'] for aux in auxlist))
            self.assertTrue(all(subcirc.depth == aux['depth'] for aux in auxlist))

        # check that the correct number of circuits of each width and depth exist by looking at aux_info
        reshaped_width_depths = []
        for width, depths in width_depths.items():
            for depth in depths:
                reshaped_width_depths.append((width, depth))

        created_width_depths = defaultdict(int)
        for subcirc, auxlist in design.aux_info.items():
            created_width_depths[(subcirc.width, subcirc.depth)] += len(auxlist)

        self.assertListEqual(sorted(reshaped_width_depths), sorted(list(created_width_depths.keys())))
        self.assertTrue(all(created_width_depths[k] == num_subcircs_per_width_depth for k in reshaped_width_depths))

        
    def test_simple_subcirc_selection_qiskit_coupling_map(self):
        try:
            import qiskit
        except:
            self.skipTest('Qiskit is required for this operation, and does not appear to be installed.')

        try:
            import qiskit_ibm_runtime
        except:
            self.skipTest('Qiskit Runtime is required for this operation, and does not appear to be installed.')


        backend = qiskit_ibm_runtime.fake_provider.FakeFez()

        qk_circ = qiskit.QuantumCircuit(4)
        qk_circ.append(qiskit.circuit.library.QFTGate(4), range(4))
        qk_circ = qiskit.transpile(qk_circ, backend=backend)

        ps_circ, _ = C.from_qiskit(qk_circ)
        ps_circ = ps_circ.delete_idling_lines()

        rand_state = _np.random.RandomState(0)

        width_depths = {2: [2,4,6],
                        3: [3,6,9]}
        
        num_subcircs_per_width_depth = 15
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingDependencyWarning)
            design = _subcircsel.sample_subcircuits(ps_circ, width_depths=width_depths,
                                                    instruction_durations=backend.instruction_durations,
                                                    coupling_map=backend.coupling_map,
                                                    num_samples_per_width_depth=num_subcircs_per_width_depth,
                                                    rand_state=rand_state
                                                    )


        # check that the width and depth metadata agree with the circuit width and depth for each circuit in the design.
        for subcirc, auxlist in design.aux_info.items():
            self.assertTrue(all(subcirc.width == aux['width'] for aux in auxlist))
            self.assertTrue(all(subcirc.depth == aux['depth'] for aux in auxlist))

        # check that the correct number of circuits of each width and depth exist by looking at aux_info
        reshaped_width_depths = []
        for width, depths in width_depths.items():
            for depth in depths:
                reshaped_width_depths.append((width, depth))

        created_width_depths = defaultdict(int)
        for subcirc, auxlist in design.aux_info.items():
            created_width_depths[(subcirc.width, subcirc.depth)] += len(auxlist)


        self.assertListEqual(sorted(reshaped_width_depths), sorted(list(created_width_depths.keys())))
        self.assertTrue(all(created_width_depths[k] == num_subcircs_per_width_depth for k in reshaped_width_depths))


    def test_simple_subcirc_selection_qiskit_instruction_durations(self):
        # Like test_simple_subcirc_selection_qiskit_coupling_map but constructs
        # `qiskit.transpiler.InstructionDurations` directly instead of pulling
        # one off an IBM-Runtime fake backend, so the test runs anywhere qiskit
        # is installed (qiskit_ibm_runtime not required).
        # Exercises the qiskit branch in subcircuit_selection (lines ~324-347):
        # `isinstance(instruction_durations, qiskit.transpiler.InstructionDurations)`
        # → True, then the gate-name conversion path and `.get(name, qubits)`
        # call on the real qiskit object.
        try:
            import qiskit
            from qiskit.transpiler import InstructionDurations
        except ImportError:
            self.skipTest('qiskit is required for this test and is not installed.')

        rand_state = _np.random.RandomState(0)
        line_labels = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4']

        # Build a 5-qubit circuit using pyGSTi gate names that have known
        # qiskit equivalents in standard_gatenames_qiskit_conversions:
        #   'Gxpi2' -> 'sx', 'Gcnot' -> 'cx'.
        layers = [
            [L('Gxpi2', ['Q0']), L('Gxpi2', ['Q1']), L('Gxpi2', ['Q2']), L('Gxpi2', ['Q3']), L('Gxpi2', ['Q4'])],
            [L('Gcnot', ['Q0', 'Q1']), L('Gcnot', ['Q2', 'Q3'])],
            [L('Gxpi2', ['Q0']), L('Gxpi2', ['Q1']), L('Gxpi2', ['Q2']), L('Gxpi2', ['Q3']), L('Gxpi2', ['Q4'])],
            [L('Gcnot', ['Q1', 'Q2']), L('Gcnot', ['Q3', 'Q4'])],
            [L('Gxpi2', ['Q0']), L('Gxpi2', ['Q1']), L('Gxpi2', ['Q2']), L('Gxpi2', ['Q3']), L('Gxpi2', ['Q4'])],
            [L('Gcnot', ['Q0', 'Q1']), L('Gcnot', ['Q3', 'Q4'])],
            [L('Gxpi2', ['Q0']), L('Gxpi2', ['Q1']), L('Gxpi2', ['Q2']), L('Gxpi2', ['Q3']), L('Gxpi2', ['Q4'])],
        ]
        circ = C(layers, line_labels=line_labels)

        # InstructionDurations is keyed by *qiskit* gate names and integer
        # qubit indices. The qiskit branch in sample_subcircuits applies the
        # standard_gatenames_qiskit_conversions mapping and strips the 'Q'
        # prefix from qubit labels before calling .get().
        durs = InstructionDurations([
            ('sx', [0], 35),
            ('sx', [1], 35),
            ('sx', [2], 35),
            ('sx', [3], 35),
            ('sx', [4], 35),
            ('cx', [0, 1], 300),
            ('cx', [1, 2], 300),
            ('cx', [2, 3], 300),
            ('cx', [3, 4], 300),
        ])

        width_depths = {2: [2, 4], 3: [3]}
        num_subcircs_per_width_depth = 5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingDependencyWarning)
            design = _subcircsel.sample_subcircuits(
                circ, width_depths=width_depths,
                instruction_durations=durs,
                coupling_map='linear',
                num_samples_per_width_depth=num_subcircs_per_width_depth,
                rand_state=rand_state)

        for subcirc, auxlist in design.aux_info.items():
            self.assertTrue(all(subcirc.width == aux['width'] for aux in auxlist))
            self.assertTrue(all(subcirc.depth == aux['depth'] for aux in auxlist))

        reshaped_width_depths = [(w, d) for w, ds in width_depths.items() for d in ds]
        created_width_depths = defaultdict(int)
        for subcirc, auxlist in design.aux_info.items():
            created_width_depths[(subcirc.width, subcirc.depth)] += len(auxlist)
        self.assertListEqual(sorted(reshaped_width_depths), sorted(created_width_depths.keys()))
        self.assertTrue(all(created_width_depths[k] == num_subcircs_per_width_depth
                            for k in reshaped_width_depths))


    def test_simple_subcirc_selection_qiskit_coupling_map_direct(self):
        # Companion to test_simple_subcirc_selection_qiskit_instruction_durations:
        # constructs a `qiskit.transpiler.CouplingMap` directly so the qiskit
        # CouplingMap branch in simple_weighted_subcirc_selection (lines
        # ~363-380: the `isinstance(coupling_map, qiskit.transpiler.CouplingMap)`
        # check, the edge iteration, and the 'Q'-prefix join) is exercised
        # without needing qiskit_ibm_runtime's fake backend.
        try:
            import qiskit
            from qiskit.transpiler import CouplingMap
        except ImportError:
            self.skipTest('qiskit is required for this test and is not installed.')

        rand_state = _np.random.RandomState(0)

        class NoDelayInstructions(object):
            def get(self, *args):
                return 0.0

        # Same circuit shape as the InstructionDurations test — line labels
        # 'Q0'..'Q4' so the 'Q'-prefix join in subcircuit_selection.py:374
        # produces labels that match `full_circ.line_labels`.
        line_labels = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4']
        layers = [
            [L('Gxpi2', ['Q0']), L('Gxpi2', ['Q1']), L('Gxpi2', ['Q2']), L('Gxpi2', ['Q3']), L('Gxpi2', ['Q4'])],
            [L('Gcnot', ['Q0', 'Q1']), L('Gcnot', ['Q2', 'Q3'])],
            [L('Gxpi2', ['Q0']), L('Gxpi2', ['Q1']), L('Gxpi2', ['Q2']), L('Gxpi2', ['Q3']), L('Gxpi2', ['Q4'])],
            [L('Gcnot', ['Q1', 'Q2']), L('Gcnot', ['Q3', 'Q4'])],
            [L('Gxpi2', ['Q0']), L('Gxpi2', ['Q1']), L('Gxpi2', ['Q2']), L('Gxpi2', ['Q3']), L('Gxpi2', ['Q4'])],
            [L('Gcnot', ['Q0', 'Q1']), L('Gcnot', ['Q3', 'Q4'])],
            [L('Gxpi2', ['Q0']), L('Gxpi2', ['Q1']), L('Gxpi2', ['Q2']), L('Gxpi2', ['Q3']), L('Gxpi2', ['Q4'])],
        ]
        circ = C(layers, line_labels=line_labels)

        # Linear connectivity expressed as a qiskit CouplingMap rather than
        # the string 'linear' or a plain list — exercises the qiskit branch.
        cmap = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])

        width_depths = {2: [2, 4], 3: [3]}
        num_subcircs_per_width_depth = 5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MissingDependencyWarning)
            design = _subcircsel.sample_subcircuits(
                circ, width_depths=width_depths,
                instruction_durations=NoDelayInstructions(),
                coupling_map=cmap,
                num_samples_per_width_depth=num_subcircs_per_width_depth,
                rand_state=rand_state)

        for subcirc, auxlist in design.aux_info.items():
            self.assertTrue(all(subcirc.width == aux['width'] for aux in auxlist))
            self.assertTrue(all(subcirc.depth == aux['depth'] for aux in auxlist))

        reshaped_width_depths = [(w, d) for w, ds in width_depths.items() for d in ds]
        created_width_depths = defaultdict(int)
        for subcirc, auxlist in design.aux_info.items():
            created_width_depths[(subcirc.width, subcirc.depth)] += len(auxlist)
        self.assertListEqual(sorted(reshaped_width_depths), sorted(created_width_depths.keys()))
        self.assertTrue(all(created_width_depths[k] == num_subcircs_per_width_depth
                            for k in reshaped_width_depths))

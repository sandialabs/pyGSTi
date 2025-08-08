from ..util import BaseCase

from pygsti.circuits import subcircuit_selection as _subcircsel, Circuit as C
from pygsti.baseobjs import Label as L

import numpy as _np
import networkx as _nx

from collections import defaultdict

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


    
    def test_simple_subcirc_selection_full_connectivity(self):

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


    def test_simple_subcirc_selection_full_connectivity(self):

        rand_state = _np.random.RandomState(0)

        class CustomInstructions(object):
            def get(self, *args):
                if args[0] == 'u':
                    return 5.0
                elif args[0] == 'cx':
                    return 12.0
                elif args[0] == 'cz':
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

        qk_circ = qiskit.circuit.library.QFT(4)
        qk_circ = qiskit.transpile(qk_circ, backend=backend)

        ps_circ, _ = C.from_qiskit(qk_circ)
        ps_circ = ps_circ.delete_idling_lines()

        rand_state = _np.random.RandomState(0)

        width_depths = {2: [2,4,6],
                        3: [3,6,9]}
        
        num_subcircs_per_width_depth = 15
        
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
        

        

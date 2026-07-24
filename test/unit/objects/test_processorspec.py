import itertools

import numpy as np

from pygsti.baseobjs.label import Label
from pygsti.processors import QuditProcessorSpec
from pygsti.processors import QubitProcessorSpec
from pygsti.tools import symplectic
from ..util import BaseCase, with_temp_path


def save_and_load(obj, pth):
    obj.write(pth + ".json")
    return obj.__class__.read(pth + '.json')


class ProcessorSpecTester(BaseCase):
    def test_argumented_symplectic_rep_factory_maps_label_args(self):
        def unitary_factory(args):
            return np.identity(4, 'd')

        unitary_factory.shape = (4, 4)

        def srep_factory(label):
            if not label.args:
                raise ValueError("Gargp labels must specify the active qubit as their first argument.")
            return symplectic.symplectic_rep_of_clifford_layer(Label('P', label.args[0]), q_labels=label.sslbls)

        pspec = QubitProcessorSpec(
            2, ['Gargp'], nonstd_gate_unitaries={'Gargp': unitary_factory},
            availability={'Gargp': [(0, 1)]}, geometry='line',
            nonstd_gate_symplecticreps={'Gargp': srep_factory},
            gate_arg_label_indices={'Gargp': (0,)})

        label = Label('Gargp', (0, 1), args=(1,))
        expected = symplectic.symplectic_rep_of_clifford_layer(Label('P', 1), q_labels=(0, 1))
        actual = pspec.clifford_symplectic_rep_of(label)
        self.assertArraysAlmostEqual(actual[0], expected[0])
        self.assertArraysAlmostEqual(actual[1], expected[1])

        mapped_label = pspec.map_gate_label_state_space(label, {0: 'Q0', 1: 'Q1'})
        self.assertEqual(mapped_label, Label('Gargp', ('Q0', 'Q1'), args=('Q1',)))

        mapped_pspec = pspec.map_qubit_labels({0: 'Q0', 1: 'Q1'})
        mapped_expected = symplectic.symplectic_rep_of_clifford_layer(
            Label('P', 'Q1'), q_labels=('Q0', 'Q1'))
        mapped_actual = mapped_pspec.clifford_symplectic_rep_of(mapped_label)
        self.assertArraysAlmostEqual(mapped_actual[0], mapped_expected[0])
        self.assertArraysAlmostEqual(mapped_actual[1], mapped_expected[1])

    @with_temp_path
    def test_label_specific_symplectic_rep_serialization(self, pth):
        label = Label('Gargp', (0, 1), args=(1,))
        srep = symplectic.symplectic_rep_of_clifford_layer(Label('P', 1), q_labels=(0, 1))
        pspec = QubitProcessorSpec(
            2, ['Gargp'], nonstd_gate_unitaries={'Gargp': np.identity(4, 'd')},
            availability={'Gargp': [(0, 1)]}, geometry='line',
            nonstd_gate_symplecticreps={label: srep})

        loaded_pspec = save_and_load(pspec, pth)
        loaded_srep = loaded_pspec.clifford_symplectic_rep_of(label)
        self.assertArraysAlmostEqual(loaded_srep[0], srep[0])
        self.assertArraysAlmostEqual(loaded_srep[1], srep[1])

    @with_temp_path
    def test_arity_only_nonstd_gate(self, pth):
        srep = (np.identity(6, dtype=int), np.zeros(6, dtype=int))
        pspec = QubitProcessorSpec(
            3, ['Gglobal'], availability={'Gglobal': [(0, 1, 2)]}, geometry='line',
            nonstd_gate_num_qubits={'Gglobal': 3},
            nonstd_gate_symplecticreps={'Gglobal': srep})

        self.assertEqual(pspec.gate_num_qubits('Gglobal'), 3)
        self.assertNotIn('Gglobal', pspec.gate_unitaries)
        self.assertEqual(pspec.compute_ops_on_qubits()[(0, 1, 2)], [Label('Gglobal', (0, 1, 2))])

        actual_srep = pspec.clifford_symplectic_rep_of(Label('Gglobal', (0, 1, 2)))
        self.assertArraysAlmostEqual(actual_srep[0], srep[0])
        self.assertArraysAlmostEqual(actual_srep[1], srep[1])

        subset_pspec = pspec.subset(['Gglobal'], [0, 1, 2])
        self.assertEqual(subset_pspec.gate_num_qubits('Gglobal'), 3)
        self.assertNotIn('Gglobal', subset_pspec.gate_unitaries)

        renamed_pspec = pspec.subset(['Gglobal'], [0, 1, 2])
        renamed_pspec.rename_gate_inplace('Gglobal', 'Grenamed')
        self.assertEqual(renamed_pspec.gate_num_qubits('Grenamed'), 3)
        self.assertNotIn('Gglobal', renamed_pspec.nonstd_gate_num_qudits)

        mapped_pspec = pspec.map_qubit_labels({0: 'Q0', 1: 'Q1', 2: 'Q2'})
        self.assertEqual(mapped_pspec.gate_num_qubits('Gglobal'), 3)
        self.assertEqual(mapped_pspec.compute_ops_on_qubits()[('Q0', 'Q1', 'Q2')],
                         [Label('Gglobal', ('Q0', 'Q1', 'Q2'))])

        loaded_pspec = save_and_load(pspec, pth)
        self.assertEqual(loaded_pspec.gate_num_qubits('Gglobal'), 3)
        self.assertNotIn('Gglobal', loaded_pspec.gate_unitaries)
        loaded_srep = loaded_pspec.clifford_symplectic_rep_of(Label('Gglobal', (0, 1, 2)))
        self.assertArraysAlmostEqual(loaded_srep[0], srep[0])
        self.assertArraysAlmostEqual(loaded_srep[1], srep[1])

    def test_arity_only_gate_requires_explicit_symplectic_rep(self):
        pspec = QubitProcessorSpec(
            3, ['Gglobal'], availability={'Gglobal': [(0, 1, 2)]}, geometry='line',
            nonstd_gate_num_qubits={'Gglobal': 3})

        with self.assertRaisesRegex(ValueError, "No unitary is available for arity-only gate"):
            pspec.clifford_symplectic_rep_of(Label('Gglobal', (0, 1, 2)))

        self.assertEqual(pspec.compute_one_qubit_gate_relations(), ({}, {}))
        self.assertEqual(pspec.compute_multiqubit_inversion_relations(), {})

    @with_temp_path
    def test_qudit_arity_only_nonstd_gate(self, pth):
        pspec = QuditProcessorSpec(
            ('Q0', 'Q1'), (3, 2), ['Gnonunitary'],
            availability={'Gnonunitary': [('Q0', 'Q1')]},
            nonstd_gate_num_qudits={'Gnonunitary': 2})

        self.assertEqual(pspec.gate_num_qudits('Gnonunitary'), 2)
        self.assertNotIn('Gnonunitary', pspec.gate_unitaries)
        self.assertEqual(pspec.available_gatelabels('Gnonunitary', ('Q0', 'Q1')),
                         (Label('Gnonunitary', ('Q0', 'Q1')),))

        subset_pspec = pspec.subset(['Gnonunitary'], ['Q0', 'Q1'])
        self.assertEqual(subset_pspec.gate_num_qudits('Gnonunitary'), 2)
        self.assertNotIn('Gnonunitary', subset_pspec.gate_unitaries)

        mapped_pspec = pspec.map_qudit_labels({'Q0': 'A', 'Q1': 'B'})
        self.assertEqual(mapped_pspec.gate_num_qudits('Gnonunitary'), 2)
        self.assertEqual(mapped_pspec.available_gatelabels('Gnonunitary', ('A', 'B')),
                         (Label('Gnonunitary', ('A', 'B')),))

        loaded_pspec = save_and_load(pspec, pth)
        self.assertEqual(loaded_pspec.gate_num_qudits('Gnonunitary'), 2)
        self.assertEqual(loaded_pspec.available_gatelabels('Gnonunitary', ('Q0', 'Q1')),
                         (Label('Gnonunitary', ('Q0', 'Q1')),))

    def test_arity_only_metadata_validation(self):
        with self.assertRaisesRegex(ValueError, "both `nonstd_gate_unitaries` and `nonstd_gate_num_qudits`"):
            QubitProcessorSpec(
                1, ['Gbad'], nonstd_gate_unitaries={'Gbad': np.identity(2, 'd')},
                nonstd_gate_num_qubits={'Gbad': 1})

        with self.assertRaisesRegex(ValueError, "Gate arity for Gbad must be positive"):
            QubitProcessorSpec(1, ['Gbad'], nonstd_gate_num_qubits={'Gbad': 0})

    @with_temp_path
    def test_instrument_with_sslbls_serialization(self, pth):
        # Regression test: an instrument whose name carries explicit state-space labels
        # (as opposed to a bare name like 'Iz' that implicitly acts on all qudits) serializes
        # its Label to a JSON list. On load, QubitProcessorSpec._from_nice_serialization must
        # turn each entry of `instrument_names` back into a hashable object; if it doesn't, the
        # unhashable list survives into `self.instrument_names` and any lookup against
        # `self.nonstd_instruments` (e.g. via `instrument_specifier`, as called from
        # `_create_explicit_model`) raises `TypeError: unhashable type: 'list'`.
        iname = Label('Iz', (0,))
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2'], geometry='line',
                                   instrument_names=(iname,),
                                   nonstd_instruments={iname: 'Iz'})

        loaded_pspec = save_and_load(pspec, pth)

        # Compound names come back as plain tuples (hash-equal to the original Labels).
        self.assertEqual(loaded_pspec.instrument_names, (('Iz', 0),))
        for loaded_iname in loaded_pspec.instrument_names:
            self.assertEqual(loaded_pspec.instrument_specifier(loaded_iname), 'Iz')

        # Exercise the reported call chain (_create_explicit_model) on a single-qubit spec,
        # where the instrument's sslbls cover the full state space.  On the two-qubit spec
        # above this raises NotImplementedError even without serialization, because
        # instruments cannot be embedded onto a subset of the qudits yet.
        pspec_1q = QubitProcessorSpec(1, ['Gxpi2', 'Gypi2'],
                                      instrument_names=(iname,),
                                      nonstd_instruments={iname: 'Iz'})
        loaded_pspec_1q = save_and_load(pspec_1q, pth)

        from pygsti.models.modelconstruction import _create_explicit_model
        mdl = _create_explicit_model(loaded_pspec_1q, None, evotype='default', simulator='auto',
                                     ideal_gate_type='static', ideal_prep_type='auto', ideal_povm_type='auto',
                                     embed_gates=False, basis='pp')
        self.assertEqual(list(mdl.instruments.keys()), [iname])

    @with_temp_path
    def test_instrument_with_custom_spec_serialization(self, pth):
        # Regression test: `nonstd_instruments` keys used to be flattened with a lossy colon-join,
        # which exploded plain-string names character-by-character ('Iparity' -> 'I:p:a:r:i:t:y'),
        # so a custom instrument spec could never be found by `instrument_specifier` after a
        # round trip through JSON.
        spec = {'plus': [('00', '00'), ('11', '11')],
                'minus': [('10', '10'), ('01', '01')]}
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2'], geometry='line',
                                   instrument_names=('Iparity',),
                                   nonstd_instruments={'Iparity': spec})

        loaded_pspec = save_and_load(pspec, pth)

        self.assertEqual(loaded_pspec.instrument_names, ('Iparity',))
        self.assertEqual(loaded_pspec.instrument_specifier('Iparity'), spec)

        from pygsti.models.modelconstruction import create_explicit_model
        mdl = create_explicit_model(loaded_pspec)
        self.assertEqual(list(mdl.instruments['Iparity'].keys()), ['plus', 'minus'])

    @with_temp_path
    def test_qudit_instrument_serialization(self, pth):
        # Regression test: QuditProcessorSpec._from_nice_serialization applied tuple() to every
        # loaded instrument name, exploding plain-string names into character tuples
        # ('Iz' -> ('I', 'z')).
        iname = Label('Iz', ('Q0',))
        pspec = QuditProcessorSpec(('Q0', 'Q1'), (2, 2), ['Gxpi2', 'Gypi2'], geometry='line',
                                   instrument_names=('Iz', iname),
                                   nonstd_instruments={iname: 'Iz'})

        loaded_pspec = save_and_load(pspec, pth)

        self.assertEqual(loaded_pspec.instrument_names, ('Iz', ('Iz', 'Q0')))
        self.assertEqual(loaded_pspec.instrument_specifier('Iz'), 'Iz')
        self.assertEqual(loaded_pspec.instrument_specifier(('Iz', 'Q0')), 'Iz')

    @with_temp_path
    def test_legacy_instrument_serialization_format(self, pth):
        # Files written before the `nonstd_instruments` format change stored a dict whose keys
        # were flattened with ':'.join(map(str, key)).  Loading must repair those keys by
        # matching them against `instrument_names`, falling back to a split on ':' for keys it
        # cannot match.
        import json

        iname = Label('Iz', (0,))
        parity_spec = {'plus': [('00', '00'), ('11', '11')],
                       'minus': [('10', '10'), ('01', '01')]}
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2'], geometry='line',
                                   instrument_names=('Iparity', iname),
                                   nonstd_instruments={'Iparity': parity_spec, iname: 'Iz'})

        pspec.write(pth + '.json')
        with open(pth + '.json') as f:
            state = json.load(f)
        state['nonstd_instruments'] = {':'.join(map(str, k)): v for k, v in state['nonstd_instruments']}
        state['nonstd_instruments']['Ighost:2'] = 'Iz'  # matches no instrument name
        with open(pth + '.json', 'w') as f:
            json.dump(state, f)

        loaded_pspec = QubitProcessorSpec.read(pth + '.json')

        self.assertEqual(loaded_pspec.instrument_names, ('Iparity', ('Iz', 0)))
        self.assertEqual(loaded_pspec.instrument_specifier('Iparity'), parity_spec)
        self.assertEqual(loaded_pspec.instrument_specifier(('Iz', 0)), 'Iz')
        # Unmatched legacy keys keep the old best-effort split-on-colon reconstruction.
        self.assertEqual(loaded_pspec.nonstd_instruments[('Ighost', '2')], 'Iz')

    @with_temp_path
    def test_with_spam(self, pth):
        pspec_defaults = QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line')

        pspec_names = QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line',
                                         prep_names=("rho1", "rho_1100"), povm_names=("Mz",))

        prep_vec = np.zeros(2**4, complex)
        prep_vec[4] = 1.0
        EA = np.zeros(2**4, complex)
        EA[14] = 1.0
        EB = np.zeros(2**4, complex)
        EB[15] = 1.0

        pspec_vecs = QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line',
                                        prep_names=("rhoA", "rhoC"), povm_names=("Ma", "Mc"),
                                        nonstd_preps={'rhoA': "rho0", 'rhoC': prep_vec},
                                        nonstd_povms={'Ma': {'0': "0000", '1': EA},
                                                      'Mc': {'OutA': "0000", 'OutB': [EA, EB]}})

        pspec_defaults = save_and_load(pspec_defaults, pth)
        pspec_names = save_and_load(pspec_names, pth)
        pspec_vecs = save_and_load(pspec_vecs, pth)

    def test_resolved_availability_contradiction(self):
        nQubits = 1
        qubit_labels = [0]
        
        gate_names = ['Ga', 'Gb']
        
        # Define two distinct dummy unitaries for the gates
        Ua = np.array([[1, 0], [0, 1]], 'd')
        Ub = np.array([[0, 1], [1, 0]], 'd')
        
        nonstd_gate_unitaries = {'Ga': Ua, 'Gb': Ub}
        
        # Both gates are available on the same qubit
        availability = {'Ga': [(0,)], 'Gb': [(0,)]}
        
        ps = QubitProcessorSpec(nQubits, gate_names, nonstd_gate_unitaries=nonstd_gate_unitaries, 
                                availability=availability, qubit_labels=qubit_labels)
        
        ga_available = ps.is_available(('Ga', 0))
        gb_available = ps.is_available(('Gb', 0))
        
        self.assertTrue(ga_available and gb_available)

    def test_compute_2Q_connectivity(self):
        qubit_labels = ['q0', 'q1', 'q2']
        gate_names = ['Gcnot']
        availability = {'Gcnot': [('q0', 'q1'), ('q1', 'q2')]}
        
        ps = QubitProcessorSpec(3, gate_names, availability=availability, qubit_labels=qubit_labels)
        
        computed_graph = ps.compute_2Q_connectivity()

        # Check that the graph is undirected and has the correct edges
        computed_undirected_edges = {frozenset(edge) for edge in computed_graph.edges()}
        expected_undirected_edges = {frozenset({'q0', 'q1'}), frozenset({'q1', 'q2'})}
        self.assertEqual(computed_undirected_edges, expected_undirected_edges)

        # Check that the nodes are correct
        self.assertEqual(set(computed_graph.node_names), set(qubit_labels))

    def test_gate_num_qubits(self):
        ps = QubitProcessorSpec(2, gate_names=['Gx', 'Gcnot'], geometry='line')
        self.assertEqual(ps.gate_num_qubits('Gx'), 1)
        self.assertEqual(ps.gate_num_qubits('Gcnot'), 2)

    def test_rename_gate_inplace(self):
        ps = QubitProcessorSpec(1, gate_names=['Gx', 'Gy'], availability={'Gx': [(0,)], 'Gy': [(0,)]})
        ps.rename_gate_inplace('Gx', 'MyGx')
        self.assertNotIn('Gx', ps.gate_names)
        self.assertIn('MyGx', ps.gate_names)
        self.assertNotIn('Gx', ps.gate_unitaries)
        self.assertIn('MyGx', ps.gate_unitaries)
        self.assertNotIn('Gx', ps.availability)
        self.assertIn('MyGx', ps.availability)

    def test_resolved_availability_modes(self):
        ps = QubitProcessorSpec(3, gate_names=['Gcnot'], availability={'Gcnot': [(0, 1)]}, geometry='line')
        self.assertEqual(ps.resolved_availability('Gcnot', 'tuple'), [(0, 1)])

        avail_fn = ps.resolved_availability('Gcnot', 'function')
        self.assertTrue(avail_fn((0, 1)))
        self.assertFalse(avail_fn((1, 0)))
        self.assertFalse(avail_fn((0, 2)))

    def test_availability_specifiers(self):
        qubit_labels = [0, 1, 2]
        # Test "all-permutations"
        ps_perm = QubitProcessorSpec(3, gate_names=['Gcnot'], availability={'Gcnot': 'all-permutations'}, qubit_labels=qubit_labels)
        self.assertEqual(set(ps_perm.resolved_availability('Gcnot', 'tuple')), set(itertools.permutations(qubit_labels, 2)))

        # Test "all-combinations"
        ps_comb = QubitProcessorSpec(3, gate_names=['Gcnot'], availability={'Gcnot': 'all-combinations'}, qubit_labels=qubit_labels)
        self.assertEqual(set(ps_comb.resolved_availability('Gcnot', 'tuple')), set(itertools.combinations(qubit_labels, 2)))

        # Test "all-edges"
        ps_edges = QubitProcessorSpec(3, gate_names=['Gcnot'], geometry='line', qubit_labels=qubit_labels)
        self.assertEqual(set(ps_edges.resolved_availability('Gcnot', 'tuple')), { (0, 1), (1, 0), (1, 2), (2, 1)})

    def test_available_gatenames(self):
        qubit_labels = [0, 1, 2]
        gate_names = ['Gx', 'Gy', 'Gcnot']
        availability = {'Gx': [(0,)], 'Gy': [(1,)], 'Gcnot': [(0, 1)]}
        ps = QubitProcessorSpec(3, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        self.assertEqual(set(ps.available_gatenames((0,))), {'Gx'})
        self.assertEqual(set(ps.available_gatenames((1,))), {'Gy'})
        self.assertEqual(set(ps.available_gatenames((0, 1))), {'Gx', 'Gy', 'Gcnot'})
        self.assertEqual(set(ps.available_gatenames((2,))), set())

    def test_available_gatelabels(self):
        qubit_labels = [0, 1, 2]
        gate_names = ['Gx', 'Gcnot']
        availability = {'Gx': [(0,), (1,)], 'Gcnot': 'all-permutations'}
        ps = QubitProcessorSpec(3, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        self.assertEqual(set(ps.available_gatelabels('Gx', (0, 1))), {Label('Gx', (0,)), Label('Gx', (1,))})
        self.assertEqual(set(ps.available_gatelabels('Gx', (0, 2))), {Label('Gx', (0,))})
        self.assertEqual(set(ps.available_gatelabels('Gcnot', (0, 1, 2))), set(map(lambda t: Label('Gcnot', t), itertools.permutations([0, 1, 2], 2))))

    def test_compute_ops_on_qudits(self):
        qubit_labels = [0, 1]
        gate_names = ['Gx', 'Gcnot']
        availability = {'Gx': [(0,)], 'Gcnot': [(0, 1)]}
        ps = QubitProcessorSpec(2, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        ops_on_qudits = ps.compute_ops_on_qudits()
        self.assertEqual(ops_on_qudits, {(0,): [Label('Gx', (0,))], (0, 1): [Label('Gcnot', (0, 1))]})

    def test_subset(self):
        qubit_labels = [0, 1, 2]
        gate_names = ['Gx', 'Gy', 'Gcnot']
        availability = {'Gx': [(0,), (1,)], 'Gy': [(1,), (2,)], 'Gcnot': [(0, 1), (1, 2)]}
        ps = QubitProcessorSpec(3, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        subset_ps = ps.subset(gate_names_to_include=['Gx', 'Gcnot'], qubit_labels_to_keep=[0, 1])

        self.assertEqual(subset_ps.gate_names, ('Gx', 'Gcnot'))
        self.assertEqual(subset_ps.qubit_labels, (0, 1))
        self.assertEqual(subset_ps.availability, {'Gx': ((0,), (1,)), 'Gcnot': ((0, 1),)})

    def test_map_qudit_labels(self):
        qubit_labels = [0, 1]
        gate_names = ['Gx', 'Gcnot']
        availability = {'Gx': [(0,)], 'Gcnot': [(0, 1)]}
        ps = QubitProcessorSpec(2, gate_names=gate_names, availability=availability, qubit_labels=qubit_labels)

        mapped_ps = ps.map_qudit_labels({0: 'a', 1: 'b'})

        self.assertEqual(mapped_ps.qubit_labels, ('a', 'b'))
        self.assertEqual(mapped_ps.availability, {'Gx': (('a',),), 'Gcnot': (('a', 'b'),)})

    def test_compute_clifford_symplectic_reps(self):
        # Create a non-Clifford gate unitary
        non_clifford_U = np.array([[1, 0], [0, np.exp(1j * np.pi / 8)]], 'D')

        ps = QubitProcessorSpec(1, gate_names=['Gh', 'Gp', 'Gnc'], 
                                nonstd_gate_unitaries={'Gnc': non_clifford_U})
        
        srep_dict = ps.compute_clifford_symplectic_reps()
        
        internal_srep_dict = symplectic.compute_internal_gate_symplectic_representations()
        
        self.assertIn('Gh', srep_dict)
        self.assertIn('Gp', srep_dict)
        self.assertNotIn('Gnc', srep_dict)
        
        expected_Gh_s, expected_Gh_p = internal_srep_dict['H']
        expected_Gp_s, expected_Gp_p = internal_srep_dict['P']
        
        actual_Gh_s, actual_Gh_p = srep_dict['Gh']
        actual_Gp_s, actual_Gp_p = srep_dict['Gp']
        
        self.assertArraysEqual(actual_Gh_s, expected_Gh_s)
        self.assertArraysEqual(actual_Gh_p, expected_Gh_p)
        self.assertArraysEqual(actual_Gp_s, expected_Gp_s)
        self.assertArraysEqual(actual_Gp_p, expected_Gp_p)

from ..util import BaseCase
from pygsti.circuits import Circuit
from pygsti.algorithms.randomcircuit import create_random_circuit, find_all_sets_of_compatible_two_q_gates
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.processors import QubitProcessorSpec
from pygsti.models.modelconstruction import create_crosstalk_free_model, create_cloud_crosstalk_model
from pygsti.baseobjs import Label, BuiltinBasis, QubitSpace, CompleteElementaryErrorgenBasis, QubitGraph
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel, LocalElementaryErrorgenLabel
from pygsti.tools import errgenproptools as _eprop
from pygsti.tools import errgenpolytools as _epoly
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.tools.matrixtools import print_mx
from itertools import product
from math import floor
from pygsti.modelpacks import smq2Q_XYCPHASE
import numpy as np
import stim
import unittest
import warnings
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
import copy
import pickle


class ErrorgenPropTester(BaseCase):

    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase':[(0,1), (1,2), (2,3), (3,0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.target_model = create_crosstalk_free_model(processor_spec = pspec)
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
        self.circuit_length_1 = create_random_circuit(pspec, 1, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
        typ = 'H'
        max_stochastic = {'S': 0.0005, 'H': 0, 'H+S': 0.0001}
        max_hamiltonian = {'S': 0, 'H': 0.00005, 'H+S': 0.0001}
        max_strengths = {1: {'S': max_stochastic[typ], 'H': max_hamiltonian[typ]},
                        2: {'S': 3*max_stochastic[typ], 'H': 3*max_hamiltonian[typ]}
                        }
        error_rates_dict = sample_error_rates_dict(pspec, max_strengths, seed=12345)
        self.error_model = create_crosstalk_free_model(pspec, lindblad_error_coeffs=error_rates_dict)

    def test_exact_propagation_probabilities(self):
        #This should simultaneously confirm that the propagation code runs
        #and also that it is giving the correct values by directly comparing
        #to the probabilities from direct forward simulation.
        error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        probabilities_exact_propagation = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit)
        probabilities_forward_simulation = probabilities_fwdsim(self.error_model, self.circuit)

        self.assertTrue(np.linalg.norm(probabilities_exact_propagation - probabilities_forward_simulation, ord=1) < 1e-10)

    def test_approx_propagation_probabilities_BCH(self):
        error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        # exercises the deprecated 'pairwise' mode.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pyGSTiDeprecationWarning)
            probabilities_BCH_order_1 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=1, bch_mode='pairwise')
            probabilities_BCH_order_2 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=2, bch_mode='pairwise')
            probabilities_BCH_order_3 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=3, bch_mode='pairwise')
            probabilities_BCH_order_4 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=4, bch_mode='pairwise')
            probabilities_BCH_order_5 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=5, bch_mode='pairwise')
        probabilities_forward_simulation = probabilities_fwdsim(self.error_model, self.circuit)

        #use a much looser constraint on the agreement between the BCH results and forward simulation. Mostly testing to catch things exploding.
        TVD_order_1 = np.linalg.norm(probabilities_BCH_order_1 - probabilities_forward_simulation, ord=1)
        TVD_order_2 = np.linalg.norm(probabilities_BCH_order_2 - probabilities_forward_simulation, ord=1)
        TVD_order_3 = np.linalg.norm(probabilities_BCH_order_3 - probabilities_forward_simulation, ord=1)
        TVD_order_4 = np.linalg.norm(probabilities_BCH_order_4 - probabilities_forward_simulation, ord=1)
        TVD_order_5 = np.linalg.norm(probabilities_BCH_order_5 - probabilities_forward_simulation, ord=1)
        
        #loose bound is just to make sure nothing exploded.
        self.assertTrue(TVD_order_1 < 1e-2)
        self.assertTrue(TVD_order_2 < 1e-2)
        self.assertTrue(TVD_order_3 < 1e-2)
        self.assertTrue(TVD_order_4 < 1e-2)
        self.assertTrue(TVD_order_5 < 1e-2)

        #also assert that the TVDs get smaller in general as you go up in order.
        self.assertTrue((TVD_order_1>TVD_order_2) and (TVD_order_2>TVD_order_3) and (TVD_order_3>TVD_order_4) and (TVD_order_4>TVD_order_5))
        
    def test_approx_propagation_probabilities_magnus(self):
        error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        probabilities_BCH_order_1 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=1, bch_mode='magnus')
        probabilities_BCH_order_2 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=2, bch_mode='magnus')
        probabilities_BCH_order_3 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=3, bch_mode='magnus')
        probabilities_forward_simulation = probabilities_fwdsim(self.error_model, self.circuit)

        #use a much looser constraint on the agreement between the BCH results and forward simulation. Mostly testing to catch things exploding.
        TVD_order_1 = np.linalg.norm(probabilities_BCH_order_1 - probabilities_forward_simulation, ord=1)
        TVD_order_2 = np.linalg.norm(probabilities_BCH_order_2 - probabilities_forward_simulation, ord=1)
        TVD_order_3 = np.linalg.norm(probabilities_BCH_order_3 - probabilities_forward_simulation, ord=1)
        
        #loose bound is just to make sure nothing exploded.
        self.assertTrue(TVD_order_1 < 1e-2)
        self.assertTrue(TVD_order_2 < 1e-2)
        self.assertTrue(TVD_order_3 < 1e-2)

        #also assert that the TVDs get smaller in general as you go up in order.
        self.assertTrue((TVD_order_1>TVD_order_2) and (TVD_order_2>TVD_order_3))
        
    def test_eoc_error_channel(self):
        error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        eoc_error_channel = error_propagator.eoc_error_channel(self.circuit)

        #manually compute end-of-circuit error generator
        ideal_channel = self.target_model.sim.product(self.circuit)
        noisy_channel_exact = self.error_model.sim.product(self.circuit)
        eoc_error_channel_exact = noisy_channel_exact@ideal_channel.conj().T  

        assert np.linalg.norm(eoc_error_channel - eoc_error_channel_exact) < 1e-10
    
    def test_propagation_length_zero_one(self):
        error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        empty_circuit = Circuit([], line_labels=(0,1,2,3))
        error_propagator.propagate_errorgens(self.circuit_length_1)
        error_propagator.propagate_errorgens(empty_circuit, include_spam=True)
        error_propagator.propagate_errorgens(empty_circuit, include_spam=False)

    def test_errorgen_transform_map(self):
        error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        errorgen_input_output_map = error_propagator.errorgen_transform_map(self.circuit, include_spam=True)

        assert errorgen_input_output_map[(_LSE('H', (stim.PauliString("+___X"),)), 1)] == (_LSE('H', (stim.PauliString("+__ZY"),)), 1.0)
        assert errorgen_input_output_map[(_LSE('S', (stim.PauliString("+X___"),)), 2)] == (_LSE('S', (stim.PauliString("+Z___"),)),  1.0)
        assert errorgen_input_output_map[(_LSE('H', (stim.PauliString("+X___"),)), 3)] == (_LSE('H', (stim.PauliString("+Z___"),)), -1.0)

    def test_explicit_model(self):
        
        target_model = smq2Q_XYCPHASE.target_model('full TP')
        noisy_model = target_model.copy()
        noisy_model = noisy_model.rotate(max_rotate = 0.01)
        noisy_model.set_all_parameterizations('GLND')
        errorgen_propagator = ErrorGeneratorPropagator(noisy_model)
        circuit_2Q = list(smq2Q_XYCPHASE.create_gst_experiment_design(4).all_circuits_needing_data)[-1]

        #make sure that the various methods don't die.
        propagated_errorgens = errorgen_propagator.propagate_errorgens(circuit_2Q)
        gate_contributors = _epoly.errorgen_gate_contributors(noisy_model, LocalElementaryErrorgenLabel('H', ['XI']), circuit_2Q, 1, include_spam=True) 

    def test_cloud_crosstalk_model(self):
        oq=['Gxpi2','Gypi2','Gzpi2']
        qbts=4
        gate_names=oq+['Gcphase']
        max_strengths = {1: {'S': 10**(-3), 'H': 10**(-2)},
                        2: {'S': (1/6)*10**(-2), 'H': 2*10**(-3)}
                        }

        #Build circuit models
        qubit_labels =range(qbts)
        gate_names = ['Gxpi2','Gzpi2','Gcphase','Gypi2']
        ps = QubitProcessorSpec(qbts, gate_names,availability= {'Gcphase':[(i,(i+1)%qbts) for i in range(qbts)]} , qubit_labels=qubit_labels)
        lindblad_error_coeffs=sample_error_rates_cloud_crosstalk(max_strengths,4,gate_names)
        mdl_cloudnoise = create_cloud_crosstalk_model(ps, lindblad_error_coeffs=lindblad_error_coeffs, errcomp_type="errorgens")
        errorgen_prop=ErrorGeneratorPropagator(mdl_cloudnoise)
        propagated_errorgens = errorgen_prop.propagate_errorgens(self.circuit)
        gate_contributors = _epoly.errorgen_gate_contributors(mdl_cloudnoise, LocalElementaryErrorgenLabel('H', ['IZZI']), self.circuit, 1, include_spam=True) 

class LocalStimErrorgenLabelTester(BaseCase):
    def setUp(self):
        self.local_eel = LocalElementaryErrorgenLabel('C', ['XX', 'YY'])
        self.global_eel = GlobalElementaryErrorgenLabel('C', ['XX', 'YY'], (0,1))
        self.sslbls = [0,1]
        self.tableau = stim.PauliString('XI').to_tableau()

    def test_cast(self):
        correct_lse = _LSE('C', [stim.PauliString('XX'), stim.PauliString('YY')])

        self.assertEqual(correct_lse, _LSE.cast(self.local_eel))
        self.assertEqual(correct_lse, _LSE.cast(self.global_eel, self.sslbls))

    def test_to_local_global_eel(self):
        lse = _LSE('C', [stim.PauliString('XX'), stim.PauliString('YY')])

        self.assertEqual(lse.to_local_eel(), self.local_eel)
        self.assertEqual(lse.to_global_eel(), self.global_eel)
    
    def test_propagate_error_gen_tableau(self):
        lse = _LSE('C', [stim.PauliString('XX'), stim.PauliString('YY')])
        propagated_lse = lse.propagate_error_gen_tableau(self.tableau, 1)
        self.assertEqual(propagated_lse, (_LSE('C', [stim.PauliString('XX'), stim.PauliString('YY')]), -1))
        
        lse = _LSE('S', [stim.PauliString('ZI')])
        propagated_lse = lse.propagate_error_gen_tableau(self.tableau, 1)
        self.assertEqual(propagated_lse, (_LSE('S', [stim.PauliString('ZI')]), 1))

    def test_type_idx(self):
        from pygsti.errorgenpropagation.localstimerrorgen import _ERRORGEN_TYPE_INDICES
        self.assertEqual(_ERRORGEN_TYPE_INDICES, {'H': 0, 'S': 1, 'C': 2, 'A': 3})
        for typ, bels in [('H', ['XI']), ('S', ['XI']), ('C', ['XI', 'YI']), ('A', ['XI', 'YI'])]:
            lse = _LSE.cast((typ, bels))
            self.assertEqual(lse.type_idx, _ERRORGEN_TYPE_INDICES[typ])
            self.assertEqual(copy.copy(lse).type_idx, lse.type_idx)
            self.assertEqual(copy.deepcopy(lse).type_idx, lse.type_idx)
            self.assertEqual(pickle.loads(pickle.dumps(lse)).type_idx, lse.type_idx)
        with self.assertRaises(ValueError):
            _LSE('Q', [stim.PauliString('XI')])

    def test_hash_and_equality(self):
        # equal labels built independently compare equal and hash equal; H_P and S_P (same Pauli)
        # must differ in both (the hash string carries the type letter); other objects are unequal.
        a = _LSE('C', [stim.PauliString('XX'), stim.PauliString('YY')])
        b = _LSE.cast(('C', ['XX', 'YY']))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)
        h = _LSE('H', [stim.PauliString('XX')])
        s = _LSE('S', [stim.PauliString('XX')])
        self.assertNotEqual(h, s)
        self.assertNotEqual(hash(h), hash(s))
        self.assertNotEqual(a, _LSE('A', [stim.PauliString('XX'), stim.PauliString('YY')]))
        self.assertNotEqual(a, LocalElementaryErrorgenLabel('C', ['XX', 'YY']))
        self.assertFalse(a == ('C', ('XX', 'YY')))
        # labels carrying propagation metadata are still equal to their plain counterpart.
        self.assertEqual(_LSE('H', [stim.PauliString('XX')], circuit_time=3, label='foo'), h)

    def test_unpickle_legacy_states(self):
        # Pickles written by older versions of LocalStimErrorgenLabel lack attributes added
        # since. Emulate them by editing the instance __dict__ before pickling (the default
        # reduce protocol pickles __dict__ verbatim) and check that __setstate__ migrates them.
        # Use a propagated label so that the stored pre-propagation initial_label differs from
        # the label itself and its preservation can be checked.
        original = _LSE('A', [stim.PauliString('XI'), stim.PauliString('YI')])
        propagated, sign = original.propagate_error_gen_tableau(stim.Tableau.from_named_gate('H') + stim.Tableau(1), 1.0)
        self.assertNotEqual(propagated, original)
        self.assertEqual(propagated.initial_label, original.to_local_eel())
        fresh_state = dict(propagated.__dict__)

        def roundtrip(state):
            legacy = copy.copy(propagated)
            legacy.__dict__.clear()
            legacy.__dict__.update(state)
            return pickle.loads(pickle.dumps(legacy))

        # (1) state written before `type_idx` existed.
        state = dict(fresh_state)
        del state['type_idx']
        restored = roundtrip(state)
        self.assertEqual(restored.type_idx, propagated.type_idx)

        # (2) state written before `initial_label` became a lazy property (plain attribute,
        #     always materialized) and before `type_idx` existed.
        state = dict(fresh_state)
        del state['type_idx']
        state['initial_label'] = state.pop('_initial_label')
        restored = roundtrip(state)
        self.assertEqual(restored.type_idx, propagated.type_idx)
        self.assertEqual(restored.initial_label, original.to_local_eel())

        # (3) as (2), and additionally without the cached hashable string representations.
        del state['_hashable_basis_element_labels']
        del state['_hashable_string_rep']
        restored = roundtrip(state)
        self.assertEqual(restored._hashable_basis_element_labels, propagated._hashable_basis_element_labels)
        self.assertEqual(restored.initial_label, original.to_local_eel())

        # (4) state with the older format of the hash/equality string (type letter used as the
        #     joiner, so absent for single-index labels): it must be rebuilt, not trusted.
        state = dict(fresh_state)
        state['_hashable_string_rep'] = state['errorgen_type'].join(state['_hashable_basis_element_labels'])
        restored = roundtrip(state)
        self.assertEqual(restored._hashable_string_rep, propagated._hashable_string_rep)

        # (5) state written before the cached support mask existed (or pickled before the mask
        #     was first built): it must be built on demand.
        state = dict(fresh_state)
        self.assertNotIn('_support_mask', state)
        restored = roundtrip(state)
        self.assertEqual(restored.support_mask, propagated.support_mask)
        self.assertIn('_support_mask', restored.__dict__)

    def test_support_mask(self):
        from pygsti.errorgenpropagation import localstimerrorgen as _lse_mod
        # bit q of the mask is set iff some basis element label is non-identity on qubit q.
        self.assertEqual(_LSE.cast(('H', ['XI'])).support_mask, 0b01)
        self.assertEqual(_LSE.cast(('S', ['IZ'])).support_mask, 0b10)
        self.assertEqual(_LSE.cast(('C', ['XI', 'IY'])).support_mask, 0b11)
        self.assertEqual(_LSE.cast(('A', ['IIX', 'IIZ'])).support_mask, 0b100)
        self.assertEqual(_LSE.cast(('H', ['I' * 70 + 'Y' + 'I' * 29])).support_mask, 1 << 70)
        # the mask is built lazily, cached, and survives copies and pickling.
        lbl = _LSE.cast(('C', ['XI', 'IY']))
        self.assertNotIn('_support_mask', lbl.__dict__)
        self.assertEqual(lbl.support_mask, 0b11)
        self.assertEqual(lbl.__dict__['_support_mask'], 0b11)
        for other in [copy.copy(lbl), copy.deepcopy(lbl), pickle.loads(pickle.dumps(lbl))]:
            self.assertEqual(other.support_mask, 0b11)
        # the pure-python builder and the one in use (cython when built) agree with an
        # independent construction from stim's `pauli_indices`, for random labels of all
        # types at a range of qubit counts (including the 64-bit chunk boundaries).
        rng = np.random.default_rng(0)
        for n in (1, 2, 3, 63, 64, 65, 100, 129):
            for _ in range(25):
                typ = rng.choice(['H', 'S', 'C', 'A'])
                bels = []
                for _ in range(1 if typ in 'HS' else 2):
                    s = ['I'] * n
                    for q in rng.choice(n, size=rng.integers(1, n + 1), replace=False):
                        s[q] = rng.choice(['X', 'Y', 'Z'])
                    bels.append(stim.PauliString(''.join(s)))
                lbl = _LSE(typ, bels)
                expected = 0
                for p in bels:
                    for q in p.pauli_indices():
                        expected |= 1 << q
                self.assertEqual(lbl.support_mask, expected)
                self.assertEqual(_lse_mod._slow_support_mask(lbl._hashable_basis_element_labels), expected)
                self.assertEqual(_lse_mod.support_mask_from_strings(lbl._hashable_basis_element_labels), expected)

class FixedLayerErrorgenPropTester(BaseCase):
    """Coverage for ``ErrorGeneratorPropagator(fixed_errorgen_layer=...)`` construction,
    validation, SPAM layer counts, copy/aliasing behavior, and the (currently broken)
    matrix output path in fixed-layer mode.
    """

    def setUp(self):
        # a 2-qubit local-label fixed error generator layer
        self.fixed_local = {
            LocalElementaryErrorgenLabel('H', ['XX']): 0.01,
            LocalElementaryErrorgenLabel('S', ['ZZ']): 0.02,
        }
        self.empty_circuit = Circuit([], line_labels=(0, 1))
        self.two_layer_circuit = Circuit([[('Gxpi2', 0)], [('Gxpi2', 1)]], line_labels=(0, 1))

    # ------------------------------------------------------------------
    # construction + label casting
    # ------------------------------------------------------------------

    def test_construct_with_local_labels_casts_to_lse(self):
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer=self.fixed_local)
        self.assertIsNone(prop.model)
        # keys are cast to LocalStimErrorgenLabel, rates preserved.
        self.assertTrue(all(isinstance(k, _LSE) for k in prop.fixed_errorgen_layer))
        self.assertEqual(prop.fixed_errorgen_layer[_LSE('H', [stim.PauliString('XX')])], 0.01)
        self.assertEqual(prop.fixed_errorgen_layer[_LSE('S', [stim.PauliString('ZZ')])], 0.02)

    def test_construct_with_global_labels_int_state_space(self):
        fixed_global = {GlobalElementaryErrorgenLabel('H', ['XX'], (0, 1)): 0.01}
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer=fixed_global, state_space_labels=2)
        self.assertEqual(list(prop.fixed_errorgen_layer.keys()), [_LSE('H', [stim.PauliString('XX')])])

    def test_construct_with_global_labels_list_state_space(self):
        fixed_global = {GlobalElementaryErrorgenLabel('H', ['XX'], (0, 1)): 0.01}
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer=fixed_global, state_space_labels=[0, 1])
        self.assertEqual(list(prop.fixed_errorgen_layer.keys()), [_LSE('H', [stim.PauliString('XX')])])

    # ------------------------------------------------------------------
    # constructor validation
    # ------------------------------------------------------------------

    def test_cannot_specify_both_model_and_fixed_layer(self):
        model = smq2Q_XYCPHASE.target_model('full TP')
        with self.assertRaises(AssertionError):
            ErrorGeneratorPropagator(model=model, fixed_errorgen_layer=self.fixed_local)

    def test_must_specify_model_or_fixed_layer(self):
        with self.assertRaises(AssertionError):
            ErrorGeneratorPropagator()

    def test_global_labels_require_state_space_labels(self):
        fixed_global = {GlobalElementaryErrorgenLabel('H', ['XX'], (0, 1)): 0.01}
        with self.assertRaises(AssertionError):
            ErrorGeneratorPropagator(fixed_errorgen_layer=fixed_global)

    def test_mixed_width_labels_raise(self):
        mixed = {
            LocalElementaryErrorgenLabel('H', ['XX']): 0.01,
            LocalElementaryErrorgenLabel('S', ['Z']): 0.02,
        }
        with self.assertRaises(AssertionError):
            ErrorGeneratorPropagator(fixed_errorgen_layer=mixed)

    # ------------------------------------------------------------------
    # construct_errorgen_layers behavior
    # ------------------------------------------------------------------

    def test_circuit_width_mismatch_raises(self):
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer=self.fixed_local)  # 2-qubit support
        three_qubit_circuit = Circuit([], line_labels=(0, 1, 2))
        with self.assertRaises(AssertionError):
            prop.construct_errorgen_layers(three_qubit_circuit, 3, include_spam=True)

    def test_spam_layer_counts(self):
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer=self.fixed_local)
        # empty circuit: 0 layers, +2 for spam.
        self.assertEqual(len(prop.construct_errorgen_layers(self.empty_circuit, 2, include_spam=True)), 2)
        self.assertEqual(len(prop.construct_errorgen_layers(self.empty_circuit, 2, include_spam=False)), 0)
        # depth-2 circuit: 2 layers, +2 for spam.
        self.assertEqual(len(prop.construct_errorgen_layers(self.two_layer_circuit, 2, include_spam=True)), 4)
        self.assertEqual(len(prop.construct_errorgen_layers(self.two_layer_circuit, 2, include_spam=False)), 2)

    def test_each_layer_matches_fixed_layer(self):
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer=self.fixed_local)
        layers = prop.construct_errorgen_layers(self.two_layer_circuit, 2, include_spam=True)
        for layer in layers:
            self.assertEqual(layer, prop.fixed_errorgen_layer)

    def test_layers_are_independent_copies(self):
        # the per-layer .copy() must keep layers (and the stored fixed layer) from aliasing.
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer=self.fixed_local)
        layers = prop.construct_errorgen_layers(self.two_layer_circuit, 2, include_spam=True)
        key = next(iter(layers[0]))
        layers[0][key] = 999.0
        self.assertNotEqual(layers[1][key], 999.0)
        self.assertNotEqual(prop.fixed_errorgen_layer[key], 999.0)

    def test_empty_fixed_layer_constructs_empty_layers(self):
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer={})
        self.assertEqual(prop.fixed_errorgen_layer, {})
        layers = prop.construct_errorgen_layers(self.empty_circuit, 2, include_spam=True)
        self.assertEqual(layers, [{}, {}])

    def test_fixed_rate_overrides_all_rates(self):
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer=self.fixed_local)
        # fixed_rate=None keeps the stored rates.
        default_layers = prop.construct_errorgen_layers(self.empty_circuit, 2, include_spam=True, fixed_rate=None)
        self.assertEqual(set(default_layers[0].values()), {0.01, 0.02})
        # fixed_rate=1 sets every rate to 1.
        ones = prop.construct_errorgen_layers(self.empty_circuit, 2, include_spam=True, fixed_rate=1)
        self.assertEqual(set(ones[0].values()), {1})
        # fixed_rate=0 sets every rate to 0 (the `is not None` behavior, not falsy fall-through).
        zeros = prop.construct_errorgen_layers(self.empty_circuit, 2, include_spam=True, fixed_rate=0)
        self.assertEqual(set(zeros[0].values()), {0})
        self.assertEqual(len(zeros[0]), len(prop.fixed_errorgen_layer))


    def test_eoc_channel_fixed_errorgen(self):
        #confirm dense EOC channel array works with fixed error gener
        prop = ErrorGeneratorPropagator(fixed_errorgen_layer=self.fixed_local)
        channel = prop.eoc_error_channel(self.empty_circuit)
        self.assertEqual(channel.shape, (4 ** 2, 4 ** 2))


#Helper Functions:
def probabilities_errorgen_prop(error_propagator, target_model, circuit, use_bch=False, bch_order=1, truncation_threshold=1e-14, bch_mode='magnus'):
    #get the eoc error channel, and the process matrix for the ideal circuit:
    if use_bch:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True, use_bch=use_bch,
                                                        bch_kwargs={'bch_order':bch_order,
                                                                    'truncation_threshold':truncation_threshold,
                                                                    'mode':bch_mode})
    else:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True)
    ideal_channel = target_model.sim.product(circuit)
    #also get the ideal state prep and povm:
    ideal_prep = target_model.circuit_layer_operator(Label('rho0'), typ='prep').copy()
    ideal_meas = target_model.circuit_layer_operator(Label('Mdefault'), typ='povm').copy()
    #calculate the probabilities.
    prob_vec = np.zeros(len(ideal_meas))
    for i, effect in enumerate(ideal_meas.values()):
        dense_effect = effect.to_dense().copy()
        dense_prep = ideal_prep.to_dense().copy()
        prob_vec[i] = np.linalg.multi_dot([dense_effect.reshape((1, -1)), eoc_channel, ideal_channel, dense_prep.reshape((-1, 1))]).item()
    return prob_vec

def probabilities_fwdsim(noise_model, circuit):
    prob_dict = noise_model.sim.probs(circuit)
    prob_vec = np.fromiter(prob_dict.values(), dtype=np.double)
    return prob_vec

def sample_error_rates_dict(pspec, strengths, seed=None):
    """
    For example:
        strengths = {1: {'S':0.001, 'H':0.01}, 
                    2: {'S':0.01,'H':0.1}}

    The 'S' and 'H' entries in the strengths dictionary give 
    the maximum possible contribution to the infidelity from a given gate.
    """
    qubits = pspec.qubit_labels
    errors_rates_dict = {}
    for gate, availability in pspec.availability.items():
        n = pspec.gate_num_qubits(gate)
        if availability == 'all-edges':
            assert(n == 1), "Currently require all 2-qubit gates have a specified availability!"
            qubits_for_gate = qubits
        else:
            qubits_for_gate = availability  
        for qs in qubits_for_gate:
            label = Label(gate, qs)
            # First, check if there's a strength specified for this specific gate.
            max_stength = strengths.get(label, None) # to get highly biased errors can set generic error rates to be low, then set it to be high for one or two particular gates.
            # Next, check if there's a strength specified for all gates with this name
            if max_stength is None:
                max_stength = strengths.get(gate, None)
            # Finally, get error rate for all gates on this number of qubits.
            if max_stength is None:
                max_stength = strengths[n]
            # Sample error rates.
            errors_rates_dict[label] = sample_error_rates(max_stength, n, seed)
    return errors_rates_dict

def sample_error_rates(strengths, n, seed = None):
    '''
    Samples an error rates dictionary for dependent gates.
    '''
    error_rates_dict = {}
    
    #create a basis to get the basis element labels.
    basis = BuiltinBasis('pp', 4**n)
    
    #set the rng
    rng = np.random.default_rng(seed)
    
    # Sample stochastic error rates. First we sample the overall stochastic error rate.
    # Then we sample (and normalize) the individual stochastic error rates
    stochastic_strength = strengths['S'] * rng.random()
    s_error_rates = rng.random(4 ** n - 1)
    s_error_rates = s_error_rates / np.sum(s_error_rates) * stochastic_strength

    hamiltonian_strength = strengths['H'] * rng.random()
    h_error_rates = rng.random(4 ** n - 1)
    h_error_rates = h_error_rates * np.sqrt(hamiltonian_strength) / np.sqrt(np.sum(h_error_rates**2))

    error_rates_dict.update({('S', basis.labels[i + 1]): s_error_rates[i] for i in range(4 ** n - 1)})
    error_rates_dict.update({('H', basis.labels[i + 1]): h_error_rates[i] for i in range(4 ** n - 1)})

    return error_rates_dict

def comm_list_to_matrix(comm_list, errorgen_matrix_dict, num_qubits):
    #if the list is empty return all zeros
    #initialize empty array for accumulation.
    mat = np.zeros((4**num_qubits, 4**num_qubits), dtype=np.complex128)
    if not comm_list:
        return mat
    
    #infer the correct label type.
    if errorgen_matrix_dict:
        first_label = next(iter(errorgen_matrix_dict))
        if isinstance(first_label, LocalElementaryErrorgenLabel):
            label_type = 'local'
        elif isinstance(first_label, GlobalElementaryErrorgenLabel):
            label_type = 'global'
        else:
            msg = f'Label type {type(first_label)} is not supported as a key for errorgen_matrix_dict.'\
                  + 'Please use either LocalElementaryErrorgenLabel or GlobalElementaryErrorgenLabel.'
            raise ValueError()
    else:
        raise ValueError('Non-empty commutatory result list, but the dictionary is empty. Cannot convert.')
        
    #loop through comm_list and accumulate the weighted error generators prescribed.
    if label_type == 'local':
        for comm_tup in comm_list:
            mat +=  comm_tup[1]*errorgen_matrix_dict[comm_tup[0].to_local_eel()]
    else:
        for comm_tup in comm_list:
            mat +=  comm_tup[1]*errorgen_matrix_dict[comm_tup[0].to_global_eel()]
            
    return mat

def error_generator_commutator_numerical(errorgen_1, errorgen_2, errorgen_matrix_dict):
    return errorgen_matrix_dict[errorgen_1]@errorgen_matrix_dict[errorgen_2] - errorgen_matrix_dict[errorgen_2]@errorgen_matrix_dict[errorgen_1]


#--------- Cloud crosstalk helper functions---------------------#
def sample_error_rates_cloud_crosstalk(strengths,qbts, gates):
    error_rates_dict = {}
    for gate in gates:
        if not gate =='Gcphase':
            for el in range(qbts):
                stochastic_strength = strengths[1]['S']*np.random.random()
                hamiltonian_strength = 2*strengths[1]['H']*np.random.random()-strengths[1]['H']        
                paulis=['X','Y','Z']
                error_rates_dict[(gate,el)]=dict()
                for pauli_label in paulis:
                    if (gate=='Gxpi2' and pauli_label=='X') or (gate=='Gypi2' and pauli_label=='Y') or (gate=='Gzpi2' and pauli_label=='Z'):
                        error_rates_dict[(gate,el)].update({('H', pauli_label+':'+str(el)):hamiltonian_strength})
                        error_rates_dict[(gate,el)].update({('S', pauli_label+':'+str(el)): stochastic_strength})
                    else:
                        error_rates_dict[(gate,el)].update({('H', pauli_label+':'+str(el)):0.0})
                        error_rates_dict[(gate,el)].update({('S', pauli_label+':'+str(el)): 0.0})
        else:
            for qbt in range(qbts):
                
                gate_lbl=('Gcphase',qbt,(qbt+1)%4)
                error_rates_dict[gate_lbl]=dict()
                for qbt1 in range(qbts):
                    for qbt2 in range(qbts):
                        if qbt1 < qbt2:
                            hamiltonian_strength = 2*strengths[2]['H']*np.random.random()-strengths[2]['H']
                            for pauli in two_qbt_pauli_str():
                                if pauli =='ZZ':
                                    error_rates_dict[gate_lbl].update({('H',pauli+':'+str(qbt1)+','+str(qbt2)):hamiltonian_strength})
                                else:
                                    error_rates_dict[gate_lbl].update({('H',pauli+':'+str(qbt1)+','+str(qbt2)):0.0})
                
                for qbt1 in range(qbts):
                    hamiltonian_strength = 2*strengths[2]['H']*np.random.random()-strengths[2]['H']
                    for pauli in ['X','Y','Z']:
                        if pauli=='Z':
                            error_rates_dict[gate_lbl].update({('H',pauli+':'+str(qbt1)):hamiltonian_strength})
                        else:
                            error_rates_dict[gate_lbl].update({('H',pauli+':'+str(qbt1)):0.0})


                stochastic_strength = strengths[2]['S']*np.random.random()
                error_rates_dict[gate_lbl].update({('S', 'ZZ:'+str(gate_lbl[1])+','+str(gate_lbl[2])): stochastic_strength})
                stochastic_strength = strengths[2]['S']*np.random.random()
                error_rates_dict[gate_lbl].update({('S', 'Z:'+str(gate_lbl[1])): stochastic_strength})
                stochastic_strength = strengths[2]['S']*np.random.random()
                error_rates_dict[gate_lbl].update({('S', 'Z:'+str(gate_lbl[2])): stochastic_strength})

    return error_rates_dict

def two_qbt_pauli_str():
    paulis=['I','X','Y','Z']
    pauli_strs=[]
    for p1 in paulis:
        for p2 in paulis:
            pauli_strs.append(p1+p2)
    pauli_strs.remove('II')
    return pauli_strs

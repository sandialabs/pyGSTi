"""Unit tests for the POVM container classes in pygsti.modelmembers.povms.

These tests focus on behaviors of the POVM (dict-of-effects) container
classes themselves -- construction, vectorization, serialization,
dict-protocol methods, and probability computation -- as opposed to the
individual POVMEffect classes (see test_spamvec.py for those).
"""
import pickle

import numpy as np

import pygsti.modelmembers.operations as operations
import pygsti.modelmembers.povms as povms
import pygsti.modelmembers.states as states
from pygsti.models import ExplicitOpModel, Model

from ...util import BaseCase


def roundtrip_povm_via_model(povm, num_qubits, prep=None):
    """Serialize `povm` by embedding it into a fresh ExplicitOpModel, dumping the model
    to (json) string form, and reloading it -- returning the reloaded POVM.  This exercises
    the to_memoized_dict/_from_memoized_dict serialization path used for model persistence.
    """
    qubit_labels = ['Q%d' % i for i in range(num_qubits)]
    mdl = ExplicitOpModel(qubit_labels, evotype='default')
    if prep is None:
        prep = states.ComputationalBasisState([0] * num_qubits, 'pp', 'default')
    mdl['rho0'] = prep
    mdl['Mdefault'] = povm
    mdl._rebuild_paramvec()
    s = mdl.dumps()
    mdl2 = Model.loads(s)
    return mdl2['Mdefault']


class DictProtocolMixin(object):
    """Mixin testing the lazy dict-protocol methods common to several POVM container classes."""

    def test_dict_protocol(self):
        keys = list(self.povm.keys())
        self.assertEqual(len(self.povm), len(keys))
        self.assertEqual(list(iter(self.povm)), keys)
        for k in keys:
            self.assertIn(k, self.povm)
        self.assertNotIn('not-a-valid-outcome-label', self.povm)

        values = list(self.povm.values())
        self.assertEqual(len(values), len(keys))
        items = list(self.povm.items())
        self.assertEqual([k for k, v in items], keys)
        for (k, v), direct_v in zip(items, values):
            self.assertArraysAlmostEqual(v.to_dense(), direct_v.to_dense())
            self.assertArraysAlmostEqual(self.povm[k].to_dense(), v.to_dense())

        with self.assertRaises(KeyError):
            self.povm['not-a-valid-outcome-label']

    def test_repeated_access_returns_cached_instance(self):
        # For POVMs whose effects are lazily constructed on first access (see _LazyEffectsPOVM),
        # repeated access to the same key should return the *same* (identical) cached effect
        # object, rather than reconstructing a new one on every access.
        key = next(iter(self.povm.keys()))
        e1 = self.povm[key]
        e2 = self.povm[key]
        self.assertIs(e1, e2)

    def test_simplify_effects(self):
        simplified = self.povm.simplify_effects(prefix="pfx")
        self.assertEqual(len(simplified), len(self.povm))
        for k in self.povm.keys():
            self.assertIn("pfx_" + k, simplified)


class ExplicitPOVMBase(object):
    """Common tests for POVMs holding explicitly-parameterized (Full/Static) effects."""

    def setUp(self):
        ExplicitOpModel._strict = False
        self.povm = self.build_povm()

    def test_num_params(self):
        self.assertEqual(self.povm.num_params, self.n_params)

    def test_num_elements(self):
        self.assertEqual(self.povm.num_elements, len(self.povm) * self.povm['0'].hilbert_schmidt_size)

    def test_vector_conversion_roundtrip(self):
        v0 = self.povm.to_vector().copy()
        dense0 = {k: e.to_dense().copy() for k, e in self.povm.items()}
        self.povm.from_vector(v0)
        self.assertArraysAlmostEqual(self.povm.to_vector(), v0)
        for k, e in self.povm.items():
            self.assertArraysAlmostEqual(e.to_dense(), dense0[k])

    def test_acton_probabilities(self):
        state = states.FullState([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], 'pp')
        probs = self.povm.acton(state)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertAlmostEqual(probs['0'], 1.0, places=6)
        self.assertAlmostEqual(probs['1'], 0.0, places=6)

    def test_setitem_blocked_after_init(self):
        with self.assertRaises(NotImplementedError):
            self.povm['0'] = self.povm['0'].copy()

    def test_pickle(self):
        pklstr = pickle.dumps(self.povm)
        povm2 = pickle.loads(pklstr)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertArraysAlmostEqual(povm2.to_vector(), self.povm.to_vector())

    def test_serialization_roundtrip(self):
        povm2 = roundtrip_povm_via_model(self.povm, 1)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertEqual(povm2.num_params, self.povm.num_params)
        for k in self.povm.keys():
            self.assertArraysAlmostEqual(povm2[k].to_dense(), self.povm[k].to_dense())


def _build_full_effects():
    v0 = np.array([[1.0], [0], [0], [1.0]]) / np.sqrt(2)
    v1 = np.array([[1.0], [0], [0], [-1.0]]) / np.sqrt(2)
    return [('0', povms.FullPOVMEffect(v0, 'pp')), ('1', povms.FullPOVMEffect(v1, 'pp'))]


class UnconstrainedPOVMTester(ExplicitPOVMBase, DictProtocolMixin, BaseCase):
    n_params = 8

    @staticmethod
    def build_povm():
        return povms.UnconstrainedPOVM(_build_full_effects())

    def test_construction_raises_on_bad_input(self):
        with self.assertRaises(ValueError):
            povms.UnconstrainedPOVM("NotAListOrDict")

    def test_transform_inplace_identity(self):
        from pygsti.models.gaugegroup import FullGaugeGroupElement
        dense0 = {k: e.to_dense().copy() for k, e in self.povm.items()}
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        self.povm.transform_inplace(S)
        for k, e in self.povm.items():
            self.assertArraysAlmostEqual(e.to_dense(), dense0[k])

    def test_depolarize(self):
        self.povm.depolarize(0.9)
        # complete depolarization (amount=1) should send all effects toward the maximally-mixed value
        # except for their first (identity) component -- just check that it runs and produces valid probs.
        state = states.FullState([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], 'pp')
        probs = self.povm.acton(state)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)


class TPPOVMTester(ExplicitPOVMBase, DictProtocolMixin, BaseCase):
    n_params = 4  # one effect is a "complement" effect with no free parameters of its own

    @staticmethod
    def build_povm():
        return povms.TPPOVM(_build_full_effects())

    def test_complement_label_set(self):
        self.assertIsNotNone(self.povm.complement_label)
        self.assertIn(self.povm.complement_label, self.povm.keys())

    def test_complement_effect_sums_to_identity(self):
        total = sum(e.to_dense() for e in self.povm.values())
        identity = np.zeros(4, 'd')
        identity[0] = 4 ** 0.25
        self.assertArraysAlmostEqual(total, identity)

    def test_transform_inplace_identity(self):
        from pygsti.models.gaugegroup import FullGaugeGroupElement
        dense0 = {k: e.to_dense().copy() for k, e in self.povm.items()}
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        self.povm.transform_inplace(S)
        for k, e in self.povm.items():
            self.assertArraysAlmostEqual(e.to_dense(), dense0[k])

    def test_depolarize(self):
        self.povm.depolarize(0.9)
        state = states.FullState([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], 'pp')
        probs = self.povm.acton(state)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)


class ComputationalBasisPOVMTester(DictProtocolMixin, BaseCase):
    def setUp(self):
        ExplicitOpModel._strict = False
        self.povm = povms.ComputationalBasisPOVM(2, 'default')

    def test_num_params(self):
        self.assertEqual(self.povm.num_params, 0)

    def test_len_and_keys(self):
        self.assertEqual(len(self.povm), 4)
        self.assertEqual(set(self.povm.keys()), {'00', '01', '10', '11'})

    def test_qubit_filter_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            povms.ComputationalBasisPOVM(2, 'default', qubit_filter=[0])

    def test_acton_probabilities(self):
        state = states.ComputationalBasisState([0, 1], 'pp', 'default')
        probs = self.povm.acton(state)
        self.assertAlmostEqual(probs['01'], 1.0, places=6)
        for k, p in probs.items():
            if k != '01':
                self.assertAlmostEqual(p, 0.0, places=6)

    def test_from_pure_vectors(self):
        v = (np.array([1, 0], 'd'), np.array([0, 1], 'd'))
        pure_vectors = {'00': np.kron(v[0], v[0]), '01': np.kron(v[0], v[1]),
                        '10': np.kron(v[1], v[0]), '11': np.kron(v[1], v[1])}
        povm2 = povms.ComputationalBasisPOVM.from_pure_vectors(pure_vectors, 'default', None)
        self.assertEqual(povm2.nqubits, 2)

    def test_from_pure_vectors_raises_on_non_zbasis(self):
        bad_vectors = {'00': np.array([1, 1, 0, 0], 'd'), '01': np.array([0, 0, 1, 0], 'd'),
                       '10': np.array([0, 0, 0, 1], 'd'), '11': np.array([0, 0, 0, 1], 'd')}
        with self.assertRaises(ValueError):
            povms.ComputationalBasisPOVM.from_pure_vectors(bad_vectors, 'default', None)

    def test_is_similar(self):
        same_nqubits = povms.ComputationalBasisPOVM(2, 'default')
        self.assertTrue(self.povm._is_similar(same_nqubits, rtol=1e-6, atol=1e-9))
        different_nqubits = povms.ComputationalBasisPOVM(3, 'default')
        self.assertFalse(self.povm._is_similar(different_nqubits, rtol=1e-6, atol=1e-9))

    def test_pickle(self):
        pklstr = pickle.dumps(self.povm)
        povm2 = pickle.loads(pklstr)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertEqual(povm2.nqubits, self.povm.nqubits)

    def test_serialization_roundtrip(self):
        povm2 = roundtrip_povm_via_model(self.povm, 2)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertEqual(povm2.nqubits, self.povm.nqubits)
        self.assertEqual(set(povm2.keys()), set(self.povm.keys()))


class MarginalizedPOVMTester(DictProtocolMixin, BaseCase):
    """Tests for MarginalizedPOVM -- prior to this test file, this class had *zero* test coverage."""

    def setUp(self):
        ExplicitOpModel._strict = False
        self.base_povm = povms.ComputationalBasisPOVM(2, 'default')
        self.povm = povms.MarginalizedPOVM(self.base_povm, (0, 1), (0,))

    def test_construction_keeps_correct_labels(self):
        # Marginalizing a 2-qubit computational POVM down to qubit 0 should give 2 outcomes
        self.assertEqual(set(self.povm.keys()), {'0', '1'})
        self.assertEqual(self.povm.num_params, 0)

    def test_acton_probabilities_match_hand_computed_marginal(self):
        # State |01>: after marginalizing away qubit 1, outcome should deterministically be '0'
        state = states.ComputationalBasisState([0, 1], 'pp', 'default')
        probs = self.povm.acton(state)
        self.assertAlmostEqual(probs['0'], 1.0, places=6)
        self.assertAlmostEqual(probs['1'], 0.0, places=6)

        state = states.ComputationalBasisState([1, 0], 'pp', 'default')
        probs = self.povm.acton(state)
        self.assertAlmostEqual(probs['0'], 0.0, places=6)
        self.assertAlmostEqual(probs['1'], 1.0, places=6)

    def test_marginalize_effect_label(self):
        self.assertEqual(self.povm.marginalize_effect_label('01'), '0')
        self.assertEqual(self.povm.marginalize_effect_label('10'), '1')

    def test_submembers(self):
        self.assertEqual(self.povm.submembers(), [self.base_povm])

    def test_simplify_effects_with_label_prefix(self):
        # MarginalizedPOVM.simplify_effects has a special code path for _Label-valued (as
        # opposed to plain string) prefixes -- exercise it explicitly (the generic
        # DictProtocolMixin.test_simplify_effects only covers the plain-string-prefix path).
        from pygsti.baseobjs.label import Label
        prefix = Label('Mdefault', (0,))
        simplified = self.povm.simplify_effects(prefix=prefix)
        self.assertEqual(len(simplified), len(self.povm))
        for k in self.povm.keys():
            self.assertIn(Label('Mdefault_' + k, (0,)), simplified)

    def test_is_similar(self):
        other = povms.MarginalizedPOVM(povms.ComputationalBasisPOVM(2, 'default'), (0, 1), (0,))
        self.assertTrue(self.povm._is_similar(other, rtol=1e-6, atol=1e-9))
        different = povms.MarginalizedPOVM(povms.ComputationalBasisPOVM(2, 'default'), (0, 1), (1,))
        self.assertFalse(self.povm._is_similar(different, rtol=1e-6, atol=1e-9))

    def test_pickle(self):
        pklstr = pickle.dumps(self.povm)
        povm2 = pickle.loads(pklstr)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertEqual(set(povm2.keys()), set(self.povm.keys()))

    def test_serialization_roundtrip(self):
        povm2 = roundtrip_povm_via_model(self.povm, 2)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertEqual(set(povm2.keys()), set(self.povm.keys()))
        self.assertEqual(tuple(povm2.sslbls_to_marginalize), tuple(self.povm.sslbls_to_marginalize))
        self.assertEqual(tuple(povm2.sslbls_after_marginalizing), tuple(self.povm.sslbls_after_marginalizing))


class TensorProductPOVMTester(DictProtocolMixin, BaseCase):
    def setUp(self):
        ExplicitOpModel._strict = False
        v0 = np.array([[1.0], [0], [0], [1.0]]) / np.sqrt(2)
        v1 = np.array([[1.0], [0], [0], [-1.0]]) / np.sqrt(2)
        self.factor0 = povms.UnconstrainedPOVM([('0', povms.FullPOVMEffect(v0.copy(), 'pp')),
                                                ('1', povms.FullPOVMEffect(v1.copy(), 'pp'))])
        self.factor1 = povms.UnconstrainedPOVM([('0', povms.FullPOVMEffect(v0.copy(), 'pp')),
                                                ('1', povms.FullPOVMEffect(v1.copy(), 'pp'))])
        self.povm = povms.TensorProductPOVM([self.factor0, self.factor1])

    def test_num_params(self):
        self.assertEqual(self.povm.num_params, 16)

    def test_keys(self):
        self.assertEqual(set(self.povm.keys()), {'00', '01', '10', '11'})

    def test_vector_conversion_roundtrip(self):
        v = self.povm.to_vector().copy()
        self.povm.from_vector(v)
        self.assertArraysAlmostEqual(self.povm.to_vector(), v)

    def test_acton_probabilities(self):
        from pygsti.modelmembers.states import TensorProductState
        state = TensorProductState([states.FullState([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], 'pp'),
                                    states.FullState([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], 'pp')],
                                   state_space=(0, 1))
        probs = self.povm.acton(state)
        self.assertAlmostEqual(probs['00'], 1.0, places=6)
        for k, p in probs.items():
            if k != '00':
                self.assertAlmostEqual(p, 0.0, places=6)

    def test_depolarize(self):
        self.povm.depolarize(0.9)
        self.assertTrue(self.povm.dirty)

    def test_submembers(self):
        self.assertEqual(self.povm.submembers(), [self.factor0, self.factor1])

    def test_pickle(self):
        pklstr = pickle.dumps(self.povm)
        povm2 = pickle.loads(pklstr)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertArraysAlmostEqual(povm2.to_vector(), self.povm.to_vector())

    def test_serialization_roundtrip(self):
        povm2 = roundtrip_povm_via_model(self.povm, 2)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertEqual(povm2.num_params, self.povm.num_params)
        self.assertEqual(set(povm2.keys()), set(self.povm.keys()))


class ComposedPOVMFactoryTester(DictProtocolMixin, BaseCase):
    """Additional coverage for ComposedPOVM not already handled in test_composed_spam.py.

    In particular, test_composed_spam.py doesn't exercise the lazy dict-protocol methods
    (__contains__, __iter__, keys, values, items, __getitem__) or simplify_effects, which
    ComposedPOVM implements via the shared _LazyEffectsPOVM mixin -- see DictProtocolMixin.
    """

    def setUp(self):
        ExplicitOpModel._strict = False
        self.base_povm = povms.ComputationalBasisPOVM(1, 'default')
        self.errormap = operations.FullArbitraryOp(np.identity(4, 'd'))
        self.povm = povms.ComposedPOVM(self.errormap, self.base_povm, 'pp')

    def test_num_params(self):
        self.assertEqual(self.povm.num_params, 16)

    def test_vector_conversion_roundtrip(self):
        v = self.povm.to_vector().copy()
        self.povm.from_vector(v)
        self.assertArraysAlmostEqual(self.povm.to_vector(), v)

    def test_submembers(self):
        self.assertEqual(self.povm.submembers(), [self.errormap, self.base_povm])

    def test_pickle(self):
        pklstr = pickle.dumps(self.povm)
        povm2 = pickle.loads(pklstr)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertArraysAlmostEqual(povm2.to_vector(), self.povm.to_vector())

    def test_serialization_roundtrip(self):
        povm2 = roundtrip_povm_via_model(self.povm, 1)
        self.assertEqual(type(povm2), type(self.povm))
        self.assertEqual(povm2.num_params, self.povm.num_params)
        for k in self.povm.keys():
            self.assertArraysAlmostEqual(povm2[k].to_dense(), self.povm[k].to_dense())


class POVMFactoryFunctionTester(BaseCase):
    """Tests for the module-level factory functions in pygsti.modelmembers.povms.__init__."""

    def setUp(self):
        ExplicitOpModel._strict = False
        v = (np.array([1, 0], 'd'), np.array([0, 1], 'd'))
        self.pure_vectors = {'0': v[0], '1': v[1]}
        self.dm_vectors = {'0': povms.create_effect_from_pure_vector(v[0], 'static').to_dense(),
                           '1': povms.create_effect_from_pure_vector(v[1], 'static').to_dense()}

    def test_create_from_pure_vectors_computational(self):
        povm = povms.create_from_pure_vectors(self.pure_vectors, 'computational')
        self.assertIsInstance(povm, povms.ComputationalBasisPOVM)
        self.assertEqual(povm.num_params, 0)

    def test_create_from_pure_vectors_full(self):
        povm = povms.create_from_pure_vectors(self.pure_vectors, 'full')
        self.assertIsInstance(povm, povms.UnconstrainedPOVM)
        self.assertGreater(povm.num_params, 0)

    def test_create_from_pure_vectors_full_tp(self):
        povm = povms.create_from_pure_vectors(self.pure_vectors, 'full TP')
        self.assertIsInstance(povm, povms.TPPOVM)

    def test_create_from_dmvecs_static(self):
        povm = povms.create_from_dmvecs(self.dm_vectors, 'static')
        self.assertIsInstance(povm, povms.UnconstrainedPOVM)
        self.assertEqual(povm.num_params, 0)

    def test_convert_effect_roundtrip(self):
        effect = povms.create_effect_from_pure_vector(self.pure_vectors['0'], 'full')
        basis = None
        converted = povms.convert_effect(effect, 'static', 'pp')
        self.assertArraysAlmostEqual(converted.to_dense(), effect.to_dense())

    def test_convert_effect_raises_on_invalid_type(self):
        effect = povms.create_effect_from_pure_vector(self.pure_vectors['0'], 'full')
        with self.assertRaises(ValueError):
            povms.convert_effect(effect, 'foobar', 'pp')

"""Unit tests for the individual POVMEffect classes in pygsti.modelmembers.povms.

These tests cover the "leaf" effect-vector classes (StaticPOVMEffect, FullPOVMEffect,
ConjugatedStatePOVMEffect, ComplementPOVMEffect, ComposedPOVMEffect, TensorProductPOVMEffect,
etc.) as opposed to the POVM container classes -- see test_povm.py for those.
"""
import pickle

import numpy as np

import pygsti.modelmembers.povms as povms
import pygsti.modelmembers.states as states
import pygsti.modelmembers.operations as operations
import pygsti.baseobjs.statespace as statespace
from pygsti.evotypes import Evotype
from pygsti.modelmembers.povms import TPPOVM, UnconstrainedPOVM
from pygsti.models import ExplicitOpModel
from pygsti.models.gaugegroup import FullGaugeGroupElement

from ...util import BaseCase
from ..test_spamvec import TensorProdStateBase


def fd_jac(f, v, eps=1e-6):
    """Central-difference Jacobian of array-valued f: returns shape f(v).shape + (len(v),)."""
    v = np.asarray(v, float)
    base = np.asarray(f(v))
    out = np.zeros(base.shape + (v.size,), dtype=complex)
    for k in range(v.size):
        vp = v.copy(); vp[k] += eps
        vm = v.copy(); vm[k] -= eps
        out[..., k] = (np.asarray(f(vp)) - np.asarray(f(vm))) / (2 * eps)
    return out


def assert_deriv_wrt_params_matches_finite_diff(testcase, vec, eps=1e-6, atol=1e-5, rtol=1e-4):
    """Checks vec.deriv_wrt_params() against a central-difference approximation of d(to_dense())/d(params)."""
    nparams = vec.num_params
    if nparams == 0:
        testcase.assertEqual(vec.deriv_wrt_params().shape, (vec.dim, 0))
        return
    v0 = vec.to_vector().copy()

    def to_dense_of(vv):
        vec.from_vector(vv)
        return vec.to_dense().copy()

    try:
        analytic = np.asarray(vec.deriv_wrt_params())
        fd = fd_jac(to_dense_of, v0, eps=eps)
    finally:
        vec.from_vector(v0)  # restore original parameter values

    testcase.assertEqual(analytic.shape, fd.shape)
    testcase.assertTrue(np.allclose(np.real(fd), np.real(analytic), atol=atol, rtol=rtol))
    testcase.assertTrue(np.allclose(np.imag(fd), np.imag(analytic), atol=atol, rtol=rtol))


class POVMEffectUtilTester(BaseCase):
    """Direct tests of the abstract POVMEffect base class's default method implementations.

    Prior to this test class, POVMEffect was *only* ever exercised indirectly through its
    subclasses (which override most of its methods), so none of these base-class defaults
    (num_params/to_vector/from_vector == 0-params, outcomes/taylor_order_terms raising
    NotImplementedError, set_dense/transform_inplace raising ValueError, etc.) had any
    direct test coverage.
    """

    @staticmethod
    def _build_raw_effect():
        state_space = statespace.default_space_for_dim(4)
        evotype = Evotype.cast('default')
        rep = evotype.create_dense_state_rep(np.zeros(4, 'd'), None, state_space)
        return povms.POVMEffect(rep, evotype)

    def test_outcomes_not_implemented(self):
        raw = self._build_raw_effect()
        with self.assertRaises(NotImplementedError):
            raw.outcomes

    def test_dim_and_hilbert_schmidt_size(self):
        raw = self._build_raw_effect()
        self.assertEqual(raw.dim, 4)
        self.assertEqual(raw.hilbert_schmidt_size, 4)

    def test_set_dense_raises(self):
        raw = self._build_raw_effect()
        with self.assertRaises(ValueError):
            raw.set_dense(np.zeros(4, 'd'))

    def test_set_time_is_noop(self):
        raw = self._build_raw_effect()
        raw.set_time(1.234)  # should just do nothing, and not raise

    def test_trivial_parameterization(self):
        raw = self._build_raw_effect()
        self.assertEqual(raw.num_params, 0)
        self.assertArraysEqual(raw.to_vector(), np.array([], 'd'))
        raw.from_vector(np.array([]))  # should succeed with an empty vector
        with self.assertRaises(AssertionError):
            raw.from_vector(np.array([1.0]))  # non-empty vector should fail the length assertion

    def test_deriv_wrt_params_default(self):
        raw = self._build_raw_effect()
        deriv = raw.deriv_wrt_params()
        self.assertEqual(deriv.shape, (raw.dim, 0))
        filtered = raw.deriv_wrt_params(wrt_filter=[])
        self.assertEqual(filtered.shape, (raw.dim, 0))

    def test_has_nonzero_hessian_default(self):
        raw = self._build_raw_effect()
        # base implementation: True iff num_params > 0; a 0-parameter raw effect has none.
        self.assertFalse(raw.has_nonzero_hessian())

    def test_taylor_order_terms_not_implemented(self):
        raw = self._build_raw_effect()
        with self.assertRaises(NotImplementedError):
            raw.taylor_order_terms(0)

    def test_highmagnitude_terms_propagates_not_implemented(self):
        # highmagnitude_terms() calls taylor_order_terms() internally, which the base
        # class doesn't implement.
        raw = self._build_raw_effect()
        with self.assertRaises(NotImplementedError):
            raw.highmagnitude_terms(min_term_mag=0.0)

    def test_taylor_order_terms_above_mag_propagates_not_implemented(self):
        raw = self._build_raw_effect()
        with self.assertRaises(NotImplementedError):
            raw.taylor_order_terms_above_mag(order=0, max_polynomial_vars=100, min_term_mag=0.0)

    def test_hessian_wrt_params_zero_param_path_is_broken(self):
        # NOTE: this documents a pre-existing bug in POVMEffect.hessian_wrt_params(): when
        # has_nonzero_hessian() is False (as it always is for a 0-parameter effect), it tries
        # to return `_np.zeros(self.size, self.num_params, self.num_params)`, but POVMEffect
        # has no `size` attribute (only `dim`/`hilbert_schmidt_size`), and even if it did,
        # _np.zeros expects a single shape tuple rather than 3 separate positional args. As a
        # result, calling hessian_wrt_params() on a 0-parameter effect currently raises
        # AttributeError instead of returning a zero array. If this is ever fixed, this test
        # should be updated to check the corrected return value instead.
        raw = self._build_raw_effect()
        self.assertFalse(raw.has_nonzero_hessian())
        with self.assertRaises(AttributeError):
            raw.hessian_wrt_params()


class ImmutableEffectTransformTester(BaseCase):
    """Covers POVMEffect.transform_inplace's generic (set_dense-based) implementation, using
    a concrete immutable effect (StaticPOVMEffect) to provide a working to_dense()."""

    def test_transform_inplace_raises_for_immutable_effect(self):
        effect = povms.StaticPOVMEffect(np.array([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], 'd'))
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            effect.transform_inplace(S)


class POVMEffectBase(object):

    def setUp(self):
        self.vec = self.build_vec()
        ExplicitOpModel._strict = False

    def test_num_params(self):
        self.assertEqual(self.vec.num_params, self.n_params)

    def test_copy(self):
        vec_copy = self.vec.copy()
        self.assertArraysAlmostEqual(vec_copy.to_dense(), self.vec.to_dense())
        self.assertEqual(type(vec_copy), type(self.vec))

    def test_get_dimension(self):
        self.assertEqual(self.vec.dim, 4)

    def test_set_value_raises_on_bad_size(self):
        with self.assertRaises(ValueError):
            self.vec.set_dense(np.zeros((1, 1), 'd'))  # bad size

    def test_vector_conversion(self):
        v = self.vec.to_vector()
        self.vec.from_vector(v)
        self.assertArraysAlmostEqual(self.vec.to_vector(), v)
        assert_deriv_wrt_params_matches_finite_diff(self, self.vec)

    def test_pickle(self):
        pklstr = pickle.dumps(self.vec)
        vec_pickle = pickle.loads(pklstr)
        self.assertArraysAlmostEqual(vec_pickle.to_dense(), self.vec.to_dense())
        self.assertEqual(type(vec_pickle), type(self.vec))


class ConjugatedStatePOVMEffectTester(POVMEffectBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        state = states.FullState([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], None, 'default', state_space=None)
        return povms.ConjugatedStatePOVMEffect(state)

    def test_set_value_raises_on_bad_size(self):
        # ConjugatedStatePOVMEffect has no set_dense of its own (inherits POVMEffect.set_dense, which
        # always raises ValueError, regardless of shape).
        with self.assertRaises(ValueError):
            self.vec.set_dense(np.zeros((1, 1), 'd'))

    def test_parameter_labels(self):
        # delegates directly to the wrapped state's parameter_labels
        self.assertArraysEqual(self.vec.parameter_labels, self.vec.state.parameter_labels)

    def test_str(self):
        s = str(self.vec)
        self.assertIn('ConjugatedStatePOVMEffect', s)

    def test_has_nonzero_hessian_delegates_to_state(self):
        # FullState.has_nonzero_hessian() always returns False (its parameterization is linear)
        self.assertEqual(self.vec.has_nonzero_hessian(), self.vec.state.has_nonzero_hessian())
        self.assertFalse(self.vec.has_nonzero_hessian())

    def test_hessian_wrt_params_zero_hessian_path_is_broken(self):
        # NOTE: documents the same pre-existing bug as
        # POVMEffectUtilTester.test_hessian_wrt_params_zero_param_path_is_broken, but reached
        # here via has_nonzero_hessian() == False (rather than num_params == 0): the base
        # State.hessian_wrt_params() also tries to build `_np.zeros(self.size, ...)`, which
        # raises AttributeError since neither State nor POVMEffect define a `size` attribute.
        with self.assertRaises(AttributeError):
            self.vec.hessian_wrt_params()

    def test_to_transformed_dense(self):
        identity = np.identity(4, 'd')
        transformed = self.vec._to_transformed_dense(identity, identity)
        self.assertArraysAlmostEqual(transformed, self.vec.to_dense())


class StaticPOVMEffectTester(POVMEffectBase, BaseCase):
    n_params = 0

    @staticmethod
    def build_vec():
        v = np.array([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], 'd')
        return povms.StaticPOVMEffect(v, None, 'default', state_space=None)

    def test_raises_on_set_value(self):
        v = self.vec.to_dense()
        with self.assertRaises(ValueError):
            self.vec.set_dense(v)

    def test_has_no_depolarize_method(self):
        # Unlike FullPOVMEffect, StaticPOVMEffect doesn't define depolarize (they share
        # __init__/set_dense via the common _StateWrappingEffectTemplate, but depolarize is
        # only added to the "Full" variants). Confirm the merge didn't leak a depolarize
        # method onto the static variant.
        self.assertFalse(hasattr(self.vec, 'depolarize'))


class FullPOVMEffectTester(POVMEffectBase, BaseCase):
    """Direct coverage of FullPOVMEffect -- previously this class was only ever exercised
    indirectly as a building block within other testers (e.g. ComplementPOVMEffectTester,
    UnconstrainedPOVMTester), despite being one of the most commonly-used effect classes."""
    n_params = 4

    @staticmethod
    def build_vec():
        v = np.array([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], 'd')
        return povms.FullPOVMEffect(v, None, 'default', state_space=None)

    def test_raises_on_bad_dimension(self):
        with self.assertRaises(ValueError):
            povms.FullPOVMEffect([[1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], [0, 0, 0, 0]])

    def test_depolarize(self):
        dense0 = self.vec.to_dense().copy()
        amount = 0.9
        self.vec.depolarize(amount)
        dense1 = self.vec.to_dense()
        # the first ("identity") component is left alone; all others are scaled by (1 - amount)
        self.assertAlmostEqual(dense1[0], dense0[0])
        self.assertArraysAlmostEqual(dense1[1:], (1 - amount) * dense0[1:])
        self.assertTrue(self.vec.dirty)

        self.vec.depolarize([0.9, 0.8, 0.7])
        # just check that a per-component amount also runs without error


class StaticPOVMPureEffectTester(POVMEffectBase, BaseCase):
    n_params = 0

    @staticmethod
    def build_vec():
        v = np.array([1.0, 0], complex)
        return povms.StaticPOVMPureEffect(v, 'pp', 'default', state_space=None)

    def test_set_value_raises_on_bad_size(self):
        with self.assertRaises(ValueError):
            self.vec.set_dense(np.zeros((1, 1), 'd'))

    def test_has_no_depolarize_method(self):
        self.assertFalse(hasattr(self.vec, 'depolarize'))


class FullPOVMPureEffectTester(POVMEffectBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        v = np.array([1.0, 0], complex)
        return povms.FullPOVMPureEffect(v, 'pp', 'default', state_space=None)

    def test_vector_conversion(self):
        # deriv_wrt_params doesn't currently work for FullPureState-based effects because of
        # dim/udim weirdness (see TensorProductStateTester in test_spamvec.py for the analogous
        # state case).
        v = self.vec.to_vector()
        self.vec.from_vector(v)
        self.assertArraysAlmostEqual(self.vec.to_vector(), v)

    def test_has_no_depolarize_method(self):
        # NOTE: unlike FullPOVMEffect, FullPOVMPureEffect does *not* define a depolarize
        # method (this asymmetry predates the introduction of _StateWrappingEffectTemplate;
        # it is not something that template merge introduced -- just a pre-existing gap
        # between the "pure" and "dense" Full effect variants worth documenting here).
        self.assertFalse(hasattr(self.vec, 'depolarize'))


class ComposedPOVMEffectTester(POVMEffectBase, BaseCase):
    n_params = 16

    @staticmethod
    def build_vec():
        base_effect = povms.StaticPOVMEffect(np.array([1.0, 0, 0, 1.0], 'd') / np.sqrt(2))
        errormap = operations.FullArbitraryOp(np.identity(4, 'd'))
        return povms.ComposedPOVMEffect(base_effect, errormap)

    def test_pickle(self):
        # ComposedPOVMEffect references a shared error-map object; pickling/unpickling it directly
        # (outside of a parent POVM/Model) isn't a supported use case.
        self.skipTest("ComposedPOVMEffect is not picklable in isolation from its parent POVM.")


class ComplementPOVMEffectTester(POVMEffectBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        v = np.ones((4, 1), 'd')
        v_id = np.zeros((4, 1), 'd')
        v_id[0] = 1.0 / np.sqrt(2)
        evotype = "default"
        basis = None
        tppovm = TPPOVM([('0', povms.FullPOVMEffect(v, basis, evotype, state_space=None)),
                         ('1', povms.FullPOVMEffect(v_id - v, basis, evotype, state_space=None))])
        return tppovm['1']  # complement POVM

    def test_vector_conversion(self):
        with self.assertRaises(RuntimeError):
            self.vec.to_vector()

    def test_has_nonzero_hessian(self):
        self.assertFalse(self.vec.has_nonzero_hessian())

    def test_submembers(self):
        # submembers() should be the "other" effects (not the internal static identity state)
        self.assertEqual(self.vec.submembers(), self.vec.other_effects)

    def test_is_similar(self):
        v = np.ones((4, 1), 'd')
        v_id = np.zeros((4, 1), 'd')
        v_id[0] = 1.0 / np.sqrt(2)
        same_identity = TPPOVM([('0', povms.FullPOVMEffect(v.copy())),
                               ('1', povms.FullPOVMEffect((v_id - v).copy()))])['1']
        self.assertTrue(self.vec._is_similar(same_identity, rtol=1e-6, atol=1e-9))

        v_id2 = np.zeros((4, 1), 'd')
        v_id2[0] = 2.0 / np.sqrt(2)  # different identity vector
        different_identity = TPPOVM([('0', povms.FullPOVMEffect(v.copy())),
                                     ('1', povms.FullPOVMEffect((v_id2 - v).copy()))])['1']
        self.assertFalse(self.vec._is_similar(different_identity, rtol=1e-6, atol=1e-9))

    def test_deriv_wrt_params_correctness(self):
        # This TPPOVM has exactly one "other" effect ('0'), so the complement effect is
        # identity - effect0, and its derivative wrt effect0's parameters should be exactly
        # -1 * (effect0's own derivative wrt its parameters).
        other_effect = self.vec.other_effects[0]
        expected = -other_effect.deriv_wrt_params()
        self.assertArraysAlmostEqual(self.vec.deriv_wrt_params(), expected)


class TensorProductPOVMEffectTester(TensorProdStateBase, POVMEffectBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        v = np.ones((4, 1), 'd')
        povm = UnconstrainedPOVM([('0', povms.FullPOVMEffect(v, evotype='default', state_space=None))])
        return povms.TensorProductPOVMEffect([povm], ['0'], state_space=(0,))

    def test_vector_conversion(self):  # because we inherit some state methods from TensorProdStateBase
        with self.assertRaises(ValueError):  # don't call to_vector on effects - call it on the POVM
            self.vec.to_vector()

import pickle

import numpy as np

import pygsti.modelmembers.povms as povms
import pygsti.modelmembers.states as states
import pygsti.baseobjs.statespace as statespace
from pygsti.evotypes import Evotype
from pygsti.modelmembers.povms import TPPOVM, UnconstrainedPOVM
from pygsti.models import ExplicitOpModel
from pygsti.baseobjs import Basis
from pygsti.models.gaugegroup import FullGaugeGroupElement
from ..util import BaseCase


class StateUtilTester(BaseCase):
    def test_convert_to_vector_raises_on_bad_input(self):
        bad_vecs = [
            'akdjsfaksdf',
            [[], [1, 2]],
            [[[]], [[1, 2]]]
        ]
        for bad_vec in bad_vecs:
            with self.assertRaises(ValueError):
                states.State._to_vector(bad_vec)
        with self.assertRaises(ValueError):
            states.State._to_vector(0.0)  # something with no len()

    def test_base_state(self):

        state_space = statespace.default_space_for_dim(4)
        evotype = Evotype.cast('default')
        rep = evotype.create_dense_state_rep(np.zeros(4, 'd'), state_space)
        raw = states.State(rep, evotype)

        T = FullGaugeGroupElement(
            np.array([[0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], 'd'))

        with self.assertRaises(NotImplementedError):
            raw.to_dense()
        with self.assertRaises(NotImplementedError):
            raw.transform_inplace(T)
        with self.assertRaises(NotImplementedError):
            raw.depolarize(0.01)


class StateBase(object):
    def setUp(self):
        self.vec = self.build_vec()
        ExplicitOpModel._strict = False

    def test_num_params(self):
        self.assertEqual(self.vec.num_params, self.n_params)

    def test_get_dimension(self):
        self.assertEqual(self.vec.dim, 4)

    def test_set_value_raises_on_bad_size(self):
        with self.assertRaises(ValueError):
            self.vec.set_dense(np.zeros((1, 1), 'd'))  # bad size

    def test_vector_conversion(self):
        v = self.vec.to_vector()
        self.vec.from_vector(v)
        deriv = self.vec.deriv_wrt_params()
        # TODO assert correctness

    def test_pickle(self):
        pklstr = pickle.dumps(self.vec)
        vec_pickle = pickle.loads(pklstr)
        self.assertArraysAlmostEqual(vec_pickle.to_dense(), self.vec.to_dense())
        self.assertEqual(type(vec_pickle), type(self.vec))

    def test_hessian(self):
        self.assertFalse(self.vec.has_nonzero_hessian())

    def test_frobeniusdist2(self):
        self.vec.frobeniusdist_squared(self.vec)
        self.vec.frobeniusdist_squared(self.vec)
        # TODO assert correctness


class DenseStateBase(StateBase):

    def test_vector_conversion(self):
        v = self.vec.to_vector()
        self.vec.from_vector(v)
        deriv = self.vec.deriv_wrt_params()
        # TODO assert correctness
    
    def test_element_accessors(self):
        a = self.vec[:]
        b = self.vec[0]
        #with self.assertRaises(ValueError):
        #    self.vec.shape = (2,2) #something that would affect the shape??

        self.vec_as_str = str(self.vec)
        a1 = self.vec[:]  # invoke getslice method
        # TODO assert correctness

    def test_copy(self):
        vec_copy = self.vec.copy()
        self.assertArraysAlmostEqual(vec_copy.to_dense(), self.vec.to_dense())
        self.assertEqual(type(vec_copy), type(self.vec))

    def test_arithmetic(self):
        result = self.vec + self.vec
        self.assertEqual(type(result), np.ndarray)
        result = self.vec + (-self.vec)
        self.assertEqual(type(result), np.ndarray)
        result = self.vec - self.vec
        self.assertEqual(type(result), np.ndarray)
        result = self.vec - abs(self.vec)
        self.assertEqual(type(result), np.ndarray)
        result = 2 * self.vec
        self.assertEqual(type(result), np.ndarray)
        result = self.vec * 2
        self.assertEqual(type(result), np.ndarray)
        result = 2 / self.vec
        self.assertEqual(type(result), np.ndarray)
        result = self.vec / 2
        self.assertEqual(type(result), np.ndarray)
        result = self.vec // 2
        self.assertEqual(type(result), np.ndarray)
        result = self.vec**2
        self.assertEqual(type(result), np.ndarray)
        result = self.vec.transpose()
        self.assertEqual(type(result), np.ndarray)

        V = np.ones((4, 1), 'd')

        result = self.vec + V
        self.assertEqual(type(result), np.ndarray)
        result = self.vec - V
        self.assertEqual(type(result), np.ndarray)
        result = V + self.vec
        self.assertEqual(type(result), np.ndarray)
        result = V - self.vec
        self.assertEqual(type(result), np.ndarray)


class MutableDenseStateBase(DenseStateBase):
    def test_set_value(self):
        v = np.asarray(self.vec)
        self.vec.set_dense(v)
        # TODO assert correctness

    def test_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        self.vec.transform_inplace(S)
        # TODO assert correctness

    def test_depolarize(self):
        self.vec.depolarize(0.9)
        self.vec.depolarize([0.9, 0.8, 0.7])
        # TODO assert correctness


class ImmutableDenseStateBase(DenseStateBase):
    def test_raises_on_set_value(self):
        v = np.asarray(self.vec)
        with self.assertRaises(ValueError):
            self.vec.set_dense(v)

    def test_raises_on_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.vec.transform_inplace(S)

    def test_raises_on_depolarize(self):
        with self.assertRaises(ValueError):
            self.vec.depolarize(0.9)


class FullStateTester(MutableDenseStateBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        return states.FullState([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)],
                                'default', state_space=None)

    def test_raises_on_bad_dimension_2(self):
        with self.assertRaises(ValueError):
            states.FullState([[1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], [0, 0, 0, 0]])

    def test_convert(self):
        basis = Basis.cast("pp", 4)
        conv = states.convert(self.vec, "full", basis)
        # TODO assert correctness

    def test_raises_on_invalid_conversion_type(self):
        basis = Basis.cast("pp", 4)
        with self.assertRaises(ValueError):
            states.convert(self.vec, "foobar", basis)


class TPStateTester(MutableDenseStateBase, BaseCase):
    n_params = 3

    @staticmethod
    def build_vec():
        return states.TPState([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], evotype='default', state_space=None)

    def test_raises_on_bad_initial_element(self):
        with self.assertRaises(ValueError):
            states.TPState([1.0, 0, 0, 0])
            # incorrect initial element for TP!
        with self.assertRaises(ValueError):
            self.vec.set_dense([1.0, 0, 0, 0])
            # incorrect initial element for TP!

    def test_convert(self):
        basis = Basis.cast("pp", 4)
        conv = states.convert(self.vec, "full TP", basis)
        # TODO assert correctness


class CPTPStateTester(MutableDenseStateBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        v_tp = np.zeros((4, 1), 'd')
        v_tp[0] = 1.0 / np.sqrt(2)
        v_tp[3] = 1.0 / np.sqrt(2) - 0.05
        return states.CPTPState(v_tp, "pp", truncate=False, evotype='default', state_space=None)

    def test_hessian(self):
        self.skipTest("Hessian computation isn't implemented for CPTPSPAMVec; remove this skip when it becomes a priority")
        self.vec.hessian_wrt_params()
        self.vec.hessian_wrt_params([0])
        self.vec.hessian_wrt_params([0], [0])
        # TODO assert correctness


class StaticStateTester(ImmutableDenseStateBase, BaseCase):
    n_params = 0
    v_tp = [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)]

    @staticmethod
    def build_vec():
        return states.StaticState(StaticStateTester.v_tp, evotype='default', state_space=None)

    def test_convert(self):
        basis = Basis.cast("pp", 4)
        conv = states.convert(self.vec, "static", basis)
        # TODO assert correctness

    def test_optimize(self):
        s = states.FullState(StaticStateTester.v_tp)
        states.optimize_state(self.vec, s)
        # TODO assert correctness


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
        deriv = self.vec.deriv_wrt_params()
        # TODO assert correctness

    def test_pickle(self):
        pklstr = pickle.dumps(self.vec)
        vec_pickle = pickle.loads(pklstr)
        self.assertArraysAlmostEqual(vec_pickle.to_dense(), self.vec.to_dense())
        self.assertEqual(type(vec_pickle), type(self.vec))


class ComplementPOVMEffectTester(POVMEffectBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        v = np.ones((4, 1), 'd')
        v_id = np.zeros((4, 1), 'd')
        v_id[0] = 1.0 / np.sqrt(2)
        evotype = "default"
        tppovm = TPPOVM([('0', povms.FullPOVMEffect(v, evotype, state_space=None)),
                         ('1', povms.FullPOVMEffect(v_id - v, evotype, state_space=None))])
        return tppovm['1']  # complement POVM

    def test_vector_conversion(self):
        with self.assertRaises(ValueError):
            self.vec.to_vector()


class TensorProdStateBase(StateBase):

    def test_copy(self):
        vec_copy = self.vec.copy()
        self.assertArraysAlmostEqual(vec_copy.to_dense(), self.vec.to_dense())
        self.assertEqual(type(vec_copy), type(self.vec))

    def test_pickle(self):
        pklstr = pickle.dumps(self.vec)
        vec_pickle = pickle.loads(pklstr)
        self.assertArraysAlmostEqual(vec_pickle.to_dense(), self.vec.to_dense())
        self.assertEqual(type(vec_pickle), type(self.vec))


class TensorProductStateTester(TensorProdStateBase, BaseCase):
    n_params = 8

    @staticmethod
    def build_vec():
        v = np.ones((2, 1), complex)
        evotype = "default"
        return states.TensorProductState([states.FullPureState(v, 'pp', evotype, state_space=None),
                                          states.FullPureState(v, 'pp', evotype, state_space=None)],
                                         state_space=(0, 1))

    def test_get_dimension(self):
        self.assertEqual(self.vec.dim, 16)  # 2-qubits!

    def test_vector_conversion(self):
        v = self.vec.to_vector()
        self.vec.from_vector(v)
        #deriv = self.vec.deriv_wrt_params()  # this doesn't work with FullPureState yet because of dim/udim weirdness
        # Once this is fixed, we should be able to remove this method and just use the base class


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

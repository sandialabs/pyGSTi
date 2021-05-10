import numpy as np
import pickle

from ..util import BaseCase

from pygsti.objects import FullGaugeGroupElement, Basis, ExplicitOpModel, TPPOVM, UnconstrainedPOVM
import pygsti.construction as pc
import pygsti.objects.spamvec as sv
import pygsti.objects.operation as op


class SpamvecUtilTester(BaseCase):
    def test_convert_to_vector_raises_on_bad_input(self):
        bad_vecs = [
            'akdjsfaksdf',
            [[], [1, 2]],
            [[[]], [[1, 2]]]
        ]
        for bad_vec in bad_vecs:
            with self.assertRaises(ValueError):
                sv.SPAMVec._to_vector(bad_vec)
        with self.assertRaises(ValueError):
            sv.SPAMVec._to_vector(0.0)  # something with no len()

    def test_base_spamvec(self):
        raw = sv.SPAMVec(4, "densitymx", "prep")

        T = FullGaugeGroupElement(
            np.array([[0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], 'd'))

        with self.assertRaises(NotImplementedError):
            raw.to_dense()
        with self.assertRaises(NotImplementedError):
            raw.transform_inplace(T, "prep")
        with self.assertRaises(NotImplementedError):
            raw.depolarize(0.01)


class SpamvecBase(object):
    def setUp(self):
        self.vec = self.build_vec()
        ExplicitOpModel._strict = False

    def test_num_params(self):
        self.assertEqual(self.vec.num_params, self.n_params)

    def test_copy(self):
        vec_copy = self.vec.copy()
        self.assertArraysAlmostEqual(vec_copy, self.vec)
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

    def test_element_accessors(self):
        a = self.vec[:]
        b = self.vec[0]
        #with self.assertRaises(ValueError):
        #    self.vec.shape = (2,2) #something that would affect the shape??

        self.vec_as_str = str(self.vec)
        a1 = self.vec[:]  # invoke getslice method
        # TODO assert correctness

    def test_pickle(self):
        pklstr = pickle.dumps(self.vec)
        vec_pickle = pickle.loads(pklstr)
        self.assertArraysAlmostEqual(vec_pickle, self.vec)
        self.assertEqual(type(vec_pickle), type(self.vec))

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

    def test_hessian(self):
        self.assertFalse(self.vec.has_nonzero_hessian())

    def test_frobeniusdist2(self):
        self.vec.frobeniusdist_squared(self.vec, "prep")
        self.vec.frobeniusdist_squared(self.vec, "effect")
        # TODO assert correctness

    def test_frobeniusdist2_raises_on_bad_type(self):
        with self.assertRaises(ValueError):
            self.vec.frobeniusdist_squared(self.vec, "foobar")


class MutableSpamvecBase(SpamvecBase):
    def test_set_value(self):
        v = np.asarray(self.vec)
        self.vec.set_dense(v)
        # TODO assert correctness

    def test_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        self.vec.transform_inplace(S, 'prep')
        self.vec.transform_inplace(S, 'effect')
        # TODO assert correctness

    def test_transform_raises_on_bad_type(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.vec.transform_inplace(S, 'foobar')

    def test_depolarize(self):
        self.vec.depolarize(0.9)
        self.vec.depolarize([0.9, 0.8, 0.7])
        # TODO assert correctness


class ImmutableSpamvecBase(SpamvecBase):
    def test_raises_on_set_value(self):
        v = np.asarray(self.vec)
        with self.assertRaises(ValueError):
            self.vec.set_dense(v)

    def test_raises_on_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.vec.transform_inplace(S, 'prep')

    def test_raises_on_depolarize(self):
        with self.assertRaises(ValueError):
            self.vec.depolarize(0.9)


class FullSpamvecTester(MutableSpamvecBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        return sv.FullSPAMVec([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)])

    def test_raises_on_bad_dimension_2(self):
        with self.assertRaises(ValueError):
            sv.FullSPAMVec([[1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], [0, 0, 0, 0]])

    def test_convert(self):
        basis = Basis.cast("pp", 4)
        conv = sv.convert(self.vec, "full", basis)
        # TODO assert correctness

    def test_raises_on_invalid_conversion_type(self):
        basis = Basis.cast("pp", 4)
        with self.assertRaises(ValueError):
            sv.convert(self.vec, "foobar", basis)


class TPSpamvecTester(MutableSpamvecBase, BaseCase):
    n_params = 3

    @staticmethod
    def build_vec():
        return sv.TPSPAMVec([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)])

    def test_raises_on_bad_initial_element(self):
        with self.assertRaises(ValueError):
            sv.TPSPAMVec([1.0, 0, 0, 0])
            # incorrect initial element for TP!
        with self.assertRaises(ValueError):
            self.vec.set_dense([1.0, 0, 0, 0])
            # incorrect initial element for TP!

    def test_convert(self):
        basis = Basis.cast("pp", 4)
        conv = sv.convert(self.vec, "TP", basis)
        # TODO assert correctness


class CPTPSpamvecTester(MutableSpamvecBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        v_tp = np.zeros((4, 1), 'd')
        v_tp[0] = 1.0 / np.sqrt(2)
        v_tp[3] = 1.0 / np.sqrt(2) - 0.05
        return sv.CPTPSPAMVec(v_tp, "pp")

    def test_hessian(self):
        self.skipTest("Hessian computation isn't implemented for CPTPSPAMVec; remove this skip when it becomes a priority")
        self.vec.hessian_wrt_params()
        self.vec.hessian_wrt_params([0])
        self.vec.hessian_wrt_params([0], [0])
        # TODO assert correctness


class StaticSpamvecTester(ImmutableSpamvecBase, BaseCase):
    n_params = 0
    v_tp = [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)]

    @staticmethod
    def build_vec():
        return sv.StaticSPAMVec(StaticSpamvecTester.v_tp)

    def test_convert(self):
        basis = Basis.cast("pp", 4)
        conv = sv.convert(self.vec, "static", basis)
        # TODO assert correctness

    def test_optimize(self):
        s = sv.FullSPAMVec(StaticSpamvecTester.v_tp)
        sv.optimize_spamvec(self.vec, s)
        # TODO assert correctness


class POVMSpamvecBase(ImmutableSpamvecBase):
    def test_vector_conversion(self):
        with self.assertRaises(ValueError):
            self.vec.to_vector()


class ComplementSpamvecTester(POVMSpamvecBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        v = np.ones((4, 1), 'd')
        v_id = np.zeros((4, 1), 'd')
        v_id[0] = 1.0 / np.sqrt(2)
        tppovm = TPPOVM([('0', sv.FullSPAMVec(v, typ="effect")),
                         ('1', sv.FullSPAMVec(v_id - v, typ="effect"))])
        return tppovm['1']  # complement POVM

    def test_vector_conversion(self):
        with self.assertRaises(ValueError):
            self.vec.to_vector()


class TensorProdSpamvecBase(ImmutableSpamvecBase):
    def test_arithmetic(self):
        with self.assertRaises(TypeError):
            self.vec + self.vec

    def test_copy(self):
        vec_copy = self.vec.copy()
        self.assertArraysAlmostEqual(vec_copy.to_dense(), self.vec.to_dense())
        self.assertEqual(type(vec_copy), type(self.vec))

    def test_element_accessors(self):
        with self.assertRaises(TypeError):
            self.vec[:]

    def test_pickle(self):
        pklstr = pickle.dumps(self.vec)
        vec_pickle = pickle.loads(pklstr)
        self.assertArraysAlmostEqual(vec_pickle.to_dense(), self.vec.to_dense())
        self.assertEqual(type(vec_pickle), type(self.vec))


class TensorProdPrepSpamvecTester(TensorProdSpamvecBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        v = np.ones((2, 1), 'd')
        return sv.TensorProdSPAMVec("prep", [sv.FullSPAMVec(v),
                                             sv.FullSPAMVec(v)])


class TensorProdEffectSpamvecTester(TensorProdSpamvecBase, POVMSpamvecBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_vec():
        v = np.ones((4, 1), 'd')
        povm = UnconstrainedPOVM([('0', sv.FullSPAMVec(v,typ="effect"))])
        return sv.TensorProdSPAMVec("effect", [povm], ['0'])

# For NoisySPAMVec, mutability can be either in noise op or SPAM vec
class MutableNoisyStaticSpamvecBase(SpamvecBase):
    # Almost MutableSpamvecBase, but no set_dense for StaticSPAMVec
    # transform_inplace is OK because allowed by StaticSPAMVec
    # depolarize is OK because it acts on the Mutable noisy op
    def test_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        self.vec.transform_inplace(S, 'prep')
        self.vec.transform_inplace(S, 'effect')
        # TODO assert correctness

    def test_transform_raises_on_bad_type(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.vec.transform_inplace(S, 'foobar')

    def test_depolarize(self):
        self.vec.depolarize(0.9)
        self.vec.depolarize([0.9, 0.8, 0.7])
        # TODO assert correctness

class FullDenseNoisyStaticSpamvecTester(MutableNoisyStaticSpamvecBase, BaseCase):
    n_params = 16

    @staticmethod
    def build_vec():
        full_depol = op.FullDenseOp(np.eye(4))
        full_depol.depolarize(0.1)
        return sv.NoisySPAMVec([1, 0, 0, 0], full_depol, 'prep')


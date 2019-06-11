import numpy as np
import pickle

from ..util import BaseCase

from pygsti.objects import FullGaugeGroupElement, Basis, ExplicitOpModel
import pygsti.construction as pc
import pygsti.objects.spamvec as sv


class SpamvecBase:
    def setUp(self):
        self.vec = self.build_vec()
        ExplicitOpModel._strict = False

    def test_num_params(self):
        self.assertEqual(self.vec.num_params(), self.n_params)

    def test_copy(self):
        vec_copy = self.vec.copy()
        self.assertArraysAlmostEqual(vec_copy, self.vec)
        self.assertEqual(type(vec_copy), type(self.vec))

    def test_get_dimension(self):
        self.assertEqual(self.vec.get_dimension(), 4)

    def test_set_value_raises_on_bad_size(self):
        with self.assertRaises(ValueError):
            self.vec.set_value(np.zeros((1, 1), 'd'))  # bad size

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


class MutableSpamvecBase(SpamvecBase):
    def test_set_value(self):
        v = np.asarray(self.vec)
        self.vec.set_value(v)
        # TODO assert correctness

    def test_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        self.vec.transform(S, 'prep')
        self.vec.transform(S, 'effect')
        # TODO assert correctness

    def test_transform_raises_on_bad_type(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.vec.transform(S, 'foobar')

    def test_depolarize(self):
        self.vec.depolarize(0.9)
        self.vec.depolarize([0.9, 0.8, 0.7])
        # TODO assert correctness


class ImmutableSpamvecBase(SpamvecBase):
    def test_raises_on_set_value(self):
        v = np.asarray(self.vec)
        with self.assertRaises(ValueError):
            self.vec.set_value(v)

    def test_raises_on_transform(self):
        S = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises(ValueError):
            self.vec.transform(S, 'prep')

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
            self.vec.set_value([1.0, 0, 0, 0])
            # incorrect initial element for TP!

    def test_convert(self):
        basis = Basis.cast("pp", 4)
        conv = sv.convert(self.vec, "TP", basis)
        # TODO assert correctness


class StaticSpamvecTester(ImmutableSpamvecBase, BaseCase):
    n_params = 0

    @staticmethod
    def build_vec():
        return sv.StaticSPAMVec([1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)])

    def test_convert(self):
        basis = Basis.cast("pp", 4)
        conv = sv.convert(self.vec, "static", basis)
        # TODO assert correctness

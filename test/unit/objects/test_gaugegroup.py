import numpy as np

from pygsti.modelmembers import operations as op
from pygsti.models import gaugegroup as ggrp
from pygsti.baseobjs.statespace import QubitSpace, QuditSpace, ExplicitStateSpace
from pygsti.baseobjs.basis import Basis, TensorProdBasis
from pygsti.tools.optools import unitary_to_superop, superop_to_unitary
from ..util import BaseCase


class GaugeGroupBase(object):

    HAS_DERIV_WRT_PARAMS = True

    def setUp(self):
        self.state_space = QubitSpace(1)
        self.rng = np.random.default_rng(0)
    
    def test_construction(self):
        params = self.gg.initial_params
        self.assertEqual(len(params), self.n_params)
        self.assertEqual(self.gg.num_params, self.n_params)
        element = self.gg.compute_element(params)
        # XXX is this necessary?  EGN: maybe not, but it asserts correctness and should be fast
        self.assertIsInstance(element, self.element_type)

    def test_element_construction(self):
        el = self.gg.compute_element(self.gg.initial_params)
        self.assertEqual(el.num_params, self.n_params)

    def test_element_get_transform_matrix(self):
        el = self.gg.compute_element(self.gg.initial_params)
        mx = el.transform_matrix
        # TODO assert correctness

    def test_element_get_transform_matrix_inverse(self):
        el = self.gg.compute_element(self.gg.initial_params)
        mx = el.transform_matrix
        inv = el.transform_matrix_inverse
        self.assertArraysAlmostEqual(np.linalg.inv(mx), inv)

    def test_element_deriv_wrt_params(self):
        if self.HAS_DERIV_WRT_PARAMS:
            el = self.gg.compute_element(self.gg.initial_params)
            deriv = el.deriv_wrt_params()
            # TODO assert correctness

    def test_element_to_from_vector(self):
        el = self.gg.compute_element(self.gg.initial_params)
        v0 = el.to_vector().copy()
        m0 = el.transform_matrix.copy()
        num_params = v0.size
        if num_params > 0:
            v1 = self.rng.random(size=(num_params,))
            el.from_vector(v1)
            m1 = el.transform_matrix.copy()
            self.assertGreater(np.linalg.norm(m1 - m0), 0.0)
            el.from_vector(v0)
            m2 = el.transform_matrix.copy()
            self.assertArraysAlmostEqual(m0, m2)
        else:
            # we just check that from_vector raises no error when provided 
            # with a vector of length zero.
            el.from_vector(v0)
        return


class GaugeGroupTester(GaugeGroupBase, BaseCase):
    # XXX do we need coverage of an abstract base class?
    # XXX should this class even be instantiatable?  EGN: no, it's just a base class.
    n_params = 0
    element_type = ggrp.GaugeGroupElement

    def setUp(self):
        self.gg = ggrp.GaugeGroup('myGaugeGroupName')

    def test_element_get_transform_matrix_inverse(self):
        el = self.gg.compute_element(self.gg.initial_params)
        inv = el.transform_matrix_inverse
        self.assertIsNone(inv)

    def test_element_to_from_vector(self):
        pass  # abstract


class OpGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 16
    element_type = ggrp.OpGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.OpGaugeGroup(op.FullArbitraryOp(np.identity(4, 'd'), state_space=self.state_space),
                                    ggrp.OpGaugeGroupElement, 'myGateGaugeGroupName')


class FullGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 16
    element_type = ggrp.FullGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.FullGaugeGroup(self.state_space)


class TPGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 12
    element_type = ggrp.TPGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.TPGaugeGroup(self.state_space)


class DiagGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 4
    element_type = ggrp.DiagGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.DiagGaugeGroup(self.state_space)


class TPDiagGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 3
    element_type = ggrp.TPDiagGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.TPDiagGaugeGroup(self.state_space)


class SpamGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 2
    element_type = ggrp.SpamGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.SpamGaugeGroup(self.state_space)


class TrivialGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 0
    element_type = ggrp.TrivialGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.TrivialGaugeGroup(self.state_space)


class DirectSumGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 3
    element_type = ggrp.DirectSumUnitaryGroupElement
    HAS_DERIV_WRT_PARAMS = False

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.state_space = ExplicitStateSpace(['dummy'],[5])
        g1 = ggrp.TrivialGaugeGroup(ExplicitStateSpace(['T0']))
        g2 = ggrp.UnitaryGaugeGroup(QubitSpace(1), 'pp')
        self.gg = ggrp.DirectSumUnitaryGroup((g1, g2), 'std')


class U1GroupTester(GaugeGroupBase, BaseCase):
    n_params = 1
    element_type = ggrp.U1GroupElement
    HAS_DERIV_WRT_PARAMS = False

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.gg = ggrp.U1Group()

    def test_identity_transform(self):
        el = self.gg.compute_element(np.array([0.0]))
        self.assertArraysAlmostEqual(el.transform_matrix, np.array([[1.0 + 0.0j]]))

    def test_transform_matrix_is_unitary(self):
        el = self.gg.compute_element(np.array([1.2]))
        M = el.transform_matrix
        self.assertArraysAlmostEqual(M @ M.conj().T, np.eye(1, dtype=complex))

    def test_angle_wrapping(self):
        angle = 0.5
        el = self.gg.compute_element(self.gg.initial_params)
        el.from_vector(np.array([angle]))
        mx1 = el.transform_matrix.copy()
        el.from_vector(np.array([angle + 2 * np.pi]))
        mx2 = el.transform_matrix.copy()
        self.assertArraysAlmostEqual(mx1, mx2)

    def test_inverse_gives_identity(self):
        el = self.gg.compute_element(np.array([0.7]))
        product = el.transform_matrix @ el.inverse().transform_matrix
        self.assertArraysAlmostEqual(product, np.eye(1, dtype=complex))


class TensorProductGaugeGroupTester(GaugeGroupBase, BaseCase):
    n_params = 6
    element_type = ggrp.TensorProductGaugeGroupElement

    def setUp(self):
        GaugeGroupBase.setUp(self)
        self.state_space = QubitSpace(2)
        self.factor = ggrp.UnitaryGaugeGroup(QubitSpace(1), 'pp')
        self.gg = ggrp.TensorProductGaugeGroup([self.factor, self.factor], self.state_space, 'pp')
        self.v = np.array([0.3, -0.2, 0.5, 0.1, 0.7, -0.4])

    def test_pp_on_qubits_needs_no_change_of_basis(self):
        self.assertIsNone(self.gg._change_of_basis)

    def test_element_is_kronecker_product_of_factors(self):
        el = self.gg.compute_element(self.v)
        S1 = self.factor.compute_element(self.v[:3]).transform_matrix
        S2 = self.factor.compute_element(self.v[3:]).transform_matrix
        self.assertArraysAlmostEqual(el.transform_matrix, np.kron(S1, S2))
        self.assertArraysAlmostEqual(el.transform_matrix_inverse, np.kron(np.linalg.inv(S1), np.linalg.inv(S2)))
        self.assertEqual(el.transform_matrix.dtype, np.dtype('d'))

    def test_element_matches_full_space_unitary(self):
        el = self.gg.compute_element(self.v)
        U1, U2 = (superop_to_unitary(m, 'pp') for m in el.factor_matrices)
        self.assertArraysAlmostEqual(el.transform_matrix, unitary_to_superop(np.kron(U1, U2), 'pp'))

    def test_tensor_product_model_basis(self):
        # Models built by pyGSTi carry a TensorProdBasis ('pp*pp') rather than the builtin pp of dim 16.
        tpb = TensorProdBasis([Basis.cast('pp', 4)] * 2)
        gg = ggrp.TensorProductGaugeGroup([self.factor, self.factor], self.state_space, tpb)
        self.assertIsNone(gg._change_of_basis)
        self.assertArraysAlmostEqual(gg.compute_element(self.v).transform_matrix,
                                     self.gg.compute_element(self.v).transform_matrix)

    def test_mixed_factor_types(self):
        tp = ggrp.TPGaugeGroup(QubitSpace(1), 'pp')
        triv = ggrp.TrivialGaugeGroup(QubitSpace(1))
        gg = ggrp.TensorProductGaugeGroup([tp, triv], self.state_space, 'pp')
        self.assertEqual(gg.num_params, tp.num_params)
        v = self.rng.random(gg.num_params)
        el = gg.compute_element(v)
        self.assertArraysAlmostEqual(el.transform_matrix, np.kron(tp.compute_element(v).transform_matrix, np.eye(4)))
        self.assertArraysAlmostEqual(el.transform_matrix_inverse @ el.transform_matrix, np.eye(16))

    def test_multi_label_factor(self):
        two_q = ggrp.UnitaryGaugeGroup(QubitSpace(2), 'pp')
        gg = ggrp.TensorProductGaugeGroup([two_q, self.factor], QubitSpace(3), 'pp')
        self.assertEqual(gg.num_params, 15 + 3)
        self.assertEqual(gg._label_runs, ((0, 1), (2,)))
        el = gg.compute_element(gg.initial_params)
        self.assertArraysAlmostEqual(el.transform_matrix, np.eye(64))

    def test_constructor_rejects_bad_factor_layouts(self):
        with self.assertRaises(ValueError):  # too few factors
            ggrp.TensorProductGaugeGroup([self.factor], self.state_space, 'pp')
        with self.assertRaises(ValueError):  # too many factors
            ggrp.TensorProductGaugeGroup([self.factor] * 3, self.state_space, 'pp')
        with self.assertRaises(ValueError):  # dimension mismatch
            ggrp.TensorProductGaugeGroup([self.factor, ggrp.UnitaryGaugeGroup(QuditSpace(1, 3), 'gm')],
                                         self.state_space, 'pp')
        with self.assertRaises(ValueError):  # multi-block state space
            ss = ExplicitStateSpace([('Q0',), ('L',)], [(2,), (1,)])
            ggrp.TensorProductGaugeGroup([self.factor], ss, 'pp')
        with self.assertRaises(ValueError):  # basis / state space dimension mismatch
            ggrp.TensorProductGaugeGroup([self.factor, self.factor], self.state_space, Basis.cast('pp', 4))

    def test_inverse_element(self):
        el = self.gg.compute_element(self.v)
        inv = el.inverse()
        self.assertArraysAlmostEqual(inv.transform_matrix @ el.transform_matrix, np.eye(16))

    def _finite_difference_deriv(self, gg, v, h=1e-6):
        el = gg.compute_element(v)
        cols = []
        for i in range(gg.num_params):
            vp, vm = v.copy(), v.copy()
            vp[i] += h
            vm[i] -= h
            cols.append(((gg.compute_element(vp).transform_matrix
                          - gg.compute_element(vm).transform_matrix) / (2 * h)).reshape(-1))
        return el, np.column_stack(cols)

    def test_deriv_wrt_params_matches_finite_differences(self):
        el, fd = self._finite_difference_deriv(self.gg, self.v)
        deriv = el.deriv_wrt_params()
        self.assertEqual(deriv.shape, (16 * 16, 6))
        self.assertArraysAlmostEqual(deriv, fd, places=6)

    def test_deriv_wrt_params_honors_filter(self):
        el = self.gg.compute_element(self.v)
        full = el.deriv_wrt_params()
        flt = [4, 1, 3, 1]
        self.assertArraysAlmostEqual(el.deriv_wrt_params(flt), full[:, flt])
        self.assertEqual(el.deriv_wrt_params([]).shape, (256, 0))

    def test_deriv_wrt_params_with_change_of_basis(self):
        qutrit = ggrp.UnitaryGaugeGroup(QuditSpace(1, 3), 'gm')
        gg = ggrp.TensorProductGaugeGroup([qutrit, qutrit], QuditSpace(2, 3), Basis.cast('gm', 81))
        self.assertIsNotNone(gg._change_of_basis)
        v = 0.3 * self.rng.normal(size=gg.num_params)
        el, fd = self._finite_difference_deriv(gg, v)
        self.assertArraysAlmostEqual(el.deriv_wrt_params(), fd, places=6)
        # and the element itself agrees with the direct unitary computation in the builtin gm basis
        Ua, Ub = (superop_to_unitary(m, 'gm') for m in el.factor_matrices)
        self.assertArraysAlmostEqual(el.transform_matrix, unitary_to_superop(np.kron(Ua, Ub), Basis.cast('gm', 81)))

    def test_inverse_element_deriv(self):
        el = self.gg.compute_element(self.v)
        inv = el.inverse()
        S_inv, dS = el.transform_matrix_inverse, el.deriv_wrt_params()
        expected = np.column_stack([(-S_inv @ dS[:, i].reshape(16, 16) @ S_inv).reshape(-1) for i in range(6)])
        self.assertArraysAlmostEqual(inv.deriv_wrt_params(), expected)

    def test_local_unitary_constructor(self):
        gg = ggrp.TensorProductGaugeGroup.local_unitary(self.state_space, 'pp')
        self.assertEqual(gg.num_params, 6)
        self.assertTrue(all(isinstance(f, ggrp.UnitaryGaugeGroup) for f in gg.factors))
        self.assertEqual([f.state_space.tensor_product_blocks_labels[0] for f in gg.factors], [(0,), (1,)])
        self.assertIsNone(gg._change_of_basis)
        self.assertArraysAlmostEqual(gg.compute_element(self.v).transform_matrix,
                                     self.gg.compute_element(self.v).transform_matrix)
        # with the TensorProdBasis that pyGSTi-built models carry
        tpb = TensorProdBasis([Basis.cast('pp', 4)] * 2)
        gg2 = ggrp.TensorProductGaugeGroup.local_unitary(self.state_space, tpb)
        self.assertIsNone(gg2._change_of_basis)
        self.assertArraysAlmostEqual(gg2.compute_element(self.v).transform_matrix,
                                     self.gg.compute_element(self.v).transform_matrix)

    def test_local_tp_constructor(self):
        gg = ggrp.TensorProductGaugeGroup.local_tp(QubitSpace(3), 'pp')
        self.assertEqual(gg.num_params, 3 * 12)
        self.assertTrue(all(isinstance(f, ggrp.TPGaugeGroup) for f in gg.factors))
        el = gg.compute_element(self.rng.random(gg.num_params))
        S = el.transform_matrix
        self.assertArraysAlmostEqual(S[0, :], np.eye(64)[0, :])  # TP: first row is e_0
        self.assertArraysAlmostEqual(el.transform_matrix_inverse @ S, np.eye(64))

    def test_local_constructors_with_dimension_one_label(self):
        ss = ExplicitStateSpace(['Q0', 'L', 'Q1'], [2, 1, 3])
        gg = ggrp.TensorProductGaugeGroup.local_unitary(ss, 'gm')
        self.assertEqual([type(f) for f in gg.factors],
                         [ggrp.UnitaryGaugeGroup, ggrp.TrivialGaugeGroup, ggrp.UnitaryGaugeGroup])
        self.assertEqual(gg._label_runs, ((0,), (1,), (2,)))
        self.assertEqual(gg.num_params, 3 + 0 + 8)
        el = gg.compute_element(0.3 * self.rng.normal(size=gg.num_params))
        self.assertArraysAlmostEqual(el.transform_matrix_inverse @ el.transform_matrix, np.eye(36))

    def test_local_constructors_reject_multi_block_space(self):
        ss = ExplicitStateSpace([('Q0',), ('L',)], [(2,), (1,)])
        with self.assertRaises(ValueError):
            ggrp.TensorProductGaugeGroup.local_unitary(ss, 'pp')

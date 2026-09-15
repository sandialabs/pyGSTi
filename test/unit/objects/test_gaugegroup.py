import unittest.mock
import warnings

import numpy as np
import scipy.sparse as sps

from pygsti.modelmembers import operations as op
from pygsti.models import gaugegroup as ggrp
from pygsti.baseobjs.statespace import QubitSpace, QuditSpace, ExplicitStateSpace
from pygsti.baseobjs.basis import Basis, TensorProdBasis
from pygsti.tools.optools import unitary_to_superop, superop_to_unitary
from pygsti.tools.exceptions import DubiousTargetWarning
from pygsti.tools.matrixtools import IdentityOperator
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
        self.assertIsInstance(self.gg._change_of_basis, IdentityOperator)

    def test_tensor_product_builtin_bases_whitelist_is_accurate(self):
        # _TENSOR_PRODUCT_BUILTIN_BASES claims B(d1*d2) == B(d1) (x) B(d2).  Verify that claim
        # directly, so the constant can't drift away from what the bases actually do.
        for name, pairs, tensors in [('pp', [(4, 4), (4, 16), (16, 16)], True),
                                     ('std', [(4, 4), (9, 9)], False),
                                     ('gm', [(4, 4), (9, 9)], False)]:
            self.assertEqual(name in ggrp._TENSOR_PRODUCT_BUILTIN_BASES, tensors, msg=name)
            for d1, d2 in pairs:
                tpb = TensorProdBasis([Basis.cast(name, d1), Basis.cast(name, d2)])
                C = np.asarray(tpb.create_transform_matrix(Basis.cast(name, d1 * d2)))
                self.assertEqual(np.allclose(C, np.eye(d1 * d2)), tensors,
                                 msg=f"{name} {d1}x{d2}")

    def test_pauli_product_basis_builds_no_transform_matrix(self):
        # The structural check must settle pp before any transform matrix is built: building one
        # is O(dim^2) and dominates construction cost.
        def boom(self, other):
            raise AssertionError("built a transform matrix for a Pauli-product basis")
        with unittest.mock.patch.object(TensorProdBasis, 'create_transform_matrix', boom), \
             unittest.mock.patch.object(TensorProdBasis, 'reverse_transform_matrix', boom):
            for ss in (QubitSpace(2), QubitSpace(5)):
                gg = ggrp.TensorProductGaugeGroup.local_unitary(ss, Basis.cast('pp', ss))
                self.assertIsInstance(gg._change_of_basis, IdentityOperator)
                self.assertIsInstance(gg._change_of_basis_inverse, IdentityOperator)

    def test_sparse_bases_keep_change_of_basis_sparse(self):
        # C and its inverse are used only via matmul, so there's no reason to densify them.
        gm9 = Basis.cast('gm', 9, sparse=True)
        factor = ggrp.UnitaryGaugeGroup(QuditSpace(1, 3), 'gm')
        gg = ggrp.TensorProductGaugeGroup([factor, factor], QuditSpace(2, 3),
                                          Basis.cast('gm', 81, sparse=True),
                                          factor_bases=[gm9, gm9])
        self.assertTrue(sps.issparse(gg._change_of_basis))
        self.assertTrue(sps.issparse(gg._change_of_basis_inverse))
        el = gg.compute_element(0.1 * self.rng.normal(size=gg.num_params))
        mx = el.transform_matrix
        self.assertIsInstance(mx, np.ndarray)  # conjugation still yields a dense transform
        self.assertArraysAlmostEqual(el.transform_matrix_inverse @ mx, np.eye(81))
        # and it agrees with the same group built from dense bases
        dense = ggrp.TensorProductGaugeGroup([factor, factor], QuditSpace(2, 3), Basis.cast('gm', 81))
        dense_el = dense.compute_element(el.to_vector())
        self.assertArraysAlmostEqual(mx, dense_el.transform_matrix)

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
        self.assertIsInstance(gg._change_of_basis, IdentityOperator)
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

    def test_inverse_element_from_vector_updates_wrapped_element(self):
        el = self.gg.compute_element(self.v)
        inv = el.inverse()
        w = self.rng.random(self.n_params)
        inv.from_vector(w)
        self.assertArraysAlmostEqual(el.to_vector(), w)
        self.assertArraysAlmostEqual(inv.to_vector(), w)
        self.assertArraysAlmostEqual(inv.transform_matrix, np.linalg.inv(self.gg.compute_element(w).transform_matrix))

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
        self.assertNotIsInstance(gg._change_of_basis, IdentityOperator)
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
        self.assertIsInstance(gg._change_of_basis, IdentityOperator)
        self.assertArraysAlmostEqual(gg.compute_element(self.v).transform_matrix,
                                     self.gg.compute_element(self.v).transform_matrix)
        # with the TensorProdBasis that pyGSTi-built models carry
        tpb = TensorProdBasis([Basis.cast('pp', 4)] * 2)
        gg2 = ggrp.TensorProductGaugeGroup.local_unitary(self.state_space, tpb)
        self.assertIsInstance(gg2._change_of_basis, IdentityOperator)
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

    def test_factor_absorbing_dimension_one_label_warns(self):
        # A dim-2 factor sitting where a dimension-1 label starts claims both labels, since
        # 1 * 2 == 2.  That's dimensionally consistent, so it's allowed, but it's warned about.
        ss = ExplicitStateSpace(['L', 'Q0'], [1, 2])
        factor = ggrp.UnitaryGaugeGroup(ExplicitStateSpace(['Q0'], [2]), 'pp')
        with self.assertWarns(DubiousTargetWarning):
            gg = ggrp.TensorProductGaugeGroup([factor], ss, Basis.cast('pp', 4))
        self.assertEqual(gg._label_runs, ((0, 1),))

    def test_dimension_one_label_with_its_own_factor_does_not_warn(self):
        ss = ExplicitStateSpace(['Q0', 'L', 'Q1'], [2, 1, 3])
        with warnings.catch_warnings():
            warnings.simplefilter("error", DubiousTargetWarning)
            ggrp.TensorProductGaugeGroup.local_unitary(ss, 'gm')

    def test_local_constructors_reject_multi_block_space(self):
        ss = ExplicitStateSpace([('Q0',), ('L',)], [(2,), (1,)])
        with self.assertRaises(ValueError):
            ggrp.TensorProductGaugeGroup.local_unitary(ss, 'pp')

    def test_serialization_round_trip(self):
        import json
        for gg in (self.gg,
                   ggrp.TensorProductGaugeGroup.local_unitary(ExplicitStateSpace(['Q0', 'L', 'Q1'], [2, 1, 3]), 'gm'),
                   ggrp.TensorProductGaugeGroup([ggrp.UnitaryGaugeGroup(QuditSpace(1, 3), 'gm')] * 2,
                                                QuditSpace(2, 3), Basis.cast('gm', 81))):
            state = gg.to_nice_serialization()
            json.dumps(state)  # "nice" means JSON-able
            gg2 = ggrp.GaugeGroup.from_nice_serialization(state)
            self.assertIsInstance(gg2, ggrp.TensorProductGaugeGroup)
            self.assertEqual(gg2.num_params, gg.num_params)
            self.assertEqual(gg2.state_space, gg.state_space)
            self.assertEqual([type(f) for f in gg2.factors], [type(f) for f in gg.factors])
            self.assertEqual(isinstance(gg2._change_of_basis, IdentityOperator),
                             isinstance(gg._change_of_basis, IdentityOperator))
            v = 0.3 * self.rng.normal(size=gg.num_params)
            el, el2 = gg.compute_element(v), gg2.compute_element(v)
            self.assertArraysAlmostEqual(el.transform_matrix, el2.transform_matrix)
            # element round trip (like other Op-based elements, this keeps the matrices, not the parameterization)
            estate = el.to_nice_serialization()
            json.dumps(estate)
            el3 = ggrp.GaugeGroupElement.from_nice_serialization(estate)
            self.assertIsInstance(el3, ggrp.TensorProductGaugeGroupElement)
            self.assertArraysAlmostEqual(el.transform_matrix, el3.transform_matrix)
            self.assertArraysAlmostEqual(el.transform_matrix_inverse, el3.transform_matrix_inverse)

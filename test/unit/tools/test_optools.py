import os
import unittest
from unittest import mock
import numpy as np
import scipy

from ..util import BaseCase

from pygsti.construction import std2Q_XXYYII
import pygsti.tools.optools as ot
import pygsti.tools.basistools as bt
from pygsti.baseobjs.basis import Basis


class OpToolsTester(BaseCase):
    def test_unitary_to_pauligate(self):
        theta = np.pi
        ex = 1j * theta * bt.sigmax / 2
        U = scipy.linalg.expm(ex)
        # U is 2x2 unitary matrix operating on single qubit in [0,1] basis (X(pi) rotation)

        op = ot.unitary_to_pauligate(U)
        op_ans = np.array([[ 1.,  0.,  0.,  0.],
                           [ 0.,  1.,  0.,  0.],
                           [ 0.,  0., -1.,  0.],
                           [ 0.,  0.,  0., -1.]], 'd')
        self.assertArraysAlmostEqual(op, op_ans)

        U_2Q = np.identity(4, 'complex'); U_2Q[2:, 2:] = U
        # U_2Q is 4x4 unitary matrix operating on isolated two-qubit space (CX(pi) rotation)

        op_2Q = ot.unitary_to_pauligate(U_2Q)
        # TODO assert correctness

    def test_decompose_gate_matrix(self):
        oneRealPair = np.array([
            [1+1j,    0, 0, 0],
            [   0, 1-1j, 0, 0],
            [   0,    0, 2, 0],
            [   0,    0, 0, 2]
        ], 'complex')
        decomp = ot.decompose_gate_matrix(oneRealPair)
        # decompose gate mx whose eigenvalues have a real but non-unit pair
        # TODO assert correctness

        dblRealPair = np.array([
            [ 3, 0, 0, 0],
            [ 0, 3, 0, 0],
            [ 0, 0, 2, 0],
            [ 0, 0, 0, 2]
        ], 'complex')
        decomp = ot.decompose_gate_matrix(dblRealPair)
        # decompose gate mx whose eigenvalues have two real but non-unit pairs
        # TODO assert correctness

    def test_decompose_gate_matrix_invalidates_on_all_complex_eigval(self):
        unpairedMx = np.array([
            [1+1j,    0,    0,      0],
            [   0, 2-1j,    0,      0],
            [   0,    0, 2+2j,      0],
            [   0,    0,    0, 1.0+3j]
        ], 'complex')
        decomp = ot.decompose_gate_matrix(unpairedMx)
        # decompose gate mx which has all complex eigenvalue -> bail out
        self.assertFalse(decomp['isValid'])

    def test_decompose_gate_matrix_invalidates_on_large_matrix(self):
        largeMx = np.identity(16, 'd')
        decomp = ot.decompose_gate_matrix(largeMx)  # can only handle 1Q mxs
        self.assertFalse(decomp['isValid'])

    def test_hack_sqrt_m(self):
        expected = np.array([
            [ 0.55368857+0.46439416j,  0.80696073-0.21242648j],
            [ 1.21044109-0.31863972j,  1.76412966+0.14575444j]
        ])
        sqrt = ot._hack_sqrtm(np.array([[1, 2], [3, 4]]))
        self.assertArraysAlmostEqual(sqrt, expected)

    def test_unitary_to_process_mx(self):
        identity = np.identity(2)
        processMx = ot.unitary_to_process_mx(identity)
        self.assertArraysAlmostEqual(processMx, np.identity(4))


class ProjectModelTester(BaseCase):
    def setUp(self):
        self.projectionTypes = ('H', 'S', 'H+S', 'LND', 'LNDF')
        self.target_model = std2Q_XXYYII.target_model()
        self.model = self.target_model.depolarize(op_noise=0.01)

    def test_log_diff_model_projection(self):
        # Optimization is expensive, fake it
        def side_effect(o, mx, **kwargs):
            return mock.MagicMock(x=mx)
        with mock.patch.object(scipy.optimize, 'minimize', side_effect=side_effect):
            proj_model, Np_dict = ot.project_model(self.model, self.target_model, self.projectionTypes, 'logG-logT')
            # TODO assert correctness

    def test_logTiG_model_projection(self):
        # TODO mock expensive calls
        proj_model, Np_dict = ot.project_model(self.model, self.target_model, self.projectionTypes, 'logTiG')
        # TODO assert correctness

    def test_logGTi_model_projection(self):
        # TODO mock expensive calls
        proj_model, Np_dict = ot.project_model(self.model, self.target_model, self.projectionTypes, 'logGTi')
        # TODO assert correctness

    def test_raises_on_basis_mismatch(self):
        with self.assertRaises(ValueError):
            mdl_target_gm = std2Q_XXYYII.target_model()
            mdl_target_gm.basis = Basis.cast("gm", 16)
            ot.project_model(self.model, mdl_target_gm, self.projectionTypes, 'logGti')  # basis mismatch


class GateOpsTester(BaseCase):
    def setUp(self):
        self.A = np.array([
            [  0.9, 0, 0.1j,   0],
            [    0, 0,    0,   0],
            [-0.1j, 0,    0,   0],
            [    0, 0,    0, 0.1]
        ], 'complex')

        self.B = np.array([
            [ 0.5,    0,    0, -0.2j],
            [   0, 0.25,    0,     0],
            [   0,    0, 0.25,     0],
            [0.2j,    0,    0,   0.1]
        ], 'complex')

    def test_frobenius_distance(self):
        self.assertAlmostEqual(ot.frobeniusdist(self.A, self.A), 0.0)
        self.assertAlmostEqual(ot.frobeniusdist(self.A, self.B), (0.430116263352+0j))

        self.assertAlmostEqual(ot.frobeniusdist2(self.A, self.A), 0.0)
        self.assertAlmostEqual(ot.frobeniusdist2(self.A, self.B), (0.185+0j))

    def test_jtrace_distance(self):
        self.assertAlmostEqual(ot.jtracedist(self.A, self.A, mxBasis="std"), 0.0)
        self.assertAlmostEqual(ot.jtracedist(self.A, self.B, mxBasis="std"), 0.26430148)  # OLD: 0.2601 ?

    @unittest.skipIf(os.getenv('SKIP_CVXPY'), "skipping cvxpy tests")
    def test_diamond_distance(self):
        self.assertAlmostEqual(ot.diamonddist(self.A, self.A, mxBasis="std"), 0.0)
        self.assertAlmostEqual(ot.diamonddist(self.A, self.B, mxBasis="std"), 0.614258836298)

    def test_frobenius_norm_equiv(self):
        from pygsti.tools import matrixtools as mt
        self.assertAlmostEqual(ot.frobeniusdist(self.A, self.B), mt.frobeniusnorm(self.A - self.B))
        self.assertAlmostEqual(ot.frobeniusdist(self.A, self.B), np.sqrt(mt.frobeniusnorm2(self.A - self.B)))

    def test_entanglement_fidelity(self):
        fidelity = ot.entanglement_fidelity(self.A, self.B)
        self.assertAlmostEqual(fidelity, 0.42686642003)

    def test_fidelity_upper_bound(self):
        upperBound = ot.fidelity_upper_bound(self.A)
        expected = (
            np.array([[ 0.25]]),
            np.array([[  1.00000000e+00,  -8.27013523e-16,   8.57305616e-33, 1.95140273e-15],
                      [ -8.27013523e-16,   1.00000000e+00,   6.28036983e-16, -8.74760501e-31],
                      [  5.68444574e-33,  -6.28036983e-16,   1.00000000e+00, -2.84689309e-16],
                      [  1.95140273e-15,  -9.27538795e-31,   2.84689309e-16, 1.00000000e+00]])
        )
        self.assertArraysAlmostEqual(upperBound[0], expected[0])
        self.assertArraysAlmostEqual(upperBound[1], expected[1])

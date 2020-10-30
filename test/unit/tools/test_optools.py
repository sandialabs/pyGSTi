import functools
import numpy as np
import scipy
from unittest import mock

from ..util import BaseCase, needs_cvxpy

from pygsti.modelpacks.legacy import std2Q_XXYYII
import pygsti.tools.optools as ot
import pygsti.tools.basistools as bt
import pygsti.tools.lindbladtools as lt
from pygsti.objects.basis import Basis


def fake_minimize(fn):
    """Mock scipy.optimize.minimize in the underlying function call to reduce optimization overhead"""
    def side_effect(o, mx, **kwargs):
        return mock.MagicMock(x=mx)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with mock.patch.object(scipy.optimize, 'minimize', side_effect=side_effect):
            return fn(*args, **kwargs)

    return wrapper


class OpToolsTester(BaseCase):
    def test_unitary_to_pauligate(self):
        theta = np.pi
        sigmax = np.array([[0, 1], [1, 0]])
        ex = 1j * theta * sigmax / 2
        U = scipy.linalg.expm(ex)
        # U is 2x2 unitary matrix operating on single qubit in [0,1] basis (X(pi) rotation)

        op = ot.unitary_to_pauligate(U)
        op_ans = np.array([[ 1.,  0.,  0.,  0.],
                           [ 0.,  1.,  0.,  0.],
                           [ 0.,  0., -1.,  0.],
                           [ 0.,  0.,  0., -1.]], 'd')
        self.assertArraysAlmostEqual(op, op_ans)

        U_2Q = np.identity(4, 'complex')
        U_2Q[2:, 2:] = U
        # U_2Q is 4x4 unitary matrix operating on isolated two-qubit space (CX(pi) rotation)

        op_2Q = ot.unitary_to_pauligate(U_2Q)
        op_2Q_inv = ot.process_mx_to_unitary(bt.change_basis(op_2Q, 'pp', 'std'))
        self.assertArraysAlmostEqual(U_2Q, op_2Q_inv)

    def test_decompose_gate_matrix(self):
        # decompose gate mx whose eigenvalues have a real but non-unit pair
        oneRealPair = np.array([
            [1+1j,    0, 0, 0],  # Angle between 0 and 1 should give rotation
            [   0, 1-1j, 0, 0],
            [   0,    0, 2, 0],  # should be picked out as fixed point (first real eigenval)
            [   0,    0, 0, 2]   # should be picked out as axis of rotation
        ], 'complex')
        decomp = ot.decompose_gate_matrix(oneRealPair)

        self.assertEqual(decomp['isValid'], True)
        self.assertEqual(decomp['isUnitary'], False)
        self.assertArraysAlmostEqual(decomp['fixed point'], [0, 0, 1, 0])
        self.assertArraysAlmostEqual(decomp['axis of rotation'], [0, 0, 0, 1])
        self.assertArraysAlmostEqual(decomp['rotating axis 1'], [1, 0, 0, 0])
        self.assertArraysAlmostEqual(decomp['rotating axis 2'], [0, 1, 0, 0])
        self.assertEqual(decomp['decay of diagonal rotation terms'], 1.0 - 2.0)
        self.assertEqual(decomp['decay of off diagonal rotation terms'], 1.0 - abs(1+1j))
        self.assertEqual(decomp['pi rotations'], np.angle(1+1j)/np.pi)

        dblRealPair = np.array([
            [ 3, 0, 0, 0],
            [ 0, 3, 0, 0],
            [ 0, 0, 2, 0], # still taken as fixed point because closest to identity (1.0)
            [ 0, 0, 0, 2]
        ], 'complex')
        decomp = ot.decompose_gate_matrix(dblRealPair)
        # decompose gate mx whose eigenvalues have two real but non-unit pairs
        
        self.assertEqual(decomp['isValid'], True)
        self.assertEqual(decomp['isUnitary'], False)
        self.assertArraysAlmostEqual(decomp['fixed point'], [0, 0, 1, 0])
        self.assertArraysAlmostEqual(decomp['axis of rotation'], [0, 0, 0, 1])
        self.assertArraysAlmostEqual(decomp['rotating axis 1'], [1, 0, 0, 0])
        self.assertArraysAlmostEqual(decomp['rotating axis 2'], [0, 1, 0, 0])
        self.assertEqual(decomp['decay of diagonal rotation terms'], 1.0 - 2.0)
        self.assertEqual(decomp['decay of off diagonal rotation terms'], 1.0 - 3.0)
        self.assertEqual(decomp['pi rotations'], np.angle(3.0)/np.pi)

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

    @fake_minimize
    def test_log_diff_model_projection(self):
        self.skipTest("project_model for logG-logT is known to be inconsistent in testing (Gxx,Gxy,Gyx,Gyy gates).  Skip tests until it gets fixed.")
        basis = self.target_model.basis
        gen_type = 'logG-logT'
        proj_model, Np_dict = ot.project_model(self.model, self.target_model, self.projectionTypes, gen_type, logG_weight=0)
        # Project a second time and ensure models don't change
        for pm1, ptype in zip(proj_model, self.projectionTypes):
            proj2, _ = ot.project_model(pm1, self.target_model, [ptype], gen_type, logG_weight=0)
            pm2 = proj2[0]
            for pm1_op, pm2_op in zip(pm1.operations.values(), pm2.operations.values()):
                self.assertArraysAlmostEqual(pm1_op, pm2_op)

    def test_logTiG_model_projection(self):
        gen_type = 'logTiG'
        proj_model, Np_dict = ot.project_model(self.model, self.target_model, self.projectionTypes, gen_type)
        # Project a second time and ensure models don't change
        for pm1, ptype in zip(proj_model, self.projectionTypes):
            proj2, _ = ot.project_model(pm1, self.target_model, [ptype], gen_type, logG_weight=0)
            pm2 = proj2[0]
            for pm1_op, pm2_op in zip(pm1.operations.values(), pm2.operations.values()):
                self.assertArraysAlmostEqual(pm1_op, pm2_op)

    def test_logGTi_model_projection(self):
        gen_type = 'logGTi'
        proj_model, Np_dict = ot.project_model(self.model, self.target_model, self.projectionTypes, gen_type)
        # Project a second time and ensure models don't change
        for pm1, ptype in zip(proj_model, self.projectionTypes):
            proj2, _ = ot.project_model(pm1, self.target_model, [ptype], gen_type, logG_weight=0)
            pm2 = proj2[0]
            for pm1_op, pm2_op in zip(pm1.operations.values(), pm2.operations.values()):
                self.assertArraysAlmostEqual(pm1_op, pm2_op)

    def test_raises_on_basis_mismatch(self):
        with self.assertRaises(ValueError):
            mdl_target_gm = std2Q_XXYYII.target_model()
            mdl_target_gm.basis = Basis.cast("gm", 16)
            ot.project_model(self.model, mdl_target_gm, self.projectionTypes, 'logGti')  # basis mismatch


class ErrorGenTester(BaseCase):
    def setUp(self):
        self.target_model = std2Q_XXYYII.target_model()
        self.mdl_datagen = self.target_model.depolarize(op_noise=0.1, spam_noise=0.001)

    def test_std_errgens(self):
        projectionTypes = ['hamiltonian', 'stochastic', 'affine']
        basisNames = ['std', 'gm', 'pp']  # , 'qt'] #dim must == 3 for qt

        for projectionType in projectionTypes:
            ot.std_scale_factor(4, projectionType)
            for basisName in basisNames:
                ot.std_error_generators(4, projectionType, basisName)

    def test_std_errgens_raise_on_bad_projection_type(self):
        with self.assertRaises(ValueError):
            ot.std_scale_factor(4, "foobar")
        with self.assertRaises(ValueError):
            ot.std_error_generators(4, "foobar", 'gm')

    def test_lind_errgens(self):
        basis = Basis.cast('gm', 4)

        normalize = False
        other_mode = "all"
        H0, O0 = ot.lindblad_error_generators(basis, basis, normalize, other_mode)
        H1, O1 = ot.lindblad_error_generators(None, basis, normalize, other_mode)
        H2, O2 = ot.lindblad_error_generators(basis, None, normalize, other_mode)
        H3, O3 = ot.lindblad_error_generators(None, None, normalize, other_mode)

        # Check lindblad generators called as expected
        for i, mi in enumerate(basis[1:]):
            Hi = lt.hamiltonian_to_lindbladian(mi)
            self.assertArraysAlmostEqual(Hi, H0[i])

            for j, mj in enumerate(basis[1:]):
                Oij = lt.nonham_lindbladian(mi, mj)
                self.assertArraysAlmostEqual(Oij, O0[i, j])

        # Check Nones handled properly
        self.assertEqual(H1, None)
        self.assertArraysAlmostEqual(O0, O1)
        self.assertArraysAlmostEqual(H0, H2)
        self.assertEqual(O2, None)
        self.assertEqual(H3, None)
        self.assertEqual(O3, None)

        normalize = True
        other_mode = "all"
        H0n, O0n = ot.lindblad_error_generators(basis, basis, normalize, other_mode)
        H1n, O1n = ot.lindblad_error_generators(None, basis, normalize, other_mode)
        H2n, O2n = ot.lindblad_error_generators(basis, None, normalize, other_mode)
        H3n, O3n = ot.lindblad_error_generators(None, None, normalize, other_mode)

        # Check normalization against unnormalized version
        for h, hn in zip(H0, H0n):
            norm = np.linalg.norm(h)
            normedh = h if np.isclose(norm, 0) else h / norm
            self.assertArraysAlmostEqual(normedh, hn)
        for row, rown in zip(O0, O0n):
            for o, on in zip(row, rown):
                norm = np.linalg.norm(o)
                normedo = o if np.isclose(norm, 0) else o / norm
                self.assertArraysAlmostEqual(normedo, on)

        # Check Nones handled properly
        self.assertEqual(H1n, None)
        self.assertArraysAlmostEqual(O0n, O1n)
        self.assertArraysAlmostEqual(H0n, H2n)
        self.assertEqual(O2n, None)
        self.assertEqual(H3n, None)
        self.assertEqual(O3n, None)

        normalize = False
        other_mode = "diagonal"
        H0d, O0d = ot.lindblad_error_generators(basis, basis, normalize, other_mode)
        H1d, O1d = ot.lindblad_error_generators(None, basis, normalize, other_mode)
        H2d, O2d = ot.lindblad_error_generators(basis, None, normalize, other_mode)
        H3d, O3d = ot.lindblad_error_generators(None, None, normalize, other_mode)

        # Check diag vs all
        self.assertArraysAlmostEqual(H0, H0d)
        for i, od in enumerate(O0d):
            self.assertArraysEqual(O0[i, i], od)

        # Check Nones handled properly
        self.assertEqual(H1d, None)
        self.assertArraysAlmostEqual(O0d, O1d)
        self.assertArraysAlmostEqual(H0d, H2d)
        self.assertEqual(O2d, None)
        self.assertEqual(H3d, None)
        self.assertEqual(O3d, None)

        normalize = False
        other_mode = "diag_affine"
        H0da, O0da = ot.lindblad_error_generators(basis, basis, normalize, other_mode)
        H1da, O1da = ot.lindblad_error_generators(None, basis, normalize, other_mode)
        H2da, O2da = ot.lindblad_error_generators(basis, None, normalize, other_mode)
        H3da, O3da = ot.lindblad_error_generators(None, None, normalize, other_mode)

        # Check diag component
        self.assertArraysAlmostEqual(H0, H0da)
        self.assertArraysAlmostEqual(O0d, O0da[0, :])
        # Check affine component called as expected
        for i, mi in enumerate(basis[1:]):
            A = lt.affine_lindbladian(mi)
            self.assertArraysAlmostEqual(A, O0da[1, i])

        # Check Nones handled properly
        self.assertEqual(H1da, None)
        self.assertArraysAlmostEqual(O0da, O1da)
        self.assertArraysAlmostEqual(H0da, H2da)
        self.assertEqual(O2da, None)
        self.assertEqual(H3da, None)
        self.assertEqual(O3da, None)

    def test_lind_errgen_projects(self):
        basis = Basis.cast('gm', 4)
        
        # Build known combination to project back to
        Hgen, Ogen = ot.lindblad_error_generators(basis, basis, True, 'all')

        href = np.array([1/4, 0, 0])
        oref = np.array([[1/4, 0, 0], [0, 0, 1/4], [0, 0, 1/4]])
        
        errgen = np.zeros_like(Hgen[0])
        for i in range(3):
            errgen += href[i]*Hgen[i]
            for j in range(3):
                errgen += oref[i,j]*Ogen[i,j]

        hc, oc = ot.lindblad_errorgen_projections(errgen, basis, basis, 'std',
                                                  normalize=True, return_generators=False,
                                                  other_mode="all", sparse=False)
        self.assertArraysAlmostEqual(href, hc)
        self.assertArraysAlmostEqual(oref, oc)

        # Test basis from name (really as base case for sparse)
        hc, oc = ot.lindblad_errorgen_projections(errgen, 'gm', 'gm', 'std',
                                       normalize=True, return_generators=False,
                                       other_mode="all", sparse=False)
        self.assertArraysAlmostEqual(href, hc)
        self.assertArraysAlmostEqual(oref, oc)

        # Test sparse version
        hc, oc = ot.lindblad_errorgen_projections(errgen, 'gm', 'gm', 'std',
                                       normalize=True, return_generators=False,
                                       other_mode="all", sparse=True)
        self.assertArraysAlmostEqual(href, hc)
        self.assertArraysAlmostEqual(oref, oc)

        # Test diagonal contributions only
        href = np.array([1/4, 0, 0])
        odiag = np.array([1/4, 1/4, 1/4])
        
        errgen = np.zeros_like(Hgen[0])
        for i in range(3):
            errgen += href[i]*Hgen[i]
            errgen += odiag[i]*Ogen[i, i]
        hc, oc = ot.lindblad_errorgen_projections(errgen, 'gm', 'gm', 'std',
                                       normalize=True, return_generators=False,
                                       other_mode="diagonal", sparse=False)
        self.assertArraysAlmostEqual(href, hc)
        self.assertArraysAlmostEqual(odiag, oc)

    @fake_minimize
    def test_err_gen(self):
        projectionTypes = ['hamiltonian', 'stochastic', 'affine']
        basisNames = ['std', 'gm', 'pp']  # , 'qt'] #dim must == 3 for qt

        for (lbl, gateTarget), gate in zip(self.target_model.operations.items(), self.mdl_datagen.operations.values()):
            errgen = ot.error_generator(gate, gateTarget, self.target_model.basis, 'logG-logT')
            altErrgen = ot.error_generator(gate, gateTarget, self.target_model.basis, 'logTiG')
            altErrgen2 = ot.error_generator(gate, gateTarget, self.target_model.basis, 'logGTi')
            with self.assertRaises(ValueError):
                ot.error_generator(gate, gateTarget, self.target_model.basis, 'adsf')

            for projectionType in projectionTypes:
                for basisName in basisNames:
                    ot.std_errorgen_projections(errgen, projectionType, basisName)

            originalGate = ot.operation_from_error_generator(errgen, gateTarget, self.target_model.basis, 'logG-logT')
            altOriginalGate = ot.operation_from_error_generator(altErrgen, gateTarget, self.target_model.basis, 'logTiG')
            altOriginalGate2 = ot.operation_from_error_generator(altErrgen, gateTarget, self.target_model.basis, 'logGTi')
            with self.assertRaises(ValueError):
                ot.operation_from_error_generator(errgen, gateTarget, self.target_model.basis, 'adsf')
            self.assertArraysAlmostEqual(originalGate, gate) # sometimes need to approximate the log for this one
            self.assertArraysAlmostEqual(altOriginalGate, gate)
            self.assertArraysAlmostEqual(altOriginalGate2, gate)

    @fake_minimize
    def test_err_gen_nonunitary(self):
        errgen_nonunitary = ot.error_generator(self.mdl_datagen.operations['Gxi'],
                                               self.mdl_datagen.operations['Gxi'],
                                               self.mdl_datagen.basis)
        # Perfect match, should get all 0s
        self.assertArraysAlmostEqual(np.zeros_like(self.mdl_datagen.operations['Gxi']), errgen_nonunitary)

    def test_err_gen_not_near_gate(self):
        # Both should warn
        with self.assertWarns(UserWarning):
            errgen_notsmall = ot.error_generator(self.mdl_datagen.operations['Gxi'], self.target_model.operations['Gix'],
                                                 self.target_model.basis, 'logTiG')

        with self.assertWarns(UserWarning):
            errgen_notsmall = ot.error_generator(self.mdl_datagen.operations['Gxi'], self.target_model.operations['Gix'],
                                                 self.target_model.basis, 'logGTi')

    def test_err_gen_raises_on_bad_type(self):
        with self.assertRaises(ValueError):
            ot.error_generator(self.mdl_datagen.operations['Gxi'], self.target_model.operations['Gxi'],
                               self.target_model.basis, 'foobar')

    def test_err_gen_assert_shape_raises_on_ndims_too_high(self):
        # Check helper routine _assert_shape
        with self.assertRaises(NotImplementedError):  # boundary case
            ot._assert_shape(np.zeros((2, 2, 2, 2, 2), 'd'), (2, 2, 2, 2, 2), sparse=True)  # ndims must be <= 4


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

        self.assertAlmostEqual(ot.frobeniusdist_squared(self.A, self.A), 0.0)
        self.assertAlmostEqual(ot.frobeniusdist_squared(self.A, self.B), (0.185+0j))

    def test_jtrace_distance(self):
        self.assertAlmostEqual(ot.jtracedist(self.A, self.A, mx_basis="std"), 0.0)
        self.assertAlmostEqual(ot.jtracedist(self.A, self.B, mx_basis="std"), 0.26430148)  # OLD: 0.2601 ?

    @needs_cvxpy
    def test_diamond_distance(self):
        self.assertAlmostEqual(ot.diamonddist(self.A, self.A, mx_basis="std"), 0.0)
        self.assertAlmostEqual(ot.diamonddist(self.A, self.B, mx_basis="std"), 0.614258836298)

    def test_frobenius_norm_equiv(self):
        from pygsti.tools import matrixtools as mt
        self.assertAlmostEqual(ot.frobeniusdist(self.A, self.B), mt.frobeniusnorm(self.A - self.B))
        self.assertAlmostEqual(ot.frobeniusdist(self.A, self.B), np.sqrt(mt.frobeniusnorm_squared(self.A - self.B)))

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

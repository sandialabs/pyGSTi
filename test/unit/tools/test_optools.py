import functools
from unittest import mock

import sys
import numpy as np
import scipy
from pygsti.baseobjs.basis import Basis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL

import pygsti.tools.basistools as bt
import pygsti.tools.lindbladtools as lt
import pygsti.tools.optools as ot
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock
from pygsti.modelpacks.legacy import std2Q_XXYYII
from ..util import BaseCase, needs_cvxpy

SKIP_DIAMONDIST_ON_WIN = True


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
        op_2Q_inv = ot.std_process_mx_to_unitary(bt.change_basis(op_2Q, 'pp', 'std'))
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

    def test_unitary_to_process_mx(self):
        identity = np.identity(2)
        processMx = ot.unitary_to_std_process_mx(identity)
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
        projectionTypes = ['H', 'S', 'C', 'A']
        basisNames = ['gm', 'pp']  # , 'qt'] #dim must == 3 for qt
        # Note: bases must have first element == identity

        for projectionType in projectionTypes:
            #REMOVE ot.std_scale_factor(4, projectionType)
            for basisName in basisNames:
                #REMOVE ot.std_error_generators(4, projectionType, basisName)
                ot.elementary_errorgens_dual(4, projectionType, basisName)

    def test_std_errgens_raise_on_bad_projection_type(self):
        with self.assertRaises(AssertionError):
            #REMOVE ot.std_error_generators(4, "foobar", 'gm')
            ot.elementary_errorgens_dual(4, "foobar", 'gm')

    def test_lind_errgens(self):

        bases = [Basis.cast('gm', 4),
                 Basis.cast('pp', 4),
                 Basis.cast('PP', 4)]

        for basis in bases:
            print(basis)
            Hblk = LindbladCoefficientBlock('ham', basis)
            Hblk_superops = Hblk.create_lindblad_term_superoperators(mx_basis='std')

            for i, mi in enumerate(basis[1:]):
                Hi = lt.create_elementary_errorgen('H', mi)
                HiB = lt.create_lindbladian_term_errorgen('H', mi)
                self.assertArraysAlmostEqual(Hi, HiB)
                self.assertArraysAlmostEqual(Hi, Hblk_superops[i])

            ODblk = LindbladCoefficientBlock('other_diagonal', basis)
            ODblk_superops = ODblk.create_lindblad_term_superoperators(mx_basis='std')

            for i, mi in enumerate(basis[1:]):
                ODi = lt.create_elementary_errorgen('S', mi)
                ODiB = lt.create_lindbladian_term_errorgen('O', mi, mi)
                self.assertArraysAlmostEqual(ODi, ODiB)
                self.assertArraysAlmostEqual(ODi, ODblk_superops[i])

            Oblk = LindbladCoefficientBlock('other', basis)
            Oblk_superops = Oblk.create_lindblad_term_superoperators(mx_basis='std')

            for i, mi in enumerate(basis[1:]):
                for j, mj in enumerate(basis[1:]):
                    Oij = lt.create_lindbladian_term_errorgen('O', mi, mj)
                    self.assertArraysAlmostEqual(Oij, Oblk_superops[i][j])

                    # C_PQ = NH_PQ + NH_QP
                    # A_PQ = i(NH_PQ - NH_QP)
                    if i < j:
                        Cij = lt.create_elementary_errorgen('C', mi, mj)
                        Aij = lt.create_elementary_errorgen('A', mi, mj)
                        self.assertArraysAlmostEqual(Oij, (Cij + 1j * Aij) / 2.0)
                    elif j < i:
                        Cji = lt.create_elementary_errorgen('C', mj, mi)
                        Aji = lt.create_elementary_errorgen('A', mj, mi)
                        self.assertArraysAlmostEqual(Oij, (Cji - 1j * Aji) / 2.0)
                    else:  # i == j
                        Sii = lt.create_elementary_errorgen('S', mi)
                        self.assertArraysAlmostEqual(Oij, Sii)

    def test_lind_errgen_projects(self):
        mx_basis = Basis.cast('pp', 4)
        basis = Basis.cast('PP', 4)
        X = basis['X']
        Y = basis['Y']
        Z = basis['Z']

        # Build known combination to project back to
        errgen = 0.1 * lt.create_elementary_errorgen('H', Z) \
            - 0.01 * lt.create_elementary_errorgen('H', X) \
            + 0.2 * lt.create_elementary_errorgen('S', X) \
            + 0.25 * lt.create_elementary_errorgen('S', Y) \
            + 0.05 * lt.create_elementary_errorgen('C', X, Y) \
            - 0.01 * lt.create_elementary_errorgen('A', X, Y)
        errgen = bt.change_basis(errgen, 'std', mx_basis)

        Hblk = LindbladCoefficientBlock('ham', basis)
        ODblk = LindbladCoefficientBlock('other_diagonal', basis)
        Oblk = LindbladCoefficientBlock('other', basis)

        Hblk.set_from_errorgen_projections(errgen, errorgen_basis=mx_basis)
        ODblk.set_from_errorgen_projections(errgen, errorgen_basis=mx_basis)
        Oblk.set_from_errorgen_projections(errgen, errorgen_basis=mx_basis)

        self.assertArraysAlmostEqual(Hblk.block_data, [-0.01, 0, 0.1])
        self.assertArraysAlmostEqual(ODblk.block_data, [0.2, 0.25, 0])
        self.assertArraysAlmostEqual(Oblk.block_data,
                                     np.array([[0.2,          0.05 + 0.01j, 0],
                                               [0.05 - 0.01j, 0.25,         0],
                                               [0,            0,            0]]))

        def dicts_equal(d, f):
            f = {LEEL.cast(k): v for k, v in f.items()}
            if set(d.keys()) != set(f.keys()): return False
            for k in d:
                if abs(d[k] - f[k]) > 1e-12: return False
            return True

        self.assertTrue(dicts_equal(Hblk.elementary_errorgens, {('H','Z'): 0.1, ('H','X'): -0.01, ('H','Y'): 0}))
        self.assertTrue(dicts_equal(ODblk.elementary_errorgens, {('S','X'): 0.2, ('S','Y'): 0.25, ('S','Z'): 0}))
        self.assertTrue(dicts_equal(Oblk.elementary_errorgens,
                                    {('S', 'X'): 0.2,
                                     ('S', 'Y'): 0.25,
                                     ('S', 'Z'): 0.0,
                                     ('C', 'X', 'Y'): 0.05,
                                     ('A', 'X', 'Y'): -0.01,
                                     ('C', 'X', 'Z'): 0,
                                     ('A', 'X', 'Z'): 0,
                                     ('C', 'Y', 'Z'): 0,
                                     ('A', 'Y', 'Z'): 0,
                                     }))

        #TODO: test with sparse bases??

        #TODO: test basis from name (seems unnecessary)?

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

            #OLD: tested above
            #for projectionType in projectionTypes:
            #    for basisName in basisNames:
            #        ot.std_errorgen_projections(errgen, projectionType, basisName)

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
        
        self.A_TP= np.array([
            [1,  0,           0,           0],
            [0, -0.74972882,  0.06641116, -0.65840432],
            [0, -0.07921032, -0.99680422, -0.01034724],
            [0, -0.65698738,  0.04439479,  0.7525933 ]])
        self.B_unitary= np.array([
            [1,  0,           0,           0        ],
            [0, -0.29719065,  0.63991085, -0.70865494],
            [0,  0.79014219, -0.2518555 , -0.55878809],
            [0, -0.5360532 , -0.72600476, -0.43077146]])
            
        self.A_TP_std= np.array([
            [ 0.87629665+0.j, -0.32849369+0.0221974j, -0.32849369-0.0221974j,  0.12370335+0.j],
            [-0.32920216+0.00517362j, -0.87326652+0.07281074j, 0.1235377 +0.00639958j,  0.32920216-0.00517362j],
            [-0.32920216-0.00517362j,  0.1235377 -0.00639958j, -0.87326652-0.07281074j,  0.32920216+0.00517362j],
            [ 0.12370335+0.j,  0.32849369-0.0221974j,0.32849369+0.0221974j ,  0.87629665+0.j]])
        
        self.B_unitary_std= np.array([
            [ 0.28461427+0.j, -0.2680266 -0.36300238j, -0.2680266 + 0.36300238j,  0.71538573+0.j],
            [-0.35432747+0.27939404j, -0.27452307-0.07511567j, -0.02266757-0.71502652j,  0.35432747-0.27939404j],
            [-0.35432747-0.27939404j, -0.02266757+0.71502652j, -0.27452307+0.07511567j,  0.35432747+0.27939404j],
            [ 0.71538573+0.j,  0.2680266 +0.36300238j, 0.2680266 -0.36300238j,  0.28461427+0.j]])

    def test_frobenius_distance(self):
        self.assertAlmostEqual(ot.frobeniusdist(self.A, self.A), 0.0)
        self.assertAlmostEqual(ot.frobeniusdist(self.A, self.B), 0.6204836823)

        self.assertAlmostEqual(ot.frobeniusdist_squared(self.A, self.A), 0.0)
        self.assertAlmostEqual(ot.frobeniusdist_squared(self.A, self.B), 0.385)

    def test_jtrace_distance(self):
        val = ot.jtracedist(self.A_TP, self.A_TP, mx_basis="pp")
        self.assertAlmostEqual(val, 0.0)
        val = ot.jtracedist(self.A_TP, self.B_unitary, mx_basis="pp")
        self.assertGreaterEqual(val, 0.5)

    @needs_cvxpy
    def test_diamond_distance(self):
        if SKIP_DIAMONDIST_ON_WIN and sys.platform.startswith('win'): return
        val = ot.diamonddist(self.A_TP, self.A_TP, mx_basis="pp")
        self.assertAlmostEqual(val, 0.0)
        val = ot.diamonddist(self.A_TP, self.B_unitary, mx_basis="pp")
        self.assertGreaterEqual(val, 0.7)

    def test_entanglement_fidelity(self):
        fidelity_TP_unitary= ot.entanglement_fidelity(self.A_TP, self.B_unitary, is_tp=True, is_unitary=True)
        fidelity_TP_unitary_no_flag= ot.entanglement_fidelity(self.A_TP, self.B_unitary)
        fidelity_TP_unitary_jam= ot.entanglement_fidelity(self.A_TP, self.B_unitary, is_tp=False, is_unitary=False)
        fidelity_TP_unitary_std= ot.entanglement_fidelity(self.A_TP_std, self.B_unitary_std, mx_basis='std')
        
        expect = 0.4804724656092404
        self.assertAlmostEqual(fidelity_TP_unitary, expect)
        self.assertAlmostEqual(fidelity_TP_unitary_no_flag, expect)
        self.assertAlmostEqual(fidelity_TP_unitary_jam, expect)
        self.assertAlmostEqual(fidelity_TP_unitary_std, expect)

    def test_leaky_entanglement_fidelity(self):
        fidelity_TP_unitary= ot.leaky_entanglement_fidelity(self.A_TP, self.B_unitary, 'pp')
        fidelity_TP_unitary_no_flag= ot.leaky_entanglement_fidelity(self.A_TP, self.B_unitary, 'pp')
        fidelity_TP_unitary_jam= ot.leaky_entanglement_fidelity(self.A_TP, self.B_unitary, 'pp')
        fidelity_TP_unitary_std= ot.leaky_entanglement_fidelity(self.A_TP_std, self.B_unitary_std, mx_basis='std')

        expect = 0.4804724656092404
        self.assertAlmostEqual(fidelity_TP_unitary, expect)
        self.assertAlmostEqual(fidelity_TP_unitary_no_flag, expect)
        self.assertAlmostEqual(fidelity_TP_unitary_jam, expect)
        self.assertAlmostEqual(fidelity_TP_unitary_std, expect)
        pass

    def test_fidelity_upper_bound(self):
        np.random.seed(0)
        Q = np.linalg.qr(np.random.randn(4,4) + 1j*np.random.randn(4,4))[0]
        Q[:, 0] = 0.0  # zero out the first column
        bad_superoperator = ot.unitary_to_superop(Q)
        upperBound, _ = ot.fidelity_upper_bound(bad_superoperator)
        self.assertAlmostEqual(upperBound, 0.75)
        np.random.seed(0)
        Q = np.linalg.qr(np.random.randn(4,4) + 1j*np.random.randn(4,4))[0]
        Q[:, 0] = 0.0  # zero out the first column
        bad_superoperator = ot.unitary_to_superop(Q)
        upperBound, _ = ot.fidelity_upper_bound(bad_superoperator)
        self.assertAlmostEqual(upperBound, 0.75)

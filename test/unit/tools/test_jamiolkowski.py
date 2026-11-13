import numpy as np

import pygsti.tools.basistools as bt
from pygsti.models.modelconstruction import create_operation
from pygsti.modelpacks.legacy import std1Q_XYI as std1Q
from pygsti.baseobjs import statespace
from pygsti.baseobjs import Basis
from pygsti.baseobjs.basis import BuiltinBasis, TensorProdBasis
from pygsti.tools.optools import unitary_to_superop
from pygsti.tools import jamiolkowski as j
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
from ..util import BaseCase, needs_cvxpy


class JamiolkowskiBasisTester(BaseCase):
    def setUp(self):
        # (This test never indexes an ExplicitOpModel, so it does not need
        # strict mode.  It used to set ``ExplicitOpModel._strict = True`` at the
        # class level with no tearDown, which leaked into and broke unrelated
        # tests -- non-deterministically under parallel/xdist execution.)

        # density matrix == 3x3 block diagonal matrix: a 2x2 block followed by a 1x1 block
        self.stateSpaceDims = [(4,), (1,)]
        self.stateSpaceUDims = [(2,), (1,)]
        self.std = Basis.cast('std', 9)
        self.gm = Basis.cast('gm', 9)
        self.stdSmall = Basis.cast('std', [4, 1])
        self.gmSmall = Basis.cast('gm', [4, 1])

        #labels which give a tensor product interp. for the states within each density matrix block
        self.stateSpaceLabels = [('Qhappy',), ('Lsad',)]

        # Adjust for deprecation of _create_operation
        self.sslbls = statespace.ExplicitStateSpace(self.stateSpaceLabels, self.stateSpaceUDims)

        #Build a test gate   -- old # X(pi,Qhappy)*LX(pi,0,2)
        self.testGate = create_operation("LX(pi,0,2)", self.sslbls, self.stdSmall)
        self.testGate_mx = self.testGate.to_dense()
        self.testGateGM_mx = bt.change_basis(self.testGate_mx, self.stdSmall, self.gmSmall)
        self.expTestGate_mx = bt.flexible_change_basis(self.testGate_mx, self.stdSmall, self.std)
        self.expTestGateGM_mx = bt.change_basis(self.expTestGate_mx, self.std, self.gm)

    def checkBasis(self, cmb):
        #Op with Jamio map on gate in std and gm bases
        Jmx1 = j.jamiolkowski_iso(self.testGate_mx, op_mx_basis=self.stdSmall,
                                  choi_mx_basis=cmb)
        Jmx2 = j.jamiolkowski_iso(self.testGateGM_mx, op_mx_basis=self.gmSmall,
                                  choi_mx_basis=cmb)
        #print("Jmx1.shape = ", Jmx1.shape)

        #Make sure these yield the same trace == 1 matrix
        self.assertArraysAlmostEqual(Jmx1, Jmx2)
        self.assertAlmostEqual(np.trace(Jmx1), 1.0)

        #Op on expanded gate in std and gm bases
        JmxExp1 = j.jamiolkowski_iso(self.expTestGate_mx, op_mx_basis=self.std, choi_mx_basis=cmb)
        JmxExp2 = j.jamiolkowski_iso(self.expTestGateGM_mx, op_mx_basis=self.gm, choi_mx_basis=cmb)
        #print("JmxExp1.shape = ", JmxExp1.shape)

        #Make sure these are the same as operating on the contracted basis
        self.assertArraysAlmostEqual(Jmx1, JmxExp1)
        self.assertArraysAlmostEqual(Jmx1, JmxExp2)

        #Reverse transform should yield back the operation matrix
        revTestGate_mx = j.jamiolkowski_iso_inv(Jmx1, choi_mx_basis=cmb,
                                                op_mx_basis=self.gmSmall)
        self.assertArraysAlmostEqual(revTestGate_mx, self.testGateGM_mx)

        #Reverse transform without specifying stateSpaceDims, then contraction, should yield same result
        revExpTestGate_mx = j.jamiolkowski_iso_inv(Jmx1, choi_mx_basis=cmb, op_mx_basis=self.std)
        self.assertArraysAlmostEqual(bt.resize_std_mx(revExpTestGate_mx, 'contract', self.std, self.stdSmall),
                                     self.testGate_mx)

    def test_std_basis(self):
        #mx_dim = sum([ int(np.sqrt(d)) for d in ])
        cmb = Basis.cast('std', self.stateSpaceDims)
        self.checkBasis(cmb)

    def test_gm_basis(self):
        #mx_dim = sum([ int(np.sqrt(d)) for d in self.stateSpaceDims])
        cmb = Basis.cast('gm', self.stateSpaceDims)
        self.checkBasis(cmb)


class JamiolkowskiOpsTester(BaseCase):
    def setUp(self):
        self.gm = Basis.cast('gm', 4)
        self.pp = Basis.cast('pp', 4)
        self.std = Basis.cast('std', 4)
        self.mxGM = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0,-1, 0, 0],
                              [0, 0, 0, 1]], 'complex')

        self.mxStd = bt.change_basis(self.mxGM, self.gm, self.std)
        self.mxPP = bt.change_basis(self.mxGM, self.gm, self.pp)

    def test_sum_of_negative_choi_evals(self):
        sumOfNeg = j.abs_sum_of_negative_choi_eigenvalues(std1Q.target_model())
        self.assertAlmostEqual(sumOfNeg, 0.0)

        sumOfNegWt = j.abs_sum_of_negative_choi_eigenvalues(std1Q.target_model(), {'Gx': 1.0, 'Gy': 0.5})
        self.assertAlmostEqual(sumOfNegWt, 0.0)

        sumsOfNeg = j.abs_sums_of_negative_choi_eigenvalues(std1Q.target_model())
        self.assertArraysAlmostEqual(sumsOfNeg, np.zeros(3, 'd'))  # 3 gates in std.target_model()

        magsOfNeg = j.magnitudes_of_negative_choi_eigenvalues(std1Q.target_model())
        self.assertArraysAlmostEqual(magsOfNeg, np.zeros(12, 'd'))  # 3 gates * 4 evals each = 12

        # Test magnitudes_of_negative_choi_eigenvalues with a genuinely non-CP model,
        # verifying that it matches eigenvalues computed directly from the Choi matrices.
        non_cp_model = std1Q.target_model().copy()
        rng = np.random.default_rng(42)
        for gl, gate in non_cp_model.operations.items():
            dense = gate.to_dense().copy()
            dense[1:, :] += rng.normal(scale=0.1, size=dense[1:, :].shape)
            non_cp_model.operations[gl] = dense

        expected_mags = []
        std_basis = non_cp_model.basis
        simple_std = std_basis.create_simple_equivalent('std')
        for gl, gate in non_cp_model.operations.items():
            J = j.jamiolkowski_iso(gate.to_dense('minimal'), std_basis, choi_mx_basis=simple_std)
            from pygsti.tools import matrixtools as mt
            evals = mt.eigenvalues(J, assume_hermitian=True)
            for ev in evals:
                expected_mags.append(-ev if ev < 0 else 0.0)

        mags = j.magnitudes_of_negative_choi_eigenvalues(non_cp_model)
        self.assertArraysAlmostEqual(np.array(mags), np.array(expected_mags))

    def test_deprecated_aliases_warn_and_delegate(self):
        model = std1Q.target_model()
        gate_mx = next(iter(model.operations.values())).to_dense()
        cases = [
            (j.sum_of_negative_choi_eigenvalues_gate,
             j.abs_sum_of_negative_choi_eigenvalues_gate, (gate_mx, model.basis)),
            (j.sum_of_negative_choi_eigenvalues,
             j.abs_sum_of_negative_choi_eigenvalues, (model,)),
            (j.sums_of_negative_choi_eigenvalues,
             j.abs_sums_of_negative_choi_eigenvalues, (model,)),
        ]
        for deprecated, replacement, args in cases:
            with self.subTest(deprecated=deprecated.__name__):
                with self.assertWarns(pyGSTiDeprecationWarning):
                    got = deprecated(*args)
                self.assertArraysAlmostEqual(np.atleast_1d(got),
                                             np.atleast_1d(replacement(*args)))

    def test_fast_jamiolkowski_iso(self):
        choiStd = j.jamiolkowski_iso(self.mxStd, self.std, self.std)
        fastChoiStd = j.fast_jamiolkowski_iso_std(self.mxStd, self.std)
        fastChoiStd2 = j.fast_jamiolkowski_iso_std(self.mxGM, self.gm)
        fastChoiStd3 = j.fast_jamiolkowski_iso_std(self.mxPP, self.pp)

        self.assertArraysAlmostEqual(choiStd, fastChoiStd) # Test against standard call
        self.assertArraysAlmostEqual(fastChoiStd, fastChoiStd2)
        self.assertArraysAlmostEqual(fastChoiStd, fastChoiStd3)

        fastGateStd = j.fast_jamiolkowski_iso_std_inv(fastChoiStd, self.std)
        fastGateGM = j.fast_jamiolkowski_iso_std_inv(fastChoiStd, self.gm)
        fastGatePP = j.fast_jamiolkowski_iso_std_inv(fastChoiStd, self.pp)

        self.assertArraysAlmostEqual(fastGateStd, self.mxStd)
        self.assertArraysAlmostEqual(fastGateGM, self.mxGM)
        self.assertArraysAlmostEqual(fastGatePP, self.mxPP)

    def test_fast_jamiolkowski_iso_std_unnormalized(self):
        fastChoiStdUnnorm = j.fast_jamiolkowski_iso_std(self.mxGM, self.gm, normalized=False)
        self.assertAlmostEqual(np.trace(fastChoiStdUnnorm), 2.0)
        self.assertArraysAlmostEqual(fastChoiStdUnnorm, 2.0 * j.fast_jamiolkowski_iso_std(self.mxGM, self.gm))

        fastGateGMUnnorm = j.fast_jamiolkowski_iso_std_inv(fastChoiStdUnnorm, self.gm, normalized=False)
        self.assertArraysAlmostEqual(fastGateGMUnnorm, self.mxGM)

    def test_jamiolkowski_iso(self):
        choiStd = j.jamiolkowski_iso(self.mxStd, self.std, self.std)
        choiStd2 = j.jamiolkowski_iso(self.mxGM, self.gm, self.std)
        choiStd3 = j.jamiolkowski_iso(self.mxPP, self.pp, self.std)

        choiGM = j.jamiolkowski_iso(self.mxStd, self.std, self.gm)
        choiGM2 = j.jamiolkowski_iso(self.mxGM, self.gm, self.gm)
        choiGM3 = j.jamiolkowski_iso(self.mxPP, self.pp, self.gm)

        choiPP = j.jamiolkowski_iso(self.mxStd, self.std, self.pp)
        choiPP2 = j.jamiolkowski_iso(self.mxGM, self.gm, self.pp)
        choiPP3 = j.jamiolkowski_iso(self.mxPP, self.pp, self.pp)

        # Reconstruct standard matrix: GS = sum_ij Jij (BSi x BSj^*)
        mxReconstruct = np.zeros_like(self.mxStd)
        M = mxReconstruct.shape[0]
        dmDim = int(round(np.sqrt(self.mxStd.shape[0]))) # Will need to undo renormalization
        Bs = self.std.elements
        for i in range(M):
            for k in range(M):
                term = choiStd[i, k] * np.kron(Bs[i], np.conjugate(Bs[k]))
                mxReconstruct += term * dmDim
        self.assertArraysAlmostEqual(mxReconstruct, self.mxStd)

        self.assertArraysAlmostEqual(choiStd, choiStd2)
        self.assertArraysAlmostEqual(choiStd, choiStd3)
        self.assertArraysAlmostEqual(choiGM, choiGM2)
        self.assertArraysAlmostEqual(choiGM, choiGM3)
        self.assertArraysAlmostEqual(choiPP, choiPP2)
        self.assertArraysAlmostEqual(choiPP, choiPP3)

        gateStd = j.jamiolkowski_iso_inv(choiStd, self.std, self.std)
        gateStd2 = j.jamiolkowski_iso_inv(choiGM, self.gm, self.std)
        gateStd3 = j.jamiolkowski_iso_inv(choiPP, self.pp, self.std)

        gateGM = j.jamiolkowski_iso_inv(choiStd, self.std, self.gm)
        gateGM2 = j.jamiolkowski_iso_inv(choiGM, self.gm, self.gm)
        gateGM3 = j.jamiolkowski_iso_inv(choiPP, self.pp, self.gm)

        gatePP = j.jamiolkowski_iso_inv(choiStd, self.std, self.pp)
        gatePP2 = j.jamiolkowski_iso_inv(choiGM, self.gm, self.pp)
        gatePP3 = j.jamiolkowski_iso_inv(choiPP, self.pp, self.pp)

        self.assertArraysAlmostEqual(gateStd, self.mxStd)
        self.assertArraysAlmostEqual(gateStd2, self.mxStd)
        self.assertArraysAlmostEqual(gateStd3, self.mxStd)

        self.assertArraysAlmostEqual(gateGM, self.mxGM)
        self.assertArraysAlmostEqual(gateGM2, self.mxGM)
        self.assertArraysAlmostEqual(gateGM3, self.mxGM)

        self.assertArraysAlmostEqual(gatePP, self.mxPP)
        self.assertArraysAlmostEqual(gatePP2, self.mxPP)
        self.assertArraysAlmostEqual(gatePP3, self.mxPP)


class JamiolkowskiCVXPYTester(BaseCase):
    """`jamiolkowski_iso` and `jamiolkowski_iso_inv` are documented to accept and
    return cvxpy Expressions, so that Choi-matrix constraints can be written directly
    against a superoperator variable (or vice versa) inside an SDP.  Only the forward
    map actually implemented it; these tests pin down that both directions do."""

    def setUp(self):
        self.bases = [Basis.cast(nm, 4) for nm in ('std', 'gm', 'pp')]
        self.mx = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]], 'd')

    @needs_cvxpy
    def test_iso_inv_accepts_expression(self):
        """Symbolic and numeric evaluation must agree exactly, in every basis pair."""
        import cvxpy as cp
        for op_basis in self.bases:
            for choi_basis in self.bases:
                for normalized in (True, False):
                    mx = bt.change_basis(self.mx, self.bases[1], op_basis)
                    choi = j.jamiolkowski_iso(mx, op_basis, choi_basis, normalized=normalized)

                    param = cp.Parameter(choi.shape, complex=np.iscomplexobj(choi))
                    param.value = choi
                    symbolic = j.jamiolkowski_iso_inv(param, choi_basis, op_basis,
                                                      normalized=normalized)
                    numeric = j.jamiolkowski_iso_inv(choi, choi_basis, op_basis,
                                                     normalized=normalized)

                    self.assertIsInstance(symbolic, cp.Expression)
                    self.assertEqual(symbolic.shape, numeric.shape)
                    # a real op basis must give a real Expression, not a complex one
                    # with a zero imaginary part -- otherwise every downstream
                    # constraint silently doubles in size.
                    self.assertEqual(symbolic.is_complex(), not op_basis.real)
                    self.assertArraysAlmostEqual(symbolic.value, numeric)
                    self.assertArraysAlmostEqual(numeric, mx)

    @needs_cvxpy
    def test_iso_inv_composes_with_iso(self):
        """iso_inv(iso(X)) is the identity map on a symbolic superoperator."""
        import cvxpy as cp
        x = cp.Variable((4, 4))
        roundtrip = j.jamiolkowski_iso_inv(j.jamiolkowski_iso(x, 'pp', 'gm'), 'gm', 'pp')
        x.value = self.mx
        self.assertArraysAlmostEqual(roundtrip.value, self.mx)

    @needs_cvxpy
    def test_iso_inv_in_sdp(self):
        """End-to-end: optimize over a Choi variable with constraints on the superop."""
        import cvxpy as cp
        target = bt.change_basis(self.mx, self.bases[1], self.bases[2])
        choi = cp.Variable((4, 4), hermitian=True)
        superop = j.jamiolkowski_iso_inv(choi, 'pp', 'pp')
        prob = cp.Problem(cp.Minimize(cp.norm(superop - target, 'fro')),
                          [choi >> 0, cp.trace(choi) == 1, superop[0, :] == np.eye(4)[0]])
        prob.solve(solver='CLARABEL')
        self.assertEqual(prob.status, 'optimal')
        # the recovered superop must be CPTP and its Choi matrix must be the solution
        self.assertArraysAlmostEqual(superop.value[0, :], np.eye(4)[0])
        self.assertArraysAlmostEqual(j.jamiolkowski_iso(superop.value, 'pp', 'pp'), choi.value)
        self.assertGreater(np.linalg.eigvalsh(choi.value).min(), -1e-7)

    @needs_cvxpy
    def test_iso_inv_block_structured_basis(self):
        """The 'contract' branch of `resize_std_mx` must stay symbolic too."""
        import cvxpy as cp
        kite = Basis.cast('std', [4, 1])
        mx = np.eye(kite.dim)
        choi = j.jamiolkowski_iso(mx, kite, 'std')
        param = cp.Parameter(choi.shape, complex=np.iscomplexobj(choi))
        param.value = choi
        symbolic = j.jamiolkowski_iso_inv(param, 'std', kite)
        self.assertIsInstance(symbolic, cp.Expression)
        self.assertArraysAlmostEqual(symbolic.value, j.jamiolkowski_iso_inv(choi, 'std', kite))

    def test_is_cvxpy_expression_on_plain_arrays(self):
        self.assertFalse(bt.is_cvxpy_expression(np.eye(4)))
        self.assertFalse(bt.is_cvxpy_expression(1.0))
        self.assertFalse(bt.is_cvxpy_expression(None))


class JamiolkowskiLeakageBasisTester(BaseCase):
    """
    Tests `jamiolkowski_iso` with a leakage tensor-product basis (e.g., pp ⊗ l2p1).
    This exercises the fallback branch in `jamiolkowski_iso` when `create_simple_equivalent`
    fails for bases with no same-name builtin equivalent.
    """

    def setUp(self):
        # 36-dimensional qubit x qutrit tensor product basis
        self.basis = TensorProdBasis((BuiltinBasis('pp', 4), BuiltinBasis('l2p1', 9)))
        self.dm_dim = 6

        # Build a 6x6 unitary (qubit flip x qutrit rotation mixing level 1 and 2)
        X_qubit = np.array([[0, 1], [1, 0]], dtype=complex)
        theta = 0.3
        R_qutrit = np.eye(3, dtype=complex)
        R_qutrit[1, 1] = np.cos(theta)
        R_qutrit[2, 2] = np.cos(theta)
        R_qutrit[1, 2] = -np.sin(theta)
        R_qutrit[2, 1] = np.sin(theta)
        U = np.kron(X_qubit, R_qutrit)

        # Convert the unitary to a superoperator in our tensor-product basis
        self.superop = unitary_to_superop(U, self.basis)

    def test_simple_equivalent_unavailable(self):
        # Precondition check: the leakage tensor-product basis does not have a same-name
        # builtin equivalent, which raises AssertionError during create_simple_equivalent().
        with self.assertRaises(AssertionError):
            self.basis.create_simple_equivalent()

    def test_iso_falls_back_to_basis_elements(self):
        # Exercises the fallback: uses choi_mx_basis.elements directly.
        J = j.jamiolkowski_iso(self.superop, self.basis, self.basis)

        self.assertEqual(J.shape, (36, 36))
        self.assertAlmostEqual(np.trace(J), 1.0)
        self.assertArraysAlmostEqual(J, J.conj().T)
        self.assertGreater(np.linalg.eigvalsh(J).min(), -1e-10)

        # Verification of correctness via reconstruction identity:
        # S_std = sum_ik J_ik * d * kron(B_i, conj(B_k))
        B = self.basis.elements
        reconstructed = np.zeros_like(self.superop, dtype=complex)
        for i in range(36):
            for k in range(36):
                reconstructed += J[i, k] * np.kron(B[i], np.conjugate(B[k])) * self.dm_dim

        # Check against standard process matrix representation
        simple_std = self.basis.create_simple_equivalent('std')
        expected_std = bt.change_basis(self.superop, self.basis, simple_std)
        self.assertArraysAlmostEqual(reconstructed, expected_std)

    def test_iso_unnormalized(self):
        J_unnorm = j.jamiolkowski_iso(self.superop, self.basis, self.basis, normalized=False)
        self.assertAlmostEqual(np.trace(J_unnorm), float(self.dm_dim))
        self.assertArraysAlmostEqual(J_unnorm, self.dm_dim * j.jamiolkowski_iso(self.superop, self.basis, self.basis))

    def test_iso_eigenvalues_match_fast_std(self):
        # Choi eigenvalues must be basis-independent.
        # fast_jamiolkowski_iso_std does not call create_simple_equivalent on the target basis,
        # so it avoids the fallback pathway but must produce the exact same eigenvalues.
        J = j.jamiolkowski_iso(self.superop, self.basis, self.basis)
        Jstd = j.fast_jamiolkowski_iso_std(self.superop, self.basis)

        ev = np.sort(np.linalg.eigvalsh(J))
        ev_std = np.sort(np.linalg.eigvalsh(Jstd))
        self.assertArraysAlmostEqual(ev, ev_std)

    def test_iso_inv_rejects_leakage_basis(self):
        # jamiolkowski_iso_inv lacks the try-except fallback in line 207,
        # so it currently raises AssertionError when given the leakage basis.
        # This test documents today's behavior.
        J = j.jamiolkowski_iso(self.superop, self.basis, self.basis)
        with self.assertRaises(AssertionError):
            j.jamiolkowski_iso_inv(J, self.basis, self.basis)

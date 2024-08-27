from unittest import mock, TestCase

import numpy as np
import pytest
import scipy.linalg as la
from pygsti.tools.su2tools import batch_normal_expm_1jscales, SU2, Spin72, distance_mod_phase


def reconstruct_eigendecomp(eigvals, eigvecs, tol=1e-14):
    n = eigvals.size
    assert la.norm(eigvecs @ eigvecs.T.conj() - np.eye(n)) <= np.sqrt(n) * tol 
    return eigvecs @ np.diag(eigvals) @ eigvecs.T.conj()


class TestFreeFunctions(TestCase):

    def test_batch_normal_expm_1jscales(self):
        pass


class BaseSU2Tests(TestCase):

    RELTOL = 1e-14

    def _test_matval_funcs_agree(self, args, our_handle, ref_handle):
        for arg in args:
            actual = our_handle(arg)
            expect = ref_handle(arg)
            abstol = self.RELTOL * max(la.norm(actual), la.norm(expect))
            self.assertLessEqual(la.norm(actual - expect), abstol)
        return

    def _test_expmiJx_single(self, su2class):
        thetas = np.linspace(0, 4 * np.pi, num=10)
        def our_handle(t):
            return su2class.expm_iJx(t)[0, :, :]
        def ref_handle(t):
            return la.expm(1j * t * su2class.Jx)
        self._test_matval_funcs_agree(thetas, our_handle, ref_handle)

    def _test_expmiJy_single(self, su2class):
        thetas = np.linspace(0, 4 * np.pi, num=10)
        def our_handle(t):
            return su2class.expm_iJy(t)[0, :, :]
        def ref_handle(t):
            return la.expm(1j * t * su2class.Jy)
        self._test_matval_funcs_agree(thetas, our_handle, ref_handle)
        

    def _test_static_variables(self, su2class):
        abstol = 2*self.RELTOL
        recon_Jx = reconstruct_eigendecomp(su2class.eigJx, su2class.VJx)
        self.assertLessEqual(la.norm(recon_Jx - su2class.Jx), abstol)
        recon_Jy = reconstruct_eigendecomp(su2class.eigJy, su2class.VJy)
        self.assertLessEqual(la.norm(recon_Jy - su2class.Jy), abstol)
        return


class TestSU2(BaseSU2Tests):

    def test_static_variables(self):
        return self._test_static_variables(SU2)
    
    def test_expmiJx_single(self):
        return self._test_expmiJx_single(SU2)

    def test_expmiJy_single(self):
        return self._test_expmiJy_single(SU2)

    def test_unitary_angles_unitary(self):
        num_trials = 20
        gen = np.random.default_rng(0)
        def make_unitary():
            temp1 = gen.standard_normal(size=(2,2))
            temp2 = gen.standard_normal(size=(2,2))
            temp3 = temp1 + 1j*temp2
            U = la.qr(temp3)[0]
            U /= np.sqrt(la.det(U))
            return U
        # TODO: probably need to make special unitary, not just unitary.
        mats = [make_unitary() for _ in range(num_trials)]
        mats.append(np.eye(2))
        mats.append(np.eye(2)[::-1,:])  # reverse the order of rows
        for U in mats:
            a,b,g = SU2.angles_from_2x2_unitary(U)
            U_recon = SU2.unitaries_from_angles(a,b,g)[0]
            dist = distance_mod_phase(U, U_recon)
            self.assertLessEqual(dist, 2*self.RELTOL)
    
    def test_angles_unitary_angles(self):
        num_trials = 10
        np.random.seed(0)
        angles = np.column_stack(SU2.random_euler_angles(num_trials))
        for abg in angles:
            U = SU2.unitaries_from_angles(*abg.tolist())[0]
            abg_recon = np.array(SU2.angles_from_2x2_unitary(U))
            self.assertLessEqual(la.norm(abg - abg_recon), self.RELTOL)
        
    def test_composition_inverse(self):
        np.random.seed(0)
        lengths = np.arange(1,16,2)
        for sequence_len in lengths:
            angles = np.column_stack(SU2.random_euler_angles(sequence_len))
            a,b,g = angles.T
            char_angles = (a[0],b[0],g[0])
            inv_angles = np.column_stack(SU2.composition_inverse(a[1:], b[1:], g[1:]))
            comp_to_char_angles = np.block([
                [  angles  ],
                [inv_angles]
            ])
            U_char_expect = SU2.composition_asmatrix(comp_to_char_angles)
            U_char_actual = SU2.unitaries_from_angles(*char_angles)[0]
            discrepency = distance_mod_phase(U_char_actual, U_char_expect)
            self.assertLessEqual(discrepency, self.RELTOL)


class TestSpin72(BaseSU2Tests):

    RELTOL = 1e-13

    def test_static_variables(self):
        return self._test_static_variables(Spin72)
    
    def test_expmiJx_single(self):
        return self._test_expmiJx_single(Spin72)

    def test_expmiJy_single(self):
        return self._test_expmiJy_single(Spin72)

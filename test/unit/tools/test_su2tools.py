from unittest import TestCase
from pygsti.tools import unitary_to_std_process_mx
import numpy as np
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

    thetas = np.linspace(0, 4 * np.pi, num=10)

    def _test_matval_funcs_agree(self, args, our_handle, ref_handle):
        for arg in args:
            actual = our_handle(arg)
            expect = ref_handle(arg)
            abstol = self.RELTOL * max(la.norm(actual), la.norm(expect))
            self.assertLessEqual(la.norm(actual - expect), abstol)
        return

    def _test_expmiJx_single(self, su2class):
        def our_handle(t):
            return su2class.expm_iJx(t)[0, :, :]
        def ref_handle(t):
            return la.expm(1j * t * su2class.Jx)
        self._test_matval_funcs_agree(self.thetas, our_handle, ref_handle)

    def _test_expmiJx_batch(self, su2class):
        actual = su2class.expm_iJx(self.thetas)
        expect = np.array([
            la.expm(1j * t * su2class.Jx) for t in self.thetas
        ])
        abstol = self.RELTOL * max(la.norm(actual), la.norm(expect))
        self.assertLessEqual(la.norm(actual - expect), abstol)
        pass

    def _test_expmiJy_single(self, su2class):
        def our_handle(t):
            return su2class.expm_iJy(t)[0, :, :]
        def ref_handle(t):
            return la.expm(1j * t * su2class.Jy)
        self._test_matval_funcs_agree(self.thetas, our_handle, ref_handle)
        

    def _test_expmiJy_batch(self, su2class):
        actual = su2class.expm_iJy(self.thetas)
        expect = np.array([
            la.expm(1j * t * su2class.Jy) for t in self.thetas
        ])
        abstol = self.RELTOL * max(la.norm(actual), la.norm(expect))
        self.assertLessEqual(la.norm(actual - expect), abstol)
        pass

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

    def test_expmiJx_batch(self):
        return self._test_expmiJx_batch(SU2)

    def test_expmiJy_batch(self):
        return self._test_expmiJy_batch(SU2)

    def test_angles_from_2x2_unitaries(self):
        angles_in = np.row_stack(SU2.random_euler_angles(20))
        Us = SU2.unitaries_from_angles(*angles_in)

        def reference_from_kevin(R):
            # Compute the euler angles from the SU(2) elements
            beta = 2*np.arccos(np.real(np.sqrt(R[0,0]*R[1,1])))

            alpha = np.angle(-1.j*R[0,0]*R[0,1]/(np.sin(beta/2)*np.cos(beta/2)))
            if alpha < 0:
                alpha += 2*np.pi
            
            gamma = np.angle(-1.j*R[0,0]*R[1,0]/(np.sin(beta/2)*np.cos(beta/2)))
            if gamma < 0:
                gamma += 2*np.pi
            if np.isclose(np.exp(1.j*(alpha+gamma)/2)*np.cos(beta/2) / R[0,0], -1):
                gamma += 2*np.pi

            return alpha, beta, gamma
        
        angles_out_expect = np.column_stack([reference_from_kevin(U) for U in Us])
        angles_out_actual = np.row_stack(SU2.angles_from_2x2_unitaries(Us))
        self.assertLessEqual(la.norm(angles_out_expect - angles_out_actual), self.RELTOL * 10)
        return
    
    def test_random_euler_angles(self):
        def reference_from_paper(N: int):
            """
            See page 11 of https://arxiv.org/abs/math-ph/0609050 for sampling
            from the Haar distribution. We multiply such samples by a global
            phase in order to get the uniform distribution on special-unitary
            matrices.
            """
            G = np.random.randn(N,2,2) + 1j*np.random.randn(N,2,2)
            Q,R = np.linalg.qr(G)
            r = np.row_stack([np.diag(rr) for rr in R])
            r /= np.abs(r)
            Q_out = Q * r[:,np.newaxis,:]
            # ^ That's now Haar distribured, but not special-unitary distributed.
            d = np.linalg.det(Q_out)
            Q_out = Q_out * np.sqrt(d.conj())[:, np.newaxis, np.newaxis]
            return SU2.angles_from_2x2_unitaries(Q_out)
        
        """
        np.random.seed(0)
        old_angles = np.row_stack(Spin72.random_euler_angles(N))
        unit_det = True
        old_unitaries = Spin72.unitaries_from_angles(*old_angles)
        new_angles = np.row_stack(new_random_euler_angles(N, unit_det, 0))
        fig, axs = plt.subplots(1,3)
        angle_names = ['alpha', 'beta', 'gamma']
        for i in range(3):
            if i == 2:
                axs[i].ecdf(old_angles[i],color='r',label='unif[0,2π], acos(unif[-1,1]), unif[0,4π]')
                axs[i].ecdf(new_angles[i],color='b',label='Haar + unitary_to_angles')
                axs[i].set_title(angle_names[i])
            axs[i].ecdf(old_angles[i],color='r')
            axs[i].ecdf(new_angles[i],color='b')
            axs[i].set_title(angle_names[i])
        fig.suptitle(f'Empirical CDFs for Euler angles of n=10,000 SU(2) elements')
        fig.legend()
        fig.tight_layout()
        plt.show()
        """

        #TODO: run KS tests for each of the three angles, comparing against SU2.random_euler_angles.
        return


class TestSpin72(BaseSU2Tests):

    RELTOL = 1e-13

    def test_static_variables(self):
        # First we check the static variables that also appear in the base SU2 class.
        self._test_static_variables(Spin72)
        # Next, check unitarity of C:= Spin72.superop_stdmx_cob
        C =  Spin72.superop_stdmx_cob
        discrepency = la.norm(C @ C.T.conj() - np.eye(64))
        self.assertLessEqual(discrepency, self.RELTOL)
        # Finally, we check if C actually block diagonalizes random SU2 elements
        # in the superop-of-spin72 representation.
        np.random.seed(0)
        aa = np.column_stack(Spin72.random_euler_angles(5))
        Us = Spin72.unitaries_from_angles(aa[:,0], aa[:,1], aa[:,2])
        for U in Us:
            V = unitary_to_std_process_mx(U)
            W = C @ V @ C.T.conj()
            start = 0
            for b_sz in Spin72.irrep_block_sizes:
                stop = start + b_sz
                block = W[start:stop, start:stop].copy()
                W[start:stop, start:stop] -= block
                start = stop
            self.assertLessEqual(la.norm(W), self.RELTOL)
        return
    
    def test_expmiJx_single(self):
        return self._test_expmiJx_single(Spin72)

    def test_expmiJy_single(self):
        return self._test_expmiJy_single(Spin72)

    def test_angles2irrepchars(self):
        # use the fact that we can compute character functions in one
        # of two ways. We can either take the traces of the blocks of the
        # block-diagonalized version of the superoperator, or we can use the nice
        # function implied by Robin's note. Obviously the latter is faster.
        # The former is nice because the block-diagonal structure can plausibly 
        # be verified by testing a single random SU(2) element, and the diagonalizer's
        # unitarity can be checked cheaply.
        np.random.seed(0)
        aa = np.column_stack(Spin72.random_euler_angles(5))
        Us = Spin72.unitaries_from_angles(aa[:,0], aa[:,1], aa[:,2])
        for i,U in enumerate(Us):
            expect = Spin72.all_characters_from_unitary(U)
            actual = Spin72.angles2irrepchars(aa[i,:])
            discrepency = la.norm(actual - expect)
            self.assertLessEqual(discrepency, 64*self.RELTOL)
        return
    
    def test_new_angles2irrepchars(self):
        # use the fact that we can compute character functions in one
        # of two ways. We can either take the traces of the blocks of the
        # block-diagonalized version of the superoperator, or we can use the nice
        # function implied by Robin's note. Obviously the latter is faster.
        # The former is nice because the block-diagonal structure can plausibly 
        # be verified by testing a single random SU(2) element, and the diagonalizer's
        # unitarity can be checked cheaply.
        np.random.seed(0)
        aa = np.column_stack(Spin72.random_euler_angles(5))
        Us = Spin72.unitaries_from_angles(aa[:,0], aa[:,1], aa[:,2])
        for i,U in enumerate(Us):
            expect = Spin72.all_characters_from_unitary(U)
            actual = Spin72.characters_from_euler_angles(aa[i,:])
            discrepency = la.norm(actual - expect)
            self.assertLessEqual(discrepency, 64*self.RELTOL)
        return

    def test_expmiJx_batch(self):
        return self._test_expmiJx_batch(Spin72)

    def test_expmiJy_batch(self):
        return self._test_expmiJy_batch(Spin72)

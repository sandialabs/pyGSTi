"""
Tools for working with representations of the special unitary group, SU(2).
"""
import numpy as np
from numpy import sqrt
import pygsti.modelmembers
import scipy.linalg as la
import scipy.sparse as spar
from pygsti.tools.optools import unitary_to_superop
from pygsti.baseobjs import basisconstructors as bcons
from pygsti.baseobjs import basis as _basis
from typing import List, Tuple
import warnings
import scipy as sp
from tqdm import tqdm


def check_su2_generators(Jx, Jy, Jz):
    tol = 1e-14
    def bracket(a,b): return a @ b - b @ a
    def is_zero(arg): return np.all(np.abs(arg) <= tol)

    # The necessary relationships between these generators are ambiguous
    # up to a scale factor in the definition of the Levi-Civita symbol.
    # 
    # We'll start by finding the only scale that could possibly result in
    # these generators satisfying the necessary relationships.
    abxy = np.abs(bracket(Jx, Jy))
    aJz  = np.abs(Jz)
    ratios = abxy[aJz > 0]/ aJz[aJz > 0]
    scale  = ratios[0]
    assert np.all(np.isclose(ratios, scale))

    diff = bracket(Jx, Jy) - 1j*scale*Jz
    assert is_zero(diff), f'\n{str(diff)}\n\tis not zero up to tolerance {tol}.'

    diff = bracket(Jx, Jz) + 1j*scale*Jy
    assert is_zero(diff), f'\n{str(diff)}\n\tis not zero up to tolerance {tol}.'

    diff = bracket(Jy, Jz) - 1j*scale*Jx
    assert is_zero(diff), f'\n{str(diff)}\n\tis not zero up to tolerance {tol}.'
    return

def batch_normal_expm_1jscales(eigvecs, eigvals, scales):
    """
    eigvecs is a square unitary matrix of order n, and eigvals is a vector of length n.
    We return a numpy array of that's equivalent to

        np.array([
            eigvecs @ np.diag(1j * s * eigvals) @ eigvecs.T.conj() for s in scales
        ])
    
    but that's constructed more efficiently.
    
    Note: The word "normal" appears in this function name because (eigvals, eigvecs)
    implicitly define the normal operator eigvecs @ diag(eigvals) @ eigvecs.T.conj().
    """
    assert eigvals.ndim == 1
    n = eigvals.size
    assert eigvecs.shape == (n, n)

    exp_eig = np.exp(1j * scales[:, np.newaxis] * eigvals[np.newaxis, :])
    eigvecs_tile = np.broadcast_to(eigvecs, (scales.size, n, n))
    left = eigvecs_tile * exp_eig[:, np.newaxis, :]
    batch_out = left @ eigvecs.T.conj()
    return batch_out

def batch_eigvals_2x2(mats):
    """
    If M is a 2-by-2 matrix, then its eigenvalues are a ± √(a² - d), where a = mean(diag(M)) and d = det(M).

    Given an array mats of shape (dim,2,2), we return a dim-by-2 array whose rows are the eigenvalues of
    the matrices mat[i,:,:]. This function's output is only accurate to the square-root of the working precision
    (so for IEEE double precision that comes out to about 1e-8).
    """
    panels = mats.shape[0]
    assert mats.ndim == 3
    assert mats.shape == (panels, 2, 2)
    a = np.trace(mats, axis1=1, axis2=2)/2
    d = mats[:,0,0]*mats[:,1,1] - mats[:,0,1]*mats[:,1,0]
    radical = np.sqrt(a**2 - d)
    out = np.zeros(panels, 2)
    out[:,0] = a + radical
    out[:,1] = a - radical
    return out

def eign(mat, tol=1e-12):
    """
    Return the eigendecomposition of a normal operator "mat": 

        mat = V @ np.diag(eigvals) @ V.T.conj().
    """
    T, V = la.schur(mat)
    offdiag_norm = la.norm(T[np.triu_indices_from(T, 1)])
    if offdiag_norm > tol:
        raise ValueError(
            f'Off-diagonal of triangular factor T from Schur decomposition had norm {offdiag_norm}, '
            f'which exceeds tolerance of {tol}.'
        )
    eigvals = np.diag(T)
    return eigvals, V


class SU2:

    Jx = bcons.sigmax / 2
    Jy = bcons.sigmay / 2
    Jz = bcons.sigmaz / 2
    check_su2_generators(Jx, Jy, Jz)
    eigJx =  np.array([0.5, -0.5])
    eigJy =  np.array([0.5, -0.5])
    VJx = np.array([[1, 1], [1,   -1]]) / np.sqrt(2)
    VJy = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
    # VJx and VJy are eigenbases for Jx and Jy, respectively,
    # with corresponding eigenvalues in eigJx and eigJy.

    @staticmethod
    def random_euler_angles(size=1):
        # Construct the Euler angles for a Haar randomly sampled element of SU(2)
        alpha = np.random.uniform(low=0,high=2*np.pi, size=size)
        beta = np.arccos(np.random.uniform(low=-1,high=1, size=size))
        gamma = np.random.uniform(low=0,high=4*np.pi, size=size)
        return alpha, beta, gamma

    @classmethod
    def random_unitaries(cls,size=1):
        # Construct a Haar random SU(2) element in terms of ZXZ Euler angles
        alpha, beta, gamma = cls.random_euler_angles(size)
        Rs = cls.unitaries_from_angles(alpha,beta,gamma)
        return Rs
    
    @classmethod
    def angles_from_unitary(cls, R, tol=1e-10):
        
        def log_unitary(U):
            T,Z = la.schur(U, output='complex')
            eigvals = np.diag(T)
            assert la.norm(T[np.triu_indices_from(T,1)]) < tol
            log_eigvals = np.log(eigvals)
            log_U = (Z * log_eigvals[np.newaxis, :]) @ Z.T.conj()
            U_recover_mine      = la.expm(log_U)
            U_recover_reference = la.expm(la.logm(U))
            assert la.norm(U_recover_mine - U_recover_reference) < tol
            assert la.norm(U_recover_mine - U) < tol
            return log_U
        
        T = log_unitary(R) / 1j
        b = T.ravel()
        A = np.column_stack([cls.Jx.ravel(), cls.Jy.ravel(), cls.Jz.ravel()])
        coeffs, residual = la.lstsq(A, b)[:2]
        assert residual < tol and la.norm(coeffs.imag) < tol
        
        u,v,w = coeffs.real
        R2x2 = la.expm(1j*(u*SU2.Jx + v*SU2.Jy + w*SU2.Jz))
        a, b, g = SU2.angles_from_2x2_unitaries(R2x2)
        return a, b, g

    @classmethod
    def expm_iJx(cls, theta):
        if not isinstance(theta, np.ndarray):
            batch = cls.expm_iJx(np.array(theta))
            return batch[0]
        return batch_normal_expm_1jscales(cls.VJx, cls.eigJx, theta)
    
    @classmethod
    def expm_iJy(cls, theta):
        if not isinstance(theta, np.ndarray):
            batch = cls.expm_iJy(np.array(theta))
            return batch[0]
        return batch_normal_expm_1jscales(cls.VJy, cls.eigJy, theta)

    @staticmethod
    def angles_from_2x2_unitaries(R):
        if R.ndim != 3:
            R = R[np.newaxis,:,:]
            a,b,g = SU2.angles_from_2x2_unitaries(R)
            return a[0], b[0], g[0]

        assert R.shape[1:] == (2, 2)
        # Compute the euler angles from the SU(2) elements
        beta = 2*np.arccos(np.real(np.sqrt(R[:,0,0]*R[:,1,1])))

        beta = np.atleast_1d(beta)
        sbh  = np.sin(beta/2)
        cbh  = np.cos(beta/2)
        scbh = sbh * cbh

        div_zero_tol = 1e-14
        safe = np.abs(scbh) >= div_zero_tol**2
        alpha = np.zeros_like(beta)
        gamma = np.zeros_like(beta) 
        alpha[safe] = np.angle(-1j * R[safe,0,0] * R[safe,0,1] / scbh[safe])
        gamma[safe] = np.angle(-1j * R[safe,0,0] * R[safe,1,0] / scbh[safe])

        two_pi = 2*np.pi
        alpha += two_pi * (alpha < 0)
        gamma += two_pi * (gamma < 0)
        gamma += two_pi * np.isclose(sbh * np.exp(0.5j*(alpha + gamma)), -R[:,0,0])
        
        return alpha, beta, gamma
    
    @classmethod
    def unitaries_from_angles(cls,alpha,beta,gamma):
        # Construct an element of SU(2) from Euler angles
        array_on_input = isinstance(alpha, np.ndarray)

        alpha = np.atleast_1d(alpha)
        beta  = np.atleast_1d(beta)
        gamma = np.atleast_1d(gamma)

        if not array_on_input:
            assert alpha.size == beta.size == gamma.size == 1

        dJz = np.diag(cls.Jz)
        right = (np.exp(1j * alpha[:, np.newaxis] * dJz[np.newaxis,:]))[:, :, np.newaxis]
        center = cls.expm_iJx(beta)
        left  = (np.exp(1j * gamma[:, np.newaxis] * dJz[np.newaxis,:]))[:, np.newaxis, :]
        out = left * center * right 
        return out

    @staticmethod
    def composition(alphas, betas, gammas):
        Rs = SU2.unitaries_from_angles(alphas.ravel(), betas.ravel(), gammas.ravel())
        R_composed = np.eye(2)
        for R in Rs:
            R_composed = R @ R_composed
        ea = SU2.angles_from_2x2_unitaries(R_composed)
        return ea


    @staticmethod
    def inverse_angles(alpha, beta, gamma):
        R = SU2.unitaries_from_angles(alpha, beta, gamma)
        R = np.transpose(R, axes=[0,2,1]).conj()
        angles = SU2.angles_from_2x2_unitaries(R)
        return angles

    
    @staticmethod
    def rb_circuits_by_angles(N: int, lengths: List[int], seed=0, invert_from=0) -> List[np.ndarray]:
        np.random.seed(seed)
        out = []

        assert 0 not in lengths
        for ell in lengths:
            base_angles = np.column_stack(SU2.random_euler_angles(ell * N))
            # ^ Each block of ell rows in base_angles defines gates for our length-(ell+1) circuits
            #   (the +1'th circuit comes from the inversion gate, which we need to construct).
            base_angles = base_angles.reshape((N, ell, 3))
            # ^ Now each angles[k,:,:] defines a circuit, for k = 0, ..., N-1.

            comps_angles = np.row_stack(SU2.composition(
                base_angles[:, invert_from:, 0],
                base_angles[:, invert_from:, 1],
                base_angles[:, invert_from:, 2]
            ))
            # ^ array of shape (3, N).
            #   Its k-th column gives the Euler angles for the composition of all SU(2) elements
            #   induced from the (ell-invert_from)-by-3 array "base_angles[k, invert_from:, :]"
            last_angles = np.column_stack(SU2.inverse_angles(*comps_angles))
            # ^ array of shape (N, 3)
            #   Its k-th row gives the Euler angles for the SU(2) element that's inverse to  
            #   the SU(2) element whose Euler angles are in comps_angles[k,:].
        
            all_angles = np.zeros((N, ell+1, 3))
            all_angles[:, 0:ell, :] = base_angles
            all_angles[:,   ell, :] = last_angles
        
            out.append(all_angles)
        # out[j][k] == out[j][k,:,:] == (lengths[j]+1)-by-3 array giving the Euler angles for the k-th RB circuit of length lenghts[j].
        return out

    @staticmethod
    def characters_from_2x2_unitaries(U, j, check_numeric=False, tol=1e-8):
        
        j = np.atleast_1d(j)
        if U.ndim == 2:
            U = U[np.newaxis,:,:]
        
        assert U.shape[1] == U.shape[2] == 2
        assert np.all(j >= 0) and np.all(2*j % 1 == 0)

        trs = np.trace(U, axis1=1, axis2=2)
        thetas = 2*np.arccos(trs/2)
        if check_numeric:
            eigs = batch_eigvals_2x2(U)
            thetas_ref = np.real(2*np.log(eigs[:, 0])/1j)
            checks = np.abs(thetas_ref - thetas)
            failed = (checks > tol).nonzero()[0]
            if np.any(failed):
                raise ValueError(
                    f'Checks for correct computation of eigenvalues failed at indices {failed}.\n'
                    f'The difference between the traces of these matrices and expected traces were {checks[failed]}.'    
                )
        thetas = thetas.reshape((-1,1))

        # Compute a matrix whose rows are of length j.size:
        #       np.sum([np.exp(1j*__j*theta) for __j in np.arange(-_j, _j+1)]) for each _j in j 
        # as theta ranges over theta.
        out = np.array([
             np.sum(np.exp(1j * thetas * np.arange(-_j, _j+1)[np.newaxis, :]), axis=1) for _j in j
        ])
        out = np.real(out).T
        assert out.shape == (thetas.size, j.size)
        return out

    
    @staticmethod
    def characters_from_angles(alphas, betas, gammas, j, tol=1e-8):
        U2x2s = SU2.unitaries_from_angles(alphas, betas, gammas)
        characters = np.zeros(len(U2x2s))
        eigs = batch_eigvals_2x2(U2x2s)
        thetas = np.real(2*np.log(eigs[:, 0])/1j)
        trs = np.trace(U2x2s, axis1=1, axis2=2)
        checks = np.abs(trs - 2*np.cos(thetas/2))
        failed = (checks > tol).nonzero()[0]
        if np.any(failed):
            raise ValueError(
                f'Checks for correct computation of eigenvalues failed at indices {failed}.\n'
                f'The difference between the traces of these matrices and expected traces were {checks}.'    
            )
        toreduce = np.array([np.exp(1j*__j*thetas) for __j in np.arange(-j, j+1)])
        characters = np.real(np.sum(toreduce, axis=1))
        return characters


def clebsh_gordan_matrix_spin72():
    """
    Return a numpy array representing the matrix "cjs" generated by the following Mathematica code

            Off[ClebschGordan::phy];
            j = 7/2;
            cjs = Transpose[
                Flatten[ 
                    Table[ Flatten[ Table[ ClebschGordan[{j, m1}, {j, m2}, {J, M}], {J, 0, 2 j}, {M, J, -J, -1}]], {m1, j, -j, -1}, {m2, j, -j, -1}],
                1]
            ];

    Ask Kevin Young what this matrix is supposed to mean.
    """
    rows = np.array([
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  2,  2,
        2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,
        4,  4,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,
        7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9, 10,
        10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
        12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15,
        15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18,
        18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21,
        21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24,
        24, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28,
        28, 28, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30,
        31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33,
        33, 34, 34, 34, 34, 35, 35, 35, 36, 36, 37, 37, 38, 38, 38, 38, 39,
        39, 39, 39, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 42, 42,
        42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44,
        44, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51,
        51, 51, 52, 52, 52, 52, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54,
        55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57,
        57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 60,
        60, 60, 60, 61, 61, 61, 62, 62, 63
    ])
    cols = np.array([
        7, 14, 21, 28, 35, 42, 49, 56,  6, 13, 20, 27, 34, 41, 48,  7, 14,
        21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57,  5, 12, 19, 26,
        33, 40,  6, 13, 20, 34, 41, 48,  7, 14, 21, 28, 35, 42, 49, 56, 15,
        22, 29, 43, 50, 57, 23, 30, 37, 44, 51, 58,  4, 11, 18, 25, 32,  5,
        12, 19, 26, 33, 40,  6, 13, 20, 27, 34, 41, 48,  7, 14, 21, 28, 35,
        42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31,
        38, 45, 52, 59,  3, 10, 17, 24,  4, 11, 25, 32,  5, 12, 19, 26, 33,
        40,  6, 13, 20, 34, 41, 48,  7, 14, 21, 28, 35, 42, 49, 56, 15, 22,
        29, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 52, 59, 39, 46, 53,
        60,  2,  9, 16,  3, 10, 17, 24,  4, 11, 18, 25, 32,  5, 12, 19, 26,
        33, 40,  6, 13, 20, 27, 34, 41, 48,  7, 14, 21, 28, 35, 42, 49, 56,
        15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 45, 52,
        59, 39, 46, 53, 60, 47, 54, 61,  1,  8,  2, 16,  3, 10, 17, 24,  4,
        11, 25, 32,  5, 12, 19, 26, 33, 40,  6, 13, 20, 34, 41, 48,  7, 14,
        21, 28, 35, 42, 49, 56, 15, 22, 29, 43, 50, 57, 23, 30, 37, 44, 51,
        58, 31, 38, 52, 59, 39, 46, 53, 60, 47, 61, 55, 62,  0,  1,  8,  2,
        9, 16,  3, 10, 17, 24,  4, 11, 18, 25, 32,  5, 12, 19, 26, 33, 40,
        6, 13, 20, 27, 34, 41, 48,  7, 14, 21, 28, 35, 42, 49, 56, 15, 22,
        29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 45, 52, 59, 39,
        46, 53, 60, 47, 54, 61, 55, 62, 63
    ])
    vals = np.array([
        0.3535533905932737, -0.3535533905932737,  0.3535533905932737,
        -0.3535533905932737,  0.3535533905932737, -0.3535533905932737,
        0.3535533905932737, -0.3535533905932737,  0.2886751345948129,
        -0.3779644730092272,  0.4225771273642583, -0.4364357804719848,
        0.4225771273642583, -0.3779644730092272,  0.2886751345948129,
        0.5400617248673217, -0.3857583749052298,  0.2314550249431379,
        -0.077151674981046 , -0.077151674981046 ,  0.2314550249431379,
        -0.3857583749052298,  0.5400617248673217,  0.2886751345948129,
        -0.3779644730092272,  0.4225771273642583, -0.4364357804719848,
        0.4225771273642583, -0.3779644730092272,  0.2886751345948129,
        0.2886751345948129, -0.4225771273642583,  0.4879500364742666,
        -0.4879500364742666,  0.4225771273642583, -0.2886751345948129,
        0.5               , -0.4364357804719848,  0.2439750182371333,
        -0.2439750182371333,  0.4364357804719848, -0.5               ,
        0.5400617248673217, -0.077151674981046 , -0.2314550249431379,
        0.3857583749052298, -0.3857583749052298,  0.2314550249431379,
        0.077151674981046 , -0.5400617248673217,  0.5               ,
        -0.4364357804719848,  0.2439750182371333, -0.2439750182371333,
        0.4364357804719848, -0.5               ,  0.2886751345948129,
        -0.4225771273642583,  0.4879500364742666, -0.4879500364742666,
        0.4225771273642583, -0.2886751345948129,  0.3256694736394648,
        -0.492365963917331 ,  0.5504818825631803, -0.492365963917331 ,
        0.3256694736394648,  0.5149286505444373, -0.4522670168666454,
        0.1740776559556979,  0.1740776559556979, -0.4522670168666454,
        0.5149286505444373,  0.5640760748177662, -0.1230914909793327,
        -0.2752409412815902,  0.4264014327112209, -0.2752409412815902,
        -0.1230914909793327,  0.5640760748177662,  0.4308202184276645,
        0.3077287274483318, -0.4308202184276645,  0.1846372364689991,
        0.1846372364689991, -0.4308202184276645,  0.3077287274483318,
        0.4308202184276645,  0.5640760748177662, -0.1230914909793327,
        -0.2752409412815902,  0.4264014327112209, -0.2752409412815902,
        -0.1230914909793327,  0.5640760748177662,  0.5149286505444373,
        -0.4522670168666454,  0.1740776559556979,  0.1740776559556979,
        -0.4522670168666454,  0.5149286505444373,  0.3256694736394648,
        -0.492365963917331 ,  0.5504818825631803, -0.492365963917331 ,
        0.3256694736394648,  0.3988620176087328, -0.5838742081211422,
        0.5838742081211422, -0.3988620176087328,  0.5640760748177662,
        -0.4264014327112209,  0.4264014327112209, -0.5640760748177662,
        0.5838742081211422, -0.056980288229819 , -0.3947710169758614,
        0.3947710169758614,  0.056980288229819 , -0.5838742081211422,
        0.4767312946227962,  0.3120938919661796, -0.4187178946793119,
        0.4187178946793119, -0.3120938919661796, -0.4767312946227962,
        0.2820380374088831,  0.5237849266164972, -0.120873444603807 ,
        -0.3626203338114211,  0.3626203338114211,  0.120873444603807 ,
        -0.5237849266164972, -0.2820380374088831,  0.4767312946227962,
        0.3120938919661796, -0.4187178946793119,  0.4187178946793119,
        -0.3120938919661796, -0.4767312946227962,  0.5838742081211422,
        -0.056980288229819 , -0.3947710169758614,  0.3947710169758614,
        0.056980288229819 , -0.5838742081211422,  0.5640760748177662,
        -0.4264014327112209,  0.4264014327112209, -0.5640760748177662,
        0.3988620176087328, -0.5838742081211422,  0.5838742081211422,
        -0.3988620176087328,  0.5188745216627708, -0.6793662204867574,
        0.5188745216627708,  0.6354889093022426, -0.3100868364730212,
        -0.3100868364730212,  0.6354889093022426,  0.5991446895152781,
        0.1132277034144596, -0.5063696835418333,  0.1132277034144596,
        0.5991446895152781,  0.4736654667156709,  0.4160251471689219,
        -0.3202563076101743, -0.3202563076101743,  0.4160251471689219,
        0.4736654667156709,  0.3100868364730212,  0.5413319619607668,
        0.0302613766334401, -0.4688072309384954,  0.0302613766334401,
        0.5413319619607668,  0.3100868364730212,  0.1497861723788195,
        0.4921545663875498,  0.3637664186342759, -0.3209703693831847,
        -0.3209703693831847,  0.3637664186342759,  0.4921545663875498,
        0.1497861723788195,  0.3100868364730212,  0.5413319619607668,
        0.0302613766334401, -0.4688072309384954,  0.0302613766334401,
        0.5413319619607668,  0.3100868364730212,  0.4736654667156709,
        0.4160251471689219, -0.3202563076101743, -0.3202563076101743,
        0.4160251471689219,  0.4736654667156709,  0.5991446895152781,
        0.1132277034144596, -0.5063696835418333,  0.1132277034144596,
        0.5991446895152781,  0.6354889093022426, -0.3100868364730212,
        -0.3100868364730212,  0.6354889093022426,  0.5188745216627708,
        -0.6793662204867574,  0.5188745216627708,  0.7071067811865475,
        -0.7071067811865475,  0.7071067811865475, -0.7071067811865475,
        0.5838742081211422,  0.3988620176087328, -0.3988620176087328,
        -0.5838742081211422,  0.4264014327112209,  0.5640760748177662,
        -0.5640760748177662, -0.4264014327112209,  0.2752409412815902,
        0.5640760748177662,  0.3256694736394648, -0.3256694736394648,
        -0.5640760748177662, -0.2752409412815902,  0.1507556722888818,
        0.4605661864718383,  0.5149286505444373, -0.5149286505444373,
        -0.4605661864718383, -0.1507556722888818,  0.0615457454896664,
        0.3077287274483318,  0.5539117094069973,  0.3077287274483318,
        -0.3077287274483318, -0.5539117094069973, -0.3077287274483318,
        -0.0615457454896664,  0.1507556722888818,  0.4605661864718383,
        0.5149286505444373, -0.5149286505444373, -0.4605661864718383,
        -0.1507556722888818,  0.2752409412815902,  0.5640760748177662,
        0.3256694736394648, -0.3256694736394648, -0.5640760748177662,
        -0.2752409412815902,  0.4264014327112209,  0.5640760748177662,
        -0.5640760748177662, -0.4264014327112209,  0.5838742081211422,
        0.3988620176087328, -0.3988620176087328, -0.5838742081211422,
        0.7071067811865475, -0.7071067811865475,  0.7071067811865475,
        -0.7071067811865475,  1.                ,  0.7071067811865475,
        0.7071067811865475,  0.4803844614152614,  0.7337993857053428,
        0.4803844614152614,  0.3100868364730212,  0.6354889093022426,
        0.6354889093022426,  0.3100868364730212,  0.1869893980016915,
        0.4947274449181537,  0.6637465183030646,  0.4947274449181537,
        0.1869893980016915,  0.1024183112998378,  0.3498251311407206,
        0.6059149009001735,  0.6059149009001735,  0.3498251311407206,
        0.1024183112998378,  0.0482804549585268,  0.2212488394343549,
        0.4947274449181537,  0.6386903850265855,  0.4947274449181537,
        0.2212488394343549,  0.0482804549585268,  0.017069718549973 ,
        0.1194880298498108,  0.3584640895494324,  0.597440149249054 ,
        0.597440149249054 ,  0.3584640895494324,  0.1194880298498108,
        0.017069718549973 ,  0.0482804549585268,  0.2212488394343549,
        0.4947274449181537,  0.6386903850265855,  0.4947274449181537,
        0.2212488394343549,  0.0482804549585268,  0.1024183112998378,
        0.3498251311407206,  0.6059149009001735,  0.6059149009001735,
        0.3498251311407206,  0.1024183112998378,  0.1869893980016915,
        0.4947274449181537,  0.6637465183030646,  0.4947274449181537,
        0.1869893980016915,  0.3100868364730212,  0.6354889093022426,
        0.6354889093022426,  0.3100868364730212,  0.4803844614152614,
        0.7337993857053428,  0.4803844614152614,  0.7071067811865475,
        0.7071067811865475,  1.                
    ])
    M = spar.coo_matrix((vals,(rows,cols))).toarray()
    return M


def irrep_projectors(dims, cob):
    start = 0
    projectors = []
    for bk_sz in dims:
        stop = start + bk_sz 
        subspace = cob[:,start:stop]
        P = subspace @ subspace.T.conj()
        projectors.append(P)
        trP = np.trace(P)
        assert abs(trP - bk_sz) < 1e-8
        start = stop 
    return projectors


class Spin72(SU2):

    SPINS = np.arange(start=7, stop=-8, step=-2)/2.0
    Jx_sup = np.array([sqrt(7.)/2, sqrt(3.), sqrt(15.)/2, 2., sqrt(15.)/2, sqrt(3.), sqrt(7.)/2])
    Jx = np.diag(Jx_sup, 1) + np.diag(Jx_sup, -1)
    Jy = np.diag(-1j*Jx_sup, 1) + np.diag(1j*Jx_sup, -1)
    Jz = np.diag(SPINS)
    check_su2_generators(Jx, Jy, Jz)
    eigJx, VJx = eign(Jx)
    eigJy, VJy = eign(Jy)

    C = clebsh_gordan_matrix_spin72()
    superop_stdmx_cob = C @ np.kron(la.expm(1j * np.pi * Jy), np.eye(8))
    irrep_block_sizes = np.array([i for i in range(1, 16, 2)])
    irrep_labels = (irrep_block_sizes-1)/2
    irrep_stdmx_projectors = irrep_projectors(irrep_block_sizes, superop_stdmx_cob)

    @staticmethod
    def mat_R(theta, phi):
        mat = np.exp(-1j * Spin72.SPINS * phi)[:, np.newaxis] * Spin72.expm_iJy(-theta)
        # ^ that line is equivalent to, but more efficient than the following.
        #       Rz_phi = np.diag(np.exp(-1j * Spin72.SPINS * phi))
        #       mat = Rz_phi @ Spin72.expm_iJy(-theta, numeric)
        return mat
    
    @staticmethod
    def stdmx_twirl(A : np.ndarray) -> np.ndarray:
        coeffs = np.array([np.vdot(A,P) for P in Spin72.irrep_stdmx_projectors])
        coeffs /= Spin72.irrep_block_sizes
        tA = np.sum([coeffs[i]*P for i,P in enumerate(Spin72.irrep_stdmx_projectors)])
        return tA

    @classmethod
    def all_characters_from_unitary(cls, U):
        """
        Equivalent to np.array([
            SU2.characters_from_angles(a, b, g, (__twojplusone-1)/2 )[0] for __twojplusone in range(1,16,2)
        ]), where (a,b,g) are the Euler angles for U.
        """
        A = pygsti.tools.unitary_to_std_process_mx(U)
        diag = np.diag(cls.superop_stdmx_cob @ A @ cls.superop_stdmx_cob.conj().T).real
        out = []
        idx = 0
        for b_sz in cls.irrep_block_sizes:
            vec = diag[idx:idx+b_sz]
            out.append(np.sum(vec))
            idx += b_sz
        out = np.array(out)
        return out


def get_M():
    # define a bunch of constants to reduce the risk of typos.
    from numpy import sqrt
    s1b2   = sqrt(1/2)
    s7b6   = sqrt(7/6)
    s1b42  = sqrt(1/42)
    s3b14  = sqrt(3/14)
    s3b22  = sqrt(3/22)
    s1b858 = sqrt(1/858)
    s1b546 = sqrt(1/546)
    s3b286 = sqrt(3/286)
    s3b182 = sqrt(3/182)
    s1b66  = sqrt(1/66)
    s7b22  = sqrt(7/22)
    s1b154 = sqrt(1/154)
    s7b78  = sqrt(7/78)

    row1 = [     s1b2,        s1b2,         s1b2,          s1b2   ]
    row2 = [     s7b6,    5 * s1b42,        s3b14,         s1b42  ]
    row3 = [     s7b6,        s1b42,        s3b14,     5 * s1b42  ]
    row4 = [ 7 * s1b66,   5 * s1b66,    7 * s1b66,         s3b22  ]
    row5 = [     s7b22,  13 * s1b154,   3 * s1b154,    9 * s1b154 ]
    row6 = [     s7b78,  23 * s1b546,  17 * s1b546,    5 * s3b182 ]
    row7 = [     s1b66,   5 * s1b66,    3 * s3b22,     5 * s1b66  ]
    row8 = [     s1b858,  7 * s1b858,   7 * s3b286,   35 * s1b858 ]
    Mhalf = np.vstack([row1, row2, row3, row4, row5, row6, row7, row8])
    M = np.hstack([Mhalf, Mhalf[:, ::-1]])
    signs = np.ones((8,8))
    signs[1, 4:8] = -1
    signs[2, 2:6] = -1
    signs[3,   :] = [1,  -1,  -1,  -1,   1,   1,   1,  -1]
    signs[4,   :] = [1,  -1,  -1,   1,   1,  -1,  -1,   1]
    signs[5,   :] = [1,  -1,   1,   1,  -1,  -1,   1,  -1]
    signs[6,   :] = [1,  -1,   1,  -1,  -1,   1,  -1,   1]
    signs[7,   :] = [1,  -1,   1,  -1,   1,  -1,   1,  -1]
    M = signs * M
    M = 0.5*M
    # A construction that's more error prone but appearently equivalent.
    #
    # row1 = [    s1b2,         s1b2,         s1b2,          s1b2,         s1b2,          s1b2,          s1b2,          s1b2]
    # row2 = [    s7b6,     5 * s1b42,        s3b14,         s1b42,   -1 * s1b42,    -1 * s3b14,    -5 * s1b42,    -1 * s7b6]
    # row3 = [    s7b6,         s1b42,   -1 * s3b14,    -5 * s1b42,   -5 * s1b42,    -1 * s3b14,         s1b42,         s7b6]
    # row4 = [7 * s1b66,   -5 * s1b66,   -7 * s1b66,    -1 * s3b22,        s3b22,     7 * s1b66,     5 * s1b66,    -7 * s1b66]
    # row5 = [    s7b22,  -13 * s1b154,  -3 * s1b154,    9 * s1b154,   9 * s1b154,   -3 * s1b154,  -13 * s1b154,        s7b22]
    # row6 = [    s7b78,  -23 * s1b546,  17 * s1b546,    5 * s3b182,  -5 * s3b182,  -17 * s1b546,   23 * s1b546,   -1 * s7b78]
    # row7 = [    s1b66,   -5 * s1b66,    3 * s3b22,    -5 * s1b66,   -5 * s1b66,     3 * s3b22,    -5 * s1b66,         s1b66]
    # row8 = [    s1b858,  -7 * s1b858,   7 * s3b286,  -35 * s1b858,  35 * s1b858,   -7 * s3b286,    7 * s1b858,   -1 * s1b858]
    # M0 = np.vstack([row1, row2, row3, row4, row5, row6, row7, row8])
    return M


def get_F():

    F = np.zeros((8,8))
    F[0,  :8] = 1.0
    F[1, 1:8] = [59 / 63, 17 / 21,   13 / 21,    23 / 63,     1 / 21,    -1 / 3,    -7 / 9    ]
    F[2, 2:8] =         [  7 / 15,    1 / 21,    -1 / 3,    -11 / 21,    -1 / 3,     7 / 15   ]
    F[3, 3:8] =                   [ -31 / 77,  -101 / 231,    1 / 77,    17 / 33,   -7 / 33   ]
    F[4, 4:8] =                              [    1 / 9,    103 / 231,   -1 / 3,     7 / 99   ]
    F[5, 5:8] =                                         [   -33 / 91,    53 / 429,  -7 / 429  ]
    F[6, 6:8] =                                                       [  -1 / 39,    1 / 429  ]
    F[7, 7:8] =                                                                   [ -1 / 6435 ]

    tril_ind = np.tril_indices(8, -1)
    F[tril_ind] = F.T[tril_ind]
    # F = F / 8 <--- that scaling is likely a mistake, per Robin's email.
    return F


def std_to_pp(dim):
    """
    Return 
    """
    from_basis = _basis.BuiltinBasis('std', dim, sparse=False)
    to_basis   = _basis.BuiltinBasis('pp',  dim, sparse=False)
    toMx   = from_basis.create_transform_matrix(to_basis)
    fromMx = to_basis.create_transform_matrix(from_basis)
    
    pass


def unitary_to_superoperator(U, basis):
    return pygsti.tools.basistools.change_basis(pygsti.tools.unitary_to_std_process_mx(U), 'std', basis)


def default_povm(d, basis):
    effects = [np.zeros((d,d)) for _ in range(d)]
    for i, e in enumerate(effects):
        e[i,i] = 1.0
    povm = np.vstack([pygsti.tools.stdmx_to_vec(e, basis).ravel() for e in effects])
    return povm


class Spin72RB:

    def __init__(self, N: int, lengths: List[int]):
        self._basis = 'pp'
        self._noise_channel = np.eye(64)
        self._povm = default_povm(8, self._basis)
        self.N = N
        self.lengths = np.array(lengths)
        pass

    def set_error_channel_exponential(self, gamma: float):
        assert gamma >= 0
        if gamma == 0:
            self._noise_channel = np.eye(64)
            return
        E_matrixunit = np.zeros((64, 64))
        for ell in range(64):
            # convert ell to a linear index; i = ell % 8, j = ell // 8
            i = ell  % 8
            j = ell // 8
            E_matrixunit[ell,ell] = np.exp(-gamma * abs(i - j))
        self._noise_channel = pygsti.tools.basistools.change_basis(E_matrixunit, 'std', self._basis)
        return

    def set_error_channel_gaussian(self, gamma: float):
        assert gamma > 0
        E_matrixunit = np.zeros((64, 64))
        for ell in range(64):
            # convert ell to a linear index; i = ell % 8, j = ell // 8
            i = ell  % 8
            j = ell // 8
            E_matrixunit[ell,ell] = np.exp(-gamma * abs(i - j)**2)
        self._noise_channel = pygsti.tools.basistools.change_basis(E_matrixunit, 'std', self._basis)
        pass

    def set_error_channel_rotate_Jz2(self, theta: float):
        U = la.expm(1j * theta * Spin72.Jz @ Spin72.Jz)
        self._noise_channel = unitary_to_superop(U, self._basis)
        pass

    def set_error_channel_exponential_compose_rotate_Jz2(self, gamma:float, theta:float):
        self.set_error_channel_exponential(gamma)
        E0 = self._noise_channel
        self.set_error_channel_rotate_Jz2(theta)
        E1 = self._noise_channel
        self._noise_channel = E1 @ E0

    def probabilities(self, seed=0):
        from tqdm import tqdm
        np.random.seed(seed)
        probs = np.zeros((8, self.lengths.size, self.N, 8))
        # probs[i,j,k,ell] = probability of measuring outcome ell 
        #    ... when running the k-th circuit of length lengths[j] 
        #    ... given preparation in state i.
        for i in range(8):
            all_circuits = SU2.rb_circuits_by_angles(self.N, self.lengths, seed + i)
            # ^ all_circuits[j][k] = (lengths[j]+1)-by-3 array.
            for j in tqdm(range(self.lengths.size), desc=f'Simulating with stateprep at |{i}>'):
                fixed_length_circuits = all_circuits[j]
                starting_state = self._povm[i, :]
                probs[i,j] = self.process_circuit_block(fixed_length_circuits, starting_state)
        return probs
    
    def process_circuit_block(self, circuits, starting_superket):
        block = np.zeros(shape=(len(circuits), 8))
        # block[k,ell] = the probability of measuring outcome j after running the k-th circuit.
        for k, angles in enumerate(circuits):
            # angles has three columns. Each row of angles specifies an element of SU(2).
            # The sequence of SU(2) elements induced by the rows of angles defines a circuit. 
            unitaries = Spin72.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
            superops = [unitary_to_superoperator(U, self._basis) for U in unitaries]
            superket = starting_superket
            for superop in superops:
                superket = self._noise_channel @ (superop @ superket)
            block[k,:] = self._povm @ superket
        return block
    
    @staticmethod
    def synspam_transform(_probs, shots_per_circuit=np.inf, seed=0):
        #  probs[i,j,k,ell] = probability of measuring outcome ell when running the k-th lengths[j] circuit given preparation in state i.
        M = get_M()
        state = np.random.default_rng(seed)
        num_lengths = _probs.shape[1]
        N = _probs.shape[2]
        pkm = np.zeros((8, num_lengths))
        plm_equivs = []
        for j in range(num_lengths):
            # m = lengths[j]
            Pmj = np.zeros((8,8))
            for i in range(8):
                distns_i = _probs[i,j,:,:]
                if shots_per_circuit < np.inf:
                    samples_i = np.zeros(8)
                    for k in range(N):
                        dist = distns_i[k]
                        assert la.norm(dist[dist < 0]) < 1e-14
                        dist[dist < 0] = 0.0
                        dist /= np.sum(dist)
                        temp = sp.stats.multinomial.rvs(shots_per_circuit, dist, random_state=state)
                        samples_i += temp
                    samples_i /= np.sum(samples_i)
                else:
                    samples_i = np.mean(distns_i, axis=0)
                Pmj[i,:] = samples_i
            plm_equivs.append(Pmj[np.diag_indices(8)])
            Qmj = M @ Pmj @ M.T
            pkm[:,j] = Qmj[np.diag_indices(8)]
        plm = np.column_stack(plm_equivs)
        return pkm, plm
    
    @staticmethod
    def synspam_character_transform(_probs, _chars, shots_per_circuit=np.inf, seed=0):
        #  probs[i,j,k,ell] = probability of measuring outcome ell when running the k-th lengths[j] circuit given preparation in state i.
        #  chars[i,j,k,ell] = the value of the ell^th irrep's character function for the "hidden" initial gate in the k-th circuit of length lengths[j] given state prep in i.
        assert _probs.shape == _chars.shape
        M = get_M()
        state = np.random.default_rng(seed)
        num_lengths = _probs.shape[1]
        N = _probs.shape[2]  # we ran this many circuits per (stateprep, rank-1 povm, length).
        pkm = np.zeros((8, num_lengths))
        plm_equivs = []
        for j in range(num_lengths):
            # m = lengths[j]
            Pmj = np.zeros((8,8))
            iterator = range(8) if shots_per_circuit == np.inf else tqdm(range(8), desc='Processing data from circuits that started at |{i}>')
            for i in iterator:
                distns_i = _probs[i,j,:,:]
                if shots_per_circuit < np.inf:
                    samples_i = np.zeros(8)
                    for k in range(N):
                        dist = distns_i[k].copy()
                        assert la.norm(dist[dist < 0]) < 1e-14
                        dist[dist < 0] = 0.0
                        dist /= dist.sum()
                        temp = sp.stats.multinomial.rvs(shots_per_circuit, dist, random_state=state)
                        samples_i += (temp * _chars[i,j,k,:])
                    samples_i *= Spin72.irrep_block_sizes
                    samples_i /= (N*shots_per_circuit)
                else:
                    distns_i = distns_i * _chars[i,j,:,:] * Spin72.irrep_block_sizes[np.newaxis, :]
                    samples_i = np.sum(distns_i, axis=0) / N  # How should I be normalizing this?
                Pmj[i,:] = samples_i
            plm_equivs.append(Pmj[np.diag_indices(8)])
            Qmj = M @ Pmj @ M.T
            pkm[:,j] = Qmj[np.diag_indices(8)]
        plm = np.column_stack(plm_equivs)
        return pkm, plm

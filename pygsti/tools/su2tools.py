"""
Tools for working with representations of the special unitary group, SU(2).
"""
import numpy as np
from numpy import sqrt
import pygsti.modelmembers
import scipy.linalg as la
import scipy.sparse as spar
from pygsti.baseobjs import basisconstructors as bcons
from typing import List, Tuple


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
    scales = np.atleast_1d(scales)

    batch_out = np.array([
        (eigvecs * np.exp(1j*s*eigvals)[np.newaxis,:]) @ eigvecs for s in scales
    ])
    return batch_out


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

    """
    The "formula" for the character function in terms of euler angles is nasty.
    Just convert to a 2-by-2 matrix represetation, compute the eigenvalues,
    I'll get e^(\pm 1j \theta /2), and that lets me read off theta.

    The character only depends on \theta and it has a specific formula for a given
    j that depends on whether j is integral or half-integral. Robin will send formula.
    """

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
    def angles_from_unitary(cls,R, tol=1e-10):
        
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
        a, b, g = SU2.angles_from_2x2_unitary(R2x2)
        return a, b, g

    @classmethod
    def expm_iJx(cls, theta):
        theta = np.atleast_1d(theta)
        return batch_normal_expm_1jscales(cls.VJx, cls.eigJx, theta)
    
    @classmethod
    def expm_iJy(cls, theta):
        theta = np.atleast_1d(theta)
        return batch_normal_expm_1jscales(cls.VJy, cls.eigJy, theta)

    @staticmethod
    def angles_from_2x2_unitary(R):
        assert R.shape == (2, 2)
        try:
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
        except ZeroDivisionError:
            return 0, 0, 0
        if np.any(np.isnan((alpha, beta, gamma))):
            return 0, 0, 0
        return alpha, beta, gamma
    
    # TODO: interrogate all the places where I'm using np.newaxis and de-vectorize in case I broke things.
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
    def composition_inverse(alphas, betas, gammas):
        R_composed = np.eye(2)
        if alphas.size > 0:
            Rs = SU2.unitaries_from_angles(alphas, betas, gammas)
            for R in Rs:
                R_composed = R @ R_composed
        invR_composed = R_composed.T.conj()
        ea = SU2.angles_from_2x2_unitary(invR_composed)
        return ea
    
    @staticmethod
    def rb_circuits_by_angles(N: int, lengths: List[int], seed=0, invert_from=0) -> np.ndarray[Tuple[float,float,float]]:
        np.random.seed(seed)
        out = []

        assert 0 not in lengths
        for ell in lengths:
            angles_for_length_ell = []
            for _ in range(N):
                alphas, betas, gammas = SU2.random_euler_angles(ell)
                inv_angles = SU2.composition_inverse(
                    alphas[invert_from:],
                    betas[invert_from:],
                    gammas[invert_from:]
                )
                ell_out = list(zip(alphas, betas, gammas))
                ell_out.append(inv_angles)
                angles_for_length_ell.append(np.array(ell_out))
            out.append(angles_for_length_ell)
        # out[i,j] = (lengths[i]+1)-by-3 array whose columns contain angles for the j-th RB circuit of length lenghts[i].
        return out

    @classmethod
    def character_from_unitary(cls, U, j=1/2):
        assert U.shape[0] == U.shape[1]
        dim = U.shape[0]
        if dim != 2:
            j = (dim - 1)/2  # ignore j if it was provided.
            a,b,g = cls.angles_from_unitary(U)
            R2x2 = SU2.unitaries_from_angles(a,b,g)[0]
            return SU2.character_from_unitary(R2x2, j)
        
        eigs = la.eigvals(U) # eigs = exp(\pm i theta /2)
        theta = np.real(2*np.log(eigs[0])/1j)
        tr = np.trace(U)
        check = tr - 2*np.cos(theta/2)
        assert abs(check) < 1e-10

        return np.real(np.sum([np.exp(1j*__j*theta) for __j in np.arange(-j, j+1)]))

    @staticmethod
    def characters_from_angles(alphas, betas, gammas, j):
        U2x2s = SU2.unitaries_from_angles(alphas, betas, gammas)
        characters = np.zeros(len(U2x2s))
        for i,U in enumerate(U2x2s):
            characters[i] = SU2.character_from_unitary(U, j)
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


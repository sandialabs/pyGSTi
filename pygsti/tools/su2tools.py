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
from typing import List, Tuple
import warnings


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


class SU2:

    Jx = bcons.sigmax / 2
    Jy = bcons.sigmay / 2
    Jz = bcons.sigmaz / 2
    check_su2_generators(Jx, Jy, Jz)

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

    @staticmethod
    def expm_iJx(theta):
        return la.expm(1j*theta*SU2.Jx)
    
    @staticmethod
    def expm_iJy(theta):
        return la.expm(1j*theta*SU2.Jy)

    @staticmethod
    def angles_from_2x2_unitary(R):
        assert R.shape == (2, 2)
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
    
    @classmethod
    def unitaries_from_angles(cls,alpha,beta,gamma):
        # Construct an element of SU(2) from Euler angles
        array_on_input = isinstance(alpha, np.ndarray)

        alpha = np.atleast_1d(alpha)
        beta  = np.atleast_1d(beta)
        gamma = np.atleast_1d(gamma)

        if not array_on_input:
            alpha.size == beta.size == gamma.size == 1

        dJz = np.diag(cls.Jz)
        angles = zip(alpha, beta, gamma)
        out = []
        for abg in angles:
            a, b, g = abg
            scaled_rows = cls.expm_iJx(b) * (np.exp(1j * a * dJz))[:, np.newaxis]
            # scaled_rows = la.expm(1j * b * Jx) * (np.exp(1j * a * dJz))[:, np.newaxis]
            mat = scaled_rows * (np.exp(1j * g * dJz))[np.newaxis, :]
            # left   = np.diag(np.exp(1j * a * dJz))
            # middle = la.expm(1j * b * Jx) 
            # right  = np.diag(np.exp(1j * g * dJz))
            # mat = left @ middle @ right
            out.append(mat)

        return out

    @classmethod
    def composition_inverse(cls, alphas, betas, gammas, as_angles=True):
        Rs = SU2.unitaries_from_angles(alphas, betas, gammas)
        R_composed = np.eye(2)
        for R in Rs:
            R_composed = R @ R_composed
        invR_composed = R_composed.T.conj()
        ea = SU2.angles_from_2x2_unitary(invR_composed)
        if as_angles:
            return ea
        else:
            invR_mat = cls.unitaries_from_angles(*ea)[0]
            return invR_mat
    
    @staticmethod
    def rb_circuits_by_angles(N: int, lengths: List[int], seed=0) -> np.ndarray[Tuple[float,float,float]]:
        np.random.seed(seed)
        out = []
    
        if lengths[0] == 0:
            angles_for_length_0 = [np.zeros((1,3)) for _ in range(N)]
            out.append(angles_for_length_0)
            remaining_lengths = lengths[1:]
        else:
            remaining_lengths = lengths

        assert 0 not in remaining_lengths
        for ell in remaining_lengths:
            angles_for_length_ell = []
            for _ in range(N):
                alphas, betas, gammas = SU2.random_euler_angles(ell)
                inv_angles = SU2.composition_inverse(alphas, betas, gammas, as_angles=True)
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

    C = clebsh_gordan_matrix_spin72()
    superop_stdmx_cob = C @ np.kron(la.expm(1j * np.pi * Jy), np.eye(8))
    irrep_block_sizes = np.array([i for i in range(1, 16, 2)])
    irrep_stdmx_projectors = irrep_projectors(irrep_block_sizes, superop_stdmx_cob)

    @staticmethod
    def expm_iJy(theta, numerically_verify=False):
        """
        Returns expm(1j * theta * Jy).
        This is equivalent to the matrix "Ry(-theta)" from the file '2023_12-22_SU2-rotate.pdf' provided by Robin.
        """
        if isinstance(theta, np.ndarray):
            assert theta.size == 1
            theta = theta.item()
        theta *= -1  # this was originally written to compute expm(-1j * theta * Jy)
        mat = np.zeros(shape=(8,8), dtype=np.double)
        c = np.power(np.cos(theta/2), np.arange(8))
        s = np.power(np.sin(theta/2), np.arange(8))
        """
        Build the lower triangle of "mat", column-by-column
        """
        mat[:,0] = c[::-1] * s * np.sqrt([1, 7, 21, 35, 35, 21, 7, 1])

        mat[1,1] = 7*c[7] - 6*c[5]
        mat[2,1] = s[1]*c[4]*(7*c[2] - 5) * sqrt(3)
        mat[3,1] = s[2]*c[3]*(7*c[2] - 4) * sqrt(5)
        mat[4,1] = s[3]*c[2]*(7*c[2] - 3) * sqrt(5)
        mat[5,1] = s[4]*c[1]*(7*c[2] - 2) * sqrt(3)
        mat[6,1] = s[5]*c[0]*(7*c[2] - 1)
        mat[7,1] = s[6]*c[1]*sqrt(7)

        mat[2,2] = 21*c[7] - 30*c[5] + 10*c[3]
        mat[3,2] = s[1]*c[2]*(7*c[4] - 8*c[2] + 2) * sqrt(15)
        mat[4,2] = s[2]*c[1]*(7*c[4] - 6*c[2] + 1) * sqrt(15)
        mat[5,2] = s[3]*c[0]*(21*c[4] - 12*c[2] + 1)
        mat[6,2] = s[4]*c[1]*(7*c[2] - 2)*sqrt(3)
        mat[7,2] = s[5]*c[2]*sqrt(21)

        mat[3,3] = s[0] * c[[7,5,3,1]] @ np.array([35, -60, 30, -4])
        mat[4,3] = s[1] * c[[6,4,2,0]] @ np.array([35, -45, 15, -1])
        mat[5,3] = s[2] * c[1]*(7*c[4] - 6*c[2] + 1) * sqrt(15)
        mat[6,3] = s[3] * c[2]*(7*c[2] - 3) * sqrt(5)
        mat[7,3] = s[4] * c[3] * sqrt(35)

        mat[4,4] = mat[3,3]
        mat[5,4] = s[1]*c[2]*(7*c[4] - 8*c[2] + 2) * sqrt(15)
        mat[6,4] = s[2]*c[3]*(7*c[2] - 4) * sqrt(5)
        mat[7,4] = s[3]*c[4] * sqrt(35)

        mat[5,5] = mat[2,2]
        mat[6,5] = s[1]*c[4]*(7*c[2] - 5) * sqrt(3)
        mat[7,5] = s[2]*c[5] * sqrt(21)
        
        mat[6,6] = mat[1,1]
        mat[7,6] = s[1]*c[6] * sqrt(7)

        mat[7,7] = mat[0,0]

        """
        Go back and build the upper triangle
        """
        mat[0, 1] = -mat[1,0]
        mat[:2,2] = mat[2,:2] * np.array([ 1, -1])
        mat[:3,3] = mat[3,:3] * np.array([-1,  1, -1])
        mat[:4,4] = mat[4,:4] * np.array([ 1, -1,  1, -1])
        mat[:5,5] = mat[5,:5] * np.array([-1,  1, -1,  1, -1])
        mat[:6,6] = mat[6,:6] * np.array([ 1, -1,  1, -1,  1, -1])
        mat[:7,7] = mat[7,:7] * np.array([-1,  1, -1,  1, -1, 1, -1])

        if numerically_verify:
            expm_arg = 1j * (-theta) * Spin72.Jy
            num_mat = la.expm(expm_arg)
            assert la.norm(mat - num_mat, 'fro') <= 1e-13, 'Numeric and symbolic exponentiation give different results'
            # ^ an extremely crude sanity check.
        return mat
    
    @staticmethod
    def expm_iJx(theta, numerically_verify=False):
        """
        Returns expm(1j * theta * Jx).
        """
        if isinstance(theta, np.ndarray):
            assert theta.size == 1
            theta = theta.item()
        theta *= -1  # this was originally written to compute expm(-1j * theta * Jx)
        mat = np.zeros(shape=(8,8), dtype=np.complex128)
        c = np.power(np.cos(theta/2), np.arange(8))
        s = np.power(np.sin(theta/2), np.arange(8))
        """
        Build the lower triangle of "mat", column-by-column
        """
        mat[:,0] = c[::-1] * s * np.sqrt([1, 7, 21, 35, 35, 21, 7, 1])
        mat[:,0] = mat[:,0] * np.array([1, -1j, -1, 1j, 1, -1j, -1, 1j])

        mat[1,1] = 7*c[7] - 6*c[5]
        mat[2,1] = s[1]*c[4]*(7*c[2] - 5) * sqrt(3) * (-1j)
        mat[3,1] = s[2]*c[3]*(7*c[2] - 4) * sqrt(5) * (-1)
        mat[4,1] = s[3]*c[2]*(7*c[2] - 3) * sqrt(5) * (1j)
        mat[5,1] = s[4]*c[1]*(7*c[2] - 2) * sqrt(3)
        mat[6,1] = s[5]*c[0]*(7*c[2] - 1) * (-1j)
        mat[7,1] = s[6]*c[1]*sqrt(7) * (-1)

        mat[2,2] = 21*c[7] - 30*c[5] + 10*c[3]
        mat[3,2] = s[1]*c[2]*(7*c[4] - 8*c[2] + 2) * sqrt(15) * (-1j)
        mat[4,2] = s[2]*c[1]*(7*c[4] - 6*c[2] + 1) * sqrt(15) * (-1)
        mat[5,2] = s[3]*c[0]*(21*c[4] - 12*c[2] + 1) * (1j)
        mat[6,2] = s[4]*c[1]*(7*c[2] - 2)*sqrt(3)
        mat[7,2] = s[5]*c[2]*sqrt(21) * (-1j)

        mat[3,3] = s[0] * c[[7,5,3,1]] @ np.array([35, -60, 30, -4])
        mat[4,3] = s[1] * c[[6,4,2,0]] @ np.array([35, -45, 15, -1]) * (-1j)
        mat[5,3] = s[2] * c[1]*(7*c[4] - 6*c[2] + 1) * sqrt(15) * (-1)
        mat[6,3] = s[3] * c[2]*(7*c[2] - 3) * sqrt(5) * (1j)
        mat[7,3] = s[4] * c[3] * sqrt(35)

        mat[4,4] = mat[3,3]
        mat[5,4] = s[1]*c[2]*(7*c[4] - 8*c[2] + 2) * sqrt(15) * (-1j)
        mat[6,4] = s[2]*c[3]*(7*c[2] - 4) * sqrt(5) * (-1)
        mat[7,4] = s[3]*c[4] * sqrt(35) * (1j)

        mat[5,5] = mat[2,2]
        mat[6,5] = s[1]*c[4]*(7*c[2] - 5) * sqrt(3) * (-1j)
        mat[7,5] = s[2]*c[5] * sqrt(21) * (-1)
        
        mat[6,6] = mat[1,1]
        mat[7,6] = s[1]*c[6] * sqrt(7) * (-1j)

        mat[7,7] = mat[0,0]

        """
        Go back and build the upper triangle. It turns out this is a lot simpler than Jy.
        """
        mat[0, 1] = mat[1,0]
        mat[:2,2] = mat[2,:2] #* np.array([ 1, 1])
        mat[:3,3] = mat[3,:3] #* np.array([ 1,  1,  1])
        mat[:4,4] = mat[4,:4] #* np.array([ 1,  1,  1,  1])
        mat[:5,5] = mat[5,:5] #* np.array([ 1,  1,  1,  1,  1])
        mat[:6,6] = mat[6,:6] #* np.array([ 1,  1,  1,  1,  1,  1])
        mat[:7,7] = mat[7,:7] #* np.array([ 1,  1,  1,  1,  1, 1, 1])

        if numerically_verify:
            expm_arg = 1j * (-theta) * Spin72.Jx
            num_mat = la.expm(expm_arg)
            assert la.norm(mat - num_mat, 'fro') <= 1e-13, 'Numeric and symbolic exponentiation give different results'
            # ^ an extremely crude sanity check.
        return mat

    @staticmethod
    def mat_R(theta, phi, numerically_verify=False):
        mat = np.exp(-1j * Spin72.SPINS * phi).reshape(8, 1) * Spin72.expm_iJy(-theta, numerically_verify)
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
        assert gamma > 0
        E_matrixunit = np.zeros((64, 64))
        for ell in range(64):
            # convert ell to a linear index; i = ell % 8, j = ell // 8
            i = ell  % 8
            j = ell // 8
            E_matrixunit[ell,ell] = np.exp(-gamma * abs(i - j))
        self._noise_channel = pygsti.tools.basistools.change_basis(E_matrixunit, 'std', self._basis)
        pass

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
            for j in tqdm(range(self.lengths.size)):
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
    

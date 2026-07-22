"""
Tests for pygsti.tools.su2tools.

Ported and adapted from the branch `su2-rb-conservative`'s `test_su2tools.py`
(which tested a hardcoded `SU2` (spin-1/2) class and a hardcoded `Spin72` (spin-7/2)
subclass) to the generalized, instance-based `SpinJ` API introduced in this phase.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as spar

from pygsti.tools.optools import unitary_to_std_process_mx
from pygsti.tools.su2tools import (
    SpinJ,
    angles_from_2x2_unitaries,
    angles_from_2x2_unitary,
    batch_normal_expm_1jscales,
    charactercores_from_euler_angles,
    composition_asmatrix,
    composition_inverse,
    distance_mod_phase,
    random_euler_angles,
    _fundamental_spinj,
)
from ..util import BaseCase


def reconstruct_eigendecomp(eigvals, eigvecs, tol=1e-14):
    n = eigvals.size
    assert la.norm(eigvecs @ eigvecs.T.conj() - np.eye(n)) <= np.sqrt(n) * tol
    return eigvecs @ np.diag(eigvals) @ eigvecs.T.conj()


# ---------------------------------------------------------------------------
# Golden data, ported verbatim from the su2-rb-conservative branch:
#   - get_M(), get_F() from pygsti/adhoc/su2rbsims.py
#   - clebsh_gordan_matrix_spin72() from pygsti/tools/su2tools.py
# ---------------------------------------------------------------------------

def _golden_get_M():
    from numpy import sqrt
    s1b2 = sqrt(1 / 2)
    s7b6 = sqrt(7 / 6)
    s1b42 = sqrt(1 / 42)
    s3b14 = sqrt(3 / 14)
    s3b22 = sqrt(3 / 22)
    s1b858 = sqrt(1 / 858)
    s1b546 = sqrt(1 / 546)
    s3b286 = sqrt(3 / 286)
    s3b182 = sqrt(3 / 182)
    s1b66 = sqrt(1 / 66)
    s7b22 = sqrt(7 / 22)
    s1b154 = sqrt(1 / 154)
    s7b78 = sqrt(7 / 78)

    row1 = [s1b2, s1b2, s1b2, s1b2]
    row2 = [s7b6, 5 * s1b42, s3b14, s1b42]
    row3 = [s7b6, s1b42, s3b14, 5 * s1b42]
    row4 = [7 * s1b66, 5 * s1b66, 7 * s1b66, s3b22]
    row5 = [s7b22, 13 * s1b154, 3 * s1b154, 9 * s1b154]
    row6 = [s7b78, 23 * s1b546, 17 * s1b546, 5 * s3b182]
    row7 = [s1b66, 5 * s1b66, 3 * s3b22, 5 * s1b66]
    row8 = [s1b858, 7 * s1b858, 7 * s3b286, 35 * s1b858]
    Mhalf = np.vstack([row1, row2, row3, row4, row5, row6, row7, row8])
    M = np.hstack([Mhalf, Mhalf[:, ::-1]])
    signs = np.ones((8, 8))
    signs[1, 4:8] = -1
    signs[2, 2:6] = -1
    signs[3, :] = [1, -1, -1, -1, 1, 1, 1, -1]
    signs[4, :] = [1, -1, -1, 1, 1, -1, -1, 1]
    signs[5, :] = [1, -1, 1, 1, -1, -1, 1, -1]
    signs[6, :] = [1, -1, 1, -1, -1, 1, -1, 1]
    signs[7, :] = [1, -1, 1, -1, 1, -1, 1, -1]
    M = signs * M
    M = 0.5 * M
    return M


def _golden_get_F():
    F = np.zeros((8, 8))
    F[0, :8] = 1.0
    F[1, 1:8] = [59 / 63, 17 / 21, 13 / 21, 23 / 63, 1 / 21, -1 / 3, -7 / 9]
    F[2, 2:8] = [7 / 15, 1 / 21, -1 / 3, -11 / 21, -1 / 3, 7 / 15]
    F[3, 3:8] = [-31 / 77, -101 / 231, 1 / 77, 17 / 33, -7 / 33]
    F[4, 4:8] = [1 / 9, 103 / 231, -1 / 3, 7 / 99]
    F[5, 5:8] = [-33 / 91, 53 / 429, -7 / 429]
    F[6, 6:8] = [-1 / 39, 1 / 429]
    F[7, 7:8] = [-1 / 6435]
    tril_ind = np.tril_indices(8, -1)
    F[tril_ind] = F.T[tril_ind]
    return F


def _golden_clebsh_gordan_matrix_spin72():
    rows = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
        4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7,
        7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10,
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
        7, 14, 21, 28, 35, 42, 49, 56, 6, 13, 20, 27, 34, 41, 48, 7, 14,
        21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 5, 12, 19, 26,
        33, 40, 6, 13, 20, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15,
        22, 29, 43, 50, 57, 23, 30, 37, 44, 51, 58, 4, 11, 18, 25, 32, 5,
        12, 19, 26, 33, 40, 6, 13, 20, 27, 34, 41, 48, 7, 14, 21, 28, 35,
        42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31,
        38, 45, 52, 59, 3, 10, 17, 24, 4, 11, 25, 32, 5, 12, 19, 26, 33,
        40, 6, 13, 20, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22,
        29, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 52, 59, 39, 46, 53,
        60, 2, 9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26,
        33, 40, 6, 13, 20, 27, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56,
        15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 45, 52,
        59, 39, 46, 53, 60, 47, 54, 61, 1, 8, 2, 16, 3, 10, 17, 24, 4,
        11, 25, 32, 5, 12, 19, 26, 33, 40, 6, 13, 20, 34, 41, 48, 7, 14,
        21, 28, 35, 42, 49, 56, 15, 22, 29, 43, 50, 57, 23, 30, 37, 44, 51,
        58, 31, 38, 52, 59, 39, 46, 53, 60, 47, 61, 55, 62, 0, 1, 8, 2,
        9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26, 33, 40,
        6, 13, 20, 27, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22,
        29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 45, 52, 59, 39,
        46, 53, 60, 47, 54, 61, 55, 62, 63
    ])
    vals = np.array([
        0.3535533905932737, -0.3535533905932737, 0.3535533905932737,
        -0.3535533905932737, 0.3535533905932737, -0.3535533905932737,
        0.3535533905932737, -0.3535533905932737, 0.2886751345948129,
        -0.3779644730092272, 0.4225771273642583, -0.4364357804719848,
        0.4225771273642583, -0.3779644730092272, 0.2886751345948129,
        0.5400617248673217, -0.3857583749052298, 0.2314550249431379,
        -0.077151674981046, -0.077151674981046, 0.2314550249431379,
        -0.3857583749052298, 0.5400617248673217, 0.2886751345948129,
        -0.3779644730092272, 0.4225771273642583, -0.4364357804719848,
        0.4225771273642583, -0.3779644730092272, 0.2886751345948129,
        0.2886751345948129, -0.4225771273642583, 0.4879500364742666,
        -0.4879500364742666, 0.4225771273642583, -0.2886751345948129,
        0.5, -0.4364357804719848, 0.2439750182371333,
        -0.2439750182371333, 0.4364357804719848, -0.5,
        0.5400617248673217, -0.077151674981046, -0.2314550249431379,
        0.3857583749052298, -0.3857583749052298, 0.2314550249431379,
        0.077151674981046, -0.5400617248673217, 0.5,
        -0.4364357804719848, 0.2439750182371333, -0.2439750182371333,
        0.4364357804719848, -0.5, 0.2886751345948129,
        -0.4225771273642583, 0.4879500364742666, -0.4879500364742666,
        0.4225771273642583, -0.2886751345948129, 0.3256694736394648,
        -0.492365963917331, 0.5504818825631803, -0.492365963917331,
        0.3256694736394648, 0.5149286505444373, -0.4522670168666454,
        0.1740776559556979, 0.1740776559556979, -0.4522670168666454,
        0.5149286505444373, 0.5640760748177662, -0.1230914909793327,
        -0.2752409412815902, 0.4264014327112209, -0.2752409412815902,
        -0.1230914909793327, 0.5640760748177662, 0.4308202184276645,
        0.3077287274483318, -0.4308202184276645, 0.1846372364689991,
        0.1846372364689991, -0.4308202184276645, 0.3077287274483318,
        0.4308202184276645, 0.5640760748177662, -0.1230914909793327,
        -0.2752409412815902, 0.4264014327112209, -0.2752409412815902,
        -0.1230914909793327, 0.5640760748177662, 0.5149286505444373,
        -0.4522670168666454, 0.1740776559556979, 0.1740776559556979,
        -0.4522670168666454, 0.5149286505444373, 0.3256694736394648,
        -0.492365963917331, 0.5504818825631803, -0.492365963917331,
        0.3256694736394648, 0.3988620176087328, -0.5838742081211422,
        0.5838742081211422, -0.3988620176087328, 0.5640760748177662,
        -0.4264014327112209, 0.4264014327112209, -0.5640760748177662,
        0.5838742081211422, -0.056980288229819, -0.3947710169758614,
        0.3947710169758614, 0.056980288229819, -0.5838742081211422,
        0.4767312946227962, 0.3120938919661796, -0.4187178946793119,
        0.4187178946793119, -0.3120938919661796, -0.4767312946227962,
        0.2820380374088831, 0.5237849266164972, -0.120873444603807,
        -0.3626203338114211, 0.3626203338114211, 0.120873444603807,
        -0.5237849266164972, -0.2820380374088831, 0.4767312946227962,
        0.3120938919661796, -0.4187178946793119, 0.4187178946793119,
        -0.3120938919661796, -0.4767312946227962, 0.5838742081211422,
        -0.056980288229819, -0.3947710169758614, 0.3947710169758614,
        0.056980288229819, -0.5838742081211422, 0.5640760748177662,
        -0.4264014327112209, 0.4264014327112209, -0.5640760748177662,
        0.3988620176087328, -0.5838742081211422, 0.5838742081211422,
        -0.3988620176087328, 0.5188745216627708, -0.6793662204867574,
        0.5188745216627708, 0.6354889093022426, -0.3100868364730212,
        -0.3100868364730212, 0.6354889093022426, 0.5991446895152781,
        0.1132277034144596, -0.5063696835418333, 0.1132277034144596,
        0.5991446895152781, 0.4736654667156709, 0.4160251471689219,
        -0.3202563076101743, -0.3202563076101743, 0.4160251471689219,
        0.4736654667156709, 0.3100868364730212, 0.5413319619607668,
        0.0302613766334401, -0.4688072309384954, 0.0302613766334401,
        0.5413319619607668, 0.3100868364730212, 0.1497861723788195,
        0.4921545663875498, 0.3637664186342759, -0.3209703693831847,
        -0.3209703693831847, 0.3637664186342759, 0.4921545663875498,
        0.1497861723788195, 0.3100868364730212, 0.5413319619607668,
        0.0302613766334401, -0.4688072309384954, 0.0302613766334401,
        0.5413319619607668, 0.3100868364730212, 0.4736654667156709,
        0.4160251471689219, -0.3202563076101743, -0.3202563076101743,
        0.4160251471689219, 0.4736654667156709, 0.5991446895152781,
        0.1132277034144596, -0.5063696835418333, 0.1132277034144596,
        0.5991446895152781, 0.6354889093022426, -0.3100868364730212,
        -0.3100868364730212, 0.6354889093022426, 0.5188745216627708,
        -0.6793662204867574, 0.5188745216627708, 0.7071067811865475,
        -0.7071067811865475, 0.7071067811865475, -0.7071067811865475,
        0.5838742081211422, 0.3988620176087328, -0.3988620176087328,
        -0.5838742081211422, 0.4264014327112209, 0.5640760748177662,
        -0.5640760748177662, -0.4264014327112209, 0.2752409412815902,
        0.5640760748177662, 0.3256694736394648, -0.3256694736394648,
        -0.5640760748177662, -0.2752409412815902, 0.1507556722888818,
        0.4605661864718383, 0.5149286505444373, -0.5149286505444373,
        -0.4605661864718383, -0.1507556722888818, 0.0615457454896664,
        0.3077287274483318, 0.5539117094069973, 0.3077287274483318,
        -0.3077287274483318, -0.5539117094069973, -0.3077287274483318,
        -0.0615457454896664, 0.1507556722888818, 0.4605661864718383,
        0.5149286505444373, -0.5149286505444373, -0.4605661864718383,
        -0.1507556722888818, 0.2752409412815902, 0.5640760748177662,
        0.3256694736394648, -0.3256694736394648, -0.5640760748177662,
        -0.2752409412815902, 0.4264014327112209, 0.5640760748177662,
        -0.5640760748177662, -0.4264014327112209, 0.5838742081211422,
        0.3988620176087328, -0.3988620176087328, -0.5838742081211422,
        0.7071067811865475, -0.7071067811865475, 0.7071067811865475,
        -0.7071067811865475, 1., 0.7071067811865475,
        0.7071067811865475, 0.4803844614152614, 0.7337993857053428,
        0.4803844614152614, 0.3100868364730212, 0.6354889093022426,
        0.6354889093022426, 0.3100868364730212, 0.1869893980016915,
        0.4947274449181537, 0.6637465183030646, 0.4947274449181537,
        0.1869893980016915, 0.1024183112998378, 0.3498251311407206,
        0.6059149009001735, 0.6059149009001735, 0.3498251311407206,
        0.1024183112998378, 0.0482804549585268, 0.2212488394343549,
        0.4947274449181537, 0.6386903850265855, 0.4947274449181537,
        0.2212488394343549, 0.0482804549585268, 0.017069718549973,
        0.1194880298498108, 0.3584640895494324, 0.597440149249054,
        0.597440149249054, 0.3584640895494324, 0.1194880298498108,
        0.017069718549973, 0.0482804549585268, 0.2212488394343549,
        0.4947274449181537, 0.6386903850265855, 0.4947274449181537,
        0.2212488394343549, 0.0482804549585268, 0.1024183112998378,
        0.3498251311407206, 0.6059149009001735, 0.6059149009001735,
        0.3498251311407206, 0.1024183112998378, 0.1869893980016915,
        0.4947274449181537, 0.6637465183030646, 0.4947274449181537,
        0.1869893980016915, 0.3100868364730212, 0.6354889093022426,
        0.6354889093022426, 0.3100868364730212, 0.4803844614152614,
        0.7337993857053428, 0.4803844614152614, 0.7071067811865475,
        0.7071067811865475, 1.
    ])
    M = spar.coo_matrix((vals, (rows, cols))).toarray()
    return M


class GoldenArrayTester(BaseCase):
    """SpinJ(7/2)'s C, M, F must match the branch's hardcoded spin-7/2 arrays."""

    def setUp(self):
        self.s = SpinJ(3.5)

    def test_synthetic_spam_matrix_matches_branch(self):
        self.assertArraysAlmostEqual(self.s.synthetic_spam_matrix, _golden_get_M(), places=13)

    def test_decay_recoupling_matrix_matches_branch_exactly(self):
        # F's entries are exact rationals, but wigner_6j() (pygsti.tools.wignersymbols)
        # produces them via a final float(...)**0.5 step that is documented (see that
        # module) to be accurate to ~1 ulp rather than bit-exact -- so we compare at
        # tight-but-not-exact tolerance rather than with assertArraysEqual.
        self.assertArraysAlmostEqual(self.s.decay_recoupling_matrix, _golden_get_F(), places=14)

    def test_clebsch_gordan_cob_matches_branch(self):
        self.assertArraysAlmostEqual(self.s.clebsch_gordan_cob, _golden_clebsh_gordan_matrix_spin72(), places=13)


class FreeFunctionTester(BaseCase):

    def test_batch_normal_expm_1jscales(self):
        rng = np.random.default_rng(0)
        n = 5
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A = A + A.conj().T  # Hermitian, hence normal.
        evals, evecs = np.linalg.eigh(A)
        scales = np.array([0.0, 0.5, 1.3, -2.1])
        actual = batch_normal_expm_1jscales(evecs, evals, scales)
        expect = np.array([la.expm(1j * s * A) for s in scales])
        self.assertArraysAlmostEqual(actual, expect, places=10)

    def test_distance_mod_phase_invariant_to_global_phase(self):
        rng = np.random.default_rng(0)
        U = la.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))[0]
        for phase in [1.0, -1.0, 1j, np.exp(1j * 0.37)]:
            self.assertAlmostEqual(distance_mod_phase(U, phase * U), 0.0, places=12)

    def test_random_euler_angles_shape_and_range(self):
        np.random.seed(0)
        alpha, beta, gamma = random_euler_angles(size=50)
        self.assertEqual(alpha.shape, (50,))
        self.assertTrue(np.all(alpha >= 0) and np.all(alpha < 2 * np.pi))
        self.assertTrue(np.all(beta >= 0) and np.all(beta <= np.pi))
        self.assertTrue(np.all(gamma >= 0) and np.all(gamma < 4 * np.pi))

    def test_random_euler_angles_with_rng(self):
        rng = np.random.default_rng(0)
        alpha, beta, gamma = random_euler_angles(size=10, rng=rng)
        self.assertEqual(alpha.shape, (10,))


class SU2GroupUtilityTester(BaseCase):
    """Tests of the representation-independent (2x2 / abstract group) utilities."""

    RELTOL = 1e-13

    def test_unitary_angles_unitary(self):
        num_trials = 20
        gen = np.random.default_rng(0)

        def make_unitary():
            temp = gen.standard_normal((2, 2)) + 1j * gen.standard_normal((2, 2))
            U = la.qr(temp)[0]
            U /= np.sqrt(la.det(U))
            return U

        mats = [make_unitary() for _ in range(num_trials)]
        mats.append(np.eye(2))
        mats.append(np.eye(2)[::-1, :])
        fundamental = _fundamental_spinj()
        for U in mats:
            a, b, g = angles_from_2x2_unitary(U)
            U_recon = fundamental.unitaries_from_angles(a, b, g)[0]
            dist = distance_mod_phase(U, U_recon)
            self.assertLessEqual(dist, 2 * self.RELTOL)

    def test_angles_unitary_angles(self):
        np.random.seed(0)
        fundamental = _fundamental_spinj()
        angles = np.column_stack(random_euler_angles(10))
        for abg in angles:
            U = fundamental.unitaries_from_angles(*abg.tolist())[0]
            abg_recon = np.array(angles_from_2x2_unitary(U))
            self.assertLessEqual(la.norm(abg - abg_recon), self.RELTOL)

    def test_angles_from_2x2_unitaries_matches_reference(self):
        angles_in = np.vstack(random_euler_angles(20))
        fundamental = _fundamental_spinj()
        Us = fundamental.unitaries_from_angles(*angles_in)

        def reference_from_kevin(R):
            beta = 2 * np.arccos(np.real(np.sqrt(R[0, 0] * R[1, 1])))
            alpha = np.angle(-1.j * R[0, 0] * R[0, 1] / (np.sin(beta / 2) * np.cos(beta / 2)))
            if alpha < 0:
                alpha += 2 * np.pi
            gamma = np.angle(-1.j * R[0, 0] * R[1, 0] / (np.sin(beta / 2) * np.cos(beta / 2)))
            if gamma < 0:
                gamma += 2 * np.pi
            if np.isclose(np.exp(1.j * (alpha + gamma) / 2) * np.cos(beta / 2) / R[0, 0], -1):
                gamma += 2 * np.pi
            return alpha, beta, gamma

        angles_out_expect = np.column_stack([reference_from_kevin(U) for U in Us])
        angles_out_actual = np.vstack(angles_from_2x2_unitaries(Us))
        self.assertLessEqual(la.norm(angles_out_expect - angles_out_actual), self.RELTOL * 10)

    def test_composition_inverse(self):
        np.random.seed(0)
        fundamental = _fundamental_spinj()
        for sequence_len in np.arange(1, 16, 2):
            angles = np.column_stack(random_euler_angles(sequence_len))
            a, b, g = angles.T
            char_angles = (a[0], b[0], g[0])
            inv_angles = np.column_stack(composition_inverse(a[1:], b[1:], g[1:]))
            comp_to_char_angles = np.block([[angles], [inv_angles]])
            U_char_expect = composition_asmatrix(comp_to_char_angles)
            U_char_actual = fundamental.unitaries_from_angles(*char_angles)[0]
            discrepancy = distance_mod_phase(U_char_actual, U_char_expect)
            self.assertLessEqual(discrepancy, self.RELTOL)

class SpinJConstructionTester(BaseCase):

    def test_rejects_non_half_integer_spin(self):
        with self.assertRaises(ValueError):
            SpinJ(0.3)

    def test_rejects_negative_spin(self):
        with self.assertRaises(ValueError):
            SpinJ(-0.5)

    def test_rejects_near_half_integer_float_without_laundering(self):
        # A float that is merely *close* to a half-integer must still be rejected --
        # no tolerance-based coercion (this mirrors the exactness convention used by
        # pygsti.tools.wignersymbols._as_half_integer).
        with self.assertRaises(ValueError):
            SpinJ(0.5000000001)
        with self.assertRaises(ValueError):
            SpinJ(3.4999999999)

    def test_accepts_int_float_and_fraction(self):
        from fractions import Fraction
        for j in (1, 1.0, Fraction(2, 1), Fraction(7, 2), 3.5):
            s = SpinJ(j)
            self.assertEqual(s.dim, round(2 * float(j)) + 1)

    def test_dim_and_spins(self):
        s = SpinJ(1.5)
        self.assertEqual(s.dim, 4)
        self.assertArraysAlmostEqual(s.spins, np.array([1.5, 0.5, -0.5, -1.5]))


class EigReconstructionTester(BaseCase):
    """`eigJx`/`VJx` and `eigJy`/`VJy` should reconstruct Jx and Jy (the 'eign'
    pattern from the branch, now routed through matrixtools.eigendecomposition
    with assume_normal=True)."""

    RELTOL = 1e-12

    def _check(self, s):
        recon_Jx = reconstruct_eigendecomp(s.eigJx, s.VJx)
        self.assertLessEqual(la.norm(recon_Jx - s.Jx), self.RELTOL)
        recon_Jy = reconstruct_eigendecomp(s.eigJy, s.VJy)
        self.assertLessEqual(la.norm(recon_Jy - s.Jy), self.RELTOL)

    def test_fundamental(self):
        self._check(_fundamental_spinj())

    def test_spin72(self):
        self._check(SpinJ(3.5))

    def test_several_spins(self):
        for j in (0.5, 1, 1.5, 2.5, 5):
            self._check(SpinJ(j))


class SpinJExpmTester(BaseCase):

    RELTOL = 1e-12
    thetas = np.linspace(0, 4 * np.pi, num=10)

    def _check_expm_iJx_single(self, s):
        for t in self.thetas:
            actual = s.expm_iJx(t)[0, :, :]
            expect = la.expm(1j * t * s.Jx)
            abstol = self.RELTOL * max(la.norm(actual), la.norm(expect), 1.0)
            self.assertLessEqual(la.norm(actual - expect), abstol)

    def _check_expm_iJx_batch(self, s):
        actual = s.expm_iJx(self.thetas)
        expect = np.array([la.expm(1j * t * s.Jx) for t in self.thetas])
        abstol = self.RELTOL * max(la.norm(actual), la.norm(expect))
        self.assertLessEqual(la.norm(actual - expect), abstol)

    def _check_expm_iJy_single(self, s):
        for t in self.thetas:
            actual = s.expm_iJy(t)[0, :, :]
            expect = la.expm(1j * t * s.Jy)
            abstol = self.RELTOL * max(la.norm(actual), la.norm(expect), 1.0)
            self.assertLessEqual(la.norm(actual - expect), abstol)

    def _check_expm_iJy_batch(self, s):
        actual = s.expm_iJy(self.thetas)
        expect = np.array([la.expm(1j * t * s.Jy) for t in self.thetas])
        abstol = self.RELTOL * max(la.norm(actual), la.norm(expect))
        self.assertLessEqual(la.norm(actual - expect), abstol)

    def test_fundamental(self):
        s = _fundamental_spinj()
        self._check_expm_iJx_single(s)
        self._check_expm_iJx_batch(s)
        self._check_expm_iJy_single(s)
        self._check_expm_iJy_batch(s)

    def test_spin72(self):
        s = SpinJ(3.5)
        self._check_expm_iJx_single(s)
        self._check_expm_iJx_batch(s)
        self._check_expm_iJy_single(s)
        self._check_expm_iJy_batch(s)


class MultiSpinPropertyTester(BaseCase):
    """Property tests of M, F, C, and the block-diagonalization/character machinery,
    at several spins."""

    SPINS = (0.5, 1, 1.5, 2.5, 3.5)

    def test_synthetic_spam_matrix_orthogonal(self):
        for j in self.SPINS:
            s = SpinJ(j)
            M = s.synthetic_spam_matrix
            self.assertArraysAlmostEqual(M @ M.T, np.eye(s.dim), places=12)

    def test_decay_recoupling_matrix_row0_allones_and_symmetric(self):
        for j in self.SPINS:
            s = SpinJ(j)
            F = s.decay_recoupling_matrix
            self.assertArraysAlmostEqual(F[0, :], np.ones(s.dim), places=12)
            self.assertArraysAlmostEqual(F, F.T, places=12)

    def test_clebsch_gordan_cob_orthogonal(self):
        for j in self.SPINS:
            s = SpinJ(j)
            C = s.clebsch_gordan_cob
            self.assertArraysAlmostEqual(C @ C.T, np.eye(s.dim ** 2), places=12)

    def test_superop_stdmx_cob_block_diagonalizes(self):
        rng = np.random.default_rng(0)
        for j in self.SPINS:
            s = SpinJ(j)
            d2 = s.dim ** 2
            mask = np.zeros((d2, d2), dtype=bool)
            idx = 0
            for bsz in s.irrep_block_sizes:
                mask[idx:idx + bsz, idx:idx + bsz] = True
                idx += bsz
            for _ in range(3):
                w = rng.standard_normal(3)
                U = la.expm(1j * (w[0] * s.Jx + w[1] * s.Jy + w[2] * s.Jz))
                V = unitary_to_std_process_mx(U)  # kron(U, conj(U)), pyGSTi's convention
                W = s.superop_stdmx_cob @ V @ s.superop_stdmx_cob.conj().T
                self.assertLessEqual(la.norm(W[~mask]), 1e-12)
                # The plan's convention card states the block-diagonalizer is applied
                # as cob @ kron(conj(U), U) @ cob^H (note the reversed tensor-factor
                # order relative to pyGSTi's unitary_to_std_process_mx); confirm that
                # convention is equally block-diagonalized by the same cob.
                V_card = np.kron(U.conj(), U)
                W_card = s.superop_stdmx_cob @ V_card @ s.superop_stdmx_cob.conj().T
                self.assertLessEqual(la.norm(W_card[~mask]), 1e-12)

    def test_irrep_stdmx_projectors_are_isotypic_projectors(self):
        # Guards against the branch's latent bug where the projector built from
        # cob's *columns* has the right rank but does not commute with the
        # representation (only cob's *rows* give a valid isotypic projector).
        rng = np.random.default_rng(1)
        for j in self.SPINS:
            s = SpinJ(j)
            projectors = s.irrep_stdmx_projectors
            total = sum(projectors)
            self.assertArraysAlmostEqual(total, np.eye(s.dim ** 2), places=10)
            for bsz, P in zip(s.irrep_block_sizes, projectors):
                self.assertAlmostEqual(np.trace(P).real, bsz, places=10)
                self.assertArraysAlmostEqual(P @ P, P, places=10)  # idempotent
                self.assertArraysAlmostEqual(P, P.conj().T, places=10)  # Hermitian
            w = rng.standard_normal(3)
            U = la.expm(1j * (w[0] * s.Jx + w[1] * s.Jy + w[2] * s.Jz))
            V = unitary_to_std_process_mx(U)
            for P in projectors:
                self.assertLessEqual(la.norm(P @ V - V @ P), 1e-10)

    def test_stdmx_twirl_preserves_identity_and_commutes(self):
        rng = np.random.default_rng(2)
        for j in self.SPINS:
            s = SpinJ(j)
            d2 = s.dim ** 2
            I = np.eye(d2)
            self.assertArraysAlmostEqual(s.stdmx_twirl(I), I, places=10)

            A = rng.standard_normal((d2, d2))
            A = A + A.T
            tA = s.stdmx_twirl(A)
            w = rng.standard_normal(3)
            U = la.expm(1j * (w[0] * s.Jx + w[1] * s.Jy + w[2] * s.Jz))
            V = unitary_to_std_process_mx(U)
            self.assertLessEqual(la.norm(tA @ V - V @ tA), 1e-9)

    def test_charactercores_are_legendre_of_cos_beta(self):
        from scipy.special import eval_legendre
        for j in self.SPINS:
            s = SpinJ(j)
            np.random.seed(0)
            aa = np.array(random_euler_angles(5)).T
            actual = charactercores_from_euler_angles(s.irrep_labels, aa)
            expect = np.array([
                [eval_legendre(k, np.cos(aa_i[1])) for k in s.irrep_labels]
                for aa_i in aa
            ])
            self.assertArraysAlmostEqual(actual, expect, places=12)

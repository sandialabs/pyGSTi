import numpy as np

import pygsti.baseobjs.basisconstructors as bc
from ..util import BaseCase


class BasisConstructorsTester(BaseCase):
    def test_GellMann(self):
        id2x2 = np.array([[1, 0], [0, 1]])
        sigmax = np.array([[0, 1], [1, 0]])
        sigmay = np.array([[0, -1.0j], [1.0j, 0]])
        sigmaz = np.array([[1, 0], [0, -1]])

        # Gell-Mann 2x2 matrices should just be the sigma matrices
        GM2_mxs = bc.gm_matrices_unnormalized(2)
        self.assertTrue(len(GM2_mxs) == 4)
        self.assertArraysAlmostEqual(GM2_mxs[0], id2x2)
        self.assertArraysAlmostEqual(GM2_mxs[1], sigmax)
        self.assertArraysAlmostEqual(GM2_mxs[2], sigmay)
        self.assertArraysAlmostEqual(GM2_mxs[3], sigmaz)
        with self.assertRaises(TypeError):
            bc.gm_matrices_unnormalized("FooBar")  # arg must be tuple,list,or int

        # Normalized Gell-Mann 2x2 matrices should just be the sigma matrices / sqrt(2)
        NGM2_mxs = bc.gm_matrices(2)
        self.assertTrue(len(NGM2_mxs) == 4)
        self.assertArraysAlmostEqual(NGM2_mxs[0], id2x2 / np.sqrt(2))
        self.assertArraysAlmostEqual(NGM2_mxs[1], sigmax / np.sqrt(2))
        self.assertArraysAlmostEqual(NGM2_mxs[2], sigmay / np.sqrt(2))
        self.assertArraysAlmostEqual(NGM2_mxs[3], sigmaz / np.sqrt(2))

        #TODO: test 4x4 matrices?

    def test_orthogonality(self):
        #Gell Mann
        dim = 5
        mxs = bc.gm_matrices(dim)
        N = len(mxs); self.assertTrue(N == dim**2)

        gm_trMx = np.zeros((N, N), 'complex')
        for i in range(N):
            for j in range(N):
                gm_trMx[i, j] = np.trace(np.dot(np.conjugate(np.transpose(mxs[i])), mxs[j]))
                #Note: conjugate transpose not needed since mxs are Hermitian
        self.assertArraysAlmostEqual(gm_trMx, np.identity(N, 'complex'))

        #Std Basis
        dim = 5
        mxs = bc.std_matrices(dim)
        N = len(mxs); self.assertTrue(N == dim**2)

        std_trMx = np.zeros((N, N), 'complex')
        for i in range(N):
            for j in range(N):
                std_trMx[i, j] = np.trace(np.dot(np.conjugate(np.transpose(mxs[i])), mxs[j]))
        self.assertArraysAlmostEqual(std_trMx, np.identity(N, 'complex'))

        #Pauli-product basis
        dim = 4
        mxs = bc.pp_matrices(dim)
        N = len(mxs); self.assertTrue(N == dim**2)

        with self.assertRaises(TypeError):
            bc.pp_matrices("Foobar")  # dim must be an int
        with self.assertRaises(ValueError):
            bc.pp_matrices(3)  # dim must be a power of 4

        specialCase = bc.pp_matrices(1)  # single 1x1 identity mx
        self.assertEqual(specialCase, [np.identity(1, 'complex')])

        pp_trMx = np.zeros((N, N), 'complex')
        for i in range(N):
            for j in range(N):
                pp_trMx[i, j] = np.trace(np.dot(np.conjugate(np.transpose(mxs[i])), mxs[j]))
                #Note: conjugate transpose not needed since mxs are Hermitian
        self.assertArraysAlmostEqual(pp_trMx, np.identity(N, 'complex'))

    def test_basis_misc(self):
        mx = bc.pp_matrices(1)  # was [1] but this shouldn't be allowed
        self.assertArraysAlmostEqual(np.identity(1, 'complex'), mx)

    def test_pp_maxweight(self):
        pp2Max1 = bc.pp_matrices(2, max_weight=1)  # using max_weight
        pp2 = bc.pp_matrices(2) # For 2x2, should match max_weight=1
        for mxMax, mx in zip(pp2Max1, pp2):
            self.assertArraysAlmostEqual(mxMax, mx)

        pp4Max1 = bc.pp_matrices(4, max_weight=1)
        pp4 = bc.pp_matrices(4)
        pp4Subset = [pp4[0], pp4[1], pp4[2], pp4[3], pp4[4], pp4[8], pp4[12]] # Pull out II,IX,IY,IZ,XI,YI,ZI
        for mxMax, mxSub in zip(pp4Max1, pp4Subset):
            self.assertArraysAlmostEqual(mxMax, mxSub)

    def test_qt_dim1(self):
        qutrit1 = bc.qt_matrices(1)  # special case when dim==1
        self.assertArraysAlmostEqual(np.identity(1, 'd'), qutrit1)

    def test_qt_orthonorm(self):
        mxs = bc.qt_matrices(3)
        for i in range(len(mxs)):
            for j in range(len(mxs)):
                dp = np.vdot(mxs[i], mxs[j])
                if i == j:
                    self.assertAlmostEqual(dp, 1.0)
                else:
                    self.assertAlmostEqual(dp, 0.0)

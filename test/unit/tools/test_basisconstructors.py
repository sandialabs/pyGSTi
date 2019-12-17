import numpy as np

from ..util import BaseCase

import pygsti.tools.basisconstructors as bc


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
        bc.pp_matrices(1)  # was [1] but this shouldn't be allowed
        # TODO assert correctness

    def test_pp_maxweight(self):
        ppMax1 = bc.pp_matrices(2, maxWeight=1)  # using maxWeight
        # TODO assert correctness

    def test_qt_dim1(self):
        qutrit1 = bc.qt_matrices(1)  # special case when dim==1
        # TODO assert correctness

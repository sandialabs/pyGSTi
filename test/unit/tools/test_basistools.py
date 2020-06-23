import numpy as np
import scipy
from functools import partial

from ..util import BaseCase

import pygsti.tools.basistools as bt
import pygsti.tools.lindbladtools as lindbladtools

from pygsti.objects.basis import Basis, ExplicitBasis, DirectSumBasis


class BasisToolsTester(BaseCase):
    def test_basis_element_labels(self):
        basisnames = ['gm', 'std', 'pp']

        # One dimensional gm
        self.assertEqual([''], bt.basis_element_labels('gm', 1))

        # Two dimensional
        expectedLabels = [
            ['I', 'X', 'Y', 'Z'],
            ['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
            ['I', 'X', 'Y', 'Z']
        ]
        labels = [bt.basis_element_labels(basisname, 4) for basisname in basisnames]
        self.assertEqual(labels, expectedLabels)

        with self.assertRaises(AssertionError):
            bt.basis_element_labels('asdklfasdf', 4)

        # Non power of two for pp labels:
        with self.assertRaises(ValueError):
            label = bt.basis_element_labels('pp', 9)
            # TODO assert correctness

        # Single list arg for pp labels
        self.assertEqual(bt.basis_element_labels('pp', 4), ['I', 'X', 'Y', 'Z'])

        # Four dimensional+
        expectedLabels = [
            ['I', 'X_{0,1}', 'X_{0,2}', 'X_{0,3}', 'X_{1,2}', 'X_{1,3}', 'X_{2,3}', 'Y_{0,1}', 'Y_{0,2}', 'Y_{0,3}',
             'Y_{1,2}', 'Y_{1,3}', 'Y_{2,3}', 'Z_{1}', 'Z_{2}', 'Z_{3}'],
            ['(0,0)', '(0,1)', '(0,2)', '(0,3)', '(1,0)', '(1,1)', '(1,2)', '(1,3)', '(2,0)', '(2,1)', '(2,2)', '(2,3)',
             '(3,0)', '(3,1)', '(3,2)', '(3,3)'],
            ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
        ]
        labels = [bt.basis_element_labels(basisname, 16) for basisname in basisnames]
        self.assertEqual(expectedLabels, labels)

    def test_basis_longname(self):
        longnames = {bt.basis_longname(b) for b in {'gm', 'std', 'pp', 'qt'}}
        self.assertEqual(longnames, {'Gell-Mann basis', 'Matrix-unit basis', 'Pauli-Product basis', 'Qutrit basis'})
        with self.assertRaises(KeyError):
            bt.basis_longname('not a basis')

    def test_expand_contract(self):
        # matrix that operates on 2x2 density matrices, but only on the 0-th and 3-rd
        # elements which correspond to the diagonals of the 2x2 density matrix.
        mxInStdBasis = np.array([[1,0,0,2],
                                 [0,0,0,0],
                                 [0,0,0,0],
                                 [3,0,0,4]], 'd')

        # Reduce to a matrix operating on a density matrix space with 2 1x1 blocks (hence [1,1])
        begin = Basis.cast('std', [1, 1])
        end = Basis.cast('std', 4)

        mxInReducedBasis = bt.resize_std_mx(mxInStdBasis, 'contract', end, begin)
        #mxInReducedBasis = bt.change_basis(mxInStdBasis, begin, end)
        notReallyContracted = bt.change_basis(mxInStdBasis, 'std', 'std')  # 4
        correctAnswer = np.array([[ 1.0,  2.0],
                                  [ 3.0,  4.0]])
        self.assertArraysAlmostEqual(mxInReducedBasis, correctAnswer)
        self.assertArraysAlmostEqual(notReallyContracted, mxInStdBasis)

        expandedMx = bt.resize_std_mx(mxInReducedBasis, 'expand', begin, end)
        #expandedMx = bt.change_basis(mxInReducedBasis, end, begin)
        expandedMxAgain = bt.change_basis(expandedMx, 'std', 'std')  # , 4)
        self.assertArraysAlmostEqual(expandedMx, mxInStdBasis)
        self.assertArraysAlmostEqual(expandedMxAgain, mxInStdBasis)

    def test_transforms(self):
        mxStd = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1]], 'complex')
        vecStd = np.array([1,0,0,0], 'complex')

        change = bt.change_basis
        mxGM = change(mxStd, 'std', 'gm')
        mxStd2 = change(mxGM, 'gm', 'std')
        self.assertArraysAlmostEqual(mxStd, mxStd2)

        vecGM = change(vecStd, 'std', 'gm')
        vecStd2 = change(vecGM, 'gm', 'std')
        self.assertArraysAlmostEqual(vecStd, vecStd2)

        mxPP = change(mxStd, 'std', 'pp')
        mxStd2 = change(mxPP, 'pp', 'std')
        self.assertArraysAlmostEqual(mxStd, mxStd2)

        vecPP = change(vecStd, 'std', 'pp')
        vecStd2 = change(vecPP, 'pp', 'std')
        self.assertArraysAlmostEqual(vecStd, vecStd2)

        mxPP2 = change(mxGM, 'gm', 'pp')
        self.assertArraysAlmostEqual(mxPP, mxPP2)

        vecPP2 = change(vecGM, 'gm', 'pp')
        self.assertArraysAlmostEqual(vecPP, vecPP2)

        mxGM2 = change(mxPP, 'pp', 'gm')
        self.assertArraysAlmostEqual(mxGM, mxGM2)

        vecGM2 = change(vecPP, 'pp', 'gm')
        self.assertArraysAlmostEqual(vecGM, vecGM2)

        non_herm_mxStd = np.array([[1,0,2,3j],
                                   [0,1,0,2],
                                   [0,0,1,0],
                                   [0,0,0,1]], 'complex')
        non_herm_vecStd = np.array([1,0,2,3j], 'complex')  # ~ non-herm 2x2 density mx
        rank3tensor = np.ones((4, 4, 4), 'd')

        with self.assertRaises(ValueError):
            change(non_herm_mxStd, 'std', 'gm')  # will result in gm mx with *imag* part
        with self.assertRaises(ValueError):
            change(non_herm_vecStd, 'std', 'gm')  # will result in gm vec with *imag* part
        with self.assertRaises(ValueError):
            change(non_herm_mxStd, 'std', 'pp')  # will result in pp mx with *imag* part
        with self.assertRaises(ValueError):
            change(non_herm_vecStd, 'std', 'pp')  # will result in pp vec with *imag* part

        with self.assertRaises(ValueError):
            change(rank3tensor, 'std', 'gm')  # only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'gm', 'std')  # only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'std', 'pp')  # only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'pp', 'std')  # only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'gm', 'pp')  # only convert rank 1 & 2 objects
        with self.assertRaises(ValueError):
            change(rank3tensor, 'pp', 'gm')  # only convert rank 1 & 2 objects

        densityMx = np.array([[1, 0], [0, -1]], 'complex')
        gmVec = bt.stdmx_to_gmvec(densityMx)
        ppVec = bt.stdmx_to_ppvec(densityMx)
        stdVec = bt.stdmx_to_stdvec(densityMx)
        self.assertArraysAlmostEqual(gmVec, np.array([[0], [0], [0], [np.sqrt(2)]], 'd'))
        self.assertArraysAlmostEqual(ppVec, np.array([[0], [0], [0], [np.sqrt(2)]], 'd'))
        self.assertArraysAlmostEqual(stdVec, np.array([[1], [0], [0], [-1]], 'complex'))

        mxFromGM = bt.gmvec_to_stdmx(gmVec)
        mxFromPP = bt.ppvec_to_stdmx(ppVec)
        mxFromStd = bt.stdvec_to_stdmx(stdVec)
        self.assertArraysAlmostEqual(mxFromGM, densityMx)
        self.assertArraysAlmostEqual(mxFromPP, densityMx)
        self.assertArraysAlmostEqual(mxFromStd, densityMx)

    def test_few_qubit_fns(self):
        state_vec = np.array([1, 0], 'complex')
        dmVec = bt.state_to_pauli_density_vec(state_vec)
        self.assertArraysAlmostEqual(dmVec, np.array([[0.70710678], [0], [0], [0.70710678]], 'complex'))

        stdMx = np.array([[1, 0], [0, 0]], 'complex')  # density matrix
        pauliVec = bt.stdmx_to_ppvec(stdMx)
        self.assertArraysAlmostEqual(pauliVec, np.array([[0.70710678], [0], [0], [0.70710678]], 'complex'))

        stdMx2 = bt.ppvec_to_stdmx(pauliVec)
        self.assertArraysAlmostEqual(stdMx, stdMx2)

    def test_vec_to_stdmx(self):
        vec = np.zeros(shape=(4,))
        for b in {'gm', 'pp', 'std'}:
            bt.vec_to_stdmx(vec, b)
        with self.assertRaises(AssertionError):
            bt.vec_to_stdmx(vec, 'akdfj;ladskf')

    def test_auto_expand(self):
        comp = Basis.cast([('std', 4,), ('std', 1)])
        std = Basis.cast('std', 9)
        mxStd = np.identity(5)
        test = bt.resize_std_mx(mxStd, 'expand', comp, std)
        # TODO assert intermediate correctness
        test2 = bt.resize_std_mx(test, 'contract', std, comp)
        self.assertArraysAlmostEqual(test2, mxStd)

    def test_flexible_change_basis(self):
        comp = Basis.cast([('gm', 4,), ('gm', 1)])
        std = Basis.cast('std', 9)
        mx = np.identity(5)
        test = bt.flexible_change_basis(mx, comp, std)
        self.assertEqual(test.shape[0], comp.elsize)
        test2 = bt.flexible_change_basis(test, std, comp)
        self.assertArraysAlmostEqual(test2, mx)

    def test_change_between_composites(self):
        a = Basis.cast('std', [4, 1])
        b = Basis.cast('gm', [4, 1])
        mxStd = np.identity(5)
        test = bt.change_basis(mxStd, a, b)
        self.assertEqual(test.shape, mxStd.shape)
        test2 = bt.change_basis(test, b, a)
        self.assertArraysAlmostEqual(test2, mxStd)

    def test_general(self):
        std = Basis.cast('std', 4)
        std4 = Basis.cast('std', 16)
        std2x2 = Basis.cast([('std', 4), ('std', 4)])
        gm = Basis.cast('gm', 4)

        from_basis, to_basis = bt.create_basis_pair(np.identity(4, 'd'), "std", "gm")
        from_basis, to_basis = bt.create_basis_pair(np.identity(4, 'd'), std, "gm")
        from_basis, to_basis = bt.create_basis_pair(np.identity(4, 'd'), "std", gm)

        mx = np.array([
            [1, 0, 0, 1],
            [0, 1, 2, 0],
            [0, 2, 1, 0],
            [1, 0, 0, 1]
        ])

        bt.change_basis(mx, 'std', 'gm')  # shortname lookup
        bt.change_basis(mx, std, gm)  # object
        bt.change_basis(mx, std, 'gm')  # combination
        bt.flexible_change_basis(mx, std, gm)  # same dimension
        I2x2 = np.identity(8, 'd')
        I4 = bt.flexible_change_basis(I2x2, std2x2, std4)
        self.assertArraysAlmostEqual(bt.flexible_change_basis(I4, std4, std2x2), I2x2)

        with self.assertRaises(AssertionError):
            bt.change_basis(mx, std, std4)  # basis size mismatch

        mxInStdBasis = np.array([[1,0,0,2],
                                 [0,0,0,0],
                                 [0,0,0,0],
                                 [3,0,0,4]], 'd')

        begin = Basis.cast('std', [1, 1])
        end = Basis.cast('std', 4)
        mxInReducedBasis = bt.resize_std_mx(mxInStdBasis, 'contract', end, begin)
        original = bt.resize_std_mx(mxInReducedBasis, 'expand', begin, end)
        # TODO assert correctness

    def test_sparse_lindblad_bases(self):
        sparsePP = Basis.cast("pp", 16, sparse=True)
        mxs = sparsePP.elements
        for lbl, mx in zip(sparsePP.labels, mxs):
            print("{}: {} matrix with {} nonzero entries (of {} total)".format(
                lbl, mx.shape, mx.nnz, mx.shape[0] * mx.shape[1]
            ))
            print(mx.toarray())
        print("{} basis elements".format(len(sparsePP)))
        self.assertEqual(len(sparsePP), 16)

        # TODO assert correctness

        M = np.ones((16, 16), 'd')
        v = np.ones(16, 'd')
        S = scipy.sparse.identity(16, 'd', 'csr')

        print("Test types after basis change by sparse basis:")
        Mout = bt.change_basis(M, sparsePP, 'std')
        vout = bt.change_basis(v, sparsePP, 'std')
        Sout = bt.change_basis(S, sparsePP, 'std')
        print("{} -> {}".format(type(M), type(Mout)))
        print("{} -> {}".format(type(v), type(vout)))
        print("{} -> {}".format(type(S), type(Sout)))
        self.assertIsInstance(Mout, np.ndarray)
        self.assertIsInstance(vout, np.ndarray)
        self.assertIsInstance(Sout, scipy.sparse.csr_matrix)

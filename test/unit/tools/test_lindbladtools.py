import numpy as np
import scipy.sparse as sps

from pygsti.tools import lindbladtools as lt
from pygsti.baseobjs import Basis
from ..util import BaseCase


class LindbladToolsTester(BaseCase):
    def test_hamiltonian_to_lindbladian(self):
        expectedLindbladian = np.array([
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]
        ])
        self.assertArraysAlmostEqual(lt.create_elementary_errorgen('H', np.zeros(shape=(2, 2))),
                                     expectedLindbladian)
        sparse = sps.csr_matrix(np.zeros(shape=(2, 2)))
        #spL = lt.hamiltonian_to_lindbladian(sparse, True)
        spL = lt.create_elementary_errorgen('H', sparse, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expectedLindbladian)

    def test_stochastic_lindbladian(self):
        a = np.array([[1, 2], [3, 4]], 'd')
        expected = np.array([
            [ 1,  2,  2,  4],
            [ 3,  4,  6,  8],
            [ 3,  6,  4,  8],
            [ 9, 12, 12, 16]
        ], 'd')
        dual_eg, norm = lt.create_elementary_errorgen_dual('S', a, normalization_factor='auto_return')
        self.assertArraysAlmostEqual(
            dual_eg * norm, expected)
        sparse = sps.csr_matrix(a)
        spL = lt.create_elementary_errorgen_dual('S', sparse, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray() * norm, expected)

    def test_nonham_lindbladian(self):
        a = np.array([[1, 2], [3, 4]], 'd')
        b = np.array([[1, 2], [3, 4]], 'd')
        expected = np.array([
            [ -9,  -5,  -5,  4],
            [ -4, -11,   6,  1],
            [ -4,   6, -11,  1],
            [  9,   5,   5, -4]
        ], 'd')
        self.assertArraysAlmostEqual(lt.create_lindbladian_term_errorgen('O', a, b), expected)
        sparsea = sps.csr_matrix(a)
        sparseb = sps.csr_matrix(b)
        spL = lt.create_lindbladian_term_errorgen('O', sparsea, sparseb, sparse=True)
        self.assertArraysAlmostEqual(spL.toarray(),
                                     expected)

    def test_elementary_errorgen_bases(self):

        bases = [Basis.cast('gm', 4),
                 Basis.cast('pp', 4),
                 Basis.cast('PP', 4)]

        for basis in bases:
            print(basis)

            primals = []; duals = []; lbls = []
            for lbl, bel in zip(basis.labels[1:], basis.elements[1:]):
                lbls.append("H_%s" % lbl)
                primals.append(lt.create_elementary_errorgen('H', bel)) 
                duals.append(lt.create_elementary_errorgen_dual('H', bel)) 
            for lbl, bel in zip(basis.labels[1:], basis.elements[1:]):
                lbls.append("S_%s" % lbl)
                primals.append(lt.create_elementary_errorgen('S', bel))
                duals.append(lt.create_elementary_errorgen_dual('S', bel)) 
            for i, (lbl, bel) in enumerate(zip(basis.labels[1:], basis.elements[1:])):
                for lbl2, bel2 in zip(basis.labels[1+i+1:], basis.elements[1+i+1:]):
                    lbls.append("C_%s_%s" % (lbl, lbl2))
                    primals.append(lt.create_elementary_errorgen('C', bel, bel2))
                    duals.append(lt.create_elementary_errorgen_dual('C', bel, bel2)) 
            for i, (lbl, bel) in enumerate(zip(basis.labels[1:], basis.elements[1:])):
                for lbl2, bel2 in zip(basis.labels[1+i+1:], basis.elements[1+i+1:]):
                    lbls.append("A_%s_%s" % (lbl, lbl2))
                    primals.append(lt.create_elementary_errorgen('A', bel, bel2))
                    duals.append(lt.create_elementary_errorgen_dual('A', bel, bel2)) 

            dot_mx = np.empty((len(duals), len(primals)), complex)
            for i, dual in enumerate(duals):
                for j, primal in enumerate(primals):
                    dot_mx[i,j] = np.vdot(dual, primal)

            self.assertTrue(np.allclose(dot_mx, np.identity(len(lbls), 'd')))

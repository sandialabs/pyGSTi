import numpy as np

from ..util import BaseCase

from pygsti.objects import basis
import pygsti.tools.basistools as bt


class BasisTester(BaseCase):
    def test_composite_basis(self):
        comp = basis.Basis.cast([('std', 4,), ('std', 1)])
        # TODO assert correctness

        a = basis.Basis.cast([('std', 4), ('std', 4)])
        b = basis.Basis.cast('std', [4, 4])
        self.assertEqual(len(a), len(b))
        self.assertArraysAlmostEqual(np.array(a.elements), np.array(b.elements))

    def test_qt(self):
        qt = basis.Basis.cast('qt', 9)
        qt = basis.Basis.cast('qt', [9])
        # TODO assert correctness

    def test_basis_casting(self):
        basis.Basis.cast('pp', 4)
        basis.Basis.cast('std', [4, 1])
        # TODO assert correctness
        with self.assertRaises(AssertionError):
            basis.Basis.cast([('std', 16), ('gm', 4)])  # inconsistent .real values of components!

        gm = basis.Basis.cast('gm', 4)
        ungm = basis.Basis.cast('gm_unnormalized', 4)
        empty = basis.Basis.cast([])  # special "empty" basis
        self.assertEqual(empty.name, "*Empty*")
        gm_mxs = gm.elements
        unnorm = basis.ExplicitBasis([gm_mxs[0], 2 * gm_mxs[1]])

        self.assertTrue(gm.is_normalized())
        self.assertFalse(ungm.is_normalized())
        self.assertFalse(unnorm.is_normalized())

        composite = basis.DirectSumBasis([gm, gm])
        # TODO assert correctness

        comp = basis.DirectSumBasis([gm, gm], name='comp', longname='CustomComposite')

        comp = basis.DirectSumBasis([gm, gm], name='comp', longname='CustomComposite')
        comp._labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # TODO: make a set_labels?

        std2x2Matrices = np.array([
            [[1, 0],
             [0, 0]],

            [[0, 1],
             [0, 0]],

            [[0, 0],
             [1, 0]],

            [[0, 0],
             [0, 1]]
        ], 'complex')

        empty = basis.ExplicitBasis([])
        alt_standard = basis.ExplicitBasis(std2x2Matrices)
        print("MXS = \n", alt_standard.elements)
        alt_standard = basis.ExplicitBasis(std2x2Matrices,
                                           name='std',
                                           longname='Standard')
        self.assertEqual(alt_standard, std2x2Matrices)

    def test_basis_object(self):
        # test a few aspects of a Basis object that other tests miss...
        b = basis.Basis.cast("pp", 4)
        beq = b.simple_equivalent()
        longnm = bt.basis_longname(b)
        lbls = bt.basis_element_labels(b, None)

        raw_mxs = bt.basis_matrices("pp", 4)
        # TODO assert correctness for all

        with self.assertRaises(AssertionError):
            bt.basis_matrices("foobar", 4)  # invalid basis name

        print("Dim = ", repr(b.dim))  # calls Dim.__repr__

    def test_sparse_basis(self):
        sparsePP = basis.Basis.cast("pp", 4, sparse=True)
        sparsePP2 = basis.Basis.cast("pp", 4, sparse=True)
        sparseBlockPP = basis.Basis.cast("pp", [4, 4], sparse=True)
        sparsePP_2Q = basis.Basis.cast("pp", 16, sparse=True)
        sparseGM_2Q = basis.Basis.cast("gm", 4, sparse=True)  # different sparsity structure than PP 2Q
        denseGM = basis.Basis.cast("gm", 4, sparse=False)

        mxs = sparsePP.elements
        block_mxs = sparseBlockPP.elements
        # TODO assert correctness

        expeq = sparsePP.simple_equivalent()
        block_expeq = sparseBlockPP.simple_equivalent()
        # TODO assert correctness

        raw_mxs = bt.basis_matrices("pp", 4, sparse=True)

        #test equality of bases with other bases and matrices
        self.assertEqual(sparsePP, sparsePP2)
        self.assertEqual(sparsePP, raw_mxs)
        self.assertNotEqual(sparsePP, sparsePP_2Q)
        self.assertNotEqual(sparsePP_2Q, sparseGM_2Q)

        #sparse transform matrix
        trans = sparsePP.transform_matrix(sparsePP2)
        self.assertArraysAlmostEqual(trans, np.identity(4, 'd'))
        trans2 = sparsePP.transform_matrix(denseGM)
        # TODO assert correctness

        #test equality for large bases
        large_sparsePP = basis.Basis.cast("pp", 256, sparse=True)
        large_sparsePP2 = basis.Basis.cast("pp", 256, sparse=True)
        self.assertEqual(large_sparsePP, large_sparsePP2)

    def test_basis_cast(self):
        pp1 = basis.Basis.cast('pp', 16)
        pp2 = basis.Basis.cast('pp', (4, 4))
        pp3 = basis.Basis.cast('pp', [(4, 4)])
        self.assertTrue(isinstance(pp1, basis.BuiltinBasis))
        self.assertTrue(isinstance(pp2, basis.DirectSumBasis))
        self.assertTrue(isinstance(pp3, basis.DirectSumBasis))
        self.assertTrue(isinstance(pp2.component_bases[0], basis.BuiltinBasis))
        self.assertTrue(isinstance(pp2.component_bases[1], basis.BuiltinBasis))
        self.assertTrue(isinstance(pp3.component_bases[0], basis.TensorProdBasis))
        self.assertTrue(isinstance(pp3.component_bases[0].component_bases[0], basis.BuiltinBasis))
        self.assertTrue(isinstance(pp3.component_bases[0].component_bases[1], basis.BuiltinBasis))

    def test_tensorprod_basis(self):
        pp1 = basis.Basis.cast('pp', 4)  # 1Q
        tpb = basis.TensorProdBasis([pp1, pp1])
        self.assertTrue(tpb.is_simple())
        self.assertEqual(pp1.dim, 4)
        self.assertEqual(tpb.dim, 4 * 4)

    def test_directsum_basis(self):
        s1 = basis.BuiltinBasis('std', 1)
        s2 = basis.BuiltinBasis('std', 4)
        dsb = basis.DirectSumBasis([s2, s1, s2])
        self.assertEqual(dsb.dim, 4 + 1 + 4)
        self.assertEqual(len(dsb.component_bases), 3)
        self.assertEqual([x.dim for x in dsb.component_bases], [4, 1, 4])

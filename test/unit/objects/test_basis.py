import numpy as np

import pygsti.baseobjs.basisconstructors as bc
import pygsti.tools.basistools as bt
from pygsti.baseobjs import basis
from ..util import BaseCase


class BasisTester(BaseCase):
    def test_composite_basis(self):
        comp = basis.Basis.cast([('std', 4,), ('std', 1)])
        b4 = basis.Basis.cast('std', 4)
        b1 = basis.Basis.cast('std', 1)
        for mcomp, mb4 in zip(comp.elements[:4], b4.elements):
            # Pad the standalone matrix up to what composite should be
            padded = np.pad(mb4, [(0,1), (0,1)])
            self.assertArraysAlmostEqual(mcomp, padded)
        for mcomp, mb1 in zip(comp.elements[4:], b1.elements):
            padded = np.pad(mb1, [(2,0), (2,0)])
            self.assertArraysAlmostEqual(mcomp, padded)

        a = basis.Basis.cast([('std', 4), ('std', 4)])
        b = basis.Basis.cast('std', [4, 4])
        self.assertEqual(len(a), len(b))
        self.assertArraysAlmostEqual(np.array(a.elements), np.array(b.elements))

    def test_qt(self):
        qt = basis.Basis.cast('qt', 9)
        mats = bc.qt_matrices(3)
        self.assertArraysAlmostEqual(qt.elements, mats)

        qt = basis.Basis.cast('qt', [9])
        self.assertArraysAlmostEqual(qt.elements, mats)

    def test_basis_casting(self):
        pp = basis.Basis.cast('pp', 4)
        # Test correctness with Pauli matrix identities (s_x^2 = s_y^2 = s_z^2 = I)
        for mx in pp.elements:
            self.assertArraysAlmostEqual(np.dot(mx, mx), 0.5*np.eye(2))
        
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
        altComposite = basis.Basis.cast([('gm', 4), ('gm', 4)])
        self.assertArraysAlmostEqual(composite.elements, altComposite.elements)

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
        beq = b.create_simple_equivalent()
        longnm = bt.basis_longname(b)
        lbls = bt.basis_element_labels(b, None)

        raw_mxs = bt.basis_matrices("pp", 4)
        self.assertArraysAlmostEqual(b.elements, raw_mxs)

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
        for mblock, mx in zip(block_mxs[:4], mxs):
            # Pad the standalone matrix up to what composite should be
            padded = np.pad(mx.todense(), [(0,2), (0,2)])
            self.assertArraysAlmostEqual(mblock.todense(), padded)
        for mblock, mx in zip(block_mxs[4:], mxs):
            padded = np.pad(mx.todense(), [(2,0), (2,0)])
            self.assertArraysAlmostEqual(mblock.todense(), padded)

        # Equivalent matrices should be the same size
        expeq = sparsePP.create_simple_equivalent()
        for eqmx, mx in zip(expeq.elements, sparsePP.elements):
            self.assertEqual(eqmx.shape, mx.shape)
        block_expeq = sparseBlockPP.create_simple_equivalent()
        for eqmx, mx in zip(block_expeq.elements, sparseBlockPP.elements):
            self.assertEqual(eqmx.shape, mx.shape)

        raw_mxs = bt.basis_matrices("pp", 4, sparse=True)

        #test equality of bases with other bases and matrices
        self.assertEqual(sparsePP, sparsePP2)
        self.assertEqual(sparsePP, raw_mxs)
        self.assertNotEqual(sparsePP, sparsePP_2Q)
        self.assertNotEqual(sparsePP_2Q, sparseGM_2Q)

        #sparse transform matrix
        trans = sparsePP.create_transform_matrix(sparsePP2)
        self.assertArraysAlmostEqual(trans, np.identity(4, 'd'))
        trans2 = sparsePP.create_transform_matrix(denseGM)
        trans3 = denseGM.reverse_transform_matrix(sparsePP)
        self.assertArraysAlmostEqual(np.dot(trans3, trans2), np.identity(4, 'd'))

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
        self.assertTrue(isinstance(pp3, basis.TensorProdBasis))
        self.assertTrue(isinstance(pp2.component_bases[0], basis.BuiltinBasis))
        self.assertTrue(isinstance(pp2.component_bases[1], basis.BuiltinBasis))
        self.assertTrue(isinstance(pp3.component_bases[0], basis.BuiltinBasis))
        self.assertTrue(isinstance(pp3.component_bases[1], basis.BuiltinBasis))

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


    def test_sv_basis(self):
        sv = basis.Basis.cast('sv', 7)
        self.assertTrue(sv.dim == 7)
        self.assertTrue(sv._get_dimension_to_pass_to_constructor() == 7)
        self.assertTrue(sv.elshape == (7,))
        self.assertTrue(sv.is_simple())
        self.assertTrue(len(sv.elements) == 7)
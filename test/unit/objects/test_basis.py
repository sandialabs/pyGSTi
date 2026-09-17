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

    def test_basis_serialization_deserialization_regression(self):
        # 1. BuiltinBasis
        b1 = basis.BuiltinBasis('pp', 4)
        state1 = b1.to_nice_serialization()
        b1_rec = basis.Basis.from_nice_serialization(state1)
        self.assertEqual(b1, b1_rec)
        self.assertEqual(b1.state_space, b1_rec.state_space)

        # 2. ExplicitBasis
        mats = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype='complex')
        b2 = basis.ExplicitBasis(mats, ['I', 'X'], name='CustomExplicit')
        state2 = b2.to_nice_serialization()
        b2_rec = basis.Basis.from_nice_serialization(state2)
        self.assertEqual(b2, b2_rec)
        self.assertEqual(b2.labels, b2_rec.labels)

        # 3. DirectSumBasis
        s1 = basis.BuiltinBasis('std', 1)
        s2 = basis.BuiltinBasis('std', 4)
        b3 = basis.DirectSumBasis([s2, s1])
        state3 = b3.to_nice_serialization()
        b3_rec = basis.Basis.from_nice_serialization(state3)
        self.assertEqual(b3, b3_rec)

        # 4. TensorProdBasis
        b4 = basis.TensorProdBasis([s1, s1])
        state4 = b4.to_nice_serialization()
        b4_rec = basis.Basis.from_nice_serialization(state4)
        self.assertEqual(b4, b4_rec)

    def test_basis_equality_and_hashing_regression(self):
        # Hash consistency and equality
        b1 = basis.BuiltinBasis('pp', 4)
        b2 = basis.BuiltinBasis('pp', 4)
        self.assertEqual(b1, b2)
        self.assertEqual(hash(b1), hash(b2))

        # Check sparse / dense hashes and equality
        b_dense = basis.Basis.cast('pp', 4, sparse=False)
        b_sparse = basis.Basis.cast('pp', 4, sparse=True)
        self.assertNotEqual(b_dense, b_sparse) # because sparseness_must_match defaults to True
        self.assertTrue(b_dense.is_equivalent(b_sparse, sparseness_must_match=False))

        # ExplicitBasis sparse vs dense hashing and precision tolerance
        mats = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype='complex')
        mats_close = np.array([[[1.0000001, 0.0], [0.0, 1.0]]], dtype='complex') # diff < 1e-6
        mats_far = np.array([[[1.00001, 0.0], [0.0, 1.0]]], dtype='complex')     # diff > 1e-6
        
        ex_dense = basis.ExplicitBasis(mats, ['I'])
        ex_close = basis.ExplicitBasis(mats_close, ['I'])
        ex_far = basis.ExplicitBasis(mats_far, ['I'])

        self.assertEqual(hash(ex_dense), hash(ex_close))
        self.assertNotEqual(hash(ex_dense), hash(ex_far))

        import scipy.sparse as sps
        mats_sparse = [sps.csr_matrix(m) for m in mats]
        ex_sparse = basis.ExplicitBasis(mats_sparse, ['I'], sparse=True)
        self.assertIsInstance(hash(ex_sparse), int)

        # DirectSum and TensorProd hashing
        ds1 = basis.DirectSumBasis([b1, b2])
        ds2 = basis.DirectSumBasis([b1, b2])
        self.assertEqual(ds1, ds2)
        self.assertEqual(hash(ds1), hash(ds2))

        tp1 = basis.TensorProdBasis([b1, b2])
        tp2 = basis.TensorProdBasis([b1, b2])
        self.assertEqual(tp1, tp2)
        self.assertEqual(hash(tp1), hash(tp2))

        # Compare large sparse bases warning triggers and returns False
        large_sparsePP_more = basis.Basis.cast("pp", 1024, sparse=True)
        with self.assertWarns(UserWarning):
            is_eq = large_sparsePP_more.is_equivalent(large_sparsePP_more.elements)
        self.assertFalse(is_eq)

    def test_basis_pseudo_inverses_and_partial_transform_regression(self):
        # Incomplete basis (size < dim)
        # pp basis size 4, dim 4. Let's make an explicit basis of size 2, dim 4
        pp = basis.BuiltinBasis('pp', 4)
        partial_mxs = pp.elements[:2] # shape: (2, 2, 2) -> each matrix size 4
        b_partial = basis.ExplicitBasis(partial_mxs, ['I', 'X'])
        self.assertTrue(b_partial.is_partial())
        self.assertFalse(b_partial.is_complete())

        # Check transform matrices
        to_std = b_partial.to_std_transform_matrix
        self.assertEqual(to_std.shape, (4, 2))
        from_std = b_partial.from_std_transform_matrix
        self.assertEqual(from_std.shape, (2, 4))

        # Pseudo-inverse property: from_std . to_std should be identity (size x size)
        # Since from_std = (A^H A)^-1 A^H and to_std = A:
        # from_std . to_std = (A^H A)^-1 A^H A = I
        self.assertArraysAlmostEqual(np.dot(from_std, to_std), np.identity(2, dtype='complex'))

        # Sparse incomplete basis
        pp_sparse = basis.BuiltinBasis('pp', 4, sparse=True)
        import scipy.sparse as sps
        partial_sparse_mxs = [sps.csr_matrix(m) for m in pp_sparse.elements[:2]]
        b_partial_sparse = basis.ExplicitBasis(partial_sparse_mxs, ['I', 'X'], sparse=True)
        self.assertTrue(b_partial_sparse.sparse)
        to_std_sparse = b_partial_sparse.to_std_transform_matrix
        self.assertEqual(to_std_sparse.shape, (4, 2))
        from_std_sparse = b_partial_sparse.from_std_transform_matrix
        self.assertEqual(from_std_sparse.shape, (2, 4))
        # from_std_sparse . to_std_sparse is identity
        prod = from_std_sparse.dot(to_std_sparse.tocsc()).toarray()
        self.assertArraysAlmostEqual(prod, np.identity(2, dtype='complex'))

        # Check simple-assuming assertion in Basis base class methods using a custom non-simple Basis subclass
        class DummyNonSimpleBasis(basis.Basis):
            @property
            def dim(self): return 4
            @property
            def size(self): return 4
            @property
            def elshape(self): return (3, 3) # elsize = 9 != dim=4
            def __hash__(self):
                return hash(('dummy', self.dim, self.elshape))

        dummy = DummyNonSimpleBasis('dummy', 'Dummy', True, False)
        self.assertFalse(dummy.is_simple())
        with self.assertRaises(AssertionError):
            dummy.to_elementstd_transform_matrix
        with self.assertRaises(AssertionError):
            dummy.create_equivalent('pp')
        with self.assertRaises(AssertionError):
            dummy.create_simple_equivalent('pp')

        # from_elementstd_transform_matrix sparse raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            pp_sparse.from_elementstd_transform_matrix

        # Complete-basis inversion path (size == dim)
        b_complete = basis.BuiltinBasis('pp', 4) # size=4, dim=4
        self.assertTrue(b_complete.is_complete())
        from_std_comp = b_complete.from_std_transform_matrix
        to_std_comp = b_complete.to_std_transform_matrix
        self.assertArraysAlmostEqual(np.dot(from_std_comp, to_std_comp), np.identity(4, dtype='complex'))

        # DirectSumBasis non-simple vector_elements vs elements shapes and embedding structure
        # (Where vector_elements and elements spaces fundamentally diverge)
        s1 = basis.BuiltinBasis('std', 1)
        s2 = basis.BuiltinBasis('std', 4)
        dsb = basis.DirectSumBasis([s2, s1])
        self.assertFalse(dsb.is_simple())
        self.assertEqual(dsb.dim, 5)
        self.assertEqual(dsb.elsize, 9)
        self.assertEqual(dsb.elements[0].shape, (3, 3))
        self.assertEqual(dsb.vector_elements[0].shape, (5,))
        # Active vector element contains standard unit [1, 0, 0, 0, 0]
        self.assertArraysAlmostEqual(dsb.vector_elements[0], np.array([1, 0, 0, 0, 0], dtype='complex'))
        # Active matrix element embeds [1, 0, 0, 0] as the upper-left 2x2 block
        expected_matrix = np.zeros((3, 3), dtype='complex')
        expected_matrix[0, 0] = 1.0
        self.assertArraysAlmostEqual(dsb.elements[0], expected_matrix)

    def test_basis_copy_and_sparsity_toggling_regression(self):
        b = basis.BuiltinBasis('pp', 4, sparse=False)
        b_copy = b.copy()
        self.assertEqual(b, b_copy)
        self.assertIsNot(b, b_copy)

        # with_sparsity
        self.assertIs(b.with_sparsity(False), b) # no-op
        b_sparse = b.with_sparsity(True)
        self.assertTrue(b_sparse.sparse)
        self.assertIs(b_sparse.with_sparsity(True), b_sparse) # no-op
        b_dense = b_sparse.with_sparsity(False)
        self.assertFalse(b_dense.sparse)

        # Sparsity toggling for other subclasses
        # ExplicitBasis
        mats = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype='complex')
        ex = basis.ExplicitBasis(mats, ['I', 'X'])
        self.assertTrue(ex.with_sparsity(True).sparse)

        # DirectSumBasis
        s1 = basis.BuiltinBasis('std', 1)
        s2 = basis.BuiltinBasis('std', 4)
        dsb = basis.DirectSumBasis([s2, s1])
        dsb_sparse = dsb.with_sparsity(True)
        self.assertTrue(dsb_sparse.sparse)

        # TensorProdBasis
        tpb = basis.TensorProdBasis([s1, s1])
        tpb_sparse = tpb.with_sparsity(True)
        self.assertTrue(tpb_sparse.sparse)

    def test_basis_dimensions_and_math_properties_regression(self):
        # elndim, elsize, is_simple, is_complete, is_partial, first_element_is_identity
        b = basis.BuiltinBasis('pp', 4)
        self.assertEqual(b.elndim, 2)
        self.assertEqual(b.elsize, 4)
        self.assertTrue(b.is_simple())
        self.assertTrue(b.is_complete())
        self.assertFalse(b.is_partial())
        self.assertTrue(b.first_element_is_identity)

        # Not proportional to identity first element
        mats = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype='complex')
        ex = basis.ExplicitBasis(mats, ['X', 'I'])
        self.assertFalse(ex.first_element_is_identity)

        # vector_elements
        vels = b.vector_elements
        self.assertEqual(len(vels), 4)
        self.assertEqual(vels[0].shape, (4,))

        # Sparse vector_elements
        b_sparse = b.with_sparsity(True)
        vels_sparse = b_sparse.vector_elements
        self.assertEqual(len(vels_sparse), 4)
        self.assertEqual(vels_sparse[0].shape, (4, 1))

        # LazyBasis string representation when not computed
        b_lazy = basis.BuiltinBasis('pp', 256)
        # Ensure it starts as not computed (elements are None initially)
        self.assertIsNone(b_lazy._elements)
        s = str(b_lazy)
        self.assertIn("not computed yet", s)
        # Computed string representation of LazyBasis (after labels are built)
        _ = b_lazy.labels
        self.assertNotIn("not computed yet", str(b_lazy))

        # sv is vector-valued (elndim == 1), so raises NotImplementedError
        sv = basis.Basis.cast('sv', 7)
        with self.assertRaises(NotImplementedError):
            sv.is_normalized()

        # Vector-valued first_element_is_identity returns False
        self.assertFalse(sv.first_element_is_identity)

        # cl classical basis behavior
        cl = basis.Basis.cast('cl', 4)
        self.assertEqual(cl.name, 'cl')
        self.assertEqual(cl.dim, 4)
        self.assertFalse(cl.first_element_is_identity)

        # elndim == 3 raises ValueError
        custom_el = np.zeros((2, 2, 2))
        custom_b = basis.ExplicitBasis([custom_el])
        with self.assertRaises(ValueError):
            custom_b.is_normalized()

    def test_basis_lookup_and_indexing_regression(self):
        b = basis.BuiltinBasis('pp', 4)
        # index lookup
        self.assertArraysAlmostEqual(b[0], b.elements[0])
        # label lookup
        self.assertArraysAlmostEqual(b['I'], b.elements[0])
        # len
        self.assertEqual(len(b), 4)

    def test_basis_casting_and_error_handling_regression(self):
        # Empty bases
        empty_none = basis.Basis.cast(None)
        empty_list = basis.Basis.cast([])
        self.assertEqual(empty_none.name, "*Empty*")
        self.assertEqual(empty_list.name, "*Empty*")

        # Incompatible dimensions / state space assertion
        b = basis.BuiltinBasis('pp', 4)
        with self.assertRaises(AssertionError):
            basis.Basis.cast_from_basis(b, dim=16)

        # cast_from_arrays assertion
        mats = np.array([[[1, 0], [0, 1]]], dtype='complex')
        with self.assertRaises(AssertionError):
            basis.Basis.cast_from_arrays(mats, dim=16)

        # Incompatible types raise an error. NOTE: on current `develop`, an unindexable arg like an
        # int raises TypeError from `arg[0]`. A separate bugfix PR hardens this to a descriptive
        # ValueError; this assertion accepts either so the test-only PR stays green on develop.
        with self.assertRaises((ValueError, TypeError)):
            basis.Basis.cast(12345)

        # Cast from a list of (subname, subdim) tuples -> DirectSumBasis
        dsb_tuple = basis.Basis.cast([('std', 4), ('std', 1)])
        self.assertTrue(isinstance(dsb_tuple, basis.DirectSumBasis))
        self.assertEqual(dsb_tuple.dim, 5)

        # StateSpace compatibility check
        ss_comp = b.state_space
        self.assertTrue(b.is_compatible_with_state_space(ss_comp))
        b_16 = basis.BuiltinBasis('pp', 16)
        ss_incomp = b_16.state_space
        self.assertFalse(b.is_compatible_with_state_space(ss_incomp))

    def test_basis_coverage_supplement_regression(self):
        import scipy.sparse as sps
        from pygsti.baseobjs.statespace import QubitSpace

        # --- Basis.cast from name + StateSpace (tensor-product and single-qubit paths) ---
        tpb = basis.Basis.cast('pp', QubitSpace(2))
        self.assertTrue(isinstance(tpb, basis.TensorProdBasis))
        self.assertEqual(tpb.dim, 16)
        single = basis.Basis.cast('pp', QubitSpace(1))
        self.assertTrue(isinstance(single, basis.BuiltinBasis))

        # --- Basis.cast where arg[0] is an ndarray -> cast_from_arrays ---
        mats = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype='complex')
        cast_arr = basis.Basis.cast(mats)
        self.assertTrue(isinstance(cast_arr, basis.ExplicitBasis))

        # --- cast_from_basis: StateSpace dim compatibility + sparsity coercion ---
        pp = basis.BuiltinBasis('pp', 4)
        same = basis.Basis.cast_from_basis(pp, dim=QubitSpace(1))
        self.assertEqual(same, pp)
        coerced = basis.Basis.cast_from_basis(pp, sparse=True)
        self.assertTrue(coerced.sparse)

        # --- cast_from_arrays sparse-flag success branch (assert sparse == b.sparse) ---
        b_sp = basis.Basis.cast_from_arrays(mats, sparse=True)
        self.assertTrue(b_sp.sparse)

        # --- is_equivalent comparing a basis to raw element lists (dense + sparse) ---
        self.assertTrue(pp.is_equivalent(pp.elements))
        pp_sparse = basis.BuiltinBasis('pp', 4, sparse=True)
        self.assertTrue(pp_sparse.is_equivalent(pp_sparse.elements))

        # --- _sparse_equal: shape mismatch, value mismatch, and true equality ---
        a = sps.csr_matrix(np.eye(2))
        self.assertFalse(basis._sparse_equal(a, sps.csr_matrix(np.eye(3))))       # shape mismatch
        self.assertFalse(basis._sparse_equal(a, sps.csr_matrix(np.diag([1, 2])))) # value mismatch
        self.assertTrue(basis._sparse_equal(a, sps.csr_matrix(np.eye(2))))

        # --- create_transform_matrix / reverse_transform_matrix (dense + by-name) ---
        gm = basis.BuiltinBasis('gm', 4)
        T = pp.create_transform_matrix(gm)
        Trev = pp.reverse_transform_matrix(gm)
        self.assertArraysAlmostEqual(np.dot(Trev, T), np.identity(4))
        self.assertEqual(pp.create_transform_matrix('gm').shape, (4, 4))
        self.assertEqual(pp.reverse_transform_matrix('gm').shape, (4, 4))

        # --- is_normalized False for an unnormalized matrix basis ---
        self.assertFalse(basis.Basis.cast('gm_unnormalized', 4).is_normalized())

        # --- LazyBasis ellookup ---
        self.assertIn('I', pp.ellookup)
        # NOTE: accessing elindlookup *after* ellookup currently raises AttributeError on develop
        # (a latent bug); that path and its fix are covered in the separate bugfix PR. The
        # elindlookup happy-path (accessed first) is exercised in supplement2 below.

        # --- ExplicitBasis: label/element count mismatch raises ValueError ---
        with self.assertRaises(ValueError):
            basis.ExplicitBasis(mats, ['only_one_label'])

        # --- ExplicitBasis: non-ndarray (nested list) elements get converted ---
        ex_list = basis.ExplicitBasis([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], ['I', 'X'])
        self.assertEqual(ex_list.elements[0].shape, (2, 2))

        # --- ExplicitBasis: sparse non-CSR element coerced to CSR ---
        ex_sp = basis.ExplicitBasis([sps.lil_matrix(np.eye(2))], ['I'], sparse=True)
        self.assertTrue(ex_sp.sparse)

        # --- ExplicitBasis with explicit (dense) vector_elements ---
        # NOTE: the sparse + explicit vector_elements path currently raises AttributeError on
        # develop (a latent bug); that path and its fix are covered in the separate bugfix PR.
        vec_basis = basis.ExplicitBasis(
            mats, ['I', 'X'], vector_elements=np.array([[1, 0, 0, 0], [0, 1, 1, 0]], 'complex'))
        self.assertEqual(vec_basis.dim, 4)
        self.assertEqual(len(vec_basis.vector_elements), 2)

        # --- DirectSumBasis: element-space transform, equivalents, inequivalence branches ---
        s1 = basis.BuiltinBasis('std', 1)
        s2 = basis.BuiltinBasis('std', 4)
        dsb = basis.DirectSumBasis([s2, s1])
        em = dsb.to_elementstd_transform_matrix
        self.assertEqual(em.shape, (dsb.elsize, dsb.size))
        self.assertTrue(isinstance(dsb.create_equivalent('gm'), basis.DirectSumBasis))
        self.assertTrue(isinstance(dsb.create_simple_equivalent('gm'), basis.BuiltinBasis))
        self.assertTrue(isinstance(dsb.create_simple_equivalent(), basis.BuiltinBasis))
        self.assertFalse(dsb.is_equivalent(pp))                                  # not a DirectSumBasis
        self.assertFalse(dsb.is_equivalent(basis.DirectSumBasis([s2, s1, s2])))  # different length

        # --- TensorProdBasis: equivalents, inequivalence branches, sparse element build ---
        pp1 = basis.BuiltinBasis('pp', 4)
        tp = basis.TensorProdBasis([pp1, pp1])
        self.assertTrue(isinstance(tp.create_simple_equivalent(), basis.BuiltinBasis))
        self.assertTrue(isinstance(tp.create_equivalent('gm'), basis.TensorProdBasis))
        self.assertFalse(tp.is_equivalent(pp))                                   # not a TensorProdBasis
        self.assertFalse(tp.is_equivalent(basis.TensorProdBasis([pp1, pp1, pp1])))  # different length
        pp1_sp = basis.BuiltinBasis('pp', 4, sparse=True)
        tp_sp = basis.TensorProdBasis([pp1_sp, pp1_sp])
        self.assertEqual(len(tp_sp.elements), 16)
        self.assertTrue(sps.issparse(tp_sp.elements[0]))
        self.assertFalse(tp_sp.is_equivalent(tp))                                # differing sparsity

    def test_basis_coverage_supplement2_regression(self):
        import scipy.sparse as sps
        from pygsti.baseobjs.statespace import ExplicitStateSpace

        # --- Abstract base-class methods raise NotImplementedError ---
        abstract = basis.Basis('n', 'l', True, False)
        for accessor in ('dim', 'size', 'elshape'):
            with self.assertRaises(NotImplementedError):
                getattr(abstract, accessor)
        with self.assertRaises(NotImplementedError):
            abstract._copy_with_toggled_sparsity()

        # --- Abstract LazyBasis lazy-builders raise NotImplementedError ---
        lazy = basis.LazyBasis('n', 'l', True, False)
        with self.assertRaises(NotImplementedError):
            lazy._lazy_build_elements()
        with self.assertRaises(NotImplementedError):
            lazy._lazy_build_labels()

        # --- _sparse_equal: differing nonzero *patterns* (index mismatch) returns False ---
        a = sps.csr_matrix(np.array([[1, 0], [0, 0]]))
        b = sps.csr_matrix(np.array([[0, 0], [0, 1]]))
        self.assertFalse(basis._sparse_equal(a, b))

        # --- cast([1,2,3]): first element has no len() and is not ndarray -> error.
        # On current `develop` this raises TypeError from `len(arg[0])`; a separate bugfix PR
        # hardens it to ValueError. Accept either so the test-only PR stays green on develop. ---
        with self.assertRaises((ValueError, TypeError)):
            basis.Basis.cast([1, 2, 3])

        # --- cast from a name + multi-block StateSpace -> DirectSumBasis ---
        multiblock = basis.Basis.cast('pp', ExplicitStateSpace([['Q0'], ['Q1']]))
        self.assertTrue(isinstance(multiblock, basis.DirectSumBasis))

        # --- BuiltinBasis.is_equivalent with a string equal to its name ---
        pp = basis.BuiltinBasis('pp', 4)
        self.assertTrue(pp.is_equivalent('pp'))
        self.assertFalse(pp.is_equivalent('gm'))

        # --- create_simple_equivalent on a simple basis with an explicit name (else branch) ---
        self.assertTrue(isinstance(pp.create_simple_equivalent('gm'), basis.BuiltinBasis))

        # --- Lazy build order: ellookup before elements, elindlookup before labels ---
        fresh1 = basis.BuiltinBasis('pp', 4)
        self.assertIn('I', fresh1.ellookup)        # triggers element+label build via ellookup
        fresh2 = basis.BuiltinBasis('pp', 4)
        self.assertEqual(fresh2.elindlookup['X'], 1)  # triggers label build via elindlookup

        # --- ExplicitBasis is_equivalent: sparseness mismatch and sparse-equal both-Basis paths ---
        mats = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype='complex')
        ex_dense = basis.ExplicitBasis(mats, ['I', 'X'])
        ex_sparse1 = basis.ExplicitBasis([sps.csr_matrix(m) for m in mats], ['I', 'X'], sparse=True)
        ex_sparse2 = basis.ExplicitBasis([sps.csr_matrix(m) for m in mats], ['I', 'X'], sparse=True)
        self.assertFalse(ex_dense.is_equivalent(ex_sparse1))   # sparseness mismatch -> False
        self.assertTrue(ex_sparse1.is_equivalent(ex_sparse2))  # both sparse, element-wise equal

        # --- reverse_transform_matrix on a sparse basis (self.sparse branch) ---
        pp_sparse = basis.BuiltinBasis('pp', 4, sparse=True)
        gm_sparse = basis.BuiltinBasis('gm', 4, sparse=True)
        rt = pp_sparse.reverse_transform_matrix(gm_sparse)
        self.assertEqual(rt.shape, (4, 4))

        # --- DirectSumBasis: labels build, dense + sparse to_std_transform_matrix ---
        s1 = basis.BuiltinBasis('std', 1)
        s2 = basis.BuiltinBasis('std', 4)
        dsb = basis.DirectSumBasis([s2, s1])
        self.assertEqual(len(dsb.labels), dsb.size)
        self.assertEqual(dsb.to_std_transform_matrix.shape, (dsb.dim, dsb.size))
        dsb_sparse = basis.DirectSumBasis([s2.with_sparsity(True), s1.with_sparsity(True)])
        self.assertEqual(dsb_sparse.to_std_transform_matrix.shape, (dsb_sparse.dim, dsb_sparse.size))

        # --- DirectSumBasis / TensorProdBasis components specified as cast-argument tuples ---
        dsb_args = basis.DirectSumBasis([('std', 4), ('std', 1)])
        self.assertEqual(dsb_args.dim, 5)
        tpb_args = basis.TensorProdBasis([('pp', 4), ('pp', 4)])
        self.assertEqual(tpb_args.dim, 16)

        # --- TensorProdBasis dense elements + labels build ---
        tp = basis.TensorProdBasis([basis.BuiltinBasis('pp', 4), basis.BuiltinBasis('pp', 4)])
        self.assertEqual(len(tp.elements), 16)
        self.assertEqual(len(tp.labels), 16)
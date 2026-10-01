import numpy as np

import pygsti.tools.basistools as bt
from pygsti.baseobjs import basis
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis, ExplicitElementaryErrorgenBasis
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel, LocalElementaryErrorgenLabel
from pygsti.baseobjs import BuiltinBasis, QubitSpace
from ..util import BaseCase


class CanonicalErrorgenBasisTester(BaseCase):
    def test_import_paths(self):
        from pygsti.baseobjs import canonical_errorgen_basis, errorgenbasis

        self.assertTrue(hasattr(errorgenbasis, 'canonical_errorgen_basis'))
        self.assertIs(errorgenbasis.canonical_errorgen_basis, canonical_errorgen_basis)
        self.assertEqual(errorgenbasis.canonical_errorgen_basis(QubitSpace(1)).name, 'PP')

    # (state space labels, subsystem dimensions)
    CASES = [(('Q0',), (2,)),
             (('Q0', 'Q1'), (2, 2)),
             (('T0',), (3,)),
             (('Q0', 'T1'), (2, 3)),
             (('T0', 'Q1'), (3, 2)),
             (('D0',), (5,))]

    @staticmethod
    def _state_space(sslbls, dims):
        from pygsti.baseobjs import ExplicitStateSpace
        return ExplicitStateSpace([sslbls], [dims])

    def test_structure(self):
        from pygsti.baseobjs import canonical_errorgen_basis
        from pygsti.baseobjs.errorgenlabel import _bel_tokens
        for sslbls, dims in self.CASES:
            b = canonical_errorgen_basis(self._state_space(sslbls, dims))
            D = int(np.prod(dims))
            els = [np.asarray(el) for el in b.elements]
            self.assertEqual(len(els), D**2)
            self.assertEqual(b.labels[0], 'I' * len(dims))
            self.assertArraysAlmostEqual(els[0], np.eye(D))
            gram = np.array([[np.trace(a.conj().T @ c) for c in els] for a in els])
            self.assertArraysAlmostEqual(gram, D * np.eye(D**2))
            for lbl, el in zip(b.labels[1:], els[1:]):
                self.assertArraysAlmostEqual(el, el.conj().T)
                self.assertAlmostEqual(abs(np.trace(el)), 0)
                self.assertEqual(len(_bel_tokens(lbl)), len(dims))

    def test_factor_bases(self):
        from pygsti.baseobjs import canonical_errorgen_basis
        b = canonical_errorgen_basis(self._state_space(('Q0',), (2,)))
        self.assertIsInstance(b, basis.BuiltinBasis)
        self.assertEqual(b.name, 'PP')
        b = canonical_errorgen_basis(self._state_space(('T0',), (3,)))
        self.assertIsInstance(b, basis.BuiltinBasis)
        self.assertEqual(b.name, 'GM')
        b = canonical_errorgen_basis(self._state_space(('T0', 'Q1'), (3, 2)))
        self.assertIsInstance(b, basis.TensorProdBasis)
        self.assertEqual([c.name for c in b.component_bases], ['GM', 'PP'])

        two_qubits = canonical_errorgen_basis(self._state_space(('Q0', 'Q1'), (2, 2)))
        pp16 = basis.BuiltinBasis('PP', 16)
        self.assertEqual(list(two_qubits.labels), list(pp16.labels))
        self.assertArraysAlmostEqual(np.array(two_qubits.elements), np.array(pp16.elements))

    def test_sparse(self):
        from pygsti.baseobjs import canonical_errorgen_basis
        b = canonical_errorgen_basis(self._state_space(('Q0', 'T1'), (2, 3)), sparse=True)
        self.assertTrue(b.sparse)
        dense = canonical_errorgen_basis(self._state_space(('Q0', 'T1'), (2, 3)))
        self.assertArraysAlmostEqual(np.array([el.toarray() for el in b.elements]), np.array(dense.elements))

    def test_rejections(self):
        from pygsti.baseobjs import canonical_errorgen_basis, ExplicitStateSpace, QubitSpace
        with self.assertRaises(TypeError):
            canonical_errorgen_basis(3)
        with self.assertRaises(ValueError):  # direct sum
            canonical_errorgen_basis(ExplicitStateSpace([('Q0',), ('L0',)], [(2,), (1,)]))
        with self.assertRaises(ValueError):  # classical factor
            canonical_errorgen_basis(ExplicitStateSpace([('Q0', 'C1')], [(2, 2)], [('Q', 'C')]))
        with self.assertRaises(ValueError):  # udim == 1
            canonical_errorgen_basis(ExplicitStateSpace([('Q0', 'L1')], [(2, 1)]))
        canonical_errorgen_basis(QubitSpace(3))  # a QubitSpace is accepted

    def test_lindblad_construction(self):
        # Compare H and S generators built from the helper's basis against the defining formulas,
        # written in pyGSTi's row-stacked standard basis.
        from pygsti.baseobjs import canonical_errorgen_basis
        from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel
        from pygsti.modelmembers.operations import LindbladErrorgen
        for sslbls, dims, lbl in [(('T0',), (3,), 'Y_{0,2}'), (('Q0', 'T1'), (2, 3), 'XZ_{2}')]:
            ss = self._state_space(sslbls, dims)
            b = canonical_errorgen_basis(ss)
            D = int(np.prod(dims))
            F = np.asarray(b.ellookup[lbl])
            formulas = {'H': lambda r: -1j * (F @ r - r @ F),
                        'S': lambda r: F @ r @ F.conj().T - 0.5 * (F.conj().T @ F @ r + r @ F.conj().T @ F)}
            for typ, f in formulas.items():
                eg = LindbladErrorgen.from_elementary_errorgens({LocalElementaryErrorgenLabel(typ, (lbl,)): 0.1},
                                                               elementary_errorgen_basis=b, mx_basis=b,
                                                               state_space=ss, parameterization='GLND')
                G = bt.change_basis(eg.to_dense(), b, basis.Basis.cast('std', D**2))
                expected = np.zeros((D**2, D**2), complex)
                for k in range(D**2):
                    E = np.zeros(D**2, complex); E[k] = 1
                    expected[:, k] = 0.1 * f(E.reshape(D, D)).reshape(-1)
                self.assertArraysAlmostEqual(G, expected)


class CompleteElementaryErrorgenBasisTester(BaseCase):
    
    def setUp(self):
        self.basis_1q = BuiltinBasis('PP', 4)
        self.state_space_1Q = QubitSpace(1)
        self.state_space_2Q = QubitSpace(2)
        
        #create a complete basis with default settings for reuse.
        self.complete_errorgen_basis_default_1Q = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q)
        self.complete_errorgen_basis_default_2Q = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_2Q)

    def test_default_construction(self):
        assert len(self.complete_errorgen_basis_default_1Q.labels) == 12
        #may as well also test the __len__ method while we're here.
        assert len(self.complete_errorgen_basis_default_1Q) == 12
    
    def test_sector_restrictions(self):
        errorgen_basis_H = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('H',))
        errorgen_basis_S = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('S',))
        errorgen_basis_C = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('C',))
        errorgen_basis_A = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('A',))
        
        for lbl in errorgen_basis_H.labels:
            assert lbl.errorgen_type == 'H'
        for lbl in errorgen_basis_S.labels:
            assert lbl.errorgen_type == 'S'
        for lbl in errorgen_basis_C.labels:
            assert lbl.errorgen_type == 'C'
        for lbl in errorgen_basis_A.labels:
            assert lbl.errorgen_type == 'A'

        assert len(errorgen_basis_H.labels) == 3
        assert len(errorgen_basis_S.labels) == 3
        assert len(errorgen_basis_C.labels) == 3
        assert len(errorgen_basis_A.labels) == 3

        #confirm multiple sectors work right too.
        errorgen_basis_HSC = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('H','S','C'))
        for lbl in errorgen_basis_HSC.labels:
            assert lbl.errorgen_type in ('H', 'S', 'C')
        assert len(errorgen_basis_HSC.labels) == 9

    def test_max_weights(self):
        errorgen_basis = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_2Q, 
                                                         max_weights = {'H':2, 'S':2, 'C':1, 'A':1})
        
        for lbl in errorgen_basis.labels:
            if lbl.errorgen_type in ('H', 'S'):
                assert len(lbl.sslbls) in (1,2)
            else:
                assert len(lbl.sslbls)==1

    def test_to_explicit_basis(self):
        explicit_errorgen_basis = self.complete_errorgen_basis_default_1Q.to_explicit_basis()

        assert self.complete_errorgen_basis_default_1Q.labels == explicit_errorgen_basis.labels

    def test_global_local_labels(self):
        global_labels = self.complete_errorgen_basis_default_1Q.global_labels()
        local_labels = self.complete_errorgen_basis_default_1Q.local_labels()

        assert isinstance(global_labels[0], GlobalElementaryErrorgenLabel)
        assert isinstance(local_labels[0], LocalElementaryErrorgenLabel)
        
    def test_sublabels(self):
        H_labels = self.complete_errorgen_basis_default_1Q.sublabels('H')
        S_labels = self.complete_errorgen_basis_default_1Q.sublabels('S')
        C_labels = self.complete_errorgen_basis_default_1Q.sublabels('C')
        A_labels = self.complete_errorgen_basis_default_1Q.sublabels('A')

        for lbl in H_labels:
            assert lbl.errorgen_type == 'H'
        for lbl in S_labels:
            assert lbl.errorgen_type == 'S'
        for lbl in C_labels:
            assert lbl.errorgen_type == 'C'
        for lbl in A_labels:
            assert lbl.errorgen_type == 'A'
    
    def test_elemgen_supports(self):
        errorgen_basis = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_2Q)

        #there should be 24 weight 1 and 216 weight 2 terms.
        elemgen_supports = errorgen_basis.elemgen_supports
        num_weight_1 = 0
        num_weight_2 = 0
        for support in elemgen_supports:
            if len(support) == 1:
                num_weight_1+=1
            elif len(support) == 2:
                num_weight_2+=1
            else:
                raise ValueError('Invalid support length for two-qubit error gen basis.')

        assert num_weight_1==24 and num_weight_2==216

    def test_elemgen_and_dual_construction(self):
        #just test for running w/o failure.
        elemgens = self.complete_errorgen_basis_default_1Q.elemgen_matrices
        duals = self.complete_errorgen_basis_default_1Q.elemgen_dual_matrices

    def test_label_index(self):
        
        #1 qubit tests
        labels = self.complete_errorgen_basis_default_1Q.labels
        test_eg = GlobalElementaryErrorgenLabel('A', ['X', 'Y'], (0,))
        lbl_idx = self.complete_errorgen_basis_default_1Q.label_index(test_eg)
        assert lbl_idx ==  labels.index(test_eg)

        #Test missing label
        test_eg_missing = GlobalElementaryErrorgenLabel('C', ['X', 'Y'], (1,))

        with self.assertRaises(KeyError):
            self.complete_errorgen_basis_default_1Q.label_index(test_eg_missing)
        assert self.complete_errorgen_basis_default_1Q.label_index(test_eg_missing, ok_if_missing=True) is None

        #Test embedding 
        test_eg = GlobalElementaryErrorgenLabel('C', ['X', 'Y'], (0,))
        test_eg_local = LocalElementaryErrorgenLabel('C', ['XI', 'YI'])
        
        lbl_idx = self.complete_errorgen_basis_default_1Q.label_index(test_eg)
        lbl_idx_1 = self.complete_errorgen_basis_default_1Q.label_index(test_eg_local)
        assert lbl_idx == lbl_idx_1
        assert lbl_idx ==  labels.index(test_eg)

        # 2 qubit tests
        labels = self.complete_errorgen_basis_default_2Q.labels

        test_eg = GlobalElementaryErrorgenLabel('A', ['X', 'Y'], (0,))
        lbl_idx = self.complete_errorgen_basis_default_2Q.label_index(test_eg)
        assert lbl_idx ==  labels.index(test_eg)

        #Test missing label
        test_eg_missing = GlobalElementaryErrorgenLabel('C', ['X', 'Y'], (2,))

        with self.assertRaises(KeyError):
            self.complete_errorgen_basis_default_2Q.label_index(test_eg_missing)
        assert self.complete_errorgen_basis_default_2Q.label_index(test_eg_missing, ok_if_missing=True) is None

        #Test embedding for 2 qubit labels
        test_eg = GlobalElementaryErrorgenLabel('C', ['X', 'Y'], (1,))
        test_eg_local = LocalElementaryErrorgenLabel('C', ['IX', 'IY'])

        lbl_idx = self.complete_errorgen_basis_default_2Q.label_index(test_eg)
        lbl_idx_1 = self.complete_errorgen_basis_default_2Q.label_index(test_eg_local)
        assert lbl_idx == lbl_idx_1
        assert lbl_idx ==  labels.index(test_eg)

    def test_create_subbasis(self):
        errorgen_basis = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_2Q)
        subbasis = errorgen_basis.create_subbasis(sslbl_overlap=(0,))

        #should have 12 weight-1 terms on zero and 216 weight 2, for 228 total in this subbasis.
        assert len(subbasis) == 228

    def test_union(self):
        errorgen_basis_H = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('H',))
        errorgen_basis_S = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('S',))

        union_basis = errorgen_basis_H.union(errorgen_basis_S)
        #should now have 6 items.
        assert len(union_basis) == 6
        for lbl in union_basis.labels:
            assert lbl.errorgen_type in ('H', 'S')

    def test_intersection(self):
        errorgen_basis_HSC = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('H','S','C'))
        errorgen_basis_H = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('H',))
        
        intersection_basis = errorgen_basis_HSC.intersection(errorgen_basis_H)
        #should now have 3 items
        assert len(intersection_basis) == 3
        for lbl in intersection_basis.labels:
            assert lbl.errorgen_type == 'H'

    def test_difference(self):
        errorgen_basis_HSC = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('H','S','C'))
        errorgen_basis_H = CompleteElementaryErrorgenBasis(self.basis_1q, self.state_space_1Q, elementary_errorgen_types=('H',))
        
        intersection_basis = errorgen_basis_HSC.difference(errorgen_basis_H)
        #should now have 6 items
        assert len(intersection_basis) == 6
        for lbl in intersection_basis.labels:
            assert lbl.errorgen_type in ('S', 'C')

class ExplicitElementaryErrorgenBasisTester(BaseCase):

    def setUp(self):
        self.basis_1q = BuiltinBasis('PP', 4)
        self.state_space_1Q = QubitSpace(1)
        self.state_space_2Q = QubitSpace(2)

        self.labels_1Q = [LocalElementaryErrorgenLabel('H', ['X']),
                          LocalElementaryErrorgenLabel('S', ['Y']),
                          LocalElementaryErrorgenLabel('C', ['X','Y']),
                          LocalElementaryErrorgenLabel('A', ['X','Y'])]
        self.labels_2Q = [LocalElementaryErrorgenLabel('H', ['XI']),
                          LocalElementaryErrorgenLabel('S', ['YY']),
                          LocalElementaryErrorgenLabel('C', ['XX','YY']),
                          LocalElementaryErrorgenLabel('A', ['XX','YY'])]
        self.labels_2Q_alt = [LocalElementaryErrorgenLabel('H', ['IX']),
                          LocalElementaryErrorgenLabel('S', ['ZZ']),
                          LocalElementaryErrorgenLabel('C', ['XX','YY']),
                          LocalElementaryErrorgenLabel('A', ['XX','YY'])]
        

        self.explicit_basis_1Q = ExplicitElementaryErrorgenBasis(self.state_space_1Q, self.labels_1Q, self.basis_1q)
        self.explicit_basis_2Q = ExplicitElementaryErrorgenBasis(self.state_space_2Q, self.labels_2Q, self.basis_1q)
        self.explicit_basis_2Q_alt = ExplicitElementaryErrorgenBasis(self.state_space_2Q, self.labels_2Q_alt, self.basis_1q)
        
        

    def test_elemgen_supports(self):
        #there should be 1 weight 1 and 3 weight 2 terms.
        elemgen_supports = self.explicit_basis_2Q.elemgen_supports
        num_weight_1 = 0
        num_weight_2 = 0
        for support in elemgen_supports:
            if len(support) == 1:
                num_weight_1+=1
            elif len(support) == 2:
                num_weight_2+=1
            else:
                raise ValueError('Invalid support length for two-qubit error gen basis.')

        assert num_weight_1==1 and num_weight_2==3

    def test_elemgen_and_dual_construction(self):
        #just test for running w/o failure.
        elemgens = self.explicit_basis_2Q.elemgen_matrices
        duals = self.explicit_basis_2Q.elemgen_dual_matrices

    def test_label_index(self):
        labels = self.explicit_basis_1Q.labels

        test_eg = LocalElementaryErrorgenLabel('C', ['X', 'Y'])
        test_eg_missing = LocalElementaryErrorgenLabel('C', ['X', 'Z'])

        lbl_idx = self.explicit_basis_1Q.label_index(test_eg)

        assert lbl_idx ==  labels.index(test_eg)

        with self.assertRaises(KeyError):
            self.explicit_basis_1Q.label_index(test_eg_missing)
        assert self.explicit_basis_1Q.label_index(test_eg_missing, ok_if_missing=True) is None

    def test_create_subbasis(self):
        subbasis = self.explicit_basis_2Q.create_subbasis(sslbl_overlap=(1,))

        #should have 3 elements remaining in the subbasis.
        assert len(subbasis) == 3

    def test_union(self):
        union_basis = self.explicit_basis_2Q.union(self.explicit_basis_2Q_alt)
        correct_union_labels = [LocalElementaryErrorgenLabel('H', ['XI']),
                          LocalElementaryErrorgenLabel('S', ['YY']),
                          LocalElementaryErrorgenLabel('H', ['IX']),
                          LocalElementaryErrorgenLabel('S', ['ZZ']),
                          LocalElementaryErrorgenLabel('C', ['XX','YY']),
                          LocalElementaryErrorgenLabel('A', ['XX','YY'])]
        #should now have 6 items.
        assert len(union_basis) == 6
        for lbl in union_basis.labels:
            assert lbl in correct_union_labels

    def test_intersection(self):
        intersection_basis = self.explicit_basis_2Q.intersection(self.explicit_basis_2Q_alt)
        correct_intersection_labels = [LocalElementaryErrorgenLabel('C', ['XX','YY']),
                                       LocalElementaryErrorgenLabel('A', ['XX','YY'])]
        #should now have 2 items.
        assert len(intersection_basis) == 2
        for lbl in intersection_basis.labels:
            assert lbl in correct_intersection_labels

    def test_difference(self):
        difference_basis = self.explicit_basis_2Q.difference(self.explicit_basis_2Q_alt)
        correct_difference_labels = [LocalElementaryErrorgenLabel('H', ['XI']),
                                     LocalElementaryErrorgenLabel('S', ['YY'])]
        #should now have 2 items.
        assert len(difference_basis) == 2
        for lbl in difference_basis.labels:
            assert lbl in correct_difference_labels



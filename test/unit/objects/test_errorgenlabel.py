from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL, GlobalElementaryErrorgenLabel as GEEL
from ..util import BaseCase

class LocalElementaryErrorgenLabelTester(BaseCase):

    def test_cast(self):
        #from local
        leel_to_cast = LEEL('H', ['X'])
        leel_cast = LEEL.cast(leel_to_cast)
        assert leel_cast is leel_to_cast

        #from global
        geel_to_cast = GEEL('H', ['X'], (0,))
        leel_cast = LEEL.cast(geel_to_cast, sslbls=(0,1))
        assert leel_cast.basis_element_labels == ('XI',)

        #from string
        string_to_cast = 'H(XX)'
        leel_cast = LEEL.cast(string_to_cast)
        assert leel_cast.errorgen_type == 'H'
        assert leel_cast.basis_element_labels == ('XX',)
        
        #from tuple
        #global style tuple
        global_tup_to_cast = ('H', ('X',), (1,))
        leel_cast = LEEL.cast(global_tup_to_cast, sslbls=(0,1))
        assert leel_cast.errorgen_type == 'H'
        assert leel_cast.basis_element_labels == ('IX',)

        local_tup_to_cast = ('H', 'IX')
        leel_cast = LEEL.cast(local_tup_to_cast) 
        assert leel_cast.errorgen_type == 'H'
        assert leel_cast.basis_element_labels == ('IX',)
        
        #different identity label
        geel_to_cast = GEEL('H', ['X'], (0,))
        leel_cast = LEEL.cast(geel_to_cast, sslbls=(0,1), identity_label='F')
        assert leel_cast.basis_element_labels == ('XF',)

    def test_eq(self):
        assert LEEL('H', ('XX',)) == LEEL('H', ('XX',))
        assert LEEL('H', ('XX',)) != LEEL('S', ('XX',))
        assert LEEL('H', ('XX',)) != LEEL('H', ('XY',))
        
    def test_support_indices(self):
        assert LEEL('H', ('XX',)).support_indices() == (0,1)
        assert LEEL('C', ['IX', 'XI']).support_indices() == (0,1)
        assert LEEL('C', ['IXI', 'XII']).support_indices() == (0,1)
        #nonstandard identity label
        assert LEEL('C', ['FXF', 'XFF']).support_indices(identity_label='F') == (0,1)
        
class GlobalElementaryErrorgenLabelTester(BaseCase):

    def test_cast(self):
        #from global
        geel_to_cast = GEEL('H', ['X'], (0,))
        geel_cast = GEEL.cast(geel_to_cast)
        assert geel_cast is geel_to_cast

        #from local
        leel_to_cast = LEEL('H', ['XI'])
        geel_cast = GEEL.cast(leel_to_cast, sslbls=(0,1))
        assert geel_cast.basis_element_labels == ('X',)
        assert geel_cast.sslbls == (0,)

        #from string
        string_to_cast = 'H(XX:0,1)'
        geel_cast = GEEL.cast(string_to_cast)
        assert geel_cast.errorgen_type == 'H'
        assert geel_cast.basis_element_labels == ('XX',)
        assert geel_cast.sslbls == (0,1)

        string_to_cast = 'SXX:0,1'
        geel_cast = GEEL.cast(string_to_cast)
        assert geel_cast.errorgen_type == 'S'
        assert geel_cast.basis_element_labels == ('XX',)
        assert geel_cast.sslbls == (0,1)

        string_to_cast = 'SXX'
        geel_cast = GEEL.cast(string_to_cast, sslbls=(0,1))
        assert geel_cast.errorgen_type == 'S'
        assert geel_cast.basis_element_labels == ('XX',)
        assert geel_cast.sslbls == (0,1)

        #from tuple
        #global style tuple
        global_tup_to_cast = ('H', ('X',), (1,))
        geel_cast = GEEL.cast(global_tup_to_cast, sslbls=(0,1))
        assert geel_cast.errorgen_type == 'H'
        assert geel_cast.basis_element_labels == ('X',)
        assert geel_cast.sslbls == (1,)
        
        local_tup_to_cast = ('H', 'IX')
        geel_cast = GEEL.cast(local_tup_to_cast, sslbls=(0,1)) 
        assert geel_cast.errorgen_type == 'H'
        assert geel_cast.basis_element_labels == ('X',)
        assert geel_cast.sslbls == (1,)

    def test_eq(self):
        assert GEEL('H', ('X',), (0,)) == GEEL('H', ('X',), (0,)) 
        assert GEEL('H', ('X',), (0,)) != GEEL('H', ('X',), (1,))
        assert GEEL('H', ('X',), (0,)) != GEEL('H', ('Y',), (0,))
   
    def test_padded_basis_element_labels(self):
        assert GEEL('H', ('X',), (0,)).padded_basis_element_labels(all_sslbls=(0,1,2)) == ('XII',)
        assert GEEL('C', ('XX','YY'), (1,2)).padded_basis_element_labels(all_sslbls=(0,1,2)) == ('IXX','IYY')
    
    def test_map_state_space_labels(self):
        geel_to_test = GEEL('C', ['XX', 'YY'], (0,1))
        #dictionary mapper
        mapper = {0:'Q0', 1:'Q1'}
        mapped_geel = geel_to_test.map_state_space_labels(mapper)
        assert mapped_geel.sslbls == ('Q0', 'Q1')

        #function mapper
        mapper = lambda x:x+10
        mapped_geel = geel_to_test.map_state_space_labels(mapper)
        assert mapped_geel.sslbls == (10, 11)

    def test_sort_sslbls(self):
        geel_to_test = GEEL('C', ['XI', 'IX'], (1,0))
        sorted_sslbl_geel = geel_to_test.sort_sslbls()

        assert sorted_sslbl_geel.sslbls == (0,1)
        assert sorted_sslbl_geel.basis_element_labels[0] == 'IX' and sorted_sslbl_geel.basis_element_labels[1] == 'XI'

    

class MultiCharacterBasisElementLabelTester(BaseCase):
    """Labels whose per-subsystem tokens are Gell-Mann labels such as 'X_{0,1}'."""

    # (state space labels, subsystem dimensions)
    CASES = [(('T0',), (3,)),
             (('Q0', 'T1'), (2, 3)),
             (('T0', 'Q1'), (3, 2)),
             (('T0', 'T1'), (3, 3)),
             (('D0',), (5,))]

    @staticmethod
    def _tokenized_labels(dims):
        """ All joined GM tensor-product labels, paired with their per-subsystem tokens (built independently). """
        from itertools import product
        from pygsti.baseobjs import Basis
        factor_labels = [Basis.cast('GM', d**2).labels for d in dims]
        return [(''.join(toks), toks) for toks in product(*factor_labels)]

    def test_labels_match_tensor_product_basis(self):
        from pygsti.baseobjs import Basis, ExplicitStateSpace
        for sslbls, dims in self.CASES:
            tpb = Basis.cast('GM', ExplicitStateSpace([sslbls], [dims]))
            self.assertEqual([lbl for lbl, _ in self._tokenized_labels(dims)], list(tpb.labels))

    def test_local_global_round_trip(self):
        for sslbls, dims in self.CASES:
            for lbl, toks in self._tokenized_labels(dims)[1:]:
                support = tuple(i for i, t in enumerate(toks) if t != 'I')
                leel = LEEL('H', (lbl,))
                self.assertEqual(leel.support_indices(), support)

                geel = GEEL.cast(leel, sslbls=sslbls)
                sorted_support = sorted(support, key=lambda i: sslbls[i])  # global labels sort their sslbls
                self.assertEqual(geel.sslbls, tuple(sslbls[i] for i in sorted_support))
                self.assertEqual(geel.basis_element_labels, (''.join(toks[i] for i in sorted_support),))
                self.assertEqual(geel.padded_basis_element_labels(sslbls), (lbl,))
                self.assertEqual(LEEL.cast(geel, sslbls=sslbls), leel)

    def test_two_label_support(self):
        leel = LEEL('C', ('X_{0,1}I', 'Z_{1}Z'))
        self.assertEqual(leel.support_indices(), (0, 1))

        geel = GEEL.cast(leel, sslbls=('T0', 'Q1'))
        self.assertEqual(geel.sslbls, ('Q1', 'T0'))  # sorted, with tokens permuted to match
        self.assertEqual(geel.basis_element_labels, ('IX_{0,1}', 'ZZ_{1}'))
        self.assertEqual(LEEL.cast(geel, sslbls=('T0', 'Q1')), leel)

    def test_sorting_permutes_tokens(self):
        geel = GEEL('A', ('X_{0,2}Y', 'IZ'), ('T1', 'Q0'), sort=False)
        sorted_geel = geel.sort_sslbls()
        self.assertEqual(sorted_geel.sslbls, ('Q0', 'T1'))
        self.assertEqual(sorted_geel.basis_element_labels, ('YX_{0,2}', 'ZI'))
        self.assertEqual(GEEL('A', ('X_{0,2}Y', 'IZ'), ('T1', 'Q0')), sorted_geel)

    def test_global_style_tuple(self):
        leel = LEEL.cast(('S', ('X_{1,2}',), ('T1',)), sslbls=('Q0', 'T1'))
        self.assertEqual(leel.basis_element_labels, ('IX_{1,2}',))

    def test_map_state_space_labels(self):
        geel = GEEL('H', ('XZ_{2}',), ('Q0', 'T1'))
        mapped = geel.map_state_space_labels({'Q0': 'Q9', 'T1': 'T3'})
        self.assertEqual(mapped.sslbls, ('Q9', 'T3'))
        self.assertEqual(mapped.basis_element_labels, ('XZ_{2}',))
        self.assertEqual(mapped.padded_basis_element_labels(('T3', 'Q9')), ('Z_{2}X',))

    def test_text_round_trip(self):
        for leel in [LEEL('H', ('X_{0,1}',)), LEEL('C', ('IX_{0,1}', 'ZZ_{1}')),
                     LEEL('A', ('X_{0,4}', 'Y_{2,3}'))]:
            self.assertEqual(LEEL.cast(str(leel)), leel)
        for geel in [GEEL('S', ('X_{0,1}',), ('T1',)), GEEL('C', ('XZ_{2}', 'YY_{0,1}'), ('Q0', 'T1')),
                     GEEL('A', ('X_{0,1}X_{0,2}', 'Z_{1}I'), (0, 1))]:
            self.assertEqual(GEEL.cast(str(geel)), geel)

    def test_pickle(self):
        import pickle
        leel = LEEL('C', ('IX_{0,1}', 'ZZ_{1}'))
        geel = GEEL.cast(leel, sslbls=('Q0', 'T1'))
        self.assertEqual(pickle.loads(pickle.dumps(leel)), leel)
        self.assertEqual(pickle.loads(pickle.dumps(geel)), geel)
        self.assertEqual(hash(pickle.loads(pickle.dumps(geel))), hash(geel))

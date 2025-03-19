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

    
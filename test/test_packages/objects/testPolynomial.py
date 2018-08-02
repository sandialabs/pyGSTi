from __future__ import print_function
import unittest
import pygsti
import numpy as np

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some gatesets that get used in this file and in testGateSets2.py
class PolynomialTestCase(BaseTestCase):

    def setUp(self):
        super(PolynomialTestCase, self).setUp()

    def test_compact_polys(self):
        
        #Test compact deriv
        p = pygsti.obj.polynomial.Polynomial({(): 1.0, (1,2): 2.0, (1,1,2): 3.0}) 
        v,c = p.compact()
        # 3x1^2 x2 + 2 x1x2 + 3
        
        q = pygsti.obj.polynomial.Polynomial({(): 4.0, (1,1): 5.0, (2,2,3): 6.0})
        v2,c2 = q.compact()
        # 6x2^2 x3 + 5 x1^2 + 4
        
        v = np.concatenate( (v,v2) )
        c = np.concatenate( (c,c2) )
        c = np.ascontiguousarray(c,complex)
        
        #print(v)
        #print(c,"\n")
        vout, cout = pygsti.obj.polynomial.compact_deriv(v,c, (1,2,3)) 
        # p deriv wrt x1 = 6x1x2 + 2x2
        #         wrt x2 = 3x1^2 + 2x1
        #         wrt x3 = 0
        # q deriv wrt x1 = 10 x1
        #         wrt x2 = 12 x2x3
        #         wrt x3 = 6 x2^2
        #print(vout)
        #print(cout)
        compact_polys = pygsti.obj.polynomial.bulk_load_compact_polys(vout,cout, keep_compact=True)
        print("Compact: \n","\n ".join(map(str,compact_polys)))

        def assertCompactPolysEqual(vctups1,vctups2):
            for (v1,c1),(v2,c2) in zip(vctups1,vctups2):
                self.assertArraysAlmostEqual(v1,v2) #integer arrays
                self.assertArraysAlmostEqual(c1,c2) # complex arrays
                
        assertCompactPolysEqual(compact_polys,
                         ((np.array([2, 2, 1, 2, 1, 2]), np.array([6.+0.j, 2.+0.j])),
                          (np.array([2, 2, 1, 1, 1, 1]), np.array([3.+0.j, 2.+0.j])),
                          (np.array([0]), np.array([], dtype=np.complex128)),
                          (np.array([1, 1, 1]), np.array([10.+0.j])),
                          (np.array([1, 2, 2, 3]), np.array([12.+0.j])),
                          (np.array([1, 2, 2, 2]), np.array([6.+0.j]))) )
        
        polys = pygsti.obj.polynomial.bulk_load_compact_polys(vout,cout)
        print("Polynomials: \n","\n ".join(map(str,polys)))
        self.assertEqual(str(polys[0]), "6.000x1x2 + 2.000x2")
        self.assertEqual(str(polys[1]), "2.000x1 + 3.000x1^2")
        self.assertEqual(str(polys[2]), "0")
        self.assertEqual(str(polys[3]), "10.000x1")
        self.assertEqual(str(polys[4]), "12.000x2x3")
        self.assertEqual(str(polys[5]), "6.000x2^2")
        
        self.assertEqual( list(vout), [2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 2, 3, 1, 2, 2, 2])
        self.assertEqual( list(cout), [ 6.+0.j,  2.+0.j,  3.+0.j,  2.+0.j, 10.+0.j, 12.+0.j,  6.+0.j] )
        print("OK!")
        # => [2 2 1 2 1 2]

        try:
            from pygsti.objects import fastgatecalc
            vout2, cout2 = fastgatecalc.fast_compact_deriv(v,c,np.array((1,2,3),int))
            self.assertEqual( list(vout2), [2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 2, 3, 1, 2, 2, 2])
            self.assertEqual( list(cout2), [ 6.+0.j,  2.+0.j,  3.+0.j,  2.+0.j, 10.+0.j, 12.+0.j,  6.+0.j] )
        except ImportError:
            pass # ok if fastgatecalc doesn't exist...



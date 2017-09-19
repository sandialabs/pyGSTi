import unittest
import pygsti
import numpy as np
import pickle

from numpy.random import random,seed

from  pygsti.objects import SPAMVec
import pygsti.construction as pc

from ..testutils import BaseTestCase, compare_files, temp_files

class SPAMVecTestCase(BaseTestCase):

    def setUp(self):
        super(SPAMVecTestCase, self).setUp()
        self.spamvec = SPAMVec(np.array([1,0]))

    def test_slice(self):
        self.spamvec[:]

    def test_bad_vec(self):
        bad_vecs = [
            'akdjsfaksdf',
            [[], [1, 2]],
            [[[]], [[1, 2]]]
        ]
        for bad_vec in bad_vecs:
            with self.assertRaises(ValueError):
                SPAMVec.convert_to_vector(bad_vec)
        
    def test_methods(self):
        pass #TODO: see testGate.py

    def test_cptp_spamvec(self):

        vec = pygsti.obj.CPTPParameterizedSPAMVec([1/np.sqrt(2),0,0,1/np.sqrt(2) - 0.1], "pp")
        print(vec)
        print(vec.base.shape)


        v = vec.to_vector()
        vec.from_vector(v)
        print(v)
        print(vec)

        vec_std = pygsti.change_basis(vec,"pp","std")
        print(vec_std)

        def analyze(spamvec):
            stdmx = pygsti.vec_to_stdmx(spamvec, "pp")
            evals = np.linalg.eigvals(stdmx)
            #print(evals)
            assert( np.all(evals > -1e-10) )
            assert( np.linalg.norm(np.imag(evals)) < 1e-8)
            return np.real(evals)

        print( analyze(vec) )

        pygsti.objects.spamvec.check_deriv_wrt_params(vec)


        seed(1234)

        #Nice cases - when parameters are small
        nRandom = 1000
        for randvec in random((nRandom,4)):
            r = 2*(randvec-0.5)
            vec.from_vector(r)
            evs = analyze(vec)
            pygsti.objects.spamvec.check_deriv_wrt_params(vec)
            #print(r, "->", evs)
        print("OK1")

        #Mean cases - when parameters are large
        nRandom = 1000
        for randvec in random((nRandom,4)):
            r = 10*(randvec-0.5)
            vec.from_vector(r)
            evs = analyze(vec)
            pygsti.objects.spamvec.check_deriv_wrt_params(vec)
            #print(r, "->", evs)
        print("OK2")



if __name__ == '__main__':
    unittest.main(verbosity=2)


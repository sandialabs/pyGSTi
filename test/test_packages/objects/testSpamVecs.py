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

    #TODO
    def test_complement_spamvec(self):
        gateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])

        E0 = gateset.povms['Mdefault']['0']
        E1 = gateset.povms['Mdefault']['1']
        Ec = pygsti.obj.ComplementSPAMVec(
            pygsti.construction.build_identity_vec([2],"pp"),
            [E0])
        print(Ec.gpindices)

        #Test TPPOVM which uses a complement evec
        gateset.povms['Mtest'] = pygsti.obj.TPPOVM( [('+',E0),('-',E1)] )
        E0 = gateset.povms['Mtest']['+']
        Ec = gateset.povms['Mtest']['-']
        
        v = gateset.to_vector()
        gateset.from_vector(v)

        #print(Ec.num_params()) #not implemented for complement vecs - only for POVM
        identity = np.array([[np.sqrt(2)], [0], [0], [0]],'d')
        print("TEST1")
        print(E0)
        print(Ec)
        print(E0 + Ec)
        self.assertArraysAlmostEqual(E0+Ec, identity)

        #TODO: add back if/when we can set parts of a POVM directly...
        #print("TEST2")
        #gateset.effects['E0'] = [1/np.sqrt(2), 0, 0.4, 0.6]
        #print(gateset.effects['E0'])
        #print(gateset.effects['E1'])
        #print(gateset.effects['E0'] + gateset.effects['E1'])
        #self.assertArraysAlmostEqual(gateset.effects['E0'] + gateset.effects['E1'], identity)
        #
        #print("TEST3")
        #gateset.effects['E0'][0,0] = 1.0 #uses dirty processing
        #gateset._update_paramvec(gateset.effects['E0'])
        #print(gateset.effects['E0'])
        #print(gateset.effects['E1'])
        #print(gateset.effects['E0'] + gateset.effects['E1'])
        #self.assertArraysAlmostEqual(gateset.effects['E0'] + gateset.effects['E1'], identity)

        

if __name__ == '__main__':
    unittest.main(verbosity=2)


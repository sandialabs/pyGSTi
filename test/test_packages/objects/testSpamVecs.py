import unittest
import pygsti
import numpy as np
import pickle
import itertools

from numpy.random import random,seed

from pygsti.construction import std1Q_XYI
from  pygsti.objects import SPAMVec, DenseSPAMVec
import pygsti.construction as pc


from ..testutils import BaseTestCase, compare_files, temp_files

class SPAMVecTestCase(BaseTestCase):

    def setUp(self):
        super(SPAMVecTestCase, self).setUp()
        self.spamvec = DenseSPAMVec(np.array([1,0]),"densitymx")

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

    def test_base_spamvec(self):
        raw = pygsti.objects.SPAMVec(4,"densitymx")

        T = pygsti.objects.FullGaugeGroupElement(
                np.array( [ [0,1,0,0],
                            [1,0,0,0],
                            [0,0,1,0],
                            [0,0,0,1] ], 'd') )


        with self.assertRaises(NotImplementedError):
            raw.todense()
        with self.assertRaises(NotImplementedError):
            raw.transform(T,"prep")
        with self.assertRaises(NotImplementedError):
            raw.depolarize(0.01)
        with self.assertRaises(ValueError):
            pygsti.objects.SPAMVec.convert_to_vector(0.0) # something with no len()
        
    def test_methods(self):
        v = np.ones((4,1),'d')
        v_tp = np.zeros((4,1),'d'); v_tp[0] = 1.0/np.sqrt(2); v_tp[3] = 1.0/np.sqrt(2) - 0.05
        v_id = np.zeros((4,1),'d'); v_id[0] = 1.0/np.sqrt(2)
        povm = pygsti.obj.UnconstrainedPOVM( [('0',pygsti.obj.FullyParameterizedSPAMVec(v))] )
        tppovm = pygsti.obj.TPPOVM( [('0',pygsti.obj.FullyParameterizedSPAMVec(v)),
                                     ('1',pygsti.obj.FullyParameterizedSPAMVec(v_id-v))] )
        compSV = tppovm['1'] #complement POVM
        self.assertTrue(isinstance(compSV,pygsti.obj.ComplementSPAMVec))

        dummyGS = pygsti.objects.GateSet()
        dummyGS.povms['Mtest'] = povm # so to/from vector work w/tensor prod of povm in tests below
        assert(povm.gpindices is not None)
        
        vecs = [ pygsti.obj.FullyParameterizedSPAMVec(v),
                 pygsti.obj.TPParameterizedSPAMVec(v_tp),
                 pygsti.obj.CPTPParameterizedSPAMVec(v_tp, "pp"),
                 pygsti.obj.StaticSPAMVec(v),
                 compSV,
                 pygsti.obj.TensorProdSPAMVec("prep", [pygsti.obj.FullyParameterizedSPAMVec(v),
                                                       pygsti.obj.FullyParameterizedSPAMVec(v)]),
                 pygsti.obj.TensorProdSPAMVec("effect", [povm], ['0'])
                ]

        with self.assertRaises(ValueError):
            pygsti.obj.TensorProdSPAMVec("foobar",
                                         [pygsti.obj.FullyParameterizedSPAMVec(v),
                                          pygsti.obj.FullyParameterizedSPAMVec(v)])

        for sv in vecs:
            print("Testing %s spam vec -------------- " % type(sv))

            try:
                Np = sv.num_params()
            except NotImplementedError:
                Np = 0 # OK, b/c complement spam vec isn't a true spamvec...
                
            sv.size
            sv_str = str(sv)

            try:
                deriv = sv.deriv_wrt_params()
            except NotImplementedError:
                pass #OK - some not implemented yet... (like TensorProdSPAMVec)
            
            if Np > 0:
                try:
                    deriv = sv.deriv_wrt_params([0])
                except NotImplementedError:
                    pass #OK - some not implemented yet...

            if sv.has_nonzero_hessian():
                try:
                    sv.hessian_wrt_params()
                    sv.hessian_wrt_params([0])
                    sv.hessian_wrt_params([0],[0])
                except NotImplementedError:
                    pass #OK for NOW -- later this should always be implemented when there's a nonzero hessian!!

            
            sv.frobeniusdist2(sv,"prep")
            sv.frobeniusdist2(sv,"effect")
            with self.assertRaises(ValueError):
                sv.frobeniusdist2(sv,"foobar")

            sv2 = sv.copy()
            try:
                vec = sv.to_vector()
                sv2.from_vector(vec)
            except ValueError:
                assert(isinstance(sv, (pygsti.obj.TensorProdSPAMVec,pygsti.obj.ComplementSPAMVec)))
                # OK for TensorProd case and Complement case (when one should *not* call to_vector)

            T = pygsti.objects.FullGaugeGroupElement(
                np.array( [ [0,1],
                            [1,0] ], 'd') )
            T2 = pygsti.objects.UnitaryGaugeGroupElement(
                np.array( [ [1,0],
                            [0,1] ], 'd') )

            try:
                sv.transform(T,"prep")
                sv.transform(T,"effect")
            except ValueError: pass #OK, as this is unallowed for some gate types
            except NotImplementedError: pass #also OK, as this signifies derived class doesn't even implement it
            
            try:
                sv.transform(T2, "prep")
                sv.transform(T2, "effect")
            except ValueError: pass #OK, as this is unallowed for some gate types
            except NotImplementedError: pass #also OK, as this signifies derived class doesn't even implement it

            try:
                with self.assertRaises(ValueError):
                    sv.transform(T,"foobar") #invalid type
            except NotImplementedError: pass #also OK
                
            try:
                sv.depolarize(0.05)
            except ValueError: pass #OK, as this is unallowed for some gate types
            except NotImplementedError: pass #also OK, as this signifies derived class doesn't even implement it


            try:
                sv.set_value( np.zeros( (sv.dim,1), 'd') )
            except ValueError:
                pass #OK - some don't allow setting value
            

    def test_convert(self):
        v_tp = np.zeros((4,1),'d'); v_tp[0] = 1.0/np.sqrt(2); v_tp[3] = 1.0/np.sqrt(2) - 0.05
        s = pygsti.obj.FullyParameterizedSPAMVec(v_tp)

        for toType in ("full","TP","CPTP","static"):
            s2 = pygsti.objects.spamvec.convert(s, toType, "pp")
            pygsti.objects.spamvec.convert(s2, toType, "pp") #no conversion necessary

        ssv = pygsti.obj.StaticSPAMVec(v_tp)
        pygsti.objects.spamvec.optimize_spamvec(ssv, s) #test opt of static spamvec (nothing to do)

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

        vec.depolarize(0.01)
        vec.depolarize((0.1,0.09,0.08))        
        

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


    def test_povms(self):
        gateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi'], ["I(Q0)"])
        gateset2Q = pygsti.construction.build_gateset(
            [4], [('Q0','Q1')],['Gi'], ["I(Q0)"])

        povm = gateset.povms['Mdefault'].copy()
        E0 = povm['0']
        E1 = povm['1']
        gateset.povms['Munconstrained'] = povm # so gpindices get setup

        with self.assertRaises(ValueError):
            pygsti.obj.povm.convert(povm, "foobar", gateset.basis)
        with self.assertRaises(ValueError):
            pygsti.obj.UnconstrainedPOVM( "NotAListOrDict" )

        povm['0'] = E0 # assignment
        tp_povm = pygsti.obj.povm.convert(povm, "TP", gateset.basis)
        tp_povm['0'] = E0 # ok
        with self.assertRaises(KeyError):
            tp_povm['1'] = E0 # can't assign complement vector
        gateset.povms['Mtp'] = tp_povm # so gpindices get setup

        factorPOVMs = [povm, povm.copy()]
        tensor_povm = pygsti.obj.TensorProdPOVM( factorPOVMs )
        gateset2Q.povms['Mtensor'] = tensor_povm # so gpindices get setup

        for i,p in enumerate([povm, tp_povm, tensor_povm]):
            print("Testing POVM of type ", type(p))
            Nels = p.num_elements()
            cpy = p.copy()
            s = str(p)
            
            s = pickle.dumps(p)
            x = pickle.loads(s)

            T = pygsti.objects.FullGaugeGroupElement(
                np.array( [ [0,1],
                            [1,0] ], 'd') )

            v = p.to_vector()
            p.from_vector(v)

            v = gateset.to_vector() if i < 2 else gateset2Q.to_vector()
            effects = p.compile_effects(prefix="ABC")
            for Evec in effects.values():
                print("inds = ",Evec.gpindices, len(v))
                Evec.from_vector(v[Evec.gpindices]) # gpindices should be setup relative to GateSet's param vec


            try:
                p.transform(T)
            except ValueError:
                pass #OK - tensorprod doesn't allow transform for instance

            try:
                p.depolarize(0.01)
            except ValueError:
                pass #OK - tensorprod doesn't allow transform for instance

    def test_compbasis_povm(self):
        cv = pygsti.obj.ComputationalSPAMVec([0,1],'densitymx')
        v = pygsti.construction.basis_build_vector("1", pygsti.obj.Basis("pp",2**2))
        self.assertTrue(np.linalg.norm(cv.todense()-v.flat) < 1e-6)

        cv = pygsti.obj.ComputationalSPAMVec([0,0,1],'densitymx')
        v = pygsti.construction.basis_build_vector("1", pygsti.obj.Basis("pp",2**3))
        self.assertTrue(np.linalg.norm(cv.todense()-v.flat) < 1e-6)

        cv = pygsti.obj.ComputationalSPAMVec([0,0,1],'densitymx')
        v = pygsti.construction.basis_build_vector("1", pygsti.obj.Basis("pp",2**3))
        self.assertTrue(np.linalg.norm(cv.todense()-v.flat) < 1e-6)

        cv = pygsti.obj.ComputationalSPAMVec([0,0,1],'densitymx')
        v = pygsti.construction.basis_build_vector("1", pygsti.obj.Basis("pp",2**3))
        self.assertTrue(np.linalg.norm(cv.todense()-v.flat) < 1e-6)

        #Only works with Python replib (only there is todense implemented)
        #cv = pygsti.obj.ComputationalSPAMVec([0,1,1],'densitymx')
        #v = pygsti.construction.basis_build_vector("3", pygsti.obj.Basis("pp",2**3))
        #s = pygsti.obj.FullyParameterizedSPAMVec(v)
        #assert(np.linalg.norm(cv.torep("effect").todense(np.empty(cv.dim,'d'))-v.flat) < 1e-6)
        #
        #cv = pygsti.obj.ComputationalSPAMVec([0,1,0,1],'densitymx')
        #v = pygsti.construction.basis_build_vector("5", pygsti.obj.Basis("pp",2**4))
        #assert(np.linalg.norm(cv.torep("effect").todense(np.empty(cv.dim,'d'))-v.flat) < 1e-6)

        nqubits = 3
        iterover = [(0,1)]*nqubits
        items = [ (''.join(map(str,outcomes)), pygsti.obj.ComputationalSPAMVec(outcomes,"densitymx"))
                  for outcomes in itertools.product(*iterover) ]
        povm = pygsti.obj.UnconstrainedPOVM(items)
        self.assertEqual(povm.num_params(),0)

        gs = std1Q_XYI.gs_target.copy()
        gs.preps['rho0'] = pygsti.obj.ComputationalSPAMVec([0],'densitymx')
        gs.povms['Mdefault'] = pygsti.obj.UnconstrainedPOVM({'0': pygsti.obj.ComputationalSPAMVec([0],'densitymx'),
                                                             '1': pygsti.obj.ComputationalSPAMVec([1],'densitymx')})

        ps0 = gs.probs(())
        ps1 = gs.probs(('Gx',))
        self.assertAlmostEqual(ps0['0'], 1.0)
        self.assertAlmostEqual(ps0['1'], 0.0)
        self.assertAlmostEqual(ps1['0'], 0.5)
        self.assertAlmostEqual(ps1['1'], 0.5)
            

        

        

if __name__ == '__main__':
    unittest.main(verbosity=2)


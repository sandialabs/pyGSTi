from __future__ import division
import unittest
import pickle
import pygsti
import numpy as np
import warnings
import os
import collections

from ..testutils import BaseTestCase, compare_files, temp_files


class TestGateSetConstructionMethods(BaseTestCase):

    def setUp(self):
        super(TestGateSetConstructionMethods, self).setUp()

        #OK for these tests, since we test user interface?
        #Set Model objects to "strict" mode for testing
        pygsti.objects.ExplicitOpModel._strict = False


    def test_build_gatesets(self):
        SQ2 = 1/np.sqrt(2)
        for defParamType in ("full", "TP", "static"):
            gateset_simple = pygsti.objects.ExplicitOpModel(['Q0'],'pp',defParamType)
            gateset_simple['rho0'] = [SQ2, 0, 0, SQ2]
            gateset_simple['Mdefault'] = pygsti.obj.UnconstrainedPOVM( [('0',[SQ2, 0, 0, -SQ2])] )
            gateset_simple['Gi'] = [ [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1] ]

            with self.assertRaises(TypeError):
                gateset_simple['rho0'] = 3.0
            with self.assertRaises(ValueError):
                gateset_simple['rho0'] = [3.0]
            with self.assertRaises(ValueError):
                gateset_simple['Gx'] = [1,2,3,4]
            with self.assertRaises(ValueError):
                gateset_simple['Gx'] = [[1,2,3,4],[5,6,7]]

        gateset_badDefParam = pygsti.objects.ExplicitOpModel(['Q0'],"pp","full")
        gateset_badDefParam.preps.default_param = "foobar"
        gateset_badDefParam.operations.default_param = "foobar"
        with self.assertRaises(ValueError):
            gateset_badDefParam['rho0'] = [1, 0, 0, 0]
        with self.assertRaises(ValueError):
            gateset_badDefParam['Gi'] = np.identity(4,'d')

        with self.assertRaises(AssertionError):
            pygsti.construction.build_identity_vec(stateSpace, basis="foobar")


        gateset_povm_first = pygsti.objects.ExplicitOpModel(['Q0']) #set effect vector first
        gateset_povm_first['Mdefault'] = pygsti.obj.TPPOVM(
            [ ('0', pygsti.construction.build_vector(stateSpace,spaceLabels,"0")),
              ('1', pygsti.construction.build_vector(stateSpace,spaceLabels,"1")) ] )

        with self.assertRaises(ValueError):
            gateset_povm_first['rhoBad'] =  np.array([1,2,3],'d') #wrong dimension
        with self.assertRaises(ValueError):
            gateset_povm_first['Mdefault'] =  pygsti.obj.UnconstrainedPOVM( [('0',np.array([1,2,3],'d'))] ) #wrong dimension

    def test_gate_object(self):

        #Build each type of gate
        gate_full = pygsti.construction.build_operation( [(4,)],[('Q0',)], "X(pi/8,Q0)","gm", parameterization="full")
        gate_linear = pygsti.construction.build_operation( [(4,)],[('Q0',)], "I(Q0)","gm", parameterization="full") # 'I' was 'D', 'full' was 'linear'
        gate_tp = pygsti.construction.build_operation( [(4,)],[('Q0',)], "Y(pi/4,Q0)","gm", parameterization="TP")
        gate_static = pygsti.construction.build_operation( [(4,)],[('Q0',)], "Z(pi/3,Q0)","gm", parameterization="static")
        gate_objs = [gate_full, gate_linear, gate_tp, gate_static]

        self.assertEqual(gate_full.num_params(), 16)
        self.assertEqual(gate_linear.num_params(), 16)
        self.assertEqual(gate_tp.num_params(), 12)
        self.assertEqual(gate_static.num_params(), 0)

        #Test gate methods
        for gate in gate_objs:
            gate_copy = gate.copy()
            self.assertArraysAlmostEqual(gate_copy, gate)
            self.assertEqual(type(gate_copy), type(gate))

            self.assertEqual(gate.get_dimension(), 4)

            M = np.asarray(gate) #gate as a matrix
            if isinstance(gate, (pygsti.obj.LinearlyParamDenseOp,pygsti.obj.StaticDenseOp)):
                with self.assertRaises(ValueError):
                    gate.set_value(M)
            else:
                gate.set_value(M)

            with self.assertRaises(ValueError):
                gate.set_value( np.zeros((1,1),'d') ) #bad size

            v = gate.to_vector()
            gate.from_vector(v)
            deriv = gate.deriv_wrt_params()
            #test results?

            T = pygsti.obj.FullGaugeGroupElement(np.identity(4,'d'))
            if type(gate) in (pygsti.obj.LinearlyParamDenseOp,
                              pygsti.obj.StaticDenseOp):
                with self.assertRaises(ValueError):
                    gate_copy.transform(T)
            else:
                gate_copy.transform(T)

            self.assertArraysAlmostEqual(gate_copy, gate)

            gate_as_str = str(gate)

            pklstr = pickle.dumps(gate)
            gate_copy = pickle.loads(pklstr)
            self.assertArraysAlmostEqual(gate_copy, gate)
            self.assertEqual(type(gate_copy), type(gate))

              #math ops
            result = gate + gate
            self.assertEqual(type(result), np.ndarray)
            result = gate + (-gate)
            self.assertEqual(type(result), np.ndarray)
            result = gate - gate
            self.assertEqual(type(result), np.ndarray)
            result = gate - abs(gate)
            self.assertEqual(type(result), np.ndarray)
            result = 2*gate
            self.assertEqual(type(result), np.ndarray)
            result = gate*2
            self.assertEqual(type(result), np.ndarray)
            result = 2/gate
            self.assertEqual(type(result), np.ndarray)
            result = gate/2
            self.assertEqual(type(result), np.ndarray)
            result = gate//2
            self.assertEqual(type(result), np.ndarray)
            result = gate**2
            self.assertEqual(type(result), np.ndarray)
            result = gate.transpose()
            self.assertEqual(type(result), np.ndarray)


            M = np.identity(4,'d')

            result = gate + M
            self.assertEqual(type(result), np.ndarray)
            result = gate - M
            self.assertEqual(type(result), np.ndarray)
            result = M + gate
            self.assertEqual(type(result), np.ndarray)
            result = M - gate
            self.assertEqual(type(result), np.ndarray)




        #Test compositions (and conversions)
        c = pygsti.obj.compose(gate_full, gate_full, "gm", "full")
        self.assertArraysAlmostEqual(c, np.dot(gate_full,gate_full) )
        self.assertEqual(type(c), pygsti.obj.FullDenseOp)

        c = pygsti.obj.compose(gate_full, gate_tp, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_full,gate_tp) )
        self.assertEqual(type(c), pygsti.obj.FullDenseOp)

        c = pygsti.obj.compose(gate_full, gate_static, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_full,gate_static) )
        self.assertEqual(type(c), pygsti.obj.FullDenseOp)

        c = pygsti.obj.compose(gate_full, gate_linear, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_full,gate_linear) )
        self.assertEqual(type(c), pygsti.obj.FullDenseOp)


        c = pygsti.obj.compose(gate_linear, gate_full, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_linear,gate_full) )
        self.assertEqual(type(c), pygsti.obj.FullDenseOp)

        #c = pygsti.obj.compose(gate_linear, gate_tp, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(gate_linear,gate_tp) )
        #self.assertEqual(type(c), pygsti.obj.TPDenseOp)

        #c = pygsti.obj.compose(gate_linear, gate_static, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(gate_linear,gate_static) )
        #self.assertEqual(type(c), pygsti.obj.LinearlyParamDenseOp)

        #c = pygsti.obj.compose(gate_linear, gate_linear, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(gate_linear,gate_linear) )
        #self.assertEqual(type(c), pygsti.obj.LinearlyParamDenseOp)


        c = pygsti.obj.compose(gate_tp, gate_full, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_tp,gate_full) )
        self.assertEqual(type(c), pygsti.obj.FullDenseOp)

        c = pygsti.obj.compose(gate_tp, gate_tp, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_tp,gate_tp) )
        self.assertEqual(type(c), pygsti.obj.TPDenseOp)

        c = pygsti.obj.compose(gate_tp, gate_static, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_tp,gate_static) )
        self.assertEqual(type(c), pygsti.obj.TPDenseOp)

        #c = pygsti.obj.compose(gate_tp, gate_linear, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(gate_tp,gate_linear) )
        #self.assertEqual(type(c), pygsti.obj.TPDenseOp)


        c = pygsti.obj.compose(gate_static, gate_full, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_static,gate_full) )
        self.assertEqual(type(c), pygsti.obj.FullDenseOp)

        c = pygsti.obj.compose(gate_static, gate_tp, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_static,gate_tp) )
        self.assertEqual(type(c), pygsti.obj.TPDenseOp)

        c = pygsti.obj.compose(gate_static, gate_static, "gm")
        self.assertArraysAlmostEqual(c, np.dot(gate_static,gate_static) )
        self.assertEqual(type(c), pygsti.obj.StaticDenseOp)

        #c = pygsti.obj.compose(gate_static, gate_linear, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(gate_static,gate_linear) )
        #self.assertEqual(type(c), pygsti.obj.LinearlyParamDenseOp)

        #Test specific conversions that don't get tested by compose
        conv = pygsti.obj.operation.convert(gate_tp, "full", "gm")
        conv = pygsti.obj.operation.convert(gate_tp, "TP", "gm")
        conv = pygsti.obj.operation.convert(gate_static, "static", "gm")

        with self.assertRaises(ValueError):
            pygsti.obj.operation.convert(gate_full, "linear", "gm") #unallowed
        with self.assertRaises(ValueError):
            pygsti.obj.operation.convert(gate_full, "foobar", "gm")


        #Test element access/setting

          #full
        e1 = gate_full[1,1]
        e2 = gate_full[1][1]
        self.assertAlmostEqual(e1,e2)

        s1 = gate_full[1,:]
        s2 = gate_full[1]
        s3 = gate_full[1][:]
        a1 = gate_full[:]
        self.assertArraysAlmostEqual(s1,s2)
        self.assertArraysAlmostEqual(s1,s3)

        s4 = gate_full[2:4,1]

        gate_full[1,1] = e1
        gate_full[1,:] = s1
        gate_full[1] = s1
        gate_full[2:4,1] = s4

        result = len(gate_full)
        with self.assertRaises(TypeError):
            result = int(gate_full) #can't convert
        with self.assertRaises(TypeError):
            result = int(gate_full) #can't convert
        with self.assertRaises(TypeError):
            result = float(gate_full) #can't convert
        with self.assertRaises(TypeError):
            result = complex(gate_full) #can't convert


          #static (same as full case)
        e1 = gate_static[1,1]
        e2 = gate_static[1][1]
        self.assertAlmostEqual(e1,e2)

        s1 = gate_static[1,:]
        s2 = gate_static[1]
        s3 = gate_static[1][:]
        a1 = gate_static[:]
        self.assertArraysAlmostEqual(s1,s2)
        self.assertArraysAlmostEqual(s1,s3)

        s4 = gate_static[2:4,1]

        gate_static[1,1] = e1
        gate_static[1,:] = s1
        gate_static[1] = s1
        gate_static[2:4,1] = s4


          #TP (can't modify first row)
        e1 = gate_tp[0,0]
        e2 = gate_tp[0][0]
        self.assertAlmostEqual(e1,e2)
        self.assertAlmostEqual(e1,1.0)

        s1 = gate_tp[1,:]
        s2 = gate_tp[1]
        s3 = gate_tp[1][:]
        a1 = gate_tp[:]
        self.assertArraysAlmostEqual(s1,s2)
        self.assertArraysAlmostEqual(s1,s3)

        s4 = gate_tp[2:4,1]

        # check that first row is read-only
        with self.assertRaises(ValueError):
            gate_tp[0,0] = e1
        with self.assertRaises(ValueError):
            gate_tp[0][0] = e1
        with self.assertRaises(ValueError):
            gate_tp[0,:] = [ e1, 0, 0, 0 ]
        with self.assertRaises(ValueError):
            gate_tp[0][:] = [ e1, 0, 0, 0 ]
        with self.assertRaises(ValueError):
            gate_tp[0,1:2] = [ 0 ]
        with self.assertRaises(ValueError):
            gate_tp[0][1:2] = [ 0 ]

        gate_tp[1,:] = s1
        gate_tp[1] = s1
        gate_tp[2:4,1] = s4


          #linear
        e1 = gate_linear[1,1]
        e2 = gate_linear[1][1]
        self.assertAlmostEqual(e1,e2)

        s1 = gate_linear[1,:]
        s2 = gate_linear[1]
        s3 = gate_linear[1][:]
        a1 = gate_linear[:]
        self.assertArraysAlmostEqual(s1,s2)
        self.assertArraysAlmostEqual(s1,s3)

        s4 = gate_linear[2:4,1]

        #Linear gate REMOVED
        ## check that cannot set anything
        #with self.assertRaises(ValueError):
        #    gate_linear[1,1] = e1
        #with self.assertRaises(ValueError):
        #    gate_linear[1,:] = s1
        #with self.assertRaises(ValueError):
        #    gate_linear[1] = s1
        #with self.assertRaises(ValueError):
        #    gate_linear[2:4,1] = s4



        #Full from scratch
        gate_full_B = pygsti.obj.FullDenseOp([[1,0],[0,1]])

        numParams = gate_full_B.num_params()
        v = gate_full_B.to_vector()
        gate_full_B.from_vector(v)
        deriv = gate_full_B.deriv_wrt_params()


        #Linear from scratch
        baseMx = np.zeros( (2,2) )
        paramArray = np.array( [1.0,1.0] )
        parameterToBaseIndicesMap = { 0: [(0,0)], 1: [(1,1)] } #parameterize only the diagonal els
        gate_linear_B = pygsti.obj.LinearlyParamDenseOp(baseMx, paramArray,
                                                             parameterToBaseIndicesMap, real=True)
        with self.assertRaises(AssertionError):
            pygsti.obj.LinearlyParamDenseOp(baseMx, np.array( [1.0+1j, 1.0] ),
                                                 parameterToBaseIndicesMap, real=True) #must be real

        numParams = gate_linear_B.num_params()
        v = gate_linear_B.to_vector()
        gate_linear_B.from_vector(v)
        deriv = gate_linear_B.deriv_wrt_params()


    def test_spamvec_object(self):
        full_spamvec = pygsti.obj.FullSPAMVec([ 1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2) ] )
        tp_spamvec = pygsti.obj.TPSPAMVec([ 1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2) ] )
        static_spamvec = pygsti.obj.StaticSPAMVec([ 1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2) ] )
        spamvec_objs = [full_spamvec, tp_spamvec, static_spamvec]

        with self.assertRaises(ValueError):
            pygsti.obj.FullSPAMVec([[ 1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2) ],[0,0,0,0]] )
            # 2nd dimension must == 1
        with self.assertRaises(ValueError):
            pygsti.obj.TPSPAMVec([ 1.0, 0, 0, 0 ])
            # incorrect initial element for TP!
        with self.assertRaises(ValueError):
            tp_spamvec.set_value([1.0 ,0, 0, 0])
            # incorrect initial element for TP!


        self.assertEqual(full_spamvec.num_params(), 4)
        self.assertEqual(tp_spamvec.num_params(), 3)
        self.assertEqual(static_spamvec.num_params(), 0)

        for svec in spamvec_objs:
            svec_copy = svec.copy()
            self.assertArraysAlmostEqual(svec_copy, svec)
            self.assertEqual(type(svec_copy), type(svec))

            self.assertEqual(svec.get_dimension(), 4)

            v = np.asarray(svec)
            if isinstance(svec, pygsti.obj.StaticSPAMVec):
                with self.assertRaises(ValueError):
                    svec.set_value(svec)
            else:
                svec.set_value(svec)

            with self.assertRaises(ValueError):
                svec.set_value( np.zeros((1,1),'d') ) #bad size

            v = svec.to_vector()
            svec.from_vector(v)
            deriv = svec.deriv_wrt_params()
            #test results?

            a = svec[:]
            b = svec[0]
            #with self.assertRaises(ValueError):
            #    svec.shape = (2,2) #something that would affect the shape??

            svec_as_str = str(svec)
            a1 = svec[:] #invoke getslice method

            pklstr = pickle.dumps(svec)
            svec_copy = pickle.loads(pklstr)
            self.assertArraysAlmostEqual(svec_copy, svec)
            self.assertEqual(type(svec_copy), type(svec))

              #math ops
            result = svec + svec
            self.assertEqual(type(result), np.ndarray)
            result = svec + (-svec)
            self.assertEqual(type(result), np.ndarray)
            result = svec - svec
            self.assertEqual(type(result), np.ndarray)
            result = svec - abs(svec)
            self.assertEqual(type(result), np.ndarray)
            result = 2*svec
            self.assertEqual(type(result), np.ndarray)
            result = svec*2
            self.assertEqual(type(result), np.ndarray)
            result = 2/svec
            self.assertEqual(type(result), np.ndarray)
            result = svec/2
            self.assertEqual(type(result), np.ndarray)
            result = svec//2
            self.assertEqual(type(result), np.ndarray)
            result = svec**2
            self.assertEqual(type(result), np.ndarray)
            result = svec.transpose()
            self.assertEqual(type(result), np.ndarray)

            V = np.ones((4,1),'d')

            result = svec + V
            self.assertEqual(type(result), np.ndarray)
            result = svec - V
            self.assertEqual(type(result), np.ndarray)
            result = V + svec
            self.assertEqual(type(result), np.ndarray)
            result = V - svec
            self.assertEqual(type(result), np.ndarray)

        #Run a few methods that won't work on static spam vecs
        for svec in (full_spamvec, tp_spamvec):
            v = svec.copy()
            S = pygsti.objects.FullGaugeGroupElement( np.identity(4,'d') )
            v.transform(S, 'prep')
            v.transform(S, 'effect')
            with self.assertRaises(ValueError):
                v.transform(S,'foobar')

            v.depolarize(0.9)
            v.depolarize([0.9,0.8,0.7])

        #Ensure we aren't allowed to tranform or depolarize a static vector
        with self.assertRaises(ValueError):
            S = pygsti.objects.FullGaugeGroupElement( np.identity(4,'d') )
            static_spamvec.transform(S,'prep')

        with self.assertRaises(ValueError):
            static_spamvec.depolarize(0.9)

        #Test conversions to own type (not tested elsewhere)
        basis = pygsti.obj.Basis.cast("pp",4)
        conv = pygsti.obj.spamvec.convert(full_spamvec, "full", basis)
        conv = pygsti.obj.spamvec.convert(tp_spamvec, "TP", basis)
        conv = pygsti.obj.spamvec.convert(static_spamvec, "static", basis)
        with self.assertRaises(ValueError):
            pygsti.obj.spamvec.convert(full_spamvec, "foobar", basis)


if __name__ == "__main__":
    unittest.main(verbosity=2)

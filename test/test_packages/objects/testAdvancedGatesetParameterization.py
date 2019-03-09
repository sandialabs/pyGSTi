from __future__ import print_function

import unittest
import pygsti
import numpy as np
import scipy.sparse as sps
        
from pygsti.construction import std1Q_XYI
from pygsti.construction import std2Q_XYICNOT
from pygsti.objects import LindbladDenseOp, ComposedDenseOp, EmbeddedDenseOp, StaticDenseOp
from pygsti.objects import LindbladOp

from ..testutils import BaseTestCase, compare_files, temp_files

class AdvancedParameterizationTestCase(BaseTestCase):

    def setUp(self):
        super(AdvancedParameterizationTestCase, self).setUp()
        #Nothing yet...

    def test_composed_embedded_param(self):
        #Test out above cell
        gs1Q = std1Q_XYI.target_model()
        
        print("START")
        
        nQubits = 3 # say
        Id_1Q = np.identity(4**1,'d')
        idleErr0 = LindbladDenseOp.from_operation_matrix(Id_1Q) # 1-qubit error generator
        idleErr1 = LindbladDenseOp.from_operation_matrix(Id_1Q) # allow different "idle" 
        idleErr2 = LindbladDenseOp.from_operation_matrix(Id_1Q) # 1Q errors on each qubit
        # so far no gpindices have been set...
        
        ss3Q = [('Q0','Q1','Q2')] #3Q state space
        basis3Q = pygsti.objects.Basis.cast('pp', 4**nQubits) #3Q basis
        Giii = ComposedDenseOp([ EmbeddedDenseOp(ss3Q, ('Q0',), idleErr0),
                              EmbeddedDenseOp(ss3Q, ('Q1',), idleErr1),
                              EmbeddedDenseOp(ss3Q, ('Q2',), idleErr2)
                            ])
        
        targetGx = StaticDenseOp(gs1Q.operations['Gx'])
        Gxii_xErr = LindbladDenseOp.from_operation_matrix(Id_1Q) 
        Gxii_xGate = ComposedDenseOp( [targetGx, idleErr0, Gxii_xErr])
        Gxii = ComposedDenseOp([ EmbeddedDenseOp(ss3Q, ('Q0',), Gxii_xGate),
                              EmbeddedDenseOp(ss3Q, ('Q1',), idleErr1),
                              EmbeddedDenseOp(ss3Q, ('Q2',), idleErr2)
                            ])
        
        def printInfo():
            def pp(x): return id(x) if (x is not None) else x #print's parent nicely
            print("INDEX INFO")
            print("idleErr0 (%d):" % id(idleErr0), idleErr0.gpindices, pp(idleErr0.parent), idleErr0.num_params())
            print("idleErr1 (%d):" % id(idleErr1), idleErr1.gpindices, pp(idleErr1.parent), idleErr1.num_params())
            print("idleErr2 (%d):" % id(idleErr2), idleErr2.gpindices, pp(idleErr2.parent), idleErr2.num_params())
            print("Gxii_xErr (%d):" % id(Gxii_xErr), Gxii_xErr.gpindices, pp(Gxii_xErr.parent), Gxii_xErr.num_params())
            print("Gxii_xGate (%d):" % id(Gxii_xGate), Gxii_xGate.gpindices, pp(Gxii_xGate.parent), Gxii_xGate.num_params())
            print("Giii (%d):" % id(Giii), Giii.gpindices, pp(Giii.parent), Giii.num_params())
            print("Gxii (%d):" % id(Gxii), Gxii.gpindices, pp(Gxii.parent), Gxii.num_params())
            print()
        
        # rubber meets road: how to assign gpindices??
        # need model.from_vector() to work, and also to_vector(), but maybe less important
        print("PREGAME")
        printInfo()
        print("BEGIN")
        mdl_constructed = pygsti.obj.ExplicitOpModel(['Q0','Q1','Q2'])
        print("Model id = ",id(mdl_constructed))
        print("INSERT1: Giii indices = ", Giii.gpindices, " parent = ", Giii.parent)
        mdl_constructed.operations['Giii'] = Giii # will set gpindices of Giii, which will set those of 
         # of it's contained EmbeddedGates which will set gpindices of idleErr0, idleErr1, & idleErr2
         # (because they're None to begin with)
        print("POST")
        Giii = mdl_constructed.operations['Giii'] #so printInfo works
        printInfo()
        print("INSERT2: Gxii indices = ", Gxii.gpindices, " parent = ", Gxii.parent)
        mdl_constructed.operations['Gxii'] = Gxii # similar, but will only set indices of Gxii_xGate (gpindices
         # of idleErr1 and idleErr2 are already set), which will only set indices of Gxii_xErr
         # (since idleErr0.gpindices is already set )
        Giii = mdl_constructed.operations['Giii'] #so printInfo works
        Gxii = mdl_constructed.operations['Gxii'] #so printInfo works
        printInfo()
        
        print("TOTAL Params = ", mdl_constructed.num_params())
        
        v = mdl_constructed.to_vector()
        print("len(v) =",len(v))
        mdl_constructed2 = mdl_constructed.copy()
        print("Copy's total params = ", mdl_constructed2.num_params())
        mdl_constructed2.from_vector(v)
        print("Diff = %g (should be 0)" % mdl_constructed.frobeniusdist(mdl_constructed2))
        self.assertAlmostEqual(mdl_constructed.frobeniusdist(mdl_constructed2),0)

    def test_sparse_lindblad_param(self):
        #Test sparse bases and Lindblad gates

        #To catch sparse warnings as error (for debugging)
        #import warnings
        #from scipy.sparse import SparseEfficiencyWarning
        #warnings.resetwarnings()
        #warnings.simplefilter('error',SparseEfficiencyWarning)
        #warnings.simplefilter('error',DeprecationWarning)
        
        sparsePP = pygsti.objects.Basis.cast("pp",16,sparse=True)
        mxs = sparsePP.elements
        #for lbl,mx in zip(sparsePP.labels,mxs):
        #    print(lbl,":",mx.shape,"matrix with",mx.nnz,"nonzero entries (of",
        #          mx.shape[0]*mx.shape[1],"total)")
        #    #print(mx.toarray())
        print("%d basis elements" % len(sparsePP))
        self.assertEqual(len(sparsePP), 16)
        
        M = np.ones((16,16),'d')
        v = np.ones(16,'d')
        S = sps.identity(16,'d','csr')

        print("Test types after basis change by sparse basis:")
        Mout = pygsti.tools.change_basis(M, sparsePP, 'std')
        vout = pygsti.tools.change_basis(v, sparsePP, 'std')
        Sout = pygsti.tools.change_basis(S, sparsePP, 'std')
        print(type(M),"->",type(Mout))
        print(type(v),"->",type(vout))
        print(type(S),"->",type(Sout))
        self.assertIsInstance(Mout, np.ndarray)
        self.assertIsInstance(vout, np.ndarray)
        self.assertIsInstance(Sout, sps.csr_matrix)
        
        print("\nGate Test:")
        SparseId = sps.identity(4**2,'d','csr')
        gate = LindbladDenseOp.from_operation_matrix( np.identity(4**2,'d') )
        print("gate Errgen type (should be dense):",type(gate.errorgen.err_gen_mx))
        self.assertIsInstance(gate.errorgen.err_gen_mx, np.ndarray)
        sparseOp = LindbladOp.from_operation_matrix( SparseId )
        print("spareGate Errgen type (should be sparse):",type(sparseOp.errorgen.err_gen_mx))
        self.assertIsInstance(sparseOp.errorgen.err_gen_mx, sps.csr_matrix)
        self.assertArraysAlmostEqual(gate.errorgen.err_gen_mx,sparseOp.errorgen.err_gen_mx.toarray())
        
        perfectG = std2Q_XYICNOT.target_model().operations['Gix'].copy()
        noisyG = std2Q_XYICNOT.target_model().operations['Gix'].copy()
        noisyG.depolarize(0.9)
        Sparse_noisyG = sps.csr_matrix(noisyG,dtype='d')
        Sparse_perfectG = sps.csr_matrix(perfectG,dtype='d')
        op2 = LindbladDenseOp.from_operation_matrix( noisyG, perfectG )
        sparseGate2 = LindbladOp.from_operation_matrix( Sparse_noisyG, Sparse_perfectG )
        print("spareGate2 Errgen type (should be sparse):",type(sparseGate2.errorgen.err_gen_mx))
        self.assertIsInstance(sparseGate2.errorgen.err_gen_mx, sps.csr_matrix)
        #print("errgen = \n"); pygsti.tools.print_mx(op2.err_gen,width=4,prec=1)
        #print("sparse errgen = \n"); pygsti.tools.print_mx(sparseGate2.err_gen.toarray(),width=4,prec=1)
        self.assertArraysAlmostEqual(op2.errorgen.err_gen_mx,sparseGate2.errorgen.err_gen_mx.toarray())

if __name__ == '__main__':
    unittest.main(verbosity=2)

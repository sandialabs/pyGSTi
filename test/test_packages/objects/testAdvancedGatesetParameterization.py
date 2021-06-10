
import unittest

import numpy as np
import scipy.sparse as sps

import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI
from pygsti.modelpacks.legacy import std2Q_XYICNOT
from pygsti.objects import LindbladDenseOp, ComposedDenseOp, EmbeddedDenseOp, StaticDenseOp
from pygsti.objects import LindbladOp
from ..testutils import BaseTestCase


class AdvancedParameterizationTestCase(BaseTestCase):

    def setUp(self):
        super(AdvancedParameterizationTestCase, self).setUp()
        #Nothing yet...

    def test_composed_embedded_param(self):
        #Test out above cell
        gs1Q = std1Q_XYI.target_model()

        print("START")

        Id_1Q = np.identity(4**1,'d')
        idleErr0 = LindbladDenseOp.from_operation_matrix(Id_1Q) # 1-qubit error generator
        idleErr1 = LindbladDenseOp.from_operation_matrix(Id_1Q) # allow different "idle"
        idleErr2 = LindbladDenseOp.from_operation_matrix(Id_1Q) # 1Q errors on each qubit
        # so far no gpindices have been set...

        ss3Q = [('Q0','Q1','Q2')] #3Q state space
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
            print("idleErr0 (%d):" % id(idleErr0), idleErr0.gpindices, pp(idleErr0.parent), idleErr0.num_params)
            print("idleErr1 (%d):" % id(idleErr1), idleErr1.gpindices, pp(idleErr1.parent), idleErr1.num_params)
            print("idleErr2 (%d):" % id(idleErr2), idleErr2.gpindices, pp(idleErr2.parent), idleErr2.num_params)
            print("Gxii_xErr (%d):" % id(Gxii_xErr), Gxii_xErr.gpindices, pp(Gxii_xErr.parent), Gxii_xErr.num_params)
            print("Gxii_xGate (%d):" % id(Gxii_xGate), Gxii_xGate.gpindices, pp(Gxii_xGate.parent), Gxii_xGate.num_params)
            print("Giii (%d):" % id(Giii), Giii.gpindices, pp(Giii.parent), Giii.num_params)
            print("Gxii (%d):" % id(Gxii), Gxii.gpindices, pp(Gxii.parent), Gxii.num_params)
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

        print("TOTAL Params = ", mdl_constructed.num_params)

        v = mdl_constructed.to_vector()
        print("len(v) =",len(v))
        mdl_constructed2 = mdl_constructed.copy()
        print("Copy's total params = ", mdl_constructed2.num_params)
        mdl_constructed2.from_vector(v)
        print("Diff = %g (should be 0)" % mdl_constructed.frobeniusdist(mdl_constructed2))
        self.assertAlmostEqual(mdl_constructed.frobeniusdist(mdl_constructed2),0)

    def test_sparse_lindblad_param(self):
        #Test sparse Lindblad gates

        print("\nGate Test:")
        SparseId = sps.identity(4**2,'d','csr')
        gate = LindbladDenseOp.from_operation_matrix( np.identity(4**2,'d') )
        print("gate Errgen type (should be dense):",type(gate.errorgen.to_dense()))
        self.assertIsInstance(gate.errorgen.to_dense(), np.ndarray)
        sparseOp = LindbladOp.from_operation_matrix( SparseId )
        print("spareGate Errgen type (should be sparse):",type(sparseOp.errorgen.to_sparse()))
        self.assertIsInstance(sparseOp.errorgen.to_sparse(), sps.csr_matrix)
        self.assertArraysAlmostEqual(gate.errorgen.to_dense(),sparseOp.errorgen.to_dense())

        perfectG = std2Q_XYICNOT.target_model().operations['Gix'].copy()
        noisyG = std2Q_XYICNOT.target_model().operations['Gix'].copy()
        noisyG.depolarize(0.9)
        Sparse_noisyG = sps.csr_matrix(noisyG,dtype='d')
        Sparse_perfectG = sps.csr_matrix(perfectG,dtype='d')
        op2 = LindbladDenseOp.from_operation_matrix( noisyG, perfectG )
        sparseGate2 = LindbladOp.from_operation_matrix( Sparse_noisyG, Sparse_perfectG )
        print("spareGate2 Errgen type (should be sparse):",type(sparseGate2.errorgen.to_sparse()))
        self.assertIsInstance(sparseGate2.errorgen.to_sparse(), sps.csr_matrix)
        #print("errgen = \n"); pygsti.tools.print_mx(op2.err_gen,width=4,prec=1)
        #print("sparse errgen = \n"); pygsti.tools.print_mx(sparseGate2.err_gen.toarray(),width=4,prec=1)
        self.assertArraysAlmostEqual(op2.errorgen.to_dense(),sparseGate2.errorgen.to_dense())

    def test_setting_lindblad_stochastic_error_rates(self):
        mdl_std1Q_HS = std1Q_XYI.target_model("H+S")

        ps_err = 0.01 # per-Pauli error - the p_i in usual operator rep of a depol channel
        d = 4 # number of Paulis (and dim of space)
        alpha = d*ps_err # per-Pauli error * num-Paulis
        depol_err = alpha * (d-1)/d # == (d-1) * ps_err
        Gx_depol = mdl_std1Q_HS.operations['Gx'].copy()
        #Gx_depol.depolarize( (alpha,alpha,0) )
        Gx_depol.set_error_rates( {('S','X'): 0.01, ('S','Y'): 0.01, ('S','Z'): 0.01} )
        print("Infidelity = ",pygsti.tools.entanglement_infidelity(Gx_depol, mdl_std1Q_HS.operations['Gx']))
        self.assertAlmostEqual(pygsti.tools.entanglement_infidelity(Gx_depol, mdl_std1Q_HS.operations['Gx']), depol_err)

        
        print("error rate of depol channel = ", depol_err)
        print("Each of %d pauli stochastic parts has error rate of " % (d-1),alpha * 1/d)

        expected_coeff = -np.log(1-d*ps_err) / d
        print("Errgen S-coeff should be:", expected_coeff )
        #pygsti.tools.print_mx(Gx_depol.todense())

        print("Coeffs are:")
        print(Gx_depol.errorgen_coefficients())
        print("Rates are:")
        print(Gx_depol.error_rates())
        
        self.assertAlmostEqual(Gx_depol.error_rates()[('S','X')],ps_err)
        self.assertAlmostEqual(Gx_depol.error_rates()[('S','Y')],ps_err)
        self.assertAlmostEqual(Gx_depol.error_rates()[('S','Z')],ps_err)
        self.assertAlmostEqual(Gx_depol.errorgen_coefficients()[('S','X')],expected_coeff)
        self.assertAlmostEqual(Gx_depol.errorgen_coefficients()[('S','Y')],expected_coeff)
        self.assertAlmostEqual(Gx_depol.errorgen_coefficients()[('S','Z')],expected_coeff)

    def test_setting_lindblad_hamiltonian_error_rates(self):
        mdl_std1Q_HS = std1Q_XYI.target_model("H+S")
        Gx_rot = mdl_std1Q_HS.operations['Gx'].copy()

        #Test 3 different ways of setting rotation angles.
        Gx_rot.rotate( (0.2,0.0,0) )
        self.assertAlmostEqual(Gx_rot.errorgen_coefficients()[('H','X')], 0.2)
        self.assertAlmostEqual(Gx_rot.error_rates()[('H','X')], 0.2)
        Gx_rot.set_error_rates({('H','Y'): 0.1})
        self.assertAlmostEqual(Gx_rot.errorgen_coefficients()[('H','Y')], 0.1)
        self.assertAlmostEqual(Gx_rot.error_rates()[('H','Y')], 0.1)
        Gx_rot.set_errorgen_coefficients({('H','Z'): 0.3})
        self.assertAlmostEqual(Gx_rot.errorgen_coefficients()[('H','Z')], 0.3)
        self.assertAlmostEqual(Gx_rot.error_rates()[('H','Z')], 0.3)
        

if __name__ == '__main__':
    unittest.main(verbosity=2)

from ..testutils import BaseTestCase, compare_files, temp_files

import pygsti
import pygsti.tools.gatetools as gatetools

from pygsti.construction import std2Q_XXYYII
from pygsti.construction import std1Q_XYI

import numpy as np
import unittest

A = np.array( [[0.9, 0, 0.1j, 0],
               [ 0,  0, 0,    0],
               [ -0.1j, 0, 0, 0],
               [ 0,  0,  0,  0.1]], 'complex')

B = np.array( [[0.5, 0, 0, -0.2j],
               [ 0,  0.25, 0,  0],
               [ 0, 0, 0.25,   0],
               [ 0.2j,  0,  0,  0.1]], 'complex')

class GateBaseTestCase(BaseTestCase):

    def test_gate_tools(self):
        oneRealPair = np.array( [[1+1j, 0, 0, 0],
                             [ 0, 1-1j,0, 0],
                             [ 0,   0, 2, 0],
                             [ 0,   0,  0, 2]], 'complex')
        decomp = pygsti.decompose_gate_matrix(oneRealPair)
            #decompose gate mx whose eigenvalues have a real but non-unit pair

        dblRealPair = np.array( [[3, 0, 0, 0],
                             [ 0, 3,0, 0],
                             [ 0,   0, 2, 0],
                             [ 0,   0,  0, 2]], 'complex')
        decomp = pygsti.decompose_gate_matrix(dblRealPair)
            #decompose gate mx whose eigenvalues have two real but non-unit pairs


        unpairedMx = np.array( [[1+1j, 0, 0, 0],
                                [ 0, 2-1j,0, 0],
                                [ 0,   0, 2+2j, 0],
                                [ 0,   0,  0,  1.0+3j]], 'complex')
        decomp = pygsti.decompose_gate_matrix(unpairedMx)
            #decompose gate mx which has all complex eigenvalue -> bail out
        self.assertFalse(decomp['isValid'])

        largeMx = np.identity(16,'d')
        decomp = pygsti.decompose_gate_matrix(largeMx) #can only handle 1Q mxs
        self.assertFalse(decomp['isValid'])

        self.assertAlmostEqual( pygsti.frobeniusdist(A,A), 0.0 )
        self.assertAlmostEqual( pygsti.jtracedist(A,A,mxBasis="std"), 0.0 )
        self.assertAlmostEqual( pygsti.diamonddist(A,A,mxBasis="std"), 0.0 )
        self.assertAlmostEqual( pygsti.frobeniusdist(A,B), (0.430116263352+0j) )
        self.assertAlmostEqual( pygsti.jtracedist(A,B,mxBasis="std"), 0.26430148 ) #OLD: 0.2601 ?
        self.assertAlmostEqual( pygsti.diamonddist(A,B,mxBasis="std"), 0.614258836298)

        self.assertAlmostEqual( pygsti.frobeniusdist(A,B), pygsti.frobeniusnorm(A-B) )
        self.assertAlmostEqual( pygsti.frobeniusdist(A,B), np.sqrt( pygsti.frobeniusnorm2(A-B) ) )

    def test_hack_sqrt_m(self):
        expected = np.array([[ 0.55368857+0.46439416j,  0.80696073-0.21242648j],
             [ 1.21044109-0.31863972j,  1.76412966+0.14575444j]]
             )
        sqrt = gatetools._hack_sqrtm(np.array([[1, 2], [3, 4]]))
        self.assertArraysAlmostEqual(sqrt, expected)

    def test_frobenius_distance(self):
        self.assertAlmostEqual( pygsti.frobeniusdist(A,A), 0.0 )
        self.assertAlmostEqual( pygsti.frobeniusdist2(A,A), 0.0 )

    def test_process_fidelity(self):
        fidelity = gatetools.process_fidelity(A, B)
        self.assertAlmostEqual(fidelity, 0.42686642003)

    def test_fidelity_upper_bound(self):
        upperBound = gatetools.fidelity_upper_bound(A)
        expected   = (np.array([[ 0.25]]),
                      np.array([[  1.00000000e+00,  -8.27013523e-16,   8.57305616e-33, 1.95140273e-15],
                                [ -8.27013523e-16,   1.00000000e+00,   6.28036983e-16, -8.74760501e-31],
                                [  5.68444574e-33,  -6.28036983e-16,   1.00000000e+00, -2.84689309e-16],
                                [  1.95140273e-15,  -9.27538795e-31,   2.84689309e-16, 1.00000000e+00]]))
        self.assertArraysAlmostEqual(upperBound[0], expected[0])
        self.assertArraysAlmostEqual(upperBound[1], expected[1])

    def test_unitary_to_process_mx(self):
        identity  = np.identity(2)
        processMx = gatetools.unitary_to_process_mx(identity)
        self.assertArraysAlmostEqual(processMx, np.identity(4))

    def test_err_gen(self):
        gs_target = std2Q_XXYYII.gs_target
        gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)

        projectionTypes = ['hamiltonian', 'stochastic', 'affine']
        basisNames      = ['std', 'gm', 'pp'] #, 'qt'] #dim must == 3 for qt

        for (lbl,gateTarget), gate in zip(gs_target.gates.items(), gs_datagen.gates.values()):
            print("Gate %s" % lbl)
            errgen    = gatetools.error_generator(gate, gateTarget, gs_target.basis, 'logG-logT')
            altErrgen = gatetools.error_generator(gate, gateTarget, gs_target.basis, 'logTiG')
            altErrgen2 = gatetools.error_generator(gate, gateTarget, gs_target.basis, 'logGTi')            
            with self.assertRaises(ValueError):
                gatetools.error_generator(gate, gateTarget, gs_target.basis, 'adsf')

            for projectionType in projectionTypes:
                for basisName in basisNames:
                    gatetools.std_errgen_projections(errgen, projectionType, basisName)

            originalGate     = gatetools.gate_from_error_generator(errgen, gateTarget, 'logG-logT')
            altOriginalGate  = gatetools.gate_from_error_generator(altErrgen, gateTarget, 'logTiG')
            altOriginalGate2 = gatetools.gate_from_error_generator(altErrgen, gateTarget, 'logGTi')
            with self.assertRaises(ValueError):
                gatetools.gate_from_error_generator(errgen, gateTarget, 'adsf')
            #self.assertArraysAlmostEqual(originalGate, gate) # sometimes need to approximate the log for this one
            self.assertArraysAlmostEqual(altOriginalGate, gate)
            self.assertArraysAlmostEqual(altOriginalGate2, gate)

        #test odd cases:

        # when target is not unitary
        errgen_nonunitary = gatetools.error_generator(gs_datagen.gates['Gxi'], gs_datagen.gates['Gxi'],
                                                      gs_datagen.basis)
        # when target is not near gate
        errgen_notsmall = gatetools.error_generator(gs_datagen.gates['Gxi'], gs_target.gates['Gix'],
                                                    gs_target.basis, 'logTiG')
        errgen_notsmall = gatetools.error_generator(gs_datagen.gates['Gxi'], gs_target.gates['Gix'],
                                                    gs_target.basis, 'logGTi')

        with self.assertRaises(ValueError):
            gatetools.error_generator(gs_datagen.gates['Gxi'], gs_target.gates['Gxi'],
                                      gs_target.basis, 'foobar')

        #Check helper routine _assert_shape
        with self.assertRaises(NotImplementedError): #boundary case
            gatetools._assert_shape(np.zeros((2,2,2,2,2),'d'), (2,2,2,2,2),sparse=True) # ndims must be <= 4

        

    def test_std_errgens(self):
        projectionTypes = ['hamiltonian', 'stochastic','affine']
        basisNames      = ['std', 'gm', 'pp'] #, 'qt'] #dim must == 3 for qt
        
        for projectionType in projectionTypes:
            gatetools.std_scale_factor(4, projectionType)
            for basisName in basisNames:
                gatetools.std_error_generators(4, projectionType, basisName)

        with self.assertRaises(ValueError):
            gatetools.std_scale_factor(4, "foobar")
        with self.assertRaises(ValueError):
            gatetools.std_error_generators(4, "foobar", 'gm')

    def test_lind_errgens(self):
        basis = pygsti.obj.Basis('gm',2)

        normalize = False
        other_diagonal_only = False
        gatetools.lindblad_error_generators(basis, basis, normalize, other_diagonal_only)
        gatetools.lindblad_error_generators(None, basis, normalize, other_diagonal_only)
        gatetools.lindblad_error_generators(basis, None, normalize, other_diagonal_only)
        gatetools.lindblad_error_generators(None, None, normalize, other_diagonal_only)                

        normalize = True
        other_diagonal_only = False
        gatetools.lindblad_error_generators(basis, basis, normalize, other_diagonal_only)
        gatetools.lindblad_error_generators(None, basis, normalize, other_diagonal_only)
        gatetools.lindblad_error_generators(basis, None, normalize, other_diagonal_only)
        gatetools.lindblad_error_generators(None, None, normalize, other_diagonal_only)                

        normalize = True
        other_diagonal_only = True
        gatetools.lindblad_error_generators(basis, basis, normalize, other_diagonal_only)
        gatetools.lindblad_error_generators(None, basis, normalize, other_diagonal_only)
        gatetools.lindblad_error_generators(basis, None, normalize, other_diagonal_only)
        gatetools.lindblad_error_generators(None, None, normalize, other_diagonal_only)


        basis = pygsti.obj.Basis('gm',4)
        mxBasis = pygsti.obj.Basis('gm',4)
        errgen = np.identity(16,'d')
        gatetools.lindblad_errgen_projections(errgen, basis, basis, mxBasis, 
                                    normalize=True, return_generators=False, 
                                    other_diagonal_only=False, sparse=False)

        gatetools.lindblad_errgen_projections(errgen, None, 'gm', mxBasis, 
                                    normalize=True, return_generators=False, 
                                    other_diagonal_only=False, sparse=False)
        gatetools.lindblad_errgen_projections(errgen, 'gm', None, mxBasis, 
                                    normalize=True, return_generators=True, 
                                    other_diagonal_only=True, sparse=False)

        basisMxs = pygsti.tools.basis_matrices('gm', 4, sparse=False) 
        gatetools.lindblad_errgen_projections(errgen, basisMxs, basisMxs, mxBasis, 
                                    normalize=True, return_generators=False, 
                                    other_diagonal_only=False, sparse=False)

        gatetools.lindblad_errgen_projections(errgen, None, None, mxBasis, 
                                              normalize=True, return_generators=False, 
                                              other_diagonal_only=False, sparse=False)
                

    def test_project_gateset(self):
        projectionTypes=('H','S','H+S','LND', 'LNDF')
        gs_target = std2Q_XXYYII.gs_target.copy()
        gs = gs_target.depolarize(gate_noise=0.01)

        for genType in ("logG-logT", "logTiG", "logGTi"):
            proj_gateset, Np_dict = gatetools.project_gateset(
                gs, gs_target, projectionTypes, genType)

        with self.assertRaises(ValueError):
            gs_target_gm = std2Q_XXYYII.gs_target.copy()
            gs_target_gm.basis = pygsti.obj.Basis("gm",4)
            gatetools.project_gateset(
                gs, gs_target_gm, projectionTypes, genType) # basis mismatch


if __name__ == '__main__':
    unittest.main(verbosity=2)

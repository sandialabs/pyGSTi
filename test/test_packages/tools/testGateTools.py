from ..testutils import BaseTestCase, compare_files, temp_files

import pygsti
import pygsti.tools.gatetools as gatetools

from pygsti.construction import std2Q_XXYYII

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
        pass
        '''
        gs_target = std2Q_XXYYII.gs_target
        gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)

        projectionTypes = ['hamiltonian', 'stochastic']
        basisNames      = ['std', 'gm', 'pp', 'qt']

        for gateTarget, gate in zip(gs_target.gates.values(), gs_datagen.gates.values()):
            errgen    = gatetools.error_generator(gate, gateTarget, gs_target.basis)
            altErrgen = gatetools.error_generator(gate, gateTarget, gs_target.basis, 'logTiG')
            with self.assertRaises(ValueError):
                gatetools.error_generator(gate, gateTarget, gs_target.basis, 'adsf')

            #std_errgen_projections(errgen, projectionType, basisName)

            originalGate    = gatetools.gate_from_error_generator(errgen, gateTarget)
            altOriginalGate = gatetools.gate_from_error_generator(altErrgen, gateTarget, 'logTiG')
            with self.assertRaises(ValueError):
                gatetools.gate_from_error_generator(errgen, gateTarget, 'adsf')
        '''
        '''
        for projectionType in projectionTypes:
            for basisName in basisNames:
                std_error_generators(4, projectionType, basisName)
        '''

if __name__ == '__main__':
    unittest.main(verbosity=2)

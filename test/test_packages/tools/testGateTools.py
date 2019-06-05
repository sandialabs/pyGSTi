from ..testutils import BaseTestCase, compare_files, temp_files

import os
import pygsti
import pygsti.tools.optools as optools

from pygsti.construction import std2Q_XXYYII
from pygsti.construction import std1Q_XYI

import numpy as np
import unittest

SKIP_CVXPY = os.getenv('SKIP_CVXPY')

A = np.array( [[0.9, 0, 0.1j, 0],
               [ 0,  0, 0,    0],
               [ -0.1j, 0, 0, 0],
               [ 0,  0,  0,  0.1]], 'complex')

B = np.array( [[0.5, 0, 0, -0.2j],
               [ 0,  0.25, 0,  0],
               [ 0, 0, 0.25,   0],
               [ 0.2j,  0,  0,  0.1]], 'complex')

class GateBaseTestCase(BaseTestCase):
    def test_err_gen(self):
        target_model = std2Q_XXYYII.target_model()
        mdl_datagen = target_model.depolarize(op_noise=0.1, spam_noise=0.001)

        projectionTypes = ['hamiltonian', 'stochastic', 'affine']
        basisNames      = ['std', 'gm', 'pp'] #, 'qt'] #dim must == 3 for qt

        for (lbl,gateTarget), gate in zip(target_model.operations.items(), mdl_datagen.operations.values()):
            print("LinearOperator %s" % lbl)
            errgen    = optools.error_generator(gate, gateTarget, target_model.basis, 'logG-logT')
            altErrgen = optools.error_generator(gate, gateTarget, target_model.basis, 'logTiG')
            altErrgen2 = optools.error_generator(gate, gateTarget, target_model.basis, 'logGTi')
            with self.assertRaises(ValueError):
                optools.error_generator(gate, gateTarget, target_model.basis, 'adsf')

            for projectionType in projectionTypes:
                for basisName in basisNames:
                    optools.std_errgen_projections(errgen, projectionType, basisName)

            originalGate     = optools.operation_from_error_generator(errgen, gateTarget, 'logG-logT')
            altOriginalGate  = optools.operation_from_error_generator(altErrgen, gateTarget, 'logTiG')
            altOriginalGate2 = optools.operation_from_error_generator(altErrgen, gateTarget, 'logGTi')
            with self.assertRaises(ValueError):
                optools.operation_from_error_generator(errgen, gateTarget, 'adsf')
            #self.assertArraysAlmostEqual(originalGate, gate) # sometimes need to approximate the log for this one
            self.assertArraysAlmostEqual(altOriginalGate, gate)
            self.assertArraysAlmostEqual(altOriginalGate2, gate)

        #test odd cases:

        # when target is not unitary
        errgen_nonunitary = optools.error_generator(mdl_datagen.operations['Gxi'], mdl_datagen.operations['Gxi'],
                                                      mdl_datagen.basis)
        # when target is not near gate
        errgen_notsmall = optools.error_generator(mdl_datagen.operations['Gxi'], target_model.operations['Gix'],
                                                    target_model.basis, 'logTiG')
        errgen_notsmall = optools.error_generator(mdl_datagen.operations['Gxi'], target_model.operations['Gix'],
                                                    target_model.basis, 'logGTi')

        with self.assertRaises(ValueError):
            optools.error_generator(mdl_datagen.operations['Gxi'], target_model.operations['Gxi'],
                                      target_model.basis, 'foobar')

        #Check helper routine _assert_shape
        with self.assertRaises(NotImplementedError): #boundary case
            optools._assert_shape(np.zeros((2,2,2,2,2),'d'), (2,2,2,2,2),sparse=True) # ndims must be <= 4



    def test_std_errgens(self):
        projectionTypes = ['hamiltonian', 'stochastic','affine']
        basisNames      = ['std', 'gm', 'pp'] #, 'qt'] #dim must == 3 for qt

        for projectionType in projectionTypes:
            optools.std_scale_factor(4, projectionType)
            for basisName in basisNames:
                optools.std_error_generators(4, projectionType, basisName)

        with self.assertRaises(ValueError):
            optools.std_scale_factor(4, "foobar")
        with self.assertRaises(ValueError):
            optools.std_error_generators(4, "foobar", 'gm')

    def test_lind_errgens(self):
        basis = pygsti.obj.Basis.cast('gm',4)

        normalize = False
        other_mode = "all"
        optools.lindblad_error_generators(basis, basis, normalize, other_mode)
        optools.lindblad_error_generators(None, basis, normalize, other_mode)
        optools.lindblad_error_generators(basis, None, normalize, other_mode)
        optools.lindblad_error_generators(None, None, normalize, other_mode)

        normalize = True
        other_mode = "all"
        optools.lindblad_error_generators(basis, basis, normalize, other_mode)
        optools.lindblad_error_generators(None, basis, normalize, other_mode)
        optools.lindblad_error_generators(basis, None, normalize, other_mode)
        optools.lindblad_error_generators(None, None, normalize, other_mode)

        normalize = True
        other_mode = "diagonal"
        optools.lindblad_error_generators(basis, basis, normalize, other_mode)
        optools.lindblad_error_generators(None, basis, normalize, other_mode)
        optools.lindblad_error_generators(basis, None, normalize, other_mode)
        optools.lindblad_error_generators(None, None, normalize, other_mode)


        basis = pygsti.obj.Basis.cast('gm',16)
        mxBasis = pygsti.obj.Basis.cast('gm',16)
        errgen = np.identity(16,'d')
        optools.lindblad_errgen_projections(errgen, basis, basis, mxBasis,
                                    normalize=True, return_generators=False,
                                    other_mode="all", sparse=False)

        optools.lindblad_errgen_projections(errgen, None, 'gm', mxBasis,
                                    normalize=True, return_generators=False,
                                    other_mode="all", sparse=False)
        optools.lindblad_errgen_projections(errgen, 'gm', None, mxBasis,
                                    normalize=True, return_generators=True,
                                    other_mode="diagonal", sparse=False)

        basisMxs = pygsti.tools.basis_matrices('gm', 16, sparse=False)
        optools.lindblad_errgen_projections(errgen, basisMxs, basisMxs, mxBasis,
                                    normalize=True, return_generators=False,
                                    other_mode="all", sparse=False)

        optools.lindblad_errgen_projections(errgen, None, None, mxBasis,
                                              normalize=True, return_generators=False,
                                              other_mode="all", sparse=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)

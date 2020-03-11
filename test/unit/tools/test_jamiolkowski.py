import os
import numpy as np

from ..util import BaseCase


from pygsti.modelpacks.legacy import std1Q_XYI as std1Q
from pygsti.construction import build_operation
import pygsti.tools.basistools as bt
from pygsti.objects import ExplicitOpModel, Basis

from pygsti.tools import jamiolkowski as j

class JamiolkowskiBasisTester(BaseCase):
    def setUp(self):
        #Set Model objects to "strict" mode for testing
        ExplicitOpModel._strict = True

        # density matrix == 3x3 block diagonal matrix: a 2x2 block followed by a 1x1 block
        self.stateSpaceDims = [(4,), (1,)]
        self.std = Basis.cast('std', 9)
        self.gm = Basis.cast('gm', 9)
        self.stdSmall = Basis.cast('std', [4, 1])
        self.gmSmall = Basis.cast('gm', [4, 1])

        #labels which give a tensor product interp. for the states within each density matrix block
        self.stateSpaceLabels = [('Qhappy',), ('Lsad',)]

        #Build a test gate   -- old # X(pi,Qhappy)*LX(pi,0,2)
        self.testGate = build_operation(self.stateSpaceDims,
                                        self.stateSpaceLabels,
                                        "LX(pi,0,2)",
                                        "std")
        self.testGateGM_mx = bt.change_basis(self.testGate, self.stdSmall, self.gmSmall)
        self.expTestGate_mx = bt.flexible_change_basis(self.testGate, self.stdSmall, self.std)
        self.expTestGateGM_mx = bt.change_basis(self.expTestGate_mx, self.std, self.gm)

    def checkBasis(self, cmb):
        #Op with Jamio map on gate in std and gm bases
        Jmx1 = j.jamiolkowski_iso(self.testGate, op_mx_basis=self.stdSmall,
                                  choi_mx_basis=cmb)
        Jmx2 = j.jamiolkowski_iso(self.testGateGM_mx, op_mx_basis=self.gmSmall,
                                  choi_mx_basis=cmb)
        print("Jmx1.shape = ", Jmx1.shape)

        #Make sure these yield the same trace == 1 matrix
        self.assertArraysAlmostEqual(Jmx1, Jmx2)
        self.assertAlmostEqual(np.trace(Jmx1), 1.0)

        #Op on expanded gate in std and gm bases
        JmxExp1 = j.jamiolkowski_iso(self.expTestGate_mx, op_mx_basis=self.std, choi_mx_basis=cmb)
        JmxExp2 = j.jamiolkowski_iso(self.expTestGateGM_mx, op_mx_basis=self.gm, choi_mx_basis=cmb)
        print("JmxExp1.shape = ", JmxExp1.shape)

        #Make sure these are the same as operating on the contracted basis
        self.assertArraysAlmostEqual(Jmx1, JmxExp1)
        self.assertArraysAlmostEqual(Jmx1, JmxExp2)

        #Reverse transform should yield back the operation matrix
        revTestGate_mx = j.jamiolkowski_iso_inv(Jmx1, choi_mx_basis=cmb,
                                                op_mx_basis=self.gmSmall)
        self.assertArraysAlmostEqual(revTestGate_mx, self.testGateGM_mx)

        #Reverse transform without specifying stateSpaceDims, then contraction, should yield same result
        revExpTestGate_mx = j.jamiolkowski_iso_inv(Jmx1, choi_mx_basis=cmb, op_mx_basis=self.std)
        self.assertArraysAlmostEqual(bt.resize_std_mx(revExpTestGate_mx, 'contract', self.std, self.stdSmall),
                                     self.testGate)

    def test_std_basis(self):
        #mx_dim = sum([ int(np.sqrt(d)) for d in ])
        cmb = Basis.cast('std', self.stateSpaceDims)
        self.checkBasis(cmb)

    def test_gm_basis(self):
        #mx_dim = sum([ int(np.sqrt(d)) for d in self.stateSpaceDims])
        cmb = Basis.cast('gm', self.stateSpaceDims)
        self.checkBasis(cmb)


class JamiolkowskiOpsTester(BaseCase):
    def setUp(self):
        self.gm = Basis.cast('gm', 4)
        self.pp = Basis.cast('pp', 4)
        self.std = Basis.cast('std', 4)
        self.mxGM = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0,-1, 0, 0],
                              [0, 0, 0, 1]], 'complex')

        self.mxStd = bt.change_basis(self.mxGM, self.gm, self.std)
        self.mxPP = bt.change_basis(self.mxGM, self.gm, self.pp)

    def test_sum_of_negative_choi_evals(self):
        sumOfNeg = j.sum_of_negative_choi_evals(std1Q.target_model())
        self.assertAlmostEqual(sumOfNeg, 0.0)

        sumOfNegWt = j.sum_of_negative_choi_evals(std1Q.target_model(), {'Gx': 1.0, 'Gy': 0.5})
        # TODO assert correctness

        sumsOfNeg = j.sums_of_negative_choi_evals(std1Q.target_model())
        self.assertArraysAlmostEqual(sumsOfNeg, np.zeros(3, 'd'))  # 3 gates in std.target_model()

        magsOfNeg = j.mags_of_negative_choi_evals(std1Q.target_model())
        self.assertArraysAlmostEqual(magsOfNeg, np.zeros(12, 'd'))  # 3 gates * 4 evals each = 12

    def test_fast_jamiolkowski_iso(self):
        fastChoiStd = j.fast_jamiolkowski_iso_std(self.mxStd, self.std)
        fastChoiStd2 = j.fast_jamiolkowski_iso_std(self.mxGM, self.gm)
        fastChoiStd3 = j.fast_jamiolkowski_iso_std(self.mxPP, self.pp)

        # TODO assert correctness
        self.assertArraysAlmostEqual(fastChoiStd, fastChoiStd2)
        self.assertArraysAlmostEqual(fastChoiStd, fastChoiStd3)

        fastGateStd = j.fast_jamiolkowski_iso_std_inv(fastChoiStd, self.std)
        fastGateGM = j.fast_jamiolkowski_iso_std_inv(fastChoiStd, self.gm)
        fastGatePP = j.fast_jamiolkowski_iso_std_inv(fastChoiStd, self.pp)

        # TODO assert correctness
        self.assertArraysAlmostEqual(fastGateStd, self.mxStd)
        self.assertArraysAlmostEqual(fastGateGM, self.mxGM)
        self.assertArraysAlmostEqual(fastGatePP, self.mxPP)

    def test_jamiolkowski_iso(self):
        choiStd = j.jamiolkowski_iso(self.mxStd, self.std, self.std)
        choiStd2 = j.jamiolkowski_iso(self.mxGM, self.gm, self.std)
        choiStd3 = j.jamiolkowski_iso(self.mxPP, self.pp, self.std)

        choiGM = j.jamiolkowski_iso(self.mxStd, self.std, self.gm)
        choiGM2 = j.jamiolkowski_iso(self.mxGM, self.gm, self.gm)
        choiGM3 = j.jamiolkowski_iso(self.mxPP, self.pp, self.gm)

        choiPP = j.jamiolkowski_iso(self.mxStd, self.std, self.pp)
        choiPP2 = j.jamiolkowski_iso(self.mxGM, self.gm, self.pp)
        choiPP3 = j.jamiolkowski_iso(self.mxPP, self.pp, self.pp)

        # TODO assert correctness
        self.assertArraysAlmostEqual(choiStd, choiStd2)
        self.assertArraysAlmostEqual(choiStd, choiStd3)
        self.assertArraysAlmostEqual(choiGM, choiGM2)
        self.assertArraysAlmostEqual(choiGM, choiGM3)
        self.assertArraysAlmostEqual(choiPP, choiPP2)
        self.assertArraysAlmostEqual(choiPP, choiPP3)

        gateStd = j.jamiolkowski_iso_inv(choiStd, self.std, self.std)
        gateStd2 = j.jamiolkowski_iso_inv(choiGM, self.gm, self.std)
        gateStd3 = j.jamiolkowski_iso_inv(choiPP, self.pp, self.std)

        gateGM = j.jamiolkowski_iso_inv(choiStd, self.std, self.gm)
        gateGM2 = j.jamiolkowski_iso_inv(choiGM, self.gm, self.gm)
        gateGM3 = j.jamiolkowski_iso_inv(choiPP, self.pp, self.gm)

        gatePP = j.jamiolkowski_iso_inv(choiStd, self.std, self.pp)
        gatePP2 = j.jamiolkowski_iso_inv(choiGM, self.gm, self.pp)
        gatePP3 = j.jamiolkowski_iso_inv(choiPP, self.pp, self.pp)

        # TODO assert correctness
        self.assertArraysAlmostEqual(gateStd, self.mxStd)
        self.assertArraysAlmostEqual(gateStd2, self.mxStd)
        self.assertArraysAlmostEqual(gateStd3, self.mxStd)

        self.assertArraysAlmostEqual(gateGM, self.mxGM)
        self.assertArraysAlmostEqual(gateGM2, self.mxGM)
        self.assertArraysAlmostEqual(gateGM3, self.mxGM)

        self.assertArraysAlmostEqual(gatePP, self.mxPP)
        self.assertArraysAlmostEqual(gatePP2, self.mxPP)
        self.assertArraysAlmostEqual(gatePP3, self.mxPP)

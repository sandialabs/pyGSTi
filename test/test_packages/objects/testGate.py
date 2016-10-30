import unittest
import pygsti
import numpy as np

from  pygsti.objects import Gate
import pygsti.construction as pc

from ..testutils import BaseTestCase, compare_files, temp_files

class GateTestCase(BaseTestCase):

    def setUp(self):
        super(GateTestCase, self).setUp()
        self.gate = Gate(np.zeros(2))

    def test_slice(self):
        self.gate[:]

    def test_bad_mx(self):
        bad_mxs = ['akdjsfaksdf',
                  [[], [1, 2]],
                  [[[]], [[1, 2]]]]
        for bad_mx in bad_mxs:
            with self.assertRaises(ValueError):
                Gate.convert_to_matrix(bad_mx)

    def test_lpg_deriv(self):
        gs_target_lp = pc.build_gateset(
            [2], [('Q0',)],['Gi','Gx'], [ "D(Q0)","X(pi/2,Q0)" ],
            prepLabels = ["rho0"], prepExpressions=["0"],
            effectLabels = ["E0"], effectExpressions=["0"], 
            spamdefs={'up': ("rho0","E0"), 'dn': ("rho0","remainder") },
            basis="pp", parameterization="linear" )

    
        gs_target_lp2 = pc.build_gateset(
            [4], [('Q0','Q1')],['Gix','Giy','Gxi','Gyi','Gcnot'], 
            [ "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CX(pi,Q0,Q1)" ],
            prepLabels = ["rho0"], prepExpressions=["0"],
            effectLabels = ["E0","E1","E2"], effectExpressions=["0","1","2"], 
            spamdefs={'upup': ("rho0","E0"), 'updn': ("rho0","E1"),
                      'dnup': ("rho0","E2"), 'dndn': ("rho0","remainder") },
            basis="pp", parameterization="linearTP" )

        gs_target_lp2.preps['rho0'] = pygsti.objects.TPParameterizedSPAMVec(gs_target_lp2['rho0'])
        #because there's no easy way to specify this TP parameterization...
        # (above is leftover from a longer feature test)
        
        check = pygsti.objects.profiler.check_deriv_wrt_params
        testDeriv = check(gs_target_lp['Gi'])
        testDeriv = check(gs_target_lp['Gx'])
        testDeriv = check(gs_target_lp2['Gix'])
        testDeriv = check(gs_target_lp2['Giy'])
        testDeriv = check(gs_target_lp2['Gxi'])
        testDeriv = check(gs_target_lp2['Gyi'])
        testDeriv = check(gs_target_lp2['Gcnot'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

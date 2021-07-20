import unittest

import numpy as np

import pygsti
from pygsti.models import modelconstruction
from ..testutils import BaseTestCase


class TestGateSetConstructionMethods(BaseTestCase):

    def setUp(self):
        super(TestGateSetConstructionMethods, self).setUp()

        #OK for these tests, since we test user interface?
        #Set Model objects to "strict" mode for testing
        pygsti.models.ExplicitOpModel._strict = False


    def test_build_gatesets(self):
        SQ2 = 1/np.sqrt(2)
        for defParamType in ("full", "full TP", "static"):
            gateset_simple = pygsti.models.ExplicitOpModel(['Q0'], 'pp', defParamType)
            gateset_simple['rho0'] = [SQ2, 0, 0, SQ2]
            gateset_simple['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM([('0', [SQ2, 0, 0, -SQ2])], evotype='default')
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

        gateset_badDefParam = pygsti.models.ExplicitOpModel(['Q0'], "pp", "full")
        gateset_badDefParam.preps.default_param = "foobar"
        gateset_badDefParam.operations.default_param = "foobar"
        with self.assertRaises(ValueError):
            gateset_badDefParam['rho0'] = [1, 0, 0, 0]
        with self.assertRaises(ValueError):
            gateset_badDefParam['Gi'] = np.identity(4,'d')

        stateSpace = [(4,)]  # density matrix is a 2x2 matrix
        spaceLabels = [('Q0',)]  # interpret the 2x2 density matrix as a single qubit named 'Q0'

        with self.assertRaises(AssertionError):
            modelconstruction._create_identity_vec(stateSpace, basis="foobar")


        gateset_povm_first = pygsti.models.ExplicitOpModel(['Q0']) #set effect vector first
        gateset_povm_first['Mdefault'] = pygsti.modelmembers.povms.TPPOVM(
            [('0', modelconstruction._create_spam_vector(stateSpace, spaceLabels, "0")),
             ('1', modelconstruction._create_spam_vector(stateSpace, spaceLabels, "1"))], evotype='default' )

        with self.assertRaises(ValueError):
            gateset_povm_first['rhoBad'] =  np.array([1,2,3],'d') #wrong dimension
        with self.assertRaises(ValueError):
            gateset_povm_first['Mdefault'] =  pygsti.modelmembers.povms.UnconstrainedPOVM(
                [('0', np.array([1, 2, 3], 'd'))], evotype='default',
                state_space=pygsti.baseobjs.StateSpace.cast([('L0',),('L1',),('L2',)])) #wrong dimension


if __name__ == "__main__":
    unittest.main(verbosity=2)

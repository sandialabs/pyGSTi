from ..testutils import BaseTestCase, compare_files, temp_files

import unittest
import pygsti
import pygsti.construction as pc
import numpy as np
import scipy

class ModelConstructionTestCase(BaseTestCase):

    def setUp(self):
        super(ModelConstructionTestCase, self).setUp()

    def test_explicit(self):
        model = pc.build_explicit_model( [('Q0',)],
                                         ['Gi','Gx','Gy'], [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"])
        self.assertEqual(set(model.operations.keys()), set(['Gi','Gx','Gy']))
        self.assertAlmostEqual(sum(model.probs( ('Gx','Gi','Gy')).values()), 1.0)
        self.assertEqual(model.num_params(), 60)

    def test_indep_localnoise(self):
        nQubits = 2
        mdl_local = pygsti.obj.LocalNoiseModel.build_from_parameterization(
            nQubits, ('Gx','Gy','Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='H+S', independent_gates=True,
            ensure_composed_gates=False, global_idle=None)

        assert( set(mdl_local.operation_blks['gates'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        assert( set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        test_circuit = ( [('Gx','qb0'),('Gy','qb1')], ('Gcnot', 'qb0', 'qb1'), [('Gx','qb1'), ('Gy','qb0')])
        self.assertAlmostEqual(sum(mdl_local.probs(test_circuit).values()), 1.0)
        self.assertEqual(mdl_local.num_params(), 108)

    def test_dep_localnoise(self):
        nQubits = 2
        mdl_local = pygsti.obj.LocalNoiseModel.build_from_parameterization(
            nQubits, ('Gx','Gy','Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='H+S', independent_gates=False,
            ensure_composed_gates=False, global_idle=None)

        assert( set(mdl_local.operation_blks['gates'].keys()) == set(["Gx","Gy","Gcnot"]))
        assert( set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')]))
        test_circuit = ( [('Gx','qb0'),('Gy','qb1')], ('Gcnot', 'qb0', 'qb1'), [('Gx','qb1'), ('Gy','qb0')])
        self.assertAlmostEqual(sum(mdl_local.probs(test_circuit).values()), 1.0)
        self.assertEqual(mdl_local.num_params(), 66)

    def test_localnoise_with1Qidle(self):
        nQubits = 2
        noisy_idle = np.array([[1,0,0,0],
                               [0,0.9,0,0],
                               [0,0,0.9,0],
                               [0,0,0,0.9]], 'd')

        mdl_local = pygsti.obj.LocalNoiseModel.build_from_parameterization(
            nQubits, ('Gx','Gy','Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='static', independent_gates=False,
            ensure_composed_gates=False, global_idle=noisy_idle)

        assert( set(mdl_local.operation_blks['gates'].keys()) == set(["Gx","Gy","Gcnot","1QIdle"]))
        assert( set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0'), 'globalIdle']))
        test_circuit = ( ('Gx','qb0'), ('Gcnot', 'qb0', 'qb1'), [], [('Gx','qb1'), ('Gy','qb0')])
        self.assertAlmostEqual(sum(mdl_local.probs(test_circuit).values()), 1.0)
        self.assertAlmostEqual(mdl_local.probs(test_circuit)['00'], 0.3576168)
        self.assertEqual(mdl_local.num_params(), 0)

    def test_localnoise_withNQidle(self):
        nQubits = 2
        noisy_idle = 0.9 * np.identity(4**nQubits,'d')
        noisy_idle[0,0] = 1.0

        mdl_local = pygsti.obj.LocalNoiseModel.build_from_parameterization(
            nQubits, ('Gx','Gy','Gcnot'), geometry="line",
            qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            parameterization='H+S+A', independent_gates=False,
            ensure_composed_gates=False, global_idle=noisy_idle)

        assert( set(mdl_local.operation_blks['gates'].keys()) == set(["Gx","Gy","Gcnot"]))
        assert( set(mdl_local.operation_blks['layers'].keys()) == set(
            [('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'), ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0'), 'globalIdle']))
        test_circuit = ( ('Gx','qb0'), ('Gcnot', 'qb0', 'qb1'), [], [('Gx','qb1'), ('Gy','qb0')])
        self.assertAlmostEqual(sum(mdl_local.probs(test_circuit).values()), 1.0)
        self.assertAlmostEqual(mdl_local.probs(test_circuit)['00'], 0.414025)
        self.assertEqual(mdl_local.num_params(), 144)

    def test_crosstalk_free_buildup(self):
        nQubits = 2

        mdl = pygsti.construction.build_crosstalk_free_model(nQubits, ('Gi','Gx','Gy','Gcnot'),
                                                               {}, ensure_composed_gates=True,
                                                               independent_gates=False)
        assert( set(mdl.operation_blks['gates'].keys()) == set(["Gi","Gx","Gy","Gcnot"]))
        assert( set(mdl.operation_blks['layers'].keys()) == set(
            [('Gi', 0), ('Gi', 1), ('Gx', 0), ('Gx', 1), ('Gy', 0), ('Gy', 1), ('Gcnot', 0, 1), ('Gcnot', 1, 0)]))
        self.assertEqual(mdl.num_params(), 0)

        addlErr = pygsti.obj.TPDenseOp(np.identity(4, 'd'))  # adds 12 params
        addlErr2 = pygsti.obj.TPDenseOp(np.identity(4, 'd'))  # adds 12 params

        mdl.operation_blks['gates']['Gx'].append(addlErr)
        mdl.operation_blks['gates']['Gy'].append(addlErr2)
        mdl.operation_blks['gates']['Gi'].append(addlErr)

        self.assertEqual(mdl.num_params(), 24)

        self.assertEqual(mdl.operation_blks['layers'][('Gx', 0)].gpindices, slice(0, 12))
        self.assertEqual(mdl.operation_blks['layers'][('Gy', 0)].gpindices, slice(12, 24))
        self.assertEqual(mdl.operation_blks['layers'][('Gi', 0)].gpindices, slice(0, 12))
        self.assertEqual(mdl.operation_blks['gates']['Gx'].gpindices, slice(0, 12))
        self.assertEqual(mdl.operation_blks['gates']['Gy'].gpindices, slice(12, 24))
        self.assertEqual(mdl.operation_blks['gates']['Gi'].gpindices, slice(0, 12))

    def test_crosstalk_free(self):
        nQubits = 2
        
        # Case: ensure_composed_gates=False, independent_gates=True
        cfmdl = pygsti.construction.build_crosstalk_free_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            {'Gx': 0.1,  # depol
             'Gy': (0.02, 0.02, 0.02),  # pauli stochastic
             'Gcnot': {('H','ZZ'): 0.01, ('S','IX'): 0.01},  # errgen: BUG? when SIX too large -> no coeff corresponding to rate?
             'idle': 0.01, 'prep': 0.01, 'povm': 0.01
            }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            ensure_composed_gates=False, independent_gates=True)

        self.assertEqual(cfmdl.num_params(), 17)

        # Case: ensure_composed_gates=True, independent_gates=False
        cfmdl2 = pygsti.construction.build_crosstalk_free_model(
            nQubits, ('Gx', 'Gy', 'Gcnot'),
            {'Gx': 0.1,  # depol
             'Gy': (0.02, 0.02, 0.02),  # pauli stochastic
             'Gcnot': {'HZZ': 0.01, 'SIX': 0.01},  # errgen: BUG? when SIX too large -> no coeff corresponding to rate?
             'idle': 0.01, 'prep': 0.01, 'povm': 0.01
            }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            ensure_composed_gates=True, independent_gates=False)
        self.assertEqual(cfmdl2.num_params(), 11)

        # Same as above but add ('Gx','qb0') to test giving qubit-specific error rates
        cfmdl3 = pygsti.construction.build_crosstalk_free_model(
            nQubits, ('Gx','Gy','Gcnot'), 
            {'Gx': 0.1,  #depol
             ('Gx','qb0'): 0.2, # adds another independent depol param for Gx:qb0
             'Gy': (0.02,0.02,0.02), # pauli stochastic 
             'Gcnot': {'HZZ': 0.01, 'SIX': 0.01}, #errgen: BUG? when SIX too large -> no coeff corresponding to rate?
             'idle': 0.01, 'prep': 0.01, 'povm': 0.01
            }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)],
            ensure_composed_gates=True, independent_gates=False)
        self.assertEqual(cfmdl3.num_params(), 12)
        

    def test_cloud_crosstalk(self):
        nQubits = 2
        ccmdl1 = pygsti.construction.build_cloud_crosstalk_model(
            nQubits, ('Gx','Gy','Gcnot'), 
            { ('Gx','qb0'): { ('H','X'): 0.01, ('S','XY:qb0,qb1'): 0.01},
              ('Gcnot','qb0','qb1'): { ('H','ZZ'): 0.02, ('S','XX:qb0,qb1'): 0.02 },
              'idle': { ('S','XX:qb0,qb1'): 0.01 },
              'prep': { ('S','XX:qb0,qb1'): 0.01 },
              'povm': { ('S','XX:qb0,qb1'): 0.01 }
            }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
        self.assertEqual(ccmdl1.num_params(), 7)

        #Using compact notation:
        ccmdl2 = pygsti.construction.build_cloud_crosstalk_model(
            nQubits, ('Gx','Gy','Gcnot'), 
            { 'Gx:0': { ('HX'): 0.01, 'SXY:0,1': 0.01},
              'Gcnot:0:1': { 'HZZ': 0.02, 'SXX:0,1': 0.02 },
              'idle': { 'SXX:0,1': 0.01 },
              'prep': { 'SXX:0,1': 0.01 },
              'povm': { 'SXX:0,1': 0.01 }
            })
        self.assertEqual(ccmdl2.num_params(), 7)

        #also using qubit_labels
        ccmdl3 = pygsti.construction.build_cloud_crosstalk_model(
            nQubits, ('Gx','Gy','Gcnot'), 
            { 'Gx:qb0': { ('HX'): 0.01, 'SXY:qb0,qb1': 0.01},
              'Gcnot:qb0:qb1': { 'HZZ': 0.02, 'SXX:qb0,qb1': 0.02 },
              'idle': { 'SXX:qb0,qb1': 0.01 },
              'prep': { 'SXX:qb0,qb1': 0.01 },
              'povm': { 'SXX:qb0,qb1': 0.01 }
            }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
        self.assertEqual(ccmdl3.num_params(), 7)

    def test_cloud_crosstalk_stencils(self):
        nQubits = 2
        ccmdl1 = pygsti.construction.build_cloud_crosstalk_model(
            nQubits, ('Gx','Gy','Gcnot'), 
            { 'Gx': { ('H','X'): 0.01, ('S','X:@0+left'): 0.01}, #('S','XX:@1+right,@0+left'): 0.02
              'Gcnot': { ('H','ZZ'): 0.02, ('S','XX:@1+right,@0+left'): 0.02 },
              'idle': { ('S','XX:qb0,qb1'): 0.01 }
            }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
        self.assertEqual(ccmdl1.num_params(), 5)

        #Using compact notation:
        ccmdl2 = pygsti.construction.build_cloud_crosstalk_model(
            nQubits, ('Gx','Gy','Gcnot'), 
            { 'Gx': { 'HX': 0.01, 'SX:@0+left': 0.01}, #('S','XX:@1+right,@0+left'): 0.02
              'Gcnot': { 'HZZ': 0.02, 'SXX:@1+right,@0+left': 0.02 },
              'idle': { 'SXX:qb0,qb1': 0.01 }
            }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)])
        self.assertEqual(ccmdl2.num_params(), 5)

    def test_cloud_crosstalk_indepgates(self):
        #Same as test_cloud_crosstalk_stencils case but set independent_gates=True
        nQubits = 2
        ccmdl1 = pygsti.construction.build_cloud_crosstalk_model(
            nQubits, ('Gx','Gy','Gcnot'), 
            { 'Gx': { ('H','X'): 0.01, ('S','X:@0+left'): 0.01}, #('S','XX:@1+right,@0+left'): 0.02
              'Gcnot': { ('H','ZZ'): 0.02, ('S','XX:@1+right,@0+left'): 0.02 },
              'idle': { ('S','XX:qb0,qb1'): 0.01 }
            }, qubit_labels=['qb{}'.format(i) for i in range(nQubits)], independent_gates=True)
        self.assertEqual(ccmdl1.num_params(), 8)

    def test_factories(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            a, = args
            sigmaZ = np.array([[1, 0], [0, -1]], 'd')
            return scipy.linalg.expm(1j * float(a) * sigmaZ)

        cfmdl = pygsti.construction.build_crosstalk_free_model(nQubits, ('Gx','Gy','Gcnot','Ga'),
                                                               {}, nonstd_gate_unitaries={'Ga': fn})
        ccmdl = pygsti.construction.build_cloud_crosstalk_model(nQubits, ('Gx','Gy','Gcnot','Ga'),
                                                                {}, nonstd_gate_unitaries={'Ga': fn})
        ps = pygsti.obj.ProcessorSpec(nQubits, ('Gx','Gy','Gcnot','Ga'), nonstd_gate_unitaries={'Ga': fn})

        c = pygsti.obj.Circuit("Gx:1Ga;0.3:1Gx:1")
        p1 = ccmdl.probs(c)
        p2 = cfmdl.probs(c)
        p3 = ps.models['target'].probs(c)

        for p in (p1, p2, p3):
            self.assertAlmostEqual(p['00'], 0.08733219254516078)
            self.assertAlmostEqual(p['01'], 0.9126678074548386)

        c2 = pygsti.obj.Circuit("Gx:1Ga;0.78539816:1Gx:1")  # a clifford: 0.78539816 = pi/4
        p4 = ps.models['clifford'].probs(c2)
        self.assertAlmostEqual(p4['00'], 0.5)
        self.assertAlmostEqual(p4['01'], 0.5)

if __name__ == '__main__':
    unittest.main(verbosity=2)

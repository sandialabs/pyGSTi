import numpy as np

from ..util import BaseCase

from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.construction.modelconstruction import create_explicit_model, _create_operation
import pygsti.objects.explicitmodel as mdl


class ExplicitOpModelStrictAccessTester(BaseCase):
    def setUp(self):
        mdl.ExplicitOpModel._strict = True
        self.model = std.target_model().randomize_with_unitary(0.001, seed=1234)

    def test_strict_access(self):
        #test strict mode, which forbids all these accesses
        with self.assertRaises(KeyError):
            self.model['identity'] = [1, 0, 0, 0]
        with self.assertRaises(KeyError):
            self.model['Gx'] = np.identity(4, 'd')
        with self.assertRaises(KeyError):
            self.model['E0'] = [1, 0, 0, 0]
        with self.assertRaises(KeyError):
            self.model['rho0'] = [1, 0, 0, 0]

        with self.assertRaises(KeyError):
            self.model['identity']
        with self.assertRaises(KeyError):
            self.model['Gx']
        with self.assertRaises(KeyError):
            self.model['E0']
        with self.assertRaises(KeyError):
            self.model['rho0']


class ExplicitOpModelToolTester(BaseCase):
    def setUp(self):
        mdl.ExplicitOpModel._strict = False
        # XXX can these be constructed directly?  EGN: yes, some model-construction tests should do it.
        self.model = create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                          ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])

        self.gateset_2q = create_explicit_model(
            [('Q0', 'Q1')], ['GIX', 'GIY', 'GXI', 'GYI', 'GCNOT'],
            ["I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CX(pi,Q0,Q1)"])

    def test_randomize_with_unitary(self):
        gateset_randu = self.model.randomize_with_unitary(0.01)
        gateset_randu = self.model.randomize_with_unitary(0.01, seed=1234)
        # TODO assert correctness

    def test_rotate_1q(self):
        rotXPi = _create_operation([(4,)], [('Q0',)], "X(pi,Q0)")
        rotXPiOv2 = _create_operation([(4,)], [('Q0',)], "X(pi/2,Q0)")
        rotYPiOv2 = _create_operation([(4,)], [('Q0',)], "Y(pi/2,Q0)")
        gateset_rot = self.model.rotate((np.pi / 2, 0, 0))  # rotate all gates by pi/2 about X axis
        self.assertArraysAlmostEqual(gateset_rot['Gi'], rotXPiOv2)
        self.assertArraysAlmostEqual(gateset_rot['Gx'], rotXPi)
        self.assertArraysAlmostEqual(gateset_rot['Gx'], np.dot(rotXPiOv2, rotXPiOv2))
        self.assertArraysAlmostEqual(gateset_rot['Gy'], np.dot(rotXPiOv2, rotYPiOv2))

    def test_rotate_2q(self):
        gateset_2q_rot = self.gateset_2q.rotate(rotate=list(np.zeros(15, 'd')))
        gateset_2q_rot_same = self.gateset_2q.rotate(rotate=(0.01,) * 15)
        gateset_2q_randu = self.gateset_2q.randomize_with_unitary(0.01)
        gateset_2q_randu = self.gateset_2q.randomize_with_unitary(0.01, seed=1234)
        # TODO assert correctness

    def test_depolarize(self):
        Gi_dep = np.array([[1, 0, 0, 0],
                           [0, 0.9, 0, 0],
                           [0, 0, 0.9, 0],
                           [0, 0, 0, 0.9]], 'd')
        Gx_dep = np.array([[1, 0, 0, 0],
                           [0, 0.9, 0, 0],
                           [0, 0, 0, -0.9],
                           [0, 0, 0.9, 0]], 'd')
        Gy_dep = np.array([[1, 0, 0, 0],
                           [0, 0, 0, 0.9],
                           [0, 0, 0.9, 0],
                           [0, -0.9, 0, 0]], 'd')
        gateset_dep = self.model.depolarize(op_noise=0.1)
        self.assertArraysAlmostEqual(gateset_dep['Gi'], Gi_dep)
        self.assertArraysAlmostEqual(gateset_dep['Gx'], Gx_dep)
        self.assertArraysAlmostEqual(gateset_dep['Gy'], Gy_dep)

    def test_depolarize_with_spam_noise(self):
        gateset_spam = self.model.depolarize(spam_noise=0.1)
        self.assertAlmostEqual(float(np.dot(self.model['Mdefault']['0'].T, self.model['rho0'])), 1.0)
        # Since np.ndarray doesn't implement __round__... (assertAlmostEqual() doesn't work)
        # Compare the single element dot product result to 0.095 instead (coverting the array's contents ([[ 0.095 ]]) to a **python** float (0.095))
        # print("DEBUG gateset_spam = ")
        # print(gateset_spam['Mdefault']['0'].T)
        # print(gateset_spam['rho0'].T)
        # print(gateset_spam)
        # print(gateset_spam['Mdefault']['0'].T)
        # print(gateset_spam['rho0'].T)
        # not 0.905 b/c effecs aren't depolarized now
        self.assertAlmostEqual(np.dot(gateset_spam['Mdefault']['0'].T, gateset_spam['rho0']).reshape(-1,)[0], 0.95)
        self.assertArraysAlmostEqual(gateset_spam['rho0'], 1 / np.sqrt(2) * np.array([1, 0, 0, 0.9]).reshape(-1, 1))
        #self.assertArraysAlmostEqual(gateset_spam['Mdefault']['0'], 1/np.sqrt(2)*np.array([1,0,0,0.9]).reshape(-1,1) ) #not depolarized now
        self.assertArraysAlmostEqual(gateset_spam['Mdefault']['0'], 1 / np.sqrt(2)
                                     * np.array([1, 0, 0, 1]).reshape(-1, 1))  # not depolarized now

    def test_random_rotate_1q(self):
        gateset_rand_rot = self.model.rotate(max_rotate=0.2)
        gateset_rand_rot = self.model.rotate(max_rotate=0.2, seed=1234)

    def test_random_rotate_2q(self):
        gateset_2q_rand_rot = self.gateset_2q.rotate(max_rotate=0.2)
        gateset_2q_rand_rot = self.gateset_2q.rotate(max_rotate=0.2, seed=1234)
        # TODO assert correctness

    def test_random_depolarize(self):
        gateset_rand_dep = self.model.depolarize(max_op_noise=0.1)
        gateset_rand_dep = self.model.depolarize(max_op_noise=0.1, seed=1234)
        # TODO assert correctness

    def test_random_depolarize_with_spam_noise(self):
        gateset_rand_spam = self.model.depolarize(max_spam_noise=0.1)
        gateset_rand_spam = self.model.depolarize(max_spam_noise=0.1, seed=1234)
        # TODO assert correctness

    def test_rotate_raises_on_bad_arg_spec(self):
        with self.assertRaises(ValueError):
            self.model.rotate(rotate=(0.2,) * 3, max_rotate=0.2)  # can't specify both
        with self.assertRaises(ValueError):
            self.model.rotate()  # must specify rotate or max_rotate
        with self.assertRaises(ValueError):
            self.gateset_2q.rotate(rotate=(0.2,) * 15, max_rotate=0.2)  # can't specify both
        with self.assertRaises(ValueError):
            self.gateset_2q.rotate()  # must specify rotate or max_rotate

    def test_rotate_raises_on_bad_dim(self):
        with self.assertRaises(AssertionError):
            self.model.rotate((1, 2, 3, 4))  # tuple must be length 3
        with self.assertRaises(AssertionError):
            self.gateset_2q.rotate(rotate=(0, 0, 0))  # wrong dimension model
        with self.assertRaises(AssertionError):
            self.gateset_2q.rotate((1, 2, 3, 4))  # tuple must be length 15
        with self.assertRaises(AssertionError):
            self.model.rotate(rotate=np.zeros(15, 'd'))  # wrong dimension model

    def test_rotate_raises_on_bad_type(self):
        with self.assertRaises(AssertionError):
            self.model.rotate("a string!")  # must be a 3-tuple
        with self.assertRaises(AssertionError):
            self.gateset_2q.rotate("a string!")  # must be a 15-tuple

    def test_depolarize_raises_on_bad_arg_spec(self):
        with self.assertRaises(ValueError):
            self.model.depolarize(op_noise=0.1, max_op_noise=0.1, spam_noise=0)  # can't specify both
        with self.assertRaises(ValueError):
            self.model.depolarize(spam_noise=0.1, max_spam_noise=0.1)  # can't specify both

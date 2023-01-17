import pickle

import sys
import numpy as np
try:
    import pygsti.evotypes.densitymx.statereps
except ModuleNotFoundError:
    print('importing pygsti.evotypes.densitymx.statereps failed. ModuleNotFoundError raised.') 
from pygsti.modelpacks import smq1Q_XYI
from pygsti.baseobjs import QubitSpace, Basis
from pygsti.modelmembers.operations import StochasticNoiseOp
from pygsti.circuits import Circuit
from pygsti.models import create_explicit_model
from pygsti.modelmembers.operations.composedop import ComposedOp
from pygsti.modelmembers.operations.staticunitaryop import StaticUnitaryOp
from pygsti.modelmembers.operations.fulltpop import FullTPOp
from pygsti.modelmembers.operations import FullUnitaryOp, FullArbitraryOp
from pygsti.forwardsims import WeakForwardSimulator, MapForwardSimulator
from pygsti.tools import create_elementary_errorgen, change_basis, unitary_to_superop

from ..util import BaseCase, needs_cvxpy


class KrausInterfaceTester(BaseCase):
    def test_stochastic_op(self):
        ss = QubitSpace(1)
        op = StochasticNoiseOp(ss, initial_rates=[0.01, 0.04, 0.16])

        expected_kops = [
            np.array([[0.888819, 0.      ],
                      [0.      , 0.888819]]),
            np.array([[0.0, 0.1],
                      [0.1, 0.0]]),
            np.array([[0.0, -0.2j],
                      [+0.2j, 0.0]]),
            np.array([[ 0.4,  0.],
                      [ 0. , -0.4]])
            ]
        for i, kop in enumerate(op.kraus_operators):
            print(np.round(kop, 3))
            self.assertArraysAlmostEqual(kop, expected_kops[i], places=5)

        #check that sum(K Kdag) == I
        kkdag = [kop @ kop.conjugate().T for kop in op.kraus_operators]
        self.assertArraysAlmostEqual(sum(kkdag), np.identity(2))

    def test_unitary_op(self):
        mdl = smq1Q_XYI.target_model('static unitary')
        #mdl.basis  # should be a BuiltinBasis

        op = mdl.operations['Gxpi2', 0]
        self.assertTrue(isinstance(op, StaticUnitaryOp))

        kraus_ops = [k.copy() for k in op.kraus_operators]  # make sure we copy
        self.assertEqual(len(kraus_ops), op.num_kraus_operators)

        #Create a new unitary op and set its Kraus ops
        op2 = FullUnitaryOp(np.identity(2, 'd'), mdl.basis)
        op2.set_kraus_operators(kraus_ops)

        self.assertArraysAlmostEqual(op.to_dense(), op2.to_dense())
        self.assertEqual(op.num_kraus_operators, 1)
        self.assertEqual(op2.num_kraus_operators, 1)

        kkdag = [kop @ kop.conjugate().T for kop in op.kraus_operators]
        self.assertArraysAlmostEqual(sum(kkdag), np.identity(2))

    def test_dense_op(self):
        mdl = smq1Q_XYI.target_model('TP').depolarize(op_noise=0.1, spam_noise=0.1)
        op = mdl.operations[()]
        self.assertTrue(isinstance(op, FullTPOp))

        kraus_ops = [k.copy() for k in op.kraus_operators]  # make sure we copy
        self.assertEqual(len(kraus_ops), op.num_kraus_operators)

        op2 = FullArbitraryOp(np.zeros((4, 4), 'd'), mdl.basis)
        op2.set_kraus_operators(kraus_ops)

        self.assertArraysAlmostEqual(op.to_dense(), op2.to_dense())
        self.assertEqual(op.num_kraus_operators, 4)
        self.assertEqual(op2.num_kraus_operators, 4)

        kkdag = [kop @ kop.conjugate().T for kop in op.kraus_operators]
        assert(np.allclose(sum(kkdag), np.identity(2)))

    def test_stochastic_errorgen_equivalence_single(self):
        #Check that StochasticOp and 'S'-type elementary errorgen give the same op
        B = Basis.cast('PP', 4)
        b = Basis.cast('pp', 4)

        std_superop = create_elementary_errorgen('S', B['X'], sparse=False)
        superop = change_basis(std_superop, 'std', b)
        #print(np.round(superop, 4))  # Should be:
        #array([[ 0.,  0.,  0., -0.],
        #       [ 0.,  0.,  0.,  0.],
        #       [ 0.,  0., -2.,  0.],
        #       [ 0.,  0.,  0., -2.]])

        superop2 = unitary_to_superop(B['X'], b) - unitary_to_superop(B['I'], b)
        #print(np.round(superop2, 4))

        self.assertArraysAlmostEqual(superop, superop2)

    def _check_equiv_nQ(self, num_qubits):
        nQ = num_qubits
        B = Basis.cast('PP', 4**nQ)
        b = Basis.cast('pp', 4**nQ)
        Ilbl = B.labels[0]
        for lbl, el in zip(B.labels, B.elements):
            #print(lbl)
            std_superop = create_elementary_errorgen('S', el, sparse=False)
            superop = change_basis(std_superop, 'std', b)
            superop2 = unitary_to_superop(el, b) - unitary_to_superop(B[Ilbl], b)
            self.assertArraysAlmostEqual(superop, superop2)

    def test_stochastic_errorgen_equivalence_1Q(self):
        self._check_equiv_nQ(1)

    def test_stochastic_errorgen_equivalence_2Q(self):
        self._check_equiv_nQ(2)

    def test_stochastic_errorgen_equivalence_3Q(self):
        self._check_equiv_nQ(3)


class KrausInterfaceModelTestBase(object):
    def setUp(self):
        mdl = smq1Q_XYI.target_model('TP', evotype='densitymx').depolarize(op_noise=0.1)
        # op_noise == 4/3(depol_rate), so depol_rate = 0.075
        self.test_circuit = Circuit('Gxpi2:0^2', line_labels=(0,))
        #self.test_circuit = Circuit('[]^2', line_labels=(0,))

        #BASE CASE to compare with - densitymx using TP gates
        self.cmp_probs = mdl.probabilities(self.test_circuit)

        # SANITY CHECK
        #Rates in X,Y,&Z direction is 0.1/4 = 0.025.  Z state is only flipped by X and Y rates (not Z) so
        # probability of flip is 2*0.025 = 0.05.  Probability of staying in 'correct' state (not flipping)
        # after 2 gates is (1 - 0.05)^2.  Probability of flipping from wrong state to correct state is 0.05^2,
        # since there 0.05 probability of being in wrong state after first gate and 0.05 probability to flip.
        # Thus, expected probability of being in correct state is:
        self.expected_prob1 = (1 - 0.05)**2 + 0.05**2
        self.assertAlmostEqual(self.cmp_probs['1'], self.expected_prob1)

        op = mdl.operations[()]

        # Kraus ops are a little weird for idle gate - I think just because there's freedom in
        # choosing the Kraus decomposition (especially for degenerate gates?) and we don't do anything
        # special to choose a nice/standard decomposition.  Could check into this later?
        #for kop in op.kraus_operators:
        #    print(np.round(kop, 3))
        # GIVES not (1-p)I + p/3(X + Y + Z) but:
        # [[0.962+0.j 0.   -0.j]
        #  [0.   -0.j 0.962+0.j]]
        # [[ 0.158+0.j  0.   -0.j]
        #  [ 0.   -0.j -0.158-0.j]]
        # [[0.   +0.j 0.224+0.j]
        #  [0.   +0.j 0.   +0.j]]
        # [[0.   +0.j 0.   +0.j]
        #  [0.224+0.j 0.   +0.j]]

        self.expected_idle_superop = np.array([[1., 0., 0., 0.],
                                               [0., 0.9, 0., 0.],
                                               [0., 0., 0.9, 0.],
                                               [0., 0., 0., 0.9]])
        self.assertArraysAlmostEqual(op.to_dense(on_space='HilbertSchmidt'), self.expected_idle_superop)

    def test_stochastic_op_creation(self):
        ss = QubitSpace(1)
        op = StochasticNoiseOp(ss, initial_rates=[0.025, 0.025, 0.025], evotype=self.evotype) # 0.025 = 0.1/4
        try:
            self.assertArraysAlmostEqual(op.to_dense(on_space='HilbertSchmidt'), self.expected_idle_superop)
        except NotImplementedError:
            pass  # ok if to_dense not implemented, as for CHP evotype

    def test_depol_model(self):
        if self.forwardsim is None:
            self.skipTest("Forward simulator could not be constructed (unavailable?)")
        pspec = smq1Q_XYI.processor_spec()
        mdl_sto = create_explicit_model(
            pspec, evotype=self.evotype,
            simulator=self.forwardsim,
            depolarization_strengths={(): 0.075,
                                      ('Gxpi2',0): 0.075,
                                      ('Gypi2',0): 0.075})  # depol rate is sum of all stochastic rates = 3 * 0.025

        ops = mdl_sto.operations
        self.assertTrue(isinstance(ops[()], ComposedOp))
        self.assertTrue(isinstance(ops[('Gxpi2', 0)], ComposedOp))
        self.assertTrue(isinstance(ops[('Gypi2', 0)], ComposedOp))

        try:
            Gx_error = mdl_sto.operations['Gxpi2', 0].factorops[1].to_dense(on_space='HilbertSchmidt')
            self.assertArraysAlmostEqual(Gx_error, self.expected_idle_superop)
        except NotImplementedError:
            pass  # ok if not implemented, as for CHP evotype

        probs = mdl_sto.probabilities(self.test_circuit)
        self.assertLess(abs(probs['1'] - self.expected_prob1), self.tolerance)

    def test_depol_model_histogram(self):
        if self.forwardsim is None:
            self.skipTest("Forward simulator could not be constructed (unavailable?)")
        pspec = smq1Q_XYI.processor_spec()
        mdl_sto = create_explicit_model(
            pspec, evotype=self.evotype,
            simulator=self.forwardsim,
            depolarization_strengths={(): 0.075,
                                      ('Gxpi2',0): 0.075,
                                      ('Gypi2',0): 0.075})  # depol rate is sum of all stochastic rates = 3 * 0.025

        npoints = self.histogram_npoints; vals = []
        for i in range(npoints):
            probs = mdl_sto.probabilities(self.test_circuit)
            vals.append(probs['1'])
        vals = np.array(vals)
        self.assertLess(abs(vals.mean() - self.expected_prob1), self.tolerance)


class KrausInterfaceDensitymxTester(KrausInterfaceModelTestBase, BaseCase):
    evotype = 'densitymx'
    forwardsim = MapForwardSimulator()
    histogram_npoints = 20
    tolerance = 1e-6


class KrausInterfaceStateVecSlowTester(KrausInterfaceModelTestBase, BaseCase):
    evotype = 'statevec_slow'
    forwardsim = WeakForwardSimulator(shots=1000, base_seed=1234)
    histogram_npoints = 20
    tolerance = 0.005


class KrausInterfaceStateVecTester(KrausInterfaceModelTestBase, BaseCase):
    evotype = 'statevec'
    forwardsim = WeakForwardSimulator(shots=1000, base_seed=1234)
    histogram_npoints = 20
    tolerance = 0.005


class KrausInterfaceCHPTester(KrausInterfaceModelTestBase, BaseCase):
    def setUp(self):
        from pygsti.evotypes import chp
        self.evotype = 'chp'
        chp_path = None #'/Users/enielse/chp/chp'  
        if chp_path is not None:
            chp.chpexe = chp_path
            self.forwardsim = WeakForwardSimulator(shots=100, base_seed=1234)
        else:
            self.forwardsim = None
        self.histogram_npoints = 4
        self.tolerance = 0.05  # very loose because we don't want to do many shots (so it doesn't take forever)
        super().setUp()

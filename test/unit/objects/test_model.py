import numpy as np
import pickle
from contextlib import contextmanager

from ..util import BaseCase, needs_cvxpy

from pygsti.objects import ExplicitOpModel, Instrument, LinearOperator, \
    Circuit, FullDenseOp, FullGaugeGroupElement, matrixforwardsim
import pygsti.construction as pc
import pygsti.objects.model as m


@contextmanager
def smallness_threshold(threshold=10):
    """Helper context for setting/resetting the matrix forward simulator smallness threshold"""
    original = matrixforwardsim.PSMALL
    try:
        matrixforwardsim.PSMALL = threshold
        yield  # yield to context
    finally:
        matrixforwardsim.PSMALL = original


class ModelBase:
    @classmethod
    def setUpClass(cls):
        #OK for these tests, since we test user interface?
        #Set Model objects to "strict" mode for testing
        ExplicitOpModel._strict = False

    def setUp(self):
        self.model = self._model.copy()

    def test_construction(self):
        # XXX what exactly does this cover and is it needed?
        self.assertIsInstance(self.model, m.Model)
        self.assertEqual(len(self.model.preps), 1)
        self.assertEqual(len(self.model.povms['Mdefault']), 2)
        self.assertEqual(list(self.model.preps.keys()), ["rho0"])
        self.assertEqual(list(self.model.povms.keys()), ["Mdefault"])

        # Test default prep/effects
        self.assertArraysAlmostEqual(self.model.prep, self.model.preps["rho0"])
        self.assertEqual(set(self.model.effects.keys()), set(['0', '1']))

        self.assertTrue(isinstance(self.model['Gi'], LinearOperator))

    def _assert_model_params(self, *, nOperations, nSPVecs, nEVecs, nParamsPerGate, nParamsPerSP):
        nParams = nOperations * nParamsPerGate + nSPVecs * nParamsPerSP + nEVecs * 4
        self.assertEqual(self.model.num_params(), nParams)
        # TODO does this actually assert correctness?

    def test_set_all_parameterizations_full(self):
        self.model.set_all_parameterizations("full")
        self._assert_model_params(
            nOperations=3,
            nSPVecs=1,
            nEVecs=2,
            nParamsPerGate=16,
            nParamsPerSP=4
        )

    def test_set_all_parameterizations_TP(self):
        self.model.set_all_parameterizations("TP")
        self._assert_model_params(
            nOperations=3,
            nSPVecs=1,
            nEVecs=1,
            nParamsPerGate=12,
            nParamsPerSP=3
        )

    def test_set_all_parameterizations_static(self):
        self.model.set_all_parameterizations("static")
        self._assert_model_params(
            nOperations=0,
            nSPVecs=0,
            nEVecs=0,
            nParamsPerGate=12,
            nParamsPerSP=3
        )

    def test_element_accessors(self):
        # XXX what does this test cover and is it useful?
        v = np.array([[1.0 / np.sqrt(2)], [0], [0], [1.0 / np.sqrt(2)]], 'd')
        self.model['rho1'] = v
        w = self.model['rho1']
        self.assertArraysAlmostEqual(w, v)

        del self.model.preps['rho1']
        # TODO assert correctness

        Iz = Instrument([('0', np.random.random((4, 4)))])
        self.model["Iz"] = Iz  # set an instrument
        Iz2 = self.model["Iz"]  # get an instrument
        # TODO assert correctness if needed (can the underlying model even mutate this?)

    def test_set_operation_matrix(self):
        # TODO no random
        Gi_test_matrix = np.random.random((4, 4))
        Gi_test_matrix[0, :] = [1, 0, 0, 0]  # so TP mode works
        self.model["Gi"] = Gi_test_matrix  # set operation matrix
        self.assertArraysAlmostEqual(self.model['Gi'], Gi_test_matrix)

        Gi_test_dense_op = FullDenseOp(Gi_test_matrix)
        self.model["Gi"] = Gi_test_dense_op  # set gate object
        self.assertArraysAlmostEqual(self.model['Gi'], Gi_test_matrix)

    def test_raise_on_set_bad_prep_key(self):
        with self.assertRaises(KeyError):
            self.model.preps['foobar'] = [1.0 / np.sqrt(2), 0, 0, 0]  # bad key prefix

    def test_raise_on_get_bad_povm_key(self):
        with self.assertRaises(KeyError):
            self.model.povms['foobar']

    def test_copy(self):
        gs2 = self.model.copy()
        # TODO assert correctness

    def test_deriv_wrt_params(self):
        deriv = self.model.deriv_wrt_params()
        # TODO assert correctness

    def test_frobeniusdist(self):
        cp = self.model.copy()
        self.assertAlmostEqual(self.model.frobeniusdist(cp), 0)
        # TODO non-trivial case

    def test_jtracedist(self):
        cp = self.model.copy()
        self.assertAlmostEqual(self.model.jtracedist(cp), 0)
        # TODO non-trivial case

    @needs_cvxpy
    def test_diamonddist(self):
        cp = self.model.copy()
        self.assertAlmostEqual(self.model.diamonddist(cp), 0)
        # TODO non-trivial case

    def test_vectorize(self):
        # TODO I think this doesn't actually test anything
        cp = self.model.copy()
        v = cp.to_vector()
        cp.from_vector(v)
        self.assertAlmostEqual(self.model.frobeniusdist(cp), 0)

    def test_pickle(self):
        # XXX what exactly does this cover and is it needed?
        p = pickle.dumps(self.model.preps)
        preps = pickle.loads(p)
        self.assertEqual(list(preps.keys()), list(self.model.preps.keys()))

        p = pickle.dumps(self.model.povms)
        povms = pickle.loads(p)
        self.assertEqual(list(povms.keys()), list(self.model.povms.keys()))

        p = pickle.dumps(self.model.operations)
        gates = pickle.loads(p)
        self.assertEqual(list(gates.keys()), list(self.model.operations.keys()))

        self.model._clean_paramvec()
        p = pickle.dumps(self.model)
        g = pickle.loads(p)
        g._clean_paramvec()
        self.assertAlmostEqual(self.model.frobeniusdist(g), 0.0)

    def test_product(self):
        circuit = ('Gx', 'Gy')
        p1 = np.dot(self.model['Gy'], self.model['Gx'])
        p2 = self.model.product(circuit, bScale=False)
        p3, scale = self.model.product(circuit, bScale=True)
        self.assertArraysAlmostEqual(p1, p2)
        self.assertArraysAlmostEqual(p1, scale * p3)

        circuit = ('Gx', 'Gy', 'Gy')
        p1 = np.dot(self.model['Gy'], np.dot(self.model['Gy'], self.model['Gx']))
        p2 = self.model.product(circuit, bScale=False)
        p3, scale = self.model.product(circuit, bScale=True)
        self.assertArraysAlmostEqual(p1, p2)
        self.assertArraysAlmostEqual(p1, scale * p3)

    def test_product_with_high_threshold(self):
        # Artificially reset the "smallness" threshold for scaling to be
        # sure to engate the scaling machinery
        with smallness_threshold(10):
            self.test_product()

    def test_bulk_product(self):
        gatestring1 = ('Gx', 'Gy')
        gatestring2 = ('Gx', 'Gy', 'Gy')
        evt, lookup, outcome_lookup = self.model.bulk_evaltree([gatestring1, gatestring2])

        p1 = np.dot(self.model['Gy'], self.model['Gx'])
        p2 = np.dot(self.model['Gy'], np.dot(self.model['Gy'], self.model['Gx']))

        bulk_prods = self.model.bulk_product(evt)
        bulk_prods_scaled, scaleVals = self.model.bulk_product(evt, bScale=True)
        bulk_prods2 = scaleVals[:, None, None] * bulk_prods_scaled
        self.assertArraysAlmostEqual(bulk_prods[0], p1)
        self.assertArraysAlmostEqual(bulk_prods[1], p2)
        self.assertArraysAlmostEqual(bulk_prods2[0], p1)
        self.assertArraysAlmostEqual(bulk_prods2[1], p2)

    def test_bulk_product_with_high_threshold(self):
        # Artificially reset the "smallness" threshold for scaling to be
        # sure to engate the scaling machinery
        with smallness_threshold(10):
            self.test_bulk_product()

    def test_dproduct(self):
        circuit = ('Gx', 'Gy')
        dp = self.model.dproduct(circuit)
        dp_flat = self.model.dproduct(circuit, flat=True)
        # TODO assert correctness for all of the above

    def test_probs(self):
        circuit = ('Gx', 'Gy')
        p0a = np.dot(np.transpose(self.model.povms['Mdefault']['0']),
                     np.dot(self.model['Gy'],
                            np.dot(self.model['Gx'],
                                   self.model.preps['rho0'])))

        probs = self.model.probs(circuit)
        p0b, p1b = probs[('0',)], probs[('1',)]
        self.assertArraysAlmostEqual(p0a, p0b)
        self.assertArraysAlmostEqual(1.0 - p0a, p1b)

        circuit = ('Gx', 'Gy', 'Gy')
        p1 = np.dot(np.transpose(self.model.povms['Mdefault']['0']),
                    np.dot(self.model['Gy'],
                           np.dot(self.model['Gy'],
                                  np.dot(self.model['Gx'],
                                         self.model.preps['rho0']))))
        p2 = self.model.probs(circuit)[('0',)]
        self.assertAlmostEqual(p1.reshape(-1)[0], p2)
        # TODO is this sufficient?

    def test_probs_map_computation(self):
        #Compare with map-based computation
        self.model.set_simtype('map')
        self.test_probs()

    def test_dprobs(self):
        circuit = ('Gx', 'Gy')

        dprobs = self.model.dprobs(circuit)
        dprobs2 = self.model.dprobs(circuit, returnPr=True)
        self.assertArraysAlmostEqual(dprobs[('0',)], dprobs2[('0',)][0])
        self.assertArraysAlmostEqual(dprobs[('1',)], dprobs2[('1',)][0])
        # TODO assert correctness

    def test_dprobs_map_computation(self):
        #Compare with map-based computation
        self.model.set_simtype('map')
        self.test_dprobs()

    def test_probs_warns_on_nan_in_input(self):
        circuit = ('Gx', 'Gy')

        self.model['rho0'][:] = np.nan
        with self.assertWarns(Warning):
            self.model.probs(circuit)

    def test_probs_map_computation_warns_on_nan_in_input(self):
        self.model.set_simtype('map')
        self.test_probs_warns_on_nan_in_input()

    def test_bulk_probs(self):
        gatestring1 = Circuit(('Gx', 'Gy'))
        gatestring2 = Circuit(('Gx', 'Gy', 'Gy'))
        # evt, lookup, outcome_lookup = self.model.bulk_evaltree([gatestring1, gatestring2])
        # mevt,mlookup,moutcome_lookup = self.mgateset.bulk_evaltree( [gatestring1,gatestring2] )

        p1 = np.dot(np.transpose(self.model.povms['Mdefault']['0']),
                    np.dot(self.model['Gy'],
                           np.dot(self.model['Gx'],
                                  self.model.preps['rho0']))).reshape(-1)[0]

        p2 = np.dot(np.transpose(self.model.povms['Mdefault']['0']),
                    np.dot(self.model['Gy'],
                           np.dot(self.model['Gy'],
                                  np.dot(self.model['Gx'],
                                         self.model.preps['rho0'])))).reshape(-1)[0]

        with self.assertNoWarns():
            bulk_probs = self.model.bulk_probs([gatestring1, gatestring2], check=True)

        self.assertAlmostEqual(p1, bulk_probs[gatestring1][('0',)])
        self.assertAlmostEqual(p2, bulk_probs[gatestring2][('0',)])
        self.assertAlmostEqual(1.0 - p1, bulk_probs[gatestring1][('1',)])
        self.assertAlmostEqual(1.0 - p2, bulk_probs[gatestring2][('1',)])

    def test_bulk_probs_map_computation(self):
        #Compare with map-based computation
        self.model.set_simtype('map')
        self.test_dprobs()


class FullModelTester(ModelBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(FullModelTester, cls).setUpClass()
        cls._model = pc.build_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"])

    def test_transform(self):
        T = np.array([[0.36862036, 0.49241519, 0.35903944, 0.90069522],
                      [0.12347698, 0.45060548, 0.61671491, 0.64854769],
                      [0.4038386, 0.89518315, 0.20206879, 0.6484708],
                      [0.44878029, 0.42095514, 0.27645424, 0.41766033]])  # some random array
        Tinv = np.linalg.inv(T)
        elT = FullGaugeGroupElement(T)
        cp = self.model.copy()
        cp.transform(elT)

        self.assertAlmostEqual(self.model.frobeniusdist(cp, T, normalize=False), 0)  # test out normalize=False
        self.assertAlmostEqual(self.model.jtracedist(cp, T), 0)

        # TODO is this needed?
        for opLabel in cp.operations:
            self.assertArraysAlmostEqual(cp[opLabel], np.dot(Tinv, np.dot(self.model[opLabel], T)))
        for prepLabel in cp.preps:
            self.assertArraysAlmostEqual(cp[prepLabel], np.dot(Tinv, self.model[prepLabel]))
        for povmLabel in cp.povms:
            for effectLabel, eVec in cp.povms[povmLabel].items():
                self.assertArraysAlmostEqual(eVec, np.dot(np.transpose(T), self.model.povms[povmLabel][effectLabel]))


class TPModelTester(ModelBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(TPModelTester, cls).setUpClass()
        cls._model = pc.build_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
            parameterization="TP")


class StaticModelTester(ModelBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(StaticModelTester, cls).setUpClass()
        cls._model = pc.build_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
            parameterization="static")

    def test_set_operation_matrix(self):
        # TODO no random
        Gi_test_matrix = np.random.random((4, 4))
        Gi_test_dense_op = FullDenseOp(Gi_test_matrix)
        self.model["Gi"] = Gi_test_dense_op  # set gate object
        self.assertArraysAlmostEqual(self.model['Gi'], Gi_test_matrix)

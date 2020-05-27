import numpy as np
import pickle
from contextlib import contextmanager
import functools

from ..util import BaseCase, needs_cvxpy

from pygsti.objects import ExplicitOpModel, Instrument, LinearOperator, \
    Circuit, FullDenseOp, FullGaugeGroupElement, matrixforwardsim
from pygsti.tools import indices
import pygsti.construction as pc
import pygsti.objects.model as m


@contextmanager
def smallness_threshold(threshold=10):
    """Helper context for setting/resetting the matrix forward simulator smallness threshold"""
    original_p = matrixforwardsim._PSMALL
    original_d = matrixforwardsim._DSMALL
    original_h = matrixforwardsim._HSMALL
    try:
        matrixforwardsim._PSMALL = threshold
        matrixforwardsim._DSMALL = threshold
        matrixforwardsim._HSMALL = threshold
        yield  # yield to context
    finally:
        matrixforwardsim._HSMALL = original_h
        matrixforwardsim._DSMALL = original_d
        matrixforwardsim._PSMALL = original_p


##
# Model base classes, controlling the parameterization of the tested model
#
class ModelBase(object):
    @classmethod
    def setUpClass(cls):
        #OK for these tests, since we test user interface?
        #Set Model objects to "strict" mode for testing
        ExplicitOpModel._strict = False
        cls._model = pc.create_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
            **cls.build_options)
        super(ModelBase, cls).setUpClass()

    def setUp(self):
        self.model = self._model.copy()
        super(ModelBase, self).setUp()

    def test_construction(self):
        # XXX what exactly does this cover and is it needed?  EGN: not exactly sure what it covers, but this seems like a good sanity check
        self.assertIsInstance(self.model, m.Model)
        self.assertEqual(len(self.model.preps), 1)
        self.assertEqual(len(self.model.povms['Mdefault']), 2)
        self.assertEqual(list(self.model.preps.keys()), ["rho0"])
        self.assertEqual(list(self.model.povms.keys()), ["Mdefault"])

        # Test default prep/effects
        self.assertArraysAlmostEqual(self.model.prep, self.model.preps["rho0"])
        self.assertEqual(set(self.model.effects.keys()), set(['0', '1']))

        self.assertTrue(isinstance(self.model['Gi'], LinearOperator))


class FullModelBase(ModelBase):
    """Base class for test cases using a full-parameterized model"""
    build_options = {'parameterization': 'full'}


class TPModelBase(ModelBase):
    """Base class for test cases using a TP-parameterized model"""
    build_options = {'parameterization': 'TP'}


class StaticModelBase(ModelBase):
    """Base class for test cases using a static-parameterized model"""
    build_options = {'parameterization': 'static'}


##
# Method base classes, controlling which methods will be tested
#
class GeneralMethodBase(object):
    def _assert_model_params(self, nOperations, nSPVecs, nEVecs, nParamsPerGate, nParamsPerSP):
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
        # XXX what does this test cover and is it useful?  EGN: covers the __getitem__/__setitem__ functions of model
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

    def test_strdiff(self):
        other = pc.create_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
            parameterization='TP'
        )
        self.model.strdiff(other)
        # TODO assert correctness

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
        # XXX what exactly does this cover and is it needed?  EGN: this tests that the individual pieces (~dicts) within a model can be pickled; it's useful for debuggin b/c often just one of these will break.
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

    def test_raises_on_get_bad_key(self):
        with self.assertRaises(KeyError):
            self.model['Non-existent-key']

    def test_raises_on_set_bad_key(self):
        with self.assertRaises(KeyError):
            self.model['Non-existent-key'] = np.zeros((4, 4), 'd')  # can't set things not in the model

    def test_raise_on_set_bad_prep_key(self):
        with self.assertRaises(KeyError):
            self.model.preps['foobar'] = [1.0 / np.sqrt(2), 0, 0, 0]  # bad key prefix

    def test_raise_on_get_bad_povm_key(self):
        with self.assertRaises(KeyError):
            self.model.povms['foobar']

    def test_raises_on_conflicting_attribute_access(self):
        self.model.preps['rho1'] = self.model.preps['rho0'].copy()
        self.model.povms['M2'] = self.model.povms['Mdefault'].copy()
        with self.assertRaises(ValueError):
            self.model.prep  # can only use this property when there's a *single* prep
        with self.assertRaises(ValueError):
            self.model.effects  # can only use this property when there's a *single* POVM

        with self.assertRaises(ValueError):
            prep, gates, povm = self.model._split_circuit(Circuit(('rho0', 'Gx')))
        with self.assertRaises(ValueError):
            prep, gates, povm = self.model._split_circuit(Circuit(('Gx', 'Mdefault')))

    def test_set_gate_raises_on_bad_dimension(self):
        with self.assertRaises(ValueError):
            self.model['Gbad'] = FullDenseOp(np.zeros((5, 5), 'd'))


class ThresholdMethodBase(object):
    """Tests for model methods affected by the mapforwardsim smallness threshold"""

    def test_product(self):
        circuit = ('Gx', 'Gy')
        p1 = np.dot(self.model['Gy'], self.model['Gx'])
        p2 = self.model.product(circuit, scale=False)
        p3, scale = self.model.product(circuit, scale=True)
        self.assertArraysAlmostEqual(p1, p2)
        self.assertArraysAlmostEqual(p1, scale * p3)

        circuit = ('Gx', 'Gy', 'Gy')
        p1 = np.dot(self.model['Gy'], np.dot(self.model['Gy'], self.model['Gx']))
        p2 = self.model.product(circuit, scale=False)
        p3, scale = self.model.product(circuit, scale=True)
        self.assertArraysAlmostEqual(p1, p2)
        self.assertArraysAlmostEqual(p1, scale * p3)

    def test_bulk_product(self):
        gatestring1 = ('Gx', 'Gy')
        gatestring2 = ('Gx', 'Gy', 'Gy')
        evt, lookup, outcome_lookup = self.model.bulk_evaltree([gatestring1, gatestring2])

        p1 = np.dot(self.model['Gy'], self.model['Gx'])
        p2 = np.dot(self.model['Gy'], np.dot(self.model['Gy'], self.model['Gx']))

        bulk_prods = self.model.bulk_product(evt)
        bulk_prods_scaled, scaleVals = self.model.bulk_product(evt, scale=True)
        bulk_prods2 = scaleVals[:, None, None] * bulk_prods_scaled
        self.assertArraysAlmostEqual(bulk_prods[0], p1)
        self.assertArraysAlmostEqual(bulk_prods[1], p2)
        self.assertArraysAlmostEqual(bulk_prods2[0], p1)
        self.assertArraysAlmostEqual(bulk_prods2[1], p2)

    def test_dproduct(self):
        circuit = ('Gx', 'Gy')
        dp = self.model.dproduct(circuit)
        dp_flat = self.model.dproduct(circuit, flat=True)
        # TODO assert correctness for all of the above

    def test_bulk_dproduct(self):
        gatestring1 = ('Gx', 'Gy')
        gatestring2 = ('Gx', 'Gy', 'Gy')
        evt, lookup, _ = self.model.bulk_evaltree([gatestring1, gatestring2])
        dp = self.model.bulk_dproduct(evt)
        dp_flat = self.model.bulk_dproduct(evt, flat=True)
        dp_scaled, scaleVals = self.model.bulk_dproduct(evt, scale=True)
        # TODO assert correctness for all of the above

    def test_hproduct(self):
        circuit = ('Gx', 'Gy')
        hp = self.model.hproduct(circuit)
        hp_flat = self.model.hproduct(circuit, flat=True)
        # TODO assert correctness for all of the above

    def test_bulk_hproduct(self):
        gatestring1 = ('Gx', 'Gy')
        gatestring2 = ('Gx', 'Gy', 'Gy')
        evt, lookup, _ = self.model.bulk_evaltree([gatestring1, gatestring2])
        hp = self.model.bulk_hproduct(evt)
        hp_flat = self.model.bulk_hproduct(evt, flat=True)
        hp_scaled, scaleVals = self.model.bulk_hproduct(evt, scale=True)
        # TODO assert correctness for all of the above


class SimMethodBase(object):
    """Tests for model methods which can use different forward sims"""
    # XXX is there any reason this shouldn't be refactored into test_forwardsim?  EGN: no, I think moving it would be fine - most model functions just defer to the fwdsim functions.

    @classmethod
    def setUpClass(cls):
        super(SimMethodBase, cls).setUpClass()
        cls.gatestring1 = ('Gx', 'Gy')
        cls.gatestring2 = ('Gx', 'Gy', 'Gy')
        cls._expected_probs = {
            cls.gatestring1: np.dot(np.transpose(cls._model.povms['Mdefault']['0']),
                                    np.dot(cls._model['Gy'],
                                           np.dot(cls._model['Gx'],
                                                  cls._model.preps['rho0']))).reshape(-1)[0],
            cls.gatestring2: np.dot(np.transpose(cls._model.povms['Mdefault']['0']),
                                    np.dot(cls._model['Gy'],
                                           np.dot(cls._model['Gy'],
                                                  np.dot(cls._model['Gx'],
                                                         cls._model.preps['rho0'])))).reshape(-1)[0]
        }
        # TODO expected dprobs & hprobs

    def test_probs(self):
        probs = self.model.probabilities(self.gatestring1)
        expected = self._expected_probs[self.gatestring1]
        actual_p0, actual_p1 = probs[('0',)], probs[('1',)]
        self.assertAlmostEqual(expected, actual_p0)
        self.assertAlmostEqual(1.0 - expected, actual_p1)

        probs = self.model.probabilities(self.gatestring2)
        expected = self._expected_probs[self.gatestring2]
        actual_p0, actual_p1 = probs[('0',)], probs[('1',)]
        self.assertAlmostEqual(expected, actual_p0)
        self.assertAlmostEqual(1.0 - expected, actual_p1)

    def test_dprobs(self):
        dprobs = self.model.dprobs(self.gatestring1)
        dprobs2 = self.model.dprobs(self.gatestring1, return_pr=True)
        self.assertArraysAlmostEqual(dprobs[('0',)], dprobs2[('0',)][0])
        self.assertArraysAlmostEqual(dprobs[('1',)], dprobs2[('1',)][0])
        # TODO assert correctness

    def test_hprobs(self):
        # TODO optimize
        hprobs = self.model.hprobs(self.gatestring1)
        # XXX is this necessary?  EGN: maybe testing so many cases is overkill?
        # Cover combinations of arguments
        variants = [
            self.model.hprobs(self.gatestring1, return_pr=True),
            self.model.hprobs(self.gatestring1, return_deriv=True),
            self.model.hprobs(self.gatestring1, return_pr=True, return_deriv=True)
        ]
        for hprobs2 in variants:
            self.assertArraysAlmostEqual(hprobs[('0',)], hprobs2[('0',)][0])
            self.assertArraysAlmostEqual(hprobs[('1',)], hprobs2[('1',)][0])
        # TODO assert correctness

    def test_bulk_probs(self):
        with self.assertNoWarns():
            bulk_probs = self.model.bulk_probs([self.gatestring1, self.gatestring2], check=True)

        expected_1 = self._expected_probs[self.gatestring1]
        expected_2 = self._expected_probs[self.gatestring2]
        self.assertAlmostEqual(expected_1, bulk_probs[self.gatestring1][('0',)])
        self.assertAlmostEqual(expected_2, bulk_probs[self.gatestring2][('0',)])
        self.assertAlmostEqual(1.0 - expected_1, bulk_probs[self.gatestring1][('1',)])
        self.assertAlmostEqual(1.0 - expected_2, bulk_probs[self.gatestring2][('1',)])

    def test_bulk_fill_probs(self):
        evt, lookup, _ = self.model.bulk_evaltree([self.gatestring1, self.gatestring2])
        nElements = evt.num_final_elements()
        probs_to_fill = np.empty(nElements, 'd')

        with self.assertNoWarns():
            self.model.bulk_fill_probs(probs_to_fill, evt, check=True)

        expected_1 = self._expected_probs[self.gatestring1]
        expected_2 = self._expected_probs[self.gatestring2]
        actual_1 = probs_to_fill[lookup[0]]
        actual_2 = probs_to_fill[lookup[1]]
        self.assertAlmostEqual(expected_1, actual_1[0])
        self.assertAlmostEqual(expected_2, actual_2[0])
        self.assertAlmostEqual(1 - expected_1, actual_1[1])
        self.assertAlmostEqual(1 - expected_2, actual_2[1])

    def test_bulk_fill_probs_with_split_tree(self):
        # XXX is this correct?  EGN: looks right to me.
        evt, lookup, _ = self.model.bulk_evaltree([self.gatestring1, self.gatestring2])
        nElements = evt.num_final_elements()
        probs_to_fill = np.empty(nElements, 'd')
        lookup_split = evt.split(lookup, num_sub_trees=2)

        with self.assertNoWarns():
            self.model.bulk_fill_probs(probs_to_fill, evt)

        expected_1 = self._expected_probs[self.gatestring1]
        expected_2 = self._expected_probs[self.gatestring2]
        actual_1 = probs_to_fill[lookup_split[0]]
        actual_2 = probs_to_fill[lookup_split[1]]
        self.assertAlmostEqual(expected_1, actual_1[0])
        self.assertAlmostEqual(expected_2, actual_2[0])
        self.assertAlmostEqual(1 - expected_1, actual_1[1])
        self.assertAlmostEqual(1 - expected_2, actual_2[1])

    def test_bulk_dprobs(self):
        with self.assertNoWarns():
            bulk_dprobs = self.model.bulk_dprobs([self.gatestring1, self.gatestring2], return_pr=False)
        # TODO assert correctness

        with self.assertNoWarns():
            bulk_dprobs = self.model.bulk_dprobs([self.gatestring1, self.gatestring2], return_pr=True)
        # TODO assert correctness

    def test_bulk_fill_dprobs(self):
        evt, lookup, _ = self.model.bulk_evaltree([self.gatestring1, self.gatestring2])
        nElements = evt.num_final_elements()
        nParams = self.model.num_params()
        dprobs_to_fill = np.empty((nElements, nParams), 'd')

        with self.assertNoWarns():
            self.model.bulk_fill_dprobs(dprobs_to_fill, evt, check=True)
        # TODO assert correctness

        probs_to_fill = np.empty(nElements, 'd')
        dprobs_to_fill = np.empty((nElements, nParams), 'd')
        with self.assertNoWarns():
            self.model.bulk_fill_dprobs(dprobs_to_fill, evt, pr_mx_to_fill=probs_to_fill, check=True)
        # TODO assert correctness

    def test_bulk_fill_dprobs_with_high_smallness_threshold(self):
        # TODO figure out better way to do this
        with smallness_threshold(10):
            evt, lookup, _ = self.model.bulk_evaltree([self.gatestring1, self.gatestring2])
            nElements = evt.num_final_elements()
            nParams = self.model.num_params()
            dprobs_to_fill = np.empty((nElements, nParams), 'd')

            self.model.bulk_fill_dprobs(dprobs_to_fill, evt, check=True)
            # TODO assert correctness

    def test_bulk_fill_dprobs_with_split_tree(self):
        evt, lookup, _ = self.model.bulk_evaltree([self.gatestring1, self.gatestring2])
        nElements = evt.num_final_elements()
        nParams = self.model.num_params()
        dprobs_to_fill = np.empty((nElements, nParams), 'd')
        lookup_split = evt.split(lookup, num_sub_trees=2)
        with self.assertNoWarns():
            self.model.bulk_fill_dprobs(dprobs_to_fill, evt, check=True)
        # TODO assert correctness

    def test_bulk_hprobs(self):
        # call normally
        with self.assertNoWarns():
            bulk_hprobs = self.model.bulk_hprobs(
                [self.gatestring1, self.gatestring2], return_pr=False, return_deriv=False)
        # TODO assert correctness

        # with probabilities
        with self.assertNoWarns():
            bulk_hprobs = self.model.bulk_hprobs([self.gatestring1, self.gatestring2], return_pr=True, return_deriv=False)
        # TODO assert correctness

        # with derivative probabilities
        with self.assertNoWarns():
            bulk_hprobs = self.model.bulk_hprobs([self.gatestring1, self.gatestring2], return_pr=False, return_deriv=True)
        # TODO assert correctness

    def test_bulk_fill_hprobs(self):
        evt, lookup, _ = self.model.bulk_evaltree([self.gatestring1, self.gatestring2])
        nElements = evt.num_final_elements()
        nParams = self.model.num_params()

        # call normally
        hprobs_to_fill = np.empty((nElements, nParams, nParams), 'd')
        with self.assertNoWarns():
            self.model.bulk_fill_hprobs(hprobs_to_fill, evt, check=True)
        # TODO assert correctness

        # also fill probabilities
        probs_to_fill = np.empty(nElements, 'd')
        with self.assertNoWarns():
            self.model.bulk_fill_hprobs(hprobs_to_fill, evt, pr_mx_to_fill=probs_to_fill, check=True)
        # TODO assert correctness

        #also fill derivative probabilities
        dprobs_to_fill = np.empty((nElements, nParams), 'd')
        hprobs_to_fill = np.empty((nElements, nParams, nParams), 'd')
        with self.assertNoWarns():
            self.model.bulk_fill_hprobs(hprobs_to_fill, evt, deriv_mx_to_fill=dprobs_to_fill, check=True)
        # TODO assert correctness

    def test_bulk_fill_hprobs_with_high_smallness_threshold(self):
        # TODO figure out better way to do this
        with smallness_threshold(10):
            evt, lookup, _ = self.model.bulk_evaltree([self.gatestring1, self.gatestring2])
            nElements = evt.num_final_elements()
            nParams = self.model.num_params()
            hprobs_to_fill = np.empty((nElements, nParams, nParams), 'd')

            self.model.bulk_fill_hprobs(hprobs_to_fill, evt, check=True)
            # TODO assert correctness

    def test_bulk_fill_hprobs_with_split_tree(self):
        evt, lookup, _ = self.model.bulk_evaltree([self.gatestring1, self.gatestring2])
        nElements = evt.num_final_elements()
        nParams = self.model.num_params()
        hprobs_to_fill = np.empty((nElements, nParams, nParams), 'd')
        lookup_split = evt.split(lookup, num_sub_trees=2)
        with self.assertNoWarns():
            self.model.bulk_fill_hprobs(hprobs_to_fill, evt, check=True)
        # TODO assert correctness

    def test_bulk_hprobs_by_block(self):
        evt, lookup, _ = self.model.bulk_evaltree([self.gatestring1, self.gatestring2])
        nP = self.model.num_params()

        hcols = []
        d12cols = []
        slicesList = [(slice(0, nP), slice(i, i + 1)) for i in range(nP)]
        for s1, s2, hprobs_col, dprobs12_col in self.model.bulk_hprobs_by_block(
                evt, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate(hcols, axis=2)  # axes = (spam+circuit, derivParam1, derivParam2)
        all_d12cols = np.concatenate(d12cols, axis=2)
        # TODO assert correctness

    def test_bulk_evaltree(self):
        # Test tree construction
        circuits = pc.to_circuits(
            [('Gx',),
             ('Gy',),
             ('Gx', 'Gy'),
             ('Gy', 'Gy'),
             ('Gy', 'Gx'),
             ('Gx', 'Gx', 'Gx'),
             ('Gx', 'Gy', 'Gx'),
             ('Gx', 'Gy', 'Gy'),
             ('Gy', 'Gy', 'Gy'),
             ('Gy', 'Gx', 'Gx')])
        evt, lookup, outcome_lookup = self.model.bulk_evaltree(circuits, max_tree_size=4)
        evt, lookup, outcome_lookup = self.model.bulk_evaltree(circuits, min_subtrees=2, max_tree_size=4)
        with self.assertWarns(Warning):
            self.model.bulk_evaltree(circuits, min_subtrees=3, max_tree_size=8)
            #balanced to trigger 2 re-splits! (Warning: could not create a tree ...)


class StandardMethodBase(GeneralMethodBase, SimMethodBase, ThresholdMethodBase):
    pass


##
# Test cases to run, built from combinations of model & method bases
#
class FullModelTester(FullModelBase, StandardMethodBase, BaseCase):
    def test_transform(self):
        T = np.array([[0.36862036, 0.49241519, 0.35903944, 0.90069522],
                      [0.12347698, 0.45060548, 0.61671491, 0.64854769],
                      [0.4038386, 0.89518315, 0.20206879, 0.6484708],
                      [0.44878029, 0.42095514, 0.27645424, 0.41766033]])  # some random array
        Tinv = np.linalg.inv(T)
        elT = FullGaugeGroupElement(T)
        cp = self.model.copy()
        cp.transform_inplace(elT)

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

    def test_gpindices(self):
        # Test instrument construction with elements whose gpindices
        # are already initialized.  Since this isn't allowed currently
        # (a future functionality), we need to do some hacking
        mdl = self.model.copy()
        mdl.operations['Gnew1'] = FullDenseOp(np.identity(4, 'd'))
        del mdl.operations['Gnew1']

        v = mdl.to_vector()
        Np = mdl.num_params()
        gate_with_gpindices = FullDenseOp(np.identity(4, 'd'))
        gate_with_gpindices[0, :] = v[0:4]
        gate_with_gpindices.set_gpindices(np.concatenate(
            (np.arange(0, 4), np.arange(Np, Np + 12))), mdl)  # manually set gpindices
        mdl.operations['Gnew2'] = gate_with_gpindices
        mdl.operations['Gnew3'] = FullDenseOp(np.identity(4, 'd'))
        del mdl.operations['Gnew3']  # this causes update of Gnew2 indices
        del mdl.operations['Gnew2']
        # TODO assert correctness

    def test_check_paramvec_raises_on_error(self):
        # XXX is this test needed?  EGN: seems to be a unit test for _check_paramvec, which is good I think.
        self.model._paramvec[:] = 0.0  # mess with paramvec to get error below
        with self.assertRaises(ValueError):
            self.model._check_paramvec(debug=True)  # param vec is now out of sync!

    def test_probs_warns_on_nan_in_input(self):
        self.model['rho0'][:] = np.nan
        with self.assertWarns(Warning):
            self.model.probabilities(self.gatestring1)


class TPModelTester(TPModelBase, StandardMethodBase, BaseCase):
    def test_tp_dist(self):
        self.assertAlmostEqual(self.model._tpdist(), 3.52633900335e-16, 5)


class StaticModelTester(StaticModelBase, StandardMethodBase, BaseCase):
    def test_set_operation_matrix(self):
        # TODO no random
        Gi_test_matrix = np.random.random((4, 4))
        Gi_test_dense_op = FullDenseOp(Gi_test_matrix)
        self.model["Gi"] = Gi_test_dense_op  # set gate object
        self.assertArraysAlmostEqual(self.model['Gi'], Gi_test_matrix)

    def test_bulk_fill_dprobs_with_high_smallness_threshold(self):
        self.skipTest("TODO should probably warn user?")

    def test_bulk_fill_hprobs_with_high_smallness_threshold(self):
        self.skipTest("TODO should probably warn user?")

    def test_bulk_hprobs_by_block(self):
        self.skipTest("TODO should probably warn user?")


class FullMapSimMethodTester(FullModelBase, SimMethodBase, BaseCase):
    def setUp(self):
        super(FullMapSimMethodTester, self).setUp()
        self.model.set_simtype('map')

    def test_bulk_evaltree(self):
        # Test tree construction
        circuits = pc.to_circuits(
            [('Gx',),
             ('Gy',),
             ('Gx', 'Gy'),
             ('Gy', 'Gy'),
             ('Gy', 'Gx'),
             ('Gx', 'Gx', 'Gx'),
             ('Gx', 'Gy', 'Gx'),
             ('Gx', 'Gy', 'Gy'),
             ('Gy', 'Gy', 'Gy'),
             ('Gy', 'Gx', 'Gx')])
        evt, lookup, outcome_lookup = self.model.bulk_evaltree(circuits, max_tree_size=4)
        evt, lookup, outcome_lookup = self.model.bulk_evaltree(circuits, min_subtrees=2, max_tree_size=4)
        with self.assertNoWarns():
            self.model.bulk_evaltree(circuits, min_subtrees=3, max_tree_size=8)
            #balanced to trigger 2 re-splits! (Warning: could not create a tree ...)


class FullHighThresholdMethodTester(FullModelBase, ThresholdMethodBase, BaseCase):
    def setUp(self):
        super(FullHighThresholdMethodTester, self).setUp()
        self._context = smallness_threshold(10)
        self._context.__enter__()

    def tearDown(self):
        self._context.__exit__(None, None, None)
        super(FullHighThresholdMethodTester, self).tearDown()


class FullBadDimensionModelTester(FullModelBase, BaseCase):
    def setUp(self):
        super(FullBadDimensionModelTester, self).setUp()
        self.model = self.model.increase_dimension(11)

    # XXX these aren't tested under normal conditions...  EGN: we should probably test them under normal conditions then.
    def test_rotate_raises(self):
        with self.assertRaises(AssertionError):
            self.model.rotate((0.1, 0.1, 0.1))

    def test_randomize_with_unitary_raises(self):
        with self.assertRaises(AssertionError):
            self.model.randomize_with_unitary(1, rand_state=np.random.RandomState())  # scale shouldn't matter

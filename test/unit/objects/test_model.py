# XXX rewrite/refactor forward-simulator tests

import pickle
from contextlib import contextmanager

import sys
import numpy as np

import pygsti.circuits as pc
import pygsti.models as models
import pygsti.models.model as m
from pygsti.forwardsims import matrixforwardsim, mapforwardsim
from pygsti.modelmembers.instruments import Instrument
from pygsti.modelmembers.operations import LinearOperator, FullArbitraryOp
from pygsti.models import ExplicitOpModel
from pygsti.circuits import Circuit
from pygsti.models.gaugegroup import FullGaugeGroupElement
from ..util import BaseCase, needs_cvxpy

SKIP_DIAMONDIST_ON_WIN = True


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
        cls._model = models.create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
            **cls.build_options)

        super(ModelBase, cls).setUpClass()

    def setUp(self):
        self.model = self._model.copy()
        self.model.sim = 'matrix'
        super(ModelBase, self).setUp()

    def test_construction(self):
        # XXX what exactly does this cover and is it needed?  EGN: not exactly sure what it covers, but this seems like a good sanity check
        self.assertIsInstance(self.model, m.Model)
        self.assertEqual(len(self.model.preps), 1)
        self.assertEqual(len(self.model.povms['Mdefault']), 2)
        self.assertEqual(list(self.model.preps.keys()), ["rho0"])
        self.assertEqual(list(self.model.povms.keys()), ["Mdefault"])

        # Test default prep/effects
        self.assertArraysAlmostEqual(self.model.prep.to_dense(), self.model.preps["rho0"].to_dense())
        self.assertEqual(set(self.model.effects.keys()), set(['0', '1']))

        self.assertTrue(isinstance(self.model['Gi'], LinearOperator))

#TODO: Add tests between more combinations of parameterizations
class FullModelBase(ModelBase):
    """Base class for test cases using a full-parameterized model"""
    build_options = {'gate_type': 'full'}


class TPModelBase(ModelBase):
    """Base class for test cases using a TP-parameterized model"""
    build_options = {'gate_type': 'full TP'}


class StaticModelBase(ModelBase):
    """Base class for test cases using a static-parameterized model"""
    build_options = {'gate_type': 'static'}

class GLNDModelBase(ModelBase):
    """Base class for test cases using a static-parameterized model"""
    build_options = {'gate_type': 'GLND'}
##
# Method base classes, controlling which methods will be tested
#
class GeneralMethodBase(object):
    def _assert_model_params(self, nOperations, nSPVecs, nEVecs, nParamsPerGate, nParamsPerSP):
        nParams = nOperations * nParamsPerGate + nSPVecs * nParamsPerSP + nEVecs * 4
        print("num params = ", self.model.num_params)
        self.assertEqual(self.model.num_params, nParams)

    def _assert_model_ops(self, oldModel):
        #test operations
        for (_, gate),(_,gate2) in zip(self.model.operations.items(), oldModel.operations.items() ):
            assert np.allclose(gate.to_dense(), gate2.to_dense()), "Discrepancy in operation process matrices when converting parameterizations"

    def _assert_model_SPAM(self, oldModel):
        for (_, povm1), (_, povm2) in zip(self.model.povms.items(), oldModel.povms.items()):
            for element1, element2 in zip(povm1.items(), povm2.items()):
                assert np.allclose(element1[1].to_dense(), element2[1].to_dense()), "Discrepancy in POVM superbra when converting parameterizations"
        
        for (_, prep1), (_, prep2) in zip(self.model.preps.items(), oldModel.preps.items()):
            assert np.allclose(prep1.to_dense(), prep2.to_dense()), "Discrepancy in state prep superket when converting parameterizations"

    def test_set_all_parameterizations_full(self):
        model_copy = self.model.copy()
        self.model.set_all_parameterizations("full")        
        self._assert_model_ops(model_copy)
        self._assert_model_SPAM(model_copy)
        self._assert_model_params(
            nOperations=3,
            nSPVecs=1,
            nEVecs=2,
            nParamsPerGate=16,
            nParamsPerSP=4
        )

    def test_set_all_parameterizations_TP(self):
        model_copy = self.model.copy()
        self.model.set_all_parameterizations("full TP")
        self._assert_model_ops(model_copy)
        self._assert_model_SPAM(model_copy)
        self._assert_model_params(
            nOperations=3,
            nSPVecs=1,
            nEVecs=1,
            nParamsPerGate=12,
            nParamsPerSP=3
        )

    def test_set_all_parameterizations_static(self):
        model_copy = self.model.copy()
        self.model.set_all_parameterizations("static")
        self._assert_model_ops(model_copy)
        self._assert_model_SPAM(model_copy)
        self._assert_model_params(
            nOperations=0,
            nSPVecs=0,
            nEVecs=0,
            nParamsPerGate=12,
            nParamsPerSP=3
        )

    def test_set_all_parameterizations_HS(self):
        model_copy = self.model.copy()
        self.model.set_all_parameterizations("H+S")
        self._assert_model_ops(model_copy)
        self._assert_model_SPAM(model_copy)
        assert self.model.num_params == 6 * (3 + 1 + 1)

    def test_set_all_parameterizations_GLND(self):
        model_copy = self.model.copy()
        self.model.set_all_parameterizations("GLND")
        self._assert_model_ops(model_copy)
        self._assert_model_SPAM(model_copy)
        assert self.model.num_params == 12 * (3 + 1 + 1)

    def test_element_accessors(self):
        # XXX what does this test cover and is it useful?  EGN: covers the __getitem__/__setitem__ functions of model
        v = np.array([[1.0 / np.sqrt(2)], [0], [0], [1.0 / np.sqrt(2)]], 'd')
        self.model['rho1'] = v
        w = self.model['rho1']
        self.assertArraysAlmostEqual(w.to_dense(), v.T)

        del self.model.preps['rho1']
        # TODO assert correctness

        Iz = Instrument([('0', np.random.random((4, 4)))])
        self.model["Iz"] = Iz  # set an instrument
        Iz2 = self.model["Iz"]  # get an instrument
        # TODO assert correctness if needed (can the underlying model even mutate this?)

    def test_strdiff(self):
        other = models.create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"],
            gate_type='full TP'
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
        if SKIP_DIAMONDIST_ON_WIN and sys.platform.startswith('win'): return
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
            prep, gates, povm = self.model.split_circuit(Circuit(('rho0', 'Gx')))
        with self.assertRaises(ValueError):
            prep, gates, povm = self.model.split_circuit(Circuit(('Gx', 'Mdefault')))

    def test_set_gate_raises_on_bad_dimension(self):
        with self.assertRaises(AssertionError):
            self.model['Gbad'] = FullArbitraryOp(np.zeros((5, 5), 'd'))

    def test_parameter_labels(self):
        self.model.set_all_parameterizations("H+s")

        self.model.parameter_labels
        if self.model.num_params > 0:
            self.model.set_parameter_label(index=0, label="My favorite parameter")
            self.assertEqual(self.model.parameter_labels[0], "My favorite parameter")

        self.model.operations['Gx'].parameter_labels  # ('Gxpi2',0)
        self.model.print_parameters_by_op()

    def test_collect_parameters(self):
        self.model.set_all_parameterizations("H+s")
        self.assertEqual(self.model.num_params, 30)

        self.model.collect_parameters([ ('Gx', 'X Hamiltonian error coefficient'),
                                  ('Gy', 'Y Hamiltonian error coefficient')],
                                new_param_label='Over-rotation')
        self.assertEqual(self.model.num_params, 29)
        self.assertTrue(bool(('Gx', 'X Hamiltonian error coefficient') not in set(self.model.parameter_labels)))
        self.assertTrue(bool(('Gy', 'Y Hamiltonian error coefficient') not in set(self.model.parameter_labels)))
        self.assertTrue(bool('Over-rotation' in set(self.model.parameter_labels)))

        # You can also use integer indices, and parameter labels can be tuples too.
        self.model.collect_parameters([3,4,5], new_param_label=("rho0", "common stochastic coefficient"))
        self.assertEqual(self.model.num_params, 27)

        lbls_save = self.model.parameter_labels.copy()
        #DEBUG print(self.model.parameter_labels) #self.model.print_parameters_by_op();  print()


        # Using "pretty" labels works too:
        self.model.collect_parameters(['Gx: Y stochastic coefficient',
                                       'Gx: Z stochastic coefficient' ],
                                      new_param_label='Gxpi2 off-axis stochastic')
        self.assertEqual(self.model.num_params, 26)
        #DEBUG print(self.model.parameter_labels)
        
        #Just make sure printing works
        self.model.parameter_labels_pretty
        self.model.print_parameters_by_op()

        self.model.uncollect_parameters('Gxpi2 off-axis stochastic')

        #DEBUG print(); print(self.model.parameter_labels)
        self.assertEqual(self.model.num_params, 27)
        self.assertEqual(set(lbls_save), set(self.model.parameter_labels))  # ok if ordering if different

    def test_parameter_bounds(self):
        self.model.set_all_parameterizations("H+S")
        self.model.num_params  # rebuild parameter vector -- but this should be done by set_all_parameterizations?!
        
        self.assertTrue(self.model.parameter_bounds is None)
        self.assertTrue(self.model['Gx'].parameter_bounds is None)

        new_bounds = np.ones((6,2), 'd')
        new_bounds[:,0] = -0.01  # lower bounds
        new_bounds[:,1] = +0.01  # upper bounds
        self.model['Gx'].parameter_bounds = new_bounds

        self.model.num_params  # should rebuild parameter vector -- maybe .parameter_bounds should (but rebuild call it!)
        Gx_indices = self.model['Gx'].gpindices
        self.assertArraysAlmostEqual(self.model.parameter_bounds[Gx_indices], new_bounds)


class ThresholdMethodBase(object):
    """Tests for model methods affected by the mapforwardsim smallness threshold"""

    def test_product(self):
        circuit = ('Gx', 'Gy')
        p1 = self.model['Gy'].to_dense() @ self.model['Gx'].to_dense()
        p2 = self.model.sim.product(circuit, scale=False)
        p3, scale = self.model.sim.product(circuit, scale=True)
        self.assertArraysAlmostEqual(p1, p2)
        self.assertArraysAlmostEqual(p1, scale * p3)

        circuit = ('Gx', 'Gy', 'Gy')
        p1 = self.model['Gy'].to_dense() @ self.model['Gy'].to_dense() @ self.model['Gx'].to_dense()
        p2 = self.model.sim.product(circuit, scale=False)
        p3, scale = self.model.sim.product(circuit, scale=True)
        self.assertArraysAlmostEqual(p1, p2)
        self.assertArraysAlmostEqual(p1, scale * p3)

    def test_bulk_product(self):
        gatestring1 = ('Gx', 'Gy')
        gatestring2 = ('Gx', 'Gy', 'Gy')
        circuits = [gatestring1, gatestring2]

        p1 = self.model['Gy'].to_dense() @ self.model['Gx'].to_dense()
        p2 = self.model['Gy'].to_dense() @ self.model['Gy'].to_dense() @ self.model['Gx'].to_dense()

        bulk_prods = self.model.sim.bulk_product(circuits)
        bulk_prods_scaled, scaleVals = self.model.sim.bulk_product(circuits, scale=True)
        bulk_prods2 = scaleVals[:, None, None] * bulk_prods_scaled
        self.assertArraysAlmostEqual(bulk_prods[0], p1)
        self.assertArraysAlmostEqual(bulk_prods[1], p2)
        self.assertArraysAlmostEqual(bulk_prods2[0], p1)
        self.assertArraysAlmostEqual(bulk_prods2[1], p2)

    def test_dproduct(self):
        circuit = ('Gx', 'Gy')
        dp = self.model.sim.dproduct(circuit)
        dp_flat = self.model.sim.dproduct(circuit, flat=True)
        # TODO assert correctness for all of the above

    def test_bulk_dproduct(self):
        gatestring1 = ('Gx', 'Gy')
        gatestring2 = ('Gx', 'Gy', 'Gy')
        circuits = [gatestring1, gatestring2]
        dp = self.model.sim.bulk_dproduct(circuits)
        dp_flat = self.model.sim.bulk_dproduct(circuits, flat=True)
        dp_scaled, scaleVals = self.model.sim.bulk_dproduct(circuits, scale=True)
        # TODO assert correctness for all of the above

    def test_hproduct(self):
        circuit = ('Gx', 'Gy')
        hp = self.model.sim.hproduct(circuit)
        hp_flat = self.model.sim.hproduct(circuit, flat=True)
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
            cls.gatestring1: (np.transpose(cls._model.povms['Mdefault']['0'].to_dense()) @
                                    cls._model['Gy'].to_dense() @
                                           cls._model['Gx'].to_dense() @
                                                  cls._model.preps['rho0'].to_dense()).reshape(-1)[0],
            cls.gatestring2: (np.transpose(cls._model.povms['Mdefault']['0'].to_dense()) @
                                    cls._model['Gy'].to_dense() @
                                           cls._model['Gy'].to_dense() @
                                                  cls._model['Gx'].to_dense() @
                                                         cls._model.preps['rho0'].to_dense()).reshape(-1)[0]
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
        dprobs = self.model.sim.dprobs(self.gatestring1)
        # TODO assert correctness

    def test_hprobs(self):
        # TODO optimize
        hprobs = self.model.sim.hprobs(self.gatestring1)
        # TODO assert correctness

    def test_bulk_probs(self):
        with self.assertNoWarns():
            bulk_probs = self.model.sim.bulk_probs([self.gatestring1, self.gatestring2])

        expected_1 = self._expected_probs[self.gatestring1]
        expected_2 = self._expected_probs[self.gatestring2]
        self.assertAlmostEqual(expected_1, bulk_probs[self.gatestring1][('0',)])
        self.assertAlmostEqual(expected_2, bulk_probs[self.gatestring2][('0',)])
        self.assertAlmostEqual(1.0 - expected_1, bulk_probs[self.gatestring1][('1',)])
        self.assertAlmostEqual(1.0 - expected_2, bulk_probs[self.gatestring2][('1',)])

    def test_bulk_fill_probs(self):
        layout = self.model.sim.create_layout([self.gatestring1, self.gatestring2])
        nElements = layout.num_elements
        probs_to_fill = np.empty(nElements, 'd')

        with self.assertNoWarns():
            self.model.sim.bulk_fill_probs(probs_to_fill, layout)

        expected_1 = self._expected_probs[self.gatestring1]
        expected_2 = self._expected_probs[self.gatestring2]
        actual_1 = probs_to_fill[layout.indices_for_index(0)]
        actual_2 = probs_to_fill[layout.indices_for_index(1)]
        zero_outcome_index1 = layout.outcomes_for_index(0).index(('0',))
        zero_outcome_index2 = layout.outcomes_for_index(1).index(('0',))
        self.assertAlmostEqual(expected_1, actual_1[zero_outcome_index1])
        self.assertAlmostEqual(expected_2, actual_2[zero_outcome_index2])
        self.assertAlmostEqual(1 - expected_1, actual_1[1-zero_outcome_index1])
        self.assertAlmostEqual(1 - expected_2, actual_2[1-zero_outcome_index2])

    def test_bulk_dprobs(self):
        with self.assertNoWarns():
            bulk_dprobs = self.model.sim.bulk_dprobs([self.gatestring1, self.gatestring2])
        # TODO assert correctness

        with self.assertNoWarns():
            bulk_dprobs = self.model.sim.bulk_dprobs([self.gatestring1, self.gatestring2])
        # TODO assert correctness

    def test_bulk_fill_dprobs(self):
        layout = self.model.sim.create_layout([self.gatestring1, self.gatestring2], array_types=('ep',))
        nElements = layout.num_elements
        nParams = self.model.num_params
        dprobs_to_fill = np.empty((nElements, nParams), 'd')

        with self.assertNoWarns():
            self.model.sim.bulk_fill_dprobs(dprobs_to_fill, layout)
        # TODO assert correctness

        probs_to_fill = np.empty(nElements, 'd')
        dprobs_to_fill = np.empty((nElements, nParams), 'd')
        with self.assertNoWarns():
            self.model.sim.bulk_fill_dprobs(dprobs_to_fill, layout, pr_array_to_fill=probs_to_fill)
        # TODO assert correctness

    def test_bulk_fill_dprobs_with_high_smallness_threshold(self):
        # TODO figure out better way to do this
        with smallness_threshold(10):
            layout = self.model.sim.create_layout([self.gatestring1, self.gatestring2], array_types=('ep',))
            nElements = layout.num_elements
            nParams = self.model.num_params
            dprobs_to_fill = np.empty((nElements, nParams), 'd')

            self.model.sim.bulk_fill_dprobs(dprobs_to_fill, layout)
            # TODO assert correctness

    def test_bulk_hprobs(self):
        # call normally
        #with self.assertNoWarns():  # - now *can* warn about inefficient evaltree (ok)
        bulk_hprobs = self.model.sim.bulk_hprobs([self.gatestring1, self.gatestring2])
        # TODO assert correctness

        #hprobs no longer has return_pr and return_deriv args - just call respective functions.
        ## with probabilities
        #with self.assertNoWarns():
        #    bulk_hprobs = self.model.bulk_hprobs([self.gatestring1, self.gatestring2], return_pr=True, return_deriv=False)
        ## TODO assert correctness
        #
        ## with derivative probabilities
        #with self.assertNoWarns():
        #    bulk_hprobs = self.model.bulk_hprobs([self.gatestring1, self.gatestring2], return_pr=False, return_deriv=True)
        ## TODO assert correctness

    def test_bulk_fill_hprobs(self):
        layout = self.model.sim.create_layout([self.gatestring1, self.gatestring2], array_types=('epp',))
        nElements = layout.num_elements
        nParams = self.model.num_params

        # call normally
        hprobs_to_fill = np.empty((nElements, nParams, nParams), 'd')
        #with self.assertNoWarns():  # - now *can* warn about inefficient evaltree (ok)
        self.model.sim.bulk_fill_hprobs(hprobs_to_fill, layout)
        # TODO assert correctness

        # also fill probabilities
        probs_to_fill = np.empty(nElements, 'd')
        #with self.assertNoWarns():  # - now *can* warn about inefficient evaltree (ok)
        self.model.sim.bulk_fill_hprobs(hprobs_to_fill, layout, pr_array_to_fill=probs_to_fill)
        # TODO assert correctness

        #also fill derivative probabilities
        dprobs_to_fill = np.empty((nElements, nParams), 'd')
        hprobs_to_fill = np.empty((nElements, nParams, nParams), 'd')
        #with self.assertNoWarns():  # - now *can* warn about inefficient evaltree (ok)
        self.model.sim.bulk_fill_hprobs(hprobs_to_fill, layout, deriv1_array_to_fill=dprobs_to_fill)
        # TODO assert correctness

    def test_bulk_fill_hprobs_with_high_smallness_threshold(self):
        # TODO figure out better way to do this
        with smallness_threshold(10):
            layout = self.model.sim.create_layout([self.gatestring1, self.gatestring2], array_types=('epp',))
            nElements = layout.num_elements
            nParams = self.model.num_params
            hprobs_to_fill = np.empty((nElements, nParams, nParams), 'd')

            self.model.sim.bulk_fill_hprobs(hprobs_to_fill, layout)
            # TODO assert correctness

    def test_iter_hprobs_by_rectangle(self):
        layout = self.model.sim.create_layout([self.gatestring1, self.gatestring2], array_types=('epp',))
        nP = self.model.num_params

        hcols = []
        d12cols = []
        slicesList = [(slice(0, nP), slice(i, i + 1)) for i in range(nP)]
        for s1, s2, hprobs_col, dprobs12_col in self.model.sim.iter_hprobs_by_rectangle(
                layout, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate(hcols, axis=2)  # axes = (spam+circuit, derivParam1, derivParam2)
        all_d12cols = np.concatenate(d12cols, axis=2)
        # TODO assert correctness

    def test_layout_construction(self):
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

        layout = self.model.sim.create_layout(circuits)

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
            self.assertArraysAlmostEqual(cp[opLabel], Tinv @ self.model[opLabel] @ T)
        for prepLabel in cp.preps:
            self.assertArraysAlmostEqual(cp[prepLabel], Tinv @ self.model[prepLabel])
        for povmLabel in cp.povms:
            for effectLabel, eVec in cp.povms[povmLabel].items():
                self.assertArraysAlmostEqual(eVec, np.transpose(T) @ self.model.povms[povmLabel][effectLabel])

    def test_gpindices(self):
        # Test instrument construction with elements whose gpindices
        # are already initialized.  Since this isn't allowed currently
        # (a future functionality), we need to do some hacking
        mdl = self.model.copy()
        mdl.operations['Gnew1'] = FullArbitraryOp(np.identity(4, 'd'))
        del mdl.operations['Gnew1']

        v = mdl.to_vector()
        Np = mdl.num_params
        gate_with_gpindices = FullArbitraryOp(np.identity(4, 'd'))
        gate_with_gpindices[0, :] = v[0:4]
        gate_with_gpindices.set_gpindices(np.concatenate(
            (np.arange(0, 4), np.arange(Np, Np + 12))), mdl)  # manually set gpindices
        mdl.operations['Gnew2'] = gate_with_gpindices
        mdl.operations['Gnew3'] = FullArbitraryOp(np.identity(4, 'd'))
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
    def test_set_operation_matrix(self):
        # TODO no random
        Gi_test_matrix = np.random.random((4, 4))
        Gi_test_matrix[0, :] = [1, 0, 0, 0]  # so TP mode works
        self.model["Gi"] = Gi_test_matrix  # set operation matrix
        self.assertArraysAlmostEqual(self.model['Gi'], Gi_test_matrix)

        Gi_test_dense_op = FullArbitraryOp(Gi_test_matrix)
        self.model["Gi"] = Gi_test_dense_op  # set gate object
        self.assertArraysAlmostEqual(self.model['Gi'], Gi_test_matrix)


class StaticModelTester(StaticModelBase, StandardMethodBase, BaseCase):
    def test_set_operation_matrix(self):
        # TODO no random
        Gi_test_matrix = np.random.random((4, 4))
        Gi_test_dense_op = FullArbitraryOp(Gi_test_matrix)
        self.model["Gi"] = Gi_test_dense_op  # set gate object
        self.assertArraysAlmostEqual(self.model['Gi'], Gi_test_matrix)

    def test_bulk_fill_dprobs_with_high_smallness_threshold(self):
        self.skipTest("TODO should probably warn user?")

    def test_bulk_fill_hprobs_with_high_smallness_threshold(self):
        self.skipTest("TODO should probably warn user?")

    def test_iter_hprobs_by_rectangle(self):
        self.skipTest("TODO should probably warn user?")

class LindbladModelTester(GLNDModelBase, StandardMethodBase, BaseCase):
    pass
class FullMapSimMethodTester(FullModelBase, SimMethodBase, BaseCase):
    def setUp(self):
        super(FullMapSimMethodTester, self).setUp()
        self.model.sim = mapforwardsim.MapForwardSimulator(self.model)

    def test_layout_construction(self):
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

        layout = self.model.sim.create_layout(circuits)

class FullHighThresholdMethodTester(FullModelBase, ThresholdMethodBase, BaseCase):
    def setUp(self):
        super(FullHighThresholdMethodTester, self).setUp()
        self._context = smallness_threshold(10)
        self._context.__enter__()

    def tearDown(self):
        self._context.__exit__(None, None, None)
        super(FullHighThresholdMethodTester, self).tearDown()


#TODO: see if this makes sense to have as a unit test... now it fails in setUp b/c 11 is a bad dimension
#class FullBadDimensionModelTester(FullModelBase, BaseCase):
#    def setUp(self):
#        super(FullBadDimensionModelTester, self).setUp()
#        self.model = self.model.increase_dimension(11)
#
#    # XXX these aren't tested under normal conditions...  EGN: we should probably test them under normal conditions then.
#    def test_rotate_raises(self):
#        with self.assertRaises(AssertionError):
#            self.model.rotate((0.1, 0.1, 0.1))
#
#    def test_randomize_with_unitary_raises(self):
#        with self.assertRaises(AssertionError):
#            self.model.randomize_with_unitary(1, rand_state=np.random.RandomState())  # scale shouldn't matter

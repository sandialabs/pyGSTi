
import numpy as np
from pygsti.baseobjs.label import Label

from pygsti.circuits.circuit import Circuit
from pygsti.modelmembers.operations import ComposedOp, EmbeddedOp
from pygsti.models.explicitmodel import ExplicitOpModel
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.processors.processorspec import QubitProcessorSpec
from pygsti.modelmembers.operations import (
    StaticArbitraryOp, ExpErrorgenOp, LindbladErrorgen
)
from pygsti.modelmembers.instruments import Instrument
from ..util import BaseCase


class LocalNoiseModelInstanceTester(BaseCase):

    def setUp(self):
        nQubits = 2
        qubit_labels = ['qb{}'.format(i) for i in range(nQubits)]
        self.pspec_2Q = QubitProcessorSpec(
            nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
            qubit_labels=qubit_labels
        )
        nQubits = 4
        qubit_labels = ['qb{}'.format(i) for i in range(nQubits)]
        self.pspec_4Q = QubitProcessorSpec(
            nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
            qubit_labels=qubit_labels
        )

    def test_indep_localnoise(self):
        mdl_local = create_crosstalk_free_model(
            self.pspec_2Q, ideal_gate_type='H+S',
            ideal_spam_type='tensor product H+S', independent_gates=True,
            ensure_composed_gates=False
        )

        self.assertEqual(set(mdl_local.operation_blks['gates'].keys()), set([
            ('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'),
            ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')
        ]))
        self.assertEqual(set(mdl_local.operation_blks['layers'].keys()), set([
            ('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'),
            ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')
        ]))
        test_circuit = ([('Gx', 'qb0'), ('Gy', 'qb1')],
                        ('Gcnot', 'qb0', 'qb1'),
                        [('Gx', 'qb1'), ('Gy', 'qb0')])
        probs = mdl_local.probabilities(test_circuit)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(mdl_local.num_params, 108)

    def test_dep_localnoise(self):
        mdl_local = create_crosstalk_free_model(
            self.pspec_2Q, ideal_gate_type='H+S',
            ideal_spam_type='exp(H+S)', independent_gates=False,
            ensure_composed_gates=False
        )

        self.assertEqual(set(mdl_local.operation_blks['gates'].keys()),
                         set(["Gx", "Gy", "Gcnot"]))
        self.assertEqual(set(mdl_local.operation_blks['layers'].keys()), set([
            ('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'),
            ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0')
        ]))
        test_circuit = ([('Gx', 'qb0'), ('Gy', 'qb1')],
                        ('Gcnot', 'qb0', 'qb1'),
                        [('Gx', 'qb1'), ('Gy', 'qb0')])
        probs = mdl_local.probabilities(test_circuit)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(mdl_local.num_params, 66)

    def test_localnoise_1Q_global_idle(self):
        nQubits = 2
        noisy_idle = StaticArbitraryOp(np.array([[1, 0, 0, 0],
                                                 [0, 0.9, 0, 0],
                                                 [0, 0, 0.9, 0],
                                                 [0, 0, 0, 0.9]], 'd'))
        qubit_labels = ['qb{}'.format(i) for i in range(nQubits)]
        pspec_2Q = QubitProcessorSpec(
            nQubits, ('Gx', 'Gy', 'Gcnot', 'Gidle'), geometry="line",
            availability={'Gidle': [('qb0',), ('qb1',)]},
            qubit_labels=qubit_labels
        )

        mdl_local = create_crosstalk_free_model(
            pspec_2Q, {'Gidle': noisy_idle}, ideal_gate_type='static',
            independent_gates=False, ensure_composed_gates=False,
            implicit_idle_mode='add_global'
        )

        self.assertEqual(set(mdl_local.operation_blks['gates'].keys()),
                         set(["Gx", "Gy", "Gcnot", "Gidle"]))
        self.assertEqual(set(mdl_local.operation_blks['layers'].keys()), set([
            ('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'),
            ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0'),
            ('Gidle', 'qb0'), ('Gidle', 'qb1'), '{auto_global_idle}'
        ]))
        test_circuit = (('Gx', 'qb0'), ('Gcnot', 'qb0', 'qb1'),
                        [], [('Gx', 'qb1'), ('Gy', 'qb0')])
        probs = mdl_local.probabilities(test_circuit)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertAlmostEqual(probs['00'], 0.3576168)
        self.assertEqual(mdl_local.num_params, 0)

        op = mdl_local.circuit_layer_operator(Label('Gx', 'qb1'))
        ref_op = ComposedOp([
            ComposedOp([
                EmbeddedOp(mdl_local.state_space, ('qb0',),
                           mdl_local.operation_blks['gates']['Gidle']),
                EmbeddedOp(mdl_local.state_space, ('qb1',),
                           mdl_local.operation_blks['gates']['Gidle']),
            ]),  # Global idle
            EmbeddedOp(mdl_local.state_space, ('qb1',),
                       mdl_local.operation_blks['gates']['Gx'])  # Gx op
        ])
        ref_op2 = ComposedOp([
            mdl_local.operation_blks['layers']['{auto_global_idle}'],
            EmbeddedOp(mdl_local.state_space, ('qb1',),
                       mdl_local.operation_blks['gates']['Gx'])
        ])
        self.assertEqual(str(op), str(ref_op))
        self.assertEqual(str(op), str(ref_op2))

    def test_localnoise_NQ_global_idle(self):
        nQubits = 2
        noisy_idle = 0.9 * np.identity(4**nQubits, 'd')
        noisy_idle[0, 0] = 1.0
        noisy_idle = ExpErrorgenOp(LindbladErrorgen.from_operation_matrix(
            noisy_idle, "H+S"
        ))

        qubit_labels = ['qb{}'.format(i) for i in range(nQubits)]
        pspec_2Q = QubitProcessorSpec(
            nQubits, ('Gx', 'Gy', 'Gcnot', 'Gidle'), geometry="line",
            qubit_labels=qubit_labels
        )

        mdl_local = create_crosstalk_free_model(
            pspec_2Q, {'Gidle': noisy_idle},
            ideal_gate_type='H+S', ideal_spam_type="exp(H+S)",
            independent_gates=False, ensure_composed_gates=False,
            implicit_idle_mode='add_global'
        )

        self.assertEqual(set(mdl_local.operation_blks['gates'].keys()),
                         set(["Gx", "Gy", "Gcnot", "Gidle"]))
        self.assertEqual(set(mdl_local.operation_blks['layers'].keys()), set([
            ('Gx', 'qb0'), ('Gx', 'qb1'), ('Gy', 'qb0'), ('Gy', 'qb1'),
            ('Gcnot', 'qb0', 'qb1'), ('Gcnot', 'qb1', 'qb0'), 'Gidle'
        ]))
        test_circuit = (('Gx', 'qb0'), ('Gcnot', 'qb0', 'qb1'), [],
                        [('Gx', 'qb1'), ('Gy', 'qb0')])
        probs = mdl_local.probabilities(test_circuit)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertAlmostEqual(probs['00'], 0.414025)
        self.assertEqual(mdl_local.num_params, 96)

        op = mdl_local.circuit_layer_operator(Label('Gx', 'qb1'))
        ref_op = ComposedOp([
            mdl_local.operation_blks['layers'][Label('Gidle')],
            EmbeddedOp(mdl_local.state_space, ('qb1',),
                       mdl_local.operation_blks['gates']['Gx'])
        ])
        self.assertEqual(str(op), str(ref_op))

    def test_marginalized_povm(self):
        mdl_local = create_crosstalk_free_model(
            self.pspec_4Q, ideal_gate_type='H+S', independent_gates=True,
            ensure_composed_gates=False
        )

        c = Circuit([('Gx', 'qb0'), ('Gx', 'qb1'), ('Gx', 'qb2'), ('Gx', 'qb3')],
                    num_lines=4)
        prob = mdl_local.probabilities(c)
        self.assertEqual(len(prob), 16)  # Full 4 qubit space

        c2 = Circuit([('Gx', 'qb0'), ('Gx', 'qb1')], num_lines=2)
        prob2 = mdl_local.probabilities(c2)
        self.assertEqual(len(prob2), 4)  # Full 2 qubit space

        c3 = Circuit([('Gx', 'qb0'), ('Gx', 'qb1')], editable=True)
        c3.insert_idling_lines_inplace(None, ['qb2', 'qb3'])
        c3.done_editing()
        prob3 = mdl_local.probabilities(c3)
        self.assertEqual(len(prob3), 16)  # Full 4 qubit space

    def test_getitem(self):
        mdl_local = create_crosstalk_free_model(
            self.pspec_2Q, ideal_gate_type='H+S',
            ideal_spam_type='tensor product H+S', independent_gates=False,
            ensure_composed_gates=False
        )

        # Test getting a prep
        self.assertIs(mdl_local['rho0'],
                      mdl_local.prep_blks['layers']['rho0'])

        # Test getting a POVM
        self.assertIs(mdl_local['Mdefault'],
                      mdl_local.povm_blks['layers']['Mdefault'])

        # Test getting a gate from operation_blks['gates']
        self.assertIs(mdl_local['Gx'],
                      mdl_local.operation_blks['gates']['Gx'])

        # Test getting a layer from operation_blks['layers']
        self.assertIs(mdl_local[('Gx', 'qb0')],
                      mdl_local.operation_blks['layers'][('Gx', 'qb0')])

        # Test getting a layer using a string with a colon
        self.assertIs(mdl_local['Gx:qb0'],
                      mdl_local.operation_blks['layers'][('Gx', 'qb0')])


class ToExplicitModelTester(BaseCase):
    """Tests for ImplicitOpModel.to_explicit_model via LocalNoiseModel."""

    def setUp(self):
        self.pspec = QubitProcessorSpec(2, ('Gx', 'Gy', 'Gcnot'),
                                        geometry="line",
                                        qubit_labels=['qb0', 'qb1'])
        self.test_circuits = [
            Circuit([('Gx', 'qb0'), ('Gy', 'qb1')],
                    line_labels=('qb0', 'qb1')),
            Circuit([('Gcnot', 'qb0', 'qb1')], line_labels=('qb0', 'qb1')),
            Circuit([('Gx', 'qb0'), ('Gcnot', 'qb0', 'qb1'),
                     ('Gy', 'qb1'), ('Gx', 'qb1')],
                    line_labels=('qb0', 'qb1')),
            Circuit([('Gy', 'qb0'), ('Gy', 'qb0'), ('Gcnot', 'qb1', 'qb0')],
                    line_labels=('qb0', 'qb1')),
        ]

    def _assert_probs_match(self, mdl_a, mdl_b, places=9):
        for c in self.test_circuits:
            pa, pb = mdl_a.probabilities(c), mdl_b.probabilities(c)
            self.assertEqual(set(pa.keys()), set(pb.keys()))
            for outcome, prob in pa.items():
                self.assertAlmostEqual(prob, pb[outcome], places=places)

    def test_returns_explicit_model_with_expected_members(self):
        mdl = create_crosstalk_free_model(
            self.pspec, ideal_gate_type='H+S',
            depolarization_strengths={'Gx': 0.01}
        )
        exp = mdl.to_explicit_model()

        self.assertIsInstance(exp, ExplicitOpModel)
        self.assertEqual(exp.state_space, mdl.state_space)
        # every stored layer operation/prep/povm is carried over
        self.assertEqual(set(exp.operations.keys()),
                         set(mdl.operation_blks['layers'].keys()))
        self.assertEqual(set(exp.preps.keys()),
                         set(mdl.prep_blks['layers'].keys()))
        self.assertEqual(set(exp.povms.keys()),
                         set(mdl.povm_blks['layers'].keys()))

    def test_reproduces_probabilities(self):
        mdl = create_crosstalk_free_model(
            self.pspec, ideal_gate_type='H+S',
            depolarization_strengths={'Gx': 0.05, 'Gcnot': 0.1},
            lindblad_error_coeffs={'Gy': {('H', 'Z'): 0.02}}
        )
        exp = mdl.to_explicit_model()
        self._assert_probs_match(mdl, exp)

    def test_preserves_parameterization(self):
        # An H+S model has parameterized (error-generator) ops; conversion
        # must not collapse them to static dense operators.
        mdl = create_crosstalk_free_model(
            self.pspec, ideal_gate_type='H+S',
            lindblad_error_coeffs={'Gy': {('H', 'Z'): 0.02}}
        )
        exp = mdl.to_explicit_model()

        self.assertGreater(mdl.num_params, 0)
        self.assertEqual(exp.num_params, mdl.num_params)

        gy = exp.operations[Label('Gy', 'qb0')]
        self.assertNotIsInstance(gy, StaticArbitraryOp)
        # error-generator coefficients survive and match the implicit model's
        imp_coeffs = mdl.operation_blks['layers'][Label('Gy', 'qb0')].errorgen_coefficients()
        exp_coeffs = gy.errorgen_coefficients()
        self.assertEqual(set(imp_coeffs.keys()), set(exp_coeffs.keys()))
        for k in imp_coeffs:
            self.assertAlmostEqual(imp_coeffs[k], exp_coeffs[k])

    def test_preserves_parameter_sharing(self):
        # With independent_gates=False a single base op is embedded on multiple
        # qubits; the shared `memo` in to_explicit_model must keep that sharing
        # so the parameter count does not inflate.
        shared = create_crosstalk_free_model(
            self.pspec, ideal_gate_type='H+S', independent_gates=False
        )
        indep = create_crosstalk_free_model(
            self.pspec, ideal_gate_type='H+S', independent_gates=True
        )
        exp_shared = shared.to_explicit_model()
        exp_indep = indep.to_explicit_model()

        self.assertEqual(exp_shared.num_params, shared.num_params)
        self.assertEqual(exp_indep.num_params, indep.num_params)
        # sharing genuinely reduces the parameter count (and survives conv)
        self.assertLess(exp_shared.num_params, exp_indep.num_params)

    def test_captures_nontrivial_parameter_values(self):
        # Perturb the implicit model's parameters, then convert: the explicit
        # model must reflect the perturbed values (not just the ideal gates).
        mdl = create_crosstalk_free_model(
            self.pspec, ideal_gate_type='H+S', independent_gates=True
        )
        rng = np.random.default_rng(12345)
        v = mdl.to_vector()
        mdl.from_vector(v + 0.01 * rng.standard_normal(len(v)))
        exp = mdl.to_explicit_model()
        self._assert_probs_match(mdl, exp)

    def test_handles_instruments(self):
        # Instruments in (crosstalk-free) implicit models act on all qudits,
        # so they are keyed without subsystem labels and stored full-dimension.
        pspec_1Q = QubitProcessorSpec(1, ('Gx', 'Gy'), qubit_labels=['qb0'])
        mdl = create_crosstalk_free_model(pspec_1Q, ideal_gate_type='static')
        inst = Instrument({'p0': np.diag([1., 0, 0, 0]),
                           'p1': np.diag([0, 0, 0, 1.])})
        mdl.instrument_blks['layers'][Label('Iz')] = inst
        exp = mdl.to_explicit_model()

        self.assertIn(Label('Iz'), exp.instruments)
        self.assertIsInstance(exp.instruments[Label('Iz')], Instrument)

    def test_raises_on_factories(self):
        mdl = create_crosstalk_free_model(self.pspec,
                                           ideal_gate_type='static')
        # inject any object into a factories block to trigger the guard
        some_op = next(iter(mdl.operation_blks['gates'].values()))
        mdl.factories['gates'][Label('Gfoo')] = some_op
        with self.assertRaises(ValueError):
            mdl.to_explicit_model()

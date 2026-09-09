"""Tests for the DesignReducer interface in pygsti.tools.edesign._reduction.

This covers the *contract*, not any particular reduction rule: what a subclass has to
return, and what happens when it returns something else. The reducers here are
deliberately trivial (take the first k, take a fixed list) so that a failure points at
the base class rather than at a selection algorithm.

The D-optimal reducer built on this interface is tested in test_blockdopt.py.
"""
import numpy as np

from pygsti.circuits.circuit import Circuit
from pygsti.protocols import CircuitListsDesign
from pygsti.tools.edesign import CallableReducer, CircuitSelection, DesignReducer

from ..util import BaseCase


def _circuits(n):
    """`n` distinct one-qubit circuits."""
    return [Circuit([('Gxpi2', 0)] * (i + 1), line_labels=(0,)) for i in range(n)]


def _design(n=10):
    circuits = _circuits(n)
    return CircuitListsDesign([circuits[:n // 2], circuits], nested=True)


def _take_the_first_two(design, num_circuits):
    """A module-level (hence importable) reducer function, for the CallableReducer tests."""
    return list(design.all_circuits_needing_data)[:2]


class _FirstK(DesignReducer):
    """Keeps the first `num_circuits` candidates in the design's own order."""

    def _select(self, design, num_circuits):
        chosen = list(design.all_circuits_needing_data)
        if num_circuits is not None:
            chosen = chosen[:num_circuits]
        return CircuitSelection(chosen, scores=np.arange(len(chosen), dtype=float),
                                score_name='position')


class _ReturnsWhatever(DesignReducer):
    """Returns exactly what it was constructed with, so a test can hand `_select` anything."""

    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def _select(self, design, num_circuits):
        return self.payload


class SelectionTester(BaseCase):
    def test_a_selection_reports_its_length_and_keeps_circuit_order(self):
        circuits = _circuits(4)
        selection = CircuitSelection(circuits)
        self.assertEqual(len(selection), 4)
        self.assertEqual(list(selection.circuits), circuits)
        self.assertIsNone(selection.scores)
        self.assertEqual(selection.metadata, {})
        self.assertIsNone(selection.reducer)

    def test_scores_become_a_float_array(self):
        selection = CircuitSelection(_circuits(3), scores=[1, 2, 3])
        self.assertIsInstance(selection.scores, np.ndarray)
        self.assertEqual(selection.scores.dtype, np.float64)

    def test_metadata_is_copied_not_aliased(self):
        """Otherwise two selections built from one dict share diagnostics."""
        shared = {'note': 'original'}
        selection = CircuitSelection(_circuits(2), metadata=shared)
        selection.metadata['note'] = 'changed'
        self.assertEqual(shared['note'], 'original')


class SelectContractTester(BaseCase):
    """What `select` guarantees on top of whatever `_select` did."""

    def setUp(self):
        super().setUp()
        self.design = _design(10)

    def test_the_selection_is_stamped_with_the_reducer_and_the_candidate_count(self):
        reducer = _FirstK()
        selection = reducer.select(self.design, 4)
        self.assertIs(selection.reducer, reducer)
        self.assertEqual(selection.metadata['num_candidates'], 10)

    def test_a_reducer_may_report_its_own_candidate_count(self):
        """`setdefault`, not `[]=`: a reducer that counts differently is not overwritten."""
        payload = CircuitSelection(_circuits(2), metadata={'num_candidates': 99})
        selection = _ReturnsWhatever(payload).select(self.design, 2)
        self.assertEqual(selection.metadata['num_candidates'], 99)

    def test_the_budget_is_clamped_to_the_candidate_count(self):
        selection = _FirstK().select(self.design, 10 ** 6)
        self.assertEqual(len(selection), 10)

    def test_no_budget_means_the_reducer_decides(self):
        self.assertEqual(len(_FirstK().select(self.design)), 10)

    def test_a_negative_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            _FirstK().select(self.design, -1)

    def test_a_zero_budget_is_allowed_and_gives_nothing(self):
        """Distinct from a reducer that returned nothing when asked for some."""
        self.assertEqual(len(_FirstK().select(self.design, 0)), 0)


class ValidationTester(BaseCase):
    """Each of these is a mistake a third-party subclass can make."""

    def setUp(self):
        super().setUp()
        self.design = _design(10)

    def _assert_raises_naming_the_subclass(self, payload, exc=ValueError):
        with self.assertRaises(exc) as ctx:
            _ReturnsWhatever(payload).select(self.design, 5)
        self.assertIn('_ReturnsWhatever', str(ctx.exception))
        return str(ctx.exception)

    def test_returning_a_bare_list_is_a_typeerror_that_says_how_to_fix_it(self):
        message = self._assert_raises_naming_the_subclass(_circuits(3), exc=TypeError)
        self.assertIn('CircuitSelection', message)

    def test_returning_more_circuits_than_the_budget_is_rejected(self):
        self._assert_raises_naming_the_subclass(CircuitSelection(_circuits(6)))

    def test_returning_fewer_circuits_than_the_budget_is_allowed(self):
        """A reducer may run out of anything worth adding before the budget is spent."""
        selection = _ReturnsWhatever(CircuitSelection(_circuits(2))).select(self.design, 5)
        self.assertEqual(len(selection), 2)

    def test_returning_a_duplicate_is_rejected(self):
        repeated = _circuits(3) + _circuits(1)
        message = self._assert_raises_naming_the_subclass(CircuitSelection(repeated))
        self.assertIn('more than once', message)

    def test_returning_a_circuit_from_outside_the_design_is_rejected(self):
        foreign = Circuit([('Gypi2', 0)] * 17, line_labels=(0,))
        payload = CircuitSelection(_circuits(2) + [foreign])
        message = self._assert_raises_naming_the_subclass(payload)
        self.assertIn('not in the design', message)

    def test_the_foreign_circuit_message_names_the_bulk_dprobs_hazard(self):
        """The specific way to produce this bug is worth naming in the message.

        `bulk_dprobs` deduplicates and reorders, so a reducer that maps Jacobian row
        blocks back through its own input list returns real circuits in wrong positions,
        and every downstream check passes.
        """
        payload = CircuitSelection([Circuit([('Gypi2', 0)] * 17, line_labels=(0,))])
        message = self._assert_raises_naming_the_subclass(payload)
        self.assertIn('jac_dict', message)

    def test_misaligned_scores_are_rejected(self):
        payload = CircuitSelection(_circuits(3), scores=[1.0, 2.0])
        message = self._assert_raises_naming_the_subclass(payload)
        self.assertIn('scores', message)

    def test_returning_nothing_when_asked_for_something_is_rejected(self):
        message = self._assert_raises_naming_the_subclass(CircuitSelection([]))
        self.assertIn('no circuits', message)

    def test_an_unimplemented_select_says_what_to_implement(self):
        with self.assertRaises(NotImplementedError) as ctx:
            DesignReducer().select(self.design, 3)
        self.assertIn('_select', str(ctx.exception))


class ReduceTester(BaseCase):
    def setUp(self):
        super().setUp()
        self.design = _design(10)

    def test_reduce_keeps_the_design_class_and_the_chosen_circuits(self):
        reduced = _FirstK().reduce(self.design, 4)
        self.assertIsInstance(reduced, CircuitListsDesign)
        self.assertEqual(set(reduced.all_circuits_needing_data), set(_circuits(4)))

    def test_reduce_does_not_modify_the_original(self):
        before = [list(cl) for cl in self.design.circuit_lists]
        _FirstK().reduce(self.design, 3)
        self.assertEqual([list(cl) for cl in self.design.circuit_lists], before)

    def test_reduce_validates_too(self):
        """`reduce` goes through `select`, so a bad subclass cannot reach truncation."""
        with self.assertRaises(ValueError):
            _ReturnsWhatever(CircuitSelection(_circuits(9))).reduce(self.design, 2)


class CastTester(BaseCase):
    def setUp(self):
        super().setUp()
        self.design = _design(10)

    def test_a_reducer_casts_to_itself(self):
        reducer = _FirstK()
        self.assertIs(DesignReducer.cast(reducer), reducer)

    def test_a_callable_returning_circuits_is_wrapped(self):
        reducer = DesignReducer.cast(
            lambda design, n: list(design.all_circuits_needing_data)[:n])
        self.assertIsInstance(reducer, CallableReducer)
        self.assertEqual(len(reducer.select(self.design, 3)), 3)

    def test_a_callable_returning_a_selection_is_passed_through(self):
        reducer = DesignReducer.cast(
            lambda design, n: CircuitSelection(_circuits(n), score_name='mine'))
        self.assertEqual(reducer.select(self.design, 3).score_name, 'mine')

    def test_a_callable_reducer_is_validated_like_any_other(self):
        reducer = DesignReducer.cast(lambda design, n: _circuits(2) + _circuits(1))
        with self.assertRaises(ValueError):
            reducer.select(self.design, 5)

    def test_casting_a_non_callable_says_what_is_accepted(self):
        with self.assertRaises(TypeError) as ctx:
            DesignReducer.cast('dopt')
        self.assertIn('callable', str(ctx.exception))


class SerializationTester(BaseCase):
    """The claim the NicelySerializable base is here to support."""

    def test_a_third_party_subclass_round_trips_without_being_registered(self):
        reducer = _FirstK()
        restored = DesignReducer.from_nice_serialization(reducer.to_nice_serialization())
        self.assertIsInstance(restored, _FirstK)
        self.assertEqual(len(restored.select(_design(10), 4)), 4)

    def test_a_selection_round_trips_with_its_scores_and_its_reducer(self):
        selection = _FirstK().select(_design(6), 4)
        restored = CircuitSelection.from_nice_serialization(selection.to_nice_serialization())
        self.assertEqual(list(restored.circuits), list(selection.circuits))
        self.assertArraysAlmostEqual(restored.scores, selection.scores)
        self.assertEqual(restored.score_name, 'position')
        self.assertEqual(restored.metadata['num_candidates'], 6)
        self.assertIsInstance(restored.reducer, _FirstK)

    def test_a_selection_with_no_scores_round_trips(self):
        selection = CircuitSelection(_circuits(3))
        restored = CircuitSelection.from_nice_serialization(selection.to_nice_serialization())
        self.assertIsNone(restored.scores)
        self.assertIsNone(restored.reducer)

    def test_a_lambda_backed_callable_reducer_does_not_round_trip(self):
        """Documented limitation, pinned so it stays documented.

        A lambda has no importable name, so this is the price of the escape hatch --
        and the reason a durable reducer should be a DesignReducer subclass.
        """
        reducer = CallableReducer(lambda design, n: [])
        state = reducer.to_nice_serialization()
        with self.assertRaises(ValueError) as ctx:
            DesignReducer.from_nice_serialization(state)
        # The raw import failure names a dotted path that is not the real problem.
        self.assertIn('lambda', str(ctx.exception))
        self.assertIn('DesignReducer', str(ctx.exception))

    def test_a_module_level_function_backed_callable_reducer_does_round_trip(self):
        """The escape hatch is not useless -- only lambdas and closures are excluded."""
        reducer = CallableReducer(_take_the_first_two)
        restored = DesignReducer.from_nice_serialization(reducer.to_nice_serialization())
        self.assertIsInstance(restored, CallableReducer)
        self.assertEqual(len(restored.select(_design(10), 5)), 2)

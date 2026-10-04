"""
Tests of the deprecation of TermForwardSimulator and its supporting term machinery (see #916).

Remove this file (together with the deprecated code) when the deprecation expires in pyGSTi 0.12;
see agent-docs/known-debt.md.
"""
import warnings

import pytest

import pygsti
from pygsti.baseobjs import Basis, QubitSpace
from pygsti.circuits import Circuit
from pygsti.forwardsims import TermForwardSimulator
from pygsti.modelpacks import smq1Q_XY as std
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
from ..util import BaseCase


def _deprecation_messages(fn):
    """ Run `fn` and return the messages of all pyGSTiDeprecationWarnings it issues. """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return [str(r.message) for r in rec if issubclass(r.category, pyGSTiDeprecationWarning)]


class TermSimDeprecationTester(BaseCase):

    def test_construction_warns(self):
        with pytest.warns(pyGSTiDeprecationWarning, match=r"TermForwardSimulator is deprecated.*0\.12.*916"):
            TermForwardSimulator(mode='taylor-order', max_order=1)

    def test_deserialization_warns_once(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyGSTiDeprecationWarning)
            state = TermForwardSimulator(mode='pruned')._to_nice_serialization()
        msgs = _deprecation_messages(lambda: TermForwardSimulator._from_nice_serialization(state))
        self.assertEqual(len(msgs), 1)
        self.assertIn("TermForwardSimulator", msgs[0])

    def test_simulation_warns_only_at_construction(self):
        def run():
            mdl = std.target_model("H+S", evotype='statevec')
            mdl.sim = TermForwardSimulator(mode='taylor-order', max_order=1)
            c = Circuit([('Gxpi2', 0), ('Gypi2', 0)], line_labels=(0,))
            mdl.probabilities(c)
            mdl.sim.bulk_dprobs([c])
            mdl.copy()  # copying the attached simulator must not re-warn either
        msgs = _deprecation_messages(run)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].startswith("TermForwardSimulator is deprecated"))

    def test_modelmember_term_methods_warn_once(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyGSTiDeprecationWarning)
            op = std.target_model("H+S", evotype='statevec').operations[('Gxpi2', 0)]

        # nested calls into deprecated methods of sub-members are not reported separately
        msgs = _deprecation_messages(lambda: op.taylor_order_terms(1))
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].startswith("ComposedOp.taylor_order_terms is deprecated"))

        msgs = _deprecation_messages(lambda: op.total_term_magnitude)  # a property
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].startswith("ComposedOp.total_term_magnitude is deprecated"))

        msgs = _deprecation_messages(lambda: op.highmagnitude_terms(1e-6))
        self.assertEqual(len(msgs), 1)
        self.assertIn("highmagnitude_terms is deprecated", msgs[0])

    def test_create_lindblad_term_objects_warns(self):
        blk = LindbladCoefficientBlock('ham', Basis.cast('pp', 4), param_mode='elements')
        evotype = pygsti.evotypes.Evotype.cast('statevec')
        with pytest.warns(pyGSTiDeprecationWarning, match="create_lindblad_term_objects is deprecated"):
            blk.create_lindblad_term_objects(0, 100, evotype, QubitSpace(1))

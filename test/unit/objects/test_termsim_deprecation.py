"""
Tests of the deprecation of TermForwardSimulator and its supporting term machinery (see #916).

Remove this file (together with the deprecated code) when the deprecation expires in pyGSTi 0.12;
see agent-docs/known-debt.md.
"""
import os
import re
import subprocess
import sys
import warnings
from unittest import mock

import pytest

import pygsti
from pygsti.tools import legacytools
from pygsti.baseobjs import Basis, QubitSpace
from pygsti.circuits import Circuit
from pygsti.forwardsims import TermForwardSimulator
from pygsti.modelpacks import smq1Q_XY as std
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
from ..util import BaseCase

try:  # use the compiled statevec evotype when available (it is absent in no-Cython installs)
    import pygsti.evotypes.statevec  # noqa: F401
    STATEVEC = 'statevec'
except ImportError:
    STATEVEC = 'statevec_slow'


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
            mdl = std.target_model("H+S", evotype=STATEVEC)
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
            op = std.target_model("H+S", evotype=STATEVEC).operations[('Gxpi2', 0)]

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
        evotype = pygsti.evotypes.Evotype.cast(STATEVEC)
        with pytest.warns(pyGSTiDeprecationWarning, match="create_lindblad_term_objects is deprecated"):
            blk.create_lindblad_term_objects(0, 100, evotype, QubitSpace(1))


_M2_SCRIPT = """\
import pygsti
from pygsti.forwardsims import TermForwardSimulator
from pygsti.modelpacks import smq1Q_XY

mdl = smq1Q_XY.target_model('H+S')
mdl.sim = TermForwardSimulator      # class passed, goes through ForwardSimulator.cast
mdl.write('termsim_model.json')

mdl2 = pygsti.models.Model.read('termsim_model.json')   # load a saved term-sim model
print("done")
"""


class TermSimWarningAttributionTester(BaseCase):

    def test_warnings_attributed_to_caller(self):
        # Indirect paths (cast, Model.read) must be attributed to this file, not to pyGSTi code.
        with self.temp_path('termsim_model.json') as path:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pyGSTiDeprecationWarning)
                mdl = std.target_model("H+S")
                mdl.sim = TermForwardSimulator
                mdl.write(path)

            def construct():
                TermForwardSimulator(mode='taylor-order', max_order=1)

            def cast():
                std.target_model("H+S").sim = TermForwardSimulator

            def read():
                pygsti.models.Model.read(path)

            for flag in ((True, False) if sys.version_info >= (3, 12) else (False,)):
                for call in (construct, cast, read):
                    with self.subTest(skip_file_prefixes=flag, call=call.__name__), \
                            mock.patch.object(legacytools, '_HAS_SKIP_FILE_PREFIXES', flag), \
                            warnings.catch_warnings(record=True) as rec:
                        warnings.simplefilter("always")
                        call()
                        rec = [r for r in rec if issubclass(r.category, pyGSTiDeprecationWarning)]
                        self.assertEqual([r.filename for r in rec], [__file__])

    def test_m2_script_warns_under_default_filters(self):
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONWARNINGS', 'PYTHONDEVMODE')}
        pkg_parent = os.path.dirname(os.path.dirname(pygsti.__file__))
        env['PYTHONPATH'] = os.pathsep.join([pkg_parent] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
        with self.temp_path('ex2.py') as script:
            with open(script, 'w') as f:
                f.write(_M2_SCRIPT)
            proc = subprocess.run([sys.executable, script], cwd=os.path.dirname(script), env=env,
                                  capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        pattern = re.compile(r"^(.*):(\d+): pyGSTiDeprecationWarning: TermForwardSimulator is deprecated")
        hits = [pattern.match(line) for line in proc.stderr.splitlines()]
        found = [(os.path.basename(m.group(1)), int(m.group(2))) for m in hits if m]
        self.assertEqual(found, [('ex2.py', 6), ('ex2.py', 9)])

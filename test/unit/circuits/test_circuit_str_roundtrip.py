"""Property tests for the parse/str round-trip invariant over the bijective grammar
subset: a Circuit built from structured labels must satisfy Circuit(c.str) == c,
under BOTH parser implementations (Cython fastcircuitparser and pure-Python
slowcircuitparser), and via the StdInputParser path that dataset/text loading uses.

The strategy deliberately stays inside the bijective subset: integer sslbls and
line labels, no '^' exponents, no overlapping sslbls within a layer, no time/args
annotations; constructs outside the subset are out of scope.

Summed circuits are covered too, since `+` used to compose the result's string rep
from the operands' cached markers while dropping the indices themselves (#758).

derandomize=True keeps CI deterministic (no flaky example discovery in PR gates).
"""
from unittest import mock

import pytest

hypothesis = pytest.importorskip('hypothesis')
from hypothesis import given, settings, strategies as st  # noqa: E402

import pygsti.circuits.circuitparser as cparser_mod  # noqa: E402
from pygsti.circuits import Circuit  # noqa: E402
from pygsti.circuits.circuitparser import slowcircuitparser  # noqa: E402
from pygsti.io import stdinput  # noqa: E402

from ..util import BaseCase  # noqa: E402

LINES    = (0, 1, 2)
GATES_1Q = ('Gx', 'Gy', 'Gz', 'Gi')
GATES_2Q = ('Gcnot', 'Gcphase')

ROUNDTRIP_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True)


@st.composite
def layer_st(draw):
    """One layer: 1-2 gates on disjoint sslbls drawn from LINES."""
    free = list(LINES)
    labels = []
    n_gates = draw(st.integers(1, 2))
    for _ in range(n_gates):
        use_2q = len(free) >= 2 and draw(st.booleans())
        if use_2q:
            name = draw(st.sampled_from(GATES_2Q))
            q0 = draw(st.sampled_from(free))
            free.remove(q0)
            q1 = draw(st.sampled_from(free))
            free.remove(q1)
            labels.append((name, q0, q1))
        else:
            # free cannot be empty here (3 lines, at most 2 gates); widening n_gates would require restoring a guard
            name = draw(st.sampled_from(GATES_1Q))
            q = draw(st.sampled_from(free))
            free.remove(q)
            labels.append((name, q))
    return labels


@st.composite
def circuit_st(draw):
    n_layers   = draw(st.integers(0, 5))
    layer_list = [draw(layer_st()) for _ in range(n_layers)]
    occurrence = draw(st.one_of(st.none(), st.integers(0, 3)))

    compilable = None
    if n_layers >= 1 and draw(st.booleans()):
        # any non-empty subset, up to and including every layer: the all-compilable
        # case used to render with no marker at all, which is the writer-side sibling
        # of #758 fixed alongside it.  1-layer circuits only have the full set, so
        # excluding it would have left them uncovered entirely.
        index_set  = draw(st.sets(st.integers(0, n_layers - 1), min_size=1, max_size=n_layers))
        compilable = tuple(sorted(index_set))

    c = Circuit(layer_list, line_labels=LINES, occurrence=occurrence,
                compilable_layer_indices=compilable)
    return c


class CircuitStrRoundtripTester(BaseCase):

    @ROUNDTRIP_SETTINGS
    @given(c=circuit_st())
    def test_str_roundtrip_default_parser(self, c):
        c2 = Circuit(c.str)
        self.assertEqual(c2, c)
        # hash/eq consistency witnessed once here; other parser tests rely on == only
        self.assertEqual(hash(c2), hash(c))
        self.assertEqual(c2.str, c.str)

    @ROUNDTRIP_SETTINGS
    @given(c=circuit_st())
    def test_str_roundtrip_slow_parser(self, c):
        slow_parse = slowcircuitparser.parse_circuit
        with mock.patch.object(cparser_mod, 'parse_circuit', slow_parse):
            c2 = Circuit(c.str)
        self.assertEqual(c2, c)
        self.assertEqual(c2.str, c.str)

    @ROUNDTRIP_SETTINGS
    @given(c=circuit_st())
    def test_str_roundtrip_stdinput_parser(self, c):
        # crosses the Circuit._fastinit path; generator output is canonically sorted,
        # so this holds despite the issue #757 bug (which needs *unsorted* source text)
        sip = stdinput.StdInputParser()
        c2 = sip.parse_circuit(c.str, create_subcircuits=False)
        self.assertEqual(c2, c)

    @ROUNDTRIP_SETTINGS
    @given(a=circuit_st(), b=circuit_st())
    def test_str_roundtrip_of_summed_circuits(self, a, b):
        s = a + b
        self.assertEqual(Circuit(s.str), s)
        self.assertEqual(hash(Circuit(s.str)), hash(s))

    @ROUNDTRIP_SETTINGS
    @given(a=circuit_st(), b=circuit_st())
    def test_add_concatenates_compilable_indices(self, a, b):
        expected = a.compilable_layer_indices \
            + tuple(i + len(a) for i in b.compilable_layer_indices)
        self.assertEqual((a + b).compilable_layer_indices, expected)

    # ---- the all-compilable case, by example ----
    #
    # Every layer compilable leaves _op_seq_to_str's uncompilable set empty, which used
    # to take the '|' branch with nothing to mark: the generated string carried no
    # marker and re-parsing gave back compilable_layer_indices == ().  '~' is now
    # preferred whenever the uncompilable set is empty.  Strings that already marked
    # something are byte-identical, since no other case reaches the changed condition.

    def test_all_compilable_circuit_emits_markers(self):
        c = Circuit([('Gx', 0), ('Gy', 0)], line_labels=(0,), compilable_layer_indices=(0, 1))
        self.assertEqual(c.str, 'Gx:0~Gy:0~@(0)')
        self.assertEqual(Circuit(c.str), c)
        self.assertEqual(hash(Circuit(c.str)), hash(c))

    def test_single_layer_compilable_circuit_roundtrips(self):
        # the sharpest case: a 1-layer circuit's only non-empty index set is the full one,
        # so before the fix such a circuit could never round-trip
        c = Circuit([('Gx', 0)], line_labels=(0,), compilable_layer_indices=(0,))
        self.assertEqual(c.str, 'Gx:0~@(0)')
        self.assertEqual(Circuit(c.str), c)

    def test_all_compilable_circuit_roundtrips_after_regenerating_its_string(self):
        # .str is cached from the parse, so a *parsed* all-compilable circuit round-tripped
        # even before the fix -- it handed back its own input.  The loss showed only on
        # paths that regenerate the string rep, of which done_editing is one.
        c = Circuit('Gz~', editable=True)
        c.done_editing()
        self.assertEqual(c.compilable_layer_indices, (0,))
        self.assertEqual(c.str, 'Gz~')
        self.assertEqual(Circuit(c.str), c)

    def test_sum_of_all_compilable_circuits_roundtrips(self):
        # `+` regenerates the string whenever the combined index set is non-empty (#758),
        # so before this fix the summed object had the right indices and a string that
        # lost them -- the one case #758's fix left open
        s = Circuit('Gz~') + Circuit('Gz~')
        self.assertEqual(s.compilable_layer_indices, (0, 1))
        self.assertEqual(Circuit(s.str), s)

    def test_fast_parser_extension_importable(self):
        reason = 'fast parser extension not built; the roundtrip tests above exercised only the slow parser'
        try:
            import pygsti.circuits.circuitparser.fastcircuitparser  # noqa: F401
        except ImportError:
            self.skipTest(reason)

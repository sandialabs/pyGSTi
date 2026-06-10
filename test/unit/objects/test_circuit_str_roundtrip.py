"""Property tests for the parse/str round-trip invariant over the bijective grammar
subset: a Circuit built from structured labels must satisfy Circuit(c.str) == c,
under BOTH parser implementations (Cython fastcircuitparser and pure-Python
slowcircuitparser), and via the StdInputParser path that dataset/text loading uses.

The strategy deliberately stays inside the bijective subset: integer sslbls and
line labels, no '^' exponents, no overlapping sslbls within a layer, no time/args
annotations. Known violations of the invariant are pinned in
test_circuit_known_bugs.py (#758); constructs outside the subset are out of scope.

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
    if n_layers >= 2 and draw(st.booleans()):
        # max_size=n_layers-1 excludes the all-layers-compilable case, a recorded
        # new-issue candidate: _op_seq_to_str emits a '|' after each *uncompilable*
        # layer when that set is smaller, so an empty uncompilable set yields NO
        # marker and Circuit(c.str) silently loses compilable_layer_indices
        # (writer-side sibling of #758; every proper subset round-trips).
        # 1-layer circuits can only have the (excluded) full set.
        index_set  = draw(st.sets(st.integers(0, n_layers - 1), min_size=1, max_size=n_layers - 1))
        compilable = tuple(sorted(index_set))

    c = Circuit(layer_list, line_labels=LINES, occurrence=occurrence,
                compilable_layer_indices=compilable)
    return c


@ROUNDTRIP_SETTINGS
@given(c=circuit_st())
def test_str_roundtrip_default_parser(c):
    c2 = Circuit(c.str)
    assert c2 == c
    assert hash(c2) == hash(c)  # hash/eq consistency witnessed once here; other parser tests rely on == only
    assert c2.str == c.str


@ROUNDTRIP_SETTINGS
@given(c=circuit_st())
def test_str_roundtrip_slow_parser(c):
    slow_parse = slowcircuitparser.parse_circuit
    with mock.patch.object(cparser_mod, 'parse_circuit', slow_parse):
        c2 = Circuit(c.str)
    assert c2 == c
    assert c2.str == c.str


@ROUNDTRIP_SETTINGS
@given(c=circuit_st())
def test_str_roundtrip_stdinput_parser(c):
    # crosses the Circuit._fastinit path; generator output is canonically sorted,
    # so this holds despite the issue #757 bug (which needs *unsorted* source text)
    sip = stdinput.StdInputParser()
    c2 = sip.parse_circuit(c.str, create_subcircuits=False)
    assert c2 == c


def test_fast_parser_extension_importable():
    pytest.importorskip('pygsti.circuits.circuitparser.fastcircuitparser', reason='fast parser extension not built; the roundtrip tests above exercised only the slow parser')

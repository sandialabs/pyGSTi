"""Pins for verified, currently-unfixed Circuit bugs (issues #757, #758, #759, #761).

Each test asserts the CURRENT (buggy) behavior, so the suite documents the bug and
the eventual fix is forced to flip the pin in the same PR that fixes it.
Convention:   # KNOWN BUG, pyGSTi issue #NNN — assertions pin the bug.
Repros taken verbatim from the issue reports (verified by execution 2026-06-10
at develop@47b3dcae5, re-verified here at 3e7dd411e).
If one of these tests goes red after your change, you have probably fixed the
referenced issue: flip or delete the pin in the same PR and note the issue number.
Newly discovered, not-yet-filed bugs are pinned as SURPRISE comments in the
module where they were found, and graduate here once filed.
"""
import pytest

from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.io import stdinput


# ---- KNOWN BUG, pyGSTi issue #757: _fastinit skips the inner-layer sort that
# ---- __init__/done_editing apply, so equality/hash depend on construction path.

def test_757_fastinit_skips_layer_sort():
    c_fast = Circuit._fastinit((Label((('Gy', 1), ('Gx', 0))),), (0, 1), False)
    c_init = Circuit([[('Gy', 1), ('Gx', 0)]], line_labels=(0, 1))
    assert c_fast != c_init            # KNOWN BUG #757: should be equal
    assert hash(c_fast) != hash(c_init)
    assert c_fast.tup != c_init.tup


def test_757_parsed_circuits_differ_from_constructed():
    # production path: stdinput builds every parsed circuit via _fastinit in source order
    sip = stdinput.StdInputParser()
    c_parsed = sip.parse_circuit("[Gy:1Gx:0]@(0,1)", create_subcircuits=False)
    c_built = Circuit("[Gy:1Gx:0]@(0,1)")
    assert c_parsed != c_built         # KNOWN BUG #757
    assert hash(c_parsed) != hash(c_built)


# ---- KNOWN BUG, pyGSTi issue #758: __add__ drops compilable_layer_indices but
# ---- concatenates cached strings including '~' markers, breaking parse(c.str)==c.

def test_758_add_drops_compilable_but_leaks_markers():
    a = Circuit("Gx~Gy@(0)")
    b = Circuit("Gz@(0)")
    s = a + b
    assert s.compilable_layer_indices == ()   # metadata hard-dropped (keep/drop policy matrix: test_circuit_metadata_policy.py)
    assert '~' in s.str                       # KNOWN BUG #758: marker leaked into cached str
    assert Circuit(s.str) != s                # KNOWN BUG #758: round-trip broken


# ---- KNOWN BUG, pyGSTi issue #759: editable layer-slices share sublist objects
# ---- with the parent (latent aliasing; landmine, not active corruption).

def test_759_editable_slice_aliases_parent_sublists():
    p = Circuit("[Gx:0Gy:1][Gz:0]", line_labels=(0, 1), editable=True)
    s = p[0:2]
    assert s._labels[0] is p._labels[0]       # KNOWN BUG #759: same list object


def test_759_editable_copy_shares_nested_compound_label_lists():
    # KNOWN BUG #759: copy(editable=True) copies per-layer lists but the nested
    # lists representing compound labels within a layer remain shared
    inner = Label((('Gx', 0), ('Gy', 1)))
    p = Circuit([[inner, ('Gz', 2)]], line_labels=(0, 1, 2), editable=True)
    q = p.copy(editable=True)
    assert q._labels       is not p._labels        # outer list copied...
    assert q._labels[0]    is not p._labels[0]     # ...per-layer list copied (one level)...
    assert q._labels[0][0] is     p._labels[0][0]  # KNOWN BUG #759: nested list shared


# ---- KNOWN BUG, pyGSTi issue #761: str-setter consistency check uses zip without
# ---- a length check, so a TRUNCATED string rep is silently accepted.

def test_761_str_setter_accepts_truncated_string():
    c = Circuit("GxGy@(0)", editable=True)
    c.str = "Gx@(0)"                          # KNOWN BUG #761: silently accepted
    assert c.str == "Gx@(0)"                  # circuit now lies about itself
    assert len(c) == 2


def test_761_str_setter_extended_string():
    # zip also stops at self._labels: extra parsed layers are unchecked (verified)
    c = Circuit("GxGy@(0)", editable=True)
    c.str = "GxGyGz@(0)"                      # KNOWN BUG #761 (adjacent): silently accepted
    assert c.str == "GxGyGz@(0)"
    assert len(c) == 2


# ---- str-setter accept/reject matrix (the non-buggy rows, pinned for reference)

def test_str_setter_rejects_same_length_mismatch():
    c = Circuit("GxGy@(0)", editable=True)
    with pytest.raises(ValueError, match=r"doesn't evaluate to GxGy@\(0\)"):
        c.str = "GxGz@(0)"


def test_str_setter_accepts_exact_match():
    c = Circuit("GxGy@(0)", editable=True)
    c.str = "GxGy@(0)"
    assert c.str == "GxGy@(0)"


def test_str_setter_refuses_static_circuit():
    c = Circuit("GxGy@(0)")
    with pytest.raises(AssertionError, match="Cannot edit a read-only circuit"):
        c.str = "GxGy@(0)"

"""Pins for verified, currently-unfixed Circuit bugs (issues #757, #758).

Fixed, with the pins flipped or graduated out of this file:
  #759 -> test_circuit_aliasing.py (slice/copy independence under every public
         in-place mutator; the two pins here became the flipped assertions there).
  #761 -> the str-setter accept/reject matrix below; the truncation and extension
         rows now assert the ValueError instead of the silent accept.


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
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.io import stdinput

from ..util import BaseCase


class CircuitKnownBugsTester(BaseCase):

    # ---- KNOWN BUG, pyGSTi issue #757: _fastinit skips the inner-layer sort that
    # ---- __init__/done_editing apply, so equality/hash depend on construction path.

    def test_757_fastinit_skips_layer_sort(self):
        c_fast = Circuit._fastinit((Label((('Gy', 1), ('Gx', 0))),), (0, 1), False)
        c_init = Circuit([[('Gy', 1), ('Gx', 0)]], line_labels=(0, 1))
        self.assertNotEqual(c_fast, c_init)            # KNOWN BUG #757: should be equal
        self.assertNotEqual(hash(c_fast), hash(c_init))
        self.assertNotEqual(c_fast.tup, c_init.tup)

    def test_757_parsed_circuits_differ_from_constructed(self):
        # production path: stdinput builds every parsed circuit via _fastinit in source order
        sip = stdinput.StdInputParser()
        c_parsed = sip.parse_circuit("[Gy:1Gx:0]@(0,1)", create_subcircuits=False)
        c_built = Circuit("[Gy:1Gx:0]@(0,1)")
        self.assertNotEqual(c_parsed, c_built)         # KNOWN BUG #757
        self.assertNotEqual(hash(c_parsed), hash(c_built))

    # ---- KNOWN BUG, pyGSTi issue #758: __add__ drops compilable_layer_indices but
    # ---- concatenates cached strings including '~' markers, breaking parse(c.str)==c.

    def test_758_add_drops_compilable_but_leaks_markers(self):
        a = Circuit("Gx~Gy@(0)")
        b = Circuit("Gz@(0)")
        s = a + b
        self.assertEqual(s.compilable_layer_indices, ())   # metadata hard-dropped (keep/drop policy matrix: test_circuit_metadata_policy.py)
        self.assertIn('~', s.str)                          # KNOWN BUG #758: marker leaked into cached str
        self.assertNotEqual(Circuit(s.str), s)             # KNOWN BUG #758: round-trip broken

    # ---- str-setter accept/reject matrix

    def test_str_setter_rejects_same_length_mismatch(self):
        c = Circuit("GxGy@(0)", editable=True)
        with self.assertRaisesRegex(ValueError, r"doesn't evaluate to GxGy@\(0\)"):
            c.str = "GxGz@(0)"

    def test_str_setter_rejects_truncated_string(self):
        # #761: zip stopped at the shorter sequence, so this was silently accepted
        # and the circuit then reported .str == 'Gx@(0)' while len(c) == 2
        c = Circuit("GxGy@(0)", editable=True)
        with self.assertRaisesRegex(ValueError, r"evaluates to 1 layer\(s\).*number of layers \(2\)"):
            c.str = "Gx@(0)"
        self.assertEqual(c.str, "GxGy@(0)")

    def test_str_setter_rejects_extended_string(self):
        # #761, the other direction: zip also stopped at self._labels
        c = Circuit("GxGy@(0)", editable=True)
        with self.assertRaisesRegex(ValueError, r"evaluates to 3 layer\(s\).*number of layers \(2\)"):
            c.str = "GxGyGz@(0)"
        self.assertEqual(c.str, "GxGy@(0)")

    def test_str_setter_accepts_exact_match(self):
        c = Circuit("GxGy@(0)", editable=True)
        c.str = "GxGy@(0)"
        self.assertEqual(c.str, "GxGy@(0)")

    def test_str_setter_refuses_static_circuit(self):
        c = Circuit("GxGy@(0)")
        with self.assertRaisesRegex(AssertionError, "Cannot edit a read-only circuit"):
            c.str = "GxGy@(0)"

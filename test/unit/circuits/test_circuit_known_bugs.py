"""Pins for verified, currently-unfixed Circuit bugs. None are open right now.

Convention:   # KNOWN BUG, pyGSTi issue #NNN -- assertions pin the *buggy* behavior,
so the suite documents the bug and the eventual fix is forced to flip the pin in the
same PR that fixes it. If one of these tests goes red after your change, you have
probably fixed the referenced issue: flip or delete the pin in the same PR and note
the issue number. Newly discovered, not-yet-filed bugs are pinned as SURPRISE
comments in the module where they were found, and graduate here once filed.

The four bugs this file was created for (#757, #758, #759, #761, pinned by PR #768
against develop@47b3dcae5) are all fixed. Their flipped assertions live with the
behavior they now describe:

  #757 -> test_circuit_identity_contract.py (parsed circuits match constructed ones;
         both parsers agree; `_fastinit`'s canonical-input precondition)
  #758 -> test_circuit_metadata_policy.py (the `add` row of the keep/drop matrix,
         plus the index propagation and string regeneration assertions)
  #759 -> test_circuit_aliasing.py (slice/copy independence under every public
         in-place mutator)
  #761 -> the str-setter accept/reject matrix below, which grew out of those pins
         and is kept here as the one complete statement of that method's contract

The str-setter matrix stays because it has no better home; everything else about
`.str` is spread across the roundtrip and identity-contract files.
"""
from pygsti.circuits import Circuit

from ..util import BaseCase


class CircuitKnownBugsTester(BaseCase):

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

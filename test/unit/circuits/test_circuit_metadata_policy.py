"""Characterization: which Circuit operations preserve vs drop the two non-layer
metadata fields that participate in identity (occurrence, compilable_layer_indices).

The CASES table below is the behavioral contract. It was pinned from code reading
at develop@3e7dd411e and reconciled by execution. Pin convention: KNOWN BUG pins
(behavior with a filed issue) live in test_circuit_known_bugs.py; SURPRISE pins
(newly found, not-yet-filed behavior, like the mul-with-occurrence ValueError
crash below) are pinned in the module where they were found and recorded for
issue filing.

`add` used to drop compilable_layer_indices while leaving the '~'/'|' markers in
the composed string rep, so a summed circuit disagreed with its own string (#758).
It now propagates them; the shifting semantics are asserted below the table.
"""
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit

from ..util import BaseCase


def _copy_editable_roundtrip(c):
    e = c.copy(editable=True)
    e.done_editing()
    return e


def _add_tail(c):
    return c + Circuit([('Gz', 0)], line_labels=(0,))


# (case id,                   operation,                                  occ kept?, cmp kept?)
CASES = [
    ('copy_static',             lambda c: c.copy(),                         True,    True ),
    ('copy_editable_roundtrip', _copy_editable_roundtrip,                   True,    True ),
    ('add',                     _add_tail,                                  False,   True ),
    ('getitem_layer_slice',     lambda c: c[0:2],                           False,   False),
    ('serialize',               lambda c: c.serialize(),                    True,    False),
    ('parallelize',             lambda c: c.parallelize(),                  True,    False),
    # SURPRISE: replace_gatename keeps occurrence but drops compilable_layer_indices.
    ('replace_gatename',        lambda c: c.replace_gatename('Gx', 'Ga'),   True,    False),
    # SURPRISE: same keep-occurrence/drop-compilable asymmetry as replace_gatename
    ('map_state_space_labels',   lambda c: c.map_state_space_labels({0: 1}),  True,    False),
]


class CircuitMetadataPolicyTester(BaseCase):

    def _base(self):
        layer_list = [('Gx', 0), ('Gy', 0), ('Gz', 0)]
        c = Circuit(layer_list, line_labels=(0,), occurrence=7, compilable_layer_indices=(1,))
        self.assertEqual(c.occurrence, 7)
        self.assertEqual(c.compilable_layer_indices, (1,))
        return c

    def test_metadata_policy(self):
        for case_id, operation, occ_kept, cmp_kept in CASES:
            with self.subTest(case=case_id):
                out = operation(self._base())
                self.assertEqual(out.occurrence, 7 if occ_kept else None)
                self.assertEqual(out.compilable_layer_indices, (1,) if cmp_kept else ())

    # ---- how `+` propagates compilable_layer_indices (#758) ----

    def test_add_shifts_the_right_operands_indices(self):
        a = Circuit("Gx~Gy@(0)")   # 2 layers, layer 0 compilable
        b = Circuit("Gz~Gi@(0)")   # 2 layers, layer 0 compilable
        self.assertEqual((a + b).compilable_layer_indices, (0, 2))
        self.assertEqual((a + b + a).compilable_layer_indices, (0, 2, 4))

    def test_add_with_label_tuple_propagates_on_both_sides(self):
        c = Circuit("Gx~Gy@(0)")
        self.assertEqual((c + (Label('Gz', 0),)).compilable_layer_indices, (0,))
        self.assertEqual(((Label('Gz', 0),) + c).compilable_layer_indices, (1,))

    def test_summed_circuit_agrees_with_its_own_string_rep(self):
        # #758: the composed string kept the '~' marker while the object dropped the
        # indices, so re-parsing produced a different circuit
        a = Circuit("Gx~Gy@(0)")
        b = Circuit("Gz@(0)")
        s = a + b
        self.assertEqual(Circuit(s.str), s)
        self.assertEqual(hash(Circuit(s.str)), hash(s))

    def test_add_regenerates_string_rather_than_splicing_mixed_markers(self):
        # _op_seq_to_str marks whichever set is smaller, so one operand can render
        # with '~' and the other with '|'; splicing those yields a string the parser
        # rejects outright ("contains both barrier and compilable layer joining")
        a = Circuit([('Gx', 0), ('Gy', 0)], line_labels=(0,), compilable_layer_indices=(0,))
        b = Circuit([('Gz', 0), ('Gi', 0), ('Gx', 0)], line_labels=(0,),
                    compilable_layer_indices=(0, 1))
        self.assertIn('~', a.str)
        self.assertIn('|', b.str)
        s = a + b
        self.assertEqual(s.compilable_layer_indices, (0, 2, 3))
        self.assertEqual(Circuit(s.str), s)

    def test_add_without_compilable_indices_keeps_the_spliced_string(self):
        # the common path is unchanged: no indices means no markers to reconcile,
        # so the operands' cached strings are still concatenated verbatim
        u = Circuit("GxGy@(0)")
        v = Circuit("GzGi@(0)")
        self.assertEqual((u + v).str, "GxGyGzGi@(0)")

    def test_mul_repeat_raises_when_occurrence_is_set(self):
        # SURPRISE: Circuit.repeat (hence __mul__) parses self.str via str.split('@')
        # and unpacks exactly two parts.  An occurrence id appends a second '@'
        # separator (e.g. 'Gx:0@(0)@7'), so multiplying any circuit that has an
        # occurrence set raises ValueError instead of returning a circuit.
        c = self._base()
        with self.assertRaisesRegex(ValueError, r"too many values to unpack"):
            c * 2

    def test_mul_repeat_drops_compilable_indices_when_no_occurrence(self):
        # Companion pin: with occurrence unset, __mul__ works and drops the
        # compilable_layer_indices metadata.
        c = Circuit([('Gx', 0), ('Gy', 0)], line_labels=(0,), compilable_layer_indices=(1,))
        out = c * 2
        self.assertIsNone(out.occurrence)
        self.assertEqual(out.compilable_layer_indices, ())

    def test_getitem_single_layer_returns_label_not_circuit(self):
        out = self._base()[1]
        self.assertIsInstance(out, Label)
        self.assertNotIsInstance(out, Circuit)

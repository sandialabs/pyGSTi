"""Characterization: which Circuit operations preserve vs drop the two non-layer
metadata fields that participate in identity (occurrence, compilable_layer_indices).

The CASES table below is the behavioral contract. It was pinned from code reading
at develop@3e7dd411e and reconciled by execution. Pin convention: KNOWN BUG pins
(behavior with a filed issue, e.g. __add__ dropping metadata while leaking string
markers) live in test_circuit_known_bugs.py; SURPRISE pins (newly found,
not-yet-filed behavior, like the mul-with-occurrence ValueError crash below) are
pinned in the module where they were found and recorded for issue filing.
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
    ('add',                     _add_tail,                                  False,   False),
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

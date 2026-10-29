"""Independence of editable Circuit slices and copies from their parent (issue #759).

Editable circuits store each layer as a mutable list, so a derived circuit that
shares those lists with its parent lets an in-place editor on one silently rewrite
the other.  Before the #759 fix, `p[0:3]` shared the parent's per-layer lists
outright and `p.copy(editable=True)` shared the nested lists used for compound
labels; `p[0:3].compress_depth_inplace()` corrupted `p`.

The matrix below is the contract: for every public in-place mutator, editing a
slice or a copy must leave the parent byte-identical.  `_labels` is compared
structurally rather than through `.str`, so a reordering within a layer counts as
a change even when the rendered circuit would look the same.
"""
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit

from ..util import BaseCase


def _snapshot(circuit):
    """Structural, hashable copy of an editable circuit's nested `_labels`."""
    def rec(el):
        return tuple(rec(x) for x in el) if isinstance(el, list) else el
    return rec(circuit._labels), tuple(circuit._line_labels)


def _parent():
    return Circuit("[Gx:0Gy:1][Gz:0][Gy:1]", line_labels=(0, 1), editable=True)


# (case id, mutator) -- every entry must run to completion on the circuit above;
# a mutator that raises fails the test rather than silently proving nothing.
MUTATORS = [
    ('compress_depth_inplace',        lambda c: c.compress_depth_inplace()),
    ('delete_idle_layers_inplace',    lambda c: c.delete_idle_layers_inplace()),
    ('sort_layer_labels_inplace',     lambda c: c.sort_layer_labels_inplace()),
    ('reverse_inplace',               lambda c: c.reverse_inplace()),
    ('clear_labels',                  lambda c: c.clear_labels(layers=[0], lines=[0])),
    ('clear',                         lambda c: c.clear()),
    ('delete_layers',                 lambda c: c.delete_layers([1])),
    ('delete_lines',                  lambda c: c.delete_lines([1], delete_straddlers=True)),
    ('set_labels',                    lambda c: c.set_labels(('Gi', 0), layers=1, lines=0)),
    ('insert_idling_layers_inplace',  lambda c: c.insert_idling_layers_inplace(1, 2)),
    ('insert_layer_inplace',          lambda c: c.insert_layer_inplace(Label('Gi', 0), 1)),
    ('insert_idling_lines_inplace',   lambda c: c.insert_idling_lines_inplace(None, [2])),
    ('replace_gatename_inplace',      lambda c: c.replace_gatename_inplace('Gx', 'Ga')),
    ('replace_gatename_with_idle',    lambda c: c.replace_gatename_with_idle_inplace('Gx')),
    ('map_names_inplace',             lambda c: c.map_names_inplace(lambda n: n + 'q')),
    ('map_state_space_labels_inplace', lambda c: c.map_state_space_labels_inplace({0: 1, 1: 0})),
    ('reorder_lines_inplace',         lambda c: c.reorder_lines_inplace([1, 0])),
    ('delete_idling_lines_inplace',   lambda c: c.delete_idling_lines_inplace()),
    ('expand_subcircuits_inplace',    lambda c: c.expand_subcircuits_inplace()),
    ('insert_implicit_idles_inplace', lambda c: c.insert_implicit_idles_inplace()),
    ('append_circuit_inplace',        lambda c: c.append_circuit_inplace(
                                          Circuit([('Gz', 1)], line_labels=(0, 1)))),
    ('insert_circuit_inplace',        lambda c: c.insert_circuit_inplace(
                                          Circuit([('Gz', 1)], line_labels=(0, 1)), 1)),
    ('replace_layer_with_circuit',    lambda c: c.replace_layer_with_circuit_inplace(
                                          Circuit([('Gz', 1)], line_labels=(0, 1)), 1)),
    ('done_editing',                  lambda c: c.done_editing()),
]


class CircuitAliasingTester(BaseCase):

    def _assert_parent_untouched(self, derive):
        for case_id, mutate in MUTATORS:
            with self.subTest(case=case_id):
                p = _parent()
                before = _snapshot(p)
                mutate(derive(p))
                self.assertEqual(_snapshot(p), before)

    def test_slice_edits_do_not_reach_parent(self):
        self._assert_parent_untouched(lambda p: p[0:3])

    def test_partial_slice_edits_do_not_reach_parent(self):
        self._assert_parent_untouched(lambda p: p[1:3])

    def test_copy_edits_do_not_reach_parent(self):
        self._assert_parent_untouched(lambda p: p.copy(editable=True))

    # ---- the specific aliasing the issue reported, asserted directly ----

    def test_editable_slice_does_not_alias_parent_sublists(self):
        p = Circuit("[Gx:0Gy:1][Gz:0]", line_labels=(0, 1), editable=True)
        s = p[0:2]
        self.assertIsNot(s._labels[0], p._labels[0])
        self.assertEqual(s._labels[0], p._labels[0])

    def test_editable_copy_does_not_share_nested_compound_label_lists(self):
        inner = Label((('Gx', 0), ('Gy', 1)))
        p = Circuit([[inner, ('Gz', 2)]], line_labels=(0, 1, 2), editable=True)
        q = p.copy(editable=True)
        self.assertIsNot(q._labels, p._labels)              # outer list
        self.assertIsNot(q._labels[0], p._labels[0])        # per-layer list
        self.assertIsNot(q._labels[0][0], p._labels[0][0])  # nested compound list
        self.assertEqual(q._labels[0][0], p._labels[0][0])

    def test_compress_depth_on_slice_leaves_parent_intact(self):
        # the witness that made #759 an active-corruption bug rather than a landmine:
        # two public calls, and the parent came back a different circuit
        p = Circuit("[Gx:0Gy:1][Gz:0][Gy:1]", line_labels=(0, 1), editable=True)
        s = p[0:3]
        s.compress_depth_inplace()
        self.assertEqual(p.copy(editable=False).str, "[Gx:0Gy:1]Gz:0Gy:1@(0,1)")
        self.assertEqual(s.copy(editable=False).str, "[Gx:0Gy:1][Gz:0Gy:1]@(0,1)")

    def test_static_paths_still_share_immutable_labels(self):
        # the fix is scoped to the editable tier; static circuits hold immutable
        # Labels in tuples, and those are still shared (deliberately -- it is what
        # makes static copy ~0.2us)
        p = Circuit("[Gx:0Gy:1][Gz:0]", line_labels=(0, 1))
        self.assertIs(p.copy()._labels, p._labels)

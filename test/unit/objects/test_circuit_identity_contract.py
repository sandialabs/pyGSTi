"""Characterization tests pinning the Circuit identity contract (tup/str/hash/eq).

These pin CURRENT behavior of pygsti.circuits.circuit.Circuit. They are
characterization tests: a failure means intended behavior changed; the change must
be deliberate, called out in the PR description, and the pin updated in the same
PR. Never "fix" a failing pin by casually changing production code.

Pinned here:
  * the .tup wire grammar:
      layertup [+ ('@',)+line_labels] [+ ('@',occurrence)] [+ ('__CMPLBL__',)+indices]
      (the line-label '@' separator always appears when occurrence does)
  * hash/eq laws (hash == hash(.tup) == hash(._hashable_tup); eq vs non-Circuit
    compares layertup only; tup metadata affects Circuit-Circuit eq; name/auxinfo don't)
  * implicit done_editing() on hashing an editable circuit (mutation + warning)
  * ordering follows tup ordering
  * the empty-circuit vs idle-layer representation distinction
"""
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.tools.exceptions import ImplicitlyDoneEditingCircuitWarning

from ..util import BaseCase


class CircuitIdentityContractTester(BaseCase):

    # ---------------------------------------------------------------- tup grammar

    def test_tup_no_line_labels(self):
        c = Circuit('GxGy')
        self.assertEqual(c.line_labels, ('*',))
        self.assertEqual(c.tup, (Label('Gx'), Label('Gy')))

    def test_tup_with_line_labels(self):
        c = Circuit('Gx:0Gy:1@(0,1)')
        self.assertEqual(c.tup, (Label(('Gx', 0)), Label(('Gy', 1)), '@', 0, 1))

    def test_tup_occurrence_without_line_labels(self):
        c = Circuit(['Gx', 'Gy'], occurrence=3)
        self.assertEqual(c.tup, (Label('Gx'), Label('Gy'), '@', '@', 3))

    def test_tup_occurrence_with_line_labels(self):
        c = Circuit([('Gx', 0)], line_labels=(0,), occurrence=3)
        self.assertEqual(c.tup, (Label(('Gx', 0)), '@', 0, '@', 3))

    def test_tup_compilable_indices(self):
        layer_list = [('Gx', 0), ('Gy', 0)]
        c = Circuit(layer_list, line_labels=(0,), compilable_layer_indices=(1,))
        self.assertEqual(c.tup, (Label(('Gx', 0)), Label(('Gy', 0)), '@', 0, '__CMPLBL__', 1))

    def test_tup_compilable_indices_without_line_labels(self):
        c = Circuit(['Gx', 'Gy'], compilable_layer_indices=(1,))
        # no '@' separator: '__CMPLBL__' directly abuts the layer labels
        self.assertEqual(c.tup, (Label('Gx'), Label('Gy'), '__CMPLBL__', 1))

    def test_tup_full_grammar(self):
        layer_list = [('Gx', 0), ('Gy', 0)]
        c = Circuit(layer_list, line_labels=(0,), occurrence=2, compilable_layer_indices=(0,))
        self.assertEqual(c.tup, (Label(('Gx', 0)), Label(('Gy', 0)), '@', 0, '@', 2, '__CMPLBL__', 0))

    def test_layertup_is_labels_alias_for_static(self):
        c = Circuit('Gx:0Gy:0@(0)')
        self.assertIs(c.layertup, c._labels)  # zero-copy alias on the static/hot path

    # ---------------------------------------------------------------- hash/eq laws

    def test_hash_chain_static(self):
        c = Circuit('Gx:0Gy:0@(0)')
        h = hash(c)
        self.assertEqual(c._hash, h)
        self.assertEqual(h, hash(c._hashable_tup))
        self.assertEqual(h, hash(c.tup))

    def test_eq_ignores_name_and_auxinfo(self):
        c1 = Circuit('Gx:0@(0)')
        c2 = Circuit('Gx:0@(0)', name='other_name')
        c2.auxinfo['key'] = 'value'
        self.assertEqual(c1, c2)
        self.assertEqual(hash(c1), hash(c2))

    def test_eq_with_none_is_false(self):
        self.assertNotEqual(Circuit('Gx'), None)  # pins the explicit None branch of __eq__

    def test_eq_with_non_circuit_compares_layertup_only(self):
        c = Circuit([('Gx', 0), ('Gy', 1)], line_labels=(0, 1, 2), occurrence=5)
        # metadata (line labels beyond sslbls, occurrence) is IGNORED vs non-Circuits
        self.assertEqual(c, (Label(('Gx', 0)), Label(('Gy', 1))))

    def test_metadata_participates_in_circuit_eq(self):
        base = Circuit([('Gx', 0)], line_labels=(0,))
        self.assertNotEqual(base, Circuit([('Gx', 0)], line_labels=(0, 1)))
        self.assertNotEqual(base, Circuit([('Gx', 0)], line_labels=(0,),  occurrence=1))
        self.assertNotEqual(base, Circuit([('Gx', 0)], line_labels=(0,),  compilable_layer_indices=(0,)))

    def test_construction_paths_agree(self):
        # all public construction paths must yield equal circuits with equal hashes
        # (Circuit._fastinit is deliberately EXCLUDED: see the issue #757 pins)
        via_string   = Circuit('Gx:0Gy:0@(0)')
        via_labels   = Circuit([Label(('Gx', 0)), Label(('Gy', 0))], line_labels=(0,))
        via_tuples   = Circuit([('Gx', 0), ('Gy', 0)],               line_labels=(0,))
        via_editable = Circuit([('Gx', 0), ('Gy', 0)],               line_labels=(0,), editable=True)
        via_editable.done_editing()
        all_paths = [via_string, via_labels, via_tuples, via_editable]
        for other in all_paths[1:]:
            with self.subTest(circuit=other):
                self.assertEqual(other, all_paths[0])
                self.assertEqual(hash(other), hash(all_paths[0]))

    def test_hashing_editable_circuit_mutates_it(self):
        c = Circuit([[('Gy', 1), ('Gx', 0)]], line_labels=(0, 1), editable=True)
        with self.assertWarns(ImplicitlyDoneEditingCircuitWarning):
            h = hash(c)
        # hashing flipped it to read-only (DataSet._collisionaction_update_circuit relies on this)
        self.assertTrue(c._static)
        self.assertEqual(c.layertup[0], Label((('Gx', 0), ('Gy', 1))))  # and canonicalized (sorted)
        self.assertEqual(h, hash(c.tup))

    def test_ordering_follows_tup(self):
        a = Circuit('Gx:0@(0)')
        b = Circuit('Gy:0@(0)')
        self.assertLess(a, b)
        self.assertEqual(a < b, a.tup < b.tup)
        self.assertEqual(a > b, a.tup > b.tup)

    # ------------------------------------------- empty circuit vs idle layer triplet

    def test_empty_circuit_representations_agree(self):
        c_str   = Circuit('{}')
        c_list  = Circuit([])
        c_tuple = Circuit(())
        self.assertEqual(len(c_str), 0)
        self.assertEqual(len(c_list), 0)
        self.assertEqual(len(c_tuple), 0)
        self.assertEqual(c_str, c_list)
        self.assertEqual(c_list, c_tuple)
        self.assertEqual(c_str.str, '{}')
        self.assertEqual(c_str.tup, ())

    def test_empty_circuit_with_line_labels_keeps_suffix(self):
        c = Circuit('{}@(0,1)')
        self.assertEqual(len(c), 0)
        self.assertEqual(c.tup, ('@', 0, 1))
        self.assertEqual(c.str, '{}@(0,1)')

    def test_idle_layer_is_not_empty_circuit(self):
        c_idle = Circuit([Label(())])  # one explicit idle (empty) layer
        self.assertEqual(len(c_idle), 1)
        self.assertEqual(c_idle.str, '[]')
        self.assertEqual(c_idle.tup, (Label(()),))
        self.assertNotEqual(c_idle, Circuit([]))
        self.assertEqual(Circuit('[]'), c_idle)  # '[]' parses to one empty layer

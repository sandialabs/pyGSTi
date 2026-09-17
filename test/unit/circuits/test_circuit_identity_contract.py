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
  * layer canonicalization: every construction path that takes user input sorts the
    simple labels within a layer, so text and code produce the same circuit (#757);
    `_fastinit` is the documented exception
"""
from unittest import mock

from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
import pygsti.circuits.circuitparser as cparser_mod
from pygsti.circuits.circuitparser import slowcircuitparser
from pygsti.io import stdinput
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
        # all public construction paths must yield equal circuits with equal hashes.
        # `Circuit._fastinit` is EXCLUDED because it is private and canonicalizes
        # nothing -- see test_fastinit_requires_canonical_input below.
        via_string   = Circuit('Gx:0Gy:0@(0)')
        via_labels   = Circuit([Label(('Gx', 0)), Label(('Gy', 0))], line_labels=(0,))
        via_tuples   = Circuit([('Gx', 0), ('Gy', 0)],               line_labels=(0,))
        via_editable = Circuit([('Gx', 0), ('Gy', 0)],               line_labels=(0,), editable=True)
        via_editable.done_editing()
        via_stdinput = stdinput.StdInputParser().parse_circuit('Gx:0Gy:0@(0)', create_subcircuits=False)
        reference = via_string
        for name, other in [('labels', via_labels), ('tuples', via_tuples),
                            ('editable', via_editable), ('stdinput', via_stdinput)]:
            self.assertEqual(other, reference, msg=name)
            self.assertEqual(hash(other), hash(reference), msg=name)

    # ------------------------------------------------- layer canonicalization (#757)

    def test_parsed_circuits_match_constructed_ones(self):
        # #757: io/stdinput.py builds every parsed circuit through `_fastinit`, which
        # applies no sort, so the same text used to yield two unequal, differently
        # hashing Circuits depending on whether it came through StdInputParser
        # (dataset/text-file loading) or the constructor. Both parsers now emit
        # canonically-sorted layer labels, so the two paths agree.
        text = "[Gy:1Gx:0]@(0,1)"
        c_parsed = stdinput.StdInputParser().parse_circuit(text, create_subcircuits=False)
        c_built = Circuit(text)
        self.assertEqual(c_parsed, c_built)
        self.assertEqual(hash(c_parsed), hash(c_built))
        self.assertEqual(c_parsed.tup[0], Label((('Gx', 0), ('Gy', 1))))

    def test_both_parsers_emit_the_same_canonical_order(self):
        # the Cython and pure-Python parsers must stay in lockstep: canonicalizing in
        # one and not the other would split behavior by whether the extension built
        cases = ["[Gy:1Gx:0]@(0,1)", "[Gz:2Gcnot:0:1]@(0,1,2)",
                 "[Gx:1!0.5Gy:0]@(0,1)", "[GxGi]@(0,1)",
                 "[Gy:q1Gx:q0]@(q0,q1)", "[Gz:1[Gy:2Gx:0]]@(0,1,2)"]
        for text in cases:
            with self.subTest(text=text):
                fast = Circuit(text)
                with mock.patch.object(cparser_mod, 'parse_circuit',
                                       slowcircuitparser.parse_circuit):
                    slow = Circuit(text)
                self.assertEqual(fast, slow)
                self.assertEqual(fast.tup, slow.tup)

    def test_parsers_reject_duplicate_sslbls_within_a_layer(self):
        # canonicalization also validates: two gates on the same line in one layer
        # have no sorted order. The constructor always rejected this; StdInputParser
        # used to accept it and produce a circuit the constructor could not build.
        text = "[Gx:0Gy:0]@(0)"
        with self.assertRaisesRegex(ValueError, 'duplicate sslbls'):
            Circuit(text)
        with self.assertRaisesRegex(ValueError, 'duplicate sslbls'):
            stdinput.StdInputParser().parse_circuit(text, create_subcircuits=False)

    def test_fastinit_does_not_flatten_nested_compound_layers(self):
        # A SEPARATE parsed-vs-constructed divergence, unrelated to layer ordering and
        # not addressed by the #757 parser fix: `__init__` flattens a nested compound
        # layer, `_fastinit` keeps the nesting. Reproduces with a hand-built, already
        # canonically-sorted label and no parser involved, so sorting is not the cause.
        # Pinned as the current behavior; recorded for issue filing.
        nested = Label((Label((Label(('Gx', 0)), Label(('Gy', 2)))), Label(('Gz', 1))))
        via_fastinit = Circuit._fastinit((nested,), (0, 1, 2), False)
        via_init = Circuit([nested], line_labels=(0, 1, 2))
        self.assertNotEqual(via_fastinit, via_init)
        self.assertEqual(via_init.tup[0], Label((('Gx', 0), ('Gz', 1), ('Gy', 2))))

    def test_fastinit_requires_canonical_input(self):
        # #757 was fixed in the parsers, not here: `_fastinit` is the ~0.5us hot
        # construction tier -- some 300x cheaper than `__init__` -- and still trusts
        # its caller to pass canonical layer tuples. Passing unsorted ones produces a
        # circuit that will not compare equal to the same circuit built any other way.
        # It is private; the only callers outside circuit.py are in io/stdinput.py,
        # and those now receive sorted labels from the parser.
        unsorted = Circuit._fastinit((Label((('Gy', 1), ('Gx', 0))),), (0, 1), False)
        canonical = Circuit([[('Gy', 1), ('Gx', 0)]], line_labels=(0, 1))
        self.assertNotEqual(unsorted, canonical)
        self.assertEqual(Circuit._fastinit(canonical.layertup, (0, 1), False), canonical)

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

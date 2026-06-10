"""Characterization tests pinning the Circuit identity contract (tup/str/hash/eq).

These pin CURRENT behavior of pygsti.circuits.circuit.Circuit. They are
characterization tests: a failure means intended behavior changed; the change must
be deliberate, called out in the PR description, and the pin updated in the same
PR. Never "fix" a failing pin by casually changing production code.

Pinned here:
  * the .tup wire grammar:
      layertup [+ ('@',)+line_labels] [+ ('@',occurrence)] [+ ('__CMPLBL__',)+indices]
  * hash/eq laws (hash == hash(.tup) == hash(._hashable_tup); eq vs non-Circuit
    compares layertup only; tup metadata affects Circuit-Circuit eq; name/auxinfo don't)
  * implicit done_editing() on hashing an editable circuit (mutation + warning)
  * ordering follows tup ordering
  * the empty-circuit vs idle-layer representation distinction
"""
import pytest

from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.tools.exceptions import ImplicitlyDoneEditingCircuitWarning


# ---------------------------------------------------------------- tup grammar

def test_tup_no_line_labels():
    c = Circuit('GxGy')
    assert c.line_labels == ('*',)
    assert c.tup == (Label('Gx'), Label('Gy'))


def test_tup_with_line_labels():
    c = Circuit('Gx:0Gy:1@(0,1)')
    assert c.tup == (Label(('Gx', 0)), Label(('Gy', 1)), '@', 0, 1)


def test_tup_occurrence_without_line_labels():
    c = Circuit(['Gx', 'Gy'], occurrence=3)
    assert c.tup == (Label('Gx'), Label('Gy'), '@', '@', 3)


def test_tup_occurrence_with_line_labels():
    c = Circuit([('Gx', 0)], line_labels=(0,), occurrence=3)
    assert c.tup == (Label(('Gx', 0)), '@', 0, '@', 3)


def test_tup_compilable_indices():
    layer_list = [('Gx', 0), ('Gy', 0)]
    c = Circuit(layer_list, line_labels=(0,), compilable_layer_indices=(1,))
    assert c.tup == (Label(('Gx', 0)), Label(('Gy', 0)), '@', 0, '__CMPLBL__', 1)


def test_tup_full_grammar():
    layer_list = [('Gx', 0), ('Gy', 0)]
    c = Circuit(layer_list, line_labels=(0,), occurrence=2, compilable_layer_indices=(0,))
    assert c.tup == (Label(('Gx', 0)), Label(('Gy', 0)), '@', 0, '@', 2, '__CMPLBL__', 0)


def test_layertup_is_labels_alias_for_static():
    c = Circuit('Gx:0Gy:0@(0)')
    assert c.layertup is c._labels  # zero-copy alias on the static/hot path


# ---------------------------------------------------------------- hash/eq laws

def test_hash_chain_static():
    c = Circuit('Gx:0Gy:0@(0)')
    assert c._hash == hash(c) == hash(c._hashable_tup) == hash(c.tup)


def test_eq_ignores_name_and_auxinfo():
    c1 = Circuit('Gx:0@(0)')
    c2 = Circuit('Gx:0@(0)', name='other_name')
    c2.auxinfo['key'] = 'value'
    assert c1 == c2
    assert hash(c1) == hash(c2)


def test_eq_with_non_circuit_compares_layertup_only():
    c = Circuit([('Gx', 0), ('Gy', 1)], line_labels=(0, 1, 2), occurrence=5)
    # metadata (line labels beyond sslbls, occurrence) is IGNORED vs non-Circuits
    assert c == (Label(('Gx', 0)), Label(('Gy', 1)))


def test_metadata_participates_in_circuit_eq():
    base = Circuit([('Gx', 0)], line_labels=(0,))
    assert base != Circuit([('Gx', 0)], line_labels=(0, 1))
    assert base != Circuit([('Gx', 0)], line_labels=(0,),  occurrence=1)
    assert base != Circuit([('Gx', 0)], line_labels=(0,),  compilable_layer_indices=(0,))


def test_construction_paths_agree():
    # all public construction paths must yield equal circuits with equal hashes
    # (Circuit._fastinit is deliberately EXCLUDED: see the issue #757 pins)
    via_string   = Circuit('Gx:0Gy:0@(0)')
    via_labels   = Circuit([Label(('Gx', 0)), Label(('Gy', 0))], line_labels=(0,))
    via_tuples   = Circuit([('Gx', 0), ('Gy', 0)],               line_labels=(0,))
    via_editable = Circuit([('Gx', 0), ('Gy', 0)],               line_labels=(0,), editable=True)
    via_editable.done_editing()
    all_paths = [via_string, via_labels, via_tuples, via_editable]
    for other in all_paths[1:]:
        assert other == all_paths[0]
        assert hash(other) == hash(all_paths[0])


def test_hashing_editable_circuit_mutates_it():
    c = Circuit([[('Gy', 1), ('Gx', 0)]], line_labels=(0, 1), editable=True)
    with pytest.warns(ImplicitlyDoneEditingCircuitWarning):
        h = hash(c)
    assert c._static  # hashing flipped it to read-only (dataset.py:1415-1432 exploits this)
    assert c.layertup[0] == Label((('Gx', 0), ('Gy', 1)))  # and canonicalized (sorted)
    assert h == hash(c.tup)


def test_ordering_follows_tup():
    a = Circuit('Gx:0@(0)')
    b = Circuit('Gy:0@(0)')
    assert (a < b) == (a.tup < b.tup)
    assert (a > b) == (a.tup > b.tup)


# ------------------------------------------- empty circuit vs idle layer triplet

def test_empty_circuit_representations_agree():
    c_str   = Circuit('{}')
    c_list  = Circuit([])
    c_tuple = Circuit(())
    assert len(c_str) == len(c_list) == len(c_tuple) == 0
    assert c_str == c_list == c_tuple
    assert c_str.str == '{}'
    assert c_str.tup == ()


def test_empty_circuit_with_line_labels_keeps_suffix():
    c = Circuit('{}@(0,1)')
    assert len(c) == 0
    assert c.tup == ('@', 0, 1)
    assert c.str == '{}@(0,1)'


def test_idle_layer_is_not_empty_circuit():
    c_idle = Circuit([Label(())])  # one explicit idle (empty) layer
    assert len(c_idle) == 1
    assert c_idle.str == '[]'
    assert c_idle.tup == (Label(()),)
    assert c_idle != Circuit([])
    assert Circuit('[]') == c_idle  # '[]' parses to one empty layer

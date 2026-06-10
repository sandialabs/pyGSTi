"""Characterization: which Circuit operations preserve vs drop the two non-layer
metadata fields that participate in identity (occurrence, compilable_layer_indices).

The CASES table below is the behavioral contract. It was pinned from code reading
at develop@3e7dd411e and reconciled by execution. Where current behavior is
arguably wrong (e.g. __add__ dropping metadata while leaking string markers) the
wrongness is pinned in test_circuit_known_bugs.py, not here.
"""
import pytest

from pygsti.circuits import Circuit


def _base():
    layer_list = [('Gx', 0), ('Gy', 0), ('Gz', 0)]
    return Circuit(layer_list, line_labels=(0,), occurrence=7, compilable_layer_indices=(1,))


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
]


@pytest.mark.parametrize('case', CASES, ids=lambda case: case[0])
def test_metadata_policy(case):
    case_id, operation, occ_kept, cmp_kept = case
    out = operation(_base())
    assert out.occurrence == (7 if occ_kept else None), case_id
    assert out.compilable_layer_indices == ((1,) if cmp_kept else ()), case_id


def test_mul_repeat_raises_when_occurrence_is_set():
    # SURPRISE: Circuit.repeat (hence __mul__) parses self.str via str.split('@')
    # and unpacks exactly two parts.  An occurrence id appends a second '@'
    # separator (e.g. 'Gx:0@(0)@7'), so multiplying any circuit that has an
    # occurrence set raises ValueError instead of returning a circuit.
    c = _base()
    with pytest.raises(ValueError):
        c * 2


def test_mul_repeat_drops_compilable_indices_when_no_occurrence():
    # Companion pin: with occurrence unset, __mul__ works and drops the
    # compilable_layer_indices metadata.
    c = Circuit([('Gx', 0), ('Gy', 0)], line_labels=(0,), compilable_layer_indices=(1,))
    out = c * 2
    assert out.occurrence is None
    assert out.compilable_layer_indices == ()


def test_getitem_single_layer_returns_label_not_circuit():
    from pygsti.baseobjs import Label
    out = _base()[1]
    assert isinstance(out, Label)
    assert not isinstance(out, Circuit)

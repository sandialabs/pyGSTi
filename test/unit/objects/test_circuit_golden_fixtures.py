"""Golden-fixture tests: the committed binary artifacts under golden/ must keep
loading and must equal freshly constructed circuits on the identity surface
(eq / hash-consistency / str / tup). The fixtures are bytes written by the code
version that baselined them; golden_circuit_defs constructs the same circuits
with current code, so any drift between the two is a contract break.
"""
import json
import os
import pickle

import pytest

from pygsti.circuits import Circuit
from pygsti.data import DataSet

from . import golden_circuit_defs

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')


def _compressed_roundtrip_form(circuit):
    """What a CompressedCircuit round trip reproduces: the layer labels (with any
    CircuitLabels expanded, because Circuit.default_expand_subcircuits is True),
    the line labels, and the occurrence id survive; compilable_layer_indices are
    silently dropped (CompressedCircuit never stores them)."""
    layers = circuit.layertup
    lines  = circuit.line_labels
    return Circuit(layers, lines, occurrence=circuit.occurrence)


def _dataset_binary_key_form(circuit):
    """What survives as a DataSet key after a write_binary/load round trip: like
    the CompressedCircuit form, but the occurrence id is dropped too (DataSet
    strips it on insertion)."""
    layers = circuit.layertup
    lines  = circuit.line_labels
    return Circuit(layers, lines)


@pytest.fixture(scope='module')
def expected():
    return golden_circuit_defs.build_golden_circuits()


@pytest.fixture(scope='module')
def manifest():
    with open(os.path.join(GOLDEN, 'golden_manifest.json')) as f:
        return json.load(f)


def test_pickled_circuits_load_and_match(expected, manifest):
    with open(os.path.join(GOLDEN, 'circuits_golden.pkl'), 'rb') as f:
        loaded = pickle.load(f)
    assert set(loaded) == set(expected)
    for key, c_old in loaded.items():
        c_new = expected[key]
        assert c_old == c_new, key
        assert hash(c_old) == hash(c_new), key
        assert c_old.str == manifest[key]['str'], key
        assert repr(c_old.tup) == manifest[key]['tup'], key
        assert len(c_old) == manifest[key]['len'], key


def test_freshly_constructed_circuits_match_manifest(expected, manifest):
    # guards against defs-module drift AND against changes to str/tup bytes
    for key, c in expected.items():
        assert c.str == manifest[key]['str'], key
        assert repr(c.tup) == manifest[key]['tup'], key


def test_compressed_circuits_expand_to_expected(expected):
    with open(os.path.join(GOLDEN, 'compressed_golden.pkl'), 'rb') as f:
        compressed = pickle.load(f)
    assert set(compressed) == set(expected)
    for key, cc in compressed.items():
        assert cc.expand() == _compressed_roundtrip_form(expected[key]), key


def test_golden_dataset_loads_with_expected_keys_and_counts(expected):
    # The binary round trip normalizes keys via _dataset_binary_key_form, so the
    # compilable_tilde and compilable_pipe entries collide into one key.  On load
    # the later row shadows the earlier one (last writer wins, NOT aggregation),
    # which the overwrite-style dict build below reproduces.
    expected_counts = {}
    for i, c in enumerate(expected.values()):
        key = _dataset_binary_key_form(c)
        expected_counts[key] = (10 + i, 90 - i)

    ds = DataSet(file_to_load_from=os.path.join(GOLDEN, 'golden_dataset.pkl.gz'))
    assert set(ds.keys()) == set(expected_counts)
    for key, (n0, n1) in expected_counts.items():
        counts = ds[key].counts
        assert counts[('0',)] == n0, key.str
        assert counts[('1',)] == n1, key.str

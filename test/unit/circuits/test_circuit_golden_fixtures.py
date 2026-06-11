"""Golden-fixture tests: the committed binary artifacts under golden/ must keep
loading and must equal freshly constructed circuits on the identity surface
(eq / hash-consistency / str / tup). The fixtures are bytes written by the code
version that baselined them; golden_circuit_defs constructs the same circuits
with current code, so any drift between the two is a contract break.

The compressed and dataset comparisons go through ``_compressed_roundtrip_form``
/ ``_dataset_binary_key_form``, which encode the known-lossy parts of those
round trips.
"""
import json
import os
import pickle

from pygsti.circuits import Circuit
from pygsti.data import DataSet

from ..util import BaseCase
from . import golden_circuit_defs

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')


def _compressed_roundtrip_form(circuit):
    """What a CompressedCircuit round trip reproduces: the layer labels (with any
    CircuitLabels expanded, because Circuit.default_expand_subcircuits is True),
    the line labels, and the occurrence id survive; compilable_layer_indices are
    silently dropped (CompressedCircuit never stores them).

    Note: both sides of the comparison pass through the CURRENT Circuit
    constructor, so constructor-level normalization changes are masked here
    (the manifest test covers fresh-constructor drift)."""
    layers = circuit.layertup
    lines  = circuit.line_labels
    return Circuit(layers, lines, occurrence=circuit.occurrence)


def _dataset_binary_key_form(circuit):
    """What survives as a DataSet key after a write_binary/load round trip: like
    the CompressedCircuit form, but the occurrence id is dropped too (DataSet
    strips it on insertion).

    Note: both sides of the comparison pass through the CURRENT Circuit
    constructor, so constructor-level normalization changes are masked here
    (the manifest test covers fresh-constructor drift)."""
    layers = circuit.layertup
    lines  = circuit.line_labels
    return Circuit(layers, lines)


class CircuitGoldenFixturesTester(BaseCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.expected = golden_circuit_defs.build_golden_circuits()
        with open(os.path.join(GOLDEN, 'golden_manifest.json')) as f:
            cls.manifest = json.load(f)

    def test_pickled_circuits_load_and_match(self):
        with open(os.path.join(GOLDEN, 'circuits_golden.pkl'), 'rb') as f:
            loaded = pickle.load(f)
        self.assertEqual(set(loaded), set(self.expected))
        for key, c_old in loaded.items():
            with self.subTest(key=key):
                c_new = self.expected[key]
                self.assertEqual(c_old, c_new)
                self.assertEqual(hash(c_old), hash(c_new))
                self.assertEqual(c_old.str, self.manifest[key]['str'])
                self.assertEqual(repr(c_old.tup), self.manifest[key]['tup'])
                self.assertEqual(len(c_old), self.manifest[key]['len'])

    def test_freshly_constructed_circuits_match_manifest(self):
        # guards against defs-module drift AND against changes to str/tup bytes
        for key, c in self.expected.items():
            with self.subTest(key=key):
                self.assertEqual(c.str, self.manifest[key]['str'])
                self.assertEqual(repr(c.tup), self.manifest[key]['tup'])

    def test_compressed_circuits_expand_to_expected(self):
        with open(os.path.join(GOLDEN, 'compressed_golden.pkl'), 'rb') as f:
            compressed = pickle.load(f)
        self.assertEqual(set(compressed), set(self.expected))
        for key, cc in compressed.items():
            with self.subTest(key=key):
                self.assertEqual(cc.expand(), _compressed_roundtrip_form(self.expected[key]))

    def test_golden_dataset_loads_with_expected_keys_and_counts(self):
        # The binary round trip normalizes keys via _dataset_binary_key_form, so the
        # compilable_tilde and compilable_pipe entries collide into one key.  On load
        # the later row shadows the earlier one (last writer wins, NOT aggregation),
        # which the overwrite-style dict build below reproduces.
        expected_counts = {}
        for i, c in enumerate(self.expected.values()):
            key = _dataset_binary_key_form(c)
            counts_i = golden_circuit_defs.golden_counts(i)
            expected_counts[key] = (counts_i['0'], counts_i['1'])

        ds = DataSet(file_to_load_from=os.path.join(GOLDEN, 'golden_dataset.pkl.gz'))
        self.assertEqual(set(ds.keys()), set(expected_counts))
        for key, (n0, n1) in expected_counts.items():
            with self.subTest(key=key.str):
                counts = ds[key].counts
                self.assertEqual(counts[('0',)], n0)
                self.assertEqual(counts[('1',)], n1)

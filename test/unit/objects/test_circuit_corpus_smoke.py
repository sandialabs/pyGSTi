"""Smoke test for the differential-corpus tool (test/performance/circuit_corpus.py):
a small corpus must build, fingerprint, and compare clean against itself in-process.
Keeps the tool importable and its fingerprint surface exercised in CI without the
cost of a full corpus run.
"""
import copy
import os
import sys

from ..util import BaseCase

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PERF_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', 'performance'))
if _PERF_DIR not in sys.path:
    sys.path.insert(0, _PERF_DIR)

import circuit_corpus  # noqa: E402


class CircuitCorpusSmokeTester(BaseCase):

    def test_smoke_corpus_self_compare(self):
        corpus = circuit_corpus.build_corpus(size='smoke')
        self.assertGreaterEqual(len(corpus), 50)
        fps = circuit_corpus.fingerprint_all(corpus)
        exceptions = [rec['id'] for rec in fps if rec['fp']['str'].startswith('EXC:')]
        self.assertEqual(exceptions, [])
        self.assertEqual(circuit_corpus.compare_fingerprints(fps, fps, allowlist=[]), [])

    def test_smoke_corpus_is_deterministic(self):
        a = circuit_corpus.fingerprint_all(circuit_corpus.build_corpus(size='smoke'))
        b = circuit_corpus.fingerprint_all(circuit_corpus.build_corpus(size='smoke'))
        self.assertEqual(circuit_corpus.compare_fingerprints(a, b, allowlist=[]), [])

    def test_smoke_compare_detects_and_allowlists_mismatch(self):
        """Negative path: a perturbed field is reported as exactly one mismatch, and
        an allowlist entry keyed on (field, baseline circuit-str repr) suppresses it.
        This also pins the repr-quoting contract for the allowlist circuit-str column.
        """
        base = circuit_corpus.fingerprint_all(circuit_corpus.build_corpus(size='smoke'))
        other = copy.deepcopy(base)
        idx = len(other) // 2
        other[idx]['fp']['hash'] = 'PERTURBED'

        mismatches = circuit_corpus.compare_fingerprints(base, other, allowlist=[])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]['field'], 'hash')
        self.assertEqual(mismatches[0]['id'], base[idx]['id'])
        self.assertEqual(mismatches[0]['str'], base[idx]['fp']['str'])

        allowlist = [('hash', base[idx]['fp']['str'], 'unit-test perturbation')]
        self.assertEqual(circuit_corpus.compare_fingerprints(base, other, allowlist=allowlist), [])

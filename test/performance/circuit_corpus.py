#!/usr/bin/env python
"""Differential behavior corpus for pygsti Circuits: generate fingerprints of
realistic circuits (~23k at --size full) under one version of the code, replay
under another, and demand that every behavioral difference is on a
human-readable allowlist.

Usage:
    # Any --out / compare path ending in '.gz' is gzip-(de)compressed
    # transparently, detected by suffix. A full uncompressed corpus is ~120 MB,
    # so prefer .gz; the tracked reference baseline is stored compressed at
    # test/performance/circuit_corpus_baseline.jsonl.gz (~2 MB).
    python test/performance/circuit_corpus.py generate \
        --out test/performance/circuit_corpus_baseline.jsonl.gz [--size full|small|smoke]
    python test/performance/circuit_corpus.py compare \
        test/performance/circuit_corpus_baseline.jsonl.gz candidate.jsonl.gz \
        --allowlist test/performance/circuit_corpus_allowlist.txt

Regenerating the baseline to the same path is byte-reproducible (fixed gzip
mtime), so a no-op regeneration leaves git clean. Uncompressed '*.jsonl' outputs
are gitignored under test/performance/.

Fingerprints include hash values, which are only process-stable under a fixed
PYTHONHASHSEED — the CLI re-execs itself with PYTHONHASHSEED=0 automatically.
Generate both fingerprint files with the same interpreter version and platform:
PYTHONHASHSEED=0 fixes hash salting, not the hash algorithm itself, which can
differ across Python versions and platforms.

Allowlist format (tab-separated, '#' comments):
    field<TAB>circuit-str<TAB>reason
A mismatch is permitted iff some line matches its (field, baseline str) exactly.
The circuit-str column is the Python repr of the baseline circuit string exactly
as shown in compare output, including the surrounding quotes.
For a reviewed systematic change (e.g. a deliberate hash change affecting every
circuit), the intended procedure is to regenerate the baseline after sign-off —
do NOT bulk-populate the allowlist; the allowlist is for narrow, named exceptions.

Known coverage gaps: the corpus does NOT contain circuits with occurrence ids,
compilable_layer_indices, CircuitLabel subcircuits (the reparse pass uses
create_subcircuits=False), the '*' default line labels, labels with args or
time, or editable circuits; line labels are (0,), (0, 1), ('Q0',), and
('Q0', 'Q1'). Those constructs are covered statically by the golden-fixture
tests in test_circuit_golden_fixtures.py, which complement this corpus.

At --size full the large 2-qubit GST designs are deterministically subsampled to
a fixed per-design cap (see SIZES and _subsample) so the committed baseline stays
~2 MB compressed; the subsample keeps the first and last circuit, so the deepest
max-length circuits stay represented. The 1-qubit designs are below the cap and
kept in full.
"""
import argparse
import gzip
import io
import json
import os
import sys

# gst_cap bounds how many circuits are kept from EACH GST design (deterministic
# even stride that keeps both endpoints; see _subsample), so the committed
# compressed baseline stays small (~2 MB gzipped) without dropping the deepest
# max-length circuits. None keeps the whole design. The 1-qubit designs are well
# under the cap and so are kept in full; only the large 2-qubit designs are thinned.
SIZES = {
    'smoke': dict(ml_1q=4,   ml_2q=None, n_random=40,   reparse_every=5,  gst_cap=None),
    'small': dict(ml_1q=8,   ml_2q=None, n_random=500,  reparse_every=10, gst_cap=None),
    'full':  dict(ml_1q=256, ml_2q=128,  n_random=4000, reparse_every=10, gst_cap=7000),
}


def _subsample(circuits, cap):
    """Deterministically thin `circuits` to at most `cap` entries via an even
    stride that always keeps the first and last element. The GST circuit lists are
    ordered by increasing max-length, so keeping the endpoints preserves the full
    depth range -- including the deepest max-length circuits -- in the slimmed
    corpus. `cap=None` (or cap >= len) keeps every circuit."""
    n = len(circuits)
    if cap is None or n <= cap:
        return list(circuits)
    if cap <= 1:
        return [circuits[0]] if n else []
    idxs = sorted({(i * (n - 1)) // (cap - 1) for i in range(cap)})
    return [circuits[i] for i in idxs]


def build_corpus(size='full'):
    """Deterministic list of (source_tag, Circuit)."""
    import numpy as np
    from pygsti.algorithms.randomcircuit import create_random_circuit
    from pygsti.io import stdinput
    from pygsti.modelpacks import smq1Q_XYI
    from pygsti.processors import QubitProcessorSpec

    cfg = SIZES[size]
    corpus = []

    cap = cfg['gst_cap']

    design = smq1Q_XYI.create_gst_experiment_design(cfg['ml_1q'])
    corpus += [('gst_1q', c) for c in _subsample(design.all_circuits_needing_data, cap)]

    # the same 1-qubit design relabeled with a string qubit label ('Q0'), so the
    # corpus exercises string line labels through every fingerprint field and the
    # reparse path -- not just the integer labels (0,) / (0, 1).
    design_q = smq1Q_XYI.create_gst_experiment_design(cfg['ml_1q'], qubit_labels=('Q0',))
    corpus += [('gst_1q_strlbl', c) for c in _subsample(design_q.all_circuits_needing_data, cap)]

    if cfg['ml_2q']:
        from pygsti.modelpacks import smq2Q_XYICNOT
        design2 = smq2Q_XYICNOT.create_gst_experiment_design(cfg['ml_2q'])
        corpus += [('gst_2q', c) for c in _subsample(design2.all_circuits_needing_data, cap)]
        design2_q = smq2Q_XYICNOT.create_gst_experiment_design(cfg['ml_2q'], qubit_labels=('Q0', 'Q1'))
        corpus += [('gst_2q_strlbl', c) for c in _subsample(design2_q.all_circuits_needing_data, cap)]

    pspec = QubitProcessorSpec(2, ['Gi', 'Gxpi2', 'Gypi2', 'Gcnot'], geometry='line')
    rng = np.random.RandomState(20260610)
    for _ in range(cfg['n_random']):
        depth   = int(rng.randint(0, 65))
        circuit = create_random_circuit(pspec, depth, rand_state=rng)
        corpus.append(('rand_2q', circuit))

    # re-parse a sample through the dataset-loading (_fastinit) path
    sip = stdinput.StdInputParser()
    for i in range(0, len(corpus), cfg['reparse_every']):
        tag, c = corpus[i]
        reparsed = sip.parse_circuit(c.str, create_subcircuits=False)
        corpus.append((tag + ':reparsed', reparsed))
    return corpus


def _outcome(fn):
    try:
        return repr(fn())
    except Exception as e:  # exceptions ARE behavior; record them
        return f'EXC:{type(e).__name__}:{e}'


def fingerprint(c):
    # len(c) is computed inside each slice lambda (not once up front) so that a
    # raising __len__ records EXC values for the slice fields instead of
    # crashing generate; the standalone 'len' field captures len() behavior.
    fp = {
        'str':          _outcome(lambda: c.str),
        'len':          _outcome(lambda: len(c)),
        'depth':        _outcome(lambda: c.depth),
        'tup':          _outcome(lambda: c.tup),
        'layertup':     _outcome(lambda: c.layertup),
        'line_labels':  _outcome(lambda: c.line_labels),
        'hash':         _outcome(lambda: hash(c)),
        'slice_head':   _outcome(lambda: c[0:min(2, len(c))].tup),
        'slice_tail':   _outcome(lambda: c[len(c) // 2:].tup),
        'concat_tup':   _outcome(lambda: (c + c).tup),
        'concat_str':   _outcome(lambda: (c + c).str),
    }
    return fp


def fingerprint_all(corpus):
    records = []
    for i, (tag, c) in enumerate(corpus):
        records.append({'id': i, 'src': tag, 'fp': fingerprint(c)})
    return records


def compare_fingerprints(base, other, allowlist):
    """Returns list of non-allowlisted mismatches as dicts."""
    mismatches = []
    if len(base) != len(other):
        mismatches.append({'id': None, 'src': 'CORPUS', 'field': 'length',
                           'base': len(base), 'other': len(other)})
        return mismatches
    # Misalignment check: if most records disagree on 'str' or 'id', the two
    # corpora were almost certainly generated differently (different size,
    # builder version, or record order) and per-field comparison is meaningless.
    n_str_diff = sum(1 for rb, ro in zip(base, other)
                     if rb['fp']['str'] != ro['fp']['str'])
    n_id_diff = sum(1 for rb, ro in zip(base, other) if rb['id'] != ro['id'])
    if n_str_diff > len(base) / 2 or n_id_diff > len(base) / 2:
        mismatches.append({'id': None, 'src': 'CORPUS', 'field': 'CORPUS_ALIGNMENT',
                           'base': f'{n_str_diff}/{len(base)} circuit strs differ',
                           'other': f'{n_id_diff}/{len(base)} record ids differ'})
    allowed = {(field, cstr) for field, cstr, _reason in allowlist}
    for rec_b, rec_o in zip(base, other):
        base_str = rec_b['fp']['str']
        for field, val_b in rec_b['fp'].items():
            val_o = rec_o['fp'].get(field)
            if val_b == val_o:
                continue
            if (field, base_str) in allowed:
                continue
            mismatch = {'id': rec_b['id'], 'src': rec_b['src'], 'field': field,
                        'str': base_str, 'base': val_b, 'other': val_o}
            mismatches.append(mismatch)
    return mismatches


def load_allowlist(path):
    entries = []
    if path and os.path.exists(path):
        with open(path) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                try:
                    field, cstr, reason = line.split('\t', 2)
                except ValueError as e:
                    raise ValueError(
                        f"malformed allowlist line {lineno} of {path} "
                        f"(expected field<TAB>circuit-str<TAB>reason): {line!r}") from e
                entries.append((field, cstr, reason))
    return entries


def _ensure_fixed_hashseed():
    if os.environ.get('PYTHONHASHSEED') != '0':
        env = dict(os.environ, PYTHONHASHSEED='0')
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


def _open_text(path, mode):
    """Open a fingerprint file for text I/O, choosing gzip by suffix: any path
    ending in '.gz' is (de)compressed transparently, so the committed baseline can
    be stored compressed and `generate`/`compare` handle it without a separate
    step. `mode` is 'rt' or 'wt'. Compressed writes fix the gzip mtime to 0 so
    regenerating identical content yields byte-identical output (git diffs of the
    committed baseline stay meaningful)."""
    if not path.endswith('.gz'):
        return open(path, mode, encoding='utf-8')
    if 'w' in mode:
        gz = gzip.GzipFile(path, 'wb', compresslevel=9, mtime=0)
    else:
        gz = gzip.GzipFile(path, 'rb')
    return io.TextIOWrapper(gz, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--out', required=True)
    gen.add_argument('--size', choices=sorted(SIZES), default='full')
    cmp_p = sub.add_parser('compare')
    cmp_p.add_argument('baseline')
    cmp_p.add_argument('candidate')
    cmp_p.add_argument('--allowlist', default=None)
    args = parser.parse_args()

    _ensure_fixed_hashseed()

    if args.cmd == 'generate':
        corpus = build_corpus(args.size)
        records = fingerprint_all(corpus)
        with _open_text(args.out, 'wt') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')
        print(f"wrote {len(records)} fingerprints ({args.size}) to {args.out}")
    else:
        with _open_text(args.baseline, 'rt') as f:
            base = [json.loads(line) for line in f]
        with _open_text(args.candidate, 'rt') as f:
            other = [json.loads(line) for line in f]
        allow = load_allowlist(args.allowlist)
        mismatches = compare_fingerprints(base, other, allow)
        if mismatches:
            if any(m['field'] == 'CORPUS_ALIGNMENT' for m in mismatches):
                print("=" * 78)
                print("WARNING: baseline and candidate corpora appear to have been")
                print("generated differently — per-field comparison below is unreliable.")
                print("Regenerate both fingerprint files from the same corpus settings.")
                print("=" * 78)
            print(f"{len(mismatches)} NON-ALLOWLISTED mismatches:")
            for m in mismatches[:50]:
                print(f"  [{m['id']} {m['src']}] {m['field']}:")
                if 'str' in m:
                    print(f"    circuit-str (paste into allowlist): {m['str']}")
                print(f"    base:   {m['base']}")
                print(f"    other:  {m['other']}")
            if len(mismatches) > 50:
                print(f"  ... and {len(mismatches) - 50} more")
            sys.exit(1)
        print(f"OK: {len(base)} fingerprints match ({len(allow)} allowlist entries available)")


if __name__ == '__main__':
    main()

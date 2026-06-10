#!/usr/bin/env python
"""Differential behavior corpus for pygsti Circuits: generate fingerprints of
~10-50k realistic circuits under one version of the code, replay under another,
and demand that every behavioral difference is on a human-readable allowlist.

Usage:
    python test/performance/circuit_corpus.py generate --out develop.jsonl [--size full|small]
    python test/performance/circuit_corpus.py compare develop.jsonl rewrite.jsonl \
        --allowlist test/performance/circuit_corpus_allowlist.txt

Fingerprints include hash values, which are only process-stable under a fixed
PYTHONHASHSEED — the CLI re-execs itself with PYTHONHASHSEED=0 automatically.

Allowlist format (tab-separated, '#' comments):
    field<TAB>circuit-str<TAB>reason
A mismatch is permitted iff some line matches its (field, baseline str) exactly.
"""
import argparse
import json
import os
import sys

SIZES = {
    'smoke': dict(ml_1q=4,  ml_2q=None, n_random=40,    reparse_every=5 ),
    'small': dict(ml_1q=8,  ml_2q=None, n_random=500,   reparse_every=10),
    'full':  dict(ml_1q=64, ml_2q=8,    n_random=10000, reparse_every=10),
}


def build_corpus(size='full'):
    """Deterministic list of (source_tag, Circuit)."""
    import numpy as np
    from pygsti.algorithms.randomcircuit import create_random_circuit
    from pygsti.io import stdinput
    from pygsti.modelpacks import smq1Q_XYI
    from pygsti.processors import QubitProcessorSpec

    cfg = SIZES[size]
    corpus = []

    design = smq1Q_XYI.create_gst_experiment_design(cfg['ml_1q'])
    corpus += [('gst_1q', c) for c in design.all_circuits_needing_data]

    if cfg['ml_2q']:
        from pygsti.modelpacks import smq2Q_XYICNOT
        design2 = smq2Q_XYICNOT.create_gst_experiment_design(cfg['ml_2q'])
        corpus += [('gst_2q', c) for c in design2.all_circuits_needing_data]

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
    n = len(c)
    fp = {
        'str':          _outcome(lambda: c.str),
        'len':          _outcome(lambda: len(c)),
        'depth':        _outcome(lambda: c.depth),
        'tup':          _outcome(lambda: c.tup),
        'layertup':     _outcome(lambda: c.layertup),
        'line_labels':  _outcome(lambda: c.line_labels),
        'hash':         _outcome(lambda: hash(c)),
        'slice_head':   _outcome(lambda: c[0:min(2, n)].tup),
        'slice_tail':   _outcome(lambda: c[n // 2:].tup),
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
                        'base': val_b, 'other': val_o}
            mismatches.append(mismatch)
    return mismatches


def load_allowlist(path):
    entries = []
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                field, cstr, reason = line.split('\t', 2)
                entries.append((field, cstr, reason))
    return entries


def _ensure_fixed_hashseed():
    if os.environ.get('PYTHONHASHSEED') != '0':
        env = dict(os.environ, PYTHONHASHSEED='0')
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


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
        with open(args.out, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')
        print(f"wrote {len(records)} fingerprints ({args.size}) to {args.out}")
    else:
        with open(args.baseline) as f:
            base = [json.loads(line) for line in f]
        with open(args.candidate) as f:
            other = [json.loads(line) for line in f]
        allow = load_allowlist(args.allowlist)
        mismatches = compare_fingerprints(base, other, allow)
        if mismatches:
            print(f"{len(mismatches)} NON-ALLOWLISTED mismatches:")
            for m in mismatches[:50]:
                print(f"  [{m['id']} {m['src']}] {m['field']}:")
                print(f"    base:   {m['base']}")
                print(f"    other:  {m['other']}")
            if len(mismatches) > 50:
                print(f"  ... and {len(mismatches) - 50} more")
            sys.exit(1)
        print(f"OK: {len(base)} fingerprints match ({len(allow)} allowlist entries available)")


if __name__ == '__main__':
    main()

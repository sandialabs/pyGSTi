"""One-time generator for the committed golden Circuit fixtures.

Run from the repo root:
    PYTHONHASHSEED=0 python test/unit/objects/generate_circuit_golden.py

Regenerating REPLACES the fixtures and resets the baseline — only do that
deliberately, in a reviewed PR that explains why.

Regeneration determinism:

* ``circuits_golden.pkl``, ``compressed_golden.pkl``, and ``golden_manifest.json``
  regenerate byte-identically under ``PYTHONHASHSEED=0`` and therefore serve as
  drift detectors: an unexpected diff in any of them means circuit construction
  or serialization behavior changed.
* ``golden_dataset.pkl.gz`` is EXPECTED to differ on every run (DataSet generates
  a random uuid that gets pickled, plus gzip embeds mtime headers). Only
  re-commit it when the dataset contract deliberately changes.
* ``PYTHONHASHSEED=0`` is load-bearing for the .pkl byte-stability — do not
  drop it.
"""
import json
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import golden_circuit_defs  # noqa: E402

from pygsti.circuits.circuit import CompressedCircuit  # noqa: E402
from pygsti.data import DataSet  # noqa: E402

OUTDIR = os.path.join(_HERE, 'golden')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    circuits = golden_circuit_defs.build_golden_circuits()

    with open(os.path.join(OUTDIR, 'circuits_golden.pkl'), 'wb') as f:
        # protocol 2: frozen format, readable by every Python 3; keeps fixture
        # bytes stable across interpreter upgrades
        pickle.dump(circuits, f, protocol=2)

    compressed = {k: CompressedCircuit(c) for k, c in circuits.items()}
    with open(os.path.join(OUTDIR, 'compressed_golden.pkl'), 'wb') as f:
        pickle.dump(compressed, f, protocol=2)  # protocol 2: see above

    manifest = {}
    for k, c in circuits.items():
        manifest[k] = {
            'str':                       c.str,
            'tup':                       repr(c.tup),
            'len':                       len(c),
            'line_labels':               repr(c.line_labels),
            'occurrence':                repr(c.occurrence),
            'compilable_layer_indices':  repr(c.compilable_layer_indices),
        }
    with open(os.path.join(OUTDIR, 'golden_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    ds = DataSet(outcome_labels=['0', '1'])
    for i, c in enumerate(circuits.values()):
        ds.add_count_dict(c, golden_circuit_defs.golden_counts(i))
    ds.done_adding_data()
    ds.write_binary(os.path.join(OUTDIR, 'golden_dataset.pkl.gz'))

    print(f"wrote {len(circuits)} golden circuits to {OUTDIR}")


if __name__ == '__main__':
    main()

# Performance Tests

This test directory is for performance tests that may not be feasible to run in the context of GitHub Actions CI,
either due to long runtimes or high resource requirements (i.e. CPU cores/memory).

However, it is still desirable to occasionally check that performance has not degraded,
particularly before official PyGSTi releases.

## MPI Scaling

In early 2021, significant effort was placed in improving the MPI performance of PyGSTi with large core counts;
in particular, the MapForwardSimulator saw major performance improvements.

The `mpi_2D_scaling` directory contains the scripts needed to generate 2D speedup plots.
These plots make it easy to see tradeoffs between MPI and shared memory usage.

The scripts here are intended to run on the Kahuna cluster through SLURM.
In order to run on your machine:

1. Open `run.sh` and set the environment properly (OpenMPI, compilers/libraries, and Python venv)

1. Run `sbatch-all.sh` to submit all jobs to SLURM. 
    1. If running interactively, modify `sbatch-all.sh` to simply run each `*.sbatch` file
    (which can be run as a script on a local machine).
    1. If on a machine with less than 64 cores, comment those lines out of `sbatch-all.sh`.

1. Once all jobs are completed, run `extract_timings.py` to generate a JSON file and the 2D speedup plot.

1. Compare to the reference values and hope nothing has gotten slower.

## Circuit microbenchmarks

`circuit_microbench.py` times the Circuit identity/creation hot paths (construction
tiers, hashing, parsing, concatenation) plus a real experiment-design macro, and
prints a markdown table — the before/after evidence format used on PRs [#445](https://github.com/sandialabs/pyGSTi/pull/445)/[#692](https://github.com/sandialabs/pyGSTi/pull/692).
Run it manually with the same interpreter before and after any change to
`pygsti/circuits/circuit.py`; it is intentionally not part of CI.

    python test/performance/circuit_microbench.py --repeats 7 --json out.json

## Circuit differential corpus

`circuit_corpus.py` builds a deterministic corpus of realistic circuits (~23k at `--size full`: 1- and 2-qubit GST experiment designs with both integer `(0,)`/`(0, 1)` and string `('Q0',)`/`('Q0', 'Q1')` qubit labels, seeded random circuits, and re-parsed variants through the dataset-loading path), fingerprints every identity-relevant behavior (str/tup/hash/slices/concat, with exceptions recorded as values), and compares fingerprint files across implementations. Differences must be listed in `circuit_corpus_allowlist.txt` (field, circuit, reason) or the compare fails. Hash stability across processes is handled via an automatic PYTHONHASHSEED=0 re-exec. Generate both fingerprint files with the same interpreter version and platform: PYTHONHASHSEED=0 fixes hash salting, not the hash algorithm itself, which can differ across Python versions and platforms.

Any `--out`/compare path ending in `.gz` is gzip-(de)compressed transparently (detected by file suffix). An uncompressed full corpus is ~120 MB, so prefer `.gz`. The tracked reference baseline lives at `circuit_corpus_baseline.jsonl.gz` (~2 MB), kept small by deterministically subsampling the large 2-qubit GST designs to a fixed per-design cap (the deepest max-length circuits are retained). Uncompressed `*.jsonl` outputs are gitignored (both under `test/performance/` and at the repo root). Regenerating the baseline to the same path is byte-reproducible (fixed gzip mtime), so re-running it after a no-op change leaves git clean — regenerate and commit it only after a reviewed corpus change.

    # Compare a candidate build against the committed baseline:
    python test/performance/circuit_corpus.py generate \
        --out candidate.jsonl.gz --size full
    python test/performance/circuit_corpus.py compare \
        test/performance/circuit_corpus_baseline.jsonl.gz candidate.jsonl.gz \
        --allowlist test/performance/circuit_corpus_allowlist.txt

    # Or regenerate the committed baseline itself (after a reviewed corpus change):
    python test/performance/circuit_corpus.py generate \
        --out test/performance/circuit_corpus_baseline.jsonl.gz --size full

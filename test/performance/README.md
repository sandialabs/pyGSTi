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

On Kahuna, these can be run with `./sbatch-all.sh` to submit all necessary jobs to SLURM.
Then `extract_timings.py` parses the output files and generates the 2D speedup plot.

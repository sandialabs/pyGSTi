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

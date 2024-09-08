#!/bin/bash

# These are Kahuna modules to get to OpenMPI
# Ensure your environment matches however Cython extensions and mpi4py was compiled
module load gcc/7.5.0
module load gcc7-support
module load openmpi/4.0.5

# Ensure that the correct Python environment is set
# Example: source <path to venv>/bin/activate

NUM_PROCS=${1:-1}
NUM_HOST_PROCS=${2:-1}

export PYGSTI_MAX_HOST_PROCS=${NUM_HOST_PROCS}
PREFIX=${NUM_PROCS}_${NUM_HOST_PROCS}

echo "Running with ${NUM_PROCS} total procs, ${NUM_HOST_PROCS} procs per 'host'"

# Note: I think only the first of these is required if using NumPy/SciPy from pip
# But doesn't really hurt to have them all
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1

# Note: This flags are useful on Kahuna to avoid error messages
# But the --mca flags are not necessary for performance
mpirun -np ${NUM_PROCS} --mca pml ucx --mca btl '^openib' \
  python ./mpi_timings.py &> ${PREFIX}.out

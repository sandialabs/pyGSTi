#!/bin/bash
cd ../..
mkdir -p output
python3 -m nose -v */test*.py mpi/mpitest*.py 2>&1 | tee output/out.txt
#OLD nosetests -v */test*.py mpi/mpitest*.py 2>&1 | tee output/out.txt

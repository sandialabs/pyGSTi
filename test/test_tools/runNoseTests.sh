#!/bin/bash
cd ..
mkdir -p output
nosetests -v */test*.py mpi/mpitest*.py 2>&1 | tee output/out.txt

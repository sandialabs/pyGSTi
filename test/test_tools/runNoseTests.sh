#!/bin/bash
cd ..
nosetests -v */*.py mpi/mpitest*.py 2>&1 | tee output/out.txt

#!/bin/bash
#cd ../..
mkdir -p output
python2 -m nose -v */test*.py mpi/testmpi*.py 2>&1 | tee output/out2.txt

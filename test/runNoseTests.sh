#!/bin/bash
nosetests -v */*.py mpi/mpitest*.py 2>&1 | tee out.txt

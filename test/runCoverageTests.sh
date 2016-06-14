#!/bin/bash
echo "Serial tests started..."
time nosetests -v --with-coverage --cover-package=pygsti --cover-erase test*.py  > coverage_tests_serial.out 2>&1
mv .coverage coverage.serial
echo "Serial Output written to coverage_tests_serial.out"

echo "Parallel tests started..."
cp setup.cfg.mpi setup.cfg #stage setup.cfg
time mpiexec -np 4 python runtests.py -v --with-coverage --cover-package=pygsti --cover-erase mpitest*.py  > coverage_tests_mpi.out 2>&1
mv .coverage coverage.mpi
rm setup.cfg #unstage setup.cfg
echo "MPI Output written to coverage_tests_mpi.out"

cp coverage.serial .coverage.serial
cp coverage.mpi .coverage.mpi
coverage combine
coverage report -m --include="*/pyGSTi/packages/pygsti/*" > coverage_tests.out 2>&1
echo "Combined Output written to coverage_tests.out"

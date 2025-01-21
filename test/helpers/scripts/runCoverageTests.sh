#!/bin/bash

cd ../../

echo "Serial tests started..."
time nosetests -v --with-coverage --cover-package=pygsti_temp --cover-erase */test*.py > output/coverage_tests_serial.out 2>&1
mv .coverage output/coverage.serial
echo "Serial Output written to coverage_tests_serial.out"

echo "Parallel tests started..."
cp mpi/setup.cfg.mpi setup.cfg #stage setup.cfg
time mpiexec -np 4 python mpi/runtests.py -v --with-coverage --cover-package=pygsti_temp --cover-erase mpi/testmpi*.py  > output/coverage_tests_mpi.out 2>&1
mv .coverage output/coverage.mpi
rm setup.cfg #unstage setup.cfg
echo "MPI Output written to coverage_tests_mpi.out"

cp output/coverage.serial .coverage.serial
cp output/coverage.mpi    .coverage.mpi
coverage combine
coverage report -m --include="*/pyGSTi/pygsti/*" > output/coverage_tests.out 2>&1
echo "Combined Output written to coverage_tests.out"

from runModule  import run_module
from message    import show_message
from benchmarks import benchmark
from tool       import tool
import os

@tool
def serial_tests():
    os.system('time nosetests -v --with-coverage --cover-package=pygsti --cover-erase */*.py > coverage_tests_serial.out 2>&1')
    os.system('mv .coverage coverage.serial')

@tool
def parallel_tests():
    os.system('cp mpi/setup.cfg.mpi setup.cfg')
    os.system('time mpiexec -np 4 python mpi/runtests.py -v --with-coverage --cover-package=pygsti --cover-erase mpi/mpitest*.py  > coverage_tests_mpi.out 2>&1')
    os.system('mv .coverage coverage.mpi')
    os.system('rm setup.cfg')

if __name__ == "__main__":

    show_message('SERIAL TESTS STARTED')

    serial_tests()

    show_message('ENDING SERIAL TESTS')
    show_message('PARALLEL TESTS STARTED')

    parallel_tests()

    show_message('ENDING PARALLEL TESTS')

'''
    os.system('cp coverage.serial .coverage.serial')
    os.system('cp coverage.mpi .coverage.mpi')
    os.system('coverage combine')
    os.system('coverage report -m --include=\"*/pyGSTi/packages/pygsti/*\" > coverage_tests.out 2>&1')

    show_message('Combined output written to coverage_tests.out')
'''
'''
#!/bin/bash
echo "Serial tests started..."
time nosetests -v --with-coverage --cover-package=pygsti --cover-erase */test*.py > coverage_tests_serial.out 2>&1
mv .coverage coverage.serial
echo "Serial Output written to coverage_tests_serial.out"

echo "Parallel tests started..."
cp mpi/setup.cfg.mpi setup.cfg #stage setup.cfg
time mpiexec -np 4 python mpi/runtests.py -v --with-coverage --cover-package=pygsti --cover-erase mpi/mpitest*.py  > coverage_tests_mpi.out 2>&1
mv .coverage coverage.mpi
rm setup.cfg #unstage setup.cfg
echo "MPI Output written to coverage_tests_mpi.out"

cp coverage.serial .coverage.serial
cp coverage.mpi .coverage.mpi
coverage combine
coverage report -m --include="*/pyGSTi/packages/pygsti/*" > coverage_tests.out 2>&1
echo "Combined Output written to coverage_tests.out"

'''

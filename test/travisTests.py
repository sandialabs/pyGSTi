#!/usr/bin/env python3
from __future__ import print_function
from runTests   import run_tests
from helpers.automation_tools import get_branchname
import os, sys

def check_env(varname):
    return os.environ.get(varname, 'False') == 'True'

doReport      = check_env('Report')
doReportB     = check_env('ReportB')
doDrivers     = check_env('Drivers')
doDefault     = check_env('Default')
doMPI         = check_env('MPI') 
doAlgorithms  = check_env('Algorithms') 
doAlgorithmsB = check_env('AlgorithmsB') 

# I'm doing string comparison rather than boolean comparison because, in python, bool('False') evaluates to true
# Build the list of packages to test depending on which portion of the build matrix is running

tests    = []       # Run nothing if no environment variables are set
parallel = True     # By default
package  = 'pygsti' # Check coverage of all of pygsti by default

# All other reports tests
if doReport:
    tests = ['report']
    package = 'pygsti.report'

elif doReportB:
    tests = ['reportb']
    package = 'pygsti.report'

elif doDrivers:
    tests = ['drivers', 'objects']

elif doAlgorithms:
    #parallel = False
    tests = ['algorithms']
    package = 'pygsti.algorithms'

elif doAlgorithmsB:
    #parallel = False
    tests = ['algorithmsb']
    package = 'pygsti.algorithms'

elif doDefault:
    tests = ['tools', 'iotest', 'optimize', 'construction','extras']
    #parallel = False #multiprocessing bug in darwin (and apparently TravisCI) causes segfault if used.

elif doMPI:
    tests = ['mpi']
    parallel = False # Not for mpi

print('Running travis tests with python%s.%s' % (sys.version_info[0], sys.version_info[1]))

coverage = True

branchname = get_branchname()
print('Branchname is %s' % branchname)

threshold = 0 # Isn't representative of full coverage anyways

if branchname == 'develop':
    coverage  = False

run_tests(tests, parallel=parallel, coverage=coverage, threshold=threshold, outputfile='../output/test.out', package=package)

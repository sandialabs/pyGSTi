#!/usr/bin/env python3
from __future__ import print_function
from runTests   import run_tests
from helpers.automation_tools import get_branchname
import os, sys

doReportA = os.environ.get('ReportA', 'False')
doReportB = os.environ.get('ReportB', 'False')
doDrivers = os.environ.get('Drivers', 'False')
doDefault = os.environ.get('Default', 'False')
doMPI     = os.environ.get('MPI',     'False')

# I'm doing string comparison rather than boolean comparison because, in python, bool('False') evaluates to true
# Build the list of packages to test depending on which portion of the build matrix is running

tests    = []   # Run nothing if no environment variables are set
parallel = True # By default

if doReportA == 'True':
    tests    = ['report/testReport.py:TestReport.test_reports_logL_TP_wCIs']
    parallel = False # Not for mpi
    # Parallel tests break individual tests !! (and will instead run everything in a package)

# All other reports tests
elif doReportB == 'True':
    tests = ['report/%s' % filename for filename in [
    'testEBFormatters.py',
    'testMetrics.py',
    'testPrecisionFormatter.py',
    'testFormatter.py',
    'testWorkspace.py',
    'testTable.py']]
    # Removed: 'testAnalysis.py',
    # Removed: 'testFigureFormatter.py',

elif doDrivers == 'True':
    tests = ['drivers']

elif doDefault == 'True':
    tests = ['objects', 'tools', 'iotest', 'optimize', 'algorithms', 'construction','extras']

elif doMPI == 'True':
    tests = ['mpi']
    parallel = False # Not for mpi

print('Running travis tests with python%s.%s' % (sys.version_info[0], sys.version_info[1]))

coverage = True

branchname = get_branchname()
print('Branchname is %s' % branchname)

threshold = 0 # Isn't representative of full coverage anyways

if branchname == 'develop':
    coverage  = False

run_tests(tests, parallel=parallel, coverage=coverage, threshold=threshold, outputfile='../output/test.out')

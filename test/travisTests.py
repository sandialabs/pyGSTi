#!/usr/bin/env python3
from __future__ import print_function
from runTests   import run_tests
import os

doReportA = os.environ.get('ReportA', 'False')
doReportB = os.environ.get('ReportB', 'False')
doDrivers = os.environ.get('Drivers', 'False')
doDefault = os.environ.get('Default', 'False')
doMPI     = os.environ.get('MPI',     'False')

# I'm doing string comparison rather than boolean comparison because, in python, bool('False') evaluates to true
# Build the list of packages to test depending on which portion of the build matrix is running

tests = [] # Run nothing if no environment variables are set

# Only testReport.py (barely finishes in time!)
if doReportA == 'True':
    tests = ['report/testReport.py:TestReport.test_reports_logL_TP_wCIs'] # Maybe this single test wont time out? :)

# All other reports tests
elif doReportB == 'True':
    tests = ['report/%s' % filename for filename in [
    'testAnalysis.py',
    'testEBFormatters.py',
    'testMetrics.py',
    'testPrecisionFormatter.py']]

# Drivers
elif doDrivers == 'True':
    tests = ['drivers']

elif doDefault == 'True':
    tests = ['objects', 'tools', 'iotest', 'optimize', 'algorithms']

elif doMPI == 'True':
    tests = ['mpi']
    run_tests(tests) # Don't run mpi tests in parallel

# Begin by running all of the packages in the current test matrix
run_tests(tests, parallel=True)


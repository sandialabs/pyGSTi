#!/usr/bin/env python3
from __future__ import print_function
from helpers.test.runPackage  import run_package
from helpers.automation_tools import directory
import subprocess, sys, os

def run_specific(commands):
    print(commands)
    try:
        output = subprocess.check_output(commands)
    except subprocess.CalledProcessError as e:
        output = e.output

    output = output.decode('utf-8')
    print(output)

    if "ERROR" in output or "FAIL" in output:
        return False
    return True

doReportA = os.environ.get('ReportA',  'False')
doReportB = os.environ.get('ReportB',  'False')
doDrivers = os.environ.get('Drivers', 'False')
doDefault = os.environ.get('Default',   'False')

# I'm doing string comparison rather than boolean comparison because, in python, bool('False') evaluates to true
# Build the list of packages to test depending on which portion of the build matrix is running

individuals    = []
travisPackages = []

# Only testReport.py (barely finishes in time!)
if doReportA == 'True':
    individuals    = [['test_packages/report/testReport.py']] # Maybe this single test wont time out? :)

# All other reports tests
elif doReportB == 'True':
    individuals = [['test_packages/report/%s' % filename] for filename in (
        'testAnalysis.py',
        'testEBFormatters.py',
        'testMetrics.py',
        'testPrecisionFormatter.py')]

# Drivers configuration: run specific drivers tests only
elif doDrivers == 'True':
    travisPackages = ['drivers']

elif doDefault == 'True':
    travisPackages = ['objects', 'tools', 'iotest', 'optimize', 'algorithms']

# Begin by running all of the packages in the current test matrix
results = []
with directory('test_packages'):
    for package in travisPackages:
        print('Running %s' % package)
        result = run_package(package)
        results.append((result, package))

failed = [(result[1], result[0][1]) for result in results if not result[0][0]]

if len(failed) > 0:
    for failure in failed:
        print('%s Failed: %s' % (failure[0], failure[1]))
    sys.exit(1)

# Now run all of the specific tests
for individual in individuals:
    print('Running %s' % individual)
    result = run_specific(['python', '-m', 'nose'] + individual)
    if not result:
        sys.exit(1)


#!/usr/bin/env python
from __future__ import print_function
from helpers.test.runPackage  import run_package
from helpers.automation_tools import directory
import subprocess, sys
travisPackages = ['tools', 'objects', 'construction', 'iotest', 'optimize', 'algorithms']

results = []
with directory('test_packages'):
    for package in travisPackages:
        result = run_package(package)
        results.append((result, package))

failed = [(result[1], result[0][1]) for result in results if not result[0][0]]

if len(failed) > 0:
    for failure in failed:
        print('%s Failed: %s' % (failure[0], failure[1]))
    sys.exit(1)

def run_specific(commands):
    try:
        output = subprocess.check_output(commands)
    except subprocess.CalledProcessError as e:
        output = e.output

    output = output.decode('utf-8')
    print(output)

    if "ERROR" in output or "FAIL" in output:
        return False
    return True

# Run specific tests in report and drivers that get large coverage over a short time
report = run_specific(['python', 'test_packages/report/testReport.py', 'TestReport.test_reports_logL_TP_wCIs'])
if not report:
    sys.exit(1)
drivers = run_specific(['python', 'test_packages/drivers/testDrivers.py', 'TestDriversMethods.test_bootstrap'])
if not drivers:
    sys.exit(1)
sys.exit(0)

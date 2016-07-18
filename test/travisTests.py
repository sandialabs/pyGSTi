#!/usr/bin/env python
from __future__ import print_function
from test_tools.runPackage import run_package
import subprocess, sys
travisPackages = ['tools', 'objects', 'construction', 'io', 'optimize', 'algorithms']

results = []
for package in travisPackages:
    result = run_package(package)
    results.append((result, package))

failed = [(result[1], result[0][1]) for result in results if not result[0][0]]

if len(failed) > 0:
    for failure in failed:
        print('%s Failed: %s' % (failure[0], failure[1]))
    sys.exit(1)

def get_test_info(filename, packageName, testCaseName, test):
    commands = ['nosetests', '-v', '--with-coverage', '--cover-package=pygsti.%s' % packageName, 
                '--cover-erase', '%s:%s.%s' % (filename, testCaseName, test), '2>&1 | tee temp.out']
    start   = time.time()
    percent = _read_coverage(' '.join(commands), 'temp.out')
    end     = time.time()
    return ((end - start), percent)

try:
    output = subprocess.check_output(['python', 'report/testReport.py', 'TestReport.test_reports_logL_TP_wCIs'])
except subprocess.CalledProcessError as e:
    output = e.output

output = output.decode('utf-8')
print('Running a single report test that maximizes coverage to time ratio (71% in six minutes!!)')
print(output)

if "ERROR" in output or "FAIL" in output:
    sys.exit(1)

sys.exit(0)





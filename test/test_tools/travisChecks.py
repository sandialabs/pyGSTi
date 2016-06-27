#!/usr/bin/python
from __future__ import print_function
from runPackage import run_package
import sys

travisPackages = ['tools', 'objects', 'construction', 'io', 'optimize', 'algorithms']

results = []
for package in travisPackages:
    result = run_package(package)
    results.append((result, package))

failed = [(result[1], result[0][1]) for result in results if not result[0][0]]

if len(failed) == 0:
    sys.exit(0)
else:
    for failure in failed:
        print('%s Failed: %s' % (failure[0], failure[1]))
    sys.exit(1)

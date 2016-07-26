#!/usr/bin/env python3
from __future__                import print_function
from helpers.test.helpers      import *
from helpers.automation_tools  import read_yaml, write_yaml, directory, get_output
from importlib import import_module
from inspect   import isclass
import unittest
import subprocess
import argparse
import time
import os

def find_individual_tests(testCase):
    istest = lambda a, case : callable(getattr(case, a)) and a.startswith('test')
    return [attr for attr in dir(testCase) if istest(attr, testCase)]

def make_coverage_output(output, amount):
    output   = output.split('-------------------------------------------------------------')
    specific = output[1]
    # Get last word of the line after the dashes, and remove the percent symbol
    percent  = int(output[2].split()[-1][:-1])
    specific = [line for line in specific.splitlines() if line != '']

    coverageDict = {}
    for line in specific:
        line          = line.split()
        filename      = line[0].rsplit('/', 1)[1]
        modulepercent = int(line[3][:-1])
        missedlines   = line[4:]
        coverageDict[filename] = (modulepercent, missedlines, amount)
    return coverageDict

def get_info(filename, packageName, testCaseName, test):
    coverageFile = 'output/individual_coverage/%s/%s%s.yml' % (packageName, testCaseName, test)
    commands = ['nosetests',
                '--with-coverage',
                '--cover-package=pygsti.%s' % packageName,
                '--cover-erase',
                'test_packages/%s/%s:%s.%s' % (packageName, filename, testCaseName, test)]

    start = time.time()
    output = get_output(commands)
    end = time.time()

    print(output)
    coverageDict = make_coverage_output(output, end-start)
    write_yaml(coverageDict, coverageFile)

def get_test_files(packageName):
    files = next(os.walk('test_packages/%s' % packageName))[2]
    is_test_file = lambda name : name.startswith('test') and name.endswith('.py')
    return [filename for filename in files if is_test_file(filename)]

is_test = lambda name, case : 'test' in name and callable(getattr(case, name))
def get_casetests(case):
    return [name for name in dir(case) if is_test(name, case)]

def get_file_tests(testFile, packageName):
    moduleName = 'test_packages.%s.%s' % (packageName, testFile[:-3])
    package    = import_module(moduleName)
    cases      = [getattr(package, name) for name in dir(package) if isclass(getattr(package, name))]
    tests      = [(case.__name__, get_casetests(case)) for case in cases]
    return tests

def get_tests(packageName):
    return [(testFile, get_file_tests(testFile, packageName)) for testFile in get_test_files(packageName)]

if __name__ == '__main__':
    tests = get_tests('tools')
    for filename, innertests in tests:
        for case, testnames in innertests:
            for testname in testnames:
                get_info(filename, 'tools', case, testname)

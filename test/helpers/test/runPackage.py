from __future__         import print_function, absolute_import
from .helpers           import *
from contextlib         import contextmanager
from ..automation_tools import directory
import os, sys
import subprocess
import pickle

last_failed_file = 'last_failed.pkl'


def get_last_failed(packageName):
    with open(last_failed_file, 'rb') as picklefile:
        return pickle.load(picklefile)

def create_last_failed(testnames, packageName):
    with open(last_failed_file, 'wb') as output:
        pickle.dump(testnames, output)

def run_subprocess(commands):
    try:
        result = subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True,  result
    except subprocess.CalledProcessError as e:
        return False, e.output

def find_failed(output):
    output = output.decode('utf-8')
    failedlines = [line for line in output.splitlines() if '... ERROR' in line or '... FAIL' in line]
    failedtests = []
    for line in failedlines:
        line      = line.split(' ')
        testname  = line[0]                         # use the first token as the test name
        testclass = line[1][len('__main__.('):-1] # remove parentheses and modulename from the second token
        failedtests.append(testclass + '.' + testname)
    return failedtests

def run_test(commands, filepath, failedtests):
    result = run_subprocess(commands)
    print('Test output:')
    print(result[1].decode('utf-8'))
    if not result[0]:
        failedtests.append((find_failed(result[1]), filepath))

def get_test_names(failedtests):
    names = [(item, testname[1]) for testname in failedtests for item in testname[0]]
    return set(names)

def run_package(packageName, precommands=['python', '-m', 'nose'], postcommand=None, lastFailed=''):
    # enter the package directory to begin running tests and leave when done
    with directory(packageName):
        failedtests = []

        if lastFailed == 'True' and os.path.isfile(last_failed_file):
            print('Running last failed tests in %s' % packageName)
            for testname in get_last_failed(packageName):
                # Run a specific test in each file, ex: python testMyCase.py MyCase.testItIsHot
                print('Running %s' % testname[1].rsplit('/', 1)[1])
                run_test(precommands + [testname[1], testname[0]], testname[1], failedtests)
        else:
            print('Running all tests in %s' % packageName)

            filenames = get_file_names()
            for filename in filenames:
                filepath = filenames[filename]
                # run ALL test files, even those that passed previously
                print('Running %s' % filename)
                commands = precommands + [filepath]
                if postcommand is not None: commands.append(postcommand)
                run_test(commands, filepath, failedtests)

        testnames = get_test_names(failedtests)
        create_last_failed(testnames, packageName)
        if len(testnames) > 0:
            return (False, testnames)
        else:
            return (True, [])

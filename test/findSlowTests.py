#!/usr/bin/env python3
from __future__         import print_function
from test_tools.helpers import *
from test_tools._getCoverage import _read_coverage
import importlib
import inspect
import time


def get_test_files(directory, extension='.py'):
    testFiles = []
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filename.endswith(extension) and filename.startswith('test'):
                testFiles.append((filepath, filename))
    return testFiles

def find_individual_tests(testCase):
    istest = lambda a, case : callable(getattr(case, a)) and a.startswith('test')
    return [attr for attr in dir(testCase) if istest(attr, testCase)]

def get_test_info(filename, packageName, testCaseName, test):
    commands = ['nosetests', '-v', '--with-coverage', '--cover-package=pygsti.%s' % packageName, 
                '--cover-erase', '%s:%s.%s' % (filename, testCaseName, test), '2>&1 | tee temp.out']
    start   = time.time()
    percent = _read_coverage(' '.join(commands), 'temp.out')
    end     = time.time()
    return ((end - start), percent)

def gen_individual_test_info(packageName):
    testsInfo = {}

    directory = os.getcwd() + '/' + packageName + '/'
    for _, filename in get_test_files(directory):
        moduleName = packageName + '.' + filename[:-3]
        print('Finding slow tests in %s' % moduleName)
        i = importlib.import_module(moduleName)
        testCases = inspect.getmembers(i, predicate=inspect.isclass)
        for testCaseName, testCase in testCases:
            tests = find_individual_tests(testCase)
            for testName in tests:
                testcase = testCase()
                testcase.setUp()
                info = get_test_info(filename, packageName, testCaseName, testName)
                testsInfo[filename[:-3] + '.' + testCaseName + '.' + testName] = info
                testcase.tearDown()
    return testsInfo

                
gen_info_on = sys.argv[1:]
infoDict = {}
for packageName in gen_info_on:
    infoDict[packageName] = gen_individual_test_info(packageName)

    infoString = ''
    infoString += '\n\n%s:\n\n' % packageName
    for testfunction in infoDict[packageName]:
        infoString += '\n    %s:\n' % testfunction
        testTime, coverage = infoDict[packageName][testfunction]
        infoString += ('        - %s\n' % str(round(testTime, 2)).ljust(5))  
        infoString += ('        - %s\n' % coverage)

    print(infoString)
    with open('output/%s_individual_test_info.yml' % packageName, 'w') as testInfo:
        testInfo.write(infoString)
    infoDict[packageName] = infoString

with open('output/all_individual_test_info.yml', 'w') as testInfo:
    info = '\n*3'.join([infoDict[packageName] for packageName in infoDict])
    testInfo.write(info)

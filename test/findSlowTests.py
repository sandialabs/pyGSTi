from __future__         import print_function
from contextlib         import contextmanager
from test_tools.helpers import *
from test_tools._getCoverage import _read_coverage
import subprocess
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
        i = importlib.import_module(packageName + '.' + filename[:-3])
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

                
gen_info_on = ['tools']
infoDict = {}
for packageName in gen_info_on:
    infoDict[packageName] = gen_individual_test_info(packageName)

for packageName in infoDict:
    print('\n\n%s:\n\n' % packageName)
    for testfunction in infoDict[packageName]:
        print('\n    %s:' % testfunction)
        time, coverage = infoDict[packageName][testfunction]
        print('        %s seconds | %s%% coverage \n' % (str(round(time, 2)).ljust(5), coverage))


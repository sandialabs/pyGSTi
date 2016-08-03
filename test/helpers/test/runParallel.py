from __future__         import print_function
from multiprocessing    import cpu_count # This is all that is used from this module. All parallellism is done with subprocess.Popen
from subprocess         import Popen     # Does all the actual parallel speedups
from time               import sleep     # Wait for processes

from ..info.genInfo     import get_tests, get_test_files, get_file_tests
from ..automation_tools import directory
from .helpers            import *
import sys
import os

get_pfilename = lambda test : '../output/%s.pout' % test.replace('/', '.')

def expand(testname, packages, files):
    with directory('..'):
        if testname in packages:
            expanded = get_tests(testname)
            expandedtests = []
            for filename, testcases in expanded:
                for case, tests in testcases:
                    for test in tests:
                        expandedtests.append('%s/%s:%s.%s' % (testname, filename, case, test))
            return expandedtests
        elif testname.count('/') > 0 and testname.rsplit('/')[-1] in files:
            with directory('..'):
                expanded = get_file_tests(testname.replace('/', '.'))
            expandedtests = []
            for case, tests in expanded:
                for test in tests:
                    expandedtests.append('%s:%s.%s' % (testname, case, test))
            return expandedtests
        else:
            return [testname]

# Push finished processes into results, keeping those we are waiting for
def get_waiting_for(processes, results):
    waiting_for = []
    for process, processname in processes:
        if process.poll() is None:
            waiting_for.append((process, processname))
        else:
            print('Finished:    %s' % processname)
            results.append((process.wait(), processname))
    return waiting_for

# Start as many processes as we have cores
def start_processes(processes, nCores, testnames, pythoncommands, postcommands):
    while len(processes) < nCores:
        if len(testnames) > 0:
            test = testnames.pop()
            with open(get_pfilename(test), 'w') as output:
                processes.append((Popen(pythoncommands + [test] + postcommands, stdout=output, stderr=output),
                                 test))
            print('Started:     %s' % test)
        else:
            print('No more processes to queue')
            break

# Break a list of tests into individual tests, run them one at a time
def run_parallel(testnames, pythoncommands, postcommands, nCores):

    packages = [name for name in get_package_names()]
    with directory('..'):
        files = [testfile for package in packages for testfile in get_test_files(package)]

    # Expand test names
    testnames = [expand(name, packages, files) for name in testnames]
    testnames = [name for subtests in testnames for name in subtests] # join lists

    if nCores is None:
        nCores    = cpu_count() # From multiprocessing (native to python)
    processes = []
    results   = []
    numTests  = len(testnames)

    print('Parallelizing tests across %s cores' % nCores)

    start_processes(processes, nCores, testnames, pythoncommands, postcommands)

    while True:
        for i in range(5):
            sleep(.1)
        processes = get_waiting_for(processes, results)
        start_processes(processes, nCores, testnames, pythoncommands, postcommands)
        if len(testnames) == 0:
            print('Done queuing processes, waiting')
            while len(processes) > 0:
                processes = get_waiting_for(processes, results)
        if len(processes) == 0:
            print('All processes finished')
            break

    failed = False
    for processreturn, processname in results:
        if processreturn > 0:
            print('The process %s failed with: %s. Output was:\n' % (processname, processreturn))
            with open(get_pfilename(processname), 'r') as processoutput:
                print(processoutput.read(), end='\n\n')
            failed = True
        else:
            print('The process %s was successful.' % processname)
    return failed

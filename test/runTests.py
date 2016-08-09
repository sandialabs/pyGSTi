#!/usr/bin/env python3
from __future__                  import print_function, division, unicode_literals, absolute_import
from helpers.automation_tools    import directory, get_changed_packages
import subprocess, argparse, sys

'''
Script for running the test suite.

see pyGSTi/doc/repotools/test.md, or try running ./runTests.py -h

'''

default   = ['tools', 'io', 'objects', 'construction', 'drivers', 'report', 'algorithms', 'optimize']
slowtests = ['report', 'drivers']

def parse_coverage_percent(output):
    output   = output.split('Missing')[1].split('Ran')[0]
    output   = output.splitlines()
    specific = output[3:-3]
    # Get last word of the line after the dashes, and remove the percent symbol
    percent  = int(output[-2].split()[-1][:-1])
    return percent


def run_tests(testnames, version=None, fast=False, changed=False,
              parallel=False, failed=False, cores=None, coverdir=None, html=False, threshold=90):

    with directory('test_packages'):

        # Don't run report or drivers
        if fast:
            for slowtest in slowtests:
                testnames.remove(slowtest)

        # Specify the versions of your test :)
        if version is None:
            # The version this file was run/imported with
            pythoncommands = ['python%s.%s' % (sys.version_info[0], sys.version_info[1])]
        else:
            # The version specified
            pythoncommands = ['python%s' % version]
        # Always use nose
        pythoncommands += ['-m', 'nose']

        # Since last commit to current branch
        if changed:
            testnames = [name for name in testnames if name in get_changed_packages()]

        if len(testnames) == 0:
            print('No tests to run')
            sys.exit(0)

        # testnames should be final at this point
        print('Running tests:\n%s' % ('\n'.join(testnames)))

        # Use the failure monitoring native to nose
        postcommands = ['--with-id']
        if failed:
            postcommands = ['--failed']# ~implies --with-id

        # Use parallelism native to nose
        if parallel:
            if cores is None:
                pythoncommands.append('--processes=-1')
                # (-1) will use all cores
            else:
                pythoncommands.append('--processes=%s' % cores)
            # Some tests take up to an hour
            pythoncommands.append('--process-timeout=14400') # Four hours

        # html coverage is prettiest
        pythoncommands += ['--with-coverage']

        if html:
            pythoncommands += ['--cover-html']

        # Build the set of covered packages automatically
        covering = set()
        for name in testnames:
            if name.count('/') > 1:
                covering.add(name.split('/')[0])
            else:
                covering.add(name)
        for coverpackage in covering:
            pythoncommands.append('--cover-package=pygsti.%s' % coverpackage)

        if coverdir is None:
            coverdir = '_'.join(covering)
        pythoncommands.append('--cover-html-dir=../output/coverage/%s' % coverdir)

        pythoncommands.append('--cover-min-percentage=%s' % threshold)

        # Make a single subprocess call

        returned = subprocess.call(pythoncommands + testnames + postcommands, stderr=subprocess.STDOUT)

        sys.exit(returned)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run tests for pygsti')
    parser.add_argument('tests', nargs='*', default=default, type=str,
                        help='list of packages to run tests for')
    parser.add_argument('--version', '-v', type=str,
                        help='version of python to run the tests under')
    parser.add_argument('--changed', '-c', action='store_true', help='run only the changed packages')
    parser.add_argument('--fast', '-f', action='store_true',
                        help='run only the faster packages')
    parser.add_argument('--failed', action='store_true',
                        help='run last failed tests only')
    parser.add_argument('--html', action='store_true',
                        help='generate html')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='run tests in parallel')
    parser.add_argument('--cores', type=int, default=None,
                        help='run tests with n cores')
    parser.add_argument('--coverdir', type=str, default='all',
                        help='put html coverage report here')
    parser.add_argument('--threshold', type=int, default=90,
                        help='coverage percentage to beat')

    parsed = parser.parse_args(sys.argv[1:])

    run_tests(parsed.tests, parsed.version, parsed.fast, parsed.changed,
              parsed.parallel, parsed.failed, parsed.cores, parsed.coverdir,
              parsed.html, parsed.threshold)

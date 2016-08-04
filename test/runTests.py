#!/usr/bin/env python3
from __future__                  import print_function, division, unicode_literals, absolute_import
from helpers.test                import *
from helpers.test.runChanged     import *
from helpers.automation_tools    import directory
import sys
import argparse

def get_excluded():
    return ['__pycache__', 'cmp_chk_files', 'temp_test_files', 'testutils']

def run_tests(testnames, version=None, fast=False, changed=False, 
              parallel=False, failed=False, cores=None):

    slowTests = ['report', 'drivers']

    print('Testnames %s' % testnames)

    packages = get_package_names()

    with directory('test_packages'):

        if fast:
            exclude += slowTests # Shave off ~3 hrs?

        # Specify the versions of your test :)
        if version is None:
            pythoncommands = ['python%s.%s' % (sys.version_info[0], sys.version_info[1])]
        else:
            pythoncommands = ['python%s' % version]
        pythoncommands += ['-m', 'nose']

        # Since last commit to current branch
        if changed:
            testnames = [name for name in get_changed_test_packages() if name in testnames] # Run only the changed packages we specify

        print('Running tests %s' % ('    \n     '.join(testnames)))

        # Use the failure monitoring native to nose
        postcommands = ['--with-id']
        if failed:
            postcommands = ['--failed']

        if parallel:
            if cores is None:
                pythoncommands.append('--processes=-1') # Let nose figure out how to parallelize things
            else:
                pythoncommands.append('--processes=%s' % cores)
            pythoncommands.append('--process-timeout=3600') # Yikes!

        pythoncommands += ['--with-coverage', '--cover-html']
        covering = set()
        for name in testnames:
            if name.count('/') > 1:
                covering.add(name.split('/')[0])
            else:
                covering.add(name)
        for coverpackage in covering:
            pythoncommands.append('--cover-package=pygsti.%s' % coverpackage)
        pythoncommands.append('--cover-html-dir=../output/coverage/%s' % '_'.join(covering))

        result = subprocess.call(pythoncommands + testnames + postcommands)
        sys.exit(result)

if __name__ == "__main__":

    with directory('test_packages'):
        defaultpackages = [name for name in get_package_names() if name not in get_excluded()]

    parser = argparse.ArgumentParser(description='Run tests for pygsti')
    parser.add_argument('tests', nargs='*', default=defaultpackages, type=str,
                        help='list of packages to run tests for')
    parser.add_argument('--version', '-v', type=str,
                        help='version of python to run the tests under')
    parser.add_argument('--changed', '-c', action='store_true',
                        help='run only the changed packages')
    parser.add_argument('--fast', '-f', action='store_true',
                        help='run only the faster packages')
    parser.add_argument('--failed', action='store_true',
                        help='run last failed tests only')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='run tests in parallel')
    parser.add_argument('--cores', type=int, default=None,
                        help='run tests with n cores')

    parsed = parser.parse_args(sys.argv[1:])

    run_tests(parsed.tests, parsed.version, parsed.fast, parsed.changed, parsed.parallel, parsed.failed, parsed.cores)

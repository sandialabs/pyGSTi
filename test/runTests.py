#!/usr/bin/env python3
from __future__                  import print_function, division, unicode_literals, absolute_import
from helpers.test                import *
from helpers.test.runChanged     import *
from helpers.test.runParallel    import run_parallel
from helpers.automation_tools    import directory
from helpers.info.genInfo        import get_tests, get_test_files, get_file_tests
import sys
import argparse

def get_excluded():
    return ['__pycache__', 'cmp_chk_files', 'temp_test_files', 'testutils']

def expand(testname, packages, files):
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

def run_tests(testnames, version=None, fast=False, changed=False, parallel=False, failed=False):
    slowTests = ['report', 'drivers']

    with directory('test_packages'):
        packages = [name for name in get_package_names() if name not in get_excluded()]
    files = [testfile for package in packages for testfile in get_test_files(package)]

    testnames = [expand(name, packages, files) for name in testnames]
    testnames = [name for subtests in testnames for name in subtests] # join lists

    print('Testnames %s' % testnames)

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
            if not run_parallel(testnames, pythoncommands, postcommands):
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            sys.exit(subprocess.call(pythoncommands + testnames + postcommands))

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
                        help='run last failed tests only')

    parsed = parser.parse_args(sys.argv[1:])

    run_tests(parsed.tests, parsed.version, parsed.fast, parsed.changed, parsed.parallel, parsed.failed)

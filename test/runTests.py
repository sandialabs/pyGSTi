#!/usr/bin/env python
from __future__                  import print_function, division, unicode_literals, absolute_import
from helpers.test                import *
from helpers.test.runChanged     import *
from helpers.test.runPackage     import run_package
from helpers.automation_tools    import read_yaml, directory
import sys
import argparse


if __name__ == "__main__":
    config = read_yaml('config/test_config.yml')
    slowTests = config['slow-tests']

    with directory('test_packages'):

        exclude = ['__pycache__', 'cmp_chk_files', 'temp_test_files']
        defaultpackages = [name for name in get_package_names() if name not in exclude]

        parser = argparse.ArgumentParser(description='Run tests for pygsti')
        parser.add_argument('packages', nargs='*', default=defaultpackages, type=str,
                            help='list of packages to run tests for')
        parser.add_argument('--version', '-v', type=float,
                            help='version of python to run the tests under')
        parser.add_argument('--changed', '-c', type=bool,
                            help='run only the changed packages')
        parser.add_argument('--fast', '-f', type=bool,
                            help='run only the faster packages')
        parser.add_argument('--nose', '-n', type=bool,
                            help='run tests with nosetests')
        parser.add_argument('--lastFailed', '-l', type=bool,
                            help='run last failed tests only')

        parsed = parser.parse_args(sys.argv[1:])

        # Setup arguments and other variables:

        # Specify the versions of your test :)
        if parsed.version is None:
            pythonCommands = ['python%s.%s' % (sys.version_info[0], sys.version_info[1])]
        else:
            pythonCommands = ['python%s' % parsed.version]

        if parsed.fast:
            exclude += slowTests # Shave off ~3 hrs?

        # Since last commit to current branch
        if parsed.changed:
            packageNames = [name for name in get_changed_test_packages() if name in parsed.packages]
        else:
            packageNames = parsed.packages

        # Only running tests
        if parsed.nose:
            pythonCommands += ['-m', 'nose']

        print('Running packages %s' % (', '.join(packageNames)))

        lastFailed = parsed.lastFailed
        for package in packageNames:
            run_package(package, precommands=pythonCommands,
                        postcommand=None, lastFailed=lastFailed)

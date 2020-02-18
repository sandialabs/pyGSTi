#!/usr/bin/env python3
from helpers.automation_tools    import directory, get_changed_packages
import subprocess, argparse, shutil, sys, os
import webbrowser

'''
Script for running the test suite.

see pyGSTi/doc/repotools/test.md, or try running ./runTests.py -h
'''

def run_mpi_coverage_tests(coverage_cmd, nproc=4):
    shutil.copy('mpi/setup.cfg.mpi', 'setup.cfg')

    #OLD: worked with python2.7, but not with 3 (where .coverage files turned to JSON)
    #mpicommands = ('time mpiexec -np %s python%s mpi/runtests.py -v ' % (str(nproc), '' if version is None else version)+
    #               '--with-coverage --cover-package=pygsti --cover-erase mpi/testmpi*.py  ' +
    #               '> ../output/coverage_tests_mpi.out 2>&1')

    mpicommands = ('time mpiexec -np %s %s run -p ' % (str(nproc), coverage_cmd) +
                   '--source=pygsti mpi/runtests.py -v mpi/testmpi*.py ' +
                   '> ../output/coverage_tests_mpi.out 2>&1')

    with open('../output/mpi_output.txt', 'w') as output:
        returned = subprocess.call(mpicommands, shell=True, stdout=output, stderr=output)
    with open('../output/mpi_output.txt', 'r') as output:
        print(output.read())
    os.remove('setup.cfg')
    return returned

def create_html(dirname, coverage_cmd):
    subprocess.call([coverage_cmd, 'html', '--directory=%s' % dirname])

default   = ['tools', 'iotest', 'objects', 'construction', 'drivers', 'report', 'reportb', 'algorithms', 'algorithmsb', 'optimize', 'extras', 'mpi']
slowtests = ['report', 'drivers']

def run_tests(testnames, version=None, fast=False, changed=False, coverage=True,
              parallel=False, failed=False, cores=None, coverdir='../output/coverage', html=False,
              threshold=90, outputfile=None, package='pygsti', scriptfile=None, timer=False):

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
        pythoncommands += ['-m', 'nose', '-v']

        # Since last commit to current branch
        if changed:
            testnames = [name for name in testnames if name in get_changed_packages()]

        if len(testnames) == 0:
            print('No tests to run')
            sys.exit(0)

        # testnames should be final at this point
        print('Running tests:\n%s' % ('\n'.join(testnames)))

        # Version-specific coverage cmds only sometimes have a dash
        #  e.g.: coverage, coverage2, coverage3, coverage-2.7, coverage-3.5
        if version is None: coverage_cmd = "coverage"
        elif "." in version: coverage_cmd = 'coverage-%s' % version
        else:                coverage_cmd = 'coverage%s' % version

        # Run mpi coverage tests differently
        covermpi = ('mpi' in testnames) and coverage 
        if covermpi:
            testnames.remove('mpi')

        postcommands = []
        # Use parallelism native to nose
        if parallel:
            if cores is None:
                pythoncommands.append('--processes=-1')
                # (-1) will use all cores
            else:
                pythoncommands.append('--processes=%s' % cores)
            # Some tests take up to an hour
            pythoncommands.append('--process-timeout=14400') # Four hours
        else:
            # Use the failure monitoring native to nose
            postcommands = ['--with-id']
            if failed:
                postcommands = ['--failed']# ~implies --with-id

        if coverage:
            # html coverage is prettiest
            pythoncommands += ['--with-coverage',
                               '--cover-erase',
                               '--cover-package={}'.format(package),
                               '--cover-min-percentage={}'.format(threshold)]

        if timer:
            pythoncommands.append('--with-timer')

        returned = 0
        if len(testnames) > 0:
            commands = pythoncommands + testnames + postcommands
            commandStr = ' '.join(commands)
            
            if scriptfile:
                #Script file runs command directly from shell so output works normally
                # (using subprocess on TravisCI gives incomplete output sometimes).  It
                # uses a sleep loop to ensure some output is printed every 9 minutes,
                # as TravisCI terminates a process when it goes 10m without output.
                with open(scriptfile, 'w') as script:
                    print("#!/usr/bin/bash",file=script)
                    print('echo "%s"' % commandStr, file=script)
                    print('while sleep 540; do echo "=====[ $SECONDS seconds ]====="; done &', file=script)
                    print(commandStr,file=script)
                    print('kill %1', file=script) # Kill background sleep loop
                print("Wrote script file %s" % os.path.join('test_packages',scriptfile)) # cwd == 'test_packages'
                sys.exit(0)
            else:
                print(commandStr)

            if outputfile is None:
                returned = subprocess.call(commands)

            else:
                with open(outputfile, 'w') as testoutput:
                    returned = subprocess.call(commands, stdout=testoutput, stderr=testoutput)
                with open(outputfile, 'r') as testoutput:
                    print(testoutput.read())

        if parallel:
            #Only combine when run in parallel mode, since this
            # causes nose tests to create .coverage.<processid>
            # files instead of just a single .coverage file, which
            # "coverage combine" will overwrite with no-data (eek!).
            subprocess.call([coverage_cmd, 'combine'])

        if covermpi:
            print('Running mpi with coverage')
            # Combine serial/parallel coverage
            serial_coverage_exists = bool(len(testnames) > 0)

            if serial_coverage_exists: 
                #In this case, nose tests have erased old coverage files
                shutil.copy2('.coverage', '../output/temp_coverage')
            else:
                #If no serial tests have run, then we need to erase old files
                subprocess.call([coverage_cmd, 'erase'])

            run_mpi_coverage_tests(coverage_cmd) #creates .coverage.xxx files

            if serial_coverage_exists: 
                shutil.copy2('../output/temp_coverage', '.coverage.serial')

            subprocess.call([coverage_cmd, 'combine']) #combine everything

        if html:
            create_html(coverdir, coverage_cmd)
            webbrowser.open(coverdir + '/index.html')

        sys.exit(returned)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run tests for pygsti')
    parser.add_argument('tests', nargs='*', default=default, type=str,
                        help='list of packages to run tests for')
    parser.add_argument('--package', type=str, default='pygsti',
                        help='package to test coverage for')
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
    parser.add_argument('--cover', action='store_true',
                        help='generate coverage')
    parser.add_argument('--cores', type=int, default=None,
                        help='run tests with n cores')
    parser.add_argument('--coverdir', type=str, default='../output/coverage',
                        help='put html coverage report here')
    parser.add_argument('--threshold', type=int, default=90,
                        help='coverage percentage to beat')
    parser.add_argument('--output', type=str, default=None,
                        help='outputfile')
    parser.add_argument('--script', type=str, default=None,
                        help='scriptfile')
    parser.add_argument('--with-timer', '-t', action='store_true',
                        help='run tests in parallel')
        

    parsed = parser.parse_args(sys.argv[1:])

    # With this many arguments, maybe this function should be refactored?
    run_tests(parsed.tests, parsed.version, parsed.fast, parsed.changed, parsed.cover,
              parsed.parallel, parsed.failed, parsed.cores, parsed.coverdir,
              parsed.html, parsed.threshold, parsed.output, parsed.package,
              parsed.script, parsed.with_timer)

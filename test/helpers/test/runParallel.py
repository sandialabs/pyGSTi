from __future__      import print_function
from multiprocessing import cpu_count # This is all that is used from this module. All parallellism is done with subprocess.Popen
from subprocess      import Popen     # Does all the actual parallel speedups
from time            import sleep     # Wait for processes
import sys

get_pfilename = lambda test : '../output/%s.pout' % test.replace('/', '.')

# Push finished processes into results, keeping those we are waiting for
def get_waiting_for(processes, results):
    waiting_for = []
    for process, processname in processes:
        if process.poll() is None:
            print('Still waiting for %s' % processname)
            waiting_for.append((process, processname))
        else:
            print('%s is done' % processname)
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
            print('Started %s' % test)
        else:
            print('No more processes to queue')
            break

# Break a list of tests into individual tests, run them one at a time
def run_parallel(testnames, pythoncommands, postcommands):

    nCores    = cpu_count() # From multiprocessing (native to python)
    processes = []
    results   = []
    numTests  = len(testnames)

    print('Parallelizing tests across %s cores' % nCores)

    start_processes(processes, nCores, testnames, pythoncommands, postcommands)

    while True:
        for i in range(5):
            print('.', end='')
            sleep(.1)
            sys.stdout.flush()
        print('')
        processes = get_waiting_for(processes, results)
        start_processes(processes, nCores, testnames, pythoncommands, postcommands)
        print('Checking processes\n')
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

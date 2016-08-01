import subprocess
import time
import sys

def are_finished(processes):
    done = True
    for process, processname in processes:
        if process.poll() is None:
            print('Still waiting for %s' % processname)
            done = False
        else:
            print('%s is done' % processname)
    return done

def run_parallel(testnames, pythoncommands, postcommands):
    processes = []
    for test in testnames:
        with open('../output/%s.pout' % test, 'w') as output:
            processes.append((subprocess.Popen(pythoncommands + [test] + postcommands, stdout=output, stderr=output),
                             test))
        print('Started %s' % test)

    while not are_finished(processes):
        for i in range(10):
            print('.', end='')
            time.sleep(1)
            sys.stdout.flush()
        print('')
        print('Checking processes\n')

    failed = False
    for process, processname in processes:
        returncode = process.wait()
        if returncode > 0:
            print('The process %s failed with: %s. Output was:\n' % (processname, returncode))
            with open('../output/%s.pout' % processname, 'r') as processoutput:
                print(processoutput.read(), end='\n\n')
            failed = True
        else:
            print('The process %s was successful.' % processname)
    return failed

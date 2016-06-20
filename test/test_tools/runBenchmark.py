from __future__ import print_function
from runModule  import run_module
from benchmarks import benchmark
from tool       import tool
import os, sys

get_bench_file = lambda benchDir, name : '%s%s.bench' % ((benchDir if benchDir != None else ''), name.replace('.py', ''))

def benchmark_template(fullpath, command, benchDir=None):
    directory, name = fullpath.rsplit('/', 1)
    benchFile = get_bench_file(benchDir, name)
    @benchmark(benchFile)
    def template():
        command(fullpath)
    template()

def benchmark_module(moduleName, benchDir=None):
    benchmark_template(moduleName, run_module, benchDir)

def benchmark_file(filename, benchDir=None):
    run_file = lambda filename : os.system('python %s' % filename)
    benchmark_template(filename, run_file, benchDir)

@tool
def run_benchmarks(names, benchDir=None, allModules=False):
    # build modulenames and filenames
    _, moduleNames, _ = os.walk(os.getcwd()).next()
    fileNames = {}
    for subdir, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith('.py') and filename.startswith('test'):
                fileNames[filename] = subdir + os.sep + filename


    for name in names:
        if name in moduleNames:
            benchmark_module(os.getcwd() + '/' + name, benchDir)
        elif name in fileNames:
            # send the full filepath to benchmark_file()
            benchmark_file(fileNames[name], benchDir)
        else:
            print('%s is neither a valid modulename, nor a valid filename' % name)


if __name__ == "__main__":
    args      = [[arg for arg in sys.argv[1:] if not arg.startswith('--')]] # create args
    optionals = [arg for arg in sys.argv[1:] if arg.startswith('--')]
    kwargs    = {}
    # create kwargs
    for optional in optionals:
        kv = optional[2:].split('=') # remove prepending '--' and seperate into key : value
        kwargs[kv[0]] = kv[1]

    run_benchmarks(*args, **kwargs)

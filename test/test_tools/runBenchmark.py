from __future__  import print_function
from runModule   import run_module
from benchmarks  import benchmark
from helpers     import *
from getCoverage import get_single_coverage
import os, sys

def benchmark_template(command, *args, **kwargs):
    @benchmark
    def template():
        return command(*args, **kwargs)
    return template()

def benchmark_module(moduleName):
    return benchmark_template(run_module, moduleName)

def benchmark_file(filename):
    run_file = lambda filename : os.system('python %s' % filename)
    return benchmark_template(run_file, filename)

def benchmark_coverage(fullpath, package=''):
    return benchmark_template(get_single_coverage, fullpath, package)

@tool
def run_benchmarks(names, output=None):
    # build modulenames and filenames
    fileNames   = get_file_names()
    moduleNames = get_module_names()

    benchDict = {}

    for name in names:
        if name in moduleNames:
            benchDict[name] = benchmark_module(os.getcwd() + '/' + name)[1]
        elif name in fileNames:
            # send the full filepath to benchmark_file()
            benchDict[name] = benchmark_file(fileNames[name])[1]
        else:
            print('%s is neither a valid modulename, nor a valid filename' % name)

    if output != None:
        write_formatted_table(output, benchDict.items())
    return benchDict

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    run_benchmarks(*args, **kwargs)

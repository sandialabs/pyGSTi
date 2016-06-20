from __future__ import print_function
from runModule  import run_module
from benchmarks import benchmark
from helpers    import *
import os, sys

def benchmark_template(fullpath, command):
    directory, name = fullpath.rsplit('/', 1)
    @benchmark
    def template():
        command(fullpath)
    _, time = template()
    return time

def benchmark_module(moduleName):
    return benchmark_template(moduleName, run_module)

def benchmark_file(filename):
    run_file = lambda filename : os.system('python %s' % filename)
    return benchmark_template(filename, run_file)

@tool
def run_benchmarks(names, output=None):
    # build modulenames and filenames
    fileNames   = get_file_names()
    moduleNames = get_module_names()

    benchDict = {}

    for name in names:
        if name in moduleNames:
            benchDict[name] = benchmark_module(os.getcwd() + '/' + name)
        elif name in fileNames:
            # send the full filepath to benchmark_file()
            benchDict[name] = benchmark_file(fileNames[name])
        else:
            print('%s is neither a valid modulename, nor a valid filename' % name)

    if output != None:
        write_formatted_table(output, benchDict.items())
    return benchDict

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    run_benchmarks(*args, **kwargs)

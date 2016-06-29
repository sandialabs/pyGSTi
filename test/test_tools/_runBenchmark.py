#!/usr/bin/python

from __future__    import print_function
from runPackage    import run_package
from benchmarks    import benchmark
from helpers       import *
from _getCoverage  import get_single_coverage

import subprocess
import os, sys

'''
A set of functions/tools for timing tests

usage: $ python runBenchmark.py testPrinter.py io --output=bench.out
  (benchmark the testPrinter.py test and io package, writing the result to bench.out)
'''

def benchmark_template(command, *args, **kwargs):
    # returns commandresult, time(seconds)
    @benchmark
    def template():
        return command(*args, **kwargs)
    result, time = template()
    time = '%s seconds | %s hours' % (time, (time/3600))
    return result, time

# A default function for benchmarking a set of tests
def benchmark_package(packageName):
    return benchmark_template(run_package, packageName)

# Default function for benchmarking a single file
def benchmark_file(filename):
    run_file = lambda filename : subprocess.call(['python', filename])
    return benchmark_template(run_file, filename)

# returns a coverage percent and a time - works for packages and individual files
def benchmark_coverage(fullpath, package=''):
    return benchmark_template(get_single_coverage, fullpath, package)

# the tool decorator makes a function act as if run from the test directory
@tool
def run_benchmarks(names, output=None):
    fileNames    = get_file_names()
    packageNames = get_package_names()

    benchDict = {}

    for name in names:
        if name in packageNames:
            benchDict[name] = benchmark_package(os.getcwd() + '/' + name)[1]
        elif name in fileNames:
            # send the full filepath to benchmark_file()
            benchDict[name] = benchmark_file(fileNames[name])[1]
        else:
            print('%s is neither a valid package, nor a valid filename' % name)

    if output != None:
        write_formatted_table(output, list(benchDict.items()))
    return benchDict

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    run_benchmarks(*args, **kwargs)

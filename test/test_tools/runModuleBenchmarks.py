from __future__ import print_function
from runModule  import run_module
from benchmarks import benchmark
from tool       import tool
import os, sys

@tool
@benchmark('modulebenchmark.txt')
def benchmark_module(moduleName):
    run_module(moduleName)

if __name__ == "__main__":
    _, directories, _ = os.walk(os.getcwd()).next()

    for directory in directories:
        print('Benchmarking the %s module' % directory)
        benchmark_module(directory)

from __future__ import print_function
from runModule  import run_module
from benchmarks import benchmark
from tool       import tool
import os, sys

@benchmark('modulebenchmark.txt')
def benchmark_module(moduleName):
    run_module(moduleName)

@tool
def bench_modules():
    _, directories, _ = os.walk(os.getcwd()).next()

    for directory in directories:
        print('Benchmarking the %s module' % directory)
        benchmark_module(directory)

if __name__ == "__main__":
    bench_modules()

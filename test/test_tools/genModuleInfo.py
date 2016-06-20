from __future__   import print_function
from helpers      import *
from runBenchmark import benchmark_coverage
import os, sys

exclude = ['benchmarks', 'output', 'cmp_chk_files', 'temp_test_files', 'Tutorials', 'test_tools']

@tool
def genModuleInfo(extra_exclude):
    excludes = exclude + extra_exclude
    moduleDict = {}
    # moduleNames = [name for name in get_module_names() if name not in excludes]
    moduleNames = ['objects', 'tools']
    for name in moduleNames:
        moduleDict[name] = benchmark_coverage(name, package=name)
    moduleDict = { key : ('%s%% coverage | %s seconds' % moduleDict[key]) for key in moduleDict }

    write_formatted_table('moduleinfo.out', moduleDict.items())

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    genModuleInfo(*args, **kwargs)

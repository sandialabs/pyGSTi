from __future__   import print_function
from helpers      import *
from getCoverage  import get_coverage
from runBenchmark import run_benchmarks
import os, sys

@tool
def gen_info(names, output=None, package=''):
    fileNames    = get_file_names()
    moduleNames  = get_module_names()

    coverageDict = get_coverage(names, package=package)
    benchDict    = run_benchmarks(names)

    combinedDict = {}

    for key in coverageDict:
        if key in benchDict:
            combinedDict[key] = '%s%% coverage | %s seconds' % (coverageDict[key], benchDict[key])
        else:
            raise ValueError('coverageDict and benchDict should have the same set of keys')

    if output != None:
        write_formatted_table(output, combinedDict.items())

    return combinedDict

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    gen_info(*args, **kwargs)

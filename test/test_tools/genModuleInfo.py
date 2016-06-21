from __future__   import print_function
from helpers      import *
from runBenchmark import *
import os, sys

exclude = ['benchmarks', 'output', 'cmp_chk_files', 'temp_test_files', 'Tutorials', 'test_tools']

@tool
def genModuleInfo(extra_exclude, infoType=''):
    excludes = exclude + extra_exclude
    moduleDict = {}
    moduleNames = [name for name in get_module_names() if name not in excludes]
    for name in moduleNames:
        if infoType   == 'coverage':
            moduleDict[name] = get_single_coverage(name, package=name)
        elif infoType == 'benchmark':
            moduleDict[name] = benchmark_module(name)
        else:
            moduleDict[name] = benchmark_coverage(name, package=name)

    if infoType == 'coverage':
        moduleDict = { key : ('%s%% coverage' % moduleDict[key]) for key in moduleDict }
    elif infoType == 'benchmark':
        moduleDict = { key : ('%s seconds'    % moduleDict[key]) for key in moduleDict }
    else:
        moduleDict = { key : ('%s%% coverage | %s seconds' % moduleDict[key]) for key in moduleDict }

    write_formatted_table('output/module%sinfo.out' % \
                           (('_%s_' % infoType) if infoType != '' else ''),
                          moduleDict.items())

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    genModuleInfo(*args, **kwargs)

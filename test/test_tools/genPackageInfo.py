#!/usr/bin/python
from __future__   import print_function
from .helpers      import *
from .runBenchmark import *
from .runChanged   import get_changed_test_packages
import os, sys

exclude = ['benchmarks', 'output', 'cmp_chk_files', 'temp_test_files', 'Tutorials', 'test_tools']

'''
A tool for generating info about each of the packages in the test directory

usage: $ python genPackageInfo.py io
  (Generate coverage and benchmark info for every package except io(and the excluded packages))

optionally, the argument --infoType='' can be sent as either:
  'coverage'  - generate only coverage info
  'benchmark' - generate only benchmark info
  generally, --infoType=benchmark should generate more accurate benchmarks, as there is no overhead from the coverage tests

'''

#the tool decorator makes the function act as if it were run from the test directory
@tool
def gen_package_info(extra_exclude, infoType='', changedOnly=''):
    # build the full list of excluded packages
    excludes = exclude + extra_exclude
    packageDict = {}
    if changedOnly == 'True':
        packageNames = [name for name in get_changed_test_packages() if name not in excludes]
    else:
        packageNames = [name for name in get_package_names() if name not in excludes]

    for name in packageNames:
        if infoType   == 'coverage':
            packageDict[name] = get_single_coverage(name, package=name)
        elif infoType == 'benchmark':
            packageDict[name] = benchmark_package(name)
        else:
            packageDict[name] = benchmark_coverage(name, package=name)

    if infoType == 'coverage':
        packageDict = { key : ('%s%% coverage' % packageDict[key]) for key in packageDict }
    elif infoType == 'benchmark':
        packageDict = { key : ('%s'            % packageDict[key]) for key in packageDict }
    else:
        packageDict = { key : ('%s%% coverage | %s' % packageDict[key]) for key in packageDict }

    # ALWAYS write a table to file
    write_formatted_table('output/package%sinfo.out' % \
                           (('_%s_' % infoType) if infoType != '' else ''),
                          list(packageDict.items()))

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    gen_package_info(*args, **kwargs)

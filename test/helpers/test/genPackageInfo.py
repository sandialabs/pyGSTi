from __future__     import print_function
from .helpers       import *
from ._runBenchmark import *
import os, sys

'''
A tool for generating info about each of the packages in the test directory

usage: $ python genPackageInfo.py io
  (Generate coverage and benchmark info for every package except io(and the excluded packages))

optionally, the argument --infoType='' can be sent as either:
  'coverage'  - generate only coverage info
  'benchmark' - generate only benchmark info
  generally, --infoType=benchmark should generate more accurate benchmarks, as there is no overhead from the coverage tests

'''

def gen_package_info(packageNames, infoType='', changedOnly=''):
    packageDict = {}

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


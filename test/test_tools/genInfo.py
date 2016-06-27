#!/usr/bin/python
from __future__   import print_function
from helpers      import *
from runBenchmark import benchmark_coverage
import os, sys

'''
A tool for generating information about a combination of packages and individual files

usage: $ python genInfo.py testHessian.py tools --output=info.out --package=pygsti
  (generate coverage and time info for testHessian.py and the tools package, and write it to info.out)

( A possible downside to this is that coverage is limited to a single package.)
'''

# the tool decorator makes genInfo act as if run from the test directory
@tool
def gen_info(names, output=None, package=''):
    fileNames    = get_file_names()
    packageNames = get_package_names()

    infoDict = {}

    for name in names:
        if name in packageNames:
            # get coverage of the package, and the time the test took
            infoDict[name] = benchmark_coverage(os.getcwd() + '/' + name, package=package)
        elif name in fileNames:
            # send the full filepath to benchmark_coverage
            infoDict[name] = benchmark_coverage(fileNames[name], package=package)
        else:
            print('%s is neither a valid package, nor a valid filename' % name)

    # some nicer formatting before the table is written
    infoDict = { key : ('%s%% coverage | %s' % infoDict[key]) for key in infoDict }

    if output != None:
        write_formatted_table(output, infoDict.items())

    return infoDict

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    gen_info(*args, **kwargs)

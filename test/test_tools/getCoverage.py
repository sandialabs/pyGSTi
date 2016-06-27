#!/usr/bin/python
from __future__ import print_function
from helpers    import *
import os, sys
import subprocess

'''
A tool for generating a coverage file and then reading the result - optionally returning the result or outputting to a file

usage: $ python getCoverage.py testAlgorithms.py construction --output=coverage.out --package=pygsti
  (generate coverage info for testAlgorithms.py and the construction package, sending the result to coverage.out)

(This is mostly collection of helper functions for the genInfo/genPackageInfo scripts)
'''

temp_coverage_file_name = '_coverage.out'

def _read_coverage(command, filename):
    os.system(command) # generate the coverage file
    percent = 0
    with open(filename, 'r') as coveragefile:
        totals = [line for line in coveragefile.read().splitlines() if 'TOTAL' in line]
        if len(totals) > 0:
            total   = totals[0] # use the first line that contains 'TOTAL'
            percent = total.rsplit('%', 1)[0][-3:] # get the three characters to the left of the rightmost % character
            try:
                percent = int(percent)
            except:
                percent = 0 # if the percent is invalid/cannot be parsed
    subprocess.call(['mv', '.coverage', (filename.split('.', 1)[0] + '.coverage')])
    return percent

# assumes name is either a package or an absolute path to a file
# the tool decorator makes the function act as if it were in the test directory
@tool
def get_single_coverage(name, package=''):
    #nosetests -v --with-coverage --cover-package=pygsti --cover-erase */test*.py > coverage_tests_serial.out 2>&1
    # build the above command with some string formatting
    package   = 'pygsti' + ('.%s' % package if package != '' else '')
    commands  = 'nosetests -v --with-coverage --cover-package=%s --cover-erase ' % package
    shortname = name
    if name.count('/') > 0:
        shortname = name.rsplit('/', 1)[1] # get the filename without full path
    if shortname.count('.') > 0:
        shortname = shortname.split('.', 1)[0] # remove file extensions
    filename = 'output/' + shortname + temp_coverage_file_name
    tempfile = ' 2>&1 | tee %s' % filename

    return _read_coverage(commands + name + tempfile, filename)

# the tool decorator makes the function act as if it were in the test directory
@tool
def get_coverage(names, output=None, package=''):
    fileNames    = get_file_names()
    packageNames = get_package_names()

    coverageDict = {}

    for name in names:
        if name in packageNames:
            coverageDict[name] = get_single_coverage(name, package)
        elif name in fileNames:
            # give the full pathname to read_coverage if name is a filename
            coverageDict[name] = get_single_coverage(fileNames[name], package)
        else:
            print('%s is neither a valid package, nor a valid filename' % name)

    if output != None:
        write_formatted_table(output, coverageDict.items())

    return coverageDict

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    get_coverage(*args, **kwargs)

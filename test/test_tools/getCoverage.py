from __future__ import print_function
from helpers    import *
import os, sys

temp_coverage_file_name = 'temp_coverage_file.out'


def _read_coverage(command):
    os.system(command) # generate the coverage file
    with open(temp_coverage_file_name, 'r') as coveragefile:
        total = [line for line in coveragefile.read().splitlines() if line != '']
        total = total[-4]
        percent = total[-4:-1]
    os.remove(temp_coverage_file_name)
    return int(percent)

# assumes name is either a module or an absolute path to a file
@tool
def get_single_coverage(name, package):
    #nosetests -v --with-coverage --cover-package=pygsti --cover-erase */test*.py > coverage_tests_serial.out 2>&1
    # build the above command with some string formatting
    package  = 'pygsti' + ('.%s' % package if package != '' else '')
    commands = 'nosetests -v --with-coverage --cover-package=%s --cover-erase ' % package
    tempfile = ' > %s 2>&1' % temp_coverage_file_name

    return _read_coverage(commands + name + tempfile)

@tool
def get_coverage(names, output=None, package=''):
    fileNames   = get_file_names()
    moduleNames = get_module_names()

    coverageDict = {}

    for name in names:
        if name in moduleNames:
            coverageDict[name] = get_single_coverage(name, package)
        elif name in fileNames:
            # give the full pathname to read_coverage if name is a filename
            coverageDict[name] = get_single_coverage(fileNames[name])
        else:
            print('%s is neither a valid modulename, nor a valid filename' % name)

    if output != None:
        write_formatted_table(output, coverageDict.items())

    return coverageDict

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    get_coverage(*args, **kwargs)

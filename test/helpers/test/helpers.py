from __future__ import print_function, absolute_import
import os

# return a list of the immediate subdirectories
def get_package_names():
    _, packageNames, _ = next(os.walk(os.getcwd()))
    return packageNames

# return a dict of filenames that correspond to full paths
def get_file_names():
    fileNames = {}
    for subdir, _, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith('.py') and filename.startswith('test'):
                fileNames[filename] = subdir + os.sep + filename
    return fileNames

# for the tools like runBenchmarks or genModuleInfoa
def write_formatted_table(filename, items):
    with open(filename, 'w') as output:
        for a, b in items:
            row = '%s | %s\n' % (a.ljust(20), b)
            output.write(row)

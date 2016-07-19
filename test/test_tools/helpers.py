from __future__ import print_function, absolute_import
import os

# creates a message ~like so:
#
##########################################################################
#
#                          message
#
##########################################################################
show_message = lambda message : print('\n\n%s\n\n%s%s\n\n%s\n\n' % ('#' * 80, ' ' * 30, message, '#' * 80))

# for functions in the test_tools directory that act like they've been run in the test directory
def tool(function):
    def wrapper(*args, **kwargs):
        owd = os.getcwd() # Handle moving between directories
        if os.getcwd().rsplit('/', 1)[1] == 'test_tools':
            os.chdir('..')
        result = function(*args, **kwargs)
        os.chdir(owd)
        return result
    return wrapper

# return args and kwargs from sys.argv
def get_args(rawArgs):
    args      = [[arg for arg in rawArgs[1:] if not arg.startswith('--')]] # create args
    optionals = [arg for arg in rawArgs[1:] if arg.startswith('--')]
    kwargs    = {}
    # create kwargs
    for optional in optionals:
        if optional.count('=') > 0:
            kv = optional[2:].split('=') # remove prepending '--' and seperate into key : value
            kwargs[kv[0]] = kv[1]
        else:
            k = optional[2:] # only remove prepending '--'
            kwargs[k] = k

    return args, kwargs

# return a list of the immediate subdirectories
def get_package_names():
    _, packageNames, _ = next(os.walk(os.getcwd()))
    return packageNames

# return a dict of filenames that correspond to full paths
def get_file_names():
    fileNames = {}
    for subdir, dirs, files in os.walk(os.getcwd()):
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

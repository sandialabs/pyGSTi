from __future__ import print_function
from contextlib import contextmanager
import subprocess, os, sys

def get_file_directory():
    return os.path.dirname(os.path.abspath(__file__))

@contextmanager
def directory(directoryName):
    oldwd = os.getcwd()
    os.chdir(directoryName)
    yield
    os.chdir(oldwd)

@contextmanager
def this_directory():
    with directory(get_file_directory()):
        yield

# Works from anywhere
def get_branchname():
    try:
        with this_directory():
            branchname = subprocess.check_output(['bash', 'get_branch'])
    except subprocess.CalledProcessError:
        branchname = 'unnamed_branch'
    branchname = os.path.basename(branchname)
    branchname = branchname.replace('\n', '')
    return branchname

# Used by git hooks, (called from top level pyGSTi directory)
def run_pylint(commands):
    with directory('test/'):
        result = subprocess.call(commands)
    return result

# creates a message ~like so:
#
##########################################################################
#
#                          message
#
##########################################################################
show_message = lambda message : print('\n\n%s\n\n%s%s\n\n%s\n\n' % ('#' * 80, ' ' * 30, message, '#' * 80))

# Decorator for making functions run in the tools directory
def tool(function):
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
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


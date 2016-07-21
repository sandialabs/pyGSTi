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

def get_branchname():
    branchname = 'unspecified'
    output     = subprocess.check_output(['git', 'branch'])
    branches   = output.decode('utf-8').splitlines()
    for branch in branches:
        if '*' in branch:
            branchname = branch.replace('*', '').replace(' ', '')
            break
    return branchname
'''
def get_branchname():
    try:
        with this_directory():
            branchname = subprocess.check_output(['bash', 'get_branch'])
    except subprocess.CalledProcessError:
        branchname = 'unnamed_branch'
    branchname = os.path.basename(branchname)
    branchname = branchname.replace('\n', '')
    return branchname
'''
def get_author(SHA=None):
    if SHA is None:
        output = subprocess.check_output(['git', 'log', 'HEAD', '-1'])
    else:
        output = subprocess.check_output(['git', 'show', SHA])
    authorline = output.decode('utf-8').splitlines()[1]
    _, author, email = authorline.split()
    return author, email

'''
lsaldyt@s1000706:~/pyGSTi/test$ git log HEAD -1
commit 51930d0a7d6be7c60ab7070479e69ed619ac73ab
Author: LSaldyt <lucassaldyt@yahoo.com>
Date:   Thu Jul 21 11:30:24 2016 -0600

    Change (Inactive) after_success hook to switch to develop branch
'''

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
    args      = [arg for arg in rawArgs[1:] if not arg.startswith('--')] # create args
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

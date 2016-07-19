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
    with directory('test/pylint/'):
        result = subprocess.call(commands)
    return result


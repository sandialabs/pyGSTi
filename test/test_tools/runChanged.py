from __future__ import print_function
from runPackage import run_package
from helpers    import *
import subprocess
import os
import sys

endings      = ['py', 'ipynb']
directory    = 'test/'

# The directories to be removed from the path of changed files
cutoffDirs   = ['packages/pygsti/', 'ipython_notebooks/']
exclude      = ['test']

# the tool decorator makes a function act as if run from the test directory
@tool
def get_changed_files():
    oldwd = os.getcwd()
    os.chdir('../')
    output = subprocess.check_output(['git', 'diff', '--name-only'])

    changedFilePaths = []

    for line in output.splitlines():
        for cutoffDir in cutoffDirs:
            if cutoffDir in line:
               line = line.replace(cutoffDir, '')
        if line.count('.') == 1:
            _, ending = line.split('.')
            if ending in endings:
	            changedFilePaths.append(line)
    os.chdir(oldwd)
    return changedFilePaths


def get_changed_packages():
    # Get the packageNames that have changed
    changedPackages = set()

    for name in get_changed_files():
        packageName, name = name.split('/')
        if packageName not in exclude:
            changedPackages.add(packageName)

    return changedPackages

@tool
def run_changed_packages():
    print('Trying to run changed packages...')
    changedPackages = get_changed_packages()

    if len(changedPackages) == 0:
        print('No packages have changed since the last commit')

    for packageName in changedPackages:
         run_package(packageName)


if __name__ == "__main__":
    run_changed_packages()

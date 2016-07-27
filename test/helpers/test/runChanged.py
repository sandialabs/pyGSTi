from __future__         import print_function
from .runPackage        import run_package
from .helpers           import *
from ..automation_tools import directory, get_output
import subprocess
import os
import sys

def get_changed_files(cutoffDirs=['packages/pygsti/', 'jupyter_notebooks/'],
                      exclude=['test'], preCommand='../', endings=['py', 'ipynb', '']):
    with directory(preCommand):
        output = get_output(['git', 'diff', '--name-only'])

    changedFilePaths = []
    for line in output.splitlines():
        if line.split('/', 1)[0] not in exclude:
            for cutoffDir in cutoffDirs:
                if cutoffDir in line:
                    line = line.replace(cutoffDir, '')
            if line.count('.') == 1:
                _, ending = line.split('.')
                if ending in endings:
                    changedFilePaths.append(line)
            elif '' in endings:
                changedFilePaths.append(line)
    return changedFilePaths

def get_changed_test_packages():
    return get_changed_packages(cutoffDirs=['test/', 'hooks/git/'],
                                exclude=['doc', 'jupyter_notebooks', 'test'],
                                endings=['py', ''])

def get_changed_packages(cutoffDirs=[], exclude=[], preCommand='../', endings=['py', '']):
    # Get the packageNames that have changed
    changedPackages = set()

    for name in get_changed_files(cutoffDirs, exclude, preCommand, endings):
        packageNames = name.rsplit('/')[:-1]
        for packageName in packageNames:
            if packageName not in exclude:
                changedPackages.add(packageName)

    return changedPackages

def run_changed_packages(cutoffDirs=[], exclude=[], preCommand='../', endings=[], lastFailed=''):
    print('Trying to run changed packages...')
    changedPackages = get_changed_packages(cutoffDirs, exclude, preCommand, endings)

    if len(changedPackages) == 0:
        print('No packages have changed since the last commit')

    for packageName in changedPackages:
        run_package(packageName, lastFailed=lastFailed)

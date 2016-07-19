from __future__ import print_function
import subprocess, os, sys

def get_branchname():
    try:
        branchname = subprocess.check_output(['bash', '.git/hooks/.get_branch'])
    except subprocess.CalledProcessError:
        branchname = 'unnamed_branch'
    branchname = os.path.basename(branchname)
    branchname = branchname.replace('\n', '')
    return branchname

def run_pylint(commands):
    old_wd = os.getcwd()
    os.chdir('test/pylint/')
    result = subprocess.call(commands)
    os.chdir(old_wd)
    return result


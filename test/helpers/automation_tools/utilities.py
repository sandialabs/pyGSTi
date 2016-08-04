from __future__ import print_function
from contextlib import contextmanager
import subprocess, os, sys

def get_files(directory):
    print(directory)
    for _, _, files in os.walk(directory):
        return files # exit early
    return []

def get_file_directory():
    return os.path.dirname(os.path.abspath(__file__))

@contextmanager
def directory(directoryName):
    oldwd = os.getcwd()
    os.chdir(directoryName)
    try:
        yield
    finally:
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

def get_author(SHA=None):
    if SHA is None:
        output = subprocess.check_output(['git', 'log', 'HEAD', '-1'])
    else:
        output = subprocess.check_output(['git', 'show', SHA])
    authorline = output.decode('utf-8').splitlines()[1]
    _, author, email = authorline.split()
    return author, email

# Used by git hooks, (called from top level pyGSTi directory)
def run_pylint(commands):
    with directory('test/'):
        result = subprocess.call(commands)
    return result

def get_output(commands):
    try:
        output = subprocess.check_output(commands, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e)
        output = e.output
    return output.decode('utf-8')

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

def get_changed_files():
    output = get_output(['git', 'diff', '--name-only'])
    return output.splitlines()

def get_changed_core_files(core='pygsti'):
    return (filename.split(core, 1)[1] for filename in get_changed_files() if core in filename)

def get_changed_packages():
    return (corefile.split('/')[1] for corefile in get_changed_core_files())



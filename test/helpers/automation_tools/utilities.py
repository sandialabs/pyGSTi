import os
import subprocess
from contextlib import contextmanager


# Get decoded output of a command, even if it fails!
def get_output(commands):
    try:
        # We'll want error output, too
        output = subprocess.check_output(commands, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e)
        output = e.output
    return output.decode('utf-8')

# Immediate subdirectories
def get_packages(directory):
    for _, packages, _ in os.walk(directory):
        return packages # exit early
    return []

# Immediate files
def get_files(directory):
    print(directory)
    for _, _, files in os.walk(directory):
        return files # exit early
    return []

# Use a directory for a task
@contextmanager
def directory(directoryName):
    oldwd = os.getcwd()
    os.chdir(directoryName)
    try:
        yield
    finally:
        os.chdir(oldwd)

# 'Parse' git branch output
def get_branchname():
    branches   = []
    output     = get_output(['git', 'branch'])
    branches   = output.splitlines()
    for branch in branches:
        if '*' in branch: # current branch is starred
            return branch.replace('*', '').replace(' ', '') # Get rid of * and whitespace

# Get all LOCAL branches
def get_branches():
    branches   = []
    output     = get_output(['git', 'branch'])
    # Remove whitespace and *
    branches   = output.replace('*', '').replace(' ', '').splitlines()
    return branches

# Finds the author of a commit
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

# return a dict of filenames that correspond to full paths
def get_file_names():
    fileNames = {}
    for subdir, _, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith('.py') and filename.startswith('test'):
                fileNames[filename] = subdir + os.sep + filename
    return fileNames

# Wrapper for git diff
def get_changed_files():
    output = get_output(['git', 'diff', '--name-only'])
    return output.splitlines()

# Changed files in /pygsti (not tests/repotools)
def get_changed_core_files(core='pygsti'):
    return (filename.split(core, 1)[1] for filename in get_changed_files() if core in filename)

# Immediate packages under pygsti that have changed (i.e. tools, drivers..)
def get_changed_packages():
    return (corefile.split('/')[1] for corefile in get_changed_core_files())

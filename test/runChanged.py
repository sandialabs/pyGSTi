from __future__ import print_function
import os
import sys

endings      = ['py', 'ipynb']
tempDiffFile = 'temp_test_files/names.txt'
directory    = 'test/'
preCommand   = 'cd ~/pyGSTi/'

# The directories to be removed from the path of changed files
cutoffDirs   = ['packages/pygsti/', 'ipython_notebooks/']

def get_changed_files():
    # Run git diff against the repository, and save all the changed names to a text file
    os.system('%s; git diff --name-only > %s/%s' % (preCommand, os.getcwd(), tempDiffFile))

    changedFilePaths = []

    with open(tempDiffFile, 'r') as changed:
	for line in changed.read().splitlines(): # discard newline characters
            for cutoffDir in cutoffDirs:
                if cutoffDir in line:
                   line = line.replace(cutoffDir, '')
            if line.count('.') == 1:
	        _, ending = line.split('.')
	        if ending in endings:
		    changedFilePaths.append(line)

    # Remove the temporary file that was created
    os.remove(tempDiffFile)
    return changedFilePaths


def get_changed_packages():
    # Get the packageNames that have changed
    changedPackages = set()

    for name in get_changed_files():
	packageName, name = name.split('/')
	changedPackages.add(packageName)

    return changedPackages

def run_changed_packages():
    for packageName in get_changed_packages():
	for subdir, dirs, files in os.walk(os.getcwd() + '/' + packageName):
	    for filename in files:
		filepath = subdir + os.sep + filename
		if filepath.endswith('.py'):
		    os.system('python %s' % filepath)


if __name__ == "__main__":
    run_changed_packages()

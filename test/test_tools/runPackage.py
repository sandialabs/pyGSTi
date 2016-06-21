from __future__ import print_function
from helpers    import tool
import os, sys
import subprocess

#tool makes the function act as if run from the test directory
@tool
def run_package(packageName, precommand='python', postcommand=''):
    os.chdir(packageName)
    for subdir, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith('.py') and filename.startswith('test'):
                print('Running %s' % filename)
                filepath = subdir + os.sep + filename
                subprocess.call([precommand, filepath, postcommand])
    os.chdir('..')

if __name__ == "__main__":
   for name in sys.argv[1:]:
       run_package(name)

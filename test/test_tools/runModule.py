from __future__ import print_function
from helpers    import tool
import os, sys
import subprocess

#@tool
def run_module(moduleName, precommand='python', postcommand=''):
    os.chdir(moduleName)
    for subdir, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith('.py') and filename.startswith('test'):
                print('Running %s' % filename)
                filepath = subdir + os.sep + filename
                subprocess.call([precommand, filepath, postcommand])
    os.chdir('..')

if __name__ == "__main__":
   for name in sys.argv[1:]:
       run_module(name)

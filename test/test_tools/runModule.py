from __future__ import print_function
import os, sys

def run_module(moduleName, precommand='python', postcommand=''):
    os.chdir(moduleName)
    for subdir, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith('.py') and filename.startswith('test'):
                print('Running %s' % filename)
                filepath = subdir + os.sep + filename
                os.system('%s %s %s' % (precommand, filepath, postcommand))
    os.chdir('..')

if __name__ == "__main__":
   for name in sys.argv[1:]:
       run_module(name)

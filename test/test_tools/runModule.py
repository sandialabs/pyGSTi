from __future__ import print_function
from tool       import tool
import os, sys

@tool
def run_module(moduleName, command='python', extension=''):
    owd = os.getcwd() # Handle moving between directories
    os.chdir(os.getcwd() +  '/' + moduleName)
    for subdir, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith('.py'):
                print('Running %s' % filename)
                filepath = subdir + os.sep + filename
                os.system('%s %s %s' % (command, filepath, extension))
    os.chdir(owd)

if __name__ == "__main__":
   for name in sys.argv[1:]:
       run_module(name)

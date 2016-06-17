from __future__ import print_function
import os, sys


def run_module(module):
    for subdir, dirs, files in os.walk(os.getcwd() + '/' + module):
        for filename in files:
            print('Running %s' % filename)
            filepath = subdir + os.sep + filename
            if filepath.endswith('.py'):
                os.system('python %s' % filepath)


if __name__ == "__main__":
   for name in sys.argv[1:]:
       run_module(name) 

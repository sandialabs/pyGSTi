#!/usr/bin/env python3
import os.path, importlib, pkgutil, sys
import pygsti

from inspect import *

def check_function(f):
    pass

def check_class(c):
    pass

def check_method(m):
    pass

def check_module(module):
    for member in getmembers(module):
        if isfunction(member):
            pass
        if ismethod(member):
            pass
        if isclass(member):
            pass
        if ismodule(member):
            check_module(member)
        print(member)

def main(args):
    check_module(pygsti)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

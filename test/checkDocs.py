#!/usr/bin/env python3
import os.path, importlib, pkgutil, sys
import pygsti

from pprint import pprint

from inspect import *
import inspect

missing = []

def check_args_in_docstring(item):
    args, _, kwargs, _ = getargspec(item)
    if kwargs is None:
        kwargs = []
    for arg in args + [k for k, v in kwargs]:
        if item.__doc__ is None or arg not in item.__doc__:
            missing.append(item)

def check_function(f):
    print('Checking function')
    check_args_in_docstring(f)

def check_class(c):
    check(c)

def check_method(m):
    check_args_in_docstring(m)

def check(module):
    for member in getmembers(module):
        if 'pygsti' in str(member[1]):
            name, member = member
            if isfunction(member):
                check_function(member)
            if ismethod(member):
                check_method(member)
            if isclass(member):
                check_class(member)
            if ismodule(member):
                check(member)

def main(args):
    check(pygsti)
    pprint(missing)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

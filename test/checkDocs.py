#!/usr/bin/env python3
import sys
from collections import defaultdict
from inspect import *

import pygsti


def fullname(o):
    try:
        if o.__module__ is None:
            mod = ''
        else:
            mod = o.__module__
        return mod + "." + o.__name__ #o.__class__.__name__
    except AttributeError:
        return ''

def parse_doc_arg_set(doc):
    args = set()
    if 'Parameters' not in doc or \
            'Returns' not in doc:
        raise ValueError('Docstring malformed')
    desc, rest      = doc.split('Parameters')
    params, returns = rest.split('Returns')

    for line in params.split('\n'):
        if ':' in line:
            arg = line.split(':')[0].strip()
            if ',' in arg:
                inner_args = set(map(lambda s : s.strip(), arg.split(',')))
            else:
                inner_args = {arg}
            for arg in inner_args:
                if ' ' not in arg and \
                   '`' not in arg:
                    args.add(arg)
    return args

found = defaultdict(set)
data  = defaultdict(dict)

def mark(item, k):
    found[k].add(fullname(item))

def check_args_in_docstring(item):
    args, _, kwargs, _ = getargspec(item)
    if kwargs is None or kwargs in ['kwargs', 'kwds']:
        kwargs = []
    argset = set(args)
    if 'self' in argset:
        argset.remove('self')
    for arg in args + [k for k, v in kwargs]:
        if item.__doc__ is None:
            mark(item, 'missing')
        else:
            if arg not in item.__doc__:
                mark(item, 'incomplete')
            try:
                docargs   = parse_doc_arg_set(item.__doc__)
                extraargs = set.difference(argset, docargs)
                if len(extraargs) > 0:
                    mark(item, 'incomplete')
                data['incomplete'][fullname(item)] = extraargs
            except ValueError:
                mark(item, 'malformed')

def check_function(f):
    #print('Checking function {}'.format(f.__name__))
    check_args_in_docstring(f)

def check_class(c):
    check(c)

def check_method(m):
    check_args_in_docstring(m)

def check(module):
    for member in getmembers(module):
        name, member = member
        if 'pygsti' in fullname(member):
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
    for k, v in found.items():
        print('{}:'.format(k))
        print('    ' + '\n    '.join(v))
    #pprint(data)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

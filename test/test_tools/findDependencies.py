#!/usr/bin/python
from __future__ import print_function
from helpers    import tool
import os
import sys

'''
A small tool for finding module dependencies within the pygsti tests.

This might be useful for binning each of the tests so that each package is covered independently
'''

# For maintaining a dictionary of dependencies
def increase_count(dictionary, item):
    if item in dictionary:
        dictionary[item] = dictionary[item] + 1
    else:
        dictionary[item] = 1

# Complicated rules for building the imports of a module.
def build_qualified_imports(line, qualifiedImports):
    # seperate by spaces and remove commas and empty strings
    tokenized = [t for t in line.split(' ') if (t != '' and t != ',')]
    # first handle a normal qualified import
    if tokenized[0] == 'import' and \
       len(tokenized) > 2       and \
       tokenized[2] == 'as':
	    qualifiedImports.append((tokenized[3], tokenized[1]))
    # then from _ import _, _, etc
    elif tokenized[0] == 'from':
	package = tokenized[1]
	if len(tokenized) < 5:
	    qualifiedImports.append((tokenized[3], package + '.' + tokenized[3]))
    # then from _ import _ as
	elif len(tokenized) > 4:
	    if tokenized[4] == 'as':
                qualifiedImports.append((tokenized[5], package + '.' + tokenized[3]))
            else:
                for token in tokenized[2:]:
                    token = token.replace(',', '')
                    qualifiedImports.append((token, package + '.' + token))

# @tool tells the function to run as if it were in the test directory
@tool
def find_dependencies(filename, packageName='pygsti'):
    dependencies = dict()
    # qualified imports contains tuples of the form (qualifier, actual)
    qualifiedImports = []
    with open(filename, 'r') as source:
        for line in source.read().splitlines():
            # Replace qualified imports in the line with their full names
            for qImport in qualifiedImports:
                if qImport[0] in line:
                    line = line.replace(qImport[0], qImport[1])

            # Add qualified imports to the list
            if 'import' in line:
                build_qualified_imports(line, qualifiedImports)

            elif packageName in line:
                 occurances = line.split(packageName)
                 for occurance in occurances[1:]:
                     module = ''
                     for char in occurance[1:]:
                         if char not in '. ()':
                             module += char
                         else:
                             break
                     increase_count(dependencies, module)
    return dependencies


if __name__ == "__main__":
    for arg in sys.argv[1:]: # Here, the arguments are filenames
        print('Dependencies for %s:' % arg)
        deps = find_dependencies(arg)
        print(deps)
        if len(deps) > 0:
            print('Most Imported: %s' % max(deps, key=deps.get))

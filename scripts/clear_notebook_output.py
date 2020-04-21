#! /usr/bin/env python
""" Simple script to strip a notebook of all cell output.

Modified from https://gist.github.com/damianavila/5305869
"""

import sys
import argparse
import nbformat

_DEFAULT_VERSION = 4


def strip_output(nb):
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell.outputs = []
            cell.execution_count = None


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('paths', action='store', nargs='+', help="Path to the IPython notebook to be stripped")
parser.add_argument('-i', '--inplace', action='store_true',
                    help="Modify notebooks in-place instead of writing to stdout")
parser.add_argument('-v', '--version', action='store', type=int, default=_DEFAULT_VERSION,
                    help="IPython notebook version to parse as (default: {}).".format(_DEFAULT_VERSION))

args = parser.parse_args()

for path in args.paths:
    with open(path, 'r') as f:
        nb = nbformat.read(f, args.version)

    strip_output(nb)

    if args.inplace:
        with open(path, 'w') as f:
            nbformat.write(nb, f)
    else:
        nbformat.write(nb, sys.stdout)

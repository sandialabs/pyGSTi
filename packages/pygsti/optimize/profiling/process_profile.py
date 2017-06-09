#!/usr/bin/env python3
import sys

def main(args):
    assert len(args) == 1
    with open(args[0]) as infile:
        lines = [line for line in infile]
    lines = [line.split() for line in lines[4:] if 'built-in' not in line]
    lines = [line for line in lines if len(line) > 0]
    to_int = lambda elem : int(elem.split('/')[0])
    headers = lines[:1]
    lines = [items for items in lines[1:] if to_int(items[0]) > 1]
    lines = headers + lines
    for line in lines[:50]:
        print(' '.join(['{:<10}'.format(elem) for elem in line]))

if __name__ == '__main__':
    main(sys.argv[1:])

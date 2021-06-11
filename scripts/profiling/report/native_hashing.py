#!/usr/bin/env python3
import pygsti

from pygsti.tools import timed_block


def main():
    digest = pygsti.tools.smartcache.digest
    native = pygsti.tools.smartcache.native_hash

    iterations = 100

    s = 'adfasdkfj;asldfa;ldfja;sdfja;sdfjas;dlkfja;sdfafad1*(@#&$)(@#*&$)(#&@'
    l = [(s, s) for i in range(50)]
    t = tuple(l)
    d = dict(l)
    keys = [s, l, t, d]
    for key in keys:
        with timed_block('digest_' + str(type(key))):
            for i in range(iterations):
                digest(key)
        with timed_block('native_' + str(type(key))):
            for i in range(iterations):
                native(key)

if __name__ == '__main__':
    main()

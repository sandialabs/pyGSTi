#!/usr/bin/env python3
import sys

import pygsti


def f(x):
    if x == 1 or x == 0:
        return 1
    else:
        return f(x - 1) + f(x - 2)

def main(args):
    comm = pygsti.mpi4py_comm()
    numbers = list(range(100))
    results = pygsti.parallel_apply(f, numbers, comm)
    if comm.Get_rank() == 0:
        print(results)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

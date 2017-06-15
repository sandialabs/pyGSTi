#!/usr/bin/env python3
from pygsti.tools.mpitools import mpidot, distribute_for_dot
import numpy as np
                  
from random import randint

from timedblock import timed_block


def main():
    size = 100

    a = np.array([[randint(0, 99999) for i in range(size)] for i in range(size)])
    b = np.array([[randint(0, 99999) for i in range(size)] for i in range(size)])

    with timed_block('np.dot'):
        for i in range(1000):
            npC = np.dot(a, b)

    from mpi4py import MPI
    Comm = MPI.COMM_WORLD.Clone()

    with timed_block('mpidot'):
        for i in range(1000):
            p = distribute_for_dot(size, Comm)
            mpiC = mpidot(a, b, p, Comm)

main()

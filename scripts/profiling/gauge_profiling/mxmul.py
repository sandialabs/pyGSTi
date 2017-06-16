#!/usr/bin/env python3
from pygsti.tools.mpitools import mpidot, distribute_for_dot
from pygsti.tools.timed_block   import timed_block
import numpy as np
                  
from random      import randint
from collections import defaultdict

def main():
    size = 100
    iterations = 100

    a = np.array([[randint(0, 99999) for i in range(size)] for i in range(size)])
    b = np.array([[randint(0, 99999) for i in range(size)] for i in range(size)])

    timeDict = defaultdict(list)

    with timed_block('np.dot', timeDict):
        for i in range(iterations):
            npC = np.dot(a, b)

    from mpi4py import MPI
    Comm = MPI.COMM_WORLD.Clone()

    with timed_block('mpidot', timeDict):
        for i in range(iterations):
            p = distribute_for_dot(size, Comm)
            mpiC = mpidot(a, b, p, Comm)

    if Comm.Get_rank() == 0:
        nProcessors = Comm.Get_size()
        avg         = lambda l : sum(l) / len(l)
        npTime      = avg(timeDict['np.dot'])
        mpiTime     = avg(timeDict['mpidot'])
        speedup     = npTime / mpiTime 
        linearity   = speedup / nProcessors
        '''
        print('np: {}s'.format(npTime))
        print('mpi: {}s'.format(mpiTime))
        print('mpidot ran {}x faster than np.dot on {} cores (linearity={})'.format(
            speedup, nProcessors, linearity))
        '''
        print('{},{},{}'.format(speedup, nProcessors, linearity))

main()

#!/usr/bin/env python3
import subprocess

import matplotlib.pyplot as plt


def main():
    avg = lambda l : sum(l) / len(list(l))
    nTests   = 10
    speedups = []
    nprocs   = []
    for i in range(1, 8):
        processorData = []
        for _ in range(nTests):
            data = subprocess.check_output('mpiexec -np {} python3 mxmul.py'.format(i), shell=True)
            data = data.decode('utf-8')
            processorData.append(tuple(float(x) for x in data.split(',')))

        speedups.append(avg(item[0] for item in processorData))
        nprocs.append(avg(item[1] for item in processorData))

    plt.plot(nprocs, speedups)
    plt.xlabel('processors')
    plt.ylabel('speedup')
    plt.show()

main()

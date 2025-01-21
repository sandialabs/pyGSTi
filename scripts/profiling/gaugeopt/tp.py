#!/usr/bin/env python3
import pygsti
from load import load
from pygsti.tools import timed_block


def main():
    gs, gs_target = load()
    #envSettings = dict(MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, OMP_NUM_THREADS=1)

    with timed_block('TP penalty gauge opt'):
        gs_gaugeopt = pygsti.gaugeopt_to_target(gs, gs_target,
                                                item_weights={'spam' : 0.0001, 'gates':1.0},
                                                TPpenalty=1.0)

if __name__ == '__main__':
    main()

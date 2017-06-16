#!/usr/bin/env python3
import pygsti
from pygsti.tools.timed_block   import timed_block

import pickle
from contextlib import contextmanager

def main():
    with open('2qbit_results.pkl', 'rb') as infile:
        results = pickle.load(infile)

    est = results.estimates['default']
    gs_target = est.gatesets['target']
    gs = est.gatesets['final iteration estimate']

    #envSettings = dict(MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, OMP_NUM_THREADS=1)

    with timed_block('Basic gauge opt:'):
        gs_gaugeopt = pygsti.gaugeopt_to_target(gs, gs_target, 
                itemWeights={'spam' : 0.0001, 'gates':1.0},
                spamMetric='frobenius',
                gatesMetric='frobenius')
    '''

    with timed_block('Gauge opt with CP Penalty:'):
        gs_gaugeopt = pygsti.gaugeopt_to_target(gs, gs_target, 
                itemWeights={'spam' : 0.0001, 'gates':1.0}, 
                CPpenalty=1.0, 
                validSpamPenalty=1.0)

    with timed_block('TP penalty gauge opt'):
        gs_gaugeopt = pygsti.gaugeopt_to_target(gs, gs_target, 
                itemWeights={'spam' : 0.0001, 'gates':1.0}, 
                TPpenalty=1.0)
    '''

if __name__ == '__main__':
    main()

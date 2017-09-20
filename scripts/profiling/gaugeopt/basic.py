#!/usr/bin/env python3
from pygsti.algorithms import gaugeopt_to_target, contract
from pygsti.tools import timed_block
from pygsti.construction import std2Q_XYICNOT

import pickle
from contextlib import contextmanager

from load import load

def main():
    gs_target  = std2Q_XYICNOT.gs_target
    gs = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001).rotate(0.1)
    gs = gs.kick(0.1, seed=1234)

    gs_target.set_all_parameterizations("TP")
    gs = contract(gs, "TP")
    gs.set_all_parameterizations("TP")

    #envSettings = dict(MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, OMP_NUM_THREADS=1)

    with timed_block('Basic gauge opt:'):
        gs_gaugeopt = gaugeopt_to_target(
            gs, gs_target,
            #method="L-BFGS-B",
            method="auto",
            itemWeights={'spam' : 0.0001, 'gates':1.0},
            spamMetric='frobenius',
            gatesMetric='frobenius',
            cptp_penalty_factor=1.0,
            spam_penalty_factor=1.0,
            verbosity=3, checkJac=True)
        print("Final Diff = ", gs_gaugeopt.frobeniusdist(gs_target, None, 1.0, 0.0001))
        print(gs_gaugeopt.strdiff(gs_target))

if __name__ == '__main__':
    main()

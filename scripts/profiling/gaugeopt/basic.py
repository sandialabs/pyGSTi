#!/usr/bin/env python3
from pygsti.algorithms import gaugeopt_to_target, contract
from pygsti.tools import timed_block
from pygsti.construction import std2Q_XYICNOT

import pickle
from contextlib import contextmanager

from mpi4py import MPI
comm = MPI.COMM_WORLD
#comm = None

def main():
    gs_target  = std2Q_XYICNOT.gs_target
    gs = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001).rotate(0.1)
    gs = gs.kick(0.1, seed=1234)

    gs_target.set_all_parameterizations("TP")
    gs = contract(gs, "TP")
    gs.set_all_parameterizations("TP")

    #del gs.spamdefs['11']
    #del gs_target.spamdefs['11']
    #del gs.preps['rho0']
    #del gs_target.preps['rho0']
    #envSettings = dict(MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, OMP_NUM_THREADS=1)

    print(gs.get_prep_labels())
    print(gs.get_effect_labels())
    print(gs.num_elements(include_povm_identity=True))
    print(gs.spamdefs)

    with timed_block('Basic gauge opt:'):
        gs_gaugeopt = gaugeopt_to_target(
            gs, gs_target,
            #method="L-BFGS-B",
            method="auto",
            item_weights={'spam' : 0.0001, 'gates':1.0},
            spam_metric='frobenius',
            gates_metric='frobenius',
            cptp_penalty_factor=1.0,
            spam_penalty_factor=1.0,
            comm=comm, verbosity=3, check_jac=True)

        if comm is None or comm.Get_rank() == 0:
            print("Final Diff = ", gs_gaugeopt.frobeniusdist(gs_target, None, 1.0, 0.0001))
            print(gs_gaugeopt.strdiff(gs_target))

if __name__ == '__main__':
    main()

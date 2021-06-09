#!/usr/bin/env python3
from mpi4py import MPI

from pygsti.algorithms import gaugeopt_to_target
from pygsti.construction import std1Q_XYI
from pygsti.tools import timed_block

comm = MPI.COMM_WORLD
#comm = None


def main():
    gs_target  = std1Q_XYI.gs_target
    gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001).rotate(0.1)

    #DEBUG
    #del gs_target.spamdefs['1']
    #del gs_datagen.spamdefs['1']
    print(gs_datagen.get_prep_labels())
    print(gs_datagen.get_effect_labels())
    print(gs_datagen.num_elements(include_povm_identity=True))
    print(gs_datagen.spamdefs)
    
    with timed_block('Basic gauge opt:'):
        gs_gaugeopt = gaugeopt_to_target(
            gs_datagen, gs_target, tol=1e-7,
            method="auto",
            #method="L-BFGS-B",
            item_weights={'spam' : 1.0, 'gates':1.0},
            spam_metric='frobenius',
            gates_metric='frobenius', check_jac=True,
            cptp_penalty_factor=1.0,
            spam_penalty_factor=1.0,
            comm=comm, verbosity=3)

        if comm is None or comm.Get_rank() == 0:
            print("Final Diff = ", gs_gaugeopt.frobeniusdist(gs_target, None, 1.0, 1.0))
            print(gs_gaugeopt.strdiff(gs_target))

if __name__ == '__main__':
    main()

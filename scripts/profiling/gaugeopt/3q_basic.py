#!/usr/bin/env python3
import pygsti
from pygsti.algorithms import gaugeopt_to_target
from pygsti.tools import timed_block

import pickle
from contextlib import contextmanager

from load import load_3q

def main():
    gs_target = pygsti.construction.build_gateset(
            [8], [('Q0','Q1','Q2')],['Gx1','Gy1','Gx2','Gy2','Gx3','Gy3','Gcnot12','Gcnot23'],
            [ "X(pi/2,Q0):I(Q1):I(Q2)", "Y(pi/2,Q0):I(Q1):I(Q2)", "I(Q0):X(pi/2,Q1):I(Q2)", "I(Q0):Y(pi/2,Q1):I(Q2)",
                  "I(Q0):I(Q1):X(pi/2,Q2)", "I(Q0):I(Q1):Y(pi/2,Q2)", "CX(pi,Q0,Q1):I(Q2)", "I(Q0):CX(pi,Q1,Q2)"],
            prep_labels=['rho0'], prep_expressions=["0"],
            effect_labels=['E0','E1','E2','E3','E4','E5','E6'], effect_expressions=["0","1","2","3","4","5","6"],
            spamdefs={'upupup': ('rho0','E0'), 'upupdn': ('rho0','E1'), 'updnup': ('rho0','E2'), 'updndn': ('rho0','E3'),
                'dnupup': ('rho0','E4'), 'dnupdn': ('rho0','E5'), 'dndnup': ('rho0','E6'), 'dndndn': ('rho0','remainder')},
            basis="pp")
    gs = load_3q()

    with timed_block('Basic gauge opt (3Q)'):
        gs_gaugeopt = gaugeopt_to_target(gs, gs_target, 
                item_weights={'spam' : 0.0001, 'gates':1.0},
                spam_metric='frobenius',
                gates_metric='frobenius')

if __name__ == '__main__':
    main()

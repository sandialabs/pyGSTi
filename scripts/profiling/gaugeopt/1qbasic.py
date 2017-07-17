#!/usr/bin/env python3
from pygsti.construction      import std1Q_XYI
from pygsti.algorithms        import gaugeopt_to_target
from pygsti.tools.timed_block import timed_block

def main():
    gs_target = std1Q_XYI.gs_target
    gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
    with timed_block('Basic gauge opt:'):
        gs_gaugeopt = gaugeopt_to_target(gs_datagen, gs_target, 
                itemWeights={'spam' : 1.0, 'gates':1.0},
                #itemWeights={'spam' : 0.0001, 'gates':1.0},
                spamMetric='frobenius',
                gatesMetric='frobenius')

if __name__ == '__main__':
    main()

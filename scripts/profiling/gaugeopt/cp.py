#!/usr/bin/env python3
import pygsti
from load import load
from pygsti.tools import timed_block


def main():
    gs, gs_target = load()
    with timed_block('Gauge opt with CP Penalty:'):
        gs_gaugeopt = pygsti.gaugeopt_to_target(gs, gs_target, 
                item_weights={'spam' : 0.0001, 'gates':1.0}, 
                CPpenalty=1.0, 
                validSpamPenalty=1.0)

if __name__ == '__main__':
    main()

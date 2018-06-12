from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np

def check_valid_gate(gate):
    
    # To do: check that gate is type Gate.
    assert(type(gate.label) is unicode), "The gate label should be unicode!"
    assert(type(gate.qubits) is list), "The list of qubits on which the gate acts should be a list!"
    
    for i in range(1,gate.number_of_qubits):
        assert(type(gate.qubits[i]) is int), "The qubits on which a gate acts should be indexed by integers!"

def check_valid_circuit_layer(layer,n):
    
    assert(type(layer) is list), "A gate layer must be a list!"
    #
    # To do: add other less trivial tests.
    #
   


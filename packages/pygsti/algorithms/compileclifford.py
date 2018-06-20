""" Clifford compilation routines """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import copy as _copy

from ..objects.circuit import Circuit as _Circuit
from ..baseobjs import Label as _Label
from ..tools import symplectic as _symp
from ..tools import matrixmod2 as _mtx

from . import compilesymplectic as _csymp

def compile_clifford(s, p, pspec=None, depth_compression=True, algorithms=['DGGE','RGGE'], 
                     costfunction='2QGC', iterations={'RGGE':4}, prefix_paulis=False, 
                     pauli_randomize=False):
    """
    Compiles a Clifford gate, described by the symplectic matrix s and vector p, into
    a circuit over the specified gateset, or, a standard gateset.
    
    Parameters
    ----------
    s : Todo: fill in

    Returns
    -------
    Circuit
        Todo: fill in
        
    """
    assert(_symp.check_valid_clifford(s,p)), "Input is not a valid Clifford!"
    n = _np.shape(s)[0]//2
    
    if pspec is not None:
        assert(n == pspec.number_of_qubits), "...."
    
    # Create a circuit that implements a Clifford with symplectic matrix s.
    circuit = _csymp.compile_symplectic(s, pspec=pspec, algorithms=algorithms,  costfunction= costfunction, 
                                 iterations=iterations, pauli_randomize=pauli_randomize, 
                                 depth_compression=depth_compression)
    
    # If we did Pauli randomization, remove the random Pauli at the end that we will add a determinstic
    # Pauli.
    #
    # Todo : put this back in -- it should be there.
    #
    #if pauli_randomize:
    #    if prefix_paulis:
    #        circuit.delete_layer(0)
    #    else:
    #        circuit.delete_layer(circuit.depth()-1)
     
    
    sreps = pspec.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
    temp_s, temp_p = _symp.symplectic_rep_of_clifford_circuit(circuit,  sreps)
        
    assert(_np.array_equal(s,temp_s))
    assert(_symp.check_valid_clifford(temp_s,temp_p))
        
    s_form = _symp.symplectic_form(n)
    
    if prefix_paulis:
        vec = _np.dot(s_form, (p - temp_p)//2)      
    else:
        vec = _np.dot(s,_np.dot(s_form, (p - temp_p)//2))
    vec = vec % 2
    
    pauli_layer = []
    for q in range(0,n):
        if vec[q] == 0 and vec[q+n] == 0:
            pauli_layer.append(('I',q))
        elif vec[q] == 0 and vec[q+n] == 1:
            pauli_layer.append(('Z',q))
        elif vec[q] == 1 and vec[q+n] == 0:
            pauli_layer.append(('X',q))
        elif vec[q] == 1 and vec[q+n] == 1:
            pauli_layer.append(('Y',q))
    
    pauli_circuit = _Circuit(gatestring=pauli_layer,num_lines=n)
    
    # Only change gate library if we have a DeviceSpec with compilations.
    if pspec is not None:
        pauli_circuit.change_gate_library(pspec.compilations['absolute'])
    
    if prefix_paulis:
        circuit.prefix_circuit(pauli_circuit)
    else:
        circuit.append_circuit(pauli_circuit)

    sreps = pspec.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(circuit, sreps)
    
    assert(_symp.check_valid_clifford(s_out,p_out))
    assert(_np.array_equal(s,s_out)) 
    assert(_np.array_equal(p,p_out)) 
    
    return circuit

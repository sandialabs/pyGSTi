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

def compile_clifford(s, p, pspec=None, iterations=20, algorithms=['DGGE','RGGE'], 
                     costfunction='2QGC', prefixpaulis=False, paulirandomize=False):
    """
    Compiles an n-qubit Clifford gate, described by the symplectic matrix s and vector p, into
    a circuit over the specified gateset, or, a standard gateset. Clifford gates can be converted
    to, or sampled in, the symplectic representation using the functions in pygsti.tools.symplectic.
        
    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers.
    
    p : array over [0,1]
        A length-2n vector over [0,1,2,3] that, together with s, defines a valid n-qubit Clifford
        gate.
        
    pspec : ProcessorSpec, optional
        An n-qubit ProcessorSpec object that encodes the device that the Clifford is being compiled
        for. If this is specified, the output circuit is over the gates available in this device.
        If this is None, the output circuit is over the "canonical" gateset of CNOT gates between
        all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation. In most circumstances, the output will be much more
        useful if a ProcessorSpec is provided.
        
    iterations : int, optional
        Some of the allowed algorithms are randomized. This is the number of iterations used in
        each of the randomized algorithms specified. If any randomized algorithms are specified,
        the time taken by this function increases linearly with `iterations`. Increasing `iterations`
        *may* improve the obtained compilation (the "cost" of the obtained circuit, as specified
        by `costfunction` may decrease towards some asymptotic value).
        
        todo : finish this.

    Returns
    -------
    Circuit
        Todo: fill in
        
    """
    assert(_symp.check_valid_clifford(s,p)), "Input is not a valid Clifford!"
    n = _np.shape(s)[0]//2
    
    if pspec is not None:
        assert(n == pspec.number_of_qubits), "The Clifford and the ProcessorSpec are for a different number of qubits!"
    
    # Create a circuit that implements a Clifford with symplectic matrix s.
    circuit = _csymp.compile_symplectic(s, pspec=pspec, iterations=iterations, algorithms=algorithms,
                                        costfunction= costfunction, paulirandomize=paulirandomize)

    temp_s, temp_p = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
    
    # Find the necessary Pauli layer to compile the correct Clifford, not just the correct
    # Clifford up to Paulis. The required Pauli layer depends on whether we pre-fix or post-fix it.
    if prefixpaulis:        
        pauli_layer = _symp.find_premultipled_pauli(s,temp_p,p)
    else:
        pauli_layer = _symp.find_postmultipled_pauli(s,temp_p,p)    
    # Turn the Pauli layer into a circuit.
    pauli_circuit = _Circuit(gatestring=pauli_layer,num_lines=n)   
    # Only change gate library of the Pauli circuit if we have a ProcessorSpec with compilations.    
    if pspec is not None:
        pauli_circuit.change_gate_library(pspec.compilations['absolute'])    
    # Prefix or post-fix the Pauli circuit to the main symplectic-generating circuit.
    if prefixpaulis:
        circuit.prefix_circuit(pauli_circuit)
    else:
        circuit.append_circuit(pauli_circuit)
    
    # Check that the correct Clifford has been compiled. This should never fail.
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)    
    assert(_symp.check_valid_clifford(s_out,p_out))
    assert(_np.array_equal(s,s_out)) 
    assert(_np.array_equal(p,p_out)) 
    
    return circuit

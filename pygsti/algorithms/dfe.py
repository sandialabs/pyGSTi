
import copy as _copy
import itertools as _itertools

import numpy as _np

from pygsti.algorithms import compilers as _cmpl
from pygsti.circuits import circuit as _cir
from pygsti.baseobjs import label as _lbl
from pygsti.tools import group as _rbobjs
from pygsti.tools import symplectic as _symp
from pygsti.tools import compilationtools as _comp
from pygsti.tools import internalgates as _gates
from pygsti.algorithms import randomcircuit as _rc

def sample_dfe_circuit(pspec, circuit, clifford_compilations, seed=None):
    """
 	... ... ... ...
    """
    qubit_labels = circuit.line_labels
    n = len(qubit_labels)
    rand_state = _np.random.RandomState(seed)  # Ok if seed is None
    
    rand_pauli, rand_sign, pauli_circuit = _rc._sample_random_pauli(n = n, pspec = pspec, qubit_labels=qubit_labels,
                                                                   absolute_compilation = clifford_compilations,
                                                                   circuit = True, include_identity = False)

    s_inputstate, p_inputstate, s_init_layer, p_init_layer, prep_circuit = _rc._sample_stabilizer(rand_pauli, rand_sign, clifford_compilations, qubit_labels)
    
    s_pc, p_pc = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec=pspec.subset(gate_names_to_include='all', qubit_labels_to_keep=qubit_labels)) #note: if the pspec contains gates not in pyGSTi, this
    
    # build the initial layer of the circuit
    full_circuit = prep_circuit.copy(editable=True)
        
    # find the symplectic matrix / phase vector of the input circuit
    s_rc, p_rc = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec.subset(gate_names_to_include='all', qubit_labels_to_keep=qubit_labels))
    
    s_composite, p_composite = _symp.compose_cliffords(s1 = s_init_layer, p1 = p_init_layer, s2 = s_rc, p2 = p_rc)

    full_circuit.append_circuit_inplace(circuit)
    
    s_outputstate, p_outputstate = _symp.apply_clifford_to_stabilizer_state(s_rc, p_rc, s_inputstate, p_inputstate)
    
    # Figure out what stabilizer of s_outputstate, rand_pauli was mapped too
    s_rc_inv, p_rc_inv = _symp.inverse_clifford(s_rc, p_rc) # U^(-1)
    s_new_pauli, p_new_pauli = _symp.compose_cliffords(s_rc_inv, p_rc_inv, s_pc, p_pc) # PU^(-1)
    s_new_pauli, p_new_pauli = _symp.compose_cliffords(s_new_pauli, p_new_pauli, s_rc, p_rc) # UPaU^(-1)
        
    pauli_vector = p_new_pauli
    pauli = [i[0] for i in _symp.find_pauli_layer(pauli_vector, [j for j in range(n)])]
    measurement = ['I' if i == 'I' else 'Z' for i in pauli]
        
    # Turn the stabilizer into an all Z and I stabilizer
    s_stab, p_stab, stab_circuit = _rc._stabilizer_to_all_zs(pauli, qubit_labels, clifford_compilations)
    
    full_circuit.append_circuit_inplace(stab_circuit)
    
    s_inv, p_inv = _symp.inverse_clifford(s_stab, p_stab)
    s_cc, p_cc = _symp.compose_cliffords(s_inv, p_inv, s_composite, p_composite)
    s_cc, p_cc = _symp.compose_cliffords(s_composite, p_composite, s_stab, p_stab) # MUPaU^(-1)M^(-1)
         
    s_outputstate, p_outputstate = _symp.apply_clifford_to_stabilizer_state(s_stab, p_stab, s_outputstate, p_outputstate)

    full_circuit.done_editing()
    sign = _rc._determine_sign(s_outputstate, p_outputstate, measurement)
    
    outcircuit = full_circuit
        
    return outcircuit, measurement, sign
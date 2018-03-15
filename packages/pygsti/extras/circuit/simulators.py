from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy

def simulate(circuit,model,inputstate=None,store=True,returnall=False):
    """
    Simulates a circuit, given a model for the gates in this circuit.
    """
    if model.mtype == 'mixedstate':
        out = mixedstate_simulator(circuit, model, input_bitstring=inputstate, store=store, returnall=returnall)
        
    elif model.mtype == 'purestate':
        out = purestate_simulator(circuit, model, inputstate=inputstate, store=store, returnall=returnall)
        
    elif model.mtype == 'pauliclifford':
        # Todo : write this efficient-in-n pauli-errors for Clifford circuits simulator.
        print("This functionality is not yet supported")
        return None
        out = pauliclifford_simulator(circuit, model, inputstate=inputstate, store=store, returnall=returnall)
                
    else:
        raise ValueError("Model type not understood/supported")
        
    return out
    
def mixedstate_simulator(circuit,model,input_bitstring=None,store=True,returnall=False):
    """
    Todo: docstring
        
    Probably should have an input as a general density matrix and measurment effect.
    """
    n = circuit.number_of_qubits
    
    if input_bitstring is None: 
        input_bitstring = _np.zeros(n,int)
        
        input_state = _np.array([1.])
        for i in range(0,n):
            input_state = _np.kron(input_state,_np.array([1.,0.,0.,(-1.)**(input_bitstring[i])]))
        input_state = input_state / _np.sqrt(2)**n                      
        
    output_state = _np.copy(input_state)
                                       
    for l in range(0,circuit.depth()):
        superoperator = model.get_layer_as_operator(circuit.get_circuit_layer(l),store=store)
        output_state = _np.dot(superoperator,output_state)
       
    probabilities = {}
    for i in range(0,2**n):
        bit_string = [0 for x in range(0,n)]
        bit_string_end =  [int(x) for x in _np.base_repr(i,2)]
        bit_string[n-len(bit_string_end):] = bit_string_end
            
        possible_outcome = _np.array([1.])
        for j in range(0,n):
            possible_outcome = _np.kron(possible_outcome,_np.array([1.,0.,0.,(-1.)**bit_string[j]]))
                    
        probabilities[tuple(bit_string)] = _np.dot(possible_outcome,output_state)/(_np.sqrt(2)**n)
        
    if returnall:
        return probabilities, output_state
    else:
        return probabilities
    
#
# Todo: add in the possibility of Pauli errors into the vector state simulator
#
def purestate_simulator(circuit,model,inputstate=None,store=True,returnall=False):
    """
    Todo
    """
    n = circuit.number_of_qubits
    if inputstate is None:           
        inputstate = _np.zeros(2**n,complex)
        inputstate[0] = 1

    outputstate = _np.copy(inputstate)
                                       
    for l in range(0,circuit.depth()):
        unitary = model.get_circuit_layer_as_operator(circuit.get_circuit_layer(l),store=store)
        outputstate = _np.dot(unitary,outputstate)

    probs = abs(outputstate)**2
    probs_as_dict = {}
    for i in range(0,2**n):
        bit_string = [0 for x in range(0,n)]
        bit_string_end =  [int(x) for x in _np.base_repr(i,2)]
        bit_string[n-len(bit_string_end):] = bit_string_end
        probs_as_dict[tuple(bit_string)] = probs[i]
        
    if returnall:
        return probs_as_dict, outputstate
    else:
        return probs_as_dict
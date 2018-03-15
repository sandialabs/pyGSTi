from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy
from scipy import mod as _mod
from ..circuit import symplectic as _symp
from ..circuit import circuit as _cir
from ..circuit import compileclifford as _comp


def circuit_layer_sampler_2QW(ds,two_qubit_weighting=0.5):

    available_gates_1q = _copy.deepcopy(ds.allgates)
    available_gates_2q = _copy.deepcopy(ds.allgates)
    xx = len(available_gates_1q)
    for i in range(0,xx):
        if available_gates_1q[xx-1-i].number_of_qubits != 1:
            del available_gates_1q[xx-1-i]
    for i in range(0,xx):
        if available_gates_2q[xx-1-i].number_of_qubits != 2:
            del available_gates_2q[xx-1-i]
            
    sampled_layer = []
    
    if two_qubit_weighting is not None:
        weighting = [1-two_qubit_weighting,two_qubit_weighting]
    
    l = ds.number_of_qubits
    remaining_qubits = list(_np.arange(0,ds.number_of_qubits))
    qubits_used = 0
    while qubits_used < l:
               
        # Pick a random qubit
        r = _np.random.randint(0,ds.number_of_qubits-qubits_used)
        q = remaining_qubits[r]
        del remaining_qubits[r]
        
        remaining_containing_q_1q = []
        ll = len(available_gates_1q)
        for i in range(0,ll):
            if q in available_gates_1q[ll-1-i].qubits:
                remaining_containing_q_1q.append(available_gates_1q[ll-1-i])
                del available_gates_1q[ll-1-i]
                
        remaining_containing_q_2q = []
        ll = len(available_gates_2q)
        for i in range(0,ll):
            if q in available_gates_2q[ll-1-i].qubits:
                remaining_containing_q_2q.append(available_gates_2q[ll-1-i])
                del available_gates_2q[ll-1-i]
                
        # Weighted choice for 1 or 2 qubit gate
        if two_qubit_weighting == None:
            nrm = len(remaining_containing_q_1q)+len(remaining_containing_q_2q)
            weighting = [len(remaining_containing_q_1q)/nrm,len(remaining_containing_q_2q)/nrm]
              
        if len(remaining_containing_q_2q) == 0:
            xx = 1
        else:
            xx = _np.random.choice([1,2],p=weighting)
        
        if xx == 1:
            r = _np.random.randint(0,len(remaining_containing_q_1q))
            sampled_layer.append(remaining_containing_q_1q[r])
            qubits_used += 1
            
        if xx == 2:
            r = _np.random.randint(0,len(remaining_containing_q_2q))
            sampled_layer.append(remaining_containing_q_2q[r])
        
            other_qubit = remaining_containing_q_2q[r].qubits[0]
            if other_qubit == q:
                other_qubit = remaining_containing_q_2q[r].qubits[1]
                
            ll = len(available_gates_1q)
            for i in range(0,ll):
                if other_qubit in available_gates_1q[ll-1-i].qubits:                       
                    del available_gates_1q[ll-1-i]
                    
            ll = len(available_gates_2q)
            for i in range(0,ll):
                if other_qubit in available_gates_2q[ll-1-i].qubits:                       
                    del available_gates_2q[ll-1-i]
                        
            del remaining_qubits[remaining_qubits.index(other_qubit)]
            
            qubits_used += 2
    
    return sampled_layer

def circuit_layer_sampler_sectors(ds, sectors, two_qubit_prob):
    
    twoqubitgates = sectors[_np.random.randint(0,len(sectors))]    
    remaining_qubits = list(_np.arange(0,ds.number_of_qubits))
    sampled_layer = []
    
    for i in range(0,len(twoqubitgates)):
        if _np.random.binomial(1,two_qubit_prob) == 1:
            gate = twoqubitgates[i]
            sampled_layer.append(gate)
            del remaining_qubits[remaining_qubits.index(gate.qubits[0])]
            del remaining_qubits[remaining_qubits.index(gate.qubits[1])]
            
    for i in range(0,len(remaining_qubits)):
        qubit = remaining_qubits[i]        
        possiblegates = ds.gatesonqubits[qubit]
        gate = possiblegates[_np.random.randint(0,len(possiblegates))]
        sampled_layer.append(gate)

    return sampled_layer

# Todo : better name for this function
#
def circuit_sampler(ds,length,sampler='weights',sampler_args={'two_qubit_weighting' : 0.5}):
    
    if sampler == 'weights':
        two_qubit_weighting = sampler_args['two_qubit_weighting']
        
    elif sampler == 'sectors':
        two_qubit_prob = sampler_args['two_qubit_prob']
        sectors = sampler_args['sectors']
        
    circuit = _cir.Circuit(n=ds.number_of_qubits)
    
    for i in range(0,length):
        
        if sampler == 'weights':
            layer = circuit_layer_sampler_2QW(ds,two_qubit_weighting=two_qubit_weighting)
            
        elif sampler == 'sectors':
            layer = circuit_layer_sampler_sectors(ds, sectors=sectors, two_qubit_prob=two_qubit_prob)
        else:
            layer = sampler(ds, length, sampler_args)
            
        circuit.insert_layer(layer,0)
             
    return circuit


def construct_grb_circuit(ds, length, sampler='weights',sampler_args={'two_qubit_weighting' : 0.5,},  
                          twirled=True, stabilizer=True, algorithm='GGE', depth_compression=True, 
                          return_partitioned = False):
    
    # Sample random circuit, and find the symplectic matrix / phase vector it implements    
    random_circuit = circuit_sampler(ds=ds, length=length, sampler=sampler,
                                     sampler_args=sampler_args)
    
    sl = ds.gateset.smatrix
    pl = ds.gateset.svector
    n = ds.number_of_qubits
    
    s_rc, p_rc = _symp.composite_clifford_from_clifford_circuit(random_circuit,s_dict=sl,p_dict=pl)
    
    if twirled:
        
        s_initial, p_initial = _symp.random_clifford(n)
        s_composite, p_composite = _symp.compose_cliffords(s_initial, p_initial, s_rc, p_rc)
        
        if stabilizer:
            initial_circuit = _comp.stabilizer_state_preparation_circuit(s_initial, p_initial, ds)            
        else:
            initial_circuit = _comp.compile_clifford(s_initial, p_initial, ds, 
                                                           depth_compression=depth_compression, 
                                                           algorithm=algorithm,
                                                           prefix_paulis=True)
        
        
    else:
        s_composite = _copy.deepcopy(s_rc)
        p_composite = _copy.deepcopy(p_rc)
        
    
    s_inverse, p_inverse = _symp.inverse_clifford(s_composite, p_composite)
    
    #
    # Put something here that allows for a random Pauli at the end of the circuit?
    #
    
    if stabilizer:
        inversion_circuit = _comp.stabilizer_measurement_preparation_circuit(s_inverse, p_inverse, ds)   
    else:
        inversion_circuit = _comp.compile_clifford(s_inverse, p_inverse, ds, 
                                                        depth_compression=depth_compression,
                                                        algorithm=algorithm,
                                                        prefix_paulis=False)

    if not return_partitioned:
        
        if twirled:
            full_circuit = _copy.deepcopy(initial_circuit)
            full_circuit.append_circuit(random_circuit)
            full_circuit.append_circuit(inversion_circuit)
        else:
            full_circuit = _copy.deepcopy(random_circuit)
            full_circuit.append_circuit(inversion_circuit)
            
        return full_circuit
    
    else:
        if twirled:
            return initial_circuit, random_circuit, inversion_circuit
        else:
            return random_circuit, inversion_circuit
        
#
# Todo: merge below function with the function above, and add two or more wrap-around functions: 
# construct_std_practice_primitive_rb_experiment()
# construct_std_practice_clifford_rb_experiment()
# construct_std_practice_interleaved_clifford_rb_experiment()
# construct_std_practice_primitive_clifford_rb_experiment()
#
def construct_cliffordrb_circuit(ds, length, algorithm='GGE', depth_compression=True):
    
   
    sl = ds.gateset.smatrix
    pl = ds.gateset.svector
    n = ds.number_of_qubits
       
    s_composite = _np.identity(2*n,int)
    p_composite = _np.zeros((2*n),int)
    
    full_circuit = _cir.Circuit(n=n)
    
    for i in range(0,length):
    
        s, p = _symp.random_clifford(n)
        circuit = _comp.compile_clifford(s, p, ds, depth_compression=depth_compression, 
                                                  algorithm=algorithm, prefix_paulis=True)
        # Keeps track of the current composite Clifford
        s_composite, p_composite = _symp.compose_cliffords(s_composite, p_composite, s, p)
        full_circuit.append_circuit(circuit)
        
    
    s_inverse, p_inverse = _symp.inverse_clifford(s_composite, p_composite)
    
    #
    # Put something here that allows for a random Pauli at the end of the circuit?
    #
    
    inversion_circuit = _comp.compile_clifford(s_inverse, p_inverse, ds, 
                                                         depth_compression=depth_compression, 
                                                  algorithm=algorithm, prefix_paulis=True)
    
    full_circuit.append_circuit(inversion_circuit)
    
            
    return full_circuit

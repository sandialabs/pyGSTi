from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy
from scipy import mod as _mod

from ...algorithms import compileclifford as _comp
from ...objects import circuit as _cir
from ...baseobjs import label as _lbl
from ...tools import symplectic as _symp

#from ... import symplectic as _symp
#from ..circuit import circuit as _cir
#from ..circuit import compileclifford as _comp

# Todo : make these methods work when the qubits have labels other than integers on 0,...,n-1.


def sample_circuit_layer_by_pairings(pspec, onequbit_gatenames=None, twoqubit_gatenames=None, 
                                     twoqubit_probability=0.5):   
    """
    A circuit layer sampler than pairs up qubits and then implements a two-qubit gate on them
    with the specified probability. Note that this function currently only works with all-to-all
    connectivity, but it does *not* check for this.
    
    """    
    n = pspec.number_of_qubits
        
    if (onequbit_gatenames is None) or (twoqubit_gatenames is None):
    
        onequbit_gatenames = []
        twoqubit_gatenames = []

        gatenames = list(pspec.models['clifford'].gates.keys())

        for gate in pspec.models['clifford'].gates:

            if gate.number_of_qubits == 1 and gate.name not in onequbit_gatenames:
                onequbit_gatenames.append(gate.name)
            if gate.number_of_qubits == 2 and gate.name not in twoqubit_gatenames:
                twoqubit_gatenames.append(gate.name)
                
    qubits = list(range(n))
    sampled_layer = []
    num_onequbit_gatenames = len(onequbit_gatenames)
    num_twoqubit_gatenames = len(twoqubit_gatenames)
    
    # If there is an odd number of qubits, begin by picking one to have a 1-qubit gate.
    if n % 2 != 0:
        q = qubits[np.random.randint(0,n)]
        name = onequbit_gatenames[_np.random.randint(0,num_onequbit_gatenames)]
        del qubits[q]       
        sampled_layer.append(_lbl.Label(name,q))
    
    for i in range(n//2):
        
        # Pick two of the remaining qubits
        index = _np.random.randint(0,len(qubits))
        q1 = qubits[index]
        del qubits[index] 
        index = _np.random.randint(0,len(qubits))
        q2 = qubits[index]
        del qubits[index] 
        
        if _np.random.binomial(1,twoqubit_probability) == 1:
            name = twoqubit_gatenames[_np.random.randint(0,num_twoqubit_gatenames)]
            sampled_layer.append(_lbl.Label(name,(q1,q2)))
        else:
            name1 = onequbit_gatenames[_np.random.randint(0,num_onequbit_gatenames)]
            name2 = onequbit_gatenames[_np.random.randint(0,num_onequbit_gatenames)]
            sampled_layer.append(_lbl.Label(name1,q1))
            sampled_layer.append(_lbl.Label(name2,q2))                     
    
    return sampled_layer


def sample_circuit_layer_by_2Qweighting(pspec,gates1Q=None,gates2Q=None,two_qubit_weighting=0.5):

    
    if gates1Q != None:
        available_gates_1q = _copy.copy(gates1Q)
    
    else:
        assert('clifford' in list(pspec.models.keys())), "The provided ProcessorSpec object `pspec` must contain a 'clifford' gateset."
        available_gates_1q = list(pspec.models['clifford'].gates.keys())
        d = len(available_gates_1q)
    
        for i in range(0,d):
            if available_gates_1q[d-1-i].number_of_qubits != 1:
                del available_gates_1q[d-1-i]
    
    if gates2Q != None:
        available_gates_2q = _copy.copy(gates2Q)
    
    else:
        assert('clifford' in list(pspec.models.keys())), "The provided ProcessorSpec object `pspec` must contain a 'clifford' gateset."
        available_gates_2q = list(pspec.models['clifford'].gates.keys())
        d = len(available_gates_2q)               
                        
        for i in range(0,d):
            if available_gates_2q[d-1-i].number_of_qubits != 2:
                del available_gates_2q[d-1-i]
    
    
    sampled_layer = [] 
    
    if two_qubit_weighting != None: 
        weighting = [1-two_qubit_weighting,two_qubit_weighting]
    
    l = pspec.number_of_qubits
    remaining_qubits = list(_np.arange(0,pspec.number_of_qubits))
    qubits_used = 0
    
    while qubits_used < l:
               
        # Pick a random qubit
        r = _np.random.randint(0,pspec.number_of_qubits-qubits_used)
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

def sample_circuit_layer_by_sectors(pspec, sectors, two_qubit_prob, singlequbitgates='all'):
    
    twoqubitgates = sectors[_np.random.randint(0,len(sectors))]    
    remaining_qubits = list(_np.arange(0,pspec.number_of_qubits))
    sampled_layer = []
    
    for i in range(0,len(twoqubitgates)):
        if _np.random.binomial(1,two_qubit_prob) == 1:
            gate = twoqubitgates[i]
            sampled_layer.append(gate)
            del remaining_qubits[remaining_qubits.index(gate.qubits[0])]
            del remaining_qubits[remaining_qubits.index(gate.qubits[1])]
            
    for i in range(0,len(remaining_qubits)):
        qubit = remaining_qubits[i]        
        possiblegates = pspec.clifford_gates_on_qubits[(qubit,)]
        
        if singlequbitgates != 'all':
            numgates = len(possiblegates)
            for k in range(1,numgates+1):
                if possiblegates[numgates-k].name not in singlequbitgates:
                    del possiblegates[numgates-k]
                
        gate = possiblegates[_np.random.randint(0,len(possiblegates))]
        sampled_layer.append(gate)

    return sampled_layer

def sample_circuit_layer_of_1Q_gates(pspec, gtype = 'primitives'):
    
    assert(gtype == 'primitives' or gtype == 'paulis'), "gtype must be'primitives' or 'paulis'"
    sampled_layer = []
    
    if gtype == 'primitives':
        for i in range(0,pspec.number_of_qubits):
            try:
                gate = pspec.clifford_gates_on_qubits[(i,)][_np.random.randint(0,len(pspec.clifford_gates_on_qubits[(i,)]))]
                sampled_layer.append(gate)
            
            except:
                raise ValueError ("There are no available 1Q gates on qubit {}. Check the ProcessorSpec is correctly initialized".format(i))
                
    if gtype == 'paulis':
        
        # Todo : currently this method breaks if the pauli gates are not in the gateset.
        
        plist = ['Gi','Gxpi','Gypi','Gzpi']
        #for pgl in plist:
        #    assert(pgl in list(pspec.models['clifford'].gates.keys())), "Currently, the Pauli gates must be natively available to use this method!"
            
        for i in range(0,pspec.number_of_qubits):
            sampled_pauli = plist[_np.random.randint(0,4)]
            sampled_layer.append(_lbl.Label(sampled_pauli,i))

    return sampled_layer

def sample_primitives_circuit(pspec, length, sampler='weights', sampler_args=[0.5,'all',None,'single'],
                              alternatewithlocal = False, localtype = 'primitives'):
    
    if sampler == 'weights':
        two_qubit_weighting = sampler_args[0]
        singlequbitgates = sampler_args[1]
        
        gates1q = list(pspec.models['clifford'].gates.keys())
        gates2q = list(pspec.models['clifford'].gates.keys())
        d = len(gates1q)   
        for i in range(0,d):
            
            if singlequbitgates == 'all':
                if gates1q[d-1-i].number_of_qubits != 1:
                    del gates1q[d-1-i]
            else:
                if gates1q[d-1-i].name not in singlequbitgates:
                    del gates1q[d-1-i]
                    
            if gates2q[d-1-i].number_of_qubits != 2:
                 del gates2q[d-1-i]
        
    elif sampler == 'sectors':
        
        twoqubitprob = sampler_args[0]
        singlequbitgates = sampler_args[1]
        customsectors = sampler_args[2]
        stdsectors = sampler_args[3]
        
        if customsectors is not None:
            sectors = customsectors
        else:
            assert(stdsectors == 'single')
            
            sectors = []           
            for gate in pspec.models['clifford'].gates:
                if gate.number_of_qubits == 2:
                    sectors.append([gate,])
                    
    elif sampler == 'pairing':
    
    
        twoqubit_probability= sampler_args[0]
        #
        # Todo: this currently doesn't use a list of one-qubit gates specified for
        # sampling over. That should probably be fixed.
        #
        onequbit_gatenames = []
        twoqubit_gatenames = []

        gatenames = list(pspec.models['clifford'].gates.keys())

        for gate in pspec.models['clifford'].gates:

            if gate.number_of_qubits == 1 and gate.name not in onequbit_gatenames:
                onequbit_gatenames.append(gate.name)
            if gate.number_of_qubits == 2 and gate.name not in twoqubit_gatenames:
                twoqubit_gatenames.append(gate.name)
                       
    circuit = _cir.Circuit(gatestring=[],num_lines=pspec.number_of_qubits)
    
    if not alternatewithlocal:
        for i in range(0,length):
  
            if sampler == 'weights':       
                        
                layer = sample_circuit_layer_by_2Qweighting(pspec, gates1Q=gates1q, gates2Q=gates2q, 
                                                            two_qubit_weighting=two_qubit_weighting)
            
            elif sampler == 'sectors':
                layer = sample_circuit_layer_by_sectors(pspec, sectors=sectors, two_qubit_prob=twoqubitprob,
                                                       singlequbitgates=singlequbitgates)
                
            elif sampler == 'pairing':
                layer = sample_circuit_layer_by_pairings(pspec, onequbit_gatenames=onequbit_gatenames,
                                                         twoqubit_gatenames=twoqubit_gatenames,
                                                         twoqubit_probability=twoqubit_probability)
                
            else:
                layer = sampler(pspec, length, sampler_args)
            
            circuit.insert_layer(layer,0)
            
    if alternatewithlocal:
        
         for i in range(0,2*length+1):
                
                # For odd layers, we uniformly sample the specified type of local gates.
                local = not bool(i % 2)
                if local:
                    layer = sample_circuit_layer_of_1Q_gates(pspec, gtype = localtype)
                
                # For even layers, we sample according to the given distribution
                else:
                    if sampler == 'weights':
                        layer = sample_circuit_layer_by_2Qweighting(pspec,two_qubit_weighting=two_qubit_weighting)
            
                    elif sampler == 'sectors':
                        layer = sample_circuit_layer_by_sectors(pspec, sectors=sectors, two_qubit_prob=twoqubitprob)
                    elif sampler == 'pairings':
                        assert(False), "This funcationality is not yet included!"
                    
                    else:
                        layer = sampler(pspec, length, sampler_args)
                        
                circuit.insert_layer(layer,0)
    
    circuit.done_editing()
    return circuit


def sample_prb_circuit(pspec, length, sampler='weights',sampler_args=[0.5,'all',None,'single'],  
                         twirled=True, stabilizer=True, compiler_algorithm='GGE', depth_compression=True, 
                         alternatewithlocal = False, localtype = 'primitives', return_partitioned = False, 
                       iterations=5,relations=None,prep_measure_pauli_randomize=False,
                      improved_CNOT_compiler=True, ICC_custom_ordering=None, ICC_std_ordering='connectivity',
                      ICC_qubitshuffle=False):
    
    #
    # Todo : allow for pauli-twirling in the prep/measure circuits
    #
    
    # Sample random circuit, and find the symplectic matrix / phase vector it implements    
    random_circuit = sample_primitives_circuit(pspec=pspec, length=length, sampler=sampler,
                                     sampler_args=sampler_args, alternatewithlocal = alternatewithlocal, 
                                               localtype = localtype)
    
    sreps = pspec.models['clifford'].get_clifford_symplectic_reps()
    n = pspec.number_of_qubits
    
    s_rc, p_rc = _symp.composite_clifford_from_clifford_circuit(random_circuit,srep_dict=sreps)
    
    if twirled:
        
        s_initial, p_initial = _symp.random_clifford(n)
        s_composite, p_composite = _symp.compose_cliffords(s_initial, p_initial, s_rc, p_rc)
        
        if stabilizer:
            initial_circuit = _comp.stabilizer_state_preparation_circuit(s_initial, p_initial, pspec, 
                                                                         iterations=iterations,
                                                                        relations=relations,
                                                                        pauli_randomize=prep_measure_pauli_randomize,
                                                                        improved_CNOT_compiler=improved_CNOT_compiler,
                                                                        ICC_custom_ordering=ICC_custom_ordering,
                                                                         ICC_std_ordering=ICC_std_ordering,
                                                                        ICC_qubitshuffle=ICC_qubitshuffle)            
        else:
            initial_circuit = _comp.compile_clifford(s_initial, p_initial, pspec, 
                                                           depth_compression=depth_compression, 
                                                           algorithm=algorithm,
                                                           prefix_paulis=True,
                                                             pauli_randomize=prep_measure_pauli_randomize)
        
        
    else:
        s_composite = _copy.deepcopy(s_rc)
        p_composite = _copy.deepcopy(p_rc)
        
    
    s_inverse, p_inverse = _symp.inverse_clifford(s_composite, p_composite)
    
    #
    # Put something here that allows for a random Pauli at the end of the circuit?
    #
    
    if stabilizer:
        inversion_circuit = _comp.stabilizer_measurement_preparation_circuit(s_inverse, p_inverse, pspec, 
                                                                             iterations=iterations,
                                                                            relations=relations,
                                                                             pauli_randomize=prep_measure_pauli_randomize,
                                                                        improved_CNOT_compiler=improved_CNOT_compiler, 
                                                                             ICC_custom_ordering=ICC_custom_ordering,
                                                                         ICC_std_ordering=ICC_std_ordering,
                                                                            ICC_qubitshuffle=ICC_qubitshuffle)   
    else:
        inversion_circuit = _comp.compile_clifford(s_inverse, p_inverse, pspec, 
                                                        depth_compression=depth_compression,
                                                        algorithm=algorithm,
                                                        prefix_paulis=False,
                                                       pauli_randomize=prep_measure_pauli_randomize)

    if not return_partitioned:
        
        if twirled:
            full_circuit = _copy.deepcopy(initial_circuit)
            full_circuit.append_circuit(random_circuit)
            full_circuit.append_circuit(inversion_circuit)
        else:
            full_circuit = _copy.deepcopy(random_circuit)
            full_circuit.append_circuit(inversion_circuit)
         
        full_circuit.done_editing()        
        return full_circuit
    
    else:
        if twirled:
            initial_circuit.done_editing()
            inversion_circuit.done_editing()
            return initial_circuit, random_circuit, inversion_circuit
        else:
            inversion_circuit.done_editing()
            return random_circuit, inversion_circuit
        
#
# Todo: merge below function with the function above, and add two or more wrap-around functions: 
# construct_std_practice_primitive_rb_experiment()
# construct_std_practice_clifford_rb_experiment()
# construct_std_practice_interleaved_clifford_rb_experiment()
# construct_std_practice_primitive_clifford_rb_experiment()
#
def sample_crb_circuit(pspec, length, algorithms=['DGGE','RGGE'],costfunction='2QGC',
                       iterations={'RGGE':4}, depth_compression=True, pauli_randomize=False):

    #sreps = pspec.models['clifford'].get_clifford_symplectic_reps()
    n = pspec.number_of_qubits
       
    s_composite = _np.identity(2*n,int)
    p_composite = _np.zeros((2*n),int)
    
    full_circuit = _cir.Circuit(gatestring=[],num_lines=n)
    
    for i in range(0,length):
    
        s, p = _symp.random_clifford(n)
        circuit = _comp.compile_clifford(s, p, pspec, depth_compression=depth_compression, algorithms=algorithms, 
                                         costfunction=costfunction, iterations=iterations, prefix_paulis=True,
                                        pauli_randomize=pauli_randomize)
        
        # Keeps track of the current composite Clifford
        s_composite, p_composite = _symp.compose_cliffords(s_composite, p_composite, s, p)
        full_circuit.append_circuit(circuit)
        
    
    s_inverse, p_inverse = _symp.inverse_clifford(s_composite, p_composite)
    
    inversion_circuit = _comp.compile_clifford(s_inverse, p_inverse, pspec, depth_compression=depth_compression, 
                                               algorithms=algorithms, costfunction=costfunction, 
                                               iterations=iterations, prefix_paulis=True, 
                                               pauli_randomize=pauli_randomize)
    
    full_circuit.append_circuit(inversion_circuit)
    full_circuit.done_editing()
            
    return full_circuit

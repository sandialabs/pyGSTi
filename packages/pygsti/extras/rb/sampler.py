from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy
from scipy import mod as _mod

from ...algorithms import compileclifford as _comp
from ...objects import circuit as _cir
from ...objects import label as _lbl
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


#
#
#
#
#
#
#
#
#
#
#
#
#
from ... import construction as _cnst
from ... import objects as _objs
from ... import io as _io
from ... import tools as _tools
from . import rbutils as _rbutils
from . import rbobjs as _rbobjs

import itertools as _itertools
import numpy as _np
from numpy import random as _rndm
from scipy.optimize import minimize as _minimize


def create_random_gatestring(m, group_or_gateset, inverse = True,
                             random_pauli = False,
                             interleaved = None, seed=None,
                             group_inverse_only = False,
                             group_prep = False,
                             compilation = None,
                             generated_group = None,
                             gateset_to_group_labels = None,
                             randState=None):
    """
    Makes a random RB sequence.
    
    Parameters
    ----------
    m : int
        The number of random gates in the sequence.

    group_or_gateset : GateSet or MatrixGroup
        Which GateSet of MatrixGroup to create the random sequence for. If
        inverse is true and this is a GateSet, the GateSet gates must form
        a group (so in this case it requires the *target gateset* rather than 
        a noisy gateset). When inverse is true, the MatrixGroup for the gateset 
        is generated. Therefore, if inverse is true and the function is called 
        multiple times, it will be much faster if the MatrixGroup is provided.
        
    inverse: Bool, optional
        If true, the random sequence is followed by its inverse gate. The gateset
        must form a group if this is true. If it is true then the sequence
        returned is length m+1 (2m+1) if interleaved is False (True).
        
    interleaved: Str, optional
        If not None, then a gatelabel string. When a gatelabel string is provided,
        every random gate is followed by this gate. So the returned sequence is of
        length 2m+1 (2m) if inverse is True (False).
        
    group_prep: bool, optional
        If group_inverse_only is True and inverse is True, setting this to true
        creates a "group pre-twirl". Does nothing otherwise (which should be changed
        at some point).

    seed : int, optional
        Seed for random number generator; optional.

    randState : numpy.random.RandomState, optional
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.
    
    Returns
    -------
    Gatestring
        The random gate string of length:
        m if inverse = False, interleaved = None
        m + 1 if inverse = True, interleaved = None
        2m if inverse = False, interleaved not None
        2m + 1 if inverse = True, interleaved not None

    """   
    assert hasattr(group_or_gateset, 'gates') or hasattr(group_or_gateset, 
                   'product'), 'group_or_gateset must be a MatrixGroup of Gateset'    
    group = None
    gateset = None
    if hasattr(group_or_gateset, 'gates'):
        gateset = group_or_gateset
    if hasattr(group_or_gateset, 'product'):
        group = group_or_gateset
        
    if randState is None:
        rndm = _rndm.RandomState(seed) # ok if seed is None
    else:
        rndm = randState
        
    if (inverse) and (not group_inverse_only):
        if gateset:
            group = _rbobjs.MatrixGroup(group_or_gateset.gates.values(),
                                  group_or_gateset.gates.keys() )
                      
        rndm_indices = rndm.randint(0,len(group),m)
        if interleaved:
            interleaved_index = group.label_indices[interleaved]
            interleaved_indices = interleaved_index*_np.ones((m,2),int)
            interleaved_indices[:,0] = rndm_indices
            rndm_indices = interleaved_indices.flatten()
        
        random_string = [ group.labels[i] for i in rndm_indices ]    
        effective_gate = group.product(random_string)
        inv = group.get_inv(effective_gate)
        random_string.append( inv )
        
    if (inverse) and (group_inverse_only):
        assert (gateset is not None), "gateset_or_group should be a GateSet!"
        assert (compilation is not None), "Compilation of group elements to gateset needs to be specified!"
        assert (generated_group is not None), "Generated group needs to be specified!"        
        if gateset_to_group_labels is None:
            gateset_to_group_labels = {}
            for gate in gateset.gates.keys():
                assert(gate in generated_group.labels), "gateset labels are not in \
                the generated group! Specify a gateset_to_group_labels dictionary." 
                gateset_to_group_labels = {'gate':'gate'}
        else:
            for gate in gateset.gates.keys():
                assert(gate in gateset_to_group_labels.keys()), "gateset to group labels \
                are invalid!"              
                assert(gateset_to_group_labels[gate] in generated_group.labels), "gateset to group labels \
                are invalid!"              
                
        rndm_indices = rndm.randint(0,len(gateset.gates.keys()),m)
        if interleaved:
                interleaved_index = gateset.gates.keys().index(interleaved)
                interleaved_indices = interleaved_index*_np.ones((m,2),int)
                interleaved_indices[:,0] = rndm_indices
                rndm_indices = interleaved_indices.flatten()
        # This bit of code is a quick hashed job. Needs to be checked at somepoint
        if group_prep:
            rndm_group_index = rndm.randint(0,len(generated_group))
            prep_random_string = compilation[generated_group.labels[rndm_group_index]]
            prep_random_string_group = [generated_group.labels[rndm_group_index],]

        random_string = [ gateset.gates.keys()[i] for i in rndm_indices ]   
        random_string_group = [ gateset_to_group_labels[gateset.gates.keys()[i]] for i in rndm_indices ] 
        # This bit of code is a quick hashed job. Needs to be checked at somepoint
        if group_prep:
            random_string = prep_random_string + random_string
            random_string_group = prep_random_string_group + random_string_group
        #print(random_string)
        inversion_group_element = generated_group.get_inv(generated_group.product(random_string_group))
        
        # This bit of code is a quick hash job, and only works when the group is the 1-qubit Cliffords
        if random_pauli:
            pauli_keys = ['Gc0','Gc3','Gc6','Gc9']
            rndm_index = rndm.randint(0,4)
            
            if rndm_index == 0 or rndm_index == 3:
                bitflip = False
            else:
                bitflip = True
            inversion_group_element = generated_group.product([inversion_group_element,pauli_keys[rndm_index]])
            
        inversion_sequence = compilation[inversion_group_element]
        #print(inversion_sequence)
        random_string.extend(inversion_sequence)
        #print(random_string)
        
    if not inverse:
        if gateset:
            rndm_indices = rndm.randint(0,len(gateset.gates.keys()),m)
            gateLabels = list(gateset.gates.keys())
            if interleaved:
                interleaved_index = gateLabels.index(interleaved)
                interleaved_indices = interleaved_index*_np.ones((m,2),int)
                interleaved_indices[:,0] = rndm_indices
                rndm_indices = interleaved_indices.flatten()           
            random_string = [gateLabels[i] for i in rndm_indices ]
            
        else:
            rndm_indices = rndm.randint(0,len(group),m)
            if interleaved:
                interleaved_index = group.label_indices[interleaved]
                interleaved_indices = interleaved_index*_np.ones((m,2),int)
                interleaved_indices[:,0] = rndm_indices
                rndm_indices = interleaved_indices.flatten()
            random_string = [ group.labels[i] for i in rndm_indices ] 
    
    if not random_pauli:
        return _objs.GateString(random_string)
    if random_pauli:
        return _objs.GateString(random_string), bitflip

def create_random_gatestrings(m_list, K_m, group_or_gateset, inverse=True, 
                              interleaved = None, alias_maps=None, seed=None, 
                              randState=None):
    """
    Makes a list of random RB sequences.
    
    Parameters
    ----------
    m_list : list or array of ints
        The set of lengths for the random sequences (with the total
        number of Cliffords in each sequence given by m_list + 1). Minimal
        allowed length is therefore 1 (a random CLifford followed by its 
        inverse).

    clifford_group : MatrixGroup
        Which Clifford group to use.

    K_m : int or dict
        If an integer, the fixed number of Clifford sequences to be sampled at
        each length m.  If a dictionary, then a mapping from Clifford
        sequence length m to number of Cliffords to be sampled at that length.
    
    alias_maps : dict of dicts, optional
        If not None, a dictionary whose keys name other gate-label-sets, e.g.
        "primitive" or "canonical", and whose values are "alias" dictionaries 
        which map the clifford labels (defined by `clifford_group`) to those
        of the corresponding gate-label-set.  For example, the key "canonical"
        might correspond to a dictionary "clifford_to_canonical" for which 
        (as one example) clifford_to_canonical['Gc1'] == ('Gy_pi2','Gy_pi2').
            
    seed : int, optional
        Seed for random number generator; optional.

    randState : numpy.random.RandomState, optional
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.
    
    Returns
    -------
    dict or list
        If `alias_maps` is not None, a dictionary of lists-of-gatestring-lists
        whose keys are 'clifford' and all of the keys of `alias_maps` (if any).
        Values are lists of `GateString` lists, one for each K_m value.  If
        `alias_maps` is None, then just the list-of-lists corresponding to the 
        clifford gate labels is returned.
    """

    if randState is None:
        rndm = _rndm.RandomState(seed) # ok if seed is None
    else:
        rndm = randState
        
    assert hasattr(group_or_gateset, 'gates') or hasattr(group_or_gateset, 
           'product'), 'group_or_gateset must be a MatrixGroup or Gateset'
    
    
    if inverse:
        if hasattr(group_or_gateset, 'gates'):
            group_or_gateset = _rbobjs.MatrixGroup(group_or_gateset.gates.values(),
                                  group_or_gateset.gates.keys())
    if isinstance(K_m,int):
        K_m_dict = {m : K_m for m in m_list }
    else: K_m_dict = K_m
    assert hasattr(K_m_dict, 'keys'),'K_m must be a dict or int!'

    string_lists = {'uncompiled': []} # GateStrings with uncompiled labels
    if alias_maps is not None:
        for gstyp in alias_maps.keys(): string_lists[gstyp] = []

    for m in m_list:
        K = K_m_dict[m]
        strs_for_this_m = [ create_random_gatestring(m, group_or_gateset,
            inverse=inverse,interleaved=interleaved,randState=rndm) for i in range(K) ]
        string_lists['uncompiled'].append(strs_for_this_m)
        if alias_maps is not None:
            for gstyp,alias_map in alias_maps.items(): 
                string_lists[gstyp].append(
                    _cnst.translate_gatestring_list(strs_for_this_m,alias_map))

    if alias_maps is None:
        return string_lists['uncompiled'] #only list of lists is uncompiled one
    else:
        return string_lists #note we also return this if alias_maps == {}

def create_random_interleaved_gatestrings(m_list, K_m, group_or_gateset, interleaved_list,
                                          inverse=True, alias_maps=None):
    
    # Currently no random number generator seed allowed, as needs to have different seed for each
    # call of create_random_gatestrings().
    all_random_string_lists = {}
    alias_maps_mod = {} if (alias_maps is None) else alias_maps      
    random_string_lists = create_random_gatestrings(m_list, K_m, 
                          group_or_gateset,inverse,interleaved = None, 
                          alias_maps = alias_maps_mod,)

    if alias_maps is None: 
        all_random_string_lists['baseline'] = random_string_lists['uncompiled']
    else:
        all_random_string_lists['baseline'] = random_string_lists
        
    for interleaved in interleaved_list:
        random_string_lists = \
                       create_random_gatestrings(m_list, K_m, group_or_gateset,inverse,
                                  interleaved = interleaved, alias_maps = alias_maps_mod)

        if alias_maps is None: 
            all_random_string_lists[interleaved] = random_string_lists['uncompiled']
        else:
            all_random_string_lists[interleaved] = random_string_lists
            
        return all_random_string_lists          

def write_empty_rb_files(filename, m_list, K_m, group_or_gateset, 
                         inverse=True, interleaved_list=None, alias_maps=None, 
                         seed=None, randState=None):
    """
    A wrapper for list_random_rb_clifford_strings which also writes output
    to disk.

    This function returns the same value as list_random_rb_clifford_strings,
    and also:

    - saves the clifford strings in an empty data set file by adding ".txt"
      to `filename`.
    - saves each set of strings to a gatestring list text file by adding
      "_<gate-label-set-name>.txt" to `filename`.  
      
    For example, if "primitive" is the only key of `alias_maps`, and 
    `filename` is set to "test", then the following files are created:

    - "test.txt" (empty dataset with clifford-labelled strings)
    - "test_clifford.txt" (gate string list with clifford-label strings)
    - "test_primitive.txt" (gate string list with primitive-label strings)

    Parameters
    ----------
    filename : str
        The base name of the files to create (see above).

    m_min : integer
        Smallest desired Clifford sequence length.
    
    m_max : integer
        Largest desired Clifford sequence length.
    
    Delta_m : integer
        Desired Clifford sequence length increment.

    clifford_group : MatrixGroup
        Which Clifford group to use.

    K_m_sched : int or dict
        If an integer, the fixed number of Clifford sequences to be sampled at
        each length m.  If a dictionary, then a mapping from Clifford
        sequence length m to number of Cliffords to be sampled at that length.
    
    alias_maps : dict of dicts, optional
        If not None, a dictionary whose keys name other gate-label-sets, e.g.
        "primitive" or "canonical", and whose values are "alias" dictionaries 
        which map the clifford labels (defined by `clifford_group`) to those
        of the corresponding gate-label-set.  For example, the key "canonical"
        might correspond to a dictionary "clifford_to_canonical" for which 
        (as one example) clifford_to_canonical['Gc1'] == ('Gy_pi2','Gy_pi2').
            
    seed : int, optional
        Seed for random number generator; optional.

    randState : numpy.random.RandomState, optional
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.
    

    Returns
    -------
    dict or list
        If `alias_maps` is not None, a dictionary of lists-of-gatestring-lists
        whose keys are 'clifford' and all of the keys of `alias_maps` (if any).
        Values are lists of `GateString` lists, one for each K_m value.  If
        `alias_maps` is None, then just the list-of-lists corresponding to the 
        clifford gate labels is returned.
    """
    base_filename = filename
    if interleaved_list is not None:
        base_filename = filename+'_baseline' 
        
    # line below ensures random_string_lists is *always* a dictionary
    alias_maps_mod = {} if (alias_maps is None) else alias_maps      
    random_string_lists = \
        create_random_gatestrings(m_list, K_m, group_or_gateset,inverse,
                                  interleaved = None, alias_maps = alias_maps_mod, 
                                  seed=seed, randState=randState)
    #always write uncompiled gates to empty dataset (in future have this be an arg?)
    _io.write_empty_dataset(base_filename+'.txt', list(
            _itertools.chain(*random_string_lists['uncompiled'])))
    for gstyp,strLists in random_string_lists.items():
        _io.write_gatestring_list(base_filename +'_%s.txt' % gstyp,
                                  list(_itertools.chain(*strLists)))
        
    if interleaved_list is None:
        if alias_maps is None: 
            return random_string_lists['uncompiled'] 
            #mimic list_random_rb_clifford_strings return value
        else: return random_string_lists
        
    else:
        all_random_string_lists = {}
        if alias_maps is None: 
            all_random_string_lists['baseline'] = random_string_lists['uncompiled']
        else:
            all_random_string_lists['baseline'] = random_string_lists
        
        for interleaved in interleaved_list:
            # No seed allowed here currently, as currently no way to make it different to
            # the seed for the baseline decay
            filename_interleaved = filename+'_interleaved_'+interleaved
            random_string_lists = \
                       create_random_gatestrings(m_list, K_m, group_or_gateset,inverse,
                                  interleaved = interleaved, alias_maps = alias_maps_mod)
            _io.write_empty_dataset(filename_interleaved+'.txt', list(
                _itertools.chain(*random_string_lists['uncompiled'])))
            for gstyp,strLists in random_string_lists.items():
                _io.write_gatestring_list(filename_interleaved +'_%s.txt' % gstyp,
                                  list(_itertools.chain(*strLists)))
                
            if alias_maps is None: 
                all_random_string_lists[interleaved] = random_string_lists['uncompiled']
            else:
                all_random_string_lists[interleaved] = random_string_lists
            
        return all_random_string_lists   

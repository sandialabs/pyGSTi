""" RB circuit sampling functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from ...algorithms import compileclifford as _comp
from ...objects import circuit as _cir
from ...objects import label as _lbl
from ...tools import symplectic as _symp

#
# ?????? #
from ... import construction as _cnst
from ... import objects as _objs
from ... import io as _io
from ... import tools as _tools


import itertools as _itertools
from numpy import random as _rndm

import numpy as _np
import copy as _copy
from scipy import mod as _mod

def std_practice_direct_rb_experiment():
    return

def std_practice_clifford_rb_experiment():
    return

def std_practice_interleaved_direct_rb_experiment():
    return

def std_practice_interleaved_clifford_rb_experiment():
    return

def circuit_layer_by_pairings(pspec, twoQprob=0.5, oneQgatenames='all', twoQgatenames='all',
                              gatesetname = 'clifford'):   
    """
    Samples a random circuit layer by pairing up qubits and picking a two-qubit gate for a pair
    with the specificed probability. This sampler *assumes* all-to-all connectivity, and does
    not check that this condition is satisfied (more generally, it assumes that all gates can be
    applied in parallel in any combination that would be well-defined). 
    
    The sampler works as follows: If there are an odd number of qubits, one qubit is chosen at 
    random to have a uniformly random 1-qubit gate applied to it (from all possible 1-qubit gates,
    or those in `oneQgatenames` if not None). Then, the remaining qubits are paired up, uniformly 
    at random. A uniformly random 2-qubit gate is then chosen for a pair with probability `twoQprob`
    (from all possible 2-qubit gates, or those in `twoQgatenames` if not None). If a 2-qubit gate 
    is not chosen to act on a pair, then each qubit is independently and uniformly randomly assigned
    a 1-qubit gate.
    
    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit layer is being sampled for. This 
       function assumes all-to-all connectivity, but does not check this is satisfied.

    twoQprob : float, optional
        A probability for a two-qubit gate to be applied to a pair of qubits. So, the expected
        number of 2-qubit gates in the sampled layer is twoQprob*floor(n/2).
        
    oneQgatenames : 'all' or list, optional
        If not 'all', a list of the names of the 1-qubit gates to be sampled from when applying 
        a 1-qubit gate to a qubit. If this is 'all', the full set of 1-qubit gate names is extracted
        from the ProcessorSpec.
        
    twoQgatenames : 'all' or list, optional
        If not 'all', a list of the names of the 2-qubit gates to be sampled from when applying 
        a 2-qubit gate to a pair of qubits. If this is 'all', the full set of 2-qubit gate names is 
        extracted from the ProcessorSpec.
        
    gatesetname : str, optional
        Only used if oneQgatenames or twoQgatenames is None. Specifies the which of the
        `pspec.models` to use to extract the gateset. The `clifford` default is suitable 
        for Clifford or direct RB, but will not use any non-Clifford gates in the gateset.
        
    Returns
    -------
    list of Labels
        A list of gate Labels that defines a "complete" circuit layer (there is one and only 
        one gate acting on each qubit).
        
    """
    n = pspec.number_of_qubits
    
    # If the one qubit and two qubit gate names are only specified as 'all', construct them.
    if (oneQgatenames is 'all') or (twoQgatenames is 'all'):    
        if oneQgatenames is 'all':
            oneQgatenames = []
        if twoQgatenames is 'all':    
            twoQgatenames = []
        
        gatenames = list(pspec.models[gatesetname].gates.keys())
        for gate in pspec.models[gatesetname].gates:
            if oneQgatenames is 'all':
                if gate.number_of_qubits == 1 and gate.name not in oneQgatenames:
                    oneQgatenames.append(gate.name)
            if twoQgatenames is 'all': 
                if gate.number_of_qubits == 2 and gate.name not in twoQgatenames:
                    twoQgatenames.append(gate.name)
    
    # Basic variables required for sampling the circuit layer.
    qubits = list(range(n))
    sampled_layer = []
    num_oneQgatenames = len(oneQgatenames)
    num_twoQgatenames = len(twoQgatenames)
    
    # If there is an odd number of qubits, begin by picking one to have a 1-qubit gate.
    if n % 2 != 0:
        q = qubits[np.random.randint(0,n)]
        name = oneQgatenames[_np.random.randint(0,num_oneQgatenames)]
        del qubits[q]       
        sampled_layer.append(_lbl.Label(name,q))
    
    # Go through n//2 times until all qubits have been paired up and gates on them sampled
    for i in range(n//2):
        
        # Pick two of the remaining qubits : each qubit that is picked is deleted from the list.
        index = _np.random.randint(0,len(qubits))
        q1 = qubits[index]
        del qubits[index] 
        index = _np.random.randint(0,len(qubits))
        q2 = qubits[index]
        del qubits[index] 
        
        # Flip a coin to decide whether to act a two-qubit gate on that qubit
        if _np.random.binomial(1,twoQprob) == 1:
            # If there is more than one two-qubit gate on the pair, pick a uniformly random one.
            name = twoQgatenames[_np.random.randint(0,num_twoQgatenames)]
            sampled_layer.append(_lbl.Label(name,(q1,q2)))
        else:
            # Independently, pick uniformly random 1-qubit gates to apply to each qubit.
            name1 = oneQgatenames[_np.random.randint(0,num_oneQgatenames)]
            name2 = oneQgatenames[_np.random.randint(0,num_oneQgatenames)]
            sampled_layer.append(_lbl.Label(name1,q1))
            sampled_layer.append(_lbl.Label(name2,q2))                     
    
    return sampled_layer

def circuit_layer_by_Qelimination(pspec, twoQprob=0.5, oneQgates='all', twoQgates='all',
                                  gatesetname='clifford'):
    """
    Samples a random circuit layer by eliminating qubits one by one. This sampler works
    with any connectivity, but the expected number of 2-qubit gates in a layer depends
    on both the specified 2-qubit gate probability and the exact connectivity graph. 
    
    This sampler is the following algorithm: List all the qubits, and repeat the 
    following steps until all qubits are deleted from this list. 1) Uniformly at random 
    pick a qubit from the list, and delete it from the list 2) Flip a coin with  bias 
    `twoQprob` to be "Heads". 3) If "Heads" then -- if there is one or more 2-qubit gates
    from this qubit to other qubits still in the list -- pick one of these at random. 
    4) If we haven't chosen a 2-qubit gate for this qubit ("Tails" or "Heads" but there 
    are no possible 2-qubit gates) then pick a uniformly random 1-qubit gate to apply to
    this qubit.
    
    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit layer is being sampled for.

    twoQprob : float or None, optional
        None or a probability for a two-qubit gate to be applied to a pair of qubits. If
        None, sampling is uniform over all gates available on a qubit. If a float, if a
        2-qubit can is still possible on a qubit at that stage of the sampling, this is
        the probability a 2-qubit gate is chosen for that qubit. The expected number of
        2-qubit gates per layer depend on this quantity and the connectivity graph of
        the device.
        
    oneQgates : 'all' or list, optional
        If not 'all', a list of the 1-qubit gates to sample from, in the form of Label 
        objects. This is *not* just gate names (e.g. "Gh"), but Labels each containing 
        the gate name and the qubit it acts on. So it is possible to specify different 
        1-qubit gatesets on different qubits. If this is 'all', the full set of possible
        1-qubit gates is extracted from the ProcessorSpec.
        
    twoQgates : 'all' or list, optional
        If not 'all', a list of the 2-qubit gates to sample from, in the form of Label 
        objects. This is *not* just gate names (e.g. "Gcnot"), but Labels each containing
        the gate name and the qubits it acts on. If this is 'all', the full set of possible
        2-qubit gates is extracted from the ProcessorSpec.
        
    gatesetname : str, optional
        Only used if oneQgatenames or twoQgatenames is None. Specifies the which of the
        `pspec.models` to use to extract the gateset. The `clifford` default is suitable 
        for Clifford or direct RB, but will not use any non-Clifford gates in the gateset.
               
    Returns
    -------
    list of gates
        A list of gate Labels that defines a "complete" circuit layer (there is one and 
        only one gate acting on each qubit).
        
    """   
    # If oneQgates is not specified, extract this list from the ProcessorSpec
    if oneQgates != None:
        oneQgates_available = _copy.copy(oneQgates)    
    else:
        oneQgates_available = list(pspec.models[gatesetname].gates.keys())
        d = len(oneQgates_available)    
        for i in range(0,d):
            if oneQgates_available[d-1-i].number_of_qubits != 1:
                del oneQgates_available[d-1-i]
    
    # If twoQgates is not specified, extract this list from the ProcessorSpec
    if twoQgates != None:
        twoQgates_available = _copy.copy(twoQgates)    
    else:
        twoQgates_available = list(pspec.models[gatesetname].gates.keys())
        d = len(twoQgates_available)                                      
        for i in range(0,d):
            if twoQgates_available[d-1-i].number_of_qubits != 2:
                del twoQgates_available[d-1-i]
  
    # If the `twoQprob` is not None, we specify a weighting towards 2-qubit gates
    if twoQprob != None: 
        weighting = [1-twoQprob,twoQprob]
        
    # Prep the sampling variables.
    sampled_layer = [] 
    n = pspec.number_of_qubits
    remaining_qubits = list(_np.arange(0,pspec.number_of_qubits))
    num_qubits_used = 0
    
    # Go through until all qubits have been assigned a gate.
    while num_qubits_used < n:
               
        # Pick a random qubit
        r = _np.random.randint(0,n-num_qubits_used)
        q = remaining_qubits[r]
        del remaining_qubits[r]
        
        # Find the 1Q gates that act on q.
        oneQgates_remaining_on_q = []
        ll = len(oneQgates_available)
        for i in range(0,ll):
            if q in oneQgates_available[ll-1-i].qubits:
                oneQgates_remaining_on_q.append(oneQgates_available[ll-1-i])
                del oneQgates_available[ll-1-i]
        
        # Find the 2Q gates that act on q and a remaining qubit.       
        twoQgates_remaining_on_q = []
        ll = len(twoQgates_available)
        for i in range(0,ll):
            if q in twoQgates_available[ll-1-i].qubits:
                twoQgates_remaining_on_q.append(twoQgates_available[ll-1-i])
                del twoQgates_available[ll-1-i]
                
        # If twoQprob is None, there is no weighting towards 2-qubit gates.
        if twoQprob is None:
            nrm = len(oneQgates_remaining_on_q)+len(twoQgates_remaining_on_q)
            weighting = [len(oneQgates_remaining_on_q)/nrm,len(twoQgates_remaining_on_q)/nrm]
         
        # Decide whether to to implement a 2-qubit gate or a 1-qubit gate.
        if len(twoQgates_remaining_on_q) == 0:
            xx = 1
        else:
            xx = _np.random.choice([1,2],p=weighting)
        
        # Implement a 1-qubit gate on qubit q.
        if xx == 1:
            # Sample the gate
            r = _np.random.randint(0,len(oneQgates_remaining_on_q))
            sampled_layer.append(oneQgates_remaining_on_q[r])
            # We have assigned gates to 1 of the remaining qubits.
            num_qubits_used += 1
        
        # Implement a 2-qubit gate on qubit q.    
        if xx == 2:
            # Sample the gate
            r = _np.random.randint(0,len(twoQgates_remaining_on_q))
            sampled_layer.append(twoQgates_remaining_on_q[r])
            
            # Find the index of the other qubit in the sampled gate.
            other_qubit = twoQgates_remaining_on_q[r].qubits[0]
            if other_qubit == q:
                other_qubit = twoQgates_remaining_on_q[r].qubits[1]
            
            # Delete the gates on this other qubit from the 1-qubit gate list.
            ll = len(oneQgates_available)
            for i in range(0,ll):
                if other_qubit in oneQgates_available[ll-1-i].qubits:                       
                    del oneQgates_available[ll-1-i]
            
            # Delete the gates on this other qubit from the 2-qubit gate list.        
            ll = len(twoQgates_available)
            for i in range(0,ll):
                if other_qubit in twoQgates_available[ll-1-i].qubits:                       
                    del twoQgates_available[ll-1-i]
            
            # Delete this other qubit from remaining qubits list.                 
            del remaining_qubits[remaining_qubits.index(other_qubit)]
            
            # We have assigned gates to 2 of the remaining qubits.
            num_qubits_used += 2
    
    return sampled_layer

def circuit_layer_by_sectors(pspec, sectors, sectorsprob='uniform', twoQprob=1.0, 
                             oneQgatenames='all', gatesetname='clifford'):
    """
    Samples a random circuit layer using the 2-qubit gate "sectors" specified. The 
    `sectors` variable should be a list of lists containing 2-qubit gates, in the
    form of gate Label objects, that can be applied in parallel (the empty list is 
    allowed). E.g., on 4 qubits with linear connectivity a valid `sectors` list is
    [[],[Label(Gcnot,(0,1)),Label(Gcnot,(2,3))]] consisting of a element containing
    zero 2-qubit gates and an element containing two 1-qubit gates that can be
    applied in parallel.
   
    Using this sampler, a circuit layer is sampled according to the algorithm:
    
    1) Pick an item (a list) from the list `sectors`, according to the distribution
    over `sectors` specified by `sectorsprob`.
    2) For each 2-qubit gate in the chosen list, apply the gate with probability
    `twoQprob`.
    3) Uniformly at random, sample 1-qubit gates to apply to all qubits that do
    not yet have a gate assigned to them, from the set specified by `oneQgatenames`.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit layer is being sampled for.
       
    sectors : list of lists of Labels
        A list of lists of 2-qubit gate Labels. Each list in `sectors` should 
        contain 2-qubit gates, in the form of Labels, that can be applied in parallel. 
        The sampler then picks one of these "sectors", and converts this into a circuit
        layer by applying the 2-qubit gates it contains with a user-specified probability
        and augmenting these with 1-qubit gate (see above).
        
    sectorsprob : str or list of floats
        If a list, they are unnormalized probabilities to sample each of the sectors. So it
        is a list of non-negative floats of the same length as `sectors`. If 'uniform', then 
        the uniform distribution over the sectors is used.

    twoQprob : float, optional
        The probability for each two-qubit gate to be applied to a pair of qubits, after a
        set of 2-qubit gates (a "sector") has been chosen. The expected number of 2-qubit
        gates in a layer is `twoQprob` times the expected number of 2-qubit gates in a
        sector sampled according to `sectorsprob`.
                
    oneQgatenames : 'all' or list of strs, optional
        If not 'all', a list of the names of the 1-qubit gates to be sampled from when applying 
        a 1-qubit gate to a qubit. If this is 'all', the full set of 1-qubit gate names is 
        extracted from the ProcessorSpec.
        
    gatesetname : str, optional
        Only used if oneQgatenames is 'all'. Specifies which of the `pspec.models` to use to 
        extract the gateset. The `clifford` default is suitable for Clifford or direct RB,
        but will not use any non-Clifford gates in the gateset.
               
    Returns
    -------
    list of gates
        A list of gate Labels that defines a "complete" circuit layer (there is one and 
        only one gate acting on each qubit).
        
    """
    assert(gatesetname == 'clifford'), "This function currently assumes sampling from a Clifford gateset!"
    # Pick the sector.
    if sectorsprob == 'uniform':
        twoqubitgates = sectors[_np.random.randint(0,len(sectors))]            
    else:
        sectorsprob = sectorsprob/np.sum(sectorsprob)
        x = list(np.random.multinomial(1,sectorsprob))
        twoqubitgates = sectors[x.index(1)]
    
    # Prep the sampling variables
    sampled_layer = []
    remaining_qubits = list(_np.arange(0,pspec.number_of_qubits))
    
    # Go through the 2-qubit gates in the sector, and apply each one with probability twoQprob
    for i in range(0,len(twoqubitgates)):
        if _np.random.binomial(1,twoQprob) == 1:
            gate = twoqubitgates[i]
            sampled_layer.append(gate)
            # Delete the qubits that have been assigned a gate.
            del remaining_qubits[remaining_qubits.index(gate.qubits[0])]
            del remaining_qubits[remaining_qubits.index(gate.qubits[1])]
    
    # Go through the qubits which don't have a 2-qubit gate assigned to them, and pick a 1-qubit gate        
    for i in range(0,len(remaining_qubits)):
        
        qubit = remaining_qubits[i] 
        
        # If the 1-qubit gate names are specified, use these.
        if oneQgatenames != 'all':
            possiblegates = [_lbl(name,(qubit,)) for name in oneQgatenames]
        
        # If the 1-qubit gate names are not specified, find the available 1-qubit gates
        else:
            if gatesetname == 'clifford':
                possiblegates = pspec.clifford_gates_on_qubits[(qubit,)]
            else:
                possiblegates = pspec.models[gatesetname].gates()
                l = len(possiblegates)
                for j in range(0,l):
                    if possiblegates[l-j].number_of_qubits != 1:
                        del possiblegates[l-j]
                    else:
                        if possiblegates[l-j].qubits[0] != qubit:
                            del possiblegates[l-j]
       
        gate = possiblegates[_np.random.randint(0,len(possiblegates))]
        sampled_layer.append(gate)

    return sampled_layer

def circuit_layer_of_1Q_gates(pspec, oneQgatenames='all', gatesetname='clifford'):
    """
    Samples a random circuit layer containing only 1-qubit gates. The allowed
    1-qubit gates are specified by `oneQgatenames`, and the 1-qubit gate is
    sampled independently and uniformly.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit layer is being sampled for.
                
    oneQgatenames : 'all' or list of strs, optional
        If not 'all', a list of the names of the 1-qubit gates to be sampled from when applying 
        a 1-qubit gate to a qubit. If this is 'all', the full set of 1-qubit gate names is 
        extracted from the ProcessorSpec.
        
    gatesetname : str, optional
        Only used if oneQgatenames is 'all'. Specifies which of the `pspec.models` to use to 
        extract the gateset. The `clifford` default is suitable for Clifford or direct RB,
        but will not use any non-Clifford gates in the gateset.
               
    Returns
    -------
    list of gates
        A list of gate Labels that defines a "complete" circuit layer (there is one and 
        only one gate acting on each qubit).
        
    """
    sampled_layer = []
    
    if oneQgatenames == 'all':
        if gatesetname == 'clifford':
            for i in range(0,pspec.number_of_qubits):
                try:
                    gate = pspec.clifford_gates_on_qubits[(i,)][_np.random.randint(0,len(pspec.clifford_gates_on_qubits[(i,)]))]
                    sampled_layer.append(gate)
                except:
                    raise ValueError ("There are no 1Q Clifford gates on qubit {}!".format(i))
        else:
            raise ValueError("Currently, 'gatesetname' must be 'clifford'")
    
    else:     
        for i in range(0,pspec.number_of_qubits):
            sampled_gatename = oneQgatenames[_np.random.randint(0,4)]
            sampled_layer.append(_lbl.Label(sampled_gatename,i))

    return sampled_layer

def circuit(pspec, length, sampler='weights', sampler_args=[0.5,'all',None,'single'],
                              alternatewithlocal = False, localtype = 'primitives'):
    
    if sampler == 'weights':
        twoQprob = sampler_args[0]
        oneQgatenames = sampler_args[1]
        
        gates1q = list(pspec.models['clifford'].gates.keys())
        gates2q = list(pspec.models['clifford'].gates.keys())
        d = len(gates1q)   
        for i in range(0,d):
            
            if oneQgatenames == 'all':
                if gates1q[d-1-i].number_of_qubits != 1:
                    del gates1q[d-1-i]
            else:
                if gates1q[d-1-i].name not in oneQgatenames:
                    del gates1q[d-1-i]
                    
            if gates2q[d-1-i].number_of_qubits != 2:
                 del gates2q[d-1-i]
        
    elif sampler == 'sectors':
        
        twoqubitprob = sampler_args[0]
        oneQgatenames = sampler_args[1]
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
    
    
        twoQprob= sampler_args[0]
        #
        # Todo: this currently doesn't use a list of one-qubit gates specified for
        # sampling over. That should probably be fixed.
        #
        oneQgatenames = []
        twoQgatenames = []

        gatenames = list(pspec.models['clifford'].gates.keys())

        for gate in pspec.models['clifford'].gates:

            if gate.number_of_qubits == 1 and gate.name not in oneQgatenames:
                oneQgatenames.append(gate.name)
            if gate.number_of_qubits == 2 and gate.name not in twoQgatenames:
                twoQgatenames.append(gate.name)
                       
    circuit = _cir.Circuit(gatestring=[],num_lines=pspec.number_of_qubits)
    
    if not alternatewithlocal:
        for i in range(0,length):
  
            if sampler == 'weights':       
                        
                layer = sample_circuit_layer_by_2Qweighting(pspec, oneQgates=gates1q, twoQgates=gates2q, 
                                                            twoQprob=twoQprob)
            
            elif sampler == 'sectors':
                layer = sample_circuit_layer_by_sectors(pspec, sectors=sectors, twoQprob=twoqubitprob,
                                                       oneQgatenames=oneQgatenames)
                
            elif sampler == 'pairing':
                layer = sample_circuit_layer_by_pairings(pspec, oneQgatenames=oneQgatenames,
                                                         twoQgatenames=twoQgatenames,
                                                         twoQprob=twoQprob)
                
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
                        layer = sample_circuit_layer_by_2Qweighting(pspec,twoQprob=twoQprob)
            
                    elif sampler == 'sectors':
                        layer = sample_circuit_layer_by_sectors(pspec, sectors=sectors, twoQprob=twoqubitprob)
                    elif sampler == 'pairings':
                        assert(False), "This funcationality is not yet included!"
                    
                    else:
                        layer = sampler(pspec, length, sampler_args)
                        
                circuit.insert_layer(layer,0)
    
    circuit.done_editing()
    return circuit


def direct_rb_circuit(pspec, length, sampler='weights',sampler_args=[0.5,'all',None,'single'],  
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
        
def clifford_rb_circuit(pspec, length, algorithms=['DGGE','RGGE'],costfunction='2QGC',
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



def oneQ_rb_sequence(m, group_or_gateset, inverse=True, random_pauli=False, interleaved=None, 
                     group_inverse_only=False, group_prep=False, compilation=None,
                     generated_group=None, gateset_to_group_labels=None, seed=None, randState=None):
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

def oneQ_rb_experiment(m_list, K_m, group_or_gateset, inverse=True, 
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
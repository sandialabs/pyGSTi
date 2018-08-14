""" RB circuit sampling functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from ...algorithms import compilers as _cmpl
from ...objects import circuit as _cir
from ...baseobjs import label as _lbl
from ...tools import symplectic as _symp
from ... import construction as _cnst
from ... import objects as _objs
from ... import io as _io
from ... import tools as _tools
from ...tools import compattools as _compat

import numpy as _np
import copy as _copy

def circuit_layer_by_pairing_qubits(pspec, subsetQs=None, twoQprob=0.5, oneQgatenames='all', 
                                    twoQgatenames='all', gatesetname='clifford'):   
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
       function assumes all-to-all connectivity, but does not check this is satisfied. Unless
       `subsetQs` is not None, a circuit layer is sampled over all the qubits in `pspec`.

    subsetQs : list, optional
        If not None, a list of the qubits to sample the circuit layer for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit layer is sampled to acton all the qubits 
        in `pspec`.

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
        Only used if oneQgatenames or twoQgatenames is None. Specifies which of the
        `pspec.models` to use to extract the gate-set. The `clifford` default is suitable 
        for Clifford or direct RB, but will not use any non-Clifford gates in the gate-set.
        
    Returns
    -------
    list of Labels
        A list of gate Labels that defines a "complete" circuit layer (there is one and only 
        one gate acting on each qubit in `pspec` or `subsetQs`).       
    """
    if subsetQs is None: n = pspec.number_of_qubits
    else: n = len(subsetQs)
    
    # If the one qubit and/or two qubit gate names are only specified as 'all', construct them.
    if (oneQgatenames == 'all') or (twoQgatenames == 'all'):    
        if oneQgatenames == 'all':
            oneQpopulate = True
            oneQgatenames = []
        else:
            oneQpopulate = False
        if twoQgatenames == 'all':   
            twoQpopulate = True
            twoQgatenames = []
        else:
            twoQpopulate = False
        
        gatelist = list(pspec.models[gatesetname].gates.keys())
        for gate in gatelist:
            if oneQpopulate:
                if (gate.number_of_qubits == 1) and (gate.name not in oneQgatenames):
                    oneQgatenames.append(gate.name)
            if twoQpopulate: 
                if (gate.number_of_qubits == 2) and (gate.name not in twoQgatenames):
                    twoQgatenames.append(gate.name)
    
    # Basic variables required for sampling the circuit layer.
    if subsetQs is None:
        qubits = pspec.qubit_labels[:] # copy this list
    else:
        qubits = subsetQs[:] # copy this list
    sampled_layer = []
    num_oneQgatenames = len(oneQgatenames)
    num_twoQgatenames = len(twoQgatenames)
    
    # If there is an odd number of qubits, begin by picking one to have a 1-qubit gate.
    if n % 2 != 0:
        q = qubits[_np.random.randint(0,n)]
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

def circuit_layer_by_Qelimination(pspec, subsetQs=None, twoQprob=0.5, oneQgates='all',
                                 twoQgates='all', gatesetname='clifford'):
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
       The ProcessorSpec for the device that the circuit layer is being sampled for. Unless
       `subsetQs` is not None, a circuit layer is sampled over all the qubits in `pspec`.

    subsetQs : list, optional
        If not None, a list of the qubits to sample the circuit layer for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit layer is sampled to acton all the qubits 
        in `pspec`.

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
        only one gate acting on each qubit in `pspec` or `subsetQs`).        
    """
    if subsetQs is None:
        n = pspec.number_of_qubits
        qubits = pspec.qubit_labels[:] # copy this list
    else:
        n = len(subsetQs)   
        qubits = subsetQs[:] # copy this list

    # If oneQgates is specified, use the given list.
    if oneQgates != 'all':
        oneQgates_available = _copy.copy(oneQgates)    
    # If oneQgates is not specified, extract this list from the ProcessorSpec
    else:
        oneQgates_available = list(pspec.models[gatesetname].gates.keys())
        d = len(oneQgates_available)    
        for i in range(0,d):
            # If it's not a 1-qubit gate, we delete it.
            if oneQgates_available[d-1-i].number_of_qubits != 1:
                del oneQgates_available[d-1-i]
            # If it's not a gate on the allowed qubits, we delete it.
            elif oneQgates_available[d-1-i].qubits[0] not in qubits:
                del oneQgates_available[d-1-i]
    
    # If twoQgates is specified, use the given list.
    if twoQgates != 'all':
        twoQgates_available = _copy.copy(twoQgates)    
    # If twoQgates is not specified, extract this list from the ProcessorSpec
    else:
        twoQgates_available = list(pspec.models[gatesetname].gates.keys())
        d = len(twoQgates_available)                                      
        for i in range(0,d):
            # If it's not a 2-qubit gate, we delete it.
            if twoQgates_available[d-1-i].number_of_qubits != 2:
                del twoQgates_available[d-1-i]
             # If it's not a gate on the allowed qubits, we delete it.
            elif not set(twoQgates_available[d-1-i].qubits).issubset(set(qubits)):
                del twoQgates_available[d-1-i]
  
    # If the `twoQprob` is not None, we specify a weighting towards 2-qubit gates
    if twoQprob != None: 
        weighting = [1-twoQprob,twoQprob]
        
    # Prep the sampling variables.
    sampled_layer = [] 
    remaining_qubits = _copy.deepcopy(qubits)
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
            
            # Find the label of the other qubit in the sampled gate.
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

def circuit_layer_by_co2Qgates(pspec, subsetQs, co2Qgates, co2Qgatesprob='uniform', twoQprob=1.0, 
                            oneQgatenames='all', gatesetname='clifford'):
    """
    Samples a random circuit layer using the specified list of "compatible two-qubit gates"
    (co2Qgates). That is, the user inputs a list (`co2Qgates`) specifying 2-qubit gates that are
    "compatible" -- meaning that they can be implemented simulatenously -- and a distribution 
    over the different compatible sets, and a layer is sampled from this via:

    1. Pick a set of compatible two-qubit gates from the list `co2Qgates`, according to the
    distribution specified by `co2Qgatesprob`.
    2. For each 2-qubit gate in the chosen set of compatible gates, with probability `twoQprob`
    add this gate to the layer.
    3. Uniformly sample 1-qubit gates for any qubits that don't yet have a gate on them,
    from those 1-qubit gates specified by `oneQgatenames`.

    For example, consider 4 qubits with linear connectivity. a valid `co2Qgates` list is
    co2Qgates = [[,],[Label(Gcphase,(0,1)),Label(Gcphase,(2,3))]] which consists of an 
    element containing zero 2-qubit gates and an element containing  two 2-qubit gates 
    that can be applied in parallel. In this example there are 5 possible sets of compatible 
    2-qubit gates:
    
    1. [,] (zero 2-qubit gates)
    2. [Label(Gcphase,(0,1)),] (one of the three 2-qubit gate)
    3. [Label(Gcphase,(1,2)),] (one of the three 2-qubit gate)
    4. [Label(Gcphase,(2,3)),] (one of the three 2-qubit gate)
    5. [Label(Gcphase,(0,1)), Label(Gcphase,(2,3)),] (the only compatible pair of 2-qubit gates).

    The list of compatible two-qubit gates `co2Qgates` can be any list containing anywhere
    from 1 to all 5 of these lists.

    In order to allow for convenient sampling of some commonly useful distributions, 
    `co2Qgates` can be a list of lists of lists of compatible 2-qubit gates ("nested" sampling). 
    In this case, a list of lists of compatible 2-qubit gates is picked according to the distribution 
    `co2Qgatesprob`, and then one of the sublists of compatible 2-qubit gates in the selected list is 
    then chosen uniformly at random. For example, this is useful for sampling a layer containing one
    uniformly random 2-qubit gate with probability p and a layer of 1-qubit gates with probability
    1-p. Here, we can specify `co2Qgates` as [[],[[the 1st 2Q-gate,],[the 2nd 2Q-gate,], ...]] and
    set `twoQprob=1` and `co2Qgatesprob  = [1-p,p].

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit layer is being sampled for. Unless
       `subsetQs` is not None, a circuit layer is sampled over all the qubits in `pspec`.

    subsetQs : list
        If not None, a list of the qubits to sample the circuit layer for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit layer is sampled to act on all the qubits 
        in `pspec`.
       
    co2Qgates : list
        This is either:

            1. A list of lists of 2-qubit gate Labels that can be applied in parallel.
            2. A list of lists of lists of 2-qubit gate Labels that can be applied in parallel.

        In case (1) each list in `co2Qgates` should contain 2-qubit gates, in the form of Labels, 
        that can be applied in parallel and act only on the qubits in `pspec` if `subsetQs` is None, 
        or act only on the qubits in  `subsetQs` if `subsetQs` is not None.  The sampler then picks 
        one of these compatible sets of gates (with probability specified by `co2Qgatesprob`, and converts 
        this into a circuit layer by applying the 2-qubit gates it contains with the user-specified 
        probability `twoQprob`, and augmenting these 2-qubit gates with 1-qubit gates on all other qubits.

        In case (2) a sublist of lists is sampled from `co2Qgates` according to `co2Qgatesprob` and then we
        proceed as in case (1) but as though `co2Qgatesprob` is the uniform distribution.
        
    co2Qgatesprob : str or list of floats
        If a list, they are unnormalized probabilities to sample each of the elements of `co2Qgates`. So it
        is a list of non-negative floats of the same length as `co2Qgates`. If 'uniform', then the uniform 
        distribution is used.

    twoQprob : float, optional
        The probability for each two-qubit gate to be applied to a pair of qubits, after a
        set of compatible 2-qubit gates has been chosen. The expected number of 2-qubit
        gates in a layer is `twoQprob` times the expected number of 2-qubit gates in a
        set of compatible 2-qubit gates sampled according to `co2Qgatesprob`.
                
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
    if _compat.isstr(co2Qgatesprob):
        assert(co2Qgatesprob == 'uniform'), "If `co2Qgatesprob` is a string it must be 'uniform!'"
        twoqubitgates_or_nestedco2Qgates = co2Qgates[_np.random.randint(0,len(co2Qgates))]            
    else:
        co2Qgatesprob = _np.array(co2Qgatesprob)/_np.sum(co2Qgatesprob)
        x = list(_np.random.multinomial(1,co2Qgatesprob))
        twoqubitgates_or_nestedco2Qgates = co2Qgates[x.index(1)]
    
    # The special case where the selected co2Qgates contains no gates or co2Qgates.
    if len(twoqubitgates_or_nestedco2Qgates) == 0:
          twoqubitgates = twoqubitgates_or_nestedco2Qgates
    # If it's a nested sector, sample uniformly from the nested co2Qgates.
    elif type(twoqubitgates_or_nestedco2Qgates[0]) == list:
        twoqubitgates = twoqubitgates_or_nestedco2Qgates[_np.random.randint(0,len(twoqubitgates_or_nestedco2Qgates))]
    # If it's not a list of "co2Qgates" (lists) then this is the list of gates to use.
    else:
        twoqubitgates = twoqubitgates_or_nestedco2Qgates
        
    # Prep the sampling variables
    sampled_layer = []
    if subsetQs is not None:    
        remaining_qubits = subsetQs[:] # copy this list
    else:
        remaining_qubits = pspec.qubit_labels[:] # copy this list
    
    # Go through the 2-qubit gates in the sector, and apply each one with probability twoQprob
    for i in range(0,len(twoqubitgates)):
        if _np.random.binomial(1,twoQprob) == 1:
            gate = twoqubitgates[i]
            # If it's a nested co2Qgates:
            sampled_layer.append(gate)
            # Delete the qubits that have been assigned a gate.
            del remaining_qubits[remaining_qubits.index(gate.qubits[0])]
            del remaining_qubits[remaining_qubits.index(gate.qubits[1])]
    
    # Go through the qubits which don't have a 2-qubit gate assigned to them, and pick a 1-qubit gate        
    for i in range(0,len(remaining_qubits)):
        
        qubit = remaining_qubits[i] 
        
        # If the 1-qubit gate names are specified, use these.
        if oneQgatenames != 'all':
            possiblegates = [_lbl.Label(name,(qubit,)) for name in oneQgatenames]
        
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

def circuit_layer_of_oneQgates(pspec, subsetQs=None, oneQgatenames='all', pdist='uniform', 
                              gatesetname='clifford'):
    """
    Samples a random circuit layer containing only 1-qubit gates. The allowed
    1-qubit gates are specified by `oneQgatenames`, and the 1-qubit gates are
    sampled independently and uniformly.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit layer is being sampled for. Unless
       `subsetQs` is not None, a circuit layer is sampled over all the qubits in `pspec`.

    subsetQs : list, optional
        If not None, a list of the qubits to sample the circuit layer for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit layer is sampled to acton all the qubits 
        in `pspec`.
                
    oneQgatenames : 'all' or list of strs, optional
        If not 'all', a list of the names of the 1-qubit gates to be sampled from when applying 
        a 1-qubit gate to a qubit. If this is 'all', the full set of 1-qubit gate names is 
        extracted from the ProcessorSpec.
        
    pdist : 'uniform' or list of floats, optional
        If a list, they are unnormalized probabilities to sample each of the 1-qubit gates
        in the list `oneQgatenames`. If this is not 'uniform', then oneQgatename` must not
        be 'all' (it must be a list so that it is unambigious which probability correpsonds
        to which gate). So if not 'uniform', `pdist` is a list of non-negative floats of the 
        same length as `oneQgatenames`. If 'uniform', then the uniform distribution over 
        the gates is used.
        
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
    if subsetQs is not None: qubits = subsetQs[:] # copy this list
    else: qubits = pspec.qubit_labels[:] # copy this list

    sampled_layer = []
    
    if _compat.isstr(pdist): assert(pdist == 'uniform'), "If pdist is not a list or numpy.array it must be 'uniform'"
    
    if oneQgatenames == 'all':
        assert(pdist == 'uniform'), "If `oneQgatenames` = 'all', pdist must be 'uniform'"
        if gatesetname == 'clifford':
            for i in qubits:
                try:
                    gate = pspec.clifford_gates_on_qubits[(i,)][_np.random.randint(0,len(pspec.clifford_gates_on_qubits[(i,)]))]
                    sampled_layer.append(gate)
                except:
                    raise ValueError ("There are no 1Q Clifford gates on qubit {}!".format(i))
        else: raise ValueError("Currently, 'gatesetname' must be 'clifford'")
    
    else:
        # A basic check for the validity of pdist.
        if not _compat.isstr(pdist): assert(len(pdist) == len(oneQgatenames)), "The pdist probability distribution is invalid!"
        
        # Find out how many 1-qubit gate names there are
        num_oneQgatenames = len(oneQgatenames)
        
        # Sample a gate for each qubit.
        for i in qubits:
            
            # If 'uniform', then sample according to the uniform dist.
            if _compat.isstr(pdist): sampled_gatename = oneQgatenames[_np.random.randint(0,num_oneQgatenames)]
            # If not 'uniform', then sample according to the user-specified dist.
            else:
                pdist = _np.array(pdist)/sum(pdist)
                x = list(_np.random.multinomial(1,pdist))
                sampled_gatename = oneQgatenames[x.index(1)]
            # Add sampled gate to the layer.
            sampled_layer.append(_lbl.Label(sampled_gatename,i))

    return sampled_layer

def random_circuit(pspec, length, subsetQs=None, sampler='Qelimination', samplerargs=[], addlocal = False, lsargs=[]):
    """
    Samples a random circuit of the specified length (or ~ twice this length), using layers 
    independently sampled according to the specified sampling distribution.
    
    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit is being sampled for. This is always
       handed to the sampler, as the first argument of the sampler function. Unless
       `subsetQs` is not None, the circuit is sampled over all the qubits in `pspec`.
                 
    length : int
        If `addlocal` is False, this is the length of the sampled circuit. If `addlocal is
        True the length of the circuits is 2*length+1 with odd-indexed layers sampled according
        to the sampler specified by `sampler`, and the the zeroth layer + the even-indexed 
        layers consisting of random 1-qubit gates (with the sampling specified by `lsargs`)

    subsetQs : list, optional
        If not None, a list of the qubits to sample the circuit for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit is sampled to act on all the qubits 
        in `pspec`.

    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function 
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates. If this is a
        function, it should be a function that takes as the first argument a ProcessorSpec, and
        returns a random circuit layer as a list of gate Label objects. Note that the default
        'Qelimination' is not necessarily the most useful in-built sampler, but it is the only
        sampler that requires no parameters beyond the ProcessorSpec *and* works for arbitrary 
        connectivity devices. See the docstrings for each of these samplers for more information.
    
    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec` and `samplerargs` lists the
        remaining arguments handed to the sampler. For some in-built samplers this is not
        optional.
        
    addlocal : bool, optional
        If False, the circuit sampled is of length `length` and each layer is independently
        sampled according to the sampler specified by `sampler`. If True, the circuit sampled
        is of length 2*`length`+1 where: the zeroth + all even layers are consisting of 
        independently random 1-qubit gates (with the sampling specified by `lsargs`); the 
        odd-indexed layers are independently sampled according to `sampler`. So `length`+1 
        layers consist only of 1-qubit gates, and `length` layers are sampled according to 
        `sampler`.
        
    lsargs : list, optional
        A list of arguments that are handed to the 1-qubit gate layers sampler 
        rb.sampler.circuit_layer_of_oneQgates for the alternating 1-qubit-only layers that are
        included in the circuit if `addlocal` is True. This argument is not used if `addlocal`
        is false. Note that `pspec` is used as the first, and only required, argument of
        rb.sampler.circuit_layer_of_oneQgates. If `lsargs` = [] then all available 1-qubit gates
        are uniformly sampled from. To uniformly sample from only a subset of the available
        1-qubit gates (e.g., the Paulis to Pauli-frame-randomize) then `lsargs` should be a
        1-element list consisting of a list of the relevant gate names (e.g., `lsargs` = ['Gi,
        'Gxpi, 'Gypi', 'Gzpi']).
        
    Returns
    -------
    Circuit
        A random circuit of length `length` (if not addlocal) or length 2*`length`+1 (if addlocal)
        with layers independently sampled using the specified sampling distribution.        
    """ 
    if _compat.isstr(sampler):
        
        if sampler == 'pairingQs': sampler = circuit_layer_by_pairing_qubits
        elif sampler == 'Qelimination': sampler = circuit_layer_by_Qelimination
        elif sampler == 'co2Qgates':
            sampler = circuit_layer_by_co2Qgates
            assert(len(samplerargs) >= 1), "The samplerargs must at least a 1-element list with the first element the 'co2Qgates' argument of the co2Qgates sampler."
        elif sampler == 'local': sampler = circuit_layer_of_oneQgates            
        else: raise ValueError("Sampler type not understood!")

    if subsetQs is not None:
        qubits = subsetQs[:] # copy this list
    else:
        qubits = pspec.qubit_labels[:] # copy this list
    
    # If we can, we use the identity in the pspec.
    if pspec.identity is not None:
        identity = pspec.identity
    else:
        identity = 'I'
    # Initialize an empty circuit, to populate with sampled layers.
    circuit = _cir.Circuit(gatestring=[], line_labels=qubits, identity=identity)
    
    # If we are not add layers of random local gates between the layers, sample 'length' layers
    # according to the sampler `sampler`.
    if not addlocal:
        for i in range(0,length):
            layer = sampler(pspec,subsetQs,*samplerargs)
            circuit.insert_layer(layer,0)
            
    # If we are adding layers of random local gates between the layers.
    if addlocal:
        for i in range(0,2*length+1):
                local = not bool(i % 2)
                # For odd layers, we uniformly sample the specified type of local gates.
                if local:
                    layer = circuit_layer_of_oneQgates(pspec,subsetQs,*lsargs)                 
                # For even layers, we sample according to the given distribution
                else:
                    layer = sampler(pspec,subsetQs,*samplerargs)
                circuit.insert_layer(layer,0)
    
    return circuit

def direct_rb_circuit(pspec, length, subsetQs=None, sampler='Qelimination', samplerargs=[], addlocal=False, lsargs=[],
                      randomizeout=True, cliffordtwirl=True, conditionaltwirl=True, citerations=20,
                      compilerargs=[], partitioned=False):
    """
    Generates a "direct randomized benchmarking" (DRB) circuit, which is the protocol introduced in
    arXiv:1807.07975 (2018). The length of the "core" sequence is given by `length` and may be any 
    integer >= 0. An n-qubit DRB circuit consists of (1) a circuit the prepares a uniformly random 
    stabilizer state; (2) a length-l circuit (specified by `length`) consisting of circuit layers sampled 
    according to some user-specified distribution (specified by `sampler`), (3) a circuit that maps the 
    output of the preceeding circuit to a computational basis state. See arXiv:1807.07975 (2018) for further
    details.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit is being sampled for, which defines the 
       "native" gate-set and the connectivity of the device. The returned DRB circuit will be over
       the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
       is always handed to the sampler, as the first argument of the sampler function (this is only
       of importance when not using an in-built sampler for the "core" of the DRB circuit). Unless
       `subsetQs` is not None, the circuit is sampled over all the qubits in `pspec`.
                         
    length : int
        The "direct RB length" of the circuit, which is closely related to the circuit depth. It
        must be an integer >= 0. Unless `addlocal` is True, it is the depth of the "core" random 
        circuit, sampled according to `sampler`, specified in step (2) above. If `addlocal` is True, 
        each layer in the "core" circuit sampled according to "sampler` is followed by a layer of 
        1-qubit gates, with sampling specified by `lsargs` (and the first layer is proceeded by a 
        layer of 1-qubit gates), and so the circuit of step (2) is length 2*`length` + 1.

    subsetQs : list, optional
        If not None, a list of the qubits to sample the circuit for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit is sampled to act on all the qubits 
        in `pspec`.
        
    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function 
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not 
        a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function]. 
        If `sampler` is a function, it should be a function that takes as the first argument a 
        ProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that 
        the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is 
        the only sampler that requires no parameters beyond the ProcessorSpec *and* works for arbitrary 
        connectivity devices. See the docstrings for each of these samplers for more information.
    
    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec`, the second argument is `subsetQs`,
        and `samplerargs` lists the remaining arguments handed to the sampler. This is not
        optional for some choices of `sampler`.

    addlocal : bool, optional
        Whether to follow each layer in the "core" circuit, sampled according to `sampler` with
        a layer of 1-qubit gates.

    lsargs : list, optional
        Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
        layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.
        
    randomizeout : bool, optional
        If False, the ideal output of the circuit (the "success" or "survival" outcome) is the all-zeros
        bit string. If True, the ideal output of the circuit is randomized to a uniformly random bit-string. 
        This setting is useful for, e.g., detecting leakage/loss/measurement-bias etc.

    cliffordtwirl : bool, optional
        Wether to begin the circuit with a sequence that generates a random stabilizer state. For
        standard DRB this should be set to True. There are a variety of reasons why it is better
        to have this set to True.

    conditionaltwirl : bool, optional
        DRB only requires that the initial/final sequences of step (1) and (3) create/measure
        a uniformly random / particular stabilizer state, rather than implement a particular unitary. 
        step (1) and (3) can be achieved by implementing a uniformly random Clifford gate and the
        unique inversion Clifford, respectively. This is implemented if `conditionaltwirl` is False.
        However, steps (1) and (3) can be implemented much more efficiently than this: the sequences
        of (1) and (3) only need to map a particular input state to a particular output state, 
        if `conditionaltwirl` is True this more efficient option is chosen -- this is option corresponds 
        to "standard" DRB. (the term "conditional" refers to the fact that in this case we essentially
        implementing a particular Clifford conditional on a known input).

    citerations : int, optional
        Some of the stabilizer state / Clifford compilation algorithms in pyGSTi (including the default 
        algorithms) are  randomized, and the lowest-cost circuit is chosen from all the circuit generated 
        in the iterations of the algorithm. This is the number of iterations used. The time required to
        generate a DRB circuit is linear in `citerations`. Lower-depth / lower 2-qubit gate count
        compilations of steps (1) and (3) are important in order to successfully implement DRB on as many
        qubits as possible. 
    
    compilerargs : list, optional
        A list of arguments that are handed to the compile_stabilier_state/measurement()functions (or the
        compile_clifford() function if `conditionaltwirl `is False). This includes all the optional 
        arguments of these functions *after* the `iterations` option (set by `citerations`). For most
        purposes the default options will be suitable (or at least near-optimal from the compilation methods
        in-built into pyGSTi). See the docstrings of these functions for more information. 
        
    partitioned : bool, optional
        If False, a single circuit is returned consisting of the full circuit. If True, three circuits
        are returned in a list consisting of: (1) the stabilizer-prep circuit, (2) the core random circuit, 
        (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1) 
        and then (3) to (1).
        
    Returns
    -------
    Circuit or list of Circuits
        If partioned is False, a random DRB circuit sampled as specified. If partioned is True, a list of
        three circuits consisting of (1) the stabilizer-prep circuit, (2) the core random circuit, 
        (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1) 
        and then (3) to (1) [except in the case of cliffordtwirl=False, when it is a list of two circuits].    

    Tuple 
        A length-n tuple of integers in [0,1], corresponding to the error-free outcome of the 
        circuit. Always all zeros if `randomizeout` is False. The ith element of the tuple 
        corresponds to the error-free outcome for the qubit labelled by: the ith element of
        `subsetQs`, if `subsetQs` is not None; the ith element of `pspec.qubit_labels`, otherwise.
        In both cases, the ith element of the tuple corresponds to the error-free outcome for the
        qubit on the ith wire of the output circuit.
    """    
    if subsetQs is not None: n = len(subsetQs)
    else: n = pspec.number_of_qubits
    # Sample a random circuit of "native gates".   
    circuit = random_circuit(pspec=pspec, length=length, subsetQs=subsetQs, sampler=sampler, 
                             samplerargs=samplerargs, addlocal=addlocal, lsargs=lsargs)   
    # find the symplectic matrix / phase vector this "native gates" circuit implements.
    s_rc, p_rc = _symp.symplectic_rep_of_clifford_circuit(circuit,pspec=pspec)
    
    # If we are clifford twirling, we do an initial random circuit that is either a uniformly random
    # cliffor or creates a uniformly random stabilizer state from the standard input.
    if cliffordtwirl:
        # Sample a uniformly random Clifford.
        s_initial, p_initial = _symp.random_clifford(n)
        # Find the composite action of this uniformly random clifford and the random circuit.
        s_composite, p_composite = _symp.compose_cliffords(s_initial, p_initial, s_rc, p_rc)
        # If conditionaltwirl we do a stabilizer prep (a conditional Clifford).
        if conditionaltwirl:
            initial_circuit = _cmpl.compile_stabilizer_state(s_initial, p_initial, pspec, subsetQs, citerations, 
                                                           *compilerargs)           
        # If not conditionaltwirl, we do a full random Clifford.
        else: initial_circuit = _cmpl.compile_clifford(s_initial, p_initial, pspec, subsetQs, citerations, 
                                                       *compilerargs)        
    # If we are not Clifford twirling, we just copy the effect of the random circuit as the effect
    # of the "composite" prep + random circuit (as here the prep circuit is the null circuit).
    else:
        s_composite = _copy.deepcopy(s_rc)
        p_composite = _copy.deepcopy(p_rc)
    
    if conditionaltwirl:
        # If we want to randomize the expected output then randomize the p vector, otherwise
        # it is left as p. Note that, unlike with compile_clifford, we don't invert (s,p)
        # before handing it to the stabilizer measurement function.
        if randomizeout: p_for_measurement = _symp.random_phase_vector(s_composite,n)
        else: p_for_measurement =  p_composite
        inversion_circuit = _cmpl.compile_stabilizer_measurement(s_composite, p_for_measurement, pspec, subsetQs,
                                                                 citerations, *compilerargs)   
    else:
        # Find the Clifford that inverts the circuit so far. We 
        s_inverse, p_inverse = _symp.inverse_clifford(s_composite, p_composite)
        # If we want to randomize the expected output then randomize the p_inverse vector, otherwise
        # do not.
        if randomizeout: p_for_inversion = _symp.random_phase_vector(s_inverse,n)
        else: p_for_inversion =  p_inverse
        # Compile the Clifford.
        inversion_circuit = _cmpl.compile_clifford(s_inverse, p_for_inversion, pspec, subsetQs, 
                                                   citerations, *compilerargs)        
    if cliffordtwirl:
        full_circuit = _copy.deepcopy(initial_circuit)
        full_circuit.append_circuit(circuit)
        full_circuit.append_circuit(inversion_circuit)
    else:
        full_circuit = _copy.deepcopy(circuit)
        full_circuit.append_circuit(inversion_circuit)         
     
    # Find the expected outcome of the circuit.
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(full_circuit,pspec=pspec)
    if conditionaltwirl: # s_out is not always the identity with a conditional twirl, only conditional on prep/measure.
        assert(_np.array_equal(s_out[:n,n:],_np.zeros((n,n),int))), "Compiler has failed!"
    else: assert(_np.array_equal(s_out,_np.identity(2*n,int))), "Compiler has failed!"

    # Find the ideal output of the circuit.
    s_inputstate, p_inputstate = _symp.prep_stabilizer_state(n, zvals=None)
    s_outstate, p_outstate = _symp.apply_clifford_to_stabilizer_state(s_out, p_out, s_inputstate, p_inputstate)
    idealout = []
    for q in range(0,n):
        measurement_out = _symp.pauli_z_measurement(s_outstate, p_outstate, q)
        bit = measurement_out[1]
        assert(bit == 0 or bit == 1), "Ideal output is not a computational basis state!"
        if not randomizeout:
            assert(bit == 0), "Ideal output is not the all 0s computational basis state!"
        idealout.append(int(measurement_out[1]))
    idealout = tuple(idealout)

    if not partitioned: outcircuit = full_circuit
    else:
        if cliffordtwirl: outcircuit = [initial_circuit, circuit, inversion_circuit]
        else: outcircuit = [circuit, inversion_circuit]

    return outcircuit, idealout

def direct_rb_experiment(pspec, lengths, circuits_per_length, subsetQs=None, sampler='Qelimination', samplerargs=[], addlocal=False, lsargs=[],
                         randomizeout=False, cliffordtwirl=True, conditionaltwirl=True, citerations=20, compilerargs=[], 
                         partitioned=False, descriptor='A DRB experiment', verbosity=1):
    """
    Generates a "direct randomized benchmarking" (DRB) experiments, which is the protocol introduced in
    arXiv:1807.07975 (2018). The set of lengths of the "core" sequence is given by `lengths` and may be 
    a list of any distinct integers >= 0.
    
    An n-qubit DRB circuit consists of (1) a circuit the prepares a uniformly random stabilizer state;
    (2) a length-l circuit (specified by `length`) consisting of circuit layers sampled according to 
    some user-specified distribution (specified by `sampler`), (3) a circuit that maps the output of
    the preceeding circuit to a computational basis state. See arXiv:1807.07975 (2018) for further
    details.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit is being sampled for, which defines the 
       "native" gate-set and the connectivity of the device. The returned DRB circuit will be over
       the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
       is always handed to the sampler, as the first argument of the sampler function (this is only
       of importance when not using an in-built sampler for the "core" of the DRB circuit). Unless
       `subsetQs` is not None, the circuit is sampled over all the qubits in `pspec`.
                         
    lengths : int
        The set of "direct RB lengths" for the circuits. The DRB lengths must be integers >= 0. 
        Unless `addlocal` is True, the DRB length is the depth of the "core" random circuit, 
        sampled according to `sampler`, specified in step (2) above. If `addlocal` is True, 
        each layer in the "core" circuit sampled according to "sampler` is followed by a layer of 
        1-qubit gates, with sampling specified by `lsargs` (and the first layer is proceeded by a 
        layer of 1-qubit gates), and so the circuit of step (2) is length 2*`length` + 1.

    circuits_per_length : int
        The number of (possibly) different DRB circuits sampled at each length.

    subsetQs : list, optional
        If not None, a list of the qubits to sample the circuit for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit is sampled to act on all the qubits 
        in `pspec`.
        
    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function 
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not 
        a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function]. 
        If `sampler` is a function, it should be a function that takes as the first argument a 
        ProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
        the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is the
        only sampler that requires no parameters beyond the ProcessorSpec *and* works for arbitrary 
        connectivity devices. See the docstrings for each of these samplers for more information.
    
    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec`, the second argument is `subsetQs`,
        and `samplerargs` lists the remaining arguments handed to the sampler. This is not
        optional for some choices of `sampler`.

    addlocal : bool, optional
        Whether to follow each layer in the "core" circuits, sampled according to `sampler` with
        a layer of 1-qubit gates.

    lsargs : list, optional
        Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
        layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.
        
    randomizeout : bool, optional
        If False, the ideal output of the circuits (the "success" or "survival" outcome) is the all-zeros
        bit string. If True, the ideal output of each circuit is randomized to a uniformly random bit-string. 
        This setting is useful for, e.g., detecting leakage/loss/measurement-bias etc.

    cliffordtwirl : bool, optional
        Wether to begin the circuitas with a sequence that generates a random stabilizer state. For
        standard DRB this should be set to True. There are a variety of reasons why it is better
        to have this set to True.

    conditionaltwirl : bool, optional
        DRB only requires that the initial/final sequences of step (1) and (3) create/measure
        a uniformly random / particular stabilizer state, rather than implement a particular unitary. 
        step (1) and (3) can be achieved by implementing a uniformly random Clifford gate and the
        unique inversion Clifford, respectively. This is implemented if `conditionaltwirl` is False.
        However, steps (1) and (3) can be implemented much more efficiently than this: the sequences
        of (1) and (3) only need to map a particular input state to a particular output state, 
        if `conditionaltwirl` is True this more efficient option is chosen -- this is option corresponds 
        to "standard" DRB. (the term "conditional" refers to the fact that in this case we essentially
        implementing a particular Clifford conditional on a known input).

    citerations : int, optional
        Some of the stabilizer state / Clifford compilation algorithms in pyGSTi (including the default 
        algorithms) are  randomized, and the lowest-cost circuit is chosen from all the circuits generated 
        in the iterations of the algorithm. This is the number of iterations used. The time required to
        generate a DRB circuit is linear in `citerations`. Lower-depth / lower 2-qubit gate count
        compilations of steps (1) and (3) are important in order to successfully implement DRB on as many
        qubits as possible. 
    
    compilerargs : list, optional
        A list of arguments that are handed to the compile_stabilier_state/measurement()functions (or the
        compile_clifford() function if `conditionaltwirl `is False). This includes all the optional 
        arguments of these functions *after* the `iterations` option (set by `citerations`). For most
        purposes the default options will be suitable (or at least near-optimal from the compilation methods
        in-built into pyGSTi). See the docstrings of these functions for more information. 
        
    partitioned : bool, optional
        If False, each circuit is returned as a single full circuit. If True, each circuit is returned as
        a list of three circuits consisting of: (1) the stabilizer-prep circuit, (2) the core random circuit, 
        (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1) 
        and then (3) to (1).

    descriptor : str, optional
        A description of the experiment being generated. Stored in the output dictionary.

    verbosity : int, optional
        If > 0 the number of circuits generated so far is shown.
        
    Returns
    -------
    Circuit or list of Circuits
        If partioned is False, a random DRB circuit sampled as specified. If partioned is True, a list of
        three circuits consisting of (1) the stabilizer-prep circuit, (2) the core random circuit, 
        (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1) 
        and then (3) to (1).     

    Tuple 
        A length-n tuple of integers in [0,1], corresponding to the error-free outcome of the 
        circuit. Always all zeros if `randomizeout` is False. The ith element of the tuple 
        corresponds to the error-free outcome for the qubit labelled by: the ith element of
        `subsetQs`, if `subsetQs` is not None; the ith element of `pspec.qubit_labels`, otherwise.
        In both cases, the ith element of the tuple corresponds to the error-free outcome for the
        qubit on the ith wire of the output circuit.
    
    Returns
    -------
    dict
        A dictionary containing the generated RB circuits, the error-free outputs of the circuit,
        and the specification used to generate the circuits. The keys are:

        - 'circuits'. A dictionary of the sampled circuits. The circuit with key(l,k) is the kth circuit 
        at DRB length l.

        - 'idealout'. A dictionary of the error-free outputs of the circuits as tuples. The tuple with 
        key(l,k) is the error-free output of the (l,k) circuit. The ith element of this tuple corresponds
        to the error-free outcome for the qubit on the ith wire of the output circuit and/or the ith element
        of the list at the key 'qubitordering'. These tuples will all be (0,0,0,...) when `randomizeout` is
        False

        - 'qubitordering'. The ordering of the qubits in the 'idealout' tuples.

        - 'spec'. A dictionary containing all of the parameters handed to this function, except `pspec`.
        This then specifies how the circuits where generated.
    """    

    experiment_dict = {}
    experiment_dict['spec'] = {}
    experiment_dict['spec']['subsetQs'] = subsetQs
    experiment_dict['spec']['sampler'] = sampler
    experiment_dict['spec']['samplerargs'] = samplerargs
    experiment_dict['spec']['addlocal'] = addlocal
    experiment_dict['spec']['lsargs'] = lsargs
    experiment_dict['spec']['randomizeout'] = randomizeout
    experiment_dict['spec']['cliffordtwirl'] = cliffordtwirl
    experiment_dict['spec']['conditionaltwirl'] = conditionaltwirl
    experiment_dict['spec']['citerations'] = citerations
    experiment_dict['spec']['compilerargs'] = compilerargs
    experiment_dict['spec']['partitioned'] = partitioned
    experiment_dict['spec']['descriptor'] = descriptor

    if subsetQs is not None: experiment_dict['qubitordering'] = tuple(subsetQs)
    else: experiment_dict['qubitordering'] = tuple(pspec.qubit_labels)
    
    experiment_dict['circuits'] = {}
    experiment_dict['idealout'] = {}

    for lnum, l in enumerate(lengths):
        if verbosity > 0:
            print('- Sampling {} circuits at DRB length {} ({} of {} lengths)'.format(circuits_per_length,l,lnum+1,len(lengths)))
            print('  - Number of circuits sampled = ',end='')
        for j in range(circuits_per_length):            
            circuit, idealout = direct_rb_circuit(pspec, l, subsetQs=subsetQs, sampler=sampler, samplerargs=samplerargs, 
                                                  addlocal=addlocal, lsargs=lsargs, randomizeout=randomizeout, 
                                                  cliffordtwirl=cliffordtwirl, conditionaltwirl=conditionaltwirl, 
                                                  citerations=citerations, compilerargs=compilerargs, partitioned=partitioned)
            experiment_dict['circuits'][l,j] = circuit
            experiment_dict['idealout'][l,j] = idealout
            if verbosity > 0: print(j+1,end=',')
        if verbosity >0: print('')

    return experiment_dict

def clifford_rb_circuit(pspec, length, subsetQs=None, randomizeout=False, citerations=20, compilerargs=[]):
    """
    Generates a "Clifford randomized benchmarking" (CRB) circuit, which is the current-standard
    RB protocol defined in "Scalable and robust randomized benchmarking of quantum processes", 
    Magesan et al. PRL 106 180504 (2011). This consists of a sequence of `length`+1 uniformly random
    n-qubit Clifford gates followed by the unique inversion Clifford, with all the Cliffords compiled 
    into the "native" gates of a device as specified by `pspec`. The circuit output by this function will
    respect the connectivity of the device, as encoded into `pspec` (see the ProcessorSpec object docstring
    for how to construct the relevant `pspec`).

    Note the convention that the the output Circuit consists of `length+2` Clifford gates, rather than the 
    more usual convention of defining the "CRB length" to be the number of Clifford gates - 1. This is for 
    consistency with the other RB functions in pyGSTi: in all RB-circuit-generating functions in pyGSTi 
    length zero corresponds to the minimum-length circuit allowed by the protocol. Note that changing the 
    "RB lengths" by a constant additive factor is irrelevant for fitting purposes (except that it changes
    the obtained "SPAM" fit parameter). 

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit is being sampled for, which defines the 
       "native" gate-set and the connectivity of the device. The returned CRB circuit will be over
       the gates in `pspec`, and will respect the connectivity encoded by `pspec`.
                
    length : int
        The "CRB length" of the circuit -- an integer >= 0 --  which is the number of Cliffords in the
        circuit - 2 *before* each Clifford is compiled into the native gate-set. 

    subsetQs : list, optional
        If not None, a list of the qubits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
        irrelevant. If desired, a circuit that explicitly idles on the other qubits can be obtained
        by using methods of the Circuit object.
        
    randomizeout : bool, optional
        If False, the ideal output of the circuit (the "success" or "survival" outcome) is the all-zeros
        bit string. This is probably considered to be the "standard" in CRB. If True, the ideal output
        of the circuit is randomized to a uniformly random bit-string. This setting is useful for, e.g.,
        detecting leakage/loss/measurement-bias etc.

    citerations : int, optional
        Some of the Clifford compilation algorithms in pyGSTi (including the default algorithm) are 
        randomized, and the lowest-cost circuit is chosen from all the circuit generated in the
        iterations of the algorithm. This is the number of iterations used. The time required to
        generate a CRB circuit is linear in `citerations` * (`length`+2). Lower-depth / lower 2-qubit
        gate count compilations of the Cliffords are important in order to successfully implement
        CRB on more qubits. 
    
    compilerargs : list, optional
        A list of arguments that are handed to compile_clifford() function, which includes all the
        optional arguments of compile_clifford() *after* the `iterations` option (set by `citerations`).
        In order, this list should be values for:
            - algorithm : str. A string that specifies the compilation algorithm. The default in
                compile_clifford() will always be whatever we consider to be the 'best' all-round
                algorith,
            - aargs : list. A list of optional arguments for the particular compilation algorithm.
            - costfunction : 'str' or function. The cost-function from which the "best" compilation
                for a Clifford is chosen from all `citerations` compilations. The default costs a
                circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
            - prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
            - paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a 
                random Pauli on each qubit (compiled into native gates). I.e., if this is True the
                native gates are Pauli-randomized. When True, this prevents any coherent errors adding
                (on average) inside the layers of each compiled Clifford, at the cost of increased
                circuit depth. Defaults to False.
        For more information on these options, see the compile_clifford() docstring.

    Returns
    -------
    Circuit
        A random CRB circuit over the "native" gate-set specified. 

    Tuple 
        A length-n tuple of integers in [0,1], corresponding to the error-free outcome of the 
        circuit. Always all zeros if `randomizeout` is False. The ith element of the tuple 
        corresponds to the error-free outcome for the qubit labelled by: the ith element of
        `subsetQs`, if `subsetQs` is not None; the ith element of `pspec.qubit_labels`, otherwise.
        In both cases, the ith element of the tuple corresponds to the error-free outcome for the
        qubit on the ith wire of the output circuit.
    """
    # Find the labels of the qubits to create the circuit for.
    if subsetQs is not None: qubits = subsetQs[:] # copy this list
    else: qubits = pspec.qubit_labels[:] # copy this list
    # The number of qubits the circuit is over.
    n = len(qubits)
      
    # Initialize the identity circuit rep.    
    s_composite = _np.identity(2*n,int)
    p_composite = _np.zeros((2*n),int)
    # Initialize an empty circuit
    full_circuit = _cir.Circuit(gatestring=[],line_labels=qubits)
    
    # Sample length+1 uniformly random Cliffords (we want a circuit of length+2 Cliffords, in total), compile 
    # them, and append them to the current circuit.
    for i in range(0,length+1):
        
        s, p = _symp.random_clifford(n)
        circuit = _cmpl.compile_clifford(s, p, pspec, subsetQs=subsetQs, iterations=citerations, *compilerargs)       
        # Keeps track of the current composite Clifford
        s_composite, p_composite = _symp.compose_cliffords(s_composite, p_composite, s, p)
        full_circuit.append_circuit(circuit)
    
    # Find the symplectic rep of the inverse clifford
    s_inverse, p_inverse = _symp.inverse_clifford(s_composite, p_composite)
    
    # If we want to randomize the expected output then randomize the p_inverse vector, so that
    # the final bit of circuit will only invert the preceeding circuit up to a random Pauli.
    if randomizeout: p_for_inversion = _symp.random_phase_vector(s_inverse,n)
    else: p_for_inversion = p_inverse
    
    # Compile the inversion circuit
    inversion_circuit = _cmpl.compile_clifford(s_inverse, p_for_inversion, pspec, subsetQs=subsetQs, iterations=citerations, 
                                               *compilerargs)    
    full_circuit.append_circuit(inversion_circuit)
    # Find the expected outcome of the circuit.
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(full_circuit,pspec=pspec)
    # Check the output is the identity up to Paulis.
    assert(_np.array_equal(s_out[:n,n:],_np.zeros((n,n),int)))
    # Find the ideal-out of the circuit, as a bit-string.
    s_inputstate, p_inputstate = _symp.prep_stabilizer_state(n, zvals=None)
    s_outstate, p_outstate = _symp.apply_clifford_to_stabilizer_state(s_out, p_out, s_inputstate, p_inputstate)
    idealout = []
    for q in range(n):
        measurement_out = _symp.pauli_z_measurement(s_outstate, p_outstate, q)
        # This is the probability of the 0 outcome (it is a float)
        bit = measurement_out[1]
        assert(_np.allclose(bit,0.) or _np.allclose(bit,1.)), "Ideal output is not a computational basis state!"
        if not randomizeout: assert(_np.allclose(bit,0.)), "Ideal output is not the all 0s computational basis state!"
        idealout.append(round(measurement_out[1]))
    # Convert ideal-out to a tuple, so that it is imutable 
    idealout = tuple(idealout)

    return full_circuit, idealout

def clifford_rb_experiment(pspec, lengths, circuits_per_length, subsetQs=None, randomizeout=False, 
                           citerations=20, compilerargs=[], descriptor='A Clifford RB experiment',
                           verbosity=1):

    """
    Generates a "Clifford randomized benchmarking" (CRB) experiment, which is the RB protocol defined 
    in "Scalable and robust randomized benchmarking of quantum processes", Magesan et al. PRL 106 180504 (2011). 
    The circuits created by this function will respect the connectivity and gate-set of the device encoded 
    by `pspec` (see the ProcessorSpec object docstring for how to construct the relevant `pspec` for a device).

    Note that this function uses the convention that a length "l" CRB circuit  consists of "l"+2 Clifford gates
    (before compilation), rather than the  more usual convention of defining the "CRB length" to be the number 
    of Clifford gates - 1. This is for consistency with the other RB functions in pyGSTi: in all RB-circuit-generating 
    in pyGSTi, length zero corresponds to the minimum-length circuit allowed by the protocol. (Note that changing the 
    "RB length" by a constant additive factor is irrelevant for fitting purposes, except that it changes
    the obtained "SPAM" fit parameter).

    This function is a wrap-around for rb.sample.clifford_rb_circuit(). 

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the CRB experiment is being generated for, which defines the 
       "native" gate-set and the connectivity of the device. The returned CRB circuits will be over
       the gates in `pspec`, and will respect the connectivity encoded by `pspec`.
                
    lengths : list of ints
        The "CRB lengths" of the circuit; a list of integers >= 0. The CRB length is the number of Cliffords 
        in the circuit - 2 *before* each Clifford is compiled into the native gate-set. 

    circuits_per_length : int
        The number of (possibly) different CRB circuits sampled at each length.

    subsetQs : list, optional
        If not None, a list of the qubits that the RB circuits are to be sampled for. This should
        be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
        irrelevant. If desired, a circuit that explicitly idles on the other qubits can be obtained
        by using methods of the Circuit object.
        
    randomizeout : bool, optional
        If False, the ideal output of the circuits (the "success" or "survival" outcome) is always
        the all-zeros bit string. This is probably considered to be the "standard" in CRB. If True, 
        the ideal output a circuit is randomized to a uniformly random bit-string. This setting is 
        useful for, e.g., detecting leakage/loss/measurement-bias etc.

    citerations : int, optional
        Some of the Clifford compilation algorithms in pyGSTi (including the default algorithm) are 
        randomized, and the lowest-cost circuit is chosen from all the circuit generated in the
        iterations of the algorithm. This is the number of iterations used. The time required to
        generate a CRB circuit is linear in `citerations` * (CRB length + 2). Lower-depth / lower 2-qubit
        gate count compilations of the Cliffords are important in order to successfully implement
        CRB on more qubits. 
    
    compilerargs : list, optional
        A list of arguments that are handed to compile_clifford() function, which includes all the
        optional arguments of compile_clifford() *after* the `iterations` option (set by `citerations`).
        In order, this list should be values for:
            - algorithm : str. A string that specifies the compilation algorithm. The default in
                compile_clifford() will always be whatever we consider to be the 'best' all-round
                algorith,
            - aargs : list. A list of optional arguments for the particular compilation algorithm.
            - costfunction : 'str' or function. The cost-function from which the "best" compilation
                for a Clifford is chosen from all `citerations` compilations. The default costs a
                circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
            - prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
            - paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a 
                random Pauli on each qubit (compiled into native gates). I.e., if this is True the
                native gates are Pauli-randomized. When True, this prevents any coherent errors adding
                (on average) inside the layers of each compiled Clifford, at the cost of increased
                circuit depth. Defaults to False.
        For more information on these options, see the compile_clifford() docstring.

    decscriptor : str, optional
        A string describing the experiment generated, which will be stored in the returned
        dictionary.

    verbosity : int, optional
        If > 0 the number of circuits generated so far is shown.

    Returns
    -------
    dict
        A dictionary containing the generated RB circuits, the error-free outputs of the circuit,
        and the specification used to generate the circuits. The keys are:

        - 'circuits'. A dictionary of the sampled circuits. The circuit with key(l,k) is the kth circuit 
        at CRB length l.

        - 'idealout'. A dictionary of the error-free outputs of the circuits as tuples. The tuple with 
        key(l,k) is the error-free output of the (l,k) circuit. The ith element of this tuple corresponds
        to the error-free outcome for the qubit on the ith wire of the output circuit and/or the ith element
        of the list at the key 'qubitordering'. These tuples will all be (0,0,0,...) when `randomizeout` is
        False

        - 'qubitordering'. The ordering of the qubits in the 'idealout' tuples.

        - 'spec'. A dictionary containing all of the parameters handed to this function, except `pspec`.
        This then specifies how the circuits where generated
    """
    experiment_dict = {}
    experiment_dict['spec'] = {}
    experiment_dict['spec']['subsetQs'] = subsetQs
    experiment_dict['spec']['randomizeout'] = randomizeout
    experiment_dict['spec']['citerations'] = citerations
    experiment_dict['spec']['compilerargs'] = compilerargs
    experiment_dict['spec']['descriptor'] = descriptor
    if subsetQs is not None: experiment_dict['qubitordering'] = tuple(subsetQs)
    else: experiment_dict['qubitordering'] = tuple(pspec.qubit_labels)
    
    experiment_dict['circuits'] = {}
    experiment_dict['idealout'] = {}

    for lnum, l in enumerate(lengths):
        if verbosity > 0:
            print('- Sampling {} circuits at CRB length {} ({} of {} lengths)'.format(circuits_per_length,l,lnum+1,len(lengths)))
            print('  - Number of circuits sampled = ',end='')
        for j in range(circuits_per_length):
            c, iout = clifford_rb_circuit(pspec, l, subsetQs=subsetQs, randomizeout=randomizeout, 
                                          citerations=citerations, compilerargs=compilerargs)           
            experiment_dict['circuits'][l,j] = c
            experiment_dict['idealout'][l,j] = iout
            if verbosity > 0: print(j+1,end=',')
        if verbosity >0: print('')

    return experiment_dict

def pauli_layer_as_compiled_circuit(pspec, subsetQs=None, keepidle=False):
    """
    Samples a uniformly random n-qubit Pauli and then converts
    it to the native gate-set of `pspec`.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device.
                
    subsetQs : list, optional
        If not None, a list of a subset of the qubits from `pspec` that 
        the pauli circuit should act on.

    keepidle : bool, optional
        Whether to always have the circuit at-least depth 1.

    Returns
    -------
    Circuit
        A circuit corresponding to a uniformly random n-qubit Pauli.
    """
    if pspec.identity is not None: identity = pspec.identity
    else: identity = 'I'
    if subsetQs is not None:qubits = subsetQs[:] # copy this list
    else: qubits = pspec.qubit_labels[:] # copy this list
    n = len(qubits)

    # The hard-coded notation for that Pauli operators
    paulis = ['I','X','Y','Z']
    
    # Samples a random Pauli layer   
    r = _np.random.randint(0,4,size=n)           
    pauli_layer_std_lbls = [_lbl.Label(paulis[r[q]],(qubits[q],)) for q in range(n)]
    # Converts the layer to a circuit, and changes to the native gateset.
    pauli_circuit = _cir.Circuit(gatestring=pauli_layer_std_lbls, parallelize=True, line_labels=qubits, identity='I')
    pauli_circuit.change_gate_library(pspec.compilations['absolute'], identity=identity)
    if keepidle:
        if pauli_circuit.depth() == 0:
            pauli_circuit.insert_layer([_lbl.Label(identity,qubits[0]),],0)

    return pauli_circuit

def oneQclifford_layer_as_compiled_circuit(pspec, subsetQs=None):
    """
    Samples a uniformly random layer of 1-qubit Cliffords on all 
    the qubits, and then converts it to the native gate-set of `pspec`.
    That is, an independent and uniformly random 1-qubit Clifford is
    sampled for each qubit.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device.
                
    subsetQs : list, optional
        If not None, a list of a subset of the qubits from `pspec` that 
        the circuit should act on.

    Returns
    -------
    Circuit
        A circuit corresponding to an independent, uniformly random 1-qubit
        Clifford gate on each qubit.
    """
    if pspec.identity is not None:
        identity = pspec.identity
    else:
        identity = 'I'
    if subsetQs is not None:
        n = len(subsetQs)
        qubits = subsetQs[:] # copy this list
    else:
        n = pspec.number_of_qubits
        qubits = pspec.qubit_labels[:] # copy this list

    # The hard-coded notation for the 1Q clifford operators
    oneQcliffords = ['C'+str(i) for i in range(24)]
           
    r = _np.random.randint(0,24,size=n)
            
    oneQclifford_layer_std_lbls = [_lbl.Label(oneQcliffords[r[q]],(qubits[q],)) for q in range(n)]
    oneQclifford_circuit = _cir.Circuit(gatestring=oneQclifford_layer_std_lbls, parallelize=True, line_labels=qubits, identity=identity)
    oneQclifford_circuit.change_gate_library(pspec.compilations['absolute'])
    
    return oneQclifford_circuit

def mirror_rb_circuit(pspec, length, subsetQs=None, sampler='Qelimination', samplerargs=[], localclifford=True,
                      paulirandomize=True):
    """
    Generates a "mirror randomized benchmarking" (MRB) circuit, for the case of Clifford gates and with the option
    of Pauli-randomization and Clifford-twirling. This RB method is currently in development; this docstring will 
    be updated in the future with further information on this technique.

    To implement mirror RB it is necessary for U^(-1) to in the gate-set for every U in the gate-set.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit is being sampled for. The `pspec` is always
       handed to the sampler, as the first argument of the sampler function.
                
    length : int
        The "mirror RB length" of the circuit, which is closely related to the circuit depth. It
        must be an even integer, and can be zero.

        - If `localclifford` and `paulirandomize` are False, this is the depth of the sampled circuit.
          The first length/2 layers are all sampled independently according to the sampler specified by
          `sampler`. The remaining half of the circuit is the "inversion" circuit that is determined
          by the first half.

        - If `paulirandomize` is True and `localclifford` is False, the depth of the circuits is 
          2*length+1 with odd-indexed layers sampled according to the sampler specified by `sampler, and
          the the zeroth layer + the even-indexed layers consisting of random 1-qubit Pauli gates.

        - If `paulirandomize` and `localclifford` are True, the depth of the circuits is 
          2*length+1 + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for 
          the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

        - If `paulirandomize` is False and `localclifford` is True, the depth of the circuits is 
          length + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for 
          the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

    subsetQs : list, optional
        If not None, a list of the qubits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
        irrelevant.
        
    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function 
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
        a valid option for n-qubit MRB -- it results in sim. 1-qubit MRB -- but it is not explicitly
        forbidden by this function]. If `sampler` is a function, it should be a function that takes 
        as the first argument a ProcessorSpec, and returns a random circuit layer as a list of gate 
        Label objects. Note that the default 'Qelimination' is not necessarily the most useful 
        in-built sampler, but it is the only sampler that requires no parameters beyond the ProcessorSpec 
        *and* works for arbitrary connectivity devices. See the docstrings for each of these samplers 
        for more information.
    
    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec` and `samplerargs` lists the
        remaining arguments handed to the sampler.
        
    localclifford: bool, optional
        Whether to start the circuit with uniformly random 1-qubit Cliffords and all of the 
        qubits (compiled into the native gates of the device).
        
    paulirandomize: bool, optional
        Whether to have uniformly random Pauli operators on all of the qubits before and 
        after all of the layers in the "out" and "back" random circuits. At length 0 there
        is a single layer of random Pauli operators (in between two layers of 1-qubit Clifford
        gates if `localclifford` is True); at length l there are 2l+1 Pauli layers as there
        are 
        
    Returns
    -------
    Circuit
        A random MRB circuit, sampled as specified, of depth:

            - `length`, if not paulirandomize and not local clifford.
            - 2*`length`+1 if paulirandomize and not local clifford.
            - `length` + X, if not paulirandomize and local clifford, where X is a random variable
              that accounts for the depth from the layers of random 1-qubit Cliffords (X = 2 if the 1
              qubit Clifford gates are "native" gates in the ProcessorSpec).
            - 2*`length`+1 + X, if paulirandomize and local clifford, where X is a random variable
              that accounts for the depth from the layers of random 1-qubit Cliffords (X = 2 if the 1
              qubit Clifford gates are "native" gates in the ProcessorSpec). 

    Tuple 
        A length-n tuple of integers in [0,1], corresponding to the error-free outcome of the 
        circuit. Always all zeros if `randomizeout` is False. The ith element of the tuple 
        corresponds to the error-free outcome for the qubit labelled by: the ith element of
        `subsetQs`, if `subsetQs` is not None; the ith element of `pspec.qubit_labels`, otherwise.
        In both cases, the ith element of the tuple corresponds to the error-free outcome for the
        qubit on the ith wire of the output circuit.       
    """    
    assert(length % 2 == 0), "The mirror rb length `length` must be even!"
    random_natives_circuit_length = length//2

    # This is used to fix the identity element in the circuits to be consistent with the pspec.identity element,
    # if there is one.
    if pspec.identity is not None: identity = pspec.identity
    else: identity = 'I'
    if subsetQs is not None:
        assert(isinstance(subsetQs,list) or isinstance(subsetQs,tuple)), "If not None, `subsetQs` must be a list!"
        subsetQs = list(subsetQs)
        n = len(subsetQs)
        allqubits = subsetQs
    else:
        n = pspec.number_of_qubits
        allqubits = pspec.qubit_labels[:] # copy this list

    # Check that the inverse of every gate is in the gateset:
    for gname in pspec.root_gate_names:
        assert(gname in list(pspec.gate_inverse.keys())), "Not every gate has its inverse in the gate-set! MRB is not possible!"
 
    # Find a random circuit according to the sampling specified; this is the "out" circuit.
    circuit = random_circuit(pspec, random_natives_circuit_length, subsetQs=subsetQs, sampler=sampler, samplerargs=samplerargs)
    # Copy the circuit, to create the "back" circuit from the "out" circuit.
    circuit_inv = circuit.copy()
    # First we reverse the circuit; then we'll replace each gate with its inverse.
    circuit_inv.reverse()
    # Go through the circuit and replace every gate with its inverse, stored in the pspec. If the circuits
    # are length 0 this is skipped.
    for i in range(n):
        for j in range(random_natives_circuit_length):
            # This will fail if the gates do not all have an inverse in the set.
            inv_name = pspec.gate_inverse[circuit_inv.line_items[i][j].name]
            qubits_for_gate = circuit_inv.line_items[i][j].qubits
            # Replace the gatename with the gatename of the inverse.
            circuit_inv.line_items[i][j] = _lbl.Label(inv_name,qubits_for_gate)

    # If we are Pauli randomizing, we add a indepedent uniformly random Pauli layer, as a compiled circuit, after 
    # every layer in the "out" and "back" circuits. If the circuits are length 0 we do nothing here.
    if paulirandomize:
        for i in range(random_natives_circuit_length):            
            pauli_circuit = pauli_layer_as_compiled_circuit(pspec, subsetQs=subsetQs, keepidle=True)
            circuit.insert_circuit(pauli_circuit,random_natives_circuit_length-i)
            pauli_circuit = pauli_layer_as_compiled_circuit(pspec, subsetQs=subsetQs, keepidle=True)
            circuit_inv.insert_circuit(pauli_circuit,random_natives_circuit_length-i)
        
    # We then append the "back" circuit to the "out" circuit. At length 0 this will be a length 0 circuit.
    circuit.append_circuit(circuit_inv)
    
    # If we Pauli randomize, There should also be a random Pauli at the start of this circuit; so we add that. If we have a 
    # length 0 circuit we now end up with a length 1 circuit (or longer, if compiled Paulis). So, there is always
    # a random Pauli.
    if paulirandomize:
        pauli_circuit = pauli_layer_as_compiled_circuit(pspec, subsetQs=subsetQs, keepidle=True)
        circuit.insert_circuit(pauli_circuit,0)

    # If we start with a random layer of 1-qubit Cliffords, we sample this here.
    if localclifford:
        # Sample a compiled 1Q Cliffords layer
        oneQclifford_circuit_out = oneQclifford_layer_as_compiled_circuit(pspec, subsetQs=subsetQs)
        # Generate the inverse in the same way as before (note that this will not be the same in some
        # circumstances as finding the inverse Cliffords and using the compilations for those. It doesn't
        # matter much which we do).
        oneQclifford_circuit_back = oneQclifford_circuit_out.copy()
        oneQclifford_circuit_back.reverse()
        for i in range(n):
            for j in range(oneQclifford_circuit_back.depth()):
                # This will fail if the gates do not all have an inverse in the set.
                inv_name = pspec.gate_inverse[oneQclifford_circuit_back.line_items[i][j].name]
                qubits_for_gate = oneQclifford_circuit_back.line_items[i][j].qubits
                # Replace the gatename with the gatename of the inverse.
                oneQclifford_circuit_back.line_items[i][j] = _lbl.Label(inv_name,qubits_for_gate)

        # Put one these 1Q clifford circuits at the start and one at then end.
        circuit.append_circuit(oneQclifford_circuit_out)
        circuit.prefix_circuit(oneQclifford_circuit_back)

    # The full circuit should be, up to a Pauli, the identity.
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(circuit,pspec=pspec)
    assert(_np.array_equal(s_out,_np.identity(2*n,int)))

    # Find the error-free output.
    s_inputstate, p_inputstate = _symp.prep_stabilizer_state(n, zvals=None)
    s_outstate, p_outstate = _symp.apply_clifford_to_stabilizer_state(s_out, p_out, s_inputstate, p_inputstate)
    idealout = []

    for q in range(n):
        measurement_out = _symp.pauli_z_measurement(s_outstate, p_outstate, q)
        bit = measurement_out[1]
        assert(bit == 0 or bit == 1), "Ideal output is not a computational basis state!"
        if not paulirandomize:
            assert(bit == 0), "Ideal output is not the all 0s computational basis state!"
        idealout.append(int(measurement_out[1]))
    idealout = tuple(idealout)
    
    return circuit, idealout

def mirror_rb_experiment(pspec, lengths, circuits_per_length, subsetQs=None, sampler='Qelimination', samplerargs=[], 
                         localclifford=True, paulirandomize=True, descriptor='A mirror RB experiment'):
    """
    Generates a "mirror randomized benchmarking" (MRB) experiment, for the case of Clifford gates and with the option
    of Pauli-randomization and Clifford-twirling. This RB method is currently in development; this docstring will 
    be updated in the future with further information on this technique.

    To implement mirror RB it is necessary for U^(-1) to in the gate-set for every U in the gate-set.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the experiment is being generated for. The `pspec` is always
       handed to the sampler, as the first argument of the sampler function.
                
    lengths : list of ints
        The "mirror RB lengths" of the circuits, which is closely related to the circuit depth. A MRB
        length must be an even integer, and can be zero.

        - If `localclifford` and `paulirandomize` are False, the depth of a sampled circuit = the MRB length.
          The first length/2 layers are all sampled independently according to the sampler specified by
          `sampler`. The remaining half of the circuit is the "inversion" circuit that is determined
          by the first half.

        - If `paulirandomize` is True and `localclifford` is False, the depth of a circuit is 
          2*length+1 with odd-indexed layers sampled according to the sampler specified by `sampler, and
          the the zeroth layer + the even-indexed layers consisting of random 1-qubit Pauli gates.

        - If `paulirandomize` and `localclifford` are True, the depth of a circuit is 
          2*length+1 + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for 
          the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

        - If `paulirandomize` is False and `localclifford` is True, the depth of a circuit is 
          length + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for 
          the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

    subsetQs : list, optional
        If not None, a list of the qubits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
        irrelevant.
        
    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function 
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
        a valid option for n-qubit MRB -- it results in sim. 1-qubit MRB -- but it is not explicitly
        forbidden by this function]. If `sampler` is a function, it should be a function that takes 
        as the first argument a ProcessorSpec, and returns a random circuit layer as a list of gate 
        Label objects. Note that the default 'Qelimination' is not necessarily the most useful 
        in-built sampler, but it is the only sampler that requires no parameters beyond the ProcessorSpec 
        *and* works for arbitrary connectivity devices. See the docstrings for each of these samplers 
        for more information.
    
    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec` and `samplerargs` lists the
        remaining arguments handed to the sampler.
        
    localclifford : bool, optional
        Whether to start the circuit with uniformly random 1-qubit Cliffords and all of the 
        qubits (compiled into the native gates of the device).
        
    paulirandomize : bool, optional
        Whether to have uniformly random Pauli operators on all of the qubits before and 
        after all of the layers in the "out" and "back" random circuits. At length 0 there
        is a single layer of random Pauli operators (in between two layers of 1-qubit Clifford
        gates if `localclifford` is True); at length l there are 2l+1 Pauli layers as there
        are 

    descriptor : str, optional
        A string describing the generated experiment. Stored in the returned dictionary.
        
    Returns
    -------
    dict
        A dictionary containing the generated RB circuits, the error-free outputs of the circuit,
        and the specification used to generate the circuits. The keys are:

        - 'circuits'. A dictionary of the sampled circuits. The circuit with key(l,k) is the kth circuit 
        at MRB length l.

        - 'idealout'. A dictionary of the error-free outputs of the circuits as tuples. The tuple with 
        key(l,k) is the error-free output of the (l,k) circuit. The ith element of this tuple corresponds
        to the error-free outcome for the qubit on the ith wire of the output circuit and/or the ith element
        of the list at the key 'qubitordering'. These tuples will all be (0,0,0,...) when `paulirandomize` is
        False

        - 'qubitordering'. The ordering of the qubits in the 'idealout' tuples.

        - 'spec'. A dictionary containing all of the parameters handed to this function, except `pspec`.
        This then specifies how the circuits where generated
    """ 
    experiment_dict = {}
    experiment_dict['spec'] = {}
    experiment_dict['spec']['subsetQs'] = subsetQs
    experiment_dict['spec']['sampler'] = sampler
    experiment_dict['spec']['samplerargs'] = samplerargs
    experiment_dict['spec']['localclifford'] = localclifford
    experiment_dict['spec']['paulirandomize'] = paulirandomize
    experiment_dict['spec']['descriptor'] = descriptor
    if subsetQs is not None: experiment_dict['qubitordering'] = tuple(subsetQs)
    else: experiment_dict['qubitordering'] = tuple(pspec.qubit_labels)
    experiment_dict['circuits'] = {}
    experiment_dict['idealout'] = {}

    for l in lengths:
        for j in range(circuits_per_length):
            c, iout = mirror_rb_circuit(pspec, l, subsetQs=subsetQs, sampler=sampler, samplerargs=samplerargs, 
                                        localclifford=localclifford, paulirandomize=paulirandomize)

            experiment_dict['circuits'][l,j] = c
            experiment_dict['idealout'][l,j] = iout

    return experiment_dict

def oneQ_generalized_rb_sequence(m, group_or_gateset, inverse=True, random_pauli=False, interleaved=None, 
                     group_inverse_only=False, group_prep=False, compilation=None,
                     generated_group=None, gateset_to_group_labels=None, seed=None, randState=None):
    """
    Makes a random 1-qubit RB sequence, with RB over an arbitrary group and with a range of other
    options that allow circuits for many types of RB to be generated, including:
    
    - Clifford RB
    - Direct RB
    - Interleaved Clifford or direct RB
    - Unitarity Clifford or direct RB

    The function can in-principle be used beyond 1-qubit RB, but it relies on explicit matrix representation
    of a group, which is infeasble for, e.g., the many-qubit Clifford group.

    Note that this function has *not* been carefully tested. This will be rectified in the future,
    or this function will be replaced.
    
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
        rndm = _np.random.RandomState(seed) # ok if seed is None
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
        random_string.extend(inversion_sequence)
        
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

# Future : possibly add this back in, but only if the other function it is a wrap-around
# for has been tested.
# def oneQ_generalized_rb_experiment(m_list, K_m, group_or_gateset, inverse=True, 
#                               interleaved = None, alias_maps=None, seed=None, 
#                               randState=None):
#     """
#     Makes a list of random RB sequences.
    
#     Parameters
#     ----------
#     m_list : list or array of ints
#         The set of lengths for the random sequences (with the total
#         number of Cliffords in each sequence given by m_list + 1). Minimal
#         allowed length is therefore 1 (a random CLifford followed by its 
#         inverse).

#     clifford_group : MatrixGroup
#         Which Clifford group to use.

#     K_m : int or dict
#         If an integer, the fixed number of Clifford sequences to be sampled at
#         each length m.  If a dictionary, then a mapping from Clifford
#         sequence length m to number of Cliffords to be sampled at that length.
    
#     alias_maps : dict of dicts, optional
#         If not None, a dictionary whose keys name other gate-label-sets, e.g.
#         "primitive" or "canonical", and whose values are "alias" dictionaries 
#         which map the clifford labels (defined by `clifford_group`) to those
#         of the corresponding gate-label-set.  For example, the key "canonical"
#         might correspond to a dictionary "clifford_to_canonical" for which 
#         (as one example) clifford_to_canonical['Gc1'] == ('Gy_pi2','Gy_pi2').
            
#     seed : int, optional
#         Seed for random number generator; optional.

#     randState : numpy.random.RandomState, optional
#         A RandomState object to generate samples from. Can be useful to set
#         instead of `seed` if you want reproducible distribution samples across
#         multiple random function calls but you don't want to bother with
#         manually incrementing seeds between those calls.
    
#     Returns
#     -------
#     dict or list
#         If `alias_maps` is not None, a dictionary of lists-of-gatestring-lists
#         whose keys are 'clifford' and all of the keys of `alias_maps` (if any).
#         Values are lists of `GateString` lists, one for each K_m value.  If
#         `alias_maps` is None, then just the list-of-lists corresponding to the 
#         clifford gate labels is returned.
#     """

#     if randState is None:
#         rndm = _np.random.RandomState(seed) # ok if seed is None
#     else:
#         rndm = randState
        
#     assert hasattr(group_or_gateset, 'gates') or hasattr(group_or_gateset, 
#            'product'), 'group_or_gateset must be a MatrixGroup or Gateset'
    
    
#     if inverse:
#         if hasattr(group_or_gateset, 'gates'):
#             group_or_gateset = _rbobjs.MatrixGroup(group_or_gateset.gates.values(),
#                                   group_or_gateset.gates.keys())
#     if isinstance(K_m,int):
#         K_m_dict = {m : K_m for m in m_list }
#     else: K_m_dict = K_m
#     assert hasattr(K_m_dict, 'keys'),'K_m must be a dict or int!'

#     string_lists = {'uncompiled': []} # GateStrings with uncompiled labels
#     if alias_maps is not None:
#         for gstyp in alias_maps.keys(): string_lists[gstyp] = []

#     for m in m_list:
#         K = K_m_dict[m]
#         strs_for_this_m = [ create_random_gatestring(m, group_or_gateset,
#             inverse=inverse,interleaved=interleaved,randState=rndm) for i in range(K) ]
#         string_lists['uncompiled'].append(strs_for_this_m)
#         if alias_maps is not None:
#             for gstyp,alias_map in alias_maps.items(): 
#                 string_lists[gstyp].append(
#                     _cnst.translate_gatestring_list(strs_for_this_m,alias_map))

#     if alias_maps is None:
#         return string_lists['uncompiled'] #only list of lists is uncompiled one
#     else:
#         return string_lists #note we also return this if alias_maps == {}

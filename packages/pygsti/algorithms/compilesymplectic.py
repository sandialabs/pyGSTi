""" Symplectic matrix compilation routines: helpers for clifford and stabilizer compiling """
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

def create_standard_cost_function(name):
    
    if name == '2QGC':
        def costfunction(circuit, junk):
            return circuit.twoqubit_gatecount()
    elif name == 'depth':
        def costfunction(circuit, junk):
            return circuit.depth()
    #elif costfunction == 'gcosts':
    #        # Todo : make this work - .gatecosts is currently not a property of a DeviceSpec object,
    #        # and circuit.cost() is currently not a method of circuit.
    #        c_cost = circuit.cost(pspec.gatecosts)
    else:
        raise ValueError("This `costfunction` string is not a valid option!")
    return costfunction

def compile_symplectic(s, pspec=None, subsetQs=None, iterations=20, algorithms=['DBGGE','RBGGE'], 
                       costfunction='2QGC', paulirandomize=False, aargs=None, check=True):    
    """
    Tim todo : docstring.

    # The cost function should take two arguments. (the 2nd one is a pspec.)
    """
    # The number of qubits the symplectic matrix is on. 
    n = _np.shape(s)[0]//2 
    if pspec is not None:
        if subsetQs is None:
            # At this point we don't write over subsetQs, as it is better to pass it as `None` to the algorithms.
            assert(pspec.number_of_qubits == n), "If all the qubits in `pspec` are to be used, `s` must be a symplectic matrix over {} qubits!".format(pspec.number_of_qubits)
        else:
            assert(len(subsetQs) == n), "The subset of qubits to compile `s` for is the wrong size for this symplectic matrix!" 

    # A list to hold the compiled circuits, from which we'll choose the best one. Each algorithm
    # only returns 1 circuit, so this will have the same length as the `algorithms` list.
    circuits = []

    # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str):
        costfunction = create_standard_cost_function(costfunction)
    
    # Todo : if aargs are specified, pass them to the algorithms.
    
    # Deterministic basic global Gaussian elimination
    if 'DBGGE' in algorithms:       
        circuit = ordered_global_gaussian_elimination(s, pspec=pspec, subsetQs=subsetQs, iterations=1, ctype='basic', check=False)
        circuits.append(circuit)      
    
    # Randomized basic global Gaussian elimination, whereby the order that the qubits are eliminated in
    # is randomized.
    if 'RBGGE' in algorithms:
        circuit = randomized_global_gaussian_elimination(s, pspec=pspec, subsetQs=subsetQs, ctype='basic', 
                                                        costfunction=costfunction, iterations = iteration, check=False) 
        circuits.append(circuit) 
        
    # The Aaraonson-Gottesman method for compiling a symplectic matrix using 5 CNOT circuits + local layers,
    # with the CNOT circuits compiled using global Gaussian elimination.
    if 'AGvGE' in algorithms:       
        circuit = aaronson_gottesman_on_symplectic(s, pspec=pspec, subsetQs=subsetQs, cnotmethod='GE', check=False)  
        circuits.append(circuit) 
        
    if 'AGvPMH' in algorithms:
        circuit = aaronson_gottesman_on_symplectic(s, pspec=pspec, subsetQs=subsetQs, cnotmethod = 'PMH', check=False)   
        circuits.append(circuit) 
        
    if 'iAGvGE' in algorithms:
        circuit = improved_aaronson_gottesman_on_symplectic(s, pspec=pspec, subsetQs=subsetQs, cnotmethod = 'GE', check=False) 
        circuits.append(circuit) 
        
    if 'iAGvPMH' in algorithms:
        circuit = improved_aaronson_gottesman_on_symplectic(s, pspec=pspec, subsetQs=subsetQs, cnotmethod = 'PMH', check=False) 
        circuits.append(circuit) 
    
    # If multiple algorithms have be called, find the lowest cost circuit.
    if len(circuits) > 1:
        bestcost = _np.inf
        for c in circuits:
            c_cost = costfunction(c,pspec)  
            if c_cost < bestcost:
                circuit = c.copy()
                cost = bestcost                 
    else:
        circuit = circuits[0]

    # At this point we set subsetQs to be the full list of qubits, for re-labelling purposes below.
    if pspec is not None:
        if subsetQs is None:
            subsetQs = pspec.qubit_labels
    
    # If we want to Pauli randomize the circuits, we insert a random compiled Pauli layer between every layer.
    if paulirandomize:     
        paulilist = ['I','X','Y','Z']
        d = circuit.depth()
        for i in range(0,d+1):
            pcircuit = _Circuit(gatestring=[_Label(paulilist[_np.random.randint(4)],k) for k in range(n)],num_lines=n)
            if pspec is not None:
                # Map the circuit to the correct qubit labels
                pcircuit.map_state_space_labels(self, {i:subsetQs[i] for i in range(n)})
                # Compile the circuit into the native gateset, using an "absolute" compilation -- Pauli-equivalent is
                # not sufficient here.
                pcircuit.change_gate_library(pspec.compilations['absolute'])
            circuit.insert_circuit(pcircuit,d-i)

    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit,pspec=pspec)            
        assert(_np.array_equal(s,implemented_s)) 
            
    return circuit

def random_order_global_gaussian_elimination(s, pspec=None, subsetQs=None, ctype='basic', costfunction='2QGC', 
                                           iterations=10, check=True):
    """
    Tim todo : docstring.

    costfunction should be a function, taking a circuit + pspec. NO!!!!

    Note that it is better to use the wrap-around, as these algorithms don't check some consistency things.
    """   
    # The number of qubits the symplectic matrix is on. 
    n = _np.shape(s)[0]//2   
    # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str):
        costfunction = create_standard_cost_function(costfunction)
    # The elimination order in terms of qubit *index*, which is randomized below.
    eliminationorder = list(range(n))
    
    lowestcost = _np.inf    
    for i in range(0,iterations):
        
        # Pick a random order to attempt the elimination in
        _np.random.shuffle(eliminationorder)
        # Call the re-ordered global Gaussian elimination, which is wrap-around for the GE algorithms to deal
        # with qubit relabeling. Check is False avoids multiple checks of success, when only the last check matters.
        circuit = reordered_global_gaussian_elimination(s, eliminationorder, pspec=pspec, subsetQs=subsetQs, ctype=ctype, check=False)
            
        # Find the cost of the circuit, and keep it if this circuit is the lowest-cost circuit so far.
        circuit_cost = costfunction(circuit, pspec)                       
        if circuit_cost < lowestcost:
            bestcircuit = circuit.copy()
            lowestcost = circuit_cost 

    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(bestcircuit,pspec=pspec)            
        assert(_np.array_equal(s,implemented_s)) 
    
    return bestcircuit
  
def ordered_global_gaussian_elimination(s, pspec=None, ctype = 'basic', costfunction='2QGC', iterations=1,
                                        check=True):
    """
    todo : docstring.
    """
    
    # The number of qubits the symplectic matrix is on. 
    n = _np.shape(s)[0] // 2
     # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str):
        costfunction = create_standard_cost_function(costfunction)
    # Get a list of lists of ordered qubits
    # Todo : update this.
    orderedqubits = pspec.costorderedqubits
    # The starting elimination order
    eliminationorder = [q for sublist in orderedqubits for q in sublist]
    
    lowestcost = _np.inf
    for i in range(0,iterations):
        # Call the re-ordered global Gaussian elimination, which is wrap-around for the GE algorithms to deal
        # with qubit relabeling. Check is False avoids multiple checks of success, when only the last check matters.       
        circuit = reordered_global_gaussian_elimination(s, eliminationorder, pspec=pspec, subsetQs=subsetQs, ctype=ctype, check=False)
        # If we don't do any randomization, then just return this circuit.
        if iterations == 1:
            return circuit

        circuit_cost = costfunction(circuit, pspec)             
        if circuit_cost < lowestcost:
            bestcircuit = circuit.copy()
            lowestcost = circuit_cost 
        
        # Randomize the order of qubits that are the same cost to eliminate.
        eliminationorder = [_np.random.shuffle(sublist) for sublist in orderedqubits]

    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(bestcircuit,pspec=pspec)            
        assert(_np.array_equal(s,implemented_s)) 
    
    return bestcircuit

def reordered_global_gaussian_elimination(s, eliminationorder, pspec=None, subsetQs=None, ctype='basic', check=True):
    """
    todo : doctstring
    """
    # Re-order the s matrix to reflect this.
    P = _np.zeros((n,n),int)
    for j in range(0,n):
        P[j,eliminationorder[j]] = 1        
    P2n = _np.zeros((2*n,2*n),int)
    P2n[0:n,0:n] = P
    P2n[n:2*n,n:2*n] = P           
    permuted_s = _mtx.dotmod2(_mtx.dotmod2(P2n,s),_np.transpose(P2n))
               
    if ctype == 'basic':
        # Check is False avoids multiple checks of success, when only the last check matters.
        circuit = basic_global_gaussian_elimination(permuted_s, check=False)

    else:
        raise ValueError("Only the `basic` compilation sub-method is currently available!")
        # The plan is to write the ctype == 'advanced':
        
    # If the subsetQs is not None, we relabel the circuit in terms of the labels of these qubits.
    if subsetQs is not None:
        circuit.map_state_space_labels({i:subsetQs[eliminationorder[i]] for i in range(n)})
        circuit.reorder_wires(subsetQs)
    # If the subsetQs is None, but there is a pspec, we relabel the circuit in terms of the full set
    # of pspec labels.
    elif pspec is not None:
        circuit.map_state_space_labels({i:pspec.qubits_labels[eliminationorder[i]] for i in range(n)})
        circuit.reorder_wires(subsetQs)
        
    # If we have a pspec, we change the gate library. We use a pauli-equivalent compilation, as it is
    # only necessary to implement each gate in this circuit up to Pauli matrices.
    if pspec is not None:
        circuit.change_gate_library(pspec.compilations['paulieq'])

    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit,pspec=pspec)            
        assert(_np.array_equal(s,implemented_s)) 

    return circuit
    
    
def basic_global_gaussian_elimination(s, check=True):
    """
    s: array
        A square symplectic matrix over [0,1]. It should have elements that are integers, 
        and it should have even dimension (i.e, must an N x N matrix with N even).
        
    Returns: list
        A list of instruction, read left to right, for how to construct a Clifford corresponding 
        to the inverse of M -- up to a final or initial Pauli gate -- using CNOT, SWAP, PHASE 
        and hadamard gates. Here, the corresponding Clifford is defined with respect to the 
        'standard' symplectic form.
    
    """
    s_updated = _np.copy(s)      
    n = _np.shape(s)[0]//2
    
    assert(_symp.check_symplectic(s,convention='standard')), "The input matrix must be symplectic!"
    
    instruction_list = []
    # Map the portion of the symplectic matrix acting on qubit j to the identity, for j = 0,...,d-1 in
    # turn, using the basic row operations corresponding to the CNOT, Hadamard, phase, and SWAP gates.    
    for j in range (0,d):
        
        # *** Step 1: Set the upper half of column j to the relevant identity column ***
        upperl_c = s_updated[:d,j]
        lowerl_c = s_updated[d:,j]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
        
        # If the jth element in the column is not 1, it needs to be set to 1.
        if j not in upperl_ones:
            
            # First try using a Hadamard gate.
            if j in lowerl_ones:
                instruction_list.append(_Label('H',j))                
                s_updated = _symp.symplectic_action(s_updated, 'H', [j,])
                
            # Then try using a swap gate, we don't try and find the best qubit to swap with.
            elif len(upperl_ones) >= 1:
                instruction_list.append(_Label('CNOT',[j,upperl_ones[0]]))
                instruction_list.append(_Label('CNOT',[upperl_ones[0],j]))
                instruction_list.append(_Label('CNOT',[j,upperl_ones[0]]))
                s_updated = _symp.symplectic_action(s_updated, 'SWAP', [j,upperl_ones[0]])
                
            # Finally, try using swap and Hadamard gates, we don't try and find the best qubit to swap with.
            else:
                instruction_list.append(_Label('H',lowerl_ones[0]))
                s_updated = _symp.symplectic_action(s_updated, 'H', [lowerl_ones[0],])
                instruction_list.append(_Label('CNOT',[j,lowerl_ones[0]]))
                instruction_list.append(_Label('CNOT',[lowerl_ones[0],j]))
                instruction_list.append(_Label('CNOT',[j,lowerl_ones[0]]))
                s_updated = _symp.symplectic_action(s_updated, 'SWAP', [j,lowerl_ones[0]])
            
            # Update the lists that keep track of where the 1s are in the column.
            upperl_c = s_updated[:d,j]
            lowerl_c = s_updated[d:,j]
            upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
            lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
            
        # Pair up qubits with 1s in the jth upper jth column, and set all but the
        # jth qubit to 0 in logarithmic depth. When there is an odd number of qubits
        # one of them is left out in the layer.
        while len(upperl_ones) >= 2: 

            num_pairs = len(upperl_ones)//2
            
            for i in range(0,num_pairs):                
                if upperl_ones[i+1] != j:
                    controlq = upperl_ones[i]
                    targetq = upperl_ones[i+1]
                    del upperl_ones[1+i]
                else:
                    controlq = upperl_ones[i+1]
                    targetq = upperl_ones[i]
                    del upperl_ones[i]
                    
                instruction_list.append(_Label('CNOT',(controlq,targetq)))                   
                s_updated = _symp.symplectic_action(s_updated, 'CNOT', [controlq,targetq])

        # *** Step 2: Set the lower half of column j to all zeros ***
        upperl_c = s_updated[:d,j]
        lowerl_c = s_updated[d:,j]        
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
        
        # If the jth element in this lower column is 1, it must be set to 0.
        if j in lowerl_ones:
            instruction_list.append(_Label('P',j))
            s_updated = _symp.symplectic_action(s_updated, 'P', [j,])
            
        # Move in the 1 from the upper part of the column, and use this to set all
        # other elements to 0, as in Step 1.
        instruction_list.append(_Label('H',j))
        s_updated =_symp.symplectic_action(s_updated, 'H', [j,])
        
        upperl_c = None
        upperl_ones = None
        lowerl_c = s_updated[d:,j]
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])

        while len(lowerl_ones) >= 2:

            num_pairs = len(lowerl_ones)//2
            
            for i in range(0,num_pairs):
                if lowerl_ones[i+1] != j:
                    controlq = lowerl_ones[i+1]
                    targetq = lowerl_ones[i]
                    del lowerl_ones[1+i]
                else:
                    controlq = lowerl_ones[i]
                    targetq = lowerl_ones[i+1]
                    del lowerl_ones[i]
                
                instruction_list.append(_Label('CNOT',(controlq,targetq)))                   
                s_updated = _symp.symplectic_action(s_updated, 'CNOT', [controlq,targetq])
      
        # Move the 1 back to the upper column.
        instruction_list.append(_Label('H',j))
        s_updated = _symp.symplectic_action(s_updated, 'H', [j,])
       
        # *** Step 3: Set the lower half of column j+d to the relevant identity column ***
        upperl_c = s_updated[:d,j+d]
        lowerl_c = s_updated[d:,j+d]      
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
             
        while len(lowerl_ones) >= 2:  

            num_pairs = len(lowerl_ones)//2
            
            for i in range(0,num_pairs):
                
                if lowerl_ones[i+1] != j:
                    controlq = lowerl_ones[i+1]
                    targetq = lowerl_ones[i]
                    del lowerl_ones[1+i]
                else:
                    controlq = lowerl_ones[i]
                    targetq = lowerl_ones[i+1]
                    del lowerl_ones[i]
                
                instruction_list.append(_Label('CNOT',(controlq,targetq)))                   
                s_updated = _symp.symplectic_action(s_updated, 'CNOT', [controlq,targetq])
                
        # *** Step 4: Set the upper half of column j+d to all zeros ***
        upperl_c = s_updated[:d,j+d]
        lowerl_c = s_updated[d:,j+d]        
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
        
        # If the jth element in the upper column is 1 it must be set to zero
        if j in upperl_ones:       
            instruction_list.append(_Label('H',j))
            s_updated = _symp.symplectic_action(s_updated,'H',[j,])           
            instruction_list.append(_Label('P',j))
            s_updated = _symp.symplectic_action(s_updated,'P',[j,])                
            instruction_list.append(_Label('H',j))
            s_updated = _symp.symplectic_action(s_updated,'H',[j,])
        
        # Switch in the 1 from the lower column
        instruction_list.append(_Label('H',j))
        s_updated = _symp.symplectic_action(s_updated,'H',[j,])
        
        upperl_c = s_updated[:d,j+d]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_c = None
        lowerl_ones = None

        while len(upperl_ones) >= 2:        
            
            num_pairs = len(upperl_ones)//2
            
            for i in range(0,num_pairs):
                if upperl_ones[i+1] != j:
                    controlq = upperl_ones[i]
                    targetq = upperl_ones[i+1] 
                    del upperl_ones[1+i]
                else:
                    controlq = upperl_ones[i+1]
                    targetq = upperl_ones[i]                                                       
                    del upperl_ones[i]
                instruction_list.append(_Label('CNOT',(controlq,targetq)))                   
                s_updated = _symp.symplectic_action(s_updated, 'CNOT', [controlq,targetq])
                    
        # Switch the 1 back to the lower column
        instruction_list.append(_Label('H',j))            
        s_updated = _symp.symplectic_action(s_updated,'H',[j], optype='row')
                
        # If the matrix has been mapped to the identity, quit the loop as we are done.
        if _np.array_equal(s_updated,_np.identity(2*n,int)):
            break
            
    assert(_np.array_equal(s_updated,_np.identity(2*n,int))), "Compilation has failed!"
            
    # Operations that are the same next to each other cancel, and this algorithm can have these. So
    # we go through and delete them.
    j = 1
    depth = len(instruction_list)
    while j < depth:
        
        if instruction_list[depth-j] == instruction_list[depth-j-1]:
            del instruction_list[depth-j]
            del instruction_list[depth-j-1]
            j = j + 2
        else:
            j = j + 1
    
    # We turn the instruction list into a circuit
    circuit = _Circuit(gatestring=instruction_list,num_lines=n)
    # That circuit implements the inverse of s (it maps s to the identity). As all the gates in this
    # set are self-inverse (up to Pauli multiplication) we just reverse the circuit to get a circuit
    # for s.
    circuit.reverse()

    # To do the depth compression, we use the 1-qubit gate relations for the standard set of gates used
    # here.
    oneQgate_relations = _symp.single_qubit_clifford_symplectic_group_relations()
    circuit.compress_depth(oneQgate_relations, verbosity=0)

    # We check that the correct Clifford -- up to Pauli operators -- has been implemented.
    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit)            
        assert(_np.array_equal(s,implemented_s))        
        
    return circuit
    
# Todo : write this function - a connectivity-aware GGE algorithm
def advanced_global_gaussian_elimination(s, shortestpaths):
    raise NotImplementedError("This method is not yet written!")
    circuit = None
    return circuit
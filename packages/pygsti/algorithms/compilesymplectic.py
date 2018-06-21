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

def compile_symplectic(s, pspec=None, iterations=20, algorithms=['DGGE','RGGE'], costfunction='2QGC',
                       paulirandomize=False):    
    """
    
    """ 
    # Todo : change the iterations methods.
    # Remove this everywhere
    depth_compression = True
    circuits = []
    
    if 'DGGE' in algorithms:
        
        circuit = ordered_global_gaussian_elimination(s, pspec=pspec, iterations=1, ctype = 'basic', 
                                               depth_compression=depth_compression)
        circuits.append(circuit)      
    
    if 'RGGE' in algorithms:
        
        try:
            iterations = iterations['RGGE']
        except:
            iterations = 4
        circuit = randomized_global_gaussian_elimination(s, pspec=pspec, ctype = 'basic',
                                                         costfunction=costfunction, iterations = iterations, 
                                                         depth_compression=depth_compression) 
        circuits.append(circuit) 
        
    if 'AGvGE' in algorithms:
        
        # Todo : write this function
        circuit = aaronson_gottesman_on_symplectic(s, pspec=pspec, cnotmethod = 'GE', 
                                                   depth_compression=depth_compression)   
        circuits.append(circuit) 
        
    if 'AGvPMH' in algorithms:
        
        # Todo : write this function
        circuit = aaronson_gottesman_on_symplectic(s, pspec=pspec, cnotmethod = 'PMH', 
                                                   depth_compression=depth_compression)   
        circuits.append(circuit) 
        
    if 'iAGvGE' in algorithms:
         
        # Todo : write this function
        circuit = improved_aaronson_gottesman_on_symplectic(s, pspec=pspec, cnotmethod = 'GE', 
                                                             depth_compression=depth_compression)   
        circuits.append(circuit) 
        
    if 'iAGvPMH' in algorithms:
        
        # Todo : write this function
        circuit = improved_aaronson_gottesman_on_symplectic(s, pspec=pspec, cnotmethod = 'PMH',
                                                             depth_compression=depth_compression)   
        circuits.append(circuit) 
    
    # If multiple algorithms have be called, find the lowest cost circuit.
    if len(circuits) > 1:
        cost = _np.inf
        for c in circuits:
            if costfunction == '2QGC':
                c_cost = c.twoqubit_gatecount()           
            elif costfunction == 'depth':
                c_cost = c.depth()  
            elif costfunction == 'gcosts':
                # Todo : make this work - .gatecosts is currently not a property of a DeviceSpec object,
                # and circuit.cost() is currently not a method of circuit.
                c_cost = circuit.cost(pspec.gatecosts)
            else:
                c_cost = costfunction(circuit,pspec)
            if c_cost < cost:
                circuit = _copy.deepcopy(c)
                cost = c_cost
                  
    else:
        circuit = circuits[0]
        
    if paulirandomize:
        
        n = pspec.number_of_qubits
        paulilist = ['I','X','Y','Z']
        d = circuit.depth()
        for i in range(0,d+1):
            pcircuit = _Circuit(gatestring=[_Label(paulilist[_np.random.randint(4)],k) for k in range(n)],num_lines=n)
            pcircuit.change_gate_library(pspec.compilations['absolute'])
            circuit.insert_circuit(pcircuit,d-i)
            
    return circuit


def randomized_global_gaussian_elimination(s, pspec=None, ctype = 'basic', costfunction='2QGC', 
                                           iterations=10, depth_compression=True, returncosts=False):
    
    lowestcost = _np.inf
    n = _np.shape(s)[0] // 2
    eliminationorder = list(range(0,n))
    allcosts = []
    
    for i in range(0,iterations):
        
        # Pick a random order to attempt the elimination in
        _np.random.shuffle(eliminationorder)
        
        # Reorder the s matrix to reflect this.
        P = _np.zeros((n,n),int)
        for j in range(0,n):
            P[j,eliminationorder[j]] = 1 
        
        P2n = _np.zeros((2*n,2*n),int)
        P2n[0:n,0:n] = P
        P2n[n:2*n,n:2*n] = P
            
        permuted_s = _mtx.dotmod2(_mtx.dotmod2(P2n,s),_np.transpose(P2n))
        
        if ctype == 'advanced':
            raise NotImplementedError("This method is not yet written!")
            #circuit = advanced_global_gaussian_elimination(permuted_s,pspec.shortestpath,depth_compression=depth_compression)
        
        if ctype == 'basic':
            circuit = basic_global_gaussian_elimination(permuted_s ,depth_compression=depth_compression)
        
        circuit.relabel_qubits(eliminationorder)
        
        if pspec is not None:
            circuit.change_gate_library(pspec.compilations['paulieq'])
            
        # Find the cost of the circuit
        if costfunction == '2QGC':
            circuit_cost = circuit.twoqubit_gatecount()           
        elif costfunction == 'depth':
            circuit_cost = circuit.depth() 
        elif costfunction == 'gcosts':
            # Todo : make this work - .gatecosts is currently not a property of a DeviceSpec object,
            # and circuit.cost() is currently not a method of circuit.
            circuit_cost = circuit.cost(pspec.gatecosts)
        else:
            circuit_cost = costfunction(circuit,pspec)
            
        allcosts.append(circuit_cost)
            
        if circuit_cost < lowestcost:
                bestcircuit = _copy.deepcopy(circuit)
                lowestcost = circuit_cost 
    
    if returncosts:
        return bestcircuit, allcosts
    else:
        return bestcircuit
    
def ordered_global_gaussian_elimination(s, pspec=None, ctype = 'basic', costfunction='2QGC', iterations=1,
                                        depth_compression=True, returncosts=False):
    
    lowestcost = _np.inf
    n = _np.shape(s)[0] // 2
    allcosts = []
    
    # Get a list of lists of ordered qubits
    orderedqubits = pspec.costorderedqubits
    
    for i in range(0,iterations):
               
        eliminationorder = [q for sublist in orderedqubits for q in sublist]

        # Reorder the s matrix to reflect this.
        P = _np.zeros((n,n),int)
        for j in range(0,n):
            P[j,eliminationorder[j]] = 1 
        
        P2n = _np.zeros((2*n,2*n),int)
        P2n[0:n,0:n] = P
        P2n[n:2*n,n:2*n] = P
            
        permuted_s = _mtx.dotmod2(_mtx.dotmod2(P2n,s),_np.transpose(P2n))
        
        if ctype == 'advanced':
            raise NotImplementedError("This method is not yet written!")
            #circuit = advanced_global_gaussian_elimination(permuted_s ,pspec.shortestpaths,depth_compression=depth_compression)
        
        if ctype == 'basic':
            circuit = basic_global_gaussian_elimination(permuted_s ,depth_compression=depth_compression)
        
        circuit.relabel_qubits(eliminationorder)
        
        if pspec is not None:
            circuit.change_gate_library(pspec.compilations['paulieq'])
        
        if iterations == 1:
            return circuit
                
        # Find the cost of the circuit
        if costfunction == '2QGC':
            circuit_cost = circuit.twoqubit_gatecount()           
        elif costfunction == 'depth':
            circuit_cost = circuit.depth() 
        elif costfunction == 'gcosts':
            # Todo : make this work - .gatecosts is currently not a property of a DeviceSpec object,
            # and circuit.cost() is currently not a method of circuit.
            circuit_cost = circuit.cost(pspec.gatecosts)
        else:
            circuit_cost = costfunction(circuit,pspec)
            
        allcosts.append(circuit_cost)
            
        if circuit_cost < lowestcost:
                bestcircuit = _copy.deepcopy(circuit)
                lowestcost = circuit_cost 
        
        # Randomize the order of qubits that are the same cost to eliminate.
        eliminationorder = [_np.random.shuffle(sublist) for sublist in orderedqubits]
    
    if returncosts:
        return bestcircuit, allcosts
    else:
        return bestcircuit
    
    
def basic_global_gaussian_elimination(M, depth_compression=True):
    """
    M: array
        A square symplectic matrix over [0,1]. It should have elements that are integers, 
        and it should have even dimension (i.e, must an N x N matrix with N even).
        
    Returns: list
        A list of instruction, read left to right, for how to construct a Clifford corresponding 
        to the inverse of M -- up to a final or initial Pauli gate -- using CNOT, SWAP, PHASE 
        and hadamard gates. Here, the corresponding Clifford is defined with respect to the 
        'standard' symplectic form.
    
    """
    M_updated = _np.copy(M)  
    dim = _np.shape(M)[0]     
    d = dim//2
    
    assert(dim/2 == d), "The input matrix dimension must be even for the matrix to be symplectic!"    
    assert(_symp.check_symplectic(M,convention='standard')), "The input matrix must be symplectic!"
    
    instruction_list = []
    
    #if connectivity is 'complete':
    #    connectivity = _np.ones((d,d),int) - _np.identity(d,int)
        
    #
    # Todo: put something here that forces the connectivity matrix to be symmetric, or checks that.
    #
    
    # Map the portion of the symplectic matrix acting on qubit j to the identity, for j = 0,...,d-1 in
    # turn, using the basic row operations corresponding to the CNOT, Hadamard, phase, and SWAP gates.    
    for j in range (0,d):
        
        # *** Step 1: Set the upper half of column j to the relevant identity column ***
        upperl_c = M_updated[:d,j]
        lowerl_c = M_updated[d:,j]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
        
        # If the jth element in the column is not 1, it needs to be set to 1.
        if j not in upperl_ones:
            
            # First try using a Hadamard gate.
            if j in lowerl_ones:
                instruction_list.append(_Label('H',j))                
                M_updated = _symp.symplectic_action(M_updated, 'H', [j,])
                
            # Then try using a swap gate
            elif len(upperl_ones) >= 1:
                # THIS DOESN'T CURRENTLY FIND THE CLOSEST QUBIT TO SWAP WITH
                instruction_list.append(_Label('CNOT',[j,upperl_ones[0]]))
                instruction_list.append(_Label('CNOT',[upperl_ones[0],j]))
                instruction_list.append(_Label('CNOT',[j,upperl_ones[0]]))
                M_updated = _symp.symplectic_action(M_updated, 'SWAP', [j,upperl_ones[0]])
                
            # Finally, try using swap and Hadamard gates.
            else:
                # THIS DOESN'T CURRENTLY FIND THE CLOSEST QUBIT TO SWAP WITH
                instruction_list.append(_Label('H',lowerl_ones[0]))
                M_updated = _symp.symplectic_action(M_updated, 'H', [lowerl_ones[0],])
                instruction_list.append(_Label('CNOT',[j,lowerl_ones[0]]))
                instruction_list.append(_Label('CNOT',[lowerl_ones[0],j]))
                instruction_list.append(_Label('CNOT',[j,lowerl_ones[0]]))
                M_updated = _symp.symplectic_action(M_updated, 'SWAP', [j,lowerl_ones[0]])
            
            # Update the lists that keep track of where the 1s are in the column.
            upperl_c = M_updated[:d,j]
            lowerl_c = M_updated[d:,j]
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
                M_updated = _symp.symplectic_action(M_updated, 'CNOT', [controlq,targetq])

        # *** Step 2: Set the lower half of column j to all zeros ***
        upperl_c = M_updated[:d,j]
        lowerl_c = M_updated[d:,j]        
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
        
        # If the jth element in this lower column is 1, it must be set to 0.
        if j in lowerl_ones:
            instruction_list.append(_Label('P',j))
            M_updated = _symp.symplectic_action(M_updated, 'P', [j,])
            
        # Move in the 1 from the upper part of the column, and use this to set all
        # other elements to 0, as in Step 1.
        instruction_list.append(_Label('H',j))
        M_updated =_symp.symplectic_action(M_updated, 'H', [j,])
        
        upperl_c = None
        upperl_ones = None
        lowerl_c = M_updated[d:,j]
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
                M_updated = _symp.symplectic_action(M_updated, 'CNOT', [controlq,targetq])
      
        # Move the 1 back to the upper column.
        instruction_list.append(_Label('H',j))
        M_updated = _symp.symplectic_action(M_updated, 'H', [j,])
       
        # *** Step 3: Set the lower half of column j+d to the relevant identity column ***
        upperl_c = M_updated[:d,j+d]
        lowerl_c = M_updated[d:,j+d]      
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
                M_updated = _symp.symplectic_action(M_updated, 'CNOT', [controlq,targetq])
                
        # *** Step 4: Set the upper half of column j+d to all zeros ***
        upperl_c = M_updated[:d,j+d]
        lowerl_c = M_updated[d:,j+d]        
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
        
        # If the jth element in the upper column is 1 it must be set to zero
        if j in upperl_ones:       
            instruction_list.append(_Label('H',j))
            M_updated = _symp.symplectic_action(M_updated,'H',[j,])           
            instruction_list.append(_Label('P',j))
            M_updated = _symp.symplectic_action(M_updated,'P',[j,])                
            instruction_list.append(_Label('H',j))
            M_updated = _symp.symplectic_action(M_updated,'H',[j,])
        
        # Switch in the 1 from the lower column
        instruction_list.append(_Label('H',j))
        M_updated = _symp.symplectic_action(M_updated,'H',[j,])
        
        upperl_c = M_updated[:d,j+d]
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
                M_updated = _symp.symplectic_action(M_updated, 'CNOT', [controlq,targetq])
                    
        # Swithc the 1 back to the lower column
        instruction_list.append(_Label('H',j))            
        M_updated = _symp.symplectic_action(M_updated,'H',[j], optype='row')
                
        # If the matrix has been mapped to the identity, quit the for loop as all
        # remaining instructions added will cancel out.
        if _np.array_equal(M_updated,_np.identity(dim,int)):
            break
            
    assert(_np.array_equal(M_updated,_np.identity(dim,int))), "Compilation has failed!"
            
    j = 1
    n = len(instruction_list)
    while j < n:
        
        if instruction_list[n-j] == instruction_list[n-j-1]:
            del instruction_list[n-j]
            del instruction_list[n-j-1]
            j = j + 2
        else:
            j = j + 1
    
    circuit = _Circuit(gatestring=instruction_list,num_lines=d)
       
    circuit.reverse()
    
    implemented_M, junk = _symp.symplectic_rep_of_clifford_circuit(circuit)        
     
    assert(_np.array_equal(M,implemented_M))        
    
    if depth_compression:
        gate_relations_1q = _symp.single_qubit_clifford_symplectic_group_relations()
        circuit.compress_depth(gate_relations_1q,max_iterations=10000,verbosity=0)
             
        implemented_M, junk = _symp.symplectic_rep_of_clifford_circuit(circuit)     
        assert(_np.array_equal(M,implemented_M))        
        
    return circuit
    
# Todo : write this function - a connectivity-aware GGE algorithm
def advanced_global_gaussian_elimination(s, shortestpaths):
    raise NotImplementedError("This method is not yet written!")
    circuit = None
    return circuit
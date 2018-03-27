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
from ..objects.label import Label as _Label
from ..tools import symplectic as _symp
from ..tools import matrixmod2 as _mtx

def relabel_qubits(circuit,order):
    """
    Todo : put this as a method of the circuit object.
    
    The quantum wire for qubit i becomes
    the quantum wire for qubit order[i]
    """
    relabelled_circuit = _copy.deepcopy(circuit)
    #for i in range(0,circuit.number_of_qubits):
    #    relabelled_circuit.line_items[order[i]] = circuit.line_items[i]
    
    depth = circuit.depth()
    for i in range(0,circuit.number_of_lines):
        for j in range(0,depth):
            gate = circuit.line_items[i][j]
            relabelled_circuit.line_items[order[i]][j] = _Label(gate.name,tuple([order[k] for k in gate.qubits]))
        
    return relabelled_circuit

def compile_clifford(s, p, ds=None, depth_compression=True, algorithms=['DGGE','RGGE'], 
                     costfunction='2QGC', iterations={'RGGE':100}, prefix_paulis=False):
    """
    Compiles a Clifford gate, described by the symplectic matrix s and vector p, into
    a circuit over the specified gateset, or, a standard gateset.
    
    Parameters
    ----------
    s : Todo: fill in

    Returns
    -------
    Circuit
        Todo: fill in
        
    """
    assert(_symp.check_valid_clifford(s,p)), "Input is not a valid Clifford!"
    n = _np.shape(s)[0]//2
    
    if ds is not None:
        assert(n == ds.number_of_qubits), "...."
    
    # Create a circuit that implements a Clifford with symplectic matrix s.
    circuit = compile_symplectic(s, ds=ds, algorithms=algorithms,  costfunction= costfunction, 
                                 iterations=iterations, depth_compression=depth_compression)
    sreps = ds.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
    temp_s, temp_p = _symp.composite_clifford_from_clifford_circuit(circuit,  sreps)
        
    assert(_np.array_equal(s,temp_s))
    assert(_symp.check_valid_clifford(temp_s,temp_p))
        
    s_form = _symp.symplectic_form(n)
    
    if prefix_paulis:
        vec = _np.dot(s_form, (p - temp_p)//2)      
    else:
        vec = _np.dot(s,_np.dot(s_form, (p - temp_p)//2))
    vec = vec % 2
    
    pauli_layer = []
    for q in range(0,n):
        if vec[q] == 0 and vec[q+n] == 0:
            pauli_layer.append(('I',q))
        elif vec[q] == 0 and vec[q+n] == 1:
            pauli_layer.append(('Z',q))
        elif vec[q] == 1 and vec[q+n] == 0:
            pauli_layer.append(('X',q))
        elif vec[q] == 1 and vec[q+n] == 1:
            pauli_layer.append(('Y',q))
    
    pauli_circuit = _Circuit(gatestring=pauli_layer,num_lines=n)
    
    # Only change gate library if we have a DeviceSpec with compilations.
    if ds is not None:
        pauli_circuit.change_gate_library(ds.compilations['absolute'])
    
    if prefix_paulis:
        circuit.prefix_circuit(pauli_circuit)
    else:
        circuit.append_circuit(pauli_circuit)

    sreps = ds.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
    s_out, p_out = _symp.composite_clifford_from_clifford_circuit(circuit, sreps)
    
    assert(_symp.check_valid_clifford(s_out,p_out))
    assert(_np.array_equal(s,s_out)) 
    assert(_np.array_equal(p,p_out)) 
    
    return circuit

def compile_symplectic(s, ds=None, algorithms=['DGGE','RGGE'], costfunction='2QGC', iterations={'RGGE':100},
                       depth_compression=True):    
    """
    
    """                            
    circuits = []
    
    if 'DGGE' in algorithms:
        
        circuit = ordered_global_gaussian_elimination(s, ds=ds, iterations=1, ctype = 'basic', 
                                               depth_compression=depth_compression)
        circuits.append(circuit)      
    
    if 'RGGE' in algorithms:
        
        try:
            iterations = iterations['RGGE']
        except:
            iterations = 100
        circuit = randomized_global_gaussian_elimination(s, ds=ds, ctype = 'basic',
                                                         costfunction=costfunction, iterations = iterations, 
                                                         depth_compression=depth_compression) 
        circuits.append(circuit) 
        
    if 'AGvGE' in algorithms:
        
        # Todo : write this function
        circuit = aaronson_gottesman_on_symplectic(s, ds=ds, cnotmethod = 'GE', 
                                                   depth_compression=depth_compression)   
        circuits.append(circuit) 
        
    if 'AGvPMH' in algorithms:
        
        # Todo : write this function
        circuit = aaronson_gottesman_on_symplectic(s, ds=ds, cnotmethod = 'PMH', 
                                                   depth_compression=depth_compression)   
        circuits.append(circuit) 
        
    if 'iAGvGE' in algorithms:
         
        # Todo : write this function
        circuit = improved_aaronson_gottesman_on_symplectic(s, ds=ds, cnotmethod = 'GE', 
                                                             depth_compression=depth_compression)   
        circuits.append(circuit) 
        
    if 'iAGvPMH' in algorithms:
        
        # Todo : write this function
        circuit = improved_aaronson_gottesman_on_symplectic(s, ds=ds, cnotmethod = 'PMH',
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
                c_cost = circuit.cost(ds.gatecosts)
            else:
                c_cost = costfunction(circuit,ds)
            if c_cost < cost:
                circuit = _copy.deepcopy(c)
                cost = c_cost
                  
    else:
        circuit = circuits[0]
        
    return circuit



def randomized_global_gaussian_elimination(s, ds=None, ctype = 'basic', costfunction='2QGC', 
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
            print("This method is not yet written!")
            circuit = advanced_global_gaussian_elimination(permuted_s ,ds.shortestpaths,depth_compression=depth_compression)
        
        if ctype == 'basic':
            circuit = basic_global_gaussian_elimination(permuted_s ,depth_compression=depth_compression)
        
        circuit = relabel_qubits(circuit,eliminationorder)
        
        if ds is not None:
            circuit.change_gate_library(ds.compilations['paulieq'])
            
        # Find the cost of the circuit
        if costfunction == '2QGC':
            circuit_cost = circuit.twoqubit_gatecount()           
        elif costfunction == 'depth':
            circuit_cost = circuit.depth() 
        elif costfunction == 'gcosts':
            # Todo : make this work - .gatecosts is currently not a property of a DeviceSpec object,
            # and circuit.cost() is currently not a method of circuit.
            circuit_cost = circuit.cost(ds.gatecosts)
        else:
            circuit_cost = costfunction(circuit,ds)
            
        allcosts.append(circuit_cost)
            
        if circuit_cost < lowestcost:
                bestcircuit = _copy.deepcopy(circuit)
                lowestcost = circuit_cost 
    
    if returncosts:
        return bestcircuit, allcosts
    else:
        return bestcircuit
    
def ordered_global_gaussian_elimination(s, ds=None, ctype = 'basic', costfunction='2QGC', iterations=1,
                                        depth_compression=True, returncosts=False):
    
    lowestcost = _np.inf
    n = _np.shape(s)[0] // 2
    allcosts = []
    
    # Get a list of lists of ordered qubits
    orderedqubits = ds.costorderedqubits
    
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
            print("This method is not yet written!")
            circuit = advanced_global_gaussian_elimination(permuted_s ,ds.shortestpaths,depth_compression=depth_compression)
        
        if ctype == 'basic':
            circuit = basic_global_gaussian_elimination(permuted_s ,depth_compression=depth_compression)
        
        circuit = relabel_qubits(circuit,eliminationorder)
        
        if ds is not None:
            circuit.change_gate_library(ds.compilations['paulieq'])
        
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
            circuit_cost = circuit.cost(ds.gatecosts)
        else:
            circuit_cost = costfunction(circuit,ds)
            
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

# Todo : write this function - a cnot-circuit GE algorithm
def basic_cnotcircuit_gaussian_elimination(s):
    circuit = None
    return circuit 

# Todo : write this function - a connectivity-aware cnot-circuit GE algorithm
def advanced_cnotcircuit_gaussian_elimination(s):
    circuit = None
    return circuit
    
    

# Todo : write this function - a connectivity-aware GGE algorithm
def advanced_global_gaussian_elimination(s, shortestpaths):
    circuit = None
    return circuit


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
    
    implemented_M, junk = _symp.composite_clifford_from_clifford_circuit(circuit)        
     
    assert(_np.array_equal(M,implemented_M))        
    
    if depth_compression:
        gate_relations_1q = _symp.single_qubit_clifford_symplectic_group_relations()
        circuit.compress_depth(gate_relations_1q,max_iterations=10000,verbosity=0)
             
        implemented_M, junk = _symp.composite_clifford_from_clifford_circuit(circuit)     
        assert(_np.array_equal(M,implemented_M))        
        
    return circuit

#
# Todo: Update to a function that works on all of s, or simplify (with this functinality
# inside the CNOT-circuit compiler) and put inside the matrix tools .py
#
def convert_invertible_to_echelon_form(matrixin, optype='row', position='upper', 
                                       paired_matrix=False, pmatrixin=None):
    """
    Todo: docstring
    """
    
    success = True
    matrix = matrixin.copy()
    d = len(matrix[0,:])
    
    if paired_matrix:
        pmatrix = pmatrixin.copy()
    
    instruction_list = []
    
    if optype == 'row':
    
        for i in range(0,d):

            if matrix[i,i] != 1:
                swap_quit = False
                j = i + 1
                while not swap_quit:
                
                    if j >= d:
                        
                        # Quit the function if the matrix is not invertable
                        success = False
                        if paired_matrix:
                            return matrix, success, instruction_list, pmatrix
                        else:
                            return matrix, success, instruction_list
                    
                    else:    
                        if matrix[j,i] == 1:
                            
                            jrow = matrix[j,:].copy()
                            irow = matrix[i,:].copy()
                            matrix[i,:] = jrow
                            matrix[j,:] = irow
                            
                            if paired_matrix:
                                pjrow = pmatrix[j,:].copy()
                                pirow = pmatrix[i,:].copy()
                                pmatrix[i,:] = pjrow
                                pmatrix[j,:] = pirow
                            
                            instruction_list.append(_Label('CNOT',(i,j)))
                            instruction_list.append(_Label('CNOT',(j,i)))
                            instruction_list.append(_Label('CNOT',(i,j)))
                            swap_quit = True
                        else:
                            j += 1
                
            for j in range(i+1,d):                
                if matrix[j,i] == 1:
                    # Add the ith row to row j, mod 2.
                    matrix[j,:] = matrix[i,:] ^ matrix[j,:]
                    if paired_matrix:
                        pmatrix[i,:] = pmatrix[i,:] ^ pmatrix[j,:]                       
                    if position == 'upper':
                        instruction_list.append(_Label('CNOT',(i,j)))
                    if position == 'lower':
                        instruction_list.append(_Label('CNOT',(j,i)))                        
                    
    if optype == 'column':
        
        for i in range(0,d):
            
            if matrix[i,i] != 1:
                
                swap_quit = False
                j = i + 1
                
                while not swap_quit:
                
                    if j >= d:
                        # Quit the function if the matrix is not invertable
                        success = False
                        if paired_matrix:
                            return matrix, success, instruction_list, pmatrix
                        else:
                            return matrix, success, instruction_list

                    else:    
                        
                        if matrix[i,j] == 1:
                            
                            jcol = matrix[:,j].copy()
                            icol = matrix[:,i].copy()
                            matrix[:,i] = jcol
                            matrix[:,j] = icol
                            instruction_list.insert(0,_Label('CNOT',(i,j)))
                            instruction_list.insert(0,_Label('CNOT',(j,i)))
                            instruction_list.insert(0,_Label('CNOT',(i,j)))
                            
                            if paired_matrix:
                                pjcol = pmatrix[:,j].copy()
                                picol = pmatrix[:,i].copy()
                                pmatrix[:,i] = pjcol
                                pmatrix[:,j] = picol
                                
                            swap_quit = True

                        else:
                            j += 1
                
            for j in range(i+1,d):
                
                if matrix[i,j] == 1:
                    
                    # Add the ith column to column j, mod 2.
                    matrix[:,j] = matrix[:,i] ^ matrix[:,j]
                    if paired_matrix:
                        pmatrix[:,j] = pmatrix[:,i] ^ pmatrix[:,j]
                        
                    instruction_list.insert(0,_Label('CNOT',(i,j)))
        
    if paired_matrix:
        return matrix, success, instruction_list, pmatrix
    else:
        return matrix, success, instruction_list

#
# Todo: Update to a function that works on all of s.
#
def convert_invertible_to_reduced_echelon_form(matrixin,optype='row',position='upper',paired_matrix=False,
                                               pmatrixin=None):
    
    # Convert to row echelon form
    
    if paired_matrix:
        matrix, success, instruction_list, pmatrix = convert_invertible_to_echelon_form(matrixin, optype=optype, 
                                                                           position= position,paired_matrix=True,
                                                                           pmatrixin=pmatrixin)
    else:
        matrix, success, instruction_list = convert_invertible_to_echelon_form(matrixin, optype=optype, 
                                                                           position = position)
    
    if not success:
        if paired_matrix:
            return matrix, success, instruction_list, pmatrix
        else:
            return matrix, success, instruction_list
    
    d = len(matrix[0,:])
    
    for i in range(0,d):
        
        k = d - 1 - i 
        
        if matrix[k,k] == 1:
            for j in range(0,k):
                l = k - 1 - j
                
                if optype == 'row':
                
                    if matrix[l,k] == 1:
                        # Add the kth row to row l, mod 2.
                        matrix[l,:] = matrix[l,:] ^ matrix[k,:]
                        if position == 'upper':
                            instruction_list.append(_Label('CNOT',(k,l)))
                        if position == 'lower':
                            instruction_list.append(_Label('CNOT',(l,k)))
                        if paired_matrix:
                            pmatrix[k,:] = pmatrix[l,:] ^ pmatrix[k,:]
                        #print(matrix)
                        
                if optype == 'column':
                
                    if matrix[k,l] == 1:
                        # Add the kth colmun to column l, mod 2.
                        matrix[:,l] = matrix[:,l] ^ matrix[:,k]
                        # When acting as a column, we need the target to be ....
                        instruction_list.insert(0,_Label('CNOT',(k,l)))
                        if paired_matrix:
                            pmatrix[:,l] = pmatrix[:,l] ^ pmatrix[:,k]
                        #print(matrix)
                        
    if paired_matrix:
        return matrix, success, instruction_list, pmatrix
    else:
        return matrix, success, instruction_list

#def CNOT_circuit_from_Gaussian_elimination(s,connectivity='complete'):
#    
#    if connectivity is 'complete':
#        connectivity = _np.ones((d,d),int) - _np.identity(d,int)
 
def stabilizer_measurement_preparation_circuit(s,p,ds,iterations=1,relations=None):
    """
    Compiles a circuit that, when followed by a projection onto <0,0,...|,
    is equivalent to implementing the Clifford C defined by the pair (s,p) followed by a
    projection onto <0,0,..|. I.e., it produces a circuit that implements some 
    Clifford C' such that <0,0,0,...|C = <0,0,0,...|C' for any computational basis state. This
    could easily be improved to allow for any computational basis state.
 
    """
    assert(_symp.check_valid_clifford(s,p)), "The input s and p are not a valid clifford."

    n = len(s[0,:])//2
    sin, pin = _symp.inverse_clifford(s,p)
    
    min_twoqubit_gatecount = _np.inf
    
    #Import the single-qubit Cliffords up-to-Pauli algebra
    gate_relations_1q = _symp.single_qubit_clifford_symplectic_group_relations()
    
    failcount = 0
    i = 0
    # Todo : remove this try-except method once compiler always works.
    while i < iterations:
        try:
            trialcircuit, trialcheck_circuit = symplectic_as_conditional_clifford_circuit_over_CHP(sin,ds,returnall=True)
            i += 1
            trialcircuit.reverse()
            # Do the depth-compression *after* the circuit is reversed
            trialcircuit.compress_depth(gate_relations_1q,max_iterations=1000,verbosity=0)
            trialcircuit.change_gate_library(ds.compilations['paulieq'])
            twoqubit_gatecount = trialcircuit.twoqubit_gatecount()
            if twoqubit_gatecount  < min_twoqubit_gatecount :
                circuit = _copy.deepcopy(trialcircuit)
                check_circuit = _copy.deepcopy(trialcheck_circuit)
                min_twoqubit_gatecount = twoqubit_gatecount
        except:
            failcount += 1
            
        assert(failcount <= 5*iterations), "Randomized compiler is failing unexpectedly often. Perhaps input DeviceSpec is not valid or does not contain the neccessary information."
         
    #check_circuit.reverse()
    #check_circuit.change_gate_library(ds.compilations['paulieq'])

    if relations is not None:
        # Do more depth-compression on the chosen circuit. Todo: This should used something already
        # constructed in DeviceSpec, instead of this ad-hoc method.
        sreps = ds.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
        sprecompression, junk =  _symp.composite_clifford_from_clifford_circuit(circuit,sreps)
        circuit.compress_depth(relations,max_iterations=1000,verbosity=0)    
        spostcompression, junk =  _symp.composite_clifford_from_clifford_circuit(circuit,sreps)
        assert(_np.array_equal(sprecompression,spostcompression)), "Gate relations are incorrect!"
    
    check_circuit.reverse()
    #check_circuit.change_gate_library(ds.compilations['paulieq'])
    check_circuit.prefix_circuit(circuit)

    sreps = ds.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
    sreps['CNOT'] = (_np.array([[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1]],int), _np.array([0,0,0,0],int))
    
    implemented_scheck, implemented_pcheck = _symp.composite_clifford_from_clifford_circuit(check_circuit, sreps)
    
    implemented_sin_check, implemented_pin_check =  _symp.inverse_clifford(implemented_scheck, implemented_pcheck) 
    
    assert(_np.array_equal(implemented_scheck[0:n,:],s[0:n,:]))
    
    # Find the needed Pauli at the start    
    pinaltered = pin.copy()
    pinaltered = _symp.construct_valid_phase_vector(implemented_sin_check,pinaltered)      
    
    s_form = _symp.symplectic_form(n)
    vec = _np.dot(implemented_sin_check,_np.dot(s_form, (pinaltered - implemented_pin_check)//2))
    vec = vec % 2
    
    pauli_layer = []
    for q in range(0,n):
        if vec[q] == 0 and vec[q+n] == 0:
            pauli_layer.append(_Label('I',q))
        elif vec[q] == 0 and vec[q+n] == 1:
            pauli_layer.append(_Label('Z',q))
        elif vec[q] == 1 and vec[q+n] == 0:
            pauli_layer.append(_Label('X',q))
        elif vec[q] == 1 and vec[q+n] == 1:
            pauli_layer.append(_Label('Y',q))
            
    paulicircuit = _Circuit(gatestring=pauli_layer,num_lines=n)
    paulicircuit.change_gate_library(ds.compilations['absolute'])
    circuit.prefix_circuit(paulicircuit)
    circuit.compress_depth(max_iterations=10,verbosity=0) 
    
    return circuit
   
    
def stabilizer_state_preparation_circuit(s,p,ds,iterations=1,relations=None):
    
    assert(_symp.check_valid_clifford(s,p)), "The input s and p are not a valid clifford."
    
    n = len(s[0,:])//2
    min_twoqubit_gatecount = _np.inf
    
    #Import the single-qubit Cliffords up-to-Pauli algebra
    gate_relations_1q = _symp.single_qubit_clifford_symplectic_group_relations()
    
    failcount = 0
    i = 0
    while i < iterations:
        try:
            trialcircuit, trialcheck_circuit = symplectic_as_conditional_clifford_circuit_over_CHP(s,ds,returnall=True)
            i += 1
            #
            # Todo: work out how much this all makes sense.
            #
            # Do the depth-compression *before* changing gate library            
            trialcircuit.compress_depth(gate_relations_1q,max_iterations=1000,verbosity=0)            
            trialcircuit.change_gate_library(ds.compilations['paulieq'])        
            twoqubit_gatecount = trialcircuit.twoqubit_gatecount()
            if twoqubit_gatecount  < min_twoqubit_gatecount :
                circuit = _copy.deepcopy(trialcircuit)
                check_circuit = _copy.deepcopy(trialcheck_circuit)
                min_twoqubit_gatecount = twoqubit_gatecount
        except:
            failcount += 1
        
        assert(failcount <= 5*iterations), "Randomized compiler is failing unexpectedly often. Perhaps input DeviceSpec is not valid or does not contain the neccessary information."
            
    if relations is not None:
        # Do more depth-compression on the chosen circuit. Todo: This should used something already
        # constructed in DeviceSpec, instead of this ad-hoc method.
        sreps = ds.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
        sprecompression, junk =  _symp.composite_clifford_from_clifford_circuit(circuit,sreps)
        circuit.compress_depth(relations,max_iterations=1000,verbosity=0)    
        spostcompression, junk =  _symp.composite_clifford_from_clifford_circuit(circuit,sreps)
        assert(_np.array_equal(sprecompression,spostcompression)), "The gate relations provided are incorrect!"
        
    #check_circuit.change_gate_library(ds.compilations['paulieq'])
    check_circuit.append_circuit(circuit)
    
    
    # Add CNOT into the dictionary, in case it isn't there.
    sreps = ds.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
    sreps2 = sreps.copy()
    sreps2['CNOT'] = (_np.array([[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1]],int), _np.array([0,0,0,0],int))
    
    implemented_s, implemented_p = _symp.composite_clifford_from_clifford_circuit(circuit, sreps)
    implemented_scheck, implemented_pcheck = _symp.composite_clifford_from_clifford_circuit(check_circuit, sreps2)
    
    # Find the needed Pauli at the end
    s_form = _symp.symplectic_form(n)
    
    paltered = p.copy()
    paltered[0:n] = _np.zeros(n,int)
    paltered = _symp.construct_valid_phase_vector(implemented_scheck,paltered)                                                  
    vec = _np.dot(implemented_scheck,_np.dot(s_form, (paltered - implemented_pcheck)//2))
    vec = vec % 2
    
    pauli_layer = []
    for q in range(0,n):
        if vec[q] == 0 and vec[q+n] == 0:
            pauli_layer.append(_Label('I',q))
        elif vec[q] == 0 and vec[q+n] == 1:
            pauli_layer.append(_Label('Z',q))
        elif vec[q] == 1 and vec[q+n] == 0:
            pauli_layer.append(_Label('X',q))
        elif vec[q] == 1 and vec[q+n] == 1:
            pauli_layer.append(_Label('Y',q))

            
    paulicircuit = _Circuit(gatestring=pauli_layer,num_lines=n)
    paulicircuit.change_gate_library(ds.compilations['absolute'])
    circuit.append_circuit(paulicircuit)
    circuit.compress_depth(max_iterations=10,verbosity=0)
    
    return circuit
   

def symplectic_as_conditional_clifford_circuit_over_CHP(s,ds=None,returnall=False):
    """
    
    """
    
    def stage1(s):
    
        h_list = []
        
        n = len(s[0,:])//2
        
        MU = s[0:n,n:2*n].copy()
        ML = s[n:2*n,n:2*n].copy()
        MU_full_rank = False
        iteration = 0
        success = False
        
        while not success:
            
            #print(iteration,end='')
            iteration += 1
            MUref, success, instructions = convert_invertible_to_echelon_form(MU)
            if not success:
                hqubit = _np.random.randint(0,n)
                MUhqubitrow = MU[hqubit,:].copy()
                MLhqubitrow = ML[hqubit,:].copy()
                MU[hqubit,:] = MLhqubitrow
                ML[hqubit,:] = MUhqubitrow
                
                if hqubit in h_list:
                    del h_list[h_list.index(hqubit)]
                else:
                    h_list.append(hqubit)
                
            if iteration > 100*n:
                break
                
        sout = s.copy()
        for hqubit in h_list:
            Shqubit_upperrow = sout[hqubit,:].copy()
            Shqubit_lowerrow = sout[n+hqubit,:].copy()
            sout[hqubit,:] = Shqubit_lowerrow
            sout[n+hqubit,:] = Shqubit_upperrow
            
        instructions = [_Label('H',i) for i in h_list]
        
        return sout, instructions, success
    
    def stage2(s):
    
        sout = s.copy()
        n = len(s[0,:])//2
        
        MU = s[0:n,n:2*n].copy()
        ML = s[n:2*n,n:2*n].copy()
        
        instructions = []
        MUout, success, instructions, MLout = convert_invertible_to_reduced_echelon_form(MU, optype='column', 
                                                                                         position='upper', 
                                                                                         paired_matrix=True,
                                                                                         pmatrixin=ML)
        #print(MUout)
        assert(success)
        sout[0:n,n:2*n] = MUout.copy()
        sout[n:2*n,n:2*n] = MLout.copy()
        
        return sout, instructions, success
    
    def stage3(s):
    
        sout = s.copy()
        n = len(s[0,:])//2
        instructions = []
        
        matrix = s[n:2*n,n:2*n].copy()
        
        for i in range(0,n):
            
            if matrix[i,i] != 1:
                matrix[i,i] = 1
                instructions.append(_Label('P',i))
                sout[i+n,i+n] = sout[i+n,i+n] ^ 1
                    
            for j in range(i+1,n):                
                if matrix[j,i] == 1:
                    # Add the ith row to row j, mod 2.
                    matrix[j,:] = matrix[i,:] ^ matrix[j,:]
        #print(matrix)
                            
        return sout, instructions, True

    
    def stage4(s):
    
    
        n = _np.shape(s)[0]//2
        sout = s.copy()    
        ML = s[n:2*n,n:2*n].copy()
        Mdecomposition = _mtx.albert_factor(ML) #_mtx.TPdecomposition(ML)
        # This was previously a bool returned by the factorization algorithm. Currently
        # just being set to True
        success = True
        
        if not success:
            return sout, [], False
        
        junk1, success, instructions = convert_invertible_to_reduced_echelon_form(Mdecomposition.T)
        instructions.reverse()
        circuit = _Circuit(gatestring=instructions,num_lines=n)
        s_of_circuit, p = _symp.composite_clifford_from_clifford_circuit(circuit)
        sout = _mtx.dotmod2(s_of_circuit,sout)

        
        MU = sout[0:n,n:2*n].copy()
        ML = sout[n:2*n,n:2*n].copy()
        success = _np.array_equal(MU,ML)
    
        return sout, instructions, success  
    
    def stage5(s):
    
        sout = s.copy()
        n = _np.shape(s)[0]//2
        MU = s[0:n,n:2*n].copy()
        ML = s[n:2*n,n:2*n].copy()
    
        MUout, success, instructions, MLout = convert_invertible_to_reduced_echelon_form(MU, optype='column', 
                                                                                     position='upper', 
                                                                                     paired_matrix=True,
                                                                                     pmatrixin=ML)
        sout[0:n,n:2*n] = MUout
        sout[n:2*n,n:2*n] = MLout
    
        return sout, instructions, success
    
    def stage6(s):
        """
        ....
        """
        sout = s.copy()
        n = _np.shape(s)[0]//2
        sout[n:2*n,n:2*n] = _np.zeros((n,n),int)
        instructions = [_Label('P',i) for i in range(0,n)]
        
        return sout, instructions, True

    def stage7(s):
    
        sout = s.copy()
        n = _np.shape(s)[0]//2
        MU = s[0:n,n:2*n].copy()
        ML = s[n:2*n,n:2*n].copy()  
        sout[0:n,n:2*n] = ML
        sout[n:2*n,n:2*n] = MU
        
        instructions = [_Label('H',i) for i in range(0,n)]
        
        return sout, instructions, True
    
    lhs_instructions = []
    rhs_instructions = []
    
    # Hadamard circuit from the LHS to make the upper right hand (URH) submatrix of s invertible
    s1, instructions1, success = stage1(s)
    instructions1.reverse()
    #print(s1)
    
    # CNOT circuit from the RHS to map the URH submatirx of s to I.
    s2, instructions2, success = stage2(s1)
    instructions2.reverse()
    assert(success)
    #print(s2)
    
    # Phase circuit from the LHS to make the lower right hand (LRH) submatrix of s invertible
    s3, instructions3, success = stage3(s2)
    instructions3.reverse()
    assert(success)
    #print(s3)
    
    # CNOT circuit from the LHS to map the URH and LRH submatrices of s to the same invertible matrix M
    s4, instructions4, success = stage4(s3)
    instructions4.reverse()
    assert(success)
    #print(s4)
    
    # CNOT circuit from the RHS to map the URH and LRH submatrices of s from M to I.
    s5, instructions5, success = stage5(s4)
    instructions5.reverse()
    assert(success)
    #print(s5)
    
    # Phase circuit from the LHS to map the LRH submatrix of s to 0.
    s6, instructions6, success = stage6(s5)
    instructions6.reverse()
    assert(success)
    #print(s6)
    
    # Hadamard circuit from the LHS to swap the LRH and URH matrices of s, (mapping them to I and 0 resp.,)
    s7, instructions7, success = stage7(s6)
    instructions7.reverse()
    assert(success)
    #print(s7)
    
    main_instructions = instructions7 + instructions6 + instructions4 + instructions3 + instructions1
    precircuit_instructions = instructions2 + instructions5
    
    n = len(s[0,:])//2
    circuit = _Circuit(gatestring=main_instructions,num_lines=n)
    implemented_s, implemented_p = _symp.composite_clifford_from_clifford_circuit(circuit)
            
    # Check for success
    #check_circuit = _Circuit(gatestring=precircuit_instructions,num_line=n)
    #check_circuit.append_circuit(circuit)
    CNOT_pre_circuit = _Circuit(gatestring=precircuit_instructions,num_lines=n)
    check_circuit = _copy.deepcopy(CNOT_pre_circuit)
    check_circuit.append_circuit(circuit)
    scheck, pcheck = _symp.composite_clifford_from_clifford_circuit(check_circuit)
    assert(_np.array_equal(scheck[:,n:2*n],s[:,n:2*n])), "Compiler has failed!"

    if returnall:
        #return circuit, check_circuit
        return circuit, CNOT_pre_circuit
    else:
        #return circuit, check_circuit
        return circuit, CNOT_pre_circuit

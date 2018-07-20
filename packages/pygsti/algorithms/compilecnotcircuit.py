""" CNOT circuit compilation routines """
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

def compile_cnot_circuit(s, pspec, subsetQs=None, algorithm='COGE', aargs=[]):
    """
    - UDGE is user-ordering Gaussian elimination.
    - ROGE is random-ordered GE.
    - COGE is connectivity-ordered GE.
    """

    if subsetQs is not None:
            qubits = list(subsetQs)
    else:
        qubits = pspec.qubit_labels

    # The qubit elimination order is just that specified by the aargs list.
    if algorithm == 'UOGE':
        set(aargs[0]) == set(qubits)
        qubitorder = _copy.copy(aargs[0])

        circuit = compile_cnot_circuit_with_connectivity_adjusted_gaussian_elimination(s, pspec, qubitorder)
        
    # The qubit qubit elimination order is by from least to most connected qubit
    elif algorithm == 'COGE': 
        
        # todo -- make this work with subsetQs
        distances = pspec.qubitgraph.shortest_path_distance_matrix()
        shortestpathmatrix = pspec.qubitgraph.shortest_path_predecessor_matrix()
        costs = _np.sum(distances,axis=0)
        qubitorder = []
        for k in range(0,n):
            # Find the most-expensive qubit
            i = _np.argmax(costs)
            qubitorder.append(i)
            costs[i] = -1

        circuit = compile_cnot_circuit_with_connectivity_adjusted_gaussian_elimination(s, pspec, qubitorder)
           
    # The qubit elimination order is random.
    elif std_ordering == 'ROGE':
        #print('random')
        qubitorder = _copy.copy(qubits)
        _np.random.shuffle(qubitorder)

        # Allow an iterations thing here, that is specified in aargs.

        circuit = compile_cnot_circuit_with_connectivity_adjusted_gaussian_elimination(s, pspec, qubitorder)

    else: raise ValueError("The choice of algorithm is invalid!")

        
# Todo : write this function - a basic cnot-circuit GE algorithm (can probably be done using the code in compilerstabilizerstate)
def compile_cnot_circuit_with_basic_gaussian_elimination(s, pspec):
    circuit = None
    return circuit 

# todo : this function currently doesn't return a circuit -- that should be changed.
def compile_cnot_circuit_with_connectivity_adjusted_gaussian_elimination(mcnot, pspec, ordering):
    """
    docstring todo.
    """
    qubitshuffle=False
    assert(qubitshuffle == False), "qubitshuffle set to True is currently not working"
    n = pspec.number_of_qubits
    rowaction_instructionlist = []
    columnaction_instructionlist = []
    remaining_qubits = list(range(n))
    sout = mcnot.copy()
    distances = pspec.qubitgraph.shortest_path_distance_matrix()
    shortestpathmatrix = pspec.qubitgraph.shortest_path_predecessor_matrix()

    #print(qubitorder)
    
    for k in range(0,n):

        # Find the most-expensive qubit
        i = qubitorder[0]
        qindex = i
        
        distances_to_qubit_i = distances[:,i].copy()  
        
        if sout[i,i] == 0:
            found = False
            dis = list(distances_to_qubit_i.copy())
            counter = 0
            while not found:
                counter += 1
                if counter>n:
                    print('Fail!')
                    break
                ii = dis.index(min(dis))
                dis[ii] = 999999
                if ii in remaining_qubits and ii != i:
                    # Only do this option if qubitshuffle
                    if sout[ii,ii] == 1 and qubitshuffle:
                        found = True
                        qindex = ii
                    elif sout[ii,i] == 1:
                        rowaction_instructionlist.append(_Label('CNOT',(ii,i)))
                        sout[i,:] = sout[i,:] ^ sout[ii,:]                                        
                        found = True
                        qindex = i
                                                         
                    elif sout[i,ii] == 1:
                        columnaction_instructionlist.append(_Label('CNOT',(i,ii)))
                        sout[:,i] = sout[:,i] ^ sout[:,ii]
                        found = True
                        qindex = i
                        
        i = qindex
        del qubitorder[qubitorder.index(qindex)]
                                                        
        assert(sout[i,i]==1)

        qubits = _copy.copy(remaining_qubits)
        del qubits[qubits.index(i)]
        

        while len(qubits) > 0:

            # complement of `remaining_qubits` = "elimintated qubits"
            eliminated_qubits = set(range(n)) - set(remaining_qubits)
            
            # Find the most distant remaining qubit
            farthest_qubit_found = False
            while not farthest_qubit_found:
                farthest_qubit = _np.argmax(distances_to_qubit_i)
                if farthest_qubit in qubits:
                    farthest_qubit_found = True
                else:
                    distances_to_qubit_i[farthest_qubit] = -1

            # Check to see if that qubits needs to a gate on it or not.         
            # Qubit does need to have a gate on it.
            #print(sout)
            if sout[farthest_qubit,i] == 1:
                #print('  - Farthest qubit needs row eliminating...')

                # Find the shortest path out from i to farthest_qubit, and do CNOTs to make that all 1s.
                if pspec.qubitgraph.shortest_path_intersect(farthest_qubit, i, eliminated_qubits):
                    # shortest path from farthest_qubit -> i includes eliminated qubits
                    rowaction_instructionlist.append(_Label('CNOT',(i,farthest_qubit)))
                    sout[farthest_qubit,:] =  sout[i,:] ^ sout[farthest_qubit,:]
                else:
                    for nextqubit, currentqubit in reversed(pspec.qubitgraph.shortest_path_edges(farthest_qubit, i)):
                        # EGN: reverse-iter to follow Tim's prior implementation
                        if sout[nextqubit,i] == 0:
                            rowaction_instructionlist.append(_Label('CNOT',(currentqubit,nextqubit)))
                            sout[nextqubit,:] = sout[nextqubit,:] ^ sout[currentqubit,:]

                    # Set the farthest qubit s-matrix element to 0 (but don't change the others)
                    quse = shortestpathmatrix[i,farthest_qubit]
                    #print(quse,i,farthest_qubit)
                    rowaction_instructionlist.append(_Label('CNOT',(quse,farthest_qubit)))
                    sout[farthest_qubit,:] =  sout[quse,:] ^ sout[farthest_qubit,:]

                    for nextqubit, currentqubit in reversed(pspec.qubitgraph.shortest_path_edges(i,farthest_qubit)):
                        # EGN: reverse-iter to follow Tim's prior implementation
                        if currentqubit not in remaining_qubits:
                            rowaction_instructionlist.append(_Label('CNOT',(nextqubit,currentqubit)))
                            sout[currentqubit,:] = sout[nextqubit,:] ^ sout[currentqubit,:]

            if sout[i,farthest_qubit] == 1:                
                # Find the shortest path out from i to farthest_qubit, and do CNOTs to make that
                # all 1s.
                if pspec.qubitgraph.shortest_path_intersect(farthest_qubit, i, eliminated_qubits):
                    # shortest path from farthest_qubit -> i includes eliminated qubits
                    columnaction_instructionlist.append(_Label('CNOT',(farthest_qubit,i)))
                    sout[:,farthest_qubit] = sout[:,i] ^ sout[:,farthest_qubit]
                else:
                    
                    for nextqubit, currentqubit in reversed(pspec.qubitgraph.shortest_path_edges(farthest_qubit, i)):
                        # EGN: reverse-iter to follow Tim's prior implementation
                        if sout[i,nextqubit] == 0:
                            columnaction_instructionlist.append(_Label('CNOT',(nextqubit,currentqubit)))
                            sout[:,nextqubit] = sout[:,nextqubit] ^ sout[:,currentqubit]

                    # Set the farthest qubit s-matrix element to 0 (but don't change the others)
                    quse = shortestpathmatrix[i,farthest_qubit]
                    columnaction_instructionlist.append(_Label('CNOT',(farthest_qubit,quse)))
                    sout[:,farthest_qubit] =  sout[:,quse] ^ sout[:,farthest_qubit]

                    for nextqubit, currentqubit in reversed(pspec.qubitgraph.shortest_path_edges(i,farthest_qubit)):
                        # EGN: reverse-iter to follow Tim's prior implementation
                        if currentqubit not in remaining_qubits:
                            columnaction_instructionlist.append(_Label('CNOT',(currentqubit,nextqubit)))
                            sout[:,currentqubit] = sout[:,nextqubit] ^ sout[:,currentqubit]

            # Delete the farthest qubit from the list -- it has been eliminated for this column.    
            del qubits[qubits.index(farthest_qubit)]

            # And set it's distance to -1, so that in the next round we find the next farthest qubit.
            distances_to_qubit_i[farthest_qubit] = -1

        # Remove from the remaining qubits list
        del remaining_qubits[remaining_qubits.index(i)]

    rowaction_instructionlist.reverse()
    columnaction_instructionlist
    full_instructionlist =  columnaction_instructionlist + rowaction_instructionlist

    return full_instructionlist



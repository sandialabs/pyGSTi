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


# Todo : write this function - a cnot-circuit GE algorithm
def basic_cnotcircuit_gaussian_elimination(s):
    circuit = None
    return circuit 

# Todo : write this function - a connectivity-aware cnot-circuit GE algorithm
def advanced_cnotcircuit_gaussian_elimination(s):
    circuit = None
    return circuit

def compile_CNOT_circuit(mcnot,pspec,custom_ordering=None,std_ordering='connectivity',
                        qubitshuffle=False):
    ##
    #
    # Tim todo : sort out this function. It's currently a hacky mess b/c of Tim/Erik's
    # code changes, and it might not always work.
    #
    #
    
    assert(qubitshuffle == False), "qubitshuffle set to True is currently not working"
    n = pspec.number_of_qubits
    rowaction_instructionlist = []
    columnaction_instructionlist = []
    remaining_qubits = list(range(n))
    sout = mcnot.copy()
    distances = pspec.qubitgraph.shortest_path_distance_matrix()
    shortestpathmatrix = pspec.qubitgraph.shortest_path_predecessor_matrix()
    
    if custom_ordering is not None:
        qubitorder = _copy.copy(custom_ordering)
    
    else:
        if std_ordering == 'connectivity':
            #
            # todo : this will not behave as expected if costs are anything other than the
            # qubit connectivity. So fix that
            #
            costs = _np.sum(distances,axis=0)
            qubitorder = []
            for k in range(0,n):

                # Find the most-expensive qubit
                i = _np.argmax(costs)
                qubitorder.append(i)
                costs[i] = -1

        elif std_ordering == 'random':
            #print('random')
            qubitorder = list(range(n))
            _np.random.shuffle(qubitorder)

        else:
            print("std_ordering is not understood!")
    
    #print(qubitorder)
    
    for k in range(0,n):

        # Find the most-expensive qubit
        i = qubitorder[0]
        qindex = i
        
        #print('----- ROUND {} ------'.format(k))
        #print(sout)
        #print('- Remaing qubits are:', remaining_qubits)
        #print('- Qubit scheduled to be eliminated:', i)
        
        distances_to_qubit_i = distances[:,i].copy()  
        
        if sout[i,i] == 0:
            #print('- s[{},{}] = 0, so addressing this...'.format(i,i))
            #print(sout)
            #print(remaining_qubits)
            found = False
            dis = list(distances_to_qubit_i.copy())
            counter = 0
            while not found:
                counter += 1
                if counter>n:
                    print('Fail!')
                    #print("Remaining qubits:",remaining_qubits)
                    #print(sout)
                    #print(i)
                    #print(k+1)
                    break
                #print(dis)
                ii = dis.index(min(dis))
                dis[ii] = 999999
                if ii in remaining_qubits and ii != i:
                    #print('     - Considering using qubit {} to rectify this'.format(ii))
                    # Only do this option if qubitshuffle
                    if sout[ii,ii] == 1 and qubitshuffle:
                        #print('     - Using qubit {} instead of qubit {}'.format(ii,i))
                        found = True
                        qindex = ii
                    elif sout[ii,i] == 1:
                        #print('     - Using row-action qubit {}'.format(ii))
                        rowaction_instructionlist.append(_Label('CNOT',(ii,i)))
                        #rowaction_instructionlist.append(_Label('CNOT',(i,ii)))
                        #rowaction_instructionlist.append(_Label('CNOT',(ii,i)))  

                        sout[i,:] = sout[i,:] ^ sout[ii,:]
                        #sout[ii,:] = sout[i,:] ^ sout[ii,:]
                        #sout[i,:] = sout[i,:] ^ sout[ii,:]
                                         
                        found = True
                        qindex = i
                                                         
                    elif sout[i,ii] == 1:
                        #print('     - Using col-action qubit {}'.format(ii))
                        columnaction_instructionlist.append(_Label('CNOT',(i,ii)))
                        #rowaction_instructionlist.append(_Label('CNOT',(ii,i)))
                        #rowaction_instructionlist.append(_Label('CNOT',(i,ii)))
                        sout[:,i] = sout[:,i] ^ sout[:,ii]
                        #sout[:,ii] = sout[:,i] ^ sout[:,ii]
                        #sout[:,i] = sout[:,i] ^ sout[:,ii]
                        found = True
                        qindex = i
                        
        #while sout[i,i] == 0:
        #    qindex += 1
        #i = qubitorder[qindex]
        i = qindex
        del qubitorder[qubitorder.index(qindex)]
        #print('- Qubit to be eliminated:'.format(i))
                                                        
        assert(sout[i,i]==1)
        #print("WARNING s[i,i] == 0 !!!!!!!!!!!!")
        #    #sout[i,i] = 1
        #print(sout)
        
        #print(remaining_qubits)


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

            #print("  - Farthest qubit is", farthest_qubit)
            # Check to see if that qubits needs to a gate on it or not.         
            # Qubit does need to have a gate on it.
            #print(sout)
            if sout[farthest_qubit,i] == 1:
                #print('  - Farthest qubit needs row eliminating...')

                # Find the shortest path out from i to farthest_qubit, and do CNOTs to make that
                # all 1s.
                if pspec.qubitgraph.shortest_path_intersect(farthest_qubit, i, eliminated_qubits):
                    # shortest path from farthest_qubit -> i includes eliminated qubits
                    #print("  - Resorting to CNOT via SWAPs!!!")
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


                #OLD CODE - TODO REMOVE - kept b/c we still need to check whether this works TIM
                #currentqubit = i       
                #while currentqubit != farthest_qubit:
                #    nextqubit = pspec.shortestpath[farthest_qubit,currentqubit]
                #    if nextqubit not in remaining_qubits:
                #        includes_eliminated_qubits = True
                #        #print("  - Resorting to CNOT via SWAPs!!!")
                #        break
                #    currentqubit = nextqubit
                #
                #if not includes_eliminated_qubits:
                #    currentqubit = i       
                #    while currentqubit != farthest_qubit:
                #
                #        nextqubit = pspec.shortestpath[farthest_qubit,currentqubit]
                #        #print(currentqubit,nextqubit)
                #
                #        if sout[nextqubit,i] == 0:
                #            rowaction_instructionlist.append(_Label('CNOT',(currentqubit,nextqubit)))
                #            sout[nextqubit,:] = sout[nextqubit,:] ^ sout[currentqubit,:]
                #        currentqubit = nextqubit
                #
                #    assert(currentqubit == farthest_qubit)
                #
                #    # Set the farthest qubit s-matrix element to 0 (but don't change the others)
                #    quse = pspec.shortestpath[i,farthest_qubit]
                #    #print(quse,i,farthest_qubit)
                #    rowaction_instructionlist.append(_Label('CNOT',(quse,farthest_qubit)))
                #    sout[farthest_qubit,:] =  sout[quse,:] ^ sout[farthest_qubit,:]
                #
                #    currentqubit = farthest_qubit
                #    while currentqubit != i:
                #
                #        nextqubit = pspec.shortestpath[i,currentqubit]
                #
                #        if currentqubit not in remaining_qubits:
                #            rowaction_instructionlist.append(_Label('CNOT',(nextqubit,currentqubit)))
                #            sout[currentqubit,:] = sout[nextqubit,:] ^ sout[currentqubit,:]
                #
                #        currentqubit = nextqubit
                #              
                #else:
                #    rowaction_instructionlist.append(_Label('CNOT',(i,farthest_qubit)))
                #    sout[farthest_qubit,:] =  sout[i,:] ^ sout[farthest_qubit,:]                    
                #     
                ##print(sout)


            if sout[i,farthest_qubit] == 1: # TIM - this condition looks the same as one above... (?)
                #print('  - Farthest qubit needs column eliminating...')
                
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
                            

                #OLD - TODO REMOVE
                #includes_eliminated_qubits = False
                #currentqubit = i       
                #while currentqubit != farthest_qubit:
                #    nextqubit = pspec.shortestpath[farthest_qubit,currentqubit]
                #    if nextqubit not in remaining_qubits:
                #        includes_eliminated_qubits = True
                #        #print("  - Resorting to CNOT via SWAPs!!!")
                #        break
                #    currentqubit = nextqubit
                #
                #if not includes_eliminated_qubits:
                #
                #    currentqubit = i       
                #    while currentqubit != farthest_qubit:
                #
                #        nextqubit = pspec.shortestpath[farthest_qubit,currentqubit]
                #        #print(currentqubit,nextqubit)
                #
                #        if sout[i,nextqubit] == 0:
                #            columnaction_instructionlist.append(_Label('CNOT',(nextqubit,currentqubit)))
                #            sout[:,nextqubit] = sout[:,nextqubit] ^ sout[:,currentqubit]
                #        currentqubit = nextqubit
                #
                #    assert(currentqubit == farthest_qubit)
                #
                #    # Set the farthest qubit s-matrix element to 0 (but don't change the others)
                #    quse = pspec.shortestpath[i,farthest_qubit]
                #    columnaction_instructionlist.append(_Label('CNOT',(farthest_qubit,quse)))
                #    sout[:,farthest_qubit] =  sout[:,quse] ^ sout[:,farthest_qubit]
                #
                #    currentqubit = farthest_qubit
                #    while currentqubit != i:
                #
                #        nextqubit = pspec.shortestpath[i,currentqubit]
                #
                #        if currentqubit not in remaining_qubits:
                #            columnaction_instructionlist.append(_Label('CNOT',(currentqubit,nextqubit)))
                #            sout[:,currentqubit] = sout[:,nextqubit] ^ sout[:,currentqubit]
                #        currentqubit = nextqubit
                #        
                #else:
                #    columnaction_instructionlist.append(_Label('CNOT',(farthest_qubit,i)))
                #    sout[:,farthest_qubit] = sout[:,i] ^ sout[:,farthest_qubit]
                    
                 
                #print(sout)

            # Delete the farthest qubit from the list -- it has been eliminated for this column.    
            del qubits[qubits.index(farthest_qubit)]

            # And set it's distance to -1, so that in the next round we find the next farthest qubit.
            distances_to_qubit_i[farthest_qubit] = -1

            #print(qubits)
            #print(instructionlist)


            # Remove from te remaining qubits list
        #print(remaining_qubits)
        del remaining_qubits[remaining_qubits.index(i)]

        #print(sout)
        #print(len(rowaction_instructionlist),len(columnaction_instructionlist))

    rowaction_instructionlist.reverse()
    columnaction_instructionlist
    full_instructionlist =  columnaction_instructionlist + rowaction_instructionlist

    return full_instructionlist



""" Stabilizer state and measurement compilation routines """
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

from . import compilecnot as _cc

#
# Todo : change to have a algorithms list for the CNOT circuit.
#
# Todo : change so that the relations are a part of the ProcessorSpec
# Todo : change so that the function is handed a costfunction
def compile_stabilizer_state(s, p, pspec, iterations=20, paulirandomize=False, 
                             algorithm = 'RCACC', aargs = [], relations=None):
    
    assert(_symp.check_valid_clifford(s,p)), "The input s and p are not a valid clifford."
    
    n = len(s[0,:])//2
    min_twoqubit_gatecount = _np.inf
    
    #Import the single-qubit Cliffords up-to-Pauli algebra
    gate_relations_1q = _symp.single_qubit_clifford_symplectic_group_relations()
    
    failcount = 0
    i = 0
    while i < iterations:

        tc, tcc = symplectic_as_conditional_clifford_circuit_over_CHP(s, pspec, algorithm, aargs)
        i += 1
            #
            # Todo: work out how much this all makes sense.
            #
            # Do the depth-compression *before* changing gate library            
        tc.compress_depth(gate_relations_1q,verbosity=0)            
        tc.change_gate_library(pspec.compilations['paulieq'])        
        twoqubit_gatecount = tc.twoqubit_gatecount()
        if twoqubit_gatecount  < min_twoqubit_gatecount :
            circuit = _copy.deepcopy(tc)
            check_circuit = _copy.deepcopy(tcc)
            min_twoqubit_gatecount = twoqubit_gatecount
 
        assert(failcount <= 5*iterations), "Randomized compiler is failing unexpectedly often. Perhaps input ProcessorSpec is not valid or does not contain the neccessary information."
            
    if relations is not None:
        # Do more depth-compression on the chosen circuit. Todo: This should used something already
        # constructed in DeviceSpec, instead of this ad-hoc method.
        sreps = pspec.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
        sprecompression, junk =  _symp.symplectic_rep_of_clifford_circuit(circuit,sreps)
        circuit.compress_depth(relations,verbosity=0)    
        spostcompression, junk =  _symp.symplectic_rep_of_clifford_circuit(circuit,sreps)
        assert(_np.array_equal(sprecompression,spostcompression)), "The gate relations provided are incorrect!"
        
    if paulirandomize:
        
        n = pspec.number_of_qubits
        paulilist = ['I','X','Y','Z']
        d = circuit.depth()
        for i in range(1,d+1):
            pcircuit = _Circuit(gatestring=[_Label(paulilist[_np.random.randint(4)],k) for k in range(n)],num_lines=n)
            pcircuit.change_gate_library(pspec.compilations['absolute'])
            circuit.insert_circuit(pcircuit,d-i)
        
    #check_circuit.change_gate_library(pspec.compilations['paulieq'])
    check_circuit.append_circuit(circuit)

    # Add CNOT into the dictionary, in case it isn't there.
    sreps = pspec.models['clifford'].get_clifford_symplectic_reps()
    sreps2 = sreps.copy()
    sreps2['CNOT'] = (_np.array([[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1]],int), _np.array([0,0,0,0],int))
    
    implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit, sreps)
    implemented_scheck, implemented_pcheck = _symp.symplectic_rep_of_clifford_circuit(check_circuit, sreps2)
    
    # Find the needed Pauli at the end    
    paltered = p.copy()
    paltered[0:n] = _np.zeros(n,int)
    paltered = _symp.construct_valid_phase_vector(implemented_scheck,paltered)                                                  
    pauli_layer = _symp.find_postmultipled_pauli(implemented_scheck,implemented_pcheck,paltered)
            
    paulicircuit = _Circuit(gatestring=pauli_layer,num_lines=n)
    paulicircuit.change_gate_library(pspec.compilations['absolute'])
    circuit.append_circuit(paulicircuit)
    
    if not paulirandomize:        
        circuit.compress_depth(verbosity=0)
    
    return circuit

#
# Todo : change to have a algorithms list for the CNOT circuit.
# Todo : use the symp.postpauli function thing
#
def compile_stabilizer_measurement(s, p, pspec, iterations=20, paulirandomize=False,
                                   algorithm = 'RCACC', aargs = [], relations=None):
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
            tc, tcc = symplectic_as_conditional_clifford_circuit_over_CHP(sin, pspec, algorithm, aargs)
            i += 1
            tc.reverse()
            # Do the depth-compression *after* the circuit is reversed
            tc.compress_depth(gate_relations_1q,verbosity=0)
            tc.change_gate_library(pspec.compilations['paulieq'])
            twoqubit_gatecount = tc.twoqubit_gatecount()
            if twoqubit_gatecount  < min_twoqubit_gatecount :
                circuit = _copy.deepcopy(tc)
                check_circuit = _copy.deepcopy(tcc)
                min_twoqubit_gatecount = twoqubit_gatecount
        except:
            failcount += 1
            
        assert(failcount <= 5*iterations), "Randomized compiler is failing unexpectedly often. Perhaps input DeviceSpec is not valid or does not contain the neccessary information."
         
    #check_circuit.reverse()
    #check_circuit.change_gate_library(pspec.compilations['paulieq'])

    if relations is not None:
        # Do more depth-compression on the chosen circuit. Todo: This should used something already
        # constructed in DeviceSpec, instead of this ad-hoc method.
        sreps = pspec.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
        sprecompression, junk =  _symp.symplectic_rep_of_clifford_circuit(circuit,sreps)
        circuit.compress_depth(relations,verbosity=0)    
        spostcompression, junk =  _symp.symplectic_rep_of_clifford_circuit(circuit,sreps)
        assert(_np.array_equal(sprecompression,spostcompression)), "Gate relations are incorrect!"
        
    if paulirandomize:
        
        n = pspec.number_of_qubits
        paulilist = ['I','X','Y','Z']
        d = circuit.depth()
        for i in range(0,d):
            pcircuit = _Circuit(gatestring=[_Label(paulilist[_np.random.randint(4)],k) for k in range(n)],num_lines=n)
            pcircuit.change_gate_library(pspec.compilations['absolute'])
            circuit.insert_circuit(pcircuit,d-i)
    
    check_circuit.reverse()
    #check_circuit.change_gate_library(pspec.compilations['paulieq'])
    check_circuit.prefix_circuit(circuit)

    sreps = pspec.models['clifford'].get_clifford_symplectic_reps() # doesn't matter which compilation, just a fn of the contained gateset
    sreps['CNOT'] = (_np.array([[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1]],int), _np.array([0,0,0,0],int))
    
    implemented_scheck, implemented_pcheck = _symp.symplectic_rep_of_clifford_circuit(check_circuit, sreps)
    
    implemented_sin_check, implemented_pin_check =  _symp.inverse_clifford(implemented_scheck, implemented_pcheck) 
    
    assert(_np.array_equal(implemented_scheck[0:n,:],s[0:n,:]))
    
    # Find the needed Pauli at the start    
    pinaltered = pin.copy()
    pinaltered = _symp.construct_valid_phase_vector(implemented_sin_check,pinaltered)      
    # Unintutively, find_postmultplied is correct here.
    pauli_layer = _symp.find_postmultipled_pauli(implemented_sin_check,implemented_pin_check,pinaltered)
    
    #s_form = _symp.symplectic_form(n)
    #vec = _np.dot(implemented_sin_check,_np.dot(s_form, (pinaltered - implemented_pin_check)//2))
    #vec = vec % 2
    
    #pauli_layer = []
    #for q in range(0,n):
    #    if vec[q] == 0 and vec[q+n] == 0:
    #        pauli_layer.append(_Label('I',q))
    #    elif vec[q] == 0 and vec[q+n] == 1:
    #        pauli_layer.append(_Label('Z',q))
    #    elif vec[q] == 1 and vec[q+n] == 0:
    #        pauli_layer.append(_Label('X',q))
    #    elif vec[q] == 1 and vec[q+n] == 1:
    #        pauli_layer.append(_Label('Y',q))
          
    paulicircuit = _Circuit(gatestring=pauli_layer,num_lines=n)
    paulicircuit.change_gate_library(pspec.compilations['absolute'])
    circuit.prefix_circuit(paulicircuit)
    
    if not paulirandomize:
        circuit.compress_depth(verbosity=0) 
    
    return circuit

#
# Todo: Update to a function that works on all of s, or simplify (with this functinality
# inside the CNOT-circuit compiler) and put inside the matrix tools .py
#
# Todo : this should be in the CNOT compilers part.
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


def symplectic_as_conditional_clifford_circuit_over_CHP(s, pspec=None, calg='RCACC', cargs=[]):
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
        s_of_circuit, p = _symp.symplectic_rep_of_clifford_circuit(circuit)
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
    
        MUout, success, instructions, MLout = convert_invertible_to_reduced_echelon_form(MU,optype='column', 
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
    
    
    
    if calg != 'BGE':
        # Let's replace the instructions for 4 with a better CNOT circuit compiler
        CNOtc = _Circuit(gatestring= instructions4,num_lines=n)
        CNOTs, CNOTp = _symp.symplectic_rep_of_clifford_circuit(CNOtc)
        if calg == 'RCACC':
            improved_instructions4 = _cc.compile_CNOT_circuit(CNOTs[:n,:n],pspec,*cargs)
        else:
            raise ValueError("calg must be 'BGE' or 'RCACC'")
        main_instructions =  instructions7 + instructions6 + improved_instructions4 + instructions3 + instructions1
        #print(CNOTs)
        nws, nwp = _symp.symplectic_rep_of_clifford_circuit(_Circuit(gatestring=improved_instructions4,num_lines=n))
        #print(nws)
    
    circuit = _Circuit(gatestring=main_instructions,num_lines=n)
    implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit)
            
    # Check for success
    #check_circuit = _Circuit(gatestring=precircuit_instructions,num_line=n)
    #check_circuit.append_circuit(circuit)
    CNOT_pre_circuit = _Circuit(gatestring=precircuit_instructions,num_lines=n)
    check_circuit = _copy.deepcopy(CNOT_pre_circuit)
    check_circuit.append_circuit(circuit)
    scheck, pcheck = _symp.symplectic_rep_of_clifford_circuit(check_circuit)
    assert(_np.array_equal(scheck[:,n:2*n],s[:,n:2*n])), "Compiler has failed!"

    #if returnall:
    #    #return circuit, check_circuit
    return circuit, CNOT_pre_circuit
    #else:
    #    #return circuit, check_circuit
    #    return circuit, CNOT_pre_circuit
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy

from . import circuit as _cir
from . import symplectic as _symp


class CompilationLibraries(object):
    
    def __init__(self):
        """
        An object to store compilations for gates. Currently it is a 
        bit hacky, but I think it is probably useful to have an object 
        for this so that we can use it to develop a standard and elegant
        method to represent compilations. Ultimately, this should probably
        store circuits that map from gateset elements to gateset elements. 
        But, I'm not sure how best to go about doing this.
        
        This object should perhaps live inside a gateset/devicespec. Because
        essentially all functionality requires a gateset + some of them are handed
        extra bits from the device spec (but those are bits that can be easily 
        regenerated).
        """
        self.paulieq = {}
        self.paulieq_helpers = {}
        self.absolute = {}
        self.absolute_helpers = {}
        
    def add_fundamental_clifford_compilation(self, gate, gateset, unitary=None, smatrix=None, 
                                             svector=None, ctype='absolute', max_iterations=10,
                                             verbosity=1):
        
        
        if gate.number_of_qubits == 1:
            
            q = gate.qubits[0]
            
            
            if ctype == 'absolute':
                compilationfound = False
                if gate.label in list(self.absolute_helpers.keys()):
                    
                    compilationfound = False
                    for compilation in self.absolute_helpers[gate.label]:
                        
                        # This method currently doesn't check the stored compilation
                        # could use all the available gates for this qubit. If it couldn't
                        # this might be suboptimal. This should be fixed, and the best way
                        # to do this is to make compilations into maps from restricted gatesets
                        # to a gate.
                        
                        flag = True
                        for glabel in compilation:
                            if gateset.availability[glabel][q] == 0:
                                flag = False 
                        
                        if flag:
                            self.absolute[gate] = _cir.Circuit(gate_list = [_cir.Gate(glabel,q) for glabel in compilation],
                                                              n = gateset.number_of_qubits)
                            compilationfound = True
                            break
                     
                if not compilationfound:
                    
                    allowedglabels = []
                    for glabel in gateset.names:
                        if gateset.size[glabel] == 1 and gateset.availability[glabel][gate.qubits[0]] == 1:
                            allowedglabels.append(glabel)
                                
                    self.add_clifford_compilation_helper(gate.label, gateset, allowedglabels=allowedglabels,
                                        unitary=unitary, smatrix=smatrix,
                                        svector=svector, ctype=ctype, verbosity=verbosity,
                                        max_iterations=max_iterations)
                    
                    compilation = self.absolute_helpers[gate.label][-1]
                    self.absolute[gate] = _cir.Circuit(gate_list = [_cir.Gate(glabel,q) for glabel in compilation],
                                                              n = gateset.number_of_qubits)
                        
                    compilationfound = True
                
            if ctype == 'paulieq':
                
                if gate.label in list(self.paulieq_helpers.keys()):
                    
                    compilationfound = False
                    for compilation in self.paulieq_helpers[gate.label]:
                        
                        # This method currently doesn't check the stored compilation
                        # could use all the available gates for this qubit. If it couldn't
                        # this might be suboptimal. This should be fixed, and the best way
                        # to do this is to make compilations into maps from restricted gatesets
                        # to a gate.
                        
                        flag = True
                        for glabel in compilation:
                            if gateset.availability[glabel][q] == 0:
                                flag = False 
                        
                        if flag:
                            self.paulieq[gate] = _cir.Circuit(gate_list = [_cir.Gate(glabel,q) for glabel in compilation],
                                                              n = gateset.number_of_qubits)
                            compilationfound = True
                            break
                                         
                if not compilationfound:
                    
                    allowedglabels = []
                    for glabel in gateset.names:
                        if gateset.size[glabel] == 1 and gateset.availability[glabel][gate.qubits[0]] == 1:
                            allowedglabels.append(glabel)
                                
                    self.add_clifford_compilation_helper(gate.label, gateset, allowedglabels=allowedglabels,
                                        unitary=unitary, smatrix=smatrix,
                                        svector=svector, ctype=ctype, verbosity=verbosity,
                                        max_iterations=max_iterations)
                    
                    compilation = self.paulieq_helpers[gate.label][-1]
                    self.paulieq[gate] = grb.Circuit(gate_list = [grb.Gate(glabel,q) for glabel in compilation],
                                                              n = gateset.number_of_qubits)
                        
                    compilationfound = True
         
        elif gate.label == 'CNOT':
            print("Not yet written!")
        
        elif gate.label == 'CPHASE':
            print("Not yet written!")
            
        elif gate.label == 'SWAP':
            print("Not yet written!")
            
        else:
            print("An automated fundamental compilation for this gate is not currently supported!")
            
    def add_general_clifford_compilation(self,gatelikeobject, circuit, ctype):
        
        if ctype == 'absolute':
            
            self.absolute[gatelikeobject] = circuit
            
        elif ctype == 'paulieq':
            
             self.paulieq[gatelikeobject] = circuit
            
    def add_clifford_compilation_helper(self, glabel, gateset, allowedglabels='all',
                                        unitary=None, smatrix=None,
                                        svector=None, ctype='absolute', verbosity=1,
                                        max_iterations=10):
        
        # If gate label does not yet have any compilations, start a list for the 
        # compilations.
        
        if ctype == 'absolute':
            if glabel not in list(self.absolute_helpers.keys()):
                self.absolute_helpers[glabel] = []
                
        if ctype == 'paulieq':
            if glabel not in list(self.paulieq_helpers.keys()):
                self.paulieq_helpers[glabel] = []
                
        #print('---',glabel)
        #print(self.absolute_helpers)
        #print(self.paulieq_helpers)
        
        # The unitary is specifed, this takes priority and we use it to construct the 
        # symplectic rep of the gate.
        
        if unitary is not None:
            
            smatrix, svector = _symp.unitary_to_symplectic(unitary,flagnonclifford=True)
        
        # If the unitary has not been provided and smatrix and svector are both None, then
        # we find them from the dictionary of standard gates.
        
        if (smatrix is None) or (svector is None):
        
            std_smatrices, std_svectors = _symp.symplectic_representation()
            try:
                smatrix = std_smatrices[glabel]
                svector = std_svectors[glabel]
            except:
                raise ValueError
                
        assert(_symp.check_valid_clifford(smatrix,svector)), "The gate is not a valid Clifford!"
        
        n = _np.shape(smatrix)[0] // 2
        
        # todo : this should ultimately be removed, but making this function work for 
        # arbitrary two or multi-qubit gates is much more complicated.
        
        assert(n == 1), "Function currently only works for single qubit gates!"
        
        if n == 1:
            
            if allowedglabels == 'all':
                allowedglabels = []
                for gl in gateset.names:
                    if gateset.size[gl] == 1: #and gateset.availability[glabel] == 1:
                        allowedglabels.append(gl)
                          
            if verbosity > 0:
                if ctype == 'absolute':
                    print("Generating an absolute compilation for the {} gate...".format(glabel),end='')
                if ctype == 'paulieq':
                    print("Generating a pauli-equivalence compilation for the {} gate...".format(glabel),end='')
                    
            obtained_smatrix = {}
            obtained_svector = {}
            
            # Find the obtained gates from length-1 circuits.
            for gl in allowedglabels:
                if gateset.size[gl] == 1:
                    # We force the key to be a tuple.
                    if (gateset.smatrix[gl] is not None) and (gateset.svector[gl] is not None):
                        obtained_smatrix[(gl,)] = gateset.smatrix[gl]  
                        obtained_svector[(gl,)] = gateset.svector[gl]
        
            # Main loop. We go through the loop at most max_iterations times
            flag = False
            for counter in range(0,max_iterations):

                #if verbosity > 0:
                #    print("")
                #    print("")
                #    print("Checking all length {} single-qubit circuits...".format(counter))
            
                # Check each of the obtained symplectic matrices, and see if it corresponds to
                # any of the gates we are still searching for compilations for.
                for gseq in list(obtained_smatrix.keys()):
                
                    # Check if we have found a compilation. Note that multiple compilations are
                    # recorded if they are found at the same length.
                    if _np.array_equal(smatrix,obtained_smatrix[gseq]):
                        # Save the compilation as a list, rather than a tuple.
                        if ctype == 'paulieq':
                            self.paulieq_helpers[glabel].append(list(gseq))
                            flag = True
                        if ctype == 'absolute':
                            if _np.array_equal(svector,obtained_svector[gseq]):
                                self.absolute_helpers[glabel].append(list(gseq))
                                flag = True
                            
                # If we have found the gate, leave the loop
                if flag:
                    if verbosity > 0:
                        print("Complete.")
                    break
                
                # If we have gone through the maximum amount of iterations, we quit before constructing
                # the circuits for the next length. 
                if counter == max_iterations-1:
                    print("** Maximum iterations reached without finding a compilation! **")
                    break
            
                # If we have not found all the gates, find the gates obtained by the next length
                # gate sequences.
                new_obtained_smatrix = {}
                new_obtained_svector = {}
                # Consider each sequence obtained so far, which is recorded as a tuple.
                for seq in list(obtained_smatrix.keys()):
                    # Consider adding each possible gate to the end of that sequence.
                    for gl in allowedglabels:
                        if gateset.size[gl] == 1:
                            if (gateset.smatrix[gl] is not None) and (gateset.svector[gl] is not None):
                                # Turn the tuple, for the gate sequence into a list, to append the next gate.
                                key = list(seq)
                                key.append(gl)
                                # Turn back into a tuple, so that it can be a dict key.
                                key = tuple(key)
                                # Find the gate implemented by the new sequence, and record.
                                new_obtained_smatrix[key], new_obtained_svector[key] = _symp.compose_cliffords(obtained_smatrix[seq], obtained_svector[seq],gateset.smatrix[gl], gateset.svector[gl])
                                
                # Write over the old obtained sequences.
                obtained_smatrix = new_obtained_smatrix
                obtained_svector = new_obtained_svector
                
                
    def add_CNOT_for_connected_qubits(self,qubits,gateset,max_iterations=6,ctype='absolute',verbosity=1):
        """
        This method generates a compilation for CNOT, up to arbitrary Pauli gates, between a
        pair of connected qubits. The method is a brute-force search algorithm, checking 
        all circuit of each length until a compilation is found.
        
        Parameters
        ----------
        q1 : int
            The control qubit label
            
        q2 : int
            The target qubit label
            
        max_iterations : int, optional
            The maximum number of iterations in the search algorithm, which corresponds to
            the maximum length of circuit checked to find compilations for all needed gates.
            The time required for the alogrithm scales exponentially in this parameter, but
            the default should be sufficient for any entangling gate that is can be converted
            to CNOT by local Cliffords (e.g., CPHASE, or CNOT with the opposite control and 
            target direction).
            
        verbosity : int, optional
            The amount of print-to-screen.
            
        Returns
        -------
        circuit : a Circuit object
            A circuit representation of the constructed CNOT compilation.
        
        """
        q1 = qubits[0]
        q2 = qubits[1]
        
        if verbosity > 0:
            print("")
            print("Attempting to generate a compilation for CNOT, up to Paulis,")
            print("with control qubit = {} and target qubit = {}".format(q1,q2))
            print("")
            
        # The obtained gates, in the symplectic rep, after each round of the algorithm                               
        obtained_s = {}
        obtained_p = {}
        
        # Import the symplectic representations of the standard gates, to get the CNOT rep.
        #std_smatrices, std_svectors = _symp.symplectic_representation()
        #CNOTsmatrix = std_smatrices['CNOT']
        #CNOTsvector = std_svectors['CNOT']
        
        CNOTsmatrix, CNOTsvector = _symp.clifford_layer_in_symplectic_rep([_cir.Gate('CNOT',(q1,q2)),],
                                                                          gateset.number_of_qubits)
        
        
        symplectic_matrix = {}
        phase_vector = {}
        
        # Construst all possible circuit layers acting on the two qubits.
        all_layers = []
        for gl in gateset.names:
            
            # Go in here if it is a single qubit gate
            if gateset.size[gl] == 1 and gateset.availability[gl][q1] == 1:
                
                # Check the gate is a Clifford. If it isn't we cannot use it in this algorithm
                if (gateset.smatrix[gl] is not None) and (gateset.svector[gl] is not None):
                    
                    for gl2 in gateset.names:
                        if gateset.size[gl2] == 1 and gateset.availability[gl2][q2] == 1:
                            if (gateset.smatrix[gl2] is not None) and (gateset.svector[gl2] is not None):
                                all_layers.append([_cir.Gate(gl,q1),_cir.Gate(gl2,q2)])
            
            # Go in here if it is a two qubit gate
            if gateset.size[gl] == 2:
                if gateset.availability[gl][q1,q2] == 1:
                    all_layers.append([_cir.Gate(gl,(q1,q2)),])                        
                if gateset.availability[gl][q2,q1] == 1:
                    all_layers.append([_cir.Gate(gl,(q2,q1)),])    
        
        # Find the symplectic action of all possible circuits of length 1 on the two qubits
        for layer in all_layers:
            s, p =  _symp.clifford_layer_in_symplectic_rep(layer, gateset.number_of_qubits,  
                                                           s_dict=gateset.smatrix, p_dict=gateset.svector)
            obtained_s[tuple(layer)] =  s
            obtained_p[tuple(layer)] =  p
        
        # Bool to flag up when we have found a compilation for CNOT.
        cnot_found = False
        
        # Loop over the number of permisable iterations, until we find CNOT or run out of iterations.
        for counter in range(0,max_iterations):
                
            if verbosity > 0:
                print("Checking all length {} two-qubit circuits...".format(counter+1))
            
            # Keep a list of all compilations for CNOT, if any, of this length.
            candidates = []
            
            # Look to see if we have found a compilation for CNOT, up to Paulis.
            for key in list(obtained_s.keys()):    
                
                if _np.array_equal(CNOTsmatrix,obtained_s[key]):
                    
                    cnot_compilation = list(key)

                    if ctype == 'paulieq':
                        candidates.append(key)                    
                        cnot_found = True
                        
                    if ctype == 'absolute':
                        if _np.array_equal(CNOTsvector,obtained_p[key]):
                            candidates.append(key)                    
                            cnot_found = True
                            
            # If there is more than one way to compile CNOT at this circuit length, pick the
            # one containing the most idle gates.
            if len(candidates) > 1:
                
                number_of_idles = 0
                max_number_of_idles = 0
                
                # Look at each sequence, and see if it has more than or equal to max_number_of_idles.
                # If so, set it to the current chosen sequence.
                for seq in candidates:
                    number_of_idles = 0
                    
                    for xx in range(0,len(seq)):
                        if seq[xx].label == 'I':
                            number_of_idles += 1 
                            
                    if number_of_idles >= max_number_of_idles:
                        max_number_of_idles = number_of_idles
                        best_sequence = seq
              
                cnot_compilation = list(best_sequence)    
                cnot_compilation_phase = obtained_p[best_sequence]
            
            # If we have found CNOT, leave the loop
            if cnot_found:
                if verbosity > 0:
                    print("Compilation for CNOT found!")
                break
            
            # If we have reached the maximum number of iterations, quit the loop
            # before we construct the symplectic rep for all sequences of a longer length.
            if (counter == max_iterations - 1) and cnot_found is False:
                print("*************************************************************************************")
                print("           Maximum iterations reached without finding a CNOT compilation !")
                print("*************************************************************************************")
                print("")
                return None
            
            # Construct the gates obtained from the next length sequences.
            new_obtained_s = {}
            new_obtained_p = {}
                             
            for seq in list(obtained_s.keys()):
                # Add all possible tensor products of single-qubit gates to the end of the sequence
                for layer in all_layers:
                        
                    # Calculate the symp rep of this parallel gate
                    sadd, padd = _symp.clifford_layer_in_symplectic_rep(layer, gateset.number_of_qubits,
                                                                         s_dict=gateset.smatrix, 
                                                                         p_dict=gateset.svector)
                        
                    # Convert to list, to append gates, and then convert back to tuple
                    key = list(seq)
                    key = key + layer
                    key = tuple(key) 
                        
                    # Calculate and record the symplectic rep of this gate sequence.
                    
                    new_obtained_s[key], new_obtained_p[key] =_symp.compose_cliffords(obtained_s[seq], 
                                                                                          obtained_p[seq],
                                                                                          sadd, padd)                          
                        
            # Overwrite old sequences.
            obtained_s = new_obtained_s
            obtained_p = new_obtained_p
                
        # Convert the gate sequence to a circuit
        circuit = _cir.Circuit(gate_list=cnot_compilation,n=gateset.number_of_qubits)
        #print(circuit)
        
        return circuit
               
    def add_CNOT_for_unconnected_qubits(self, qubits, shortestpathmatrix, distancematrix, 
                                        gateset, verbosity=1, ctype='absolute', 
                                        compile_cnots=False):
        """
        This method generates a compilation for CNOT, up to arbitrary Pauli gates, between a
        pair of unconnected qubits. This function converts this CNOT into a circuit of CNOT
        gates between connected qubits, using a fixed circuit form. This compilation is not 
        optimal in at least some circumstances.
        
        Parameters
        ----------
        q1 : int
            The control qubit label
            
        q2 : int
            The target qubit label
            
        verbosity : int, optional
            The amount of print-to-screen.
            
        Returns
        -------
        circuit : a Circuit object
            A circuit representation of the constructed CNOT compilation.        
        
        """
        q1 = qubits[0]
        q2 = qubits[1]
        
        if verbosity > 0:
            print("")
            print("Attempting to generate a compilation for CNOT, up to Paulis,")
            print("with control qubit = {} and target qubit = {}".format(q1,q2))
            print("")
            print("Distance between qubits is = {}".format(distancematrix[q1,q2]))
            
        assert(shortestpathmatrix[q1,q2] >= 0), "There is no path between the qubits!"
        
        # Find the shortest path between q1 and q2, from self.shortest_path_matrix
        # We do this by following the chain in the shortest_path_matrix until we arrive at q1
        shortestpath = [q2]
        current_node = q2
                
        while current_node != q1:            
            preceeding_node = shortestpathmatrix[q1,current_node]
            shortestpath.insert(0,preceeding_node)
            current_node = preceeding_node
            
        # If the qubits are connected, this algorithm may not behave well.
        assert(len(shortestpath)> 2), "Qubits are connected! Algorithm is not needed or valid."
        
        # Part 1 of the circuit is CNOTs along the shortest path from q1 to q2.
        # To do: describe the circuit.
        part_1 = []            
        for i in range(0,len(shortestpath)-1):
            part_1.append(_cir.Gate('CNOT',[shortestpath[i],shortestpath[i+1]]))
        
        # Part 2 is...
        # To do: describe the circuit.
        part_2 = _copy.deepcopy(part_1)
        part_2.reverse()
        del part_2[0]
        
        # To do: describe the circuit.
        part_3 = _copy.deepcopy(part_1)
        del part_3[0]
        
        # To do: describe the circuit.
        part_4 = _copy.deepcopy(part_3)
        del part_4[len(part_3)-1]
        part_4.reverse()
        
        # Add the lists of gates together, in order
        cnot_circuit = part_1 + part_2 + part_3 + part_4
        
        # Convert the gatelist to a circuit.
        circuit = _cir.Circuit(gate_list=cnot_circuit,n=gateset.number_of_qubits)
        
        # If we are not compiling the CNOTs between connected qubits into native gates
        # then we are done.        
        if not compile_cnots:
            print(circuit)
            return circuit
                
        # If we are compiling the CNOTs between connected qubits into native gates
        # then we now do this.
        else:
            
            if ctype == 'paulieq':
            
                # Import the gate relations of the single-qubit Cliffords, so that circuit
                # compression can be used. Todo: this should probably be handed to this 
                # function, as it is should be something that is possibly in the device spec.
                grels = _symp.single_qubit_clifford_symplectic_group_relations()
                # To do: add an assert that checks that compilations of CNOTs between all
                # connected qubits have been generated.
                #
                # Change into the native gates, using the compilation for CNOTs between
                # connected qubits.
                circuit.change_gate_library(self.paulieq)
                
            if ctype == "absolute":
                circuit.change_gate_library(self.absolute)
            
            # Calculate the symplectic matrix implemented by this circuit, to check the compilation
            # is ok, below.
            s, p = _symp.composite_clifford_from_clifford_circuit(circuit, s_dict=gateset.smatrix, 
                                                            p_dict=gateset.svector)
            
            # Construct the symplectic rep of CNOT between this pair of qubits, to compare to s.
            s_cnot, p_cnot = _symp.clifford_layer_in_symplectic_rep([_cir.Gate('CNOT',(q1,q2)),],
                                                                          gateset.number_of_qubits)


            assert(_np.array_equal(s,s_cnot)), "Compilation has failed!"
            if ctype == "absolute":
                assert(_np.array_equal(p,p_cnot)), "Compilation has failed!"
                
            return circuit            
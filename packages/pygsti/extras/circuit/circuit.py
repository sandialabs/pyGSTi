from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy
                     
from . import tools as _tls
from . import simulators as _sim

class Gate(object):
  
    def __init__(self,label,qubits):
        """
        An object which encapsulates a gate. It consists of a gate label, and a list
        of qubits on which the gate acts. 
        
        Parameters
        ----------
        label : str
            The gate name. E.g., 'CNOT' or 'H'.
            
        qubits : int or list of ints
            The qubits on which this gate acts. The ordering in the list defines
            the 'direction' of the gate.
        
        """        
        
        # Todo : add some kind of check back in here -- this fails on python 3, as that
        # seems to not know what unicode is.
        #assert(type(label) is unicode), "The gate label should be a unicode string!"
        #
        # To do: add this back in after appropriate fix. Currently, it raises an error
        # if it is handed a type of integer that is not a plain python int.
        #
        #assert((type(qubits) is list) or (type(qubits) is tuple) or (type(qubits) is int)), "The input qubits should be a list, tuple, or int!"
        
        if qubits is list:
            for q in qubits:
                assert(type(q) is int), "The qubits should be labelled by integers!"
                                 
        
        # Regardless of whether the input is a list, tuple, or int, the qubits that the 
        # gate acts on are stored as a tuple (because tuples are immutable).
        
        if (type(qubits) is list) or (type(qubits) is tuple):
            self.qubits = tuple(qubits)            
        else:
            self.qubits = (qubits,)
            
        self.label = label          
        self.number_of_qubits = len(self.qubits)
                
    def __str__(self):
        """
        Defines how a Gate is printed out, so that it appears like a tuple.
        """
        s = '(' + self.label + ', ' +  str(self.qubits)[1:]
        return s    
    
    def __eq__(self,other):
        """
        Defines equality between gates, so that they are equal if their values
        are equal.
        """
        return self.label == other.label and self.qubits == other.qubits
    
    def __hash__(self):
        
        return hash((self.label,self.qubits))

class CircuitLayer(object):
    
    def __init__(self, gatelist, n):
        self.layer = 0

class Circuit(object):
    
    def __init__(self, circuit=None, gate_list=None, n=None):
        """
        An object with encapsulates a quantum circuit, and which contains a range of methods for
        manipulating quantum circuits. E.g., basic depth compression algorithms.
    
        """
        
        assert((circuit is not None) or (gate_list is not None) or (n is not None)), "At least one of the inputs must be not None!"
        
        if circuit is not None:
            self.circuit = circuit
            self.number_of_qubits = len(self.circuit)
            
        elif gate_list is not None:
            assert(type(n) is int), "Number of qubits needs to be specified! This cannot be extracted from a gate list!"
            self.number_of_qubits = n
            self.initialize_circuit_from_gate_list(gate_list)
            
        elif n is not None:       
            assert(type(n) is int), "Number of qubits needs to be specified!"
            self.number_of_qubits = n
            self.initialize_empty_circuit()
        
        self.layers_as_superoperators = {}
        self.layers_as_unitaries = {}
          
    def initialize_empty_circuit(self):
        """
        Creates an empty circuit, which consists of a dictionary
        of empty lists.
        
        """
        self.circuit = []
        for i in range(0,self.number_of_qubits):
            self.circuit.append([])
                        
    def initialize_circuit_from_gate_list(self,gate_list):
        """
        Creates a circuit from a list of Gate objects. This function assumes that 
        all gates act on 1 or 2 qubits only, but this could be easily updated if 
        necessary.
        
        Parameters
        ----------
        gate_list : list
            A list of Gate objects.
        
        """
        assert(type(gate_list) is list), "Input must be a list of Gate objects!"
        
        # If the list is empty, return an empty circuit
        if len(gate_list) == 0:
            self.initialize_empty_circuit()
        
        else:
            # If the sequence is of tuples or lists, convert them to Gate objects
            if (type(gate_list[0]) is list) or (type(gate_list[0]) is tuple):
                for i in range(0,len(gate_list)):
                    gate_list[i] = Gate(gate_list[i][0],tuple(gate_list[i][1:]))
            
            # Create an empty circuit, to populate from gate_list
            self.initialize_empty_circuit()
            
            # keeps track of which gates have been added to the circuit from the list.
            j = 0
            # keeps track of the circuit layer number
            layer_number = 0
        
            while j < len(gate_list):
                
                # The gates that are going into this layer.
                layer = []
                # The number of gates beyond j that are going into this layer.
                k = 0
                # initialized two ints to arbitrary values.
                q1 = -1
                q2 = -1
                # The qubits used in this layer.
                used_qubits = []
    
                while (q1 not in used_qubits) and (q2 not in used_qubits):
                    # look at the j+kth gate and include in this layer
                    gate = gate_list[j+k]
                    assert(gate.number_of_qubits == 1 or gate.number_of_qubits == 2), "All gates in the list must be 1 or 2 qubit gates!"
                    layer.append(gate)
                    # Add the qubits in this gate to the used qubits list
                    for qubit in gate.qubits:
                        used_qubits.append(qubit)
                        
                    # look at the next gate in the list, which will be
                    # added to the layer if it does not act on any qubits
                    # with a gate already in this layer
                    k += 1
                
                    if j+k >= len(gate_list):
                        break
                    else:
                        gate = gate_list[j+k]
                        q1 = gate.qubits[0]
                        if gate.number_of_qubits == 2:
                            q2 = gate.qubits[1]
                        else:
                            q2 = -1
                
                # Insert the layer into the circuit.
                self.insert_layer(layer,layer_number)
                
                # Move on to the next gate not in included in the circuit.
                j += k
                # Update the layer number.
                layer_number += 1
    
    def insert_gate(self,gate,j):
        """
        Inserts a gate into a circuit.
        
        Parameters
        ----------
        gate : Gate
            A Gate object, to insert.
            
        j : int
            The circuit layer at which to insert the gate.
            
        """
        # Check input is a valid gate
        _tls.check_valid_gate(gate)
        
        # Add an idle layer.
        for i in range(0,self.number_of_qubits):
            self.circuit[i].insert(j,Gate('I',i))
            
        # Put the gate in
        for i in gate.qubits:
            self.circuit[i][j] = gate
                
    def insert_layer(self,circuit_layer,j):
        """
        Inserts a layer into a circuit. The input layer does not
        need to contain a gate that acts on every qubit. But,
        it should be a valid layer, meaning that the layer
        does not contain more than one gate on a qubit.
        
        Parameters
        ----------
        circuit_layer : Gate
            A list of gate objects, to insert as a circuit layer.
            
        j : int
            The depth at which to insert the circuit layer.
            
        """
        _tls.check_valid_circuit_layer(circuit_layer,self.number_of_qubits)
        
        # Add an idle layer.
        for i in range(0,self.number_of_qubits):
            self.circuit[i].insert(j,Gate('I',i))
            
        # Put the gates in.
        for i in range(0,self.number_of_qubits):
            for ii in range(0,len(circuit_layer)):
                if i in circuit_layer[ii].qubits:
                    self.circuit[i][j] = circuit_layer[ii]
                    
    def insert_circuit(self,circuit,j):
        """
        Inserts a circuit into this circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be inserted.
            
        j : int
            The depth at which to insert the circuit layer.
            
        """
        # Copy the input circuit, so that there is no chance of altering it, 
        # although I don't think this is actually necessary.
        in_circuit = _copy.deepcopy(circuit)
        
        assert(self.number_of_qubits == circuit.number_of_qubits), "The circuits must act on the same number of qubits!"
        
        for q in range(0,self.number_of_qubits):
            self.circuit[q] = self.circuit[q][0:j] + in_circuit.circuit[q] + self.circuit[q][j:]
                            
    def append_circuit(self,circuit):
        """
        Append a circuit to the end of this circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be appended.
 
        """
        # Copy the input circuit, so that there is no chance of altering it, 
        # although I don't think this is actually necessary.
        in_circuit = _copy.deepcopy(circuit)
        
        assert(self.number_of_qubits == in_circuit.number_of_qubits), "The circuits must act on the same number of qubits!"
        
        for q in range(0,self.number_of_qubits):
            self.circuit[q] = self.circuit[q] + in_circuit.circuit[q]
            
    def prefix_circuit(self,circuit):
        """
        Prefix a circuit to the end of this circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be prefixed.
 
        """   
        # Copy the input circuit, so that there is no chance of altering it, 
        # although I don't think this is actually necessary.
        in_circuit = _copy.deepcopy(circuit)
        
        assert(self.number_of_qubits == circuit.number_of_qubits), "The circuits must act on the same number of qubits!"
        
        for q in range(0,self.number_of_qubits):
            self.circuit[q] = in_circuit.circuit[q] + self.circuit[q]
            
    def replace_gate_with_circuit(self,circuit,q,j):
        """
        Replace a gate with a circuit. As other gates in the
        layer might not be the idle gate, the circuit is inserted
        so that it starts at layer j+1.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be inserted in place of the gate.
            
        q : int
            The qubit on which the gate is to be replaced
            
        j : int
            The depth at which the gate is to be replaced.
 
        """
        n = self.circuit[q][j].number_of_qubits 
        assert(n == 1 or n == 2), "Only circuits with only 1 and 2 qubit gates supported!" 
        
        # Replace the gate with an idle
        if n == 1:
            self.circuit[q][j] = Gate('I',q)
            
        else:
           
            q1 = self.circuit[q][j].qubits[0]
            q2 = self.circuit[q][j].qubits[1]
            self.circuit[q1][j] = Gate('I',int(q1))
            self.circuit[q2][j] = Gate('I',int(q2))
        
        # Insert the circuit
        self.insert_circuit(circuit,j+1)
        
        
    def replace_layer_with_layer(self,circuit_layer,j):
        """
        Replace a layer with a layer. The input layer does not
        need to contain a gate that acts on every qubit. But,
        it should be a valid layer, meaning that the layer
        does not contain more than one gate on a qubit.
        
        Parameters
        ----------
        circuit_layer : List
            A list of Gate objects, defining a valid layer.
            
        j : int
            The position of the layer to be replaced.
 
        """
        _tls.check_valid_circuit_layer(circuit_layer,self.number_of_qubits)
        
        # Replace all gates with idles, in the layer to be replaced.
        for q in range(0,self.number_of_qubits):
            self.circuit[q][j] = Gate('I',q)
        
        # Write in the gates, from the layer to be inserted.
        for q in range(0,self.number_of_qubits):
            for qq in range(0,len(circuit_layer)):
                if q in circuit_layer[qq].qubits:
                    self.circuit[q][j] = circuit_layer[qq]   
                                       
    def replace_layer_with_circuit(self,circuit,j):
        """
        Replace a layer with a circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be inserted in place of the layer.
  
        j : int
            The depth at which the layer is to be replaced.
 
        """
        depth = circuit.depth()
        
        # Replace the layer with the first layer of the input circuit.
        layer = circuit.get_circuit_layer(0)
        self.replace_layer_with_layer(layer,j)
        
        # Insert the other layers of the input circuit.
        for i in range(1,depth):
            layer = circuit.get_circuit_layer(i)
            self.insert_layer(layer,j)
        
    def change_gate_library(self,compilation,depth_compression=True,
                            gate_relations_1q=None):
        """
        Re-express a circuit over a different gateset.
        
        Parameters
        ----------
        compilation : dict
            A dictionary whereby the keys are all of the gates that
            appear in the circuit, and the values are replacement
            circuits that are compilations for each of these gates.
            
        depth_compression : bool, opt
            If True then depth compression is implemented on the output
            circuit. Without this being set to True, the output circuit
            will be much larger than necessary, as gates will generically
            not have been parallelized.
  
        gate_relations_1q : dict, opt
            Gate relations for the one-qubit gates in the new gate library,
            that are used in the depth compression, to cancel / combine
            gates. E.g., one key-value pair might be 'H','H' : 'I', to
            signify that two Hadamards compose to an idle gate.
            
 
        """        
        in_compilation = _copy.deepcopy(compilation)
        
        d = self.depth()
        n = self.number_of_qubits
        for l in range(0,d):
            for q in range(0,n):
                gate = self.circuit[q][d-1-l]
                # We ignore idle gates, and do not compile them.
                # To do: is this the behaviour we want?
                if gate.label != 'I':
                    if gate in list(in_compilation.keys()):
                        self.replace_gate_with_circuit(in_compilation[gate],q,d-1-l)
                    else:
                        # To do: replace with a suitable assert somewhere.
                        print('Error!')
        
        # If specified, perform the depth compression.
        if depth_compression:            
            self.compress_depth(gate_relations_1q=gate_relations_1q,verbosity=0)
        
                                                      
    def delete_layer(self,j):
        """
        Delete a layer from the circuit.
        
        Parameters
        ----------
        j : int
            The depth at which the layer is to be deleted.
 
        """
        for q in range(0,self.number_of_qubits):
            del self.circuit[q][j]
                    
    def get_circuit_layer(self,j):
        """
        Returns the layer at depth j.

        """       
        assert(j >= 0 and j < self.depth()), "Circuit layer label invalid! Circuit is only of depth {}".format(self.depth())
        
        layer = []
        for i in range(0,self.number_of_qubits):
            layer.append(self.circuit[i][j])
            
        return layer
    
    def reverse(self):
        """
        Reverse the order of the circuit.

        """         
        for q in range(0,self.number_of_qubits):
            self.circuit[q].reverse()
    
    def depth(self):
        """
        Returns the circuit depth.
        
        """        
        return len(self.circuit[0])
    
    def size(self):
        """
        Returns the circuit size, whereby a gate defined as acting
        on n-qubits of size n, with the exception of the idle gate
        which is of size 0. Hence, the circuit size is circuit
        depth X number of qubits - number of idle gates in the
        circuit.
        """
        size = 0
        for q in range(0,self.number_of_qubits):
            for j in range(0,self.depth()):
                if self.circuit[q][j].label != 'I':
                    size += 1
        return size
    
    def twoqubit_gatecount(self):
        """
        Returns the number of two-qubit gates in the circuit
        
        """           
        count = 0
        for q in range(0,self.number_of_qubits):
            for j in range(0,self.depth()):
                if self.circuit[q][j].number_of_qubits == 2:
                    count += 1
        return count//2

    def __str__(self):
        """
        To do: replace this with a __str__ method.
        
        """
        s = ''
        for i in range(0,self.number_of_qubits):
            s += 'Qubit {} ---'.format(i)
            for j in range(0,self.depth()):
                if self.circuit[i][j].label == 'I':
                    # Replace with special idle print at some point
                    s += '|  |-'
                elif self.circuit[i][j].number_of_qubits == 1:
                    s += '|' + self.circuit[i][j].label + ' |-'
                elif self.circuit[i][j].label == 'CNOT':
                    if self.circuit[i][j].qubits[0] is i:
                        s += '|'+'C' + str(self.circuit[i][j].qubits[1]) + '|-'
                    else:
                        s += '|'+'T' + str(self.circuit[i][j].qubits[0]) + '|-'
                else:
                    s += self.circuit[i][j].__str__() + '|-|' 
            s += '--\n'

        return s
    
    def write_qcircuit_tex(self,filename):
        
        n = self.number_of_qubits
        d = self.depth()
        
        f = open(filename+'.tex','w') 
        f.write("\documentclass{article}\n")
        f.write("\\usepackage{mathtools}\n")
        f.write("\\usepackage{xcolor}\n")
        f.write("\\usepackage[paperwidth="+str(5.+d*.3)+"in, paperheight="+str(2+n*0.2)+"in,margin=0.5in]{geometry}")
        f.write("\input{Qcircuit}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{equation*}\n") 
        #f.write("\Qcircuit @C=1.2em @R=0.5em {\n")
        f.write("\Qcircuit @C=1.0em @R=0.5em {\n")
        
        n = self.number_of_qubits
        for q in range(0,n):
            qstring = '&'
            circuit_for_q = self.circuit[q]
            for gate in circuit_for_q:
                if gate.label == 'I':
                    qstring += ' \qw &'
                elif gate.label == 'CNOT':
                    if gate.qubits[0] == q:
                        qstring += ' \ctrl{'+str(gate.qubits[1]-q)+'} &'
                    else:
                        qstring += ' \\targ &'
                elif gate.label == 'CPHASE':
                    if gate.qubits[0] == q:
                        qstring += ' \ctrl{'+str(gate.qubits[1]-q)+'} &'
                    else:
                        qstring += ' \control \qw &'
            
                else:
                    qstring += ' \gate{'+str(gate.label)+'} &'
                    
            qstring += ' \qw & \\'+'\\ \n'
            f.write(qstring)
        
        f.write("}\end{equation*}\n")
        f.write("\end{document}")
        f.close() 
    
    #
    # To do: makes this work? Redo as a __copy__ method?
    #
    #For some reason this doesn't work so is commented out
    #def copy(self):
    #    
    #    return _copy.deepcopy(self)
    
    
    def combine_1q_gates(self,gate_relations):
        """
        Compresses the number of non-idle one-qubit gates in the circuit, 
        using the provided gate relations. 
        
        Parameters
        ----------
        gate_relations : dict
            Gate relations 
            
        To do: better docstring.
        
        Returns
        -------
        flag : bool
            False if no compression achieved; True otherwise.
        """ 
        
        # A flag that is turned to True if any non-trivial re-arranging is implemented
        # by this method.
        flag = False
        
        def single_qubit_gate_combined(glabel_1,glabel_2):
            # Combines two gate labels, using the provided rules on how the labels combine.
            assert((glabel_1,glabel_2) in list(gate_relations.keys())), "Gate relations provided are invalid!"  
            return gate_relations[glabel_1,glabel_2]
        
        # Loop through all the qubits
        for q in range(0,self.number_of_qubits):
            j = 0
            while j < self.depth()-1:
                k = j
                if self.circuit[q][k].number_of_qubits == 1:
                    for i in range(k+1,self.depth()):
                        if self.circuit[q][i].number_of_qubits > 1:
                            j += 1
                            break
                        else:
                            # Flag set to True if a non-trivial shift/combination is to be implemented.
                            if not self.circuit[q][i].label == 'I':
                                flag = True
                                
                            # Find the new label of the gate, according to the combination rules.
                            gl1 = self.circuit[q][k].label
                            gl2 = self.circuit[q][i].label
                            new_label = single_qubit_gate_combined(gl1,gl2)
                            
                            # Becuase of the way hashing is defining for the Gate object, we could
                            # just change the gates labels. But it seems like better practice to make
                            # new Gate objects.
                            self.circuit[q][k] = Gate(new_label,q)
                            self.circuit[q][i] = Gate('I',q)
                            j += 1
                else:
                    j += 1                    
        return flag
    
    def shift_1q_gates_forward(self):
        """
        To do: docstring
        
        """               
        flag = False
        
        for q in range(0,self.number_of_qubits):
            for j in range(1,self.depth()):
                
                if self.circuit[q][j].number_of_qubits == 1:                  
                    
                    if self.circuit[q][j].label != 'I':
                    
                        for k in range(1,j+1):
                            idle = True
                            if self.circuit[q][j-k].label != 'I':
                                idle = False
                                if not idle:
                                    k = k - 1
                                    break
                                
                        if k > 0:
                            flag = True
                            self.circuit[q][j-k] = self.circuit[q][j]
                            self.circuit[q][j] = Gate('I',q)
        return flag
    
    def shift_2q_gates_forward(self):
        """
        To do: docstring
        
        """                
        flag = False
        
        for q in range(0,self.number_of_qubits):
            for j in range(1,self.depth()):
                if self.circuit[q][j].number_of_qubits == 2:                  
                    # Only try to do a compression if q is the first qubit of the pair, as we only 
                    # need to consider one of the two.
                    if self.circuit[q][j].qubits[0] == q:
                        target_label = self.circuit[q][j].qubits[1]
                        
                        for k in range(1,j+1):
                            both_idle = True
                            if self.circuit[q][j-k].label != 'I':
                                both_idle = False
                            if self.circuit[target_label][j-k].label != 'I':
                                both_idle = False
                            if not both_idle:
                                k = k - 1
                                break
                        if k > 0:
                            flag = True
                            self.circuit[q][j-k] = self.circuit[q][j]
                            self.circuit[target_label][j-k] = self.circuit[target_label][j]
                            self.circuit[q][j] = Gate('I',q)
                            self.circuit[target_label][j] = Gate('I',int(target_label))
        return flag
    
    
    def delete_idle_layers(self):
        """
        To do: docstring
        
        """        
        flag = False
        
        d = self.depth()
        for i in range(0,d):
            
            layer = self.get_circuit_layer(d-1-i)
            all_idle = True
            
            for q in range(0,self.number_of_qubits):
                if layer[q].label != 'I':
                    all_idle = False
                    
            if all_idle:
                flag = True
                self.delete_layer(d-1-i)
                
        return flag
    
    
    def compress_depth(self,gate_relations_1q=None,max_iterations=10000,verbosity=1):
    
        if verbosity > 0:
            print("--------------------------------------------------")
            print("***** Implementing circuit depth compression *****")
            print("--------------------------------------------------")
          
        if verbosity > 0:
            print("")
            print("Circuit depth before compression is {}".format(self.depth()))
    
        flag1 = True
        flag2 = True
        flag3 = True
        counter = 0
    
        while flag1 or flag2 or flag3:
            
            if gate_relations_1q is not None:                            
                flag1 = self.combine_1q_gates(gate_relations_1q)
            else:
                flag1 = self.shift_1q_gates_forward()     
            flag2 = self.shift_2q_gates_forward()
            flag3 = self.delete_idle_layers()

            counter += 1
            if counter > max_iterations:
                print("")
                print('*** Compression algorthim reached the maximum interations of {} ***'.format(max_iterations))
                print("")
                break
    
        if verbosity > 0:
            print("")
            print("Circuit depth after compression is {}".format(self.depth()))
            print("")   
            print("Number of loops used in compression algorithm was {}".format(counter))
            print("")
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            print("") 
    
    
    def simulate(self,model,inputstate=None,store=True,returnall=False):        
        """
        A wrap-around for the circuit simulators in simulators.py. I like
        being able to do circuit.simulate()...!
        """            
        return _sim.simulate(self,model,inputstate=inputstate,store=store,returnall=returnall)


    
    #
    #
    #
    # FUNCTION NEEDS UPDATING! 
    #
    #
    #
    #def construct_layer_as_superoperator(self,layer_index,gate_superoperators,store=True):
    #    """
    #    This function expects a complete circuit layer, whereby every
    #    qubit has a gate, so it is a list with a length that is the
    #    number of qubits.
    #    """
    #    
    #    #
    #    # TO DO : THIS SHOULD FIRST CHECK IF THE LAYER IS ALREADY STORED AS A SUPEROPERATOR!!!!!
    #    #
    #    #
    #    n = self.number_of_qubits
    #    nn = 4**n
    #    layer = self.get_circuit_layer(layer_index)
    #    layer_superoperator = _np.zeros((nn,nn),float)
    #    
    #    # This is a list of those qubits for which there is a 1 gate in the layer
    #    qubits_with_local_gate = []
    #    # This ....
    ##    qubit_pairs_with_2qubit_gate = []
    #    
    #    for i in range(0,n):
    #        if layer[i].number_of_qubits == 1:
    #            qubits_with_local_gate.append(layer[i].qubits[0])
    #        if layer[i].number_of_qubits == 2:
    #            if (layer[i].qubits[0],layer[i].qubits[1]) not in qubit_pairs_with_2qubit_gate:
    #                qubit_pairs_with_2qubit_gate.append((layer[i].qubits[0],layer[i].qubits[1]))
    #    
    #    n_local = len(qubits_with_local_gate)
    #    n_2q = len(qubit_pairs_with_2qubit_gate)
    #    #print(n_2q,n_local)
    # 
    #    # This contains the superoperators in the layer, to be quickly accessed below 
    #    gates_in_layer = {}
    #    for gate in layer:
    #        #print(gate)
    #        if gate.number_of_qubits == 1:
    #            gates_in_layer[gate.qubits[0]] = gate_superoperators[gate]
    ##        else:
    #            if (gate.qubits[0],gate.qubits[1]) in qubit_pairs_with_2qubit_gate:
    #                gates_in_layer[(gate.qubits[0],gate.qubits[1])] = gate_superoperators[gate]
    #            
    #    #print(gates_in_layer)
    #    #print(nn)
    #    for i in range(0,nn):
    #        for j in range(0,nn):
    #            
    #            ii = [0 for x in range(0,n)]
    #            base_rep =  [int(x) for x in _np.base_repr(i,4)]
    #            ii[n-len(base_rep):] = base_rep
    #            jj = [0 for x in range(0,n)]
    #            base_rep =  [int(x) for x in _np.base_repr(j,4)]
    #            jj[n-len(base_rep):] = base_rep
    #             
    #            # To store the values that are multiplied together to get the
    #            # value of layer_superoperator[i,j].
    #            elements = _np.zeros(n_local+n_2q,float)
    #            
    #            # Find the values for the 1 qubit gates.
    #            for k in range(0,n_local):
    #                q = qubits_with_local_gate[k]
    #                elements[k] = gates_in_layer[q][ii[q],jj[q]]
    #                
    #            # Find the values for the 2 qubit gates
    #            for k in range(0,n_2q):
    #                q1q2 = qubit_pairs_with_2qubit_gate[k]
    #                q1 = q1q2[0]
    #                q2 = q1q2[1]
    #                elements[k+n_local] = gates_in_layer[q1q2][4*ii[q1]+ii[q2],4*jj[q1]+jj[q2]]
    #                            
    #            layer_superoperator[i,j] =  _np.prod(elements)
    #    
    #    if store:
    #        self.layers_as_superoperators[tuple(layer)] = layer_superoperator
    #    
    #    return layer_superoperator
    
    #def construct_layer_as_unitary(self,layer_index,gate_unitaries,store=True):
    #    """
    #   This function expects a complete circuit layer, whereby every
    #    qubit has a gate, so it is a list with a length that is the
    #    number of qubits.
    #    """
    #    #
    #    # TO DO : THIS SHOULD FIRST CHECK IF THE LAYER IS ALREADY STORED AS A SUPEROPERATOR!!!!!
    #    #
    #    #
    #    n = self.number_of_qubits
    #    nn = 2**n
    #    layer = self.get_circuit_layer(layer_index)
    #    layer_unitary = _np.zeros((nn,nn),complex)
    #    
    #    # This is a list of those qubits for which there is a 1 gate in the layer
    #    qubits_with_local_gate = []
    #    # This ....
    #    qubit_pairs_with_2qubit_gate = []
    #    
    #    for i in range(0,n):
    #        if layer[i].number_of_qubits == 1:
    #            qubits_with_local_gate.append(layer[i].qubits[0])
    #        if layer[i].number_of_qubits == 2:
    #            if (layer[i].qubits[0],layer[i].qubits[1]) not in qubit_pairs_with_2qubit_gate:
    #                qubit_pairs_with_2qubit_gate.append((layer[i].qubits[0],layer[i].qubits[1]))
    #    
    #    n_local = len(qubits_with_local_gate)
    #    n_2q = len(qubit_pairs_with_2qubit_gate)
    #
    #    gates_in_layer = {}
    #    for gate in layer:
    #        if gate.number_of_qubits == 1:
    #            gates_in_layer[gate.qubits[0]] = gate_unitaries[gate]
    #        else:
    #            if (gate.qubits[0],gate.qubits[1]) in qubit_pairs_with_2qubit_gate:
    #                gates_in_layer[(gate.qubits[0],gate.qubits[1])] = gate_unitaries[gate]
    #            
    #    for i in range(0,nn):
    #        for j in range(0,nn):
    #            
    #            ii = [0 for x in range(0,n)]
    #            base_rep =  [int(x) for x in _np.base_repr(i,2)]
    #            ii[n-len(base_rep):] = base_rep
    #            jj = [0 for x in range(0,n)]
    #            base_rep =  [int(x) for x in _np.base_repr(j,2)]
    #            jj[n-len(base_rep):] = base_rep
    #             
    #            # To store the values that are multiplied together to get the
    #            # value of layer_superoperator[i,j].
    #            elements = _np.zeros(n_local+n_2q,complex)
    #            
    #            # Find the values for the 1 qubit gates.
    #            for k in range(0,n_local):
    #                q = qubits_with_local_gate[k]
    #                elements[k] = gates_in_layer[q][ii[q],jj[q]]
    #                
    #            # Find the values for the 2 qubit gates
    #            for k in range(0,n_2q):
    #                q1q2 = qubit_pairs_with_2qubit_gate[k]
    #                q1 = q1q2[0]
    #                q2 = q1q2[1]
    #                elements[k+n_local] = gates_in_layer[q1q2][2*ii[q1]+ii[q2],2*jj[q1]+jj[q2]]
    #                            
    #            layer_unitary[i,j] =  _np.prod(elements)
    #    
    #    if store:
    #        self.layers_as_unitaries[tuple(layer)] = layer_unitary
    #    
    #    return layer_unitary

    #def get_layer_as_superoperator(self,layer_index,gate_superoperators,store=True):
    #    """
    #    Todo
    #    """
    #    layer = self.get_circuit_layer(layer_index)
    #    
    #    try:
    #        LAS = self.layers_as_superoperator[tuple(layer)]
    #    except:
    #        LAS = self.construct_layer_as_superoperator(layer_index,gate_superoperators,store)        
    #
    #    return LAS
    
    #def get_layer_as_unitary(self,layer_index,gate_unitaries,store=True):
    #    """
    #    Todo
    #    """
    #    layer = self.get_circuit_layer(layer_index)
    #    
    #    try:
    #        LAU = self.layers_as_unitaries[tuple(layer)]
    #    except:     
    #        LAU = self.construct_layer_as_unitary(layer_index,gate_unitaries,store)
    #
    #    return LAU
    
    #def simulate(self,ds,simtype='purestate',inputstate=None,store=True,returnall=False):
    #            
    #    if simtype == 'densitymatrix':
    #        out = self.densitymatrix_simulator(ds, input_bitstring=inputstate, store=store, returnall=returnall)
    #    
    #    if simtype == 'purestate':
    #        out = self.vectorstate_simulator(ds, inputstate=inputstate, store=store, returnall=returnall)
    #    
    #    return out
    
    #def densitymatrix_simulator(self,ds,input_bitstring=None,store=True,returnall=False):
    #    """
    #    Todo: docstring
    #    
    #    Probably should have an input as a bitstring. Should allow a general density matrix and measurment effect.
    #    """
    #    n = self.number_of_qubits
    #    if input_bitstring is None:           
    #        input_bitstring = _np.zeros(n,int)
    #    
    #        input_state = _np.array([1.])
    #        for i in range(0,n):
    #            input_state = _np.kron(input_state,_np.array([1.,0.,0.,(-1.)**(input_bitstring[i])]))
    ##        input_state = input_state / _np.sqrt(2)**n                      
    #    
    #    output_state = _np.copy(input_state)
    ##                                   
    #    for l in range(0,self.depth()):
    #        superoperator = ds.get_layer_as_superoperator(self.get_circuit_layer(l),store=store)
    #        output_state = _np.dot(superoperator,output_state)
    #   
    #    probabilities = {}
    #    for i in range(0,2**n):
    #        bit_string = [0 for x in range(0,n)]
    #        bit_string_end =  [int(x) for x in _np.base_repr(i,2)]
    #        bit_string[n-len(bit_string_end):] = bit_string_end
    #        
    #        possible_outcome = _np.array([1.])
    #        for j in range(0,n):
    #            possible_outcome = _np.kron(possible_outcome,_np.array([1.,0.,0.,(-1.)**bit_string[j]]))
    #                
    #        probabilities[tuple(bit_string)] = _np.dot(possible_outcome,output_state)/(_np.sqrt(2)**n)
    #    
    #    if returnall:
    #        return probabilities, output_state
    #    else:
    #        return probabilities
    
    #
    # Todo: add in the possibility of Pauli errors into the vector state simulator
    #
    #def vectorstate_simulator(self,ds,inputstate=None,store=True,returnall=False):
    #    """
    #    Todo
    #    """
    #    n = self.number_of_qubits
    #    if inputstate is None:           
    #        inputstate = _np.zeros(2**n,complex)
    #        inputstate[0] = 1
    #
    #    outputstate = _np.copy(inputstate)
    #                                   
    #    for l in range(0,self.depth()):
    #        unitary = ds.get_circuit_layer_as_unitary(self.get_circuit_layer(l),store=store)
    #        outputstate = _np.dot(unitary,outputstate)
    #
    #    probs = abs(outputstate)**2
    #    probs_as_dict = {}
    #    for i in range(0,2**n):
    #        bit_string = [0 for x in range(0,n)]
    #        bit_string_end =  [int(x) for x in _np.base_repr(i,2)]
    #        bit_string[n-len(bit_string_end):] = bit_string_end
    #        probs_as_dict[tuple(bit_string)] = probs[i]
    #    
    #    if returnall:
    #        return probs_as_dict, outputstate
    #    else:
    #        return probs_as_dict
    
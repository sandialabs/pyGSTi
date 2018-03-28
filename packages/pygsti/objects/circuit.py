""" Defines the Circuit class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numbers as _numbers
import numpy as _np
import copy as _copy

from . import gatestring as _gstr
from .label import Label as _Label

IDENT = 'I' # Identity-get sentinel
#FUTURE?
#class CircuitLayer(object):
#    
#    def __init__(self, gatelist, n):
#        self.layer = 0

def check_valid_gate(gate):
    
    # To do: check that gate is type Gate.
    assert(type(gate.name) is unicode), "The gate label should be unicode!"
    assert(type(gate.qubits) is list), "The list of qubits on which the gate acts should be a list!"
    
    for i in range(1,gate.number_of_qubits):
        assert(type(gate.qubits[i]) is int), "The qubits on which a gate acts should be indexed by integers!"

def check_valid_circuit_layer(layer,n):
    
    assert(type(layer) is list), "A gate layer must be a list!"
    #
    # To do: add other less trivial tests.
    #

class Circuit(_gstr.GateString):
    """
    TODO - docstring
    A circuit is a more structured representation of a sequence of preparation,
    gate, and measurement operations.  It is composed of some number of "lines",
    typically one per qubit.
    """
    
    def __init__(self, line_items=None, gatestring=None, num_lines=None, line_labels=None):
        """
        An object with encapsulates a quantum circuit, and which contains a range of methods for
        manipulating quantum circuits. E.g., basic depth compression algorithms.
    
        """
        assert((line_items is not None) or (gatestring is not None) or (num_lines is not None) or (line_labels is not None)), \
            "At least one argument must be not None!"
        self._static = False

        if (num_lines is not None) and (line_labels is not None):
            assert(num_lines == len(line_labels)), "Inconsistent `num_lines` and `line_labels`!"
                
        if line_items is not None:
            assert(gatestring is None), "Cannot specify both `line_items` and `gatestring`"
            self.line_items = line_items
            self.number_of_lines = len(self.line_items)
            if num_lines is not None:
                assert(num_lines == self.number_of_lines), "Inconsistent `line_items` and `num_lines` arguments!"
            if line_labels is not None:
                assert(len(line_labels) == self.number_of_lines), "Inconsistent `line_items` and `line_labels` arguments!"
                self.line_labels = line_labels
            else:
                self.line_labels = list(range(self.number_of_lines))

        else:
            assert(num_lines is not None or line_labels is not None), \
                "`num_lines` or `line_labels`  must be specified whenever `line_items` is None!"
            if num_lines is not None:
                assert(isinstance(num_lines, _numbers.Integral)), "`num_lines` must be an integer!"
                self.number_of_lines = num_lines
            if line_labels is not None:
                self.line_labels = line_labels
                self.number_of_lines = len(line_labels)
            else:
                self.line_labels = list(range(self.number_of_lines))
            
            if gatestring is not None:
                self.initialize_circuit_from_gatestring(gatestring)
            else:
                self.initialize_empty_circuit()

        self._reinit_base()

    def _reinit_base(self):
        """ Re-initialize the members of the GateString base class """

        if self.number_of_lines <= 0:
            super(Circuit, self).__init__(())
            return

        label_list = []
        nlayers = max( [len(line) for line in self.line_items] )
        
        #Add gates
        for j in range(nlayers): # j = layer index
            processed_lines = set()
            for line_lbl,line in zip(self.line_labels, self.line_items):
                if line_lbl in processed_lines: continue # we've already added the gate/item on this line (e.g. 2Q gates)
                if len(line) <= j: continue # this line doesn't have a j-th layer (is this possible?)

                lbl = line[j]
                if line[j].name != IDENT:
                    label_list.append( line[j] ) # don't include exact identities
                actson = lbl.qubits if (lbl.qubits is not None) else self.line_labels # None == "all lines"
                processed_lines.update(actson)

        strRep = None #don't worry about string representation currently - just auto-compute it.
        super(Circuit, self).__init__(tuple(label_list), strRep)

    def __hash__(self):
        if self._static:
            _warnings.warning(("Editable circuit is being converted to read-only"
                               " mode in order to hash it.  You should call"
                               " circuit.done_editing() beforehand."))
            self.done_editing()
        return super(Circuit,self).__hash__()
                      
    def initialize_empty_circuit(self):
        """
        Creates an empty circuit, which consists of a dictionary
        of empty lists.
        
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        self.line_items = []
        for i in range(0,self.number_of_lines):
            self.line_items.append([])
                        
    def initialize_circuit_from_gatestring(self,gatestring):
        """
        Creates a circuit from a list of Gate objects. This function assumes that 
        all gates act on 1 or 2 qubits only, but this could be easily updated if 
        necessary.
        
        Parameters
        ----------
        gatestring : GateString or tuple
            A sequence of state preparation (optional), gates, and measurement
            (optional), given by a :class:`GateString` object.
        
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        
        # Get a proper GateString object if one isn't given to us
        if not isinstance(gatestring, _gstr.GateString):
            gatestring = _gstr.GateString(gatestring)

        # Note: elements of gatestring are guaranteed to be Label objs b/c GateString enforces this.
            
        # Create an empty circuit, to populate from gatestring
        self.initialize_empty_circuit()

        # keeps track of which gates have been added to the circuit from the list.
        j = 0
        # keeps track of the circuit layer number
        layer_number = 0
        
        while j < len(gatestring):
            
            # The gates that are going into this layer.
            layer = []
            # The number of gates beyond j that are going into this layer.
            k = 0
            # The qubits used in this layer.
            used_qubits = set()
    
            while j+k < len(gatestring):
                
                # look at the j+kth gate and include in this layer
                gate = gatestring[j+k]
                gate_qubits = gate.qubits if (gate.qubits is not None) \
                              else self.line_labels  # then gate uses *all* lines

                if len(used_qubits.intersection(gate_qubits)) > 0:
                    break # `gate` can't fit in this layer
                    
                layer.append(gate)
                used_qubits.update(gate_qubits)                    
                    
                # look at the next gate in the list, which will be
                # added to the layer if it does not act on any qubits
                # with a gate already in this layer
                k += 1                
            
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
        check_valid_gate(gate)
        
        # Add an idle layer.
        for i in range(0,self.number_of_lines):
            self.line_items[i].insert(j,_Label(IDENT,self.line_labels[i]))
            
        # Put the gate in
        for i in gate.qubits:
            self.line_items[i][j] = gate
            
        self._reinit_base() #REINIT

        
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
        check_valid_circuit_layer(circuit_layer,self.number_of_lines)
        
        # Add an idle layer.
        for i in range(0,self.number_of_lines):
            self.line_items[i].insert(j,_Label(IDENT,self.line_labels[i]))
            
        # Put the gates in.
        for i,line_label in enumerate(self.line_labels):
            for gate in circuit_layer:
                gate_qubits = gate.qubits if (gate.qubits is not None) \
                              else self.line_labels
                if line_label in gate_qubits:
                    self.line_items[i][j] = gate
                    
        self._reinit_base() #REINIT
                    
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
        assert(self.number_of_lines == circuit.number_of_lines), "The circuits must act on the same number of qubits!"
        
        for q in range(0,self.number_of_lines):
            self.line_items[q] = self.line_items[q][0:j] + circuit.line_items[q][:] + self.line_items[q][j:]
            
        self._reinit_base() #REINIT
                            
    def append_circuit(self,circuit):
        """
        Append a circuit to the end of this circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be appended.
 
        """        
        assert(self.number_of_lines == circuit.number_of_lines), "The circuits must act on the same number of qubits!"
        
        for q in range(0,self.number_of_lines):
            self.line_items[q] = self.line_items[q] + circuit.line_items[q][:]
        self._reinit_base() #REINIT
            
    def prefix_circuit(self,circuit):
        """
        Prefix a circuit to the end of this circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be prefixed.
 
        """           
        assert(self.number_of_lines == circuit.number_of_lines), "The circuits must act on the same number of qubits!"
        
        for q in range(0,self.number_of_lines):
            self.line_items[q] = circuit.line_items[q][:] + self.line_items[q]
        self._reinit_base() #REINIT

        
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
        n = self.line_items[q][j].number_of_qubits 
        assert(n == 1 or n == 2), "Only circuits with only 1 and 2 qubit gates supported!" 
        
        # Replace the gate with an idle
        if n == 1:
            self.line_items[q][j] = _Label(IDENT,self.line_labels[q])
            
        else:
           
            q1 = self.line_items[q][j].qubits[0]
            q2 = self.line_items[q][j].qubits[1]
            self.line_items[q1][j] = _Label(IDENT,self.line_labels[int(q1)])
            self.line_items[q2][j] = _Label(IDENT,self.line_labels[int(q2)])
        
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
        check_valid_circuit_layer(circuit_layer,self.number_of_lines)
        
        # Replace all gates with idles, in the layer to be replaced.
        for q in range(0,self.number_of_lines):
            self.line_items[q][j] = _Label(IDENT,self.line_labels[q])
        
        # Write in the gates, from the layer to be inserted.
        for q in range(0,self.number_of_lines):
            for qq in range(0,len(circuit_layer)):
                if q in circuit_layer[qq].qubits:
                    self.line_items[q][j] = circuit_layer[qq]
        self._reinit_base() #REINIT

        
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
            
            
    def replace_gatename(self,old_gatename,new_gatename):
        """
        Changes the name of a gate. E.g., can change all instances
        of 'I' to 'Gi'.
        """
        depth = self.depth()
        for q in range(self.number_of_lines):
            for l in range(depth):
                if self.line_items[q][l].name == old_gatename:
                    self.line_items[q][l].name = new_gatename #_Label(new_gatename, self.line_items[q][l].qubits)
        
        self._reinit_base() #REINIT
        
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
        in_compilation = compilation # _copy.deepcopy(compilation)
        
        d = self.depth()
        n = self.number_of_lines
        for l in range(0,d):
            for q in range(0,n):
                gate = self.line_items[q][d-1-l]
                # We ignore idle gates, and do not compile them.
                # To do: is this the behaviour we want?
                if gate.name != IDENT:
                    if gate in in_compilation.keys():
                        self.replace_gate_with_circuit(in_compilation[gate],q,d-1-l)
                    else:
                        # To do: replace with a suitable assert somewhere.
                        print('Error: could not find ', gate, ' in ', list(in_compilation.keys()))
        
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
        for q in range(0,self.number_of_lines):
            del self.line_items[q][j]
        self._reinit_base() #REINIT
                    
    def get_circuit_layer(self,j):
        """
        Returns the layer at depth j.

        """       
        assert(j >= 0 and j < self.depth()), "Circuit layer label invalid! Circuit is only of depth {}".format(self.depth())
        
        layer = []
        for i in range(0,self.number_of_lines):
            layer.append(self.line_items[i][j])
            
        return layer
    
    def reverse(self):
        """
        Reverse the order of the circuit.

        """         
        for q in range(0,self.number_of_lines):
            self.line_items[q].reverse()
        self._reinit_base() #REINIT
    
    def depth(self):
        """
        Returns the circuit depth.
        
        """        
        return len(self.line_items[0])
    
    def size(self):
        """
        Returns the circuit size, whereby a gate defined as acting
        on n-qubits of size n, with the exception of the idle gate
        which is of size 0. Hence, the circuit size is circuit
        depth X number of qubits - number of idle gates in the
        circuit.
        """
        size = 0
        for q in range(0,self.number_of_lines):
            for j in range(0,self.depth()):
                if self.line_items[q][j].name != IDENT:
                    size += 1
        return size
    
    def twoqubit_gatecount(self):
        """
        Returns the number of two-qubit gates in the circuit
        
        """           
        count = 0
        for q in range(0,self.number_of_lines):
            for j in range(0,self.depth()):
                if self.line_items[q][j].number_of_qubits == 2:
                    count += 1
        return count//2

    def __str__(self):
        """
        To do: replace this with a __str__ method.
        
        """
        s = ''
        for i in range(0,self.number_of_lines):
            s += 'Qubit {} ---'.format(i)
            for j in range(0,self.depth()):
                if self.line_items[i][j].name == IDENT:
                    # Replace with special idle print at some point
                    s += '|  |-'
                elif self.line_items[i][j].number_of_qubits == 1:
                    s += '|' + self.line_items[i][j].name + ' |-'
                elif self.line_items[i][j].name == 'CNOT':
                    if self.line_items[i][j].qubits[0] is i:
                        s += '|'+'C' + str(self.line_items[i][j].qubits[1]) + '|-'
                    else:
                        s += '|'+'T' + str(self.line_items[i][j].qubits[0]) + '|-'
                else:
                    s += self.line_items[i][j].__str__() + '|-|' 
            s += '--\n'

        return s
    
    def write_qcircuit_tex(self,filename):
        
        n = self.number_of_lines
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
        
        n = self.number_of_lines
        for q in range(0,n):
            qstring = '&'
            circuit_for_q = self.line_items[q]
            for gate in circuit_for_q:
                if gate.name == IDENT:
                    qstring += ' \qw &'
                elif gate.name == 'CNOT':
                    if gate.qubits[0] == q:
                        qstring += ' \ctrl{'+str(gate.qubits[1]-q)+'} &'
                    else:
                        qstring += ' \\targ &'
                elif gate.name == 'CPHASE':
                    if gate.qubits[0] == q:
                        qstring += ' \ctrl{'+str(gate.qubits[1]-q)+'} &'
                    else:
                        qstring += ' \control \qw &'
            
                else:
                    qstring += ' \gate{'+str(gate.name)+'} &'
                    
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
        for q in range(0,self.number_of_lines):
            j = 0
            while j < self.depth()-1:
                k = j
                if self.line_items[q][k].number_of_qubits == 1:
                    for i in range(k+1,self.depth()):
                        if self.line_items[q][i].number_of_qubits > 1:
                            j += 1
                            break
                        else:
                            # Flag set to True if a non-trivial shift/combination is to be implemented.
                            if not self.line_items[q][i].name == IDENT:
                                flag = True
                                
                            # Find the new label of the gate, according to the combination rules.
                            gl1 = self.line_items[q][k].name
                            gl2 = self.line_items[q][i].name
                            new_label = single_qubit_gate_combined(gl1,gl2)
                            
                            # Becuase of the way hashing is defining for the Gate object, we could
                            # just change the gates labels. But it seems like better practice to make
                            # new Gate objects.
                            self.line_items[q][k] = _Label(new_label,self.line_labels[q])
                            self.line_items[q][i] = _Label(IDENT,self.line_labels[q])
                            j += 1
                else:
                    j += 1

        self._reinit_base() #REINIT
        return flag
    
    def shift_1q_gates_forward(self):
        """
        To do: docstring
        
        """               
        flag = False
        
        for q in range(0,self.number_of_lines):
            for j in range(1,self.depth()):
                
                if self.line_items[q][j].number_of_qubits == 1:                  
                    
                    if self.line_items[q][j].name != IDENT:
                    
                        for k in range(1,j+1):
                            idle = True
                            if self.line_items[q][j-k].name != IDENT:
                                idle = False
                                if not idle:
                                    k = k - 1
                                    break
                                
                        if k > 0:
                            flag = True
                            self.line_items[q][j-k] = self.line_items[q][j]
                            self.line_items[q][j] = _Label(IDENT,self.line_labels[q])
        self._reinit_base() #REINIT
        return flag
    
    def shift_2q_gates_forward(self):
        """
        To do: docstring
        
        """                
        flag = False
        
        for q in range(0,self.number_of_lines):
            for j in range(1,self.depth()):
                if self.line_items[q][j].number_of_qubits == 2:                  
                    # Only try to do a compression if q is the first qubit of the pair, as we only 
                    # need to consider one of the two.
                    if self.line_items[q][j].qubits[0] == q:
                        target_label = self.line_labels.index(self.line_items[q][j].qubits[1]) # TODO: inefficient...
                        
                        for k in range(1,j+1):
                            both_idle = True
                            if self.line_items[q][j-k].name != IDENT:
                                both_idle = False
                            if self.line_items[target_label][j-k].name != IDENT:
                                both_idle = False
                            if not both_idle:
                                k = k - 1
                                break
                        if k > 0:
                            flag = True
                            self.line_items[q][j-k] = self.line_items[q][j]
                            self.line_items[target_label][j-k] = self.line_items[target_label][j]
                            self.line_items[q][j] = _Label(IDENT,self.line_labels[q])
                            self.line_items[target_label][j] = _Label(IDENT,self.line_labels[int(target_label)])

        self._reinit_base() #REINIT
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
            
            for q in range(0,self.number_of_lines):
                if layer[q].name != IDENT:
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
    
    
    def simulate(self,gateset): #,inputstate=None,store=True,returnall=False):        
        """
        A wrap-around for the circuit simulators in simulators.py. I like
        being able to do circuit.simulate()...!
        """
        return gateset.probs(self)
        #return _sim.simulate(self,model,inputstate=inputstate,store=store,returnall=returnall)

    def done_editing(self):
        """
        Make this Circuit read-only, so that it can be hashed (used as a
        dictionary key).
        """
        self._static = True

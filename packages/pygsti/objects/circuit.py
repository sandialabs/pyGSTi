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
from ..baseobjs import Label as _Label

IDENT = 'I' # Identity-get sentinel
#FUTURE?
#class CircuitLayer(object):
#    
#    def __init__(self, gatelist, n):
#        self.layer = 0


#OLD: checks whether a Label is valid - could update this & move to Label?
#def check_valid_gate(gate):
#    # To do: check that gate is type Gate.
#    assert(type(gate.name) is unicode), "The gate label should be unicode!"
#    assert(type(gate.qubits) is list), "The list of qubits on which the gate acts should be a list!"
#    
#    for i in range(1,gate.number_of_qubits):
#        assert(type(gate.qubits[i]) is int), "The qubits on which a gate acts should be indexed by integers!"

#OLD - could add logic like this to Circuit?
#def check_valid_circuit_layer(layer,n):
#    
#    assert(type(layer) is list), "A gate layer must be a list!"
#    #
#    # To do: add other less trivial tests.
#    #

class Circuit(_gstr.GateString):
    """
    A Circuit is a more structured representation of a sequence of preparation,
    gate, and measurement operations.  It is composed of some number of "lines",
    typically one per qubit, and permits a richer set of operations than 
    :class:`GateString`.  These operations include a range of methods for
    manipulating quantum circuits. E.g., basic depth compression algorithms.
    """
    
    def __init__(self, line_items=None, gatestring=None, num_lines=None,
                 line_labels=None, parallelize=False):
        """
        Creates a new Circuit object, encapsulating a quantum circuit.

        You can supply at most one of `line_items` and `gatestring` (alternate
        ways of specifying the gates).  If neither are specified an empty
        Circuit is created.  Unless `line_items` is specified, you must also
        specify the number and label for each line via `num_lines` and/or
        `line_labels` (if labels aren't specified they default to the integers
        beggining with 0).

        Parameters
        ----------
        line_items : list, optional
            A list of lists of Label objects, giving the gate labels present on
            each "line" (usually "qubit") of the circuit.  When a gate occupies
            more than one line its label is present in the lists of *all* the
            lines it occupies.

        gatestring : list, optional
            A list of Label objects, specifying a serial sequence of gates to
            be made into a circuit.  Gates are placed in as few "layers" as 
            possible, meaning that gates acting on different qubits are 
            done in parallel (placed at the same position in their respective
            lines).  This is equivalent to saying the list for each line is 
            made as short as possible.

        num_lines : int, optional
            The number of lines (usually qubits).  You only need to specify 
            this when it cannot be inferred, that is, when `line_items` and
            `line_labels` are not given.

        line_labels : list, optional
            A list of strings or integers labelling each of the lines.  The
            length of this list equals the number of lines in the circuit, 
            and if `line_labels` is not given these labels default to the 
            integers starting with 0.

        parallelize : bool, optional
            Only used when initializing from `gatestring`.  When True, automatic
            parallelization is performed: consecutive gates in `gatestring`
            acting on disjoint sets of qubits are be placed in the same layer.
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
                self._initialize_from_gatestring(gatestring, parallelize)
            else:
                self.clear() # initializes an empty circuit

            """ Initialize the members of the GateString base class """
            tup = self._flatten_to_tup()
            strRep = None  #don't worry about string representation currently - just auto-compute it.
            super(Circuit, self).__init__(tup, strRep)

            self._tup_dirty = False # keep track of when we need to _flatten_to_tup
            self._str_dirty = True # keep track of when we need to auto-compute string rep

    def _reinit_base(self):
        """ Re-initialize the members of the base GateString object """
        if self._tup_dirty:
            self._tup = self._flatten_to_tup()
            self._tup_dirty = False
        if self._str_dirty:
            self._str = _gstr._gateSeqToStr(self._tup)
            self._str_dirty = False
        
    def _flatten_to_tup(self):
        """ Flatten self.line_items into a serial tuple of gate labels """
        #print("DB: flattening to tuple!!")
        if self.number_of_lines <= 0:
            return ()

        label_list = []
        nlayers = max( [len(line) for line in self.line_items] )
        
        #Add gates
        for j in range(nlayers): # j = layer index
            layer_list = []
            processed_lines = set()
            for line_lbl,line in zip(self.line_labels, self.line_items):
                if line_lbl in processed_lines: continue # we've already added the gate/item on this line (e.g. 2Q gates)
                if len(line) <= j: continue # this line doesn't have a j-th layer (is this possible?)

                lbl = line[j]
                if line[j].name != IDENT: #Note: it's OK to use .name on line items (all are *simple* labels)
                    layer_list.append( line[j] ) # don't include exact identities
                actson = lbl.qubits if (lbl.qubits is not None) else self.line_labels # None == "all lines"
                processed_lines.update(actson)

            if len(layer_list) > 0:
                label_list.append( _Label(layer_list) )
                
        return tuple(label_list)

    @property
    def tup(self):
        """ This Circuit as a standard Python tuple of Labels."""
        # Note: this overrides GateString method so we can compute tuple in on-demand way
        #print("Circuit.tup accessed")
        self._reinit_base()
        return self._tup

    @tup.setter
    def tup(self, value):
        """ This Circuit as a standard Python tuple of Labels."""
        #print("Circuit.tup setter accessed")
        self._reinit_base()
        self._tup = value

    @property
    def str(self):
        """ The Python string representation of this GateString."""
        #print("Circuit.str accessed")
        self._reinit_base()
        return self._str

    @str.setter
    def str(self, value):
        """ The Python string representation of this GateString."""
        #print("Circuit.str setter accessed")
        self._reinit_base()
        self._str = value


    def __hash__(self):
        if self._static:
            _warnings.warning(("Editable circuit is being converted to read-only"
                               " mode in order to hash it.  You should call"
                               " circuit.done_editing() beforehand."))
            self.done_editing()
        return super(Circuit,self).__hash__()


    def map_state_space_labels(self, mapper): # a gate string method that we need to implement correctly for Circuit TODO
        """
        Return a copy of this gate string with all of the state-space-labels
        (often just qubit labels) updated according to a mapping function.

        For example, calling this function with `mapper = {0: 1, 1: 3}`
        on the string "Gx:0Gy:1Gx:1" would return "Gx:1Gy:3Gx:3".

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing state-space-label values
            and whose value are the new labels, or a function which takes a
            single (existing label) argument and returns a new label.

        Returns
        -------
        GateString
        """
        raise NotImplementedError("TODO")

                      
    def clear(self):
        """
        Removes all the gates in a circuit (preserving the number of lines).
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        self.line_items = []
        for i in range(0,self.number_of_lines):
            self.line_items.append([])
        self._tup_dirty = self._str_dirty = True
                        
    def _initialize_from_gatestring(self,gatestring,parallelize):
        """
        Initializes self.line_items from a sequence of Label objects.
        
        Parameters
        ----------
        gatestring : GateString or tuple
            A sequence of state preparation (optional), gates, and measurement
            (optional), given by a :class:`GateString` object.

        parallelize : bool
            Whether or not automatic parallelization should be performed, where
            subsequent gates in `gatestring` acting on disjoint sets of qubits
            should be placed in the same layer.
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        
        # Get a proper GateString object if one isn't given to us
        if not isinstance(gatestring, _gstr.GateString):
            gatestring = _gstr.GateString(gatestring)

        # Note: elements of gatestring are guaranteed to be Label objs b/c GateString enforces this.
            
        # Create an empty circuit, to populate from gatestring
        self.clear() # inits self.line_items

        # keeps track of which gates have been added to the circuit from the list.
        j = 0
        
        # keeps track of the circuit layer number
        layer_number = 0
        
        while j < len(gatestring):
            layer = [] # The gates that are going into this layer.

            if parallelize:
                k = 0 # The number of gates beyond j that are going into this layer.
                used_qubits = set() # The qubits used in this layer.
                while j+k < len(gatestring):
                    
                    # look at the j+kth gate and include in this layer
                    gate = gatestring[j+k] # really a gate *label*
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
            else: # just add the next gate label as the next layer
                k = 1 # The number of gates beyond j that are going into this layer.
                layer.append(gatestring[j])
                
            # Insert the layer into the circuit.
            self.insert_layer(layer,layer_number)
            
            # Move on to the next gate not in included in the circuit.
            j += k
            # Update the layer number.
            layer_number += 1
    
    def insert_gate(self,gatelbl,j):
        """
        Inserts a gate into a circuit.
        
        Parameters
        ----------
        gatelbl : Label
            The gate label to insert.
            
        j : int
            The layer index (depth) at which to insert the gate.

        Returns
        -------
        None
        """
        #OLD Check input is a valid gate
        #OLD check_valid_gate(gatelbl)
        
        # Add an idle layer.
        for i in range(0,self.number_of_lines):
            self.line_items[i].insert(j,_Label(IDENT,self.line_labels[i]))
            
        # Put the gate label in - note this label may
        # be a "parallel-gate" label and have mulitple components.
        for gl_comp in gatelbl.components:
            gate_qubits = gl_comp.qubits if (gl_comp.qubits is not None) \
                          else self.line_labels
            for i in gate_qubits:
                self.line_items[i][j] = gl_comp

        self._tup_dirty = self._str_dirty = True

        
    def insert_layer(self,circuit_layer,j):
        """
        Inserts a layer into a circuit. The input layer does not
        need to contain a gate that acts on every qubit. But,
        it should be a valid layer, meaning that the layer
        does not contain more than one gate on a qubit.
        
        Parameters
        ----------
        circuit_layer : list
            A list of Label objects, to insert as a circuit layer.
            
        j : int
            The layer index (depth) at which to insert the circuit layer.

        Returns
        -------
        None
        """
        #OLD check_valid_circuit_layer(circuit_layer,self.number_of_lines)
        
        # Add an idle layer.
        for i in range(0,self.number_of_lines):
            self.line_items[i].insert(j,_Label(IDENT,self.line_labels[i]))
            
        # Put the gates in.
        for i,line_label in enumerate(self.line_labels):
            for gatelbl in circuit_layer:
                # circuit layer can contain "parallel" gate layers, unlike
                # the values of self.line_items which are all simple labels
                for gl_comp in gatelbl.components:
                    gate_qubits = gl_comp.qubits if (gl_comp.qubits is not None) \
                              else self.line_labels
                    if line_label in gate_qubits:
                        self.line_items[i][j] = gl_comp
                    
        self._tup_dirty = self._str_dirty = True
                    
    def insert_circuit(self,circuit,j):
        """
        Inserts a circuit into this circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be inserted.
            
        j : int
            The layer index (depth) at which to insert the circuit.
            
        Returns
        -------
        None
        """        
        assert(self.number_of_lines == circuit.number_of_lines), "The circuits must act on the same number of qubits!"
        
        for q in range(0,self.number_of_lines):
            self.line_items[q] = self.line_items[q][0:j] + circuit.line_items[q][:] + self.line_items[q][j:]
            
        self._tup_dirty = self._str_dirty = True
                            
    def append_circuit(self,circuit):
        """
        Append a circuit to the end of this circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be appended.
 
        Returns
        -------
        None
        """        
        assert(self.number_of_lines == circuit.number_of_lines), "The circuits must act on the same number of qubits!"
        
        for q in range(0,self.number_of_lines):
            self.line_items[q] = self.line_items[q] + circuit.line_items[q][:]
        self._tup_dirty = self._str_dirty = True
            
    def prefix_circuit(self,circuit):
        """
        Prefix a circuit to the end of this circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be prefixed.
 
        Returns
        -------
        None
        """           
        assert(self.number_of_lines == circuit.number_of_lines), "The circuits must act on the same number of qubits!"
        
        for q in range(0,self.number_of_lines):
            self.line_items[q] = circuit.line_items[q][:] + self.line_items[q]
        self._tup_dirty = self._str_dirty = True

        
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
            The layer index (depth) of the gate to be replaced.
 
        Returns
        -------
        None
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
            The index (depth) of the layer to be replaced.
 
        Returns
        -------
        None
        """
        #OLD check_valid_circuit_layer(circuit_layer,self.number_of_lines)
        
        # Replace all gates with idles, in the layer to be replaced.
        for q in range(0,self.number_of_lines):
            self.line_items[q][j] = _Label(IDENT,self.line_labels[q])
        
        # Write in the gates, from the layer to be inserted.
        for q in range(0,self.number_of_lines):
            for gatelbl in circuit_layer:
                for sub_gl in gatelbl.components:
                    if q in sub_gl.qubits:
                        self.line_items[q][j] = sub_gl
        self._tup_dirty = self._str_dirty = True

        
    def replace_layer_with_circuit(self,circuit,j):
        """
        Replace a layer with a circuit.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be inserted in place of the layer.
  
        j : int
            The index (depth) of the layer to be replaced.
 
        Returns
        -------
        None
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
        Changes the *name* of a gate throughout this Circuit.

        Note that the "name" is only a part of the "label" identifying each
        gate, and doesn't include the lines (qubits) a gate acts upon.  For
        example, the "Gx:0" and "Gx:1" labels both have the same name but 
        act on different qubits.

        Another possible use example is the ability to change all of the
        default idle gates (`'I'`), which are added as "padding" during
        circuit construction methods, to something else like `'Gi'`.

        Parameters
        ----------
        old_gatename, new_gatename : string
            The gate name to find and the gate name to replace the found
            name with.

        Returns
        -------
        None
        """
        depth = self.depth()
        for q in range(self.number_of_lines):
            for l in range(depth):
                if self.line_items[q][l].name == old_gatename:
                    #self.line_items[q][l].name = new_gatename # This doesn't work now for some reason.
                    self.line_items[q][l] = _Label(new_gatename, self.line_items[q][l].qubits)
        self._tup_dirty = self._str_dirty = True

        
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
            
        Returns
        -------
        None 
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
        
    def relabel_qubits(self,order):
        """
        Todo : docstring

            The quantum wire for qubit i becomes
            the quantum wire for qubit order[i]
        """
        original_circuit = _copy.deepcopy(self.line_items)
        #for i in range(0,circuit.number_of_qubits):
        #    relabelled_circuit.line_items[order[i]] = circuit.line_items[i]

        depth = self.depth()
        for i in range(0,self.number_of_lines):
            for j in range(0,depth):
                gate = original_circuit[i][j]
                self.line_items[order[i]][j] = _Label(gate.name,tuple([order[k] for k in gate.qubits]))

    def delete_layer(self,j):
        """
        Delete a layer from the circuit.
        
        Parameters
        ----------
        j : int
            The index (depth) of the layer to be deleted.
 
        Returns
        -------
        None
        """
        for q in range(0,self.number_of_lines):
            del self.line_items[q][j]
        self._tup_dirty = self._str_dirty = True

        
    def get_circuit_layer(self,j):
        """
        Returns the layer at depth j.

        Returns
        -------
        None
        """       
        assert(j >= 0 and j < self.depth()), "Circuit layer label invalid! Circuit is only of depth {}".format(self.depth())
        
        layer = []
        for i in range(0,self.number_of_lines):
            layer.append(self.line_items[i][j])
            
        return layer
    
    def reverse(self):
        """
        Reverse the order of the circuit.

        Returns
        -------
        None
        """         
        for q in range(0,self.number_of_lines):
            self.line_items[q].reverse()
        self._tup_dirty = self._str_dirty = True

        
    def depth(self):
        """
        The circuit depth.
        
        Returns
        -------
        int
        """        
        return len(self.line_items[0])
    
    def size(self):
        """
        Computes the circuit size, whereby a gate defined as acting on
        n-qubits of size n, with the exception of the special `I` idle 
        gate which is of size 0. Hence, the circuit size is circuit
        depth X number of qubits - number of idle gates in the
        circuit.
        
        Returns
        -------
        int        
        """
        size = 0
        for q in range(0,self.number_of_lines):
            for j in range(0,self.depth()):
                if self.line_items[q][j].name != IDENT:
                    size += 1
        return size
    
    def twoqubit_gatecount(self):
        """
        The number of two-qubit gates in the circuit.
        
        Returns
        -------
        int
        """           
        count = 0
        for q in range(0,self.number_of_lines):
            for j in range(0,self.depth()):
                if self.line_items[q][j].number_of_qubits == 2:
                    count += 1
        return count//2
    
    def predicted_infidelity(self,fidelities):
        
        f = 1.
        
        gatestring = self._flatten_to_tup()
        for label in gatestring:
            
            f = f*fidelities[label]
        
        return 1 - f

    def __str__(self):
        """
        A text rendering of the circuit.
        """
        s = ''

        def abbrev(lbl,k): #assumes a simple label w/ name & qubits
            """ Returns what to print on line 'k' for label 'lbl' """
            if lbl.number_of_qubits == 1:
                return lbl.name
            elif lbl.name in ('CNOT','Gcnot'): # qubit indices = (control,target)
                # Tim: display *other* CNOT qubit on each line
                if k == lbl.qubits[0]: return 'C' + str(lbl.qubits[1])
                else:                  return 'T' + str(lbl.qubits[0])          
            else:
                return str(lbl)
        
        max_labellen = [ max([ len(abbrev(self.line_items[i][j],i))
                               for i in range(0,self.number_of_lines)])
                         for j in range(0,self.depth()) ]

        for i in range(0,self.number_of_lines):
            s += 'Qubit {} ---'.format(i)
            for j,maxlbllen in enumerate(max_labellen):
                if self.line_items[i][j].name == IDENT:
                    # Replace with special idle print at some point
                    s += '-'*(maxlbllen+3) # 1 for each pipe, 1 for joining dash
                else:
                    lbl = abbrev(self.line_items[i][j],i)
                    pad = maxlbllen - len(lbl)
                    s += '|' + lbl + '|-' + '-'*pad
            s += '--\n'

        return s
    
    def write_qcircuit_tex(self,filename):
        """
        Renders this circuit as LaTeX (using Qcircuit).

        Returns
        -------
        str
        """
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

        TODO: more detail
        
        Parameters
        ----------
        gate_relations : dict
            Gate relations            
        
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
            assert((glabel_1,glabel_2) in list(gate_relations.keys())), "Gate relations provided are invalid! Does not contain the required relations for {} and {}".format(glabel_1,glabel_2)  
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

        self._tup_dirty = self._str_dirty = True
        return flag
    
    def shift_1q_gates_forward(self):
        """
        TODO: docstring (TIM)
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
        self._tup_dirty = self._str_dirty = True
        return flag
    
    def shift_2q_gates_forward(self):
        """
        TODO: docstring (TIM)
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

        self._tup_dirty = self._str_dirty = True
        return flag
    
    
    def delete_idle_layers(self):
        """
        TODO: docstring (TIM)
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
        """
        TODO: docstring (TIM)
        """        
    
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
        Compute the outcome probabilities of this Circuit using `gateset` as a
        model for the gates.

        Parameters
        ----------
        gateset : GateSet
            A description of the gate and SPAM operations corresponding to the
            labels stored in this Circuit.

        Returns
        -------
        probs : dictionary
            A dictionary with keys equal to the possible outcomes and values
            that are float probabilities.
        """
        return gateset.probs(self)

    
    def done_editing(self):
        """
        Make this Circuit read-only, so that it can be hashed (used as a
        dictionary key).  

        This is done automatically when attempting to hash a Circuit for the
        first time, so there's usually no need to call this function.

        Returns
        -------
        None
        """
        self._static = True

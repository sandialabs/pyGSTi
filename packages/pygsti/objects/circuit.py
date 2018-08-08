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
import sys as _sys

from . import gatestring as _gstr
from . import labeldicts as _ld
from ..baseobjs import Label as _Label
from ..tools import internalgates as _itgs
from ..tools import compattools as _compat


class Circuit(_gstr.GateString):
    """
    A Circuit is a more structured representation of a sequence of preparation,
    gate, and measurement operations.  It is composed of some number of "lines",
    typically one per qubit, and permits a richer set of operations than 
    :class:`GateString`.  These operations include a range of methods for
    manipulating quantum circuits. E.g., basic depth compression algorithms.
    """   
    def __init__(self, line_items=None, gatestring=None, num_lines=None,
                 line_labels=None, parallelize=False, identity='I'):
        """
        Creates a new Circuit object, encapsulating a quantum circuit.

        You can supply at most one of `line_items` and `gatestring` (alternate
        ways of specifying the gates).  If neither are specified an empty
        Circuit is created.  Unless `line_items` is specified, you must also
        specify the number of lines via `num_lines` and/or `line_labels`. If
        labels aren't specified they default to the integers starting at 0.

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
            this when it cannot be inferred, that is, when `line_items` or
            `line_labels` is not given.

        line_labels : list, optional
            A list of strings or integers labelling each of the lines.  The
            length of this list equals the number of lines in the circuit, 
            and if `line_labels` is not given these labels default to the 
            integers starting with 0.

        parallelize : bool, optional
            Only used when initializing from `gatestring`.  When True, automatic
            parallelization is performed: consecutive gates in `gatestring`
            acting on disjoint sets of qubits are placed in the same layer.

        identity : str, optional
            The "name" for the identity gate. This is the gate name that will
            be used to "pad" layers: qubits that are without a gate in
            a layer when the circuit is constructed will be assigned an idle gate
            with a name fixed by this string. Unlike all other gates, the created
            circuit "knows" that this particular gate is an idle. So, the circuit
            will potentially drop idle gates when doing depth-compression, whereas 
            all other gates are treat as having an unspecified action, by all methods
            of the circuit, unless the relationship between the gates is given to 
            a method of circuit.        
        """
        assert(_compat.isstr(identity)), "The identity name must be a string!"
        self.identity = identity

        assert((line_items is not None) or (gatestring is not None) or (num_lines is not None) or (line_labels is not None)), \
            "At least one of these arguments must be not None!"

        self._static = False

        if (num_lines is not None) and (line_labels is not None):
            assert(num_lines == len(line_labels)), "Inconsistent `num_lines` and `line_labels`!"
                
        if line_items is not None:
            assert(gatestring is None), "Cannot specify both `line_items` and `gatestring`"
            self.line_items = line_items
            if num_lines is not None:
                assert(num_lines == len(line_items)), "Inconsistent `line_items` and `num_lines` arguments!"
            if line_labels is not None:
                assert(len(line_labels) == len(line_items)), "Inconsistent `line_items` and `line_labels` arguments!"
                self.line_labels = list(line_labels)
            else:
                self.line_labels = list(range(len(line_items)))
                
        else:
            assert(num_lines is not None or line_labels is not None), \
                "`num_lines` or `line_labels`  must be specified whenever `line_items` is None!"
            if num_lines is not None:
                assert(isinstance(num_lines, _numbers.Integral)), "`num_lines` must be an integer!"
            if line_labels is not None:
                self.line_labels = list(line_labels)
                num_lines = len(list(line_labels))
            else:
                self.line_labels = list(range(num_lines))
            
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

        #future : implement this.
        #self.barriers = _np.zeros(self.depth()+1,bool)

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
        if self.number_of_lines() <= 0:
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
                if line[j].name != self.identity: #Note: it's OK to use .name on line items (all are *simple* labels)
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
   
    def number_of_lines(self):
        """
        The number of lines in the circuit
        """
        return len(self.line_labels)

    def copy(self):
        """
        Returns a copy of the circuit.
        """       
        return _copy.deepcopy(self)
                     
    def clear(self):
        """
        Removes all the gates in a circuit (preserving the number of lines).
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        self.line_items = []
        for i in range(0,self.number_of_lines()):
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

                    # Label *can* have None as its .qubits, this in interpeted
                    # to mean the label applies to all the lines.
                    gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
                    
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
                gate = gatestring[j]
                layer.append(gate)
                
            # Insert the layer into the circuit.
            self.insert_layer(layer,layer_number)
            
            # Move on to the next gate not included in the circuit.
            j += k
            # Update the layer number.
            layer_number += 1
    
    def insert_gate(self, gatelbl, j):
        """
        Inserts a gate into a circuit. If all the lines (e.g., qubits) that the gate
        acts on are idling -- as specified by circuit entries with names self.identity 
        -- then the gate replaces those idle gates. Otherwise, a new layer is inserted
        at depth j with this gate inserted into that layer (and all other lines will
        be idling).

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
        assert(not self._static),"Cannot edit a read-only circuit!"       
        # If all lines have idles on them in this layer, we put it into this layer, without moving
        # gate on other lines around. But otherwise we add in an idle layer at this point.
        gatelbl_qubits = gatelbl.qubits if (gatelbl.qubits is not None) else self.line_labels
        if not self.lines_are_idle_at_layer(gatelbl_qubits,j):    
            # Add an idle layer.
            for i in range(0,self.number_of_lines()):
                self.line_items[i].insert(j,_Label(self.identity,self.line_labels[i]))
            
        # Put the gate label in at the jth layer - note this label may
        # be a "parallel-gate" label and have mulitple components.
        for gl_comp in gatelbl.components:
            gate_qubits = gl_comp.qubits if (gl_comp.qubits is not None) \
                        else self.line_labels
            for i in gate_qubits:
                self.line_items[self.line_labels.index(i)][j] = gl_comp

        self._tup_dirty = self._str_dirty = True

    def replace_gate_with_circuit(self, circuit, q, j):
        """
        Replace a gate with a circuit. This gate is replaced with an idle and
        the circuit is inserted between this layer and the following circuit layer. 
        As such there is no restrictions on the lines on which this circuit can act non-trivially.
        `circuit` need not be a circuit over all the qubits in this circuit, but it must satisfying
        the requirements of the `insert_circuit()` method. 
       
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be inserted in place of the gate.
            
        q : int
            The qubit on which the gate is to be replaced.
            
        j : int
            The layer index (depth) of the gate to be replaced.
 
        Returns
        -------
        None
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        gate_to_replace = self.line_items[q][j]

        # Replace the gate with identity
        gate_qubits = self.line_labels if (gate_to_replace.qubits is None) \
                      else gate_to_replace.qubits
        for q in gate_qubits:
            self.line_items[self.line_labels.index(q)][j] = _Label(self.identity,q)
        
        # Inserts the circuit after the layer this gate was in.
        self.insert_circuit(circuit,j+1)

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
        assert(not self._static),"Cannot edit a read-only circuit!"        
        # Add an idle layer.
        for i in range(0,self.number_of_lines()):
            self.line_items[i].insert(j,_Label(self.identity,self.line_labels[i]))
            
        # Put the gates in.
        for i,line_label in enumerate(self.line_labels):
            for gatelbl in circuit_layer:
                # circuit layer can contain "parallel" gate layers, unlike
                # the values of self.line_items which are all simple labels
                for gl_comp in gatelbl.components:
                    gate_qubits = gl_comp.qubits if (gl_comp.qubits is not None) \
                              else self.line_labels
                    assert(set(gate_qubits).issubset(set(self.line_labels))), "Some of the of the elements in the layer to insert are on lines (qubits) that are not part of this circuit!"
                    if line_label in gate_qubits:
                        self.line_items[i][j] = gl_comp
                    
        self._tup_dirty = self._str_dirty = True

    def replace_layer_with_layer(self,circuit_layer,j):
        """
        Replace a layer with a layer. The input layer does not
        need to contain a gate that acts on every qubit. But,
        it should be a valid layer, meaning that the layer
        does not contain more than one gate on a qubit. Note that
        *all* gates in the previous layer are over-written, and
        lines without a gate in `circuit_layer` will be idling.
        
        Parameters
        ----------
        circuit_layer : List
            A list of Label objects, defining a valid layer.
            
        j : int
            The index (depth) of the layer to be replaced.
 
        Returns
        -------
        None
        """
        assert(not self._static),"Cannot edit a read-only circuit!"        
        self.delete_layer(j)
        self.insert_layer(circuit_layer,j)
        
    def replace_layer_with_circuit(self,circuit,j):
        """
        Replace a layer with a circuit. This circuit must satisfy the requirements
        of the `insert_circuit()` method. See that method for more details.
        
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
        assert(not self._static),"Cannot edit a read-only circuit!"        
        self.delete_layer(j)
        self.insert_circuit(circuit,j)
       
    def delete_layer(self,j):
        """
        Deletes a layer from the circuit.
        
        Parameters
        ----------
        j : int
            The index (depth) of the layer to be deleted.
 
        Returns
        -------
        None
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        for q in range(0,self.number_of_lines()):
            del self.line_items[q][j]

        self._tup_dirty = self._str_dirty = True
                    
    def insert_circuit(self,circuit,j):
        """
        Inserts a circuit into this circuit. The circuit to insert can be over more qubits than
        this circuit, as long as all qubits that are not part of this circuit are idling. In this
        case, the idling qubits are all discarded. The circuit to insert can also be on less qubits
        than this circuit: all other qubits are set to idling. So, the labels of the circuit to insert 
        for all non-idling qubits must be a subset of the labels of this circuit.

        Note the "idling" for the inserted circuit is decided according to the circuit.identity label,
        and that all auxillary properties of this circuit are unchanged (e.g., the self.identity
        over-rides the circuit.identity if they are different).
        
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
        assert(not self._static),"Cannot edit a read-only circuit!" 

        # Keeps track of lines that are not in the circuit to insert -- so we can pad them with identities
        lines_to_pad = self.line_labels[:] # copy this list
        for q in range(circuit.number_of_lines()):
            llabel = circuit.line_labels[q]
            # If there are lines in the circuit to insert that aren't in this circuit, we go in here.
            if llabel not in self.line_labels:
                # Fails only if the lines in the circuit to insert that aren't part of this circuit are not idling layers.
                assert(circuit.is_idling_qubit(llabel)), "There are non-idling lines in the circuit to insert that are *not* lines in this circuit!"
            # We insert lines from that circuit that are in this circuit.
            else:
                line_index = self.line_labels.index(llabel)
                self.line_items[line_index] = self.line_items[line_index][0:j] + circuit.line_items[q][:] + self.line_items[line_index][j:]
                # This lines doesn't need padding, so we delete it from the lines_to_pad list.
                del lines_to_pad[lines_to_pad.index(llabel)]

        # We now go through and pad with identities any lines in this circuit that did not have anything inserted on them.
        depth = circuit.depth()
        for llabel in lines_to_pad:
            line_index = self.line_labels.index(llabel)
            self.line_items[line_index] = self.line_items[line_index][0:j] + [_Label(self.identity,llabel) for i in range(depth)] + self.line_items[line_index][j:]
            
        self._tup_dirty = self._str_dirty = True
                            
    def append_circuit(self,circuit):
        """
        Append a circuit to the end of this circuit. This circuit must satisfy the requirements
        of the `insert_circuit()` method. See that method for more details.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be appended.
 
        Returns
        -------
        None
        """           
        self.insert_circuit(circuit,self.depth())
            
    def prefix_circuit(self,circuit):
        """
        Prefix a circuit to the end of this circuit. This circuit must satisfy the requirements
        of the `insert_circuit()` method. See that method for more details.
        
        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be prefixed.
 
        Returns
        -------
        None

        """  
        self.insert_circuit(circuit,0)

    def tensor_circuit(self, circuit, line_order=None):
        """
        Tensors a circuit to this circuit. It creates a circuit that consists of the current
        circuit in parallel with the circuit of `circuit`. The line_labels of `circuit` must
        be disjoint from the line_labels of this circuit, as otherwise applying the circuits
        in parallel does not make sense. Note that the `identity` label of this circuit must
        be the same as the `identity` label of `circuit`.

        Parameters
        ----------
        circuit : A Circuit object
            The circuit to be prefixed.

        line_order : List, optional
            A list of all the line labels specifying the order of the circuit in the updated
            circuit. If None, the lines of `circuit` are added below the lines of this circuit.
            Note that, for many purposes, the ordering of lines of the circuit is irrelevant.

        Returns
        -------
        None  
        """
        assert(self.identity == circuit.identity), "The identity labels must be the same!"
        for line_label in circuit.line_labels:
            assert(line_label not in self.line_labels), "The line labels of the circuit to tensor must be distinct from the line labels of this circuit!"

        if line_order is not None:
            assert(set(line_order).issubset(set(self.line_labels + circuit.line_labels))), "The line order `line_order`, if not None, must be a list containing all and only the line labels of the two circuits!"
            assert(set(self.line_labels + circuit.line_labels).issubset(set(line_order))), "The line order `line_order`, if not None, must be a list containing all and only the line labels of the two circuits!"
            new_line_labels = line_order
        else:
            new_line_labels = self.line_labels + circuit.line_labels

        # Make the circuits the same depth, by padding the end of whichever (if either) circuit is shorter.
        cdepth = circuit.depth()
        sdepth = self.depth()
        if cdepth > sdepth:
            for q in range(self.number_of_lines()):
                self.line_items[q] += [_Label(self.identity,q) for i in range(cdepth-sdepth)]
        elif cdepth < sdepth:
            for q in range(circuit.number_of_lines()):
                circuit.line_items[q] += [_Label(circuit.identity,q) for i in range(sdepth-cdepth)]
        
        self.insert_idling_wires(new_line_labels)

        for llabel in circuit.line_labels:
            lindex = self.line_labels.index(llabel)
            self.line_items[lindex] = _copy.deepcopy(circuit.get_line(llabel))
                       
    def replace_gatename(self, old_gatename, new_gatename):
        """
        Changes the *name* of a gate throughout this Circuit.

        Note that the "name" is only a part of the "label" identifying each
        gate, and doesn't include the lines (qubits) a gate acts upon.  For
        example, the "Gx:0" and "Gx:1" labels both have the same name but 
        act on different qubits.

        Parameters
        ----------
        old_gatename, new_gatename : string
            The gate name to find and the gate name to replace the found
            name with.

        Returns
        -------
        None
        """
        assert(not self._static),"Cannot edit a read-only circuit!"

        depth = self.depth()
        for q in range(self.number_of_lines()):
            for l in range(depth):
                gatelbl = self.line_items[q][l]
                if gatelbl.name == old_gatename:
                    self.line_items[q][l] = _Label(new_gatename, gatelbl.qubits)
                      #Note: OK if gatelbl.qubits is None

        self._tup_dirty = self._str_dirty = True

    def replace_identity(self, identity):
        """
        Changes the *name* of the idle/identity gate in the circuit. This replaces
        the name of the identity element in the circuit by setting self.identity = identity
        *and* by changing the names of all the gates that had the old self.identity name.

        Parameters
        ----------
        identity : string
            The new name for the identity gate.

        Returns
        -------
        None
        """
        self.replace_gatename(self.identity, identity)
        self.identity = identity

    def change_gate_library(self, compilation, allowed_filter=None, allow_unchanged_gates=False, depth_compression=True, 
                            oneQgate_relations=None, identity=None):
        """
        Re-express a circuit over a different gateset.
        
        Parameters
        ----------
        compilation : dict or CompilationLibrary.
            If a dictionary, the keys are some or all of the gates that appear in the circuit, and the values are 
            replacement circuits that are normally compilations for each of these gates (if they are not, the action 
            of the circuit will be changed). The circuits need not be on all of the qubits, and need only satisfy
            the requirements of the `insert_circuit` method. There must be a key for every gate except the self.identity
            gate, unless `allow_unchanged_gates` is False. In that case, gate that aren't a key in this dictionary are
            left unchanged.

            If a CompilationLibrary, this will be queried via the get_compilation_of() method to find compilations
            for all of the gates in the circuit. So this CompilationLibrary must contain or be able to auto-generate
            compilations for the requested gates, except when `allow_unchanged_gates` is True. In that case, gates
            that a compilation is not returned for are left unchanged.

        allowed_filter : dict or set, optional
            Specifies which gates are allowed to be used when generating compilations from `compilation`. Can only be
            not None if `compilation` is a CompilationLibrary. If a `dict`, keys must be gate names (like `"Gcnot"`) and
            values :class:`QubitGraph` objects indicating where that gate (if it's present in the library) may be used.
            If a `set`, then it specifies a set of qubits and any gate in the current library that is confined within that 
            set is allowed. If None, then all gates within the library are allowed.

        depth_compression : bool, optional
            Whether to perform depth compression after changing the gate library. If oneQgate_relations is None this
            will only remove idle layers and compress the circuit by moving everything as far forward as is possible
            without knowledge of the action of any gates other than self.identity. See the `depth_compression` method
            for more details. Under most circumstances this should be true; if it is False changing gate library will
            often result in a massive increase in circuit depth.
  
        oneQgate_relations : dict, optional
            Gate relations for the one-qubit gates in the new gate library, that are used in the  depth compression, to 
            cancel / combine gates. E.g., one key-value pair might be  ('Gh','Gh') : 'I', to signify that two Hadamards c
            ompose to the idle gate 'Gi'. See the depth_compression() method for more details.

        identity : str, optional
            A new name for the identity/idle gate. If not None, we set self.identity = identity and rename any gates
            that have the old identity name with the new identity name -- unless there is a compilation provided for
            gates with the old identity name; compilations for the old identity take precedence over this renaming
            method.
       
        Returns
        -------
        None
        """ 
        assert(not self._static),"Cannot edit a read-only circuit!"

        # If it's a CompilationLibrary, it has this attribute. When it's a CompilationLibrary we use the
        # .get_compilation_of method, which will look to see if a compilation for a gate is already available (with
        # `allowed_filter` taken account of) and if not it will attempt to construct it.
        if hasattr(compilation, 'templates'):
            # The function we query to find compilations
            def get_compilation(gate):
                # Use try, because it will fail if it cannot construct a compilation, and this is fine under some
                # circumstances
                try:
                    circuit = compilation.get_compilation_of(gate, allowed_filter=allowed_filter, verbosity=0)
                    return circuit
                except:
                    return None
        # Otherwise, we assume it's a dict.
        else:
            assert(allowed_filter is None), "`allowed_filter` can only been not None if the compilation is a CompilationLibrary!"
            # The function we query to find compilations
            def get_compilation(gate):
                return compilation.get(gate,None)
        
        d = self.depth()
        n = self.number_of_lines()
        for l in range(0,d):
            for q in range(0,n):
                gate = self.line_items[q][d-1-l]
                circuit = get_compilation(gate)
                # We don't check if it's the identity gate: we compile everything that there is a compilation for.
                if circuit is not None:
                    # Replace the gate with a circuit, using the wrap-around for `insert_circuit()`.
                    self.replace_gate_with_circuit(circuit,q,d-1-l)
                else:
                    # We never consider not having a compilation for the identity to be a failure.
                    if gate.name != self.identity and not allow_unchanged_gates:
                        raise ValueError("`compilation` does not contain, or cannot generate a compilation for {}!".format(gate))

        # If we are given a potentially new identity label, change to this. We do this after changing gate library, as
        # it's useful to be able to treat gates that have the *old* identity label differently: we don't fail if there is
        # no compilation given for them, but we do change the name if `identity` is specifed. Because `replace_gate_with_circuit`
        # can add idles at various points it's also important to sweep the entire circuit and remove any idles.
        if identity is not None:
            self.replace_identity(identity)

        # If specified, perform the depth compression. It is better to do this *after* the identity name has been changed.
        if depth_compression:            
            self.compress_depth(oneQgate_relations=oneQgate_relations, verbosity=0)

        self._tup_dirty = self._str_dirty = True
    
    def map_state_space_labels(self, mapper):
        """
        The labels of all of the lines (wires/qubits) are updated according to 
        the mapping function `mapper`.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing self.line_labels values
            and whose values are the new labels (ints or strings), or a function 
            which takes a single (existing label) argument and returns a new label.

        Returns
        -------
        None
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        # If the mapper is a dict, turn it into a function
        if type(mapper) == dict:
            def mapper_func(qlabel):
                return mapper[qlabel]
        else:
            mapper_func = mapper

        self.line_labels = [mapper_func(l) for l in self.line_labels]

        depth = self.depth()
        for i in range(0,self.number_of_lines()):
            for j in range(0,depth):
                gate = self.line_items[i][j]
                gate_qubits = [mapper_func(l) for l in gate.qubits] \
                              if (gate.qubits is not None) else None
                self.line_items[i][j] = _Label(gate.name,gate_qubits)

        self._tup_dirty = self._str_dirty = True

    def reorder_wires(self, order):
        """
        Reorders the lines (wires/qubits) of the circuit. Note that the ordering of the
        lines is not important for most purposes.

        Parameters
        ----------
        order : list
            A list containing all of the circuit line labels (self.line_labels) in the
            order that the should be converted to.

        Returns
        -------
        None
        """
        assert(set(order) == set(self.line_labels)), "The line labels must be the same!"
        old_line_items = _copy.deepcopy(self.line_items)
        self.line_items = []
        for i in range(0,self.number_of_lines()):
            self.line_items.append(old_line_items[self.line_labels.index(order[i])])
        # Also need to change the line_labels
        self.line_labels = order

    def delete_idling_wires(self):
        """
        Removes from the circuit all wires that are idling at every layer. These
        are the lines in the circuit whereby every gate has the name self.identity,
        which is by default 'Gi'.
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        # Find the idle lines
        allidle_list = []
        num_lines = self.number_of_lines()
        for q in range(num_lines):
            allidle = True
            for i in range(self.depth()):
                if self.line_items[q][i].name != self.identity:
                    allidle = False
                    break
            allidle_list.append(allidle)

        # Delete the idle lines
        for q in range(num_lines):
            if allidle_list[num_lines-1-q]:
                del self.line_items[num_lines-1-q]
                del self.line_labels[num_lines-1-q]

        self._tup_dirty = self._str_dirty = True

    def insert_idling_wires(self, all_line_labels):
        """
        Creates more lines (wires/qubits) in the circuit, with these new lines 
        consisting of idle gates (with the name self.identity)

        Parameters
        ----------
        all_line_labels : list
            A list containing all of the current circuit line labels (self.line_labels),
            along with further labels for all of the additional all-idle lines. The circuit
            has lines that are ordered according to the ordering of this list (but note that
            line ordering is irrevelant for most purposes).

        Returns
        -------
        None
        """
        assert(not self._static),"Cannot edit a read-only circuit!"

        old_line_items = _copy.deepcopy(self.line_items)
        old_line_labels = _copy.deepcopy(self.line_labels)
        depth = self.depth()

        self.line_labels = all_line_labels
        self.line_items = []

        for llabel in all_line_labels:
            if llabel in old_line_labels:
                self.line_items.append(old_line_items[old_line_labels.index(llabel)])
            else:
                self.line_items.append([_Label(self.identity,llabel) for i in range(depth)])
 
        self._tup_dirty = self._str_dirty = True        
         
    def reverse(self):
        """
        Reverses the order of the circuit.

        Returns
        -------
        None
        """
        assert(not self._static),"Cannot edit a read-only circuit!"         
        for q in range(0,self.number_of_lines()):
            self.line_items[q].reverse()
        self._tup_dirty = self._str_dirty = True

    def combine_oneQgates(self, oneQgate_relations, return_flag=False):
        """
        Compresses sequences of 1-qubit gates in the circuit, using the provided gate relations.
        One of the steps of the depth_compression() method, and in most cases that method will
        be more useful.
        
        Parameters
        ----------
        oneQgate_relations : dict
            Keys that are pairs of strings, corresponding to 1-qubit gate names, with values that are
            a single string, also corresponding to a 1-qubit gate name. Whenever a 1-qubit gate with 
            name `name1` is followed in the circuit by a 1-qubit gate with `name2` then, if 
            oneQgate_relations[name1,name2] = name3, name1 -> name3 and name2 -> self.identity, the
            identity name in the circuit. Moreover, this is still implemented when there are self.identity
            gates between these 1-qubit gates, and it is implemented iteratively in the sense that if there
            is a sequence of 1-qubit gates with names name1, name2, name3, ... and there are relations
            for all of (name1,name2) -> name12, (name12,name3) -> name123 etc then the entire sequence of
            1-qubit gates will be compressed into a single possibly non-idle 1-qubit gate followed by 
            idle gates in place of the previous 1-qubit gates.

            If a ProcessorSpec object has been created for the gates/device in question, the
            ProcessorSpec.oneQgate_relations is the appropriate (and auto-generated) `oneQgate_relations`.

            Note that this function will not compress sequences of 1-qubit gates that cannot be compressed by 
            independently inspecting sequential non-idle pairs (as would be the case with, for example, 
            Gxpi Gzpi Gxpi Gzpi, if the relation did not know that (Gxpi,Gzpi) -> Gypi, even though the sequence
            is the identity).
        
        return_flag : bool, optional
            If True, then a bool is returned. If False, None is returned.

        Returns
        -------
        bool or None
            If a bool, it is  False if the circuit is unchanged, and True otherwise.
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        # A flag that is turned to True if any non-trivial re-arranging is implemented by this method.
        compression_implemented = False        
        # Loop through all the qubits, to try and compress squences of 1-qubit gates on the qubit in question.
        for q in range(0,self.number_of_lines()):
            # j keeps track of the layer of the *next* gate that we are going to try and combine with later gates.
            j = 0
            while j < self.depth()-1:
                # This is the label of the current gate that we are trying to combine with later gates.
                k = j
                # Check that the gate is a 1-qubit gate, because this function can only combine pairs of 1-qubit gates.
                nqubits = self.line_items[q][k].number_of_qubits
                if nqubits is None: nqubits = len(self.line_labels)
                
                if nqubits == 1:
                    # Loop through the gates following this gate on qubit q.
                    for i in range(k+1,self.depth()):
                        # For each gate we try and combine we iterate j by 1: so we'll start the next loop at the gate
                        # we failed to combine with an earlier gate (unless we iterate j further)
                        j += 1
                        # If the next gate is not a 1-qubit gate, we leave the loop and try to combine the gate after it
                        # (the jth gate) with later gates. So we iterate j by 1 before leaving the loop
                        nqubits2 = self.line_items[q][i].number_of_qubits
                        if nqubits2 is None: nqubits2 = len(self.line_labels)
                        if nqubits2 > 1:
                            j += 1
                            break
                        # If the next gate is a 1-qubit gate, we see if we can compress it with gate k.
                        else:                              
                            # The names of the gates to try and combine
                            gl1 = self.line_items[q][k].name
                            gl2 = self.line_items[q][i].name
                            # If the later gate is the identity we skip and move onto the next gate, because the combination would 
                            # be trivial.
                            if gl2 != self.identity:
                                # Try to find a label they combine to; if the pair is not in the algebra dictionary we don't succeed.
                                try:  
                                    new_gl1 = oneQgate_relations[gl1,gl2]
                                    new_gl2 = self.identity
                                    # Write in the new gate names.
                                    self.line_items[q][k] = _Label(new_gl1,self.line_labels[q])
                                    self.line_items[q][i] = _Label(new_gl2,self.line_labels[q])
                                    # Record that a compression has been implemented : the circuit has been changed.
                                    compression_implemented = True
                                # If we can't combine the gates we quit the loop -- because we can't try and combine the gate with
                                # a gate that is past this gate. But we don't iterate j, because perhaps this 1-qubit gate can be
                                # combined with later 1-qubit gates.
                                except:
                                    break
                # If the gate is not a 1-qubit gate we move on to the gate at the next circuit layer: so we iterate j by 1.
                else:
                    j += 1
        # Only if we've changed anything do we need to set the "dirty" atributes to True.
        if compression_implemented:
            self._tup_dirty = self._str_dirty = True
        # If requested, returns the flag that tells us whether the algorithm achieved anything.
        if return_flag:
            return compression_implemented
    
    def shift_gates_forward(self, return_flag=False):
        """
        All gates are shifted forwarded as far as is possible without any knowledge of what 
        any of the gates are, except that the self.identity gates (idle gates) can be replaced.
        One of the steps of the depth_compression() method.
        
        Parameters
        ----------
        return_flag : bool, optional
            If True, then a bool is returned. If False, None is returned.

        Returns
        -------
        bool or None
            If a bool, it is  False if the circuit is unchanged, and True otherwise.
        """
        assert(not self._static),"Cannot edit a read-only circuit!"
        # Keeps track of whether any changes have been made to the circuit.
        compression_implemented = False
        # If the circuit is depth 0, we quit as there is nothing to do (and the code below 
        # fails in this case)
        if self.depth() == 0:
            if return_flag:
                return False
            else:
                return
        # Stores which layer we can move the current gate forwarded to.
        can_move_to_layer = _np.zeros(self.number_of_lines(),int) 
        # If the first layer isn't an idle, we set this to 1.
        for q in range(0,self.number_of_lines()):
            gate = self.line_items[q][0]
            if gate.name != self.identity:
                can_move_to_layer[q] = 1

        # Look at the gates in each circuit layer, and move them forward if we can
        for j in range(1,self.depth()):
            # Look at each line in turn
            for q in range(0,self.number_of_lines()):
                #print(j,q)
                gate = self.line_items[q][j]
                gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
                
                # If the gate isn't the identity, we try and move it forward. If it
                # is the identity, we don't change can_move_to_layer[q].
                if gate.name != self.identity:
                    # This stores which layer we can move the gate to. Starts at 0,
                    # and is increased as necessary
                    move_to_layer = 0
                    # Look at each qubit in the gate, and find out how far forward we
                    # can move a gate on that qubit
                    for qlabel in gate_qubits:
                        qindex = self.line_labels.index(qlabel)
                        # Update the layer we can move to to the worst-case out of the
                        # qubits the gate acts on -- as it has to be shifted forward on
                        # all of these qubits
                        move_to_layer = max(move_to_layer,can_move_to_layer[qindex])
                    
                    # If the layer we can move it to isn't the current layer we do that.
                    if move_to_layer < j:
                        # Go through the qubits the gate acts on and move it for all of them
                        for qlabel in gate_qubits:
                            qindex = self.line_labels.index(qlabel)
                            # Write the gate in where it is move to
                            self.line_items[qindex][move_to_layer] = self.line_items[qindex][j]
                            # Turn the old location of the gate into an identity on the qubit.
                            self.line_items[qindex][j] = _Label(self.identity,qlabel)
                            # The layer 1 after the layer we moved to is now the earliest available
                            # layer for a moving a gate to for that qubit, so we update this.
                            can_move_to_layer[qindex] = move_to_layer + 1
                            # We've changed the circuit, so record that in the bool.
                            compression_implemented = True
                    # If no compression can be implemented, this gate is now the "road-block" for the
                    # qubits it acts on, so update `can_move_to_layer` for these qubits to the layer
                    # *after* this. This might not be an identity, but if it isn't then this will be
                    # iterated on again so that `an_move_to_layer[qindex]` will always be the same layer
                    # as the gate we are considering trying to move forward unless this corresponds to
                    # an idle laye (it stops iterating foward as soon as we hit an idle layer on the qubit).
                    else:
                        for qlabel in gate_qubits:
                            qindex = self.line_labels.index(qlabel)
                            can_move_to_layer[qindex] = j + 1

        # Only if we've changed anything do we need to set the "dirty" atributes to True.
        if compression_implemented:
            self._tup_dirty = self._str_dirty = True
        # Only return the bool if requested
        if return_flag:     
            return compression_implemented
        else:
            return
    
    def delete_idle_layers(self, return_flag=False):
        """
        Deletes all layers in the circuit that consist of only idle layers. One of the steps of the
        depth_compression() method. 

        Parameters
        ----------
        return_flag : bool, optional
            If True, then a bool is returned. If False, None is returned.

        Returns
        -------
        bool or None
            If a bool, it is  False if the circuit is unchanged, and True otherwise.
        """
        assert(not self._static),"Cannot edit a read-only circuit!"        
        compression_implemented = False
        
        d = self.depth()
        for i in range(0,d):
            
            layer = self.get_layer(d-1-i)
            # If it's an idle layer, an empty list is returned.
            if len(layer) == 0:
                compression_implemented = True
                self.delete_layer(d-1-i)
        
        # Only if we've changed anything do we need to set the "dirty" atributes to True.
        if compression_implemented:
            self._tup_dirty = self._str_dirty = True
        # Only return the bool if requested
        if return_flag:     
            return compression_implemented
        else:
            return
    
    def compress_depth(self, oneQgate_relations=None, verbosity=0):
        """
        Compresses the depth of a circuit using very simple re-write rules. 

        1. If `oneQgate_relations` is provided, all sequences of 1-qubit gates in the  circuit 
           are compressed as far as is possible using only the pair-wise combination rules
           provided by this dict (see below).
        2. All gates are shifted forwarded as far as is possible without any knowledge of what 
           any of the gates are, except that the self.identity gates (idle gates) can be replaced.
        3. All idle-only layers are deleted.
        
        Parameters
        ----------
        oneQgate_relations : dict
            Keys that are pairs of strings, corresponding to 1-qubit gate names, with values that are
            a single string, also corresponding to a 1-qubit gate name. Whenever a 1-qubit gate with 
            name `name1` is followed in the circuit by a 1-qubit gate with `name2` then, if 
            oneQgate_relations[name1,name2] = name3, name1 -> name3 and name2 -> self.identity, the
            identity name in the circuit. Moreover, this is still implemented when there are self.identity
            gates between these 1-qubit gates, and it is implemented iteratively in the sense that if there
            is a sequence of 1-qubit gates with names name1, name2, name3, ... and there are relations
            for all of (name1,name2) -> name12, (name12,name3) -> name123 etc then the entire sequence of
            1-qubit gates will be compressed into a single possibly non-idle 1-qubit gate followed by 
            idle gates in place of the previous 1-qubit gates.

            If a ProcessorSpec object has been created for the gates/device in question, the
            ProcessorSpec.oneQgate_relations is the appropriate (and auto-generated) `oneQgate_relations`.

            Note that this function will not compress sequences of 1-qubit gates that cannot be compressed by 
            independently inspecting sequential non-idle pairs (as would be the case with, for example, 
            Gxpi Gzpi Gxpi Gzpi, if the relation did not know that (Gxpi,Gzpi) -> Gypi, even though the sequence
            is the identity).

        verbosity : int, optional
            If > 0, information about the depth compression is printed to screen.
        
        Returns
        -------
        None
        """ 
        assert(not self._static),"Cannot edit a read-only circuit!"       
    
        if verbosity > 0:
            print("- Implementing circuit depth compression")
            print("  - Circuit depth before compression is {}".format(self.depth()))
               
        #try:
        flag1 = False
        if oneQgate_relations is not None:                            
            flag1 = self.combine_oneQgates(oneQgate_relations,return_flag=True)
        flag2 = self.shift_gates_forward(return_flag=True)   
        flag3 = self.delete_idle_layers(return_flag=True)
        #except:
        #    print(self.number_of_lines, len(self.line_labels), self.line_labels, len(self.line_items),len(self.line_items[0]))
        #    print(self.line_items)
        #    assert(False)

        if verbosity > 0:
            if not (flag1 or flag2 or flag3):
                print("  - Circuit unchanged by depth compression algorithm")       
            print("  - Circuit depth after compression is {}".format(self.depth()))  
       
    def get_layer(self,j):
        """
        Returns the layer, as a list, at depth j. This list contains all gates
        in the layer except self.identity gates, and contains each gate only
        once (although multi-qubit gates appear on multiple lines of the circuit).

        Parameters
        ----------
        j : int
            The index (depth) of the layer to be returned

        Returns
        -------
        List of Labels
            Each gate in the layer, except self.identity gates, once and only once.
        """      
        assert(j >= 0 and j < self.depth()), "Circuit layer label invalid! Circuit is only of depth {}".format(self.depth())
        
        layer = []
        qubits_used = []
        for i in range(0,self.number_of_lines()):
            
            gate = self.line_items[i][j]
            gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
            
            # Checks every element is a Label object.
            assert((isinstance(gate,_Label))), "The elements of the layer should be Label objects!"
            # Checks that a Label appears in all the lines it should act on.
            for q in gate_qubits:
                assert(self.line_items[self.line_labels.index(q)][j] == gate), "This is an invalid circuit layer!"

            # We only record non-identity gates.
            if gate not in layer and gate.name != self.identity:
                # Checks that we have not already assigned a gate to this qubit
                assert(not set(gate_qubits).issubset(set(qubits_used))), "There is more than one gate on some qubits in the layer; layer invalid!"
                qubits_used.extend( gate_qubits )
                layer.append(gate)
            
        return layer

    def get_layer_with_idles(self,j):
        """
        Returns the layer, as a list, at depth j. This list contains all gates
        in the layer *including* self.identity gates, and contains each gate only
        once (although multi-qubit gates appear on multiple lines of the circuit).
        To get a layer without the self.identity gates, use the `get_layer()` method.

        Parameters
        ----------
        j : int
            The index (depth) of the layer to be returned

        Returns
        -------
        List of Labels
            Each gate in the layer, except self.identity gates, once and only once.
        """      
        assert(j >= 0 and j < self.depth()), "Circuit layer label invalid! Circuit is only of depth {}".format(self.depth())
        
        layer = []
        qubits_used = []
        for i in range(0,self.number_of_lines()):
            
            gate = self.line_items[i][j]
            gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
            
            # Checks every element is a Label object.
            assert((isinstance(gate,_Label))), "The elements of the layer should be Label objects!"
            # Checks that a Label appears in all the lines it should act on.
            for q in gate_qubits:
                assert(self.line_items[self.line_labels.index(q)][j] == gate), "This is an invalid circuit layer!"

            if gate not in layer:
                # Checks that we have not already assigned a gate to this qubit
                assert(not set(gate_qubits).issubset(set(qubits_used))), "There is more than one gate on some qubits in the layer; layer invalid!"
                qubits_used.extend( gate_qubits )
                layer.append(gate)
            
        return layer

    def is_valid_circuit(self):
        """
        Checks whether the circuit satisfies all of the criteria to be a valid circuit. These are:

        1. Every element in the circuit is a Label object.
        2. A label object is contained on all and only those lines that it acts on, as specified by
           the label.qubits tuple. 

        This function returns None, but it fails if the circuit is invalid.

        Note that a circuit object does not know what any of the gate names refer to, so, e.g.,
        if a CNOT gate has been included that acts on 3 qubits this test cannot check that.
        """

        depth = self.depth()
        for l in range(depth):
            # This checks that it's a valid layer, so that does the trick.
            layer = self.get_layer(l)
         
    def get_line(self,line_label):
        """
        Returns the line (wire) with the label `line_labels`

        Parameters
        ----------
        line_label : int or str
            The label of the line to return.

        Returns
        -------
        List of Labels
        """      
        layer_index = self.line_labels.index(line_label)        
        line = self.line_items[layer_index]            
        return line
    
    def is_idling_qubit(self,line_label):
        """
        Wether the "wire" in question is idling in *every* cirucuit layer.
        Idling is specified by gate with the name self.identity. All other
        gates are not considered to be an idle.
        
        Parameters
        ----------
        line_label : list
            The label of the line (i.e., "wire" or qubit).

        Returns
        -------
        Bool
            True if the line is idling. False otherwise.
        """
        q = self.line_labels.index(line_label)
        for i in range(self.depth()):
            if self.line_items[q][i].name != self.identity:
                 return False
        
        return True

    def lines_are_idle_at_layer(self, line_labels_subset, j):
        """
        Wether all the wires at a particular layer (depth) are
        all idling, with "idling" specified by the self.identity
        string.
        
        Parameters
        ----------
        line_labels_subset : list
            A list of line labels. Should consist of some or all
            of the element of self.line_labels.
            
        j : int
            The layer index (depth) at which to check to see
            if all the lines are idling.

        Returns
        -------
        Bool
            True if all the lines in question have an idle gate
            at this layer. False otherwise.
        """
        for llabel in line_labels_subset:
            if self.line_items[self.line_labels.index(llabel)][j].name != self.identity:
                return False
        return True

    def depth(self):
        """
        The circuit depth. This is the number of layers in the circuit.
        
        Returns
        -------
        int
        """ 
        return len(self.line_items[0])
    
    def size(self):
        """
        Returns the circuit size, which is the sum of the sizes of all the
        gates in the circuit. A gate that acts on n-qubits has a size of n, 
        with the exception of the special idle with the name self.identity,
        which has a size of 0. Hence, the circuit size = (circuit depth) X 
        (the number of lines) - (the number of self.identity gates in the
        circuit).
        
        Returns
        -------
        int        
        """
        size = 0
        for q in range(0,self.number_of_lines()):
            for j in range(0,self.depth()):
                if self.line_items[q][j].name != self.identity:
                    size += 1
        return size
    
    def twoQgate_count(self):
        """
        The number of two-qubit gates in the circuit. (Note that this cannot
        distinguish between "true" 2-qubit gates and gate that have been defined
        to act on two qubits but that represent some tensor-product gate.)
        
        Returns
        -------
        int
        """           
        count = 0
        for q in range(0,self.number_of_lines()):
            for j in range(0,self.depth()):
                nqubits = self.line_items[q][j].number_of_qubits
                if nqubits is None: nqubits = len(self.line_labels)
                if nqubits == 2:
                    count += 1
        return count//2

    def multiQgate_count(self):
        """
        The number of multi-qubit gates in the circuit. (Note that this cannot
        distinguish between "true" multi-qubit gates and gate that have been defined
        to act on more than one qubit but that represent some tensor-product gate.)
        
        Returns
        -------
        int
        """           
        count = 0
        for q in range(0,self.number_of_lines()):
            for j in range(0,self.depth()):
                gatelbl = self.line_items[q][j]
                if gatelbl.number_of_qubits is None:
                    if q == 0 and self.number_of_lines() > 1:
                        count += 1
                elif gatelbl.number_of_qubits >= 2:
                    if gatelbl.qubits[0] == self.line_labels[q]:
                        count += 1
        return count
    
    def predicted_error_probability(self, gate_error_probabilities):
        """
        Predicts the probability that one or more errors occur in the circuit
        if the gates have the error probabilities specified by in the input
        dictionary. Given correct error rates for the gates and stochastic errors, 
        this is predictive of the probability of an error in the circuit. But note
        that that is generally *not* the same as the probability that the circuit 
        implemented is incorrect (e.g., stochastic errors can cancel).
        
        Parameters
        ----------
        gate_error_probabilities : dict
            A dictionary where the keys are the labels that appear in the circuit, and
            the value is the error probability for that gate.
 
        Returns
        -------
        float
            The probability that there is one or more errors in the circuit.
        """
        f = 1.
        depth = self.depth()
        for i in range(0,self.number_of_lines()):
            for j in range(0,depth):
                gatelbl = self.line_items[i][j]
                
                # So that we don't include multi-qubit gates more than once.
                if gatelbl.qubits is None:
                    if i == 0: 
                        f = f*(1-gate_error_probabilities[gatelbl])
                elif gatelbl.qubits[0] == self.line_labels[i]:
                    f = f*(1-gate_error_probabilities[gatelbl])       
        return 1 - f

    def __str__(self):
        """
        A text rendering of the circuit.
        """
        s = ''
        Ctxt = 'C' if _sys.version_info <= (3, 0) else '\u25CF' # No unicode in
        Ttxt = 'T' if _sys.version_info <= (3, 0) else '\u2295' #  Python 2

        def abbrev(lbl,k): #assumes a simple label w/ name & qubits
            """ Returns what to print on line 'k' for label 'lbl' """
            lbl_qubits = lbl.qubits if (lbl.qubits is not None) else self.line_labels
            nqubits = len(lbl_qubits)
            if nqubits == 1:
                return lbl.name
            elif lbl.name in ('CNOT','Gcnot') and nqubits == 2: # qubit indices = (control,target)
                if k == self.line_labels.index(lbl_qubits[0]):
                    return Ctxt + str(lbl_qubits[1])
                else:
                    return Ttxt + str(lbl_qubits[0])
            elif lbl.name in ('CPHASE', 'Gcphase') and nqubits == 2:
                if k == self.line_labels.index(lbl_qubits[0]):
                    otherqubit = lbl_qubits[1]
                else:
                    otherqubit = lbl_qubits[0]
                return Ctxt + str(otherqubit)
            else:
                return str(lbl)
        
        max_labellen = [ max([ len(abbrev(self.line_items[i][j],i))
                               for i in range(0,self.number_of_lines())])
                         for j in range(0,self.depth()) ]

        max_linelabellen = max([len(str(llabel)) for llabel in self.line_labels])

        for i in range(self.number_of_lines()):
            s += 'Qubit {} '.format(self.line_labels[i]) + ' '*(max_linelabellen - len(str(self.line_labels[i]))) + '---'
            for j,maxlbllen in enumerate(max_labellen):
                if self.line_items[i][j].name == self.identity:
                    # Replace with special idle print at some point
                    #s += '-'*(maxlbllen+3) # 1 for each pipe, 1 for joining dash
                    s += '|'+' '*(maxlbllen) + '|-' 
                else:
                    lbl = abbrev(self.line_items[i][j],i)
                    pad = maxlbllen - len(lbl)
                    s += '|' + ' '*int(_np.floor(pad/2)) + lbl + ' '*int(_np.ceil(pad/2)) + '|-' #+ '-'*pad
            s += '--\n'

        return s
    
    def write_Qcircuit_tex(self, filename):
        """
        Writes this circuit into a file, containing LaTex that will diplay this circuit using the 
        Qcircuit.tex LaTex import (running the LaTex requires the Qcircuit.tex file).
        
        Parameters
        ----------
        filename : str
            The file to write the LaTex into. Should end with '.tex'

        Returns
        -------
        None
        """
        n = self.number_of_lines()
        d = self.depth()
        
        f = open(filename,'w') 
        f.write("\documentclass{article}\n")
        f.write("\\usepackage{mathtools}\n")
        f.write("\\usepackage{xcolor}\n")
        f.write("\\usepackage[paperwidth="+str(5.+d*.3)+"in, paperheight="+str(2+n*0.2)+"in,margin=0.5in]{geometry}")
        f.write("\input{Qcircuit}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{equation*}\n") 
        f.write("\Qcircuit @C=1.0em @R=0.5em {\n")
        
        for q in range(0,n):
            qstring = '&'
            # The quantum wire for qubit q
            circuit_for_q = self.line_items[q]
            for gate in circuit_for_q:
                gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
                nqubits = len(gate_qubits)
                if gate.name == self.identity:
                    qstring += ' \qw &'
                elif gate.name in ('CNOT','Gcnot') and nqubits == 2:
                    if gate_qubits[0] == q:
                        qstring += ' \ctrl{'+str(gate_qubits[1]-q)+'} &'
                    else:
                        qstring += ' \\targ &'
                elif gate.name in ('CPHASE','Gcphase') and nqubits == 2:
                    if gate_qubits[0] == q:
                        qstring += ' \ctrl{'+str(gate_qubits[1]-q)+'} &'
                    else:
                        qstring += ' \control \qw &'
            
                else:
                    qstring += ' \gate{'+str(gate.name)+'} &'
                    
            qstring += ' \qw & \\'+'\\ \n'
            f.write(qstring)
        
        f.write("}\end{equation*}\n")
        f.write("\end{document}")
        f.close()    

    def convert_to_quil(self, gatename_conversion=None, qubit_conversion=None):
        """
        Converts a circuit to a quil string.

        Parameters
        ----------
        gatename_conversion : dict, optional
            If not None, a dictionary that converts the gatenames in the circuit to the
            gatenames that will appear in the quil output. If only standard pyGSTi names 
            are used (e.g., 'Gh', 'Gp', 'Gcnot', 'Gcphase', etc) this dictionary need not 
            be specified, and an automatic conversion to the standard quil names will be
            implemented.

            * Currently some standard pyGSTi names do not have an inbuilt conversion to quil names. 
            This will be fixed in the future *

        qubit_conversion : dict, optional
            If not None, a dictionary converting the qubit labels in the circuit to the 
            desired qubit labels in the quil output. Can be left as None if the qubit
            labels are either (1) integers, or (2) of the form 'Qi' for integer i. In
            this case they are converted to integers (i.e., for (1) the mapping is trivial,
            for (2) the mapping strips the 'Q'). 

        Returns
        -------
        str
            A quil string.
        """
        # create standard conversations.
        if gatename_conversion is None:
            gatename_conversion = _itgs.get_standard_gatenames_quil_conversions()
        if qubit_conversion is None:
            # To tell us whether we have found a standard qubit labelling type.
            standardtype = False
            # Must first check they are strings, because cannot query q[0] for int q.
            if all([isinstance(q,str) for q in self.line_labels]):
                if all([q[0] == 'Q' for q in self.line_labels]):
                    standardtype = True
                    qubit_conversion = {llabel : int(llabel[1:]) for llabel in self.line_labels}
            if all([isinstance(q,int) for q in self.line_labels]):
                   qubit_conversion = {q : q for q in self.line_labels}
                   standardtype = True
            if not standardtype:
                raise ValueError("No standard qubit labelling conversion is available! Please provide `qubit_conversion`.")
         
        # Init the quil string.
        quil = ''
        depth = self.depth()
        
        # Go through the layers, and add the quil for each layer in turn.
        for l in range(depth):
            
            # Get the layer, without identity gates and containing each gate only once.
            layer = self.get_layer(l)
            # For keeping track of which qubits have a gate on them in the layer.
            qubits_used = []
            
            # Go through the (non-self.identity) gates in the layer and convert them to quil
            for gate in layer:
                gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
                assert(len(gate_qubits) <= 2 or gate.qubits is None), 'Gate on more than 2 qubits given; this is currently not supported!'
                
                
                # Find the quil for the gate.
                quil_for_gate = gatename_conversion[gate.name]
                
                #If gate.qubits is None, gate is assumed to be single-qubit gate
                #acting in parallel on all qubits.
                if gate.qubits is None:
                    if quil_for_gate == 'I':
                        quil += 'PRAGMA PRESERVE_BLOCK\n'
                        for q in gate_qubits:
                            quil += quil_for_gate + ' ' + str(qubit_conversion[q]) + '\n'
                        quil += 'PRAGMA END_PRESERVE_BLOCK\n'
                    else:
                        for q in gate_qubits:
                            quil += quil_for_gate + ' ' + str(qubit_conversion[q]) + '\n'
                                        
                #If gate.qubits is not None, then apply the one- or multi-qubit gate to
                #the explicitly specified qubits.
                else:                
                    for q in gate_qubits: quil_for_gate += ' ' + str(qubit_conversion[q]) 
                    quil_for_gate += '\n'
                    # Add the quil for the gate to the quil string.
                    quil += quil_for_gate
                
                # Keeps track of the qubits that have been accounted for, and checks that hadn't been used
                # although that should already be checked in the .get_layer(), which checks for its a valid 
                # circuit layer.
                assert(not set(gate_qubits).issubset(set(qubits_used)))
                qubits_used.extend(gate_qubits)
            
            # All gates that don't have a non-idle gate acting on them get an idle in the layer.
            for q in self.line_labels:
                if q not in qubits_used:
                    quil += 'I' + ' ' + str(qubit_conversion[q]) +'\n'
                    
            # Add in a barrier after every circuit layer. Future: Should make this optional at some
            # point and/or to agree with the "barriers" in the circuit (to be added).
            quil += 'PRAGMA PRESERVE_BLOCK\nPRAGMA END_PRESERVE_BLOCK\n'
        
        # Add in a measurement at the end.
        for q in self.line_labels:
            quil += "MEASURE {0} [{1}]\n".format(str(qubit_conversion[q]),str(qubit_conversion[q]))
            
        return quil  

    def convert_to_openqasm(self, gatename_conversion=None, qubit_conversion=None):
        """
        Converts a circuit to an openqasm string.

        Parameters
        ----------
        gatename_conversion : dict, optional
            If not None, a dictionary that converts the gatenames in the circuit to the
            gatenames that will appear in the openqasm output. If only standard pyGSTi names 
            are used (e.g., 'Gh', 'Gp', 'Gcnot', 'Gcphase', etc) this dictionary need not 
            be specified, and an automatic conversion to the standard openqasm names will be
            implemented.

        qubit_conversion : dict, optional
            If not None, a dictionary converting the qubit labels in the circuit to the 
            desired qubit labels in the openqasm output. Can be left as None if the qubit
            labels are either (1) integers, or (2) of the form 'Qi' for integer i. In
            this case they are converted to integers (i.e., for (1) the mapping is trivial,
            for (2) the mapping strips the 'Q'). 

        Returns
        -------
        str
            An openqasm string.
        """
        # create standard conversations.
        if gatename_conversion is None:
            gatename_conversion = _itgs.get_standard_gatenames_openqasm_conversions()
        if qubit_conversion is None:
            # To tell us whether we have found a standard qubit labelling type.
            standardtype = False
            # Must first check they are strings, because cannot query q[0] for int q.
            if all([isinstance(q,str) for q in self.line_labels]):
                if all([q[0] == 'Q' for q in self.line_labels]):
                    standardtype = True
                    qubit_conversion = {llabel : int(llabel[1:]) for llabel in self.line_labels}
            if all([isinstance(q,int) for q in self.line_labels]):
                   qubit_conversion = {q : q for q in self.line_labels}
                   standardtype = True
            if not standardtype:
                raise ValueError("No standard qubit labelling conversion is available! Please provide `qubit_conversion`.")
        
        num_qubits = len(self.line_labels)
        
        # Init the openqasm string.
        openqasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n\n'
        
        openqasm += 'qreg q[{0}];\n'.format(str(num_qubits))
        openqasm += 'creg cr[{0}];\n'.format(str(num_qubits))
        openqasm += '\n'
        
        depth = self.depth()
        
        # Go through the layers, and add the openqasm for each layer in turn.
        for l in range(depth):
            
            # Get the layer, without identity gates and containing each gate only once.
            layer = self.get_layer(l)
            # For keeping track of which qubits have a gate on them in the layer.
            qubits_used = []
            
            # Go through the (non-self.identity) gates in the layer and convert them to openqasm
            for gate in layer:
                gate_qubits = gate.qubits if (gate.qubits is not None) else self.line_labels
                assert(len(gate_qubits) <= 2), 'Gates on more than 2 qubits given; this is currently not supported!'
                
                # Find the openqasm for the gate.
                openqasm_for_gate = gatename_conversion[gate.name]
                for q in gate_qubits: 
                    openqasm_for_gate += ' q[' + str(qubit_conversion[q])+']'
                    if q != gate_qubits[-1]:
                        openqasm_for_gate += ', '
                openqasm_for_gate += ';\n'
                # Add the openqasm for the gate to the openqasm string.
                openqasm += openqasm_for_gate
                
                # Keeps track of the qubits that have been accounted for, and checks that hadn't been used
                # although that should already be checked in the .get_layer(), which checks for its a valid 
                # circuit layer.
                assert(not set(gate_qubits).issubset(set(qubits_used)))
                qubits_used.extend(gate_qubits)
            
            # All gates that don't have a non-idle gate acting on them get an idle in the layer.
            for q in self.line_labels:
                if q not in qubits_used:
                    openqasm += 'id' + ' q[' + str(qubit_conversion[q]) +'];\n'
                    
            # Add in a barrier after every circuit layer. Future: Should make this optional at some
            # point and/or to agree with the "barriers" in the circuit (to be added).
            openqasm += 'barrier '
            for q in self.line_labels[:-1]:
                openqasm += 'q[{0}], '.format(str(qubit_conversion[q]))
            openqasm += 'q[{0}];\n'.format(str(qubit_conversion[self.line_labels[-1]]))
#            openqasm += ';'
        
        # Add in a measurement at the end.
        for q in self.line_labels:
            openqasm += "measure q[{0}] -> cr[{1}];\n".format(str(qubit_conversion[q]),str(qubit_conversion[q]))
            
        return openqasm

    def simulate(self, gateset, return_all_outcomes=False): 
        """
        Compute the outcome probabilities of this Circuit using `gateset` as a
        model for the gates. The order of the outcome strings (e.g., '0100') is
        w.r.t to the ordering of the qubits in the circuit. That is, the ith element
        of the outcome string corresponds to the qubit with label self.qubit_labels[i].

        Parameters
        ----------
        gateset : GateSet
            A description of the gate and SPAM operations corresponding to the
            labels stored in this Circuit. If this gateset is over more qubits 
            than the circuit, the output will be the probabilities for the qubits 
            in the circuit marginalized over the other qubits. But, the simulation
            is over the full set of qubits in the gateset, and so the time taken for
            the simulation scales with the number of qubits in the gateset. For
            models whereby "spectator" qubits do not affect the qubits in this
            circuit (such as with perfect gates), more efficient simulations will
            be obtained by first creating a gateset only over the qubits in this 
            circuit.

        return_all_outcomes: bool, optional
            Whether to include outcomes in the returned dictionary that have zero
            probability. When False, the threshold for discarding an outcome as z
            ero probability is 10^-12.

        Returns
        -------
        probs : dictionary
            A dictionary with keys equal to all (`return_all_outcomes` is True) or 
            possibly only some (`return_all_outcomes` is False) of the possible 
            outcomes, and values that are float probabilities.
        """
        # These results is a dict with strings of outcomes (normally bits) ordered according to the
        # state space ordering in the gateset.
        results = gateset.probs(self)

        # Mapping from the state-space labels of the gateset to their indices.
        # (e.g. if gateset.stateSpaceLabels is [('Qa','Qb')] then sslInds['Qb'] = 1
        # (and 'Qb' may be a circuit line label)
        sslInds = { sslbl:i for i,sslbl in enumerate(gateset.stateSpaceLabels.labels[0]) }
          # Note: we ignore all but the first tensor product block of the state space.
        
        def process_outcome(outcome):
            """Relabels an outcome tuple and drops state space labels not in the circuit."""
            processed_outcome = []
            for lbl in outcome: # lbl is a string - an instrument element or POVM effect label, e.g. '010'
                relbl = ''.join([ lbl[sslInds[ll]] for ll in self.line_labels ])
                processed_outcome.append(relbl)
                #Note: above code *assumes* that each state-space label (and so circuit line label)
                # corresponds to a *single* letter of the instrument/POVM label `lbl`.  This is almost
                # always the case, as state space labels are usually qubits and so effect labels are
                # composed of '0's and '1's.
            return tuple(processed_outcome)
        
        processed_results = _ld.OutcomeLabelDict()
        for outcome,pr in results.items():
            if return_all_outcomes or pr > 1e-12: # then process & accumulate pr
                p_outcome = process_outcome(outcome) # rearranges and drops parts of `outcome`
                if p_outcome in processed_results: # (may occur b/c processing can map many-to-one)
                    processed_results[p_outcome] += pr # adding marginalizes the results.
                else:
                    processed_results[p_outcome] = pr

        return processed_results
    
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

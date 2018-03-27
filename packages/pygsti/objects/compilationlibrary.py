""" Defines CompilationLibrary class and supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import copy as _copy
import collections as _collections
import itertools as _itertools
from scipy.sparse.csgraph import floyd_warshall as _fw

from ..tools import symplectic as _symp
from .label import Label as _Label
from .gate import EmbeddedCliffordGate as _EmbeddedCliffordGate
from .circuit import Circuit as _Circuit


def sorted_partitions(n):
    """ Interator over all sorted (decreasing) partitions of integer n """

    p = _np.zeros(n,'i')
    k = 0    # Index of last element in a partition
    p[k] = n # Initialize first partition as number itself
 
    # This loop first yields current partition, then generates next
    # partition. The loop stops when the current partition has all 1s
    while True:
        yield p[0:k+1]
 
        # Find the rightmost non-one value in p[]. Also, update the
        # rem_val so that we know how much value can be accommodated
        rem_val = 0;
        while k >= 0 and p[k] == 1:
            rem_val += p[k]
            k -= 1
 
        # if k < 0, all the values are 1 so there are no more partitions
        if k < 0: return
 
        # Decrease the p[k] found above and adjust the rem_val
        p[k] -= 1
        rem_val += 1
 
        # If rem_val is more, then the sorted order is violated.  Divide
        # rem_val in different values of size p[k] and copy these values at
        # different positions after p[k]
        while rem_val > p[k]:
            p[k+1] = p[k]
            rem_val -= p[k]
            k += 1
 
        # Copy rem_val to next position and increment position
        p[k+1] = rem_val
        k += 1

def partitions(n):
    """ Interator over all partitions of integer n """
    for p in sorted_partitions(n):
        previous = tuple()
        for pp in _itertools.permutations(p):
            if pp > previous: # only *unique* permutations
                previous = pp # (relies in itertools implementatin detail that
                yield pp      # any permutations of a sorted iterable are in
                # sorted order unless they are duplicates of prior permutations

class CompilationError(Exception):
    pass
                
class CompilationLibrary(_collections.OrderedDict):
    
    def __init__(self, clifford_gateset, ctyp="absolute", items=[]):
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
        self.gateset = clifford_gateset # gateset of (all Clifford) gates to compile requested gates into
        self.ctype = ctyp
        self.templates = _collections.defaultdict(list)
        self.connectivity = {} # (connectivity, distance, shortestpath) of
                               # gates currently compiled in library (key=gate_name)                        
        super(CompilationLibrary,self).__init__(items)

    def get_symplectic_reps(self, gatelabel_filter=None):
        """TODO: docstring """
        if gatelabel_filter is None:
            gatelabels = list(self.gateset.gates.keys())
        else:
            gatelabels = gatelabel_filter
              # could check that all these are in self.gateset.gates.keys (FUTURE?)

        srep_dict = {}
        for gl in gatelabels:
            assert(isinstance(self.gateset.gates[gl], _EmbeddedCliffordGate))
            srep = (self.gateset.gates[gl].embedded_gate.smatrix,
                    self.gateset.gates[gl].embedded_gate.svector)
            if gl.name in srep_dict:
                assert(srep == srep_dict[gl.name]), "Inconsistent symplectic reps for %s root!" % gl.name
            else:
                srep_dict[gl.name] = srep

        return srep_dict


    def add_local_compilation_of(self, gatelabel, unitary=None, srep=None, max_iterations=10, force=False, verbosity=1):

        # Template compilations always use integer qubit labels: 0 to N
        #  where N is the number of qubits in the template's overall label
        #  (i.e. its key in self.templates)

        if not force and gatelabel in self:
            return #don't re-compute unless we're told to

        def to_real_label(template_label):
            """ Convert a "template" gate label (which uses integer qubit labels
                0 to N) to a "real" label for a potential gate in self.gateset. """
            qlabels = [gatelabel.qubits[i] for i in template_label.qubits]
            return _Label(template_label.name, qlabels)

        def to_template_label(real_label):
            """ The reverse (qubits in template == gatelabel.qubits) """
            qlabels = [gatelabel.qubits.index(lbl) for lbl in real_label.qubits]
            return _Label(real_label.name, qlabels)

        def is_local_compilation_feasible(template_labels):
            """ Whether template_labels can possibly be enough
                gates to compile a template for gateLabel with """
            if gatelabel.number_of_qubits <= 1:
                return len(template_labels) > 0 #1Q gates, anything is ok
            elif gatelabel.number_of_qubits == 2:
                # 2Q gates need a compilation gate that is also 2Q (can't do with just 1Q gates!)
                return max([lbl.number_of_qubits for lbl in template_labels]) == 2
            else:
                # >2Q gates need to make sure there's some connected path
                return True # TODO LATER, using graphs stuff?
                                          
        template_to_use = None
        
        for template_compilation in self.templates.get(gatelabel.name,[]):
            #Check availability of gates in self.gateset to determine
            # whether template_compilation can be applied.
            if all([(gl in self.gateset.gates) for gl in map(to_real_label,
                                                             template_compilation) ]):
                template_to_use = template_compilation
                if verbosity > 0: print("Existing template found!")
                break  # compilation found!

        else: # no existing templates can be applied, so make a new one

            #construct a list of the available gates on the qubits of
            # `gatelabel` (or a subset of them)
            available_glabels = list( filter(lambda gl: set(gl.qubits).issubset(gatelabel.qubits),
                                             self.gateset.gates.keys()) )
            available_template_labels = set(map(to_template_label, available_glabels))
            available_srep_dict = self.get_symplectic_reps(available_glabels)
                
            if is_local_compilation_feasible(available_template_labels):
                template_to_use = self.add_clifford_compilation_template(
                    gatelabel.name, gatelabel.number_of_qubits, unitary, srep,
                    available_template_labels, available_srep_dict,
                    verbosity=verbosity, max_iterations=max_iterations)

        #If a template has been found, use it.
        if template_to_use is not None:
            gstr = list( map(to_real_label, template_to_use) )
            self[gatelabel] = _Circuit(gatestring=gstr,
                                       line_labels=self.gateset.stateSpaceLabels.labels[0])
        else:
            if verbosity > 0: print("Cannot locally compile %s" % str(gatelabel))
            raise CompilationError("Cannot locally compile %s" % str(gatelabel))


    def add_clifford_compilation_template(self, gate_name, nqubits, unitary, srep,
                                          available_glabels, available_sreps,
                                          verbosity=1, max_iterations=10):

        #Get the total number of qubits in the 
        #WRONG? nQ = int(round(np.log2(self.gateset.dim))) # assumes *unitary* mode (OK?)
        
        # The unitary is specifed, this takes priority and we use it to construct the 
        # symplectic rep of the gate.
        
        if unitary is not None:
            srep = _symp.unitary_to_symplectic(unitary,flagnonclifford=True)
        
        # If the unitary has not been provided and smatrix and svector are both None, then
        # we find them from the dictionary of standard gates.
        
        if srep is None:
            template_lbl = _Label(gate_name,tuple(range(nqubits))) # integer ascending qubit labels
            smatrix, svector = _symp.clifford_layer_in_symplectic_rep([template_lbl],nqubits)
        else:
            smatrix, svector = srep
                
        assert(_symp.check_valid_clifford(smatrix,svector)), "The gate is not a valid Clifford!"
        
        assert(_np.shape(smatrix)[0] // 2 == nqubits)
        
        ## todo : this should ultimately be removed, but making this function work for 
        ## arbitrary two or multi-qubit gates is much more complicated.        
        #assert(nqubits == 1), "Function currently only works for single qubit gates!"
        
        #if available_glabels == 'all':
        #    FUTURE / REMOVE?
                          
        if verbosity > 0:
            if self.ctype == 'absolute':
                print("Generating template for an absolute compilation of the {} gate...".format(gate_name),end='\n')
            elif self.ctype == 'paulieq':
                print("Generating template for a pauli-equivalence compilation of the {} gate...".format(gate_name),end='\n')
                
        obtained_sreps = {}
        
        #Separate the available gate labels by their number of qubits
        # Note: could add "IDENT" gate to length-1 list?
        available_glabels_by_qubit = _collections.defaultdict(list)
        for gl in available_glabels:
            available_glabels_by_qubit[tuple(sorted(gl.qubits))].append(gl)
              #sort qubit labels b/c order doesn't matter and can't hash sets
    
        # Construst all possible circuit layers acting on the qubits.
        all_layers = []

        #Loop over all partitions of the nqubits
        for p in partitions(nqubits):
            pi = _np.concatenate(([0],_np.cumsum(p)))
            to_iter_over = [ available_glabels_by_qubit[tuple(range(pi[i],pi[i+1]))] for i in range(len(p)) ]
            for gls_in_layer in _itertools.product(*to_iter_over):
                all_layers.append( gls_in_layer )

        # Find the symplectic action of all possible circuits of length 1 on the qubits
        for layer in all_layers:
            obtained_sreps[layer] = _symp.clifford_layer_in_symplectic_rep(
                layer, nqubits, available_sreps)

        #TODO: REMOVE, but above should do this for 1Q case...
        ## Find the obtained gates from length-1 circuits.
        #for gl in available_glabels:
        #    if gateset.size[gl] == 1:
        #        # We force the key to be a tuple.
        #        if (gateset.smatrix[gl] is not None) and (gateset.svector[gl] is not None):
        #            obtained_smatrix[(gl,)] = gateset.smatrix[gl]  #TODO
        #            obtained_svector[(gl,)] = gateset.svector[gl]

                
        # Main loop. We go through the loop at most max_iterations times
        found = False
        for counter in range(0,max_iterations):

            if verbosity > 0:
                print("Checking all length {} {}-qubit circuits... ({})".format(counter+1,nqubits,len(obtained_sreps)))
        
            candidates = [] # all valid compilations, if any, of this length.

            # Look to see if we have found a compilation
            for seq,(s,p) in obtained_sreps.items():    
                if _np.array_equal(smatrix,s):
                    compilation = seq
                    
                    if self.ctype == 'paulieq' or \
                       (self.ctype == 'absolute' and  _np.array_equal(svector,p)):
                        candidates.append(seq)
                        found = True
                            
            # If there is more than one way to compile gate at this circuit length, pick the
            # one containing the most idle gates.
            if len(candidates) > 1:
                
                number_of_idles = 0
                max_number_of_idles = 0
                
                # Look at each sequence, and see if it has more than or equal to max_number_of_idles.
                # If so, set it to the current chosen sequence.
                for seq in candidates:
                    number_of_idles = len([x for x in seq if x.name == "IDENT"]) # or 'Gi'?
                            
                    if number_of_idles >= max_number_of_idles:
                        max_number_of_idles = number_of_idles
                        compilation = seq
            elif len(candidates) == 1:
                compilation = candidates[0]
            
            # If we have found a compilation, leave the loop
            if found:
                if verbosity > 0: print("Compilation template created!")
                break
            
            # If we have reached the maximum number of iterations, quit the loop
            # before we construct the symplectic rep for all sequences of a longer length.
            if (counter == max_iterations - 1):
                print("*************************************************************************************")
                print("           Maximum iterations reached without finding a compilation !")
                print("*************************************************************************************")
                print("")
                return None
            
            # Construct the gates obtained from the next length sequences.
            new_obtained_sreps = {}
                             
            for seq,(s,p) in obtained_sreps.items():
                # Add all possible tensor products of single-qubit gates to the end of the sequence
                for layer in all_layers:
                        
                    # Calculate the symp rep of this parallel gate
                    sadd, padd = _symp.clifford_layer_in_symplectic_rep(
                        layer, nqubits, available_sreps)
                    key = seq + layer # tuple/GateString concatenation
                        
                    # Calculate and record the symplectic rep of this gate sequence.
                    new_obtained_sreps[key] =_symp.compose_cliffords(s, p, sadd, padd)                    
                        
            # Update list of potential compilations
            obtained_sreps = new_obtained_sreps

        #Store & return template that was found
        self.templates[gate_name].append(compilation)
    
        return compilation

        
    def compute_connectivity_of(self, gate_name):
        """ TODO: docstring """
        nQ = int(round(_np.log2(self.gateset.dim))) # assumes *unitary* mode (OK?)
        d = { qlbl: i for i,qlbl in enumerate(self.gateset.stateSpaceLabels.labels[0]) }
        
        connectivity = _np.zeros( (nQ,nQ), dtype=bool )
        for compiled_gatelabel in self.keys():
            if compiled_gatelabel.name == gate_name:
                for p in _itertools.permutations(compiled_gatelabel.qubits,2):
                    connectivity[d[p[0]],d[p[1]]] = True
                      # Note: d converts from qubit labels to integer indices
                      
        distance, shortestpath = _fw(connectivity,return_predecessors=True, 
                                     directed=True, unweighted=False)
        self.connectivity[gate_name] = (connectivity, distance, shortestpath)


    def add_nonlocal_compilation_of(self, gatelabel, force=False, verbosity=1, check=True):
        """ TODO: docstring

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
        """

        if not force and gatelabel in self:
            return #don't re-compute unless we're told to

        assert(gatelabel.number_of_qubits > 1),"1-qubit gates can't be non-local!"
        assert(gatelabel.name == "CNOT" and gatelabel.number_of_qubits == 2), \
            "Only non-local CNOT compilation is currently supported."

        #Get connectivity of this gate
        if gatelabel.name not in self.connectivity: #need to recompute
            self.compute_connectivity_of(gatelabel.name)
            
        _, distancematrix, shortestpathmatrix = self.connectivity[gatelabel.name]

        #CNOT specific
        q1 = gatelabel.qubits[0]
        q2 = gatelabel.qubits[1]
        
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
            part_1.append(_Label('CNOT',[shortestpath[i],shortestpath[i+1]]))
        
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
        circuit = _Circuit(gatestring=cnot_circuit, line_labels=self.gateset.stateSpaceLabels.labels[0])

        #UNUSED - we always compile
        ## If we are not compiling the CNOTs between connected qubits into native gates
        ## then we are done.        
        #if not compile_cnots:
        #    print(circuit)
        #    return circuit
                
        # If we are compiling the CNOTs between connected qubits into native gates
        # then we now do this.
        
        #UNUSED - FUTURE IMPROVEMENT for just self.ctype == 'paulieq' case:
        ## Import the gate relations of the single-qubit Cliffords, so that circuit
        ## compression can be used. Todo: this should probably be handed to this 
        ## function, as it is should be something that is possibly in the device spec.
        #grels = _symp.single_qubit_clifford_symplectic_group_relations()
        ## To do: add an assert that checks that compilations of CNOTs between all
        ## connected qubits have been generated.
        ##
        ## Change into the native gates, using the compilation for CNOTs between
        ## connected qubits.
        circuit.change_gate_library(self)

        if check:
            # Calculate the symplectic matrix implemented by this circuit, to check the compilation
            # is ok, below.
            sreps = self.get_symplectic_reps()
            s, p = _symp.composite_clifford_from_clifford_circuit(circuit,sreps)
            
            # Construct the symplectic rep of CNOT between this pair of qubits, to compare to s.
            nQ = int(round(_np.log2(self.gateset.dim))) # assumes *unitary* mode (OK?)
            iq1 = self.gateset.stateSpaceLabels.labels[0].index(q1) # assumes single tensor-prod term
            iq2 = self.gateset.stateSpaceLabels.labels[0].index(q2) # assumes single tensor-prod term
            s_cnot, p_cnot = _symp.clifford_layer_in_symplectic_rep([_Label('CNOT',(iq1,iq2)),],nQ)
    
            assert(_np.array_equal(s,s_cnot)), "Compilation has failed!"
            if self.ctype == "absolute":
                assert(_np.array_equal(p,p_cnot)), "Compilation has failed!"

        self[gatelabel] = circuit


        
#    def add_fundamental_clifford_compilation(self, gate, gateset, unitary=None, smatrix=None, 
#                                             svector=None, ctype='absolute', max_iterations=10,
#                                             verbosity=1):
#        
#        
#        if gate.number_of_qubits == 1:
#            
#            q = gate.qubits[0]
#            
#            
#            if ctype == 'absolute':
#                compilationfound = False
#                if gate.label in list(self.absolute_helpers.keys()):
#                    
#                    compilationfound = False
#                    for compilation in self.absolute_helpers[gate.label]:
#                        
#                        # This method currently doesn't check the stored compilation
#                        # could use all the available gates for this qubit. If it couldn't
#                        # this might be suboptimal. This should be fixed, and the best way
#                        # to do this is to make compilations into maps from restricted gatesets
#                        # to a gate.
#                        
#                        flag = True
#                        for glabel in compilation:
#                            if gateset.availability[glabel][q] == 0:
#                                flag = False 
#                        
#                        if flag:
#                            self.absolute[gate] = _cir.Circuit(gate_list = [_cir.Gate(glabel,q) for glabel in compilation],
#                                                              n = gateset.number_of_qubits)
#                            compilationfound = True
#                            break
#                     
#                if not compilationfound:
#                    
#                    allowedglabels = []
#                    for glabel in gateset.names:
#                        if gateset.size[glabel] == 1 and gateset.availability[glabel][gate.qubits[0]] == 1:
#                            allowedglabels.append(glabel)
#                                
#                    self.add_clifford_compilation_helper(gate.label, gateset, allowedglabels=allowedglabels,
#                                        unitary=unitary, smatrix=smatrix,
#                                        svector=svector, ctype=ctype, verbosity=verbosity,
#                                        max_iterations=max_iterations)
#                    
#                    compilation = self.absolute_helpers[gate.label][-1]
#                    self.absolute[gate] = _cir.Circuit(gate_list = [_cir.Gate(glabel,q) for glabel in compilation],
#                                                              n = gateset.number_of_qubits)
#                        
#                    compilationfound = True
#                
#            if ctype == 'paulieq':
#                
#                if gate.label in list(self.paulieq_helpers.keys()):
#                    
#                    compilationfound = False
#                    for compilation in self.paulieq_helpers[gate.label]:
#                        
#                        # This method currently doesn't check the stored compilation
#                        # could use all the available gates for this qubit. If it couldn't
#                        # this might be suboptimal. This should be fixed, and the best way
#                        # to do this is to make compilations into maps from restricted gatesets
#                        # to a gate.
#                        
#                        flag = True
#                        for glabel in compilation:
#                            if gateset.availability[glabel][q] == 0:
#                                flag = False 
#                        
#                        if flag:
#                            self.paulieq[gate] = _cir.Circuit(gate_list = [_cir.Gate(glabel,q) for glabel in compilation],
#                                                              n = gateset.number_of_qubits)
#                            compilationfound = True
#                            break
#                                         
#                if not compilationfound:
#                    
#                    allowedglabels = []
#                    for glabel in gateset.names:
#                        if gateset.size[glabel] == 1 and gateset.availability[glabel][gate.qubits[0]] == 1:
#                            allowedglabels.append(glabel)
#                                
#                    self.add_clifford_compilation_helper(gate.label, gateset, allowedglabels=allowedglabels,
#                                        unitary=unitary, smatrix=smatrix,
#                                        svector=svector, ctype=ctype, verbosity=verbosity,
#                                        max_iterations=max_iterations)
#                    
#                    compilation = self.paulieq_helpers[gate.label][-1]
#                    self.paulieq[gate] = grb.Circuit(gate_list = [grb.Gate(glabel,q) for glabel in compilation],
#                                                              n = gateset.number_of_qubits)
#                        
#                    compilationfound = True
#         
#        elif gate.label == 'CNOT':
#            print("Not yet written!")
#        
#        elif gate.label == 'CPHASE':
#            print("Not yet written!")
#            
#        elif gate.label == 'SWAP':
#            print("Not yet written!")
#            
#        else:
#            print("An automated fundamental compilation for this gate is not currently supported!")
#
#    # Just setitem
#    #def add_general_clifford_compilation(self,gatelikeobject, circuit, ctype):
#    #    
#    #    if ctype == 'absolute':
#    #        
#    #        self.absolute[gatelikeobject] = circuit
#    #        
#    #    elif ctype == 'paulieq':
#    #        
#    #         self.paulieq[gatelikeobject] = circuit
#            
#    def add_clifford_compilation_helper(self, gate_name, gateset, allowedglabels='all',
#                                        unitary=None, smatrix=None,
#                                        svector=None, ctype='absolute', verbosity=1,
#                                        max_iterations=10):
#        
#        # If gate label does not yet have any compilations, start a list for the 
#        # compilations.
#        
#        if ctype == 'absolute':
#            if glabel not in list(self.absolute_helpers.keys()):
#                self.absolute_helpers[glabel] = []
#                
#        if ctype == 'paulieq':
#            if glabel not in list(self.paulieq_helpers.keys()):
#                self.paulieq_helpers[glabel] = []
#                
#        #print('---',glabel)
#        #print(self.absolute_helpers)
#        #print(self.paulieq_helpers)
#        
#        # The unitary is specifed, this takes priority and we use it to construct the 
#        # symplectic rep of the gate.
#        
#        if unitary is not None:
#            
#            smatrix, svector = _symp.unitary_to_symplectic(unitary,flagnonclifford=True)
#        
#        # If the unitary has not been provided and smatrix and svector are both None, then
#        # we find them from the dictionary of standard gates.
#        
#        if (smatrix is None) or (svector is None):
#        
#            std_smatrices, std_svectors = _symp.symplectic_representation()
#            try:
#                smatrix = std_smatrices[glabel]
#                svector = std_svectors[glabel]
#            except:
#                raise ValueError
#                
#        assert(_symp.check_valid_clifford(smatrix,svector)), "The gate is not a valid Clifford!"
#        
#        n = _np.shape(smatrix)[0] // 2
#        
#        # todo : this should ultimately be removed, but making this function work for 
#        # arbitrary two or multi-qubit gates is much more complicated.
#        
#        assert(n == 1), "Function currently only works for single qubit gates!"
#        
#        if n == 1:
#            
#            if allowedglabels == 'all':
#                allowedglabels = []
#                for gl in gateset.names:
#                    if gateset.size[gl] == 1: #and gateset.availability[glabel] == 1:
#                        allowedglabels.append(gl)
#                          
#            if verbosity > 0:
#                if ctype == 'absolute':
#                    print("Generating an absolute compilation for the {} gate...".format(glabel),end='')
#                if ctype == 'paulieq':
#                    print("Generating a pauli-equivalence compilation for the {} gate...".format(glabel),end='')
#                    
#            obtained_smatrix = {}
#            obtained_svector = {}
#            
#            # Find the obtained gates from length-1 circuits.
#            for gl in allowedglabels:
#                if gateset.size[gl] == 1:
#                    # We force the key to be a tuple.
#                    if (gateset.smatrix[gl] is not None) and (gateset.svector[gl] is not None):
#                        obtained_smatrix[(gl,)] = gateset.smatrix[gl]  
#                        obtained_svector[(gl,)] = gateset.svector[gl]
#        
#            # Main loop. We go through the loop at most max_iterations times
#            flag = False
#            for counter in range(0,max_iterations):
#
#                #if verbosity > 0:
#                #    print("")
#                #    print("")
#                #    print("Checking all length {} single-qubit circuits...".format(counter))
#            
#                # Check each of the obtained symplectic matrices, and see if it corresponds to
#                # any of the gates we are still searching for compilations for.
#                for gseq in list(obtained_smatrix.keys()):
#                
#                    # Check if we have found a compilation. Note that multiple compilations are
#                    # recorded if they are found at the same length.
#                    if _np.array_equal(smatrix,obtained_smatrix[gseq]):
#                        # Save the compilation as a list, rather than a tuple.
#                        if ctype == 'paulieq':
#                            self.paulieq_helpers[glabel].append(list(gseq))
#                            flag = True
#                        if ctype == 'absolute':
#                            if _np.array_equal(svector,obtained_svector[gseq]):
#                                self.absolute_helpers[glabel].append(list(gseq))
#                                flag = True
#                            
#                # If we have found the gate, leave the loop
#                if flag:
#                    if verbosity > 0:
#                        print("Complete.")
#                    break
#                
#                # If we have gone through the maximum amount of iterations, we quit before constructing
#                # the circuits for the next length. 
#                if counter == max_iterations-1:
#                    print("** Maximum iterations reached without finding a compilation! **")
#                    break
#            
#                # If we have not found all the gates, find the gates obtained by the next length
#                # gate sequences.
#                new_obtained_smatrix = {}
#                new_obtained_svector = {}
#                # Consider each sequence obtained so far, which is recorded as a tuple.
#                for seq in list(obtained_smatrix.keys()):
#                    # Consider adding each possible gate to the end of that sequence.
#                    for gl in allowedglabels:
#                        if gateset.size[gl] == 1:
#                            if (gateset.smatrix[gl] is not None) and (gateset.svector[gl] is not None):
#                                # Turn the tuple, for the gate sequence into a list, to append the next gate.
#                                key = list(seq)
#                                key.append(gl)
#                                # Turn back into a tuple, so that it can be a dict key.
#                                key = tuple(key)
#                                # Find the gate implemented by the new sequence, and record.
#                                new_obtained_smatrix[key], new_obtained_svector[key] = _symp.compose_cliffords(obtained_smatrix[seq], obtained_svector[seq],gateset.smatrix[gl], gateset.svector[gl])
#                                
#                # Write over the old obtained sequences.
#                obtained_smatrix = new_obtained_smatrix
#                obtained_svector = new_obtained_svector
#                
#                
#    def add_CNOT_for_connected_qubits(self,qubits,gateset,max_iterations=6,ctype='absolute',verbosity=1):
#        """
#        This method generates a compilation for CNOT, up to arbitrary Pauli gates, between a
#        pair of connected qubits. The method is a brute-force search algorithm, checking 
#        all circuit of each length until a compilation is found.
#        
#        Parameters
#        ----------
#        q1 : int
#            The control qubit label
#            
#        q2 : int
#            The target qubit label
#            
#        max_iterations : int, optional
#            The maximum number of iterations in the search algorithm, which corresponds to
#            the maximum length of circuit checked to find compilations for all needed gates.
#            The time required for the alogrithm scales exponentially in this parameter, but
#            the default should be sufficient for any entangling gate that is can be converted
#            to CNOT by local Cliffords (e.g., CPHASE, or CNOT with the opposite control and 
#            target direction).
#            
#        verbosity : int, optional
#            The amount of print-to-screen.
#            
#        Returns
#        -------
#        circuit : a Circuit object
#            A circuit representation of the constructed CNOT compilation.
#        
#        """
#        q1 = qubits[0]
#        q2 = qubits[1]
#        
#        if verbosity > 0:
#            print("")
#            print("Attempting to generate a compilation for CNOT, up to Paulis,")
#            print("with control qubit = {} and target qubit = {}".format(q1,q2))
#            print("")
#            
#        # The obtained gates, in the symplectic rep, after each round of the algorithm                               
#        obtained_s = {}
#        obtained_p = {}
#        
#        # Import the symplectic representations of the standard gates, to get the CNOT rep.
#        #std_smatrices, std_svectors = _symp.symplectic_representation()
#        #CNOTsmatrix = std_smatrices['CNOT']
#        #CNOTsvector = std_svectors['CNOT']
#        
#        CNOTsmatrix, CNOTsvector = _symp.clifford_layer_in_symplectic_rep([_cir.Gate('CNOT',(q1,q2)),],
#                                                                          gateset.number_of_qubits)
#        
#        
#        symplectic_matrix = {}
#        phase_vector = {}
#        
#        # Construst all possible circuit layers acting on the two qubits.
#        all_layers = []
#        for gl in gateset.names:
#            
#            # Go in here if it is a single qubit gate
#            if gateset.size[gl] == 1 and gateset.availability[gl][q1] == 1:
#                
#                # Check the gate is a Clifford. If it isn't we cannot use it in this algorithm
#                if (gateset.smatrix[gl] is not None) and (gateset.svector[gl] is not None):
#                    
#                    for gl2 in gateset.names:
#                        if gateset.size[gl2] == 1 and gateset.availability[gl2][q2] == 1:
#                            if (gateset.smatrix[gl2] is not None) and (gateset.svector[gl2] is not None):
#                                all_layers.append([_cir.Gate(gl,q1),_cir.Gate(gl2,q2)])
#            
#            # Go in here if it is a two qubit gate
#            if gateset.size[gl] == 2:
#                if gateset.availability[gl][q1,q2] == 1:
#                    all_layers.append([_cir.Gate(gl,(q1,q2)),])                        
#                if gateset.availability[gl][q2,q1] == 1:
#                    all_layers.append([_cir.Gate(gl,(q2,q1)),])    
#        
#        # Find the symplectic action of all possible circuits of length 1 on the two qubits
#        for layer in all_layers:
#            s, p =  _symp.clifford_layer_in_symplectic_rep(layer, gateset.number_of_qubits,  
#                                                           s_dict=gateset.smatrix, p_dict=gateset.svector)
#            obtained_s[tuple(layer)] =  s
#            obtained_p[tuple(layer)] =  p
#        
#        # Bool to flag up when we have found a compilation for CNOT.
#        cnot_found = False
#        
#        # Loop over the number of permisable iterations, until we find CNOT or run out of iterations.
#        for counter in range(0,max_iterations):
#                
#            if verbosity > 0:
#                print("Checking all length {} two-qubit circuits...".format(counter+1))
#            
#            # Keep a list of all compilations for CNOT, if any, of this length.
#            candidates = []
#            
#            # Look to see if we have found a compilation for CNOT, up to Paulis.
#            for key in list(obtained_s.keys()):    
#                
#                if _np.array_equal(CNOTsmatrix,obtained_s[key]):
#                    
#                    cnot_compilation = list(key)
#
#                    if ctype == 'paulieq':
#                        candidates.append(key)                    
#                        cnot_found = True
#                        
#                    if ctype == 'absolute':
#                        if _np.array_equal(CNOTsvector,obtained_p[key]):
#                            candidates.append(key)                    
#                            cnot_found = True
#                            
#            # If there is more than one way to compile CNOT at this circuit length, pick the
#            # one containing the most idle gates.
#            if len(candidates) > 1:
#                
#                number_of_idles = 0
#                max_number_of_idles = 0
#                
#                # Look at each sequence, and see if it has more than or equal to max_number_of_idles.
#                # If so, set it to the current chosen sequence.
#                for seq in candidates:
#                    number_of_idles = 0
#                    
#                    for xx in range(0,len(seq)):
#                        if seq[xx].label == 'I':
#                            number_of_idles += 1 
#                            
#                    if number_of_idles >= max_number_of_idles:
#                        max_number_of_idles = number_of_idles
#                        best_sequence = seq
#              
#                cnot_compilation = list(best_sequence)    
#                cnot_compilation_phase = obtained_p[best_sequence]
#            
#            # If we have found CNOT, leave the loop
#            if cnot_found:
#                if verbosity > 0:
#                    print("Compilation for CNOT found!")
#                break
#            
#            # If we have reached the maximum number of iterations, quit the loop
#            # before we construct the symplectic rep for all sequences of a longer length.
#            if (counter == max_iterations - 1) and cnot_found is False:
#                print("*************************************************************************************")
#                print("           Maximum iterations reached without finding a CNOT compilation !")
#                print("*************************************************************************************")
#                print("")
#                return None
#            
#            # Construct the gates obtained from the next length sequences.
#            new_obtained_s = {}
#            new_obtained_p = {}
#                             
#            for seq in list(obtained_s.keys()):
#                # Add all possible tensor products of single-qubit gates to the end of the sequence
#                for layer in all_layers:
#                        
#                    # Calculate the symp rep of this parallel gate
#                    sadd, padd = _symp.clifford_layer_in_symplectic_rep(layer, gateset.number_of_qubits,
#                                                                         s_dict=gateset.smatrix, 
#                                                                         p_dict=gateset.svector)
#                        
#                    # Convert to list, to append gates, and then convert back to tuple
#                    key = list(seq)
#                    key = key + layer
#                    key = tuple(key) 
#                        
#                    # Calculate and record the symplectic rep of this gate sequence.
#                    
#                    new_obtained_s[key], new_obtained_p[key] =_symp.compose_cliffords(obtained_s[seq], 
#                                                                                          obtained_p[seq],
#                                                                                          sadd, padd)                          
#                        
#            # Overwrite old sequences.
#            obtained_s = new_obtained_s
#            obtained_p = new_obtained_p
#                
#        # Convert the gate sequence to a circuit
#        circuit = _cir.Circuit(gate_list=cnot_compilation,n=gateset.number_of_qubits)
#        #print(circuit)
#        
#        return circuit
#               
#    def add_CNOT_for_unconnected_qubits(self, qubits, shortestpathmatrix, distancematrix, 
#                                        gateset, verbosity=1, ctype='absolute', 
#                                        compile_cnots=False):
#        """
#        This method generates a compilation for CNOT, up to arbitrary Pauli gates, between a
#        pair of unconnected qubits. This function converts this CNOT into a circuit of CNOT
#        gates between connected qubits, using a fixed circuit form. This compilation is not 
#        optimal in at least some circumstances.
#        
#        Parameters
#        ----------
#        q1 : int
#            The control qubit label
#            
#        q2 : int
#            The target qubit label
#            
#        verbosity : int, optional
#            The amount of print-to-screen.
#            
#        Returns
#        -------
#        circuit : a Circuit object
#            A circuit representation of the constructed CNOT compilation.        
#        
#        """
#        q1 = qubits[0]
#        q2 = qubits[1]
#        
#        if verbosity > 0:
#            print("")
#            print("Attempting to generate a compilation for CNOT, up to Paulis,")
#            print("with control qubit = {} and target qubit = {}".format(q1,q2))
#            print("")
#            print("Distance between qubits is = {}".format(distancematrix[q1,q2]))
#            
#        assert(shortestpathmatrix[q1,q2] >= 0), "There is no path between the qubits!"
#        
#        # Find the shortest path between q1 and q2, from self.shortest_path_matrix
#        # We do this by following the chain in the shortest_path_matrix until we arrive at q1
#        shortestpath = [q2]
#        current_node = q2
#                
#        while current_node != q1:            
#            preceeding_node = shortestpathmatrix[q1,current_node]
#            shortestpath.insert(0,preceeding_node)
#            current_node = preceeding_node
#            
#        # If the qubits are connected, this algorithm may not behave well.
#        assert(len(shortestpath)> 2), "Qubits are connected! Algorithm is not needed or valid."
#        
#        # Part 1 of the circuit is CNOTs along the shortest path from q1 to q2.
#        # To do: describe the circuit.
#        part_1 = []            
#        for i in range(0,len(shortestpath)-1):
#            part_1.append(_cir.Gate('CNOT',[shortestpath[i],shortestpath[i+1]]))
#        
#        # Part 2 is...
#        # To do: describe the circuit.
#        part_2 = _copy.deepcopy(part_1)
#        part_2.reverse()
#        del part_2[0]
#        
#        # To do: describe the circuit.
#        part_3 = _copy.deepcopy(part_1)
#        del part_3[0]
#        
#        # To do: describe the circuit.
#        part_4 = _copy.deepcopy(part_3)
#        del part_4[len(part_3)-1]
#        part_4.reverse()
#        
#        # Add the lists of gates together, in order
#        cnot_circuit = part_1 + part_2 + part_3 + part_4
#        
#        # Convert the gatelist to a circuit.
#        circuit = _cir.Circuit(gate_list=cnot_circuit,n=gateset.number_of_qubits)
#        
#        # If we are not compiling the CNOTs between connected qubits into native gates
#        # then we are done.        
#        if not compile_cnots:
#            print(circuit)
#            return circuit
#                
#        # If we are compiling the CNOTs between connected qubits into native gates
#        # then we now do this.
#        else:
#            
#            if ctype == 'paulieq':
#            
#                # Import the gate relations of the single-qubit Cliffords, so that circuit
#                # compression can be used. Todo: this should probably be handed to this 
#                # function, as it is should be something that is possibly in the device spec.
#                grels = _symp.single_qubit_clifford_symplectic_group_relations()
#                # To do: add an assert that checks that compilations of CNOTs between all
#                # connected qubits have been generated.
#                #
#                # Change into the native gates, using the compilation for CNOTs between
#                # connected qubits.
#                circuit.change_gate_library(self.paulieq)
#                
#            if ctype == "absolute":
#                circuit.change_gate_library(self.absolute)
#            
#            # Calculate the symplectic matrix implemented by this circuit, to check the compilation
#            # is ok, below.
#            s, p = _symp.composite_clifford_from_clifford_circuit(circuit, s_dict=gateset.smatrix, 
#                                                            p_dict=gateset.svector)
#            
#            # Construct the symplectic rep of CNOT between this pair of qubits, to compare to s.
#            s_cnot, p_cnot = _symp.clifford_layer_in_symplectic_rep([_cir.Gate('CNOT',(q1,q2)),],
#                                                                          gateset.number_of_qubits)
#
#
#            assert(_np.array_equal(s,s_cnot)), "Compilation has failed!"
#            if ctype == "absolute":
#                assert(_np.array_equal(p,p_cnot)), "Compilation has failed!"
#                
#            return circuit            

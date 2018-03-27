""" Defines the ProcessorSpec class and supporting functionality."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import itertools as _itertools
from scipy.sparse.csgraph import floyd_warshall as _fw

from .compilationlibrary import CompilationLibrary as _CompilationLibrary
from .compilationlibrary import CompilationError as _CompilationError
from .label import Label as _Label

class ProcessorSpec(object):
    """ TODO: docstring """
    
    def __init__(self, nQubits, gate_names, nonstd_gate_unitaries=None,
                 availability=None, parameterization='static', sim_type="dmmap",
                 construct_std_compilations=True,verbosity=1):
        """
        An object that can be used to encapsulate the device specification for a one or more qubit 
        quantum computer.
    
        The most basic information required for a ProcessorSpec object is the number of qubits in the 
        device, and the library of "native" target gates. This is a list of unitary operators, acting 
        on ordinary pure state  vectors (so they are 2^k by 2^k complex arrays, where k is the number 
        of qubits that gate acts upon), defined with respect to the standard computational basis. 
    
        This gate library should include all native gates, and they need not -- and generically should 
        not -- be unitaries acting on all of the qubits. E.g., an example of a gate library would be  
        {'H' = 2x2 matrix defining the Hadamard gate, 'CNOT' : 4x4 matrix defining the CNOT gate, ...}.
        
        Parameters
        ----------
        nQubits : int
            The number of qubits in the device.
        
        gate_library: dictionary of numpy arrays
            A library consisting of all (target) native gates that can be implemented in the device, as
            unitaries acting on ordinary pure-state-vectors, in the standard basis. These unitaries need
            not, and in most circumstances should not, be unitaries acting on all of the qubits. E.g., an 
            example of  a gate library would be  {'H' = 2x2 matrix defining the Hadamard gate, 
            'CNOT' : 4x4 matrix  defining the CNOT gate, 'P' = 2x2 matrix defining the Phase gate}. 
            The keys of this dictionary must be strings, and are the gate names.

        TODO: docstring
        """
        from .. import construction as _cnst
        assert(type(nQubits) is int), "The number of qubits, n, should be an integer!"
        
        self.number_of_qubits = nQubits
        self.gateset = _cnst.build_nqubit_standard_gateset(
            nQubits, gate_names, nonstd_gate_unitaries, availability,
            parameterization, sim_type)

        # *** Done in build_nqubit_standard_gateset ***
        ## A list of all available gates, in terms of Gate objects
        #self.allgates = []
        #
        ## A dictionary of all available gates on each qubit, and each pair of qubits
        #self.gatesonqubits = {}
        #
        ## Initilizes these dictionaries
        #for q1 in range (0,self.number_of_qubits):
        #    self.gatesonqubits[q1] = []
        #    for q2 in range (0,self.number_of_qubits):
        #        self.gatesonqubits[q1,q2] = []
        #
        ## Populates these dictionaries.
        #for glabel in self.gateset.names:
        #    
        #    # Adds the one-qubit gates.
        #    if self.gateset.size[glabel] == 1:
        #        for q in range(0,self.number_of_qubits):
        #            if self.gateset.availability[glabel][q] == 1:
        #                self.allgates.append(_cir.Gate(glabel,q))
        #                self.gatesonqubits[q].append(_cir.Gate(glabel,q))
        #    
        #    # Adds the two-qubit gates.
        #    elif self.gateset.size[glabel] == 2:
        #        for q1 in range(0,self.number_of_qubits):
        #            for q2 in range(0,self.number_of_qubits):
        #                if self.gateset.availability[glabel][q1,q2] == 1:
        #                    self.allgates.append(_cir.Gate(glabel,(q1,q2)))
        #                    self.gatesonqubits[q1,q2].append(_cir.Gate(glabel,(q1,q2)))
        #                    
        #    else:
        #        raise ValueError("Gates on > 2 qubits not currently recorded!")

        # Construct matrices related to the cost of doing a two-qubit gate between a pair
        # of qubits. This makes default values which compilers will use. But, this is a
        # function so that the user can call it and update them.
        self.construct_compiler_costs()

        # Compilations are stored here. This initalizes empty CompilationLibrary objects
        self.compilations = {'absolute': _CompilationLibrary(self.gateset, 'absolute'),
                             'paulieq':  _CompilationLibrary(self.gateset, 'paulieq') }
        
        # A dictionary of models for the device (e.g., imperfect unitaries, process matrices etc).
        self.models = {} # TODO: just other gatesets??
        
        # Constructs the standard compilations, if requested.
        if construct_std_compilations:        
            self.construct_std_compilations(verbosity=verbosity)

            
    def construct_std_compilations(self,verbosity=1):

        #Hard-coded gates we need to compile from the native (clifford) gates in order to compile
        # arbitrary (e.g. random) cliffords, since our Clifford compiler outputs circuits in terms
        # of these elements.
        singlequbit = {'paulieq': ['H','P','PH','HP','HPH'],
                       'absolute': ['I','X','Y','Z'] }
        twoqubit = {'paulieq': ['CNOT'],
                    'absolute': [] }
        descs = {'paulieq': 'up to Pauli gates',
                 'absolute': '' }

        for compile_type in ('paulieq','absolute'):
            desc = descs[compile_type]
            
            #Stage1:
            # Compile 1Q gates "locally" - i.e., out of native gates which act only
            #  on the target qubit of the gate being compiled.
            for q in range(0,self.number_of_qubits):
                for gname in singlequbit[compile_type]:
                    if verbosity > 0:
                        print("Creating a circuit to implement {} {} on qubit {}...".format(gname,desc,q))
                    self.compilations[compile_type].add_local_compilation_of(
                        _Label(gname,q), verbosity=verbosity)

                if verbosity > 0: print("Complete.")
                
            #Stage2:
            # Compile 2Q gates locally, if possible.  Keep track of what can't be compiled.
            not_locally_compilable = []
            
            for q1 in range(0,self.number_of_qubits):
                for q2 in range(0,self.number_of_qubits):
                    if q1 == q2: continue # 2Q gates must be on different qubits!
                    for gname in twoqubit[compile_type]:
                        if verbosity > 0:
                            print("Creating a circuit to implement {} {} on qubits {}...".format(gname,desc,(q1,q2)))
                        try:
                            self.compilations[compile_type].add_local_compilation_of(
                                _Label(gname,(q1,q2)), verbosity=verbosity)
                        except _CompilationError:
                            not_locally_compilable.append( (gname,q1,q2) )
                            
            #Stage3:
            # Try to compile remaining 2Q gates non-locally using specific algorithms
            non_compilable = []
            for gname,q1,q2 in not_locally_compilable:
                # This method is currently a bit of a hack.
                self.compilations[compile_type].add_nonlocal_compilation_of(_Label(gname,(q1,q2)),
                                                                            verbosity=verbosity)
                                 
    def construct_compiler_costs(self, custom_connectivity=None):

        # A matrix that stores whether there is any gate between a pair of qubits
        if custom_connectivity is not None:
            assert(custom_connectivity.shape == (self.number_of_qubits,self.number_of_qubits))
            self.connectivity = custom_connectivity
        else:
            self.connectivity = _np.zeros((self.number_of_qubits,self.number_of_qubits),dtype=bool)
            for gatelabel in self.gateset.gates:
                if gatelabel.number_of_qubits > 1:
                    for p in _itertools.permutations(gatelabel.qubits, 2):
                        self.connectivity[p] = True
        
        self.distance, self.shortestpath = _fw(self.connectivity,return_predecessors=True, 
                                               directed=True, unweighted=False)
        
        self.qubitcosts = {}
        
        #
        # todo -- I'm not sure whether this makes sense when the graph is directed.
        #
        for i in range(0,self.number_of_qubits):
            self.qubitcosts[i] = _np.sum(self.distance[i,:])
        
        temp_distances = list(_np.sum(self.distance,0))
        temp_qubits = list(_np.arange(0,self.number_of_qubits))
        self.costorderedqubits = []
               
        while len(temp_distances) > 0:
            
            longest_remaining_distance = max(temp_distances)
            qubits_at_this_distance = []
            
            while longest_remaining_distance == max(temp_distances):
                
                index = _np.argmax(temp_distances)
                qubits_at_this_distance.append(temp_qubits[index])
                del temp_distances[index]
                del temp_qubits[index]
                
                if len(temp_distances) == 0:
                    break
        
            self.costorderedqubits.append(qubits_at_this_distance)
            
            
    def simulate(self,circuit,modelname):
        """
        A wrap-around for the circuit simulators in simulators.py 
        """       
        return self.models[modelname].probs(circuit)

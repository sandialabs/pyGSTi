from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy
from scipy.sparse.csgraph import floyd_warshall as _fw

from . import symplectic as _symp
from . import stdgates as _std
from . import circuit as _cir
from . import compilefundamental as _cf
from . import simulators as _sim

class BasicGateSet(object):
    
    #
    # Erik : this object is intended to be replaced by a minimal version of a pyGSTi GateSet,
    # containing a list of gate names and the corresponding unitaries, etc. (see below).
    # For GST purposes, but also for other purposes, it would probably make sense to include
    # SPAM operations explicitly in this object. At some point circuits should probably optionally
    # consist of SPAM operations as well as gates.
    #
    
    def __init__(self, n, gllist=[], unitaries={}, availability={}, clifford=False): 
        """
        Todo:
        """
        assert(type(n) is int), "The number of qubits, n, should be an integer!"
        self.number_of_qubits = n
        self.names = [] #gllist + list(unitaries.keys())
        self.unitaries = {}
        self.availability = {}
        self.size = {}
        self.clifford = clifford
        
        if clifford:
            self.smatrix = {}
            self.svector = {}
            
        standard_unitaries = _std.standard_unitaries()
    
        for glabel in gllist:
            
            gunitary = standard_unitaries[glabel]
            if glabel in availability.keys():
                gavailability = availability[glabel]
            else:
                numqubits = int(_np.log2(_np.shape(gunitary)[0]))
                if numqubits == 1:
                     gavailability = _np.ones(n,int)
                elif numqubits == 2:
                    gavailability = _np.ones((n,n),int) - _np.identity(n,int)
                else:
                    print("Only one and two qubit gates currently supported!")
                    #mshape = tuple([n for i in range(0,numqubits)])
                    #gavailability = np.ones(mshape,int) - ?
                    
            self.add_gate_to_gateset(glabel,gunitary,gavailability)
                             
        
        for glabel in list(unitaries.keys()):
            
            gunitary = unitaries[glabel]
            if glabel in availability.keys():
                gavailability = availability[glabel]
            else:
                numqubits = int(_np.log2(_np.shape(gunitary)[0]))
                if numqubits == 1:
                     gavailability = _np.ones(n,int)
                elif numqubits == 2:
                    gavailability = _np.ones((n,n),int) - _np.identity(n,int)
                else:
                    print("Only one and two qubit gates currently supported!")
                    #mshape = tuple([n for i in range(0,numqubits)])
                    #gavailability = np.ones(mshape,int) - ?
            
            self.add_gate_to_gateset(glabel,gunitary,gavailability)
    
    def add_gate_to_gateset(self,glabel,unitary,availability):
        
        #print(glabel)
        #print(self.names)
        assert(glabel not in self.names), "A gate with this label is already in the gateset!"        
        self.names.append(glabel)
        self.unitaries[glabel] = unitary
        self.availability[glabel]= availability
        self.size[glabel] = int(_np.log2(_np.shape(unitary)[0]))
        
        #
        # todo: here put checks that the added gate does not clash with internally defined gates.
        #
        
        if self.clifford:
            # The symplectic representation the gates -- if they are Clifford and on two or less qubits. 
            # If they are not Clifford, None is stored.
            
            try:
                type(self.smatrix) == dict
                type(self.svector) == dict
            except:
                self.smatrix = {}
                self.svector = {}
            
            u = self.unitaries[glabel]
            self.smatrix[glabel], self.svector[glabel] = _symp.unitary_to_symplectic(u, flagnonclifford=False)
    
    #def get_subgateset(qtuple):
        
    #    gates_dict = {}
        
        #
        # todo: returns a dictionary where the keys are the gate labels and elements are the unitaries.
        #
        
    #    return gates_dict
                     
class DeviceSpec(object):

    def __init__(self, n, gllist=[], unitaries={}, availability={}, clifford=True,
                construct_std_compilations=True,verbosity=1):
        """
        An object that can be used to encapsulate the device specification for a one or more qubit 
        quantum computer.
    
        The most basic information required for a DeviceSpec object is the number of qubits in the 
        device, and the library of "native" target gates. This is a list of unitary operators, acting 
        on ordinary pure state  vectors (so they are 2^k by 2^k complex arrays, where k is the number 
        of qubits that gate acts upon), defined with respect to the standard computational basis. 
    
        This gate library should include all native gates, and they need not -- and generically should 
        not -- be unitaries acting on all of the qubits. E.g., an example of a gate library would be  
        {'H' = 2x2 matrix defining the Hadamard gate, 'CNOT' : 4x4 matrix defining the CNOT gate, ...}.
        
        Parameters
        ----------
        n : int
            The number of qubits in the device.
        
        gate_library: dictionary of numpy arrays
            A library consisting of all (target) native gates that can be implemented in the device, as
            unitaries acting on ordinary pure-state-vectors, in the standard basis. These unitaries need
            not, and in most circumstances should not, be unitaries acting on all of the qubits. E.g., an 
            example of  a gate library would be  {'H' = 2x2 matrix defining the Hadamard gate, 
            'CNOT' : 4x4 matrix  defining the CNOT gate, 'P' = 2x2 matrix defining the Phase gate}. 
            The keys of this dictionary must be strings, and are the gate names.
            
        
        """
        assert(type(n) is int), "The number of qubits, n, should be an integer!"
        
        self.number_of_qubits = n
        self.gateset = BasicGateSet(n, gllist, unitaries, availability, clifford=clifford)
        
        # A list of all available gates, in terms of Gate objects
        self.allgates = []
        
        # A dictionary of all available gates on each qubit, and each pair of qubits
        self.gatesonqubits = {}
        
        # Initilizes these dictionaries
        for q1 in range (0,self.number_of_qubits):
            self.gatesonqubits[q1] = []
            for q2 in range (0,self.number_of_qubits):
                self.gatesonqubits[q1,q2] = []
        
        # Populates these dictionaries.
        for glabel in self.gateset.names:
            
            # Adds the one-qubit gates.
            if self.gateset.size[glabel] == 1:
                for q in range(0,self.number_of_qubits):
                    if self.gateset.availability[glabel][q] == 1:
                        self.allgates.append(_cir.Gate(glabel,q))
                        self.gatesonqubits[q].append(_cir.Gate(glabel,q))
            
            # Adds the two-qubit gates.
            elif self.gateset.size[glabel] == 2:
                for q1 in range(0,self.number_of_qubits):
                    for q2 in range(0,self.number_of_qubits):
                        if self.gateset.availability[glabel][q1,q2] == 1:
                            self.allgates.append(_cir.Gate(glabel,(q1,q2)))
                            self.gatesonqubits[q1,q2].append(_cir.Gate(glabel,(q1,q2)))
                            
            else:
                raise ValueError, "Gates on > 2 qubits not currently recorded!"
        
        # A matrix that stores whether there is any two-qubit gate between a pair of qubits
        self.connectivity = _np.zeros((n,n),int)
                          
        for gate in self.allgates:
            if gate.number_of_qubits == 2:
                qubits = gate.qubits
                self.connectivity[gate.qubits] = 1
                
        self.connectivity = self.connectivity + _np.transpose(self.connectivity)       
        self.connectivity[self.connectivity != 0] = 1                  
        
        # Construct matrices related to the cost of doing a two-qubit gate between a pair
        # of qubits. This makes default values which compilers will use. But, this is a
        # function so that the user can call it and update them.
        self.construct_preferred_2Qgate_paths(self.connectivity)

        # Compilations are stored here. This initalizes an empty CompilationLibraries object
        self.compilations = _cf.CompilationLibraries()
        
        # A dictionary of models for the device (e.g., imperfect unitaries, process matrices etc).
        self.models = {}
        
        # Constructs the standard compilations, if requested.
        if construct_std_compilations:
        
            self.construct_std_compilations(verbosity=verbosity)
            
    def construct_std_compilations(self,verbosity=1):
            
        singlequbit_paulieq = ['H','P','PH','HP','HPH']
        singlequbit_absolute = ['I','X','Y','Z']
        
        twoqubit_paulieq = ['CNOT','CPHASE','SWAP']
        twoqubit_aboslute = []
            
        for glabel in singlequbit_paulieq:
            self.compilations.add_clifford_compilation_helper(glabel,self.gateset,ctype='paulieq',verbosity=verbosity)
        for glabel in singlequbit_absolute:
            self.compilations.add_clifford_compilation_helper(glabel,self.gateset,ctype='absolute',verbosity=verbosity)
                
        for q in range(0,self.number_of_qubits):
            for glabel in singlequbit_paulieq:
                if verbosity > 0:
                    print("Creating a circuit to implement {} up to Pauli gates on qubit {}...".format(glabel,q),end="")
                self.compilations.add_fundamental_clifford_compilation(_cir.Gate(glabel,q),self.gateset, 
                                                                       ctype='paulieq', verbosity=verbosity)
                if verbosity > 0:
                    print("Complete.")
            for glabel in singlequbit_absolute:
                if verbosity > 0:
                    print("Creating a circuit to implement {} on qubit {}...".format(glabel,q),end="")
                self.compilations.add_fundamental_clifford_compilation(_cir.Gate(glabel,q),self.gateset, 
                                                                       ctype='absolute', verbosity=verbosity)
                if verbosity > 0:
                    print("Complete.")  
         
        for q1 in range(0,self.number_of_qubits):
            for q2 in range(0,self.number_of_qubits):
                if self.connectivity[q1,q2] == 1:
                    
                    # This method is currently a bit of a hack.
                    circuit = self.compilations.add_CNOT_for_connected_qubits((q1,q2),self.gateset, 
                                                                       ctype='paulieq', verbosity=verbosity)
                    self.compilations.paulieq[_cir.Gate('CNOT',(q1,q2))] = circuit

                     
        for q1 in range(0,self.number_of_qubits):
            for q2 in range(0,self.number_of_qubits):
                if (self.connectivity[q1,q2] != 1) and (q1 != q2):
                    
                    # This method is currently a bit of a hack.
                    circuit = self.compilations.add_CNOT_for_unconnected_qubits((q1,q2),  self.shortestpath, 
                                                                                self.distance, self.gateset, 
                                                                       ctype='paulieq', verbosity=verbosity,
                                                                               compile_cnots=True)
                    
                    self.compilations.paulieq[_cir.Gate('CNOT',(q1,q2))] = circuit
                                 
    def construct_preferred_2Qgate_paths(self, weighted_paths):
        
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
            
            
    def simulate(self,circuit,modelname,inputstate=None,store=True,returnall=False):        
        """
        A wrap-around for the circuit simulators in simulators.py 
        """        
        #model = self.model[modelname]
        #        
        #if model.mtype == 'mixedstate':
        #    out = _sim.mixedstate_simulator(circuit, model, input_bitstring=inputstate, store=store, returnall=returnall)
        #
        #elif model.mtype == 'purestate':
        #    out = _sim.vectorstate_simulator(circuit, model, inputstate=inputstate, store=store, returnall=returnall)
        #
        #elif model.mtype == 'pauli-clifford':
        #    out = _sim.pauliclifford_simulator(circuit, model, inputstate=inputstate, store=store, returnall=returnall)
        #
        #else:
        #    raise ValueError, "Model type not understood by the simulator."
        #
        #return out
    
        return _sim.simulate(circuit,model[modelname],inputstate=inputstate,store=store,returnall=returnall)

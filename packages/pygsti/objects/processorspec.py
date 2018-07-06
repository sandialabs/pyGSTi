""" Defines the ProcessorSpec class and supporting functionality."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import itertools as _itertools
import collections as _collections
from scipy.sparse.csgraph import floyd_warshall as _fw

from .compilationlibrary import CompilationLibrary as _CompilationLibrary
from .compilationlibrary import CompilationError as _CompilationError
from .qubitgraph import QubitGraph as _QubitGraph
from ..baseobjs import Label as _Label
from . import gate as _gate

class ProcessorSpec(object):
    """
    An object that can be used to encapsulate the device specification for a one or more qubit 
    quantum computer. This is objected is geared towards multi-qubit devices; many of the contained 
    structures are superfluous in the case of a single qubit.

    """
    def __init__(self, nQubits, gate_names, nonstd_gate_unitaries={}, availability={}, 
                qubit_labels=None, construct_models=('clifford','target'), 
                construct_clifford_compilations = {'paulieq' : ('1Qcliffords','cnots'), 
                'absolute': ('paulis',)}, verbosity=0):
        """
        Initializes a ProcessorSpec object.
    
        The most basic information required for a ProcessorSpec object is the number of qubits in the 
        device, and the library of "native" target gates, which are specified as unitary matrices
        (using `nonstd_gate_unitaries`) or using default gate-names known to pyGSTi, such as 'Gcnot'
        for the CNOT gate (using `gate_names`). The other core information is the availability of the
        gates (specified via `availability`).
        
        Parameters
        ----------
        nQubits : int
            The number of qubits in the device.

        gate_names : list of strings
            The names of gates in the device that correspond to standard gates known by pyGSTi. 
            The set of standard gates includes, but is not limited to:

            - 'Gi' : the 1Q idle operation
            - 'Gx','Gy','Gz' : 1-qubit pi/2 rotations
            - 'Gxpi','Gypi','Gzpi' : 1-qubit pi rotations
            - 'Gh' : Hadamard
            - 'Gp' : phase or S-gate (i.e., ((1,0),(0,i)))
            - 'Gcphase','Gcnot','Gswap' : standard 2-qubit gates

            Alternative names can be used for all or any of these gates, but then they must be explicitly defined 
            in the `nonstd_gate_unitaries` dictionary.
        
        nonstd_gate_unitaries: dictionary of numpy arrays
            A dictionary with keys that are gate names (strings) and values that are numpy arrays specifying 
            quantum gates in terms of unitary matrices. Together with the gates specified in the `gate_names`
            list, these matrices specify the set of all (target) native gates that can be implemented in the device.
            as unitaries acting on ordinary pure-state-vectors, in the standard computationl basis. These unitaries 
            need not, and often should not, be unitaries acting on all of the qubits. E.g., a CNOT gate is specified 
            by a key that is the desired name for CNOT, and a value that is the standard 4 x 4 complex matrix for CNOT.
            It is advisable, although not essential, to use gate names that start with 'G'.
        
        availability : dict, optional
            A dictionary whose keys are some subset of the keys (which are gate names) `nonstd_gate_unitaries` 
            and the strings (which are gate names) in `gate_names` and whose values are lists of qubit-label-tuples.  
            Each qubit-label-tuple must have length equal to the number of qubits the corresponding gate acts upon, 
            and causes that gate to be available to act on the specified qubits. Instead of a list of tuples, values 
            of `availability` may take the special values `"all-permutations"` and `"all-combinations"`, which as their names
            imply, equate to all possible permutations and combinations of the appropriate number of qubit labels (deterined 
            by the gate's dimension). If a gate name is not present in `availability`, the default is `"all-permutations"`.
            So, the availability of a gate only needs to be specified when it cannot act in every valid way on the qubits
            (e.g., the device does not have all-to-all connectivity).

        construct_models : tuple, optional


        TODO: docstring
        """
        assert(type(nQubits) is int), "The number of qubits, n, should be an integer!"

        #Store inputs for adding models later
        self.number_of_qubits = nQubits
        self.root_gate_names = gate_names
        self.nonstd_gate_unitaries = nonstd_gate_unitaries.copy()
        self.root_gate_names += list(self.nonstd_gate_unitaries.keys())    
        self.availability = availability.copy()

        # If no qubit labels are provided it defaults to integers from 0 to nQubits-1.    
        if qubit_labels is None:
            self.qubit_labels = list(range(nQubits)) 
        else:
            assert(len(qubit_labels) == nQubits) 
            self.qubit_labels = list(qubit_labels)

        # A dictionary of models for the device (e.g., imperfect unitaries, process matrices etc).
        self.models = _collections.OrderedDict()

        # Compilations are stored here.
        self.compilations = _collections.OrderedDict()

        #Compiler-cost variables (set in construct_compiler_costs)
        self.qubitgraph = None
        # todo : delete these if not using them.
        # self.qubitcosts = None
        # self.costorderedqubits = None
        #self.connectivity['clifford'] =
        #self.compiler

        self.oneQgate_algebra = {}
        self.oneQgate_inversions = {}

        # Add initial models
        for model_name in construct_models:
            self.add_std_model(model_name)

        
        if 'clifford' in construct_models:

            # Add compilations for the "basic" Clifford gates specified, which are used for, e.g., Clifford compiler algorithms.
            # We only do this if there is a 'clifford' gateset, as these are Clifford compilations.
            for ctype in list(construct_clifford_compilations.keys()):
                
                # We construct the requested Pauli-equivalent compilations.
                if ctype == 'paulieq':
                    oneQgates = [] # A list of the 1-qubit gates to compile, in the std names understood inside the compilers.
                    twoQgates = [] # A list of the 2-qubit gates to compile, in the std names understood inside the compilers.
                    for subctype in construct_clifford_compilations[ctype]:
                        if subctype == '1Qcliffords':
                            oneQgates += ['H','P','PH','HP','HPH']
                        elif subctype == 'cnots':
                            twoQgates += ['CNOT']
                        else:
                            raise ValueError("One of the values for the key `" + ctype + "` to `construct_clifford_compilations` is not a valid option!")
                    self.add_std_compilations(ctype, oneQgates, twoQgates, verbosity)

                # We construct the requested `absolute` (i.e., not only up to Paulis) compilations.
                elif ctype == 'absolute':
                    oneQgates = [] # A list of the 1-qubit gates to compile, in the std names understood inside the compilers.
                    twoQgates = [] # A list of the 2-qubit gates to compile, in the std names understood inside the compilers.
                    for subctype in construct_clifford_compilations[ctype]:
                        if subctype == 'paulis':
                            oneQgates += ['I','X','Y','Z']
                        elif subctype == '1Qcliffords':
                            oneQgates += ['C'+str(q) for q in range(24)] # todo : implement this.
                            raise ValueError("This is not yet implemented")
                        else:
                            raise ValueError("One of the values for the key `" + ctype + "` to `construct_clifford_compilations` is not a valid option!")
                    self.add_std_compilations(ctype, oneQgates, twoQgates, verbosity)

                else:
                    raise ValueError("The keys to `construct_clifford_compilations` can only be `paulieq` and `absolute!")

            # Generates the QubitGraph for the multi-qubit Clifford gates. If there are multi-qubit gates which are not Clifford gates
            # then these are not counted as "connections".
            connectivity = _np.zeros((self.number_of_qubits,self.number_of_qubits),dtype=bool)
            for gatelabel in self.models['clifford'].gates:
                # This treats non-entangling 2-qubit gates as making qubits connected. Stopping that is
                # something we may need to do at some point.
                if gatelabel.number_of_qubits > 1:
                    for p in _itertools.permutations(gatelabel.qubits, 2):
                        connectivity[self.qubit_labels.index(p[0]),self.qubit_labels.index(p[1])] = True
            
            self.qubitgraph = _QubitGraph(list(range(self.number_of_qubits)), connectivity)

        # todo : store this in a less clumsy way.
        if 'clifford' in self.models:
            # Compute the gate labels that act on an entire set of qubits
            self.clifford_gates_on_qubits =  _collections.defaultdict(list)
            for gl in self.models['clifford'].gates:
                for p in _itertools.permutations(gl.qubits):
                    self.clifford_gates_on_qubits[p].append(gl)
        else:
            self.clifford_gates_on_qubits = None
                                  
        return # done with __init__(...)
               
    def add_std_model(self, model_name, parameterization='auto', sim_type='auto'):
        """ 
        Erik todo: docstring 

        """
        from .. import construction as _cnst
        
        if model_name == 'clifford':
            assert(parameterization in ('auto','clifford')), "Clifford model must use 'clifford' parameterizations"
            assert(sim_type in ('auto','map')), "Clifford model must use 'map' simulation type"
            model = _cnst.build_nqubit_standard_gateset(
                self.number_of_qubits, self.root_gate_names,
                self.nonstd_gate_unitaries, self.availability, self.qubit_labels,
                parameterization='clifford', sim_type=sim_type,
                on_construction_error='warn') # *drop* gates that aren't cliffords

        elif model_name in ('target','Target','static','TP','full'):
            param = model_name if (parameterization == 'auto') \
                    else parameterization
            if param in ('target','Target'): param = 'static' # special case for 'target' model
            
            model = _cnst.build_nqubit_standard_gateset(
                self.number_of_qubits, self.root_gate_names,
                self.nonstd_gate_unitaries, self.availability, self.qubit_labels,
                param, sim_type)
            
        else: # unknown model name, so require parameterization
            if parameterization == 'auto':
                raise ValueError("Non-std model name '%s' means you must specify `parameterization` argument!" % model_name)
            model = _cnst.build_nqubit_standard_gateset(
                self.number_of_qubits, self.root_gate_names,
                self.nonstd_gate_unitaries, self.availability, self.qubit_labels,
                parameterization, sim_type)
            
        self.models[model_name] = model
                  
    def add_std_compilations(self, compile_type, oneQgates, twoQgates, verbosity=0):
        """ 
        Tim todo : docstring
        
        """
        #Hard-coded gates we need to compile from the native (clifford) gates in order to compile
        # arbitrary (e.g. random) cliffords, since our Clifford compiler outputs circuits in terms
        # of these elements.
        descs = {'paulieq': 'up to paulis', 'absolute':''}
        
        if 'clifford' not in self.models:
            raise ValueError("Cannot create standard compilations without a 'clifford' model")
        library = _CompilationLibrary(self.models['clifford'], compile_type) # empty library to fill
        desc = descs[compile_type]

        #Stage1:
        # Compile 1Q gates "locally" - i.e., out of native gates which act only
        #  on the target qubit of the gate being compiled.
        for q in self.qubit_labels:
            for gname in oneQgates:
                if verbosity > 0:
                    print("Creating a circuit to implement {} {} on qubit {}...".format(gname,desc,q))
                library.add_local_compilation_of(_Label(gname,q), verbosity=verbosity)

            if verbosity > 0: print("Complete.")

        for gate in twoQgates:
            assert(gate == 'CNOT'), "Only CNOT compilations are currently possible"
        
            if compile_type == 'paulieq':
                #print('Adding in the standard CNOT compilations')
                if 'Gcnot' in self.root_gate_names:
                    library.templates['CNOT'] = [(_Label('Gcnot',(0,1)),)]
                    if 'Gh' in self.root_gate_names:
                        library.templates['CNOT'].append((_Label('Gh', 0),_Label('Gh', 1),_Label('Gcnot', (1, 0)), _Label('Gh', 0),_Label('Gh', 1)))
                    else:
                        for gate in self.models['clifford'].gates:
                            if (isinstance(self.models['clifford'][gate], _gate.EmbeddedGateMap) and 
                                    _np.array_equal(self.models['clifford'][gate].embedded_gate.smatrix,_np.array([[0,1],[1,0]]))) or \
                                   (isinstance(self.models['clifford'][gate], _gate.CliffordGate) and
                                    _np.array_equal(self.models['clifford'][gate].smatrix,_np.array([[0,1],[1,0]]))):
                                #Note: use of gate.name assumes that a Hadamard will have a simple (non-"parallel") gate label
                                library.templates['CNOT'].append( (_Label(gate.name, 0),_Label(gate.name, 1),_Label('Gcnot', (1, 0)),
                                                                       _Label(gate.name, 0),_Label(gate.name, 1)))
                                #print('Hadamard or an equivalent gate found! It is ' + gate.name)
                                break
                if 'Gcphase' in self.root_gate_names:
                    library.templates['CNOT'] = [(_Label('Gcnot',(0,1)),)]
                    if 'Gh' in self.root_gate_names:
                        library.templates['CNOT'].append((_Label('Gi', 0),_Label('Gh', 1),_Label('Gcphase', (0, 1)), _Label('Gi', 0),_Label('Gh', 1)))
                        library.templates['CNOT'].append((_Label('Gi', 0),_Label('Gh', 1),_Label('Gcphase', (1,0)), _Label('Gi', 0),_Label('Gh', 1)))

                    else:
                        for gate in self.models['clifford'].gates:
                            if (isinstance(self.models['clifford'][gate], _gate.EmbeddedGateMap) and 
                                    _np.array_equal(self.models['clifford'][gate].embedded_gate.smatrix,_np.array([[0,1],[1,0]]))) or \
                                   (isinstance(self.models['clifford'][gate], _gate.CliffordGate) and
                                    _np.array_equal(self.models['clifford'][gate].smatrix,_np.array([[0,1],[1,0]]))):
                                #Note: use of gate.name assumes that a Hadamard will have a simple (non-"parallel") gate label
                                library.templates['CNOT'].append((_Label('Gi', 0),_Label(gate.name, 1),_Label('Gcphase', (0, 1)), _Label('Gi', 0),_Label(gate.name, 1)))
                                library.templates['CNOT'].append((_Label('Gi', 0),_Label(gate.name, 1),_Label('Gcphase', (1,0)), _Label('Gi', 0),_Label(gate.name, 1)))
                                #print('Hadamard or an equivalent gate found! It is ' + gate.name)

                              
            #Stage2:
            # Compile 2Q gates locally, if possible.  Keep track of what can't be compiled.
            not_locally_compilable = []
            
            for q1 in self.qubit_labels:
                for q2 in self.qubit_labels:
                    if q1 == q2: continue # 2Q gates must be on different qubits!
                    for gname in twoQgates:
                        if verbosity > 0:
                            print("Creating a circuit to implement {} {} on qubits {}...".format(gname,desc,(q1,q2)))
                        try:
                            library.add_local_compilation_of(
                                _Label(gname,(q1,q2)), verbosity=verbosity)
                        except _CompilationError:
                            not_locally_compilable.append( (gname,q1,q2) )
                            
            #Stage3:
            # Try to compile remaining 2Q gates non-locally using specific algorithms
            non_compilable = []
            for gname,q1,q2 in not_locally_compilable:
                library.add_nonlocal_compilation_of(_Label(gname,(q1,q2)),
                                                    verbosity=verbosity)
        
        self.compilations[compile_type] = library


    def add_oneQgate_algebra(modelname):
        # todo
        return

    def add_oneQgate_inversions(modelname):
        # todo
        return


    # For finding the inversion dict.
# inverse_dict = {}
# for gl1 in pspec.models['target'].gates:
#     gate1 = pspec.models['target'][gl1]
#     gate1 = gate1.convert_to_matrix(gate1.embedded_gate)
#     for gl2 in pspec.models['target'].gates:
#         gate2 = pspec.models['target'][gl2]
#         gate2 = gate2.convert_to_matrix(gate2.embedded_gate)
#         #print(gate1,gate2)
#         if gl1.number_of_qubits == gl2.number_of_qubits:
#             #print(np.dot(gate1,gate2))
#             if np.allclose(np.dot(gate1,gate2),np.identity(4**gl1.number_of_qubits,float)):
#                 print(gl1,gl2)
    
    # Tim is going to replace this at some point with a useful way to specify how "costly" using different qubits/gates is estimated to be, so that
    # Clifford compilers etc can take this into account.                        
    # def construct_compiler_costs(self, custom_connectivity=None):
    #     """ 

    #     """
    #     self.qubitcosts = {}
        
    #     #
    #     # todo -- I'm not sure whether this makes sense when the graph is directed.
    #     #
    #     distances = self.qubitgraph.shortest_path_distance_matrix()
    #     for i in range(0,self.number_of_qubits):
    #         self.qubitcosts[i] = _np.sum(distances[i,:])
        
    #     temp_distances = list(_np.sum(distances,0))
    #     temp_qubits = list(_np.arange(0,self.number_of_qubits))
    #     self.costorderedqubits = []
               
    #     while len(temp_distances) > 0:
            
    #         longest_remaining_distance = max(temp_distances)
    #         qubits_at_this_distance = []
            
    #         while longest_remaining_distance == max(temp_distances):
                
    #             index = _np.argmax(temp_distances)
    #             qubits_at_this_distance.append(temp_qubits[index])
    #             del temp_distances[index]
    #             del temp_qubits[index]
                
    #             if len(temp_distances) == 0:
    #                 break
        
    #         self.costorderedqubits.append(qubits_at_this_distance)
            
            
    def simulate(self,circuit,modelname):
        """
        TODO: docstring
        A wrap-around for the circuit simulators in simulators.py 
        """       
        return self.models[modelname].probs(circuit)

""" Clifford compilation routines """
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

def compile_clifford(s, p, pspec=None, subsetQs=None, iterations=20, algorithm='ROGGE', aargs = [],
                     costfunction='2QGC', prefixpaulis=False, paulirandomize=False):
    """
    Compiles an n-qubit Clifford gate, described by the symplectic matrix s and vector p, into
    a circuit over the specified gateset, or, a standard gateset. Clifford gates/circuits can be converted
    to, or sampled in, the symplectic representation using the functions in pygsti.tools.symplectic.

    This circuit will be over a user-specified gateset and respecting any desired connectivity, if a 
    ProcessorSpec object is provided. Otherwise, it is over a canonical gateset containing all-to-all CNOTs, 
    Hadamard, Phase, 3 products of Hadamard and Phase, and the Pauli gates.
        
    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers.
    
    p : array over [0,1]
        A length-2n vector over [0,1,2,3] that, together with s, defines a valid n-qubit Clifford
        gate.
        
    pspec : ProcessorSpec, optional
        An nbar-qubit ProcessorSpec object that encodes the device that the Clifford is being compiled
        for, where nbar >= n. If this is specified, the output circuit is over the gates available 
        in this device. If this is None, the output circuit is over the "canonical" gateset of CNOT gates 
        between all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation. In most circumstances, the output will be much more
        useful if a ProcessorSpec is provided.

        If nbar > n it is necessary to provide `subsetQs`, that specifies which of the qubits in `pspec`
        the Clifford acts on (all other qubits will not be part of the returned circuit, regardless of 
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input Clifford needs to be ``padded'' to be the identity
        on those qubits).

        The ordering of the qubits in (`s`,`p`) is assumed to be the same as that in the list pspec.qubit_labels, 
        unless `subsetQs` is specified. Then, the ordering is taken w.r.t the ordering of the list `subsetQs`.

    subsetQs : List, optional
        Required if the Clifford to compile is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of pspec.qubit_labels.
        The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.
        
    iterations : int, optional
        Some of the allowed algorithms are randomized. This is the number of iterations used in
        algorithm if it is a randomized algorithm specified. If any randomized algorithms are specified,
        the time taken by this function increases linearly with `iterations`. Increasing `iterations`
        will often improve the obtained compilation (the "cost" of the obtained circuit, as specified
        by `costfunction` may decrease towards some asymptotic value).
        
    algorithm : str, optional
        Specifies the algorithm used for the core part of the compilation: finding a circuit that is a Clifford
        with `s` the symplectic matrix in it's symplectic representation (a circuit that implements that desired
        Clifford up to Pauli operators). The allowed values of this are:

        - 'BGGE' A basic, deterministic global Gaussian elimination algorithm. Circuits obtained from this algorithm 
           contain, in expectation, O(n^2) 2-qubit gates. Although the returned circuit will respect device 
           connectivity, this algorithm does *not* take connectivity into account in an intelligient way. More details
           on this algorithm are given in `compile_symplectic_with_ordered_global_gaussian_elimination()`; it is the
           algorithm described in that docstring but with the qubit ordering fixed to the order in the input `s`.

        - 'ROGGE A randomized elimination order global Gaussian elimination algorithm. This is essentially the
           same algorithm as 'BGGE' except that the order that the qubits are eliminated in is randomized. This
           results in significantly lower-cost circuits than the 'BGGE' method (given sufficiently many iterations).
           More details are given in the `compile_symplectic_with_random_ordered_global_gaussian_elimination()` docstring.

    aargs : list, optional
        If the algorithm can take optional arguments, not already specified as separate arguments above, then 
        this list is passed to the compile_symplectic algorithm as its final arguments.

    costfunction : function or string, optional
        If a function, it is a function that takes a circuit and `pspec` as the first and second inputs and 
        returns a cost (a float) for the circuit. The circuit input to this function will be over the gates in
        `pspec`, if a `pspec` has been provided, and as described above if not. This costfunction is used to decide 
        between different compilations when randomized algorithms are used: the lowest cost circuit is chosen. If 
        a string it must be one of:

            - '2QGC' : the cost of the circuit is the number of 2-qubit gates it contains.
            - 'depth' : the cost of the circuit is the depth of the circuit.

    prefixpauli : bool, optional
        The circuits are constructed by finding a circuit that implements the correct Clifford up to Pauli
        operators, and then attaching a (compiled) Pauli layer to the start or end of this circuit. If this
        bool is True (False) the Pauli layer is (pre-) post-fixed.

    paulirandomize : bool, optional
        If True then independent, uniformly random Pauli layers (a Pauli on each qubit) are inserted in between
        every layer in the circuit. These Paulis are then compiled into the gates in `pspec`, if `pspec` is provided.
        That is, this Pauli-frame-randomizes / Pauli-twirls the internal layers of this Clifford circuit. This can
        be useful for preventing coherent addition of errors in the circuit.

    Returns
    -------
    Circuit
        A circuit implementing the input Clifford gate/circuit.      
    """
    assert(_symp.check_valid_clifford(s,p)), "Input is not a valid Clifford!"
    n = _np.shape(s)[0]//2
    
    if pspec is not None:
        if subsetQs is None:
            assert(pspec.number_of_qubits == n), "If all the qubits in `pspec` are to be used, the Clifford must be over all {} qubits!".format(pspec.number_of_qubits)
            qubit_labels = pspec.qubit_labels
        else:
            assert(len(subsetQs) == n), "The subset of qubits to compile for is the wrong size for this CLifford!!" 
            qubit_labels = subsetQs
    else:
        assert(subsetQs == None), "subsetQs can only be specified if `pspec` is not None!"
        qubit_labels = list(range(n))
    
    # Create a circuit that implements a Clifford with symplectic matrix s.
    circuit = compile_symplectic(s, pspec=pspec, subsetQs=subsetQs, iterations=iterations, algorithms=[algorithm,],
                                 costfunction=costfunction, paulirandomize=paulirandomize, aargs={'algorithm':aargs}, 
                                 check=False)

    temp_s, temp_p = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
    
    # Find the necessary Pauli layer to compile the correct Clifford, not just the correct
    # Clifford up to Paulis. The required Pauli layer depends on whether we pre-fix or post-fix it.
    if prefixpaulis:        
        pauli_layer = _symp.find_premultipled_pauli(s,temp_p,p,qubit_labels=qubit_labels)
    else:
        pauli_layer = _symp.find_postmultipled_pauli(s,temp_p,p,qubit_labels=qubit_labels)    
    # Turn the Pauli layer into a circuit.
    pauli_circuit = _Circuit(gatestring=pauli_layer, line_labels=qubit_labels, identity='I')   
    # Only change gate library of the Pauli circuit if we have a ProcessorSpec with compilations.    
    if pspec is not None:
        pauli_circuit.change_gate_library(pspec.compilations['absolute'], identity=pspec.identity, oneQgate_relations=pspec.oneQgate_relations)    
    # Prefix or post-fix the Pauli circuit to the main symplectic-generating circuit.
    if prefixpaulis:
        circuit.prefix_circuit(pauli_circuit)
    else:
        circuit.append_circuit(pauli_circuit)
    
    # Check that the correct Clifford has been compiled. This should never fail.
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)    
    assert(_symp.check_valid_clifford(s_out,p_out))
    assert(_np.array_equal(s,s_out)) 
    assert(_np.array_equal(p,p_out)) 
    
    return circuit

def create_standard_cost_function(name):
    """
    Creates the standard costfunctions from an input string, used in the 
    Clifford compilation algorithms.

    Parameters
    ----------
    name : str
        Allowed values are:
            - '2QGC' : the cost of the circuit is the number of 2-qubit gates it contains.
            - 'depth' : the cost of the circuit is the depth of the circuit.
    Returns
    -------
    function
        A function that takes a circuit as the first arguement, a ProcessorSpec as the second
        argument (or a "junk" input when a ProcessorSpec is not needed), and returns the cost
        of the circuit.
    """
    if name == '2QGC':
        def costfunction(circuit, junk):
            return circuit.twoqubit_gatecount()
    elif name == 'depth':
        def costfunction(circuit, junk):
            return circuit.depth()
    #elif costfunction == 'gcosts':
    #        # Todo : make this work - .gatecosts is currently not a property of a DeviceSpec object,
    #        # and circuit.cost() is currently not a method of circuit.
    #        c_cost = circuit.cost(pspec.gatecosts)
    else: raise ValueError("This `costfunction` string is not a valid option!")
    return costfunction

def compile_symplectic(s, pspec=None, subsetQs=None, iterations=20, algorithms=['DGGE','RoGGE'], 
                       costfunction='2QGC', paulirandomize=False, aargs={}, check=True):    
    """
    Tim todo : docstring.

    # The cost function should take two arguments. (the 2nd one is a pspec.)

     iterations : int, optional
        Some of the allowed algorithms are randomized. This is the number of iterations used in each of
        those algorithm that is a randomized.
    """
    # The number of qubits the symplectic matrix is on. 
    n = _np.shape(s)[0]//2 
    if pspec is not None:
        if subsetQs is None:
            assert(pspec.number_of_qubits == n), "If all the qubits in `pspec` are to be used, `s` must be a symplectic matrix over {} qubits!".format(pspec.number_of_qubits)
        else:
            assert(len(subsetQs) == n), "The subset of qubits to compile `s` for is the wrong size for this symplectic matrix!" 
    else:
        assert(subsetQs == None), "subsetQs can only be specified if `pspec` is not None!"

    all_algorithms = ['BGGE','ROGGE','AGvGE','AGvPMH','iAGvGE','iAGvPMH']
    assert(set(algorithms).issubset(set(all_algorithms))), "One or more algorithms names are invalid!"

    # A list to hold the compiled circuits, from which we'll choose the best one. Each algorithm
    # only returns 1 circuit, so this will have the same length as the `algorithms` list.
    circuits = []

    # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str):
        costfunction = create_standard_cost_function(costfunction)
        
    # Deterministic basic global Gaussian elimination
    if 'BGGE' in algorithms:       
        if subsetQs is not None:
            eliminationorder = list(range(len(subsetQs)))
        elif pspec is not None:
            eliminationorder = list(range(len(pspec.qubit_labels)))
        else:
            eliminationorder = list(range(n))
        circuit = compile_symplectic_with_ordered_global_gaussian_elimination(s, eliminationorder=eliminationorder, pspec=pspec, subsetQs=subsetQs, 
                                                      ctype='basic', check=False)
        circuits.append(circuit)  

    # if 'COGGE' in algorithms:
    #     circuit = ordered_global_gaussian_elimination(s, pspec=pspec, subsetQs=subsetQs, 
    #                                                   iterations=1, ctype='basic', check=False)
    #     circuits.append(circuit) 
    
    # Randomized basic global Gaussian elimination, whereby the order that the qubits are eliminated in
    # is randomized.
    if 'ROGGE' in algorithms:
        circuit = compile_symplectic_with_random_ordered_global_gaussian_elimination(s, pspec=pspec, subsetQs=subsetQs, ctype='basic', 
                                                        costfunction=costfunction, iterations = iterations, check=False) 
        circuits.append(circuit) 
        
    # The Aaraonson-Gottesman method for compiling a symplectic matrix using 5 CNOT circuits + local layers,
    # with the CNOT circuits compiled using global Gaussian elimination.
    if 'AGvGE' in algorithms:       
        circuit = compile_symplectic_with_aaronson_gottesman_algorithm(s, pspec=pspec, subsetQs=subsetQs, cnotmethod='GE', check=False)  
        circuits.append(circuit) 
        
    if 'AGvPMH' in algorithms:
        circuit = compile_symplectic_with_aaronson_gottesman_algorithm(s, pspec=pspec, subsetQs=subsetQs, cnotmethod = 'PMH', check=False)   
        circuits.append(circuit) 
        
    if 'iAGvGE' in algorithms:
        circuit = compile_symplectic_with_improved_aaronson_gottesman_algorithm(s, pspec=pspec, subsetQs=subsetQs, cnotmethod = 'GE', check=False) 
        circuits.append(circuit) 
        
    if 'iAGvPMH' in algorithms:
        circuit = compile_symplectic_with_improved_aaronson_gottesman_algorithm(s, pspec=pspec, subsetQs=subsetQs, cnotmethod = 'PMH', check=False) 
        circuits.append(circuit) 
    
    # If multiple algorithms have be called, find the lowest cost circuit.
    if len(circuits) > 1:
        bestcost = _np.inf
        for c in circuits:
            c_cost = costfunction(c,pspec)  
            if c_cost < bestcost:
                circuit = c.copy()
                cost = bestcost                 
    else:
        circuit = circuits[0]

    # At this point we set subsetQs to be the full list of qubits, for re-labelling purposes below.
    if pspec is not None:
        if subsetQs is None:
            subsetQs = pspec.qubit_labels
    
    # If we want to Pauli randomize the circuits, we insert a random compiled Pauli layer between every layer.
    if paulirandomize:     
        paulilist = ['I','X','Y','Z']
        d = circuit.depth()
        for i in range(0,d+1):
            pcircuit = _Circuit(gatestring=[_Label(paulilist[_np.random.randint(4)],k) for k in range(n)],num_lines=n,identity='I')
            if pspec is not None:
                # Map the circuit to the correct qubit labels
                if subsetQs is not None:
                    pcircuit.map_state_space_labels({i:subsetQs[i] for i in range(n)})
                else:
                    pcircuit.map_state_space_labels({i:pspec.qubit_labels[i] for i in range(n)})
                # Compile the circuit into the native gateset, using an "absolute" compilation -- Pauli-equivalent is
                # not sufficient here.
                pcircuit.change_gate_library(pspec.compilations['absolute'], identity=pspec.identity, oneQgate_relations=pspec.oneQgate_relations)   
            circuit.insert_circuit(pcircuit,d-i)

    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit,pspec=pspec)            
        assert(_np.array_equal(s,implemented_s)) 
            
    return circuit

def compile_symplectic_with_random_ordered_global_gaussian_elimination(s, pspec=None, subsetQs=None, ctype='basic', costfunction='2QGC', 
                                           iterations=10, check=True):
    """
    Tim todo : docstring.

    costfunction should be a function, taking a circuit + pspec. NO!!!!

    Note that it is better to use the wrap-around, as these algorithms don't check some consistency things.
    """   
    # The number of qubits the symplectic matrix is on. 
    n = _np.shape(s)[0]//2   
    # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str):
        costfunction = create_standard_cost_function(costfunction)

    # The elimination order in terms of qubit *index*, which is randomized below.
    if subsetQs is not None:
            eliminationorder = list(range(len(subsetQs)))
    elif pspec is not None:
        eliminationorder = list(range(len(pspec.qubit_labels))) 
    else:
        eliminationorder = list(range(n))
    
    lowestcost = _np.inf    
    for i in range(0,iterations):
        
        # Pick a random order to attempt the elimination in
        _np.random.shuffle(eliminationorder)
        # Call the re-ordered global Gaussian elimination, which is wrap-around for the GE algorithms to deal
        # with qubit relabeling. Check is False avoids multiple checks of success, when only the last check matters.
        circuit = compile_symplectic_with_ordered_global_gaussian_elimination(s, eliminationorder, pspec=pspec, subsetQs=subsetQs, ctype=ctype, check=False)          
        # Find the cost of the circuit, and keep it if this circuit is the lowest-cost circuit so far.
        circuit_cost = costfunction(circuit, pspec)                       
        if circuit_cost < lowestcost:
            bestcircuit = circuit.copy()
            lowestcost = circuit_cost 

    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(bestcircuit,pspec=pspec)            
        assert(_np.array_equal(s,implemented_s)) 
    
    return bestcircuit
  
def compile_symplectic_with_ordered_global_gaussian_elimination(s, eliminationorder, pspec=None, subsetQs=None, ctype='basic', check=True):
    """
    todo : doctstring
    """
    
    # Re-order the s matrix to reflect the order we want to eliminate the qubits in,
    # because we hand the symp. matrix to a function that eliminates them in a fixed order.
    n = _np.shape(s)[0]//2
    P = _np.zeros((n,n),int)
    for j in range(0,n):
        P[j,eliminationorder[j]] = 1        
    P2n = _np.zeros((2*n,2*n),int)
    P2n[0:n,0:n] = P
    P2n[n:2*n,n:2*n] = P           
    permuted_s = _mtx.dotmod2(_mtx.dotmod2(P2n,s),_np.transpose(P2n))
               
    if ctype == 'basic':
        # Check is False avoids multiple checks of success, when only the last check matters.
        circuit = compile_symplectic_with_basic_global_gaussian_elimination(permuted_s, check=False)
    else: raise ValueError("The compilation sub-method is not valid!")
        # The plan is to write the ctype == 'advanced':
        
    # If the subsetQs is not None, we relabel the circuit in terms of the labels of these qubits.
    if subsetQs is not None:
        assert(len(eliminationorder) == len(subsetQs)), "`subsetQs` must be the same length as `elimintionorder`! The mapping to qubit labels is ambigiuous!"
        circuit.map_state_space_labels({i:subsetQs[eliminationorder[i]] for i in range(n)})
        circuit.reorder_wires(subsetQs)
    # If the subsetQs is None, but there is a pspec, we relabel the circuit in terms of the full set
    # of pspec labels.
    elif pspec is not None:
        assert(len(eliminationorder) == len(pspec.qubit_labels)), "If `subsetQs` is not specified `s` should be over all the qubits in `pspec`!"
        circuit.map_state_space_labels({i:pspec.qubit_labels[eliminationorder[i]] for i in range(n)})
        circuit.reorder_wires(pspec.qubit_labels)
    else:
        circuit.map_state_space_labels({i:eliminationorder[i] for i in range(n)})
        circuit.reorder_wires(list(range(n)))

    # If we have a pspec, we change the gate library. We use a pauli-equivalent compilation, as it is
    # only necessary to implement each gate in this circuit up to Pauli matrices.
    if pspec is not None:
        if subsetQs is None:
            circuit.change_gate_library(pspec.compilations['paulieq'],identity=pspec.identity, oneQgate_relations=pspec.oneQgate_relations)
        else:
            circuit.change_gate_library(pspec.compilations['paulieq'], allowed_filter=set(subsetQs), identity=pspec.identity, 
                                        oneQgate_relations=pspec.oneQgate_relations)
    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit,pspec=pspec)            
        assert(_np.array_equal(s,implemented_s)) 

    return circuit
      
def compile_symplectic_with_basic_global_gaussian_elimination(s, check=True):
    """
    Creates a circuit over 'I','H','HP','PH','HPH', and 'CNOT' that implements a Clifford
    gate with `s` as its symplectic matrix in the symplectic representation (and with any
    phase vector). This circuit is generated using a basic Gaussian elimination algorithm.

    Todo : improve this docstring.

    This algorithm is more conveniently accessed via the `compile_symplectic()`
    or `compile_clifford()` functions.

    s: array
        A 2n X 2n symplectic matrix over [0,1] for any positive integer n. The returned
        circuit is over n qubits. (a 2n x 2n symplectic matrix is a [partial] rep of an
        n-qubit Clifford gate).

    check : bool, optional
        Whether to check that the generated circuit does implement `s`. This is set to 
        False when
        
    Returns
    -------
    Circuit
       A circuit that implements a Clifford with the input symplectic matrix.   
    """
    current_s = _np.copy(s) # Copy so that we don't change the input s.      
    n = _np.shape(s)[0]//2
    
    assert(_symp.check_symplectic(s,convention='standard')), "The input matrix must be symplectic!"
    
    instruction_list = []
    # Map the portion of the symplectic matrix acting on qubit j to the identity, for j = 0,...,d-1 in
    # turn, using the basic row operations corresponding to the CNOT, Hadamard, phase, and SWAP gates.    
    for j in range (n):
        
        # *** Step 1: Set the upper half of column j to the relevant identity column ***
        upperl_c = current_s[:n,j]
        lowerl_c = current_s[n:,j]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
        
        # If the jth element in the column is not 1, it needs to be set to 1.
        if j not in upperl_ones:
            
            # First try using a Hadamard gate.
            if j in lowerl_ones:
                instruction_list.append(_Label('H',j))                
                _symp.apply_internal_gate_to_symplectic(current_s, 'H', [j,])
                
            # Then try using a swap gate, we don't try and find the best qubit to swap with.
            elif len(upperl_ones) >= 1:
                instruction_list.append(_Label('CNOT',[j,upperl_ones[0]]))
                instruction_list.append(_Label('CNOT',[upperl_ones[0],j]))
                instruction_list.append(_Label('CNOT',[j,upperl_ones[0]]))
                _symp.apply_internal_gate_to_symplectic(current_s, 'SWAP', [j,upperl_ones[0]])
                
            # Finally, try using swap and Hadamard gates, we don't try and find the best qubit to swap with.
            else:
                instruction_list.append(_Label('H',lowerl_ones[0]))
                _symp.apply_internal_gate_to_symplectic(current_s, 'H', [lowerl_ones[0],])
                instruction_list.append(_Label('CNOT',[j,lowerl_ones[0]]))
                instruction_list.append(_Label('CNOT',[lowerl_ones[0],j]))
                instruction_list.append(_Label('CNOT',[j,lowerl_ones[0]]))
                _symp.apply_internal_gate_to_symplectic(current_s, 'SWAP', [j,lowerl_ones[0]])
            
            # Update the lists that keep track of where the 1s are in the column.
            upperl_c = current_s[:n,j]
            lowerl_c = current_s[n:,j]
            upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
            lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
            
        # Pair up qubits with 1s in the jth upper jth column, and set all but the
        # jth qubit to 0 in logarithmic depth. When there is an odd number of qubits
        # one of them is left out in the layer.
        while len(upperl_ones) >= 2: 

            num_pairs = len(upperl_ones)//2
            
            for i in range(0,num_pairs):                
                if upperl_ones[i+1] != j:
                    controlq = upperl_ones[i]
                    targetq = upperl_ones[i+1]
                    del upperl_ones[1+i]
                else:
                    controlq = upperl_ones[i+1]
                    targetq = upperl_ones[i]
                    del upperl_ones[i]
                    
                instruction_list.append(_Label('CNOT',(controlq,targetq)))                   
                _symp.apply_internal_gate_to_symplectic(current_s, 'CNOT', [controlq,targetq])

        # *** Step 2: Set the lower half of column j to all zeros ***
        upperl_c = current_s[:n,j]
        lowerl_c = current_s[n:,j]        
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
        
        # If the jth element in this lower column is 1, it must be set to 0.
        if j in lowerl_ones:
            instruction_list.append(_Label('P',j))
            _symp.apply_internal_gate_to_symplectic(current_s, 'P', [j,])
            
        # Move in the 1 from the upper part of the column, and use this to set all
        # other elements to 0, as in Step 1.
        instruction_list.append(_Label('H',j))
        _symp.apply_internal_gate_to_symplectic(current_s, 'H', [j,])
        
        upperl_c = None
        upperl_ones = None
        lowerl_c = current_s[n:,j]
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])

        while len(lowerl_ones) >= 2:

            num_pairs = len(lowerl_ones)//2
            
            for i in range(0,num_pairs):
                if lowerl_ones[i+1] != j:
                    controlq = lowerl_ones[i+1]
                    targetq = lowerl_ones[i]
                    del lowerl_ones[1+i]
                else:
                    controlq = lowerl_ones[i]
                    targetq = lowerl_ones[i+1]
                    del lowerl_ones[i]
                
                instruction_list.append(_Label('CNOT',(controlq,targetq)))                   
                _symp.apply_internal_gate_to_symplectic(current_s, 'CNOT', [controlq,targetq])
      
        # Move the 1 back to the upper column.
        instruction_list.append(_Label('H',j))
        _symp.apply_internal_gate_to_symplectic(current_s, 'H', [j,])
       
        # *** Step 3: Set the lower half of column j+d to the relevant identity column ***
        upperl_c = current_s[:n,j+n]
        lowerl_c = current_s[n:,j+n]      
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
             
        while len(lowerl_ones) >= 2:  

            num_pairs = len(lowerl_ones)//2
            
            for i in range(0,num_pairs):
                
                if lowerl_ones[i+1] != j:
                    controlq = lowerl_ones[i+1]
                    targetq = lowerl_ones[i]
                    del lowerl_ones[1+i]
                else:
                    controlq = lowerl_ones[i]
                    targetq = lowerl_ones[i+1]
                    del lowerl_ones[i]
                
                instruction_list.append(_Label('CNOT',(controlq,targetq)))                   
                _symp.apply_internal_gate_to_symplectic(current_s, 'CNOT', [controlq,targetq])
                
        # *** Step 4: Set the upper half of column j+d to all zeros ***
        upperl_c = current_s[:n,j+n]
        lowerl_c = current_s[n:,j+n]        
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])
        
        # If the jth element in the upper column is 1 it must be set to zero
        if j in upperl_ones:       
            instruction_list.append(_Label('H',j))
            _symp.apply_internal_gate_to_symplectic(current_s,'H',[j,])           
            instruction_list.append(_Label('P',j))
            _symp.apply_internal_gate_to_symplectic(current_s,'P',[j,])                
            instruction_list.append(_Label('H',j))
            _symp.apply_internal_gate_to_symplectic(current_s,'H',[j,])
        
        # Switch in the 1 from the lower column
        instruction_list.append(_Label('H',j))
        _symp.apply_internal_gate_to_symplectic(current_s,'H',[j,])
        
        upperl_c = current_s[:n,j+n]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_c = None
        lowerl_ones = None

        while len(upperl_ones) >= 2:        
            
            num_pairs = len(upperl_ones)//2
            
            for i in range(0,num_pairs):
                if upperl_ones[i+1] != j:
                    controlq = upperl_ones[i]
                    targetq = upperl_ones[i+1] 
                    del upperl_ones[1+i]
                else:
                    controlq = upperl_ones[i+1]
                    targetq = upperl_ones[i]                                                       
                    del upperl_ones[i]
                instruction_list.append(_Label('CNOT',(controlq,targetq)))                   
                _symp.apply_internal_gate_to_symplectic(current_s, 'CNOT', [controlq,targetq])

        # Switch the 1 back to the lower column
        instruction_list.append(_Label('H',j))            
        _symp.apply_internal_gate_to_symplectic(current_s,'H',[j], optype='row') 

        # If the matrix has been mapped to the identity, quit the loop as we are done.
        if _np.array_equal(current_s,_np.identity(2*n,int)):
            break
            
    assert(_np.array_equal(current_s,_np.identity(2*n,int))), "Compilation has failed!"          
    # Operations that are the same next to each other cancel, and this algorithm can have these. So
    # we go through and delete them.
    j = 1
    depth = len(instruction_list)
    while j < depth:
        
        if instruction_list[depth-j] == instruction_list[depth-j-1]:
            del instruction_list[depth-j]
            del instruction_list[depth-j-1]
            j = j + 2
        else:
            j = j + 1   
    # We turn the instruction list into a circuit over the internal gates.
    circuit = _Circuit(gatestring=instruction_list,num_lines=n,identity='I')
    # That circuit implements the inverse of s (it maps s to the identity). As all the gates in this
    # set are self-inverse (up to Pauli multiplication) we just reverse the circuit to get a circuit
    # for s.
    circuit.reverse()
    # To do the depth compression, we use the 1-qubit gate relations for the standard set of gates used
    # here.
    oneQgate_relations = _symp.oneQclifford_symplectic_group_relations()
    circuit.compress_depth(oneQgate_relations=oneQgate_relations, verbosity=0)
    # We check that the correct Clifford -- up to Pauli operators -- has been implemented.
    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit)            
        assert(_np.array_equal(s,implemented_s))        
        
    return circuit

def compile_symplectic_with_improved_aaronson_gottesman_algorithm(s, pspec=None, subsetQs=None, cnotmethod='PMH', check=False):
    raise NotImplementedError("This method is not yet written!")
    circuit = None
    return circuit

def compile_symplectic_with_improved_aaronson_gottesman_algorithm(s, pspec=None, subsetQs=None, cnotmethod='PMH', check=False):
    raise NotImplementedError("This method is not yet written!")
    circuit = None
    return circuit

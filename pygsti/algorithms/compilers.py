"""
Clifford circuit, CNOT circuit, and stabilizer state/measurement generation compilation routines
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import copy as _copy

from ..objects.circuit import Circuit as _Circuit
from ..objects.label import Label as _Label
from ..tools import symplectic as _symp
from ..tools import matrixmod2 as _mtx


def _create_standard_costfunction(name):
    """
    Creates the standard 'costfunctions' from an input string.

    This is used for calculating the "cost" of a circuit created by some compilation algorithms.

    Parameters
    ----------
    name : str
        Allowed values are:
            - '2QGC' : the cost of the circuit is the number of 2-qubit gates it contains.
            - 'depth' : the cost of the circuit is the depth of the circuit.
            - '2QGC:x:depth:y' : the cost of the circuit is x * the number of 2-qubit gates in the circuit +
                y * the depth of the circuit, where x and y are integers.

    Returns
    -------
    function
        A function that takes a circuit as the first argument, a ProcessorSpec as the second
        argument (or a "junk" input when a ProcessorSpec is not needed), and returns the cost
        of the circuit.
    """
    if name == '2QGC':
        def costfunction(circuit, junk):  # Junk input as no processorspec is needed here.
            return circuit.two_q_gate_count()
    elif name == 'depth':
        def costfunction(circuit, junk):  # Junk input as no processorspec is needed here.
            return circuit.depth()

    # This allows for '2QGC:x:depth:y' strings
    elif name[:4] == '2QGC':

        s = name.split(":")
        try: twoQGCfactor = int(s[1])
        except: raise ValueError("This `costfunction` string is not a valid option!")
        assert(s[2] == 'depth'), "This `costfunction` string is not a valid option!"
        try: depthfactor = int(s[3])
        except: raise ValueError("This `costfunction` string is not a valid option!")

        def costfunction(circuit, junk):  # Junk input as no processorspec is needed here.
            return twoQGCfactor * circuit.two_q_gate_count() + depthfactor * circuit.depth()

    else: raise ValueError("This `costfunction` string is not a valid option!")
    return costfunction


def compile_clifford(s, p, pspec=None, qubit_labels=None, iterations=20, algorithm='ROGGE', aargs=[],
                     costfunction='2QGC:10:depth:1', prefixpaulis=False, paulirandomize=False):
    """
    Compiles an n-qubit Clifford gate into a circuit over a given model.

    Compiles an n-qubit Clifford gate, described by the symplectic matrix s and vector p, into
    a circuit over the specified model, or, a standard model. Clifford gates/circuits can be converted
    to, or sampled in, the symplectic representation using the functions in pygsti.tools.symplectic.

    The circuit created by this function will be over a user-specified model and respects any desired
    connectivity, if a ProcessorSpec object is provided. Otherwise, it is over a canonical model containing
    all-to-all CNOTs, Hadamard, Phase, 3 products of Hadamard and Phase, and the Pauli gates.

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
        in this device. If this is None, the output circuit is over the "canonical" model of CNOT gates
        between all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation. In most circumstances, the output will be more useful if a
        ProcessorSpec is provided.

        If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the qubits in `pspec`
        the Clifford acts on. (All other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input Clifford needs to be ``padded'' to be the identity
        on those qubits).

        The ordering of the indices in (`s`,`p`) is w.r.t to ordering of the qubit labels in pspec.qubit_labels,
        unless `qubit_labels` is specified. Then, the ordering is taken w.r.t the ordering of the list `qubit_labels`.

    qubit_labels : list, optional
        Required if the Clifford to compile is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of
        pspec.qubit_labels. The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    iterations : int, optional
        Some of the allowed algorithms are randomized. This is the number of iterations used in
        algorithm if it is a randomized algorithm specified. If any randomized algorithms are specified,
        the time taken by this function increases linearly with `iterations`. Increasing `iterations`
        will often improve the obtained compilation (the "cost" of the obtained circuit, as specified
        by `costfunction` may decrease towards some asymptotic value).

    algorithm : str, optional
        Specifies the algorithm used for the core part of the compilation: finding a circuit that is a Clifford
        with `s` the symplectic matrix in its symplectic representation (a circuit that implements that desired
        Clifford up to Pauli operators). The allowed values of this are:

        - 'BGGE': A basic, deterministic global Gaussian elimination algorithm. Circuits obtained from this algorithm
           contain, in expectation, O(n^2) 2-qubit gates. Although the returned circuit will respect device
           connectivity, this algorithm does *not* take connectivity into account in an intelligient way. More details
           on this algorithm are given in `compile_symplectic_with_ordered_global_gaussian_elimination()`; it is the
           algorithm described in that docstring but with the qubit ordering fixed to the order in the input `s`.

        - 'ROGGE': A randomized elimination order global Gaussian elimination algorithm. This is the same algorithm as
           'BGGE' except that the order that the qubits are eliminated in is randomized. This results in significantly
           lower-cost circuits than the 'BGGE' method (given sufficiently many iterations). More details are given in
           the `compile_symplectic_with_random_ordered_global_gaussian_elimination()` docstring.

         - 'iAGvGE': Our improved version of the Aaraonson-Gottesman method for compiling a Clifford circuit, which
           uses 3 CNOT circuits and 3 1Q-gate layers (rather than the 5 CNOT circuit used in the algorithm of AG in
           Phys. Rev. A 70 052328 (2004)), with the CNOT circuits compiled using Gaussian elimination. Note that this
           algorithm appears to perform substantially worse than 'ROGGE', even though with an upgraded CNOT compiler
           it is asymptotically optimal (unlike any of the GGE methods). Also, note that this algorithm is randomized:
           there are many possible CNOT circuits (with non-equivalent action, individually) for the 2 of the 3 CNOT
           stages, and we randomize over those possible circuits. This randomization is equivalent to the randomization
           used in the stabilizer state/measurement compilers.

    aargs : list, optional
        If the algorithm can take optional arguments, not already specified as separate arguments above, then
        this list is passed to the compile_symplectic algorithm as its final arguments.

    costfunction : function or string, optional
        If a function, it is a function that takes a circuit and `pspec` as the first and second inputs and
        returns a 'cost' (a float) for the circuit. The circuit input to this function will be over the gates in
        `pspec`, if a `pspec` has been provided, and as described above if not. This costfunction is used to decide
        between different compilations when randomized algorithms are used: the lowest cost circuit is chosen. If
        a string it must be one of:

            - '2QGC' : the cost of the circuit is the number of 2-qubit gates it contains.
            - 'depth' : the cost of the circuit is the depth of the circuit.
            - '2QGC:x:depth:y' : the cost of the circuit is x * the number of 2-qubit gates in the circuit +
                y * the depth of the circuit, where x and y are integers.

    prefixpaulis : bool, optional
        A Pauli layer is needed to compile the correct Clifford (and not just the correct Clifford up to Paulis).  When
        `prefixpaulis = True` this Pauli layer is placed at the beginning of the circuit; when `False`, it is placed
        at the end. Note that the required Pauli layer depends on whether we pre-fix or post-fix it to the main
        symplectic-generating circuit.

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
    assert(_symp.check_valid_clifford(s, p)), "Input is not a valid Clifford!"
    n = _np.shape(s)[0] // 2

    if pspec is not None:
        if qubit_labels is None:
            assert(pspec.number_of_qubits == n), \
                ("If all the qubits in `pspec` are to be used, "
                 "the Clifford must be over all {} qubits!".format(pspec.number_of_qubits))
            qubit_labels = pspec.qubit_labels
        else:
            assert(len(qubit_labels) == n), "The subset of qubits to compile for is the wrong size for this CLifford!!"
            qubit_labels = qubit_labels
    else:
        assert(qubit_labels is None), "qubit_labels can only be specified if `pspec` is not None!"
        #qubit_labels = list(range(n))  #EGN commented this out b/c it leads to assertion error in compile_simplectic

    # Create a circuit that implements a Clifford with symplectic matrix s. This is the core
    # of this compiler, and is the part that can be implemented with different algorithms.
    circuit = compile_symplectic(s, pspec=pspec, qubit_labels=qubit_labels, iterations=iterations,
                                 algorithms=[algorithm], costfunction=costfunction,
                                 paulirandomize=paulirandomize, aargs={'algorithm': aargs}, check=False)
    circuit = circuit.copy(editable=True)

    temp_s, temp_p = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)

    # Find the necessary Pauli layer to compile the correct Clifford, not just the correct
    # Clifford up to Paulis. The required Pauli layer depends on whether we pre-fix or post-fix it.
    if qubit_labels is None: qubit_labels = tuple(range(n))
    if prefixpaulis: pauli_layer = _symp.find_premultipled_pauli(s, temp_p, p, qubit_labels=qubit_labels)
    else: pauli_layer = _symp.find_postmultipled_pauli(s, temp_p, p, qubit_labels=qubit_labels)
    # Turn the Pauli layer into a circuit.
    pauli_circuit = _Circuit(layer_labels=pauli_layer, line_labels=qubit_labels, editable=True)
    # Only change gate library of the Pauli circuit if we have a ProcessorSpec with compilations.
    if pspec is not None:
        pauli_circuit.change_gate_library(
            pspec.compilations['absolute'], one_q_gate_relations=pspec.oneQgate_relations)  # identity=pspec.identity,
    # Prefix or post-fix the Pauli circuit to the main symplectic-generating circuit.
    if prefixpaulis: circuit.prefix_circuit(pauli_circuit)
    else: circuit.append_circuit(pauli_circuit)

    # If we aren't Pauli-randomizing, do a final bit of depth compression
    if pspec is not None: circuit.compress_depth_inplace(one_q_gate_relations=pspec.oneQgate_relations, verbosity=0)
    else: circuit.compress_depth_inplace(verbosity=0)

    # Check that the correct Clifford has been compiled. This should never fail, but could if
    # the compilation provided for the internal gates is incorrect (the alternative is a mistake in this algorithm).
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
    assert(_np.array_equal(s, s_out))
    assert(_np.array_equal(p, p_out))

    return circuit


def compile_symplectic(s, pspec=None, qubit_labels=None, iterations=20, algorithms=['ROGGE'],
                       costfunction='2QGC:10:depth:1', paulirandomize=False, aargs={}, check=True):
    """
    Creates a :class:`Circuit` that implements a Clifford gate given in the symplectic representation.

    Returns an n-qubit circuit that implements an n-qubit Clifford gate that is described by the symplectic
    matrix `s` and *some* vector `p`. The circuit created by this function will be over a user-specified model
    and respecting any desired connectivity, if a ProcessorSpec object is provided. Otherwise, it is over a
    canonical model containing all-to-all CNOTs, Hadamard, Phase, 3 products of Hadamard and Phase, and the
    Pauli gates.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers.

    pspec : ProcessorSpec, optional
        An nbar-qubit ProcessorSpec object that encodes the device that `s` is being compiled
        for, where nbar >= n. If this is specified, the output circuit is over the gates available
        in this device. If this is None, the output circuit is over the "canonical" model of CNOT gates
        between all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation.

        If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the qubits in `pspec`
        the Clifford acts on. (All other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input `s`  needs to be ``padded'' to be the identity
        on those qubits).

        The indexing `s` is assumed to be the same as that in the list pspec.qubit_labels, unless `qubit_labels`
        is specified. Then, the ordering is taken w.r.t the ordering of the list `qubit_labels`.

    qubit_labels : list, optional
        Required if the Clifford to compile is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of pspec.qubit_labels.
        The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    iterations : int, optional
        Some of the allowed algorithms are randomized. This is the number of iterations used in
        each algorithm specified that is a randomized algorithm.

    algorithms : list of strings, optional
        Specifies the algorithms used. If more than one algorithm is specified, then all the algorithms
        are implemented and the lowest "cost" circuit obtained from all the algorithms (and iterations of
        those algorithms, if randomized) is returned.

        The allowed elements of this list are:

        - 'BGGE': A basic, deterministic global Gaussian elimination algorithm. Circuits obtained from this algorithm
           contain, in expectation, O(n^2) 2-qubit gates. Although the returned circuit will respect device
           connectivity, this algorithm does *not* take connectivity into account in an intelligient way. More details
           on this algorithm are given in `compile_symplectic_with_ordered_global_gaussian_elimination()`; it is the
           algorithm described in that docstring but with the qubit ordering fixed to the order in the input `s`.

        - 'ROGGE': A randomized elimination order global Gaussian elimination algorithm. This is the same algorithm as
           'BGGE' except that the order that the qubits are eliminated in is randomized. This results in significantly
           lower-cost circuits than the 'BGGE' method (given sufficiently many iterations). More details are given in
           the `compile_symplectic_with_random_ordered_global_gaussian_elimination()` docstring.

         - 'iAGvGE': Our improved version of the Aaraonson-Gottesman method for compiling a symplectic matrix, which
           uses 3 CNOT circuits and 3 1Q-gate layers (rather than the 5 CNOT circuit used in the algorithm of AG in
           Phys. Rev. A 70 052328 (2004)), with the CNOT circuits compiled using Gaussian elimination. Note that this
           algorithm appears to perform substantially worse than 'ROGGE', even though with an upgraded CNOT compiler
           it is asymptotically optimal (unlike any of the GGE methods). Also, note that this algorithm is randomized:
           there are many possible CNOT circuits (with non-equivalent action, individually) for the 2 of the 3 CNOT
           stages, and we randomize over those possible circuits. This randomization is equivalent to the randomization
           used in the stabilizer state/measurement compilers.

    costfunction : function or string, optional
        If a function, it is a function that takes a circuit and `pspec` as the first and second inputs and
        returns a cost (a float) for the circuit. The circuit input to this function will be over the gates in
        `pspec`, if a `pspec` has been provided, and as described above if not. This costfunction is used to decide
        between different compilations when randomized algorithms are used: the lowest cost circuit is chosen. If
        a string it must be one of:

            - '2QGC' : the cost of the circuit is the number of 2-qubit gates it contains.
            - 'depth' : the cost of the circuit is the depth of the circuit.
            - '2QGC:x:depth:y' : the cost of the circuit is x * the number of 2-qubit gates in the circuit +
                y * the depth of the circuit, where x and y are integers.

    paulirandomize : bool, optional
        If True then independent, uniformly random Pauli layers (a Pauli on each qubit) are inserted in between
        every layer in the circuit. These Paulis are then compiled into the gates in `pspec`, if `pspec` is provided.
        That is, this Pauli-frame-randomizes / Pauli-twirls the internal layers of this Clifford circuit. This can
        be useful for preventing coherent addition of errors in the circuit.

    aargs : dict, optional
        If the algorithm can take optional arguments, not already specified as separate arguments above, then
        the list arrgs[algorithmname] is passed to the compile_symplectic algorithm as its final arguments, where
        `algorithmname` is the name of algorithm specified in the list `algorithms`.

    check : bool, optional
        Whether to check that the output circuit implements the correct symplectic matrix (i.e., tests for algorithm
        success).

    Returns
    -------
    Circuit
        A circuit implementing the input Clifford gate/circuit.
    """
    # The number of qubits the symplectic matrix is on.
    n = _np.shape(s)[0] // 2
    if pspec is not None:
        if qubit_labels is None:
            assert(pspec.number_of_qubits == n), \
                ("If all the qubits in `pspec` are to be used, "
                 "`s` must be a symplectic matrix over {} qubits!".format(pspec.number_of_qubits))
        else:
            assert(len(qubit_labels) == n), \
                "The subset of qubits to compile `s` for is the wrong size for this symplectic matrix!"
    else:
        if qubit_labels is not None:
            import bpdb; bpdb.set_trace()
            pass
        assert(qubit_labels is None), "qubit_labels can only be specified if `pspec` is not None!"

    all_algorithms = ['BGGE', 'ROGGE', 'iAGvGE']  # Future: ['AGvGE','AGvPMH','iAGvPMH']
    assert(set(algorithms).issubset(set(all_algorithms))), "One or more algorithms names are invalid!"

    # A list to hold the compiled circuits, from which we'll choose the best one. Each algorithm
    # only returns 1 circuit, so this will have the same length as the `algorithms` list.
    circuits = []

    # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str):
        costfunction = _create_standard_costfunction(costfunction)

    # Deterministic basic global Gaussian elimination
    if 'BGGE' in algorithms:
        if qubit_labels is not None:
            eliminationorder = list(range(len(qubit_labels)))
        elif pspec is not None:
            eliminationorder = list(range(len(pspec.qubit_labels)))
        else:
            eliminationorder = list(range(n))
        circuit = _compile_symplectic_using_ogge_algorithm(s, eliminationorder=eliminationorder, pspec=pspec,
                                                           qubit_labels=qubit_labels, ctype='basic', check=False)
        circuits.append(circuit)

    # Randomized basic global Gaussian elimination, whereby the order that the qubits are eliminated in
    # is randomized.
    if 'ROGGE' in algorithms:
        circuit = _compile_symplectic_using_rogge_algorithm(s, pspec=pspec, qubit_labels=qubit_labels, ctype='basic',
                                                            costfunction=costfunction, iterations=iterations,
                                                            check=False)
        circuits.append(circuit)

    # Future:
    # The Aaraonson-Gottesman method for compiling a symplectic matrix using 5 CNOT circuits + local layers,
    # with the CNOT circuits compiled using Gaussian elimination.
    # if 'AGvGE' in algorithms:
    #     circuit = _compile_symplectic_using_ag_algorithm(s, pspec=pspec, qubit_labels=qubit_labels, cnotmethod='GE',
    #                                                     check=False)
    #     circuits.append(circuit)

    # Future
    # The Aaraonson-Gottesman method for compiling a symplectic matrix using 5 CNOT circuits + local layers,
    # with the CNOT circuits compiled using the asymptotically optimal O(n^2/logn) CNOT circuit algorithm of
    # PMH.
    # if 'AGvPMH' in algorithms:
    #     circuit = _compile_symplectic_using_ag_algorithm(s, pspec=pspec, qubit_labels=qubit_labels,
    #                                                      cnotmethod = 'PMH', check=False)
    #     circuits.append(circuit)

    # Our improved version of the Aaraonson-Gottesman method for compiling a symplectic matrix, which uses 3
    # CNOT circuits and 3 1Q-gate layers, with the CNOT circuits compiled using Gaussian elimination.
    if 'iAGvGE' in algorithms:
        # This defaults to what we think is the best Gauss. elimin. based CNOT compiler in pyGSTi (this one may actual
        # not be the best one though). Note that this is a randomized version of the algorithm (using the albert-factor
        # randomization).
        circuit = _compile_symplectic_using_riag_algoritm(s, pspec, qubit_labels=qubit_labels, iterations=iterations,
                                                          cnotalg='COiCAGE', cargs=[], costfunction=costfunction,
                                                          check=False)
        circuits.append(circuit)

    # Future
    # The Aaraonson-Gottesman method for compiling a symplectic matrix using 5 CNOT circuits + local layers,
    # with the CNOT circuits compiled using the asymptotically optimal O(n^2/logn) CNOT circuit algorithm of
    # PMH.
    # if 'iAGvPMH' in algorithms:
    #     circuit = compile_symplectic_with_iAG_algorithm(s, pspec=pspec, qubit_labels=qubit_labels, cnotmethod = 'PMH',
    #                                                     check=False)
    #     circuits.append(circuit)

    # If multiple algorithms have be called, find the lowest cost circuit.
    if len(circuits) > 1:
        bestcost = _np.inf
        for c in circuits:
            c_cost = costfunction(c, pspec)
            if c_cost < bestcost:
                circuit = c.copy()
                bestcost = c_cost
    else: circuit = circuits[0]

    # If we want to Pauli randomize the circuits, we insert a random compiled Pauli layer between every layer.
    if paulirandomize:
        paulilist = ['I', 'X', 'Y', 'Z']
        d = circuit.depth()
        for i in range(0, d + 1):
            # Different labelling depending on qubit_labels and pspec.
            if pspec is None:
                pcircuit = _Circuit(layer_labels=[_Label(paulilist[_np.random.randint(4)], k)
                                                  for k in range(n)], num_lines=n, identity='I')
            else:
                # Map the circuit to the correct qubit labels
                if qubit_labels is not None:
                    pcircuit = _Circuit(layer_labels=[_Label(paulilist[_np.random.randint(4)], qubit_labels[k])
                                                      for k in range(n)],
                                        line_labels=qubit_labels, editable=True)  # , identity=pspec.identity)
                else:
                    pcircuit = _Circuit(layer_labels=[_Label(paulilist[_np.random.randint(4)], pspec.qubit_labels[k])
                                                      for k in range(n)],
                                        line_labels=pspec.qubit_labels, editable=True)  # , identity=pspec.identity)
                # Compile the circuit into the native model, using an "absolute" compilation -- Pauli-equivalent is
                # not sufficient here.
                # identity=pspec.identity,
                pcircuit.change_gate_library(pspec.compilations['absolute'],
                                             one_q_gate_relations=pspec.oneQgate_relations)
            circuit.insert_circuit(pcircuit, d - i)

    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
        assert(_np.array_equal(s, implemented_s))

    return circuit


def _compile_symplectic_using_rogge_algorithm(s, pspec=None, qubit_labels=None, ctype='basic',
                                              costfunction='2QGC:10:depth:1', iterations=10, check=True):
    """
    Creates a :class:`Circuit` that implements a Clifford gate using the ROGGE algorithm.

    The order global Gaussian elimiation algorithm of _compile_symplectic_using_ogge_algorithm() with the
    qubit elimination order randomized. See that function for further details on the algorithm, This algorithm
    is more conveniently and flexibly accessed via the `compile_symplectic()` or  `compile_clifford()` wrap-around
    functions.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers.

    pspec : ProcessorSpec, optional
        An nbar-qubit ProcessorSpec object that encodes the device that `s` is being compiled
        for, where nbar >= n. If this is specified, the output circuit is over the gates available
        in this device. If this is None, the output circuit is over the "canonical" model of CNOT gates
        between all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation.

        If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the qubits in `pspec`
        the Clifford acts on. (All other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input `s`  needs to be ``padded'' to be the identity
        on those qubits).

        The indexing `s` is assumed to be the same as that in the list pspec.qubit_labels, unless `qubit_labels`
        is specified. Then, the ordering is taken w.r.t the ordering of the list `qubit_labels`.

    qubit_labels : list, optional
        Required if the Clifford to compile is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of pspec.qubit_labels.
        The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    ctype : str, optional
        The particular variant on the global Gaussian elimiation core algorithm. Currently there is only one
        such variant, corresponding to the string "basic".

    costfunction : function or string, optional
        If a function, it is a function that takes a circuit and `pspec` as the first and second inputs and
        returns a cost (a float) for the circuit. The circuit input to this function will be over the gates in
        `pspec`, if a `pspec` has been provided, and as described above if not. This costfunction is used to decide
        between different compilations when randomized algorithms are used: the lowest cost circuit is chosen. If
        a string it must be one of:

            - '2QGC' : the cost of the circuit is the number of 2-qubit gates it contains.
            - 'depth' : the cost of the circuit is the depth of the circuit.
            - '2QGC:x:depth:y' : the cost of the circuit is x * the number of 2-qubit gates in the circuit +
                y * the depth of the circuit, where x and y are integers.

    iterations : int, optional
        The number of different random orderings tried. The lowest "cost" circuit obtained from the
        different orderings is what is returned.

    check : bool, optional
        Whether to check that the output circuit implements the correct symplectic matrix (i.e., tests for algorithm
        success).

    Returns
    -------
    Circuit
        A circuit implementing the input symplectic matrix.
    """
    # The number of qubits the symplectic matrix is on.
    n = _np.shape(s)[0] // 2
    # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str):
        costfunction = _create_standard_costfunction(costfunction)

    # The elimination order in terms of qubit *index*, which is randomized below.
    if qubit_labels is not None:
        eliminationorder = list(range(len(qubit_labels)))
    elif pspec is not None:
        eliminationorder = list(range(len(pspec.qubit_labels)))
    else:
        eliminationorder = list(range(n))

    lowestcost = _np.inf
    for i in range(0, iterations):
        # Pick a random order to attempt the elimination in
        _np.random.shuffle(eliminationorder)
        # Call the re-ordered global Gaussian elimination, which is wrap-around for the GE algorithms to deal
        # with qubit relabeling. Check is False avoids multiple checks of success, when only the last check matters.
        circuit = _compile_symplectic_using_ogge_algorithm(
            s, eliminationorder, pspec=pspec, qubit_labels=qubit_labels, ctype=ctype, check=False)
        # Find the cost of the circuit, and keep it if this circuit is the lowest-cost circuit so far.
        circuit_cost = costfunction(circuit, pspec)
        if circuit_cost < lowestcost:
            bestcircuit = circuit.copy()
            lowestcost = circuit_cost

    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(bestcircuit, pspec=pspec)
        assert(_np.array_equal(s, implemented_s))

    return bestcircuit


def _compile_symplectic_using_ogge_algorithm(s, eliminationorder, pspec=None, qubit_labels=None,
                                             ctype='basic', check=True):
    """
    Creates a :class:`Circuit` that implements a Clifford gate using the OGGE algorithm.

    An ordered global Gaussian elimiation algorithm for creating a circuit that implements a Clifford that is
    represented by the symplectic matrix `s` (and *some* phase vector). This algorithm is more conveniently and flexibly
    accessed via the `compile_symplectic()` or `compile_clifford()` wrap-around functions.

    The algorithm works as follows:

    1. The `s` matrix is permuted so that the index for the jth qubit to eliminate becomes index j.
    2. The "global Gaussian elimination" algorithm of E. Hostens, J. Dehaene, and B. De Moor, PRA 71 042315 (2015) is
       implemented, which can be used to decompose `s` into a circuit over CNOT, SWAP, H, and P. That algorithm is for
       d-dimensional qudits, and simplifies significantly for d=2 (qubits), which is the case implemented here. However,
       we are unware of anywhere else that this algorithm is clearly stated for the qubit case (although this basic
       algorithm is widely known).

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers.

    eliminationorder : list
        The elimination order for the qubits. If `pspec` is specified, this should be a list consisting of the
        qubit labels that `s` is over (this can be a subset of all qubits in `pspec` is `qubit_labels` is None). If
        `pspec` is not specified this list should consist of the integers between 0 and n-1 in any order, corresponding
        to the indices of `s`.

    pspec : ProcessorSpec, optional
        An nbar-qubit ProcessorSpec object that encodes the device that `s` is being compiled
        for, where nbar >= n. If this is specified, the output circuit is over the gates available
        in this device. If this is None, the output circuit is over the "canonical" model of CNOT gates
        between all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation.

        If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the qubits in `pspec`
        the Clifford acts on. (All other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input `s`  needs to be ``padded'' to be the identity
        on those qubits).

        The indexing `s` is assumed to be the same as that in the list pspec.qubit_labels, unless `qubit_labels`
        is specified. Then, the ordering is taken w.r.t the ordering of the list `qubit_labels`.

    qubit_labels : list, optional
        Required if the Clifford to compile is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of pspec.qubit_labels.
        The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    ctype : str, optional
        The particular variant on the global Gaussian elimiation core algorithm. Currently there is only one
        such variant, corresponding to the string "basic".

    check : bool, optional
        Whether to check that the output circuit implements the correct symplectic matrix (i.e., tests for algorithm
        success).

    Returns
    -------
    Circuit
        A circuit implementing the input symplectic matrix.
    """
    # Re-order the s matrix to reflect the order we want to eliminate the qubits in,
    # because we hand the symp. matrix to a function that eliminates them in a fixed order.
    n = _np.shape(s)[0] // 2
    P = _np.zeros((n, n), int)
    for j in range(0, n):
        P[j, eliminationorder[j]] = 1
    P2n = _np.zeros((2 * n, 2 * n), int)
    P2n[0:n, 0:n] = P
    P2n[n:2 * n, n:2 * n] = P
    permuted_s = _mtx.dot_mod2(_mtx.dot_mod2(P2n, s), _np.transpose(P2n))

    if ctype == 'basic':
        # Check is False avoids multiple checks of success, when only the last check matters.
        circuit = _compile_symplectic_using_gge_core(permuted_s, check=False)
        circuit = circuit.copy(editable=True)  # make editable - maybe make `editable` a param of above fn call?
    else: raise ValueError("The compilation sub-method is not valid!")
    # Futures: write a connectivity-adjusted algorithm, similar to the COGE/iCAGE CNOT compilers.

    # If the qubit_labels is not None, we relabel the circuit in terms of the labels of these qubits.
    if qubit_labels is not None:
        assert(len(eliminationorder) == len(qubit_labels)), \
            "`qubit_labels` must be the same length as `elimintionorder`! The mapping to qubit labels is ambigiuous!"
        circuit.map_state_space_labels_inplace({i: qubit_labels[eliminationorder[i]] for i in range(n)})
        circuit.reorder_lines(qubit_labels)
    # If the qubit_labels is None, but there is a pspec, we relabel the circuit in terms of the full set
    # of pspec labels.
    elif pspec is not None:
        assert(len(eliminationorder) == len(pspec.qubit_labels)
               ), "If `qubit_labels` is not specified `s` should be over all the qubits in `pspec`!"
        circuit.map_state_space_labels_inplace({i: pspec.qubit_labels[eliminationorder[i]] for i in range(n)})
        circuit.reorder_lines(pspec.qubit_labels)
    else:
        circuit.map_state_space_labels_inplace({i: eliminationorder[i] for i in range(n)})
        circuit.reorder_lines(list(range(n)))

    # If we have a pspec, we change the gate library. We use a pauli-equivalent compilation, as it is
    # only necessary to implement each gate in this circuit up to Pauli matrices.
    if pspec is not None:
        if qubit_labels is None:
            # ,identity=pspec.identity,
            circuit.change_gate_library(pspec.compilations['paulieq'], one_q_gate_relations=pspec.oneQgate_relations)
        else:
            # identity=pspec.identity,
            circuit.change_gate_library(pspec.compilations['paulieq'], allowed_filter=set(qubit_labels),
                                        one_q_gate_relations=pspec.oneQgate_relations)
    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
        assert(_np.array_equal(s, implemented_s))

    return circuit


def _compile_symplectic_using_gge_core(s, check=True):
    """
    Creates a :class:`Circuit` that implements a Clifford gate using the GGE algorithm.

    Creates a circuit over 'I','H','HP','PH','HPH', and 'CNOT' that implements a Clifford
    gate with `s` as its symplectic matrix in the symplectic representation (and with any
    phase vector). This circuit is generated using a basic Gaussian elimination algorithm,
    which is described in more detail in _compile_symplectic_using_ogge_algorithm(), which
    is a wrap-around for this algorithm that implements a more flexible compilation method.

    This algorithm is more conveniently accessed via the `compile_symplectic()` or
    `compile_clifford()` functions.

    Parameters
    ----------
    s : array
        A 2n X 2n symplectic matrix over [0,1] for any positive integer n. The returned
        circuit is over n qubits.

    check : bool, optional
        Whether to check that the generated circuit does implement `s`.

    Returns
    -------
    Circuit
        A circuit that implements a Clifford that is represented by the symplectic matrix `s`.
    """
    sout = _np.copy(s)  # Copy so that we don't change the input s.
    n = _np.shape(s)[0] // 2

    assert(_symp.check_symplectic(s, convention='standard')), "The input matrix must be symplectic!"

    instruction_list = []
    # Map the portion of the symplectic matrix acting on qubit j to the identity, for j = 0,...,d-1 in
    # turn, using the basic row operations corresponding to the CNOT, Hadamard, phase, and SWAP gates.
    for j in range(n):

        # *** Step 1: Set the upper half of column j to the relevant identity column ***
        upperl_c = sout[:n, j]
        lowerl_c = sout[n:, j]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])

        # If the jth element in the column is not 1, it needs to be set to 1.
        if j not in upperl_ones:

            # First try using a Hadamard gate.
            if j in lowerl_ones:
                instruction_list.append(_Label('H', j))
                _symp.apply_internal_gate_to_symplectic(sout, 'H', [j, ])

            # Then try using a swap gate, we don't try and find the best qubit to swap with.
            elif len(upperl_ones) >= 1:
                instruction_list.append(_Label('CNOT', [j, upperl_ones[0]]))
                instruction_list.append(_Label('CNOT', [upperl_ones[0], j]))
                instruction_list.append(_Label('CNOT', [j, upperl_ones[0]]))
                _symp.apply_internal_gate_to_symplectic(sout, 'SWAP', [j, upperl_ones[0]])

            # Finally, try using swap and Hadamard gates, we don't try and find the best qubit to swap with.
            else:
                instruction_list.append(_Label('H', lowerl_ones[0]))
                _symp.apply_internal_gate_to_symplectic(sout, 'H', [lowerl_ones[0], ])
                instruction_list.append(_Label('CNOT', [j, lowerl_ones[0]]))
                instruction_list.append(_Label('CNOT', [lowerl_ones[0], j]))
                instruction_list.append(_Label('CNOT', [j, lowerl_ones[0]]))
                _symp.apply_internal_gate_to_symplectic(sout, 'SWAP', [j, lowerl_ones[0]])

            # Update the lists that keep track of where the 1s are in the column.
            upperl_c = sout[:n, j]
            lowerl_c = sout[n:, j]
            upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
            lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])

        # Pair up qubits with 1s in the jth upper jth column, and set all but the
        # jth qubit to 0 in logarithmic depth. When there is an odd number of qubits
        # one of them is left out in the layer.
        while len(upperl_ones) >= 2:

            num_pairs = len(upperl_ones) // 2

            for i in range(0, num_pairs):
                if upperl_ones[i + 1] != j:
                    controlq = upperl_ones[i]
                    targetq = upperl_ones[i + 1]
                    del upperl_ones[1 + i]
                else:
                    controlq = upperl_ones[i + 1]
                    targetq = upperl_ones[i]
                    del upperl_ones[i]

                instruction_list.append(_Label('CNOT', (controlq, targetq)))
                _symp.apply_internal_gate_to_symplectic(sout, 'CNOT', [controlq, targetq])

        # *** Step 2: Set the lower half of column j to all zeros ***
        upperl_c = sout[:n, j]
        lowerl_c = sout[n:, j]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])

        # If the jth element in this lower column is 1, it must be set to 0.
        if j in lowerl_ones:
            instruction_list.append(_Label('P', j))
            _symp.apply_internal_gate_to_symplectic(sout, 'P', [j, ])

        # Move in the 1 from the upper part of the column, and use this to set all
        # other elements to 0, as in Step 1.
        instruction_list.append(_Label('H', j))
        _symp.apply_internal_gate_to_symplectic(sout, 'H', [j, ])

        upperl_c = None
        upperl_ones = None
        lowerl_c = sout[n:, j]
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])

        while len(lowerl_ones) >= 2:

            num_pairs = len(lowerl_ones) // 2

            for i in range(0, num_pairs):
                if lowerl_ones[i + 1] != j:
                    controlq = lowerl_ones[i + 1]
                    targetq = lowerl_ones[i]
                    del lowerl_ones[1 + i]
                else:
                    controlq = lowerl_ones[i]
                    targetq = lowerl_ones[i + 1]
                    del lowerl_ones[i]

                instruction_list.append(_Label('CNOT', (controlq, targetq)))
                _symp.apply_internal_gate_to_symplectic(sout, 'CNOT', [controlq, targetq])

        # Move the 1 back to the upper column.
        instruction_list.append(_Label('H', j))
        _symp.apply_internal_gate_to_symplectic(sout, 'H', [j, ])

        # *** Step 3: Set the lower half of column j+d to the relevant identity column ***
        upperl_c = sout[:n, j + n]
        lowerl_c = sout[n:, j + n]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])

        while len(lowerl_ones) >= 2:

            num_pairs = len(lowerl_ones) // 2

            for i in range(0, num_pairs):

                if lowerl_ones[i + 1] != j:
                    controlq = lowerl_ones[i + 1]
                    targetq = lowerl_ones[i]
                    del lowerl_ones[1 + i]
                else:
                    controlq = lowerl_ones[i]
                    targetq = lowerl_ones[i + 1]
                    del lowerl_ones[i]

                instruction_list.append(_Label('CNOT', (controlq, targetq)))
                _symp.apply_internal_gate_to_symplectic(sout, 'CNOT', [controlq, targetq])

        # *** Step 4: Set the upper half of column j+d to all zeros ***
        upperl_c = sout[:n, j + n]
        lowerl_c = sout[n:, j + n]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_ones = list(_np.nonzero(lowerl_c == 1)[0])

        # If the jth element in the upper column is 1 it must be set to zero
        if j in upperl_ones:
            instruction_list.append(_Label('H', j))
            _symp.apply_internal_gate_to_symplectic(sout, 'H', [j, ])
            instruction_list.append(_Label('P', j))
            _symp.apply_internal_gate_to_symplectic(sout, 'P', [j, ])
            instruction_list.append(_Label('H', j))
            _symp.apply_internal_gate_to_symplectic(sout, 'H', [j, ])

        # Switch in the 1 from the lower column
        instruction_list.append(_Label('H', j))
        _symp.apply_internal_gate_to_symplectic(sout, 'H', [j, ])

        upperl_c = sout[:n, j + n]
        upperl_ones = list(_np.nonzero(upperl_c == 1)[0])
        lowerl_c = None
        lowerl_ones = None

        while len(upperl_ones) >= 2:

            num_pairs = len(upperl_ones) // 2

            for i in range(0, num_pairs):
                if upperl_ones[i + 1] != j:
                    controlq = upperl_ones[i]
                    targetq = upperl_ones[i + 1]
                    del upperl_ones[1 + i]
                else:
                    controlq = upperl_ones[i + 1]
                    targetq = upperl_ones[i]
                    del upperl_ones[i]
                instruction_list.append(_Label('CNOT', (controlq, targetq)))
                _symp.apply_internal_gate_to_symplectic(sout, 'CNOT', [controlq, targetq])

        # Switch the 1 back to the lower column
        instruction_list.append(_Label('H', j))
        _symp.apply_internal_gate_to_symplectic(sout, 'H', [j], optype='row')

        # If the matrix has been mapped to the identity, quit the loop as we are done.
        if _np.array_equal(sout, _np.identity(2 * n, int)):
            break

    assert(_np.array_equal(sout, _np.identity(2 * n, int))), "Compilation has failed!"
    # Operations that are the same next to each other cancel, and this algorithm can have these. So
    # we go through and delete them.
    j = 1
    depth = len(instruction_list)
    while j < depth:

        if instruction_list[depth - j] == instruction_list[depth - j - 1]:
            del instruction_list[depth - j]
            del instruction_list[depth - j - 1]
            j = j + 2
        else:
            j = j + 1
    # We turn the instruction list into a circuit over the internal gates.
    circuit = _Circuit(layer_labels=instruction_list, num_lines=n, editable=True)  # ,identity='I')
    # That circuit implements the inverse of s (it maps s to the identity). As all the gates in this
    # set are self-inverse (up to Pauli multiplication) we just reverse the circuit to get a circuit
    # for s.
    circuit.reverse_inplace()
    # To do the depth compression, we use the 1-qubit gate relations for the standard set of gates used
    # here.
    oneQgate_relations = _symp.one_q_clifford_symplectic_group_relations()
    circuit.compress_depth_inplace(one_q_gate_relations=oneQgate_relations, verbosity=0)
    # We check that the correct Clifford -- up to Pauli operators -- has been implemented.
    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit)
        assert(_np.array_equal(s, implemented_s))

    circuit.done_editing()
    return circuit


def _compile_symplectic_using_ag_algorithm(s, pspec=None, qubit_labels=None, cnotmethod='PMH', check=False):
    """
    Creates a :class:`Circuit` that implements a Clifford gate using the AG algorithm.

    The Aaraonson-Gottesman method for compiling a symplectic matrix using 5 CNOT circuits + local layers.
    This algorithm is presented in PRA 70 052328 (2014).

    - If `cnotmethod` = `GE` then the CNOT circuits are compiled using Gaussian elimination (which is O(n^2)).
      There are multiple GE algorithms for compiling a CNOT in pyGSTi. This function has the over-all best
      variant of this algorithm hard-coded into this function.

    - If `cnotmethod` = `PMH` then the CNOT circuits are compiled using the asymptotically optimal
      O(n^2/logn) CNOT circuit algorithm of PMH.

    *** This function has not yet been implemented ***

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers.

    pspec : ProcessorSpec, optional
        An nbar-qubit ProcessorSpec object that encodes the device that `s` is being compiled
        for, where nbar >= n.

    qubit_labels : list, optional
        Required if the Clifford to compile is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of pspec.qubit_labels.
        The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    ctype : str, optional
        The particular variant on the global Gaussian elimiation core algorithm. Currently there is only one
        such variant, corresponding to the string "basic".

    cnotmethod : {"GE", "PMH"}
        See above.

    check : bool, optional
        Whether to check that the generated circuit does implement `s`.

    Returns
    -------
    Circuit
        A circuit that implements a Clifford that is represented by the symplectic matrix `s`.
    """
    raise NotImplementedError("This method is not yet written!")
    circuit = None
    return circuit


def _compile_symplectic_using_riag_algoritm(s, pspec, qubit_labels=None, iterations=20, cnotalg='COiCAGE',
                                            cargs=[], costfunction='2QGC:10:depth:1', check=True):
    """
    Creates a :class:`Circuit` that implements a Clifford gate using the RIAG algorithm.

    Our improved version of Aaraonson-Gottesman method [PRA 70 052328 (2014)] for compiling a symplectic matrix
    using 5 CNOT circuits + local layers. Our version of this algorithm uses 3 CNOT circuits, and 3 layers of
    1-qubit gates. Also, note that this algorithm is randomized: there are many possible CNOT circuits (with
    non-equivalent action, individually) for the 2 of the 3 CNOT stages, and we randomize over those possible
    circuits. This randomization is equivalent to the randomization used in the stabilizer state/measurement
    compilers.

    Note that this algorithm currently performs substantially worse than 'ROGGE', even though with an upgraded
    CNOT compiler it is asymptotically optimal (unlike any of the GGE methods).

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers.

    pspec : ProcessorSpec
        An nbar-qubit ProcessorSpec object that encodes the device that `s` is being compiled
        for, where nbar >= n. If this is specified, the output circuit is over the gates available
        in this device. If this is None, the output circuit is over the "canonical" model of CNOT gates
        between all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation.

        If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the qubits in `pspec`
        the Clifford acts on. (All other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input `s`  needs to be ``padded'' to be the identity
        on those qubits).

        The indexing `s` is assumed to be the same as that in the list pspec.qubit_labels, unless `qubit_labels`
        is specified. Then, the ordering is taken w.r.t the ordering of the list `qubit_labels`.

    qubit_labels : List, optional
        Required if the Clifford to compile is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of pspec.qubit_labels.
        The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    iterations : int, optional
        The number of different random orderings tried. The lowest "cost" circuit obtained from the
        different orderings is what is returned.

    cnotalg : str, optional
        The CNOT compiler to use. See `compile_cnot_circuit()` for the options. The default is *probably*
        the current best CNOT circuit compiler in pyGSTI.

    cargs : list, optional
        Arguments handed to the CNOT compilation algorithm. For some choices of `cnotalg` this is not
        optional.

    costfunction : function or string, optional
        If a function, it is a function that takes a circuit and `pspec` as the first and second inputs and
        returns a cost (a float) for the circuit. The circuit input to this function will be over the gates in
        `pspec`, if a `pspec` has been provided, and as described above if not. This costfunction is used to decide
        between different compilations when randomized algorithms are used: the lowest cost circuit is chosen. If
        a string it must be one of:

            - '2QGC' : the cost of the circuit is the number of 2-qubit gates it contains.
            - 'depth' : the cost of the circuit is the depth of the circuit.
            - '2QGC:x:depth:y' : the cost of the circuit is x * the number of 2-qubit gates in the circuit +
                y * the depth of the circuit, where x and y are integers.

    check : bool, optional
        Whether to check that the output circuit implements the correct symplectic matrix (i.e., tests for algorithm
        success).

    Returns
    -------
    Circuit
        A circuit implementing the input symplectic matrix.
    """
    # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str): costfunction = _create_standard_costfunction(costfunction)

    mincost = _np.inf
    for i in range(iterations):
        circuit = _compile_symplectic_using_iag_algorithm(
            s, pspec, qubit_labels=qubit_labels, cnotalg=cnotalg, cargs=cargs, check=False)

        # Change to the native gate library
        if pspec is not None:  # Currently pspec is not optional, so this always happens.
            circuit = circuit.copy(editable=True)
            if qubit_labels is None:
                # ,identity=pspec.identity
                circuit.change_gate_library(pspec.compilations['paulieq'],
                                            one_q_gate_relations=pspec.oneQgate_relations)
            else:
                # identity=pspec.identity,
                circuit.change_gate_library(pspec.compilations['paulieq'], allowed_filter=set(qubit_labels),
                                            one_q_gate_relations=pspec.oneQgate_relations)

        # Calculate the cost after changing gate library.
        cost = costfunction(circuit, pspec)
        if cost < mincost:
            mincost = cost
            bestcircuit = circuit.copy(editable=False)

    if check:
        implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(bestcircuit, pspec=pspec)
        assert(_np.array_equal(s, implemented_s))

    return bestcircuit


def _compile_symplectic_using_iag_algorithm(s, pspec, qubit_labels=None, cnotalg='COCAGE', cargs=[], check=True):
    """
    Creates a :class:`Circuit` that implements a Clifford gate using the IAG algorithm.

    A single iteration of the algorithm in _compile_symplectic_using_riag_algoritm(). See that functions
    docstring for more information. Note that it is normallly better to access this algorithm through that
    function even when only a single iteration of the randomization is desired: this function does *not* change
    into the native model.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers.

    pspec : ProcessorSpec, optional
        An nbar-qubit ProcessorSpec object that encodes the device that `s` is being compiled
        for, where nbar >= n.

    qubit_labels : list, optional
        Required if the Clifford to compile is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of pspec.qubit_labels.
        The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    cnotalg : str, optional
        The `algorithm` argument to pass internally to :function:`compile_cnot_circuit`
        when compiling CNOT gates.

    cargs : various, optional
        The `aargs` argument to pass internally to :function:`compile_cnot_circuit`
        when compiling CNOT gates.

    check : bool, optional
        Whether to check that the generated circuit does implement `s`.

    Returns
    -------
    Circuit
        A circuit that implements a Clifford that is represented by the symplectic matrix `s`.
    """
    assert(pspec is not None), "`pspec` cannot be None with this algorithm!"
    n = _np.shape(s)[0] // 2

    if qubit_labels is not None:
        assert(len(qubit_labels) == n), "The length of `qubit_labels` is inconsisent with the size of `s`!"
        qubit_labels = qubit_labels
    else:
        qubit_labels = pspec.qubit_labels
        assert(len(qubit_labels) == n), \
            ("The number of qubits is inconsisent with the size of `s`! "
             "If `s` is over a subset, `qubit_labels` must be specified!")

    # A matrix to keep track of the current state of s.
    sout = s.copy()

    # Stage 1: Hadamard gates from the LHS to make the UR submatrix of s invertible.
    sout, LHS1_Hsome_layer = _make_submatrix_invertable_using_hadamards(sout, 'row', 'UR', qubit_labels)
    assert(_symp.check_symplectic(sout))
    # Stage 2: CNOT circuit from the RHS to map the UR submatrix of s to I.
    sout, RHS1A_CNOTs, success = _submatrix_gaussian_elimination_using_cnots(sout, 'column', 'UR', qubit_labels)
    assert(success), \
        "The 1st Gaussian elimination stage of the algorithm has failed! Perhaps the input was not a symplectic matrix."
    assert(_symp.check_symplectic(sout))
    # Stage 3: Phase circuit from the LHS to make the LR submatrix of s invertible
    sout, LHS2_Psome_layer = _make_submatrix_invertable_using_phases_and_idsubmatrix(sout, 'row', 'LR', qubit_labels)
    assert(_symp.check_symplectic(sout))
    # Stage 4: CNOT circuit from the LHS to map the UR and LR submatrices of s to the same invertible matrix M
    sout, LHS3_CNOTs = find_albert_factorization_transform_using_cnots(sout, 'row', 'LR', qubit_labels)
    assert(_symp.check_symplectic(sout))
    # Stage 5: A CNOT circuit from the RHS to map the URH and LRH submatrices of s from M to I.
    sout, RHS1B_CNOTs, success = _submatrix_gaussian_elimination_using_cnots(sout, 'column', 'UR', qubit_labels)
    assert(success), \
        "The 3rd Gaussian elimination stage of the algorithm has failed! Perhaps the input was not a symplectic matrix."
    assert(_symp.check_symplectic(sout))
    # Stage 6: Phase gates on all qubits acting from the LHS to map the LR submatrix of s to 0.
    sout, LHS4_Pall_layer = _apply_phase_to_all_qubits(sout, 'row', qubit_labels)
    assert(_symp.check_symplectic(sout))
    # Stage 7: Hadamard gates on all qubits acting from the LHS to swap the LR and UR matrices
    # of s, (mapping them to I and 0 resp.,).
    sout, LHS5_Hall_layer = _apply_hadamard_to_all_qubits(sout, 'row', qubit_labels)
    assert(_symp.check_symplectic(sout))
    # Stage 8 : Phase circuit from the LHS to make the LL submatrix of s invertible
    sout, LHS6_Psome_layer = _make_submatrix_invertable_using_phases_and_idsubmatrix(sout, 'row', 'LL', qubit_labels)
    assert(_symp.check_symplectic(sout))
    # Stage 9: CNOT circuit from the RHS to map the UR and LR submatrices of s to the same invertible matrix M
    sout, RHS1C_CNOTs = find_albert_factorization_transform_using_cnots(sout, 'column', 'LL', qubit_labels)
    assert(_symp.check_symplectic(sout))
    # Stage 10: Phase gates on all qubits acting from the RHS to map the LL submatrix of s to 0.
    sout, RHS2_Pall_layer = _apply_phase_to_all_qubits(sout, 'column', qubit_labels)
    assert(_symp.check_symplectic(sout))
    # Stage 11: A CNOT circuit from the LHS to map the to I. (can do the GE on either UL or LR)
    sout, LHS7_CNOTs, success = _submatrix_gaussian_elimination_using_cnots(sout, 'row', 'UL', qubit_labels)
    assert(_symp.check_symplectic(sout))
    assert(_np.array_equal(sout, _np.identity(2 * n, int))), "Algorithm has failed!"

    RHS1A_CNOTs.reverse()
    RHS1B_CNOTs.reverse()
    RHS1C_CNOTs.reverse()
    circuit_1_cnots = _Circuit(layer_labels=RHS1A_CNOTs + RHS1B_CNOTs + RHS1C_CNOTs,
                               line_labels=qubit_labels).parallelize()
    circuit_1_local = _Circuit(layer_labels=RHS2_Pall_layer, line_labels=qubit_labels).parallelize()
    LHS7_CNOTs.reverse()
    circuit_2_cnots = _Circuit(layer_labels=LHS7_CNOTs, line_labels=qubit_labels).parallelize()
    circuit_2_local = _Circuit(layer_labels=LHS6_Psome_layer + LHS5_Hall_layer + LHS4_Pall_layer,
                               line_labels=qubit_labels).parallelize()
    LHS3_CNOTs.reverse()
    circuit_3_cnots = _Circuit(layer_labels=LHS3_CNOTs, line_labels=qubit_labels)
    circuit_3_local = _Circuit(layer_labels=LHS2_Psome_layer + LHS1_Hsome_layer, line_labels=qubit_labels)

    cnot1_s, junk = _symp.symplectic_rep_of_clifford_circuit(circuit_1_cnots)
    cnot2_s, junk = _symp.symplectic_rep_of_clifford_circuit(circuit_2_cnots)
    cnot3_s, junk = _symp.symplectic_rep_of_clifford_circuit(circuit_3_cnots)

    # clname is set to None so that the function doesn't change the circuit into the native gate library.
    circuit_1_cnots = compile_cnot_circuit(cnot1_s, pspec, qubit_labels=qubit_labels,
                                           algorithm=cnotalg, clname=None, check=False, aargs=cargs)
    circuit_2_cnots = compile_cnot_circuit(cnot2_s, pspec, qubit_labels=qubit_labels,
                                           algorithm=cnotalg, clname=None, check=False, aargs=cargs)
    circuit_3_cnots = compile_cnot_circuit(cnot3_s, pspec, qubit_labels=qubit_labels,
                                           algorithm=cnotalg, clname=None, check=False, aargs=cargs)

    circuit = circuit_1_cnots.copy(editable=True)
    circuit.append_circuit(circuit_1_local)
    circuit.append_circuit(circuit_2_cnots)
    circuit.append_circuit(circuit_2_local)
    circuit.append_circuit(circuit_3_cnots)
    circuit.append_circuit(circuit_3_local)
    circuit.done_editing()

    if check:
        scheck, pcheck = _symp.symplectic_rep_of_clifford_circuit(circuit)
        assert(_np.array_equal(s, scheck)), "Compiler has failed!"

    return circuit


def compile_cnot_circuit(s, pspec, qubit_labels=None, algorithm='COiCAGE', clname=None, check=True, aargs=[]):
    """
    A CNOT circuit compiler.

    Takes an arbitrary CNOT circuit, input as a symplectic matrix `s` that corresponds
    to the matrix portion of the symplectic representation of this Clifford circuit, and decomposes it into a
    sequences of gates from the processor spec `pspec`.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers that represents a Clifford circuit: so
        it must be block-diagonal. Specifically, it has the form s = ((A,0),(0,B)) where B is the
        inverse transpose of A (over [0,1] mod 2).

    pspec : ProcessorSpec
        An nbar-qubit ProcessorSpec object that encodes the device that `s` is being compiled
        for, where nbar >= n. If this is specified, the output circuit is over the gates available
        in this device. If this is None, the output circuit is over the "canonical" model of CNOT gates
        between all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation.

        If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the qubits in `pspec`
        the Clifford acts on. (All other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input `s`  needs to be ``padded'' to be the identity
        on those qubits).

        The indexing `s` is assumed to be the same as that in the list pspec.qubit_labels, unless `qubit_labels`
        is specified. Then, the ordering is taken w.r.t the ordering of the list `qubit_labels`.

    qubit_labels : list, optional
        Required if the Clifford to compile is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of pspec.qubit_labels.
        The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    algorithm : str, optional
        The algorithm to use. The optionas are:

            - 'BGE' : A basic Gaussian elimination algorithm, that uses CNOT to perform row-reduction on the upper
                LHS (or lower RHS) of `s`. This algorithm does not take device connectivity into account.
            - 'OCAGE' : User-ordered connectivity-adjusted Gaussian elimination. The qubits are eliminated in the
                specified order; the first element of arrgs must be a list specify this order. The algorithm is
                also "connectivity-adjusted" in the sense that it uses the connectivity graph (in pspec.qubitgraph)
                to try and avoid using CNOTs between unconnected pairs whenever possible, and to decide the order
                of various operations.
            - 'OiCAGE' : The same as 'OCAGE' except that it has some improvements, and it requires connectivity
                graph of the remaining qubits, after each qubit has been 'eliminated', to be connected. In the current
                format this algorithm is only slightly better-performing than 'OCAGE', and only on average. (This
                algorithm will possibly be improved in the future, whereas 'OCAGE' will remain as-is for reproducability
                of previously obtained results.)
            - 'ROCAGE': Same as 'OCAGE' except that the elimination order is chosen at random, rather than user-
                specified.
             - 'COCAGE', 'COiCAGE' : The same as 'OCAGE' and 'OiCAGE', respectively, except that the elimination order
                is fixed to eliminate qubits with the worse connectivity before those with better connectivity.

    clname : str, optional
        A name for a CompilationLibrary in `pspec`, i.e., a key to the dict `pspec.compilationlibraries`. If
        specified, the output circuit is over the gates in `pspec` (rather than over CNOT), with the replacement
        according to this specified CompilationLibrary. If it is only necessary to implement the correct circuit
        up to paulis this would be `paulieq`. To obtain a compilation that exactly implements the input CNOT circuit
        this should be set to `absolute`. (Although user-added compilation libraries are also fine).

    check : bool, optional
        Whether to check the output is correct.

    aargs : list, optional
        A list of arguments handed to the CNOT compiler algorithm. For some choices of algorithm (e.g., 'OCAGE') this
        list must not be empty. For algorithms where there are X non-optional arguements *after* `s` and `pspec`
        these are specified as the first X arguments of aargs. The remaining elements in `aargs`, if any, are handed
        to the algorithm as the arguments after the optional `qubit_labels` and `check` arguments (the first of which is
        set by the input `qubit_labels` in this function).

    Returns
    -------
    Circuit
        A circuit that implements the same unitary as the CNOT circuit represented by `s`.
    """

    if qubit_labels is not None: qubits = list(qubit_labels)
    else: qubits = pspec.qubit_labels
    n = len(qubits)
    assert(n == _np.shape(s)[0] // 2), "The CNOT circuit is over the wrong number of qubits!"
    assert(_np.array_equal(s[:n, n:2 * n], _np.zeros((n, n), int))
           ), "`s` is not block-diagonal and so does not rep. a valid CNOT circuit!"
    assert(_np.array_equal(s[n:2 * n, :n], _np.zeros((n, n), int))
           ), "`s` is not block-diagonal and so does not rep. a valid CNOT circuit!"
    assert(_symp.check_symplectic(s)), "`s` is not symplectic, so it does not rep. a valid CNOT circuit!"

    # basic GE
    if algorithm == 'BGE': circuit = _compile_cnot_circuit_using_bge_algorithm(s, pspec, qubit_labels=qubit_labels)

    # ordered GE with the qubit elimination order specified by the aargs list.
    elif algorithm == 'OCAGE' or algorithm == 'OiCAGE':
        assert(set(aargs[0]) == set(qubits)), \
            'With the `OCAGE` algorithm, `arrgs` must be a length-1 list/tuple containing a list of all the qubits!'
        qubitorder = _copy.copy(aargs[0])
        if algorithm == 'OCAGE':
            circuit = _compile_cnot_circuit_using_ocage_algorithm(
                s, pspec, qubitorder, qubit_labels=qubit_labels, check=False, *aargs[1:])
        if algorithm == 'OiCAGE':
            circuit = _compile_cnot_circuit_using_oicage_algorithm(
                s, pspec, qubitorder, qubit_labels=qubit_labels, check=False, *aargs[1:])

    # ordered GE with the qubit elimination order from least to most connected qubit
    elif algorithm == 'COCAGE' or algorithm == 'COiCAGE':

        remaining_qubits = _copy.copy(qubits)

        qubitorder = []
        for k in range(n):
            # Find the distance matrix for the remaining qubits
            distances = pspec.qubitgraph.subgraph(remaining_qubits).shortest_path_distance_matrix()
            # Cost them on the total distance to all other qubits.
            costs = _np.sum(distances, axis=0)
            # Find the most-expensive qubit, and put that next in the list
            qindex = _np.argmax(costs)
            qlabel = remaining_qubits[qindex]
            qubitorder.append(qlabel)
            # Remove this qubit from the remaining qubits, so we cost the next qubit by the
            # connectivity graph of the remaining qubits.
            del remaining_qubits[qindex]

        if algorithm == 'COCAGE':
            circuit = _compile_cnot_circuit_using_ocage_algorithm(
                s, pspec, qubitorder, qubit_labels=qubit_labels, check=False, *aargs)
        if algorithm == 'COiCAGE':
            circuit = _compile_cnot_circuit_using_oicage_algorithm(
                s, pspec, qubitorder, qubit_labels=qubit_labels, check=False, *aargs)

    # ordered GE with the qubit elimination order random. This is likely a pretty stupid algorithm to use
    # when device connectivity is not all-to-all
    elif algorithm == 'ROCAGE':
        # future : add an iterations option?
        qubitorder = _copy.copy(qubits)
        _np.random.shuffle(qubitorder)
        if algorithm == 'ROCAGE':
            circuit = _compile_cnot_circuit_using_ocage_algorithm(
                s, pspec, qubitorder, qubit_labels=qubit_labels, check=True, *aargs)

    else: raise ValueError("The choice of algorithm is invalid!")

    # If a compilation is specified, we compile into the native model.
    if clname is not None:
        circuit = circuit.copy(editable=True)
        circuit.change_gate_library(pspec.compilations[clname], allowed_filter=qubit_labels,
                                    one_q_gate_relations=pspec.oneQgate_relations)  # , identity=pspec.identity)
    if check:
        s_implemented, p_implemented = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
        # This only checks its correct upto the phase vector, so that we can use the algorithm
        # with paulieq compilations and it won't fail when check is True.
        assert(_np.array_equal(s, s_implemented)), "Algorithm has failed!"

    circuit.done_editing()
    return circuit


def _compile_cnot_circuit_using_bge_algorithm(s, pspec, qubit_labels=None, clname=None, check=True):
    """
    Compile a CNOT circuit.

    Compilation uses a basic Gaussian elimination algorithm, that uses CNOT to perform row-reduction
    on the upper LHS (or lower RHS) of `s`. This algorithm does not take device connectivity into account.

    See the docstring of `compile_cnot_circuit()` for information on the parameters. This function
    should normally be accessed via `compile_cnot_circuit()`.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers, that represents the CNOT circuit. This should
        be a (2x2) block-diagonal matrix, whereby the upper left block is any invertable transformation over
        [0,1]^n, and the lower right block is the inverse transpose of this transformation.

    pspec : ProcessorSpec
        An nbar-qubit ProcessorSpec object that encodes the device that the CNOT circuit is being compiled
        for, where nbar >= n. The algorithm is takes into account the connectivity of the device as specified
        by pspec.qubitgraph. If nbar > n it is necessary to provide `qubit_labels`, to specify which qubits in
        `pspec` the CNOT circuit acts on (all other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input CNOT circuit needs to be ``padded'' to be the identity
        on those qubits). The ordering of the qubits in `s` is assumed to be the same as that in the list
        pspec.qubit_labels, unless `qubit_labels` is specified. Then, the ordering is taken w.r.t the ordering of
        the list `qubit_labels`.

    qubit_labels : list, optional
        Required if the CNOT circuit to compile is over less qubits than in `pspec`. In this case this is a
        list of the qubits to compile the CNOT circuit for; it should be a subset of the elements of
        pspec.qubit_labels. The ordering of the qubits in `s` is taken w.r.t the ordering of this list.

    clname : str, optional
        Unused.

    check : bool, optional
        Whether to check the algorithm was successful.

    Returns
    -------
    Circuit
        A circuit implementing the input CNOT circuit.
    """
    # Basic gaussian elimination algorithm
    if qubit_labels is not None:
        qubit_labels = list(qubit_labels)
    else:
        qubit_labels = pspec.qubit_labels

    n = _np.shape(s)[0] // 2

    assert(len(qubit_labels) == n), "The CNOT circuit is over the wrong number of qubits!"
    # We can just use this more general function for this task.
    sout, instructions, success = _submatrix_gaussian_elimination_using_cnots(s, 'row', 'UL', qubit_labels)
    assert(_np.array_equal(sout, _np.identity(2 * n, int))
           ), "Algorithm has failed! Perhaps the input wasn't a CNOT circuit."
    # The instructions returned are for mapping s -> I, so we need to reverse them.
    instructions.reverse()
    circuit = _Circuit(gatesring=instructions, line_labels=qubit_labels).parallelize()
    if check:
        s_implemented, p_implemented = _symp.symplectic_rep_of_clifford_circuit(circuit)
        assert(_np.array_equal(s_implemented, s)), "Algorithm has failed! Perhaps the input wasn't a CNOT circuit."
    return circuit


def _add_cnot(qubitgraph, controllabel, targetlabel):
    """
    A helper function for CNOT circuit compilations.

    Returns an instruction list that is CNOTs along the shortest path from the qubits with
    label `controlabel` and `targetlabel`.

    Parameters
    ----------
    qubitgraph : QubitGraph
        The qubit graph from which the shortest path between the qubits is extracted.

    controllabel : str
        The label for the control qubit.

    targetlabel : str
        The label for the target qubit.

    Returns
    -------
    list
        The list of instructions for implementing the requested CNOT via the available
        CNOT gates, as specified by `qubitgraph`.
    """
    def add_cnots_along_path(path):
        # Adds the CNOTs along a given path.
        for currentqubit, nextqubit in path: instructionlist.append(_Label('CNOT', (currentqubit, nextqubit)))

    instructionlist = []
    path = _copy.copy(qubitgraph.shortest_path_edges(controllabel, targetlabel))
    add_cnots_along_path(path)
    path.reverse()
    add_cnots_along_path(path[1:])
    path.reverse()
    add_cnots_along_path(path[1:])
    path.reverse()
    add_cnots_along_path(path[1:-1])

    return instructionlist

# This algorithm is *not* to be improved, at least currently. Because this is the algorithm
# we used in the "direct RB" paper, so it needs to be available to reproduce those results.
# The OiCAGE algorithm is there to be an improved version of this algorithm (currently that
# algorithm is only slightly better on average than this one).


def _compile_cnot_circuit_using_ocage_algorithm(s, pspec, qubitorder, qubit_labels=None, check=True,
                                                respect_connectivity=True):
    """
    An ordered and connectivity-adjusted Gaussian-elimination (OCAGE) algorithm for compiling a CNOT circuit.

    The algorithm takes as input a symplectic matrix `s`, that defines the action of a CNOT circuit, and it
    generates a CNOT circuit (converted to a native model, if requested) that implements the same unitary.

    The algorithm works by mapping s -> identity using CNOTs acting from the LHS and RHS. I.e., it finds two
    CNOT circuits Ccnot1, Ccnot2 such that symp(Ccnot1) * s * symp(Ccnot2) = identity (where symp(c) is the
    symplectic matrix representing the circuit c). A circuit implementing s is then obtained as
    rev(Cnot1)rev(Cnot2) where rev(c) denotes the reverse of circuit c.

    To find such circuits Ccnot1/2 we use the following algorithm. We eliminate the qubits in the order
    specified, whereby "eliminating" a qubit means mapping the column and row of `s` associated with that
    qubit to the identity column and row. To eliminate the ith qubit in the list we:

    1. Look at the current value of s[i,i]. If s[i,i] = 1 continue. Else, find the closest qubit to
    do a CNOT between i and that qubit to make s[i,i] = 1.
    2. List the remaining qubits to eliminate (the i+1th qubit onwards), and going from the qubit
       in this list that is further from i to the closest implement the following steps:
        2.1. Denote this qubit by ii.
        2.1 If s[ii,i] = 0, pass. Else, map s[ii,i] -> 0 with the following method:
            (a) find the shortest path from i -> ii,
            (b) If the shortest path contains already eliminated qubits, using a LHS-action SWAP-like
                set of chains of CNOTs along the shortest path i -> ii to do a CNOT from i -> ii whilst
                leaving all other qubits unchanged. Then skip to 2.3.
            (c) If the shortest path doesn't contain already eliminated qubits, do LHS-action CNOTs
                between neighbouring qubits along this path to set all the s matrix elements in the
                column s[:,i] for the qubits along this path to 1.
            (d) Use the qubit next to ii in this path to set s[ii,i] using a LHS-action CNOT, and don't
                undo any of the changes to the other qubits along that path.
       2.3. If s[i,ii] = 0, pass. Else, map s[i,ii] -> 0 as in step 2.3 except that now we use RHS-action
            CNOTs.

    Steps 1 - 3 do not change the already eliminated qubits, so after steps 1 - 2 are repeated for each
    of the yet-to-be-eliminated qubits, s should be some smaller matrix s' between the yet-to-be-eliminated
    qubits embedded in a identity matrix on the already eliminated qubits.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers, that represents the CNOT circuit. This should
        be a (2x2) block-diagonal matrix, whereby the upper left block is any invertable transformation over
        [0,1]^n, and the lower right block is the inverse transpose of this transformation.

    pspec : ProcessorSpec
        An nbar-qubit ProcessorSpec object that encodes the device that the CNOT circuit is being compiled
        for, where nbar >= n. The algorithm is takes into account the connectivity of the device as specified
        by pspec.qubitgraph. The output circuit  is over the gates available  in this device is `clname` is
        not None. If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the qubits in `pspec`
        the CNOT circuit acts on (all other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input CNOT circuit needs to be ``padded'' to be the identity
        on those qubits). The ordering of the qubits in `s` is assumed to be the same as that in the list
        pspec.qubit_labels, unless `qubit_labels` is specified. Then, the ordering is taken w.r.t the ordering of
        the list `qubit_labels`.

    qubitorder : list
        A list of the qubit labels in the order in which they are to be eliminated. For auto-generated
        orderings use the compile_cnot_circuit() wrap-around function.

    qubit_labels : list, optional
        Required if the CNOT circuit to compile is over less qubits than in `pspec`. In this case this is a
        list of the qubits to compile the CNOT circuit for; it should be a subset of the elements of
        pspec.qubit_labels. The ordering of the qubits in `s` is taken w.r.t the ordering of this list.

    check : bool, optional
        Whether to check the algorithm was successful.

    respect_connectivity : bool, optional
        This algorithm takes the connectivity into account, but sometimes resorts to 'non-local' CNOTs. If
        `respect_connectivity` is True these gates are re-expressed over the available gates using a SWAP-like
        decomposition. If False, the algorithm does not compile these gates automatically. However, they will
        still be converted to native gates from `pspec` if `clname` is True, using whatever the user specified
        algorithm for compiling non-local CNOT gates this implies.

    Returns
    -------
    Circuit
        A circuit implementing the input CNOT circuit.
    """
    # The number of qubits the CNOT circuit is over.
    n = _np.shape(s)[0] // 2
    # The list of operations that we do from the LHS of the input, to map it to the identity opererator.
    rowaction_instructionlist = []
    # The list of operations that we do from the RHS of the input, to map it to the identity opererator.
    columnaction_instructionlist = []
    # The qubits remaining to be eliminated.
    remaining_qubits = _copy.copy(qubitorder)
    # This is the current `s` matrix, after the operations so far recorded have been applied to the input `s`.
    # We only correctly update the (nxn) matrix in the top LHS of `sout`, because the lower RHS of the matrix
    # will just be the transpose inverse of this, so it's a waste of time to update it properly.
    sout = s.copy()
    # `allqubits` is a list of all the qubit labels, where the ith index of `s` corresponds to the ith qubit label.
    if qubit_labels is not None: allqubits = qubit_labels
    else: allqubits = pspec.qubit_labels

    # Find the correct qubit graph to take into account.
    if qubit_labels is None: qubitgraph = pspec.qubitgraph
    else: qubitgraph = pspec.qubitgraph.subgraph(qubit_labels)
    nodenames = qubitgraph.node_names()

    # Find the distances and the shortest path predecessor matrix for this set of qubits.
    # The indexing here is w.r.t. the order of nodenames.
    distances = qubitgraph.shortest_path_distance_matrix()
    shortestpathmatrix = qubitgraph.shortest_path_predecessor_matrix()

    # Loop through the qubits and eliminate them in turn.
    for k in range(n):

        # The label of the next qubit to eliminate.
        q = qubitorder[k]
        # The index (w.r.t s) of the next qubit to eliminate.
        qindex = allqubits.index(q)
        # The distances from this qubit to the other qubits.
        distances_to_qubit_q = distances[:, nodenames.index(q)].copy()

        # The algorithm requires this element to be 1 at each round, so if it isn't use a CNOT from
        # the nearest qubit to make it 1.
        if sout[qindex, qindex] == 0:
            found = False
            dis = list(distances_to_qubit_q.copy())
            # Go through all the qubits from closest to farthest to see if each one can be used
            # to set sout[i,i] = 1.
            for kk in range(n):
                # Find the index of the closest qubit.
                qqgraphindex = dis.index(min(dis))
                qq = nodenames[qqgraphindex]
                qqindex = allqubits.index(qq)
                # Check it's one of the remaining qubits, as otherwise using it here will ruin
                # things. It also cannot be i, obviously.
                if qq in remaining_qubits and qq != q:
                    # We first look to see if a row-action CNOT will map sout[i,i] -> 1.
                    if sout[qqindex, qindex] == 1:
                        if not respect_connectivity:
                            rowaction_instructionlist.append(_Label('CNOT', (qq, q)))
                        else:
                            rowaction_instructionlist += _add_cnot(qubitgraph, qq, q)
                        # Do the row-action of CNOT
                        sout[qindex, :] = sout[qindex, :] ^ sout[qqindex, :]
                        found = True
                    # We then look to see if a column-action CNOT will map sout[i,i] -> 1.
                    elif sout[qindex, qqindex] == 1:
                        if not respect_connectivity:
                            columnaction_instructionlist.append(_Label('CNOT', (q, qq)))

                        else:
                            columnaction_instructionlist += _add_cnot(qubitgraph, q, qq)
                        # Do the column action of sout.
                        sout[:, qindex] = sout[:, qindex] ^ sout[:, qqindex]
                        found = True
                # If success then we leave the loop and go onto the main part of this elimination round.
                if found:
                    break
                else:
                    # Set the distance of i in this temp array to inf, as we've now tried this qubit so
                    # we don't want to select it again in the next round of the loop.
                    dis[qqgraphindex] = _np.inf

            # It should always be possible to map s[i,i] -> 1, so if we haven't managed to something has gone wrong.
            assert(found is True), 'CNOT compilation algorithm failed! Perhaps the input was invalid.'

        # This is the list of all the qubits that qubit i will need to interact with this round. (except where
        # we resort to SWAP-like methods).
        remaining_Qs_for_round = _copy.copy(remaining_qubits)
        # We don't need to interact a qubit with itself, so delete ilabel from this list.
        del remaining_Qs_for_round[remaining_Qs_for_round.index(q)]

        # Go through and remove every qubit from this list.
        while len(remaining_Qs_for_round) > 0:

            # complement of `remaining_qubits` = "eliminated qubits"
            eliminated_qubits = set(allqubits) - set(remaining_qubits)

            # Find the most distant qubit still to be dealt with in this round
            mostdistantQ_found = False
            while not mostdistantQ_found:
                mostdistantQ_graphindex = _np.argmax(distances_to_qubit_q)
                mostdistantQ = nodenames[mostdistantQ_graphindex]
                # If the this index is for a qubit we need to deal with in this round we move on.
                if mostdistantQ in remaining_Qs_for_round:
                    mostdistantQ_found = True
                    mostdistantQ_index = allqubits.index(mostdistantQ)
                # If not, we set the distance to -1 so that we don't select this index in the next iteration, either
                # of search for the next most distance qubit or in the next round of the `while` loop.
                else: distances_to_qubit_q[mostdistantQ_graphindex] = -1

            # We must set out[mostdistantQ_index,i] = 0. There is no need to do anything here if that alreadys holds.
            if sout[mostdistantQ_index, qindex] == 1:
                # Find the shortest path out from i to mostdistantQ_index, and do CNOTs to make that all 1s. If it
                # includes eliminated qubits we just do the CNOT between them, as we can't use the more clever
                # implementation below where we implement a CNOT + screw with the other qubits that aren't yet
                # eliminated. This CNOT won't respect connectivity but we can enforce that later.

                if qubitgraph.shortest_path_intersect(mostdistantQ, q, eliminated_qubits):

                    if not respect_connectivity: rowaction_instructionlist.append(_Label('CNOT', (q, mostdistantQ)))
                    else: rowaction_instructionlist += _add_cnot(qubitgraph, q, mostdistantQ)
                    sout[mostdistantQ_index, :] = sout[qindex, :] ^ sout[mostdistantQ_index, :]

                # If the shortest path doesn't include eliminated qubits, we can use a method for mapping
                # out[mostdistantQ_index,i] -> 0 that isn't equivalent to only doing a CNOT between i and
                # `mostdistantQ_index`.
                else:
                    # First, we make sure that all of the values in the chain from i to `mostdistantQ_index` have are 1,
                    # converting them to 1 where necessary.
                    for nextqubit, currentqubit in reversed(qubitgraph.shortest_path_edges(mostdistantQ, q)):
                        nextqubitindex = allqubits.index(nextqubit)
                        if sout[nextqubitindex, qindex] == 0:
                            rowaction_instructionlist.append(_Label('CNOT', (currentqubit, nextqubit)))
                            currentqubitindex = allqubits.index(currentqubit)
                            sout[nextqubitindex, :] = sout[nextqubitindex, :] ^ sout[currentqubitindex, :]

                    # Then we set the `mostdistantQ_index` s-matrix element to 0 (but don't change the others).
                    quse_graphindex = shortestpathmatrix[nodenames.index(q), mostdistantQ_graphindex]
                    quse = nodenames[quse_graphindex]
                    quseindex = allqubits.index(quse)
                    rowaction_instructionlist.append(_Label('CNOT', (quse, mostdistantQ)))
                    sout[mostdistantQ_index, :] = sout[quseindex, :] ^ sout[mostdistantQ_index, :]

            # We must set out[i,mostdistantQ_index] = 0. There is no need to do anything here if that alreadys holds.
            # This follows the method above, except that now we use column-action CNOTs.
            if sout[qindex, mostdistantQ_index] == 1:

                if qubitgraph.shortest_path_intersect(mostdistantQ, q, eliminated_qubits):

                    if not respect_connectivity: columnaction_instructionlist.append(_Label('CNOT', (mostdistantQ, q)))
                    else: columnaction_instructionlist += _add_cnot(qubitgraph, mostdistantQ, q)
                    sout[:, mostdistantQ_index] = sout[:, qindex] ^ sout[:, mostdistantQ_index]
                else:
                    for nextqubit, currentqubit in reversed(qubitgraph.shortest_path_edges(mostdistantQ, q)):
                        nextqubitindex = allqubits.index(nextqubit)
                        if sout[qindex, nextqubitindex] == 0:
                            columnaction_instructionlist.append(_Label('CNOT', (nextqubit, currentqubit)))
                            currentqubitindex = allqubits.index(currentqubit)
                            sout[:, nextqubitindex] = sout[:, nextqubitindex] ^ sout[:, currentqubitindex]

                    quse_graphindex = shortestpathmatrix[nodenames.index(q), mostdistantQ_graphindex]
                    quse = nodenames[quse_graphindex]
                    quseindex = allqubits.index(quse)
                    columnaction_instructionlist.append(_Label('CNOT', (mostdistantQ, quse)))
                    sout[:, mostdistantQ_index] = sout[:, quseindex] ^ sout[:, mostdistantQ_index]

            # Delete the farthest qubit from the list -- `q` and this qubit will nolonger need to interact, so it is
            # done for this round in which we eliminated the qubit `q`.
            del remaining_Qs_for_round[remaining_Qs_for_round.index(mostdistantQ)]
            # And set it's distance to -1, so that in the next round we find the next farthest qubit.
            distances_to_qubit_q[mostdistantQ_graphindex] = -1

        # Remove `q` from the remaining qubits list, because we've eliminated it.
        del remaining_qubits[remaining_qubits.index(q)]

    # We reverse the row-action list, append it to the column-action list, and then we have a sequence
    # that implements this CNOT circuit.
    rowaction_instructionlist.reverse()
    columnaction_instructionlist
    full_instructionlist = columnaction_instructionlist + rowaction_instructionlist
    # We convert it to a circuit
    cnot_circuit = _Circuit(layer_labels=full_instructionlist, line_labels=allqubits)

    if check:
        s_implemented, p_implemented = _symp.symplectic_rep_of_clifford_circuit(cnot_circuit)
        # This only checks its correct upto the phase vector, so that we can use the algorithm
        # with paulieq compilations and it won't fail when check is True.
        assert(_np.array_equal(s, s_implemented)), "Algorithm has failed!"

    return cnot_circuit


def _compile_cnot_circuit_using_oicage_algorithm(s, pspec, qubitorder, qubit_labels=None, clname=None, check=True):
    """
    An improved, ordered and connectivity-adjusted Gaussian-elimination (OiCAGE) algorithm for compiling a CNOT circuit.

    This is a *slight* improvement (for some CNOT circuits), on the algorithm in
    :function:`_compile_cnot_circuit_using_ocage_algorithm()`, which is the meaning of the "improved". See the docstring
    for that function for information on the parameters of this function and the basic outline of the algorithm.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers, that represents the CNOT circuit. This should
        be a (2x2) block-diagonal matrix, whereby the upper left block is any invertable transformation over
        [0,1]^n, and the lower right block is the inverse transpose of this transformation.

    pspec : ProcessorSpec
        An nbar-qubit ProcessorSpec object that encodes the device that the CNOT circuit is being compiled
        for, where nbar >= n. The algorithm is takes into account the connectivity of the device as specified
        by pspec.qubitgraph. If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the
        qubits in `pspec` the CNOT circuit acts on (all other qubits will not be part of the returned circuit,
        regardless of whether that means an over-head is required to avoid using gates that act on those qubits.
        If these additional qubits should be used, then the input CNOT circuit needs to be ``padded'' to be the identity
        on those qubits). The ordering of the qubits in `s` is assumed to be the same as that in the list
        pspec.qubit_labels, unless `qubit_labels` is specified. Then, the ordering is taken w.r.t the ordering of
        the list `qubit_labels`.

    qubitorder : list
        A list of the qubit labels in the order in which they are to be eliminated. For auto-generated
        orderings use the compile_cnot_circuit() wrap-around function.

    qubit_labels : list, optional
        Required if the CNOT circuit to compile is over less qubits than in `pspec`. In this case this is a
        list of the qubits to compile the CNOT circuit for; it should be a subset of the elements of
        pspec.qubit_labels. The ordering of the qubits in `s` is taken w.r.t the ordering of this list.

    clname : str, optional
        Unused.

    check : bool, optional
        Whether to check the algorithm was successful.

    Returns
    -------
    Circuit
        A circuit implementing the input CNOT circuit.
    """
    # The number of qubits the CNOT circuit is over.
    n = _np.shape(s)[0] // 2
    # The list of operations that we do from the LHS of the input, to map it to the identity opererator.
    rowaction_instructionlist = []
    # The list of operations that we do from the RHS of the input, to map it to the identity opererator.
    columnaction_instructionlist = []
    # The qubits remaining to be eliminated.
    remaining_qubits = _copy.copy(qubitorder)
    # This is the current `s` matrix, after the operations so far recorded have been applied to the input `s`.
    # We only correctly update the (nxn) matrix in the top LHS of `sout`, because the lower RHS of the matrix
    # will just be the transpose inverse of this, so it's a waste of time to update it properly.
    sout = s.copy()
    # `allqubits` is a list of all the qubit labels, where the ith index of `s` corresponds to the ith qubit label.
    if qubit_labels is not None:
        allqubits = qubit_labels
    else:
        allqubits = pspec.qubit_labels

    # Find the correct qubit graph to take into account.
    if qubit_labels is None:
        qubitgraph = pspec.qubitgraph
    else:
        qubitgraph = pspec.qubitgraph.subgraph(qubit_labels)

    # Loop through the qubits and eliminate them in turn.
    for k in range(n):

        # The label of the next qubit to eliminate.
        q = qubitorder[k]
        # The index of the next qubit to eliminate.
        qindex = allqubits.index(q)
        # A qubit graph over the remaining qubits
        rQsgraph = qubitgraph.subgraph(remaining_qubits)
        rQsgraph_llist = list(rQsgraph.node_names())
        q_rQSgraph_index = rQsgraph_llist.index(q)

        # The distances from this qubit to the other qubits.
        distances_to_qubit_q = rQsgraph.shortest_path_distance_matrix()[:, q_rQSgraph_index].copy()

        # The algorithm requires this element to be 1 at each round, so if it isn't use a CNOT from
        # the nearest qubit to make it 1.
        if sout[qindex, qindex] == 0:
            found = False
            dis = list(distances_to_qubit_q.copy())
            # Go through all the qubits from closest to farthest to see if each one can be used
            # to set sout[i,i] = 1.
            for kk in range(n - k):
                # Find the index of the closest qubit.
                qq_rQsgraph_index = dis.index(min(dis))
                qq = rQsgraph_llist[qq_rQsgraph_index]
                qqindex = allqubits.index(qq)
                #print(qq_rQsgraph_index,qq,qqindex)
                # Check it's one of the remaining qubits, as otherwise using it here will ruin
                # things. It also cannot be i, obviously.
                if qq != q:
                    # We first look to see if a row-action CNOT will map sout[i,i] -> 1.
                    if sout[qqindex, qindex] == 1:
                        #future : try this method as well, and pick the better one.
                        #rowaction_instructionlist.append(_Label('CNOT',(qq,q)))
                        #sout[qindex,:] = sout[qindex,:] ^ sout[qqindex,:]
                        found = True
                        # Find the shortest path from qq to q, and do CNOTs to make that all 1s.
                        for control, target in rQsgraph.shortest_path_edges(qq, q):
                            controlindex = allqubits.index(control)
                            targetindex = allqubits.index(target)
                            rowaction_instructionlist.append(_Label('CNOT', (control, target)))
                            sout[targetindex, :] = sout[targetindex, :] ^ sout[controlindex, :]

                    # We then look to see if a column-action CNOT will map sout[i,i] -> 1.
                    elif sout[qindex, qqindex] == 1:
                        #future : try this method as well, and pick the better one.
                        #columnaction_instructionlist.append(_Label('CNOT',(q,qq)))
                        #sout[:,qindex] = sout[:,qindex] ^ sout[:,qqindex]
                        found = True
                        # Find the shortest path from q to qq, and do CNOTs to make that all 1s.
                        for target, control in rQsgraph.shortest_path_edges(qq, q):
                            targetindex = allqubits.index(target)
                            controlindex = allqubits.index(control)
                            # This should always hold, otherwise there is a closer qubit we could have used.
                            #if sout[nextqubitindex,qindex] == 0:
                            columnaction_instructionlist.append(_Label('CNOT', (control, target)))
                            sout[:, controlindex] = sout[:, targetindex] ^ sout[:, controlindex]

                # If success then we leave the loop and go onto the main part of this elimination round.
                if found:
                    break
                else:
                    # Set the distance of i in this temp array to inf, as we've now tried this qubit so
                    # we don't want to select it again in the next round of the loop.
                    dis[qq_rQsgraph_index] = _np.inf

            # It should always be possible to map s[i,i] -> 1, so if we haven't managed to something has gone wrong.
            assert(found is True), 'CNOT compilation algorithm failed! Perhaps the input was invalid.'

        # This is the list of all the qubits that qubit i will need to interact with this round.
        remaining_Qs_for_round = _copy.copy(remaining_qubits)
        # We don't need to interact a qubit with itself, so delete qlabel from this list.
        del remaining_Qs_for_round[remaining_Qs_for_round.index(q)]

        # Go through and remove every qubit from this list.
        while len(remaining_Qs_for_round) > 0:

            # Find the most distant qubit still to be dealt with in this round
            mostdistantQ_rQsgraph_index = _np.argmax(distances_to_qubit_q)
            mostdistantQ = rQsgraph_llist[mostdistantQ_rQsgraph_index]
            mostdistantQ_index = allqubits.index(mostdistantQ)

            # We must set out[mostdistantQ_index,i] = 0. There is no need to do anything here if that alreadys holds.
            if sout[mostdistantQ_index, qindex] == 1:
                # Find the shortest path out from i to mostdistantQ_index, and do CNOTs to make that all 1s.
                for nextqubit, currentqubit in reversed(rQsgraph.shortest_path_edges(mostdistantQ, q)):
                    nextqubitindex = allqubits.index(nextqubit)
                    if sout[nextqubitindex, qindex] == 0:
                        rowaction_instructionlist.append(_Label('CNOT', (currentqubit, nextqubit)))
                        currentqubitindex = allqubits.index(currentqubit)
                        sout[nextqubitindex, :] = sout[nextqubitindex, :] ^ sout[currentqubitindex, :]

                # Then we set the `mostdistantQ_index` s-matrix element to 0 (but don't change the others).
                quse = rQsgraph.shortest_path_edges(q, mostdistantQ)[-1][0]
                quseindex = allqubits.index(quse)
                rowaction_instructionlist.append(_Label('CNOT', (quse, mostdistantQ)))
                sout[mostdistantQ_index, :] = sout[quseindex, :] ^ sout[mostdistantQ_index, :]

            # We must set out[i,mostdistantQ_index] = 0. There is no need to do anything here if that alreadys holds.
            if sout[qindex, mostdistantQ_index] == 1:
                # Find the shortest path out from i to mostdistantQ_index, and do CNOTs to make that all 1s.
                for nextqubit, currentqubit in reversed(rQsgraph.shortest_path_edges(mostdistantQ, q)):
                    nextqubitindex = allqubits.index(nextqubit)
                    if sout[qindex, nextqubitindex] == 0:
                        columnaction_instructionlist.append(_Label('CNOT', (nextqubit, currentqubit)))
                        currentqubitindex = allqubits.index(currentqubit)
                        sout[:, nextqubitindex] = sout[:, nextqubitindex] ^ sout[:, currentqubitindex]
                # Then we set the `mostdistantQ_index` s-matrix element to 0 (but don't change the others).
                quse = rQsgraph.shortest_path_edges(q, mostdistantQ)[-1][0]
                quseindex = allqubits.index(quse)
                columnaction_instructionlist.append(_Label('CNOT', (mostdistantQ, quse)))
                sout[:, mostdistantQ_index] = sout[:, quseindex] ^ sout[:, mostdistantQ_index]

            # Delete the farthest qubit from the list -- `i` and this qubit will nolonger need to interact, so it is
            # done for this round in which we eliminated the qubit with index `i`.
            del remaining_Qs_for_round[remaining_Qs_for_round.index(mostdistantQ)]
            # And set it's distance to -1, so that in the next round we find the next farthest qubit.
            distances_to_qubit_q[mostdistantQ_rQsgraph_index] = -1

        # Remove `i` from the remaining qubits list, because we've eliminated it.
        del remaining_qubits[remaining_qubits.index(q)]

    # We reverse the row-action list, append it to the column-action list, and then we have a sequence
    # that implements this CNOT circuit.
    rowaction_instructionlist.reverse()
    columnaction_instructionlist
    full_instructionlist = columnaction_instructionlist + rowaction_instructionlist
    # We convert it to a circuit
    cnot_circuit = _Circuit(layer_labels=full_instructionlist, line_labels=allqubits)

    if check:
        s_implemented, p_implemented = _symp.symplectic_rep_of_clifford_circuit(cnot_circuit, pspec=pspec)
        # This only checks its correct upto the phase vector, so that we can use the algorithm
        # with paulieq compilations and it won't fail when check is True.
        assert(_np.array_equal(s, s_implemented)), "Algorithm has failed!"

    return cnot_circuit


def compile_stabilizer_state(s, p, pspec, qubit_labels=None, iterations=20, paulirandomize=False,
                             algorithm='COiCAGE', aargs=[], costfunction='2QGC:10:depth:1'):
    """
    Generates a circuit to create the stabilizer state from the standard input state |0,0,0,...>.

    The stabilizer state is specified by `s` and `p`. The circuit returned is over the model of
    the processor spec `pspec`.  See :function:`compile_stabilizer_state()` for the inverse of this.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers. This is a symplectic matrix representing
        any Clifford gate that, when acting on |0,0,0,...>, generates the desired stabilizer state.
        So `s` is not unique.

    p : array over [0,1]
        A length-2n vector over [0,1,2,3] that, together with s, defines a valid n-qubit Clifford
        gate. This phase vector matrix should, together with `s`, represent any Clifford gate that,
        when acting on |0,0,0,...>, generates the desired stabilizer state. So `p` is not unique.

    pspec : ProcessorSpec, optional
        An nbar-qubit ProcessorSpec object that encodes the device that the stabilizer is being compiled
        for, where nbar >= n. If this is specified, the output circuit is over the gates available
        in this device. If this is None, the output circuit is over the "canonical" model of CNOT gates
        between all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation. In most circumstances, the output will be more useful if a
        ProcessorSpec is provided.

        If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the qubits in `pspec`
        the stabilizer is over. (All other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input (s,p) needs to be ``padded'' to be the identity
        on those qubits).

        The ordering of the indices in (`s`,`p`) is w.r.t to ordering of the qubit labels in pspec.qubit_labels,
        unless `qubit_labels` is specified. Then, the ordering is taken w.r.t the ordering of the list `qubit_labels`.

    qubit_labels : List, optional
        Required if the stabilizer state is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of
        pspec.qubit_labels. The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    iterations : int, optional
        This algorithm is randomized. This is the number of iterations used in the algorithm.
        the time taken by this function increases linearly with `iterations`. Increasing `iterations`
        will often improve the obtained compilation (the "cost" of the obtained circuit, as specified
        by `costfunction` may decrease towards some asymptotic value).

    paulirandomize : bool, optional
        If True then independent, uniformly random Pauli layers (a Pauli on each qubit) are inserted in between
        every layer in the circuit. These Paulis are then compiled into the gates in `pspec`, if `pspec` is provided.
        That is, this Pauli-frame-randomizes / Pauli-twirls the internal layers of this circuit. This can
        be useful for preventing coherent addition of errors in the circuit.

    algorithm : str, optional
        Our algorithm finds a circuit consisting of 1Q-gates - a CNOT circuit - 1Q-gates. The CNOT circuit
        is found using Gaussian elimination, and it can then be recompiled using a CNOT-circuit compiler.
        `algorithm` specifies the CNOT-compilation algorithm to use. The allowe values are all those algorithms
        that permisable in the `compile_cnot_circuit()` function. See the docstring of that function for more
        information. The default is likely to be the best out of the in-built CNOT compilers under most
        circumstances.

    aargs : list, optional
        If the CNOT compilation algorithm can take optional arguments, these are specified here. This is passed
        to compile_cnot_circuit() as `aarg`.

    costfunction : function or string, optional
        If a function, it is a function that takes a circuit and `pspec` as the first and second inputs and
        returns a 'cost' (a float) for the circuit. The circuit input to this function will be over the gates in
        `pspec`, if a `pspec` has been provided, and as described above if not. This costfunction is used to decide
        between different compilations when randomized algorithms are used: the lowest cost circuit is chosen. If
        a string it must be one of:

            - '2QGC' : the cost of the circuit is the number of 2-qubit gates it contains.
            - 'depth' : the cost of the circuit is the depth of the circuit.
            - '2QGC:x:depth:y' : the cost of the circuit is x * the number of 2-qubit gates in the circuit +
                y * the depth of the circuit, where x and y are integers.

    Returns
    -------
    Circuit
        A circuit that creates the specified stabilizer state from |0,0,0,...>
    """
    assert(_symp.check_valid_clifford(s, p)), "The input s and p are not a valid clifford."

    if qubit_labels is None: qubit_labels = pspec.qubit_labels

    n = _np.shape(s)[0] // 2
    assert(n == len(qubit_labels)), \
        "The input `s` is the wrong size for the number of qubits specified by `pspec` or `qubit_labels`!"

    # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str): costfunction = _create_standard_costfunction(costfunction)

    #Import the single-qubit Cliffords up-to-Pauli algebra
    oneQgate_relations = _symp.one_q_clifford_symplectic_group_relations()
    # The best 2Q gate count so far found.
    mincost = _np.inf
    failcount, i = 0, 0

    # Repeatedly find compilations for the symplectic, and pick the best one.
    while i < iterations:

        try:
            tc, tcc = compile_conditional_symplectic(
                s, pspec, qubit_labels=qubit_labels, calg=algorithm, cargs=aargs, check=False)
            tc = tc.copy(editable=True)
            i += 1
            # Do the depth-compression *before* changing gate library
            tc.compress_depth_inplace(one_q_gate_relations=oneQgate_relations, verbosity=0)
            tc.change_gate_library(pspec.compilations['paulieq'])  # ,identity=pspec.identity)
            cost = costfunction(tc, pspec)
            # If this is the best circuit so far, then save it.
            if cost < mincost:
                circuit = tc.copy()
                check_circuit = tcc.copy(editable=True)
                mincost = cost
        except:
            failcount += 1

        assert(failcount <= 5 * iterations), \
            ("Randomized compiler is failing unexpectedly often. "
             "Perhaps input ProcessorSpec is not valid or does not contain the neccessary information.")

    if paulirandomize:
        paulilist = ['I', 'X', 'Y', 'Z']
        d = circuit.depth()
        for i in range(1, d + 1):
            pcircuit = _Circuit(layer_labels=[_Label(paulilist[_np.random.randint(4)], qubit_labels[k])
                                              for k in range(n)],
                                line_labels=qubit_labels, editable=True)
            pcircuit.change_gate_library(pspec.compilations['absolute'])  # ,identity=pspec.identity)
            circuit.insert_circuit(pcircuit, d - i)

    implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)

    check_circuit.append_circuit(circuit)
    # Add CNOT into the dictionary, because the gates in check_circuit are 'CNOT'.
    sreps = pspec.models['clifford'].compute_clifford_symplectic_reps()
    sreps['CNOT'] = (_np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1],
                                [0, 0, 0, 1]], int), _np.array([0, 0, 0, 0], int))
    implemented_scheck, implemented_pcheck = _symp.symplectic_rep_of_clifford_circuit(check_circuit, srep_dict=sreps)

    # We should have a circuit with the same RHS as `s`. This is only true when we have prefixed the additional
    # CNOT circuit returned by `compile_conditional_symplectic`. Testing `circuit` is correct directly is more complex.
    assert(_np.array_equal(implemented_scheck[:, n:2 * n], s[:, n:2 * n])
           ), "Algorithm has failed! Perhaps the input was not a symplectic matrix."

    # Find the needed Pauli at the end.
    pauli_layer = _symp.find_postmultipled_pauli(implemented_scheck, implemented_pcheck, p, qubit_labels=qubit_labels)
    paulicircuit = _Circuit(layer_labels=pauli_layer, line_labels=qubit_labels, editable=True)
    paulicircuit.change_gate_library(pspec.compilations['absolute'])  # ,identity=pspec.identity)
    circuit.append_circuit(paulicircuit)

    if not paulirandomize: circuit.compress_depth_inplace(one_q_gate_relations=pspec.oneQgate_relations, verbosity=0)

    circuit.done_editing()
    return circuit


def compile_stabilizer_measurement(s, p, pspec, qubit_labels=None, iterations=20, paulirandomize=False,
                                   algorithm='COCAGE', aargs=[], costfunction='2QGC:10:depth:1'):
    """
    Generates a circuit to map the stabilizer state to the standard state |0,0,0,...>.

    The stabilizer state is specified by `s` and `p`.  The circuit returned is over the model of the
    processor spec `pspec`.  See :function"`compile_stabilizer_state()` for the inverse of this. So,
    this circuit followed by a Z-basis measurement can be used to simulate a projection onto the
    stabilizer state C|0,0,0,...> where C is the Clifford represented by `s` and `p`.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers. This is a symplectic matrix representing
        any Clifford gate that, when acting on |0,0,0,...>, generates the stabilizer state that we need
        to map to |0,0,0,...>. So `s` is not unique.

    p : array over [0,1]
        A length-2n vector over [0,1,2,3] that, together with s, defines a valid n-qubit Clifford
        gate. This phase vector matrix should, together with `s`, represent any Clifford gate that,
        when acting on |0,0,0,...>, generates the stabilizer state that we need to map to |0,0,0,...>.
        So `p` is not unique.

    pspec : ProcessorSpec, optional
        An nbar-qubit ProcessorSpec object that encodes the device that the stabilizer is being compiled
        for, where nbar >= n. If this is specified, the output circuit is over the gates available
        in this device. If this is None, the output circuit is over the "canonical" model of CNOT gates
        between all qubits, consisting of "H", "HP", "PH", "HPH", "I", "X", "Y" and "Z", which is the set
        used internally for the compilation. In most circumstances, the output will be more useful if a
        ProcessorSpec is provided.

        If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the qubits in `pspec`
        the stabilizer is over. (All other qubits will not be part of the returned circuit, regardless of
        whether that means an over-head is required to avoid using gates that act on those qubits. If these
        additional qubits should be used, then the input (s,p) needs to be ``padded'' to be the identity
        on those qubits).

        The ordering of the indices in (`s`,`p`) is w.r.t to ordering of the qubit labels in pspec.qubit_labels,
        unless `qubit_labels` is specified. Then, the ordering is taken w.r.t the ordering of the list `qubit_labels`.

    qubit_labels : List, optional
        Required if the stabilizer state is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of
        pspec.qubit_labels. The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    iterations : int, optional
        This algorithm is randomized. This is the number of iterations used in the algorithm.
        the time taken by this function increases linearly with `iterations`. Increasing `iterations`
        will often improve the obtained compilation (the "cost" of the obtained circuit, as specified
        by `costfunction` may decrease towards some asymptotic value).

    paulirandomize : bool, optional
        If True then independent, uniformly random Pauli layers (a Pauli on each qubit) are inserted in between
        every layer in the circuit. These Paulis are then compiled into the gates in `pspec`, if `pspec` is provided.
        That is, this Pauli-frame-randomizes / Pauli-twirls the internal layers of this circuit. This can
        be useful for preventing coherent addition of errors in the circuit.

    algorithm : str, optional
        Our algorithm finds a circuit consisting of 1Q-gates - a CNOT circuit - 1Q-gates. The CNOT circuit
        is found using Gaussian elimination, and it can then be recompiled using a CNOT-circuit compiler.
        `algorithm` specifies the CNOT-compilation algorithm to use. The allowe values are all those algorithms
        that permisable in the `compile_cnot_circuit()` function. See the docstring of that function for more
        information. The default is likely to be the best out of the in-built CNOT compilers under most
        circumstances.

    aargs : list, optional
        If the CNOT compilation algorithm can take optional arguments, these are specified here. This is passed
        to compile_cnot_circuit() as `aarg`.

    costfunction : function or string, optional
        If a function, it is a function that takes a circuit and `pspec` as the first and second inputs and
        returns a 'cost' (a float) for the circuit. The circuit input to this function will be over the gates in
        `pspec`, if a `pspec` has been provided, and as described above if not. This costfunction is used to decide
        between different compilations when randomized algorithms are used: the lowest cost circuit is chosen. If
        a string it must be one of:

            - '2QGC' : the cost of the circuit is the number of 2-qubit gates it contains.
            - 'depth' : the cost of the circuit is the depth of the circuit.
            - '2QGC:x:depth:y' : the cost of the circuit is x * the number of 2-qubit gates in the circuit +
                y * the depth of the circuit, where x and y are integers.

    Returns
    -------
    Circuit
        A circuit that maps the specified stabilizer state to |0,0,0,...>
    """
    assert(_symp.check_valid_clifford(s, p)), "The input s and p are not a valid clifford."

    if qubit_labels is not None: qubit_labels = qubit_labels
    else: qubit_labels = pspec.qubit_labels

    n = _np.shape(s)[0] // 2
    assert(n == len(qubit_labels)), \
        "The input `s` is the wrong size for the number of qubits specified by `pspec` or `qubit_labels`!"

    # Because we're compiling a measurement, we need a circuit to implement s inverse
    sin, pin = _symp.inverse_clifford(s, p)

    # If the costfunction is a string, create the relevant "standard" costfunction function.
    if isinstance(costfunction, str): costfunction = _create_standard_costfunction(costfunction)

    #Import the single-qubit Cliffords up-to-Pauli algebra
    oneQgate_relations = _symp.one_q_clifford_symplectic_group_relations()
    # The best 2Q gate count so far found.
    mincost = _np.inf
    failcount, i = 0, 0

    # Repeatedly find compilations for the symplectic, and pick the best one.
    while i < iterations:
        try:
            i += 1
            # Find a circuit to conditionally implement s, then reverse it to conditionally implement sin (all gates are
            # self-inverse up to Paulis in CNOT, H, and P).
            tc, tcc = compile_conditional_symplectic(
                s, pspec, qubit_labels=qubit_labels, calg=algorithm, cargs=aargs, check=False)
            tc = tc.copy(editable=True)
            tc.reverse_inplace()
            # Do the depth-compression *after* the circuit is reversed (after this, reversing circuit doesn't implement
            # inverse).
            tc.compress_depth_inplace(one_q_gate_relations=oneQgate_relations, verbosity=0)
            # Change into the gates of pspec.
            tc.change_gate_library(pspec.compilations['paulieq'])  # ,identity=pspec.identity)
            # If this is the best circuit so far, then save it.
            cost = costfunction(tc, pspec)
            if cost < mincost:
                circuit = tc.copy()
                check_circuit = tcc.copy(editable=True)
                mincost = cost
        except:
            failcount += 1

        assert(failcount <= 5 * iterations), \
            ("Randomized compiler is failing unexpectedly often. "
             "Perhaps input DeviceSpec is not valid or does not contain the neccessary information.")

    if paulirandomize:
        paulilist = ['I', 'X', 'Y', 'Z']
        d = circuit.depth()
        for i in range(0, d):
            pcircuit = _Circuit(layer_labels=[_Label(paulilist[_np.random.randint(4)], qubit_labels[k])
                                              for k in range(n)],
                                line_labels=qubit_labels, editable=True)
            pcircuit.change_gate_library(pspec.compilations['absolute'])  # ,identity=pspec.identity)
            circuit.insert_circuit(pcircuit, d - i)

    # We didn't reverse tcc, so reverse check_circuit now. This circuit contains CNOTs, and is the circuit we'd need
    # to do after `circuit` in order to correctly generate the top half of s inverse (which we don't actually need to
    # do), rather than only correctly do this up to a CNOT circuit.
    check_circuit.reverse_inplace()
    check_circuit.prefix_circuit(circuit)

    sreps = pspec.models['clifford'].compute_clifford_symplectic_reps()
    sreps['CNOT'] = (_np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1],
                                [0, 0, 0, 1]], int), _np.array([0, 0, 0, 0], int))
    #implemented_s, implemented_p = _symp.symplectic_rep_of_clifford_circuit(circuit, srep_dict=sreps)

    # Find the (s,p) implemented by the circuit when the additional CNOT circuit is at the end.
    implemented_scheck, implemented_pcheck = _symp.symplectic_rep_of_clifford_circuit(check_circuit, srep_dict=sreps)
    # Find the (s,p) that would be implemented if we did this all in reverse (+ gates conjugated).
    implemented_sin_check, implemented_pin_check = _symp.inverse_clifford(implemented_scheck, implemented_pcheck)
    # We check we have correctly implemented s inverse (this is only easily checked when the additional circuit is
    # there).
    assert(_np.array_equal(implemented_scheck[0:n, :], sin[0:n, :])
           ), "Algorithm has failed! Perhaps the input was not a symplectic matrix."
    # We need to put Paulis on the circuit. We'll prefix them, so we look to see what Paulis we'd need to post-fix if
    # the circuit was run in reverse (+ gates conjugated) to create a stabilizer state. This is then what we need to do
    # at the start of the circuit in the opposite direction, that we are using for the measurement.
    pauli_layer = _symp.find_postmultipled_pauli(
        implemented_sin_check, implemented_pin_check, p, qubit_labels=qubit_labels)
    # Get the Pauli layer as a circuit, find in pspec gate library, and prefix to current circuit.
    paulicircuit = _Circuit(layer_labels=pauli_layer, line_labels=qubit_labels, editable=True)
    paulicircuit.change_gate_library(pspec.compilations['absolute'])  # ,identity=pspec.identity)
    circuit.prefix_circuit(paulicircuit)
    # We can only do depth compression again if we haven't Pauli-randomized. Otherwise we'd potentially undo this
    # randomization.
    if not paulirandomize: circuit.compress_depth_inplace(one_q_gate_relations=pspec.oneQgate_relations, verbosity=0)

    circuit.done_editing()
    return circuit


def _convert_submatrix_to_echelon_form_using_cnots(s, optype, position, qubit_labels):
    """
    Converts a submatrix of `s` to row- or column-echelon form.

    Converts one of the 4 submatrices of the symplectic matrix `s` to row- or
    column-echelon form, using a CNOT circuit acting after or before this circuit.
    The conversion is only performed *if* the submatrix is invertable,
    and otherwise the function reports that this submatrix is not invertable.

    The function does not alter `s`, but rather returns an updated version of s.

    Parameters
    ----------
    s : np.array
        The (2n,2n) matrix over [0,1] that we are transforming one of the 4 (n,n) submatrices
        of to row/column echolon form.

    optype : 'row' or 'column'
        If 'row', we peform row-operation CNOTs (CNOTs from the LHS) to convert the the submatrix to row-echelon form,
        If 'column', we peform column-operation CNOTs (CNOTs from the RHS) to convert the submatrix to
        column-echelon form.

    position : 'UL', 'UR', 'LL', or 'LR'
        The submatrix to perform the row/column reduction on. 'UL' and 'UR' correspond to the upper left and
        right submatrices, respecively. 'LL' and 'LR' correspond to the lower left and right submatrices, respecively.

    qubit_labels : list
        The qubit labels corresponding to the indices of `s`. This is required because othewise
        it is ambigious as to what the 'name' of a qubit associated with each indices is, so it
        is not possible to return a suitable list of CNOTs.

    Returns
    -------
    np.array
        The updated s matrix. If the submatrix is not invertable, then the updated s at the point
        at which this becomes apparent is returned.
    bool
        True if the row/column reduction has managed to create a matrix with 1s on the diagonal, or
        equivalently when the submatrix is invertable (over [0,1] mod 2)
    list
        A list of the CNOT instructions used to convert s to row/column echolon form. If `optype`='row', this
        list is the CNOTs in the order that they should be applied *after* a unitary represented by
        `s` so that the composite unitary has row-echolon form in the specified submatrix of its `s` matrix.
        If `otptype `='column' then this list is the CNOTs in the order they should be applied *before* a
        unitary represented by `s` so that the composite unitary has row-echolon form in the specified
        submatrix of its `s` matrix.

        When the returned bool is False, this list is None.
    """
    sout = s.copy()
    n = _np.shape(sout)[0] // 2

    if position == 'UL': cs, rs, = 0, 0
    if position == 'LL': cs, rs, = 0, n
    if position == 'UR': cs, rs, = n, 0
    if position == 'LR': cs, rs, = n, n

    instruction_list = []

    for i in range(n):

        if sout[rs + i, cs + i] != 1:
            # If it's the last qubit, there is no row/column below/totheright to try swapping with.
            if i == n - 1: return sout, None, False
            # Otherwise, we try swapping the row/column with all row/columns below/totheright.
            for j in range(i + 1, n):
                # If it's a row op, the column is fixed to i (setting ci = i), and we vary the row
                # (setting ri = j): we're trying to add rows to make this below i the identity column.
                if optype == 'row': ci, ri = i, j
                # If it's a column op, the row is fixed to i (setting ri = j), and we vary the column
                # (setting ci = j): we're trying to add rows to make this to the RHS of i the identity row.
                elif optype == 'column': ci, ri = j, i

                if sout[rs + ri, cs + ci] == 1:
                    # We've found a row/column to swap with, so do so.
                    _symp.apply_internal_gate_to_symplectic(sout, 'SWAP', (i, j), optype=optype)
                    instruction_list.append(_Label('CNOT', (qubit_labels[i], qubit_labels[j])))
                    instruction_list.append(_Label('CNOT', (qubit_labels[j], qubit_labels[i])))
                    instruction_list.append(_Label('CNOT', (qubit_labels[i], qubit_labels[j])))
                    # Once we've found a row/column to swap with, we quit the loop.
                    break

                # If we've gone through all the qubits and failed to find one to swap in, the
                # algorithm fails (the portion of the matrix we are operationg on must be
                # non-invertable). Note that this doesn't raise an error, as we want to sometimes
                # use this algorithm when that is going to happen, and to just report it.
                if j == n - 1: return sout, None, False

        for j in range(i + 1, n):
            if optype == 'row': ci, ri = i, j
            elif optype == 'column': ci, ri = j, i
            if sout[rs + ri, cs + ci] == 1:
                # Add the ith row/column to row/column j, mod 2.
                if position == 'UL': pair = (ci, ri)
                elif position == 'UR':
                    if optype == 'row': pair = (ci, ri)
                    else: pair = (ri, ci)
                elif position == 'LL':
                    if optype == 'row': pair = (ri, ci)
                    else: pair = (ci, ri)
                elif position == 'LR': pair = (ri, ci)

                _symp.apply_internal_gate_to_symplectic(sout, 'CNOT', pair, optype=optype)
                instruction_list.append(_Label('CNOT', (qubit_labels[pair[0]], qubit_labels[pair[1]])))

    # If it's a column action, we reverse the instruction list, so that, when applied in this order
    # *before* a circuit with action `s` it will do the desired conversion of `s`. (reversing this
    # rather than the row list is due to the different between circuit and matrix-multiply ordering).
    if optype == 'column': instruction_list.reverse()

    return sout, instruction_list, True


def _submatrix_gaussian_elimination_using_cnots(s, optype, position, qubit_labels):
    """
    Converts a submatrix of `s` to the identity matrix using CNOTs.

    Converts one of the 4 submatrices of the symplectic matrix `s` to the identity matrix,
    using a CNOT circuit acting after or before this circuit. The CNOT circuit is found
    using Gaussian elimination. The CNOT circuit acts before this circuit if `optype` is 'column'
    and acts after this if `optype` is 'row'. The conversion is only performed *if* the submatrix
    is invertable (otherwise it is not possible), and otherwise the function reports that this
    submatrix is not invertable.

    The function does not alter `s`, but rather returns an updated version of s.

    Parameters
    ----------
    s : np.array
        The (2n,2n) matrix over [0,1] that we are transforming one of the 4 (n,n) submatrices
        of to I.

    optype : 'row' or 'column'
        If 'row', we peform row-operation CNOTs (CNOTs from the LHS) to convert the the submatrix to I,
        If 'column', we peform column-operation CNOTs (CNOTs from the RHS) to convert the submatrix to I.

    position : 'UL', 'UR', 'LL', or 'LR'
        The submatrix to perform the transformation on. 'UL' and 'UR' correspond to the upper left and
        right submatrices, respecively. 'LL' and 'LR' correspond to the lower left and right submatrices,
        respecively.

    qubit_labels : list
        The qubit labels corresponding to the indices of `s`. This is required because othewise
        it is ambigious as to what the 'name' of a qubit associated with each indices is, so it
        is not possible to return a suitable list of CNOTs.

    Returns
    -------
    np.array
        The updated s matrix. If the submatrix is not invertable, then the updated s at the point
        at which this becomes apparent is returned.
    bool
        True if the transformation was successful (the submatrix was invertable)
    list
        A list of the CNOT instructions used to convert s to I. If `optype`='row', this
        list is the CNOTs in the order that they should be applied *after* a unitary represented by
        `s` so that the composite unitary has I as the specified submatrix of its `s` matrix.
        If `otptype `='column' then this list is the CNOTs in the order they should be applied *before* a
        unitary represented by `s` so that the composite unitary has I as the specified submatrix of
        its `s` matrix.

        When the returned bool is False, this list is None.
    """
    # First, we convert to row/column echelon form.
    sout, instruction_list, success = _convert_submatrix_to_echelon_form_using_cnots(s, optype, position, qubit_labels)
    # If converting to row/column echelon form fails, we quit and return a fail message.
    if not success: return sout, None, False

    n = _np.shape(sout)[0] // 2

    if position == 'UL': cs, rs, = 0, 0
    if position == 'LL': cs, rs, = 0, n
    if position == 'UR': cs, rs, = n, 0
    if position == 'LR': cs, rs, = n, n

    additional_instructions = []

    for i in reversed(range(n)):
        for j in reversed(range(i)):
            # If it's a row op, the column is fixed to i and we vary the row: we're trying to add rows
            # to make this the identity column.
            if optype == 'row': ci, ri = i, j
            # If it's a column op, the row is fixed to i, and we vary the column we're trying to add
            # rows to make this to the RHS of i the identity row.
            elif optype == 'column': ci, ri = j, i

            # Look to see if we need to set this element to 0.
            if sout[rs + ri, cs + ci] == 1:
                if position == 'UL': pair = (ci, ri)
                elif position == 'UR':
                    if optype == 'row': pair = (ci, ri)
                    else: pair = (ri, ci)
                elif position == 'LL':
                    if optype == 'row': pair = (ri, ci)
                    else: pair = (ci, ri)
                elif position == 'LR': pair = (ri, ci)

                _symp.apply_internal_gate_to_symplectic(sout, 'CNOT', pair, optype=optype)
                additional_instructions.append(_Label('CNOT', (qubit_labels[pair[0]], qubit_labels[pair[1]])))

    if optype == 'row':
        instruction_list = instruction_list + additional_instructions
    if optype == 'column':
        additional_instructions.reverse()
        instruction_list = additional_instructions + instruction_list

    return sout, instruction_list, True


def _make_submatrix_invertable_using_hadamards(s, optype, position, qubit_labels):
    """
    Uses row-action or column-action Hadamard gates to make a submatrix of `s` invertable.

    The function does not alter `s`, but rather returns an updated version of s.

    Parameters
    ----------
    s : np.array
        A (2n,2n) matrix over [0,1].

    optype : 'row' or 'column'
        If 'row', we use row-operation Hadamards (Hs from the LHS).
        If 'column', we use column-operation Hadamards (Hs from the RHS).

    position : 'UL', 'UR', 'LL', or 'LR'
        The submatrix to perform the transformation on. 'UL' and 'UR' correspond to the upper left and
        right submatrices, respecively. 'LL' and 'LR' correspond to the lower left and right submatrices,
        respecively.

    qubit_labels : list
        The qubit labels corresponding to the indices of `s`. This is required because othewise
        it is ambigious as to what the 'name' of a qubit associated with each indices is, so it
        is not possible to return a suitable list of CNOTs.

    Returns
    -------
    np.array
        The updated s matrix. If the submatrix is not invertable, then the updated s at the point
        at which this becomes apparent is returned.
    list
        A list of the Hadamards.
    """
    n = _np.shape(s)[0] // 2
    sout = s.copy()
    # The algorithm is randomized; but the number of iterations allowed is limited.
    iteration = 0
    # Set to True once we find a suitable set of Hadamards
    success = False
    # A list of the qubits on which to do Hadamards.
    h_list = []

    while not success:

        iteration += 1
        # This returns success = True if the matrix is invertable. Note it doesn't actual matter
        # what 'optype' or 'qubit_labels' is here, but we just set them to the input of this function.
        sref, junk, success = _convert_submatrix_to_echelon_form_using_cnots(sout, optype, position, qubit_labels)
        # If this didn't succed, the matrix isn't currently invertable
        if not success:
            # Pick a random qubit.
            hqubit = _np.random.randint(n)
            # Update sout with the action of Hadamard on this qubit.
            _symp.apply_internal_gate_to_symplectic(sout, 'H', (hqubit,), optype=optype)
            # If hqubit was already in the list, we remove it.
            if hqubit in h_list: del h_list[h_list.index(hqubit)]
            # If it wasn't in the list, we add it.
            else: h_list.append(hqubit)

        # Fail after a certain number of iterations (this algorithm shouldn't need many iterations). Future: replace
        # with an algorithm that tries all possible H combinations (or a deterministic algorithm), as this algorithm
        # can fail when the input is symplectic.
        if iteration > 10 * n + 100:
            raise ValueError("Randomized algorithm has failed! "
                             "This is possible but unlikely if `s` is symplectic, so perhaps the input was invalid.")

    # Create the instruction list, now we've found a suitable set of Hadamards
    instructions = [_Label('H', qubit_labels[i]) for i in h_list]

    return sout, instructions


def _make_submatrix_invertable_using_phases_and_idsubmatrix(s, optype, position, qubit_labels):
    """
    Uses row-action or column-action Phase gates to make a submatrix of `s` invertable.

    This uses an identity submatrix in `s`, and does not alter `s`, but rather returns an
    updated version of s.

    Parameters
    ----------
    s : np.array
        A (2n,2n) matrix over [0,1].

    optype : 'row' or 'column'
        If 'row', we use row-operation Ps (Ps from the LHS). In that case 'position' must be
        'LL' or 'LR' and it is submatrix above `optype` that must be the identity for this
        function to have the desired action.
        If 'column', we use column-operation Ps (Ps from the RHS). In that case 'position' must be
        'UL' or 'LL' and it is submatrix to the RHS of `optype` that must be the identity for this
        function to have the desired action.

    position : 'UL', 'UR', 'LL', or 'LR'
        The submatrix to perform the transformation on. 'UL' and 'UR' correspond to the upper left and
        right submatrices, respecively. 'LL' and 'LR' correspond to the lower left and right submatrices,
        respecively.

    qubit_labels : list
        The qubit labels corresponding to the indices of `s`. This is required because othewise
        it is ambigious as to what the 'name' of a qubit associated with each indices is, so it
        is not possible to return a suitable list of CNOTs.

    Returns
    -------
    np.array
        The updated s matrix. If the submatrix is not invertable, then the updated s at the point
        at which this becomes apparent is returned.
    list
        A list of the phase gates.
    """
    sout = s.copy()
    n = len(s[0, :]) // 2

    # Because of the action of P, this only works with particular position-optype pairs. Note that this algorithm
    # implicitly assumes that the relevant submatrix of `s` is the identity. If 'row' we are assuming that the
    # submatrix above 'position' is the identity. If 'column' we are assuming that the submatrix to the RHS of
    # 'position' is the identity.
    if optype == 'row':
        assert(position == 'LL' or position == 'LR'), "If `optype` ='row' then the position must be a lower submatrix!"
        if position == 'LL': rs, cs = n, 0  # These specify the position of the submatrix.
        elif position == 'LR': rs, cs = n, n  # These specify the position of the submatrix.
    if optype == 'column':
        assert(position == 'UL' or position == 'LL'), "If `optype` ='column' then the position must be a LHS submatrix!"
        if position == 'UL': rs, cs = 0, 0  # These specify the position of the submatrix.
        elif position == 'LL': rs, cs = n, 0  # These specify the position of the submatrix.

    instructions = []
    # The submatrix to turn into an invertable submatrix. We do Gauss elim. on this matrix to found
    # out where to put phase gates.
    matrix = sout[rs:rs + n, cs:cs + n].copy()

    # Perform row-reduction on `matrix`. Whenever we get a '0' on the diagonal, we add a P to the instruction list,
    # and perform its action of sout. (note: we don't do the row-reducation on sout; the purposes of the row-reduction
    # is to find where we need the Ps, not to actually do Gauss elim. on the input `s`.)
    for i in range(n):

        if matrix[i, i] != 1:
            # This updates `sout`: editing `matrix` is *not* editing `sout` (or doing the same
            # operations as this).
            _symp.apply_internal_gate_to_symplectic(sout, 'P', (i,), optype=optype)
            instructions.append(_Label('P', qubit_labels[i]))

        # The row/column-reduction for index i.
        if optype == 'row':
            for j in range(i + 1, n):
                if matrix[j, i] == 1:
                    # Add the ith row to row j.
                    matrix[j, :] = matrix[i, :] ^ matrix[j, :]
        if optype == 'column':
            for j in range(i + 1, n):
                if matrix[i, j] == 1:
                    # Add the ith column to column j
                    matrix[:, j] = matrix[:, i] ^ matrix[:, j]

    return sout, instructions


def find_albert_factorization_transform_using_cnots(s, optype, position, qubit_labels):
    """
    Performs an Albert factorization transform on `s`.

    Given a symplectic `s` matrix of the form ((A,B),(C,D)) with the submatrix in the
    position specified by `position` symmetric, this function

    1. Finds an *invertable* M such that F = M M.T where F is the submatrix in position
    `position`, i.e., F is one of A, B, C and D. (this is known as an albert factorization).

    2. Applies a CNOT circuit from the LHS (if `optype` = 'row')  or RHS (if `optyp`='colum'))
    to `s` so that F - > M.

    For example, if s = ((A,I),(C,D)), `position` = 'LR' and `optype` = 'column' then it
    finds an M such that we may write s = ((A,I),(C,M M.T)) and it applies the CNOT circuit
    ((M,0),(0,M^(-1)^T) to the RHS of s, mapping s to s = ((AM,M),(CM^(-1)^T) ,M)), so that
    the upper RHS and lower RHS matrix of the new s are the same and are invertable.

    This function returns a CNOT circuit that performs this action, with this CNOT circuit
    obtained from basic Gaussian elimination on M. Note that neither the returned `s` nor
    the CNOT circuit is deterministic: Both depend on M from albert factorisation, but this
    M is non-unique and our algorithm for finding such an M is randomized.

    This function does not alter `s`, but rather returns an updated version of s.

    Parameters
    ----------
    s : np.array
        A (2n,2n) matrix over [0,1].

    optype : 'row' or 'column'
        If 'row', we use row-operation CNOTs (CNOTs from the LHS).
        If 'column', we use column-operation CNOTs (CNOTs from the RHS).

    position : 'UL', 'UR', 'LL', or 'LR'
        The submatrix to perform the transformation on. 'UL' and 'UR' correspond to the upper left and
        right submatrices, respecively. 'LL' and 'LR' correspond to the lower left and right submatrices,
        respecively.

    qubit_labels : list
        The qubit labels corresponding to the indices of `s`. This is required because othewise
        it is ambigious as to what the 'name' of a qubit associated with each indices is, so it
        is not possible to return a suitable list of CNOTs.

    Returns
    -------
    np.array
        The updated s matrix.
    list
        A list of CNOT gates to implement the transformation requested.
    """
    n = _np.shape(s)[0] // 2
    sout = s.copy()

    if position == 'UL': cs, rs = 0, 0
    elif position == 'LL': cs, rs = 0, n
    elif position == 'UR': cs, rs = n, 0
    elif position == 'LR': cs, rs = n, n

    D = s[rs:rs + n, cs:cs + n].copy()
    assert(_np.array_equal(D, D.T)), "The matrix D to find an albert factorization of is not invertable!"
    # Return an invertable matrix M such that D = M M.T
    M = _mtx.albert_factor(D)

    # Temp reset the submatrix quadrant at 'position' to M.T or M: so the GE maps that quadrant to I.
    # If it's a row-action (from the LHS) we're mapping D = M M.T -> M.T
    if optype == 'row': sout[rs:rs + n, cs:cs + n] = M
    # If it's a column-action (from the RHS) we're mapping D = M M.T -> M
    if optype == 'column': sout[rs:rs + n, cs:cs + n] = M.T
    # Do GE to map that quadrant of sout to I. (which is then replaced with Mdecomposition).
    sout, instructions, success = _submatrix_gaussian_elimination_using_cnots(sout, optype, position, qubit_labels)
    # Correct the submatrix quadrant of sout:
    # put what the quadrant is actually mapped to given it was M M.T not M or M.T
    # If it's a row-action (from the LHS) we're mapping D = M M.T -> M.T
    if optype == 'row': sout[rs:rs + n, cs:cs + n] = M.T
    # If it's a column-action (from the RHS) we're mapping D = M M.T -> M
    if optype == 'column': sout[rs:rs + n, cs:cs + n] = M

    return sout, instructions


def _apply_phase_to_all_qubits(s, optype, qubit_labels):
    """
    Applies phase gates to all qubits

    Parameters
    ----------
    s : np.array
        A (2n,2n) matrix over [0,1].

    optype : 'row' or 'column'
        If 'row', we use row-operation phase gates.
        If 'column', we use column-operation phase gates.

    qubit_labels : list
        The qubit labels corresponding to the indices of `s`.

    Returns
    -------
    np.array
        The updated s matrix.
    list
        A list containing phase gates on all the qubits.
    """
    n = _np.shape(s)[0] // 2
    sout = s.copy()

    # As a row-action, phase on all qubits adds the upper half to the lower half of s.
    if optype == 'row': sout[n:2 * n, :] = sout[n:2 * n, :] ^ sout[0:n, :]
    # As a row-action, phase on all qubits adds the RHS of s to the LHS of s.
    elif optype == 'column': sout[:, 0:n] = sout[:, 0:n] ^ sout[:, n:2 * n]
    else: raise ValueError("optype must be 'row' or 'column!")

    instructions = [_Label('P', qubit_labels[i]) for i in range(n)]

    return sout, instructions


def _apply_hadamard_to_all_qubits(s, optype, qubit_labels):
    """
    Applies Hadamard gates to all qubits

    Parameters
    ----------
    s : np.array
        A (2n,2n) matrix over [0,1].

    optype : 'row' or 'column'
        If 'row', we use row-operation Hadamard gates.
        If 'column', we use column-operation Hadamard gates.

    qubit_labels : list
        The qubit labels corresponding to the indices of `s`.

    Returns
    -------
    np.array
        The updated s matrix.
    list
        A list containing Hadamard gates on all the qubits.
    """
    n = _np.shape(s)[0] // 2
    sout = s.copy()

    # As a row-action, Hadamards on all qubits swaps the upper and lower
    # halves of s.
    if optype == 'row':
        s_upper = sout[0:n, :].copy()
        s_lower = sout[n:2 * n, :].copy()
        sout[0:n, :] = s_lower
        sout[n:2 * n:] = s_upper

    # As a column-action, Hadamards on all qubits swaps the LHS and RHS
    # halves of s.
    elif optype == 'column':
        s_LHS = sout[:, :n].copy()
        s_RHS = sout[:, n:2 * n].copy()
        sout[:, 0:n] = s_RHS
        sout[:, n:2 * n] = s_LHS

    else: raise ValueError("optype must be 'row' or 'column!")

    instructions = [_Label('H', qubit_labels[i]) for i in range(n)]

    return sout, instructions


def compile_conditional_symplectic(s, pspec, qubit_labels=None, calg='COiCAGE', cargs=[], check=True):
    """
    Finds circuits that partially (conditional on the input) implement the Clifford given by `s`.

    The core of the `compile_stabilizer_state()` and `compile_stabilizer_measurement()` functions.
    Finds circuits C1 and C2 so that:

    1. C1 is a CNOT circuit
    2. C2 is a circuit with the form 1-qubit-gates -- CNOT circuit -- 1-qubit gates.
    3. The symplectic rep of the circuit consisting of C1 followed by C2 has the form ((.,B)(.,D))
    when s has the form ((A,B),(C,D)).

    Therefore, the circuit C2 acting on |0,0,0,...> generates the same stabilizer state (up to Paulis)
    as a circuit that has the symplectic rep (s,p) for any valid p. The circuit is only "conditionally"
    equivalent to another circuit with the rep (s,p) -- conditional on the input state -- which is the
    meaning of the name `compile_conditional_symplectic`.

    Parameters
    ----------
    s : array over [0,1]
        An (2n X 2n) symplectic matrix of 0s and 1s integers.

    pspec : ProcessorSpec, optional
        An nbar-qubit ProcessorSpec object that encodes the device that `s` is being "conditionally" compiled
        for, where nbar >= n. If nbar > n it is necessary to provide `qubit_labels`, that specifies which of the
        qubits in `pspec` the stabilizer is over. (All other qubits will not be part of the returned circuit,
        regardless of whether that means an over-head is required to avoid using gates that act on those qubits.
        If these additional qubits should be used, then the input (s,p) needs to be ``padded'' to be the identity
        on those qubits). The ordering of the indices in (`s`,`p`) is w.r.t to ordering of the qubit labels in
        pspec.qubit_labels, unless `qubit_labels` is specified. Then, the ordering is taken w.r.t the ordering of the
        list `qubit_labels`.

    qubit_labels : List, optional
        Required if the `s` is over less qubits than `pspec`. In this case this is a
        list of the qubits to compile the Clifford for; it should be a subset of the elements of
        pspec.qubit_labels. The ordering of the qubits in (`s`,`p`) is taken w.r.t the ordering of this list.

    calg : str, optional
        Our algorithm finds a circuit consisting of 1Q-gates - a CNOT circuit - 1Q-gates. The CNOT circuit
        is found using Gaussian elimination, and it can then be recompiled using a CNOT-circuit compiler.
        `calg` specifies the CNOT-compilation algorithm to use. The allowed values are all those algorithms
        that are permisable in the `compile_cnot_circuit()` function. See the docstring of that function for more
        information. The default is likely to be the best out of the in-built CNOT compilers under most
        circumstances. Note that this is only used to recompile the CNOT circuit in C2, because the main use
        for this function only needs C1 for reference, rather than as a part of a final output.

    cargs : list, optional
        If the CNOT compilation algorithm can take optional arguments, these are specified here. This is passed
        to compile_cnot_circuit() as `aarg`.

    check : bool, optional
        Whether to check that the output is correct.

    Returns
    -------
    Circuit
        The circuit C2 described above.
    Circuit
        The circuit C1 described above.
    """
    n = _np.shape(s)[0] // 2

    if qubit_labels is not None:
        assert(len(qubit_labels) == n), "The length of `qubit_labels` is inconsisent with the size of `s`!"
        qubit_labels = qubit_labels
    else:
        qubit_labels = pspec.qubit_labels
        assert(len(qubit_labels) == n), \
            ("The number of qubits is inconsisent with the size of `s`! "
             "If `s` is over a subset, `qubit_labels` must be specified!")

    # A matrix to keep track of the current state of s.
    sout = s.copy()

    # Stage 1: Hadamard gates from the LHS to make the UR submatrix of s invertible.
    sout, Hsome_layer = _make_submatrix_invertable_using_hadamards(sout, 'row', 'UR', qubit_labels)

    if n > 1:
        # Stage 2: CNOT circuit from the RHS to map the UR submatrix of s to I.
        sout, CNOTs_RHS1, success = _submatrix_gaussian_elimination_using_cnots(sout, 'column', 'UR', qubit_labels)
        # We reverse the list, because its a list doing the GE on s, and we want to do the inverse of that on I.
        CNOTs_RHS1.reverse()
        assert(success), \
            ("The 1st Gaussian elimination stage of the algorithm has failed! "
             "Perhaps the input was not a symplectic matrix.")

    # Stage 3: Phase circuit from the LHS to make the LR submatrix of s invertible
    sout, Psome_layer = _make_submatrix_invertable_using_phases_and_idsubmatrix(sout, 'row', 'LR', qubit_labels)

    if n > 1:
        # Stage 4: CNOT circuit from the LHS to map the UR and LR submatrices of s to the same invertible matrix M
        sout, CNOTs = find_albert_factorization_transform_using_cnots(sout, 'row', 'LR', qubit_labels)
        # We reverse the list, because its a list doing the GE on s, and we want to do the inverse of that on I.
        CNOTs.reverse()

        # Stage 5: A CNOT circuit from the RHS to map the URH and LRH submatrices of s from M to I.
        sout, CNOTs_RHS2, success = _submatrix_gaussian_elimination_using_cnots(sout, 'column', 'UR', qubit_labels)
        # We reverse the list, because its a list doing the GE on s, and we want to do the inverse of that on I.
        CNOTs_RHS2.reverse()
        assert(success), \
            ("The 3rd Gaussian elimination stage of the algorithm has failed! "
             "Perhaps the input was not a symplectic matrix.")

    # Stage 6: Phase gates on all qubits acting from the LHS to map the LR submatrix of s to 0.
    sout, Pall_layer = _apply_phase_to_all_qubits(sout, 'row', qubit_labels)

    # Stage 7: Hadamard gates on all qubits acting from the LHS to swap the LR and UR matrices
    # of s, (mapping them to I and 0 resp.,).
    sout, Hall_layer = _apply_hadamard_to_all_qubits(s, 'row', qubit_labels)

    if n > 1:
        # If we're using the basic Gauss. elimin. algorithm for the CNOT circuit, we just keep this CNOTs list
        # as this is what the above algorithm has provided.
        if calg == 'BGE': circuit = _Circuit(layer_labels=CNOTs, line_labels=qubit_labels).parallelize()
        # Otherwise, we recompile the CNOTs circuit. (note : it would be more efficient if we used this algorithm
        # to start with inside `find_albert_factorization_transform_using_cnots`, so this is something we can do in
        # the future if the speed of this algorithm matters).
        else:
            # Finds the CNOT circuit we are trying to compile in the symplectic rep.
            cnot_s, cnot_p = _symp.symplectic_rep_of_clifford_circuit(
                _Circuit(layer_labels=CNOTs, line_labels=qubit_labels).parallelize())
            # clname is set to None so that the function doesn't change the circuit into the native gate library.
            circuit = compile_cnot_circuit(cnot_s, pspec, qubit_labels=qubit_labels,
                                           algorithm=calg, clname=None, check=False, aargs=cargs)
        circuit = circuit.copy(editable=True)
    else:
        circuit = _Circuit(layer_labels=[], line_labels=qubit_labels, editable=True)

    # Circuit starts with the all Hs layer followed by the all Ps layer.
    circuit.insert_layer(Pall_layer, 0)
    circuit.insert_layer(Hall_layer, 0)
    # circuit ends the the some Ps layer followed by the some Hs layer.
    if len(Psome_layer) > 0: circuit.insert_layer(Psome_layer, circuit.depth())
    if len(Hsome_layer) > 0: circuit.insert_layer(Hsome_layer, circuit.depth())

    # This pre-circuit is necessary for working out what Pauli's need to be pre/post fixed to `circuit` to generate
    # the requested stabilizer state.
    if n > 1: precircuit = _Circuit(layer_labels=CNOTs_RHS1 + CNOTs_RHS2, line_labels=qubit_labels)
    else: precircuit = _Circuit(layer_labels=[], line_labels=qubit_labels)

    if check:
        # Only the circuit with the precircuit prefixed as format that can be easily checked as corrected. That
        # circuit should have an s-matrix with the RHS equal to the RHS of `s`.
        checkcircuit = precircuit.copy()
        checkcircuit.append_circuit(circuit)
        scheck, pcheck = _symp.symplectic_rep_of_clifford_circuit(checkcircuit)
        assert(_np.array_equal(scheck[:, n:2 * n], s[:, n:2 * n])), "Compiler has failed!"

    return circuit, precircuit

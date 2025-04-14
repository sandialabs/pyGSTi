"""
Random circuit sampling functions.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
import itertools as _itertools

import numpy as _np

from pygsti.algorithms import compilers as _cmpl
from pygsti.circuits import circuit as _cir
from pygsti.baseobjs import label as _lbl
from pygsti.tools import group as _rbobjs
from pygsti.tools import symplectic as _symp
from pygsti.tools import compilationtools as _comp
from pygsti.tools import internalgates as _gates

################################
#### BEGIN CODE FROM JORDAN ######
################################


def sample_haar_random_one_qubit_unitary_parameters():
    """
    TODO: docstring
    """
    psi, chi = 2 * _np.pi * _np.random.rand(2)
    psi = psi - _np.pi
    chi = chi - _np.pi
    phi = _np.arcsin(_np.sqrt(_np.random.rand(1)))[0]
    #U = _np.exp(1j*alpha)*_np.array([[_np.exp(1j*psi)*_np.cos(phi), _np.exp(1j*chi)*_np.sin(phi)],[-1*_np.exp(-1j*chi)
    # *_np.sin(phi), _np.exp(-1j*psi)*_np.cos(phi)]])
    #this needs to be decomposed in the form Zrot(theta3) Xpi/2 Zrot(theta2) Xpi/2 Zrot(theta1)
    theta1 = _comp.mod_2pi(psi - chi + _np.pi)
    theta2 = _comp.mod_2pi(_np.pi - 2 * phi)
    theta3 = _comp.mod_2pi(psi + chi)
    return (theta1, theta2, theta3)


def sample_random_clifford_one_qubit_unitary_parameters():
    """
    TODO: docstring
    """
    theta1 = _comp.mod_2pi(_np.random.randint(4) * _np.pi / 2)
    theta2 = _comp.mod_2pi(_np.random.randint(4) * _np.pi / 2)
    theta3 = _comp.mod_2pi(_np.random.randint(4) * _np.pi / 2)
    return (theta1, theta2, theta3)


def sample_compiled_haar_random_one_qubit_gates_zxzxz_circuit(pspec, zname='Gzr', xname='Gxpi2', qubit_labels=None):
    """
    TODO: docstring  #generate layer of random unitaries and make a series of circuit layers with the compiled versions
    of these
    """
    if qubit_labels is not None:
        n = len(qubit_labels)
        qubits = qubit_labels[:]  # copy this list
    else:
        n = pspec.num_qubits
        qubits = pspec.qubit_labels[:]  # copy this list

    Xpi2layer = _cir.Circuit(layer_labels=[[(xname, qubits[t]) for t in range(n)], ])

    # samples random rotation angles.
    rot_angles = [sample_haar_random_one_qubit_unitary_parameters() for q in qubits]

    circ = _cir.Circuit(layer_labels=[[_lbl.Label(zname, qubits[t], args=(str(rot_angles[t][0]),))
                                       for t in range(n)], ], editable=True)
    circ.append_circuit_inplace(Xpi2layer)
    circ.append_circuit_inplace(_cir.Circuit(layer_labels=[[_lbl.Label(zname, qubits[t], args=(str(rot_angles[t][1]),))
                                                           for t in range(n)], ]))
    circ.append_circuit_inplace(Xpi2layer)
    circ.append_circuit_inplace(_cir.Circuit(layer_labels=[[_lbl.Label(zname, qubits[t], args=(str(rot_angles[t][2]),))
                                                            for t in range(n)], ]))
    circ.done_editing()
    return circ


def sample_compiled_random_clifford_one_qubit_gates_zxzxz_circuit(pspec, zname='Gzr', xname='Gxpi2', qubit_labels=None):
    """
    TODO: docstring  #generate layer of random unitaries and make a series of circuit layers with the compiled versions
    of these
    """
    if qubit_labels is not None:
        n = len(qubit_labels)
        qubits = qubit_labels[:]  # copy this list
    else:
        n = pspec.num_qubits
        qubits = pspec.qubit_labels[:]  # copy this list

    Xpi2layer = _cir.Circuit(layer_labels=[[(xname, qubits[t]) for t in range(n)], ])

    # samples random rotation angles.
    rot_angles = [sample_random_clifford_one_qubit_unitary_parameters() for q in qubits]

    circ = _cir.Circuit(layer_labels=[[_lbl.Label(zname, qubits[t], args=(str(rot_angles[t][0]),))
                                       for t in range(n)], ], editable=True)
    circ.append_circuit_inplace(Xpi2layer)
    circ.append_circuit_inplace(_cir.Circuit(layer_labels=[[_lbl.Label(zname, qubits[t], args=(str(rot_angles[t][1]),))
                                                           for t in range(n)], ]))
    circ.append_circuit_inplace(Xpi2layer)
    circ.append_circuit_inplace(_cir.Circuit(layer_labels=[[_lbl.Label(zname, qubits[t], args=(str(rot_angles[t][2]),))
                                                            for t in range(n)], ]))
    circ.done_editing()
    return circ


def sample_random_cz_zxzxz_circuit(pspec, length, qubit_labels=None, two_q_gate_density=0.25,
                                   one_q_gate_type='haar',
                                   two_q_gate_args_lists=None):
    '''
    TODO: docstring
    Generates a forward circuits with benchmark depth d for non-clifford mirror randomized benchmarking.
    The circuits alternate Haar-random 1q unitaries and layers of Gczr gates.

    If two_q_gate_args_lists is None, then we set it to {'Gczr': [(str(_np.pi / 2),), (str(-_np.pi / 2),)]}.
    '''
    if two_q_gate_args_lists is None:
        two_q_gate_args_lists = {'Gczr': [(str(_np.pi / 2),), (str(-_np.pi / 2),)]}
    #choose length to be the number of (2Q layer, 1Q layer) blocks
    circuit = _cir.Circuit(layer_labels=[], line_labels=qubit_labels, editable=True)
    for a in range(length):
        #generate random 1q unitary layer
        if one_q_gate_type == 'haar':
            new_layer = sample_compiled_haar_random_one_qubit_gates_zxzxz_circuit(pspec, qubit_labels=qubit_labels)
        elif one_q_gate_type == 'clifford':
            new_layer = sample_compiled_random_clifford_one_qubit_gates_zxzxz_circuit(pspec, qubit_labels=qubit_labels)
        else:
            raise ValueError("Unknown value {} for `one_q_gate_type`!".format(one_q_gate_type))
        #append new layer to circuit
        circuit.append_circuit_inplace(new_layer)
        #generate 2q gate layer
        sampled_layer = sample_circuit_layer_by_edgegrab(pspec, qubit_labels=qubit_labels,
                                                         two_q_gate_density=two_q_gate_density,
                                                         one_q_gate_names=[], gate_args_lists=two_q_gate_args_lists)
        if sampled_layer == []: new_layer = _cir.Circuit(layer_labels=[[]], line_labels=qubit_labels)
        else: new_layer = _cir.Circuit([sampled_layer])
        #append new layer to circuit
        circuit.append_circuit_inplace(new_layer)
    #add one more layer of Haar-random 1Q unitaries
    if one_q_gate_type == 'haar':
        new_layer = sample_compiled_haar_random_one_qubit_gates_zxzxz_circuit(pspec, qubit_labels=qubit_labels)
    elif one_q_gate_type == 'clifford':
        new_layer = sample_compiled_random_clifford_one_qubit_gates_zxzxz_circuit(pspec, qubit_labels=qubit_labels)
    else:
        raise ValueError("Unknown value {} for `one_q_gate_type`!".format(one_q_gate_type))
    circuit.append_circuit_inplace(new_layer)
    circuit.done_editing()
    return circuit


def find_all_sets_of_compatible_two_q_gates(edgelist, n, gatename='Gcnot', aslabel=False):
    """
    TODO: docstring
    <TODO summary>

    Parameters
    ----------
    edgelist : <TODO typ>
        <TODO description>

    n : int
        The number of two-qubit gates to have in the set.

    gatename : <TODO typ>, optional
        <TODO description>

    aslabel : <TODO typ>, optional
        <TODO description>

    Returns
    -------
    <TODO typ>
    """
    co2Qgates = []

    # Go for all combinations of n two-qubit gates from the edgelist.
    for npairs in _itertools.combinations(edgelist, n):

        # Make a list of the qubits involved in the gates
        flat_list = [item for sublist in npairs for item in sublist]

        # If no qubit is involved in more than one gate we accept the combination
        if len(flat_list) == len(set(flat_list)):
            if aslabel:
                co2Qgates.append([_lbl.Label(gatename, pair) for pair in npairs])
            else:
                co2Qgates.append([gatename + ':' + pair[0] + ':' + pair[1] for pair in npairs])

    return co2Qgates


def sample_circuit_layer_by_edgegrab(pspec, qubit_labels=None, two_q_gate_density=0.25, one_q_gate_names=None,
                                     gate_args_lists=None, rand_state=None):
    """
    TODO: docstring
    <TODO summary>

    Parameters
    ----------
    pspec : <TODO typ>
        <TODO description>

    qubit_labels : <TODO typ>, optional
        <TODO description>

    mean_two_q_gates : <TODO typ>, optional
        <TODO description>

    modelname : <TODO typ>, optional
        <TODO description>

    rand_state: RandomState, optional
        A np.random.RandomState object for seeding RNG

    Returns
    -------
    <TODO typ>
    """
    if gate_args_lists is None: gate_args_lists = {}
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list
    if rand_state is None:
        rand_state = _np.random.RandomState()

    # Prep the sampling variables.
    sampled_layer = []
    edgelist = pspec.compute_2Q_connectivity().edges()
    edgelist = [e for e in edgelist if all([q in qubits for q in e])]
    selectededges = []

    # Go through until all qubits have been assigned a gate.
    while len(edgelist) > 0:

        edge = edgelist[rand_state.randint(0, len(edgelist))]
        selectededges.append(edge)
        # Delete all edges containing these qubits.
        edgelist = [e for e in edgelist if not any([q in e for q in edge])]

    num2Qgates = len(selectededges)
    if len(qubits) > 1:
        mean_two_q_gates = len(qubits) * two_q_gate_density / 2
    else:
        mean_two_q_gates = 0
    assert(num2Qgates >= mean_two_q_gates), "Device has insufficient connectivity!"

    if mean_two_q_gates > 0:
        twoQprob = mean_two_q_gates / num2Qgates
    else:
        twoQprob = 0

    unusedqubits = _copy.copy(qubits)
    ops_on_qubits = pspec.compute_ops_on_qubits()
    for edge in selectededges:
        if bool(rand_state.binomial(1, twoQprob)):
            # The two-qubit gates on that edge.
            possibleops = ops_on_qubits[edge]
            argless_gate_label = possibleops[rand_state.randint(0, len(possibleops))]
            if argless_gate_label.name not in gate_args_lists.keys():
                sampled_layer.append(argless_gate_label)
            else:
                possibleargs = gate_args_lists[argless_gate_label.name]
                args = possibleargs[rand_state.randint(0, len(possibleargs))]
                sampled_layer.append(_lbl.Label(argless_gate_label.name, edge, args=args))

            for q in edge:
                del unusedqubits[unusedqubits.index(q)]

    if one_q_gate_names is None or len(one_q_gate_names) > 0:
        for q in unusedqubits:
            if one_q_gate_names is None:
                possibleops = ops_on_qubits[(q,)]
            else:
                possibleops = [gate_lbl for gate_lbl in ops_on_qubits[(q,)] if gate_lbl.name in one_q_gate_names]
            gate_label = possibleops[rand_state.randint(0, len(possibleops))]
            sampled_layer.append(gate_label)

    return sampled_layer


def sample_circuit_layer_by_q_elimination(pspec, qubit_labels=None, two_q_prob=0.5, rand_state=None):
    """
    Samples a random circuit layer by eliminating qubits one by one.

    This sampler works with any connectivity, but the expected number of 2-qubit gates
    in a layer depends on both the specified 2-qubit gate probability and the exact
    connectivity graph.

    This sampler is the following algorithm: List all the qubits, and repeat the
    following steps until all qubits are deleted from this list. 1) Uniformly at random
    pick a qubit from the list, and delete it from the list 2) Flip a coin with  bias
    `two_q_prob` to be "Heads". 3) If "Heads" then -- if there is one or more 2-qubit gates
    from this qubit to other qubits still in the list -- pick one of these at random.
    4) If we haven't chosen a 2-qubit gate for this qubit ("Tails" or "Heads" but there
    are no possible 2-qubit gates) then pick a uniformly random 1-qubit gate to apply to
    this qubit.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit layer is being sampled for. Unless
        `qubit_labels` is not None, a circuit layer is sampled over all the qubits in `pspec`.

    qubit_labels : list, optional
        If not None, a list of the qubits to sample the circuit layer for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit layer is sampled to acton all the qubits
        in `pspec`.

    two_q_prob : float, optional
        If a 2-qubit can is still possible on a qubit at that stage of the sampling, this is
        the probability a 2-qubit gate is chosen for that qubit. The expected number of
        2-qubit gates per layer depend on this quantity and the connectivity graph of
        the device.

    rand_state: RandomState, optional
        A np.random.RandomState object for seeding RNG

    Returns
    -------
    list of gates
        A list of gate Labels that defines a "complete" circuit layer (there is one and
        only one gate acting on each qubit in `pspec` or `qubit_labels`).
    """
    if qubit_labels is None:
        n = pspec.num_qubits
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        n = len(qubit_labels)
        qubits = list(qubit_labels[:])  # copy this list

    if rand_state is None:
        rand_state = _np.random.RandomState()

    possible_ops = pspec.compute_ops_on_qubits()

    # Prep the sampling variables.
    sampled_layer = []
    remaining_qubits = _copy.deepcopy(qubits)
    num_qubits_used = 0

    # Go through until all qubits have been assigned a gate.
    while num_qubits_used < n:

        # Pick a random qubit
        r = rand_state.randint(0, n - num_qubits_used)
        q = remaining_qubits[r]
        del remaining_qubits[r]

        oneq_ops_on_q = possible_ops[(q,)]
        twoq_ops_on_q = []
        for q2 in remaining_qubits:
            twoq_ops_on_q += possible_ops[(q, q2)]
            twoq_ops_on_q += possible_ops[(q2, q)]

        # Decide whether to to implement a 2-qubit gate or a 1-qubit gate.
        if len(twoq_ops_on_q) == 0:
            do_twoq_gate = False
        else:
            do_twoq_gate = rand_state.choice([False, True], p=[1 - two_q_prob, two_q_prob])

        # Implement a random 1-qubit gate on qubit q.
        if not do_twoq_gate:
            sampled_layer.append(oneq_ops_on_q[rand_state.randint(0, len(oneq_ops_on_q))])
            num_qubits_used += 1.  # We have assigned gates to 1 of the remaining qubits.

        # Implement a 2-qubit gate on qubit q.
        else:
            lbl = twoq_ops_on_q[rand_state.randint(0, len(twoq_ops_on_q))]
            sampled_layer.append(lbl)

            # Find the label of the other qubit in the sampled gate.
            other_qubit = lbl.qubits[0]
            if other_qubit == q:
                other_qubit = lbl.qubits[1]

            del remaining_qubits[remaining_qubits.index(other_qubit)]
            num_qubits_used += 2

    return sampled_layer


def sample_circuit_layer_by_co2_q_gates(pspec, qubit_labels, co2_q_gates, co2_q_gates_prob='uniform', two_q_prob=1.0,
                                        one_q_gate_names='all', rand_state=None):
    """
    Samples a random circuit layer using the specified list of "compatible two-qubit gates" (co2_q_gates).

    That is, the user inputs a list (`co2_q_gates`) specifying 2-qubit gates that are
    "compatible" -- meaning that they can be implemented simulatenously -- and a distribution
    over the different compatible sets, and a layer is sampled from this via:

    1. Pick a set of compatible two-qubit gates from the list `co2_q_gates`, according to the
    distribution specified by `co2_q_gates_prob`.
    2. For each 2-qubit gate in the chosen set of compatible gates, with probability `two_q_prob`
    add this gate to the layer.
    3. Uniformly sample 1-qubit gates for any qubits that don't yet have a gate on them,
    from those 1-qubit gates specified by `one_q_gate_names`.

    For example, consider 4 qubits with linear connectivity. a valid `co2_q_gates` list is
    `co2_q_gates = [[,],[Label(Gcphase,(0,1)),Label(Gcphase,(2,3))]]` which consists of an
    element containing zero 2-qubit gates and an element containing  two 2-qubit gates
    that can be applied in parallel. In this example there are 5 possible sets of compatible
    2-qubit gates:

    1. [,] (zero 2-qubit gates)
    2. [Label(Gcphase,(0,1)),] (one of the three 2-qubit gate)
    3. [Label(Gcphase,(1,2)),] (one of the three 2-qubit gate)
    4. [Label(Gcphase,(2,3)),] (one of the three 2-qubit gate)
    5. [Label(Gcphase,(0,1)), Label(Gcphase,(2,3)),] (the only compatible pair of 2-qubit gates).

    The list of compatible two-qubit gates `co2_q_gates` can be any list containing anywhere
    from 1 to all 5 of these lists.

    In order to allow for convenient sampling of some commonly useful distributions, 
    `co2_q_gates` can be a list of lists of lists of compatible 2-qubit gates ("nested" sampling). 
    In this case, a list of lists of compatible 2-qubit gates is picked according to the distribution 
    `co2_q_gates_prob`, and then one of the sublists of compatible 2-qubit gates in the selected list is 
    then chosen uniformly at random. For example, this is useful for sampling a layer containing one 
    uniformly random 2-qubit gate with probability p and a layer of 1-qubit gates with probability 
    1-p. Here, we can specify `co2_q_gates` as `[[],[[the 1st 2Q-gate,],[the 2nd 2Q-gate,], ...]]` and 
    set `two_q_prob=1` and `co2_q_gates_prob  = [1-p,p]`.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit layer is being sampled for. Unless
        `qubit_labels` is not None, a circuit layer is sampled over all the qubits in `pspec`.

    qubit_labels : list
        If not None, a list of the qubits to sample the circuit layer for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit layer is sampled to act on all the qubits
        in `pspec`.

    co2_q_gates : list
        This is either:

            1. A list of lists of 2-qubit gate Labels that can be applied in parallel.
            2. A list of lists of lists of 2-qubit gate Labels that can be applied in parallel.

        In case (1) each list in `co2_q_gates` should contain 2-qubit gates, in the form of Labels,
        that can be applied in parallel and act only on the qubits in `pspec` if `qubit_labels` is None,
        or act only on the qubits in  `qubit_labels` if `qubit_labels` is not None.  The sampler then picks
        one of these compatible sets of gates (with probability specified by `co2_q_gates_prob`, and converts
        this into a circuit layer by applying the 2-qubit gates it contains with the user-specified
        probability `two_q_prob`, and augmenting these 2-qubit gates with 1-qubit gates on all other qubits.

        In case (2) a sublist of lists is sampled from `co2_q_gates` according to `co2_q_gates_prob` and then we
        proceed as in case (1) but as though `co2_q_gates_prob` is the uniform distribution.

    co2_q_gates_prob : str or list of floats
        If a list, they are unnormalized probabilities to sample each of the elements of `co2_q_gates`. So it
        is a list of non-negative floats of the same length as `co2_q_gates`. If 'uniform', then the uniform
        distribution is used.

    two_q_prob : float, optional
        The probability for each two-qubit gate to be applied to a pair of qubits, after a
        set of compatible 2-qubit gates has been chosen. The expected number of 2-qubit
        gates in a layer is `two_q_prob` times the expected number of 2-qubit gates in a
        set of compatible 2-qubit gates sampled according to `co2_q_gates_prob`.

    one_q_gate_names : 'all' or list of strs, optional
        If not 'all', a list of the names of the 1-qubit gates to be sampled from when applying
        a 1-qubit gate to a qubit. If this is 'all', the full set of 1-qubit gate names is
        extracted from the QubitProcessorSpec.

    rand_state: RandomState, optional
        A np.random.RandomState object for seeding RNG

    Returns
    -------
    list of gates
        A list of gate Labels that defines a "complete" circuit layer (there is one and
        only one gate acting on each qubit).
    """
    if rand_state is None:
        rand_state = _np.random.RandomState()

    # Pick the sector.
    if isinstance(co2_q_gates_prob, str):
        assert(co2_q_gates_prob == 'uniform'), "If `co2_q_gates_prob` is a string it must be 'uniform!'"
        twoqubitgates_or_nestedco2Qgates = co2_q_gates[rand_state.randint(0, len(co2_q_gates))]
    else:
        co2_q_gates_prob = _np.array(co2_q_gates_prob) / _np.sum(co2_q_gates_prob)
        x = list(rand_state.multinomial(1, co2_q_gates_prob))
        twoqubitgates_or_nestedco2Qgates = co2_q_gates[x.index(1)]

    # The special case where the selected co2_q_gates contains no gates or co2_q_gates.
    if len(twoqubitgates_or_nestedco2Qgates) == 0:
        twoqubitgates = twoqubitgates_or_nestedco2Qgates
    # If it's a nested sector, sample uniformly from the nested co2_q_gates.
    elif type(twoqubitgates_or_nestedco2Qgates[0]) == list:
        twoqubitgates = twoqubitgates_or_nestedco2Qgates[rand_state.randint(0, len(twoqubitgates_or_nestedco2Qgates))]
    # If it's not a list of "co2_q_gates" (lists) then this is the list of gates to use.
    else:
        twoqubitgates = twoqubitgates_or_nestedco2Qgates

    # Prep the sampling variables
    sampled_layer = []
    if qubit_labels is not None:
        assert(isinstance(qubit_labels, list) or isinstance(qubit_labels, tuple)), "SubsetQs must be a list or a tuple!"
        remaining_qubits = list(qubit_labels[:])  # copy this list
    else:
        remaining_qubits = list(pspec.qubit_labels[:])  # copy this list

    # Go through the 2-qubit gates in the sector, and apply each one with probability two_q_prob
    for i in range(0, len(twoqubitgates)):
        if rand_state.binomial(1, two_q_prob) == 1:
            gate = twoqubitgates[i]
            # If it's a nested co2_q_gates:
            sampled_layer.append(gate)
            # Delete the qubits that have been assigned a gate.
            del remaining_qubits[remaining_qubits.index(gate.qubits[0])]
            del remaining_qubits[remaining_qubits.index(gate.qubits[1])]

    # Go through the qubits which don't have a 2-qubit gate assigned to them, and pick a 1-qubit gate
    clifford_ops_on_qubits = pspec.compute_clifford_ops_on_qubits()
    for i in range(0, len(remaining_qubits)):

        qubit = remaining_qubits[i]

        # If the 1-qubit gate names are specified, use these.
        if one_q_gate_names != 'all':
            possibleops = [_lbl.Label(name, (qubit,)) for name in one_q_gate_names]

        # If the 1-qubit gate names are not specified, find the available 1-qubit gates
        else:
            #if modelname == 'clifford':
            possibleops = clifford_ops_on_qubits[(qubit,)]
            #else:
            #    possibleops = pspec.models[modelname].primitive_op_labels
            #    l = len(possibleops)
            #    for j in range(0, l):
            #        if possibleops[l - j].num_qubits != 1:
            #            del possibleops[l - j]
            #        else:
            #            if possibleops[l - j].qubits[0] != qubit:
            #                del possibleops[l - j]

        gate = possibleops[rand_state.randint(0, len(possibleops))]
        sampled_layer.append(gate)

    return sampled_layer


def sample_circuit_layer_of_one_q_gates(pspec, qubit_labels=None, one_q_gate_names='all', pdist='uniform',
                                        modelname='clifford', rand_state=None):
    """
    Samples a random circuit layer containing only 1-qubit gates.

    The allowed 1-qubit gates are specified by `one_q_gate_names`, and the 1-qubit gates are
    sampled independently and uniformly.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit layer is being sampled for. Unless
        `qubit_labels` is not None, a circuit layer is sampled over all the qubits in `pspec`.

    qubit_labels : list, optional
        If not None, a list of the qubits to sample the circuit layer for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit layer is sampled to acton all the qubits
        in `pspec`.

    one_q_gate_names : 'all' or list of strs, optional
        If not 'all', a list of the names of the 1-qubit gates to be sampled from when applying
        a 1-qubit gate to a qubit. If this is 'all', the full set of 1-qubit gate names is
        extracted from the QubitProcessorSpec.

    pdist : 'uniform' or list of floats, optional
        If a list, they are unnormalized probabilities to sample each of the 1-qubit gates
        in the list `one_q_gate_names`. If this is not 'uniform', then oneQgatename` must not
        be 'all' (it must be a list so that it is unambigious which probability correpsonds
        to which gate). So if not 'uniform', `pdist` is a list of non-negative floats of the
        same length as `one_q_gate_names`. If 'uniform', then the uniform distribution over
        the gates is used.

    modelname : str, optional
        Only used if one_q_gate_names is 'all'. Specifies which of the `pspec.models` to use to
        extract the model. The `clifford` default is suitable for Clifford or direct RB,
        but will not use any non-Clifford gates in the model.

    rand_state: RandomState, optional
        A np.random.RandomState object for seeding RNG

    Returns
    -------
    list of gates
        A list of gate Labels that defines a "complete" circuit layer (there is one and
        only one gate acting on each qubit).
    """
    if qubit_labels is not None:
        assert(isinstance(qubit_labels, list) or isinstance(qubit_labels, tuple)), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list
    else:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    if rand_state is None:
        rand_state = _np.random.RandomState()

    sampled_layer = []

    if isinstance(pdist, str): assert(pdist == 'uniform'), "If pdist is not a list or numpy.array it must be 'uniform'"

    if one_q_gate_names == 'all':
        assert(pdist == 'uniform'), "If `one_q_gate_names` = 'all', pdist must be 'uniform'"
        clifford_ops_on_qubits = pspec.compute_clifford_ops_on_qubits()
        if modelname == 'clifford':
            for i in qubits:
                try:
                    gate = clifford_ops_on_qubits[(i,)][rand_state.randint(
                        0, len(clifford_ops_on_qubits[(i,)]))]
                    sampled_layer.append(gate)
                except:
                    raise ValueError("There are no 1Q Clifford gates on qubit {}!".format(i))
        else: raise ValueError("Currently, 'modelname' must be 'clifford'")

    else:
        # A basic check for the validity of pdist.
        if not isinstance(pdist, str):
            assert(len(pdist) == len(one_q_gate_names)), "The pdist probability distribution is invalid!"

        # Find out how many 1-qubit gate names there are
        num_oneQgatenames = len(one_q_gate_names)

        # Sample a gate for each qubit.
        for i in qubits:

            # If 'uniform', then sample according to the uniform dist.
            if isinstance(pdist, str): sampled_gatename = one_q_gate_names[rand_state.randint(0, num_oneQgatenames)]
            # If not 'uniform', then sample according to the user-specified dist.
            else:
                pdist = _np.array(pdist) / sum(pdist)
                x = list(rand_state.multinomial(1, pdist))
                sampled_gatename = one_q_gate_names[x.index(1)]
            # Add sampled gate to the layer.
            sampled_layer.append(_lbl.Label(sampled_gatename, i))

    return sampled_layer


def create_random_circuit(pspec, length, qubit_labels=None, sampler='Qelimination', samplerargs=None,
                          addlocal=False, lsargs=None, rand_state=None):
    """
    Samples a random circuit of the specified length (or ~ twice this length).

    The created circuit's layers are independently sampled according to the specified
    sampling distribution.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit is being sampled for. This is always
        handed to the sampler, as the first argument of the sampler function. Unless
        `qubit_labels` is not None, the circuit is sampled over all the qubits in `pspec`.

    length : int
        If `addlocal` is False, this is the length of the sampled circuit. If `addlocal is
        True the length of the circuits is 2*length+1 with odd-indexed layers sampled according
        to the sampler specified by `sampler`, and the the zeroth layer + the even-indexed
        layers consisting of random 1-qubit gates (with the sampling specified by `lsargs`)

    qubit_labels : list, optional
        If not None, a list of the qubits to sample the circuit for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit is sampled to act on all the qubits
        in `pspec`.

    sampler : str or function, optional
        If a string, this should be one of: {'edgegrab'', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates. If this is a
        function, it should be a function that takes as the first argument a QubitProcessorSpec, and
        returns a random circuit layer as a list of gate Label objects. Note that the default
        'Qelimination' is not necessarily the most useful in-built sampler, but it is the only
        sampler that requires no parameters beyond the QubitProcessorSpec *and* works for arbitrary
        connectivity devices. See the docstrings for each of these samplers for more information.

    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec` and `samplerargs` lists the
        remaining arguments handed to the sampler. For some in-built samplers this is not
        optional.

    addlocal : bool, optional
        If False, the circuit sampled is of length `length` and each layer is independently
        sampled according to the sampler specified by `sampler`. If True, the circuit sampled
        is of length 2*`length`+1 where: the zeroth + all even layers are consisting of
        independently random 1-qubit gates (with the sampling specified by `lsargs`); the
        odd-indexed layers are independently sampled according to `sampler`. So `length`+1
        layers consist only of 1-qubit gates, and `length` layers are sampled according to
        `sampler`.

    lsargs : list, optional
        A list of arguments that are handed to the 1-qubit gate layers sampler
        rb.sampler.circuit_layer_of_oneQgates for the alternating 1-qubit-only layers that are
        included in the circuit if `addlocal` is True. This argument is not used if `addlocal`
        is false. Note that `pspec` is used as the first, and only required, argument of
        rb.sampler.circuit_layer_of_oneQgates. If `lsargs` = [] then all available 1-qubit gates
        are uniformly sampled from. To uniformly sample from only a subset of the available
        1-qubit gates (e.g., the Paulis to Pauli-frame-randomize) then `lsargs` should be a
        1-element list consisting of a list of the relevant gate names (e.g., `lsargs` = ['Gi,
        'Gxpi, 'Gypi', 'Gzpi']).

    rand_state: RandomState, optional
        A np.random.RandomState object for seeding RNG

    Returns
    -------
    Circuit
        A random circuit of length `length` (if not addlocal) or length 2*`length`+1 (if addlocal)
        with layers independently sampled using the specified sampling distribution.
    """
    if samplerargs is None:
        samplerargs = []
    if lsargs is None:
        lsargs = []
    if rand_state is None:
        rand_state = _np.random.RandomState()

    if isinstance(sampler, str):

        # Removed redundant sampler
        #if sampler == 'pairingQs': sampler = sample_circuit_layer_by_pairing_qubits
        if sampler == 'Qelimination': sampler = sample_circuit_layer_by_q_elimination
        elif sampler == 'co2Qgates':
            sampler = sample_circuit_layer_by_co2_q_gates
            assert(len(samplerargs) >= 1), \
                ("The samplerargs must at least a 1-element list with the first element "
                 "the 'co2Qgates' argument of the co2Qgates sampler.")
        elif sampler == 'edgegrab':
            sampler = sample_circuit_layer_by_edgegrab
            assert(len(samplerargs) >= 1), \
                ("The samplerargs must at least a 1-element list")
        elif sampler == 'local': sampler = sample_circuit_layer_of_one_q_gates
        else: raise ValueError("Sampler type not understood!")

    if qubit_labels is not None:
        assert(isinstance(qubit_labels, list) or isinstance(qubit_labels, tuple)), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list
    else:
        qubits = list(pspec.qubit_labels[:])  # copy this list

    # Initialize an empty circuit, to populate with sampled layers.
    circuit = _cir.Circuit(layer_labels=[], line_labels=qubits, editable=True)

    # If we are not add layers of random local gates between the layers, sample 'length' layers
    # according to the sampler `sampler`.
    if not addlocal:
        for i in range(0, length):
            layer = sampler(pspec, qubit_labels, *samplerargs, rand_state=rand_state)
            circuit.insert_layer_inplace(layer, 0)

    # If we are adding layers of random local gates between the layers.
    if addlocal:
        for i in range(0, 2 * length + 1):
            local = not bool(i % 2)
            # For odd layers, we uniformly sample the specified type of local gates.
            if local:
                layer = sample_circuit_layer_of_one_q_gates(pspec, qubit_labels, *lsargs, rand_state=rand_state)
            # For even layers, we sample according to the given distribution
            else:
                layer = sampler(pspec, qubit_labels, *samplerargs, rand_state=rand_state)
            circuit.insert_layer_inplace(layer, 0)

    circuit.done_editing()
    return circuit


def create_direct_rb_circuit(pspec, clifford_compilations, length, qubit_labels=None, sampler='Qelimination',
                             samplerargs=None, addlocal=False, lsargs=None, randomizeout=True, cliffordtwirl=True,
                             conditionaltwirl=True, citerations=20, compilerargs=None, partitioned=False, seed=None):
    """
    Generates a "direct randomized benchmarking" (DRB) circuit.

    DRB is the protocol introduced in arXiv:1807.07975 (2018). The length of the "core" circuit is
    given by `length` and may be any integer >= 0. An n-qubit DRB circuit consists of (1) a circuit
    the prepares a uniformly random stabilizer state; (2) a length-l circuit (specified by `length`)
    consisting of circuit layers sampled according to some user-specified distribution (specified by
    `sampler`), (3) a circuit that maps the output of the preceeding circuit to a computational
    basis state. See arXiv:1807.07975 (2018) for further details.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit is being sampled for, which defines the
        "native" gate-set and the connectivity of the device. The returned DRB circuit will be over
        the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
        is always handed to the sampler, as the first argument of the sampler function (this is only
        of importance when not using an in-built sampler for the "core" of the DRB circuit). Unless
        `qubit_labels` is not None, the circuit is sampled over all the qubits in `pspec`.

    clifford_compilation : CompilationRules
        Rules for compiling the "native" gates of `pspec` into Clifford gates.

    length : int
        The "direct RB length" of the circuit, which is closely related to the circuit depth. It
        must be an integer >= 0. Unless `addlocal` is True, it is the depth of the "core" random
        circuit, sampled according to `sampler`, specified in step (2) above. If `addlocal` is True,
        each layer in the "core" circuit sampled according to "sampler` is followed by a layer of
        1-qubit gates, with sampling specified by `lsargs` (and the first layer is proceeded by a
        layer of 1-qubit gates), and so the circuit of step (2) is length 2*`length` + 1.

    qubit_labels : list, optional
        If not None, a list of the qubits to sample the circuit for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit is sampled to act on all the qubits
        in `pspec`.

    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
        a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function].
        If `sampler` is a function, it should be a function that takes as the first argument a
        QubitProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
        the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is
        the only sampler that requires no parameters beyond the QubitProcessorSpec *and* works for arbitrary
        connectivity devices. See the docstrings for each of these samplers for more information.

    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec`, the second argument is `qubit_labels`,
        and `samplerargs` lists the remaining arguments handed to the sampler. This is not
        optional for some choices of `sampler`.

    addlocal : bool, optional
        Whether to follow each layer in the "core" circuit, sampled according to `sampler` with
        a layer of 1-qubit gates.

    lsargs : list, optional
        Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
        layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.

    randomizeout : bool, optional
        If False, the ideal output of the circuit (the "success" or "survival" outcome) is the all-zeros
        bit string. If True, the ideal output of the circuit is randomized to a uniformly random bit-string.
        This setting is useful for, e.g., detecting leakage/loss/measurement-bias etc.

    cliffordtwirl : bool, optional
        Wether to begin the circuit with a sequence that generates a random stabilizer state. For
        standard DRB this should be set to True. There are a variety of reasons why it is better
        to have this set to True.

    conditionaltwirl : bool, optional
        DRB only requires that the initial/final sequences of step (1) and (3) create/measure
        a uniformly random / particular stabilizer state, rather than implement a particular unitary.
        step (1) and (3) can be achieved by implementing a uniformly random Clifford gate and the
        unique inversion Clifford, respectively. This is implemented if `conditionaltwirl` is False.
        However, steps (1) and (3) can be implemented much more efficiently than this: the sequences
        of (1) and (3) only need to map a particular input state to a particular output state,
        if `conditionaltwirl` is True this more efficient option is chosen -- this is option corresponds
        to "standard" DRB. (the term "conditional" refers to the fact that in this case we essentially
        implementing a particular Clifford conditional on a known input).

    citerations : int, optional
        Some of the stabilizer state / Clifford compilation algorithms in pyGSTi (including the default
        algorithms) are  randomized, and the lowest-cost circuit is chosen from all the circuit generated
        in the iterations of the algorithm. This is the number of iterations used. The time required to
        generate a DRB circuit is linear in `citerations`. Lower-depth / lower 2-qubit gate count
        compilations of steps (1) and (3) are important in order to successfully implement DRB on as many
        qubits as possible.

    compilerargs : list, optional
        A list of arguments that are handed to the compile_stabilier_state/measurement()functions (or the
        compile_clifford() function if `conditionaltwirl `is False). This includes all the optional
        arguments of these functions *after* the `iterations` option (set by `citerations`). For most
        purposes the default options will be suitable (or at least near-optimal from the compilation methods
        in-built into pyGSTi). See the docstrings of these functions for more information.

    partitioned : bool, optional
        If False, a single circuit is returned consisting of the full circuit. If True, three circuits
        are returned in a list consisting of: (1) the stabilizer-prep circuit, (2) the core random circuit,
        (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1)
        and then (3) to (1).

    seed : int, optional
        A seed to initialize the random number generator used for creating random clifford
        circuits.

    Returns
    -------
    Circuit or list of Circuits
        If partioned is False, a random DRB circuit sampled as specified. If partioned is True, a list of
        three circuits consisting of (1) the stabilizer-prep circuit, (2) the core random circuit,
        (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1)
        and then (3) to (1) [except in the case of cliffordtwirl=False, when it is a list of two circuits].
    Tuple
        A length-n tuple of integers in [0,1], corresponding to the error-free outcome of the
        circuit. Always all zeros if `randomizeout` is False. The ith element of the tuple
        corresponds to the error-free outcome for the qubit labelled by: the ith element of
        `qubit_labels`, if `qubit_labels` is not None; the ith element of `pspec.qubit_labels`, otherwise.
        In both cases, the ith element of the tuple corresponds to the error-free outcome for the
        qubit on the ith wire of the output circuit.
    """
    if samplerargs is None:
        samplerargs = []
    if compilerargs is None:
        compilerargs = []
    if lsargs is None:
        lsargs = []
    if qubit_labels is not None: n = len(qubit_labels)
    else: n = pspec.num_qubits

    rand_state = _np.random.RandomState(seed)  # Ok if seed is None

    # Sample a random circuit of "native gates".
    circuit = create_random_circuit(pspec=pspec, length=length, qubit_labels=qubit_labels, sampler=sampler,
                                    samplerargs=samplerargs, addlocal=addlocal, lsargs=lsargs, rand_state=rand_state)
    # find the symplectic matrix / phase vector this "native gates" circuit implements.
    s_rc, p_rc = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)

    # If we are clifford twirling, we do an initial random circuit that is either a uniformly random
    # cliffor or creates a uniformly random stabilizer state from the standard input.
    if cliffordtwirl:
        # Sample a uniformly random Clifford.
        s_initial, p_initial = _symp.random_clifford(n, rand_state=rand_state)
        # Find the composite action of this uniformly random clifford and the random circuit.
        s_composite, p_composite = _symp.compose_cliffords(s_initial, p_initial, s_rc, p_rc)
        # If conditionaltwirl we do a stabilizer prep (a conditional Clifford).
        if conditionaltwirl:
            initial_circuit = _cmpl.compile_stabilizer_state(s_initial, p_initial, pspec,
                                                             clifford_compilations.get('absolute', None),
                                                             clifford_compilations.get('paulieq', None),
                                                             qubit_labels, citerations,
                                                             *compilerargs, rand_state=rand_state)
        # If not conditionaltwirl, we do a full random Clifford.
        else:
            initial_circuit = _cmpl.compile_clifford(s_initial, p_initial, pspec,
                                                     clifford_compilations.get('absolute', None),
                                                     clifford_compilations.get('paulieq', None),
                                                     qubit_labels, citerations,
                                                     *compilerargs, rand_state=rand_state)
    # If we are not Clifford twirling, we just copy the effect of the random circuit as the effect
    # of the "composite" prep + random circuit (as here the prep circuit is the null circuit).
    else:
        s_composite = _copy.deepcopy(s_rc)
        p_composite = _copy.deepcopy(p_rc)

    if conditionaltwirl:
        # If we want to randomize the expected output then randomize the p vector, otherwise
        # it is left as p. Note that, unlike with compile_clifford, we don't invert (s,p)
        # before handing it to the stabilizer measurement function.
        if randomizeout: p_for_measurement = _symp.random_phase_vector(s_composite, n, rand_state=rand_state)
        else: p_for_measurement = p_composite
        inversion_circuit = _cmpl.compile_stabilizer_measurement(s_composite, p_for_measurement, pspec,
                                                                 clifford_compilations.get('absolute', None),
                                                                 clifford_compilations.get('paulieq', None),
                                                                 qubit_labels,
                                                                 citerations, *compilerargs, rand_state=rand_state)
    else:
        # Find the Clifford that inverts the circuit so far. We
        s_inverse, p_inverse = _symp.inverse_clifford(s_composite, p_composite)
        # If we want to randomize the expected output then randomize the p_inverse vector, otherwise
        # do not.
        if randomizeout: p_for_inversion = _symp.random_phase_vector(s_inverse, n, rand_state=rand_state)
        else: p_for_inversion = p_inverse
        # Compile the Clifford.
        inversion_circuit = _cmpl.compile_clifford(s_inverse, p_for_inversion, pspec,
                                                   clifford_compilations.get('absolute', None),
                                                   clifford_compilations.get('paulieq', None),
                                                   qubit_labels, citerations, *compilerargs, rand_state=rand_state)
    if cliffordtwirl:
        full_circuit = initial_circuit.copy(editable=True)
        full_circuit.append_circuit_inplace(circuit)
        full_circuit.append_circuit_inplace(inversion_circuit)
    else:
        full_circuit = circuit.copy(editable=True)
        full_circuit.append_circuit_inplace(inversion_circuit)

    full_circuit.done_editing()

    # Find the expected outcome of the circuit.
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(full_circuit, pspec=pspec)
    if conditionaltwirl:  # s_out is not always the identity with a conditional twirl, only conditional on prep/measure.
        assert(_np.array_equal(s_out[:n, n:], _np.zeros((n, n), _np.int64))), "Compiler has failed!"
    else: assert(_np.array_equal(s_out, _np.identity(2 * n, _np.int64))), "Compiler has failed!"

    # Find the ideal output of the circuit.
    s_inputstate, p_inputstate = _symp.prep_stabilizer_state(n, zvals=None)
    s_outstate, p_outstate = _symp.apply_clifford_to_stabilizer_state(s_out, p_out, s_inputstate, p_inputstate)
    idealout = []
    for q in range(0, n):
        measurement_out = _symp.pauli_z_measurement(s_outstate, p_outstate, q)
        bit = measurement_out[1]
        assert(bit == 0 or bit == 1), "Ideal output is not a computational basis state!"
        if not randomizeout:
            assert(bit == 0), "Ideal output is not the all 0s computational basis state!"
        idealout.append(int(measurement_out[1]))
    idealout = tuple(idealout)

    if not partitioned: outcircuit = full_circuit
    else:
        if cliffordtwirl: outcircuit = [initial_circuit, circuit, inversion_circuit]
        else: outcircuit = [circuit, inversion_circuit]

    return outcircuit, idealout


def _sample_clifford_circuit(pspec, clifford_compilations, qubit_labels, citerations,
        compilerargs, exact_compilation_key, srep_cache, rand_state):
    """Helper function to compile a random Clifford circuit.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit is being sampled for, which defines the
        "native" gate-set and the connectivity of the device. The returned CRB circuit will be over
        the gates in `pspec`, and will respect the connectivity encoded by `pspec`.

    clifford_compilations : dict
        A dictionary with at least the potential keys `'absolute'` and `'paulieq'` and corresponding
        :class:`CompilationRules` values.  These compilation rules specify how to compile the
        "native" gates of `pspec` into Clifford gates. Additional :class:`CompilationRules` can be
        provided, particularly for use with `exact_compilation_key`.

    qubit_labels : list
        A list of the qubits that the RB circuit is to be sampled for.

    citerations : int
        Some of the Clifford compilation algorithms in pyGSTi (including the default algorithm) are
        randomized, and the lowest-cost circuit is chosen from all the circuit generated in the
        iterations of the algorithm. This is the number of iterations used. The time required to
        generate a CRB circuit is linear in `citerations` * (`length` + 2). Lower-depth / lower 2-qubit
        gate count compilations of the Cliffords are important in order to successfully implement
        CRB on more qubits.

    compilerargs : list
        A list of arguments that are handed to compile_clifford() function, which includes all the
        optional arguments of compile_clifford() *after* the `iterations` option (set by `citerations`).
        In order, this list should be values for:
        
        algorithm : str. A string that specifies the compilation algorithm. The default in
        compile_clifford() will always be whatever we consider to be the 'best' all-round
        algorithm
        
        aargs : list. A list of optional arguments for the particular compilation algorithm.
        
        costfunction : 'str' or function. The cost-function from which the "best" compilation
        for a Clifford is chosen from all `citerations` compilations. The default costs a
        circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
        
        prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
        
        paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a
        random Pauli on each qubit (compiled into native gates). I.e., if this is True the
        native gates are Pauli-randomized. When True, this prevents any coherent errors adding
        (on average) inside the layers of each compiled Clifford, at the cost of increased
        circuit depth. Defaults to False.
        
        For more information on these options, see the `:func:compile_clifford()` docstring.
    
    exact_compilation_key: str, optional
        The key into `clifford_compilations` to use for exact deterministic complation of Cliffords.
        The underlying :class:`CompilationRules` object must provide compilations for all possible
        n-qubit Cliffords that will be generated. This also requires the pspec is able to generate the
        symplectic representations for all n-qubit Cliffords in :meth:`compute_clifford_symplectic_reps`.
        This is currently generally intended for use out-of-the-box with 1-qubit Clifford RB;
        however, larger number of qubits can be used so long as the user specifies the processor spec and
        compilation rules properly.

    srep_cache: dict
        Keys are gate labels and values are precomputed symplectic representations.
    
    rand_state: np.random.RandomState
        A RandomState to use for RNG

    Returns
    -------
    clifford_circuit : Circuit
        The compiled Clifford circuit
    
    s:
        The symplectic matrix of the Clifford
    
    p:
        The symplectic phase vector of the Clifford
    """
    # Find the labels of the qubits to create the circuit for.
    if qubit_labels is not None: qubits = qubit_labels[:]  # copy this list
    else: qubits = pspec.qubit_labels[:]  # copy this list
    # The number of qubits the circuit is over.
    n = len(qubits)

    if exact_compilation_key is not None:
        # Deterministic compilation based on a provided clifford compilation
        assert exact_compilation_key in clifford_compilations, \
                f"{exact_compilation_key} not provided in `clifford_compilations`"
        
        # Pick clifford
        cidx = rand_state.randint(_symp.compute_num_cliffords(n))
        lbl = _lbl.Label(f'C{cidx}', qubits)
        
        # Try to do deterministic compilation
        try:
            circuit = clifford_compilations[exact_compilation_key].retrieve_compilation_of(lbl)
        except AssertionError:
            raise ValueError(
                f"Failed to compile n-qubit Clifford 'C{cidx}'. Ensure this is provided in the " + \
                "compilation rules, or use a compilation algorithm to synthesize it by not " + \
                "specifying `exact_compilation_key`."
            )
    
        # compute the symplectic rep of the chosen clifford
        # TODO: Note that this is inefficient. For speed, we could implement the pair to
        # _symp.compute_symplectic_matrix and just calculate s and p directly
        s, p = _symp.symplectic_rep_of_clifford_circuit(circuit, srep_cache)
    else:
        # Random compilation
        s, p = _symp.random_clifford(n, rand_state=rand_state)
        circuit = _cmpl.compile_clifford(s, p, pspec,
                                        clifford_compilations.get('absolute', None),
                                        clifford_compilations.get('paulieq', None),
                                        qubit_labels=qubit_labels, iterations=citerations, *compilerargs,
                                            rand_state=rand_state)
    
    return circuit, s, p


def create_clifford_rb_circuit(pspec, clifford_compilations, length, qubit_labels=None, randomizeout=False,
                               citerations=20, compilerargs=None, interleaved_circuit=None, seed=None,
                               return_native_gate_counts=False, exact_compilation_key=None):
    """
    Generates a "Clifford randomized benchmarking" (CRB) circuit.

    CRB is the current-standard RB protocol defined in "Scalable and robust randomized benchmarking of quantum
    processes", Magesan et al. PRL 106 180504 (2011). This consists of a circuit of `length`+1 uniformly
    random n-qubit Clifford gates followed by the unique inversion Clifford, with all the Cliffords compiled
    into the "native" gates of a device as specified by `pspec`. The circuit output by this function will
    respect the connectivity of the device, as encoded into `pspec` (see the QubitProcessorSpec object docstring
    for how to construct the relevant `pspec`).

    Note the convention that the the output Circuit consists of `length+2` Clifford gates, rather than the
    more usual convention of defining the "CRB length" to be the number of Clifford gates - 1. This is for
    consistency with the other RB functions in pyGSTi: in all RB-circuit-generating functions in pyGSTi
    length zero corresponds to the minimum-length circuit allowed by the protocol. Note that changing the
    "RB depths" by a constant additive factor is irrelevant for fitting purposes (except that it changes
    the obtained "SPAM" fit parameter).

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit is being sampled for, which defines the
        "native" gate-set and the connectivity of the device. The returned CRB circuit will be over
        the gates in `pspec`, and will respect the connectivity encoded by `pspec`.

    clifford_compilations : dict
        A dictionary with at least the potential keys `'absolute'` and `'paulieq'` and corresponding
        :class:`CompilationRules` values.  These compilation rules specify how to compile the
        "native" gates of `pspec` into Clifford gates. Additional :class:`CompilationRules` can be
        provided, particularly for use with `exact_compilation_key`.

    length : int
        The "CRB length" of the circuit -- an integer >= 0 --  which is the number of Cliffords in the
        circuit - 2 *before* each Clifford is compiled into the native gate-set.

    qubit_labels : list, optional
        If not None, a list of the qubits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the "wires" in the returned circuit, but is otherwise
        irrelevant. If desired, a circuit that explicitly idles on the other qubits can be obtained
        by using methods of the Circuit object.

    randomizeout : bool, optional
        If False, the ideal output of the circuit (the "success" or "survival" outcome) is the all-zeros
        bit string. This is probably considered to be the "standard" in CRB. If True, the ideal output
        of the circuit is randomized to a uniformly random bit-string. This setting is useful for, e.g.,
        detecting leakage/loss/measurement-bias etc.

    citerations : int, optional
        Some of the Clifford compilation algorithms in pyGSTi (including the default algorithm) are
        randomized, and the lowest-cost circuit is chosen from all the circuit generated in the
        iterations of the algorithm. This is the number of iterations used. The time required to
        generate a CRB circuit is linear in `citerations` * (`length` + 2). Lower-depth / lower 2-qubit
        gate count compilations of the Cliffords are important in order to successfully implement
        CRB on more qubits.

    compilerargs : list, optional
        A list of arguments that are handed to compile_clifford() function, which includes all the
        optional arguments of compile_clifford() *after* the `iterations` option (set by `citerations`).
        In order, this list should be values for:
        
        algorithm : str. A string that specifies the compilation algorithm. The default in
        compile_clifford() will always be whatever we consider to be the 'best' all-round
        algorithm
        
        aargs : list. A list of optional arguments for the particular compilation algorithm.
        
        costfunction : 'str' or function. The cost-function from which the "best" compilation
        for a Clifford is chosen from all `citerations` compilations. The default costs a
        circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
        
        prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
        
        paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a
        random Pauli on each qubit (compiled into native gates). I.e., if this is True the
        native gates are Pauli-randomized. When True, this prevents any coherent errors adding
        (on average) inside the layers of each compiled Clifford, at the cost of increased
        circuit depth. Defaults to False.
        
        For more information on these options, see the `:func:compile_clifford()` docstring.

    seed : int, optional
        A seed to initialize the random number generator used for creating random clifford
        circuits.
    
    return_native_gate_counts: bool, optional
        Whether to return the number of native gates in the first `length`+1 compiled Cliffords
    
    exact_compilation_key: str, optional
        The key into `clifford_compilations` to use for exact deterministic complation of Cliffords.
        The underlying :class:`CompilationRules` object must provide compilations for all possible
        n-qubit Cliffords that will be generated. This also requires the pspec is able to generate the
        symplectic representations for all n-qubit Cliffords in :meth:`compute_clifford_symplectic_reps`.
        This is currently generally intended for use out-of-the-box with 1-qubit Clifford RB;
        however, larger number of qubits can be used so long as the user specifies the processor spec and
        compilation rules properly.

    Returns
    -------
    full_circuit : Circuit
        A random CRB circuit over the "native" gate-set specified.
        
    idealout : tuple
        A length-n tuple of integers in [0,1], corresponding to the error-free outcome of the
        circuit. Always all zeros if `randomizeout` is False. The ith element of the tuple
        corresponds to the error-free outcome for the qubit labelled by: the ith element of
        `qubit_labels`, if `qubit_labels` is not None; the ith element of `pspec.qubit_labels`, otherwise.
        In both cases, the ith element of the tuple corresponds to the error-free outcome for the
        qubit on the ith wire of the output circuit.
    
    native_gate_counts: dict
        Total number of native gates, native 2q gates, and native circuit size in the
        first `length`+1 compiled Cliffords. Only returned when `return_num_native_gates` is True
    """
    if compilerargs is None:
        compilerargs = []
    # Find the labels of the qubits to create the circuit for.
    if qubit_labels is not None: qubits = qubit_labels[:]  # copy this list
    else: qubits = pspec.qubit_labels[:]  # copy this list
    # The number of qubits the circuit is over.
    n = len(qubits)

    srep_cache = {}
    if exact_compilation_key is not None:
        # Precompute some of the symplectic reps if we are doing exact compilation
        srep_cache = _symp.compute_internal_gate_symplectic_representations()
        srep_cache.update(pspec.compute_clifford_symplectic_reps())

    rand_state = _np.random.RandomState(seed)  # OK if seed is None

    # Initialize the identity circuit rep.
    s_composite = _np.identity(2 * n, _np.int64)
    p_composite = _np.zeros((2 * n), _np.int64)
    # Initialize an empty circuit
    full_circuit = _cir.Circuit(layer_labels=[], line_labels=qubits, editable=True)

    # Sample length+1 uniformly random Cliffords (we want a circuit of length+2 Cliffords, in total), compile
    # them, and append them to the current circuit.
    num_native_gates = 0
    num_native_2q_gates = 0
    native_size = 0
    for _ in range(0, length + 1):
        # Perform sampling
        circuit, s, p = _sample_clifford_circuit(pspec, clifford_compilations, qubit_labels, citerations,
                                 compilerargs, exact_compilation_key, srep_cache, rand_state)
        num_native_gates += circuit.num_gates
        num_native_2q_gates += circuit.num_nq_gates(2)
        native_size += circuit.size

        # Keeps track of the current composite Clifford
        s_composite, p_composite = _symp.compose_cliffords(s_composite, p_composite, s, p)
        full_circuit.append_circuit_inplace(circuit)
        if interleaved_circuit is not None:
            s, p = _symp.symplectic_rep_of_clifford_circuit(interleaved_circuit, pspec=pspec)
            s_composite, p_composite = _symp.compose_cliffords(s_composite, p_composite, s, p)
            full_circuit.append_circuit_inplace(interleaved_circuit)

    # Find the symplectic rep of the inverse clifford
    s_inverse, p_inverse = _symp.inverse_clifford(s_composite, p_composite)

    # If we want to randomize the expected output then randomize the p_inverse vector, so that
    # the final bit of circuit will only invert the preceeding circuit up to a random Pauli.
    if randomizeout: p_for_inversion = _symp.random_phase_vector(s_inverse, n, rand_state=rand_state)
    else: p_for_inversion = p_inverse

    # Compile the inversion circuit
    inversion_circuit = _cmpl.compile_clifford(s_inverse, p_for_inversion, pspec,
                                               clifford_compilations.get('absolute', None),
                                               clifford_compilations.get('paulieq', None),
                                               qubit_labels=qubit_labels,
                                               iterations=citerations, *compilerargs, rand_state=rand_state)
    full_circuit.append_circuit_inplace(inversion_circuit)
    full_circuit.done_editing()
    # Find the expected outcome of the circuit.
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(full_circuit, pspec=pspec)
    # Check the output is the identity up to Paulis.
    assert(_np.array_equal(s_out[:n, n:], _np.zeros((n, n), _np.int64)))
    # Find the ideal-out of the circuit, as a bit-string.
    s_inputstate, p_inputstate = _symp.prep_stabilizer_state(n, zvals=None)
    s_outstate, p_outstate = _symp.apply_clifford_to_stabilizer_state(s_out, p_out, s_inputstate, p_inputstate)
    idealout = []
    for q in range(n):
        measurement_out = _symp.pauli_z_measurement(s_outstate, p_outstate, q)
        # This is the probability of the 0 outcome (it is a float)
        bit = measurement_out[1]
        assert(_np.allclose(bit, 0.) or _np.allclose(bit, 1.)), "Ideal output is not a computational basis state!"
        if not randomizeout: assert(_np.allclose(bit, 0.)), "Ideal output is not the all 0s computational basis state!"
        idealout.append(round(measurement_out[1]))
    # Convert ideal-out to a tuple, so that it is imutable
    idealout = tuple(idealout)

    full_circuit.done_editing()

    native_gate_counts = {
        "native_gate_count": num_native_gates,
        "native_2q_gate_count": num_native_2q_gates,
        "native_size": native_size
    }

    if return_native_gate_counts:
        return full_circuit, idealout, native_gate_counts
    return full_circuit, idealout


def sample_pauli_layer_as_compiled_circuit(pspec, absolute_compilation, qubit_labels=None, keepidle=False,
                                           rand_state=None):
    """
    Samples a uniformly random n-qubit Pauli and converts it to the gate-set of `pspec`.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device.

    absolute_compilation : CompilationRules
        Rules for exactly (absolutely) compiling the "native" gates of `pspec` into clifford gates.

    qubit_labels : list, optional
        If not None, a list of a subset of the qubits from `pspec` that
        the pauli circuit should act on.

    keepidle : bool, optional
        Whether to always have the circuit at-least depth 1.

    rand_state: RandomState, optional
        A np.random.RandomState object for seeding RNG

    Returns
    -------
    Circuit
        A circuit corresponding to a uniformly random n-qubit Pauli,
        converted to the native gate-set of `pspec`.
    """
    if qubit_labels is not None: qubits = qubit_labels[:]  # copy this list
    else: qubits = pspec.qubit_labels[:]  # copy this list
    n = len(qubits)

    if rand_state is None:
        rand_state = _np.random.RandomState()

    # The hard-coded notation for that Pauli operators
    paulis = ['I', 'X', 'Y', 'Z']

    # Samples a random Pauli layer
    r = rand_state.randint(0, 4, size=n)
    pauli_layer_std_lbls = [_lbl.Label(paulis[r[q]], (qubits[q],)) for q in range(n)]
    # Converts the layer to a circuit, and changes to the native model.
    pauli_circuit = _cir.Circuit(layer_labels=pauli_layer_std_lbls, line_labels=qubits).parallelize()
    pauli_circuit = pauli_circuit.copy(editable=True)
    pauli_circuit.change_gate_library(absolute_compilation)
    if keepidle:
        if pauli_circuit.depth == 0:
            pauli_circuit.insert_layer_inplace([_lbl.Label(())], 0)

    pauli_circuit.done_editing()
    return pauli_circuit


def sample_one_q_clifford_layer_as_compiled_circuit(pspec, absolute_compilation, qubit_labels=None, rand_state=None):
    """
    Samples a uniformly random layer of 1-qubit Cliffords.

    Create a uniformly random layer of 1-qubit Cliffords on all
    the qubits, and then converts it to the native gate-set of `pspec`.
    That is, an independent and uniformly random 1-qubit Clifford is
    sampled for each qubit.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device.

    absolute_compilation : CompilationRules
        Rules for exactly (absolutely) compiling the "native" gates of `pspec` into clifford gates.

    qubit_labels : list, optional
        If not None, a list of a subset of the qubits from `pspec` that
        the circuit should act on.

    rand_state: RandomState, optional
        A np.random.RandomState object for seeding RNG

    Returns
    -------
    Circuit
        A circuit corresponding to an independent, uniformly random 1-qubit
        Clifford gate on each qubit.
    """
    if qubit_labels is not None:
        n = len(qubit_labels)
        qubits = qubit_labels[:]  # copy this list
    else:
        n = pspec.num_qubits
        qubits = pspec.qubit_labels[:]  # copy this list

    # The hard-coded notation for the 1Q clifford operators
    oneQcliffords = ['C' + str(i) for i in range(24)]

    r = rand_state.randint(0, 24, size=n)

    oneQclifford_layer_std_lbls = [_lbl.Label(oneQcliffords[r[q]], (qubits[q],)) for q in range(n)]
    oneQclifford_circuit = _cir.Circuit(layer_labels=oneQclifford_layer_std_lbls, line_labels=qubits).parallelize()
    oneQclifford_circuit = oneQclifford_circuit.copy(editable=True)
    oneQclifford_circuit.change_gate_library(absolute_compilation)
    oneQclifford_circuit.done_editing()

    if len(oneQclifford_circuit) == 0:
        oneQclifford_circuit = _cir.Circuit(([],), line_labels=qubits)

    return oneQclifford_circuit


def create_mirror_rb_circuit(pspec, absolute_compilation, length, qubit_labels=None, sampler='Qelimination',
                             samplerargs=None, localclifford=True, paulirandomize=True, seed=None):
    """
    Generates a "mirror randomized benchmarking" (MRB) circuit.

    This is specific to the case of Clifford gates and can be performed, optionally, with Pauli-randomization
    and Clifford-twirling. This RB method is currently in development; this docstring will be updated in the
    future with further information on this technique.

    To implement mirror RB it is necessary for U^(-1) to in the gate-set for every U in the gate-set.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit is being sampled for. The `pspec` is always
        handed to the sampler, as the first argument of the sampler function.

    absolute_compilation : CompilationRules
        Rules for exactly (absolutely) compiling the "native" gates of `pspec` into clifford gates.

    length : int
        The "mirror RB length" of the circuit, which is closely related to the circuit depth. It
        must be an even integer, and can be zero.

        If `localclifford` and `paulirandomize` are False, this is the depth of the sampled circuit.
        The first length/2 layers are all sampled independently according to the sampler specified by
        `sampler`. The remaining half of the circuit is the "inversion" circuit that is determined
        by the first half.

        If `paulirandomize` is True and `localclifford` is False, the depth of the circuits is
        2*length+1 with odd-indexed layers sampled according to the sampler specified by `sampler`, and
        the the zeroth layer + the even-indexed layers consisting of random 1-qubit Pauli gates.

        If `paulirandomize` and `localclifford` are True, the depth of the circuits is
        2*length+1 + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for
        the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

        If `paulirandomize` is False and `localclifford` is True, the depth of the circuits is
        length + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for
        the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

    qubit_labels : list, optional
        If not None, a list of the qubits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the "wires" in the returned circuit, but is otherwise
        irrelevant.

    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
        a valid option for n-qubit MRB -- it results in sim. 1-qubit MRB -- but it is not explicitly
        forbidden by this function]. If `sampler` is a function, it should be a function that takes
        as the first argument a QubitProcessorSpec, and returns a random circuit layer as a list of gate
        Label objects. Note that the default 'Qelimination' is not necessarily the most useful
        in-built sampler, but it is the only sampler that requires no parameters beyond the QubitProcessorSpec
        *and* works for arbitrary connectivity devices. See the docstrings for each of these samplers
        for more information.

    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec` and `samplerargs` lists the
        remaining arguments handed to the sampler.

    localclifford : bool, optional
        Whether to start the circuit with uniformly random 1-qubit Cliffords and all of the
        qubits (compiled into the native gates of the device).

    paulirandomize : bool, optional
        Whether to have uniformly random Pauli operators on all of the qubits before and
        after all of the layers in the "out" and "back" random circuits. At length 0 there
        is a single layer of random Pauli operators (in between two layers of 1-qubit Clifford
        gates if `localclifford` is True); at length l there are 2l+1 Pauli layers as there
        are

    seed : int, optional
        A seed to initialize the random number generator used for creating random clifford
        circuits.

    Returns
    -------
    Circuit
        A random MRB circuit, sampled as specified, of depth:

        `length`, if not paulirandomize and not local clifford.
        
        2*`length`+1 if paulirandomize and not local clifford.
        
        `length` + X, if not paulirandomize and local clifford, where X is a random variable
        that accounts for the depth from the layers of random 1-qubit Cliffords (X = 2 if the 1
        qubit Clifford gates are "native" gates in the QubitProcessorSpec).
        
        2*`length`+1 + X, if paulirandomize and local clifford, where X is a random variable
        that accounts for the depth from the layers of random 1-qubit Cliffords (X = 2 if the 1
        qubit Clifford gates are "native" gates in the QubitProcessorSpec).
        
    Tuple
        A length-n tuple of integers in [0,1], corresponding to the error-free outcome of the
        circuit. Always all zeros if `randomizeout` is False. The ith element of the tuple
        corresponds to the error-free outcome for the qubit labelled by: the ith element of
        `qubit_labels`, if `qubit_labels` is not None; the ith element of `pspec.qubit_labels`, otherwise.
        In both cases, the ith element of the tuple corresponds to the error-free outcome for the
        qubit on the ith wire of the output circuit.
    """
    if samplerargs is None:
        samplerargs = []
    assert(length % 2 == 0), "The mirror rb length `length` must be even!"
    random_natives_circuit_length = length // 2

    rand_state = _np.random.RandomState(seed)  # OK if seed is None

    if qubit_labels is not None:
        assert(isinstance(qubit_labels, list) or isinstance(qubit_labels, tuple)
               ), "If not None, `qubit_labels` must be a list!"
        qubit_labels = list(qubit_labels)
        n = len(qubit_labels)
    else:
        n = pspec.num_qubits

    # Check that the inverse of every gate is in the model:
    _, gate_inverse = pspec.compute_one_qubit_gate_relations()
    gate_inverse.update(pspec.compute_multiqubit_inversion_relations())  # add multiQ inverses
    for gname in pspec.gate_names:
        assert(gname in gate_inverse), \
            "%s gate does not have an inverse in the gate-set! MRB is not possible!" % gname

    # Find a random circuit according to the sampling specified; this is the "out" circuit.
    circuit = create_random_circuit(pspec, random_natives_circuit_length, qubit_labels=qubit_labels,
                                    sampler=sampler, samplerargs=samplerargs, rand_state=rand_state)
    circuit = circuit.copy(editable=True)
    # Copy the circuit, to create the "back" circuit from the "out" circuit.
    circuit_inv = circuit.copy(editable=True)
    # First we reverse the circuit; then we'll replace each gate with its inverse.
    circuit_inv.reverse_inplace()
    # Go through the circuit and replace every gate with its inverse, stored in the pspec. If the circuits
    # are length 0 this is skipped.
    circuit_inv.map_names_inplace(gate_inverse)

    # If we are Pauli randomizing, we add a indepedent uniformly random Pauli layer, as a compiled circuit, after
    # every layer in the "out" and "back" circuits. If the circuits are length 0 we do nothing here.
    if paulirandomize:
        for i in range(random_natives_circuit_length):
            pauli_circuit = sample_pauli_layer_as_compiled_circuit(pspec, absolute_compilation,
                                                                   qubit_labels=qubit_labels, keepidle=True,
                                                                   rand_state=rand_state)
            circuit.insert_circuit_inplace(pauli_circuit, random_natives_circuit_length - i)
            pauli_circuit = sample_pauli_layer_as_compiled_circuit(pspec, absolute_compilation,
                                                                   qubit_labels=qubit_labels, keepidle=True,
                                                                   rand_state=rand_state)
            circuit_inv.insert_circuit_inplace(pauli_circuit, random_natives_circuit_length - i)

    # We then append the "back" circuit to the "out" circuit. At length 0 this will be a length 0 circuit.
    circuit.append_circuit_inplace(circuit_inv)

    # If we Pauli randomize, There should also be a random Pauli at the start of this circuit; so we add that. If we
    # have a length 0 circuit we now end up with a length 1 circuit (or longer, if compiled Paulis). So, there is always
    # a random Pauli.
    if paulirandomize:
        pauli_circuit = sample_pauli_layer_as_compiled_circuit(pspec, absolute_compilation, qubit_labels=qubit_labels,
                                                               keepidle=True, rand_state=rand_state)
        circuit.insert_circuit_inplace(pauli_circuit, 0)

    # If we start with a random layer of 1-qubit Cliffords, we sample this here.
    if localclifford:
        # Sample a compiled 1Q Cliffords layer
        oneQclifford_circuit_out = sample_one_q_clifford_layer_as_compiled_circuit(pspec, absolute_compilation,
                                                                                   qubit_labels=qubit_labels,
                                                                                   rand_state=rand_state)
        # Generate the inverse in the same way as before (note that this will not be the same in some
        # circumstances as finding the inverse Cliffords and using the compilations for those. It doesn't
        # matter much which we do).
        oneQclifford_circuit_back = oneQclifford_circuit_out.copy(editable=True)
        oneQclifford_circuit_back.reverse_inplace()
        oneQclifford_circuit_back.map_names_inplace(gate_inverse)

        # Put one these 1Q clifford circuits at the start and one at then end.
        circuit.append_circuit_inplace(oneQclifford_circuit_out)
        circuit.prefix_circuit_inplace(oneQclifford_circuit_back)

    circuit.done_editing()

    # The full circuit should be, up to a Pauli, the identity.
    s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
    assert(_np.array_equal(s_out, _np.identity(2 * n, _np.int64)))

    # Find the error-free output.
    s_inputstate, p_inputstate = _symp.prep_stabilizer_state(n, zvals=None)
    s_outstate, p_outstate = _symp.apply_clifford_to_stabilizer_state(s_out, p_out, s_inputstate, p_inputstate)
    idealout = []

    for q in range(n):
        measurement_out = _symp.pauli_z_measurement(s_outstate, p_outstate, q)
        bit = measurement_out[1]
        assert(bit == 0 or bit == 1), "Ideal output is not a computational basis state!"
        if not paulirandomize:
            assert(bit == 0), "Ideal output is not the all 0s computational basis state!"
        idealout.append(int(measurement_out[1]))
    idealout = tuple(idealout)

    return circuit, idealout


def create_random_germ(pspec, depths, interacting_qs_density, qubit_labels, rand_state=None):
    """
    TODO: docstring
    <TODO summary>

    Parameters
    ----------
    pspec : <TODO typ>
        <TODO description>

    depths : <TODO typ>
        <TODO description>

    interacting_qs_density : <TODO typ>
        <TODO description>

    qubit_labels : <TODO typ>
        <TODO description>

    Returns
    -------
    <TODO typ>
    """
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        qubits = list(qubit_labels[:])  # copy this list

    if rand_state is None:
        rand_state = _np.random.RandomState()

    width = len(qubits)

    if width == 1:
        interacting_qs_density = 0

    germcircuit = _cir.Circuit(layer_labels=[], line_labels=qubits, editable=True)

    rand = rand_state.rand()
    if rand < 4 / 8:
        max_subgerm_depth = 1
    elif rand < 6 / 8:
        max_subgerm_depth = 2
    elif rand < 7 / 8:
        max_subgerm_depth = 4
    else:
        max_subgerm_depth = 8

    if interacting_qs_density > 0:
        required_num_2Q_locations = max_subgerm_depth * width * interacting_qs_density
        R = int(_np.ceil(2 / required_num_2Q_locations))
    else:
        R = 1

    germ_depth = R * max_subgerm_depth

    subgerm_depth = {}
    for q in qubits:
        subgerm_depth_power = 0
        while (rand_state.binomial(1, 0.5) == 1) and (2 ** subgerm_depth_power < max_subgerm_depth):
            subgerm_depth_power += 1
        subgerm_depth[q] = 2 ** subgerm_depth_power

    subgerm = {}
    repeated_subgerm = {}
    clifford_ops_on_qubits = pspec.compute_clifford_ops_on_qubits()

    for q in qubits:
        subgerm[q] = []
        possibleops = clifford_ops_on_qubits[(q,)]
        subgerm[q] = [possibleops[rand_state.randint(0, len(possibleops))] for l in range(subgerm_depth[q])]
        repeated_subgerm[q] = (germ_depth // subgerm_depth[q]) * subgerm[q]

    for l in range(germ_depth):
        layer = [repeated_subgerm[q][l] for q in qubits]
        germcircuit.insert_layer_inplace(layer, 0)

    if interacting_qs_density > 0:

        assert(germ_depth * width * interacting_qs_density >= 2)
        #print(len(qubits))
        num2Qtoadd = int(_np.floor(germ_depth * width * interacting_qs_density / 2))
        #print(num2Qtoadd)

        edgelistdict = {}
        clifford_qubit_graph = pspec.compute_clifford_2Q_connectivity()
        for l in range(len(germcircuit)):

            # Prep the sampling variables.
            edgelist = clifford_qubit_graph.edges()
            edgelist = [e for e in edgelist if all([q in qubits for q in e])]
            selectededges = []

            # Go through until all qubits have been assigned a gate.
            while len(edgelist) > 0:

                edge = edgelist[rand_state.randint(0, len(edgelist))]
                selectededges.append(edge)
                # Delete all edges containing these qubits.
                edgelist = [e for e in edgelist if not any([q in e for q in edge])]

            edgelistdict[l] = selectededges

        edge_and_depth_list = []
        for l in edgelistdict.keys():
            edge_and_depth_list += [(l, edge) for edge in edgelistdict[l]]

        clifford_ops_on_qubits = pspec.compute_clifford_ops_on_qubits()
        for i in range(num2Qtoadd):

            sampind = rand_state.randint(0, len(edge_and_depth_list))
            (depthposition, edge) = edge_and_depth_list[sampind]
            del edge_and_depth_list[sampind]

            # The two-qubit gates on that edge.
            possibleops = clifford_ops_on_qubits[edge]
            op = possibleops[rand_state.randint(0, len(possibleops))]

            newlayer = []
            newlayer = [op] + [gate for gate in germcircuit[depthposition] if gate.qubits[0] not in edge]
            germcircuit.delete_layers(depthposition)
            germcircuit.insert_layer_inplace(newlayer, depthposition)

        germcircuit.done_editing()

    return germcircuit


def create_random_germpower_circuits(pspec, depths, interacting_qs_density, qubit_labels, fixed_versus_depth=False,
                                     rand_state=None):

    #import numpy as _np
    #from pygsti.circuits import circuit as _cir

    """
    TODO: docstring
    <TODO summary>

    Parameters
    ----------
    pspec : <TODO typ>
        <TODO description>

    depths : <TODO typ>
        <TODO description>

    interacting_qs_density : <TODO typ>
        <TODO description>

    qubit_labels : <TODO typ>
        <TODO description>

    fixed_versus_depth : <TODO typ>, optional
        <TODO description>

    rand_state: RandomState, optional
        A np.random.RandomState object for seeding RNG
    """
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        qubits = list(qubit_labels[:])  # copy this list

    if fixed_versus_depth:
        germcircuit = create_random_germ(pspec, depths, interacting_qs_density, qubit_labels, rand_state=rand_state)
    else:
        germcircuits = []

    if rand_state is None:
        rand_state = _np.random.RandomState()

    circs = []
    #germpowers = []
    for length in depths:
        gdepth = 0
        fullcircuit = _cir.Circuit(layer_labels=[], line_labels=qubits, editable=True)
        if not fixed_versus_depth:
            germcircuit = create_random_germ(pspec, depths, interacting_qs_density, qubit_labels, rand_state=rand_state)
            germcircuits.append(germcircuit)
        while len(fullcircuit) < length:
            fullcircuit.append_circuit_inplace(germcircuit)
            gdepth += 1

        while len(fullcircuit) > length:
            fullcircuit.delete_layers(len(fullcircuit) - 1)

        circs.append(fullcircuit)
        #germpowers.append(gdepth)

    aux = {  # 'germ_powers': germpowers,
        #'subgerm_depth': subgerm_depth,
        #'max_subgerm_depth': max_subgerm_depth
    }

    if fixed_versus_depth:
        aux['germ'] = germcircuit
    else:
        aux['germ'] = germcircuits

    return circs, aux


def create_random_germpower_mirror_circuits(pspec, absolute_compilation, depths, qubit_labels=None, localclifford=True,
                                            paulirandomize=True, interacting_qs_density=1 / 8, fixed_versus_depth=False,
                                            seed=None):
    """
    TODO: docstring
    length : consistent with RB length.

    Parameters
    ----------
    pspec : <TODO typ>
        <TODO description>

    absolute_compilation : CompilationRules
        Rules for exactly (absolutely) compiling the "native" gates of `pspec` into clifford gates.

    depths : <TODO typ>
        <TODO description>

    qubit_labels : <TODO typ>, optional
        <TODO description>

    localclifford : <TODO typ>, optional
        <TODO description>

    paulirandomize : <TODO typ>, optional
        <TODO description>

    interacting_qs_density : <TODO typ>, optional
        <TODO description>

    fixed_versus_depth : <TODO typ>, optional
        <TODO description>

    Returns
    -------
    <TODO typ>
    """
    from pygsti.tools import symplectic as _symp

    import numpy as _np
    #assert(length % 2 == 0), "The mirror rb length `length` must be even!"

    rand_state = _np.random.RandomState(seed)  # OK if seed is None

    if qubit_labels is not None:
        assert(isinstance(qubit_labels, list) or isinstance(qubit_labels, tuple)
               ), "If not None, `qubit_labels` must be a list!"
        qubit_labels = list(qubit_labels)
        n = len(qubit_labels)
    else:
        n = pspec.num_qubits

    # Check that the inverse of every gate is in the model:
    _, gate_inverse = pspec.compute_one_qubit_gate_relations()
    gate_inverse.update(pspec.compute_multiqubit_inversion_relations())  # add multiQ inverses
    for gname in pspec.gate_names:
        assert(gname in gate_inverse), \
            "%s gate does not have an inverse in the gate-set! MRB is not possible!" % gname

    circuits, aux = create_random_germpower_circuits(pspec, depths, interacting_qs_density=interacting_qs_density,
                                                     qubit_labels=qubit_labels, fixed_versus_depth=fixed_versus_depth,
                                                     rand_state=rand_state)

    if paulirandomize:
        pauli_circuit = sample_pauli_layer_as_compiled_circuit(pspec, absolute_compilation, qubit_labels=qubit_labels,
                                                               keepidle=True, rand_state=rand_state)

    if localclifford:
        # Sample a compiled 1Q Cliffords layer
        oneQclifford_circuit_out = sample_one_q_clifford_layer_as_compiled_circuit(pspec, absolute_compilation,
                                                                                   qubit_labels=qubit_labels,
                                                                                   rand_state=rand_state)
        # Generate the inverse in the same way as before (note that this will not be the same in some
        # circumstances as finding the inverse Cliffords and using the compilations for those. It doesn't
        # matter much which we do).
        oneQclifford_circuit_back = oneQclifford_circuit_out.copy(editable=True)
        oneQclifford_circuit_back.reverse_inplace()
        oneQclifford_circuit_back.map_names_inplace(gate_inverse)

    circlist = []
    outlist = []

    for circuit in circuits:
        circuit = circuit.copy(editable=True)
        circuit_inv = circuit.copy(editable=True)
        circuit_inv.reverse_inplace()
        circuit_inv.map_names_inplace(gate_inverse)

        if paulirandomize:
            # If .....
            if not fixed_versus_depth:
                pauli_circuit = sample_pauli_layer_as_compiled_circuit(pspec, absolute_compilation,
                                                                       qubit_labels=qubit_labels, keepidle=True,
                                                                       rand_state=rand_state)

            circuit.append_circuit_inplace(pauli_circuit)
            circuit.append_circuit_inplace(circuit_inv)

        # If we start with a random layer of 1-qubit Cliffords, we sample this here.
        if localclifford:
            # If .....
            if not fixed_versus_depth:
                # Sample a compiled 1Q Cliffords layer
                oneQclifford_circuit_out = sample_one_q_clifford_layer_as_compiled_circuit(pspec, absolute_compilation,
                                                                                           qubit_labels=qubit_labels,
                                                                                           rand_state=rand_state)
                # Generate the inverse in the same way as before (note that this will not be the same in some
                # circumstances as finding the inverse Cliffords and using the compilations for those. It doesn't
                # matter much which we do).
                oneQclifford_circuit_back = oneQclifford_circuit_out.copy(editable=True)
                oneQclifford_circuit_back.reverse_inplace()
                oneQclifford_circuit_back.map_names_inplace(gate_inverse)

            # Put one these 1Q clifford circuits at the start and one at then end.
            circuit.append_circuit_inplace(oneQclifford_circuit_out)
            circuit.prefix_circuit_inplace(oneQclifford_circuit_back)

        circuit.done_editing()
        circlist.append(circuit)

        # The full circuit should be, up to a Pauli, the identity.
        s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
        assert(_np.array_equal(s_out, _np.identity(2 * n, _np.int64)))

        # Find the error-free output.
        s_inputstate, p_inputstate = _symp.prep_stabilizer_state(n, zvals=None)
        s_outstate, p_outstate = _symp.apply_clifford_to_stabilizer_state(s_out, p_out, s_inputstate, p_inputstate)
        idealout = []

        for q in range(n):
            measurement_out = _symp.pauli_z_measurement(s_outstate, p_outstate, q)
            bit = measurement_out[1]
            assert(bit == 0 or bit == 1), "Ideal output is not a computational basis state!"
            if not paulirandomize:
                assert(bit == 0), "Ideal output is not the all 0s computational basis state!"
            idealout.append(int(measurement_out[1]))
        idealout = tuple(idealout)

        outlist.append(idealout)

    return circlist, outlist, aux
    
###begin BiRB tools###
def _stabilizer_to_all_zs(stabilizer, qubit_labels, absolute_compilation, seed=None):
    #inputs: 
    #   - stabilizer: An n-qubit Pauli, represented as a string of X, Y, Z, and Is.
    #   - qubit_labels: list/tuple of qubit labels
    #   - absolute_compilation: Absolute Clifford compilation rules for desired gate set
    #returns: A layer of single-qubit gates that transforms the input stabilizer 
    #   into a Pauli with only Zs and Is.
    #   - s, layer, p_layer: Symplectic representation of the layer
    #   - stab_circuit: Layer as a compiled circuit
    rng = _np.random.default_rng(seed)

    n = len(stabilizer)
    
    symp_reps = _symp.compute_internal_gate_symplectic_representations()
    
    s_inv_p, p_inv_p = _symp.inverse_clifford(symp_reps['P'][0],symp_reps['P'][1])
    s_h, p_h = symp_reps['H']
    s_y, p_y = symp_reps['C1']
    
    stab_layer = []
    c = []
    
    for i in range(len(qubit_labels)):
        if stabilizer[i] == 'Y':
            stab_layer.append((s_y, p_y))
            c.append(_lbl.Label('C1','{}'.format(qubit_labels[i])))
        elif stabilizer[i] == 'X':
            stab_layer.append((s_h, p_h))
            c.append(_lbl.Label('C12','{}'.format(qubit_labels[i])))
        elif stabilizer[i] == 'I':
            rand_clifford = str(rng.choice(_np.arange(24)))
            s_rand, p_rand = symp_reps['C'+rand_clifford]
            stab_layer.append((s_rand, p_rand))
            c.append(_lbl.Label('C'+rand_clifford,'{}'.format(qubit_labels[i])))
        else:
            s_rand, p_rand = symp_reps['C0']
            stab_layer.append((s_rand, p_rand))
            c.append(_lbl.Label('C0', '{}'.format(qubit_labels[i])))
            
    s_layer, p_layer = _symp.symplectic_kronecker(stab_layer)
    stab_circuit = _cir.Circuit([c], editable=True)
    stab_circuit.change_gate_library(absolute_compilation)
    if stab_circuit.depth == 0:
        stab_circuit.insert_layer_inplace([_lbl.Label(())], 0)
    stab_circuit.done_editing()
    
    return s_layer, p_layer, stab_circuit

# TO DO: Update DRB to use this new function

def _symplectic_to_pauli(s,p):
    # Takes in the symplectic representation of a Pauli (ie a 2n bitstring in the Hostens notation) and converts it into a list of characters
    # representing the corresponding stabilizer.
    #     - s: Length 2n bitstring.
    #     - p: The "global" phase.
    # Returns: A list of characters ('I','Y','Z','X') representing the stabilizer that corresponds to s.
    
    n = int(len(s)/2)
    pauli = []
    for i in range(n):
        x_pow = s[i]
        z_pow = s[n+i]
        if x_pow != 0 and z_pow != 0: # Have XZ in the i-th slot, ie product is a Y
            #print('need to undo a Y, apply HP^(-1)')
            pauli.append('Y')
        elif x_pow != 0 and z_pow == 0: # Have X in the i-th slot, ie product is an X
            #print('need to undo an X, so apply inverse Hadamard, ie a Hadamard')
            pauli.append('X')
        elif x_pow == 0 and z_pow != 0: # Have Z or I in the i-th slot, so nothing needs to be done
            #print('need to undo a Z or I, ie leave it be')
            pauli.append('Z')
        else:
            pauli.append('I')
            
    return pauli

def _sample_random_pauli(n,pspec = None, absolute_compilation = None, qubit_labels = None, circuit = False, include_identity = False, seed=None):
    # Samples a random Pauli along with a +-1 phase. Returns the Pauli as a list or as a circuit depending 
    # upon the value of "circuit"
    #     - n: Number of qubits
    #     - pspec: Processor spec
    #     - absolute_compilation: compilation rules 
    #     - qubit_labels:
    #     - circuit: Boolean that determines if a list of single-qubit Paulis or a compiled circuit is returned.
    
    if circuit is True:
        if qubit_labels is not None: qubits = qubit_labels[:]  # copy this list
        else: qubits = pspec.qubit_labels[:]
    
    rng = _np.random.default_rng(seed)

    pauli_list = ['I','X','Y','Z']
    
    if include_identity is False:
        while True: 
            rand_ints = rng.integers(0, 4, n)
            if sum(rand_ints) != 0: # make sure we don't get all identities
                break
    else:
        rand_ints = rng.integers(0, 4, n)
            
    pauli = [pauli_list[i] for i in rand_ints]
    if set(pauli) != set('I'): sign = rng.choice([-1,1])
    else: sign = 1
    
    if circuit is False:
        return pauli, sign
    else:
        pauli_layer_std_lbls = [_lbl.Label(pauli_list[rand_ints[q]], (qubits[q],)) for q in range(n)]
        # Converts the layer to a circuit, and changes to the native model.
        pauli_circuit = _cir.Circuit(layer_labels=pauli_layer_std_lbls, line_labels=qubits).parallelize()
        pauli_circuit = pauli_circuit.copy(editable=True)
        pauli_circuit.change_gate_library(absolute_compilation)
        if pauli_circuit.depth == 0:
            pauli_circuit.insert_layer_inplace([_lbl.Label(())], 0)
        pauli_circuit.done_editing()

    return pauli, sign, pauli_circuit

      
def _select_neg_evecs(pauli, sign, seed=None):
    # Selects the entries in an n-qubit that will be turned be given a -1 1Q eigenstates
    #     - pauli: The n-qubit Pauli
    #     - sign: Whether you want a -1 or +1 eigenvector
    # Returns: A bitstring whose 0/1 entries specify if you have a +1 or -1 1Q eigenstate
    rng = _np.random.default_rng(seed)
    
    n = len(pauli)
    identity_bitstring = [0 if i == 'I' else 1 for i in pauli]
    nonzero_indices = _np.nonzero(identity_bitstring)[0]
    num_nid = len(nonzero_indices)
    if num_nid % 2 == 0:
        if sign == 1:
            choices = _np.arange(start = 0, stop = num_nid+1, step = 2)
        else:
            choices = _np.arange(start = 1, stop = num_nid, step = 2)
    else:
        if sign == 1:
            choices = _np.arange(start = 0, stop = num_nid, step = 2)
        else:
            choices = _np.arange(start = 1, stop = num_nid+1, step = 2)
    num_neg_evecs = rng.choice(choices)
    assert((-1)**num_neg_evecs == sign)
    
    neg_evecs = rng.choice(nonzero_indices, num_neg_evecs, replace = False)
    
    bit_evecs = _np.zeros(n)
    bit_evecs[neg_evecs] = 1
    assert('I' not in _np.array(pauli)[nonzero_indices])
    
    return bit_evecs

def _compose_initial_cliffords(prep_circuit):
    #Composes initial random Clifford gates with a second layer of Clifford gates
    #that changes the sign of the target stabilizer
    #   - prep_circuit:  a list of the form [sign_layer, prep_layer], where sign_layer and prep_layer
    #   are lists of gates.
    #Returns a list of gates found by composing the corresponding elements of sign_layer
    #and prep_layer
    composition_rules = {'C0': 'C3',
                         'C2': 'C5',
                         'C12': 'C15'} #supposed to give Gc# * X
    
    sign_layer = prep_circuit[0]
    circ_layer = prep_circuit[1]
    composed_layer = []
    
    for i in range(len(sign_layer)):
        sign_gate = sign_layer[i]
        circ_gate = circ_layer[i]
        new_gate = circ_gate
        if sign_gate == 'C3': # we know that what follows must prep a X, Y, or Z stablizer
            new_gate = composition_rules[circ_gate]
        composed_layer.append(new_gate)
    return composed_layer

def _sample_stabilizer(pauli, sign, absolute_compilation, qubit_labels, seed=None):
    # Samples a random stabilizer of a Pauli, s = s_1 \otimes ... \otimes s_n. For each s_i,
    # we perform the following gates:
    #     - s_i = X: H
    #     - s_i = Y: PH
    #     - s_i = Z: I
    #     - s_i = I: A random 1Q Clifford
    # Also creates the circuit layer that is needed to prepare
    # the stabilizer state. 
    #     - pauli: a list of 1Q paulis whose tensor product gives the n-qubit Pauli
    #     - sign: The overall phase of the pauli. This will determine how many -1 1Q eigenvectors 
    #             our stabilizer state must have.
    # Returns: The symplectic representation of the stabilizer state, symplectic representation of the 
    #          preparation circuit, and a pygsti circuit representation of the prep circuit


    rng = _np.random.default_rng(seed)

    n = len(pauli)
    neg_evecs = _select_neg_evecs(pauli, sign, seed=seed)
    assert((-1)**sum(neg_evecs) == sign)
    zvals = [0 if neg_evecs[i] == 0 else -1 for i in range(n)]
    
    
    init_stab, init_phase = _symp.prep_stabilizer_state(n)
    
    symp_reps = _symp.compute_internal_gate_symplectic_representations()
    
    layer_dict = {'X': symp_reps['H'],
                'Y': tuple(_symp.compose_cliffords(symp_reps['H'][0] 
                                                   ,symp_reps['H'][1]
                                                   ,symp_reps['P'][0]
                                                   ,symp_reps['P'][1])), 
                'Z': symp_reps['I']}
    circ_dict = {'X': 'C12',
                 'Y': 'C2',
                 'Z': 'C0'}
    
    x_layer = [symp_reps['I'] if zvals[i] == 0 else symp_reps['X'] for i in range(len(zvals))]
    circ_layer = [circ_dict[i] if i in circ_dict.keys() else 'C'+str(rng.integers(24)) for i in pauli]

    init_layer = [symp_reps[circ_layer[i]] for i in range(len(pauli))]
    
    x_layer_rep, x_layer_phase = _symp.symplectic_kronecker(x_layer)

    layer_rep, layer_phase = _symp.symplectic_kronecker(init_layer)


    s_prep, p_prep = _symp.compose_cliffords(x_layer_rep,x_layer_phase, layer_rep, layer_phase)
    
    
    stab_state, stab_phase =  _symp.apply_clifford_to_stabilizer_state(s_prep, p_prep,
                                                                      init_stab, init_phase) 
    sign_layer = ['C0' if zvals[i] == 0 else 'C3' for i in range(len(zvals))]
    
    layer_gates = _compose_initial_cliffords([sign_layer, circ_layer])#composition of circ layer and sign layer, with sign layer first

    layer = _cir.Circuit([[_lbl.Label(layer_gates[i], qubit_labels[i]) for i in range(len(qubit_labels))]], line_labels=qubit_labels).parallelize()
    compiled_layer = layer.copy(editable=True)
    compiled_layer.change_gate_library(absolute_compilation)
    if compiled_layer.depth == 0:
        compiled_layer.insert_layer_inplace([_lbl.Label(())], 0)
            #compiled_layer.insert_layer_inplace([_lbl.Label(idle_name, q) for q in qubit_labels], 0)

    return stab_state, stab_phase, s_prep, p_prep, compiled_layer

def _measure(s_state, p_state): 
    #Inputs
    #   - s_state, p_state: The symplectic representation of an n-qubit state
    #Outputs: 
    #   An n-qubit bit string resulting from measuiring the input state in the computational basis
    num_qubits = len(p_state) // 2
    outcome = []
    for i in range(num_qubits):
        p0, p1, ss0, ss1, sp0, sp1 = _symp.pauli_z_measurement(s_state, p_state, i)
        # could cache these results in a FUTURE _stabilizer_measurement_probs function?
        if p0 != 0:
            outcome.append(0)
            s_state, p_state = ss0, sp0 % 4
        else:
            outcome.append(1)
            s_state, p_state = ss1, sp1 % 4
    
    return outcome

def _determine_sign(s_state, p_state, measurement):
    #Inputs:
    #   - s_state, p_state: The symplectic representation of an n-qubit state
    #   - measurement: The target measurement, which is a length-n string of 'Z' and 'I'
    #Outputs: +/-1, the sign of the result of measuring the "measurement" Pauuli on the input state
    an_outcome = _measure(s_state, p_state)
    sign = [-1 if bit == 1 and pauli == 'Z' else 1 for bit, pauli in zip(an_outcome, measurement)]
    
    return _np.prod(sign) 


def create_binary_rb_circuit(pspec, clifford_compilations, length, qubit_labels=None, layer_sampling = 'mixed1q2q', sampler='Qelimination',
                             samplerargs=None, addlocal=False, lsargs=None, seed=None):
    """
    Generates a "binary randomized benchmarking" (BiRB) circuit.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit is being sampled for. The `pspec` is always
        handed to the sampler, as the first argument of the sampler function.

    clifford_compilations : CompilationRules
        Rules for exactly (absolutely) compiling the "native" gates of `pspec` into Clifford gates.

    length : int
        The "benchmark depth" of the circuit, which is the number of randomly sampled layers of gates in 
        the core circuit. The full BiRB circuit has depth=length+2. 

    qubit_labels : list, optional
        If not None, a list of the qubits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
        irrelevant.

    layer_sampling : str, optional
        Determines the structure of the randomly sampled layers of gates:
            1. 'mixed1q2q': Layers contain radomly-sampled two-qubit gates and randomly-sampled 
            single-qubit gates on all remaining qubits. 
            2. 'alternating1q2q': Each layer consists of radomly-sampled two-qubit gates, with 
            all other qubits idling, followed by randomly sampled single-qubit gates on all qubits. 

    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
        a valid option for n-qubit BiRB, but it is not explicitly forbidden by this function]. 
        If `sampler` is a function, it should be a function that takes
        as the first argument a QubitProcessorSpec, and returns a random circuit layer as a list of gate
        Label objects. Note that the default 'Qelimination' is not necessarily the most useful
        in-built sampler, but it is the only sampler that requires no parameters beyond the QubitProcessorSpec
        *and* works for arbitrary connectivity devices. See the docstrings for each of these samplers
        for more information.

    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec` and `samplerargs` lists the
        remaining arguments handed to the sampler.

    seed : int, optional
        A seed to initialize the random number generator used for creating random clifford
        circuits.

     addlocal : bool, optional
         Whether to follow each layer in the "core" circuits, sampled according to `sampler` with
         a layer of 1-qubit gates.

    lsargs : list, optional
        Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
        layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.


    Returns
    -------
    Circuit
        A random BiRB circuit.
    String
        A length-n string of 'Z's and 'I's, which describes the target Pauli measurement for the BiRB circuit.
        The circuit, when run without errors, produces an eigenstate of the target Pauli operator.  
    Int (Either 1 or -1)
        Specifies the sign of the target Pauli measurement.
    """
    if lsargs is None:
        lsargs = []
    if samplerargs is None:
        samplerargs = []
    if qubit_labels is not None: n = len(qubit_labels)
    else: 
        n = pspec.num_qubits
        qubit_labels = pspec.qubit_labels

    rand_state = _np.random.RandomState(seed)  # Ok if seed is None
    

    rand_pauli, rand_sign, pauli_circuit = _sample_random_pauli(n = n, pspec = pspec, qubit_labels=qubit_labels,
                                                                absolute_compilation = clifford_compilations,
                                                                circuit = True, include_identity = False, seed=seed+42)

    s_inputstate, p_inputstate, s_init_layer, p_init_layer, prep_circuit = _sample_stabilizer(rand_pauli, rand_sign, clifford_compilations, 
                                                                                              qubit_labels, seed=seed+43)
    
    s_pc, p_pc = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec=pspec.subset(gate_names_to_include='all', 
                                                                                            qubit_labels_to_keep=qubit_labels)) #note: if the pspec contains gates not in pyGSTi, this
    
    # build the initial layer of the circuit
    full_circuit = prep_circuit.copy(editable=True)

    # Sample a random circuit of "native gates".
    if layer_sampling == 'alternating1q2q':
        circuit = random_alternating_clifford_circ(pspec, length, qubit_labels=qubit_labels, two_q_gate_density=samplerargs[0], rand_state=rand_state) 
    elif layer_sampling == 'mixed1q2q':
        circuit = create_random_circuit(pspec=pspec, length=length, qubit_labels=qubit_labels, sampler=sampler,
                                    samplerargs=samplerargs, addlocal=addlocal, lsargs=lsargs, rand_state=rand_state)
    else:
        raise ValueError(f'{layer_sampling} is not a known layer type')
        
    # find the symplectic matrix / phase vector this "native gates" circuit implements.
    s_rc, p_rc = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec.subset(gate_names_to_include='all', qubit_labels_to_keep=qubit_labels))
    
    s_composite, p_composite = _symp.compose_cliffords(s1 = s_init_layer, p1 = p_init_layer, s2 = s_rc, p2 = p_rc)

    # Apply the random circuit to the initial state (either the all 0s or a random stabilizer state)
    
    full_circuit.append_circuit_inplace(circuit)
    
    s_outputstate, p_outputstate = _symp.apply_clifford_to_stabilizer_state(s_rc, p_rc, 
                                                                            s_inputstate, p_inputstate)
    

    # Figure out what stabilizer of s_outputstate, rand_pauli was mapped too
    s_rc_inv, p_rc_inv = _symp.inverse_clifford(s_rc, p_rc) # U^(-1)
    s_new_pauli, p_new_pauli = _symp.compose_cliffords(s_rc_inv, p_rc_inv, s_pc, p_pc) # PU^(-1)
    s_new_pauli, p_new_pauli = _symp.compose_cliffords(s_new_pauli, p_new_pauli, s_rc, p_rc) # UPaU^(-1)
        
    pauli_vector = p_new_pauli
    pauli = [i[0] for i in _symp.find_pauli_layer(pauli_vector, [j for j in range(n)])]
    measurement, phase = ['I' if i == 'I' else 'Z' for i in pauli], None #not needed
        
    # Turn the stabilizer into an all Z and I stabilizer. Append this to the circuit.
    
    s_stab, p_stab, stab_circuit = _stabilizer_to_all_zs(pauli, qubit_labels, clifford_compilations, seed=seed+404)
    
    full_circuit.append_circuit_inplace(stab_circuit)
    
    s_inv, p_inv = _symp.inverse_clifford(s_stab, p_stab)
    s_cc, p_cc = _symp.compose_cliffords(s_inv, p_inv, s_composite, p_composite)
    s_cc, p_cc = _symp.compose_cliffords(s_composite, p_composite, s_stab, p_stab) # MUPaU^(-1)M^(-1)
    
    meas = [i[0] for i in _symp.find_pauli_layer(p_cc, [j for j in range(n)])] # not needed
     
    s_outputstate, p_outputstate = _symp.apply_clifford_to_stabilizer_state(s_stab, p_stab, s_outputstate, p_outputstate)

    full_circuit.done_editing()
    sign = _determine_sign(s_outputstate, p_outputstate, measurement)
    
    outcircuit = full_circuit
        
    return outcircuit, measurement, sign

def random_alternating_clifford_circ(pspec, depth, qubit_labels=None, two_q_gate_density=0.25, rand_state=None):
    """
    Generates a random circuit with composite layers cponsisting of a layer of two-qubit gates followed by 
    a layer of of single-qubit gates.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The QubitProcessorSpec for the device that the circuit is being sampled for. The `pspec` is always
        handed to the sampler, as the first argument of the sampler function.

    depth : int
        The number of composite layers in the final circuit.

    qubit_labels : list, optional
        If not None, a list of the qubFalseits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
        irrelevant.


    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec` and `samplerargs` lists the
        remaining arguments handed to the sampler.


    Returns
    -------
    Circuit
        A random circuit with 2*depth layers
    """ 
    if qubit_labels == None:
        qubit_labels = pspec.qubit_labels
    #sample # benchmarking layers = depth
    circ = _cir.Circuit(layer_labels=[], line_labels=qubit_labels, editable=True)
    for _ in range(depth):
        oneQ_layer = _cir.Circuit([sample_circuit_layer_of_one_q_gates(pspec, qubit_labels = qubit_labels, rand_state=rand_state)]).parallelize()
        twoQ_layer = _cir.Circuit(sample_circuit_layer_by_edgegrab(pspec, qubit_labels=qubit_labels, two_q_gate_density=two_q_gate_density, 
                                                                   rand_state=rand_state)).parallelize()
        circ.append_circuit_inplace(twoQ_layer)
        circ.append_circuit_inplace(oneQ_layer)
    circ.done_editing()

    return circ

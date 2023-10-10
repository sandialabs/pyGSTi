"""
Random circuit sampling functions.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
import itertools as _itertools

import numpy as _np

import pygsti.models as _models 


from pygsti.algorithms import compilers as _cmpl
from pygsti.circuits import circuit as _cir
from pygsti.baseobjs import label as _lbl
from pygsti.tools import group as _rbobjs
from pygsti.tools import symplectic as _symp
from pygsti.tools import compilationtools as _comp
from pygsti.tools import internalgates as _gates

try: import qsearch as _qsearch
except: _qsearch = None
from scipy.stats import unitary_group

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
                                   two_q_gate_args_lists={'Gczr': [(str(_np.pi / 2),), (str(-_np.pi / 2),)]}):
    '''
    TODO: docstring
    Generates a forward circuits with benchmark depth d for non-clifford mirror randomized benchmarking.
    The circuits alternate Haar-random 1q unitaries and layers of Gczr gates
    '''
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


# TJP: I am not aware this is ever used anymore, and it's functionality is possibly even included in the
# other samplers.
# def sample_circuit_layer_by_pairing_qubits(pspec, qubit_labels=None, two_q_prob=0.5, one_q_gate_names='all',
#                                            two_q_gate_names='all', modelname='clifford', rand_state=None):
#     """
#     Creates a circuit by randomply placing 2-qubit gates on qubit pairs.

#     Samples a random circuit layer by pairing up qubits and picking a two-qubit gate for a pair
#     with the specificed probability. This sampler *assumes* all-to-all connectivity, and does
#     not check that this condition is satisfied (more generally, it assumes that all gates can be
#     applied in parallel in any combination that would be well-defined).

#     The sampler works as follows: If there are an odd number of qubits, one qubit is chosen at
#     random to have a uniformly random 1-qubit gate applied to it (from all possible 1-qubit gates,
#     or those in `one_q_gate_names` if not None). Then, the remaining qubits are paired up, uniformly
#     at random. A uniformly random 2-qubit gate is then chosen for a pair with probability `two_q_prob`
#     (from all possible 2-qubit gates, or those in `two_q_gate_names` if not None). If a 2-qubit gate
#     is not chosen to act on a pair, then each qubit is independently and uniformly randomly assigned
#     a 1-qubit gate.

#     Parameters
#     ----------
#     pspec : QubitProcessorSpec
#         The QubitProcessorSpec for the device that the circuit layer is being sampled for. This
#         function assumes all-to-all connectivity, but does not check this is satisfied. Unless
#         `qubit_labels` is not None, a circuit layer is sampled over all the qubits in `pspec`.

#     qubit_labels : list, optional
#         If not None, a list of the qubits to sample the circuit layer for. This is a subset of
#         `pspec.qubit_labels`. If None, the circuit layer is sampled to acton all the qubits
#         in `pspec`.

#     two_q_prob : float, optional
#         A probability for a two-qubit gate to be applied to a pair of qubits. So, the expected
#         number of 2-qubit gates in the sampled layer is two_q_prob*floor(n/2).

#     one_q_gate_names : 'all' or list, optional
#         If not 'all', a list of the names of the 1-qubit gates to be sampled from when applying
#         a 1-qubit gate to a qubit. If this is 'all', the full set of 1-qubit gate names is extracted
#         from the QubitProcessorSpec.

#     two_q_gate_names : 'all' or list, optional
#         If not 'all', a list of the names of the 2-qubit gates to be sampled from when applying
#         a 2-qubit gate to a pair of qubits. If this is 'all', the full set of 2-qubit gate names is
#         extracted from the QubitProcessorSpec.

#     modelname : str, optional
#         Only used if one_q_gate_names or two_q_gate_names is None. Specifies which of the
#         `pspec.models` to use to extract the gate-set. The `clifford` default is suitable
#         for Clifford or direct RB, but will not use any non-Clifford gates in the gate-set.

#     rand_state: RandomState, optional
#         A np.random.RandomState object for seeding RNG

#     Returns
#     -------
#     list of Labels
#         A list of gate Labels that defines a "complete" circuit layer (there is one and only
#         one gate acting on each qubit in `pspec` or `qubit_labels`).
#     """
#     if rand_state is None:
#         rand_state = _np.random.RandomState()

#     if qubit_labels is None: n = pspec.num_qubits
#     else:
#         assert(isinstance(qubit_labels, list) or isinstance(qubit_labels, tuple)), \
#             "SubsetQs must be a list or a tuple!"
#         n = len(qubit_labels)

#     # If the one qubit and/or two qubit gate names are only specified as 'all', construct them.
#     if (one_q_gate_names == 'all') or (two_q_gate_names == 'all'):
#         if one_q_gate_names == 'all':
#             oneQpopulate = True
#             one_q_gate_names = []
#         else:
#             oneQpopulate = False
#         if two_q_gate_names == 'all':
#             twoQpopulate = True
#             two_q_gate_names = []
#         else:
#             twoQpopulate = False

#         operationlist = pspec.models[modelname].primitive_op_labels
#         for gate in operationlist:
#             if oneQpopulate:
#                 if (gate.num_qubits == 1) and (gate.name not in one_q_gate_names):
#                     one_q_gate_names.append(gate.name)
#             if twoQpopulate:
#                 if (gate.num_qubits == 2) and (gate.name not in two_q_gate_names):
#                     two_q_gate_names.append(gate.name)

#     # Basic variables required for sampling the circuit layer.
#     if qubit_labels is None:
#         qubits = list(pspec.qubit_labels[:])  # copy this list
#     else:
#         qubits = list(qubit_labels[:])  # copy this list
#     sampled_layer = []
#     num_oneQgatenames = len(one_q_gate_names)
#     num_twoQgatenames = len(two_q_gate_names)

#     # If there is an odd number of qubits, begin by picking one to have a 1-qubit gate.
#     if n % 2 != 0:
#         q = qubits[rand_state.randint(0, n)]
#         name = one_q_gate_names[rand_state.randint(0, num_oneQgatenames)]
#         del qubits[q]  # XXX is this correct?
#         sampled_layer.append(_lbl.Label(name, q))

#     # Go through n//2 times until all qubits have been paired up and gates on them sampled
#     for i in range(n // 2):

#         # Pick two of the remaining qubits : each qubit that is picked is deleted from the list.
#         index = rand_state.randint(0, len(qubits))
#         q1 = qubits[index]
#         del qubits[index]
#         index = rand_state.randint(0, len(qubits))
#         q2 = qubits[index]
#         del qubits[index]

#         # Flip a coin to decide whether to act a two-qubit gate on that qubit
#         if rand_state.binomial(1, two_q_prob) == 1:
#             # If there is more than one two-qubit gate on the pair, pick a uniformly random one.
#             name = two_q_gate_names[rand_state.randint(0, num_twoQgatenames)]
#             sampled_layer.append(_lbl.Label(name, (q1, q2)))
#         else:
#             # Independently, pick uniformly random 1-qubit gates to apply to each qubit.
#             name1 = one_q_gate_names[rand_state.randint(0, num_oneQgatenames)]
#             name2 = one_q_gate_names[rand_state.randint(0, num_oneQgatenames)]
#             sampled_layer.append(_lbl.Label(name1, q1))
#             sampled_layer.append(_lbl.Label(name2, q2))

#     return sampled_layer


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
                print(one_q_gate_names)
                print(ops_on_qubits[(q,)])
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


def create_random_circuit(pspec, length, qubit_labels=None, sampler='Qelimination', samplerargs=[],
                          addlocal=False, lsargs=[], rand_state=None):
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
                layer = sampler(pspec, qubit_labels, *samplerargs)
            circuit.insert_layer_inplace(layer, 0)

    circuit.done_editing()
    return circuit

#### Commented out as this code has not been tested since a much older version of pyGSTi and it is probably
#### not being used.
# def sample_simultaneous_random_circuit(pspec, length, structure='1Q', sampler='Qelimination', samplerargs=[],
#                                        addlocal=False, lsargs=[]):
#     """
#     Generates a random circuit of the specified length.

#     Parameters
#     ----------
#     pspec : QubitProcessorSpec
#         The QubitProcessorSpec for the device that the circuit is being sampled for, which defines the
#         "native" gate-set and the connectivity of the device. The returned circuit will be over
#         the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
#         is always handed to the sampler, as the first argument of the sampler function (this is only
#         of importance when not using an in-built sampler).

#     length : int
#         The length of the circuit. Todo: update for varying length in different subsets.

#     structure : str or tuple, optional
#         todo.

#     sampler : str or function, optional
#         If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
#         Except for 'local', this corresponds to sampling layers according to the sampling function
#         in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
#         corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates.
#         If `sampler` is a function, it should be a function that takes as the first argument a
#         QubitProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
#         the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is
#         the only sampler that requires no parameters beyond the QubitProcessorSpec *and* works for arbitrary
#         connectivity devices. See the docstrings for each of these samplers for more information.

#     samplerargs : list, optional
#         A list of arguments that are handed to the sampler function, specified by `sampler`.
#         The first argument handed to the sampler is `pspec`, the second argument is `qubit_labels`,
#         and `samplerargs` lists the remaining arguments handed to the sampler. This is not
#         optional for some choices of `sampler`.

#     addlocal : bool, optional
#         Whether to follow each layer in the circuit, sampled according to `sampler` with
#         a layer of 1-qubit gates. If this is True then the length of the circuit is double
#         the requested length.

#     lsargs : list, optional
#         Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
#         layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.

#     Returns
#     -------
#     Circuit
#         A random circuit sampled as specified.
#     Tuple
#         A length-n tuple of floats in [0,1], corresponding to the error-free *marginalized* probabilities
#         for the "1" outcome of a computational basis measurement at the end of this circuit, with the standard
#         input state (with the outcomes ordered to be the same as the wires in the circuit).
#     """
#     if isinstance(structure, str):
#         assert(structure == '1Q'), "The only default `structure` option is the string '1Q'"
#         structure = tuple([(q,) for q in pspec.qubit_labels])
#         n = pspec.num_qubits
#     else:
#         assert(isinstance(structure, list) or isinstance(structure, tuple)
#                ), "If not a string, `structure` must be a list or tuple."
#         qubits_used = []
#         for qubit_labels in structure:
#             assert(isinstance(qubit_labels, list) or isinstance(
#                 qubit_labels, tuple)), "SubsetQs must be a list or a tuple!"
#             qubits_used = qubits_used + list(qubit_labels)
#             assert(len(set(qubits_used)) == len(qubits_used)
#                    ), "The qubits in the tuples/lists of `structure must all be unique!"

#         assert(set(qubits_used).issubset(set(pspec.qubit_labels))
#                ), "The qubits to benchmark must all be in the QubitProcessorSpec `pspec`!"
#         n = len(qubits_used)

#     # Creates a empty circuit over no wires
#     circuit = _cir.Circuit(num_lines=0, editable=True)

#     s_rc_dict = {}
#     p_rc_dict = {}
#     circuit_dict = {}

#     if isinstance(length, _np.int64):
#         length_per_subset = [length for i in range(len(structure))]
#     else:
#         length_per_subset = length
#         assert(len(length) == len(structure)), "If `length` is a list it must be the same length as `structure`"

#     for ssQs_ind, qubit_labels in enumerate(structure):
#         qubit_labels = tuple(qubit_labels)
#         # Sample a random circuit of "native gates" over this set of qubits, with the
#         # specified sampling.
#         subset_circuit = create_random_circuit(pspec=pspec, length=length_per_subset[ssQs_ind],
#                                                qubit_labels=qubit_labels, sampler=sampler, samplerargs=samplerargs,
#                                                addlocal=addlocal, lsargs=lsargs)
#         circuit_dict[qubit_labels] = subset_circuit
#         # find the symplectic matrix / phase vector this circuit implements.
#         s_rc_dict[qubit_labels], p_rc_dict[qubit_labels] = _symp.symplectic_rep_of_clifford_circuit(
#             subset_circuit, pspec=pspec)
#         # Tensors this circuit with the current circuit
#         circuit.tensor_circuit_inplace(subset_circuit)

#     circuit.done_editing()

#     # Find the expected outcome of the circuit.
#     s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
#     s_inputstate, p_inputstate = _symp.prep_stabilizer_state(n, zvals=None)
#     s_outstate, p_outstate = _symp.apply_clifford_to_stabilizer_state(s_out, p_out, s_inputstate, p_inputstate)
#     idealout = []
#     for qubit_labels in structure:
#         subset_idealout = []
#         for q in qubit_labels:
#             qind = circuit.line_labels.index(q)
#             measurement_out = _symp.pauli_z_measurement(s_outstate, p_outstate, qind)
#             subset_idealout.append(measurement_out[1])
#         idealout.append(tuple(subset_idealout))
#     idealout = tuple(idealout)

#     return circuit, idealout


# def _get_setting(l, circuitindex, substructure, depths, circuits_per_length, structure):

#     lind = depths.index(l)
#     settingDict = {}

#     for s in structure:
#         if s in substructure:
#             settingDict[s] = len(depths) + lind * circuits_per_length + circuitindex
#         else:
#             settingDict[s] = lind

#     return settingDict


# def create_simultaneous_random_circuits_experiment(pspec, depths, circuits_per_length, structure='1Q',
#                                                    sampler='Qelimination', samplerargs=[], addlocal=False,
#                                                    lsargs=[], set_isolated=True, setcomplement_isolated=False,
#                                                    descriptor='A set of simultaneous random circuits', verbosity=1):
#     """
#     Generates a set of simultaneous random circuits of the specified depths.

#     Parameters
#     ----------
#     pspec : QubitProcessorSpec
#         The QubitProcessorSpec for the device that the circuit is being sampled for, which defines the
#         "native" gate-set and the connectivity of the device. The returned circuit will be over
#         the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
#         is always handed to the sampler, as the first argument of the sampler function (this is only
#         of importance when not using an in-built sampler).

#     depths : int
#         Todo : update (needs to include list option)
#         The set of depths for the circuits.

#     circuits_per_length : int
#         The number of (possibly) different circuits sampled at each length.

#     structure : str or tuple.
#         Defines the "structure" of the simultaneous circuit. TODO : more details.

#     sampler : str or function, optional
#         If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
#         Except for 'local', this corresponds to sampling layers according to the sampling function
#         in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
#         corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates.
#         If `sampler` is a function, it should be a function that takes as the first argument a
#         QubitProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
#         the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is the
#         only sampler that requires no parameters beyond the QubitProcessorSpec *and* works for arbitrary
#         connectivity devices. See the docstrings for each of these samplers for more information.

#     samplerargs : list, optional
#         A list of arguments that are handed to the sampler function, specified by `sampler`.
#         The first argument handed to the sampler is `pspec`, the second argument is `qubit_labels`,
#         and `samplerargs` lists the remaining arguments handed to the sampler. This is not
#         optional for some choices of `sampler`.

#     addlocal : bool, optional
#         Whether to follow each layer in the "core" circuits, sampled according to `sampler` with
#         a layer of 1-qubit gates.

#     lsargs : list, optional
#         Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
#         layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.

#     set_isolated : bool, optional
#         Todo

#     setcomplement_isolated : bool, optional
#         Todo

#     descriptor : str, optional
#         A description of the experiment being generated. Stored in the output dictionary.

#     verbosity : int, optional
#         If > 0 the number of circuits generated so far is shown.

#     Returns
#     -------
#     dict
#         A dictionary containing the generated random circuits, the error-free outputs of the circuit,
#         and the specification used to generate the circuits. The keys are:

#         - 'circuits'. A dictionary of the sampled circuits. The circuit with key(l,k) is the kth circuit
#         at length l.

#         - 'probs'. A dictionary of the error-free *marginalized* probabilities for the "1" outcome of
#         a computational basis measurement at the end of each circuit, with the standard input state.
#         The ith element of this tuple corresponds to this probability for the qubit on the ith wire of
#         the output circuit.

#         - 'qubitordering'. The ordering of the qubits in the 'target' tuples.

#         - 'spec'. A dictionary containing all of the parameters handed to this function, except `pspec`.
#         This then specifies how the circuits where generated.
#     """
#     experiment_dict = {}
#     experiment_dict['spec'] = {}
#     experiment_dict['spec']['depths'] = depths
#     experiment_dict['spec']['circuits_per_length'] = circuits_per_length
#     experiment_dict['spec']['sampler'] = sampler
#     experiment_dict['spec']['samplerargs'] = samplerargs
#     experiment_dict['spec']['addlocal'] = addlocal
#     experiment_dict['spec']['lsargs'] = lsargs
#     experiment_dict['spec']['descriptor'] = descriptor
#     experiment_dict['spec']['createdby'] = 'extras.rb.sample.simultaneous_random_circuits_experiment'

#     if isinstance(structure, str):
#         assert(structure == '1Q'), "The only default `structure` option is the string '1Q'"
#         structure = tuple([(q,) for q in pspec.qubit_labels])
#     else:
#         assert(isinstance(structure, list) or isinstance(structure, tuple)), \
#             "If not a string, `structure` must be a list or tuple."
#         qubits_used = []
#         for qubit_labels in structure:
#             assert(isinstance(qubit_labels, list) or isinstance(
#                 qubit_labels, tuple)), "SubsetQs must be a list or a tuple!"
#             qubits_used = qubits_used + list(qubit_labels)
#             assert(len(set(qubits_used)) == len(qubits_used)), \
#                 "The qubits in the tuples/lists of `structure must all be unique!"

#         assert(set(qubits_used).issubset(set(pspec.qubit_labels))), \
#             "The qubits to benchmark must all be in the QubitProcessorSpec `pspec`!"

#     experiment_dict['spec']['structure'] = structure
#     experiment_dict['circuits'] = {}
#     experiment_dict['probs'] = {}
#     experiment_dict['settings'] = {}

#     for lnum, l in enumerate(depths):
#         if verbosity > 0:
#             print('- Sampling {} circuits at length {} ({} of {} depths)'.format(circuits_per_length, l,
#                                                                                  lnum + 1, len(depths)))
#             print('  - Number of circuits sampled = ', end='')
#         for j in range(circuits_per_length):
#             circuit, idealout = sample_simultaneous_random_circuit(pspec, l, structure=structure, sampler=sampler,
#                                                                    samplerargs=samplerargs, addlocal=addlocal,
#                                                                    lsargs=lsargs)

#             if (not set_isolated) and (not setcomplement_isolated):
#                 experiment_dict['circuits'][l, j] = circuit
#                 experiment_dict['probs'][l, j] = idealout
#                 experiment_dict['settings'][l, j] = {
#                     s: len(depths) + lnum * circuits_per_length + j for s in tuple(structure)}
#             else:
#                 experiment_dict['circuits'][l, j] = {}
#                 experiment_dict['probs'][l, j] = {}
#                 experiment_dict['settings'][l, j] = {}
#                 experiment_dict['circuits'][l, j][tuple(structure)] = circuit
#                 experiment_dict['probs'][l, j][tuple(structure)] = idealout
#                 experiment_dict['settings'][l, j][tuple(structure)] = _get_setting(l, j, structure, depths,
#                                                                                    circuits_per_length, structure)
#             if set_isolated:
#                 for subset_ind, subset in enumerate(structure):
#                     subset_circuit = circuit.copy(editable=True)
#                     #print(subset)
#                     for q in circuit.line_labels:
#                         if q not in subset:
#                             #print(subset_circuit, q)
#                             subset_circuit.replace_with_idling_line_inplace(q)
#                     subset_circuit.done_editing()
#                     experiment_dict['circuits'][l, j][(tuple(subset),)] = subset_circuit
#                     experiment_dict['probs'][l, j][(tuple(subset),)] = idealout[subset_ind]
#                     # setting = {}
#                     # for s in structure:
#                     #     if s in subset:
#                     #         setting[s] =  len(depths) + lnum*circuits_per_length + j
#                     #     else:
#                     #         setting[s] =  lnum
#                     experiment_dict['settings'][l, j][(tuple(subset),)] = _get_setting(l, j, (tuple(subset),), depths,
#                                                                                        circuits_per_length, structure)
#                     # print(subset)
#                     # print(_get_setting(l, j, subset, depths, circuits_per_length, structure))

#             if setcomplement_isolated:
#                 for subset_ind, subset in enumerate(structure):
#                     subsetcomplement_circuit = circuit.copy(editable=True)
#                     for q in circuit.line_labels:
#                         if q in subset:
#                             subsetcomplement_circuit.replace_with_idling_line_inplace(q)
#                     subsetcomplement_circuit.done_editing()
#                     subsetcomplement = list(_copy.copy(structure))
#                     subsetcomplement_idealout = list(_copy.copy(idealout))
#                     del subsetcomplement[subset_ind]
#                     del subsetcomplement_idealout[subset_ind]
#                     subsetcomplement = tuple(subsetcomplement)
#                     subsetcomplement_idealout = tuple(subsetcomplement_idealout)
#                     experiment_dict['circuits'][l, j][subsetcomplement] = subsetcomplement_circuit
#                     experiment_dict['probs'][l, j][subsetcomplement] = subsetcomplement_idealout

#                     # for s in structure:
#                     #     if s in subsetcomplement:
#                     #         setting[s] =  len(depths) + lnum*circuits_per_length + j
#                     #     else:
#                     #         setting[s] =  lnum
#                     experiment_dict['settings'][l, j][subsetcomplement] = _get_setting(l, j, subsetcomplement, depths,
#                                                                                        circuits_per_length, structure)

#             if verbosity > 0: print(j + 1, end=',')
#         if verbosity > 0: print('')

#     return experiment_dict


# def create_exhaustive_independent_random_circuits_experiment(pspec, allowed_depths, circuits_per_subset,
#                                                              structure='1Q',
#                                                              sampler='Qelimination', samplerargs=[], descriptor='',
#                                                              verbosity=1, seed=None):
#     """
#     Todo

#     Parameters
#     ----------
#     pspec : QubitProcessorSpec
#         The QubitProcessorSpec for the device that the circuit is being sampled for, which defines the
#         "native" gate-set and the connectivity of the device. The returned circuit will be over
#         the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
#         is always handed to the sampler, as the first argument of the sampler function (this is only
#         of importance when not using an in-built sampler).

#     allowed_depths : <TODO typ>
#         <TODO description>

#     circuits_per_subset : <TODO typ>
#         <TODO description>

#     structure : str or tuple.
#         Defines the "structure" of the simultaneous circuit. TODO : more details.

#     sampler : str or function, optional
#         If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
#         Except for 'local', this corresponds to sampling layers according to the sampling function
#         in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
#         corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates.
#         If `sampler` is a function, it should be a function that takes as the first argument a
#         QubitProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
#         the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is the
#         only sampler that requires no parameters beyond the QubitProcessorSpec *and* works for arbitrary
#         connectivity devices. See the docstrings for each of these samplers for more information.

#     samplerargs : list, optional
#         A list of arguments that are handed to the sampler function, specified by `sampler`.
#         The first argument handed to the sampler is `pspec`, the second argument is `qubit_labels`,
#         and `samplerargs` lists the remaining arguments handed to the sampler. This is not
#         optional for some choices of `sampler`.

#     descriptor : str, optional
#         A description of the experiment being generated. Stored in the output dictionary.

#     verbosity : int, optional
#         How much output to sent to stdout.

#     seed : int, optional
#         Seed for RNG

#     Returns
#     -------
#     dict
#     """
#     experiment_dict = {}
#     experiment_dict['spec'] = {}
#     experiment_dict['spec']['allowed_depths'] = allowed_depths
#     experiment_dict['spec']['circuits_per_subset'] = circuits_per_subset
#     experiment_dict['spec']['sampler'] = sampler
#     experiment_dict['spec']['samplerargs'] = samplerargs
#     experiment_dict['spec']['descriptor'] = descriptor

#     if isinstance(structure, str):
#         assert(structure == '1Q'), "The only default `structure` option is the string '1Q'"
#         structure = tuple([(q,) for q in pspec.qubit_labels])
#     else:
#         assert(isinstance(structure, list) or isinstance(structure, tuple)), \
#             "If not a string, `structure` must be a list or tuple."
#         qubits_used = []
#         for qubit_labels in structure:
#             assert(isinstance(qubit_labels, list) or isinstance(
#                 qubit_labels, tuple)), "SubsetQs must be a list or a tuple!"
#             qubits_used = qubits_used + list(qubit_labels)
#             assert(len(set(qubits_used)) == len(qubits_used)), \
#                 "The qubits in the tuples/lists of `structure must all be unique!"

#         assert(set(qubits_used).issubset(set(pspec.qubit_labels))), \
#             "The qubits to benchmark must all be in the QubitProcessorSpec `pspec`!"

#     rand_state = _np.random.RandomState(seed)  # OK if seed is None

#     experiment_dict['spec']['structure'] = structure
#     experiment_dict['circuits'] = {}
#     experiment_dict['probs'] = {}

#     if circuits_per_subset**len(structure) >> 10000:
#         print("Warning: {} circuits are going to be generated by this function!".format(
#             circuits_per_subset**len(structure)))

#     circuits = {}

#     for ssQs_ind, qubit_labels in enumerate(structure):
#         circuits[qubit_labels] = []
#         for i in range(circuits_per_subset):
#             l = allowed_depths[rand_state.randint(len(allowed_depths))]
#             circuits[qubit_labels].append(create_random_circuit(pspec, l, qubit_labels=qubit_labels,
#                                                                 sampler=sampler, samplerargs=samplerargs))

#     experiment_dict['subset_circuits'] = circuits

#     parallel_circuits = {}
#     it = [range(circuits_per_subset) for i in range(len(structure))]
#     for setting_comb in _itertools.product(*it):
#         pcircuit = _cir.Circuit(num_lines=0, editable=True)
#         for ssQs_ind, qubit_labels in enumerate(structure):
#             pcircuit.tensor_circuit_inplace(circuits[qubit_labels][setting_comb[ssQs_ind]])
#             pcircuit.done_editing()  # TIM: is this indented properly?
#             parallel_circuits[setting_comb] = pcircuit

#     experiment_dict['circuits'] = parallel_circuits

#     return experiment_dict


def create_direct_rb_circuit(pspec, clifford_compilations, length, qubit_labels=None, sampler='Qelimination',
                             samplerargs=[], addlocal=False, lsargs=[], randomizeout=True, cliffordtwirl=True,
                             conditionaltwirl=True, citerations=20, compilerargs=[], partitioned=False, seed=None):
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

#### Commented out as all of this functionality should be reproducable using simulataneous experiment designs applied
#### to DirectRB experiment designs.
# def sample_simultaneous_direct_rb_circuit(pspec, clifford_compilations, length, structure='1Q',
#                                           sampler='Qelimination',
#                                           samplerargs=[], addlocal=False, lsargs=[], randomizeout=True,
#                                           cliffordtwirl=True, conditionaltwirl=True, citerations=20, compilerargs=[],
#                                           partitioned=False, seed=1234):
#     """
#     Generates a simultaneous "direct randomized benchmarking" (DRB) circuit.

#     DRB is the protocol introduced in arXiv:1807.07975 (2018). An n-qubit DRB circuit consists of
#     (1) a circuit the prepares a uniformly random stabilizer state; (2) a length-l circuit
#     (specified by `length`) consisting of circuit layers sampled according to some user-specified
#     distribution (specified by `sampler`), (3) a circuit that maps the output of the preceeding
#     circuit to a computational basis state. See arXiv:1807.07975 (2018) for further details. Todo :
#     what SDRB is.

#     Parameters
#     ----------
#     pspec : QubitProcessorSpec
#         The QubitProcessorSpec for the device that the circuit is being sampled for, which defines the
#         "native" gate-set and the connectivity of the device. The returned DRB circuit will be over
#         the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
#         is always handed to the sampler, as the first argument of the sampler function (this is only
#         of importance when not using an in-built sampler for the "core" of the DRB circuit). Unless
#         `qubit_labels` is not None, the circuit is sampled over all the qubits in `pspec`.

#     clifford_compilations : dict
#         A dictionary with the potential keys `'absolute'` and `'paulieq'` and corresponding
#         :class:`CompilationRules` values.  These compilation rules specify how to compile the
#         "native" gates of `pspec` into Clifford gates.

#     length : int
#         The "direct RB length" of the circuit, which is closely related to the circuit depth. It
#         must be an integer >= 0. Unless `addlocal` is True, it is the depth of the "core" random
#         circuit, sampled according to `sampler`, specified in step (2) above. If `addlocal` is True,
#         each layer in the "core" circuit sampled according to "sampler` is followed by a layer of
#         1-qubit gates, with sampling specified by `lsargs` (and the first layer is proceeded by a
#         layer of 1-qubit gates), and so the circuit of step (2) is length 2*`length` + 1.

#     structure : str or tuple, optional
#         todo.

#     sampler : str or function, optional
#         If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
#         Except for 'local', this corresponds to sampling layers according to the sampling function
#         in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
#         corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
#         a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function].
#         If `sampler` is a function, it should be a function that takes as the first argument a
#         QubitProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
#         the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is
#         the only sampler that requires no parameters beyond the QubitProcessorSpec *and* works for arbitrary
#         connectivity devices. See the docstrings for each of these samplers for more information.

#     samplerargs : list, optional
#         A list of arguments that are handed to the sampler function, specified by `sampler`.
#         The first argument handed to the sampler is `pspec`, the second argument is `qubit_labels`,
#         and `samplerargs` lists the remaining arguments handed to the sampler. This is not
#         optional for some choices of `sampler`.

#     addlocal : bool, optional
#         Whether to follow each layer in the "core" circuit, sampled according to `sampler` with
#         a layer of 1-qubit gates.

#     lsargs : list, optional
#         Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
#         layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.

#     randomizeout : bool, optional
#         If False, the ideal output of the circuit (the "success" or "survival" outcome) is the all-zeros
#         bit string. If True, the ideal output of the circuit is randomized to a uniformly random bit-string.
#         This setting is useful for, e.g., detecting leakage/loss/measurement-bias etc.

#     cliffordtwirl : bool, optional
#         Wether to begin the circuit with a sequence that generates a random stabilizer state. For
#         standard DRB this should be set to True. There are a variety of reasons why it is better
#         to have this set to True.

#     conditionaltwirl : bool, optional
#         DRB only requires that the initial/final sequences of step (1) and (3) create/measure
#         a uniformly random / particular stabilizer state, rather than implement a particular unitary.
#         step (1) and (3) can be achieved by implementing a uniformly random Clifford gate and the
#         unique inversion Clifford, respectively. This is implemented if `conditionaltwirl` is False.
#         However, steps (1) and (3) can be implemented much more efficiently than this: the sequences
#         of (1) and (3) only need to map a particular input state to a particular output state,
#         if `conditionaltwirl` is True this more efficient option is chosen -- this is option corresponds
#         to "standard" DRB. (the term "conditional" refers to the fact that in this case we essentially
#         implementing a particular Clifford conditional on a known input).

#     citerations : int, optional
#         Some of the stabilizer state / Clifford compilation algorithms in pyGSTi (including the default
#         algorithms) are  randomized, and the lowest-cost circuit is chosen from all the circuit generated
#         in the iterations of the algorithm. This is the number of iterations used. The time required to
#         generate a DRB circuit is linear in `citerations`. Lower-depth / lower 2-qubit gate count
#         compilations of steps (1) and (3) are important in order to successfully implement DRB on as many
#         qubits as possible.

#     compilerargs : list, optional
#         A list of arguments that are handed to the compile_stabilier_state/measurement()functions (or the
#         compile_clifford() function if `conditionaltwirl `is False). This includes all the optional
#         arguments of these functions *after* the `iterations` option (set by `citerations`). For most
#         purposes the default options will be suitable (or at least near-optimal from the compilation methods
#         in-built into pyGSTi). See the docstrings of these functions for more information.

#     partitioned : bool, optional
#         If False, a single circuit is returned consisting of the full circuit. If True, three circuits
#         are returned in a list consisting of: (1) the stabilizer-prep circuit, (2) the core random circuit,
#         (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1)
#         and then (3) to (1).

#     seed: int, optional
#         Seed for RNG

#     Returns
#     -------
#     Circuit or list of Circuits
#         If partioned is False, a random DRB circuit sampled as specified. If partioned is True, a list of
#         three circuits consisting of (1) the stabilizer-prep circuit, (2) the core random circuit,
#         (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1)
#         and then (3) to (1) [except in the case of cliffordtwirl=False, when it is a list of two circuits].
#     Tuple
#         A length-n tuple of integers in [0,1], corresponding to the error-free outcome of the
#         circuit. Always all zeros if `randomizeout` is False. The ith element of the tuple
#         corresponds to the error-free outcome for the qubit labelled by: the ith element of
#         `qubit_labels`, if `qubit_labels` is not None; the ith element of `pspec.qubit_labels`, otherwise.
#         In both cases, the ith element of the tuple corresponds to the error-free outcome for the
#         qubit on the ith wire of the output circuit.
#     """
#     if isinstance(structure, str):
#         assert(structure == '1Q'), "The only default `structure` option is the string '1Q'"
#         structure = tuple([(q,) for q in pspec.qubit_labels])
#         n = pspec.num_qubits
#     else:
#         assert(isinstance(structure, list) or isinstance(structure, tuple)
#                ), "If not a string, `structure` must be a list or tuple."
#         qubits_used = []
#         for qubit_labels in structure:
#             assert(isinstance(qubit_labels, list) or isinstance(
#                 qubit_labels, tuple)), "SubsetQs must be a list or a tuple!"
#             qubits_used = qubits_used + list(qubit_labels)
#             assert(len(set(qubits_used)) == len(qubits_used)
#                    ), "The qubits in the tuples/lists of `structure must all be unique!"

#         assert(set(qubits_used).issubset(set(pspec.qubit_labels))
#                ), "The qubits to benchmark must all be in the QubitProcessorSpec `pspec`!"
#         n = len(qubits_used)

#     for qubit_labels in structure:
#         subgraph = pspec.qubit_graph.subgraph(list(qubit_labels))  # or pspec.compute_clifford_2Q_connectivity?
#         assert(subgraph.is_connected_graph()), "Each subset of qubits in `structure` must be connected!"

#     rand_state = _np.random.RandomState(seed)  # OK if seed is None

#     # Creates a empty circuit over no wires
#     circuit = _cir.Circuit(num_lines=0, editable=True)

#     s_rc_dict = {}
#     p_rc_dict = {}
#     circuit_dict = {}

#     for qubit_labels in structure:
#         qubit_labels = tuple(qubit_labels)
#         # Sample a random circuit of "native gates" over this set of qubits, with the
#         # specified sampling.
#         subset_circuit = create_random_circuit(pspec=pspec, length=length, qubit_labels=qubit_labels, sampler=sampler,
#                                                samplerargs=samplerargs, addlocal=addlocal, lsargs=lsargs,
#                                                rand_state=rand_state)
#         circuit_dict[qubit_labels] = subset_circuit
#         # find the symplectic matrix / phase vector this circuit implements.
#         s_rc_dict[qubit_labels], p_rc_dict[qubit_labels] = _symp.symplectic_rep_of_clifford_circuit(
#             subset_circuit, pspec=pspec)
#         # Tensors this circuit with the current circuit
#         circuit.tensor_circuit_inplace(subset_circuit)

#     # Creates empty circuits over no wires
#     inversion_circuit = _cir.Circuit(num_lines=0, editable=True)
#     if cliffordtwirl:
#         initial_circuit = _cir.Circuit(num_lines=0, editable=True)

#     for qubit_labels in structure:
#         qubit_labels = tuple(qubit_labels)
#         subset_n = len(qubit_labels)
#         # If we are clifford twirling, we do an initial random circuit that is either a uniformly random
#         # cliffor or creates a uniformly random stabilizer state from the standard input.
#         if cliffordtwirl:

#             # Sample a uniformly random Clifford.
#             s_initial, p_initial = _symp.random_clifford(subset_n, rand_state=rand_state)
#             # Find the composite action of this uniformly random clifford and the random circuit.
#             s_composite, p_composite = _symp.compose_cliffords(s_initial, p_initial, s_rc_dict[qubit_labels],
#                                                                p_rc_dict[qubit_labels])

#             # If conditionaltwirl we do a stabilizer prep (a conditional Clifford).
#             if conditionaltwirl:
#                 subset_initial_circuit = _cmpl.compile_stabilizer_state(s_initial, p_initial, pspec,
#                                                                         clifford_compilations.get('absolute', None),
#                                                                         clifford_compilations.get('paulieq', None),
#                                                                         qubit_labels,
#                                                                         citerations, *compilerargs,
#                                                                         rand_state=rand_state)
#             # If not conditionaltwirl, we do a full random Clifford.
#             else:
#                 subset_initial_circuit = _cmpl.compile_clifford(s_initial, p_initial, pspec,
#                                                                 clifford_compilations.get('absolute', None),
#                                                                 clifford_compilations.get('paulieq', None),
#                                                                 qubit_labels, citerations,
#                                                                 *compilerargs, rand_state=rand_state)

#             initial_circuit.tensor_circuit_inplace(subset_initial_circuit)

#         # If we are not Clifford twirling, we just copy the effect of the random circuit as the effect
#         # of the "composite" prep + random circuit (as here the prep circuit is the null circuit).
#         else:
#             s_composite = _copy.deepcopy(s_rc_dict[qubit_labels])
#             p_composite = _copy.deepcopy(p_rc_dict[qubit_labels])

#         if conditionaltwirl:
#             # If we want to randomize the expected output then randomize the p vector, otherwise
#             # it is left as p. Note that, unlike with compile_clifford, we don't invert (s,p)
#             # before handing it to the stabilizer measurement function.
#             if randomizeout: p_for_measurement = _symp.random_phase_vector(s_composite, subset_n,
#                                                                            rand_state=rand_state)
#             else: p_for_measurement = p_composite
#             subset_inversion_circuit = _cmpl.compile_stabilizer_measurement(
#                 s_composite, p_for_measurement, pspec,
#                 clifford_compilations.get('absolute', None),
#                 clifford_compilations.get('paulieq', None),
#                 qubit_labels, citerations, *compilerargs,
#                 rand_state=rand_state)
#         else:
#             # Find the Clifford that inverts the circuit so far. We
#             s_inverse, p_inverse = _symp.inverse_clifford(s_composite, p_composite)
#             # If we want to randomize the expected output then randomize the p_inverse vector, otherwise
#             # do not.
#             if randomizeout: p_for_inversion = _symp.random_phase_vector(s_inverse, subset_n, rand_state=rand_state)
#             else: p_for_inversion = p_inverse
#             # Compile the Clifford.
#             subset_inversion_circuit = _cmpl.compile_clifford(s_inverse, p_for_inversion, pspec,
#                                                               clifford_compilations.get('absolute', None),
#                                                               clifford_compilations.get('paulieq', None),
#                                                               qubit_labels, citerations, *compilerargs,
#                                                               rand_state=rand_state)

#         inversion_circuit.tensor_circuit_inplace(subset_inversion_circuit)

#     inversion_circuit.done_editing()

#     if cliffordtwirl:
#         full_circuit = initial_circuit.copy(editable=True)
#         full_circuit.append_circuit_inplace(circuit)
#         full_circuit.append_circuit_inplace(inversion_circuit)
#     else:
#         full_circuit = _copy.deepcopy(circuit)
#         full_circuit.append_circuit_inplace(inversion_circuit)

#     full_circuit.done_editing()

#     # Find the expected outcome of the circuit.
#     s_out, p_out = _symp.symplectic_rep_of_clifford_circuit(full_circuit, pspec=pspec)
#     if conditionaltwirl:  # s_out is not always the identity with a conditional twirl,
#         # only conditional on prep/measure.
#         assert(_np.array_equal(s_out[:n, n:], _np.zeros((n, n), _np.int64))), "Compiler has failed!"
#     else: assert(_np.array_equal(s_out, _np.identity(2 * n, _np.int64))), "Compiler has failed!"

#     # Find the ideal output of the circuit.
#     s_inputstate, p_inputstate = _symp.prep_stabilizer_state(n, zvals=None)
#     s_outstate, p_outstate = _symp.apply_clifford_to_stabilizer_state(s_out, p_out, s_inputstate, p_inputstate)
#     idealout = []
#     for qubit_labels in structure:
#         subset_idealout = []
#         for q in qubit_labels:
#             qind = circuit.line_labels.index(q)
#             measurement_out = _symp.pauli_z_measurement(s_outstate, p_outstate, qind)
#             bit = measurement_out[1]
#             assert(bit == 0 or bit == 1), "Ideal output is not a computational basis state!"
#             if not randomizeout:
#                 assert(bit == 0), "Ideal output is not the all 0s computational basis state!"
#             subset_idealout.append(int(bit))
#         idealout.append(tuple(subset_idealout))
#     idealout = tuple(idealout)

#     if not partitioned: outcircuit = full_circuit
#     else:
#         if cliffordtwirl: outcircuit = [initial_circuit, circuit, inversion_circuit]
#         else: outcircuit = [circuit, inversion_circuit]

#     return outcircuit, idealout


# def create_simultaneous_direct_rb_experiment(pspec, depths, circuits_per_length, structure='1Q',
#                                              sampler='Qelimination',
#                                              samplerargs=[], addlocal=False, lsargs=[], randomizeout=False,
#                                              cliffordtwirl=True, conditionaltwirl=True, citerations=20,
#                                              compilerargs=[],
#                                              partitioned=False, set_isolated=True, setcomplement_isolated=False,
#                                              descriptor='A set of simultaneous DRB experiments', verbosity=1,
#                                              seed=1234):
#     """
#     Generates a simultaneous "direct randomized benchmarking" (DRB) experiments (circuits).

#     DRB is the protocol introduced in arXiv:1807.07975 (2018).
#     An n-qubit DRB circuit consists of (1) a circuit the prepares a uniformly random stabilizer state;
#     (2) a length-l circuit (specified by `length`) consisting of circuit layers sampled according to
#     some user-specified distribution (specified by `sampler`), (3) a circuit that maps the output of
#     the preceeding circuit to a computational basis state. See arXiv:1807.07975 (2018) for further
#     details. In simultaneous DRB ...... <TODO more description>.

#     Parameters
#     ----------
#     pspec : QubitProcessorSpec
#         The QubitProcessorSpec for the device that the circuit is being sampled for, which defines the
#         "native" gate-set and the connectivity of the device. The returned DRB circuit will be over
#         the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
#         is always handed to the sampler, as the first argument of the sampler function (this is only
#         of importance when not using an in-built sampler for the "core" of the DRB circuit). Unless
#         `qubit_labels` is not None, the circuit is sampled over all the qubits in `pspec`.

#     depths : int
#         The set of "direct RB depths" for the circuits. The DRB depths must be integers >= 0.
#         Unless `addlocal` is True, the DRB length is the depth of the "core" random circuit,
#         sampled according to `sampler`, specified in step (2) above. If `addlocal` is True,
#         each layer in the "core" circuit sampled according to "sampler` is followed by a layer of
#         1-qubit gates, with sampling specified by `lsargs` (and the first layer is proceeded by a
#         layer of 1-qubit gates), and so the circuit of step (2) is length 2*`length` + 1.

#     circuits_per_length : int
#         The number of (possibly) different DRB circuits sampled at each length.

#     structure : str or tuple.
#         Defines the "structure" of the simultaneous DRB experiment. TODO : more details.

#     sampler : str or function, optional
#         If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
#         Except for 'local', this corresponds to sampling layers according to the sampling function
#         in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
#         corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
#         a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function].
#         If `sampler` is a function, it should be a function that takes as the first argument a
#         QubitProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
#         the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is the
#         only sampler that requires no parameters beyond the QubitProcessorSpec *and* works for arbitrary
#         connectivity devices. See the docstrings for each of these samplers for more information.

#     samplerargs : list, optional
#         A list of arguments that are handed to the sampler function, specified by `sampler`.
#         The first argument handed to the sampler is `pspec`, the second argument is `qubit_labels`,
#         and `samplerargs` lists the remaining arguments handed to the sampler. This is not
#         optional for some choices of `sampler`.

#     addlocal : bool, optional
#         Whether to follow each layer in the "core" circuits, sampled according to `sampler` with
#         a layer of 1-qubit gates.

#     lsargs : list, optional
#         Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
#         layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.

#     randomizeout : bool, optional
#         If False, the ideal output of the circuits (the "success" or "survival" outcome) is the all-zeros
#         bit string. If True, the ideal output of each circuit is randomized to a uniformly random bit-string.
#         This setting is useful for, e.g., detecting leakage/loss/measurement-bias etc.

#     cliffordtwirl : bool, optional
#         Wether to begin the circuit with a sequence that generates a random stabilizer state. For
#         standard DRB this should be set to True. There are a variety of reasons why it is better
#         to have this set to True.

#     conditionaltwirl : bool, optional
#         DRB only requires that the initial/final sequences of step (1) and (3) create/measure
#         a uniformly random / particular stabilizer state, rather than implement a particular unitary.
#         step (1) and (3) can be achieved by implementing a uniformly random Clifford gate and the
#         unique inversion Clifford, respectively. This is implemented if `conditionaltwirl` is False.
#         However, steps (1) and (3) can be implemented much more efficiently than this: the sequences
#         of (1) and (3) only need to map a particular input state to a particular output state,
#         if `conditionaltwirl` is True this more efficient option is chosen -- this is option corresponds
#         to "standard" DRB. (the term "conditional" refers to the fact that in this case we essentially
#         implementing a particular Clifford conditional on a known input).

#     citerations : int, optional
#         Some of the stabilizer state / Clifford compilation algorithms in pyGSTi (including the default
#         algorithms) are  randomized, and the lowest-cost circuit is chosen from all the circuits generated
#         in the iterations of the algorithm. This is the number of iterations used. The time required to
#         generate a DRB circuit is linear in `citerations`. Lower-depth / lower 2-qubit gate count
#         compilations of steps (1) and (3) are important in order to successfully implement DRB on as many
#         qubits as possible.

#     compilerargs : list, optional
#         A list of arguments that are handed to the compile_stabilier_state/measurement()functions (or the
#         compile_clifford() function if `conditionaltwirl `is False). This includes all the optional
#         arguments of these functions *after* the `iterations` option (set by `citerations`). For most
#         purposes the default options will be suitable (or at least near-optimal from the compilation methods
#         in-built into pyGSTi). See the docstrings of these functions for more information.

#     partitioned : bool, optional
#         If False, each circuit is returned as a single full circuit. If True, each circuit is returned as
#         a list of three circuits consisting of: (1) the stabilizer-prep circuit, (2) the core random circuit,
#         (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1)
#         and then (3) to (1).

#     set_isolated : bool, optional
#         Todo

#     setcomplement_isolated : bool, optional
#         Todo

#     descriptor : str, optional
#         A description of the experiment being generated. Stored in the output dictionary.

#     verbosity : int, optional
#         If > 0 the number of circuits generated so far is shown.

#     seed: int, optional
#         Seed for RNG

#     Returns
#     -------
#     Circuit or list of Circuits
#         If partioned is False, a random DRB circuit sampled as specified. If partioned is True, a list of
#         three circuits consisting of (1) the stabilizer-prep circuit, (2) the core random circuit,
#         (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1)
#         and then (3) to (1).
#     Tuple
#         A length-n tuple of integers in [0,1], corresponding to the error-free outcome of the
#         circuit. Always all zeros if `randomizeout` is False. The ith element of the tuple
#         corresponds to the error-free outcome for the qubit labelled by: the ith element of
#         `qubit_labels`, if `qubit_labels` is not None; the ith element of `pspec.qubit_labels`, otherwise.
#         In both cases, the ith element of the tuple corresponds to the error-free outcome for the
#         qubit on the ith wire of the output circuit.
#     dict
#         A dictionary containing the generated RB circuits, the error-free outputs of the circuit,
#         and the specification used to generate the circuits. The keys are:

#         - 'circuits'. A dictionary of the sampled circuits. The circuit with key(l,k) is the kth circuit
#         at DRB length l.

#         - 'idealout'. A dictionary of the error-free outputs of the circuits as tuples. The tuple with
#         key(l,k) is the error-free output of the (l,k) circuit. The ith element of this tuple corresponds
#         to the error-free outcome for the qubit on the ith wire of the output circuit and/or the ith element
#         of the list at the key 'qubitordering'. These tuples will all be (0,0,0,...) when `randomizeout` is
#         False

#         - 'qubitordering'. The ordering of the qubits in the 'idealout' tuples.

#         - 'spec'. A dictionary containing all of the parameters handed to this function, except `pspec`.
#         This then specifies how the circuits where generated.
#     """

#     experiment_dict = {}
#     experiment_dict['spec'] = {}
#     experiment_dict['spec']['depths'] = depths
#     experiment_dict['spec']['circuits_per_length'] = circuits_per_length
#     experiment_dict['spec']['sampler'] = sampler
#     experiment_dict['spec']['samplerargs'] = samplerargs
#     experiment_dict['spec']['addlocal'] = addlocal
#     experiment_dict['spec']['lsargs'] = lsargs
#     experiment_dict['spec']['randomizeout'] = randomizeout
#     experiment_dict['spec']['cliffordtwirl'] = cliffordtwirl
#     experiment_dict['spec']['conditionaltwirl'] = conditionaltwirl
#     experiment_dict['spec']['citerations'] = citerations
#     experiment_dict['spec']['compilerargs'] = compilerargs
#     experiment_dict['spec']['partitioned'] = partitioned
#     experiment_dict['spec']['descriptor'] = descriptor
#     experiment_dict['spec']['createdby'] = 'extras.rb.sample.simultaneous_direct_rb_experiment'

#     #rand_state = _np.random.RandomState(seed) # OK if seed is None

#     if isinstance(structure, str):
#         assert(structure == '1Q'), "The only default `structure` option is the string '1Q'"
#         structure = tuple([(q,) for q in pspec.qubit_labels])
#     else:
#         assert(isinstance(structure, list) or isinstance(structure, tuple)), \
#             "If not a string, `structure` must be a list or tuple."
#         qubits_used = []
#         for qubit_labels in structure:
#             assert(isinstance(qubit_labels, list) or isinstance(
#                 qubit_labels, tuple)), "SubsetQs must be a list or a tuple!"
#             qubits_used = qubits_used + list(qubit_labels)
#             assert(len(set(qubits_used)) == len(qubits_used)), \
#                 "The qubits in the tuples/lists of `structure must all be unique!"

#         assert(set(qubits_used).issubset(set(pspec.qubit_labels))), \
#             "The qubits to benchmark must all be in the QubitProcessorSpec `pspec`!"

#     experiment_dict['spec']['structure'] = structure
#     experiment_dict['circuits'] = {}
#     experiment_dict['target'] = {}
#     experiment_dict['settings'] = {}

#     for qubit_labels in structure:
#         subgraph = pspec.qubit_graph.subgraph(list(qubit_labels))  # or pspec.compute_clifford_2Q_connectivity?
#         assert(subgraph.is_connected_graph()), "Each subset of qubits in `structure` must be connected!"

#     for lnum, l in enumerate(depths):
#         lseed = seed + lnum * circuits_per_length
#         if verbosity > 0:
#             print('- Sampling {} circuits at DRB length {} ({} of {} depths) with seed {}'.format(circuits_per_length,
#                                                                                                   l, lnum + 1,
#                                                                                                   len(depths), lseed))
#             print('  - Number of circuits sampled = ', end='')
#         for j in range(circuits_per_length):
#             circuit, idealout = sample_simultaneous_direct_rb_circuit(pspec, l, structure=structure, sampler=sampler,
#                                                                       samplerargs=samplerargs, addlocal=addlocal,
#                                                                       lsargs=lsargs, randomizeout=randomizeout,
#                                                                       cliffordtwirl=cliffordtwirl,
#                                                                       conditionaltwirl=conditionaltwirl,
#                                                                       citerations=citerations,
#                                                                       compilerargs=compilerargs,
#                                                                       partitioned=partitioned,
#                                                                       seed=lseed + j)

#             if (not set_isolated) and (not setcomplement_isolated):
#                 experiment_dict['circuits'][l, j] = circuit
#                 experiment_dict['target'][l, j] = idealout

#             else:
#                 experiment_dict['circuits'][l, j] = {}
#                 experiment_dict['target'][l, j] = {}
#                 experiment_dict['settings'][l, j] = {}
#                 experiment_dict['circuits'][l, j][tuple(structure)] = circuit
#                 experiment_dict['target'][l, j][tuple(structure)] = idealout
#                 experiment_dict['settings'][l, j][tuple(structure)] = _get_setting(l, j, structure, depths,
#                                                                                    circuits_per_length, structure)

#             if set_isolated:
#                 for subset_ind, subset in enumerate(structure):
#                     subset_circuit = circuit.copy(editable=True)
#                     for q in circuit.line_labels:
#                         if q not in subset:
#                             subset_circuit.replace_with_idling_line_inplace(q)
#                     subset_circuit.done_editing()
#                     experiment_dict['circuits'][l, j][(tuple(subset),)] = subset_circuit
#                     experiment_dict['target'][l, j][(tuple(subset),)] = (idealout[subset_ind],)
#                     experiment_dict['settings'][l, j][(tuple(subset),)] = _get_setting(l, j, (tuple(subset),), depths,
#                                                                                        circuits_per_length, structure)

#             if setcomplement_isolated:
#                 for subset_ind, subset in enumerate(structure):
#                     subsetcomplement_circuit = circuit.copy(editable=True)
#                     for q in circuit.line_labels:
#                         if q in subset:
#                             subsetcomplement_circuit.replace_with_idling_line_inplace(q)
#                     subsetcomplement_circuit.done_editing()
#                     subsetcomplement = list(_copy.copy(structure))
#                     subsetcomplement_idealout = list(_copy.copy(idealout))
#                     del subsetcomplement[subset_ind]
#                     del subsetcomplement_idealout[subset_ind]
#                     subsetcomplement = tuple(subsetcomplement)
#                     subsetcomplement_idealout = tuple(subsetcomplement_idealout)
#                     experiment_dict['circuits'][l, j][subsetcomplement] = subsetcomplement_circuit
#                     experiment_dict['target'][l, j][subsetcomplement] = subsetcomplement_idealout
#                     experiment_dict['settings'][l, j][subsetcomplement] = _get_setting(l, j, subsetcomplement, depths,
#                                                                                        circuits_per_length, structure)

#             if verbosity > 0: print(j + 1, end=',')
#         if verbosity > 0: print('')

#     return experiment_dict


def create_clifford_rb_circuit(pspec, clifford_compilations, length, qubit_labels=None, randomizeout=False,
                               citerations=20, compilerargs=[], interleaved_circuit=None, seed=None):
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
        A dictionary with the potential keys `'absolute'` and `'paulieq'` and corresponding
        :class:`CompilationRules` values.  These compilation rules specify how to compile the
        "native" gates of `pspec` into Clifford gates.

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
    """
    
    # Find the labels of the qubits to create the circuit for.
    if qubit_labels is not None: qubits = qubit_labels[:]  # copy this list
    else: qubits = pspec.qubit_labels[:]  # copy this list
    # The number of qubits the circuit is over.
    n = len(qubits)

    rand_state = _np.random.RandomState(seed)  # OK if seed is None

    # Initialize the identity circuit rep.
    s_composite = _np.identity(2 * n, _np.int64)
    p_composite = _np.zeros((2 * n), _np.int64)
    # Initialize an empty circuit
    full_circuit = _cir.Circuit(layer_labels=[], line_labels=qubits, editable=True)

    # Sample length+1 uniformly random Cliffords (we want a circuit of length+2 Cliffords, in total), compile
    # them, and append them to the current circuit.
    for i in range(0, length + 1):

        s, p = _symp.random_clifford(n, rand_state=rand_state)
        circuit = _cmpl.compile_clifford(s, p, pspec,
                                         clifford_compilations.get('absolute', None),
                                         clifford_compilations.get('paulieq', None),
                                         qubit_labels=qubit_labels, iterations=citerations, *compilerargs,
                                         rand_state=rand_state)
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
                             samplerargs=[], localclifford=True, paulirandomize=True, seed=None):
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

#### Commented out as most of this functionality can be found elsewhere and this code has not been tested recently.
# def sample_one_q_generalized_rb_circuit(m, group_or_model, inverse=True, random_pauli=False, interleaved=None,
#                                         group_inverse_only=False, group_prep=False, compilation=None,
#                                         generated_group=None, model_to_group_labels=None, seed=None, rand_state=None):
#     """
#     Makes a random 1-qubit RB circuit, with RB over an arbitrary group.

#     This function also contains a range of other options that allow circuits for many
#     types of RB to be generated, including:

#     - Clifford RB
#     - Direct RB
#     - Interleaved Clifford or direct RB
#     - Unitarity Clifford or direct RB

#     The function can in-principle be used beyond 1-qubit RB, but it relies on explicit matrix representation
#     of a group, which is infeasble for, e.g., the many-qubit Clifford group.

#     Note that this function has *not* been carefully tested. This will be rectified in the future,
#     or this function will be replaced.

#     Parameters
#     ----------
#     m : int
#         The number of random gates in the circuit.

#     group_or_model : Model or MatrixGroup
#         Which Model of MatrixGroup to create the random circuit for. If
#         inverse is true and this is a Model, the Model gates must form
#         a group (so in this case it requires the *target model* rather than
#         a noisy model). When inverse is true, the MatrixGroup for the model
#         is generated. Therefore, if inverse is true and the function is called
#         multiple times, it will be much faster if the MatrixGroup is provided.

#     inverse : Bool, optional
#         If true, the random circuit is followed by its inverse gate. The model
#         must form a group if this is true. If it is true then the circuit
#         returned is length m+1 (2m+1) if interleaved is False (True).

#     random_pauli : <TODO typ>, optional
#         <TODO description>

#     interleaved : Str, optional
#         If not None, then a oplabel string. When a oplabel string is provided,
#         every random gate is followed by this gate. So the returned circuit is of
#         length 2m+1 (2m) if inverse is True (False).

#     group_inverse_only : <TODO typ>, optional
#         <TODO description>

#     group_prep : bool, optional
#         If group_inverse_only is True and inverse is True, setting this to true
#         creates a "group pre-twirl". Does nothing otherwise (which should be changed
#         at some point).

#     compilation : <TODO typ>, optional
#         <TODO description>

#     generated_group : <TODO typ>, optional
#         <TODO description>

#     model_to_group_labels : <TODO typ>, optional
#         <TODO description>

#     seed : int, optional
#         Seed for random number generator; optional.

#     rand_state : numpy.random.RandomState, optional
#         A RandomState object to generate samples from. Can be useful to set
#         instead of `seed` if you want reproducible distribution samples across
#         multiple random function calls but you don't want to bother with
#         manually incrementing seeds between those calls.

#     Returns
#     -------
#     Circuit
#         The random circuit of length:
#         m if inverse = False, interleaved = None
#         m + 1 if inverse = True, interleaved = None
#         2m if inverse = False, interleaved not None
#         2m + 1 if inverse = True, interleaved not None
#     """
#     assert hasattr(group_or_model, 'gates') or hasattr(group_or_model,
#                                                        'product'), 'group_or_model must be a MatrixGroup of Model'
#     group = None
#     model = None
#     if hasattr(group_or_model, 'gates'):
#         model = group_or_model
#     if hasattr(group_or_model, 'product'):
#         group = group_or_model

#     if rand_state is None:
#         rndm = _np.random.RandomState(seed)  # ok if seed is None
#     else:
#         rndm = rand_state

#     if (inverse) and (not group_inverse_only):
#         if model:
#             group = _rbobjs.MatrixGroup(group_or_model.operations.values(),
#                                         group_or_model.operations.keys())

#         rndm_indices = rndm.randint(0, len(group), m)
#         if interleaved:
#             interleaved_index = group.label_indices[interleaved]
#             interleaved_indices = interleaved_index * _np.ones((m, 2), _np.int64)
#             interleaved_indices[:, 0] = rndm_indices
#             rndm_indices = interleaved_indices.flatten()

#         random_string = [group.labels[i] for i in rndm_indices]
#         effective_op = group.product(random_string)
#         inv = group.inverse_index(effective_op)
#         random_string.append(inv)

#     if (inverse) and (group_inverse_only):
#         assert (model is not None), "gateset_or_group should be a Model!"
#         assert (compilation is not None), "Compilation of group elements to model needs to be specified!"
#         assert (generated_group is not None), "Generated group needs to be specified!"
#         if model_to_group_labels is None:
#             model_to_group_labels = {}
#             for gate in model.primitive_op_labels:
#                 assert(gate in generated_group.labels), "model labels are not in \
#                 the generated group! Specify a model_to_group_labels dictionary."
#                 model_to_group_labels = {'gate': 'gate'}
#         else:
#             for gate in model.primitive_op_labels:
#                 assert(gate in model_to_group_labels.keys()), "model to group labels \
#                 are invalid!"
#                 assert(model_to_group_labels[gate] in generated_group.labels), "model to group labels \
#                 are invalid!"

#         opLabels = model.primitive_op_labels
#         rndm_indices = rndm.randint(0, len(opLabels), m)
#         if interleaved:
#             interleaved_index = opLabels.index(interleaved)
#             interleaved_indices = interleaved_index * _np.ones((m, 2), _np.int64)
#             interleaved_indices[:, 0] = rndm_indices
#             rndm_indices = interleaved_indices.flatten()

#         # This bit of code is a quick hashed job. Needs to be checked at somepoint
#         if group_prep:
#             rndm_group_index = rndm.randint(0, len(generated_group))
#             prep_random_string = compilation[generated_group.labels[rndm_group_index]]
#             prep_random_string_group = [generated_group.labels[rndm_group_index], ]

#         random_string = [opLabels[i] for i in rndm_indices]
#         random_string_group = [model_to_group_labels[opLabels[i]] for i in rndm_indices]
#         # This bit of code is a quick hashed job. Needs to be checked at somepoint
#         if group_prep:
#             random_string = prep_random_string + random_string
#             random_string_group = prep_random_string_group + random_string_group
#         #print(random_string)
#         inversion_group_element = generated_group.inverse_index(generated_group.product(random_string_group))

#         # This bit of code is a quick hash job, and only works when the group is the 1-qubit Cliffords
#         if random_pauli:
#             pauli_keys = ['Gc0', 'Gc3', 'Gc6', 'Gc9']
#             rndm_index = rndm.randint(0, 4)

#             if rndm_index == 0 or rndm_index == 3:
#                 bitflip = False
#             else:
#                 bitflip = True
#             inversion_group_element = generated_group.product([inversion_group_element, pauli_keys[rndm_index]])

#         inversion_sequence = compilation[inversion_group_element]
#         random_string.extend(inversion_sequence)

#     if not inverse:
#         if model:
#             opLabels = model.primitive_op_labels
#             rndm_indices = rndm.randint(0, len(opLabels), m)
#             if interleaved:
#                 interleaved_index = opLabels.index(interleaved)
#                 interleaved_indices = interleaved_index * _np.ones((m, 2), _np.int64)
#                 interleaved_indices[:, 0] = rndm_indices
#                 rndm_indices = interleaved_indices.flatten()
#             random_string = [opLabels[i] for i in rndm_indices]

#         else:
#             rndm_indices = rndm.randint(0, len(group), m)
#             if interleaved:
#                 interleaved_index = group.label_indices[interleaved]
#                 interleaved_indices = interleaved_index * _np.ones((m, 2), _np.int64)
#                 interleaved_indices[:, 0] = rndm_indices
#                 rndm_indices = interleaved_indices.flatten()
#             random_string = [group.labels[i] for i in rndm_indices]

#     if not random_pauli:
#         return _cir.Circuit(random_string)
#     if random_pauli:
#         return _cir.Circuit(random_string), bitflip


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


def create_udrb_circuit(pspec, length, qubit_labels = None, layer_type='cz-zxzxz',twoQ_gate_density = 1/2, angles=[_np.pi/2, -1*_np.pi/2]):
    #generates a unitary direct RB circuit

    if qubit_labels == None:
        qubit_labels = pspec.qubit_labels
        
    n = len(qubit_labels)
    
    #pspec.add_std_model("standard_unitary", parameterization='static unitary')
    model = _models.create_crosstalk_free_model(pspec, evotype='statevec', simulator='matrix') #pspec.models['standard_unitary']
    
    if n == 1:
        mean_two_q_gates = 0
        ###for 1Q, currently sampling Haar-random unitaries with pyGSTi terms###
        #params = _rc.sample_haar_random_one_qubit_unitary_parameters()

        drb_circ = sample_compiled_haar_random_one_qubit_gates_zxzxz_circuit(pspec)
        #print(drb_circ)
        u = model.sim.product(drb_circ)
        drb_circ = drb_circ.copy(editable=True)
    else:
        mean_two_q_gates = twoQ_gate_density*n/2
        u = unitary_group.rvs(2**n) #generate Haar-random unitary
        ###use the compiler###
        drb_circ = compile_unitary_qsearch(u, qubit_labels = qubit_labels).copy(editable = True)

    
    #generate d random layers, alternating Haar-random 1Q unitaries and 2Q gates
    #circuit = _cir.Circuit(layer_labels=[], line_labels=qubit_labels, editable=True)
    
####replace with existing pygsti machinery for generating these circuits#######
    #have option to sample alternating laters or mixed layers
    if layer_type=='cz-zxzxz':
        circuit2 = sample_random_cz_zxzxz_circuit(pspec, length, qubit_labels=qubit_labels, two_q_gate_density=twoQ_gate_density,
                                   one_q_gate_type='haar',
                                   two_q_gate_args_lists={'Gczr': [(str(a),) for a in angles]})
    #consider adding in a check for continuous-arg gates
    #elif layer_type=='standard':                                   
    #    circuit2 = create_random_circuit(pspec, length, qubit_labels=qubit_labels, sampler='edgegrab', samplerargs=[twoQ_gate_density])                                  
                                   
    #for a in range(length):
    #   #generate random 1q unitary layer
    #    new_layer = sample_1q_unitary_layer(pspec, qubit_labels)
    #    #append new layer to circuit
    #    circuit.append_circuit_inplace(new_layer)
    #    #generate 2q gate layer
    #    sampled_layer = sample_cz_layer_by_edgegrab(pspec, qubit_labels=qubit_labels, mean_two_q_gates=mean_two_q_gates, angles=angles)
    #    if sampled_layer == []: new_layer = _cir.Circuit(layer_labels=[], line_labels=qubit_labels, editable=True)
    #    else: new_layer = _cir.Circuit([sampled_layer])
    #    #append new layer to circuit
    #    circuit.append_circuit_inplace(new_layer)
    ####
    
    #circuit.done_editing()
    
    #get unitary for circuit 
    circuit = circuit2.copy(editable=True)
    circuit.delete_idle_layers_inplace()

    print(circuit)

    circ_unitary = model.sim.product(circuit)
    
    drb_circ.append_circuit_inplace(circuit)
    
    #multiply u by new circ operation
    u_net =_np.dot(circ_unitary, u)
    
    #invert this matrix
    u_inv =_np.linalg.inv(u_net)
    
    #pick a random n-qubit Pauli to be the net Pauli
    paulis = [_gates.standard_gatename_unitaries()['Gc'+str(3*k)] for k in range(4)]
    pauli_choice = [_np.random.randint(4) for _ in range(n)]
    random_pauli = paulis[pauli_choice[0]]
    for i in pauli_choice[1:]:
    	random_pauli = _np.kron(random_pauli, paulis[i])
    #random_pauli = functools.reduce(_np.kron, [paulis[i] for i in pauli_choice])
    inv_mat =_np.dot(random_pauli, u_inv) #multiply in random pauli
    
    ####USE THE COMPILER###
    #single-qubit edge case
    if n == 1:
        inv_circ = compile_1q_unitary(inv_mat, qubit_labels = qubit_labels) #.copy(editable=True)
        #print(np.dot(np.linalg.inv(inv_mat), model.sim.product(inv_circ)))
    else:
        inv_circ = compile_unitary_qsearch(inv_mat, qubit_labels = qubit_labels).copy(editable = True)
    drb_circ.append_circuit_inplace(inv_circ)
    drb_circ.done_editing()
    #####################
    
    #compute ideal output
    idealout = ''.join(['0' if p==0 or p==3 else '1' for p in pauli_choice])
    
    
    #return circuit and target bitstring
    return (drb_circ, idealout)

def compile_1q_unitary(u, qubit_labels = None):
    if qubit_labels == None:
        q = 'Q0'
    else:
        q = qubit_labels[0]
    alpha = 0.5*(_np.angle(u[0][0])+_np.angle(u[1][1]))
    #print(u)
    u *=_np.exp(-1j*alpha)
    #print(u)
    psi =_np.angle(u[0][0])
    chi =_np.angle(u[0][1])
    phi =_np.arctan2((np.exp(-1j*chi)*u[0][1]).real,(np.exp(-1j*psi)*u[0][0]).real)
    #print(phi)
    #print(psi)
    #print(chi)
    
    #print([[np.exp(1j*psi)*np.cos(phi),_np.exp(1j*chi)*np.sin(phi)],[-1*np.exp(-1j*chi)*np.sin(phi),_np.exp(-1j*psi)*np.cos(phi)]])
    
    theta1 = _mod_2pi(psi - chi +_np.pi)
    theta2 = _mod_2pi(_np.pi - 2*phi)
    theta3 = _mod_2pi(psi + chi)
    #print(theta1, theta2, theta3)
    
    #print([[np.exp(0.5*1j*(theta1+theta3))*np.sin(theta2/2),_np.exp(-1j*(0.5*(theta1-theta3)-np.pi))*np.cos(theta2/2)], 
          #[np.exp(-1j*(0.5*(theta3-theta1)+np.pi))*np.cos(theta2/2), -1*np.exp(-0.5*1j*(theta1+theta3))*np.sin(theta2/2)]])
    
    gate = 'Gzr;{t1}:{q}Gc16:{q}Gzr;{t2}:{q}Gc16:{q}Gzr;{t3}:{q}'.format(q=q, t1=theta1, t2=theta2, t3=theta3)
            
    gate_as_circ = _cir.Circuit(None, stringrep=gate)
    
    #print(model.sim.product(gate_as_circ))
    #print(np.dot(np.linalg.inv(model.sim.product(gate_as_circ)), u))
    return gate_as_circ

def compile_unitary_qsearch(u, qubit_labels = None):
    if qubit_labels == None:
        n = int(_np.log2(u.shape[0]))
        qubit_labels = ['Q'+str(i) for i in range(n)]
        
    options = _qsearch.options.Options()
    options.__setattr__('gateset', _qsearch.gatesets.U3CNOTLinear())
    options.__setattr__('target', u)
    compiler = _qsearch.SearchCompiler(options)
    result = compiler.compile()
    
    assembler_options = _qsearch.options.Options()
    assembler_options.__setattr__('assemblydict', _qsearch.assemblers.assemblydict_openqasm)
    assembler = _qsearch.assemblers.DictionaryAssembler(options = assembler_options)
    result = assembler.assemble(result)
    
    #remove initial junk
    result = result[result.find(';')+1:]
    result = result[result.find(';')+1:]
    #initialize a circuit
    c = ''
    #add gates one-by-one
    while result != '\n':
        new_gate = result[:result.find(';')] #get openQASM string for gate
        result = result[(result.find(';')+1):]
        #convert gate name to pyGSTi
        qasm_gate = new_gate[:new_gate.find('(')]
        if qasm_gate == '\nU':
            params = new_gate[new_gate.find('(')+1:new_gate.find(')')].split(', ')
            theta1 = -1*float(params[2])
            theta2 = -1*float(params[0]) +_np.pi
            theta3 = -1*float(params[1]) +_np.pi
            new_gate_q = int(new_gate[new_gate.find('q[')+2:new_gate.find(']')])
            new_q_label = qubit_labels[new_gate_q]
            gate = 'Gzr;{t1}:{q}Gc16:{q}Gzr;{t2}:{q}Gc16:{q}Gzr;{t3}:{q}'.format(q=new_q_label, t1=theta1, t2=theta2, t3=theta3)
            
            
        else:
            new_gate_qs = new_gate[4:]
            #now should have 'q[i] q[j]'
            new_gate_qs = new_gate_qs.replace('q[', '').replace(']', '')
            qubits = list(map(int, new_gate_qs.split(', ')))
            q1 = qubit_labels[qubits[0]]
            q2 = qubit_labels[qubits[1]]
            t1 = str(_np.pi)
            t2 = str(_np.pi/2)
            h = 'Gzr;{t1}:{q2}Gc16:{q2}Gzr;{t2}:{q2}Gc16:{q2}Gzr;{t1}:{q2}'.format(q2 = q2, t1 = t1, t2=t2)
            cphase = 'Gcphase:{q2}:{q1}'.format(q1=q1, q2=q2)
            gate=h+cphase+h
        
        #add gate to circuit
        c += gate

    circ = _cir.Circuit(None, stringrep=c).parallelize()

    return circ
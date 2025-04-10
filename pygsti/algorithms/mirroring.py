"""
circuit mirroring functions.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import copy as _copy
import random as _random

from pygsti.circuits import circuit as _cir
from pygsti.baseobjs import label as _lbl
from pygsti.tools import symplectic as _symp
from pygsti.tools import compilationtools as _comp

from . import randomcircuit as _rc

# ### TODO: THIS IS TIMS OLD CODE WHICH SHOULD PERHAPS ALSO BE AN OPTION IN THE `CREATE_MIRROR_CIRCUIT` FUNCTION
# def create_mirror_circuit(circ, pspec, circtype='Clifford+Gzr', pauli_labels=None, pluspi_prob=0.):
#     """
#     *****************************************************************
#     Function currently has the following limitations that need fixing:

#        - A layer contains only Clifford or Gzr gates on ALL the qubits.
#        - all of the Clifford gates are self inverse
#        - The qubits are labelled "Q0" through "Qn-1" -- THIS SHOULD NOW BE FIXED!
#        - Pauli's are labelled by "Gi", "Gxpi", "Gypi" and "Gzpi".
#        - There's no option for randomized prep/meas
#        - There's no option for randomly adding +/-pi to the Z rotation angles.
#        - There's no option for adding "barriers"
#        - There's no test that the 'Gzr' gate has the "correct" convention for a rotation angle
#          (a rotation by pi must be a Z gate) or that it's a rotation around Z.
#     *****************************************************************
#     """
#     assert(circtype == 'Clifford+Gzr' or circtype == 'Clifford')
#     n = circ.width
#     d = circ.depth
#     if pauli_labels is None: pauli_labels = ['Gi', 'Gxpi', 'Gypi', 'Gzpi']
#     qubits = circ.line_labels
#     identity = _np.identity(2 * n, _np.int64)
#     zrotname = 'Gzr'
#     # qubit_labels = ['G{}'.format(i) for i in range(n)]

#     _, gate_inverse = pspec.compute_one_qubit_gate_relations()
#     gate_inverse.update(pspec.compute_multiqubit_inversion_relations())  # add multiQ inverses

#     quasi_inverse_circ = []
#     central_pauli_circ = _cir.Circuit([[_lbl.Label(pauli_labels[_np.random.randint(0, 4)], q) for q in qubits]])
#     #telescoping_pauli = central_pauli_layer.copy()
#     # The telescoping Pauli in the symplectic rep.
#     telp_s, telp_p = _symp.symplectic_rep_of_clifford_circuit(central_pauli_circ, pspec=pspec)
#     assert(_np.sum(_np.abs(telp_s - identity)) <= 1e-8)  # Check that it's a Pauli.

#     for d_ind in range(d):
#         layer = circ.layer(d - d_ind - 1)
#         if layer[0].name == zrotname:
#             quasi_inverse_layer = []
#             for gate in layer:

#                 q_int = qubits.index(gate.qubits[0])
#                 angle = float(gate.args[0])

#                 if telp_p[n + q_int] == 0: rotation_sign = -1.  # If the Pauli is Z or I.
#                 else: rotation_sign = +1  # If the Pauli is X or Y.

#                 # Sets the quasi inversion angle to + or - the original angle, depending on the Paul
#                 quasi_inverse_angle = rotation_sign * angle
#                 # Decides whether to add with to add +/- pi to the rotation angle.
#                 if _np.random.binomial(1, pluspi_prob) == 1:
#                     quasi_inverse_angle += _np.pi * (-1)**_np.random.binomial(1, 0.5)
#                     quasi_inverse_angle = _comp.mod_2pi(quasi_inverse_angle)
#                     # Updates the telescoping Pauli (in the symplectic rep_, to include this added pi-rotation,
#                     # as we need to include it as we keep collapsing the circuit down.
#                     telp_p[q_int] = (telp_p[q_int] + 2) % 4
#                 # Constructs the quasi-inverse gate.
#                 quasi_inverse_gate = _lbl.Label(zrotname, gate.qubits, args=(str(quasi_inverse_angle),))
#                 quasi_inverse_layer.append(quasi_inverse_gate)

#             # We don't have to update the telescoping Pauli as it's unchanged, but when we update
#             # this it'll need to change.
#             #telp_p = telp_p

#         else:
#             quasi_inverse_layer = [_lbl.Label(gate_inverse[gate.name], gate.qubits) for gate in layer]
#             telp_layer = _symp.find_pauli_layer(telp_p, pauli_labels, qubits)
#             conjugation_circ = _cir.Circuit([layer, telp_layer, quasi_inverse_layer])
#             # We calculate what the new telescoping Pauli is, in the symplectic rep.
#             telp_s, telp_p = _symp.symplectic_rep_of_clifford_circuit(conjugation_circ, pspec=pspec)

#         # Check that the layer -- pauli -- quasi-inverse circuit implements a Pauli.
#         assert(_np.sum(_np.abs(telp_s - identity)) <= 1e-10)
#         # Add the quasi inverse layer that we've constructed to the end of the quasi inverse circuit.
#         quasi_inverse_circ.append(quasi_inverse_layer)

#     # now that we've completed the quasi inverse circuit we convert it to a Circuit object
#     quasi_inverse_circ = _cir.Circuit(quasi_inverse_circ)

#     # Calculate the bit string that this mirror circuit should output, from the final telescoped Pauli.
#     target_bitstring = ''.join(['1' if p == 2 else '0' for p in telp_p[n:]])
#     mirror_circuit = circ + central_pauli_circ + quasi_inverse_circ

#     return mirror_circuit, target_bitstring


def create_mirror_circuit(circ, pspec, circ_type='clifford+zxzxz'):
    """

    circ_type : clifford+zxzxz, cz(theta)+zxzxz
    """

    n = circ.width
    d = circ.depth

    pauli_labels = ['I', 'X', 'Y', 'Z']
    qubits = circ.line_labels

    _, gate_inverse = pspec.compute_one_qubit_gate_relations()
    gate_inverse.update(pspec.compute_multiqubit_inversion_relations())  # add multiQ inverse

    assert(circ_type in ('clifford+zxzxz', 'cz(theta)+zxzxz')), '{} not a valid circ_type!'.format(circ_type)

    def compute_gate_inverse(gate_label):

        if gate_label.name in gate_inverse.keys():
            return _lbl.Label(gate_inverse[gate_label.name], gate_label.qubits)
        else:
            if gate_label.name == 'Gzr' or gate_label.name == 'Gczr':
                return _lbl.Label(gate_label.name, gate_label.qubits, args=(str(-1 * float(gate_label.args[0])),))
            else:
                raise ValueError("Cannot invert gate with name {}".format(gate_label.name))

    srep_dict = _symp.compute_internal_gate_symplectic_representations(gllist=['I', 'X', 'Y', 'Z'])
    # the `callable` part is a workaround to remove gates with args, defined by functions.
    srep_dict.update(pspec.compute_clifford_symplectic_reps(tuple((gn for gn, u in pspec.gate_unitaries.items()
                                                                   if not callable(u)))))

    if 'Gxpi2' in pspec.gate_names:
        xname = 'Gxpi2'
    elif 'Gc16' in pspec.gate_names:
        xname = 'Gc16'
    else:
        raise ValueError(("There must be an X(pi/2) gate in the processor spec's gate set,"
                          " and it must be called Gxpi2 or Gc16!"))

    assert('Gzr' in pspec.gate_names), \
        "There must be an Z(theta) gate in the processor spec's gate set, and it must be called Gzr!"
    zrotname = 'Gzr'

    if circ_type == 'cz(theta)+zxzxz':
        assert('Gczr' in pspec.gate_names), \
            "There must be an controlled-Z(theta) gate in the processor spec's gate set, and it must be called Gczr!"
        czrotname = 'Gczr'

    Xpi2layer = [_lbl.Label(xname, q) for q in qubits]

    #make an editable copy of the circuit to add the inverse on to
    c = circ.copy(editable=True)

    #build the inverse
    d_ind = 0
    while d_ind < d:
        layer = circ.layer(d - d_ind - 1)
        if len(layer) > 0 and layer[0].name == zrotname:  # ask if it's a Zrot layer.
            # It's necessary for the whole layer to have Zrot gates
            #get the entire arbitrary 1q unitaries: Zrot-Xpi/2-Zrot-Xpi/2-Zrot
            current_layers = circ[d - d_ind - 5: d - d_ind]
            #recompile inverse of current layer
            for i in range(n):
                if n == 1:
                    old_params = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]),
                                   float(current_layers[4].args[0])) for i in range(n)]
                else:
                    old_params = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]),
                                   float(current_layers[4][i].args[0])) for i in range(n)]
                layer_new_params = [_comp.inv_recompile_unitary(*p) for p in old_params]
                theta1_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][0]),))
                                for i in range(len(layer_new_params))]
                theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),))
                                for i in range(len(layer_new_params))]
                theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),))
                                for i in range(len(layer_new_params))]

            #add to mirror circuit
            c.append_circuit_inplace(_cir.Circuit([theta3_layer], line_labels=circ.line_labels))
            c.append_circuit_inplace(_cir.Circuit([Xpi2layer], line_labels=circ.line_labels))
            c.append_circuit_inplace(_cir.Circuit([theta2_layer], line_labels=circ.line_labels))
            c.append_circuit_inplace(_cir.Circuit([Xpi2layer], line_labels=circ.line_labels))
            c.append_circuit_inplace(_cir.Circuit([theta1_layer], line_labels=circ.line_labels))

            d_ind += 5

        else:
            inverse_layer = [compute_gate_inverse(gate_label) for gate_label in layer]
            c.append_circuit_inplace(_cir.Circuit([inverse_layer], line_labels=circ.line_labels))
            d_ind += 1

    #now that we've built the simple mirror circuit, let's add pauli frame randomization
    d_ind = 0
    mc = []
    net_paulis = {q: 0 for q in qubits}
    d = c.depth
    correction_angles = {q: 0 for q in qubits}  # corrections used in the cz(theta) case, which do nothing otherwise.

    while d_ind < d:
        layer = c.layer(d_ind)
        if len(layer) > 0 and layer[0].name == zrotname:  # ask if it's a Zrot layer.
            #It's necessary for the whole layer to have Zrot gates
            #if the layer is 1Q unitaries, pauli randomize
            current_layers = c[d_ind:d_ind + 5]

            #generate random pauli
            new_paulis = {q: _np.random.randint(0, 4) for q in qubits}
            new_paulis_as_layer = [_lbl.Label(pauli_labels[new_paulis[q]], q) for q in qubits]

            net_paulis_as_layer = [_lbl.Label(pauli_labels[net_paulis[q]], q) for q in qubits]
            #compute new net pauli based on previous pauli
            net_pauli_numbers = _symp.find_pauli_number(_symp.symplectic_rep_of_clifford_circuit(
                _cir.Circuit(new_paulis_as_layer + net_paulis_as_layer, line_labels=circ.line_labels),
                srep_dict=srep_dict)[1])

            # THIS WAS THE (THETA) VERSIONS
            #net_paulis_as_layer = [_lbl.Label(pauli_labels[net_paulis[q]], q) for q in qubits]
            #net_pauli_numbers = _symp.find_pauli_number(_symp.symplectic_rep_of_clifford_circuit(_cir.Circuit(
            #                                        new_paulis_as_layer+net_paulis_as_layer), pspec=pspec)[1])
            net_paulis = {qubits[i]: net_pauli_numbers[i] for i in range(n)}

            #depending on what the net pauli before the U gate is, might need to change parameters on the U gate
            # to commute the pauli through
            #recompile current layer to account for this and recompile with these paulis
            if n == 1:
                old_params_and_paulis = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]),
                                          float(current_layers[4].args[0]), net_paulis[qubits[i]],
                                          new_paulis[qubits[i]]) for i in range(n)]
            else:
                old_params_and_paulis = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]),
                                          float(current_layers[4][i].args[0]), net_paulis[qubits[i]],
                                          new_paulis[qubits[i]]) for i in range(n)]

            layer_new_params = [_comp.pauli_frame_randomize_unitary(*p) for p in old_params_and_paulis]
            #recompile any zrotation corrections from the previous Czr into the first zr of this layer. This correction
            # will be zero if there are no Czr gates (when it's clifford+zxzxz)
            theta1_layer = [_lbl.Label(zrotname, qubits[i],
                                       args=(str(layer_new_params[i][0] + correction_angles[qubits[i]]),))
                            for i in range(len(layer_new_params))]
            theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),))
                            for i in range(len(layer_new_params))]
            theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),))
                            for i in range(len(layer_new_params))]

            #add to mirror circuit
            mc.append([theta1_layer])
            mc.append([Xpi2layer])
            mc.append([theta2_layer])
            mc.append([Xpi2layer])
            mc.append([theta3_layer])

            d_ind += 5
            # reset the correction angles.
            correction_angles = {q: 0 for q in qubits}

        else:
            if circ_type == 'clifford+zxzxz':
                net_paulis_as_layer = [_lbl.Label(pauli_labels[net_paulis[qubits[i]]], qubits[i]) for i in range(n)]
                circ_sandwich = _cir.Circuit([layer, net_paulis_as_layer, layer], line_labels=circ.line_labels)
                net_paulis = {qubits[i]: pn
                              for i, pn in enumerate(_symp.find_pauli_number(_symp.symplectic_rep_of_clifford_circuit(
                                                     circ_sandwich, srep_dict=srep_dict)[1]))}
                mc.append(layer)
                #we need to account for how the net pauli changes when it gets passed through the clifford layers

            if circ_type == 'cz(theta)+zxzxz':
                quasi_inv_layer = []
                #recompile layer taking into acount paulis
                for g in layer:
                    if g.name == czrotname:
                        #get the qubits, figure out net pauli on those qubits
                        gate_qubits = g.qubits
                        net_paulis_for_gate = (net_paulis[gate_qubits[0]], net_paulis[gate_qubits[1]])
                        theta = float(g.args[0])
                        if ((net_paulis_for_gate[0] % 3 != 0 and net_paulis_for_gate[1] % 3 == 0)
                           or (net_paulis_for_gate[0] % 3 == 0 and net_paulis_for_gate[1] % 3 != 0)):
                            theta *= -1
                        quasi_inv_layer.append(_lbl.Label(czrotname, gate_qubits, args=(str(theta),)))
                        #for each X or Y, do a Zrotation by -theta on the other qubit after the 2Q gate.
                        for q in gate_qubits:
                            if net_paulis[q] == 1 or net_paulis[q] == 2:
                                for q2 in gate_qubits:
                                    if q2 != q:
                                        correction_angles[q2] += -1 * theta
                    else:
                        quasi_inv_layer.append(_lbl.Label(compute_gate_inverse(g)))
                #add to circuit
                mc.append([quasi_inv_layer])

            #increment position in circuit
            d_ind += 1

    #update the target pauli
    #pauli_layer = [_lbl.Label(pauli_labels[net_paulis[i]], qubits[i]) for i in range(len(qubits))]
    # The version from (THETA)
    pauli_layer = [_lbl.Label(pauli_labels[net_paulis[q]], q) for q in qubits]
    conjugation_circ = _cir.Circuit([pauli_layer], line_labels=circ.line_labels)
    telp_s, telp_p = _symp.symplectic_rep_of_clifford_circuit(conjugation_circ, srep_dict=srep_dict)

    # Calculate the bit string that this mirror circuit should output, from the final telescoped Pauli.
    target_bitstring = ''.join(['1' if p == 2 else '0' for p in telp_p[n:]])
    mirror_circuit = _cir.Circuit(mc, line_labels=circ.line_labels)

    return mirror_circuit, target_bitstring


# #generate mirror circuits with pauli frame randomization. no random +pi needed
# #as we construct the quasi-inverse, we generate random pauli layers, and compile them into the unitaries
# #we'll need to recompute the angles needed for the z rotations

# def create_nc_mirror_circuit(circ, pspec, circtype='Clifford+Gzr'):

#     assert(circtype == 'Clifford+Gzr' or circtype == 'Clifford')
#     n = circ.width
#     d = circ.depth
#     pauli_labels = ['I', 'X', 'Y', 'Z']
#     qubits = circ.line_labels
#     identity = _np.identity(2 * n, _np.int64)
#     zrotname = 'Gzr'
#     # qubit_labels = ['G{}'.format(i) for i in range(n)]

#     _, gate_inverse = pspec.compute_one_qubit_gate_relations()
#     gate_inverse.update(pspec.compute_multiqubit_inversion_relations())  # add multiQ inverses
#     #for gname in pspec.gate_names:
#     #    assert(gname in gate_inverse), \
#     #        "%s gate does not have an inverse in the gate-set! MRB is not possible!" % gname

#     quasi_inverse_circ = []

#     Xpi2layer = [_lbl.Label('Gc16', qubits[t]) for t in range(n)]
#     c = circ.copy(editable=True)

#    #build the inverse
#     d_ind = 0
#     while d_ind<d:
#         layer = circ.layer(d - d_ind - 1)
#         if layer[0].name == zrotname: #ask if it's a Zrot layer. It's necessary for the whole layer to have Zrot gates

#             current_layers = circ[d-d_ind-5:d-d_ind]
#             #recompile inverse of current layer
#             for i in range(n):
#                 #print((i, float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]),
#                 # float(current_layers[4][i].args[0])))
#                 if n==1:
#                     old_params = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]),
#                                    float(current_layers[4].args[0])) for i in range(n)]
#                 else:
#                     old_params = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]),
#                                    float(current_layers[4][i].args[0])) for i in range(n)]
#                 layer_new_params = [_comp.inv_recompile_unitary(*p) for p in old_params] #need to write this function
#                 theta1_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][0]),))
#                                 for i in range(len(layer_new_params))]
#                 theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),))
#                                 for i in range(len(layer_new_params))]
#                 theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),))
#                                 for i in range(len(layer_new_params))]

#             #add to mirror circuit
#             c.append_circuit_inplace(_cir.Circuit([theta3_layer]))
#             c.append_circuit_inplace(_cir.Circuit([Xpi2layer]))
#             c.append_circuit_inplace(_cir.Circuit([theta2_layer]))
#             c.append_circuit_inplace(_cir.Circuit([Xpi2layer]))
#             c.append_circuit_inplace(_cir.Circuit([theta1_layer]))
#             d_ind += 5

#         else:
#             inverse_layer = [_lbl.Label(gate_inverse[gate.name], gate.qubits) for gate in layer]
#             #create quasi-inverse. Right now, it's ust inverting every gate in the original layer, so a simple inverse
#             # Add the inverse layer that we've constructed to the end of the circuit
#             c.append_circuit_inplace(_cir.Circuit([inverse_layer]))
#             d_ind += 1
#         #now that we've built the simple mirror circuit, let's add pauli frame randomization
#     d_ind = 0
#     mc = []
#     net_paulis = [0 for q in qubits]
#     d = c.depth


#     srep_dict = _symp.compute_internal_gate_symplectic_representations(gllist=['I', 'X', 'Y', 'Z'])
#     # the `callable` part is a workaround to remove gates with args, defined by functions.
#     srep_dict.update(pspec.compute_clifford_symplectic_reps([gn for gn, u in pspec.gate_unitaries.items()
#                      if not callable(u)]))

#     while d_ind<d:
#         layer = c.layer(d_ind)
#         if layer[0].name == zrotname: #ask if it's a Zrot layer. It's necessary for the whole layer to have Zrot gates
#             #if the layer is 1Q unitaries, pauli randomize
#             current_layers = c[d_ind:d_ind+5]
#             #generate random pauli
#             new_paulis = [_np.random.randint(0, 4) for q in qubits]
#             new_paulis_as_layer = [_lbl.Label(pauli_labels[new_paulis[i]], qubits[i]) for i in range(n)]
#             #compute new net pauli based on previous pauli
#             net_paulis_as_layer = [_lbl.Label(pauli_labels[net_paulis[i]], qubits[i]) for i in range(n)]
#             net_paulis = _symp.find_pauli_number(_symp.symplectic_rep_of_clifford_circuit(_cir.Circuit(
#                                                  new_paulis_as_layer+net_paulis_as_layer), srep_dict=srep_dict)[1])
#             #depending on what the net pauli before the U gate is, might need to change parameters on the U gate to
#             # commute the pauli through
#             #recompile current layer to account for this and recompile with these paulis
#             if n == 1:
#                 old_params_and_paulis = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]),
#                                           float(current_layers[4].args[0]), net_paulis[i], new_paulis[i])
#                                          for i in range(n)]
#             else:
#                 old_params_and_paulis = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]),
#                                           float(current_layers[4][i].args[0]), net_paulis[i], new_paulis[i])
#                                           for i in range(n)]
#             layer_new_params = [_comp.pauli_frame_randomize_unitary(*p) for p in old_params_and_paulis]
#             theta1_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][0]),))
#                             for i in range(len(layer_new_params))]
#             theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),))
#                             for i in range(len(layer_new_params))]
#             theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),))
#                             for i in range(len(layer_new_params))]

#             #add to mirror circuit
#             mc.append([theta1_layer])
#             mc.append([Xpi2layer])
#             mc.append([theta2_layer])
#             mc.append([Xpi2layer])
#             mc.append([theta3_layer])

#             d_ind += 5

#         else:
#             net_paulis_as_layer = [_lbl.Label(pauli_labels[net_paulis[i]], qubits[i]) for i in range(n)]
#             net_paulis = _symp.find_pauli_number(_symp.symplectic_rep_of_clifford_circuit(_cir.Circuit([layer
#                                                  , net_paulis_as_layer, layer]), srep_dict=srep_dict)[1])
#             mc.append(layer)
#             #we need to account for how the net pauli changes when it gets passed through the clifford layers
#             d_ind += 1

#     #update the target pauli
#     pauli_layer = [_lbl.Label(pauli_labels[net_paulis[i]], qubits[i]) for i in range(len(qubits))]
#     conjugation_circ = _cir.Circuit([pauli_layer])
#     telp_s, telp_p = _symp.symplectic_rep_of_clifford_circuit(conjugation_circ, srep_dict=srep_dict)

#     # Calculate the bit string that this mirror circuit should output, from the final telescoped Pauli.
#     target_bitstring = ''.join(['1' if p == 2 else '0' for p in telp_p[n:]])

#     mirror_circuit = _cir.Circuit(mc)

#     return mirror_circuit, target_bitstring

# #
# def create_cz_mirror_circuit(circ, pspec, circtype='GCzr+Gzr', pauli_labels=None):
#     '''
#     Makes a mirror circuit with Pauli frame randomization from a forward circuits consisting of only Haar-random 1Q
#     unitary layers and CZRot layers
#     The 1Q unitaries must be decomposed as Zr-Xpi/2-Zr-Xpi/2-Zr
#     The CZRot layers must contain only Gc0/Gi and Gczr gates
#     '''

#     assert(circtype == 'GCzr+Gzr')
#     n = circ.width
#     d = circ.depth
#     if pauli_labels is None: pauli_labels = ['Gc0', 'Gc3', 'Gc6', 'Gc9']
#     qubits = circ.line_labels
#     zrotname = 'Gzr'
#     czrotname = 'Gczr'

#     Xpi2layer = [_lbl.Label('Gc16', q) for q in qubits]

#     #make an editable copy of the circuit to add the inverse on to
#     c = circ.copy(editable=True)
#    #build the inverse
#     d_ind = 0
#     while d_ind<d:
#         layer = circ.layer(d - d_ind - 1)
#         if layer[0].name == zrotname: #ask if it's a Zrot layer. It's necessary for the whole layer to have Zrot gates
#             #get the entire arbitrary 1q unitaries: Zrot-Xpi/2-Zrot-Xpi/2-Zrot
#             current_layers = circ[d-d_ind-5:d-d_ind]
#             #recompile inverse of current layer
#             for i in range(n):
#                 if n==1:
#                     old_params = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]),
#                                    float(current_layers[4].args[0])) for i in range(n)]
#                 else:
#                     old_params = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]),
#                                    float(current_layers[4][i].args[0])) for i in range(n)]
#                 layer_new_params = [_comp.inv_recompile_unitary(*p) for p in old_params] #generates parameters for
#                 # the inverse of this layer
#                 theta1_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][0]),))
#                                 for i in range(len(layer_new_params))]
#                 theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),))
#                                 for i in range(len(layer_new_params))]
#                 theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),))
#                                 for i in range(len(layer_new_params))]

#             #add to mirror circuit
#             c.append_circuit_inplace(_cir.Circuit([theta3_layer]))
#             c.append_circuit_inplace(_cir.Circuit([Xpi2layer]))
#             c.append_circuit_inplace(_cir.Circuit([theta2_layer]))
#             c.append_circuit_inplace(_cir.Circuit([Xpi2layer]))
#             c.append_circuit_inplace(_cir.Circuit([theta1_layer]))

#             d_ind += 5

#         if layer[0].name == czrotname or layer[0].name == 'Gc0':
#             invlayer = []
#             for g in layer:
#                 if g.name == czrotname:
#                     gate_qubits = g.qubits
#                     #get gate args
#                     theta = float(g.args[0])
#                     invlayer.append(_lbl.Label(czrotname, gate_qubits, args=(str(-1*theta),)))
#                 else:
#                     invlayer.append(g)
#             c.append_circuit_inplace(_cir.Circuit([invlayer]))
#             d_ind += 1

#     #now that we've built the simple mirror circuit, let's add pauli frame randomization
#     d_ind = 0
#     mc = []
#     net_paulis = {q:0 for q in qubits} #dictionary keeping track of the random paulis
#     d = c.depth
#     correction_angles = {q: 0 for q in qubits}

#     while d_ind<d:
#         layer = c.layer(d_ind)
#         if layer[0].name == zrotname:
#             #if the layer is 1Q unitaries, pauli randomize
#             current_layers = c[d_ind:d_ind+5]

#             #generate random pauli
#             new_paulis = {q: _np.random.randint(0, 4) for q in qubits}
#             new_paulis_as_layer = [_lbl.Label(pauli_labels[new_paulis[q]], q) for q in qubits]

#             #compute new net pauli based on previous pauli
#             net_paulis_as_layer = [_lbl.Label(pauli_labels[net_paulis[q]], q) for q in qubits]

#             net_pauli_numbers = _symp.find_pauli_number(_symp.symplectic_rep_of_clifford_circuit(_cir.Circuit(
#                                    new_paulis_as_layer+net_paulis_as_layer), pspec=pspec)[1])
#             net_paulis = {qubits[i]: net_pauli_numbers[i] for i in range(n)}

#             #depending on what the net pauli before the U gate is, might need to change parameters on the U gate to
#             # commute the pauli through
#             #recompile current layer to account for this and recompile with these paulis
#             if n == 1:
#                 old_params_and_paulis = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]),
#                    float(current_layers[4].args[0]), net_paulis[qubits[i]], new_paulis[qubits[i]]) for i in range(n)]
#             else:
#                 #problem:ordering of qubits in the layer isn't always consistent
#                 old_params_and_paulis = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]),
#                    float(current_layers[4][i].args[0]), net_paulis[qubits[i]],
#                    new_paulis[qubits[i]]) for i in range(n)]

#             layer_new_params = [_comp.pauli_frame_randomize_unitary(*p) for p in old_params_and_paulis] #need to write
#             # this function
#             #recompile any zrotation corrections from the previous Czr into the first zr of this layer
#             theta1_layer = [_lbl.Label(zrotname, qubits[i],
#               args=(str(layer_new_params[i][0]+correction_angles[qubits[i]]),)) for i in range(len(layer_new_params))]
#             theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),))
#                             for i in range(len(layer_new_params))]
#             theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),))
#                             for i in range(len(layer_new_params))]
#             #add to mirror circuit
#             mc.append([theta1_layer])
#             mc.append([Xpi2layer])
#             mc.append([theta2_layer])
#             mc.append([Xpi2layer])
#             mc.append([theta3_layer])

#             correction_angles = {q: 0 for q in qubits}
#             d_ind += 5

#         if layer[0].name == czrotname or layer[0].name == 'Gc0':
#             quasi_inv_layer = []

#             #recompile layer taking into acount paulis
#             for g in layer:
#                 if g.name == czrotname:
#                     #get the qubits, figure out net pauli on those qubits
#                     gate_qubits = g.qubits
#                     net_paulis_for_gate = (net_paulis[gate_qubits[0]], net_paulis[gate_qubits[1]])
#                     theta = float(g.args[0])
#                     if ((net_paulis_for_gate[0] % 3 != 0 and net_paulis_for_gate[1] % 3 == 0)
#                        or (net_paulis_for_gate[0] % 3 == 0 and net_paulis_for_gate[1] % 3 != 0)):
#                         theta *= -1
#                     quasi_inv_layer.append(_lbl.Label(czrotname, gate_qubits, args=(str(theta),)))
#                     #for each X or Y, do a Zrotation by -theta on the other qubit after the 2Q gate.
#                     for q in gate_qubits:
#                         if net_paulis[q] == 1 or net_paulis[q] == 2:
#                             for q2 in gate_qubits:
#                                 if q2 != q:
#                                     correction_angles[q2] += -1*theta
#                 else:
#                     gate_qubit = g.qubits
#                     quasi_inv_layer.append(_lbl.Label('Gc0', gate_qubit))
#             #add to circuit
#             mc.append([quasi_inv_layer])

#             #increment position in circuit
#             d_ind += 1

#     #update the target pauli
#     pauli_layer = [_lbl.Label(pauli_labels[net_paulis[q]], q) for q in qubits]
#     conjugation_circ = _cir.Circuit([pauli_layer]) #conjugation_circ = _cir.Circuit([random_stateprep_layer,
#                             pauli_layer, random_meas_layer])
#     telp_s, telp_p = _symp.symplectic_rep_of_clifford_circuit(conjugation_circ, pspec=pspec)

#     # Calculate the bit string that this mirror circuit should output, from the final Pauli.
#     target_bitstring = ''.join(['1' if p == 2 else '0' for p in telp_p[n:]])

#     mirror_circuit = _cir.Circuit(mc)

#     return mirror_circuit, target_bitstring

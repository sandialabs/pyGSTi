"""
circuit mirroring functions.
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
import random as _random

from pygsti.circuits import circuit as _cir
from pygsti.baseobjs import label as _lbl
from pygsti.tools import symplectic as _symp


def _pvec_to_pauli_layer(pvec, pauli_labels, qubit_labels):
    """
    TODO
    """
    n = len(pvec) // 2
    v = (pvec[0:n] // 2) + 2 * (pvec[n:] // 2)
    #  Rearrange to I, Z, X, Y (because [0,0]=I, [2,0]=Z, [0,2]=X,and [2,2]=Y).
    rearranged_pauli_labels = [pauli_labels[0], pauli_labels[3], pauli_labels[1], pauli_labels[2]]
    paulis = [rearranged_pauli_labels[i] for i in v]
    return [(pl, q) for pl, q in zip(paulis, qubit_labels)]

def _mod_2pi(theta):
    while (theta > _np.pi or theta <= -1 * _np.pi):
        if theta > _np.pi:
            theta = theta - 2 * _np.pi
        elif theta <= -1 * _np.pi:
            theta = theta + 2 * _np.pi
    return theta


def create_mirror_circuit(circ, pspec, circtype='Clifford+Gzr', pauli_labels=None, pluspi_prob=0.):
    """
    *****************************************************************
    Function currently has the following limitations that need fixing:

       - A layer contains only Clifford or Gzr gates on ALL the qubits.
       - all of the Clifford gates are self inverse
       - The qubits are labelled "Q0" through "Qn-1" -- THIS SHOULD NOW BE FIXED!
       - Pauli's are labelled by "Gi", "Gxpi", "Gypi" and "Gzpi".
       - There's no option for randomized prep/meas
       - There's no option for randomly adding +/-pi to the Z rotation angles.
       - There's no option for adding "barriers"
       - There's no test that the 'Gzr' gate has the "correct" convention for a rotation angle
         (a rotation by pi must be a Z gate) or that it's a rotation around Z.
    *****************************************************************
    """
    assert(circtype == 'Clifford+Gzr' or circtype == 'Clifford')
    n = circ.width
    d = circ.depth
    if pauli_labels is None: pauli_labels = ['Gi', 'Gxpi', 'Gypi', 'Gzpi']
    qubits = circ.line_labels
    identity = _np.identity(2 * n, int)
    zrotname = 'Gzr'
    # qubit_labels = ['G{}'.format(i) for i in range(n)]

    quasi_inverse_circ = []
    central_pauli_circ = _cir.Circuit([[_lbl.Label(pauli_labels[_np.random.randint(0, 4)], q) for q in qubits]])
    #telescoping_pauli = central_pauli_layer.copy()
    # The telescoping Pauli in the symplectic rep.
    telp_s, telp_p = _symp.symplectic_rep_of_clifford_circuit(central_pauli_circ, pspec=pspec)
    assert(_np.sum(_np.abs(telp_s - identity)) <= 1e-8)  # Check that it's a Pauli.

    for d_ind in range(d):
        layer = circ.layer(d - d_ind - 1)
        if layer[0].name == zrotname:
            quasi_inverse_layer = []
            for gate in layer:

                q_int = qubits.index(gate.qubits[0])
                angle = float(gate.args[0])

                if telp_p[n + q_int] == 0: rotation_sign = -1.  # If the Pauli is Z or I.
                else: rotation_sign = +1  # If the Pauli is X or Y.

                # Sets the quasi inversion angle to + or - the original angle, depending on the Paul
                quasi_inverse_angle = rotation_sign * angle
                # Decides whether to add with to add +/- pi to the rotation angle.
                if _np.random.binomial(1, pluspi_prob) == 1:
                    quasi_inverse_angle += _np.pi * (-1)**_np.random.binomial(1, 0.5)
                    quasi_inverse_angle = _mod_2pi(quasi_inverse_angle)
                    # Updates the telescoping Pauli (in the symplectic rep_, to include this added pi-rotation,
                    # as we need to include it as we keep collapsing the circuit down.
                    telp_p[q_int] = (telp_p[q_int] + 2) % 4
                # Constructs the quasi-inverse gate.
                quasi_inverse_gate = _lbl.Label(zrotname, gate.qubits, args=(str(quasi_inverse_angle),))
                quasi_inverse_layer.append(quasi_inverse_gate)

            # We don't have to update the telescoping Pauli as it's unchanged, but when we update
            # this it'll need to change.
            #telp_p = telp_p

        else:
            quasi_inverse_layer = [_lbl.Label(pspec.gate_inverse[gate.name], gate.qubits) for gate in layer]
            telp_layer = _pvec_to_pauli_layer(telp_p, pauli_labels, qubits)
            conjugation_circ = _cir.Circuit([layer, telp_layer, quasi_inverse_layer])
            # We calculate what the new telescoping Pauli is, in the symplectic rep.
            telp_s, telp_p = _symp.symplectic_rep_of_clifford_circuit(conjugation_circ, pspec=pspec)

        # Check that the layer -- pauli -- quasi-inverse circuit implements a Pauli.
        assert(_np.sum(_np.abs(telp_s - identity)) <= 1e-10)
        # Add the quasi inverse layer that we've constructed to the end of the quasi inverse circuit.
        quasi_inverse_circ.append(quasi_inverse_layer)

    # now that we've completed the quasi inverse circuit we convert it to a Circuit object
    quasi_inverse_circ = _cir.Circuit(quasi_inverse_circ)

    # Calculate the bit string that this mirror circuit should output, from the final telescoped Pauli.
    target_bitstring = ''.join(['1' if p == 2 else '0' for p in telp_p[n:]])
    mirror_circuit = circ + central_pauli_circ + quasi_inverse_circ

    return mirror_circuit, target_bitstring

###

def sample_2q_gate_layer_by_edgegrab(pspec, qubit_labels=None, mean_two_q_gates=1, modelname='clifford'):
    """
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

    Returns
    -------
    <TODO typ>
    """
    assert(modelname == 'clifford'), "This function currently assumes sampling from a Clifford model!"
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list

    # Prep the sampling variables.
    sampled_layer = []
    edgelist = pspec.qubitgraph.edges()
    edgelist = [e for e in edgelist if all([q in qubits for q in e])]
    selectededges = []

    # Go through until all qubits have been assigned a gate.
    while len(edgelist) > 0:

        edge = edgelist[_np.random.randint(0, len(edgelist))]
        selectededges.append(edge)
        # Delete all edges containing these qubits.
        edgelist = [e for e in edgelist if not any([q in e for q in edge])]

    num2Qgates = len(selectededges)
    assert(num2Qgates >= mean_two_q_gates), "Device has insufficient connectivity!"

    if mean_two_q_gates > 0:
        twoQprob = mean_two_q_gates / num2Qgates
    else:
        twoQprob = 0

    unusedqubits = _copy.copy(qubits)
    for edge in selectededges:
        if bool(_np.random.binomial(1, twoQprob)):

            # The two-qubit gates on that edge.
            possibleops = pspec.clifford_ops_on_qubits[edge]
            sampled_layer.append(possibleops[_np.random.randint(0, len(possibleops))])
            for q in edge:
                del unusedqubits[unusedqubits.index(q)]
        
    for q in unusedqubits:
        sampled_layer.append(_lbl.Label('Gc0', (q)))

    return sampled_layer

def generate_random_1q_unitary():
    psi, chi = 2*_np.pi*_np.random.rand(2)
    psi = psi - _np.pi
    chi = chi - _np.pi
    phi = _np.arcsin(_np.sqrt(_np.random.rand(1)))[0]
    #U = _np.exp(1j*alpha)*_np.array([[_np.exp(1j*psi)*_np.cos(phi), _np.exp(1j*chi)*_np.sin(phi)],[-1*_np.exp(-1j*chi)*_np.sin(phi), _np.exp(-1j*psi)*_np.cos(phi)]])
    #this needs to be decomposed in the form Zrot(theta3) Xpi/2 Zrot(theta2) Xpi/2 Zrot(theta1)
    theta1 = _mod_2pi(psi - chi + _np.pi)
    theta2 = _mod_2pi(_np.pi - 2*phi)
    theta3 = _mod_2pi(psi + chi)
    return (theta1, theta2, theta3)

#generate layer of random unitaries and make a series of circuit layers with the compiled versions of these
def sample_1q_unitary_layer(pspec, qubit_labels=None):
    if qubit_labels is not None:
        n = len(qubit_labels)
        qubits = qubit_labels[:]  # copy this list
    else:
        n = pspec.number_of_qubits
        qubits = pspec.qubit_labels[:]  # copy this list
    zrotname = 'Gzr'
    #generate random rotation angles from 0 to 2pi
    
    Xpi2layer = _cir.Circuit(layer_labels=[_lbl.Label('Gc16', qubits[t]) for t in range(n)], line_labels=qubits, editable=True).parallelize()
    theta = []
    theta1 = []
    theta2 = []
    theta3 = []
    for q in qubits:
        theta = generate_random_1q_unitary()
        #theta_quasi_inv = pauli_randomize_unitary(*theta)
        theta1.append(theta[0])
        theta2.append(theta[1])
        theta3.append(theta[2])
        
    circuit = _cir.Circuit(layer_labels=[_lbl.Label(zrotname, qubits[t], args=(str(theta1[t]),)) for t in range(n)], line_labels=qubits, editable=True) #constructs the gate labels
    circuit.append_circuit_inplace(Xpi2layer)
    circuit.append_circuit_inplace(_cir.Circuit(layer_labels=[_lbl.Label(zrotname, qubits[t], args=(str(theta2[t]),)) for t in range(n)], line_labels=qubits))
    circuit.append_circuit_inplace(Xpi2layer)
    circuit.append_circuit_inplace(_cir.Circuit(layer_labels=[_lbl.Label(zrotname, qubits[t], args=(str(theta3[t]),)) for t in range(n)], line_labels=qubits))
    circuit = circuit.parallelize()
    return circuit

def create_random_1q_nc_circuit(pspec, length, qubit_labels=None, mean_two_q_gates=1):
    circuit = _cir.Circuit(layer_labels=[], line_labels=qubit_labels, editable=True)
    for a in range(length):
        #generate 2q gate layer
        new_layer = sample_1q_unitary_layer(pspec, qubit_labels)
        circuit.append_circuit_inplace(new_layer)
            
        #generate random 1q unitary layer
        sampled_layer = sample_2q_gate_layer_by_edgegrab(pspec, qubit_labels=qubit_labels, mean_two_q_gates=mean_two_q_gates)
        if sampled_layer == []: new_layer = _cir.Circuit(layer_labels=[], line_labels=qubit_labels, editable=True)
        else: new_layer = _cir.Circuit([sampled_layer])
        #append new layer to circuit
        circuit.append_circuit_inplace(new_layer)
    new_layer = sample_1q_unitary_layer(pspec, qubit_labels)
    circuit.append_circuit_inplace(new_layer)
    circuit.done_editing()
    return circuit

def pvec_to_pauli_number(pvec):
    """
    TODO
    """
    n = len(pvec) // 2
    v = (pvec[0:n] // 2) + 2 * (pvec[n:] // 2)
    #  Rearrange to I, Z, X, Y (because [0,0]=I, [2,0]=Z, [0,2]=X,and [2,2]=Y).
    rearranged_pauli_labels = [0, 3, 1, 2]
    paulis = [rearranged_pauli_labels[i] for i in v]
    return paulis

def pfr_v2_unitary(theta1, theta2, theta3, net_pauli, recomp_pauli):
    #takes the z rotation angles for the compiled version of a random unitary and finds the angles for the compiled version of the pauli frame randomized unitary
    #redefine the values so that when the net pauli commutes through, we get the original parameters
    if net_pauli == 1 or net_pauli == 3:
        theta2 *= -1
    if net_pauli == 1 or net_pauli == 2:
        theta3 *= -1
        theta1 *= -1
        
    #change angles to recompile the new pauli into the gate
    if recomp_pauli == 1 or recomp_pauli == 2: #if x or y
        theta1 = -theta1 + _np.pi
        theta2 = theta2 + _np.pi
    if recomp_pauli ==2 or recomp_pauli ==3: #if y or z
        theta1 = theta1 + _np.pi
    
    #make everything between -pi and pi.
    theta1 = _mod_2pi(theta1)
    theta2 = _mod_2pi(theta2)
    theta3 = _mod_2pi(theta3)
    
    return (theta1, theta2, theta3)
    
def inv_recompile_unitary(theta1, theta2, theta3):
    #makes a compiled version of the inverse of a compiled general unitary
    #negate angles for inverse based on central pauli, account for recompiling the X(-pi/2) into X(pi/2)
    theta1 = _mod_2pi(_np.pi -theta1)
    theta2 = _mod_2pi(-theta2)
    theta3 = _mod_2pi(-theta3+_np.pi)
    
    return (theta1, theta2, theta3)

#generate mirror circuits with pauli frame randomization. no random +pi needed
#as we construct the quasi-inverse, we generate random pauli layers, and compile them into the unitaries
#we'll need to recompute the angles needed for the z rotations

def create_nc_mirror_circuit(circ, pspec, circtype='Clifford+Gzr', pauli_labels=None):

    assert(circtype == 'Clifford+Gzr' or circtype == 'Clifford')
    n = circ.width
    d = circ.depth
    if pauli_labels is None: pauli_labels = ['Gc0', 'Gc3', 'Gc6', 'Gc9']
    qubits = circ.line_labels
    identity = _np.identity(2 * n, int)
    zrotname = 'Gzr'
    # qubit_labels = ['G{}'.format(i) for i in range(n)]

    quasi_inverse_circ = []
    
    Xpi2layer = [_lbl.Label('Gc16', qubits[t]) for t in range(n)]
    c = circ.copy(editable=True)

   #build the inverse
    d_ind = 0
    while d_ind<d:
        layer = circ.layer(d - d_ind - 1)
        if layer[0].name == zrotname: #ask if it's a Zrot layer. It's necessary for the whole layer to have Zrot gates
        
            current_layers = circ[d-d_ind-5:d-d_ind]
            #recompile inverse of current layer 
            for i in range(n): 
                #print((i, float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]), float(current_layers[4][i].args[0])))
                if n==1:
                    old_params = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]), float(current_layers[4].args[0])) for i in range(n)]
                else:
                    old_params = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]), float(current_layers[4][i].args[0])) for i in range(n)]
                layer_new_params = [inv_recompile_unitary(*p) for p in old_params] #need to write this function
                theta1_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][0]),)) for i in range(len(layer_new_params))]
                theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),)) for i in range(len(layer_new_params))]
                theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),)) for i in range(len(layer_new_params))]
    
            #add to mirror circuit
            c.append_circuit_inplace(_cir.Circuit([theta3_layer]))
            c.append_circuit_inplace(_cir.Circuit([Xpi2layer]))
            c.append_circuit_inplace(_cir.Circuit([theta2_layer]))
            c.append_circuit_inplace(_cir.Circuit([Xpi2layer]))
            c.append_circuit_inplace(_cir.Circuit([theta1_layer]))
            
            d_ind += 5

        else:
            inverse_layer = [_lbl.Label(pspec.gate_inverse[gate.name], gate.qubits) for gate in layer] #create quasi-inverse. Right now, it's ust inverting every gate in the original layer, so a simple inverse
            # Add the inverse layer that we've constructed to the end of the circuit
            c.append_circuit_inplace(_cir.Circuit([inverse_layer]))
            d_ind += 1
            
        #now that we've built the simple mirror circuit, let's add pauli frame randomization
    d_ind = 0
    mc = []
    net_paulis = [0 for q in qubits]
    d = c.depth
    
    while d_ind<d:
        layer = c.layer(d_ind)
        if layer[0].name == zrotname: #ask if it's a Zrot layer. It's necessary for the whole layer to have Zrot gates
            #if the layer is 1Q unitaries, pauli randomize
            current_layers = c[d_ind:d_ind+5]
            
            #generate random pauli
            new_paulis = [_np.random.randint(0, 4) for q in qubits]
            new_paulis_as_layer = [_lbl.Label(pauli_labels[new_paulis[i]], qubits[i]) for i in range(n)]
                
            #compute new net pauli based on previous pauli
            net_paulis_as_layer = [_lbl.Label(pauli_labels[net_paulis[i]], qubits[i]) for i in range(n)]
            net_paulis = pvec_to_pauli_number(_symp.symplectic_rep_of_clifford_circuit(_cir.Circuit(new_paulis_as_layer+net_paulis_as_layer), pspec=pspec)[1])
            #depending on what the net pauli before the U gate is, might need to change parameters on the U gate to commute the pauli through
            #recompile current layer to account for this and recompile with these paulis
            if n == 1:
                old_params_and_paulis = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]), float(current_layers[4].args[0]), net_paulis[i], new_paulis[i]) for i in range(n)]
            else:
                old_params_and_paulis = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]), float(current_layers[4][i].args[0]), net_paulis[i], new_paulis[i]) for i in range(n)]
            layer_new_params = [pfr_v2_unitary(*p) for p in old_params_and_paulis] #need to write this function
            theta1_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][0]),)) for i in range(len(layer_new_params))]
            theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),)) for i in range(len(layer_new_params))]
            theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),)) for i in range(len(layer_new_params))]
            
            #add to mirror circuit
            mc.append([theta1_layer])
            mc.append([Xpi2layer])
            mc.append([theta2_layer])
            mc.append([Xpi2layer])
            mc.append([theta3_layer])
            
            
            d_ind += 5
    
        else:
            net_paulis_as_layer = [_lbl.Label(pauli_labels[net_paulis[i]], qubits[i]) for i in range(n)]
            net_paulis = pvec_to_pauli_number(_symp.symplectic_rep_of_clifford_circuit(_cir.Circuit([layer, net_paulis_as_layer, layer]), pspec=pspec)[1])
            
            mc.append(layer)
            #we need to account for how the net pauli changes when it gets passed through the clifford layers
            d_ind += 1
            

    #update the target pauli
    pauli_layer = [_lbl.Label(pauli_labels[net_paulis[i]], qubits[i]) for i in range(len(qubits))]
    conjugation_circ = _cir.Circuit([pauli_layer])
    telp_s, telp_p = _symp.symplectic_rep_of_clifford_circuit(conjugation_circ, pspec=pspec)
    
    # Calculate the bit string that this mirror circuit should output, from the final telescoped Pauli.
    target_bitstring = ''.join(['1' if p == 2 else '0' for p in telp_p[n:]])

    mirror_circuit = _cir.Circuit(mc)
                
    return mirror_circuit, target_bitstring

def sample_cz_layer_by_edgegrab(pspec, qubit_labels=None, mean_two_q_gates=1, angles = [_np.pi/2, -1*_np.pi/2]):
    czrotname = 'Gczr'
    
    if qubit_labels is None:
        qubits = list(pspec.qubit_labels[:])  # copy this list
    else:
        assert(isinstance(qubit_labels, (list, tuple))), "SubsetQs must be a list or a tuple!"
        qubits = list(qubit_labels[:])  # copy this list

    # Prep the sampling variables.
    sampled_layer = []
    edgelist = pspec.availability[czrotname]
    edgelist = [e for e in edgelist if all([q in qubits for q in e])]
    selectededges = []

    # Go through until all qubits have been assigned a gate.
    while len(edgelist) > 0:

        edge = edgelist[_np.random.randint(0, len(edgelist))]
        selectededges.append(edge)
        # Delete all edges containing these qubits.
        edgelist = [e for e in edgelist if not any([q in e for q in edge])]

    num2Qgates = len(selectededges)

    if mean_two_q_gates > 0:
        twoQprob = mean_two_q_gates / num2Qgates
    else:
        twoQprob = 0
    
    unusedqubits = _copy.copy(qubits)
    #put czrot gates on the selected edges
    for edge in selectededges:
        if bool(_np.random.binomial(1, twoQprob)):
            theta = _random.choice(angles)
            sampled_layer.append(_lbl.Label(czrotname, edge, args=(str(theta),)))
            for q in edge:
                del unusedqubits[unusedqubits.index(q)]

    for q in unusedqubits:
        sampled_layer.append(_lbl.Label('Gc0', (q)))
        
    return sampled_layer


def generate_cz_forward_circ(pspec, length, qubit_labels=None, mean_two_q_gates=1, angles=[_np.pi/2, -1*_np.pi/2]):
    '''
    Generates a forward circuits with benchmark depth d for non-clifford mirror randomized benchmarking. 
    The circuits alternate Haar-random 1q unitaries and layers of Gczr gates
    '''
    #choose length to be the number of (2Q layer, 1Q layer) blocks
    circuit = _cir.Circuit(layer_labels=[], line_labels=qubit_labels, editable=True)
    for a in range(length):
        #generate random 1q unitary layer
        new_layer = sample_1q_unitary_layer(pspec, qubit_labels)
        #append new layer to circuit
        circuit.append_circuit_inplace(new_layer)
        #generate 2q gate layer
        sampled_layer = sample_cz_layer_by_edgegrab(pspec, qubit_labels=qubit_labels, mean_two_q_gates=mean_two_q_gates, angles=angles)
        if sampled_layer == []: new_layer = _cir.Circuit(layer_labels=[], line_labels=qubit_labels, editable=True)
        else: new_layer = _cir.Circuit([sampled_layer])
        #append new layer to circuit
        circuit.append_circuit_inplace(new_layer)
    #add one more layer of Haar-random 1Q unitaries
    new_layer = sample_1q_unitary_layer(pspec, qubit_labels)
    circuit.append_circuit_inplace(new_layer)

    circuit.done_editing()
    return circuit

#
def create_cz_mirror_circuit(circ, pspec, circtype='GCzr+Gzr', pauli_labels=None):
    '''
    Makes a mirror circuit with Pauli frame randomization from a forward circuits consisting of only Haar-random 1Q unitary layers and CZRot layers
    The 1Q unitaries must be decomposed as Zr-Xpi/2-Zr-Xpi/2-Zr
    The CZRot layers must contain only Gc0/Gi and Gczr gates
    '''

    assert(circtype == 'GCzr+Gzr')
    n = circ.width
    d = circ.depth
    if pauli_labels is None: pauli_labels = ['Gc0', 'Gc3', 'Gc6', 'Gc9']
    qubits = circ.line_labels
    zrotname = 'Gzr'
    czrotname = 'Gczr'
    
    Xpi2layer = [_lbl.Label('Gc16', q) for q in qubits]
    
    #make an editable copy of the circuit to add the inverse on to
    c = circ.copy(editable=True)
   #build the inverse
    d_ind = 0
    while d_ind<d:
        layer = circ.layer(d - d_ind - 1)
        if layer[0].name == zrotname: #ask if it's a Zrot layer. It's necessary for the whole layer to have Zrot gates
            #get the entire arbitrary 1q unitaries: Zrot-Xpi/2-Zrot-Xpi/2-Zrot
            current_layers = circ[d-d_ind-5:d-d_ind]
            #recompile inverse of current layer 
            for i in range(n): 
                if n==1:
                    old_params = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]), float(current_layers[4].args[0])) for i in range(n)]
                else:
                    old_params = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]), float(current_layers[4][i].args[0])) for i in range(n)]
                layer_new_params = [inv_recompile_unitary(*p) for p in old_params] #generates parameters for the inverse of this layer
                theta1_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][0]),)) for i in range(len(layer_new_params))]
                theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),)) for i in range(len(layer_new_params))]
                theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),)) for i in range(len(layer_new_params))]
    
            #add to mirror circuit
            c.append_circuit_inplace(_cir.Circuit([theta3_layer]))
            c.append_circuit_inplace(_cir.Circuit([Xpi2layer]))
            c.append_circuit_inplace(_cir.Circuit([theta2_layer]))
            c.append_circuit_inplace(_cir.Circuit([Xpi2layer]))
            c.append_circuit_inplace(_cir.Circuit([theta1_layer]))
            
            d_ind += 5
            
        if layer[0].name == czrotname or layer[0].name == 'Gc0': 
            invlayer = []
            for g in layer:
                if g.name == czrotname: 
                    gate_qubits = g.qubits
                    #get gate args
                    theta = float(g.args[0])
                    invlayer.append(_lbl.Label(czrotname, gate_qubits, args=(str(-1*theta),)))
                else:
                    invlayer.append(g)
            c.append_circuit_inplace(_cir.Circuit([invlayer]))
            d_ind += 1
    
    #now that we've built the simple mirror circuit, let's add pauli frame randomization
    d_ind = 0
    mc = []
    net_paulis = {q:0 for q in qubits} #dictionary keeping track of the random paulis
    d = c.depth
    correction_angles = {q: 0 for q in qubits}

    while d_ind<d:
        layer = c.layer(d_ind)
        if layer[0].name == zrotname: 
            #if the layer is 1Q unitaries, pauli randomize
            current_layers = c[d_ind:d_ind+5]
            
            #generate random pauli
            new_paulis = {q: _np.random.randint(0, 4) for q in qubits}
            new_paulis_as_layer = [_lbl.Label(pauli_labels[new_paulis[q]], q) for q in qubits]
                
            #compute new net pauli based on previous pauli
            net_paulis_as_layer = [_lbl.Label(pauli_labels[net_paulis[q]], q) for q in qubits]

            net_pauli_numbers = pvec_to_pauli_number(_symp.symplectic_rep_of_clifford_circuit(_cir.Circuit(new_paulis_as_layer+net_paulis_as_layer), pspec=pspec)[1])
            net_paulis = {qubits[i]: net_pauli_numbers[i] for i in range(n)}

            #depending on what the net pauli before the U gate is, might need to change parameters on the U gate to commute the pauli through
            #recompile current layer to account for this and recompile with these paulis
            if n == 1:
                old_params_and_paulis = [(float(current_layers[0].args[0]), float(current_layers[2].args[0]), float(current_layers[4].args[0]), net_paulis[qubits[i]], new_paulis[qubits[i]]) for i in range(n)]
            else:
                #problem:ordering of qubits in the layer isn't always consistent
                old_params_and_paulis = [(float(current_layers[0][i].args[0]), float(current_layers[2][i].args[0]), float(current_layers[4][i].args[0]), net_paulis[qubits[i]], new_paulis[qubits[i]]) for i in range(n)]

            layer_new_params = [pfr_v2_unitary(*p) for p in old_params_and_paulis] #need to write this function
            #recompile any zrotation corrections from the previous Czr into the first zr of this layer
            theta1_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][0]+correction_angles[qubits[i]]),)) for i in range(len(layer_new_params))]
            theta2_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][1]),)) for i in range(len(layer_new_params))]
            theta3_layer = [_lbl.Label(zrotname, qubits[i], args=(str(layer_new_params[i][2]),)) for i in range(len(layer_new_params))]
            
            #add to mirror circuit
            mc.append([theta1_layer])
            mc.append([Xpi2layer])
            mc.append([theta2_layer])
            mc.append([Xpi2layer])
            mc.append([theta3_layer])
            
            correction_angles = {q: 0 for q in qubits}
            d_ind += 5
            
        if layer[0].name == czrotname or layer[0].name == 'Gc0': 
            quasi_inv_layer = []

            #recompile layer taking into acount paulis
            for g in layer: 
                if g.name == czrotname:
                    #get the qubits, figure out net pauli on those qubits
                    gate_qubits = g.qubits
                    net_paulis_for_gate = (net_paulis[gate_qubits[0]], net_paulis[gate_qubits[1]])
                    theta = float(g.args[0])
                    if (net_paulis_for_gate[0] % 3 != 0 and net_paulis_for_gate[1] % 3 == 0) or (net_paulis_for_gate[0] % 3 == 0 and net_paulis_for_gate[1] % 3 != 0):
                        theta *= -1
                    quasi_inv_layer.append(_lbl.Label(czrotname, gate_qubits, args=(str(theta),)))
                    #for each X or Y, do a Zrotation by -theta on the other qubit after the 2Q gate.
                    for q in gate_qubits:
                        if net_paulis[q] == 1 or net_paulis[q] == 2: 
                            for q2 in gate_qubits: 
                                if q2 != q:
                                    correction_angles[q2] += -1*theta
                else:
                    gate_qubit = g.qubits
                    quasi_inv_layer.append(_lbl.Label('Gc0', gate_qubit))
            #add to circuit
            mc.append([quasi_inv_layer])
            
            #increment position in circuit    
            d_ind += 1
            
    #update the target pauli
    pauli_layer = [_lbl.Label(pauli_labels[net_paulis[q]], q) for q in qubits]
    conjugation_circ = _cir.Circuit([pauli_layer]) #conjugation_circ = _cir.Circuit([random_stateprep_layer, pauli_layer, random_meas_layer])
    telp_s, telp_p = _symp.symplectic_rep_of_clifford_circuit(conjugation_circ, pspec=pspec)
    
    # Calculate the bit string that this mirror circuit should output, from the final Pauli.
    target_bitstring = ''.join(['1' if p == 2 else '0' for p in telp_p[n:]])

    mirror_circuit = _cir.Circuit(mc)

    return mirror_circuit, target_bitstring

def generate_czr_experiment_design(num_circs, depths ,qubit_set, pspec ,twoQgate_density=0, angles=[_np.pi/2, -1*_np.pi/2]):
    '''
    generates a randomized mirror benchmarking experiment design for circuits with controlled Z rotations for the two-qubit gates
    '''
    depths = depths
    qs = qubit_set
    k = num_circs #number of circuits to create
    pspec = pspec


    # The exact details of the random circuit sampling distribution.
    sampler = 'edgegrab'
    

    #make the experiment design
    mcs_by_depth = {d:[] for d in depths}
    for d in depths:
        #generate k circuits
        twoQmean =  len(qs) * twoQgate_density / 2
        circs = [generate_cz_forward_circ(pspec, int(d/2), qubit_labels=qs, mean_two_q_gates=twoQmean, angles=angles) for _ in range(k)]
        mcs = [(a,[b]) for a,b in [create_cz_mirror_circuit(c, pspec) for c in circs]]
        mcs_by_depth[d].extend(mcs)
    edesign = _protocols.MirrorRBDesign.from_existing_circuits(mcs_by_depth)
    return edesign


def generate_1q_nc_experiment_design(num_circs, depths ,qubit_set, pspec ,twoQgate_density=0):
    depths = depths
    qs = qubit_set
    k = num_circs #number of circuits to create
    pspec = pspec
    n = len(qs)


    # The exact details of the random circuit sampling distribution.
    sampler = 'edgegrab'
    twoQmean = n * twoQgate_density / 2

    #make the experiment design
    mcs_by_depth = {d:[] for d in depths}
    for d in depths:
        #generate k circuits
        circs = [create_random_1q_nc_circuit(pspec, int(d/2), qubit_labels=qs, mean_two_q_gates=twoQmean) for _ in range(k)]
        mcs = [(a,[b]) for a,b in [create_nc_mirror_circuit(c, pspec) for c in circs]]
        mcs_by_depth[d].extend(mcs)
    edesign = _protocols.MirrorRBDesign.from_existing_circuits(mcs_by_depth)
    return edesign
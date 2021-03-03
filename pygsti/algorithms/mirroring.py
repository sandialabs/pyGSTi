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

from ..objects import circuit as _cir
from ..objects import label as _lbl
from ..tools import symplectic as _symp
#from .. import construction as _cnst
from .. import objects as _objs
#from .. import io as _io
from .. import tools as _tools
#from ..tools import group as _rbobjs

import numpy as _np
import copy as _copy
#import itertools as _itertools


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
    if theta > _np.pi:
        return theta - 2 * _np.pi
    elif theta <= -1 * _np.pi:
        return theta + 2 * _np.pi
    else:
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
    zrotname = 'Gz'
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

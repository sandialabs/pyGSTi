import numpy as _np

from pygsti.circuits.circuit import Circuit as _Circuit

import utils #TODO: integrate all the needed functionality from utils into pygsti

#TODO: OOP-ify this code?

def pauli_randomize_circuit(circ, return_target_pauli=False, rand_state=None):
    if rand_state is None:
        rand_state = _np.random.RandomState()

    d = circ.depth
    n = circ.width
    p = _np.zeros(2*n, int)
    
    qubits = circ.line_labels

    layers = []

    for i in range(d):

        layer = circ.layer_label(i).components

        if layer[0].name in ['Gi', 'Gdelay']: #making this explicit for the sake of clarity
            # should we be tacking a Pauli on at the end (as though this were RC of u3(0,0,0) layers?)
            layers.append(layer)

        elif len(layer) == 0 or layer[0].name == 'Gu3':
            q = 2 * rand_state.randint(0, 2, 2*n)
            # update_u3_parameters now twirls/adds label for empty qubits, so don't prepad for speed
            #padded_layer = pad_layer(layer, qubits)
            rc_layer = utils.update_u3_parameters(layer, p, q, qubits)
            layers.append(rc_layer)
            p = q
            
        else:
            layers.append(layer)
            for g in layer:
                if g.name == 'Gcnot':
                    (control, target) = g.qubits
                    p[qubits.index(control)] = (p[qubits.index(control)] + p[qubits.index(target)]) % 4
                    p[n + qubits.index(target)] = (p[n + qubits.index(control)] + p[n + qubits.index(target)]) % 4
                else:
                    raise ValueError("Circuit can only contain Gcnot, Gu3, and Gi gates in separate layers!")

    bs = ''.join([str(b // 2) for b in q[n:]])

    # Avoid checks for speed
    rc_circ = _Circuit(layers, line_labels=circ.line_labels, check=False, expand_subcircuits=False)

    if not return_target_pauli:
        return rc_circ, bs
    else:
        return rc_circ, bs, q
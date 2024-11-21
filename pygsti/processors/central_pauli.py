import numpy as _np

from pygsti.circuits.circuit import Circuit as _Circuit

import utils #TODO: integrate all the needed functionality from utils into pygsti

#TODO: OOP-ify this code?

def central_pauli_mirror_circuit(test_circ, ref_circ=None, randomized_state_preparation=True, rand_state=None, new=False): #want this function to stay backwards compatible, but it does need modified (or another function needs created)
    if rand_state is None:
        rand_state = _np.random.RandomState()

    qubits = test_circ.line_labels

    if new:
        circuit_to_mirror = ref_circ
    else:
        if randomized_state_preparation:
            prep_circ = _Circuit([utils.haar_random_u3_layer(qubits, rand_state)], line_labels=qubits)
            circuit_to_mirror = prep_circ + test_circ
        else: 
            circuit_to_mirror = test_circ

    n = circuit_to_mirror.width
    d = circuit_to_mirror.depth

    central_pauli = 2 * rand_state.randint(0, 2, 2*n)
    central_pauli_layer = utils.pauli_vector_to_u3_layer(central_pauli, qubits)
    q = central_pauli.copy()

    quasi_inverse_circ = _Circuit(line_labels=test_circ.line_labels, editable=True)

    for i in range(d):
        
        layer = circuit_to_mirror.layer_label(d - i - 1).components

        #print(layer)
        quasi_inverse_layer = [utils.gate_inverse(gate_label) for gate_label in layer]

        #print(quasi_inverse_layer)
        
        # Update the u3 gates.
        if len(layer) == 0 or layer[0].name == 'Gu3':
            # update_u3_parameters now twirls/adds label for empty qubits, so don't prepad for speed
            #padded_layer = pad_layer(quasi_inverse_layer, qubits)
            quasi_inverse_layer = utils.update_u3_parameters(quasi_inverse_layer, q, q, qubits)

        # Update q based on the CNOTs in the layer.
        else:
            for g in layer:
                if g.name == 'Gcnot':
                    (control, target) = g.qubits
                    q[qubits.index(control)] = (q[qubits.index(control)] + q[qubits.index(target)]) % 4
                    q[n + qubits.index(target)] = (q[n + qubits.index(control)] + q[n + qubits.index(target)]) % 4
                else:
                    raise ValueError("Circuit can only contain Gcnot and Gu3 gates in separate layers!")
                
        #print(quasi_inverse_layer)

        quasi_inverse_circ.insert_layer_inplace(quasi_inverse_layer, i)

        #print(quasi_inverse_circ)

        if new:
            mc = test_circ + _Circuit([central_pauli_layer], line_labels=test_circ.line_labels) + quasi_inverse_circ
        else:
            mc = circuit_to_mirror + _Circuit([central_pauli_layer], line_labels=test_circ.line_labels) + quasi_inverse_circ  
          
    mc.done_editing()

    bs = ''.join([str(b // 2) for b in q[n:]])

    return mc, bs


def randomize_central_pauli(circ: _Circuit, rand_state=None):
    #this is a modified version of the central_pauli_mirror_circuit() function that only uses the reference circuit and commutes the central Pauli through it.
    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    qubits = circ.line_labels

    n = circ.width
    n = circ.depth

    central_pauli = 2 * rand_state.randint(0, 2, 2*n)
    central_pauli_layer = utils.pauli_vector_to_u3_layer(central_pauli, qubits)
    q = central_pauli.copy()


def new_central_pauli_mirror_circuit(forward_circ: _Circuit, reverse_circ: _Circuit, rand_state=None):
    # the differences between this function and central_pauli_mirror_circuit (legacy) are:
        # accepting different arguments (test_circ and ref_circ_inverse instead of circuit_to_mirror)
        # state prep layer is implicitly a part of test_circ and ref_circ_inverse. There is not an option in this function to do randomized state prep, though there is in the legacy function.
        # ref_circ_inverse is already inverted, which changes the for loop from traversing back to front (d - i - 1) to front to back (d).

    if rand_state is None:
        rand_state = _np.random.RandomState()

    qubits = forward_circ.line_labels

    n = reverse_circ.width
    d = reverse_circ.depth

    central_pauli = 2 * rand_state.randint(0, 2, 2*n)
    central_pauli_layer = utils.pauli_vector_to_u3_layer(central_pauli, qubits)
    q = central_pauli.copy()

    quasi_inverse_circ = _Circuit(line_labels=forward_circ.line_labels, editable=True)

    for i in range(d):

        layer = reverse_circ.layer_label(i).components

        if layer[0].name in ['Gi', 'Gdelay']:
            continue

        # Update the u3 gates.
        elif len(layer) == 0 or layer[0].name == 'Gu3':
            # update_u3_parameters now twirls/adds label for empty qubits, so don't prepad for speed
            #padded_layer = pad_layer(quasi_inverse_layer, qubits)
            layer = utils.update_u3_parameters(layer, q, q, qubits)


        # Update q based on the CNOTs in the layer.
        else:
            for g in layer:
                if g.name == 'Gcnot':
                    (control, target) = g.qubits
                    q[qubits.index(control)] = (q[qubits.index(control)] + q[qubits.index(target)]) % 4
                    q[n + qubits.index(target)] = (q[n + qubits.index(control)] + q[n + qubits.index(target)]) % 4
                else:
                    raise ValueError("Circuit can only contain Gcnot and Gu3 gates in separate layers!")
                
        #print(layer)

        quasi_inverse_circ.insert_layer_inplace(layer, i)

        #print(quasi_inverse_circ)

    mc = forward_circ + _Circuit([central_pauli_layer], line_labels=forward_circ.line_labels) + quasi_inverse_circ
    mc = mc.reorder_lines(forward_circ.line_labels)
    mc.done_editing()

    bs = ''.join([str(b // 2) for b in q[n:]])

    return mc, bs
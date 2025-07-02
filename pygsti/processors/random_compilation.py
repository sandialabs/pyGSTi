import numpy as _np

from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.unitarygatefunction import UnitaryGateFunction as _UnitaryGateFunction
from pygsti.tools.internalgates import standard_gatename_unitaries as _standard_gatename_unitaries

#TODO: OOP-ify this code?

class RandomCompilation(object):
    def __init__(self, rc_strategy=None, return_bs=False, testing=False, rand_state=None): #pauli_rc, central_pauli are currently the supported options
        # do I need to call super().__init__?
        # is this even the right class to inherit from?
        self.rc_strategy = rc_strategy if rc_strategy is not None else "pauli_rc"
        self.return_bs = return_bs
        self.testing = testing


        if isinstance(rand_state, _np.random.RandomState):
            self.rand_state = rand_state
        elif isinstance(rand_state, int):
            self.rand_state = _np.random.RandomState(seed=rand_state)
        else:
            self.rand_state = _np.random.RandomState()


    def compile(self, circ: _Circuit, test_layers=None): # may need to a kwarg dict parameter to this function to allow for things to be passed to a given compilation choice
        # d = circ.depth
        # n = circ.width
        # framedata = {}

        if self.rc_strategy == 'pauli_rc':
            return_bs = False
            return_target_pauli = False
            insert_test_layers = False
            if self.return_bs:
                return_bs = True
            if self.testing:
                insert_test_layers = True
                return_target_pauli = True
                return_bs = True

            return pauli_randomize_circuit(circ=circ,
                                           rand_state=self.rand_state,
                                           return_bs=return_bs,
                                           return_target_pauli=return_target_pauli,
                                           insert_test_layers=insert_test_layers,
                                           test_layers=test_layers
                                           )
        
        elif self.rc_strategy == 'central_pauli':
            return_bs = False
            return_target_pauli = False
            insert_test_layer = False
            if self.return_bs:
                return_bs = True
            if self.testing:
                insert_test_layer = True
                return_target_pauli = True
                return_bs = True
                
            return randomize_central_pauli(circ=circ,
                                           rand_state=self.rand_state,
                                           return_bs=return_bs,
                                           return_target_pauli=return_target_pauli,
                                           insert_test_layer=insert_test_layer,
                                           test_layer=test_layers
                                           )
        else:
            raise ValueError(f"unknown compilation strategy '{self.rc_strategy}'!")



def pauli_randomize_circuit(circ, rand_state=None, return_bs=False, return_target_pauli=False, insert_test_layers=False, test_layers=None):
    if rand_state is None:
        rand_state = _np.random.RandomState()

    d = circ.depth
    n = circ.width
    p = _np.zeros(2*n, int)
    # q = _np.zeros(2*n, int) # fixes a bug that occurs if there are no U3 layers in the entire circuit (but tbh this should be fixed a different way)

    return_0_bs = False

    if insert_test_layers:
        num_u3_layers = 0
        for i in range(d):
            layer = circ.layer_label(i).components
            if layer[0].name == 'Gu3':
                num_u3_layers += 1
        # print(len(test_layers))
        # print(num_u3_layers)
        assert len(test_layers) == num_u3_layers, f'expected {num_u3_layers} Pauli vectors but got {len(test_layers)} instead'

    if return_bs:
        layer_0 = circ.layer_label(0).components
        if layer_0[0].name == 'Gcnot':
            return_0_bs = True


    qubits = circ.line_labels

    layers = []

    last_layer_u3 = False

    for i in range(d):

        layer = circ.layer_label(i).components

        # print(layer)

        if layer[0].name in ['Gi', 'Gdelay']: #making this explicit for the sake of clarity
            # should we be tacking a Pauli on at the end (as though this were RC of u3(0,0,0) layers?)
            layers.append(layer)

        elif len(layer) == 0 or layer[0].name == 'Gu3':
            if insert_test_layers:
                q = test_layers.pop(0)
                if len(q) != 2*n:
                    raise ValueError(f"test layer {q} should have length {2*n} but has length {len(q)} instead")
                # print(q)

            else:
                q = 2 * rand_state.randint(0, 2, 2*n)
            # update_u3_parameters now twirls/adds label for empty qubits, so don't prepad for speed
            #padded_layer = pad_layer(layer, qubits)
            rc_layer = update_u3_parameters(layer, p, q, qubits)
            # print(rc_layer)
            layers.append(rc_layer)
            p = q
            # last_layer_u3 = True
            
        else: # this must be a layer of 2Q gates. this implementation allows for Gcnot and Gcphase gates in the same layer.
            layers.append(layer)
            for g in layer:
                if g.name == 'Gcnot':
                    (control, target) = g.qubits
                    p[qubits.index(control)] = (p[qubits.index(control)] + p[qubits.index(target)]) % 4
                    p[n + qubits.index(target)] = (p[n + qubits.index(control)] + p[n + qubits.index(target)]) % 4
                    # q = p
                elif g.name == 'Gcphase':
                    (control, target) = g.qubits
                    p[qubits.index(control)] = (p[qubits.index(control)] + p[n + qubits.index(target)]) % 4
                    p[qubits.index(target)] = (p[n + qubits.index(control)] + p[qubits.index(target)]) % 4
                else:
                    raise ValueError("Circuit can only contain Gcnot, Gcphase, Gu3, and Gi gates in separate layers!")
            # last_layer_u3 = False

    # if last_layer_u3 == False:
    #     final_layer = pauli_vector_to_u3_layer(p, qubits)
    #     layers.append(final_layer)


    bs = ''.join([str(b // 2) for b in p[n:]])

    # Avoid checks for speed
    rc_circ = _Circuit(layers, line_labels=circ.line_labels, check=False, expand_subcircuits=False)

    out = [rc_circ]

    if return_bs:
        out.append(bs)
    if return_target_pauli:
        out.append(p)

    return out
    

# def central_pauli_mirror_circuit(self, circ: _Circuit): #want this function to stay backwards compatible, but it does need modified (or another function needs created)

#     qubits = circ.line_labels

#     n = circ.width
#     d = circ.depth

#     central_pauli = 2 * self.rand_state.randint(0, 2, 2*n)
#     central_pauli_layer = pauli_vector_to_u3_layer(central_pauli, qubits)
#     q = central_pauli.copy()

#     quasi_inverse_circ = _Circuit(line_labels=qubits, editable=True)

#     for i in range(d):
#         layer = circ.layer_label(d - i - 1).components

#         #print(layer)
#         quasi_inverse_layer = [gate_inverse(gate_label) for gate_label in layer]

#         #print(quasi_inverse_layer)
        
#         # Update the u3 gates.
#         if len(layer) == 0 or layer[0].name == 'Gu3':
#             # update_u3_parameters now twirls/adds label for empty qubits, so don't prepad for speed
#             #padded_layer = pad_layer(quasi_inverse_layer, qubits)
#             quasi_inverse_layer = update_u3_parameters(quasi_inverse_layer, q, q, qubits)

#         # Update q based on the CNOTs in the layer.
#         else:
#             for g in layer:
#                 if g.name == 'Gcnot':
#                     (control, target) = g.qubits
#                     q[qubits.index(control)] = (q[qubits.index(control)] + q[qubits.index(target)]) % 4
#                     q[n + qubits.index(target)] = (q[n + qubits.index(control)] + q[n + qubits.index(target)]) % 4
#                 else:
#                     raise ValueError("Circuit can only contain Gcnot and Gu3 gates in separate layers!")
                
#         #print(quasi_inverse_layer)

#         quasi_inverse_circ.insert_layer_inplace(quasi_inverse_layer, i)

#         #print(quasi_inverse_circ)

#         if new:
#             mc = test_circ + _Circuit([central_pauli_layer], line_labels=test_circ.line_labels) + quasi_inverse_circ
#         else:
#             mc = circuit_to_mirror + _Circuit([central_pauli_layer], line_labels=test_circ.line_labels) + quasi_inverse_circ  
        
#     mc.done_editing()

#     bs = ''.join([str(b // 2) for b in q[n:]])

#     return mc, bs


def randomize_central_pauli(circ: _Circuit, rand_state=None, return_bs=False, return_target_pauli=False, insert_test_layer=False, test_layer=None):
    #this is a modified version of the central_pauli_mirror_circuit() function that only uses the reference circuit and commutes the central Pauli through it.
    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    qubits = circ.line_labels

    n = circ.width
    d = circ.depth

    if insert_test_layer:
        assert len(test_layer) == 2*n, f"Central Pauli vector must be length {2*n} but test_layer has length {len(test_layer)}"
        central_pauli = test_layer
    else:
        central_pauli = 2 * rand_state.randint(0, 2, 2*n)

    central_pauli_layer = pauli_vector_to_u3_layer(central_pauli, qubits)
    p = central_pauli.copy()

    layers = [central_pauli_layer]

    for i in range(d):

        layer = circ.layer_label(i).components

        if layer[0].name in ['Gi', 'Gdelay']: #making this explicit for the sake of clarity
            # should we be tacking a Pauli on at the end (as though this were RC of u3(0,0,0) layers?)
            layers.append(layer)

        elif len(layer) == 0 or layer[0].name == 'Gu3':
            # update_u3_parameters now twirls/adds label for empty qubits, so don't prepad for speed
            #padded_layer = pad_layer(layer, qubits)
            rc_layer = update_u3_parameters(layer, p, p, qubits)
            layers.append(rc_layer)
            
        else:
            layers.append(layer)
            for g in layer:
                if g.name == 'Gcnot':
                    (control, target) = g.qubits
                    p[qubits.index(control)] = (p[qubits.index(control)] + p[qubits.index(target)]) % 4
                    p[n + qubits.index(target)] = (p[n + qubits.index(control)] + p[n + qubits.index(target)]) % 4
                elif g.name == 'Gcphase':
                    (control, target) = g.qubits
                    p[qubits.index(control)] = (p[qubits.index(control)] + p[n + qubits.index(target)]) % 4
                    p[qubits.index(target)] = (p[n + qubits.index(control)] + p[qubits.index(target)]) % 4
                else:
                    raise ValueError("Circuit can only contain Gcnot, Gcphase, Gu3, and Gi gates in separate layers!")

    bs = ''.join([str(b // 2) for b in p[n:]])

    # Avoid checks for speed
    cp_circ = _Circuit(layers, line_labels=circ.line_labels, check=False, expand_subcircuits=False)

    out = [cp_circ]

    if return_bs:
        out.append(bs)
    if return_target_pauli:
        out.append(p)

    return out

# def new_central_pauli_mirror_circuit(forward_circ: _Circuit, reverse_circ: _Circuit, rand_state=None):
#     # the differences between this function and central_pauli_mirror_circuit (legacy) are:
#         # accepting different arguments (test_circ and ref_circ_inverse instead of circuit_to_mirror)
#         # state prep layer is implicitly a part of test_circ and ref_circ_inverse. There is not an option in this function to do randomized state prep, though there is in the legacy function.
#         # ref_circ_inverse is already inverted, which changes the for loop from traversing back to front (d - i - 1) to front to back (d).

#     if rand_state is None:
#         rand_state = _np.random.RandomState()

#     qubits = forward_circ.line_labels

#     n = reverse_circ.width
#     d = reverse_circ.depth

#     central_pauli = 2 * rand_state.randint(0, 2, 2*n)
#     central_pauli_layer = pauli_vector_to_u3_layer(central_pauli, qubits)
#     q = central_pauli.copy()

#     quasi_inverse_circ = _Circuit(line_labels=forward_circ.line_labels, editable=True)

#     for i in range(d):

#         layer = reverse_circ.layer_label(i).components

#         if layer[0].name in ['Gi', 'Gdelay']:
#             continue

#         # Update the u3 gates.
#         elif len(layer) == 0 or layer[0].name == 'Gu3':
#             # update_u3_parameters now twirls/adds label for empty qubits, so don't prepad for speed
#             #padded_layer = pad_layer(quasi_inverse_layer, qubits)
#             layer = update_u3_parameters(layer, q, q, qubits)


#         # Update q based on the CNOTs in the layer.
#         else:
#             for g in layer:
#                 if g.name == 'Gcnot':
#                     (control, target) = g.qubits
#                     q[qubits.index(control)] = (q[qubits.index(control)] + q[qubits.index(target)]) % 4
#                     q[n + qubits.index(target)] = (q[n + qubits.index(control)] + q[n + qubits.index(target)]) % 4
#                 else:
#                     raise ValueError("Circuit can only contain Gcnot and Gu3 gates in separate layers!")
                
#         #print(layer)

#         quasi_inverse_circ.insert_layer_inplace(layer, i)

#         #print(quasi_inverse_circ)

#     mc = forward_circ + _Circuit([central_pauli_layer], line_labels=forward_circ.line_labels) + quasi_inverse_circ
#     mc = mc.reorder_lines(forward_circ.line_labels)
#     mc.done_editing()

#     bs = ''.join([str(b // 2) for b in q[n:]])

#     return mc, bs



def update_u3_parameters(layer, p, q, qubits):
    """
    Takes a layer containing u3 gates, and finds a new layer containing
    u3 gates that implements p * layer * q (p followed by layer followed by
    q, so q * layer * p in matrix order), where p and q are vectors  describing layers of paulis.

    """
    used_qubits = []

    new_layer = []
    n = len(qubits)

    for g in layer:
        assert(g.name == 'Gu3')
        (theta, phi, lamb) = (float(g.args[0]), float(g.args[1]), float(g.args[2]))
        qubit_index = qubits.index(g.qubits[0])
        if p[qubit_index] == 2:   # Z gate preceeding the layer
            lamb = lamb + _np.pi
        if q[qubit_index] == 2:   # Z gate following the layer
            phi = phi + _np.pi
        if p[n + qubit_index] == 2:  # X gate preceeding the layer
            theta = theta - _np.pi
            phi = phi
            lamb = -lamb - _np.pi
        if q[n + qubit_index] == 2:  # X gate following the layer
            theta = theta - _np.pi
            phi = -phi - _np.pi
            lamb = lamb

        new_args = (mod_2pi(theta), mod_2pi(phi), mod_2pi(lamb))
        new_label = _Label('Gu3', g.qubits[0], args=new_args)
        new_layer.append(new_label)
        used_qubits.append(g.qubits[0])
    
    for qubit_index, qubit in enumerate(qubits):
        if qubit in used_qubits:
            continue

        # Insert twirled idle on unpadded qubit
        (theta, phi, lamb) = (0.0, 0.0, 0.0)
        if p[qubit_index] == 2:   # Z gate preceeding the layer
            lamb = lamb + _np.pi
        if q[qubit_index] == 2:   # Z gate following the layer
            phi = phi + _np.pi
        if p[n + qubit_index] == 2:  # X gate preceeding the layer
            theta = theta - _np.pi
            phi = phi
            lamb = -lamb - _np.pi
        if q[n + qubit_index] == 2:  # X gate following the layer
            theta = theta - _np.pi
            phi = -phi - _np.pi
            lamb = lamb
        
        new_args = (mod_2pi(theta), mod_2pi(phi), mod_2pi(lamb))
        new_label = _Label('Gu3', qubit, args=new_args)
        new_layer.append(new_label)
        used_qubits.append(qubit)

    assert(set(used_qubits) == set(qubits))

    return new_layer

def mod_2pi(theta):
    while (theta > _np.pi or theta <= -1 * _np.pi):
        if theta > _np.pi:
            theta = theta - 2 * _np.pi
        elif theta <= -1 * _np.pi:
            theta = theta + 2 * _np.pi
    return theta


def pauli_vector_to_u3_layer(p, qubits):
    
    n = len(qubits)
    layer = []
    for i, q in enumerate(qubits):

        if p[i] == 0 and p[i+n] == 0:  # I
            theta = 0.0
            phi = 0.0
            lamb = 0.0
        if p[i] == 2 and p[i+n] == 0:  # Z
            theta = 0.0
            phi = _np.pi / 2
            lamb = _np.pi / 2
        if p[i] == 0 and p[i+n] == 2:  # X
            theta = _np.pi
            phi = 0.0
            lamb = _np.pi
        if p[i] == 2 and p[i+n] == 2:  # Y
            theta = _np.pi
            phi = _np.pi / 2
            lamb = _np.pi / 2

        layer.append(_Label('Gu3', q, args=(theta, phi, lamb)))

    return _Label(layer)

def haar_random_u3_layer(qubits, rand_state=None):
    
    return _Label([haar_random_u3(q, rand_state) for q in qubits])

def haar_random_u3(q, rand_state=None):
    if rand_state is None:
        rand_state = _np.random.RandomState()

    a, b = 2 * _np.pi * rand_state.rand(2)
    theta = mod_2pi(2 * _np.arcsin(_np.sqrt(rand_state.rand(1)))[0])
    phi = mod_2pi(a - b + _np.pi)
    lamb = mod_2pi(-1 * (a + b + _np.pi))
    return _Label('Gu3', q, args=(theta, phi, lamb))


def u3_cx_inv(circ: _Circuit) -> _Circuit:
    # Surely this method exists already. Surely.
    # I need to optimize this a little, it is bugging me
    inverse_layers = []
    d = circ.depth

    for j in range(d):
        layer = circ.layer(j)
        inverse_layer = [gate_inverse(gate_label) for gate_label in layer]
        #not doing padding. See subcirc-vb/mirror_circuits.py, bare_mirror_circuit()
        #not sure if it would be an optimization, but it is possible to read the layers in reverse and write in order instead of reading layers in order and writing in reverse
        inverse_layers.insert(0, inverse_layer)

    inverse_circ = _Circuit(inverse_layers, line_labels = circ.line_labels, check=False, expand_subcircuits=False)

    # print("inverse circuit:")
    # print(inverse_circ)

    return inverse_circ

def gate_inverse(label):
    if label.name == 'Gcnot':
        return label
    elif label.name == 'Gcphase':
        return label
    elif label.name == 'Gu3':
        return _Label('Gu3', label.qubits, args=inverse_u3(label.args))
    elif label.name in ['Gi', 'Gdelay']:
        return label
    else:
        raise RuntimeError(f'cannot compute gate inverse for {label}')

def inverse_u3(args):
    theta_inv = mod_2pi(-float(args[0]))
    phi_inv = mod_2pi(-float(args[2]))
    lambda_inv = mod_2pi(-float(args[1]))
    return (theta_inv, phi_inv, lambda_inv)


def pad_layer(layer, qubits):

    padded_layer = list(layer)
    used_qubits = []
    for g in layer:
        for q in g.qubits:
            used_qubits.append(q)

    for q in qubits:
        if q not in used_qubits:
            padded_layer.append(_Label('Gu3', (q,), args=(0.0, 0.0, 0.0)))

    return padded_layer

class Gu3(_UnitaryGateFunction):
    shape = (2,2)
    def __call__(self, arg):
        theta, phi, lamb = (float(arg[0]), float(arg[1]), float(arg[2]))
        return _np.array([[_np.cos(theta/2), -_np.exp(1j*lamb)*_np.sin(theta/2)],
                         [_np.exp(1j*phi)*_np.sin(theta/2), _np.exp(1j*(phi + lamb))*_np.cos(theta/2)]])

def get_clifford_from_unitary(U):
    clifford_unitaries = {k: v for k, v in _standard_gatename_unitaries().items()
                          if 'Gc' in k and v.shape == (2, 2)}
    for k,v in clifford_unitaries.items():
        for phase in [1, -1, 1j, -1j]:
            if _np.allclose(U, phase*v):
                return k
            
    raise RuntimeError(f'Failed to look up Clifford for unitary:\n{U}')
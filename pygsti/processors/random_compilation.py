"""
Randomized circuit compilation via random compiling and central Pauli propagation.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
from typing import Literal, Optional, Union, List, Iterable, Dict, Tuple

import numpy as _np

from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.baseobjs.label import Label as _Label

class RandomCompilation(object):
    """
    A class for performing randomized circuit compilation.

    Attributes
    ----------
    rc_strategy : str
        The strategy used for randomized compilation. Currently,
        'pauli_rc' (pauli randomized compiling on a U3-CX-CZ gate set) and
        'central_pauli' (central Pauli propagation for a U3-CX-CZ gate set)
        are supported.

    return_bs : bool
        If True, the `compile` method will return the target bitstring for
        the randomly compiled circuit.

    testing : bool
        Flag for unit testing. If True, the user can provide test Pauli layers for
        random compilation instead of the layers being randomly generated.

    rand_state : np.random.RandomState
        A random state for reproducibility of random operations.
    """

    def __init__(self,
                 rc_strategy: Optional[Literal['rc', 'cp']] = None,
                 return_bs: Optional[bool] = False,
                 testing: Optional[bool] = False,
                 rand_state: Optional[_np.random.RandomState] = None):
        """
        Initialize the RandomCompilation object.

        Parameters
        ----------
        rc_strategy : str
            The strategy used for randomized compilation. Currently,
            'pauli_rc' (pauli randomized compiling on a U3-CX-CZ gate set, see
            https://arxiv.org/abs/2204.07568) and 'central_pauli'
            (central Pauli propagation for a U3-CX-CZ gate set, see 
            https://www.nature.com/articles/s41567-021-01409-7)
            are supported. Defaults to 'pauli_rc'.

        return_bs : bool
            If True, the `compile` method will return the target bitstring for
            the randomly compiled circuit. Default is False.

        testing : bool
            Flag for unit testing. If True, the user can provide test Pauli layers for
            random compilation instead of the layers being randomly generated. Default is False.

        rand_state : np.random.RandomState
            A random state for reproducibility of random operations. Default is None.
        """

        self.rc_strategy = rc_strategy if rc_strategy is not None else "pauli_rc"
        self.return_bs = return_bs
        self.testing = testing


        if isinstance(rand_state, _np.random.RandomState):
            self.rand_state = rand_state
        elif isinstance(rand_state, int):
            self.rand_state = _np.random.RandomState(seed=rand_state)
        else:
            self.rand_state = _np.random.RandomState()


    def compile(self,
                circ: _Circuit,
                test_layers: Optional[Union[List[_np.ndarray], _np.ndarray]] = None
                ) -> _Circuit:
        """
        Compiles the given circuit using the specified randomized compilation strategy.

        Parameters
        -------------
        circ : pygsti.circuits.Circuit
            The n-qubit circuit to be compiled.

        test_layers : list[np.ndarray[int]], optional
            A list of test layers to be used in the random compilation
            if `self.testing` is True. Layers are specified by a length-2*n array
            whose entries are either 0 or 2. Indices 0:n correspond to Pauli Z errors:
            a 2 indicates the presence an error. Likewise indices n:2*n indicate a
            Pauli Z error. If using central Pauli, only one layer must be provided.
            If using random compilation, a number of layers equal to the number of 
            layers of single-qubit gates must be provided. Default is None.

        Returns
        --------
        list[pygsti.circuits.Circuit, str (optional), np.ndarray (optional)]
            A list containing the randomized circuit, and optionally the bitstring and target Pauli vector.
        """

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


def pauli_randomize_circuit(circ: _Circuit,
                            rand_state: Optional[_np.random.RandomState] = None,
                            return_bs: bool = False,
                            return_target_pauli: bool = False,
                            insert_test_layers: bool = False,
                            test_layers: Optional[List[_np.ndarray]] = None
                            ) -> _Circuit:
    """
    Performs random compilation on a given circuit by inserting Pauli gates between layers.

    Parameters
    -------------
    circ : pygsti.circuits.Circuit
        The circuit to be randomized.

    rand_state : np.random.RandomState, optional
        A random state for reproducibility. Default is None, which initializes a new random state.

    return_bs : bool, optional
        If True, returns the target bitstring for the randomly compiled circuit. Default is False.

    return_target_pauli : bool, optional
        If True, returns the target Pauli vector for the circuit. Default is False.

    insert_test_layers : bool, optional
        If True, uses `test_layers` as the Pauli layers to randomly compile instead of
        randomly generating Pauli layers.

    test_layers : list[np.ndarray[int]], optional
        A list of length-2*n arrays representing the test layers to be inserted
        if `insert_test_layers `is True. The number of test layers must equal
        the number of U3 layers in the circuit. Default is None.

    Returns
    --------
    list[pygsti.circuits.Circuit, str (optional), np.ndarray (optional)]
        A list containing the randomized circuit, and optionally the bitstring and target Pauli vector if specified.
    """

    if rand_state is None:
        rand_state = _np.random.RandomState()

    d = circ.depth
    n = circ.width
    p = _np.zeros(2*n, int)


    if insert_test_layers:
        num_u3_layers = 0
        for i in range(d):
            layer = circ.layer_label(i).components
            if layer[0].name == 'Gu3':
                num_u3_layers += 1
        assert len(test_layers) == num_u3_layers, f'expected {num_u3_layers} Pauli vectors but got {len(test_layers)} instead'

    if return_bs:
        layer_0 = circ.layer_label(0).components
        if layer_0[0].name == 'Gcnot':
            return_0_bs = True

    qubit_map = {j:i for i,j in enumerate(circ.line_labels)}

    layers = []

    for i in range(d):
        layer = circ.layer_label(i).components

        if layer[0].name in ['Gi', 'Gdelay']: # making this explicit for the sake of clarity
            layers.append(layer)

        elif len(layer) == 0 or layer[0].name == 'Gu3':
            if insert_test_layers:
                q = test_layers.pop(0)
                if len(q) != 2*n:
                    raise ValueError(f"test layer {q} should have length {2*n} but has length {len(q)} instead")

            else:
                q = 2 * rand_state.randint(0, 2, 2*n)
            # update_u3_parameters now twirls/adds label for empty qubits, so don't prepad for speed
            rc_layer = update_u3_parameters(layer, p, q, qubit_map)
            layers.append(rc_layer)
            p = q
            
        else: # we must have a layer of 2Q gates. this implementation allows for Gcnot and Gcphase gates in the same layer.
            layers.append(layer)
            for g in layer:
                if g.name == 'Gcnot':
                    (control, target) = g.qubits
                    p[qubit_map[control]] = (p[qubit_map[control]] + p[qubit_map[target]]) % 4
                    p[n + qubit_map[target]] = (p[n + qubit_map[control]] + p[n + qubit_map[target]]) % 4
                elif g.name == 'Gcphase':
                    (control, target) = g.qubits
                    p[qubit_map[control]] = (p[qubit_map[control]] + p[n + qubit_map[target]]) % 4
                    p[qubit_map[target]] = (p[n + qubit_map[control]] + p[qubit_map[target]]) % 4
                else:
                    raise ValueError("Circuit can only contain Gcnot, Gcphase, Gu3, and Gi gates in separate layers!")

    bs = ''.join([str(b // 2) for b in p[n:]])

    # Avoid checks for speed
    rc_circ = _Circuit(layers, line_labels=circ.line_labels, check=False, expand_subcircuits=False)

    out = [rc_circ]

    if return_bs:
        out.append(bs)
    if return_target_pauli:
        out.append(p)

    return out
    

def randomize_central_pauli(circ: _Circuit,
                            rand_state: Optional[_np.random.RandomState] = None,
                            return_bs: bool = False,
                            return_target_pauli: bool = False,
                            insert_test_layer: bool = False,
                            test_layer: Optional[_np.ndarray] = None
                            ) -> _Circuit:
    """
    Perform circuit randomization by propagating a central Pauli layer through the circuit.
    This function is designed to handle the "back half" of the mirror circuit: i.e., given a circuit C
    whose fidelity is to be estimated using central Pauli mirroring, this function should be passed
    C_inv + L_inv, where L_inv is Haar-random U3 layer. Refer to `make_mirror_edesign` in `protocols/mirror_design.py`
    for more information.

    Parameters
    -------------
    circ : Circuit
        The circuit through which the central Pauli layer is to be propagated.

    rand_state : np.random.RandomState, optional
        A random state for reproducibility. Default is None, which initializes a new random state.

    return_bs : bool, optional
        If True, returns the target bitstring for the full mirror central Pauli circuit. Default is False.

    return_target_pauli : bool, optional
        If True, returns the target Pauli vector that has been propagated
        through the circuit. Default is False.

    insert_test_layer : bool, optional
        If True, uses `test_layer` as the central Pauli layer
        instead of randomly generating a central Pauli layer.

    test_layer : np.ndarray[int], optional
        A length-2*n array representing the test layer to be inserted
        if `insert_test_layer `is True. Default is None.

    Returns
    --------
    list[pygsti.circuits.Circuit, str (optional), np.ndarray (optional)]
        A list containing the randomized circuit, and optionally the bitstring and target Pauli vector if specified.
    """
    
    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    qubits = circ.line_labels

    qubit_map = {j:i for i,j in enumerate(circ.line_labels)}

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
            layers.append(layer)

        elif len(layer) == 0 or layer[0].name == 'Gu3':
            rc_layer = update_u3_parameters(layer, p, p, qubit_map)
            layers.append(rc_layer)
            
        else:
            layers.append(layer)
            for g in layer:
                if g.name == 'Gcnot':
                    (control, target) = g.qubits
                    p[qubit_map[control]] = (p[qubit_map[control]] + p[qubit_map[target]]) % 4
                    p[n + qubit_map[target]] = (p[n + qubit_map[control]] + p[n + qubit_map[target]]) % 4
                elif g.name == 'Gcphase':
                    (control, target) = g.qubits
                    p[qubit_map[control]] = (p[qubit_map[control]] + p[n + qubit_map[target]]) % 4
                    p[qubit_map[target]] = (p[n + qubit_map[control]] + p[qubit_map[target]]) % 4
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


def update_u3_parameters(layer: Iterable[_Label],
                         p: _np.ndarray,
                         q: _np.ndarray,
                         qubit_map: Union[Dict[str, int], Dict[int, int]]
                         ) -> List[_Label]:
    """
    Updates the parameters of U3 gates in a given layer based on the provided Pauli random compiling vectors.

    Parameters
    -------------
    layer : iterable[pygsti.baseobjs.Label]
        A list of gate labels representing the layer containing U3 gates.

    p : np.ndarray[int]
        A vector describing the Pauli gates preceding the layer.
        For an n-qubit layer, p is a length-2n array.
        p[0:n] indicates Pauli-Z (2 is yes Z, 0 is no Z), p[n:2*n] is Pauli-X (2 yes, 0 no).
        E.g., if n = 5, p[3] = 2, and p[8] = 2, then there is a Y gate on qubit 3.

    p : np.ndarray[int]
        A vector describing the Pauli gates foloowing the layer.
        For an n-qubit layer, q is a length-2n array.
        q[0:n] indicates Pauli-Z (2 is yes Z, 0 is no Z), q[n:2*n] is Pauli-X (2 yes, 0 no).
        E.g., if n = 5, p[1] = 0, and p[6] = 2, then there is an X gate on qubit 3.

    qubit_map : dict[str, int]
        A mapping of qubit labels to their corresponding indices.

    Returns
    --------
    list
        A new layer containing updated U3 gates based on the applied Pauli gates.
    """
    used_qubits = set()

    new_layer = []
    n = len(qubit_map)

    for g in layer:
        assert(g.name == 'Gu3')
        (theta, phi, lamb) = (float(g.args[0]), float(g.args[1]), float(g.args[2]))
        qubit = g.qubits[0]
        qubit_index = qubit_map[qubit]
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
        used_qubits.add(qubit)
    
    for qubit, qubit_index in qubit_map.items():
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

        if _np.allclose((theta, phi, lamb), (0.0, 0.0, 0.0)):
            new_label = _Label('Gi', qubit, args=None)
        else:
            new_args = (mod_2pi(theta), mod_2pi(phi), mod_2pi(lamb))
            new_label = _Label('Gu3', qubit, args=new_args)
        new_layer.append(new_label)
        used_qubits.add(qubit)

    assert(set(used_qubits) == set(qubit_map.keys()))

    return new_layer

def mod_2pi(theta: float) -> float:
    """
    Modifies an angle to be within the range of -π to π.

    Parameters
    -------------
    theta : float
        The angle in radians to be modified.

    Returns
    --------
    float
        The modified angle within the range of -π to π.
    """

    while (theta > _np.pi or theta <= -1 * _np.pi):
        if theta > _np.pi:
            theta = theta - 2 * _np.pi
        elif theta <= -1 * _np.pi:
            theta = theta + 2 * _np.pi
    return theta


def pauli_vector_to_u3_layer(p: _np.ndarray,
                             qubits: Union[List[str], List[int]]
                             ) -> _Label:
    """
    Converts a Pauli vector into a corresponding layer of U3 gates.

    Parameters
    -------------
    p : np.ndarray[int]
        A vector representing the Pauli gates to be converted.
        For an n-qubit layer, p is a length-2n array.
        p[0:n] indicates Pauli-Z (2 is yes Z, 0 is no Z), p[n:2*n] is Pauli-X (2 yes, 0 no).
        E.g., if n = 5, p[3] = 2, and p[8] = 2, then there is a Y gate on qubit 3.

    qubits : list[str]
        A list of qubit labels corresponding to the Pauli vector.

    Returns
    --------
    pygsti.baseobjs.Label
        Label containing the layer of U3 gates derived from the Pauli vector.
    """
    
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

def haar_random_u3_layer(qubits: Union[List[str], List[int]],
                         rand_state: Optional[_np.random.RandomState] = None
                         ) -> _Label:
    """
    Generates a layer of Haar-random U3 gates.

    Parameters
    -------------
    qubits : list[str]
        A list of qubit labels for which to generate U3 gates.
    rand_state : np.random.RandomState, optional
        A random state for reproducibility. Default is None, which initializes a new random state.

    Returns
    --------
    pygsti.baseobjs.Label
        A label containing the layer of randomly generated U3 gates.
    """
    
    return _Label([haar_random_u3(q, rand_state) for q in qubits])

def haar_random_u3(q: Union[str, int],
                   rand_state: Optional[_np.random.RandomState] = None
                   ) -> _Label:
    """
    Generates a Haar-random U3 gate.

    Parameters
    -------------
    q : str
        The qubit label for which to generate the U3 gate.
    rand_state : np.random.RandomState, optional
        A random state for reproducibility. Default is None, which initializes a new random state.

    Returns
    --------
    pygsti.baseobjs.Label
        A label representing the randomly generated U3 gate for the specified qubit.
    """

    if rand_state is None:
        rand_state = _np.random.RandomState()

    a, b = 2 * _np.pi * rand_state.rand(2)
    theta = mod_2pi(2 * _np.arcsin(_np.sqrt(rand_state.rand(1)))[0])
    phi = mod_2pi(a - b + _np.pi)
    lamb = mod_2pi(-1 * (a + b + _np.pi))
    return _Label('Gu3', q, args=(theta, phi, lamb))


def u3_cx_cz_inv(circ: _Circuit) -> _Circuit:
    """
    Computes the inverse of a circuit composed of U3, CX and CZ gates.

    Parameters
    -------------
    circ : pygsti.circuits.Circuit
        The circuit for which to compute the inverse.

    Returns
    --------
    pygsti.circuits.Circuit
        A new circuit representing the inverse of the input circuit.
    """

    inverse_layers = []
    d = circ.depth

    for j in range(d):
        layer = circ.layer(j)
        inverse_layer = [gate_inverse(gate_label) for gate_label in layer]
        inverse_layers.insert(0, inverse_layer)

    inverse_circ = _Circuit(inverse_layers, line_labels = circ.line_labels, check=False, expand_subcircuits=False)

    return inverse_circ

def gate_inverse(label: _Label) -> _Label:
    """
    Computes the inverse of a given gate label.

    Parameters
    -------------
    label : pygsti.baseobjs.Label
        The gate label for which to compute the inverse.

    Returns
    --------
    pygsti.baseobjs.Label
        A new label representing the inverse of the input gate.
    """

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

def inverse_u3(args: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Computes the inverse parameters for a U3 gate given its parameters.

    Parameters
    -------------
    args : tuple[float]
        A tuple containing the parameters (theta, phi, lambda) of the U3 gate.

    Returns
    --------
    tuple[float]
        A tuple containing the parameters of the inverse U3 gate.
    """

    theta_inv = mod_2pi(-float(args[0]))
    phi_inv = mod_2pi(-float(args[2]))
    lambda_inv = mod_2pi(-float(args[1]))
    return (theta_inv, phi_inv, lambda_inv)


def pad_layer(layer: Iterable[_Label],
              qubits: Union[List[str], List[int]]
              ) -> List[_Label]:
    """
    Pads a layer of gates with idle gates for any unused qubits.

    Parameters
    -------------
    layer : list[pygsti.baseobjs.Label]
        A list of gate labels representing the layer to be padded.
    qubits : list[str]
        A list of qubit labels to ensure all qubits are represented in the padded layer.

    Returns
    --------
    list[pygsti.baseobjs.Label]
        A new layer containing the original gates and idle gates for unused qubits.
    """

    padded_layer = list(layer)
    used_qubits = []
    for g in layer:
        for q in g.qubits:
            used_qubits.append(q)

    for q in qubits:
        if q not in used_qubits:
            padded_layer.append(_Label('Gu3', (q,), args=(0.0, 0.0, 0.0)))

    return padded_layer
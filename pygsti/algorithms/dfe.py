"""
Direct fidelity estimation (DFE) circuit sampling
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

from pygsti.tools import symplectic as _symp
from pygsti.algorithms import randomcircuit as _rc


def sample_dfe_circuit(pspec, circuit, clifford_compilations, create_same_measurement_reference=False, seed=None,
                       fixed_identity_locations=None):
    """
    Dresses a Clifford circuit for direct fidelity estimation (DFE).

    Samples a random Pauli P, prepends to `circuit` a layer that prepares a random
    stabilizer state of P, and appends a layer that maps the image of P under `circuit`
    onto a Pauli consisting only of Zs and Is, so that the resulting circuit's process
    fidelity can be estimated from computational-basis measurements. The returned
    measurement and sign specify the observable to estimate: the expected outcome of the
    circuit is the parity of the bits at the qubits marked 'Z' in `measurement`, equal to
    `sign` in the absence of error.

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The processor specification that `circuit` is defined on.

    circuit : Circuit
        The Clifford circuit to estimate the fidelity of. Must be a Clifford circuit,
        as its symplectic representation is computed.

    clifford_compilations : CompilationRules
        The absolute compilation rules used to compile the Pauli and stabilizer layers
        into the native gates of `pspec`.

    create_same_measurement_reference : bool, optional
        Not implemented. Passing True raises NotImplementedError.

    seed : int, optional
        A seed for the random number generator, for reproducible sampling.

    fixed_identity_locations : list of bool, optional
        If given, must have one entry per qubit of `circuit`. Qubits whose entry is True
        are always assigned the identity Pauli, and the remaining qubits are assigned a
        uniformly random non-identity Pauli. If None, a uniformly random non-identity
        Pauli is sampled over all the qubits.

    Returns
    -------
    Circuit
        `circuit` with the state-preparation and measurement-basis-change layers attached.

    list
        The measurement, as a list of 'I' and 'Z', one per qubit.

    int
        The sign, +1 or -1, of the expected value of the measurement for an ideal circuit.
    """
    if create_same_measurement_reference:
        raise NotImplementedError("create_same_measurement_reference is not implemented!")

    qubit_labels = circuit.line_labels
    n = len(qubit_labels)

    # Each of the three samplers below is seeded from this generator, rather than all of them
    # being given `seed`, so that they do not draw correlated randomness.
    rand_state = _np.random.RandomState(seed)  # Ok if seed is None

    if fixed_identity_locations is None:
        rand_pauli, rand_sign, pauli_circuit = _rc._sample_random_pauli(
            n=n, pspec=pspec, qubit_labels=qubit_labels, absolute_compilation=clifford_compilations,
            circuit=True, include_identity=False, seed=rand_state.randint(0, 2**31))
    else:
        rand_pauli, rand_sign, pauli_circuit = _rc._sample_random_pauli_with_identities_fixed(
            n=n, is_identity=fixed_identity_locations, pspec=pspec, qubit_labels=qubit_labels,
            absolute_compilation=clifford_compilations, circuit=True, seed=rand_state.randint(0, 2**31))

    # randomize_for_identity=False leaves the qubits with an identity Pauli idle. A random
    # Clifford there would change the width and two-qubit gate density of the circuit, which
    # must be held fixed for featuremetric benchmarking.
    s_inputstate, p_inputstate, s_init_layer, p_init_layer, prep_circuit = _rc._sample_stabilizer(
        rand_pauli, rand_sign, clifford_compilations, qubit_labels, randomize_for_identity=False,
        seed=rand_state.randint(0, 2**31))

    pspec_subset = pspec.subset(gate_names_to_include='all', qubit_labels_to_keep=qubit_labels)
    # Note: if the pspec contains gates not in pyGSTi, this will fail.
    s_pc, p_pc = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec=pspec_subset)

    # Build the initial layer of the circuit.
    full_circuit = prep_circuit.copy(editable=True)

    # Find the symplectic matrix / phase vector of the input circuit.
    s_rc, p_rc = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec_subset)

    full_circuit.append_circuit_inplace(circuit)

    s_outputstate, p_outputstate = _symp.apply_clifford_to_stabilizer_state(s_rc, p_rc, s_inputstate, p_inputstate)

    # Figure out which stabilizer of s_outputstate rand_pauli was mapped to.
    s_rc_inv, p_rc_inv = _symp.inverse_clifford(s_rc, p_rc)  # U^(-1)
    s_new_pauli, p_new_pauli = _symp.compose_cliffords(s_rc_inv, p_rc_inv, s_pc, p_pc)  # PU^(-1)
    s_new_pauli, p_new_pauli = _symp.compose_cliffords(s_new_pauli, p_new_pauli, s_rc, p_rc)  # UPaU^(-1)

    pauli_vector = p_new_pauli
    pauli = [i[0] for i in _symp.find_pauli_layer(pauli_vector, [j for j in range(n)])]
    measurement = ['I' if i == 'I' else 'Z' for i in pauli]

    # Turn the stabilizer into an all Z and I stabilizer.
    s_stab, p_stab, stab_circuit = _rc._stabilizer_to_all_zs(pauli, qubit_labels, clifford_compilations,
                                                             seed=rand_state.randint(0, 2**31))

    full_circuit.append_circuit_inplace(stab_circuit)

    s_outputstate, p_outputstate = _symp.apply_clifford_to_stabilizer_state(s_stab, p_stab,
                                                                           s_outputstate, p_outputstate)

    full_circuit.done_editing()
    sign = _rc._determine_sign(s_outputstate, p_outputstate, measurement)

    return full_circuit, measurement, sign

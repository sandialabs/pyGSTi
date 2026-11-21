"""
Direct fidelity estimation (DFE) circuit sampling
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.tools import symplectic as _symp
from pygsti.tools import matrixmod2 as _mtx
from pygsti.algorithms import randomcircuit as _rc


def _sample_eigenstate_bits(support, sign, seed=None):
    # Uniform over all bitstrings whose parity on `support` (bools, True where the Pauli isn't the
    # identity, at least one True) gives `sign`; the bits off the support are free.
    rand_state = _np.random.RandomState(seed)  # Ok if seed is None
    bits = rand_state.randint(0, 2, len(support))
    idx = [i for i, s in enumerate(support) if s]
    bits[idx] = _mtx.random_bitstring(len(idx), 0 if sign == 1 else 1, rand_state=rand_state)
    return bits


def sample_dfe_circuit(pspec, circuit, clifford_compilations, reference=None, seed=None):
    """
    Dresses a Clifford circuit for direct fidelity estimation (DFE).

    Samples a random Pauli P, prepends to `circuit` a layer that prepares every qubit in a
    uniformly random eigenstate of its Pauli (|0> or |1> where the Pauli is the identity), with
    the sign of P the parity of the bits on its support, and appends a layer that maps the
    image of P under `circuit` onto a Pauli consisting only of Zs and Is, so that the resulting
    circuit's process fidelity can be estimated from computational-basis measurements. The
    returned measurement and sign specify the observable to estimate: the expected outcome of
    the circuit is the parity of the bits at the qubits marked 'Z' in `measurement`, equal to
    `sign` in the absence of error. An identity qubit of P carries an X half the time, while
    the final layer leaves the qubits where the image of P is the identity idle.

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

    reference : None, 'paired' or 'readout', optional
        If not None, also returns a reference circuit with the dressed circuit's measurement
        and ideal sign, and no target circuit in between. A `'paired'` reference is a random
        stabilizer state of the image of the sampled Pauli under `circuit` (its preparation also
        uses random bits on every qubit), followed by the same final layer as the dressed circuit.
        A `'readout'` reference is a uniformly random bitstring with the parity of the measured
        Z string, prepared with X gates and followed by no final layer. The dressed circuit is the
        same with or without a reference. Any other value raises ValueError.

    seed : int, optional
        A seed for the random number generator, for reproducible sampling.

    Returns
    -------
    Circuit
        `circuit` with the state-preparation and measurement-basis-change layers attached.

    list
        The measurement, as a list of 'I' and 'Z', one per qubit.

    int
        The sign, +1 or -1, of the expected value of the measurement for an ideal circuit.

    Circuit
        The reference circuit, only returned when `reference` is not None.
    """
    if reference not in (None, 'paired', 'readout'):
        raise ValueError("reference must be None, 'paired' or 'readout', not %r." % (reference,))

    qubit_labels = circuit.line_labels
    n = len(qubit_labels)

    # Each of the three samplers below is seeded from this generator, rather than all of them
    # being given `seed`, so that they do not draw correlated randomness.
    rand_state = _np.random.RandomState(seed)  # Ok if seed is None

    rand_pauli, rand_sign, pauli_circuit = _rc._sample_random_pauli(
        n=n, pspec=pspec, qubit_labels=qubit_labels, absolute_compilation=clifford_compilations,
        circuit=True, include_identity=False, seed=rand_state.randint(0, 2**31))

    # Every qubit starts in a uniformly random eigenstate bit, with the Pauli's sign the parity of
    # the bits on its support. An identity qubit gets |0> or |1> (an X half the time) and no random
    # Clifford, which would change the width and two-qubit gate density of the circuit, held fixed
    # for featuremetric benchmarking.
    bits = _sample_eigenstate_bits([p != 'I' for p in rand_pauli], rand_sign, seed=rand_state.randint(0, 2**31))
    s_inputstate, p_inputstate, s_init_layer, p_init_layer, prep_circuit = _rc._sample_stabilizer(
        rand_pauli, rand_sign, clifford_compilations, qubit_labels, randomize_for_identity=False, bits=bits)

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
    s_new_pauli, p_new_pauli = _symp.compose_cliffords(s_new_pauli, p_new_pauli, s_rc, p_rc)  # UPU^(-1)

    pauli_vector = p_new_pauli
    pauli = [i[0] for i in _symp.find_pauli_layer(pauli_vector, [j for j in range(n)])]
    measurement = ['I' if i == 'I' else 'Z' for i in pauli]

    # Turn the stabilizer into an all Z and I stabilizer. randomize_for_identity=False leaves the
    # qubits where the image Pauli is the identity idle, so that the width and two-qubit gate
    # density do not depend on the sampled Pauli.
    s_stab, p_stab, stab_circuit = _rc._stabilizer_to_all_zs(pauli, qubit_labels, clifford_compilations,
                                                             seed=rand_state.randint(0, 2**31),
                                                             randomize_for_identity=False)

    full_circuit.append_circuit_inplace(stab_circuit)

    s_outputstate, p_outputstate = _symp.apply_clifford_to_stabilizer_state(s_stab, p_stab,
                                                                           s_outputstate, p_outputstate)

    full_circuit.done_editing()
    sign = _rc._determine_sign(s_outputstate, p_outputstate, measurement)

    if reference is None:
        return full_circuit, measurement, sign

    # Drawn after every other random number, so the dressed circuit is the same with or without
    # a reference.
    ref_seed = rand_state.randint(0, 2**31)

    if reference == 'readout':
        # A uniformly random bitstring with the measured Z string's parity, then the measurement.
        # With the Z string as the Pauli, each qubit's layer is an X where its bit is 1.
        ref_bits = _sample_eigenstate_bits([m == 'Z' for m in measurement], sign, seed=ref_seed)
        _, _, _, _, ref_prep = _rc._sample_stabilizer(
            measurement, sign, clifford_compilations, qubit_labels, randomize_for_identity=False,
            bits=ref_bits)
        reference_circuit = ref_prep.copy(editable=True)
        reference_circuit.done_editing()
        return full_circuit, measurement, sign, reference_circuit

    # A stabilizer state of Q = C P C^-1, then the dressed circuit's final layer.
    ref_sign = 1
    for _ in range(2):
        ref_bits = _sample_eigenstate_bits([q != 'I' for q in pauli], ref_sign, seed=ref_seed)
        s_ref, p_ref, _, _, ref_prep = _rc._sample_stabilizer(
            pauli, ref_sign, clifford_compilations, qubit_labels, randomize_for_identity=False,
            bits=ref_bits)
        s_ref, p_ref = _symp.apply_clifford_to_stabilizer_state(s_stab, p_stab, s_ref, p_ref)
        if _rc._determine_sign(s_ref, p_ref, measurement) == sign:
            break
        # The final layer maps Q to the measured Z string with a fixed phase, so the opposite
        # prepared sign gives the opposite final sign.
        ref_sign = -1

    reference_circuit = ref_prep.copy(editable=True)
    reference_circuit.append_circuit_inplace(stab_circuit)
    reference_circuit.done_editing()

    return full_circuit, measurement, sign, reference_circuit

""" Clifford circuits with Pauli errors simulation functions """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import os as _os
import numpy as _np
import time as _time
from ...tools import symplectic as _symp
from ...objects.label import Label as _Lbl
from ... import objects as _obj
from . import sample as _samp


def random_paulierror_in_chp(q):
    """
    todo.
    """
    i = _np.random.randint(0, 3)

    if i == 0:
        return ('p ' + str(q) + '\n') * 2

    elif i == 1:
        return 'h ' + str(q) + '\n' + ('p ' + str(q) + '\n') * 2 + 'h ' + str(q) + '\n'

    else:
        return ('p ' + str(q) + '\n') * 2 + 'h ' + str(q) + '\n' + ('p ' + str(q) + '\n') * 2 + 'h ' + str(q) + '\n'


def random_pauli_in_chp(q):
    """
    todo.
    """
    i = _np.random.randint(0, 4)

    if i == 0:
        return ''

    elif i == 1:
        return ('p ' + str(q) + '\n') * 2

    elif i == 2:
        return 'h ' + str(q) + '\n' + ('p ' + str(q) + '\n') * 2 + 'h ' + str(q) + '\n'

    else:
        return ('p ' + str(q) + '\n') * 2 + 'h ' + str(q) + '\n' + ('p ' + str(q) + '\n') * 2 + 'h ' + str(q) + '\n'


def stdgate_to_chp(gate, chpqubits):
    """
    todo
    Converts any of the standard Clifford gates to a chp string.
    """
    gatestr = str(gate).split(':')
    name = gatestr[0]
    qubits = [chpqubits[q] for q in gatestr[1:]]

    if name == 'Gi':
        return ''

    elif name == 'Gxpi':
        s = 'h ' + str(qubits[0]) + '\n'
        s += ('p ' + str(qubits[0]) + '\n') * 2
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gypi':
        s = ('p ' + str(qubits[0]) + '\n') * 2
        s += 'h ' + str(qubits[0]) + '\n'
        s += ('p ' + str(qubits[0]) + '\n') * 2
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gzpi':
        return ('p ' + str(qubits[0]) + '\n') * 2

    elif name == 'Gxpi2':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gxmpi2':
        s = 'h ' + str(qubits[0]) + '\n'
        s += ('p ' + str(qubits[0]) + '\n') * 3
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gzpi2' or name == 'Gp':
        return 'p ' + str(qubits[0]) + '\n'

    elif name == 'Gzmpi2' or name == 'Gpdag':
        return ('p ' + str(qubits[0]) + '\n') * 3

    elif name == 'Gh':
        return ('h ' + str(qubits[0]) + '\n')

    elif name == 'Gc0':
        return ''

    elif name == 'Gc1':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc2':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc3':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc4':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc5':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc6':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc7':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc8':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc9':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc10':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc11':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc12':
        s = 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc13':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc14':
        s = 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc15':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc16':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc17':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc18':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc19':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc20':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc21':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc22':
        s = 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'h ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gc23':
        s = 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        s += 'p ' + str(qubits[0]) + '\n'
        return s

    elif name == 'Gcnot':
        return 'c ' + str(qubits[0]) + ' ' + str(qubits[1]) + '\n'

    elif name == 'Gcphase':
        s = 'h ' + str(qubits[1]) + '\n'
        s += 'c ' + str(qubits[0]) + ' ' + str(qubits[1]) + '\n'
        s = 'h ' + str(qubits[1]) + '\n'
        return s

    else:
        raise ValueError("{} is an unknown gate! You must make your own `gateinchp` function!".format(gate))


class IndDepolErrorModel(object):
    """
    todo

    """

    def __init__(self, gate_errors, readout_errors):
        """
        todo

        """
        self.gate_errors = gate_errors
        self.readout_errors = readout_errors

    def layer_uniform_pauli_probability(self, layer, qubitorder):
        """
        todo.

        """
        if len(layer) > 0:
            return 1 - _np.prod([1 - _np.array([4 * self.gate_errors[gate].get(q, 0) / 3
                                                for q in qubitorder]) for gate in layer], axis=0)
        else:
            return _np.zeros(len(qubitorder), float)

    def readout_uniform_pauli_probability(self, qubitorder):
        """
        todo

        """
        return 1 - _np.prod([1 - _np.array([4 * self.readout_errors[q1].get(q2, 0) / 3
                                            for q2 in qubitorder]) for q1 in qubitorder], axis=0)


def depolarizing_errors_circuit_simulator(circuitlist, shots, errormodel, gate_to_chp=None,
                                          aux_info_list=None, collision_action='keepseparate',
                                          outdir='', perge_chp_files=True, returnds=True,
                                          verbosity=1):
    """
    todo.

    """
    if returnds:
        ds = _obj.DataSet(collision_action=collision_action)
    else:
        ds = []
    assert(_os.path.isfile("chp")), "This simulator uses the chp.c code.\n" + \
        "It must be compiled to an executable called `chp` and situated in this folder!"

    try:
        _os.mkdir(outdir)
        if perge_chp_files:
            perge_dir = True
        else:
            perge_dir = False
    except:
        perge_dir = False
        pass

    time0 = _time.time()

    if gate_to_chp is None: gate_to_chp = stdgate_to_chp

    percentdone = 0
    for cind, circuit in enumerate(circuitlist):
        print(cind)
        time1 = _time.time()

        if verbosity > 0:
            if verbosity > 1:
                print("{0:.2f}".format(cind / len(circuitlist)), end=' ')
            else:
                if int(_np.floor(cind / len(circuitlist))) > percentdone:
                    percentdone += 1
                    print("  - Simulation {} percent complete.".format(percentdone))

        # Todo : this is a temp hack to get around a bug in Circuit.
        if circuit[-1].name[0] == '#':
            circuit = circuit.copy(editable=True)
            circuit.delete_layers(-1)
            circuit.delete_lines('*')

        n = circuit.number_of_lines()
        depth = circuit.depth()

        # Set up the CHP qubit labels: could be different CHP labels for each circuit.
        aschpq = {label: str(i) for i, label in enumerate(circuit.line_labels)}

        # A list containing the CHP string for each error-free circuit layer.
        perfect_chpstrings = [''.join([gate_to_chp(gate, aschpq) for gate in circuit.layer(i)])
                              for i in range(depth)]

        # Find the probability of error in each circuit layer.
        errorprobs = [errormodel.layer_uniform_pauli_probability(circuit.layer(lind), circuit.line_labels)
                      for lind in range(circuit.depth())]
        # Add measurement error at the end
        errorprobs.append(errormodel.readout_uniform_pauli_probability(circuit.line_labels))

        time2 = _time.time()

        for sample in range(shots):

            # Sample errors for the circuit. Note that if 1 then a uniformly random Pauli is sampled, so
            # there is a 1/4 chance of no error even if this is 1. This is correct.
            #print(errorprobs)
            haserror = [_np.random.binomial(1, ep) for ep in errorprobs]
            #for lind in range(depth):
            #    print('-', lind)
            #    print(haserror[lind])
            #    print([random_pauli_in_chp(q) for q in range(n) if haserror[lind][q] == 1])
            # Construct the CHP string for each error layer.
            error_chpstrings = [''.join([''] + [random_pauli_in_chp(q) for q in range(n) if haserror[lind][q] == 1])
                                for lind in range(depth)]

            # Interleave the perfect and error CHP strings and then join.
            chpstring = '#\n' + ''.join([val for pair in zip(perfect_chpstrings, error_chpstrings) for val in pair])

            # Add the readout error
            chpstring += ''.join([''] + [random_pauli_in_chp(q) for q in range(n) if haserror[depth][q] == 1])

            # Add a measurement on all the qubits.
            chpstring += '\n'.join(['m ' + aschpq[q] for q in circuit.line_labels]) + '\n'
            #print(chpstring)
            with open(outdir + "/circuit-{}-instance-{}.chp".format(cind, sample), 'w') as f:
                f.write(chpstring)

            # Run CHP on this file.
            _os.system("./chp " + outdir + "/circuit-{}-instance-{}.chp > ".format(cind, sample)
                       + outdir + "/circuit-{}-instance-{}-out.txt".format(cind, sample))

        countdict = {}
        for sample in range(shots):

            with open(outdir + "/circuit-{}-instance-{}-out.txt".format(cind, sample), 'r') as f:
                #print(cind,sample)
                outasdict = {}
                for i, line in enumerate(f):
                    if i > 3:
                        line = line.strip(' \n')
                        line = line.split(' ')
                        # todo : this assumes definite outcome circuits, so fix that
                        # by instead counting forward from the start of the line.
                        outasdict[circuit.line_labels[int(line[-2][:-1])]] = line[-1]

                #print(outasdict)
                bitstring = ''.join([outasdict[q] for q in circuit.line_labels])

            if perge_chp_files:
                _os.system("rm " + outdir + "/circuit-{}-instance-{}.chp".format(cind, sample))
                _os.system("rm " + outdir + "/circuit-{}-instance-{}-out.txt".format(cind, sample))

            try:
                countdict[bitstring] += 1
            except:
                countdict[bitstring] = 1

            if aux_info_list is not None:
                aux = aux_info_list[cind]
            else:
                aux = None

        if returnds:
            ds.add_count_dict(circuit, countdict, record_zero_counts=False, aux=aux)
        else:
            ds.append(countdict)

        time3 = _time.time()

        if verbosity > 1:
            print("({0:.2f}, {1:.2f})".format(time2 - time1, time3 - time2), end=', ')
    print("Total time: {0:.2f})".format(time3 - time0))

    if perge_dir:
        _os.system("rmdir " + outdir)

    return ds


def circuit_simulator_for_tensored_independent_pauli_errors(circuit, pspec, errormodel, counts,
                                                            alloutcomes=False, idle1q_placeholder='I'):
    """
    A Clifford circuit simulator for an error model whereby each gate in the circuit induces independent Pauli
    errors on some or all of the qubits, with user-specified error probabilities that can vary between gate
    and between Pauli. State preparation and measurements errors are restricted to bit-flip errors on the output.

    This simulator is a stochastic-unravelling simulator that uses an efficient-in-qubit-number representation
    of the action of Clifford gates on Paulis. Specifically, it samples Pauli errors according to the error
    statistics provided, and propogates them through the layers of Clifford gates in the circuit using the
    conjugation action of the Cliffords on Paulis (as represented by 2n X 2n symplectic matrices for n qubits).
    This is repeated for the number of counts (`counts`) requested. So, this function takes a time to run that
    scales as (counts * n^2 * circuit depth). Therefore, this method will be slower than the pyGSTi density-matrix
    simulators at low qubit number and high `counts`.

    Parameters
    ----------
    circuit : Circuit
        The circuit to simulate. It should only contain gates that are also contained  within the provided
        ProcessorSpec `pspec` and are Clifford gates.

    pspec : ProcessorSpec
        The ProcessorSpec that defines the device. The Clifford model in ProcessorSpec should contain all of
        the gates that are in the circuit.

    errormodel : dict
        A dictionary defining the error model. This errormodel should have keys that are Label objects (the
        elements of the circuit). The values for a particular Label is an (n,4) numpy array of floats, that
        encodes the errors caused by the gate specified by that Label. The (i,j) value in the array is the
        probability that this gate is followed by Pauli i where Pauli 0 = identity, Pauli 1 = X, Pauli 2 = Y
        and Pauli 3 = Z. So, if the arrray is [1.,0.,0.,0.] in every row then there is no errors, if it is
        [1-p,p/3,p/3,p/3] in row j then there is equal probability of each Pauli error on qubit j with an
        error probability of p.

        Some simple error models can be auto-constructed using `create_locally_gate_independent_pauli_error_model()`
        or create_iid_pauli_error_model()`.

    counts : The number of counts, i.e., the number of repeats of the circuit that data should be generated for.

    alloutcomes : bool, optional
        If True then a dictionary is returned where the keys are all possible outcomes (i.e., all length n
        bit strings) and the values are the counts for all of the outcomes. If False, then the returned
        dictionary only contains keys for those outcomes that happen at least once.

    TODO: docstring: idle1q_placeholder

    Returns
    -------
    dict
        A dictionary of simulated measurement outcome counts.
    """
    n = circuit.number_of_lines()
    #TODO REMOVE
    #if circuit.identity != idle_name:
    #    circuit.replace_gatename(circuit.identity,idle_name)

    if set(circuit.line_labels) != set(pspec.qubit_labels):
        assert(set(circuit.line_labels).issubset(set(pspec.qubit_labels)))
        reduced_errormodel = errormodel.copy()
        mask = _np.zeros(pspec.number_of_qubits, bool)
        for i in range(pspec.number_of_qubits):
            if pspec.qubit_labels[i] in circuit.line_labels:
                mask[i] = True
        for key in list(reduced_errormodel.keys()):
            errormatrix = reduced_errormodel[key]
            assert(_np.shape(errormatrix)[0] == pspec.number_of_qubits), "Format of `errormodel` incorrect!"
            if len(_np.shape(errormatrix)) == 2:
                reduced_errormodel[key] = errormatrix[mask, :]
            elif len(_np.shape(errormatrix)) == 1:
                reduced_errormodel[key] = errormatrix[mask]
            else: raise ValueError("Format of `errormodel` incorrect!")
    else:
        reduced_errormodel = errormodel

    results = {}

    if alloutcomes:
        for i in range(2**n):
            result = tuple(_symp.int_to_bitstring(i, n))
            results[result] = 0

    for i in range(0, counts):
        result = oneshot_circuit_simulator_for_tensored_independent_pauli_errors(
            circuit, pspec, reduced_errormodel, idle1q_placeholder)
        try: results[result] += 1
        except: results[result] = 1

    return results


def oneshot_circuit_simulator_for_tensored_independent_pauli_errors(circuit, pspec, errormodel, idle1q_placeholder='I'):
    """
    Generates a single measurement result for the `circuit_simulator_for_tensored_independent_pauli_errors()`
    simulator

    Parameters
    ----------
    circuit : Circuit
        The circuit to simulate. It should only contain gates that are also contained  within the provided
        ProcessorSpec `pspec` and are Clifford gates.

    pspec : ProcessorSpec
        The ProcessorSpec that defines the device. The Clifford model in ProcessorSpec should contain all of
        the gates that are in the circuit.

    errormodel : dict
        A dictionary defining the error model. This errormodel should have keys that are Label objects (the
        elements of the circuit). The values for a particular Label is an (n,4) numpy array of floats, that
        encodes the errors caused by the gate specified by that Label. The (i,j) value in the array is the
        probability that this gate is followed by Pauli i where Pauli 0 = identity, Pauli 1 = X, Pauli 2 = Y
        and Pauli 3 = Z. So, if the arrray is [1.,0.,0.,0.] in every row then there is no errors, if it is
        [1-p,p/3,p/3,p/3] in row j then there is equal probability of each Pauli error on qubit j with an
        error probability of p.

    TODO: docstring: idle1q_placeholder

    Returns
    -------
    tuple
        A tuple of values that are 0 or 1, corresponding to the results of a z-measurement on all the qubits.
        The ordering of this tuple corresponds to the ordering of the wires in the circuit.
    """
    n = circuit.number_of_lines()
    depth = circuit.depth()
    sout, pout = _symp.prep_stabilizer_state(n, zvals=None)
    srep = pspec.models['clifford'].compute_clifford_symplectic_reps()
    I = _np.identity(2 * n, int)

    for l in range(depth):

        layer = circuit.layer_with_idles(l, idle_gate_name=idle1q_placeholder)
        s, p = _symp.symplectic_rep_of_clifford_layer(layer, n, q_labels=circuit.line_labels, srep_dict=srep)
        # Apply the perfect layer to the current state.
        sout, pout = _symp.apply_clifford_to_stabilizer_state(s, p, sout, pout)

        # Consider each gate in the layer, and apply Pauli errors with the relevant probs.
        for gate in layer:
            # Sample a pauli vector for the gate
            gerror_p = _np.zeros(2 * n, int)
            sampledvec = _np.array([list(_np.random.multinomial(1, pp)) for pp in errormodel[gate]])
            # Z and Y both map X - > -X under conjugation, which is encoded with the upper half of
            # the p vector being set to 2.
            gerror_p[:n] = 2 * (sampledvec[:, 3] ^ sampledvec[:, 2])
            # X and Y both map Z - > -Z under conjugation, which is encoded with the lower half of
            # the p vector being set to 2.
            gerror_p[n:] = 2 * (sampledvec[:, 1] ^ sampledvec[:, 2])

            sout, pout = _symp.apply_clifford_to_stabilizer_state(I, gerror_p, sout, pout)

    output = []
    for q in range(n):
        measurement_out = _symp.pauli_z_measurement(sout, pout, q)
        # The probability of the '1' outcome
        oneprob = measurement_out[1]
        # Sample a bit with that probability to be 1.
        bit = _np.random.binomial(1, oneprob)
        output.append(bit)

    # Add measurement errors, by bit-flipping with some probability
    try:
        measurement_errors = errormodel['measure']
    except:
        measurement_errors = [0 for i in range(n)]

    add_to_outcome = _np.array([_np.random.binomial(1, p) for p in measurement_errors])
    output = tuple(_np.array(output) ^ add_to_outcome)
    outputasstring = ''
    for s in output: outputasstring += str(s)
    return outputasstring


def rb_with_pauli_errors(pspec, errormodel, lengths, k, counts, qubit_subset=None, filename=None, rbtype='DRB',
                         rbspec=[], returndata=True, appenddata=False, verbosity=0, idle1q_placeholder='I'):
    """
    Simulates RB with Pauli errors. Can be used to simulated Clifford RB, direct RB and mirror RB. This
    function:

        1) Samples RB circuits
        2) Simulates the RB circuit with the specified Pauli-errors error model
        3) Records the summary RB data to file and/or returns this RB data.

    Step 1 is implemented using the in-built RB samplers. For more information see rb.sample. Step 2
    is implemented using the `circuit_simulator_for_tensored_independent_pauli_errors()` stochastic
    errors circuit simulator. See that function for more details.

    Parameters
    ----------
    pspec : ProcessorSpec
        The ProcessorSpec that defines the device.

    errormodel : dict
        A dictionary defining the error model. This errormodel should have keys that are Label objects
        corresponding to the gates in `pspec`. The values for a particular Label is an (n,4) numpy array of
        floats, that encodes the errors caused by the gate specified by that Label. The (i,j) value in the
        array is the probability that this gate is followed by Pauli i where Pauli 0 = identity, Pauli 1 = X,
        Pauli 2 = Y and Pauli 3 = Z. So, if the arrray is [1.,0.,0.,0.] in every row then there is no errors,
        if it is [1-p,p/3,p/3,p/3] in row j then there is equal probability of each Pauli error on qubit j with an
        error probability of p.

        Some simple error models can be auto-constructed using `create_locally_gate_independent_pauli_error_model()`
        or create_iid_pauli_error_model()`.

    lengths : list
        A list of the RB lengths to sample and simulate circuits at. E.g., for Clifford RB this is the number
        of Cliffords in the uncompiled circuit - 2 (see `rb.sample.clifford_rb_circuit()`).

    k : int
        The number of circuits to sample and simulate at each RB length.

    counts : int
        The number of counts for each circuit.

    qubit_subset : list
        If not None, a list of qubit labels that the RB experiment should be over, that is a subset of the
        qubits in `pspec`.

    filename : str, optional
        A filename for where to save the data (if None, the data is not saved to file).

    rbtype : {'DRB', 'CRB', 'MRB'}
        The RB type to simulate. 'DRB' corresponds to direct RB, 'CRB' corresponds to Clifford RB,
        and 'MRB' corresponds to mirror RB.

    rbspec : list, optional
        Handed to the RB sampling function for all arguments after `pspec` and the RB lengths, which are the first
        two arguments handed to the relevant function. See the relevant RB circuit sampling functions for details.

    returndata : bool, optional
        Whether to return the data

    appenddata : bool, optional
        If writing to file (i.e., `filename` is not None), whether to append the data to an already existing file
        or to write over any existing file.

    verbosity : int, optional
        The amount of print-to-screen.

    Returns
    -------
    None or RBSummaryDataset
        If `returndata` an RBSummaryDataset containing the results. Else, None

    """
    assert(rbtype == 'CRB' or rbtype == 'DRB' or rbtype == 'MRB'), "RB type not valid!"

    if filename is not None:
        if not appenddata:
            with open(filename, 'w') as f:
                f.write('# Results from a {} simulation\n'.format(rbtype))
                f.write('# Number of qubits\n')
                if qubit_subset is None: f.write(str(pspec.number_of_qubits))
                else: f.write(str(len(qubit_subset)))
                f.write('\n# RB length // Success counts // Total counts '
                        '// Circuit depth // Circuit two-qubit gate count\n')

    n = pspec.number_of_qubits
    lengthslist = []
    scounts = []
    cdepths = []
    c2Qgcounts = []

    for i in range(k):

        if verbosity > 0:
            print("- Sampling and simulating circuit {} of {} at each of {} lengths".format(i + 1, k, len(lengths)))
            print("  - Number of circuits complete = ", end='')

        for lind, l in enumerate(lengths):

            lengthslist.append(l)

            if rbtype == 'DRB':
                c, idealout = _samp.direct_rb_circuit(pspec, l, qubit_subset, *rbspec)
            elif rbtype == 'CRB':
                c, idealout = _samp.clifford_rb_circuit(pspec, l, qubit_subset, *rbspec)
            elif rbtype == 'MRB':
                c, idealout = _samp.mirror_rb_circuit(pspec, l, qubit_subset, *rbspec)

            #if verbosity > 0:
            #    print(" complete")
            #    print(" - Simulating circuit...",end='')

            outcome = circuit_simulator_for_tensored_independent_pauli_errors(
                c, pspec, errormodel, counts, alloutcomes=False, idle1q_placeholder=idle1q_placeholder)
            #EGN: Hardcoded 'I' here. Could make this into an arg, but there's really
            #  no need for the user to modify this unless they use 'I' as a gate label.
            if verbosity > 0: print(lind + 1, end=',')

            # Add the number of success counts to the list
#            scounts.append(outcome.get(idealout,0))
            scounts.append(outcome.get(''.join(str(idealbit) for idealbit in idealout), 0))
            cdepths.append(c.depth())
            c2Qgcounts.append(c.two_q_gate_count())

            # Write the data to file in each round.
            if filename is not None:
                with open(filename, 'a') as f:
                    f.write('{} {} {} {} {}\n'.format(l, scounts[-1], counts, cdepths[-1], c2Qgcounts[-1]))

        if verbosity > 0: print('')
    if returndata:

        from . import results as _res
        data = _res.RBSummaryDataset(n, lengthslist, success_counts=scounts, total_counts=counts,
                                     circuit_depths=cdepths, circuit_twoQgate_counts=c2Qgcounts)
        return data


def create_iid_pauli_error_model(pspec, one_qubit_gate_errorrate, two_qubit_gate_errorrate, idle_errorrate,
                                 measurement_errorrate=0., ptype='uniform', idle1q_placeholder='I'):
    """
    Returns a dictionary encoding a Pauli-stochastic error model whereby the errors are the same on all the
    1-qubit gates, and the same on all 2-qubit gates. The probability of the 3 different Pauli errors on each
    qubit is specified by `ptype` and can either be uniform, or always X, Y, or Z errors.

    The dictionary returned is in the appropriate format for the
    `circuit_simulator_for_tensored_independent_pauli_errors()` circuit simulator function.

    Parameters
    ----------
    pspec : ProcessorSpec
        The ProcessorSpec that defines the device.

    one_qubit_gate_errorrate : float
        The 1-qubit gates error rate (the probability of a Pauli error on the target qubit) not including
        idle gates.

    two_qubit_gate_errorrate : float
        The 2-qubit gates error rate (the total probability of a Pauli error on either qubit the gate acts
        on -- each qubit has independent errors with equal probabilities).

    idle_errorrate : float
        The idle gates error rate.

    measurement_errorrate : flip
        The measurement error rate for all of the qubits. This is the probability that a qubits measurement
        result is bit-flipped.

    ptype : str, optional
        Can be 'uniform', 'X', 'Y' or 'Z'. If 'uniform' then 3 Pauli errors are equally likely, if 'X', 'Y' or
        'Z' then the errors are always Pauli X, Y or Z errors, respectively.

    TODO: docstring idle1q_placeholder

    Returns
    -------
    dict
        An dict that encodes the error model described above in the format required for the simulator
        `circuit_simulator_for_tensored_independent_pauli_errors()`.

    """
    if ptype == 'uniform':
        def error_row(er): return _np.array([1 - er, er / 3, er / 3, er / 3])

    elif ptype == 'X':
        def error_row(er): return _np.array([1 - er, er, 0., 0.])

    elif ptype == 'Y':
        def error_row(er): return _np.array([1 - er, 0., er, 0.])

    elif ptype == 'Z':
        def error_row(er): return _np.array([1 - er, 0., 0., er])
    else:
        raise ValueError("Error model type not understood! Set `ptype` to a valid option.")

    perQ_twoQ_errorrate = 1 - (1 - two_qubit_gate_errorrate)**(1 / 2)
    n = pspec.number_of_qubits

    errormodel = {}

    if idle1q_placeholder is not None:
        #Added by EGN: special behavior needed when Model has
        # an gate name used to designate a perfect 1-qubit idle op (used as placeholder).
        # This translates to a set of "<gatename>:X" operation labels all w/idle_errorrate
        idleLbl = idle1q_placeholder
        for q in pspec.qubit_labels:
            gl = _Lbl(idleLbl, q)
            errormodel[gl] = _np.zeros((n, 4), float)
            errormodel[gl][:, 0] = _np.ones(n, float)
            errormodel[gl][pspec.qubit_labels.index(q), :] = error_row(idle_errorrate)

    for gate in pspec.models['clifford'].primitive_op_labels():
        errormodel[gate] = _np.zeros((n, 4), float)
        errormodel[gate][:, 0] = _np.ones(n, float)

        # If not a CNOT, it is a 1-qubit gate / idle.
        if gate.number_of_qubits == 2:
            q1 = gate.qubits[0]
            q2 = gate.qubits[1]
            er = perQ_twoQ_errorrate
            errormodel[gate][pspec.qubit_labels.index(q1), :] = error_row(er)
            errormodel[gate][pspec.qubit_labels.index(q2), :] = error_row(er)

        elif gate.number_of_qubits == 1:
            q = gate.qubits[0]

            # If the idle gate, use the idle error rate
            #TODO REMOVE if gate.name == pspec.identity: er = idle_errorrate
            er = one_qubit_gate_errorrate

            errormodel[gate][pspec.qubit_labels.index(q), :] = error_row(er)

        else:
            raise ValueError("The ProcessorSpec must only contain 1- and 2- qubit gates!")

    errormodel['measure'] = _np.array([measurement_errorrate for q in range(n)])

    return errormodel


def create_locally_gate_independent_pauli_error_model(pspec, gate_errorrate_dict, measurement_errorrate_dict={},
                                                      ptype='uniform', idle1q_placeholder='I'):
    """
    Returns a dictionary encoding a Pauli-stochastic error model whereby the errors are independent of the gates,
    with a qubit subject to an error after a circuit layer with the probabilities specified by the dict
    `gate_errorrate_dict`. The probability of the 3 different Pauli errors on each qubit is specified by
    `ptype` and can either be uniform, or always X, Y, or Z errors.

    The dictionary returned is in the appropriate format for the
    `circuit_simulator_for_tensored_independent_pauli_errors()` circuit simulator function.

    Parameters
    ----------
    pspec : ProcessorSpec
        The ProcessorSpec that defines the device.

    gate_errorrate_dict : dict
        A dict where the keys are elements of pspec.qubit_labels and the values are floats in [0,1].
        The element for qubit with label `q` is the error probability for that qubit.

    measurement_errorrate_dict : dict
        A dict where the keys are elements of pspec.qubit_labels and the values are floats in [0,1].
        The element for qubit with label `q` is the measurement bit-flip error probability for that qubit.
        All qubits that do not have a measurement error rate specified are assumed to have perfect measurements.

    ptype : str, optional
        Can be 'uniform', 'X', 'Y' or 'Z'. If 'uniform' then 3 Pauli errors are equally likely, if 'X', 'Y' or
        'Z' then the errors are always Pauli X, Y or Z errors, respectively.

    TODO: docstring: idle1q_placeholder

    Returns
    -------
    dict
        An dict that encodes the error model described above in the format required for the simulator
        `circuit_simulator_for_tensored_independent_pauli_errors()`.

    """
    if ptype == 'uniform':
        def error_row(er): return _np.array([1 - er, er / 3, er / 3, er / 3])

    elif ptype == 'X':
        def error_row(er): return _np.array([1 - er, er, 0., 0.])

    elif ptype == 'Y':
        def error_row(er): return _np.array([1 - er, 0., er, 0.])

    elif ptype == 'Z':
        def error_row(er): return _np.array([1 - er, 0., 0., er])
    else:
        raise ValueError("Error model type not understood! Set `ptype` to a valid option.")

    n = pspec.number_of_qubits

    errormodel = {}

    if idle1q_placeholder is not None:
        #Added by EGN: special behavior needed when Model has
        # an gate name used to designate a perfect 1-qubit idle op (used as placeholder).
        # This translates to a set of "<gatename>:X" operation labels all w/appropriate errorrate
        idleLbl = idle1q_placeholder
        for q in pspec.qubit_labels:
            gl = _Lbl(idleLbl, q)
            er = gate_errorrate_dict[q]
            errormodel[gl] = _np.zeros((n, 4), float)
            errormodel[gl][:, 0] = _np.ones(n, float)
            errormodel[gl][pspec.qubit_labels.index(q), :] = error_row(er)

    for gate in pspec.models['clifford'].primitive_op_labels():
        errormodel[gate] = _np.zeros((n, 4), float)
        errormodel[gate][:, 0] = _np.ones(n, float)

        for q in gate.qubits:
            er = gate_errorrate_dict[q]
            errormodel[gate][pspec.qubit_labels.index(q)] = error_row(er)

    errormodel['measure'] = _np.array([measurement_errorrate_dict.get(q, 0.) for q in pspec.qubit_labels])

    return errormodel

#
# TODO : DOES THIS NEED AND IDLE PLACEHOLDER?
#


def create_local_pauli_error_model(pspec, one_qubit_gate_errorrate_dict, two_qubit_gate_errorrate_dict,
                                   measurement_errorrate_dict={}, ptype='uniform'):
    """
    Returns a dictionary encoding a Pauli-stochastic error model whereby the errors caused by a gate act
    only on the "target" qubits of the gate, all the 1-qubit gates on a qubit have the same error rate,
    and all the 2-qubit gates on a qubit have the same error rate. The probability of the 3 different Pauli
    errors on each qubit is specified by `ptype` and can either be uniform, or always X, Y, or Z errors.

    The dictionary returned is in the appropriate format for the
    `circuit_simulator_for_tensored_independent_pauli_errors()` circuit simulator function.

    Parameters
    ----------
    pspec : ProcessorSpec
        The ProcessorSpec that defines the device.

    one_qubit_gate_errorrate_dict : dict
        A dict where the keys are elements of pspec.qubit_labels and the values are floats in [0,1].
        The element for qubit with label `q` is the error probability for all 1-qubit gates on that qubit

    two_qubit_gate_errorrate_dict : dict
        A dict where the keys are 2-qubit gates in pspec and the values are floats in [0,1]. This is the
        error probability for the 2-qubit gate, split evenly into independent Pauli errors on each of the
        qubits the gate is intended to act on.

    measurement_errorrate_dict : dict
        A dict where the keys are elements of pspec.qubit_labels and the values are floats in [0,1].
        The element for qubit with label `q` is the measurement bit-flip error probability for that qubit.
        All qubits that do not have a measurement error rate specified are assumed to have perfect measurements.


    ptype : str, optional
        Can be 'uniform', 'X', 'Y' or 'Z'. If 'uniform' then 3 Pauli errors are equally likely, if 'X', 'Y' or
        'Z' then the errors are always Pauli X, Y or Z errors, respectively.

    Returns
    -------
    dict
        An dict that encodes the error model described above in the format required for the simulator
        `circuit_simulator_for_tensored_independent_pauli_errors()`.

    """
    if ptype == 'uniform':
        def error_row(er): return _np.array([1 - er, er / 3, er / 3, er / 3])

    elif ptype == 'X':
        def error_row(er): return _np.array([1 - er, er, 0., 0.])

    elif ptype == 'Y':
        def error_row(er): return _np.array([1 - er, 0., er, 0.])

    elif ptype == 'Z':
        def error_row(er): return _np.array([1 - er, 0., 0., er])
    else:
        raise ValueError("Error model type not understood! Set `ptype` to a valid option.")

    n = pspec.number_of_qubits

    errormodel = {}
    for gate in list(pspec.models['clifford'].primitive_op_labels()):
        errormodel[gate] = _np.zeros((n, 4), float)
        errormodel[gate][:, 0] = _np.ones(n, float)

        if gate.number_of_qubits == 1:
            er = one_qubit_gate_errorrate_dict[gate.qubits[0]]
        elif gate.number_of_qubits == 2:
            er = 1 - (1 - two_qubit_gate_errorrate_dict[gate])**(0.5)
        else: raise ValueError("Only 1- and 2-qubit gates supported!")

        for q in gate.qubits:
            errormodel[gate][pspec.qubit_labels.index(q)] = error_row(er)

    errormodel['measure'] = _np.array([measurement_errorrate_dict.get(q, 0.) for q in pspec.qubit_labels])

    return errormodel

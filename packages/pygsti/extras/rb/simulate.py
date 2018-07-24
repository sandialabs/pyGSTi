""" Clifford circuits with Pauli errors simulation functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
from ...tools import symplectic as _symp
from . import sample as _samp
from . import results as _res

def circuit_simulator_for_tensored_independent_pauli_errors(circuit, pspec, errormodel, counts, alloutcomes=False, 
                                                            idle_name='Gi'):
    """
    A Clifford circuit simulator for an error model whereby each gate in the circuit induces independent Pauli 
    errors on some or all of the qubits, with user-specified error probabilities that can vary between gate 
    and between Pauli. State preparation and measurements errors are restricted to bit-flip errors on the output.

    This simulator is a stochastic-unravelling simulator that uses an efficient-in-qubit-number representation
    of the action of Clifford gates on Paulis. Specifically, it samples Pauli errors according to the error
    statistics provided, and propogates them through the layers of Clifford gates in the circuit using the 
    conjugation action of the Cliffords on Paulis (as represented by 2n X 2n symplectic matrices for n qubits). 
    This is repeated for the number of counts (`counts`) requested. So, this function takes a time to run that 
    scales as (counts X n^2 X circuit depth). Therefore, this method will be slower than the pyGSTi density-matrix 
    simulators at low n and high counts.

    Parameters
    ----------
    circuit : Circuit
        A Circuit object that is the circuit to simulate. It should only contain gates that are also contained 
        within the provided ProcessorSpec `pspec` and are Clifford gates (except that it may also contain the 
        circuit default idle gate circuit.identity, by default named "I").

    pspec : ProcessorSpec
        The ProcessorSpec that defines the device. The Clifford gateset in ProcessorSpec should contain all of 
        the gates that are in the circuit.

    errormodel : dict
        A dictionary defining the error model. This errormodel should have keys that are Label objects (the 
        elements of the circuit). The values for a particular Label is an (n,4) numpy array of floats, that
        encodes the errors caused by the gate specified by that Label. The (i,j) value in the array is the
        probability that this gate is followed by Pauli i where Pauli 0 = identity, Pauli 1 = X, Pauli 2 = Y
        and Pauli 3 = Z. So, if the arrray is [1.,0.,0.,0.] in every row then there is no errors, if it is
        [1-p,p/3,p/3,p/3] in row j then there is equal probability of each Pauli error on qubit j with an 
        error probability of p.

    counts : The number of counts, i.e., the number of repeats of the circuit that data should be generated for.

    alloutcomes : bool, optional
        If True then a dictionary is returned where the keys are all possible outcomes (i.e., all length n 
        bit strings) and the values are the counts for all of the outcomes. If False, then the returned
        dictionary only contains keys for those outcomes that happen at least once.

    idle_name, str, optional
        Todo: The name of the idle gate in the `pspec` with which 
    
    Returns
    -------
    dict
        A dictionary of simulated measurement outcome counts.
    
    """    
    n = circuit.number_of_lines
    # If the identity in the circuit isn't the idle_name, we replace those identities with idle_name
    if circuit.identity != idle_name:
        circuit.replace_gatename(circuit.identity,idle_name)
    results = {}
    
    if alloutcomes:
        for i in range(2**n):
            result = tuple(_symp.int_to_bitstring(i,n))
            results[result] = 0
 
    for i in range(0,counts):
        result = oneshot_circuit_simulator_for_tensored_independent_pauli_errors(circuit, pspec, errormodel)
        try:
            results[tuple(result)] += 1
        except:
            results[tuple(result)] = 1

    return results

def  oneshot_circuit_simulator_for_tensored_independent_pauli_errors(circuit, pspec, errormodel):
    
    n = circuit.number_of_lines
    depth = circuit.depth()
    sout, pout = _symp.prep_stabilizer_state(n, zvals=None)
    I = _np.identity(2*n,int)
    srep=pspec.models['clifford'].get_clifford_symplectic_reps()
    
    for l in range(0,depth):
        
        layer = circuit.get_circuit_layer(l)
        s, p = _symp.symplectic_rep_of_clifford_layer(layer,n,srep_dict=srep)        
        # Apply the layer
        sout, pout = _symp.apply_clifford_to_stabilizer_state(s, p, sout, pout)
        
        for q in range(0,n):
            gate = layer[q]

            # This stops us including multi-qubit gates more than once.
            if gate.qubits[0] == q: 
                    
                # Sample a pauli vector for the gate
                gerror_p = _np.zeros(2*n,int)
                sampledvec = _np.array([list(_np.random.multinomial(1,pp)) for pp in errormodel[gate]]) 
                # Z and Y both map X - > -X under conjugation, which is encoded with the upper half of
                # the p vector being set to 2.
                gerror_p[:n] = 2*(sampledvec[:,3] ^ sampledvec[:,2])
                # X and Y both map Z - > -Z under conjugation, which is encoded with the lower half of
                # the p vector being set to 2.
                gerror_p[n:] = 2*(sampledvec[:,1] ^ sampledvec[:,2])
                    
                sout, pout = _symp.apply_clifford_to_stabilizer_state(I, gerror_p, sout, pout)


    output = []
    for q in range(0,n):
        # This is the (0,1) outcome probability
        measurement_out = _symp.pauli_z_measurement(sout, pout, q)
        # Todo : make sure that this makes sense.
        # Sample 0/1 valued outcome from this probabilitiy
        # Todo : make this work with probabilistic measurement outcomes.
        bit = measurement_out[1]
        bit = _np.random.binomial(1,bit)
        output.append(bit)

    try:
        measurement_errors = errormodel['measure']
    except:
        measurement_errors = [0 for i in range(n)]

    add_to_outcome = _np.array([_np.random.binomial(1,p) for p in measurement_errors])
    output = list(_np.array(output) ^  add_to_outcome)

    return output

def rb_with_pauli_errors(pspec, errormodel, lengths, k, counts, filename=None, rbtype='DRB', rbspec =[], 
                         idle_name='Gi', returndata=True, appenddata=False, verbosity=0):
    """
    

    """      
    assert(rbtype == 'CRB' or rbtype == 'DRB' or rbtype == 'MRB'), "RB type not valid!"

    if filename is not None:    
        if not appenddata:
            with open(filename,'w') as f:
                f.write('# Results from a {} simulation\n'.format(rbtype))
                f.write('# Number of qubits\n')
                f.write(str(pspec.number_of_qubits))
                f.write('\n# RB length // Success counts // Total counts // Circuit depth // Circuit two-qubit gate count\n')
            
    n = pspec.number_of_qubits
    lengthslist = []
    scounts = []
    cdepths = []
    c2Qgcounts = []
    
    for i in range(k):   
        for l in lengths:

            lengthslist.append(l)

            if rbtype == 'DRB':
                c, idealout = _samp.direct_rb_circuit(pspec, l, *rbspec)
            if rbtype == 'CRB':
                c, idealout = _samp.clifford_rb_circuit(pspec, l, *rbspec)
            if rbtype == 'MRB':
                c, idealout = _samp.mirror_rb_circuit(pspec, l, *rbspec)

            outcome = circuit_simulator_for_tensored_independent_pauli_errors(c, pspec, errormodel, counts, 
                                                                          alloutcomes=False, idle_name=idle_name)

            # If idealout is a count that happens at least once, we find the number of counts
            try:
                scounts.append(outcome[tuple(idealout)])
            # If idealout does not happen, it won't be a key in the dict, so we manually append 0.
            except:
                scounts.append(0)

            cdepths.append(c.depth())
            c2Qgcounts.append(c.twoQgate_count())

            # Write the data to file in each round.
            if filename is not None:    
                with open(filename,'a') as f:
                    f.write('{} {} {} {} {}\n'.format(l,scounts[-1],counts,cdepths[-1],c2Qgcounts[-1]))
                
    if returndata:

        data = _res.RBSummaryDatset(n, lengthslist, successcounts=scounts, totalcounts=counts, circuitdepths=cdepths, 
                        circuit2Qgcounts=c2Qgcounts)
        return data

def create_iid_pauli_error_model(pspec, oneQgate_errorrate, twoQgate_errorrate, idle_errorrate,
                                 measurement_errorrate=0., ptype='uniform', idle_name='Gi'):
    """
    Returns a dictionary encoding a Pauli-stochastic error model whereby the errors are the same on all the
    1-qubit gates, and the same on all 2-qubit gates.
    `gate_errorrate_list`. The probability of the 3 different Pauli errors on each qubit is specified by 
    `ptype` and can either be uniform, or always X, Y, or Z errors.

    The dictionary returned is in the appropriate format for the `circuit_simulator_for_tensored_independent_pauli_errors()` 
    circuit simulator function.

    Parameters
    ----------
    pspec : ProcessorSpec
        The ProcessorSpec that defines the device.

    oneQgate_errorrate_list : float
        The 1-qubit gates error rate (the probability of a Pauli error on the target qubit) not including 
        idle gates. 

    twoQgate_errorrate_list : float
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

    idle_name : str, optional
        The name of the idle gate.
    
    Returns
    -------
    dict
        An dict that encodes the error model described above in the format required for the simulator
        `circuit_simulator_for_tensored_independent_pauli_errors()`.

    """
    if ptype == 'uniform':
        def error_row(er):
            return _np.array([1-er,er/3,er/3,er/3])

    elif ptype == 'X':
        def error_row(er):
            return _np.array([1-er,er,0.,0.])

    elif ptype == 'Y':
        def error_row(er):
            return _np.array([1-er,0.,er,0.])

    elif ptype == 'Z':
        def error_row(er):
            return _np.array([1-er,0.,0.,er])
    else:
        raise ValueError("Error model type not understood! Set `ptype` to a valid option.")

    perQ_twoQ_errorrate = 1 - (1-twoQgate_errorrate)**(1/2)
    n = pspec.number_of_qubits

    errormodel = {}
    for gate in list(pspec.models['clifford'].gates.keys()):
        errormodel[gate] = _np.zeros((n,4),float)
        errormodel[gate][:,0] = _np.ones(n,float)
    
        # If not a CNOT, it is a 1-qubit gate / idle.
        if gate.number_of_qubits == 2:
        # If the idle gate, use the idle error rate
            q1 = gate.qubits[0]
            q2 = gate.qubits[1]
            er = perQ_twoQ_errorrate
            errormodel[gate][q1,:] =  error_row(er)
            errormodel[gate][q2,:] =  error_row(er)

        elif gate.number_of_qubits == 1:
            q = gate.qubits[0]
            
            if gate.name == idle_name:
                er = idle_errorrate
            else:
                er = oneQgate_errorrate
            
            errormodel[gate][q,:] =  error_row(er)

        else:    
            raise ValueError("The ProcessorSpec must only contain 1- and 2- qubit gates!")
 
    errormodel['measure'] = [ measurement_errorrate for q in range(n)]

    return errormodel

def create_locally_gate_independent_pauli_error_model(pspec, gate_errorrate_list, measurement_errorrate_list=None, 
                                                      ptype='uniform'):
    """
    Returns a dictionary encoding a Pauli-stochastic error model whereby the errors are independent of the gates,
    with a qubit subject to an error after a circuit layer with the probabilities  specified by the list
    `gate_errorrate_list`. The probability of the 3 different Pauli errors on each qubit is specified by 
    `ptype` and can either be uniform, or always X, Y, or Z errors.

    The dictionary returned is in the appropriate format for the `circuit_simulator_for_tensored_independent_pauli_errors()` 
    circuit simulator function.

    Parameters
    ----------
    pspec : ProcessorSpec
        The ProcessorSpec that defines the device.

    gate_errorrate_list : list 
        A list of length pspec.number_of_qubits of floats in [0,1]. The qth element is the error probability
        for the qth qubit.

    measurement_errorrate_list : list
        A list of length pspec.number_of_qubits of floats in [0,1]. The qth element is the measurement bit-flip
        error probability for the qth qubit.

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
        def error_row(er):
            return _np.array([1-er,er/3,er/3,er/3])

    elif ptype == 'X':
        def error_row(er):
            return _np.array([1-er,er,0.,0.])

    elif ptype == 'Y':
        def error_row(er):
            return _np.array([1-er,0.,er,0.])

    elif ptype == 'Z':
        def error_row(er):
            return _np.array([1-er,0.,0.,er])
    else:
        raise ValueError("Error model type not understood! Set `ptype` to a valid option.")

    n = pspec.number_of_qubits
    if measurement_errorrate_list is None:
        measurement_errorrate_list = [0. for i in range(n)]

    errormodel = {}
    for gate in list(pspec.models['clifford'].gates.keys()):
        errormodel[gate] = _np.zeros((n,4),float)
        errormodel[gate][:,0] = _np.ones(n,float)
    
        for q in gate.qubits:
            er = gate_errorrate_list[q]
            errormodel[gate][q] =  error_row(er)
    if measurement_errorrate_list is not None:       
        errormodel['measure'] = measurement_errorrate_list
    else:
        errormodel['measure'] = [0 for q in range(n)]
    return errormodel

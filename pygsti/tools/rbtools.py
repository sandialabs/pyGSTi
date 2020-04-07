""" Tools for analyzing RB data"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np


def p_to_r(p, d, rtype='EI'):
    """
    Converts an RB decay constant (p) to the RB error rate (r), where
    p is (normally) obtained from fitting data to A + Bp^m. There are
    two 'types' of RB error rate corresponding to different rescalings
    of 1 - p. These are the entanglement infidelity (EI) type r and
    the average gate infidelity (AGI) type r. The EI-type r is given by:

    r =  (d^2 - 1)(1 - p)/d^2,

    where d is the dimension of the system (i.e., 2^n for n qubits).
    The AGI-type r is given by

    r =  (d - 1)(1 - p)/d.

    For RB on gates whereby every gate is followed by an n-qubit
    uniform depolarizing channel (the most idealized RB scenario)
    then the EI-type (AGI-type) r corresponds to the EI (AGI) of
    the depolarizing channel to the identity channel.

    The default (EI) is the convention used in direct RB, and is perhaps
    the most well-motivated as then r corresponds to the error probablity
    of the gates (in the idealized pauli-errors scenario). AGI is
    the convention used throughout Clifford RB theory.

    Parameters
    ----------
    p : float
        Fit parameter p from P_m = A + B*p**m.

    d : int
        Number of dimensions of the Hilbert space

    rtype : {'EI','AGI'}, optional
        The RB error rate rescaling convention.

    Returns
    -------
    r : float
        The RB error rate
    """
    if rtype == 'AGI': r = (1 - p) * (d - 1) / d
    elif rtype == 'EI': r = (d**2 - 1) * (1 - p) / d**2
    else:
        raise ValueError("rtype must be `EI` (for entanglement infidelity) or `AGI` (for average gate infidelity)")

    return r


def r_to_p(r, d, rtype='EI'):
    """
    Inverse of the p_to_r function.

    Parameters
    ----------
    r : float
        The RB error rate

    d : int
        Number of dimensions of the Hilbert space

    rtype : {'EI','AGI'}, optional
        The RB error rate rescaling convention.

    Returns
    -------
    p : float
        The RB decay constant
    """
    if rtype == 'AGI': p = 1 - d * r / (d - 1)
    elif rtype == 'EI': p = 1 - d**2 * r / (d**2 - 1)
    else:
        raise ValueError("rtype must be `EI` (for entanglement infidelity) or `AGI` (for average gate infidelity)")

    return p


def adjusted_success_probability(hamming_distance_pdf):
    """
    todo

    """
    #adjSP = _np.sum([(-1 / 2)**n * hamming_distance_counts[n] for n in range(numqubits + 1)]) / total_counts
    adjSP = _np.sum([(-1 / 2)**n * hamming_distance_pdf[n] for n in range(len(hamming_distance_pdf))])

    return adjSP


def marginalized_success_counts(dsrow, circ, target, qubits):
    """
    todo

    """
    if dsrow.total == 0:
        return 0
    else:
        # The rows of the circuit that we are interested in
        indices = [circ.line_labels.index(q) for q in qubits]
        # The ordering of this must be the same as what we compare it to.
        margtarget = ''.join([target[i] for i in indices])

        if qubits == circ.line_labels:
            try:
                return dsrow.counts[target]
            except:
                return 0

        else:

            success_counts = 0

            for (outbitstring,), counts in dsrow.counts.items():
                if ''.join([outbitstring[i] for i in indices]) == margtarget:
                    success_counts += counts

            return success_counts


def hamming_distance(bs1, bs2):
    """
    todo

    """
    return _np.sum([b1 != b2 for b1, b2 in zip(bs1, bs2)])


def marginalized_hamming_distance_counts(dsrow, circ, target, qubits):
    """
    todo

    """
    if dsrow.total == 0:
        hamming_distance_counts = [0 for i in range(len(qubits) + 1)]
    else:
        # The rows of the circuit that we are interested in
        indices = [circ.line_labels.index(q) for q in qubits]
        # The ordering of this must be the same as what we compare it to.
        margtarget = ''.join([target[i] for i in indices])

        hamming_distance_counts = _np.zeros(len(qubits) + 1, float)

        for (outbitstring,), counts in dsrow.counts.items():
            #print(outbitstring)
            hamming_distance_counts[hamming_distance(''.join([outbitstring[i] for i in indices]), margtarget)] += counts

        hamming_distance_counts = list(hamming_distance_counts)

    return hamming_distance_counts


def rescaling_factor(lengths, quantity, offset=2):
    """
    Finds a rescaling value alpha that can be used to map the Clifford RB decay constant
    p to p_(rescaled) = p^(1/alpha) for finding e.g., a "CRB r per CNOT" or a "CRB r per
    compiled Clifford depth".

    Parameters
    ----------
    lengths : list
        A list of the RB lengths, which each value in 'quantity' will be rescaled by.

    quantity : list
        A list, of the same length as `lengths`, that contains lists of values of the quantity
        that the rescaling factor is extracted from.

    offset : int, optional
        A constant offset to add to lengths.

    Returns
        mean over i of [mean(quantity[i])/(lengths[i]+offset)]
    """
    assert(len(lengths) == len(quantity)), "Data format incorrect!"
    rescaling_factor = []

    for i in range(len(lengths)):
        rescaling_factor.append(_np.mean(_np.array(quantity[i]) / (lengths[i] + offset)))

    rescaling_factor = _np.mean(_np.array(rescaling_factor))

    return rescaling_factor

"""
Tools for analyzing MCFE data
"""

# TODO: add copyright statement

from __future__ import annotations
from typing import Optional

import numpy as _np

from pygsti.tools.rbtools import hamming_distance

from pygsti.data.dataset import _DataSetRow
from pygsti.circuits import Circuit as _Circuit

def success_probability_to_polarization(s: float, n: int) -> float:
    """
    Utility function for MCFE VBDataFrame creation.

    Maps a success probablity `s` for an n-qubit circuit to
    the polarization `s`, defined by `p = (s - 1/4^n)/(1 - 1/4^n)`.
    For large n, the difference between `p` and `s` is negligible
    and the calculation of 4**n is prohibitive, so we impose
    a cutoff above which we assert `p = s`.
    
    Parameters
    ------------
    s : float
        success probability for the circuit.
    
    n : int
        number of qubits in the circuit.


    Returns
    -----------
    float
        circuit polarization.
    """

    if n < 20:
        return (s - 1 / 4**n) / (1 - 1 / 4**n) 
    else:
        return s
    

def polarization_to_success_probability(p: float, n: int) -> float:
    """
    Utility function for MCFE VBDataFrame creation.

    Maps a polarization `p` for an n-qubit circuit to
    the success probability `s`, defined by `s = p * (1 - 1/4**n) + 1/4**n`.
    For large n, the difference between `p` and `s` is negligible
    and the calculation of 2**n is prohibitive, so we impose
    a cutoff above which we assert `s = p`.
    
    Parameters
    ------------
    p : float
        circuit polarization
    
    n : int
        number of qubits in the circuit.


    Returns
    -----------
    float
        circuit success probability.
    """

    if n < 20:
        return p * (1 - 1 / 4**n) + 1 / 4**n
    else:
        return p


def polarization_to_fidelity(p: float, n: int) -> float:
    """
    Utility function for MCFE VBDataFrame creation.

    Maps a polarization `p` for an n-qubit circuit to
    the process fidelity `f`, defined by `f = 1 - (4**n - 1)*(1 - p)/4**n`.
    For large n, the difference between `p` and `f` is negligible
    and the calculation of 4**n is prohibitive, so we impose
    a cutoff above which we assert `f = p`.
    
    Parameters
    ------------
    p : float
        circuit polarization
    
    n : int
        number of qubits in the circuit.


    Returns
    -----------
    float
        circuit process fidelity.
    """

    if n < 20: 
        return 1 - (4**n - 1)*(1 - p)/4**n
    else:
        return p


def fidelity_to_polarization(f: float, n: int) -> float:
    """
    Utility function for MCFE VBDataFrame creation.

    Maps a process fidelity `f` for an n-qubit circuit to
    the polarization `p`, defined by `p = 1 - (4**n)*(1 - f)/(4**n - 1)`.
    For large n, the difference between `p` and `f` is negligible
    and the calculation of 4**n is prohibitive, so we impose
    a cutoff above which we assert `p = f`.
    
    Parameters
    ------------
    p : float
        circuit polarization
    
    n : int
        number of qubits in the circuit.


    Returns
    -----------
    float
        circuit process fidelity.
    """
    
    if n < 20:
        return 1 - (4**n)*(1 - f)/(4**n - 1)
    else:
        return f


def hamming_distance_counts(dsrow: _DataSetRow, circ: _Circuit, idealout: str) -> _np.ndarray:
    """
    Utility function for MCFE VBDataFrame creation.

    Compute the hamming distance counts for the outcomes
    of a circuit on `n` qubits.

    Hamming distance is defined as the number of bits
    that are different between two bitstrings of the same
    length. E.g., the Hamming distance between 0010 and 1110
    is 2.

    Parameters
    ------------
    dsrow : pygsti.data.dataset.DataSetRow
        Row in dataset containing outcome information
        for all mirror circuits.

    circ : pygsti.circuits.Circuit
        pyGSTi circuit to which the outcome counts in
        `dsrow` pertain.

    idealout : string
        length-`n` string that defines the target
        bitstring for the circuit. All Hamming distances
        are calculated relative to this string.

        
    Returns
    --------------
    np.ndarray[float]
        Array whose indices correspond to Hamming distances
        from `idealout`, and whose values at each index
        correspond to the number of counts in `dsrow` that
        have that Hamming distance.
    """

    nQ = len(circ.line_labels)  # number of qubits
    assert nQ == len(idealout[-1]), f'{nQ} != {len(idealout[-1])}'
    hamming_distance_counts = _np.zeros(nQ + 1, float)
    if dsrow.total > 0:
        for outcome_lbl, counts in dsrow.counts.items():
            outbitstring = outcome_lbl[-1]
            hamming_distance_counts[hamming_distance(outbitstring, idealout[-1])] += counts
    return hamming_distance_counts


def adjusted_success_probability(hamming_distance_counts: _np.ndarray) -> float:
    """
    Utility function for MCFE VBDataFrame creation.

    Compute the adjusted success probability `adjSP` of a circuit
    from its Hamming distance counts, according to the formula

    adjSP = \sum_{k=0}^n (-1/2)^k * h_k,

    where h_k is the probability of Hamming distance k.

    Parameters
    ------------
    hamming_distance_counts : np.ndarray[float]
        Array whose indices correspond to Hamming distances,
        and whose values at each index correspond to the number of 
        counts of that Hamming distance in the circuit outcome
        data from which `hamming_distance_counts` was derived.

    Returns
    -----------
    float
        adjusted success probability.
    """

    if _np.sum(hamming_distance_counts) == 0.: 
        return 0.
    else:
        hamming_distance_pdf = _np.array(hamming_distance_counts) / _np.sum(hamming_distance_counts)
        adjSP = _np.sum([(-1 / 2)**n * hamming_distance_pdf[n] for n in range(len(hamming_distance_pdf))])
        return adjSP
    

def effective_polarization(hamming_distance_counts: _np.ndarray) -> float:
    """
    Utility function for MCFE VBDataFrame creation.

    Compute the effective polarization `p` of a circuit
    from its Hamming distance counts, according to the formula

    `p = (4**n * asp - 1)/(4**n - 1)`

    where `asp` is the adjusted success probability of
    the circuit.

    Parameters
    ------------
    hamming_distance_counts : np.ndarray[float]
        Array whose indices correspond to Hamming distances,
        and whose values at each index correspond to the number of 
        counts of that Hamming distance in the circuit outcome
        data from which `hamming_distance_counts` was derived.

    Returns
    -----------
    float
        polarization.
    """

    n = len(hamming_distance_counts) - 1 
    asp = adjusted_success_probability(hamming_distance_counts)

    if n < 20:
        return (4**n * asp - 1)/(4**n - 1)
    else:
        return asp


def rc_predicted_process_fidelity(bare_rc_effective_pols: _np.ndarray,
                                  rc_rc_effective_pols: _np.ndarray,
                                  reference_effective_pols: _np.ndarray,
                                  n: int) -> float:
    """
    Utility function for MCFE VBDataFrame creation.

    Compute the process fidelity `f` for a circuit on `n` qubits according to the formula

    `f = 1 - (4**n - 1)/4**n * (1 - p)`,

    Where `p` is the effective polarization of the circuit, given by

    `p = E[p(M_1)] / sqrt( E[p(M_2)] * E[p(M_3)] )`.

    Here, M_i refers to the i-th family of mirror circuits used in mirror circuit fidelity
    estimation with randomized compiling. See https://arxiv.org/pdf/2204.07568 for more
    information.

    The process fidelity estimate is clamped to the [0.0, 1.0] range.

    Parameters
    ------------
    bare_rc_effective_pols : np.ndarray[float]
        list of effective polarizations for bare_rc (M_1) mirror circuits.

    rc_rc_effective_pols : np.ndarray[float]
        list of effective polarizations for rc_rc (M_2) mirror circuits.

    reference_effective_pols : np.ndarray[float]
        list of effective polarizations for SPAM reference (M_3) mirror circuits.

    n : int
        number of qubits in the quantum circuit.

    
    Returns:
    ----------
    float
        process fidelity estimate.
    """

    a = _np.mean(bare_rc_effective_pols)
    b = _np.mean(rc_rc_effective_pols)
    c = _np.mean(reference_effective_pols)

    # print(a)

    # print(b)

    # print(c)
    
    if c <= 0.:
        return _np.nan  # raise ValueError("Reference effective polarization zero or smaller! Cannot estimate the process fidelity")
    elif b <= 0:
        return 0.
    else:
        pfid = polarization_to_fidelity(a / _np.sqrt(b * c), n)
        if pfid < 0.0:
            return 0.0
        elif pfid > 1.0:
            return 1.0
        else:
            return pfid
        # return pfid


def predicted_process_fidelity_for_central_pauli_mcs(central_pauli_effective_pols: _np.ndarray,
                                                     reference_effective_pols: _np.ndarray,
                                                     n: int) -> float:
    """
    Utility function for MCFE VBDataFrame creation.

    Compute the process fidelity `f` for a circuit on `n` qubits according to the formula

    `f = 1 - (4**n - 1)/4**n * (1 - p)`,

    Where `p` is the effective polarization of the circuit, given by

    `p = sqrt( E[p(M_1)] / E[p(M_2)])

    Here, M_1 refers to central Pauli quasi-mirror circuits and 
    M_2 refers to SPAM reference circuits.

    Parameters
    ------------
    central_effective_pols : np.ndarray[float]
        list of effective polarizations for central Pauli (M_1) mirror circuits.


    reference_effective_pols : np.ndarray[float]
        list of effective polarizations for SPAM reference (M_2) mirror circuits.

    n : int
        number of qubits in the quantum circuit.

    
    Returns:
    ----------
    float
        process fidelity estimate.
    """

    a = _np.mean(central_pauli_effective_pols)
    c = _np.mean(reference_effective_pols)
    if c <= 0.:
        return _np.nan  # raise ValueError("Reference effective polarization zero or smaller! Cannot estimate the process fidelity")
    elif a <= 0:
        return 0.
    else:
        return polarization_to_fidelity(_np.sqrt(a / c), n)
    

def rc_bootstrap_predicted_pfid(brs: _np.ndarray,
                                rrs: _np.ndarray,
                                refs: _np.ndarray,
                                n: int,
                                num_bootstraps: Optional[int] = 50,
                                rand_state: Optional[_np.random.RandomState] = None
                                ) -> float:
    """
    Utility function for MCFE VBDataFrame creation.

    Compute bootstrapped error bars for the MCFE random compilation process fidelity
    estimate.

    Parameters
    ------------
    brs : np.ndarray[float]
        list of effective polarizations for bare_rc (M_1) mirror circuits.

    rrs : np.ndarray[float]
        list of effective polarizations for rc_rc (M_2) mirror circuits.

    refs : np.ndarray[float]
        list of effective polarizations for SPAM reference (M_3) mirror circuits.

    n : int
        number of qubits in the quantum circuit.

    num_bootstraps : int
        number of samples to take from the bootstrapped distribution.

    rand_state : optional, np.random.RandomState
        random state to be used for bootstrapepd distribution sampling.

            
    Returns:
    ----------
    float
        estimate for standard deviation of process fidelity estimate.
    """

    if rand_state is None:
        rand_state = _np.random.RandomState()

    # print(num_bootstraps)

    pfid_samples = []
    for _ in range(num_bootstraps):
        br_sample = rand_state.choice(brs, len(brs), replace=True)
        rr_sample = rand_state.choice(rrs, len(rrs), replace=True)
        ref_sample = rand_state.choice(refs, len(refs), replace=True)

        pfid = rc_predicted_process_fidelity(
            br_sample,
            rr_sample,
            ref_sample,
            n)

        pfid_samples.append(pfid)

    bootstrapped_stdev = _np.std(pfid_samples)

    return bootstrapped_stdev    
"""
Ideal stabilizer baselines, tableau fidelities, and probability / Pauli expectation corrections
and approximations for error generator propagation.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from __future__ import annotations
import warnings as _warnings

try:
    import stim
except ImportError:
    msg = "Stim is required for use of the error generator propagation tools module, " \
          "and it does not appear to be installed. If you intend to use this module please update" \
          " your environment."
    _warnings.warn(msg)

import numpy as _np
from itertools import product
from typing import Optional, Union, Sequence, TYPE_CHECKING

from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL, LocalElementaryErrorgenLabel as _LEEL, ElementaryErrorgenLabel as _EEL
from pygsti.circuits import Circuit as _Circuit
from pygsti.errorgenpropagation.localstimerrorgen import canonicalize_errorgen_layer as _canonicalize_errorgen_layer
from pygsti.tools.mptools import SharedWorkerPool as _SharedWorkerPool, _pool_scope

from .errgentypes import _ErrorgenDict
from .parallelutils import EGPROP_PREWARM_PACKAGES
from . import expansions as _exp
from . import sensitivities as _sens
from .sensitivities import bulk_alpha, bulk_alpha_pauli, _real_if_close

if TYPE_CHECKING:
    import multiprocessing.pool as _mppool

__all__ = [
    'tableau_fidelities',
    'tableau_fidelity',
    'bitstring_to_tableau',
    'stabilizer_probabilities',
    'stabilizer_probability',
    'stabilizer_pauli_expectation',
    'stabilizer_probability_correction',
    'stabilizer_pauli_expectation_correction',
    'approximate_stabilizer_probability',
    'approximate_stabilizer_pauli_expectation',
    'approximate_stabilizer_probabilities',
    'approximate_stabilizer_pauli_expectations',
]


def tableau_fidelities(target_tableau: stim.Tableau, compared_tableaus: Sequence[stim.Tableau]) -> _np.ndarray:
    """
    Calculate the fidelities between a single target stabilizer state and a collection
    of tableaus for stabilizer states to compare with

    This function amortizes the overhead of target tableau inversion and TableauSimulator
    instantiation across all compared tableaus.

    Parameters
    ----------
    target_tableau : stim.Tableau
        Stim tableau corresponding to the reference/target stabilizer state.
    compared_tableaus : sequence of stim.Tableau
        Collection of stim tableaus corresponding to stabilizer states to compare against.

    Returns
    -------
    numpy.ndarray
        1D array of floats representing the fidelity (in probability units) for each compared tableau.
    """
    n_compared = len(compared_tableaus)
    if n_compared == 0:
        return _np.empty(0, dtype=_np.float64)

    n_qubits = len(target_tableau)
    inv_target = target_tableau.inverse()
    sim = stim.TableauSimulator()
    out = _np.empty(n_compared, dtype=_np.float64)

    for i, comp in enumerate(compared_tableaus):
        if len(comp) != n_qubits:
            raise ValueError(f"Compared tableau qubit count ({len(comp)}) must match target tableau ({n_qubits}).")
        t3 = inv_target * comp
        sim.set_inverse_tableau(t3)
        p = 1.0
        for q in range(n_qubits):
            e = sim.peek_z(q)
            if e == -1:
                p = 0.0
                break
            if e == 0:
                p *= 0.5
                sim.postselect_z(q, desired_value=False)
        out[i] = p

    return out


def tableau_fidelity(tableau1: stim.Tableau, tableau2: stim.Tableau) -> float:
    """
    Calculate the fidelity between the stabilizer states corresponding to the given stim
    tableaus. This returns a result in units of probability (so this may be squared
    fidelity depending on your convention).

    Parameters
    ----------
    tableau1 : stim.Tableau
        Stim tableau for first stabilizer state.
    tableau2 : stim.Tableau
        Stim tableau for second stabilizer state.

    Returns
    -------
    float
        Fidelity between the two stabilizer states.
    """
    return float(tableau_fidelities(tableau1, [tableau2])[0])


def bitstring_to_tableau(bitstring: str) -> stim.Tableau:
    """
    Map a computational basis bit string into a corresponding Tableau which maps the all zero
    state into that state.

    Parameters
    ----------
    bitstring : str
        String of 0's and 1's corresponding to the computational basis state to prepare the Tableau for.

    Returns
    -------
    stim.Tableau
        Tableau which maps the all zero string to this computational basis state
    """
    pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in bitstring]))
    # convert this to a stim.Tableau
    pauli_tableau = pauli_string.to_tableau()
    return pauli_tableau


def stabilizer_probability_correction(errorgen_dict: _ErrorgenDict, tableau: stim.Tableau, desired_bitstring: str,
                                      order: int = 1, truncation_threshold: float = 1e-14,
                                      num_workers: Union[int, str, None] = None,
                                      pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> float:
    """
    Compute the kth-order correction to the probability of the specified bit string.

    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.

    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.

    desired_bitstring : str
        String of 0's and 1's corresponding to the output bitstring being measured.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding rates
        are below this value.

    num_workers : int, str, or None, optional
        Target worker count for parallel evaluation.

    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.

    Returns
    -------
    correction : float
        float corresponding to the correction to the output probability for the
        desired bitstring induced by the error generator (to specified order).
    """
    num_random = _sens.random_support(tableau)
    if num_random > 2148:
        raise RuntimeError('Number of random bits is greater than 1074, magnitude of probability scale will underflow!')
    scale = 1/2**(num_random)

    #accumulate the terms across orders (with short circuit logic for order 1 to save time):
    # Share one (lazily started) pool between the Taylor expansion and the sensitivity calculation.
    with _pool_scope(num_workers, pool, prewarm_packages=EGPROP_PREWARM_PACKAGES) as stage_pool:
        if order == 1:
            combined_taylor_dict = errorgen_dict
        else:
            #compute the taylor series approximation to the desired order.
            taylor_expansion = _exp.error_generator_taylor_expansion(errorgen_dict, order, truncation_threshold,
                                                                     num_workers=num_workers, pool=stage_pool)
            # Accumulate all of the dictionaries in taylor expansion into a single one, summing overlapping terms.
            combined_taylor_dict = {}
            get = combined_taylor_dict.get
            for order_dict in taylor_expansion:
                for lbl, rate in order_dict.items():
                    combined_taylor_dict[lbl] = get(lbl, 0) + rate.real

        #can now do the correction computation in a single-shot.
        alphas = bulk_alpha(combined_taylor_dict, tableau, [desired_bitstring], num_workers=num_workers, pool=stage_pool)
    rates = _np.fromiter(combined_taylor_dict.values(), dtype=_np.double)

    alpha_errgen_prods = alphas*rates
    correction = scale*_np.sum(alpha_errgen_prods)

    return correction

# TODO: The implementations for the pauli expectation value correction and probability correction
# are basically identical modulo some additional scale factors and the alpha function used. Should be able to combine
# the implementations into one function.
def stabilizer_pauli_expectation_correction(errorgen_dict: _ErrorgenDict, tableau: stim.Tableau, pauli: stim.PauliString,
                                            order: int = 1, truncation_threshold: float = 1e-14,
                                            num_workers: Union[int, str, None] = None,
                                            pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> float:
    """
    Compute the kth-order correction to the expectation value of the specified pauli.

    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.

    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.

    pauli : stim.PauliString
        Pauli operator to compute expectation value correction for.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding rates
        are below this value.

    num_workers : int, str, or None, optional
        Target worker count for parallel evaluation.

    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.

    Returns
    -------
    correction : float
        float corresponding to the correction to the expectation value for the
        selected pauli operator induced by the error generator (to specified order).
    """
    #accumulate the terms across orders (with short circuit logic for order 1 to save time):
    # Share one (lazily started) pool between the Taylor expansion and the sensitivity calculation.
    with _pool_scope(num_workers, pool, prewarm_packages=EGPROP_PREWARM_PACKAGES) as stage_pool:
        if order == 1:
            combined_taylor_dict = errorgen_dict
        else:
            #compute the taylor series approximation to the desired order.
            taylor_expansion = _exp.error_generator_taylor_expansion(errorgen_dict, order, truncation_threshold,
                                                                     num_workers=num_workers, pool=stage_pool)
            # Accumulate all of the dictionaries in taylor expansion into a single one, summing overlapping terms.
            combined_taylor_dict = {}
            get = combined_taylor_dict.get
            for order_dict in taylor_expansion:
                for lbl, rate in order_dict.items():
                    combined_taylor_dict[lbl] = get(lbl, 0) + rate.real

        #can now do the correction computation in a single-shot.
        alphas = bulk_alpha_pauli(combined_taylor_dict, tableau, [pauli], num_workers=num_workers, pool=stage_pool)
    rates = _np.fromiter(combined_taylor_dict.values(), dtype=_np.double)
    alpha_errgen_prods = alphas*rates
    correction = _np.sum(alpha_errgen_prods)

    return correction


def stabilizer_probabilities(tableau: stim.Tableau, bitstrings: Sequence[str]) -> _np.ndarray:
    """
    Calculate the output probabilities for a sequence of specified bitstrings.

    Parameters
    ----------
    tableau : stim.Tableau
        Stim tableau for the stabilizer state being measured.
    bitstrings : sequence of str
        Sequence of bitstrings (e.g. ['00', '01', ...]) to evaluate output probabilities for.

    Returns
    -------
    numpy.ndarray
        1D array of output probabilities.
    """
    if len(bitstrings) == 0:
        return _np.empty(0, dtype=_np.float64)
    bitstring_tableaus = [bitstring_to_tableau(b) for b in bitstrings]
    return tableau_fidelities(tableau, bitstring_tableaus)


def stabilizer_probability(tableau: stim.Tableau, desired_bitstring: str) -> float:
    """
    Calculate the output probability for the specified output bitstring.

    Parameters
    ----------
    tableau : stim.Tableau
        Stim tableau for the stabilizer state being measured.

    desired_bitstring : str
        String of 0's and 1's corresponding to the output bitstring being measured.

    Returns
    -------
    p : float
        probability of desired bitstring.
    """
    return float(stabilizer_probabilities(tableau, [desired_bitstring])[0])


def stabilizer_pauli_expectation(tableau: stim.Tableau, pauli: stim.PauliString) -> float:
    """
    Calculate the output probability for the specified output bitstring.

    Parameters
    ----------
    tableau : stim.Tableau
        Stim tableau for the stabilizer state being measured.

    pauli : stim.PauliString
        Pauli operator to compute expectation value for.

    Returns
    -------
    expected_value : float
        Expectation value of specified pauli
    """
    if pauli.sign != 1:
        pauli_sign = pauli.sign
        unsigned_pauli = pauli/pauli_sign
    else:
        pauli_sign = 1
        unsigned_pauli = pauli

    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
    expectation  = pauli_sign*sim.peek_observable_expectation(unsigned_pauli)
    return expectation


def approximate_stabilizer_probability(errorgen_dict: dict[_EEL, float], circuit: Union[_Circuit, stim.Tableau],
                                       desired_bitstring: str, order: int = 1,
                                       truncation_threshold: float = 1e-14,
                                       num_workers: Union[int, str, None] = None,
                                       pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> float:
    """
    Calculate the approximate probability of a desired bit string using an nth-order taylor series approximation.

    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.

    circuit : `Circuit` or `stim.Tableau`
        A pygsti `Circuit` or a stim.Tableau to compute the output probability for. In either
        case this should be a Clifford circuit and convertible to a stim.Tableau.

    desired_bitstring : str
        String of 0's and 1's corresponding to the output bitstring being measured.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value. (Used internally in computation of probability corrections)

    num_workers : int, str, or None, optional
        Target worker count for parallel evaluation.

    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.

    Returns
    -------
    p : float
        Approximate output probability for desired bitstring.
    """

    if isinstance(circuit, _Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit
    else:
        raise ValueError('`circuit` should either be a pygsti `Circuit` or a stim.Tableau.')

    # recast keys to local stim ones if needed (in canonical basis element label order; a
    # GlobalElementaryErrorgenLabel key was never supported here since no sslbls are available).
    first_lbl = next(iter(errorgen_dict))
    if isinstance(first_lbl, (_GEEL, _LEEL)):
        errorgen_dict = _canonicalize_errorgen_layer(errorgen_dict)

    ideal_prob = stabilizer_probability(tableau, desired_bitstring)
    correction = stabilizer_probability_correction(errorgen_dict, tableau, desired_bitstring, order, truncation_threshold,
                                                   num_workers=num_workers, pool=pool)
    return ideal_prob + correction


def approximate_stabilizer_pauli_expectation(errorgen_dict: dict[_EEL, float], circuit: Union[_Circuit, stim.Tableau],
                                             pauli: Union[str, stim.PauliString], order: int = 1,
                                             truncation_threshold: float = 1e-14,
                                             num_workers: Union[int, str, None] = None,
                                             pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> float:
    """
    Calculate the approximate probability of a desired bit string using a first-order approximation.

    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.

    circuit : `Circuit` or `stim.Tableau`
        A pygsti `Circuit` or a stim.Tableau to compute the output probability for. In either
        case this should be a Clifford circuit and convertible to a stim.Tableau.

    pauli : str or stim.PauliString
        Pauli operator to compute expectation value for.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value. (Used internally in computation of probability corrections)

    num_workers : int, str, or None, optional
        Target worker count for parallel evaluation.

    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.

    Returns
    -------
    expectation_value : float
        Approximate expectation value for desired pauli.
    """

    if isinstance(circuit, _Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit
    else:
        raise ValueError('`circuit` should either be a pygsti `Circuit` or a stim.Tableau.')

    if isinstance(pauli, str):
        pauli = stim.PauliString(pauli)

    # recast keys to local stim ones if needed (in canonical basis element label order; a
    # GlobalElementaryErrorgenLabel key was never supported here since no sslbls are available).
    first_lbl = next(iter(errorgen_dict))
    if isinstance(first_lbl, (_GEEL, _LEEL)):
        errorgen_dict = _canonicalize_errorgen_layer(errorgen_dict)

    ideal_expectation = stabilizer_pauli_expectation(tableau, pauli)
    correction = stabilizer_pauli_expectation_correction(errorgen_dict, tableau, pauli, order, truncation_threshold,
                                                         num_workers=num_workers, pool=pool)
    return ideal_expectation + correction


def approximate_stabilizer_probabilities(
    errorgen_dict: _ErrorgenDict,
    circuit: Union[_Circuit, stim.Tableau],
    bitstrings: Optional[Sequence[str]] = None,
    order: int = 1,
    truncation_threshold: float = 1e-14,
    num_workers: Union[int, str, None] = None,
    pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None,
) -> _np.ndarray:
    """
    Calculate the approximate probability distribution over bitstrings using a Taylor series approximation.

    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` or `LocalElementaryErrorgenLabel`
        and whose values are corresponding rates. `LocalElementaryErrorgenLabel` keys are recast
        (in canonical basis element label order) to `LocalStimErrorgenLabel`.
    circuit : `Circuit` or `stim.Tableau`
        A pygsti `Circuit` or a stim.Tableau to compute the output probability for. In either
        case this should be a Clifford circuit and convertible to a stim.Tableau.
    bitstrings : sequence of str, optional
        Specific bitstrings to compute probabilities for. If None (default), computes
        the exhaustive probability distribution over all 2^n bitstrings. If n > 10
        and bitstrings is None, a UserWarning alerting to exponential scaling is issued.

        Note on backwards compatibility: in pyGSTi 0.10 and earlier this function had no
        `bitstrings` argument and its third positional argument was `order`. Callers
        that passed `order` positionally should be updated.
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value.
    num_workers : int, str, or None, optional
        Target worker count for parallel evaluation across sensitivity grid.
    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.

    Returns
    -------
    p : numpy.ndarray
        Array of approximate output probabilities.
    """
    if isinstance(circuit, _Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit
    else:
        raise ValueError('`circuit` should either be a pygsti `Circuit` or a stim.Tableau.')

    # recast keys to local stim ones if needed (in canonical basis element label order; a
    # GlobalElementaryErrorgenLabel key was never supported here since no sslbls are available).
    if errorgen_dict and isinstance(next(iter(errorgen_dict)), (_GEEL, _LEEL)):
        errorgen_dict = _canonicalize_errorgen_layer(errorgen_dict)

    num_random = _sens.random_support(tableau)
    if num_random > 2148:
        raise RuntimeError('Number of random bits is greater than 1074, magnitude of probability scale will underflow!')

    num_qubits = len(tableau)
    if bitstrings is None:
        if num_qubits > 10:
            _warnings.warn(
                f"approximate_stabilizer_probabilities requested exhaustive distribution for {num_qubits} qubits "
                f"(2^{num_qubits} = {2**num_qubits} bitstrings). This scales exponentially. "
                "Consider specifying the 'bitstrings' parameter explicitly for large systems.",
                UserWarning,
                stacklevel=2,
            )
        bitstrings_list = ["".join(bitstring) for bitstring in product(['0', '1'], repeat=num_qubits)]
    else:
        bitstrings_list = list(bitstrings)

    if len(bitstrings_list) == 0:
        return _np.empty(0, dtype=_np.double)

    # Compute the Taylor series expansion once for all bitstrings, sharing one (lazily started) pool
    # between the Taylor expansion and the sensitivity calculation.
    with _pool_scope(num_workers, pool, prewarm_packages=EGPROP_PREWARM_PACKAGES) as stage_pool:
        if order == 1:
            combined_taylor_dict = errorgen_dict
        else:
            taylor_expansion = _exp.error_generator_taylor_expansion(
                errorgen_dict, order, truncation_threshold, num_workers=num_workers, pool=stage_pool
            )
            combined_taylor_dict = {}
            get = combined_taylor_dict.get
            for order_dict in taylor_expansion:
                for lbl, rate in order_dict.items():
                    combined_taylor_dict[lbl] = get(lbl, 0) + rate.real

        # Single bulk_alpha call (vectorized Cython or 2D parallel grid)
        alphas = bulk_alpha(combined_taylor_dict, tableau, bitstrings_list, num_workers=num_workers, pool=stage_pool)
    rates = _np.fromiter(combined_taylor_dict.values(), dtype=_np.double)
    scale = 1.0 / (2.0 ** num_random)
    corrections = scale * (alphas @ rates)

    ideal_probs = stabilizer_probabilities(tableau, bitstrings_list)
    return ideal_probs + corrections


def approximate_stabilizer_pauli_expectations(
    errorgen_dict: _ErrorgenDict,
    circuit: Union[_Circuit, stim.Tableau],
    paulis: Sequence[Union[str, stim.PauliString]],
    order: int = 1,
    truncation_threshold: float = 1e-14,
    num_workers: Union[int, str, None] = None,
    pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None,
) -> _np.ndarray:
    """
    Calculate the approximate expectation values for a sequence of Pauli observables.

    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` or `LocalElementaryErrorgenLabel`
        and whose values are corresponding rates. `LocalElementaryErrorgenLabel` keys are recast
        (in canonical basis element label order) to `LocalStimErrorgenLabel`.
    circuit : `Circuit` or `stim.Tableau`
        A pygsti `Circuit` or stim.Tableau to compute the expectation values for.
    paulis : sequence of str or stim.PauliString
        Pauli operators to compute expectation values for.
    order : int, optional (default 1)
        Order of the Taylor series expansion for the exponentiated error generator.
    truncation_threshold : float, optional (default 1e-14)
        Truncation threshold for discarding small rates.
    num_workers : int, str, or None, optional
        Target worker count for parallel evaluation across sensitivity grid.
    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.

    Returns
    -------
    numpy.ndarray
        Array of approximate expectation values for the requested Pauli observables.
    """
    if isinstance(circuit, _Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit
    else:
        raise ValueError('`circuit` should either be a pygsti `Circuit` or a stim.Tableau.')

    # recast keys to local stim ones if needed (in canonical basis element label order; a
    # GlobalElementaryErrorgenLabel key was never supported here since no sslbls are available).
    if errorgen_dict and isinstance(next(iter(errorgen_dict)), (_GEEL, _LEEL)):
        errorgen_dict = _canonicalize_errorgen_layer(errorgen_dict)

    pauli_objs = [stim.PauliString(p) if isinstance(p, str) else p for p in paulis]
    if len(pauli_objs) == 0:
        return _np.empty(0, dtype=_np.double)

    # Compute the Taylor series expansion once for all Paulis, sharing one (lazily started) pool
    # between the Taylor expansion and the sensitivity calculation.
    with _pool_scope(num_workers, pool, prewarm_packages=EGPROP_PREWARM_PACKAGES) as stage_pool:
        if order == 1:
            combined_taylor_dict = errorgen_dict
        else:
            taylor_expansion = _exp.error_generator_taylor_expansion(
                errorgen_dict, order, truncation_threshold, num_workers=num_workers, pool=stage_pool
            )
            combined_taylor_dict = {}
            get = combined_taylor_dict.get
            for order_dict in taylor_expansion:
                for lbl, rate in order_dict.items():
                    combined_taylor_dict[lbl] = get(lbl, 0) + rate.real

        # Single bulk_alpha_pauli call (vectorized Cython or 2D parallel grid)
        alphas = bulk_alpha_pauli(combined_taylor_dict, tableau, pauli_objs, num_workers=num_workers, pool=stage_pool)
    rates = _np.fromiter(combined_taylor_dict.values(), dtype=_np.double)
    corrections = alphas @ rates

    # Ideal expectations evaluated serially using a single pre-inverted simulator
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
    ideal_expectations = _np.empty(len(pauli_objs), dtype=_np.double)
    for i, p in enumerate(pauli_objs):
        if p.sign != 1:
            s = p.sign
            up = p / s
        else:
            s = 1
            up = p
        ideal_expectations[i] = _real_if_close(s * sim.peek_observable_expectation(up))

    return ideal_expectations + corrections

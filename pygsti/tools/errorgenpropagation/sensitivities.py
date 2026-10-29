"""
Stabilizer support/amplitude tableau math and linear sensitivities of probabilities and
Pauli expectation values to elementary error generators (`phi`, `alpha`, `alpha_pauli`).
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
from typing import Optional, Union, Iterable, TYPE_CHECKING
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as _LEEL
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.tools.mptools import (
    determine_num_workers as _determine_num_workers,
    SharedWorkerPool as _SharedWorkerPool,
    parallel_2d_grid_map as _parallel_2d_grid_map,
    warn_if_insufficient_memory as _warn_if_insufficient_memory,
    resolve_pool_workers as _resolve_pool_workers,
)
from .errgentypes import (
    PauliPhaseUpdater,
    PauliPhaseZerosUpdater,
    AmplitudeOfStateType,
    BulkAmplitudeOfStateType,
    BulkPhiType,
    BulkAlphaType,
    BulkAlphaPauliType,
)
from .algebra import com, pauli_product, errorgen_pauli_action
from .parallelutils import (
    _with_cyclic_gc_paused,
    EGPROP_PREWARM_PACKAGES,
    estimate_bulk_alpha_peak_memory,
    _should_parallelize_alpha,
    _BulkAlphaWorker,
    _BulkAlphaPauliWorker,
)

if TYPE_CHECKING:
    import multiprocessing.pool as _mppool

__all__ = [
    'random_support',
    'in_stabilizer_support',
    'compute_phase_reference',
    'PAULI_PHASES_0',
    'PAULI_PHASES_1',
    'PAULI_PHASES_0_DUAL',
    'PAULI_PHASES_1_DUAL',
    'PAULI_FLIPS',
    'slow_pauli_phase_update',
    'slow_pauli_phase_update_all_zeros',
    'pauli_phase_update',
    'pauli_phase_update_all_zeros',
    'slow_amplitude_of_state',
    'slow_bulk_amplitude_of_state',
    'amplitude_of_state',
    'bulk_amplitude_of_state',
    'phi',
    'slow_bulk_phi',
    'bulk_phi',
    'alpha',
    'slow_bulk_alpha',
    'bulk_alpha',
    'alpha_pauli',
    'slow_bulk_alpha_pauli',
    'bulk_alpha_pauli',
]


def random_support(tableau: Union[stim.Tableau, stim.TableauSimulator], return_support: bool=False):
    """ 
    Compute the number of bits over which the stabilizer state corresponding to this stim tableau
    would have measurement outcomes which are random.
    
    Parameters
    ----------
    tableau : Union[stim.Tableau, stim.TableauSimulator]
        stim.Tableau corresponding to the stabilizer state we want the random support
        for.
    
    return_support : bool, optional (default False)
        If True also returns a list of qubit indices over which the distribution of outcome
        bit strings is random.

    Returns
    -------
    num_random : int
        Number of bit on which stabilize state is random.

    support : list of bool
        A list of boolean values which can be used as a bitmask
        and corresponds to the bits found to be random.
    """
    # TODO Test for correctness on support
    if isinstance(tableau, stim.Tableau):
        sim = stim.TableauSimulator()
        orig_tableau_inverse = tableau**-1
        sim.set_inverse_tableau(orig_tableau_inverse)
    elif isinstance(tableau, stim.TableauSimulator):
        sim = tableau
        orig_tableau_inverse = sim.current_inverse_tableau()
    else:
        raise ValueError(f'Unsupported input type {type(tableau)} for `tableau`. Supported options are stim.Tableau and stim.TableauSimulator')
    
    n = sim.num_qubits

    num_random = 0
    support = [False]*n
    for i in range(n):
        z = sim.peek_z(i)
        if z == 0:
            num_random+=1
            support[i] = True
            #  For a phase reference, use the smallest state with non-zero amplitude.
        forced_bit = z == -1
        sim.postselect_z(i, desired_value=forced_bit)
    if isinstance(tableau, stim.TableauSimulator):
        tableau.set_inverse_tableau(orig_tableau_inverse)
    return (num_random, support) if return_support else num_random

# Modified from Gidney 
# https://quantumcomputing.stackexchange.com/questions/34610/get-the-amplitude-of-a-computational-basis-in-stim
def slow_amplitude_of_state(tableau: Union[stim.Tableau, stim.TableauSimulator], desired_state: str, only_phase: bool) -> complex:
    """
    Get the amplitude of a particular computational basis state for given
    stabilizer state.

    Note: This is a pure python implementation of this function. For best performance see the
    optimized cython implementation pygsti.tools.errorgenpropagation.fasterrgencalc.fast_amplitude_of_state.

    Parameters
    ----------
    tableau : stim.Tableau or stim.TableauSimulator
        Stim tableau corresponding to the stabilizer state we wish to extract
        the amplitude from. If a stim.TableauSimulator it is assumed that this
        simulator has already had the appropriate inverse tableau value instantiated.
    
    desired_state : str
        String of 0's and 1's corresponding to the computational basis state to extract the amplitude for.

    only_phase : bool
        If True then on the phase of the complex amplitude is returned. In many cases this phase
        is the only information required, and for many qubits amplitude of any given state may
        underflow.        
        
    Returns
    -------
    amplitude : complex
        Amplitude of the desired computational basis state for a given stabilizer state.
    """
    amplitude = slow_bulk_amplitude_of_state(tableau, [desired_state], only_phase)[0]    
    return amplitude

def slow_bulk_amplitude_of_state(tableau: Union[stim.Tableau, stim.TableauSimulator], desired_states: Iterable[Union[str, stim.PauliString]], 
                                 only_phase: bool) -> list[complex]:
    """
    Get the amplitudes particular computational basis state fors given
    stabilizer state.

    Note: This is a pure python implementation of this function. For best performance see the
    optimized cython implementation pygsti.tools.errorgenpropagation.fasterrgencalc.fast_bulk_amplitude_of_state.

    Parameters
    ----------
    tableau : stim.Tableau or stim.TableauSimulator
        Stim tableau corresponding to the stabilizer state we wish to extract
        the amplitude from. If a stim.TableauSimulator it is assumed that this
        simulator has already had the appropriate inverse tableau value instantiated.
    
    desired_states : iterable of str or stim.PauliString
        If a string, then a series of of 0's and 1's corresponding to the computational basis 
        state to extract the amplitude for. If a stim.PauliString then the paulis operator which
        maps the all-zero state to the target computational basis state.

    only_phase : bool
        If True then on the phase of the complex amplitude is returned. In many cases this phase
        is the only information required, and for many qubits amplitude of any given state may
        underflow.        
        
    Returns
    -------
    amplitude : complex
        Amplitude of the desired computational basis state for a given stabilizer state.
    """
    if isinstance(tableau, stim.Tableau):
        sim = stim.TableauSimulator()
        orig_tableau_inverse = tableau**-1
        sim.set_inverse_tableau(orig_tableau_inverse)
    elif isinstance(tableau, stim.TableauSimulator):
        sim = tableau
        orig_tableau_inverse = sim.current_inverse_tableau()
    else:
        raise ValueError(f'Unsupported input type {type(tableau)} for `tableau`. Supported options are stim.Tableau and stim.TableauSimulator')
    
    n = sim.num_qubits
    
    num_random, is_random = random_support(sim, return_support=True)
    
    if only_phase:
        magnitude = 1
    else:
        if num_random > 2148:
            raise RuntimeError('Number of random bits is greater than 2148, magnitude of amplitude will underflow!')
        magnitude = 2**-(num_random / 2)
    
    # For a phase reference, use the smallest state with non-zero amplitude.
    # Initialize to None and instantiate the first time it is needed (only when
    # the magnitude is non-zero.
    ref_state = None
    
    phase_factors = [1]*len(desired_states)
    magnitudes = [0]*len(desired_states)
    for i, desired_state in enumerate(desired_states):        
        if in_stabilizer_support(sim, desired_state):
            magnitudes[i] = magnitude
            if ref_state is None:
                ref_state = compute_phase_reference(sim)
        else:
            continue
            
        # convert desired state into a list of bools
        if isinstance(desired_state, str):
            desired_state = [desired_state[j] == '1' for j in range(n)]
        else:
            desired_state = [desired_state[j] == 1 for j in range(n)]
            
        if ref_state == desired_state:
            continue
        #  Postselect away states that aren't the desired or reference states.
        #  Also move the ref state to |00..00> and the desired state to |00..01>.
        found_difference = False
        for q in range(n):
            desired_bit =  desired_state[q]
            ref_bit = ref_state[q]
            if desired_bit == ref_bit:
                if is_random[q]:
                    sim.postselect_z(q, desired_value=ref_bit)
                if desired_bit:
                    sim.x(q)
            elif not found_difference:
                found_difference = True
                if q:
                    sim.swap(0, q)
                if ref_bit:
                    sim.x(0)
            else:
                #  Remove difference between target state and ref state at this bit.
                sim.cnot(0, q)
                sim.postselect_z(q, desired_value=ref_bit)

        #  The phase difference between |00..00> and |00..01> is what we want.
        #  Since other states are gone, this is the bloch vector phase of qubit 0.
        assert found_difference
        s = str(sim.peek_bloch(0))

        if s == "+X":
            phase_factors[i] = 1
        if s == "-X":
            phase_factors[i] = -1
        if s == "+Y":
            phase_factors[i] = 1j
        if s == "-Y":
            phase_factors[i] = -1j
            
        sim.set_inverse_tableau(orig_tableau_inverse)
        
    return _np.fromiter([phase_factor*magnitude for phase_factor, magnitude in zip(phase_factors, magnitudes)], dtype=_np.complex128)

def in_stabilizer_support(tableau: Union[stim.Tableau, stim.TableauSimulator], desired_state: Union[str, stim.PauliString]):
    """
    Return whether or not the desired bitstring is in the support of the stabilizer state 
    corresponding to the input tableau.

    Parameters
    ----------
    tableau : stim.Tableau or stim.TableauSimulator
        Stim tableau corresponding to the stabilizer state we wish to extract
        the amplitude from. If a stim.TableauSimulator it is assumed that this
        simulator has already had the appropriate inverse tableau value instantiated.
    
    desired_state : str or stim.PauliString
        If a string, then a series of of 0's and 1's corresponding to the computational basis 
        state to extract the amplitude for. If a stim.PauliString then the paulis operator which
        maps the all-zero state to the target computational basis state.

    Return
    ------
    success: bool
        A boolean corresponding to True when the desired state is part of the support, and False otherwise.
    """
    if isinstance(tableau, stim.Tableau):
        sim = stim.TableauSimulator()
        orig_tableau_inverse = tableau**-1
        sim.set_inverse_tableau(orig_tableau_inverse)
    elif isinstance(tableau, stim.TableauSimulator):
        sim = tableau
        orig_tableau_inverse = sim.current_inverse_tableau()
    else:
        raise ValueError(f'Unsupported input type {type(tableau)} for `tableau`. Supported options are stim.Tableau and stim.TableauSimulator')

    # start by getting the pauli string which maps the all-zeros string to the target bitstring.
    if isinstance(desired_state, str):
        initial_pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in desired_state]))
    else:
        initial_pauli_string = desired_state
    
    n = sim.num_qubits
    # map the target state to all zero (if present)
    sim.do_pauli_string(initial_pauli_string)
    try:
        sim.postselect_z(range(n), desired_value=False)
        success = True
    except ValueError:
        sim.set_inverse_tableau(orig_tableau_inverse)
        success = False
    if isinstance(tableau, stim.TableauSimulator):
        sim.set_inverse_tableau(orig_tableau_inverse)
    return success

def compute_phase_reference(tableau: Union[stim.Tableau, stim.TableauSimulator]) -> list[bool]:
    """ 
    Compute a canonical state, corresponding to the smallest state with non-zero amplitude, to use
    as a phase reference in computing the phases of components of this stabilizer state. 
    
    Parameters
    ----------
    tableau : Union[stim.Tableau, stim.TableauSimulator]
        stim.Tableau or stim.TableauSimulator corresponding to the stabilizer state we want the random support
        for.
        
    Returns
    -------
    ref_state : list[bool]
        A list of boolean values corresponding to a bitstring for the phase reference.
    """
    if isinstance(tableau, stim.Tableau):
        sim = stim.TableauSimulator()
        orig_tableau_inverse = tableau**-1
        sim.set_inverse_tableau(orig_tableau_inverse)
    elif isinstance(tableau, stim.TableauSimulator):
        sim = tableau
        orig_tableau_inverse = sim.current_inverse_tableau()
    else:
        raise ValueError(f'Unsupported input type {type(tableau)} for `tableau`. Supported options are stim.Tableau and stim.TableauSimulator')
    
    n = sim.num_qubits

    ref_state = [False]*n
    for q in range(n):
        z = sim.peek_z(q)
        forced_bit = z == -1
        ref_state[q] = forced_bit
        if z == 0:
            sim.postselect_z(q, desired_value=forced_bit)
    if isinstance(tableau, stim.TableauSimulator):
        tableau.set_inverse_tableau(orig_tableau_inverse)
        
    return ref_state

#define module-wide constants for pauli-phase updates
PAULI_PHASES_0 = (1, 1, 1j, 1)
PAULI_PHASES_1 = (1, 1, -1j, -1)
PAULI_PHASES_0_DUAL = (1, 1, -1j, 1)
PAULI_PHASES_1_DUAL = (1, 1, 1j, -1)
PAULI_FLIPS = (False, True, True, False)

def slow_pauli_phase_update(pauli: Union[str, stim.PauliString], bitstring: str, dual: bool=False) -> tuple[complex, str]:
    """
    Takes as input a pauli and a bit string and computes the output bitstring
    and the overall phase that bit string accumulates.

    Note: This is a pure python implementation of this function. For best performance see the
    optimized cython implementation pygsti.tools.errorgenpropagation.fasterrgencalc.fast_pauli_phase_update.
    
    Parameters
    ----------
    pauli : str or stim.PauliString
        Pauli to apply
    
    bitstring : str
        String of 0's and 1's representing the bit string to apply the pauli to.
    
    dual : bool, optional (default False)
        If True then then the pauli is acting to the left on a row vector.
    Returns
    -------
    Tuple whose first element is the phase accumulated, and whose second element
    is a string corresponding to the updated bit string.
    """
    
    if isinstance(pauli, str):
        pauli = stim.PauliString(pauli)
    
    bitstring = [False if bit=='0' else True for bit in bitstring]
    if not dual:
        # list of phase correction for each pauli (conditional on 0)
        # Read [I, X, Y, Z]
        pauli_phases_0 = PAULI_PHASES_0
        
        # list of the phase correction for each pauli (conditional on 1)
        # Read [I, X, Y, Z]
        pauli_phases_1 = PAULI_PHASES_1
    else:
        # list of phase correction for each pauli (conditional on 0)
        # Read [I, X, Y, Z]
        pauli_phases_0 = PAULI_PHASES_0_DUAL
        
        # list of the phase correction for each pauli (conditional on 1)
        # Read [I, X, Y, Z]
        pauli_phases_1 = PAULI_PHASES_1_DUAL

    # list of bools corresponding to whether each pauli flips the target bit
    pauli_flips = PAULI_FLIPS
    
    overall_phase = 1
    indices_to_flip = []
    for i, (elem, bit) in enumerate(zip(pauli, bitstring)):
        if bit:
            overall_phase*=pauli_phases_1[elem]
        else:
            overall_phase*=pauli_phases_0[elem]
        if pauli_flips[elem]:
            indices_to_flip.append(i)
    # if the input pauli had any overall phase associated with it add that back
    # in too.
    overall_phase*=pauli.sign
    # apply the flips to get the output bit string.
    for idx in indices_to_flip:
        bitstring[idx] = not bitstring[idx]
    # turn this back into a string
    output_bitstring = ''.join(['1' if bit else '0' for bit in bitstring])
    
    return overall_phase, output_bitstring

def slow_pauli_phase_update_all_zeros(pauli: Union[str, stim.PauliString], dual: bool=False) -> tuple[complex, str]:
    """
    Specialized version of `pauli_phase_update` for the case of the all-zeros
    bitstring which is more computationally efficient. Takes as input a pauli and
    computes the output bitstring and overall phase accumulated when applied to the
    all-zeros bit string.

    Note: This is a pure python implementation of this function. For best performance see the
    optimized cython implementation pygsti.tools.errorgenpropagation.fasterrgencalc.fast_pauli_phase_update.
    
    Parameters
    ----------
    pauli : str or stim.PauliString
        Pauli to apply
        
    dual : bool, optional (default False)
        If True then then the pauli is acting to the left on a row vector.
    Returns
    -------
    Tuple[complex, str]
        A tuple (overall_phase, output_bitstring) where overall_phase is the accumulated phase,
        and output_bitstring is the updated bitstring.
    """
    if isinstance(pauli, str):
        pauli = stim.PauliString(pauli)
    
    n = len(pauli)
    overall_phase = 1
    # We start with all zeros as a list of characters.
    output_chars = ['0'] * n

    # Select the proper phase table for bits which are all False.
    if not dual:
        phases = PAULI_PHASES_0
    else:
        phases = PAULI_PHASES_0_DUAL

    # For an all zero bitstring, each bit is False; so we use phases[elem] for each pauli element.
    for i, elem in enumerate(pauli):
        overall_phase *= phases[elem]
        # Flip the bit if the pauli at this index indicates a flip.
        if PAULI_FLIPS[elem]:
            output_chars[i] = '1'
    overall_phase *= pauli.sign
    return overall_phase, "".join(output_chars)

def phi(tableau: Union[stim.Tableau, stim.TableauSimulator], desired_bitstring: str, P: Union[str, stim.PauliString], Q: Union[str, stim.PauliString]) -> complex:
    """
    This function computes a quantity whose value is used in expression for the sensitivity of probabilities to error generators.
    
    Parameters
    ----------
    tableau : stim.Tableau or stim.TableauSimulator
        A stim Tableau or stim TableauSimulator corresponding to the input stabilizer state.
        If a stim.TableauSimulator it is assumed that this simulator has already had the appropriate 
        inverse tableau value instantiated.

    desired_bitstring : str
        A string of zeros and ones corresponding to the bit string being measured.

    P : str or stim.PauliString
        The first pauli string index.

    Q : str or stim.PauliString
        The second pauli string index.
        
    Returns
    -------
    A complex number corresponding to the value of the phi function.
    """
    
    # start by getting the pauli string which maps the all-zeros string to the target bitstring.
    initial_pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in desired_bitstring]))
    # map P and Q to stim.PauliString if needed.
    if isinstance(P, str):
        P = stim.PauliString(P)
    if isinstance(Q, str):
        Q = stim.PauliString(Q)
    
    # combine this initial pauli string with the two input paulis
    eff_P = initial_pauli_string*P
    eff_Q = Q*initial_pauli_string

    # now get the bit strings which need their amplitudes extracted from the input stabilizer state and get
    # the corresponding phase corrections.
    all_zeros = '0'*len(eff_P)
    phase1, bitstring1 = pauli_phase_update(str(eff_P), all_zeros, dual=True)
    phase2, bitstring2 = pauli_phase_update(str(eff_Q), all_zeros)

    amp1 = amplitude_of_state(tableau, bitstring1, only_phase=True)
    amp2 = amplitude_of_state(tableau, bitstring2, only_phase=True).conjugate()  # The second amplitude also needs a complex conjugate applied

    # now apply the phase corrections. 
    amp1*=phase1
    amp2*=phase2
     
    # calculate phi.
    phi = amp1*amp2
    return phi

def slow_bulk_phi(tableau: Union[stim.Tableau, stim.TableauSimulator],
             desired_bitstring: str,
             Ps: list[Union[str, stim.PauliString]],
             Qs: list[Union[str, stim.PauliString]]) -> _np.ndarray[_np.complex128]:
    """
    Computes the phi function for multiple (P, Q) pairs at once while caching and reusing intermediate
    values computed via pauli_phase_update_all_zeros and amplitude_of_state.

    Parameters
    ----------
    tableau : stim.Tableau or stim.TableauSimulator
        A stim Tableau or TableauSimulator corresponding to the input stabilizer state.
    
    desired_bitstring : str
        A string of zeros and ones corresponding to the measured bitstring.
    
    Ps : list[Union[str, stim.PauliString]]
        List of Pauli string indices for the first operator in each phi computation.
        Can be either a string or a stim.PauliString (but all values are assumed to be
        the same type).
    
    Qs : list[Union[str, stim.PauliString]]
        List of Pauli string indices for the second operator in each phi computation.
        Can be either a string or a stim.PauliString (but all values are assumed to be
        the same type).

    Returns
    -------
    np.ndarray[np.complex128]
        An array of computed phi values. Each phi will be one of {0, ±1, ±i}.
    """
    if len(Ps) != len(Qs):
        raise ValueError("Lists of Ps and Qs must be of the same length.")
    if len(Ps) == 0:
        return []

    num_qubits = len(desired_bitstring)

    # (1) Build the initial pauli string mapping the all-zeros state to the desired_bitstring.
    initial_pauli_str = stim.PauliString(''.join('I' if bit == '0' else 'X' for bit in desired_bitstring))

    # (2) Convert the input Ps and Qs to stim.PauliString objects and cache unique ones.
    unique_Ps: dict[str, stim.PauliString] = {}
    unique_Qs: dict[str, stim.PauliString] = {}

    list_P_str = []  # will store canonical representation of the original P for every pair
    list_Q_str = []  # same for Q

    if not isinstance(Ps[0], stim.PauliString):
        Ps = [stim.PauliString(P_val) for P_val in Ps]
    if not isinstance(Qs[0], stim.PauliString):
        Qs = [stim.PauliString(Q_val) for Q_val in Qs]
    
    for P_val, Q_val in zip(Ps, Qs):
        # Use their string representation as canonical keys.
        key_P = str(P_val)
        key_Q = str(Q_val)
        list_P_str.append(key_P)
        list_Q_str.append(key_Q)
        unique_Ps[key_P] = P_val
        unique_Qs[key_Q] = Q_val

    # (3) Compute effective pauli strings for each unique P and unique Q.
    #    They are given by:
    #         effective P = initial_pauli_str * P  with dual=True when calling pauli_phase_update.
    #         effective Q = Q * initial_pauli_str  (default dual=False).
    eff_P_phase_cache: dict[str, tuple[complex, str]] = {}
    eff_Q_phase_cache: dict[str, tuple[complex, str]] = {}

    # For each unique P, compute effective pauli string and then call pauli_phase_update.
    unique_eff_Ps_by_unique_Ps = {}
    for key, P_obj in unique_Ps.items():
        eff_P = initial_pauli_str * P_obj
        # Use canonical string representation of effective pauli as key.
        key_eff = str(eff_P)
        # Process pauli_phase_update for effective P with dual=True.
        eff_P_phase_cache[key_eff] = pauli_phase_update_all_zeros(key_eff, dual=True)
        unique_eff_Ps_by_unique_Ps[key] = key_eff

    # For each unique Q, do similar.
    unique_eff_Qs_by_unique_Qs = {}
    for key, Q_obj in unique_Qs.items():
        eff_Q = Q_obj * initial_pauli_str
        key_eff = str(eff_Q)
        eff_Q_phase_cache[key_eff] = pauli_phase_update_all_zeros(key_eff)
        unique_eff_Qs_by_unique_Qs[key] = key_eff        

    # (4) Now, from the phase update results, collect all unique output bitstrings.
    unique_bitstrings = set()
    for phase, bitstr in eff_P_phase_cache.values():
        unique_bitstrings.add(bitstr)
    for phase, bitstr in eff_Q_phase_cache.values():
        unique_bitstrings.add(bitstr)

    # Cache amplitude_of_state for each unique bitstring.
    cached_amplitudes: dict[str, complex] = {}
    unique_amplitudes = bulk_amplitude_of_state(tableau, unique_bitstrings, True)
    for bitstr, amp in zip(unique_bitstrings, unique_amplitudes):
        cached_amplitudes[bitstr] = amp
        
    # (5) Assemble the result for each (P, Q) pair.
    # Retrieve the effective pauli's phase update using the already-computed caches.
    result_phis = _np.empty(len(list_P_str), dtype= _np.complex128)
    for i, (key_P, key_Q) in enumerate(zip(list_P_str, list_Q_str)):
        # Get the effective pauli for P corresponding to the original P key.
        key_eff_P = unique_eff_Ps_by_unique_Ps[key_P]
        phase1, bitstring1 = eff_P_phase_cache[key_eff_P]

        # Similarly for Q.
        key_eff_Q = unique_eff_Qs_by_unique_Qs[key_Q]
        phase2, bitstring2 = eff_Q_phase_cache[key_eff_Q]

        amp1 = cached_amplitudes[bitstring1]
        amp2 = cached_amplitudes[bitstring2]

        # Note the conjugation for Q's amplitude per phi logic.
        amp_val = (phase1 * amp1) * (phase2 * amp2.conjugate())
        result_phis[i] = amp_val

    return result_phis

def alpha(errorgen: Union[_LSE, _LEEL], tableau: Union[stim.Tableau, stim.TableauSimulator], desired_bitstring: str) -> float:
    """
    First-order error generator sensitivity function for probability.
    
    Parameters
    ----------
    errorgen : `LocalStimElementaryErrorgenLabel` or `LocalElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau or stim.TableauSimulator
        Stim Tableau or TableauSimulator corresponding to the stabilizer state to calculate the sensitivity for.
        
    desired_bitstring : str
        Bit string to calculate the sensitivity for.

    Returns
    -------
    sensitivity : float
        Linear sensitivity of the probability of the desired bitstring to the
        specified elementary error generator for the given stabilizer state.
    """
    sensitivity = slow_bulk_alpha([errorgen], tableau, [desired_bitstring]).item()
    
    return sensitivity

def slow_bulk_alpha(errorgens: Iterable[_LSE], tableau: Union[stim.Tableau, stim.TableauSimulator], desired_bitstrings: list[str]) -> _np.ndarray[_np.double]:
    """
    First-order error generator sensitivity function for probability.
    
    Parameters
    ----------
    errorgens : iterable of `LocalStimErrogenLabels`.
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau or stim.TableauSimulator
        Stim Tableau or TableauSimulator corresponding to the stabilizer state to calculate the sensitivity for.
        
    desired_bitstrings : list of str
        Bit string to calculate the sensitivity for.

    Returns
    -------
    sensitivities_by_bitstring : np.ndarray[np.double]
        Linear sensitivities of the probability of each desired bitstring to the
        specified elementary error generators for the given stabilizer state.
        Result is returned as a two-dimensional numpy array, with each row being
        indexed by a bitstring, and each column by an error generator.
    """
    if not errorgens or not desired_bitstrings:
        return _np.array([], dtype=_np.double)

    #pre-compute the stim.TableauSimulator we'll need for all of the computations.
    if isinstance(tableau, stim.TableauSimulator):
        # call chain that touches sim is slow_bulk_alpha -> bulk_phi -> bulk_amplitude_of_state
        # bulk_amplitude_of_state resets simulator state after use, so safe not to do so here.
        sim = tableau
    else:
        sim = stim.TableauSimulator()
        sim.set_inverse_tableau(tableau**-1)

    #pre-compute an appropriate length identity pauli string.
    identity_pauli = stim.PauliString('I'*sim.num_qubits)

    #gather all of the Ps and Qs we need for all of the error generators by looping through the errorgens list
    #so we can compute all of the phi values all at once for each bitstring.
    errgen_types = []
    Ps = []
    Qs = []
    for errorgen in errorgens:
        errgen_type = errorgen.errorgen_type
        basis_element_labels = errorgen.basis_element_labels
        if not isinstance(basis_element_labels[0], stim.PauliString):
            basis_element_labels = tuple([stim.PauliString(lbl) for lbl in basis_element_labels])

        if errgen_type == 'H':
            Ps.append(basis_element_labels[0])
            Qs.append(identity_pauli)
            errgen_types.append(errgen_type)
        elif errgen_type == 'S':
            Ps.append(basis_element_labels[0])
            Qs.append(basis_element_labels[0])
            Ps.append(identity_pauli)
            Qs.append(identity_pauli)
            errgen_types.append(errgen_type)
        elif errgen_type == 'C':
            Ps.append(basis_element_labels[0])
            Qs.append(basis_element_labels[1])
            if basis_element_labels[0].commutes(basis_element_labels[1]):
                Ps.append(basis_element_labels[0]*basis_element_labels[1])
                Qs.append(identity_pauli)
                errgen_types.append('C1') #label this differently as a flag we need the second term later on
            else:
                errgen_types.append(errgen_type)        
        else: # A
            Ps.append(basis_element_labels[1])
            Qs.append(basis_element_labels[0])
            if not basis_element_labels[0].commutes(basis_element_labels[1]):
                Ps.append(basis_element_labels[1]*basis_element_labels[0])
                Qs.append(identity_pauli)
                errgen_types.append('A1') #label this differently as a flag we need the second term later on
            else:
                errgen_types.append(errgen_type)

    sensitivities_by_bitstring = _np.empty((len(desired_bitstrings), len(errorgens)), dtype=_np.double)
    #bulk compute the phi values for each bitstring
    for i, desired_bitstring in enumerate(desired_bitstrings):
        phis = bulk_phi(sim, desired_bitstring, Ps, Qs)

        #loop through all of the phi values and get the sensitivity for each error generator.
        running_phi_index = 0
        for j, errgen_type in enumerate(errgen_types):            
            if errgen_type == 'H':
                sensitivity = 2*phis[running_phi_index].imag
                running_phi_index+=1
            elif errgen_type == 'S':
                sensitivity = (phis[running_phi_index] - phis[running_phi_index+1]).real
                running_phi_index+=2                    
            elif errgen_type == 'C': 
                sensitivity = 2*phis[running_phi_index].real
                running_phi_index+=1
            elif errgen_type == 'C1': #includes additional term because P and Q commuted
                sensitivity = 2*phis[running_phi_index].real - 2*phis[running_phi_index+1].real
                running_phi_index+=2
            elif errgen_type=='A':
                sensitivity = 2*phis[running_phi_index].imag
                running_phi_index+=1
            else: # A1, includes additional term because P and Q anticommuted
                sensitivity = 2*(phis[running_phi_index] + phis[running_phi_index+1]).imag
                running_phi_index+=2
            sensitivities_by_bitstring[i,j] = sensitivity

    return sensitivities_by_bitstring

def alpha_pauli(errorgen: _LSE, tableau: Union[stim.Tableau, stim.TableauSimulator], pauli: stim.PauliString) -> float:
    """
    First-order error generator sensitivity function for pauli expectations.
    
    Parameters
    ----------
    errorgen : `LocalStimElementaryErrorgenLabel` or `LocalElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau or stim.TableauSimulator
        Stim Tableau or TableauSimulator corresponding to the stabilizer state to calculate the sensitivity for.
        
    pauli : stim.PauliString
        Pauli to calculate the sensitivity for.

    Returns
    -------
    float
        Linear sensitivity of the expectation value of the desired pauli observable to the
        specified elementary error generator for the given stabilizer state.
    """
    sensitivity = slow_bulk_alpha_pauli([errorgen], tableau, [pauli]).item()

    return sensitivity

def _real_if_close(val: complex) -> float:
    """
    Helper function which returns the real part of a complex number and raises an exception
    if the imaginary part is non-negligible (greater than 1e-14).

    Parameters
    ----------
    val : complex
        Complex number to convert to a real float.
    """
    val_imag = val.imag
    if val_imag > 1e-14 or val_imag < -1e-14:
        raise ValueError(f'Imaginary part of val is {val_imag}, and is too large (abs(val.imag)>1e-14) to cast to real.')
    else:
        return val.real

def slow_bulk_alpha_pauli(errorgens: Iterable[_LSE], tableau: Union[stim.Tableau, stim.TableauSimulator], paulis: list[stim.PauliString]) -> _np.ndarray[_np.double]:
    """
    First-order error generator sensitivity function for pauli expectations.
    
    Parameters
    ----------
    errorgens : iterable of `LocalStimElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau or stim.TableauSimulator
        Stim Tableau or TableauSimulator corresponding to the stabilizer state to calculate the sensitivity for.
        
    pauli : stim.PauliString
        Pauli to calculate the sensitivity for.

    Returns
    -------
    np.ndarray[np.double]
        Linear sensitivities of the expectation values of the desired pauli observables to the
        specified elementary error generators for the given stabilizer state. Returned as a
        two dimensional numpy array, with rows indexed by paulis, and columns indexed by error
        generators.
    """
    #pre-compute the stim.TableauSimulator we'll need for all of the computations.
    if isinstance(tableau, stim.TableauSimulator):
        # sim is only touched by peek_observable_expectation which doesn't modify state
        # so safe not to reset following use.
        sim = tableau
    else:
        sim = stim.TableauSimulator()
        sim.set_inverse_tableau(tableau**-1)

    sensitivities_by_pauli = _np.empty((len(paulis), len(errorgens)), dtype=_np.double)

    for i, pauli in enumerate(paulis):
        for j, errorgen in enumerate(errorgens):
            errgen_type = errorgen.errorgen_type
            basis_element_labels = errorgen.basis_element_labels
            if errgen_type == 'H':
                pauli_bel_0_comm = com(pauli, basis_element_labels[0])
                if pauli_bel_0_comm is not None:
                    sign = -1j*pauli_bel_0_comm[0]
                    expectation  = sim.peek_observable_expectation(pauli_bel_0_comm[1])
                    sensitivities_by_pauli[i,j] = _real_if_close(sign*expectation)
                else: 
                    sensitivities_by_pauli[i,j] = 0 
            elif errgen_type == 'S':
                if pauli.commutes(basis_element_labels[0]):
                    sensitivities_by_pauli[i,j] = 0
                else:
                    expectation  = sim.peek_observable_expectation(pauli)
                    sensitivities_by_pauli[i,j] = _real_if_close(-2*expectation)
            elif errgen_type == 'C': 
                A = basis_element_labels[0]
                B = basis_element_labels[1]
                com_AP = A.commutes(pauli)
                if A.commutes(B):
                    if com_AP:
                        sensitivities_by_pauli[i,j] = 0
                    else:
                        com_BP = B.commutes(pauli)
                        if com_BP:
                            sensitivities_by_pauli[i,j] = 0
                        else:
                            ABP = pauli_product(A*B, pauli)
                            expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli[i,j] = _real_if_close(-4*expectation)
                else: # {A,B} = 0
                    if com_AP:
                        com_BP = B.commutes(pauli)
                        if com_BP:
                            sensitivities_by_pauli[i,j] = 0
                        else:
                            ABP = pauli_product(A*B, pauli)
                            expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli[i,j] = _real_if_close(-2*expectation)
                    else:
                        com_BP = B.commutes(pauli)
                        if com_BP:
                            ABP = pauli_product(A*B, pauli)
                            expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli[i,j] = _real_if_close(2*expectation)
                        else:
                            sensitivities_by_pauli[i,j] = 0
            else: # A
                A = basis_element_labels[0]
                B = basis_element_labels[1]
                com_AP = A.commutes(pauli)
                if A.commutes(B):
                    com_BP = B.commutes(pauli)
                    if com_AP:
                        if com_BP:
                            sensitivities_by_pauli[i,j] = 0
                        else:
                            ABP = pauli_product(A*B, pauli)
                            expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli[i,j] = _real_if_close(1j*2*expectation)
                    else:
                        if com_BP:
                            ABP = pauli_product(A*B, pauli)
                            expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli[i,j] = _real_if_close(-1j*2*expectation)
                        else:
                            sensitivities_by_pauli[i,j] = 0
                else: # {A,B} = 0
                    if com_AP:
                        sensitivities_by_pauli[i,j] = 0
                    else:
                        com_BP = B.commutes(pauli)
                        if com_BP:
                            sensitivities_by_pauli[i,j] = 0
                        else:
                            ABP = pauli_product(A*B, pauli)
                            expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli[i,j] = _real_if_close(1j*4*expectation)
    return sensitivities_by_pauli            

def _bitstring_to_int(bitstring: Union[str, tuple]) -> int:
    if isinstance(bitstring, str):
        #  If the input is a string, convert it directly
        return int(bitstring, 2)
    elif isinstance(bitstring, tuple):
        #  If the input is a tuple, join the elements to form a string
        return int(''.join(bitstring), 2)
    else:
        raise ValueError("Input must be either a string or a tuple of '0's and '1's")

# alias in cython implementations.
try:
    from pygsti.tools.errorgenpropagation import fasterrgencalc as _fc
    pauli_phase_update_all_zeros: PauliPhaseZerosUpdater = _fc.fast_pauli_phase_update_all_zeros
    pauli_phase_update: PauliPhaseUpdater = _fc.fast_pauli_phase_update
    amplitude_of_state: AmplitudeOfStateType = _fc.fast_amplitude_of_state
    bulk_amplitude_of_state: BulkAmplitudeOfStateType = _fc.fast_bulk_amplitude_of_state
    bulk_phi: BulkPhiType = _fc.fast_bulk_phi
    _raw_bulk_alpha: BulkAlphaType = _fc.fast_bulk_alpha
    _raw_bulk_alpha_pauli: BulkAlphaPauliType = _fc.fast_bulk_alpha_pauli    
except ImportError:
    msg = 'Could not import cython module `fasterrgencalc`. This may indicate that your cython extensions for pyGSTi failed to '\
          +'properly build. Lack of cython extensions can result in significant performance degredation so we recommend trying to rebuild them. '\
           'Falling back to python implementation for: pauli_phase_update_all_zeros, pauli_phase_update, amplitude_of_state, ' \
           'bulk_phi, bulk_alpha and bulk_alpha_pauli.'
    _warnings.warn(msg)
    pauli_phase_update_all_zeros: PauliPhaseZerosUpdater = slow_pauli_phase_update_all_zeros
    pauli_phase_update: PauliPhaseUpdater = slow_pauli_phase_update
    amplitude_of_state: AmplitudeOfStateType = slow_amplitude_of_state
    bulk_amplitude_of_state: BulkAmplitudeOfStateType = slow_bulk_amplitude_of_state
    bulk_phi: BulkPhiType = slow_bulk_phi
    _raw_bulk_alpha: BulkAlphaType = slow_bulk_alpha
    _raw_bulk_alpha_pauli: BulkAlphaPauliType = slow_bulk_alpha_pauli 


def bulk_alpha(
    errorgens: Iterable[_LSE],
    tableau: Union[stim.Tableau, stim.TableauSimulator],
    desired_bitstrings: list[str],
    num_workers: Union[int, str, None] = None,
    pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None,
) -> _np.ndarray:
    """
    First-order error generator sensitivity function for probability.

    Calculates the sensitivity of stabilizer measurement probabilities across bitstrings
    to specified elementary error generators. Supports both serial execution
    and parallel 2D grid partitioning across bitstrings and generators.

    Parameters
    ----------
    errorgens : iterable of `LocalStimErrorgenLabel`
        Error generator labels for which to calculate sensitivities.
    tableau : stim.Tableau or stim.TableauSimulator
        Stabilizer state tableau being measured.
    desired_bitstrings : list of str
        Bitstrings to calculate sensitivities for.
    num_workers : int, str, or None, optional
        Target worker count (defaults to None for sequential execution; pass 'auto'
        or an integer to enable parallel execution).
    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse across sequential workflow stages.

    Returns
    -------
    numpy.ndarray
        2D array of sensitivities with shape `(len(desired_bitstrings), len(errorgens))`.
    """
    gens = list(errorgens)
    n_e = len(gens)
    n_b = len(desired_bitstrings)
    if n_e == 0 or n_b == 0:
        return _np.empty((n_b, n_e), dtype=_np.double)

    workers = _determine_num_workers(num_workers) if pool is None else _resolve_pool_workers(pool)
    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")

    n_qubits = len(desired_bitstrings[0])
    _warn_if_insufficient_memory(
        estimate_bulk_alpha_peak_memory(n_qubits, n_e, n_b, workers),
        task_name="bulk_alpha"
    )

    if workers == 1 or num_workers is None and pool is None:
        return _raw_bulk_alpha(gens, tableau, desired_bitstrings)

    if not _should_parallelize_alpha(n_qubits, n_e, n_b, num_workers=workers):
        return _raw_bulk_alpha(gens, tableau, desired_bitstrings)

    worker = _BulkAlphaWorker(tableau)
    return _parallel_2d_grid_map(worker, desired_bitstrings, gens, num_workers=workers, pool=pool)


def bulk_alpha_pauli(
    errorgens: Iterable[_LSE],
    tableau: Union[stim.Tableau, stim.TableauSimulator],
    paulis: list[stim.PauliString],
    num_workers: Union[int, str, None] = None,
    pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None,
) -> _np.ndarray:
    """
    First-order error generator sensitivity function for Pauli expectation values.

    Parameters
    ----------
    errorgens : iterable of `LocalStimErrorgenLabel`
        Error generator labels for which to calculate sensitivities.
    tableau : stim.Tableau or stim.TableauSimulator
        Stabilizer state tableau being measured.
    paulis : list of stim.PauliString
        Pauli observables to compute expectation sensitivities for.
    num_workers : int, str, or None, optional
        Target worker count (defaults to None for sequential execution; pass 'auto'
        or an integer to enable parallel execution).
    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse across sequential workflow stages.

    Returns
    -------
    numpy.ndarray
        2D array of sensitivities with shape `(len(paulis), len(errorgens))`.
    """
    gens = list(errorgens)
    n_e = len(gens)
    n_p = len(paulis)
    if n_e == 0 or n_p == 0:
        return _np.empty((n_p, n_e), dtype=_np.double)

    workers = _determine_num_workers(num_workers) if pool is None else _resolve_pool_workers(pool)
    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")

    n_qubits = len(paulis[0])
    _warn_if_insufficient_memory(
        estimate_bulk_alpha_peak_memory(n_qubits, n_e, n_p, workers),
        task_name="bulk_alpha_pauli"
    )

    if workers == 1 or num_workers is None and pool is None:
        return _raw_bulk_alpha_pauli(gens, tableau, paulis)

    if not _should_parallelize_alpha(n_qubits, n_e, n_p, num_workers=workers):
        return _raw_bulk_alpha_pauli(gens, tableau, paulis)

    worker = _BulkAlphaPauliWorker(tableau)
    return _parallel_2d_grid_map(worker, paulis, gens, num_workers=workers, pool=pool)

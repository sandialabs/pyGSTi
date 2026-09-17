# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# filename: fasterrgencalc.pyx
# cython: debug=False

#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as np
cimport numpy as np
cimport cython
from cpython.unicode cimport PyUnicode_FromStringAndSize, PyUnicode_AsUTF8
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import stim
from libc.math cimport pow

_POSSIBLE_PHASE_VALUES_INDEXED_NEGJ_J_NEG1 = np.array([1, -1, 1j, -1j,
                                            -1, 1, -1j, 1j,
                                            -1j, 1j, 1, -1,
                                            1j, -1j, -1, 1,
                                            -1, 1, -1j, 1j,
                                            1, -1, 1j, -1j,
                                            1j, -1j, -1, 1,
                                            -1j, 1j, 1, -1], dtype=np.complex128)
_POSSIBLE_PHASE_VALUES_INDEXED_NEGJ_J_NEG1.flags.writeable = False                               

cdef const np.complex128_t[::1] POSSIBLE_PHASE_VALUES_INDEXED_NEGJ_J_NEG1 = _POSSIBLE_PHASE_VALUES_INDEXED_NEGJ_J_NEG1

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline np.complex128_t get_phase_from_array(int count_negj, int count_j, int count_negone):
    return POSSIBLE_PHASE_VALUES_INDEXED_NEGJ_J_NEG1[count_negone + 2*count_j + 8*count_negj]

# Optimized implementations of functions from errgenproptools.py

# Optimized version of pauli_phase_update_all_zeros,
# reworked to work on the string representation of a stim.PauliString.
#
# The expected format is:
#    prefix + operator_chars
#
# Accepted prefixes and their ASCII codes:
#    '+' (43)  -> +1,
#    '-' (45)  -> -1,
#    "+i" (43,105) -> +1j,
#    "-i" (45,105) -> -1j.
#
# Following the prefix, each operator character (one per qubit) is one of:
#    '_' (95) -> identity,
#    'X' (88) -> X,
#    'Y' (89) -> Y,
#    'Z' (90) -> Z.
#
# We now avoid converting these operator characters to an intermediate integer code:
# instead, we pass the ASCII code directly into the inline helper get_phase0_ascii().

cdef inline np.complex128_t get_phase0_ascii(unsigned char op, bint dual):
    """
    Returns the phase correction for an operator character when the input bit is '0'.
    
    For op:
      '_' -> 1
      'X' -> 1
      'Y' -> 1j  if dual is False, or -1j if dual is True.
      'Z' -> 1
    For any unrecognized operator, defaults to 1.
    """
    if op != 89:  # 'Y' ASCII 89
        return 1
    elif dual:
        return -1j
    return 1j

cdef inline bint pauli_flip_ascii(unsigned char op):
    """
    Returns whether the operator (given by its ASCII code) causes a bit flip.
    
    For op:
      '_' -> False,
      'X' -> True,
      'Y' -> True,
      'Z' -> False.
    Any unrecognized operator is treated as no flip.
    """
    return (op == 88 or op == 89)  # 'X' (88) or 'Y' (89)

@cython.wraparound(False)   # Deactivate negative indexing.
cpdef tuple fast_pauli_phase_update_all_zeros(str pauli_str, bint dual=False):
    """
    Optimized specialized version of pauli_phase_update_all_zeros specialized to the case
    where the input bitstring is all zeros.
    
    Parameters
    ----------
    pauli_str : str
        The pauli operator as a string. It may have an optional sign prefix:
             "+" or "-", or "+i" or "-i".
        The remainder of the string contains operator characters (one per qubit):
             '_' (identity), 'X', 'Y', or 'Z'.
    dual : bool, optional
        If True, the dual phase rules are applied (which swap the sign of Y's phase).
    
    Returns
    -------
    Tuple[complex, str]
         A tuple (overall_phase, output_bitstring):
            overall_phase : complex - the cumulative phase factor.
            output_bitstring : str - the updated bitstring after applying the pauli
                                     (with bits flipped where the pauli indicates a flip).
    """
    cdef:
        int prefix_len = 0
        int i, n, pauli_len
        int count_y = 0
        np.complex128_t overall_phase = 1.0
        np.complex128_t sign = 1.0
        unsigned char op   # declare op outside the loop
        char* out_buffer

    # Convert pauli_str to ASCII bytes. (UTF-8 is backwards compatible with ASCII)
    cdef const char* p = PyUnicode_AsUTF8(pauli_str)

    pauli_len = len(pauli_str)

    # Determine the sign from the prefix.
    # The possible prefixes are: "+", "-", "+i", "-i".
    if pauli_len > 0: 
        if p[0] == ord('+'):
            if p[1] == ord('i'):
                sign = 1j
                prefix_len = 2
            else:
                sign = 1
                prefix_len = 1
        elif p[0] == ord('-'):
            if p[1] == ord('i'):
                sign = -1j
                prefix_len = 2
            else:
                sign = -1
                prefix_len = 1
        else:
            # No explicit sign provided; default to positive.
            sign = 1
            prefix_len = 0
    else:
        # empty pauli string
        raise ValueError("pauli_str must not be empty")

    n = pauli_len - prefix_len  # Number of operator characters.
    # Allocate a C char buffer for output with space for n characters plus a null terminator.
    out_buffer = <char*> PyMem_Malloc((n + 1) * sizeof(char))
    if not out_buffer:
        raise MemoryError("Failed to allocate memory for output buffer")

    # We do not need to initialize the buffer. Just specify the answer in the loop.

    for i in range(n):
        op = p[prefix_len + i]
        if (op == 89):
            count_y += (op == 89)
            count_y &= 0x03 # we only need 0,1,2,3.
            out_buffer[i] = 49
        elif (op == 88):
            out_buffer[i] = 49
        else:
            out_buffer[i] = 48
    out_buffer[n] = 0  # null termination

    # Look up the result of the matrix multiplication.    
    if dual:
        if count_y == 1:
            overall_phase = -1j
        elif count_y == 2:
            overall_phase = -1
        elif count_y == 3:
            overall_phase = 1j
    else:
        if count_y == 1:
            overall_phase = 1j
        elif count_y == 2:
            overall_phase = -1
        elif count_y == 3:
            overall_phase = -1j

    overall_phase *= sign
    
    # Create a Python string from the out_buffer.
    cdef object out_pystr = PyUnicode_FromStringAndSize(out_buffer, n)
    PyMem_Free(out_buffer)
    
    return overall_phase, out_pystr

# This function is an optimized implementation of pauli_phase_update.
# It applies a pauli operator to a given bitstring (both provided as Python str).
#
# The pauli string may optionally include a sign prefix:
#     "+"  → +1, "-" → -1, "+i" → +1j, and "-i" → -1j.
# If a sign prefix is present, it is removed from the operator portion and its
# corresponding complex value multiplied into the overall phase.
#
# After the optional sign prefix, the pauli string must contain operator characters
# one per qubit. Each operator must be one of:
#
#     '_'  (ASCII 95)  → identity (1)
#     'X'  (ASCII 88)  → X (flip bit, phase = 1)
#     'Y'  (ASCII 89)  → Y (flip bit, phase = 1j if bit==0, -1j if bit==1 for non-dual)
#     'Z'  (ASCII 90)  → Z (no flip, phase = 1 if bit==0, -1 if bit==1)
#
# When dual is True, the sign of the Y phase is swapped.
#
# The bitstring is a string of '0's and '1's.
#
# The function returns a tuple (overall_phase, output_bitstring), where:
#  - overall_phase is the product of all phase factors,
#  - output_bitstring is the updated bitstring after flipping bits where required.

cdef inline np.complex128_t get_phase_ascii(unsigned char op, bint bit, bint dual):
    """
    Returns the phase factor at one qubit given:
       op: the operator ASCII code,
       bit: current input bit (0 if '0', 1 if '1'),
       dual: if True, uses dual phase rules.
    
    Mapping for dual==False:
      '_' (95): always 1.
      'X' (88): always 1.
      'Y' (89): if bit==0 then 1j, if bit==1 then -1j.
      'Z' (90): if bit==0 then 1, if bit==1 then -1.
    
    For dual==True:
      '_' (95): always 1.
      'X' (88): always 1.
      'Y' (89): if bit==0 then -1j, if bit==1 then 1j.
      'Z' (90): same as above.
    
    For any unrecognized op, returns 1.
    """
    if op == 95:       # '_' ASCII 95
        return 1
    elif op == 88:     # 'X' ASCII 88
        return 1
    elif op == 89:     # 'Y' ASCII 89
        if not dual:
            if not bit:
                return 1j
            else:
                return -1j
        else:
            if not bit:
                return -1j
            else:
                return 1j
    elif op == 90:     # 'Z' ASCII 90
        if not bit:
            return 1
        else:
            return -1
    else:
        return 1

@cython.wraparound(False)   # Deactivate negative indexing.
@cython.boundscheck(False)
cpdef tuple fast_pauli_phase_update(str pauli_str, str bit_str, bint dual=False):
    """
    Optimized specialized version of pauli_phase_update.
    
    Parameters
    ----------
    pauli_str : str
        The pauli operator as a string. It may have an optional sign prefix:
             "+" or "-", or "+i" or "-i".
        The remainder of the string contains operator characters (one per qubit):
             '_' (identity), 'X', 'Y', or 'Z'.
    bit_str : str
        The input bitstring as a string of '0's and '1's.
    dual : bool, optional
        If True, the dual phase rules are applied (which swap the sign of Y's phase).
    
    Returns
    -------
    Tuple[complex, str]
         A tuple (overall_phase, output_bitstring):
            overall_phase : complex - the cumulative phase factor.
            output_bitstring : str - the updated bitstring after applying the pauli
                                     (with bits flipped where the pauli indicates a flip).
    """
    cdef:
        int prefix_len = 0
        int count_negj = 0
        int count_j = 0
        int count_negone = 0
        int i, n, pauli_len, bit_len
        np.complex128_t overall_phase = 1.0
        unsigned char op   # for each operator character
        bint bit_val     # current bit value, 0 or 1
        char* out_buffer

    # Convert pauli_str and bit_str to ASCII bytes. (UTF-8 is backwards compatible with ASCII)
    cdef const char* p = PyUnicode_AsUTF8(pauli_str)
    cdef const char* b = PyUnicode_AsUTF8(bit_str)

    pauli_len = len(pauli_str)
    bit_len = len(bit_str)

    # Parse sign prefix from pauli_str.
    # The possible prefixes are: "+", "-", "+i", "-i".
    if pauli_len > 0:
        if p[0] == ord('+'):
            if pauli_len >= 2 and p[1] == ord('i'):
                count_j = 1
                prefix_len = 2
            else:
                prefix_len = 1
        elif p[0] == ord('-'):
            if pauli_len >= 2 and p[1] == ord('i'):
                count_negj = 1
                prefix_len = 2
            else:
                count_negone = 1
                prefix_len = 1
        else:
            # No recognized sign prefix.
            prefix_len = 0
    else:
        # Empty pauli string.
        raise ValueError("pauli_str must not be empty")

    # Determine n, the number of operator characters.
    n = pauli_len - prefix_len
    if bit_len != n:
        raise ValueError("Length of bit_str must equal the number of operator characters in pauli_str (after sign prefix).")

    # Allocate a C char buffer for the output bitstring.
    out_buffer = <char*> PyMem_Malloc((n + 1) * sizeof(char))
    if not out_buffer:
        raise MemoryError("Failed to allocate memory for output buffer")

    if dual: # specializing on dual.
        for i in range(n):
            op = p[prefix_len + i]
            bit_val = b[i] - 48
            if (op == 89) and bit_val: # Y ASCII 89
                count_j = (count_j + 1) & 0x03
            elif (op == 89):
                count_negj = (count_negj + 1) & 0x03
            elif (op == 90) and bit_val:  # Z ASCII 90
                count_negone = (count_negone + 1) & 0x01

    else:
        for i in range(n):
            op = p[prefix_len + i]
            bit_val = b[i] - 48
            if (op == 89) and bit_val:
                count_negj = (count_negj + 1) & 0x03
            elif (op == 89):
                count_j = (count_j + 1) & 0x03
            elif (op == 90) and bit_val:
                count_negone = (count_negone + 1) & 0x01

    # Process each qubit position.
    for i in range(n):
        # Get the operator character from the pointer.
        op = p[prefix_len + i]
        # Get the corresponding input bit from bit_str.
        # (Assume bit_str characters are '0' or '1')
        # 49 ASCII '1'
        bit_val = b[i] - 48
        # Determine the output bit.
        # If the pauli operator flips, then output the inverted bit.
        if pauli_flip_ascii(op):
            # Flip: output '1' if input was '0' and vice versa.
            out_buffer[i] = (not bit_val) + 48
        else:
            # No flip: just copy the input bit.
            out_buffer[i] = b[i]
    out_buffer[n] = 0  # null-terminate output C string

    # Since each term only has phases of +/-1, +/-1j, just precompute the options and store in a table.
    overall_phase = get_phase_from_array(count_negj, count_j, count_negone)

    # Create a Python string from the out_buffer.
    cdef object out_pystr = PyUnicode_FromStringAndSize(out_buffer, n)
    PyMem_Free(out_buffer)

    return overall_phase, out_pystr

cdef inline void _restore_sim(object sim, object orig_tableau_inverse, bint had_simulator):
    if had_simulator:
        sim.set_inverse_tableau(orig_tableau_inverse)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef int _random_support(object sim, object orig_tableau_inverse, int n, unsigned char* is_random):
    """
    Compute (num_random, is_random_mask) for Z measurements in the canonical
    left-to-right sequence *on the original state*.

    IMPORTANT: This routine must restore simulator state before returning
    (because it uses postselects to compute the canonical reference branch).
    """
    cdef int q, z, num_random
    cdef object peek_z = sim.peek_z
    cdef object postselect_z = sim.postselect_z

    num_random = 0
    for q in range(n):
        z = peek_z(q)
        if z == 0:
            num_random += 1
            is_random[q] = 1
        else:
            is_random[q] = 0
        # Follow your Python random_support behavior: postselect to forced_bit
        # to propagate constraints deterministically and make later peek_z values canonical.
        postselect_z(q, desired_value=(z == -1))

    # restore        
    sim.set_inverse_tableau(orig_tableau_inverse)

    return num_random


@cython.wraparound(False)
@cython.boundscheck(False)
cdef unsigned char* _compute_phase_reference(object sim,
                                             object orig_tableau_inverse,
                                             int n):
    """
    Compute reference bitstring as an unsigned char array of length n.
    Matches updated Python compute_phase_reference: record forced bits; postselect only on random qubits.

    Returns: malloc'd buffer (caller must PyMem_Free).
    """
    cdef int q, z
    cdef unsigned char* refarr
    cdef object peek_z = sim.peek_z
    cdef object postselect_z = sim.postselect_z

    refarr = <unsigned char*> PyMem_Malloc(n * sizeof(unsigned char))
    if not refarr:
        raise MemoryError("allocating refarr failed")

    for q in range(n):
        z = peek_z(q)
        refarr[q] = (z == -1)
        if z == 0:
            postselect_z(q, desired_value=refarr[q])

    # restore for caller convenience (bulk code assumes it can start clean)
    sim.set_inverse_tableau(orig_tableau_inverse)
    return refarr

cdef inline bint _in_support_by_postselecting_all_zero(object sim,
                                                       object orig_tableau_inverse,
                                                       object ps,
                                                       int n,
                                                       object range_n):
    """
    Support test in the style of in_stabilizer_support:
      - Apply X on qubits where desired bit is 1, mapping |desired> -> |0..0>
      - Attempt to postselect all Z outcomes to 0 in one bulk call.

    Returns True/False. Always restores simulator.
    """
    cdef object postselect_z = sim.postselect_z
    # Map desired -> all-zero by flipping 1 bits.
    sim.do_pauli_string(ps)

    try:
        # Bulk postselect (faster in practice for Stim).
        postselect_z(range_n, desired_value=False)
    except ValueError:
        sim.set_inverse_tableau(orig_tableau_inverse)
        return False

    sim.set_inverse_tableau(orig_tableau_inverse)
    return True

@cython.wraparound(False)
@cython.boundscheck(False)
cdef object _get_ix_pauli_from_mask(const unsigned char* bs,
                                    int n):
    """
    Returns a stim.PauliString consisting only of I/X (no prefix needed),
    with X where bs[q]==1.
    """
    cdef int q
    cdef char* buf
    cdef object ps
    cdef str key

    # Build IX... string
    buf = <char*> PyMem_Malloc((n + 1) * sizeof(char))
    if not buf:
        raise MemoryError("Failed to allocate IX buffer")
    for q in range(n):
        buf[q] = 88 if bs[q] else 73   # 'X' or 'I'
    buf[n] = 0
    key = PyUnicode_FromStringAndSize(buf, n)
    PyMem_Free(buf)

    ps = stim.PauliString(key)
    return ps

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline np.complex128_t _phase_isolation_pass(object sim,
                                                  object orig_tableau_inverse,
                                                  const unsigned char* bs,
                                                  const unsigned char* refarr,
                                                  const unsigned char* is_random,
                                                  int n):
    """
    Implements your third pass:
      - postselect away non-(ref or desired) components
      - map ref -> |0..0>, desired -> |0..01> (difference on qubit 0)
      - return phase from peek_bloch(0) as ±1,±i
    Assumes simulator is already set to orig_tableau_inverse by caller.
    """
    cdef int q
    cdef unsigned char desired_bit, ref_bit
    cdef bint found = False
    cdef object postselect_z = sim.postselect_z
    cdef object swap_op = sim.swap
    cdef object x_op = sim.x
    cdef object cnot_op = sim.cnot
    cdef str s

    for q in range(n):
        desired_bit = bs[q]
        ref_bit = refarr[q]

        if desired_bit == ref_bit:
            # In Python bulk version you only postselect if is_random[q] in this branch.
            # Using that here reduces postselects (and should be correct because deterministic
            # measurements don't need conditioning).
            if is_random[q]:
                postselect_z(q, desired_value=ref_bit)
            if desired_bit:
                x_op(q)

        elif not found:
            found = True
            if q != 0:
                swap_op(0, q)
            if ref_bit:
                x_op(0)

        else:
            cnot_op(0, q)
            postselect_z(q, desired_value=ref_bit)

    # If desired differs from ref, found must be true.
    # If it isn't, something went wrong.
    if not found:
        sim.set_inverse_tableau(orig_tableau_inverse)
        raise RuntimeError('found should be True during normal operation, something went wrong.')

    s = str(sim.peek_bloch(0))
    sim.set_inverse_tableau(orig_tableau_inverse)
    
    if s == "+X":
        return 1.0
    elif s == "-X":
        return -1.0
    elif s == "+Y":
        return 1j
    else:
        return -1j


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.complex128_t, ndim=1] fast_bulk_amplitude_of_state(object tableau,
                                                                      list desired_states,
                                                                      bint only_phase):
    """
    Bulk version of amplitude extraction, modeled after updated_python_amplitude_implementations.slow_bulk_amplitude_of_state.
    desired_states is a list of bitstrings (str of 0/1). (You can extend to accept PauliString later.)
    Returns numpy array of complex128 amplitudes.
    """
    cdef:
        bint had_simulator
        object sim, orig_tableau_inverse
        int n, m, i, q
        int num_random
        double magnitude
        unsigned char* is_random
        unsigned char* refarr = NULL
        np.ndarray[np.complex128_t, ndim=1] out
        np.complex128_t[::1] out_view
        const char* dptr
        unsigned char* bs
        bint same
        np.complex128_t phase_factor
        bint is_str = False
        bint is_pauli_str = False
        str ps_str
        const char* pptr
        int ps_len
        object ps
        unsigned char ch
        int prefix_len
    
    m = len(desired_states)
    out = np.empty(m, dtype=np.complex128)
    out_view = out

    if m == 0:
        return out

    #Validate the first entry of desired_states
    if isinstance(desired_states[0], str):
        is_str = True
    elif isinstance(desired_states[0], stim.PauliString):
        is_pauli_str = True
    else:
        raise ValueError(f'Unsupported input type {type(desired_states[0])} for desired_states, expected string or stim.PauliString entries.')

    # 1) Instantiate/reuse sim
    if isinstance(tableau, stim.Tableau):
        sim = stim.TableauSimulator()
        orig_tableau_inverse = tableau**-1
        sim.set_inverse_tableau(orig_tableau_inverse)
        had_simulator = False
    elif isinstance(tableau, stim.TableauSimulator):
        sim = tableau
        orig_tableau_inverse = sim.current_inverse_tableau()
        had_simulator = True
    else:
        raise ValueError(
            f"Unsupported input type {type(tableau)}; "
            "expected stim.Tableau or stim.TableauSimulator"
        )

    n = sim.num_qubits
    if len(desired_states[0]) != n:
        raise ValueError("desired_state length must equal number of qubits")
    cdef object range_n = range(n)

    # 2) Precompute random support (once)
    is_random = <unsigned char*> PyMem_Malloc(n * sizeof(unsigned char))
    if not is_random:
        raise MemoryError("allocating is_random failed")
    
    num_random = _random_support(sim, orig_tableau_inverse, n, is_random)

    if only_phase:
        magnitude = 1.0
    else:
        if num_random > 2148:
            PyMem_Free(is_random)
            _restore_sim(sim, orig_tableau_inverse, had_simulator)
            raise RuntimeError("Number of random bits is greater than 2148; magnitude will underflow.")
        magnitude = pow(2.0, - (num_random / 2.0))

    # 3) Loop desired states
    for i in range(m):
        # Convert desired_state to bs buffer
        bs = <unsigned char*> PyMem_Malloc(n * sizeof(unsigned char))
        if not bs:
            PyMem_Free(is_random)
            if refarr != NULL:
                PyMem_Free(refarr)
            _restore_sim(sim, orig_tableau_inverse, had_simulator)
            raise MemoryError("allocating desired_state buffer failed")
        if is_str:
            dptr = PyUnicode_AsUTF8(desired_states[i])
            for q in range(n):
                bs[q] = dptr[q] - 48  # '0'/'1' -> 0/1
            ps = _get_ix_pauli_from_mask(bs, n)
        else:
            # stim.PauliString case. We interpret it as an X-mask mapping |0..0> -> |desired>.
            # We parse str(ps) and treat positions with 'X' as 1 and all others as 0.
            ps = desired_states[i]
            ps_str = str(ps)
            pptr = PyUnicode_AsUTF8(ps_str)
            ps_len = len(ps_str)
            prefix_len = 0

            # parse optional prefix: '+', '-', '+i', '-i'
            if ps_len == n + 2 and (pptr[0] == 43 or pptr[0] == 45) and pptr[1] == 105:  # ±i
                prefix_len = 2
            elif ps_len == n + 1 and (pptr[0] == 43 or pptr[0] == 45):  # ±
                prefix_len = 1
            elif ps_len == n:
                prefix_len = 0
            else:
                PyMem_Free(bs)
                PyMem_Free(is_random)
                if refarr != NULL:
                    PyMem_Free(refarr)
                _restore_sim(sim, orig_tableau_inverse, had_simulator)
                raise ValueError("PauliString length does not match n qubits")

            for q in range(n):
                ch = <unsigned char>pptr[prefix_len + q]
                if ch == 88:  # 'X'
                    bs[q] = 1
                elif ch == 95:  # '_'
                    bs[q] = 0
                else:
                    PyMem_Free(bs)
                    PyMem_Free(is_random)
                    if refarr != NULL:
                        PyMem_Free(refarr)
                    _restore_sim(sim, orig_tableau_inverse, had_simulator)
                    raise ValueError("PauliString desired_state must contain only I/X (up to global phase)")

        # Support check
        if not _in_support_by_postselecting_all_zero(sim, orig_tableau_inverse, ps, n, range_n):
            out_view[i] = 0.0
            PyMem_Free(bs)
            continue

        # supported: ensure ref exists
        if refarr == NULL:
            refarr = _compute_phase_reference(sim, orig_tableau_inverse, n)

        # if desired == ref: phase=1
        same = True
        for q in range(n):
            if bs[q] != refarr[q]:
                same = False
                break
        if same:
            out_view[i] = magnitude
            PyMem_Free(bs)
            continue

        # Phase isolation pass
        phase_factor = _phase_isolation_pass(sim, orig_tableau_inverse, bs, refarr, is_random, n)
        out_view[i] = phase_factor * magnitude
        PyMem_Free(bs)

    # cleanup
    PyMem_Free(is_random)
    if refarr != NULL:
        PyMem_Free(refarr)

    return out


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.complex128_t fast_amplitude_of_state(object tableau, object desired_state, bint only_phase):
    """
    Refactored: delegate to bulk version for a single state.
    """
    cdef np.ndarray[np.complex128_t, ndim=1] arr
    arr = fast_bulk_amplitude_of_state(tableau, [desired_state], only_phase)
    return arr[0]


"""
Cython implementation of bulk_phi.

This function computes the phi function for multiple (P, Q) pairs at once,
caching intermediate values computed via pauli_phase_update_all_zeros() and amplitude_of_state().

The algorithm:
  (1) Build an initial pauli string mapping the all-zeros state to the desired bitstring.
  (2) Convert the input Ps and Qs to stim.PauliString objects (if needed) and cache their canonical
      string representations.
  (3) For each unique P (and similarly for Q), compute the effective pauli string by multiplying by the
      initial pauli string. Then compute its (phase update, bitstring) tuple via pauli_phase_update_all_zeros(),
      caching its canonical string. Also record a mapping from the original canonical string to the effective canonical string.
  (4) Collect all unique output bitstrings and cache the amplitude (via fast_amplitude_of_state) for each.
  (5) Assemble the final phi values by reusing the cached data.
"""

@cython.wraparound(False)   # Deactivate negative indexing.
@cython.boundscheck(False)
cpdef np.ndarray[np.complex128_t, ndim=1] fast_bulk_phi(object tableau, str desired_bitstring, list Ps, list Qs):
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
    list[complex]
        A list of computed phi values. Each phi will be one of {0, ±1, ±i}.
    """
    
    cdef int numPs, i, num_qubits
    cdef object P_val, Q_val 
    cdef str key_P, key_Q
    cdef list list_P_str = []
    cdef list list_Q_str = []
    cdef dict unique_Ps = {}   # maps canonical string from P -> stim.PauliString, later overridden with a tuple
    cdef dict unique_Qs = {}   # similar for Q

    # (0) Basic length check.
    numPs = len(Ps)
    if numPs != len(Qs):
        raise ValueError("Lists of Ps and Qs must be of the same length.")
    if numPs == 0:
        return np.array([], dtype=np.complex128)

    num_qubits = len(desired_bitstring)

    # Convert pauli_str and bit_str to ASCII bytes. (UTF-8 is backwards compatible with ASCII)
    cdef const char* b = PyUnicode_AsUTF8(desired_bitstring)
    cdef char* pauli_chars = <char*> PyMem_Malloc((num_qubits+1) * sizeof(char))
    if not pauli_chars:
        raise MemoryError('Failed to allocate memory.')

    # (1) Build the initial pauli string mapping all-zeros to desired_bitstring.
    for i in range(num_qubits):
        if b[i] == 48: #ASCII 0
            pauli_chars[i] = 73 # ASCII I
        else:
            pauli_chars[i] = 88 # ASCII X
    pauli_chars[num_qubits] = 0 # null-terminate
    cdef str initial_string = PyUnicode_FromStringAndSize(pauli_chars, num_qubits)
    PyMem_Free(pauli_chars)
    cdef object initial_pauli_str = stim.PauliString(initial_string)

    # (2) Convert input Ps and Qs to stim.PauliString objects if needed.
    cdef list temp_list  # temporary holder for conversion
    if not isinstance(Ps[0], stim.PauliString):
        temp_list = []
        for P_val in Ps:
            temp_list.append(stim.PauliString(P_val))
        Ps = temp_list
    if not isinstance(Qs[0], stim.PauliString):
        temp_list = []
        for Q_val in Qs:
            temp_list.append(stim.PauliString(Q_val))
        Qs = temp_list

    # Build unique dictionaries and record canonical key for each entry.
    for i in range(numPs):
        P_val = Ps[i]
        Q_val = Qs[i]
        key_P = str(P_val)
        key_Q = str(Q_val)
        list_P_str.append(key_P)
        list_Q_str.append(key_Q)
        unique_Ps[key_P] = P_val    # override unconditionally
        unique_Qs[key_Q] = Q_val

    # (3) Compute effective pauli strings for each unique P and Q.
    # Instead of storing just the effective pauli object, we store a tuple:
    #   (effective pauli, effective pauli's canonical string)
    cdef dict eff_P_phase_cache = {}  # maps effective pauli canonical string -> (phase, bitstr) tuple.
    cdef dict eff_Q_phase_cache = {}
    cdef dict unique_eff_Ps_by_unique_Ps = {}  # maps original P key -> effective pauli canonical string
    cdef dict unique_eff_Qs_by_unique_Qs = {}
    cdef object eff_P, eff_Q 
    cdef str key_eff

    for key_P, P_val in unique_Ps.items():
        eff_P = initial_pauli_str * P_val
        key_eff = str(eff_P)
        eff_P_phase_cache[key_eff] = fast_pauli_phase_update_all_zeros(key_eff, dual=True)
        unique_eff_Ps_by_unique_Ps[key_P] = key_eff

    for key_Q, Q_val in unique_Qs.items():
        eff_Q = Q_val * initial_pauli_str
        key_eff = str(eff_Q)
        eff_Q_phase_cache[key_eff] = fast_pauli_phase_update_all_zeros(key_eff)
        unique_eff_Qs_by_unique_Qs[key_Q] = key_eff

    # (4) Collect all unique output bitstrings from the phase caches.
    cdef set unique_bitstrings = set()
    cdef tuple phase_bit
    for phase_bit in eff_P_phase_cache.values():
        unique_bitstrings.add(phase_bit[1])
    for phase_bit in eff_Q_phase_cache.values():
        unique_bitstrings.add(phase_bit[1])

    cdef list unique_bitstrings_list = list(unique_bitstrings)

    # Cache amplitude_of_state for each unique bitstring.
    cdef dict cached_amplitudes = {}
    cdef str bitstr
    cdef np.ndarray[np.complex128_t, ndim=1] unique_amplitudes = fast_bulk_amplitude_of_state(tableau, unique_bitstrings_list, True)
    cdef np.complex128_t[::1] unique_amplitudes_view = unique_amplitudes
    cdef int num_unique_amplitudes = len(unique_amplitudes)

    for i in range(num_unique_amplitudes):
        cached_amplitudes[unique_bitstrings_list[i]] = unique_amplitudes_view[i]

    # (5) Assemble the result for each (P, Q) pair.
    cdef np.ndarray[np.complex128_t, ndim=1] result_phis = np.empty(numPs, dtype=np.complex128)
    cdef np.complex128_t[::1] result_phis_view = result_phis

    cdef np.complex128_t phase1, phase2, amp1, amp2, amp_val, norm_phi
    cdef str key_eff_P, key_eff_Q, str1, str2
    for i in range(numPs):
        key_P = list_P_str[i]
        key_Q = list_Q_str[i]
        key_eff_P = unique_eff_Ps_by_unique_Ps[key_P]
        phase1, str1 = eff_P_phase_cache[key_eff_P]
        key_eff_Q = unique_eff_Qs_by_unique_Qs[key_Q]
        phase2, str2 = eff_Q_phase_cache[key_eff_Q]
        amp1 = cached_amplitudes[str1]
        amp2 = cached_amplitudes[str2]
        # The Q amplitude gets conjugated per phi logic.
        amp_val = (phase1 * amp1) * (phase2 * amp2.conjugate())
        result_phis_view[i] = amp_val

    return result_phis

# Enumerate error‐gen types to small ints (no string compares later).
cdef enum ErrType:
    ET_H = 0
    ET_S = 1
    ET_C = 2
    ET_C1= 3
    ET_A = 4
    ET_A1= 5

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] fast_bulk_alpha(object errorgens_iter,
                                                  object tableau,
                                                  list desired_bitstrings):
    """
    Cython‐optimized bulk_alpha. Returns List[List[float]] of sensitivities.
    """
    cdef list errorgens = list(errorgens_iter)
    cdef int n_e = len(errorgens)
    cdef int n_b = len(desired_bitstrings)
    if n_e == 0 or n_b == 0:
        return np.array([], dtype=np.double)        
        
    #
    # 1) Build the simulator & remember how to restore it if needed.
    #
    cdef object sim
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)

    #
    # 2) Pre‐allocate identity pauli for reuse.
    #
    cdef int num_q = sim.num_qubits
    cdef object stim_PauliString = stim.PauliString
    cdef object identity_pauli = stim_PauliString('I' * num_q)

    #
    # 3) Convert each errorgen into 0–5 code and build Ps/Qs lists.
    #
    #    ET_H:   one phi
    #    ET_S:   two phis, type S
    #    ET_C:   one phi, type C
    #    ET_C1:  two phis, type C1
    #    ET_A:   one phi, type A
    #    ET_A1:  two phis, type A1
    #
    cdef int *codes = <int*> PyMem_Malloc(n_e * sizeof(int))
    if not codes:
        raise MemoryError('Could not allocate memory for error generator codes.')
    cdef list Ps = []
    cdef list Qs = []
    cdef int i
    cdef str t
    cdef tuple bl
    cdef list new_bls

    for i in range(n_e):
        t = errorgens[i].errorgen_type
        bl = errorgens[i].basis_element_labels
        # ensure PauliString objects
        if not isinstance(bl[0], stim_PauliString):
            new_bls = []
            for x in bl:
                new_bls.append(stim_PauliString(x))
            bl = tuple(new_bls)

        if t == 'H':
            Ps.append(bl[0])
            Qs.append(identity_pauli)
            codes[i] = ET_H
        elif t == 'S':
            Ps.append(bl[0])               
            Qs.append(bl[0])
            Ps.append(identity_pauli)      
            Qs.append(identity_pauli)
            codes[i] = ET_S
        elif t == 'C':
            Ps.append(bl[0]);               
            Qs.append(bl[1])
            if bl[0].commutes(bl[1]):
                Ps.append(bl[0] * bl[1])   
                Qs.append(identity_pauli)
                codes[i] = ET_C1
            else:
                codes[i] = ET_C
        else:  # 'A'
            Ps.append(bl[1]);               
            Qs.append(bl[0])
            if not bl[0].commutes(bl[1]):
                Ps.append(bl[1] * bl[0]);   
                Qs.append(identity_pauli)
                codes[i] = ET_A1
            else:
                codes[i] = ET_A

    #
    # 4) Pre‐allocate the output sensitivity arrays:
    #

    cdef np.ndarray[double, ndim=2] sensitivities_by_bitstring = np.empty((n_b, n_e), dtype=np.double)
    cdef double[:,::1] sensitivities_by_bitstring_view = sensitivities_by_bitstring
    
    #
    # 5) Inner loops: for each desired bitstring, call bulk_phi once,
    #    then walk through codes[] to assemble sensitivities.
    #
    cdef np.ndarray[np.complex128_t, ndim=1] phis
    cdef np.complex128_t[::1] phis_view
    cdef int run_idx, code
    cdef complex v0, v1
    for i in range(n_b):
        phis = fast_bulk_phi(sim, desired_bitstrings[i], Ps, Qs)
        phis_view = phis 
        run_idx = 0
        for j in range(n_e):
            code = codes[j]
            if code == ET_H:
                v0 = phis_view[run_idx]
                sensitivities_by_bitstring_view[i, j] = 2 * v0.imag
                run_idx += 1
            elif code == ET_S:
                v0 = phis_view[run_idx]
                v1 = phis_view[run_idx + 1]
                sensitivities_by_bitstring_view[i, j] = (v0 - v1).real
                run_idx += 2
            elif code == ET_C:
                v0 = phis_view[run_idx]
                sensitivities_by_bitstring_view[i, j] = 2 * v0.real
                run_idx += 1
            elif code == ET_C1:
                v0 = phis_view[run_idx]
                v1 = phis_view[run_idx + 1]
                sensitivities_by_bitstring_view[i, j] = 2 * v0.real - 2 * v1.real
                run_idx += 2
            elif code == ET_A:
                v0 = phis_view[run_idx]
                sensitivities_by_bitstring_view[i, j] = 2 * v0.imag
                run_idx += 1
            else:  # ET_A1
                v0 = phis_view[run_idx]
                v1 = phis_view[run_idx + 1]
                sensitivities_by_bitstring_view[i, j] = 2 * (v0 + v1).imag
                run_idx += 2

    PyMem_Free(codes)
    return sensitivities_by_bitstring

cdef inline double _real_if_close(complex val):
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

cdef inline tuple _com(object P1, object P2):
    # P1 and P2 will be stim.PauliString
    # P1 and P2 either commute or anticommute.
    if P1.commutes(P2):
        return None
    else:
        P3 = P1*P2
        return (P3.sign*2, P3 / P3.sign)

cdef inline tuple _pauli_product(object P1, object P2):
    # P1 and P2 will be stim.PauliString 
    P3 = P1*P2
    return (P3.sign, P3 / P3.sign)

@cython.wraparound(False)   # Deactivate negative indexing.
@cython.boundscheck(False)
cpdef np.ndarray[double, ndim=2] fast_bulk_alpha_pauli(object errorgens_iter, object tableau, list paulis):
    """
    First-order error generator sensitivity function for pauli expectations.
    
    Parameters
    ----------
    errorgens : iterable of LocalStimElementaryErrorgenLabel
        Error generator labels for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state.
        
    paulis : list of stim.PauliString
        List of paulis to calculate the sensitivity for.
    
    Returns
    -------
    np.ndarray[np.double]
        A two-dimensional numpy array of sensitivities such that rows correspond
        to entries in `paulis` and columns correspond to error generators.
    """
    cdef:
        list errorgens = list(errorgens_iter)
        object sim
        int n_paulis, n_errorgens
        np.ndarray[double, ndim=2] sensitivities_by_pauli  # our output array
        double[:, ::1] sensitivities_by_pauli_view  # typed memory view for element access
        int i, j
        str errgen_type
        tuple basis_element_labels
        object errorgen
        object A, B, ABP
        bint com_AP, com_BP
        complex expectation, sign, tmp
        object res  # temporary for the result of com()
    
    # Build the simulator and set its inverse tableau.
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
        
    n_paulis    = len(paulis)
    n_errorgens = len(errorgens)
    
    # Pre-allocate the output array and obtain a memoryview for fast writes.
    sensitivities_by_pauli = np.empty((n_paulis, n_errorgens), dtype=np.double)
    sensitivities_by_pauli_view = sensitivities_by_pauli  # typed memory view
    
    # Loop over each pauli observable and error generator.
    for i in range(n_paulis):
        for j in range(n_errorgens):
            errorgen = errorgens[j]
            errgen_type = errorgen.errorgen_type
            basis_element_labels = errorgen.basis_element_labels
            
            if errgen_type == 'H':
                # For H-type error generators, call the helper function com.
                # com(...) is assumed to return either None or a tuple.
                res = _com(paulis[i], basis_element_labels[0])
                if res is not None:
                    # Multiply the first element of res by -1j to generate a sign.
                    sign = -1j * res[0]
                    expectation = sim.peek_observable_expectation(res[1])
                    sensitivities_by_pauli_view[i, j] = _real_if_close(sign * expectation)
                else:
                    sensitivities_by_pauli_view[i, j] = 0.0

            elif errgen_type == 'S':
                if paulis[i].commutes(basis_element_labels[0]):
                    sensitivities_by_pauli_view[i, j] = 0.0
                else:
                    expectation = sim.peek_observable_expectation(paulis[i])
                    sensitivities_by_pauli_view[i, j] = _real_if_close(-2 * expectation)

            elif errgen_type == 'C':
                A = basis_element_labels[0]
                B = basis_element_labels[1]
                com_AP = A.commutes(paulis[i])
                if A.commutes(B):
                    if com_AP:
                        sensitivities_by_pauli_view[i, j] = 0.0
                    else:
                        com_BP = B.commutes(paulis[i])
                        if com_BP:
                            sensitivities_by_pauli_view[i, j] = 0.0
                        else:
                            ABP = _pauli_product(A * B, paulis[i])
                            expectation = ABP[0] * sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli_view[i, j] = _real_if_close(-4 * expectation)
                else:  # non-commuting A and B.
                    if com_AP:
                        com_BP = B.commutes(paulis[i])
                        if com_BP:
                            sensitivities_by_pauli_view[i, j] = 0.0
                        else:
                            ABP = _pauli_product(A * B, paulis[i])
                            expectation = ABP[0] * sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli_view[i, j] = _real_if_close(-2 * expectation)
                    else:
                        com_BP = B.commutes(paulis[i])
                        if com_BP:
                            ABP = _pauli_product(A * B, paulis[i])
                            expectation = ABP[0] * sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli_view[i, j] = _real_if_close(2 * expectation)
                        else:
                            sensitivities_by_pauli_view[i, j] = 0.0

            else:  # for error generator type 'A'
                A = basis_element_labels[0]
                B = basis_element_labels[1]
                com_AP = A.commutes(paulis[i])
                if A.commutes(B):
                    com_BP = B.commutes(paulis[i])
                    if com_AP:
                        if com_BP:
                            sensitivities_by_pauli_view[i, j] = 0.0
                        else:
                            ABP = _pauli_product(A * B, paulis[i])
                            expectation = ABP[0] * sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli_view[i, j] = _real_if_close(1j * 2 * expectation)
                    else:
                        if com_BP:
                            ABP = _pauli_product(A * B, paulis[i])
                            expectation = ABP[0] * sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli_view[i, j] = _real_if_close(-1j * 2 * expectation)
                        else:
                            sensitivities_by_pauli_view[i, j] = 0.0
                else:
                    if com_AP:
                        sensitivities_by_pauli_view[i, j] = 0.0
                    else:
                        com_BP = B.commutes(paulis[i])
                        if com_BP:
                            sensitivities_by_pauli_view[i, j] = 0.0
                        else:
                            ABP = _pauli_product(A * B, paulis[i])
                            expectation = ABP[0] * sim.peek_observable_expectation(ABP[1])
                            sensitivities_by_pauli_view[i, j] = _real_if_close(1j * 4 * expectation)
                            
    return sensitivities_by_pauli
    
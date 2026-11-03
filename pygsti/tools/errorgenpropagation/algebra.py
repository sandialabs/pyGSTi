"""
Elementary analytical error generator and Pauli algebra routines.

This module contains the closed-form commutator, composition, anticommutator and Pauli
conjugation rules for elementary error generators (Hamiltonian, Stochastic, Correlation and
Affine types), together with the Pauli product and commutator helpers they rely on. These
routines were extracted from `pygsti.tools.errgenproptools`.
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
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL, LocalElementaryErrorgenLabel as _LEEL
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE, bel_str as _bel_str, bel_product_str as _bel_product_str
from pygsti.tools.errorgenpropagation.errgentypes import _ErrorgenTerms, _SignedPauli, _PairHandler
from typing import Optional, Union, Callable

__all__ = [
    'errgen_coeff_label_to_stim_pauli_strs',
    'error_generator_commutator',
    'error_generator_composition',
    'error_generator_anticommutator',
    'pauli_conjugation_composition',
    'com',
    'acom',
    'pauli_product',
    'errorgen_pauli_action',
    'iterative_error_generator_composition',
]

def errgen_coeff_label_to_stim_pauli_strs(err_gen_coeff_label: Union[_GEEL, _LEEL],
                                          num_qubits: int) -> tuple[stim.PauliString, ...]:
    """
    Converts an input `GlobalElementaryErrorgenLabel` to a tuple of stim.PauliString
    objects, padded with an appropriate number of identities.

    Parameters
    ----------
    err_gen_coeff_label : `GlobalElementaryErrorgenLabel` or `LocalElementaryErrorgenLabel`
        The error generator coefficient label to construct the tuple of pauli
        strings for.

    num_qubits : int
        Number of total qubits to use for the Pauli strings. Used to determine
        the number of identities added when padding.

    Returns
    -------
    tuple of stim.PauliString
        A tuple of either length 1 (for H and S) or length 2 (for C and A)
        whose entries are stim.PauliString representations of the indices for the
        input error generator label, padded with an appropriate number of identities
        given the support of the error generator label.

    """

    if isinstance(err_gen_coeff_label, _LEEL):
        return tuple([stim.PauliString(bel) for bel in err_gen_coeff_label.basis_element_labels])

    elif isinstance(err_gen_coeff_label, _GEEL):
        # the coefficient label is a tuple with 3 elements. 
        # The first element is the error generator type.
        # the second element is a tuple of paulis either of length 1 or 2 depending on the error gen type.
        # the third element is a tuple of subsystem labels.
        errorgen_typ = err_gen_coeff_label.errorgen_type
        pauli_lbls = err_gen_coeff_label.basis_element_labels
        sslbls = err_gen_coeff_label.support

        # double check that the number of qubits specified is greater than or equal to the length of the
        # basis element labels.
        # assert len(pauli_lbls) >= num_qubits, 'Specified `num_qubits` is less than the length of the basis element labels.'

        if errorgen_typ == 'H' or errorgen_typ == 'S':
            pauli_string = num_qubits*['I']
            pauli_lbl = pauli_lbls[0]
            for i, sslbl in enumerate(sslbls):
                pauli_string[sslbl] = pauli_lbl[i]
            pauli_string = stim.PauliString(''.join(pauli_string))
            return (pauli_string,)
        elif errorgen_typ == 'C' or errorgen_typ == 'A':
            pauli_strings = []
            for pauli_lbl in pauli_lbls: # iterate through both pauli labels
                pauli_string = num_qubits*['I']
                for i, sslbl in enumerate(sslbls):
                    pauli_string[sslbl] = pauli_lbl[i]
                pauli_strings.append(stim.PauliString(''.join(pauli_string)))
            return tuple(pauli_strings)
        else:
            raise ValueError(f'Unsupported error generator type {errorgen_typ}')
    else:
        raise ValueError('Only `GlobalElementaryErrorgenLabel and LocalElementaryErrorgenLabel is currently supported.')

# ------- Error Generator Math -------------# 

"""
Extended elementary error generator conventions
-----------------------------------------------
The analytic formulas for the commutator and the composition of two elementary error
generators (implemented by `error_generator_commutator` and `error_generator_composition`)
are written in terms of an *extended* family of elementary error generators (EEGs) whose
indices are products, commutators or anticommutators of the input Paulis. Such an index
carries a phase, may be the identity and may coincide with the other index of the same
term. The code transcribes the formulas term by term through the four emitters `_H`,
`_S`, `_C` and `_A`, which reduce each extended term to a canonical
`LocalStimErrorgenLabel` (or drop it if it is zero) using the identities below. Keep
these in mind when comparing the code with the formulas, e.g. those in the Supplemental
Note "Formulae for efficiently manipulating elementary error generators" of *Approximate
simulation of Clifford circuits with small Markovian errors* (whose notation is used
throughout).

Signed Paulis
    A product of Paulis is represented by the pair `(phase, P)` returned by
    `pauli_product`, `com` and `acom`: `P` is an unsigned `stim.PauliString` and `phase`
    is +1, -1, +i or -i. For `com` and `acom` the pair represents `P1 P2 -/+ P2 P1`, so
    `phase` is +-2 or +-2i (and they return `None` when the (anti)commutator vanishes;
    every emitter treats a `None` index as a zero term). An index may also be given as
    the triple `(phase, P, s)` with `s = bel_str(P)` already rendered; this is used for
    the input Paulis, whose strings the input labels already hold, so that they are not
    rendered again. An unsigned input Pauli is thus written `(1, P, s)`. Phases are folded
    into the rate of the emitted term as

        H_{wP}       = w   H_P
        S_{wP}       = w w* S_P = S_P  (S_L = L . L^dag - ½{L^dag L, .} contains L twice,
                                        once conjugated, so a unit phase contributes
                                        |w|^2 = 1; the paper writes w^2, its w being +-1)
        C_{wP,vQ}    = w v C_{P,Q} (Assumes phases are real-valued)
        A_{wP,vQ}    = w v A_{P,Q} (Assumes phases are real-valued)

Identity indices
    H_I = S_I = 0,   C_{I,Q} = C_{P,I} = 0,   A_{I,Q} = H_Q,   A_{P,I} = -H_P.

Repeated indices
    C_{P,P} = 2 S_P,   A_{P,P} = 0.

Canonical ordering
    The two basis element labels of a 'C' or 'A' label are stored sorted, lexicographically
    on their 'I'-padded strings with 'I' < 'X' < 'Y' < 'Z' (`bel_str`, `bel_less_than`).
    C is symmetric, so swapping is free; A is antisymmetric, so a swap negates the rate:
    C_{Q,P} = C_{P,Q},  A_{Q,P} = -A_{P,Q}.

Operand order and weights
    `error_generator_commutator(e1, e2)` computes [e1, e2],
    `error_generator_anticommutator(e1, e2)` computes {e1, e2} = e1[e2[.]] + e2[e1[.]] and
    `error_generator_composition(e1, e2)` computes e1[e2[.]] (e1 applied after e2). All
    return a list of `(LocalStimErrorgenLabel, rate)` pairs, each rate already multiplied
    by the `weight` argument; the same label may appear more than once and the caller is
    expected to accumulate.

Anticommutators and Pauli conjugation
    Every term of a composition e1[e2[.]] recurs in the reversed composition e2[e1[.]] with
    the same label and a coefficient of the same magnitude, so it lands in exactly one of
    the commutator (opposite signs) and the anticommutator (equal signs):
    e1 o e2 = ½([e1, e2] + {e1, e2}) term by term. The anticommutation relations are thus
    the even-sign complements of the commutation relations.

    The stochastic generator is conjugation minus identity, S_Q = 𝒬 - 1 with 𝒬[rho] = Q rho Q,
    and 𝒬 applied to any elementary error generator X is again a (trace-annihilating)
    combination of elementary error generators, `pauli_conjugation_composition(Q, X)`:

        𝒬 H_P     = -A_{QP,Q} if [Q,P] = 0,  -i C_{QP,Q} if {Q,P} = 0
        𝒬 S_P     = S_R - S_Q,  R = QP / phase(QP)
        𝒬 C_{P,R}, 𝒬 A_{P,R} : the S_Q o C / S_Q o A composition tables without their
                               leading -C_{P,R} / -A_{P,R} term

    (The S row: Q (P rho P - rho) Q = R rho R - Q rho Q = (R rho R - rho) - (Q rho Q - rho);
    the phase of QP drops out because rho is conjugated by (QP) and (QP)^dag. Conjugation
    composed with *conjugation*, 𝒬 o 𝒫 = ℛ = S_R + 1, is unital and is deliberately not
    offered: it is not a combination of error generators.)

    Writing an elementary error generator as a sum of monomials rho -> U rho V, conjugating
    a monomial by 𝒬 on both sides multiplies it by (-1)^{ω(Q, UV)}, where ω(Q, T) is 0 if the
    Paulis commute and 1 if they anticommute. All monomials of one generator share the
    product UV up to phase - its *total Pauli* T_X: P for H_P, I for S_P, PR for C_{P,R} and
    A_{P,R}. Hence 𝒬 X 𝒬 = (-1)^{ω(Q,T_X)} X and, since X o 𝒬 = 𝒬 o (𝒬 X 𝒬),

        S_Q o X  = 𝒬X - X,      X o S_Q = (-1)^{ω(Q,T_X)} 𝒬X - X,
        {S_Q, X} = (1 + (-1)^{ω(Q,T_X)}) 𝒬X - 2X .

    The -X / -2X terms (the -1 inside S_Q acting on the partner) are called the *bleed*
    terms below; when Q anticommutes with T_X the anticommutator is pure bleed, {S_Q, X} =
    -2X. `_commuting_product` (used by the Taylor expansion) exploits this to recover all
    bleed terms of a product in closed form instead of emitting them label by label.
"""

# ---------------------------------------------------------------------------------------
# Term emitters. See "Extended elementary error generator conventions" in the module
# docstring for the identities they apply.
#
# Each emitter appends one term of a formula, coeff * G_{index(es)}, to `terms` as a
# (LocalStimErrorgenLabel, rate) pair - or appends nothing when the term is zero. An index
# is a *signed Pauli* w P, passed as
#     (w, P)        as returned by the public `pauli_product`, `com` and `acom`,
#     (w, P, s)     the same with the 'I'-padded string s = bel_str(P): the input Paulis carry
#                   the strings their labels cache (`_index`), and the private `_prod`, `_com`,
#                   `_acom` derive the product's string from its operands' strings
#                   (`bel_product_str`) instead of rendering the stim product, or
#     None          a vanishing (anti)commutator, as returned by `com`/`acom`.
# Strings not supplied are rendered once here; they serve the identity check (a compare
# against the all-'I' string `identity`) and the canonical ordering, and are handed to the
# label constructor via `pauli_str_reps`.
# ---------------------------------------------------------------------------------------

def _H(terms: _ErrorgenTerms, pauli: Optional[_SignedPauli], coeff: complex, identity: str) -> None:
    """
    Append coeff * H_{wP} = (coeff w) H_P to `terms`; nothing if P = I (H_I = 0).

    Parameters
    ----------
    terms : list
        Accumulator of (LocalStimErrorgenLabel, rate) pairs, appended to in place.

    pauli : tuple or None
        The signed Pauli index wP as `(w, P)` or `(w, P, s)`; None is a zero index.

    coeff : complex
        Prefactor of the term in the formula, including the overall weight.

    identity : str
        The all-identity Pauli string `'I'*n` for the number of qubits n.
    """
    if pauli is None:
        return
    w, P = pauli[0], pauli[1]
    sP = pauli[2] if len(pauli) == 3 else _bel_str(P)
    if sP == identity:
        return
    terms.append((_LSE('H', (P,), pauli_str_reps=(sP,)), w * coeff))


def _S(terms: _ErrorgenTerms, pauli: Optional[_SignedPauli], coeff: complex, identity: str) -> None:
    """
    Append coeff * S_{wP} = coeff S_P to `terms`; nothing if P = I (S_I = 0). The unit
    phase w contributes w w* = 1 (it enters S_L = L . L^dag - ½{L^dag L, .} twice, once
    conjugated).

    Parameters
    ----------
    terms : list
        Accumulator of (LocalStimErrorgenLabel, rate) pairs, appended to in place.

    pauli : tuple or None
        The signed Pauli index wP as `(w, P)` or `(w, P, s)`; None is a zero index.

    coeff : complex
        Prefactor of the term in the formula, including the overall weight.

    identity : str
        The all-identity Pauli string `'I'*n` for the number of qubits n.
    """
    if pauli is None:
        return
    P = pauli[1]
    sP = pauli[2] if len(pauli) == 3 else _bel_str(P)
    if sP == identity:
        return
    terms.append((_LSE('S', (P,), pauli_str_reps=(sP,)), coeff))


def _C(terms: _ErrorgenTerms, pauli_1: Optional[_SignedPauli], pauli_2: Optional[_SignedPauli], coeff: complex,
       identity: str) -> None:
    """
    Append coeff * C_{wP,vQ} = (coeff w v) C_{P,Q} to `terms`, reduced as follows:
    C_{P,P} = 2 S_P; nothing if P = I or Q = I (C_{I,Q} = C_{P,I} = 0); the two basis
    element labels are stored in canonical order (C is symmetric, so this is free).

    Parameters
    ----------
    terms : list
        Accumulator of (LocalStimErrorgenLabel, rate) pairs, appended to in place.

    pauli_1, pauli_2 : tuple or None
        The signed Pauli indices wP and vQ, each as `(w, P)` or `(w, P, s)`; None is a
        zero index.

    coeff : complex
        Prefactor of the term in the formula, including the overall weight.

    identity : str
        The all-identity Pauli string `'I'*n` for the number of qubits n.
    """
    if pauli_1 is None or pauli_2 is None:
        return
    w, P = pauli_1[0], pauli_1[1]
    v, Q = pauli_2[0], pauli_2[1]
    sP = pauli_1[2] if len(pauli_1) == 3 else _bel_str(P)
    sQ = pauli_2[2] if len(pauli_2) == 3 else _bel_str(Q)
    # The identity string sorts first, so of an ordered pair only the smaller can be identity.
    if sP == sQ:
        if sP == identity:
            return
        terms.append((_LSE('S', (P,), pauli_str_reps=(sP,)), 2 * w * v * coeff))
    elif sP < sQ:
        if sP == identity:
            return
        terms.append((_LSE('C', (P, Q), pauli_str_reps=(sP, sQ)), w * v * coeff))
    else:
        if sQ == identity:
            return
        terms.append((_LSE('C', (Q, P), pauli_str_reps=(sQ, sP)), w * v * coeff))


def _A(terms: _ErrorgenTerms, pauli_1: Optional[_SignedPauli], pauli_2: Optional[_SignedPauli], coeff: complex,
       identity: str) -> None:
    """
    Append coeff * A_{wP,vQ} = (coeff w v) A_{P,Q} to `terms`, reduced as follows:
    nothing if P = Q (A_{P,P} = 0); A_{I,Q} = H_Q and A_{P,I} = -H_P; the two basis
    element labels are stored in canonical order, which negates the rate when they have1860
    to be swapped (A is antisymmetric).

    Parameters
    ----------
    terms : list
        Accumulator of (LocalStimErrorgenLabel, rate) pairs, appended to in place.

    pauli_1, pauli_2 : tuple or None
        The signed Pauli indices wP and vQ, each as `(w, P)` or `(w, P, s)`; None is a
        zero index.

    coeff : complex
        Prefactor of the term in the formula, including the overall weight.

    identity : str
        The all-identity Pauli string `'I'*n` for the number of qubits n.
    """
    if pauli_1 is None or pauli_2 is None:
        return
    w, P = pauli_1[0], pauli_1[1]
    v, Q = pauli_2[0], pauli_2[1]
    sP = pauli_1[2] if len(pauli_1) == 3 else _bel_str(P)
    sQ = pauli_2[2] if len(pauli_2) == 3 else _bel_str(Q)
    # The identity string sorts first, so of an ordered pair only the smaller can be identity.
    if sP == sQ:
        return
    elif sP < sQ:
        if sP == identity:
            terms.append((_LSE('H', (Q,), pauli_str_reps=(sQ,)), w * v * coeff))
        else:
            terms.append((_LSE('A', (P, Q), pauli_str_reps=(sP, sQ)), w * v * coeff))
    else:
        if sQ == identity:
            terms.append((_LSE('H', (P,), pauli_str_reps=(sP,)), -w * v * coeff))
        else:
            terms.append((_LSE('A', (Q, P), pauli_str_reps=(sQ, sP)), -w * v * coeff))


# Signed-Pauli arithmetic for the handlers below. Arguments and results are signed Paulis
# `(w, P)` / `(w, P, s)` as accepted by the emitters (the string, if any, is not propagated),
# or None, which propagates: a vanishing (anti)commutator anywhere inside a nested index
# makes the whole index, and hence the term, vanish.

def _prod(pauli_1: Optional[_SignedPauli], pauli_2: Optional[_SignedPauli]) -> Optional[_SignedPauli]:
    """
    Product of two signed Paulis, `(w v phase, PQ)` with `PQ` unsigned; None if either is None.
    When both operands carry their string, the product's string is derived from them
    (`bel_product_str`) and returned as a third element, so the emitters need not render `PQ`.
    """
    if pauli_1 is None or pauli_2 is None:
        return None
    PQ = pauli_1[1] * pauli_2[1]
    phase = PQ.sign
    PQ.sign = 1
    if len(pauli_1) == 3 and len(pauli_2) == 3:
        return (pauli_1[0] * pauli_2[0] * phase, PQ, _bel_product_str(pauli_1[2], pauli_2[2]))
    return (pauli_1[0] * pauli_2[0] * phase, PQ)


def _com(pauli_1: Optional[_SignedPauli], pauli_2: Optional[_SignedPauli]) -> Optional[_SignedPauli]:
    """
    Commutator [pauli_1, pauli_2] of two signed Paulis as a signed Pauli (phase +-2, +-2i);
    None if either is None or they commute.
    """
    if pauli_1 is None or pauli_2 is None or pauli_1[1].commutes(pauli_2[1]):
        return None
    PQ = pauli_1[1] * pauli_2[1]
    phase = 2 * PQ.sign
    PQ.sign = 1
    if len(pauli_1) == 3 and len(pauli_2) == 3:
        return (pauli_1[0] * pauli_2[0] * phase, PQ, _bel_product_str(pauli_1[2], pauli_2[2]))
    return (pauli_1[0] * pauli_2[0] * phase, PQ)


def _acom(pauli_1: Optional[_SignedPauli], pauli_2: Optional[_SignedPauli]) -> Optional[_SignedPauli]:
    """
    Anticommutator {pauli_1, pauli_2} of two signed Paulis as a signed Pauli (phase +-2, +-2i);
    None if either is None or they anticommute.
    """
    if pauli_1 is None or pauli_2 is None or not pauli_1[1].commutes(pauli_2[1]):
        return None
    PQ = pauli_1[1] * pauli_2[1]
    phase = 2 * PQ.sign
    PQ.sign = 1
    if len(pauli_1) == 3 and len(pauli_2) == 3:
        return (pauli_1[0] * pauli_2[0] * phase, PQ, _bel_product_str(pauli_1[2], pauli_2[2]))
    return (pauli_1[0] * pauli_2[0] * phase, PQ)


def _index(errorgen: _LSE, k: int) -> tuple[int, stim.PauliString, str]:
    """
    The k-th basis element label of `errorgen` as the signed Pauli `(1, P, s)` with its
    cached string, ready for the emitters and the signed-Pauli helpers.
    """
    return (1, errorgen.basis_element_labels[k], errorgen._hashable_basis_element_labels[k])

def error_generator_commutator(errorgen_1: _LSE, errorgen_2: _LSE, weight: complex = 1.0,
                               identity: Optional[str] = None) -> _ErrorgenTerms:
    """
    Returns the commutator of two error generators. I.e. [errorgen_1, errorgen_2].

    The result is assembled from the analytic commutation relations of the elementary
    error generators; see "Extended elementary error generator conventions" in the module
    docstring for how the terms of those formulas map onto the returned labels.

    Parameters
    ----------
    errorgen_1 : `LocalStimErrorgenLabel`
        First error generator.

    errorgen_2 : `LocalStimErrorgenLabel`
        Second error generator

    weight : float or complex, optional (default 1.0)
        An optional weighting value to apply to the value of the commutator.

    identity : str, optional (default None)
        The all-identity Pauli string `'I'*n` for the number of qubits n, used to detect
        identity indices. Built from `errorgen_1` if not given; passing it avoids
        rebuilding it when calling this function many times.

    Returns
    -------
    list of tuples. The first element of each tuple is a `LocalStimErrorgenLabel`
    corresponding to a component of the commutator of the two input error generators.
    The second element is the rate of that term, additionally weighted by the specified
    value of `weight`. The same label may appear in more than one tuple.

    Error generators supported on disjoint sets of qubits commute exactly, but this function
    does not special-case them: the commutation relations are evaluated and the returned
    terms cancel only once aggregated. Callers looping over many pairs (at large qubit counts
    almost every pair is of this kind) should skip such pairs beforehand by testing the labels'
    `support_mask`s for a common bit, as `_accumulate_layer_pairwise_commutators` does.
    """
    if identity is None:
        identity = 'I' * len(errorgen_1._hashable_basis_element_labels[0])
    return _COMMUTATOR_HANDLERS[4 * errorgen_1.type_idx + errorgen_2.type_idx](errorgen_1, errorgen_2, weight, identity)


def error_generator_composition(errorgen_1: _LSE, errorgen_2: _LSE, weight: complex = 1.0,
                                identity: Optional[str] = None) -> _ErrorgenTerms:
    r"""
    Returns the composition of two error generators. I.e. errorgen_1[errorgen_2[\cdot]].

    The result is assembled from the analytic composition rules of the elementary error
    generators; see "Extended elementary error generator conventions" in the module
    docstring for how the terms of those formulas map onto the returned labels.

    Parameters
    ----------
    errorgen_1 : `LocalStimErrorgenLabel`
        First error generator (applied second).

    errorgen_2 : `LocalStimErrorgenLabel`
        Second error generator (applied first).

    weight : float or complex, optional (default 1.0)
        An optional weighting value to apply to the value of the composition.

    identity : str, optional (default None)
        The all-identity Pauli string `'I'*n` for the number of qubits n, used to detect
        identity indices. Built from `errorgen_1` if not given; passing it avoids
        rebuilding it when calling this function many times.

    Returns
    -------
    list of tuples. The first element of each tuple is a `LocalStimErrorgenLabel`
    corresponding to a component of the composition of the two input error generators.
    The second element is the rate of that term, additionally weighted by the specified
    value of `weight`. The same label may appear in more than one tuple.
    """
    if identity is None:
        identity = 'I' * len(errorgen_1._hashable_basis_element_labels[0])
    return _COMPOSITION_HANDLERS[4 * errorgen_1.type_idx + errorgen_2.type_idx](errorgen_1, errorgen_2, weight, identity)


def error_generator_anticommutator(errorgen_1: _LSE, errorgen_2: _LSE, weight: complex = 1.0,
                                   identity: Optional[str] = None) -> _ErrorgenTerms:
    r"""
    Returns the anticommutator of two error generators. I.e. {errorgen_1, errorgen_2} =
    errorgen_1[errorgen_2[\cdot]] + errorgen_2[errorgen_1[\cdot]].

    The result is assembled from the analytic anticommutation relations of the elementary
    error generators (the even-sign complements of the commutation relations; see
    "Anticommutators and Pauli conjugation" in the module docstring). 
    Together with `error_generator_commutator` it
    decomposes the composition: e1 o e2 = ½([e1, e2] + {e1, e2}), term by term. The
    anticommutator emits fewer terms than either composition, because the terms of a
    composition that change sign under reversal (those making up the commutator) are
    absent from it.

    The anticommutator is symmetric in its arguments, and this is used directly: the ten
    unordered type pairs carry the formulas, and the handlers of the six reversed pairs
    delegate to them with the arguments swapped, with no sign.

    Parameters
    ----------
    errorgen_1 : `LocalStimErrorgenLabel`
        First error generator.

    errorgen_2 : `LocalStimErrorgenLabel`
        Second error generator.

    weight : float or complex, optional (default 1.0)
        An optional weighting value to apply to the value of the anticommutator.

    identity : str, optional (default None)
        The all-identity Pauli string `'I'*n` for the number of qubits n, used to detect
        identity indices. Built from `errorgen_1` if not given; passing it avoids
        rebuilding it when calling this function many times.

    Returns
    -------
    list of tuples. The first element of each tuple is a `LocalStimErrorgenLabel`
    corresponding to a component of the anticommutator of the two input error generators.
    The second element is the rate of that term, additionally weighted by the specified
    value of `weight`. The same label may appear in more than one tuple.

    Unlike the commutator, the anticommutator of error generators with disjoint supports
    does not vanish: it is twice their composition.
    """
    if identity is None:
        identity = 'I' * len(errorgen_1._hashable_basis_element_labels[0])
    return _ANTICOMMUTATOR_HANDLERS[4 * errorgen_1.type_idx + errorgen_2.type_idx](errorgen_1, errorgen_2, weight, identity)


def pauli_conjugation_composition(pauli: stim.PauliString, errorgen: _LSE, weight: complex = 1.0,
                                  identity: Optional[str] = None, pauli_str: Optional[str] = None) -> _ErrorgenTerms:
    r"""
    Returns the composition of the Pauli conjugation superoperator 𝒬[\rho] = Q \rho Q with
    an elementary error generator, i.e. 𝒬[errorgen[\cdot]].

    The conjugation by a Pauli is the stochastic generator plus the identity,
    𝒬 = S_Q + 1, so this is `error_generator_composition(S_Q, errorgen)` with the
    `-errorgen` term removed; see "Anticommutators and Pauli conjugation" in the module
    docstring for the four relations (one per type of `errorgen`). The result is a combination of
    elementary error generators for every type of `errorgen`, including S
    (𝒬 S_P = S_R - S_Q with R the phase-stripped product QP): conjugating a
    trace-annihilating map leaves it trace annihilating.

    Parameters
    ----------
    pauli : stim.PauliString
        The conjugating Pauli Q. Its sign is irrelevant (Q and -Q conjugate identically)
        and is ignored. Must not be the identity.

    errorgen : `LocalStimErrorgenLabel`
        The error generator being conjugated.

    weight : float or complex, optional (default 1.0)
        An optional weighting value to apply to the result.

    identity : str, optional (default None)
        The all-identity Pauli string `'I'*n` for the number of qubits n. Built from
        `errorgen` if not given.

    pauli_str : str, optional (default None)
        The 'I'-padded string of `pauli` (`bel_str(pauli)`), if already available - e.g. the
        cached string of an S-type label's Pauli. Rendered from `pauli` if not given; passing
        it avoids rendering it again when calling this function many times with the same
        Pauli.

    Returns
    -------
    list of tuples. The first element of each tuple is a `LocalStimErrorgenLabel`
    corresponding to a component of 𝒬[errorgen[\cdot]], the second its rate, weighted by
    `weight`. The same label may appear in more than one tuple.
    """
    if identity is None:
        identity = 'I' * len(errorgen._hashable_basis_element_labels[0])
    if pauli.sign != 1:
        pauli = pauli.copy()
        pauli.sign = 1
    Q = (1, pauli, pauli_str if pauli_str is not None else _bel_str(pauli))
    return _CONJUGATION_HANDLERS[errorgen.type_idx](Q, errorgen, weight, identity)



# ---------------------------------------------------------------------------------------
# Commutator handlers, one per ordered type pair, called as handler(errorgen_1, errorgen_2,
# weight, identity) with `identity` the 'I'*n string. Each forward handler is headed by the
# commutation relation it transcribes (notation of the paper's Supplemental Note; where the
# paper and the previously validated implementation (see v0.10 for previous validated implementation)
# differ, the latter is kept). The six reversed pairs use [X, Y] = -[Y, X].
#
# Pairs of terms whose indices are the same two products in opposite order, such as
# A_{AP,QB} + A_{BQ,PA}, are emitted as a single term. Writing s_XY = +1 if the Paulis X, Y
# commute and -1 if they anticommute, reversing a product only changes its sign
# (QB = s_BQ BQ, PA = s_AP AP), and bilinearity plus the (anti)symmetry of C/A give
#     A_{AP,QB} + A_{BQ,PA} = (s_BQ - s_AP) A_{AP,BQ},   C_{AP,QB} - C_{PA,BQ} = (s_BQ - s_AP) C_{AP,BQ},
# i.e. zero unless exactly one of the pairs (A,P), (B,Q) anticommutes, and then twice a
# single term. Without this, every such pair costs two label constructions that cancel
# only after aggregation (at 100 qubits, 99% of the terms these handlers emitted).
#
# The same four cross-pair booleans s_AP, s_AQ, s_BP, s_BQ decide every other commutator in
# these handlers, because commutation signs are multiplicative: a Pauli anticommutes with a
# product XY iff it anticommutes with exactly one of X, Y. With X = {A,B} or [A,B] (whichever
# is nonzero is proportional to AB) and Y = {P,Q} or [P,Q] (proportional to PQ),
#     [P, X] != 0 iff s_AP != s_BP,   [Q, X] != 0 iff s_AQ != s_BQ,
#     [Y, A] != 0 iff s_AP != s_AQ,   [Y, B] != 0 iff s_BP != s_BQ,
#     [X, Y] != 0 iff an odd number of the four cross pairs anticommute,
# and a nonzero commutator of two Paulis is twice their product. So after the four
# `commutes()` tests those commutators are formed with `_prod` (no further commutation test)
# under the matching condition, with the factor of 2 folded into the coefficient; only the
# own-pair (anti)commutators X and Y themselves still go through `_com`/`_acom`.
# ---------------------------------------------------------------------------------------

def _commutator_HH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [H_P, H_Q] = -i H_{[P,Q]}
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_2, 0)
    terms = []
    _H(terms, _com(P, Q), -1j*w, identity)
    return terms


def _commutator_HS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [H_P, S_Q] = i C_{Q,[Q,P]}
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_2, 0)
    terms = []
    _C(terms, Q, _com(Q, P), 1j*w, identity)
    return terms


def _commutator_SH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [S_P, H_Q] = -[H_Q, S_P]
    return _commutator_HS(errorgen_2, errorgen_1, -w, identity)


def _commutator_HC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [H_P, C_{A,B}] = i (C_{[A,P],B} + C_{[B,P],A})
    P = _index(errorgen_1, 0)
    A = _index(errorgen_2, 0)
    B = _index(errorgen_2, 1)
    terms = []
    _C(terms, _com(A, P), B, 1j*w, identity)
    _C(terms, _com(B, P), A, 1j*w, identity)
    return terms


def _commutator_CH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [C_{A,B}, H_P] = -[H_P, C_{A,B}]
    return _commutator_HC(errorgen_2, errorgen_1, -w, identity)


def _commutator_HA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [H_P, A_{A,B}] = -i (A_{[P,A],B} + A_{A,[P,B]})
    P = _index(errorgen_1, 0)
    A = _index(errorgen_2, 0)
    B = _index(errorgen_2, 1)
    terms = []
    _A(terms, _com(P, A), B, -1j*w, identity)
    _A(terms, A, _com(P, B), -1j*w, identity)
    return terms


def _commutator_AH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [A_{A,B}, H_P] = -[H_P, A_{A,B}]
    return _commutator_HA(errorgen_2, errorgen_1, -w, identity)


def _commutator_SS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [S_P, S_Q] = 0
    return []


def _commutator_SC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [S_P, C_{A,B}] = -i (A_{PA,BP} + A_{PB,AP}) - i/2 (A_{{A,B}P,P} + A_{P,P{A,B}})
    P = _index(errorgen_1, 0)
    A = _index(errorgen_2, 0)
    B = _index(errorgen_2, 1)
    terms = []
    # Both groups vanish unless P anticommutes with exactly one of A, B:
    #   A_{PA,BP} + A_{PB,AP} = (s_PB - s_PA) A_{PA,PB}, and with X = {A,B} (which is
    #   proportional to AB, so s_XP = s_PA s_PB), A_{XP,P} + A_{P,PX} = (1 - s_XP) A_{XP,P}.
    cPA = P[1].commutes(A[1])
    cPB = P[1].commutes(B[1])
    if cPA != cPB:
        _A(terms, _prod(P, A), _prod(P, B), -2j*w if cPB else 2j*w, identity)
        X = _acom(A, B)
        if X is not None:
            _A(terms, _prod(X, P), P, -1j*w, identity)
    return terms


def _commutator_CS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [C_{A,B}, S_P] = -[S_P, C_{A,B}]
    return _commutator_SC(errorgen_2, errorgen_1, -w, identity)


def _commutator_SA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [S_P, A_{A,B}] = i (C_{PA,BP} - C_{PB,AP}) - 1/2 A_{P,[P,[A,B]]}
    P = _index(errorgen_1, 0)
    A = _index(errorgen_2, 0)
    B = _index(errorgen_2, 1)
    terms = []
    # Both groups vanish unless P anticommutes with exactly one of A, B:
    #   C_{PA,BP} - C_{PB,AP} = (s_PB - s_PA) C_{PA,PB}, and [P,[A,B]] = 0 unless P
    #   anticommutes with AB, i.e. s_PA s_PB = -1. Under that condition [P, Y] = 2 PY for
    #   Y = [A,B] (see the section comment), so only Y itself needs a commutation test.
    cPA = P[1].commutes(A[1])
    cPB = P[1].commutes(B[1])
    if cPA != cPB:
        _C(terms, _prod(P, A), _prod(P, B), 2j*w if cPB else -2j*w, identity)
        _A(terms, P, _prod(P, _com(A, B)), -1.0*w, identity)
    return terms


def _commutator_AS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [A_{A,B}, S_P] = -[S_P, A_{A,B}]
    return _commutator_SA(errorgen_2, errorgen_1, -w, identity)


def _commutator_CC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [C_{A,B}, C_{P,Q}] = -i (A_{AP,QB} + A_{AQ,PB} + A_{BP,QA} + A_{BQ,PA})
    #                      - i/2 (A_{[P,{A,B}],Q} + A_{[Q,{A,B}],P} + A_{[{P,Q},A],B} + A_{[{P,Q},B],A})
    #                      + i/4 H_{[{A,B},{P,Q}]}
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    X = _acom(A, B)
    Y = _acom(P, Q)
    terms = []
    # A_{AP,QB} + A_{BQ,PA} = (s_BQ - s_AP) A_{AP,BQ};  A_{AQ,PB} + A_{BP,QA} = (s_BP - s_AQ) A_{AQ,BP}
    cAP = A[1].commutes(P[1])
    cAQ = A[1].commutes(Q[1])
    cBP = B[1].commutes(P[1])
    cBQ = B[1].commutes(Q[1])
    if cAP != cBQ:
        _A(terms, _prod(A, P), _prod(B, Q), -2j*w if cBQ else 2j*w, identity)
    if cAQ != cBP:
        _A(terms, _prod(A, Q), _prod(B, P), -2j*w if cBP else 2j*w, identity)
    # The remaining commutators are decided by the same four booleans (see the section comment):
    # [P,X] != 0 iff cAP != cBP, [Q,X] iff cAQ != cBQ, [Y,A] iff cAP != cAQ, [Y,B] iff cBP != cBQ,
    # [X,Y] iff an odd number of the cross pairs anticommute; each nonzero commutator is twice the
    # product, and that factor of 2 is folded into the coefficients below.
    if X is not None:
        if cAP != cBP:
            _A(terms, _prod(P, X), Q, -1j*w, identity)
        if cAQ != cBQ:
            _A(terms, _prod(Q, X), P, -1j*w, identity)
    if Y is not None:
        if cAP != cAQ:
            _A(terms, _prod(Y, A), B, -1j*w, identity)
        if cBP != cBQ:
            _A(terms, _prod(Y, B), A, -1j*w, identity)
    if X is not None and Y is not None and (cAP ^ cAQ ^ cBP ^ cBQ):
        _H(terms, _prod(X, Y), 0.5j*w, identity)
    return terms


def _commutator_CA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [C_{A,B}, A_{P,Q}] = i (C_{AP,QB} - C_{AQ,PB} + C_{BP,QA} - C_{PA,BQ})
    #                      + 1/2 (A_{[A,[P,Q]],B} + A_{[B,[P,Q]],A} + i C_{[P,{A,B}],Q} - i C_{[Q,{A,B}],P})
    #                      - 1/4 H_{[[P,Q],{A,B}]}
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    X = _acom(A, B)
    Y = _com(P, Q)
    terms = []
    # C_{AP,QB} - C_{PA,BQ} = (s_BQ - s_AP) C_{AP,BQ};  C_{BP,QA} - C_{AQ,PB} = (s_AQ - s_BP) C_{AQ,BP}
    cAP = A[1].commutes(P[1])
    cAQ = A[1].commutes(Q[1])
    cBP = B[1].commutes(P[1])
    cBQ = B[1].commutes(Q[1])
    if cAP != cBQ:
        _C(terms, _prod(A, P), _prod(B, Q), 2j*w if cBQ else -2j*w, identity)
    if cAQ != cBP:
        _C(terms, _prod(A, Q), _prod(B, P), 2j*w if cAQ else -2j*w, identity)
    # The remaining commutators are decided by the same four booleans (see the section comment):
    # [P,X] != 0 iff cAP != cBP, [Q,X] iff cAQ != cBQ, [Y,A] iff cAP != cAQ, [Y,B] iff cBP != cBQ,
    # [X,Y] iff an odd number of the cross pairs anticommute; each nonzero commutator is twice the
    # product, and that factor of 2 is folded into the coefficients below.
    if Y is not None:
        if cAP != cAQ:
            _A(terms, _prod(A, Y), B, 1.0*w, identity)
        if cBP != cBQ:
            _A(terms, _prod(B, Y), A, 1.0*w, identity)
    if X is not None:
        if cAP != cBP:
            _C(terms, _prod(P, X), Q, 1j*w, identity)
        if cAQ != cBQ:
            _C(terms, _prod(Q, X), P, -1j*w, identity)
    if X is not None and Y is not None and (cAP ^ cAQ ^ cBP ^ cBQ):
        _H(terms, _prod(Y, X), -0.5*w, identity)
    return terms


def _commutator_AC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [A_{A,B}, C_{P,Q}] = -[C_{P,Q}, A_{A,B}]
    return _commutator_CA(errorgen_2, errorgen_1, -w, identity)


def _commutator_AA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # [A_{A,B}, A_{P,Q}] = -i (A_{QB,AP} + A_{PA,BQ} + A_{BP,QA} + A_{AQ,PB})
    #                      + 1/2 (C_{[B,[P,Q]],A} - C_{[A,[P,Q]],B} + C_{[P,[A,B]],Q} - C_{[Q,[A,B]],P})
    #                      + i/4 H_{[[P,Q],[A,B]]}
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    X = _com(A, B)
    Y = _com(P, Q)
    terms = []
    # A_{QB,AP} + A_{PA,BQ} = (s_AP - s_BQ) A_{AP,BQ};  A_{BP,QA} + A_{AQ,PB} = (s_BP - s_AQ) A_{AQ,BP}
    cAP = A[1].commutes(P[1])
    cAQ = A[1].commutes(Q[1])
    cBP = B[1].commutes(P[1])
    cBQ = B[1].commutes(Q[1])
    if cAP != cBQ:
        _A(terms, _prod(A, P), _prod(B, Q), -2j*w if cAP else 2j*w, identity)
    if cAQ != cBP:
        _A(terms, _prod(A, Q), _prod(B, P), -2j*w if cBP else 2j*w, identity)
    # The remaining commutators are decided by the same four booleans (see the section comment):
    # [P,X] != 0 iff cAP != cBP, [Q,X] iff cAQ != cBQ, [Y,A] iff cAP != cAQ, [Y,B] iff cBP != cBQ,
    # [X,Y] iff an odd number of the cross pairs anticommute; each nonzero commutator is twice the
    # product, and that factor of 2 is folded into the coefficients below.
    if Y is not None:
        if cBP != cBQ:
            _C(terms, _prod(B, Y), A, 1.0*w, identity)
        if cAP != cAQ:
            _C(terms, _prod(A, Y), B, -1.0*w, identity)
    if X is not None:
        if cAP != cBP:
            _C(terms, _prod(P, X), Q, 1.0*w, identity)
        if cAQ != cBQ:
            _C(terms, _prod(Q, X), P, -1.0*w, identity)
    if X is not None and Y is not None and (cAP ^ cAQ ^ cBP ^ cBQ):
        _H(terms, _prod(Y, X), 0.5j*w, identity)
    return terms



# ---------------------------------------------------------------------------------------
# Composition handlers, one per ordered type pair, called as handler(errorgen_1, errorgen_2,
# weight, identity) and computing errorgen_1[errorgen_2[.]]. These have no closed formulas
# in the paper; each handler is the case table of the previously validated implementation,
# written per *slot*: an output term whose indices are fixed products of the input Paulis
# and whose type and prefactor depend on which input pairs commute. Notation used in the
# comment tables (c = the pair commutes, a = it anticommutes):
#   "(PA,QB) by (A,P),(B,Q):  cc: +C   ca: -iA   ac: +iA   aa: -C"
#       the term is +C_{PA,QB} if both pairs commute, -i A_{PA,QB} if (A,P) commutes and
#       (B,Q) anticommutes, +i A_{PA,QB} in the opposite case and -C_{PA,QB} if both anticommute;
#   "(APQ,B) by (A,PQ):  c: -C   a: +iA"
#       keys on whether A commutes with the product PQ, i.e. on whether the pairs (A,P) and
#       (A,Q) commute alike.
# Slots built on the (anti)commutator of a generator's own index pair ({A,B} for C_{A,B},
# [A,B] for A_{A,B}, likewise for P,Q) exist only when it is nonzero; the H_{ABPQ} term
# additionally requires an odd number of the four cross pairs to anticommute.
# ---------------------------------------------------------------------------------------

def _composition_HH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # H_P[H_Q] = C_{P,Q} - i/2 H_{[P,Q]}
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_2, 0)
    terms = []
    _C(terms, P, Q, w, identity)
    _H(terms, _com(P, Q), -0.5j*w, identity)
    return terms


def _composition_HS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # H_P[S_Q] = -H_P - A_{PQ,Q}    if [P,Q] = 0
    #          = -H_P - i C_{PQ,Q}  if {P,Q} = 0
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_2, 0)
    PQ = _prod(P, Q)
    terms = []
    _H(terms, P, -w, identity)
    if P[1].commutes(Q[1]):
        _A(terms, PQ, Q, -w, identity)
    else:
        _C(terms, PQ, Q, -1j*w, identity)
    return terms


def _composition_SH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # S_P[H_Q] = -H_Q - A_{PQ,P}    if [P,Q] = 0
    #          = -H_Q - i C_{PQ,P}  if {P,Q} = 0
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_2, 0)
    PQ = _prod(P, Q)
    terms = []
    _H(terms, Q, -w, identity)
    if P[1].commutes(Q[1]):
        _A(terms, PQ, P, -w, identity)
    else:
        _C(terms, PQ, P, -1j*w, identity)
    return terms


def _composition_SS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # S_P[S_Q] = S_{R} - S_P - S_Q; R = PQ/sign(PQ)
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_2, 0)
    terms = []
    _S(terms, _prod(P, Q), w, identity)
    _S(terms, P, -w, identity)
    _S(terms, Q, -w, identity)
    return terms


def _composition_HC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # H_A[C_{P,Q}]:
    #   (PA,Q) by (A,P):  c: -A   a: +iC
    #   (QA,P) by (A,Q):  c: -A   a: +iC
    #   if {P,Q} != 0:  -A_{PQ,A},  and  -H_{APQ} if (A,P), (A,Q) commute alike
    A = _index(errorgen_1, 0)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    terms = []
    if com_AP:
        _A(terms, PA, Q, -w, identity)
    else:
        _C(terms, PA, Q, 1j*w, identity)
    if com_AQ:
        _A(terms, QA, P, -w, identity)
    else:
        _C(terms, QA, P, 1j*w, identity)
    if P[1].commutes(Q[1]):
        PQ = _prod(P, Q)
        _A(terms, PQ, A, -w, identity)
        if com_AP == com_AQ:
            _H(terms, _prod(A, PQ), -w, identity)
    return terms


def _composition_HA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # H_A[A_{P,Q}]:
    #   (PA,Q) by (A,P):  c: +C   a: +iA
    #   (QA,P) by (A,Q):  c: -C   a: -iA
    #   if [P,Q] != 0:  +i A_{PQ,A},  and  +i H_{APQ} if (A,P), (A,Q) commute alike
    A = _index(errorgen_1, 0)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    terms = []
    if com_AP:
        _C(terms, PA, Q, w, identity)
    else:
        _A(terms, PA, Q, 1j*w, identity)
    if com_AQ:
        _C(terms, QA, P, -w, identity)
    else:
        _A(terms, QA, P, -1j*w, identity)
    if not P[1].commutes(Q[1]):
        PQ = _prod(P, Q)
        _A(terms, PQ, A, 1j*w, identity)
        if com_AP == com_AQ:
            _H(terms, _prod(A, PQ), 1j*w, identity)
    return terms


def _composition_SC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # S_A[C_{P,Q}] = -C_{P,Q} + ...
    #   (PA,QA) by (A,P),(A,Q):  cc: +C   ca: -iA   ac: +iA   aa: -C
    #   if {P,Q} != 0:  (APQ,A) by (A,PQ):  c: -C   a: +iA
    A = _index(errorgen_1, 0)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    terms = []
    _C(terms, P, Q, -w, identity)
    if com_AP == com_AQ:
        _C(terms, PA, QA, w if com_AP else -w, identity)
    else:
        _A(terms, PA, QA, -1j*w if com_AP else 1j*w, identity)
    if P[1].commutes(Q[1]):
        APQ = _prod(A, _prod(P, Q))
        if com_AP == com_AQ:
            _C(terms, APQ, A, -w, identity)
        else:
            _A(terms, APQ, A, 1j*w, identity)
    return terms


def _composition_SA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # S_A[A_{P,Q}] = -A_{P,Q} + ...
    #   (PA,QA) by (A,P),(A,Q):  cc: +A   ca: +iC   ac: -iC   aa: -A
    #   if [P,Q] != 0:  (APQ,A) by (A,PQ):  c: +iC   a: +A
    A = _index(errorgen_1, 0)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    terms = []
    _A(terms, P, Q, -w, identity)
    if com_AP == com_AQ:
        _A(terms, PA, QA, w if com_AP else -w, identity)
    else:
        _C(terms, PA, QA, 1j*w if com_AP else -1j*w, identity)
    if not P[1].commutes(Q[1]):
        APQ = _prod(A, _prod(P, Q))
        if com_AP == com_AQ:
            _C(terms, APQ, A, 1j*w, identity)
        else:
            _A(terms, APQ, A, w, identity)
    return terms


def _composition_CH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # C_{P,Q}[H_A]:
    #   (PA,Q) by (A,P):  c: -A   a: -iC
    #   (QA,P) by (A,Q):  c: -A   a: -iC
    #   if {P,Q} != 0:  -A_{PQ,A},  and  -H_{APQ} if (A,P), (A,Q) commute alike
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_1, 1)
    A = _index(errorgen_2, 0)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    terms = []
    if com_AP:
        _A(terms, PA, Q, -w, identity)
    else:
        _C(terms, PA, Q, -1j*w, identity)
    if com_AQ:
        _A(terms, QA, P, -w, identity)
    else:
        _C(terms, QA, P, -1j*w, identity)
    if P[1].commutes(Q[1]):
        PQ = _prod(P, Q)
        _A(terms, PQ, A, -w, identity)
        if com_AP == com_AQ:
            _H(terms, _prod(A, PQ), -w, identity)
    return terms


def _composition_CS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # C_{P,Q}[S_A] = -C_{P,Q} + ...
    #   (PA,QA) by (A,P),(A,Q):  cc: +C   ca: +iA   ac: -iA   aa: -C
    #   if {P,Q} != 0:  (APQ,A) by (A,PQ):  c: -C   a: -iA
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_1, 1)
    A = _index(errorgen_2, 0)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    terms = []
    _C(terms, P, Q, -w, identity)
    if com_AP == com_AQ:
        _C(terms, PA, QA, w if com_AP else -w, identity)
    else:
        _A(terms, PA, QA, 1j*w if com_AP else -1j*w, identity)
    if P[1].commutes(Q[1]):
        APQ = _prod(A, _prod(P, Q))
        if com_AP == com_AQ:
            _C(terms, APQ, A, -w, identity)
        else:
            _A(terms, APQ, A, -1j*w, identity)
    return terms


def _composition_AH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # A_{P,Q}[H_A]:
    #   (PA,Q) by (A,P):  c: +C   a: -iA
    #   (QA,P) by (A,Q):  c: -C   a: +iA
    #   if [P,Q] != 0:  +i A_{PQ,A},  and  +i H_{APQ} if (A,P), (A,Q) commute alike
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_1, 1)
    A = _index(errorgen_2, 0)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    terms = []
    if com_AP:
        _C(terms, PA, Q, w, identity)
    else:
        _A(terms, PA, Q, -1j*w, identity)
    if com_AQ:
        _C(terms, QA, P, -w, identity)
    else:
        _A(terms, QA, P, 1j*w, identity)
    if not P[1].commutes(Q[1]):
        PQ = _prod(P, Q)
        _A(terms, PQ, A, 1j*w, identity)
        if com_AP == com_AQ:
            _H(terms, _prod(A, PQ), 1j*w, identity)
    return terms


def _composition_AS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # A_{P,Q}[S_A] = -A_{P,Q} + ...
    #   (PA,QA) by (A,P),(A,Q):  cc: +A   ca: -iC   ac: +iC   aa: -A
    #   if [P,Q] != 0:  (APQ,A) by (A,PQ):  c: +iC   a: -A
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_1, 1)
    A = _index(errorgen_2, 0)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    terms = []
    _A(terms, P, Q, -w, identity)
    if com_AP == com_AQ:
        _A(terms, PA, QA, w if com_AP else -w, identity)
    else:
        _C(terms, PA, QA, -1j*w if com_AP else 1j*w, identity)
    if not P[1].commutes(Q[1]):
        APQ = _prod(A, _prod(P, Q))
        if com_AP == com_AQ:
            _C(terms, APQ, A, 1j*w, identity)
        else:
            _A(terms, APQ, A, -w, identity)
    return terms


def _composition_CC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # C_{A,B}[C_{P,Q}]; the {A,B} and {P,Q} slots exist only when those anticommutators are nonzero.
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AB = A[1].commutes(B[1])
    com_PQ = P[1].commutes(Q[1])
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    com_BP = B[1].commutes(P[1])
    com_BQ = B[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    PB = _prod(P, B)
    QB = _prod(Q, B)
    terms = []
    # (PA,QB) by (A,P),(B,Q):  cc: +C   ca: -iA   ac: +iA   aa: -C
    if com_AP == com_BQ:
        _C(terms, PA, QB, w if com_AP else -w, identity)
    else:
        _A(terms, PA, QB, -1j*w if com_AP else 1j*w, identity)
    # (QA,PB) by (A,Q),(B,P):  cc: +C   ca: -iA   ac: +iA   aa: -C
    if com_AQ == com_BP:
        _C(terms, QA, PB, w if com_AQ else -w, identity)
    else:
        _A(terms, QA, PB, -1j*w if com_AQ else 1j*w, identity)
    if com_PQ:  # {P,Q} != 0
        PQ = _prod(P, Q)
        APQ = _prod(A, PQ)
        BPQ = _prod(B, PQ)
        # (APQ,B) by (A,PQ):  c: -C   a: +iA
        if com_AP == com_AQ:
            _C(terms, APQ, B, -w, identity)
        else:
            _A(terms, APQ, B, 1j*w, identity)
        # (BPQ,A) by (B,PQ):  c: -C   a: +iA
        if com_BP == com_BQ:
            _C(terms, BPQ, A, -w, identity)
        else:
            _A(terms, BPQ, A, 1j*w, identity)
    if com_AB:  # {A,B} != 0
        AB = _prod(A, B)
        PAB = _prod(P, AB)
        QAB = _prod(Q, AB)
        # (PAB,Q) by (P,AB):  c: -C   a: -iA
        if com_AP == com_BP:
            _C(terms, PAB, Q, -w, identity)
        else:
            _A(terms, PAB, Q, -1j*w, identity)
        # (QAB,P) by (Q,AB):  c: -C   a: -iA
        if com_AQ == com_BQ:
            _C(terms, QAB, P, -w, identity)
        else:
            _A(terms, QAB, P, -1j*w, identity)
        if com_PQ:  # both anticommutators nonzero
            # (PQ,AB):  +C
            _C(terms, PQ, AB, w, identity)
            # H_{ABPQ}:  +iH, present iff an odd number of (A,P), (A,Q), (B,P), (B,Q) anticommute
            if (com_AP + com_AQ + com_BP + com_BQ) % 2 == 1:
                _H(terms, _prod(AB, PQ), 1j*w, identity)
    return terms


def _composition_CA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # C_{A,B}[A_{P,Q}]; the {A,B} and [P,Q] slots exist only when the anticommutator and the commutator are nonzero.
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AB = A[1].commutes(B[1])
    com_PQ = P[1].commutes(Q[1])
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    com_BP = B[1].commutes(P[1])
    com_BQ = B[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    PB = _prod(P, B)
    QB = _prod(Q, B)
    terms = []
    # (PA,QB) by (A,P),(B,Q):  cc: +A   ca: +iC   ac: -iC   aa: -A
    if com_AP == com_BQ:
        _A(terms, PA, QB, w if com_AP else -w, identity)
    else:
        _C(terms, PA, QB, 1j*w if com_AP else -1j*w, identity)
    # (QA,PB) by (A,Q),(B,P):  cc: -A   ca: -iC   ac: +iC   aa: +A
    if com_AQ == com_BP:
        _A(terms, QA, PB, -w if com_AQ else w, identity)
    else:
        _C(terms, QA, PB, -1j*w if com_AQ else 1j*w, identity)
    if not com_PQ:  # [P,Q] != 0
        PQ = _prod(P, Q)
        APQ = _prod(A, PQ)
        BPQ = _prod(B, PQ)
        # (APQ,B) by (A,PQ):  c: +iC   a: +A
        if com_AP == com_AQ:
            _C(terms, APQ, B, 1j*w, identity)
        else:
            _A(terms, APQ, B, w, identity)
        # (BPQ,A) by (B,PQ):  c: +iC   a: +A
        if com_BP == com_BQ:
            _C(terms, BPQ, A, 1j*w, identity)
        else:
            _A(terms, BPQ, A, w, identity)
    if com_AB:  # {A,B} != 0
        AB = _prod(A, B)
        PAB = _prod(P, AB)
        QAB = _prod(Q, AB)
        # (PAB,Q) by (P,AB):  c: -A   a: +iC
        if com_AP == com_BP:
            _A(terms, PAB, Q, -w, identity)
        else:
            _C(terms, PAB, Q, 1j*w, identity)
        # (QAB,P) by (Q,AB):  c: +A   a: -iC
        if com_AQ == com_BQ:
            _A(terms, QAB, P, w, identity)
        else:
            _C(terms, QAB, P, -1j*w, identity)
        if not com_PQ:  # anticommutator and commutator both nonzero
            # (PQ,AB):  -iC
            _C(terms, PQ, AB, -1j*w, identity)
            # H_{ABPQ}:  +H, present iff an odd number of (A,P), (A,Q), (B,P), (B,Q) anticommute
            if (com_AP + com_AQ + com_BP + com_BQ) % 2 == 1:
                _H(terms, _prod(AB, PQ), w, identity)
    return terms


def _composition_AC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # A_{A,B}[C_{P,Q}]; the [A,B] and {P,Q} slots exist only when the commutator and the anticommutator are nonzero.
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AB = A[1].commutes(B[1])
    com_PQ = P[1].commutes(Q[1])
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    com_BP = B[1].commutes(P[1])
    com_BQ = B[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    PB = _prod(P, B)
    QB = _prod(Q, B)
    terms = []
    # (PA,QB) by (A,P),(B,Q):  cc: +A   ca: +iC   ac: -iC   aa: -A
    if com_AP == com_BQ:
        _A(terms, PA, QB, w if com_AP else -w, identity)
    else:
        _C(terms, PA, QB, 1j*w if com_AP else -1j*w, identity)
    # (QA,PB) by (A,Q),(B,P):  cc: +A   ca: +iC   ac: -iC   aa: -A
    if com_AQ == com_BP:
        _A(terms, QA, PB, w if com_AQ else -w, identity)
    else:
        _C(terms, QA, PB, 1j*w if com_AQ else -1j*w, identity)
    if com_PQ:  # {P,Q} != 0
        PQ = _prod(P, Q)
        APQ = _prod(A, PQ)
        BPQ = _prod(B, PQ)
        # (APQ,B) by (A,PQ):  c: -A   a: -iC
        if com_AP == com_AQ:
            _A(terms, APQ, B, -w, identity)
        else:
            _C(terms, APQ, B, -1j*w, identity)
        # (BPQ,A) by (B,PQ):  c: +A   a: +iC
        if com_BP == com_BQ:
            _A(terms, BPQ, A, w, identity)
        else:
            _C(terms, BPQ, A, 1j*w, identity)
    if not com_AB:  # [A,B] != 0
        AB = _prod(A, B)
        PAB = _prod(P, AB)
        QAB = _prod(Q, AB)
        # (PAB,Q) by (P,AB):  c: +iC   a: -A
        if com_AP == com_BP:
            _C(terms, PAB, Q, 1j*w, identity)
        else:
            _A(terms, PAB, Q, -w, identity)
        # (QAB,P) by (Q,AB):  c: +iC   a: -A
        if com_AQ == com_BQ:
            _C(terms, QAB, P, 1j*w, identity)
        else:
            _A(terms, QAB, P, -w, identity)
        if com_PQ:  # commutator and anticommutator both nonzero
            # (PQ,AB):  -iC
            _C(terms, PQ, AB, -1j*w, identity)
            # H_{ABPQ}:  +H, present iff an odd number of (A,P), (A,Q), (B,P), (B,Q) anticommute
            if (com_AP + com_AQ + com_BP + com_BQ) % 2 == 1:
                _H(terms, _prod(AB, PQ), w, identity)
    return terms


def _composition_AA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # A_{A,B}[A_{P,Q}]; the [A,B] and [P,Q] slots exist only when those commutators are nonzero.
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AB = A[1].commutes(B[1])
    com_PQ = P[1].commutes(Q[1])
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    com_BP = B[1].commutes(P[1])
    com_BQ = B[1].commutes(Q[1])
    PA = _prod(P, A)
    QA = _prod(Q, A)
    PB = _prod(P, B)
    QB = _prod(Q, B)
    terms = []
    # (PA,QB) by (A,P),(B,Q):  cc: -C   ca: +iA   ac: -iA   aa: +C
    if com_AP == com_BQ:
        _C(terms, PA, QB, -w if com_AP else w, identity)
    else:
        _A(terms, PA, QB, 1j*w if com_AP else -1j*w, identity)
    # (QA,PB) by (A,Q),(B,P):  cc: +C   ca: -iA   ac: +iA   aa: -C
    if com_AQ == com_BP:
        _C(terms, QA, PB, w if com_AQ else -w, identity)
    else:
        _A(terms, QA, PB, -1j*w if com_AQ else 1j*w, identity)
    if not com_PQ:  # [P,Q] != 0
        PQ = _prod(P, Q)
        APQ = _prod(A, PQ)
        BPQ = _prod(B, PQ)
        # (APQ,B) by (A,PQ):  c: +iA   a: -C
        if com_AP == com_AQ:
            _A(terms, APQ, B, 1j*w, identity)
        else:
            _C(terms, APQ, B, -w, identity)
        # (BPQ,A) by (B,PQ):  c: -iA   a: +C
        if com_BP == com_BQ:
            _A(terms, BPQ, A, -1j*w, identity)
        else:
            _C(terms, BPQ, A, w, identity)
    if not com_AB:  # [A,B] != 0
        AB = _prod(A, B)
        PAB = _prod(P, AB)
        QAB = _prod(Q, AB)
        # (PAB,Q) by (P,AB):  c: +iA   a: +C
        if com_AP == com_BP:
            _A(terms, PAB, Q, 1j*w, identity)
        else:
            _C(terms, PAB, Q, w, identity)
        # (QAB,P) by (Q,AB):  c: -iA   a: -C
        if com_AQ == com_BQ:
            _A(terms, QAB, P, -1j*w, identity)
        else:
            _C(terms, QAB, P, -w, identity)
        if not com_PQ:  # both commutators nonzero
            # (PQ,AB):  -C
            _C(terms, PQ, AB, -w, identity)
            # H_{ABPQ}:  -iH, present iff an odd number of (A,P), (A,Q), (B,P), (B,Q) anticommute
            if (com_AP + com_AQ + com_BP + com_BQ) % 2 == 1:
                _H(terms, _prod(AB, PQ), -1j*w, identity)
    return terms



# ---------------------------------------------------------------------------------------
# Pauli conjugation handlers, one per type of the conjugated generator, called as
# handler(Q, errorgen, weight, identity) with Q the conjugating Pauli as a signed-Pauli
# triple (1, Q, s) and computing 𝒬[errorgen[.]], 𝒬[rho] = Q rho Q. Since S_Q = 𝒬 - 1, each
# is the S_Q o X composition table (`_composition_SH/SS/SC/SA`) with its leading -X term
# removed; the case tables are written in the same slot notation as the composition
# handlers. Each handler is headed by its relation.
# ---------------------------------------------------------------------------------------

def _conjugation_H(Q: _SignedPauli, errorgen: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # 𝒬 H_P = -A_{QP,Q}    if [Q,P] = 0
    #       = -i C_{QP,Q}  if {Q,P} = 0
    P = _index(errorgen, 0)
    QP = _prod(Q, P)
    terms = []
    if Q[1].commutes(P[1]):
        _A(terms, QP, Q, -w, identity)
    else:
        _C(terms, QP, Q, -1j*w, identity)
    return terms


def _conjugation_S(Q: _SignedPauli, errorgen: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # 𝒬 S_P = S_R - S_Q,  R = QP / phase(QP)
    # (Q (P rho P - rho) Q = R rho R - Q rho Q, the phase dropping out because rho is
    # conjugated by QP and its adjoint; `_S` discards the phase. For P = Q, R = I and only
    # -S_Q remains: 𝒬 S_Q = 𝒬(𝒬 - 1) = 1 - 𝒬 = -S_Q.)
    P = _index(errorgen, 0)
    terms = []
    _S(terms, _prod(Q, P), w, identity)
    _S(terms, Q, -w, identity)
    return terms


def _conjugation_C(Q: _SignedPauli, errorgen: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # 𝒬 C_{P,R} = ½ (C_{QP,RQ} + C_{QR,PQ}) - i/2 (A_{QP,RQ} + A_{QR,PQ}) - ¼ C_{{Q,{P,R}},Q} + i/4 A_{[Q,{P,R}],Q}
    #   (PQ,RQ) by (Q,P),(Q,R):  cc: +C   ca: -iA   ac: +iA   aa: -C
    #   if {P,R} != 0:  (QPR,Q) by (Q,PR):  c: -C   a: +iA
    P = _index(errorgen, 0)
    R = _index(errorgen, 1)
    com_QP = Q[1].commutes(P[1])
    com_QR = Q[1].commutes(R[1])
    PQ = _prod(P, Q)
    RQ = _prod(R, Q)
    terms = []
    if com_QP == com_QR:
        _C(terms, PQ, RQ, w if com_QP else -w, identity)
    else:
        _A(terms, PQ, RQ, -1j*w if com_QP else 1j*w, identity)
    if P[1].commutes(R[1]):
        QPR = _prod(Q, _prod(P, R))
        if com_QP == com_QR:
            _C(terms, QPR, Q, -w, identity)
        else:
            _A(terms, QPR, Q, 1j*w, identity)
    return terms


def _conjugation_A(Q: _SignedPauli, errorgen: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # 𝒬 A_{P,R} = ½ (A_{QP,RQ} - A_{QR,PQ}) + i/2 (C_{QP,RQ} - C_{QR,PQ}) + i/4 C_{{Q,[P,R]},Q} + ¼ A_{[Q,[P,R]],Q}
    #   (PQ,RQ) by (Q,P),(Q,R):  cc: +A   ca: +iC   ac: -iC   aa: -A
    #   if [P,R] != 0:  (QPR,Q) by (Q,PR):  c: +iC   a: +A
    P = _index(errorgen, 0)
    R = _index(errorgen, 1)
    com_QP = Q[1].commutes(P[1])
    com_QR = Q[1].commutes(R[1])
    PQ = _prod(P, Q)
    RQ = _prod(R, Q)
    terms = []
    if com_QP == com_QR:
        _A(terms, PQ, RQ, w if com_QP else -w, identity)
    else:
        _C(terms, PQ, RQ, 1j*w if com_QP else -1j*w, identity)
    if not P[1].commutes(R[1]):
        QPR = _prod(Q, _prod(P, R))
        if com_QP == com_QR:
            _C(terms, QPR, Q, 1j*w, identity)
        else:
            _A(terms, QPR, Q, w, identity)
    return terms



# ---------------------------------------------------------------------------------------
# Anticommutator handlers, one per *unordered* type pair (the anticommutator is symmetric;
# `error_generator_anticommutator` orders its arguments by type before dispatch, so each
# handler sees the lower type index first). Called as handler(errorgen_1, errorgen_2,
# weight, identity). Each is headed by its relation and
# its Pauli case table in the slot notation of the composition handlers. The tables are the
# even-sign complements of the commutator's: a two-index slot such as (PA,QB) contributes
# C_{PA,QB} + C_{AP,BQ} = (1 + s_AP s_BQ) C_{PA,QB}, i.e. twice one term when the pairs (A,P)
# and (B,Q) commute alike and nothing otherwise (the commutator takes the other case), and
# the (anti)commutator slots are present when their Pauli commutes with the product, i.e. when
# the corresponding pairs commute alike (see the section comment of the commutator handlers
# for why the four cross-pair booleans decide every commutator). The H_{ABPQ} term of the C/A
# compositions always cancels in the anticommutator and is never formed. Products are formed
# inside the gate of the slot that uses them (each two-index product serves exactly one slot
# here), never up front as in the composition handlers.
#
# Handlers with an S operand emit the "bleed" term -2X (the -1 in S_Q = 𝒬 - 1 acting on the
# partner X, see the module docstring) as a fresh label through the emitters, like every
# other term, and then the conjugation part (1 + (-1)^ω(Q,T_X)) 𝒬X through the conjugation
# handlers above; the ω test is on the pairs (Q, P), (Q, R) commuting alike. This keeps the
# emission structure of these handlers identical to the commutator handlers'. It is not the
# cheapest way to obtain a *sum* of anticommutators: `_commuting_product` recovers all the
# bleed terms of L o M in closed form (a rescaling of L and M) and never calls these four.
# ---------------------------------------------------------------------------------------

def _anticommutator_HH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {H_P, H_Q} = 2 C_{P,Q}
    P = _index(errorgen_1, 0)
    Q = _index(errorgen_2, 0)
    terms = []
    _C(terms, P, Q, 2*w, identity)
    return terms


def _anticommutator_HS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {H_P, S_Q} = -2 H_P - A_{{P,Q},Q}
    #            = -2 H_P + (1 + (-1)^ω(Q,P)) 𝒬 H_P
    P = _index(errorgen_1, 0)
    terms = []
    _H(terms, P, -2*w, identity)
    Q = _index(errorgen_2, 0)
    if Q[1].commutes(P[1]):
        terms += _conjugation_H(Q, errorgen_1, 2*w, identity)
    return terms


def _anticommutator_SH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {S_P, H_Q} = {H_Q, S_P}
    return _anticommutator_HS(errorgen_2, errorgen_1, w, identity)


def _anticommutator_SS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {S_P, S_Q} = 2 S_{PQ} - 2 S_P - 2 S_Q
    #            = -2 S_P + 2 𝒬 S_P   (ω(Q, I) = 0: never pure bleed)
    P = _index(errorgen_1, 0)
    terms = []
    _S(terms, P, -2*w, identity)
    terms += _conjugation_S(_index(errorgen_2, 0), errorgen_1, 2*w, identity)
    return terms


def _anticommutator_HC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {H_A, C_{P,Q}} = -(A_{{A,P},Q} + A_{{A,Q},P}) + A_{A,{P,Q}} - ½ H_{{A,{P,Q}}}
    #   (AP,Q) if [A,P] = 0:  -2A;   (AQ,P) if [A,Q] = 0:  -2A
    #   if {P,Q} != 0:  +2 A_{A,PQ},  and  -2 H_{APQ} if (A,P), (A,Q) commute alike
    A = _index(errorgen_1, 0)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    terms = []
    if com_AP:
        _A(terms, _prod(A, P), Q, -2*w, identity)
    if com_AQ:
        _A(terms, _prod(A, Q), P, -2*w, identity)
    if P[1].commutes(Q[1]):
        PQ = _prod(P, Q)
        _A(terms, A, PQ, 2*w, identity)
        if com_AP == com_AQ:
            _H(terms, _prod(A, PQ), -2*w, identity)
    return terms


def _anticommutator_CH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {C_{A,B}, H_P} = {H_P, C_{A,B}}
    return _anticommutator_HC(errorgen_2, errorgen_1, w, identity)


def _anticommutator_HA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {H_A, A_{P,Q}} = (C_{{A,P},Q} - C_{{A,Q},P}) - i A_{A,[P,Q]} + i/2 H_{{A,[P,Q]}}
    #   (AP,Q) if [A,P] = 0:  +2C;   (AQ,P) if [A,Q] = 0:  -2C
    #   if [P,Q] != 0:  -2i A_{A,PQ},  and  +2i H_{APQ} if (A,P), (A,Q) commute alike
    A = _index(errorgen_1, 0)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    terms = []
    if com_AP:
        _C(terms, _prod(A, P), Q, 2*w, identity)
    if com_AQ:
        _C(terms, _prod(A, Q), P, -2*w, identity)
    if not P[1].commutes(Q[1]):
        PQ = _prod(P, Q)
        _A(terms, A, PQ, -2j*w, identity)
        if com_AP == com_AQ:
            _H(terms, _prod(A, PQ), 2j*w, identity)
    return terms


def _anticommutator_AH(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {A_{A,B}, H_P} = {H_P, A_{A,B}}
    return _anticommutator_HA(errorgen_2, errorgen_1, w, identity)


def _anticommutator_SC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {S_A, C_{P,Q}} = -2 C_{P,Q} + C_{AP,QA} + C_{AQ,PA} - ½ C_{{A,{P,Q}},A}
    #                = -2 C_{P,Q} + (1 + (-1)^ω(A,PQ)) 𝒬_A C_{P,Q}
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    terms = []
    _C(terms, P, Q, -2*w, identity)
    A = _index(errorgen_1, 0)
    if A[1].commutes(P[1]) == A[1].commutes(Q[1]):  # ω(A, PQ) = 0
        terms += _conjugation_C(A, errorgen_2, 2*w, identity)
    return terms


def _anticommutator_CS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {C_{A,B}, S_P} = {S_P, C_{A,B}}
    return _anticommutator_SC(errorgen_2, errorgen_1, w, identity)


def _anticommutator_SA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {S_A, A_{P,Q}} = -2 A_{P,Q} + A_{AP,QA} - A_{AQ,PA} + i/2 C_{{A,[P,Q]},A}
    #                = -2 A_{P,Q} + (1 + (-1)^ω(A,PQ)) 𝒬_A A_{P,Q}
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    terms = []
    _A(terms, P, Q, -2*w, identity)
    A = _index(errorgen_1, 0)
    if A[1].commutes(P[1]) == A[1].commutes(Q[1]):  # ω(A, PQ) = 0
        terms += _conjugation_A(A, errorgen_2, 2*w, identity)
    return terms


def _anticommutator_AS(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {A_{A,B}, S_P} = {S_P, A_{A,B}}
    return _anticommutator_SA(errorgen_2, errorgen_1, w, identity)


def _anticommutator_CC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {C_{A,B}, C_{P,Q}} = (C_{AP,QB} + C_{BQ,PA} + C_{AQ,PB} + C_{BP,QA})
    #                      - ½ (C_{{A,{P,Q}},B} + C_{{B,{P,Q}},A}) - ½ (C_{{{A,B},P},Q} + C_{{{A,B},Q},P})
    #                      + ½ C_{{A,B},{P,Q}}
    #   (PA,QB) by (A,P),(B,Q):  cc: +2C   aa: -2C;   (QA,PB) by (A,Q),(B,P):  cc: +2C   aa: -2C
    #   if {P,Q} != 0:  (APQ,B): -2C if (A,P),(A,Q) alike;  (BPQ,A): -2C if (B,P),(B,Q) alike
    #   if {A,B} != 0:  (PAB,Q): -2C if (A,P),(B,P) alike;  (QAB,P): -2C if (A,Q),(B,Q) alike
    #   if both:        (PQ,AB): +2C
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    com_BP = B[1].commutes(P[1])
    com_BQ = B[1].commutes(Q[1])
    terms = []
    if com_AP == com_BQ:
        _C(terms, _prod(P, A), _prod(Q, B), 2*w if com_AP else -2*w, identity)
    if com_AQ == com_BP:
        _C(terms, _prod(Q, A), _prod(P, B), 2*w if com_AQ else -2*w, identity)
    com_PQ = P[1].commutes(Q[1])
    if com_PQ:  # {P,Q} != 0
        PQ = _prod(P, Q)
        if com_AP == com_AQ:
            _C(terms, _prod(A, PQ), B, -2*w, identity)
        if com_BP == com_BQ:
            _C(terms, _prod(B, PQ), A, -2*w, identity)
    if A[1].commutes(B[1]):  # {A,B} != 0
        AB = _prod(A, B)
        if com_AP == com_BP:
            _C(terms, _prod(P, AB), Q, -2*w, identity)
        if com_AQ == com_BQ:
            _C(terms, _prod(Q, AB), P, -2*w, identity)
        if com_PQ:
            _C(terms, PQ, AB, 2*w, identity)
    return terms


def _anticommutator_CA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {C_{A,B}, A_{P,Q}} = (A_{AP,QB} - A_{BQ,PA} - A_{AQ,PB} + A_{BP,QA})
    #                      + i/2 (C_{{A,[P,Q]},B} + C_{{B,[P,Q]},A}) - ½ (A_{{{A,B},P},Q} - A_{{{A,B},Q},P})
    #                      - i/2 C_{{A,B},[P,Q]}
    #   (PA,QB) by (A,P),(B,Q):  cc: +2A   aa: -2A;   (QA,PB) by (A,Q),(B,P):  cc: -2A   aa: +2A
    #   if [P,Q] != 0:  (APQ,B): +2iC if (A,P),(A,Q) alike;  (BPQ,A): +2iC if (B,P),(B,Q) alike
    #   if {A,B} != 0:  (PAB,Q): -2A if (A,P),(B,P) alike;   (QAB,P): +2A if (A,Q),(B,Q) alike
    #   if both:        (PQ,AB): -2iC
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    com_BP = B[1].commutes(P[1])
    com_BQ = B[1].commutes(Q[1])
    terms = []
    if com_AP == com_BQ:
        _A(terms, _prod(P, A), _prod(Q, B), 2*w if com_AP else -2*w, identity)
    if com_AQ == com_BP:
        _A(terms, _prod(Q, A), _prod(P, B), -2*w if com_AQ else 2*w, identity)
    com_PQ = P[1].commutes(Q[1])
    if not com_PQ:  # [P,Q] != 0
        PQ = _prod(P, Q)
        if com_AP == com_AQ:
            _C(terms, _prod(A, PQ), B, 2j*w, identity)
        if com_BP == com_BQ:
            _C(terms, _prod(B, PQ), A, 2j*w, identity)
    if A[1].commutes(B[1]):  # {A,B} != 0
        AB = _prod(A, B)
        if com_AP == com_BP:
            _A(terms, _prod(P, AB), Q, -2*w, identity)
        if com_AQ == com_BQ:
            _A(terms, _prod(Q, AB), P, 2*w, identity)
        if not com_PQ:
            _C(terms, PQ, AB, -2j*w, identity)
    return terms


def _anticommutator_AC(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {A_{A,B}, C_{P,Q}} = {C_{P,Q}, A_{A,B}}
    return _anticommutator_CA(errorgen_2, errorgen_1, w, identity)


def _anticommutator_AA(errorgen_1: _LSE, errorgen_2: _LSE, w: complex, identity: str) -> _ErrorgenTerms:
    # {A_{A,B}, A_{P,Q}} = -(C_{AP,QB} + C_{BQ,PA} - C_{AQ,PB} - C_{BP,QA})
    #                      + i/2 (A_{{A,[P,Q]},B} - A_{{B,[P,Q]},A}) + i/2 (A_{{[A,B],P},Q} - A_{{[A,B],Q},P})
    #                      - ½ C_{[A,B],[P,Q]}
    #   (PA,QB) by (A,P),(B,Q):  cc: -2C   aa: +2C;   (QA,PB) by (A,Q),(B,P):  cc: +2C   aa: -2C
    #   if [P,Q] != 0:  (APQ,B): +2iA if (A,P),(A,Q) alike;  (BPQ,A): -2iA if (B,P),(B,Q) alike
    #   if [A,B] != 0:  (PAB,Q): +2iA if (A,P),(B,P) alike;  (QAB,P): -2iA if (A,Q),(B,Q) alike
    #   if both:        (PQ,AB): -2C
    A = _index(errorgen_1, 0)
    B = _index(errorgen_1, 1)
    P = _index(errorgen_2, 0)
    Q = _index(errorgen_2, 1)
    com_AP = A[1].commutes(P[1])
    com_AQ = A[1].commutes(Q[1])
    com_BP = B[1].commutes(P[1])
    com_BQ = B[1].commutes(Q[1])
    terms = []
    if com_AP == com_BQ:
        _C(terms, _prod(P, A), _prod(Q, B), -2*w if com_AP else 2*w, identity)
    if com_AQ == com_BP:
        _C(terms, _prod(Q, A), _prod(P, B), 2*w if com_AQ else -2*w, identity)
    com_PQ = P[1].commutes(Q[1])
    if not com_PQ:  # [P,Q] != 0
        PQ = _prod(P, Q)
        if com_AP == com_AQ:
            _A(terms, _prod(A, PQ), B, 2j*w, identity)
        if com_BP == com_BQ:
            _A(terms, _prod(B, PQ), A, -2j*w, identity)
    if not A[1].commutes(B[1]):  # [A,B] != 0
        AB = _prod(A, B)
        if com_AP == com_BP:
            _A(terms, _prod(P, AB), Q, 2j*w, identity)
        if com_AQ == com_BQ:
            _A(terms, _prod(Q, AB), P, -2j*w, identity)
        if not com_PQ:
            _C(terms, PQ, AB, -2*w, identity)
    return terms

# Dispatch tables for the error generator commutator and composition, indexed by
# 4*errorgen_1.type_idx + errorgen_2.type_idx with the type order H=0, S=1, C=2, A=3 of
# `pygsti.errorgenpropagation.localstimerrorgen._ERRORGEN_TYPE_INDICES`. Each entry is the
# handler for one ordered type pair, called as handler(errorgen_1, errorgen_2, weight, identity)
# (identity = the 'I'*n string) and returning the list of (LocalStimErrorgenLabel, rate) terms.
_COMMUTATOR_HANDLERS: tuple[_PairHandler, ...] = (
    _commutator_HH, _commutator_HS, _commutator_HC, _commutator_HA,
    _commutator_SH, _commutator_SS, _commutator_SC, _commutator_SA,
    _commutator_CH, _commutator_CS, _commutator_CC, _commutator_CA,
    _commutator_AH, _commutator_AS, _commutator_AC, _commutator_AA,
)

_COMPOSITION_HANDLERS: tuple[_PairHandler, ...] = (
    _composition_HH, _composition_HS, _composition_HC, _composition_HA,
    _composition_SH, _composition_SS, _composition_SC, _composition_SA,
    _composition_CH, _composition_CS, _composition_CC, _composition_CA,
    _composition_AH, _composition_AS, _composition_AC, _composition_AA,
)

# The anticommutator is symmetric, so the ten unordered type pairs carry the formulas and
# the six reversed pairs (SH, CH, AH, CS, AS, AC) delegate to them with the arguments swapped.
_ANTICOMMUTATOR_HANDLERS: tuple[_PairHandler, ...] = (
    _anticommutator_HH, _anticommutator_HS, _anticommutator_HC, _anticommutator_HA,
    _anticommutator_SH, _anticommutator_SS, _anticommutator_SC, _anticommutator_SA,
    _anticommutator_CH, _anticommutator_CS, _anticommutator_CC, _anticommutator_CA,
    _anticommutator_AH, _anticommutator_AS, _anticommutator_AC, _anticommutator_AA,
)

# Pauli conjugation handlers, indexed by the type of the conjugated generator, called as
# handler(Q, errorgen, weight, identity) with Q a signed-Pauli triple (1, Q, s).
_CONJUGATION_HANDLERS: tuple[Callable[[_SignedPauli, _LSE, complex, str], _ErrorgenTerms], ...] = (
    _conjugation_H, _conjugation_S, _conjugation_C, _conjugation_A,
)

def com(P1: stim.PauliString, P2: stim.PauliString) -> Optional[tuple[complex, stim.PauliString]]:
    """
    Commutator of two Paulis, [P1, P2] = P1 P2 - P2 P1.

    Returns None if `P1` and `P2` commute (the commutator is zero), otherwise a tuple
    `(phase, P3)` such that [P1, P2] = phase * P3, where `P3` is the sign-free product
    P1 P2 and `phase` (one of +-2, +-2i) is twice the sign of that product.
    """
    if P1.commutes(P2):
        return None
    P3 = P1*P2
    phase = 2*P3.sign
    P3.sign = 1
    return (phase, P3)


def acom(P1: stim.PauliString, P2: stim.PauliString) -> Optional[tuple[complex, stim.PauliString]]:
    """
    Anticommutator of two Paulis, {P1, P2} = P1 P2 + P2 P1.

    Returns None if `P1` and `P2` anticommute (the anticommutator is zero), otherwise a
    tuple `(phase, P3)` such that {P1, P2} = phase * P3, where `P3` is the sign-free product
    P1 P2 and `phase` (one of +-2, +-2i) is twice the sign of that product.
    """
    if not P1.commutes(P2):
        return None
    P3 = P1*P2
    phase = 2*P3.sign
    P3.sign = 1
    return (phase, P3)


def pauli_product(P1: stim.PauliString, P2: stim.PauliString) -> tuple[complex, stim.PauliString]:
    """
    Product of two Paulis, returned as a tuple `(phase, P3)` such that P1 P2 = phase * P3,
    where `P3` is the sign-free product and `phase` is one of +-1, +-i.
    """
    P3 = P1*P2
    phase = P3.sign
    P3.sign = 1
    return (phase, P3)


def errorgen_pauli_action(errorgen: _LSE, pauli: stim.PauliString) -> tuple[float, stim.PauliString]:
    """
    Apply the specified error generator to a given Pauli operator.

    Parameters
    ----------
    errorgen : `LocalStimErrorgenLabel`
        A label specifying the error generator which should be applied to the specified Pauli operator.
    
    pauli : stim.PauliString
        The pauli operator to apply the error generator to
    
    Returns:
    --------
    A tuple whose first value is the (generally complex) weight of the resulting pauli operator, and whose second
    value is the unsigned Pauli operator itself. 
    
    If the specified error generator annihilates the given pauli then this instead returns None.
    """
    errgen_type = errorgen.errorgen_type
    basis_element_labels = errorgen.basis_element_labels

    # H_P[A] = -i[P,A]
    if errgen_type == 'H':
        ret = com(basis_element_labels[0], pauli)
        if ret is not None:
            ret = (_np.real_if_close(-1j*ret[0]).item(), ret[1]) 
    # S_P[A] = PAP - A
    elif errgen_type == 'S':
        # if P and A commute this gives 0. if they anticommute you get -2A.
        if pauli.commutes(basis_element_labels[0]):
            ret = None
        else:
            ret = (-2, pauli)
    # C_P,Q[A] = PAQ + QAP - (1/2){{P,Q},A}
    elif errgen_type == 'C':
        P = basis_element_labels[0]
        Q = basis_element_labels[1]
        if P.commutes(Q):
            if not P.commutes(pauli) and not Q.commutes(pauli):
                PA = pauli_product(P, pauli)
                PAQ = pauli_product(PA[0]*PA[1], Q)
                ret = (_np.real_if_close(4*PAQ[0]).item(), PAQ[1])
            else:
                ret = None
        else:
            if P.commutes(pauli) ^ Q.commutes(pauli): # xor
                PA = pauli_product(P, pauli)
                PAQ = pauli_product(PA[0]*PA[1], Q)
                ret = (_np.real_if_close(2*PAQ[0]).item(), PAQ[1])
            else:
                ret = None
    # A_P,Q[A] = i(PAQ - QAP + (1/2){[P,Q],A})
    elif errgen_type == 'A':
        P = basis_element_labels[0]
        Q = basis_element_labels[1]
        if P.commutes(Q):
            if P.commutes(pauli) ^ Q.commutes(pauli): # xor
                PA = pauli_product(P, pauli)
                PAQ = pauli_product(PA[0]*PA[1], Q)
                ret = (_np.real_if_close(2*1j*PAQ[0]).item(), PAQ[1])
            else:
                ret = None
        else:
            if P.commutes(pauli) and Q.commutes(pauli):
                PA = pauli_product(P, pauli)
                PAQ = pauli_product(PA[0]*PA[1], Q)
                ret = (_np.real_if_close(4*1j*PAQ[0]).item(), PAQ[1])
            else:
                ret = None
    else:
        raise ValueError(f'Unsupported error generator type {errgen_type}.')
    
    return ret

def iterative_error_generator_composition(errorgen_labels: tuple[_LSE, ...], rates: tuple[complex, ...]) -> _ErrorgenTerms:
    """
    Iteratively compute error generator compositions. Each error generator
    composition in general returns a list of multiple new error generators,
    so this function manages the distribution and recursive application
    of the compositions for two-or-more error generator labels.
    
    Parameters
    ----------
    errorgen_labels : tuple of `LocalStimErrorgenLabel`
        A tuple of the elementary error generator labels to be composed.
    
    rates : tuple of float
        A tuple of corresponding error generator rates of the same length as the tuple
        of error generator labels.

    Returns
    -------
    List of tuples, the first element of each tuple is a `LocalStimErrorgenLabel`.
    The second element of each tuple is the final rate for that term.
    """

    if len(errorgen_labels) == 1:
        return [(errorgen_labels[0], rates[0])]
    else:
        label_tuples_to_process = [errorgen_labels]
        rate_tuples_to_process = [rates]
    
    fully_processed_label_rate_tuples = []    
    while label_tuples_to_process:
        new_label_tuples_to_process = []
        new_rate_tuples_to_process = []

        for label_tup, rate_tup in zip(label_tuples_to_process, rate_tuples_to_process):
            # grab the last two elements of each of these and do the composition.
            new_labels_and_rates = error_generator_composition(label_tup[-2], label_tup[-1], rate_tup[-2]*rate_tup[-1])

            # if the new labels and rates sum to zero overall then we can kill this branch of the tree.
            aggregated_labels_and_rates_dict = dict()
            for lbl, rate in new_labels_and_rates:
                if aggregated_labels_and_rates_dict.get(lbl, None) is None:
                    aggregated_labels_and_rates_dict[lbl] = rate
                else:
                    aggregated_labels_and_rates_dict[lbl] += rate
            if all([abs(val)<1e-15 for val in aggregated_labels_and_rates_dict.values()]):
                continue

            label_tup_remainder = label_tup[:-2]
            rate_tup_remainder = rate_tup[:-2]
            if label_tup_remainder:
                for new_label, new_rate in aggregated_labels_and_rates_dict.items():
                    new_label_tup = label_tup_remainder + (new_label,)
                    new_rate_tup = rate_tup_remainder + (new_rate,)
                    new_label_tuples_to_process.append(new_label_tup)
                    new_rate_tuples_to_process.append(new_rate_tup)
            else:
                for new_label_rate_tup in aggregated_labels_and_rates_dict.items():
                    fully_processed_label_rate_tuples.append(new_label_rate_tup)
        label_tuples_to_process = new_label_tuples_to_process
        rate_tuples_to_process = new_rate_tuples_to_process  
    
    return fully_processed_label_rate_tuples

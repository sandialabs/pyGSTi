"""
Tools for the propagation of error generators through circuits.
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
import warnings
import gc as _gc
from contextlib import contextmanager as _contextmanager
try:
    import stim
except ImportError:
    msg = "Stim is required for use of the error generator propagation tools module, " \
          "and it does not appear to be installed. If you intend to use this module please update" \
          " your environment."
    warnings.warn(msg)

import numpy as _np
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL, LocalElementaryErrorgenLabel as _LEEL, ElementaryErrorgenLabel as _EEL
from pygsti.baseobjs import QubitSpace as _QubitSpace
from pygsti.baseobjs.basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis as _CompleteElementaryErrorgenBasis, ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE, bel_str as _bel_str, bel_product_str as _bel_product_str, canonicalize_errorgen_layer as _canonicalize_errorgen_layer
import pygsti.errorgenpropagation.errorpropagator as _epropagator
from pygsti.circuits import Circuit as _Circuit
from pygsti.tools.optools import create_elementary_errorgen_nqudit, state_to_dmvec
from functools import wraps as _wraps
from itertools import chain, product
from math import factorial
from typing import Literal, Optional, Union, Callable, Iterable, Iterator, TypeVar, cast as _cast

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

# A list of (error generator label, rate) pairs, as produced by the commutator and
# composition routines below. Rates may be complex prior to aggregation.
_ErrorgenTerms = list[tuple[_LSE, complex]]

# A dictionary of error generator rates keyed by label, as taken and returned by the drivers
# (BCH, Magnus, Zassenhaus, Taylor) for one layer or one order. Their intermediate per-order
# accumulators hold complex rates (`_Rate`) until the final merge takes the real part.
_ErrorgenDict = dict[_LSE, float]
_Rate = TypeVar('_Rate', float, complex)

# A signed Pauli w P as accepted by the term emitters and returned by `pauli_product`, `com`
# and `acom`: `(w, P)` with P an unsigned stim.PauliString and w the phase, or `(w, P, s)`
# with s = bel_str(P) already rendered. See "Signed Paulis" in the module docstring.
_SignedPauli = Union[tuple[complex, stim.PauliString], tuple[complex, stim.PauliString, str]]

# The signature shared by the sixteen `_commutator_XY` and sixteen `_composition_XY`
# handlers: (errorgen_1, errorgen_2, weight, identity string) -> terms.
_PairHandler = Callable[[_LSE, _LSE, complex, str], _ErrorgenTerms]

_F = TypeVar('_F', bound=Callable)

@_contextmanager
def _cyclic_gc_paused() -> Iterator[None]:
    """
    Suspend Python's cyclic garbage collector for the duration of a block, restoring its
    previous state afterwards. The drivers below allocate millions of small, acyclic
    containers (labels, term tuples, dicts); every full collection traverses all of them,
    which costs ~20 % of the run time at 100 qubits while never finding anything to free.
    Reference counting is unaffected, so memory use does not change.
    """
    was_enabled = _gc.isenabled()
    _gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            _gc.enable()


def _with_cyclic_gc_paused(fn: _F) -> _F:
    """Decorator form of `_cyclic_gc_paused` for the driver functions."""
    @_wraps(fn)
    def wrapper(*args, **kwargs):
        with _cyclic_gc_paused():
            return fn(*args, **kwargs)
    return _cast(_F, wrapper)


@_with_cyclic_gc_paused
def bch_approximation(errgen_layer_1: _ErrorgenDict, errgen_layer_2: _ErrorgenDict, bch_order: Literal[1,2,3,4,5] = 1,
                      truncation_threshold: float = 1e-14) -> _ErrorgenDict:
    """
    Apply the BCH approximation at the given order to combine the input dictionaries
    of  error generator rates.

    Parameters
    ----------
    errgen_layer_1 : dict
        Dictionary of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.
    
    errgen_layer_2 : list of dicts
        See errgen_layer_1.

    bch_order : int, optional (default 1)
        Order of the BCH approximation to use. Currently support for up to fifth order.
    
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.

    Returns
    -------
    combined_errgen_layer : dict
        A dictionary with the same general structure as `errgen_layer_1` and `errgen_layer_2`, but with the
        rates combined according to the selected order of the BCH approximation.

    """
    new_errorgen_layer=[]
    for curr_order in range(0, bch_order):
        # add first order terms into new layer
        if curr_order == 0:
            # Get a combined set of error generator coefficient labels for these two
            # dictionaries.
            current_combined_coeff_lbls = {key: None for key in chain(errgen_layer_1, errgen_layer_2)}            

            first_order_dict = dict()
            # loop through the combined set of coefficient labels and add them to the new dictionary for the current BCH
            # approximation order. If present in both we sum the rates.
            for coeff_lbl in current_combined_coeff_lbls:
                # only add to the first order dictionary if the coefficient exceeds the truncation threshold.
                first_order_rate = errgen_layer_1.get(coeff_lbl, 0) + errgen_layer_2.get(coeff_lbl, 0)
                if abs(first_order_rate) > truncation_threshold:
                    first_order_dict[coeff_lbl] = first_order_rate
            
            # allow short circuiting to avoid an expensive bunch of recombination logic when only using first order BCH
            # which will likely be a common use case.
            if bch_order==1:
                return first_order_dict
            new_errorgen_layer.append(first_order_dict)
        
        # second order BCH terms.
        #  (1/2)*[X,Y]
        elif curr_order == 1:
            # calculate the pairwise commutators between each of the error generators in current_errgen_dict_1 and
            # current_errgen_dict_2.
            # precompute the all-identity Pauli string for comparisons in commutator calculations.
            identity = 'I'*len(next(iter(errgen_layer_1)).basis_element_labels[0]) if errgen_layer_1 else None
            second_order_comm_dict = {}
            _accumulate_layer_pairwise_commutators(second_order_comm_dict, errgen_layer_1, errgen_layer_2, identity,
                                                   addl_weight=0.5, truncation_threshold=truncation_threshold)
            # truncate any terms which are below the truncation threshold following aggregation.
            second_order_comm_dict = _truncated(second_order_comm_dict, truncation_threshold)

            new_errorgen_layer.append(second_order_comm_dict)

        # third order BCH terms
        #  (1/12)*([X,[X,Y]] - [Y,[X,Y]])
        # TODO: Can make this more efficient by using linearity of commutators
        elif curr_order == 2:
            # we've already calculated (1/2)*[X,Y] in the previous order, so reuse this result.
            # two different lists for the two different commutators so that we can more easily reuse
            # this at higher order if needed.
            # kept as two separate dictionaries (untruncated) because the fourth- and fifth-order terms reuse them.
            third_order_comm_dict_1 = {}
            third_order_comm_dict_2 = {}
            # only a factor of 1/6 (resp. -1/6) is needed because second_order_comm_dict is 1/2 the commutator.
            _accumulate_layer_pairwise_commutators(third_order_comm_dict_1, errgen_layer_1, second_order_comm_dict, identity,
                                                   addl_weight=(1/6), truncation_threshold=truncation_threshold)
            _accumulate_layer_pairwise_commutators(third_order_comm_dict_2, errgen_layer_2, second_order_comm_dict, identity,
                                                   addl_weight=-(1/6), truncation_threshold=truncation_threshold)
            # finally sum these two dictionaries, keeping only terms which are greater than the threshold.
            third_order_comm_dict = third_order_comm_dict_1.copy()
            for lbl, rate in third_order_comm_dict_2.items():
                third_order_comm_dict[lbl] = third_order_comm_dict.get(lbl, 0) + rate
            third_order_comm_dict = _truncated(third_order_comm_dict, truncation_threshold)
            new_errorgen_layer.append(third_order_comm_dict)
                         
        # fourth order BCH terms
        #  -(1/24)*[Y,[X,[X,Y]]]
        elif curr_order == 3:
            # we've already calculated (1/12)*[X,[X,Y]] so reuse this result.
            # this is stored in third_order_comm_dict_1
            # only need a factor of -1/2 because third_order_comm_dict_1 is 1/12 the nested commutator
            fourth_order_comm_dict = {}
            _accumulate_layer_pairwise_commutators(fourth_order_comm_dict, errgen_layer_2, third_order_comm_dict_1, identity,
                                                   addl_weight=-0.5, truncation_threshold=truncation_threshold)
            # drop any terms below the truncation threshold after aggregation
            fourth_order_comm_dict = _truncated(fourth_order_comm_dict, truncation_threshold)
            new_errorgen_layer.append(fourth_order_comm_dict)

        # Note for fifth order and beyond we can save a bunch of commutators
        # by using the results of https://doi.org/10.1016/j.laa.2003.09.010
        # Revisit this if going up to high-order ever becomes a regular computation.
        # fifth-order BCH terms:
        # -(1/720)*([X,F] - [Y, E]) + (1/360)*([Y,F] - [X,E]) + (1/120)*([Y,G] - [X,D])
        #  Where: E = [Y,C]; F = [X,B]; G=[X,C]
        #  B = [X,[X,Y]]; C = [Y,[X,Y]]; D = [Y,[X,[X,Y]]]
        #  B, C and D have all been previously calculated (up to the leading constant). 
        #  B is proportional to third_order_comm_dict_1, C is proportional to third_order_comm_dict_2
        #  D is proportional to fourth_order_comm_dict
        #  This gives 9 new commutators to calculate (7 if you used linearity, and even fewer would be needed
        #  using the result from the paper above, but we won't here atm).
        elif curr_order == 4:
            B = third_order_comm_dict_1 # has a factor of 1/12 folded in already.
            C = third_order_comm_dict_2 # has a factor of -1/12 folded in already.
            D = fourth_order_comm_dict  # has a factor of -1/24 folded in already.
            # Compute the new commutators E, F and G as defined above (no weight adjustments at this
            # stage; they are applied in the next round), each truncated after aggregation.
            E_comm_dict, F_comm_dict, G_comm_dict = {}, {}, {}
            _accumulate_layer_pairwise_commutators(E_comm_dict, errgen_layer_2, C, identity, truncation_threshold=truncation_threshold)
            _accumulate_layer_pairwise_commutators(F_comm_dict, errgen_layer_1, B, identity, truncation_threshold=truncation_threshold)
            _accumulate_layer_pairwise_commutators(G_comm_dict, errgen_layer_1, C, identity, truncation_threshold=truncation_threshold)
            E_comm_dict = _truncated(E_comm_dict, truncation_threshold)
            F_comm_dict = _truncated(F_comm_dict, truncation_threshold)
            G_comm_dict = _truncated(G_comm_dict, truncation_threshold)
            # -(1/720)*([X,F] - [Y, E]) + (1/360)*([Y,F] - [X,E]) + (1/120)*([Y,G] - [X,D])
            # Now do the next round of 6 commutators: [X,F], [Y,E], [Y,F], [X,E], [Y,G] and [X,D], all
            # accumulated into the fifth-order term. We also need the following weight factors. F has a
            # leading factor of (1/12); E and G have a leading factor of (-1/12); D has a leading factor
            # of (-1/24). This gives the following additional weight multipliers:
            # [X,F] = (-1/60); [Y,E] = (-1/60); [Y,F]= (1/30); [X,E]= (1/30); [Y,G] = (-1/10); [X,D] = (1/5)
            fifth_order_comm_dict = {}
            for layer, comm_dict, addl_weight in [(errgen_layer_1, F_comm_dict, -(1/60)), (errgen_layer_2, E_comm_dict, -(1/60)),
                                                  (errgen_layer_2, F_comm_dict, (1/30)),  (errgen_layer_1, E_comm_dict, (1/30)),
                                                  (errgen_layer_2, G_comm_dict, -0.1),    (errgen_layer_1, D, 0.2)]:
                _accumulate_layer_pairwise_commutators(fifth_order_comm_dict, layer, comm_dict, identity,
                                                       addl_weight=addl_weight, truncation_threshold=truncation_threshold)
            # keep only terms which are greater than the threshold.
            fifth_order_comm_dict = _truncated(fifth_order_comm_dict, truncation_threshold)
            new_errorgen_layer.append(fifth_order_comm_dict)

        else:
            raise NotImplementedError("Higher orders beyond fifth order are not implemented yet.")

    # Finally accumulate all of the dictionaries in new_errorgen_layer into a single one, summing overlapping terms.
    # The per-order dictionaries are each truncated, but their sums can still fall below the threshold (or cancel
    # exactly) when the orders combine, so the merged result is truncated once more.
    new_errorgen_layer_dict = {}
    get = new_errorgen_layer_dict.get
    for order_dict in new_errorgen_layer:
        for lbl, rate in order_dict.items():
            new_errorgen_layer_dict[lbl] = get(lbl, 0) + rate.real

    return _truncated(new_errorgen_layer_dict, truncation_threshold)

@_with_cyclic_gc_paused
def magnus_expansion(errorgen_layers: list[_ErrorgenDict], magnus_order: Literal[1,2,3] = 1,
                     truncation_threshold: float = 1e-14) -> _ErrorgenDict:
    """
    Function for computing the nth-order magnus expansion for a set of error generator layers.
    Please see https://arxiv.org/abs/0810.5488 or https://en.wikipedia.org/wiki/Magnus_expansion
    for more information on this approximation.

    Parameters
    ----------
    errorgen_layers : list of dicts
        List of dictionaries of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.
       
    magnus_order : int, optional (default 1)
        Order of the magnus expansion to apply. Currently supports up to third order.
    
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.
        
    Returns
    -------
    magnus_expansion_dict : dict
        A dictionary with the same general structure as those in `errorgen_layers`, but with the
        rates combined according to the selected order of the magnus expansion.
    """

    if magnus_order > 3:
        raise NotImplementedError("Magnus expansions beyond third order are not implemented yet.")
    if magnus_order < 1:
        raise ValueError(f'Invalid magnus_order <1: {magnus_order}')

    # first-order magnus term: \sum_{t1} A(t1)
    # Get a combined set of error generator coefficient labels for the list of dictionaries.
    current_combined_coeff_lbls = {key: None for key in chain(*errorgen_layers)}

    first_order_dict = dict()
    # loop through the combined set of coefficient labels and add them to the new dictionary for the current BCH
    # approximation order. If present in both we sum the rates.
    for coeff_lbl in current_combined_coeff_lbls:
        # only add to the first order dictionary if the coefficient exceeds the truncation threshold.
        first_order_rate = sum([errgen_layer.get(coeff_lbl, 0) for errgen_layer in errorgen_layers])
        if abs(first_order_rate) > truncation_threshold:
            first_order_dict[coeff_lbl] = first_order_rate

    # allow short circuiting to avoid an expensive bunch of recombination logic when only using first order BCH
    # which will likely be a common use case.
    if magnus_order == 1:
        return first_order_dict
    new_errorgen_layer = [first_order_dict]

    # precompute an identity string for comparisons in commutator calculations (None if every layer is empty,
    # in which case no commutators are computed).
    identity = None
    for layer in errorgen_layers:
        if layer:
            identity = 'I'*len(next(iter(layer)).basis_element_labels[0])
            break

    if magnus_order == 2:
        # second-order magnus term: (1/2)\sum_{t1=1}^n \sum_{t2=1}^{t1-1} [A(t1), A(t2)]
        new_errorgen_layer.append(_second_order_magnus_term(errorgen_layers, identity, truncation_threshold))
    else:
        # the third-order term is built from the same pairwise layer commutators as the second-order one,
        # so both are produced by a single pass over the layer pairs.
        new_errorgen_layer.extend(_second_and_third_order_magnus_terms(errorgen_layers, identity, truncation_threshold))

    # Finally accumulate all of the dictionaries in new_errorgen_layer into a single one, summing overlapping terms.
    # The per-order dictionaries are each truncated, but their sums can still fall below the threshold (or cancel
    # exactly) when the orders combine, so the merged result is truncated once more.
    new_errorgen_layer_dict = {}
    get = new_errorgen_layer_dict.get
    for order_dict in new_errorgen_layer:
        for lbl, rate in order_dict.items():
            new_errorgen_layer_dict[lbl] = get(lbl, 0) + rate.real

    return _truncated(new_errorgen_layer_dict, truncation_threshold)

def _second_order_magnus_term(errorgen_layers: list[_ErrorgenDict], identity: Optional[str],
                              truncation_threshold: float = 1e-14) -> _ErrorgenDict:
    r"""
    Helper function for computing the second-order correction term in the
    magnus expansion.

    (1/2)\sum_{t1=1}^n \sum_{t2=1}^{t1-1} [A(t1), A(t2)]

    Parameters:
    ----------
    errorgen_layers : list of dicts
        List of dictionaries of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.

    identity : str, optional (default None)
        The all-identity Pauli string `'I'*n` for the number of qubits n, used to detect
        identity indices in the commutator calculations. Built if not given.
        
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.

    Returns
    -------
    second_order_comm_dict : dict
        A dictionary with the same general structure as those in `errorgen_layers`, but with the
        rates combined according to the second order of the magnus expansion.
    """
    errorgen_pairs = []
    for i in range(len(errorgen_layers)):
        for j in range(i):
            errorgen_pairs.append((errorgen_layers[i], errorgen_layers[j]))
    
    # precompute an identity string for comparisons in commutator calculations if one is not provided.
    if identity is None and errorgen_layers:
        for layer in errorgen_layers:
            if layer:
                identity = 'I'*len(next(iter(layer)).basis_element_labels[0])
                break
    
    # accumulate the second-order correction, (1/2)[A(t1), A(t2)] over all layer pairs, label by label.
    second_order_comm_dict = {}
    for errorgen_layer_1, errorgen_layer_2 in errorgen_pairs:
        _accumulate_layer_pairwise_commutators(second_order_comm_dict, errorgen_layer_1, errorgen_layer_2, identity,
                                               addl_weight=0.5, truncation_threshold=truncation_threshold)
    # truncate any terms which are below the truncation threshold following aggregation.
    second_order_comm_dict = _truncated(second_order_comm_dict, truncation_threshold)
    return second_order_comm_dict

def _second_and_third_order_magnus_terms(errorgen_layers: list[_ErrorgenDict], identity: Optional[str],
                                         truncation_threshold: float = 1e-14) -> tuple[_ErrorgenDict, _ErrorgenDict]:
    r"""
    Helper function for computing the second- and third-order correction terms in the
    magnus expansion together, from a single pass over the pairwise layer commutators.

    Omega_2 = (1/2)\sum_{t1=1}^n \sum_{t2=1}^{t1-1} [A(t1), A(t2)]
    Omega_3 = (1/6)\sum_{t1=1}^n \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} ( [A(t1), [A(t2), A(t3)]] - [A(t3), [A(t1), A(t2)]] )

    where terms of Omega_3 with coinciding indices carry an extra factor of 1/2 from the
    discretization of the time-ordered integral.

    Parameters:
    ----------
    errorgen_layers : list of dicts
        List of dictionaries of the error generator coefficients and rates for a circuit layer.
        The error generator coefficients are represented using LocalStimErrorgenLabel.

    identity : str, optional (default None)
        The all-identity Pauli string `'I'*n` for the number of qubits n, used to detect
        identity indices in the commutator calculations. May be None only if every layer is
        empty.

    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.

    Returns
    -------
    second_order_comm_dict, third_order_comm_dict : dict
        Dictionaries with the same general structure as those in `errorgen_layers`, holding the
        second- and third-order terms of the magnus expansion respectively.
    """
    # Write P_ik = [A(i), A(k)] for i > k for the pairwise layer commutators and group them by
    # their larger or their smaller index:
    #
    #   row_i = \sum_{k<i} P_ik          col_k = \sum_{i>k} P_ik
    #
    # The commutators with coinciding indices vanish, so Omega_2 = (1/2) \sum_i row_i. In Omega_3,
    # using linearity, the first term is a sum over t1 of A(t1) commuted with a PREFIX of the rows
    # and (after reordering the sums so that t3 is outermost) the second is a sum over t3 of A(t3)
    # commuted with a SUFFIX of the columns, the coinciding-index row/column entering at half weight:
    #
    #   Omega_3 = (1/6) \sum_i [ A(i),  \sum_{j<i} row_j + (1/2) row_i ]
    #           - (1/6) \sum_k [ A(k),  \sum_{j>k} col_j + (1/2) col_k ]
    #
    # Every P_ik lies in exactly one row and one column, so a single forward pass over i computes
    # each P_ik once and files it into both. row_i is consumed on the spot (second-order term,
    # running prefix sum and the outer commutator with A(i)); col_k is complete only when the
    # pass ends (its last contribution comes from i = n-1), so the columns are held until then
    # and consumed by a descending pass over k. The running sums are truncated at the threshold
    # before each outer commutator; the held columns are small (the disjoint-support skip in
    # `_accumulate_layer_pairwise_commutators` removes the pairs that would otherwise dominate
    # them) and are truncated when consumed.
    num_layers = len(errorgen_layers)
    second_order_comm_dict = {}
    third_order_comm_dict = {}
    held_columns = [{} for _ in range(num_layers)]

    # forward pass: rows, consumed immediately.
    running_row_sum = {}  # \sum_{j<i} row_j at weight 1/6, plus (1/12) row_i during the outer commutator.
    for i in range(num_layers):
        row = {}
        for k in range(i):
            pairwise_comm = {}
            _accumulate_layer_pairwise_commutators(pairwise_comm, errorgen_layers[i], errorgen_layers[k], identity,
                                                   truncation_threshold=truncation_threshold)
            column = held_columns[k]
            for lbl, rate in pairwise_comm.items():
                row[lbl] = row.get(lbl, 0) + rate
                column[lbl] = column.get(lbl, 0) + rate
        for lbl, rate in row.items():
            second_order_comm_dict[lbl] = second_order_comm_dict.get(lbl, 0) + 0.5*rate
            running_row_sum[lbl] = running_row_sum.get(lbl, 0) + (1/12)*rate
        _accumulate_layer_pairwise_commutators(third_order_comm_dict, errorgen_layers[i],
                                               _truncated(running_row_sum, truncation_threshold), identity,
                                               truncation_threshold=truncation_threshold)
        # bring row_i up to the full weight of 1/6 for the later iterations.
        for lbl, rate in row.items():
            running_row_sum[lbl] += (1/12)*rate
        running_row_sum = _truncated(running_row_sum, truncation_threshold)

    # descending pass: columns (col_{n-1} is empty).
    running_column_sum = {}  # -\sum_{j>k} col_j at weight 1/6, plus -(1/12) col_k during the outer commutator.
    for k in range(num_layers-2, -1, -1):
        column = _truncated(held_columns[k], truncation_threshold)
        held_columns[k] = None
        for lbl, rate in column.items():
            running_column_sum[lbl] = running_column_sum.get(lbl, 0) - (1/12)*rate
        _accumulate_layer_pairwise_commutators(third_order_comm_dict, errorgen_layers[k],
                                               _truncated(running_column_sum, truncation_threshold), identity,
                                               truncation_threshold=truncation_threshold)
        for lbl, rate in column.items():
            running_column_sum[lbl] -= (1/12)*rate
        running_column_sum = _truncated(running_column_sum, truncation_threshold)

    return (_truncated(second_order_comm_dict, truncation_threshold),
            _truncated(third_order_comm_dict, truncation_threshold))

@_with_cyclic_gc_paused
def zassenhaus_formula(errorgen_groups: list[_ErrorgenDict], zassenhaus_order: Literal[1,2] = 1,
                      truncation_threshold: float = 1e-14) -> list[_ErrorgenDict]:
    r"""
    Function for computing the nth-order Zassenhaus formula for a set of error generators.
    Please see https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula#Zassenhaus_formula
    for more information on this approximation.

    Given an exponentiated sum of operators exp(X1+X2+...+Xn) the Zassenhaus formula allows one to disentangle this
    exponentiated sum into a product of exponentiated operators given by exp(X1)exp(X2)...exp(Xn)\prod_{k=2}^\infty exp(W_k) where
    the W_k's are Lie polynomials (nested commutators) in the operators {X1, ..., Xn}, and the value of k we go up to gives the order of the
    approximation. 

    errorgen_groups : list of dicts
        List of dictionaries of the error generator coefficients and rates for a group of error generators corresponding
        to each of the operators in the sum to perform Zassenhaus with respect to. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.
    
    zassenhaus_order : int, optional (default 1)
        Order of the Zassenahaus formula to compute. Currently supports up to second order. Note that
        zassenhaus_order = 1 corresponds simply to the original list of error generator in
        errorgen_groups, and in this case we simply return errorgen_groups as-is (not a copy).
    
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.

    Returns
    -------
    zassenhaus_formula_dicts : list of dicts
        A list of dictionaries, each corresponding to one of the operators which is exponentiated in 
        the product as output by the Zassenhaus formula.
    """
    zassenhaus_formula_dicts = []
    
    # first-order zassenhaus terms are just the original list of error generators in errorgen_groups.
    if zassenhaus_order >= 1:
        # allow short-circuiting for zassenhaus_order==1.
        if zassenhaus_order==1:
            return errorgen_groups
        else:
            zassenhaus_formula_dicts.extend(errorgen_groups)

    # second-order zassenhaus term: (1/2)\sum_{1<=i<j<=n}[X_j,X_i]
    # this is identical to the second order magnus term, so reuse that code.
    if zassenhaus_order>=2:
        # precompute an identity string for comparisons in commutator calculations.
        if errorgen_groups:
            for layer in errorgen_groups:
                if layer:
                    identity = 'I'*len(next(iter(layer)).basis_element_labels[0])
                    break
        second_order_comm_dict = _second_order_magnus_term(errorgen_groups, identity, truncation_threshold)
        zassenhaus_formula_dicts.append(second_order_comm_dict)

    if zassenhaus_order>=3:
        raise NotImplementedError('The Zassenhaus formula is currently only implemented up to second-order.')  
         
    return zassenhaus_formula_dicts

# TODO: Refactor a bunch of the code in this module to use this helper function.
# define a helper function to do a layerwise commutator accumulating all of the pairwise terms into a single list.
def _accumulate_layer_pairwise_commutators(target: dict[_LSE, complex], errorgen_layer_1: dict[_LSE, _Rate],
                                           errorgen_layer_2: dict[_LSE, _Rate], identity: Optional[str],
                                           addl_weight: float = 1.0, truncation_threshold: float = 1e-14) -> None:
    """
    Add addl_weight * rate_1 * rate_2 * [e1, e2] to `target`, a dict of label -> rate, for every
    e1 in `errorgen_layer_1` and e2 in `errorgen_layer_2` (dicts of label -> rate). Terms are
    accumulated as they are produced: the same label typically arises from many pairs, and
    aggregating on the fly keeps one label object per distinct key instead of a list of every
    term. `target` is not truncated; callers
    apply their threshold after all contributions are in. `identity` is the 'I'*n string
    (callers pass None only when `errorgen_layer_1` is empty, in which case nothing is computed).

    Pairs of labels with disjoint supports act on different tensor factors and so commute
    exactly; they are skipped here, before calling `error_generator_commutator` (which does
    not test for this itself and would emit terms that only cancel after aggregation). At
    large qubit counts these are the vast majority of pairs and, previously, the source of
    most of the entries of `target`. The support bitmasks of the inner layer are gathered
    once instead of being re-read for each label of the outer layer. No analogous shortcut
    exists for compositions: the composition of two error generators on disjoint supports is
    a genuine (non-zero) term.
    """
    get = target.get
    layer_2_items = [(error2, error2_val, error2.support_mask) for error2, error2_val in errorgen_layer_2.items()]
    for error1, error1_val in errorgen_layer_1.items():
        mask1 = error1.support_mask
        for error2, error2_val, mask2 in layer_2_items:
            # disjoint supports: the commutator is identically zero.
            if mask1 & mask2 == 0:
                continue
            weight = addl_weight*error1_val*error2_val
            # avoid computing commutators which will be effectively zero.
            if abs(weight) < truncation_threshold:
                continue
            for lbl, rate in error_generator_commutator(error1, error2, weight=weight, identity=identity):
                target[lbl] = get(lbl, 0) + rate


def _truncated(errorgen_dict: dict[_LSE, _Rate], truncation_threshold: float) -> dict[_LSE, _Rate]:
    """The entries of `errorgen_dict` (label -> rate) whose rate exceeds `truncation_threshold` in magnitude."""
    return {lbl: rate for lbl, rate in errorgen_dict.items() if abs(rate) > truncation_threshold}


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

def errorgen_layer_to_matrix(errorgen_layer: Union[list[tuple[_EEL, float]], tuple[tuple[_EEL, float], ...], dict[_EEL, float]],
                             num_qubits: int, errorgen_matrix_dict: Optional[dict[_EEL, _np.ndarray]] = None,
                             sslbls: Optional[Union[list, tuple]] = None) -> _np.ndarray:
    """
    Converts an iterable over error generator coefficients and rates into the corresponding
    dense numpy array representation.

    Parameters
    ----------
    errorgen_layer : list, tuple or dict
        An iterable over error generator coefficient and rates. If a list or a tuple the
        elements should correspond to two-element tuples, the first value being an `ElementaryErrorgenLabel`
        and the second value the rate. If a dictionary the keys should be `ElementaryErrorgenLabel` and the
        values the rates.

    num_qubits : int
        Number of qubits for the error generator matrix being constructed.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.
    
    sslbls : list or tuple, optional (default None)
        A tuple or list of state space labels corresponding to the qubits upon which the error generators
        can supported. Only required when passing in a value of `errorgen_matrix_dict` with
        `GlobalElementaryErrogenLabel` keys in conjunction with an `errorgen_layer` with labels
        which are `LocalElementaryErrorgenLabel` (or vice-versa).
    
    Returns
    -------
    errorgen_mat : ndarray 
        ndarray for the dense representation of the specified error generator in the standard basis.
    """

    # if the list is empty return all zeros
    # initialize empty array for accumulation.
    mat = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
    if not errorgen_layer:
        return mat
    
    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    # infer the correct label type.
    if errorgen_matrix_dict:
        first_label = next(iter(errorgen_matrix_dict))
        if isinstance(first_label, _LEEL):
            label_type = 'local'
        elif isinstance(first_label, _GEEL):
            label_type = 'global'
        else:
            msg = f'Label type {type(first_label)} is not supported as a key for errorgen_matrix_dict.'\
                  + 'Please use either LocalElementaryErrorgenLabel or GlobalElementaryErrorgenLabel.'
            raise ValueError()
    else:
        raise ValueError('Non-empty errorgen_layer, but errorgen_matrix_dict is empty. Cannot convert.')
        
    # loop through errorgen_layer and accumulate the weighted error generators prescribed.
    if isinstance(errorgen_layer, (list, tuple)):
        first_coefficient_lbl = errorgen_layer[0][0]
        errorgen_layer_iter = errorgen_layer
    elif isinstance(errorgen_layer, dict):
        first_coefficient_lbl = next(iter(errorgen_layer))
        errorgen_layer_iter = errorgen_layer.items()
    else:
        raise ValueError(f'errorgen_layer should be either a list, tuple or dict. {type(errorgen_layer)=}')

    if ((isinstance(first_coefficient_lbl, _LEEL) and label_type == 'global') \
        or (isinstance(first_coefficient_lbl, _GEEL) and label_type == 'local')) and sslbls is None:
        msg = "You have passed in an `errogen_layer` with `LocalElementaryErrorgenLabel` coefficients, and " \
              +"an `errorgen_matrix_dict` with keys which are `GlobalElementaryErrorgenLabel` (or vice-versa). When using this "\
              +"combination you must also specify the state space labels with `sslbls`."
        raise ValueError(msg)

    if isinstance(first_coefficient_lbl, _LSE):
        if label_type == 'local':
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl.to_local_eel()]
        else:
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl.to_global_eel()]
    elif isinstance(first_coefficient_lbl, _LEEL):
        if label_type == 'local':
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl]
        else:
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[_GEEL.cast(lbl, sslbls=sslbls)]
    elif isinstance(first_coefficient_lbl, _GEEL):
        if label_type == 'local':
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[_LEEL.cast(lbl, sslbls=sslbls)]
        else:
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl]
    else:
        raise ValueError('The coefficient labels in `errorgen_layer` should be either `LocalStimErrorgenLabel`, `LocalElementaryErrorgenLabel` or `GlobalElementaryErrorgenLabel`.')
    
    return mat

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

# Helper functions for doing numeric commutators, compositions and BCH.

def error_generator_commutator_numerical(errorgen1: Union[_LEEL, _LSE], errorgen2: Union[_LEEL, _LSE],
                                         errorgen_matrix_dict: Optional[dict[Union[_LEEL, _LSE], _np.ndarray]] = None,
                                         num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Numerically compute the commutator of the two specified elementary error generators.

    Parameters
    ----------
    errorgen1 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `ElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        Second error generator.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator commutator being computed. Only required if `errorgen_matrix_dict` is None.
    
    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of the commutator of the input error generators in the standard basis.
    """

    assert isinstance(errorgen1, (_LEEL, _LSE)) and isinstance(errorgen2, (_LEEL, _LSE))
    assert type(errorgen1) == type(errorgen2), "The elementary error generator labels have mismatched types."
    
    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))
    
    if isinstance(first_label, _LEEL):
        if isinstance(errorgen1, _LEEL):
            comm = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2] - errorgen_matrix_dict[errorgen2]@errorgen_matrix_dict[errorgen1]
        else:
            comm = errorgen_matrix_dict[errorgen1.to_local_eel()]@errorgen_matrix_dict[errorgen2.to_local_eel()]\
                  - errorgen_matrix_dict[errorgen2.to_local_eel()]@errorgen_matrix_dict[errorgen1.to_local_eel()]
    else:
        if isinstance(errorgen1, _LSE):
            comm = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2] - errorgen_matrix_dict[errorgen2]@errorgen_matrix_dict[errorgen1]
        else:
            comm = errorgen_matrix_dict[_LSE.cast(errorgen1)]@errorgen_matrix_dict[_LSE.cast(errorgen2)]\
                  - errorgen_matrix_dict[_LSE.cast(errorgen2)]@errorgen_matrix_dict[_LSE.cast(errorgen1)]
    return comm

def error_generator_composition_numerical(errorgen1: Union[_LEEL, _LSE], errorgen2: Union[_LEEL, _LSE],
                                          errorgen_matrix_dict: Optional[dict[Union[_LEEL, _LSE], _np.ndarray]] = None,
                                          num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Numerically compute the composition of the two specified elementary error generators.

    Parameters
    ----------
    errorgen1 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        Second error generator.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator commutator being computed. Only required if `errorgen_matrix_dict` is None.
    
    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of the composition of the input error generators in the standard basis.
        
    """
    assert isinstance(errorgen1, (_LEEL, _LSE)) and isinstance(errorgen2, (_LEEL, _LSE))
    assert type(errorgen1) == type(errorgen2), "The elementary error generator labels have mismatched types."
    
    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))
    
    if isinstance(first_label, _LEEL):
        if isinstance(errorgen1, _LEEL):
            comp = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2]
        else:
            comp = errorgen_matrix_dict[errorgen1.to_local_eel()]@errorgen_matrix_dict[errorgen2.to_local_eel()]
    else:
        if isinstance(errorgen1, _LSE):
            comp = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2]
        else:
            comp = errorgen_matrix_dict[_LSE.cast(errorgen1)]@errorgen_matrix_dict[_LSE.cast(errorgen2)]
    return comp


def error_generator_anticommutator_numerical(errorgen1: Union[_LEEL, _LSE], errorgen2: Union[_LEEL,_LSE],
                                             errorgen_matrix_dict: Optional[dict[Union[_LEEL, _LSE], _np.ndarray]] = None,
                                             num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Numerically compute the anticommutator of the two specified elementary error generators.

    Parameters
    ----------
    errorgen1 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        Second error generator.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator anticommutator being computed. Only required if `errorgen_matrix_dict` is None.

    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of the anticommutator of the input error generators in the standard basis.
    """
    assert isinstance(errorgen1, (_LEEL, _LSE)) and isinstance(errorgen2, (_LEEL, _LSE))
    assert type(errorgen1) == type(errorgen2), "The elementary error generator labels have mismatched types."

    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')

        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))

    if isinstance(first_label, _LEEL):
        key1, key2 = (errorgen1, errorgen2) if isinstance(errorgen1, _LEEL) else (errorgen1.to_local_eel(), errorgen2.to_local_eel())
    else:
        key1, key2 = (errorgen1, errorgen2) if isinstance(errorgen1, _LSE) else (_LSE.cast(errorgen1), _LSE.cast(errorgen2))
    mat1, mat2 = errorgen_matrix_dict[key1], errorgen_matrix_dict[key2]
    return mat1@mat2 + mat2@mat1


def pauli_conjugation_composition_numerical(pauli: Union[str, stim.PauliString], errorgen: Union[_LEEL, _LSE],
                                            errorgen_matrix_dict: Optional[dict[Union[_LEEL, _LSE], _np.ndarray]] = None,
                                            num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Numerically compute the composition of the Pauli conjugation superoperator rho -> Q rho Q
    with the specified elementary error generator, i.e. the dense representation of
    Q[errorgen[.]]. The conjugation is formed as the dense stochastic generator plus the
    identity, Q = S_Q + 1.

    Parameters
    ----------
    pauli : str or stim.PauliString
        The conjugating Pauli Q, as an 'I'-padded string such as 'XIZ' or a `stim.PauliString`
        (its sign is ignored). Must not be the identity.

    errorgen : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        The error generator being conjugated.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits. Only required if `errorgen_matrix_dict` is None.

    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of Q[errorgen[.]] in the standard basis.
    """
    assert isinstance(errorgen, (_LEEL, _LSE))
    pauli_str = _bel_str(pauli) if isinstance(pauli, stim.PauliString) else pauli
    assert set(pauli_str) != {'I'}, "The conjugating Pauli must not be the identity."

    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')

        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))

    if isinstance(first_label, _LEEL):
        s_key = _LEEL('S', (pauli_str,))
        x_key = errorgen if isinstance(errorgen, _LEEL) else errorgen.to_local_eel()
    else:
        s_key = _LSE('S', (stim.PauliString(pauli_str),))
        x_key = errorgen if isinstance(errorgen, _LSE) else _LSE.cast(errorgen)
    s_mat = errorgen_matrix_dict[s_key]
    conjugation = s_mat + _np.eye(s_mat.shape[0])
    return conjugation@errorgen_matrix_dict[x_key]


def bch_numerical(propagated_errorgen_layers: list[_np.ndarray],
                  error_propagator: _epropagator.ErrorGeneratorPropagator,
                  bch_order: int = 1) -> _np.ndarray:
    """
    Iteratively compute effective error generator layer produced by applying the BCH approximation
    to the list of input error generator matrices. Note this is primarily intended
    as part of testing and validation infrastructure.

    Parameters
    ----------
    propagated_errorgen_layers : list of numpy.ndarrays
        List of the error generator layers to combine using the BCH approximation (in circuit ordering)

    error_propagator : `ErrorGeneratorPropagator`
        An `ErrorGeneratorPropagator` instance to use as part of the BCH calculation.

    bch_order : int, optional (default 1)
        Order of the BCH approximation to apply (up to 5 is supported currently).

    Returns
    -------
    numpy.ndarray
        A dense numpy array corresponding to the result of the iterative application of the BCH
        approximation.
    """
    # Need to build an appropriate basis for getting the error generator matrices.
    # accumulate the error generator coefficients needed.
    collected_coeffs = []
    for layer in propagated_errorgen_layers:
        for coeff in layer.keys():
            collected_coeffs.append(coeff.to_local_eel())
    # only want the unique ones.
    unique_coeffs = list(set(collected_coeffs))
    
    num_qubits = len(error_propagator.model.state_space.qubit_labels)
    
    errorgen_basis = _ExplicitElementaryErrorgenBasis(_QubitSpace(num_qubits), unique_coeffs, basis_1q=_BuiltinBasis('PP', 4))
    errorgen_lbl_matrix_dict = {lbl:mat for lbl,mat in zip(errorgen_basis.labels, errorgen_basis.elemgen_matrices)}
    
    # iterate through each of the propagated error generator layers and turn these into dense numpy arrays
    errorgen_layer_mats = []
    for layer in propagated_errorgen_layers:
        errorgen_layer_mats.append(error_propagator.errorgen_layer_dict_to_errorgen(layer, mx_basis='pp'))
    
    # initialize a matrix for storing the result of doing BCH.
    bch_result = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
    
    if len(errorgen_layer_mats)==1:
        return errorgen_layer_mats[0]
        
    # otherwise iterate through in reverse order (the propagated layers are
    # in circuit ordering and not matrix multiplication ordering at the moment)
    # and combine the terms pairwise
    combined_err_layer = errorgen_layer_mats[-1]
    for i in range(len(errorgen_layer_mats)-2, -1, -1):
        combined_err_layer = pairwise_bch_numerical(combined_err_layer, errorgen_layer_mats[i], order=bch_order)
        
    return combined_err_layer  

def pairwise_bch_numerical(mat1: _np.ndarray, mat2: _np.ndarray, order: Literal[1,2,3,4,5]=1) -> _np.ndarray[_np.complex128]:
    """
    Helper function for doing the numerical BCH in a pairwise fashion. Note this function is primarily intended
    for numerical validations as part of testing infrastructure.
    """
    if not 1 <= order <= 5:
        raise ValueError('BCH order must be between 1 and 5 (inclusive), got %s' % str(order))

    bch_result = _np.zeros(mat1.shape, dtype=_np.complex128)
    if order >= 1:
        bch_result += mat1 + mat2
    if order >= 2:
        commutator12 = _matrix_commutator(mat1, mat2)
        bch_result += 0.5*commutator12
    if order >= 3:
        commutator112 = _matrix_commutator(mat1, commutator12)
        commutator212 = _matrix_commutator(mat2, commutator12)
        bch_result += (1/12)*(commutator112-commutator212)
    if order >= 4:
        commutator2112 = _matrix_commutator(mat2, commutator112)
        bch_result += (-1/24)*commutator2112
    if order == 5:
        commutator1112 = _matrix_commutator(mat1, commutator112)
        commutator2212 = _matrix_commutator(mat2, commutator212)
        
        commutator22212 = _matrix_commutator(mat2, commutator2212)
        commutator11112 = _matrix_commutator(mat1, commutator1112)
        commutator12212 = _matrix_commutator(mat1, commutator2212)
        commutator21112 = _matrix_commutator(mat2, commutator1112)
        commutator21212 = _matrix_commutator(mat2, _matrix_commutator(mat1, commutator212))
        commutator12112 = _matrix_commutator(mat1, commutator2112)
        
        bch_result += (-1/720)*(commutator11112 - commutator22212)
        bch_result += (1/360)*(commutator21112 - commutator12212)
        bch_result += (1/120)*(commutator21212 - commutator12112)
    return bch_result

def magnus_numerical(propagated_errorgen_layers: list[dict[_EEL, float]], error_propagator: _epropagator.ErrorGeneratorPropagator, 
                     magnus_order: Literal[1,2,3] = 1) -> _np.ndarray:
    """
    Compute effective error generator layer produced by applying the magnus expansions
    to the list of input error generator matrices. Note this is primarily intended
    as part of testing and validation infrastructure.

    Parameters
    ----------
    propagated_errorgen_layers : list of dictionaries
        List of the error generator layers (in circuit ordering) in the form of dictionaries
        whose keys are elementary error generator labels and whose values are their corresponding
        rates. These dictionaries are in the format produced by the `ErrorGeneratorPropagator` class's
        `propagate_errorgens` method.

    error_propagator : `ErrorGeneratorPropagator`
        An `ErrorGeneratorPropagator` instance to use as part of the Magnus calculation.

    magnus_order : int, optional (default 1)
        Order of the Magnus expansion to apply (up to 3 is supported currently).

    Returns
    -------
    numpy.ndarray
        A dense numpy array corresponding to the result of Magnus expansion.
    """

    # Need to build an appropriate basis for getting the error generator matrices.
    # accumulate the error generator coefficients needed.
    collected_coeffs = []
    for layer in propagated_errorgen_layers:
        for coeff in layer.keys():
            collected_coeffs.append(coeff.to_local_eel())
    # only want the unique ones.
    unique_coeffs = list(set(collected_coeffs))
    
    num_qubits = len(error_propagator.model.state_space.qubit_labels)
    
    errorgen_basis = _ExplicitElementaryErrorgenBasis(_QubitSpace(num_qubits), unique_coeffs, basis_1q=_BuiltinBasis('PP', 4))
    
    # iterate through each of the propagated error generator layers and turn these into dense numpy arrays
    errorgen_layer_mats = []
    for layer in propagated_errorgen_layers:
        errorgen_layer_mats.append(error_propagator.errorgen_layer_dict_to_errorgen(layer, mx_basis='pp'))
    
    # initialize a matrix for storing the result of doing magnus.
    magnus = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
    
    for curr_order in range(magnus_order):
        # first-order magnus terms:
        # \sum_{t1} A_{t1}
        if curr_order == 0:
            for mat in errorgen_layer_mats:
                magnus += mat
        
        # second-order magnus terms:
        # (1/2) \sum_{t1=1}^n \sum_{t2=1}^{t2} [A(t1), A(t2)]
        elif curr_order == 1:
            errorgen_pairs = []
            for i in range(len(errorgen_layer_mats)):
                for j in range(i):
                    errorgen_pairs.append((errorgen_layer_mats[i], errorgen_layer_mats[j]))
            for errorgen_pair in errorgen_pairs:
                magnus += .5*_matrix_commutator(errorgen_pair[0], errorgen_pair[1])
        
        # third-order magnus terms:
        # (1/6) \sum_{t1=1}^{n} \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} ([A(t1), [A(t2),A(t3)]] + [A(t3), [A(t2), A(t1)]])
        elif curr_order == 2:
            for i in range(len(errorgen_layer_mats)):
                for j in range(i+1):
                    for k in range(j+1):
                        if i==j:
                            magnus += (1/12)*_matrix_commutator(errorgen_layer_mats[i], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[k]))
                        else:
                            magnus += (1/6)*_matrix_commutator(errorgen_layer_mats[i], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[k]))
                        if j==k:
                            magnus += (1/12)*_matrix_commutator(errorgen_layer_mats[k], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[i]))
                        else:
                            magnus += (1/6)*_matrix_commutator(errorgen_layer_mats[k], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[i]))
        else:
            raise NotImplementedError('Magnus beyond third order is not currently implemented.')
        
    return magnus  

def errorgen_pauli_action_numerical(errorgen: _EEL, pauli: stim.PauliString) -> _np.ndarray:
    """
    Apply the specified error generator to a given Pauli operator. This implementation
    performs the application of the error generator numerically and is primarily
    intended for use in testing. 

    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel` or numpy.ndarray
        A label specifying the error generator which should be applied to the specified Pauli operator
        or else a dense numpy array for the error generator (in the standard basis).
    
    pauli : stim.PauliString
        The pauli operator to apply the error generator to.
    
    
    Returns:
    --------
    numpy.ndarray
        Dense representation of the weighted pauli operator resulting from the application
        of the specified error generator to the input pauli.
    """

    # also get the superoperator (in the standard basis) corresponding to the elementary error generator
    if isinstance(errorgen, _LSE):
        local_eel = errorgen.to_local_eel()
    elif isinstance(errorgen, _GEEL):
        local_eel = _LEEL.cast(errorgen)
    else:
        local_eel = errorgen
    
    errgen_type = local_eel.errorgen_type
    basis_element_labels = local_eel.basis_element_labels
    basis_1q = _BuiltinBasis('PP', 4)
    errorgen_superop = create_elementary_errorgen_nqudit(errgen_type, basis_element_labels, basis_1q, normalize=False, sparse=False,
                                                         tensorprod_basis=False)

    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    pauli_unitary_shape = pauli_unitary.shape

    weighted_pauli = (errorgen_superop@pauli_unitary.ravel()).reshape(pauli_unitary_shape)

    return weighted_pauli

def zassenhaus_formula_numerical(errorgen_groups: list[dict[_EEL, float]], error_propagator: _epropagator.ErrorGeneratorPropagator, 
                                 zassenhaus_order: Literal[1,2] = 1) -> list[_np.ndarray]:
    """
    Function for numerically computing the nth-order Zassenhaus formula for a set of error generators.
    Please see https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula#Zassenhaus_formula
    for more information on this approximation.

    Due to the numerical nature of this implementation it is not meant for efficient computation and
    primarily supports testing.

    Parameters
    ----------
    errorgen_groups : list of dicts
        List of dictionaries of the error generator coefficients and rates for a group of error generators corresponding
        to each of the operators in the sum to perform Zassenhaus with respect to. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.
    
    zassenhaus_order : int, optional (default 1)
        Order of the Zassenahaus formula to compute, currently supports up to second order.
    
    Returns
    -------
    zassenhaus_formula_arrays : list of numpy.ndarrays
        A list of numpy arrays, each corresponding to one of the operators which is exponentiated in 
        the product as output by the Zassenhaus formula.    
    """

    # Need to build an appropriate basis for getting the error generator matrices.
    # accumulate the error generator coefficients needed.
    collected_coeffs = []
    for layer in errorgen_groups:
        for coeff in layer.keys():
            collected_coeffs.append(coeff.to_local_eel())
    # only want the unique ones.
    unique_coeffs = list(set(collected_coeffs))
    
    num_qubits = len(error_propagator.model.state_space.qubit_labels)
    
    errorgen_basis = _ExplicitElementaryErrorgenBasis(_QubitSpace(num_qubits), unique_coeffs, basis_1q=_BuiltinBasis('PP', 4))
    
    # iterate through each of the propagated error generator layers and turn these into dense numpy arrays
    errorgen_group_mats = []
    for layer in errorgen_groups:
        errorgen_group_mats.append(error_propagator.errorgen_layer_dict_to_errorgen(layer, mx_basis='pp'))

    zassenhaus_formula_arrays = []

    if zassenhaus_order>=1:
        if zassenhaus_order==1:
            return errorgen_group_mats
        else:
            zassenhaus_formula_arrays.extend(errorgen_group_mats)

    if zassenhaus_order>=2:
        second_order_correction_array = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
        errorgen_pairs = []
        for i in range(len(errorgen_groups)):
            for j in range(i):
                errorgen_pairs.append((errorgen_group_mats[i], errorgen_group_mats[j]))
        for errorgen_pair in errorgen_pairs:
            second_order_correction_array += .5*_matrix_commutator(errorgen_pair[0], errorgen_pair[1])
        zassenhaus_formula_arrays.append(second_order_correction_array)

    if zassenhaus_order>=3:
        raise NotImplementedError('The Zassenhaus formula is currently only implemented up to second-order.')    

    return zassenhaus_formula_arrays
    
def _matrix_commutator(mat1: _np.ndarray, mat2: _np.ndarray) -> _np.ndarray:
    return mat1@mat2 - mat2@mat1

def iterative_error_generator_composition_numerical(errorgen_labels: tuple[_LEEL, ...], rates: tuple[float, ...],
                                                    errorgen_matrix_dict: Optional[dict[_LEEL, _np.ndarray]] = None,
                                                    num_qubits: Optional[int] = None) -> _np.ndarray:
    """
    Iteratively compute error generator compositions. The function computes a dense representation of this composition
    numerically and is primarily intended as part of testing infrastructure.
    
    Parameters
    ----------
    errorgen_labels : tuple of `LocalElementaryErrorgenLabel`s
        A tuple of the elementary error generator labels to be composed.
    
    rates : tuple of float
        A tuple of corresponding error generator rates of the same length as the tuple
        of error generator labels.
        
    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator commutator being computed. Only required if `errorgen_matrix_dict` is None.

    Returns
    -------
    numpy.ndarray
        Dense numpy array representation of the super operator corresponding to the iterated composition written in 
        the standard basis.
    """
    
    if errorgen_matrix_dict is None:
        # create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        # use this basis to construct a dictionary from error generator labels to their
        # matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    composition = errorgen_matrix_dict[errorgen_labels[0]]
    for lbl in errorgen_labels[1:]:
        composition = composition@errorgen_matrix_dict[lbl]
    composition *= _np.prod(rates)
    return composition

# -----------First-Order Approximate Error Generator Probabilities and Expectation Values---------------# 

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

# Courtesy of Gidney 
# https://quantumcomputing.stackexchange.com/questions/38826/how-do-i-efficiently-compute-the-fidelity-between-two-stabilizer-tableau-states
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
    """
    t3 = tableau2**-1 * tableau1
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(t3)
    p = 1
    # note to future selves: stim uses little endian convention by default, and we typically use
    # big endian. That doesn't make a difference in this case, but does elsewhere to be mindful to
    # save on grief.
    for q in range(len(t3)):
        e = sim.peek_z(q)
        if e == -1:
            return 0
        if e == 0:
            p *= 0.5
            sim.postselect_z(q, desired_value=False)
    return p

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

# Modified from Gidney 
# https://quantumcomputing.stackexchange.com/questions/34610/get-the-amplitude-of-a-computational-basis-in-stim
def slow_amplitude_of_state(tableau: Union[stim.Tableau, stim.TableauSimulator], desired_state: str, only_phase: bool) -> complex:
    """
    Get the amplitude of a particular computational basis state for given
    stabilizer state.

    Note: This is a pure python implementation of this function. For best performance see the
    optimized cython implementation tools.fasterrgencalc.fast_amplitude_of_state.

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
    optimized cython implementation tools.fasterrgencalc.fast_bulk_amplitude_of_state.

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
    optimized cython implementation tools.fasterrgencalc.fast_pauli_phase_update.
    
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
    optimized cython implementation tools.fasterrgencalc.fast_pauli_phase_update.
    
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

# helper function for numerically computing phi, primarily used for testing.
def phi_numerical(tableau: stim.Tableau, desired_bitstring: str, P: Union[str, stim.PauliString],
                  Q: Union[str, stim.PauliString]) -> _np.ndarray:
    """
    This function computes a quantity whose value is used in expression for the sensitivity of probabilities to error generators.
    (This version does this calculation numerically and is primarily intended for testing infrastructure.)
    
    Parameters
    ----------
    tableau : stim.Tableau
        A stim Tableau corresponding to the input stabilizer state.
        
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
    initial_pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in desired_bitstring])).to_unitary_matrix(endian = 'big')

    # map P and Q to stim.PauliString if needed.
    if isinstance(P, str):
        P = stim.PauliString(P)
    if isinstance(Q, str):
        Q = stim.PauliString(Q)
    
    stabilizer_state = tableau.to_state_vector(endian = 'big')
    stabilizer_state.reshape((len(stabilizer_state),1))
    # combine this initial pauli string with the two input paulis
    eff_P = initial_pauli_string@P.to_unitary_matrix(endian = 'big')
    eff_Q = Q.to_unitary_matrix(endian = 'big')@initial_pauli_string
    
    # now get the bit strings which need their amplitudes extracted from the input stabilizer state and get
    # the corresponding phase corrections.
    # all_zeros = '0'*len(eff_P)
    all_zeros = _np.zeros((2**len(desired_bitstring),1))
    all_zeros[0] = 1  
    # calculate phi.
    # The second amplitude also needs a complex conjugate applied
    phi = (all_zeros.T@eff_P@stabilizer_state) * (stabilizer_state.conj().T@eff_Q@all_zeros)
    
    num_random = random_support(tableau)
    scale = 2**(num_random)

    return phi*scale

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

def alpha_numerical(errorgen: Union[_LSE, _EEL], tableau: stim.Tableau, desired_bitstring: str) -> float:
    """
    First-order error generator sensitivity function for probability. This implementation calculates
    this quantity numerically, and as such is primarily intended for used as parting of testing
    infrastructure. 
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    desired_bitstring : str
        Bit string to calculate the sensitivity for.
    """
    
    # get the stabilizer state corresponding to the tableau.
    stabilizer_state = tableau.to_state_vector(endian='big')
    stabilizer_state_dmvec = state_to_dmvec(stabilizer_state)
    stabilizer_state_dmvec.reshape((len(stabilizer_state_dmvec),1))
    # also get the superoperator (in the standard basis) corresponding to the elementary error generator
    if isinstance(errorgen, _LSE):
        local_eel = errorgen.to_local_eel()
    elif isinstance(errorgen, _GEEL):
        local_eel = _LEEL.cast(errorgen)
    else:
        local_eel = errorgen
    
    errgen_type = local_eel.errorgen_type
    basis_element_labels = local_eel.basis_element_labels
    basis_1q = _BuiltinBasis('PP', 4)
    errorgen_superop = create_elementary_errorgen_nqudit(errgen_type, basis_element_labels, basis_1q, normalize=False, sparse=False,
                                                         tensorprod_basis=False)
    
    # also need a superbra for the desired bitstring.
    desired_bitstring_vec = _np.zeros(2**len(desired_bitstring))
    desired_bitstring_vec[_bitstring_to_int(desired_bitstring)] = 1
    desired_bitstring_dmvec = state_to_dmvec(desired_bitstring_vec)
    desired_bitstring_dmvec.reshape((1, len(desired_bitstring_dmvec)))
    num_random = random_support(tableau)
    scale = 2**(num_random)
    
    # compute the needed trace inner product.
    alpha = _np.real_if_close(scale*(desired_bitstring_dmvec.conj().T@errorgen_superop@stabilizer_state_dmvec))
    
    return alpha

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

def alpha_pauli_numerical(errorgen: Union[_LSE, _LEEL], tableau: stim.Tableau, pauli: stim.PauliString):
    """
    First-order error generator sensitivity function for pauli expectations. This implementation calculates
    this quantity numerically, and as such is primarily intended for used as parting of testing
    infrastructure. 
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    pauli : stim.PauliString
        Pauli to calculate the sensitivity for.
    """
    
    # get the stabilizer state corresponding to the tableau.
    stabilizer_state = tableau.to_state_vector(endian='big')
    stabilizer_state_dmvec = state_to_dmvec(stabilizer_state)
    stabilizer_state_dmvec.reshape((len(stabilizer_state_dmvec),1))
    # also get the superoperator (in the standard basis) corresponding to the elementary error generator
    if isinstance(errorgen, _LSE):
        local_eel = errorgen.to_local_eel()
    elif isinstance(errorgen, _GEEL):
        local_eel = _LEEL.cast(errorgen)
    else:
        local_eel = errorgen
    
    errgen_type = local_eel.errorgen_type
    basis_element_labels = local_eel.basis_element_labels
    basis_1q = _BuiltinBasis('PP', 4)
    errorgen_superop = create_elementary_errorgen_nqudit(errgen_type, basis_element_labels, basis_1q, normalize=False, sparse=False,
                                                         tensorprod_basis=False)
    
    # finally need the superoperator for the selected pauli.
    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    # flatten this row-wise
    pauli_vec = _np.ravel(pauli_unitary)
    pauli_vec.reshape((len(pauli_vec),1))
    
    # compute the needed trace inner product.
    alpha = _np.real_if_close(pauli_vec.conj().T@errorgen_superop@stabilizer_state_dmvec).item()
    
    return alpha

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

def stabilizer_probability_correction(errorgen_dict: _ErrorgenDict, tableau: stim.Tableau, desired_bitstring: str,
                                      order: int = 1, truncation_threshold: float = 1e-14) -> float:
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

    Returns
    -------
    correction : float
        float corresponding to the correction to the output probability for the
        desired bitstring induced by the error generator (to specified order).
    """    
    num_random = random_support(tableau)
    if num_random > 2148:
        raise RuntimeError('Number of random bits is greater than 1074, magnitude of probability scale will underflow!')
    scale = 1/2**(num_random) 
    
    #accumulate the terms across orders (with short circuit logic for order 1 to save time):
    if order == 1:
        combined_taylor_dict = errorgen_dict
    else:
        #compute the taylor series approximation to the desired order.
        taylor_expansion = error_generator_taylor_expansion(errorgen_dict, order, truncation_threshold)
        # Accumulate all of the dictionaries in taylor expansion into a single one, summing overlapping terms.
        combined_taylor_dict = {}
        get = combined_taylor_dict.get
        for order_dict in taylor_expansion:
            for lbl, rate in order_dict.items():
                combined_taylor_dict[lbl] = get(lbl, 0) + rate.real

    #can now do the correction computation in a single-shot.
    alphas = bulk_alpha(combined_taylor_dict, tableau, [desired_bitstring])
    rates = _np.fromiter(combined_taylor_dict.values(), dtype=_np.double)

    alpha_errgen_prods = alphas*rates
    correction = scale*_np.sum(alpha_errgen_prods)

    return correction

# TODO: The implementations for the pauli expectation value correction and probability correction
# are basically identical modulo some additional scale factors and the alpha function used. Should be able to combine
# the implementations into one function.
def stabilizer_pauli_expectation_correction(errorgen_dict: _ErrorgenDict, tableau: stim.Tableau, pauli: stim.PauliString,
                                            order: int = 1, truncation_threshold: float = 1e-14) -> float:
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

    Returns
    -------
    correction : float
        float corresponding to the correction to the expectation value for the
        selected pauli operator induced by the error generator (to specified order).
    """
    #accumulate the terms across orders (with short circuit logic for order 1 to save time):
    if order == 1:
        combined_taylor_dict = errorgen_dict
    else:
        #compute the taylor series approximation to the desired order.
        taylor_expansion = error_generator_taylor_expansion(errorgen_dict, order, truncation_threshold)
        # Accumulate all of the dictionaries in taylor expansion into a single one, summing overlapping terms.
        combined_taylor_dict = {}
        get = combined_taylor_dict.get
        for order_dict in taylor_expansion:
            for lbl, rate in order_dict.items():
                combined_taylor_dict[lbl] = get(lbl, 0) + rate.real

    #can now do the correction computation in a single-shot.
    alphas = bulk_alpha_pauli(combined_taylor_dict, tableau, [pauli])
    rates = _np.fromiter(combined_taylor_dict.values(), dtype=_np.double)
    alpha_errgen_prods = alphas*rates
    correction = _np.sum(alpha_errgen_prods)

    return correction

def stabilizer_pauli_expectation_correction_numerical(errorgen_dict: dict[_EEL, float],
                                                      errorgen_propagator: _epropagator.ErrorGeneratorPropagator,
                                                      circuit: _Circuit, pauli: stim.PauliString,
                                                      order: int = 1) -> float:
    """
    Compute the kth-order correction to the expectation value of the specified pauli.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
    
    errorgen_propagator : `ErrorGeneratorPropagator`
        Error generator propagator used for constructing dense representation of the error generator dictionary.
    
    circuit : `Circuit`
        Circuit the expectation value is being measured against.
        
    pauli : stim.PauliString
        Pauli operator to compute expectation value correction for.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    Returns
    -------
    correction : float
        float corresponding to the correction to the expectation value for the
        selected pauli operator induced by the error generator (to specified order).
    """
    tableau = circuit.convert_to_stim_tableau()
    
    stabilizer_state = tableau.to_state_vector(endian='big')
    stabilizer_state_dmvec = state_to_dmvec(stabilizer_state)
    stabilizer_state_dmvec.reshape((len(stabilizer_state_dmvec),1))
    
    # also get the superoperator (in the standard basis) corresponding to the taylor series
    # expansion of the specified error generator dictionary.
    taylor_expanded_errorgen = error_generator_taylor_expansion_numerical(errorgen_dict, errorgen_propagator, order=order, mx_basis='std')
    
    # finally need the superoperator for the selected pauli.
    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    # flatten this row-wise
    pauli_vec = _np.ravel(pauli_unitary)
    pauli_vec.reshape((len(pauli_vec),1))
    
    expectation_correction = _np.linalg.multi_dot([pauli_vec.conj().T, taylor_expanded_errorgen,stabilizer_state_dmvec]).item()
    return expectation_correction

def stabilizer_probability(tableau: stim.Tableau, desired_bitstring: str) -> float:
    """
    Calculate the output probability for the specified output bitstring.
    
    TODO: Should be able to do this more efficiently for many bitstrings
    by looking at the structure of the random support.
    
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
    # compute what Gidney calls the tableau fidelity (which in this case gives the probability).
    return tableau_fidelity(tableau, bitstring_to_tableau(desired_bitstring))

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
                                       truncation_threshold: float = 1e-14) -> float:
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
    correction = stabilizer_probability_correction(errorgen_dict, tableau, desired_bitstring, order, truncation_threshold)
    return ideal_prob + correction

def approximate_stabilizer_pauli_expectation(errorgen_dict: dict[_EEL, float], circuit: Union[_Circuit, stim.Tableau],
                                             pauli: Union[str, stim.PauliString], order: int = 1,
                                             truncation_threshold: float = 1e-14) -> float:
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
    correction = stabilizer_pauli_expectation_correction(errorgen_dict, tableau, pauli, order, truncation_threshold)
    return ideal_expectation + correction

def approximate_stabilizer_pauli_expectation_numerical(errorgen_dict: dict[_EEL, float],
                                                       errorgen_propagator: _epropagator.ErrorGeneratorPropagator,
                                                       circuit: _Circuit, pauli: stim.PauliString,
                                                       order: int = 1) -> float:
    """
    Calculate the approximate probability of a desired bit string using a first-order approximation.
    This function performs the corrections numerically and so it primarily intended for testing
    infrastructure.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.

    errorgen_propagator : `ErrorGeneratorPropagator`
        Error generator propagator used for constructing dense representation of the error generator dictionary.
    
    circuit : `Circuit`
        A pygsti `Circuit` or a stim.Tableau to compute the output pauli expectation value for.
        
    pauli : stim.PauliString
        Pauli operator to compute expectation value for.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value. (Used internally in computation of probability corrections)
    
    Returns
    -------
    expectation_value : float
        Approximate expectation value for desired pauli.
    """
    
    tableau = circuit.convert_to_stim_tableau()

    # recast keys to local stim ones if needed (in canonical basis element label order; a
    # GlobalElementaryErrorgenLabel key was never supported here since no sslbls are available).
    first_lbl = next(iter(errorgen_dict))
    if isinstance(first_lbl, (_GEEL, _LEEL)):
        errorgen_dict = _canonicalize_errorgen_layer(errorgen_dict)

    ideal_expectation = stabilizer_pauli_expectation(tableau, pauli)
    correction = stabilizer_pauli_expectation_correction_numerical(errorgen_dict, errorgen_propagator, circuit, pauli, order)
    return ideal_expectation + correction

def approximate_stabilizer_probabilities(errorgen_dict: dict[_EEL, float], circuit: Union[_Circuit, stim.Tableau],
                                         order: int = 1, truncation_threshold: float = 1e-14) -> _np.ndarray:
    """
    Calculate the approximate probability distribution over all bitstrings using a first-order approximation.
    Note the size of this distribution scales exponentially in the qubit count, so this is very inefficient for
    any more than a few qubits.

    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.
    
    circuit : `Circuit` or `stim.Tableau`
        A pygsti `Circuit` or a stim.Tableau to compute the output probability for. In either
        case this should be a Clifford circuit and convertible to a stim.Tableau.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value. (Used internally in computation of probability corrections)
    
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

    # get set of all bit strings
    num_qubits = len(tableau)
    bitstrings = ["".join(bitstring) for bitstring in product(['0','1'], repeat=num_qubits)]

    # initialize an array for the probabilities
    probs = _np.zeros(2**num_qubits)

    for i, bitstring in enumerate(bitstrings):
        probs[i] = approximate_stabilizer_probability(errorgen_dict, tableau, bitstring, order, truncation_threshold)

    return probs

@_with_cyclic_gc_paused
def error_generator_taylor_expansion(errorgen_dict: _ErrorgenDict, order: int = 1,
                                     truncation_threshold: float = 1e-14) -> list[_ErrorgenDict]:
    """
    Compute the nth-order taylor expansion for the exponentiation of the error generator described by the input
    error generator dictionary. (Excluding the zeroth-order identity).
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding rates
        are below this value.

    Returns
    -------
    list of dictionaries
        List of dictionaries whose keys are error generator labels and whose values are rates (including
        whatever scaling comes from order of taylor expansion). Each list corresponds to an order
        of the taylor expansion.
    """
       
 
    taylor_order_terms = [dict() for _ in range(order)]

    for lbl, rate in errorgen_dict.items():
        if abs(rate) > truncation_threshold:
            taylor_order_terms[0][lbl] = rate

    if order > 1 and errorgen_dict:
        # The k-th order term is (1/k!) L^k with L = sum_i rate_i * L_i. Composition is bilinear, so
        # the k-th power is built from the aggregated (k-1)-th power, L^k = L o L^(k-1): every
        # generator of `errorgen_dict` is composed with every term of the previous power once,
        # instead of re-composing the tail of each k-tuple of generators (which repeats the same
        # compositions for every leading generator). L and L^(k-1) commute, so the product is
        # formed by `_commuting_product` from anticommutators, L o L^(k-1) = ½{L, L^(k-1)} (see its
        # docstring). The powers are kept untruncated between orders (only the returned copies
        # are truncated), which is what makes [L, L^(k-1)] = 0 exact. Only the square gets the
        # unordered-pair halving of `_commuting_product`; forming even powers from two equal
        # factors instead (L^4 = ½{L^2, L^2}) would extend it, but costs ½|L^2|^2 pairs against
        # |L||L^3| and was measured at 0.8-0.9x on inputs with few generators and 2.3x slower on
        # a 75-generator input, so every power is L o L^(k-1).
        identity = 'I' * len(next(iter(errorgen_dict))._hashable_basis_element_labels[0])
        previous_power = errorgen_dict
        for current_order in range(2, order + 1):
            order_scale = 1 / factorial(current_order)
            current_power = _commuting_product(errorgen_dict, previous_power, identity)
            taylor_order_terms[current_order - 1] = {lbl: order_scale * rate for lbl, rate in current_power.items()
                                                     if order_scale * abs(rate) > truncation_threshold}
            previous_power = current_power

    return taylor_order_terms


def _commuting_product(errorgen_dict_1: dict[_LSE, _Rate], errorgen_dict_2: dict[_LSE, _Rate],
                       identity: str) -> dict[_LSE, complex]:
    r"""
    The composition L o M of two error generators given as label -> rate dictionaries, for
    error generators that COMMUTE, [L, M] = 0 (e.g. M a power of L, as in the Taylor
    expansion). The precondition is not checked.

    For commuting L and M the composition is half the anticommutator, L o M = ½{L, M}, and
    by bilinearity ½ Σ_ij r_i m_j {E_i, F_j} over the terms of the two dictionaries. The
    anticommutator of two elementary error generators emits fewer terms than either of their
    compositions (see `error_generator_anticommutator`), and when the two dictionaries are
    the same object the symmetry {E_i, E_j} = {E_j, E_i} halves the pair loop.

    The pairs with a stochastic member are not sent to the anticommutator handlers. Writing
    S_Q = 𝒬 - 1 (𝒬 the conjugation rho -> Q rho Q) and T_X for the total Pauli of X (P for
    H_P, I for S_P, PR for C_{P,R} and A_{P,R}; see "Anticommutators and Pauli conjugation"
    in the module docstring),

        ½ {S_Q, X} = [ω(Q, T_X) = 0] 𝒬X  -  X,

    i.e. the conjugation image of the partner when Q commutes with its total Pauli, nothing
    otherwise, and in either case minus the partner itself (the "bleed" of the -1 in S_Q).
    Splitting L = L^⊥ + L_S (L_S the S-type terms, R_L = Σ_{i∈S} r_i their rate sum) and
    likewise M = M^⊥ + M_S, R_M, the product is assembled block by block:

        L o M = ½ Σ_{i∈⊥, j∈⊥} r_i m_j {E_i, F_j}                              (1: anticommutators)
              + Σ_{i∈⊥, j∈S} [ω(P_j, T_{E_i}) = 0] r_i m_j 𝒫_j E_i   - R_M L^⊥   (2: conjugation or skip)
              + Σ_{i∈S, j∈⊥} [ω(Q_i, T_{F_j}) = 0] r_i m_j 𝒬_i F_j   - R_L M^⊥   (3: mirror of 2)
              + Σ_{i∈S, j∈S} r_i m_j S_{R_ij}   - R_M L_S - R_L M_S,   R_ij = Q_i P_j / phase   (4)
              = [handler, conjugation and S_R terms]  -  R_M L  -  R_L M .

    The bleed of all four blocks thus collapses to a rescaling of the *existing* terms of L
    and M (no label is constructed for it), which is why this is cheaper than summing the
    full anticommutators: every S pair would otherwise emit its partner as a fresh label,
    to be found again only by the aggregation. Block 4 uses the S-S relation directly,
    {S_Q, S_P} = 2 S_R - 2 S_Q - 2 S_P: 𝒬 S_P = S_R - S_Q would serve too but is asymmetric
    in the pair, which would break the unordered-pair form below; the symmetric S_R term is
    what allows the S-S bleed to be absorbed into -R_M L_S - R_L M_S. No identity term ever
    appears - each ½{S_Q, X} above is trace annihilating on its own - although the
    intermediate 𝒬 𝒫 = S_R + 1 is unital; that object is never formed. The convention
    T_{S_P} = I (ω = 0, never skipped) is what makes the S-X formula hold for X = S_P too.

    For M = L (the same object passed twice) block 1 runs over unordered pairs with weights
    r_i r_j (i < j) and ½ r_i² (i = j), blocks 2 and 3 coincide and are done once with weight
    2 r_i r_j, block 4 runs over i < j with weight 2 r_i r_j (the diagonal gives S_I = 0) and
    the bleed is -2 R_L L. For M != L no analogous halving exists.

    Parameters
    ----------
    errorgen_dict_1, errorgen_dict_2 : dict
        The two commuting error generators, as dictionaries from `LocalStimErrorgenLabel`
        to rate. Pass the same object twice to compute a square.

    identity : str
        The all-identity Pauli string `'I'*n` for the number of qubits n.

    Returns
    -------
    dict
        The composition L o M as a dictionary from `LocalStimErrorgenLabel` to (complex)
        rate, aggregated but not truncated.
    """
    same = errorgen_dict_2 is errorgen_dict_1
    perp_1 = [item for item in errorgen_dict_1.items() if item[0].type_idx != 1]
    stoch_1 = [item for item in errorgen_dict_1.items() if item[0].type_idx == 1]
    if same:
        perp_2, stoch_2 = perp_1, stoch_1
    else:
        perp_2 = [item for item in errorgen_dict_2.items() if item[0].type_idx != 1]
        stoch_2 = [item for item in errorgen_dict_2.items() if item[0].type_idx == 1]

    product: dict[_LSE, complex] = {}
    get = product.get

    # block 1: ½ Σ r_i m_j {E_i, F_j} over the non-stochastic terms.
    if same:
        for i, (lbl_1, rate_1) in enumerate(perp_1):
            for j in range(i, len(perp_1)):
                lbl_2, rate_2 = perp_1[j]
                weight = rate_1 * rate_2 if j != i else 0.5 * rate_1 * rate_1
                for lbl, rate in error_generator_anticommutator(lbl_1, lbl_2, weight, identity):
                    product[lbl] = get(lbl, 0) + rate
    else:
        for lbl_1, rate_1 in perp_1:
            for lbl_2, rate_2 in perp_2:
                for lbl, rate in error_generator_anticommutator(lbl_1, lbl_2, 0.5 * rate_1 * rate_2, identity):
                    product[lbl] = get(lbl, 0) + rate

    # blocks 2 and 3: the conjugation image r m 𝒬X of each non-stochastic term X by each
    # stochastic partner S_Q whose Q commutes with the total Pauli of X. For H that is one
    # commutation test; for C_{P,R} / A_{P,R} the product PR is not formed: Q commutes with
    # PR iff it commutes with both or neither of P, R.
    def conjugation_block(perp, stoch, scale):
        for s_lbl, s_rate in stoch:
            Q_pauli = s_lbl.basis_element_labels[0]
            Q_str = s_lbl._hashable_basis_element_labels[0]
            for x_lbl, x_rate in perp:
                bels = x_lbl.basis_element_labels
                if x_lbl.type_idx == 0:
                    commutes_with_total = Q_pauli.commutes(bels[0])
                else:
                    commutes_with_total = Q_pauli.commutes(bels[0]) == Q_pauli.commutes(bels[1])
                if commutes_with_total:
                    for lbl, rate in pauli_conjugation_composition(Q_pauli, x_lbl, scale * s_rate * x_rate,
                                                                   identity, pauli_str=Q_str):
                        product[lbl] = get(lbl, 0) + rate

    if same:
        conjugation_block(perp_1, stoch_1, 2.0)
    else:
        conjugation_block(perp_1, stoch_2, 1.0)
        conjugation_block(perp_2, stoch_1, 1.0)

    # block 4: r m S_R per stochastic pair, R the phase-stripped product of the two Paulis.
    # The phase is irrelevant here, so R is built from the two cached strings alone
    # (`bel_product_str`) and the label's stim Pauli parsed from the result - cheaper than
    # the stim product plus rendering; accumulated one row at a time (R = I only for P = Q,
    # which cannot occur across distinct keys of one dictionary but can across two).
    def stochastic_pair_terms(lbl_1, rate_1, partners, scale):
        sP = lbl_1._hashable_basis_element_labels[0]
        terms = []
        for lbl_2, rate_2 in partners:
            sR = _bel_product_str(sP, lbl_2._hashable_basis_element_labels[0])
            if sR != identity:
                terms.append((_LSE('S', (stim.PauliString(sR),), pauli_str_reps=(sR,)), scale * rate_1 * rate_2))
        for lbl, rate in terms:
            product[lbl] = get(lbl, 0) + rate

    if same:
        # unordered pairs i < j, each with weight 2 r_i r_j
        for i, (lbl_1, rate_1) in enumerate(stoch_1):
            stochastic_pair_terms(lbl_1, rate_1, stoch_1[i + 1:], 2.0)
    else:
        for lbl_1, rate_1 in stoch_1:
            stochastic_pair_terms(lbl_1, rate_1, stoch_2, 1.0)

    # the bleed of all blocks, -R_M L - R_L M, on the existing keys (nothing to do for an
    # operand without stochastic terms).
    rate_sum_1 = sum(rate for _, rate in stoch_1)
    rate_sum_2 = rate_sum_1 if same else sum(rate for _, rate in stoch_2)
    if same:
        if rate_sum_1 != 0:
            _accumulate_scaled(product, errorgen_dict_1, -2.0 * rate_sum_1)
    else:
        if rate_sum_1 != 0:
            _accumulate_scaled(product, errorgen_dict_2, -rate_sum_1)
        if rate_sum_2 != 0:
            _accumulate_scaled(product, errorgen_dict_1, -rate_sum_2)
    return product


def _accumulate_scaled(target: dict[_LSE, complex], errorgen_dict: dict[_LSE, _Rate], coeff: complex) -> None:
    """
    Add coeff * errorgen_dict to `target` in place, on the existing label objects (no label is
    constructed). Relies on the keys being in canonical basis element label order, which every
    `LocalStimErrorgenLabel` is (see its class docstring): the handlers emit canonical labels,
    so a non-canonical key here would sit beside the handlers' key for the same generator.
    """
    get = target.get
    for lbl, rate in errorgen_dict.items():
        target[lbl] = get(lbl, 0) + coeff * rate


def error_generator_taylor_expansion_numerical(errorgen_dict: dict[_EEL, float],
                                               errorgen_propagator: _epropagator.ErrorGeneratorPropagator,
                                               order: int = 1,
                                               mx_basis: Union[str, _Basis] = 'pp') -> _np.ndarray:
    """
    Compute the nth-order taylor expansion for the exponentiation of the error generator described by the input
    error generator dictionary. (Excluding the zeroth-order identity). This function computes a dense representation
    of this taylor expansion as a numpy array and is primarily intended for testing infrastructure.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.

    errorgen_propagator : `ErrorGeneratorPropagator`
        Error generator propagator used for constructing dense representation of the error generator dictionary.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    mx_basis : `Basis` or str, optional (default 'pp')
        Basis in which to return the matrix.

    Returns
    -------
    numpy.ndarray
        A dense numpy array corresponding to the nth order taylor expansion of the specified error generator.
    """
       
    errorgen_mat = errorgen_propagator.errorgen_layer_dict_to_errorgen(errorgen_dict, mx_basis)
    taylor_expansion = _np.zeros(errorgen_mat.shape, dtype=_np.complex128)
    for i in range(1, order+1):
        taylor_expansion += 1/factorial(i)*_np.linalg.matrix_power(errorgen_mat, i)

    return taylor_expansion


PauliPhaseUpdater = Callable[[str,str,Optional[bool]],tuple[complex,str]]
PauliPhaseZerosUpdater = Callable[[str,Optional[bool]],tuple[complex,str]]
AmplitudeOfStateType = Callable[[Union[stim.Tableau, stim.TableauSimulator],str,bool], complex]
BulkAmplitudeOfStateType = Callable[[Union[stim.Tableau, stim.TableauSimulator], list[Union[str, stim.PauliString]],bool], _np.ndarray]
BulkPhiType = Callable[[Union[stim.Tableau, stim.TableauSimulator],str,list[Union[str,stim.PauliString]],list[Union[str,stim.PauliString]]], _np.ndarray]
BulkAlphaType = Callable[[Iterable[_LSE],Union[stim.Tableau, stim.TableauSimulator],list[str]], _np.ndarray]
BulkAlphaPauliType = Callable[[Iterable[_LSE],Union[stim.Tableau, stim.TableauSimulator],list[stim.PauliString]], _np.ndarray]

#alias in cython implementations.
try:
    from pygsti.tools import fasterrgencalc as _fc
    pauli_phase_update_all_zeros: PauliPhaseZerosUpdater = _fc.fast_pauli_phase_update_all_zeros
    pauli_phase_update: PauliPhaseUpdater = _fc.fast_pauli_phase_update
    amplitude_of_state: AmplitudeOfStateType = _fc.fast_amplitude_of_state
    bulk_amplitude_of_state: BulkAmplitudeOfStateType = _fc.fast_bulk_amplitude_of_state
    bulk_phi: BulkPhiType = _fc.fast_bulk_phi
    bulk_alpha: BulkAlphaType = _fc.fast_bulk_alpha
    bulk_alpha_pauli: BulkAlphaPauliType = _fc.fast_bulk_alpha_pauli    
except ImportError:
    msg = 'Could not import cython module `fastcalc`. This may indicate that your cython extensions for pyGSTi failed to '\
          +'properly build. Lack of cython extensions can result in significant performance degredation so we recommend trying to rebuild them. '\
           'Falling back to python implementation for: pauli_phase_update_all_zeros, pauli_phase_update, amplitude_of_state, ' \
           'bulk_phi, bulk_alpha and bulk_alpha_pauli.'
    warnings.warn(msg)
    pauli_phase_update_all_zeros: PauliPhaseZerosUpdater = slow_pauli_phase_update_all_zeros
    pauli_phase_update: PauliPhaseUpdater = slow_pauli_phase_update
    amplitude_of_state: AmplitudeOfStateType = slow_amplitude_of_state
    bulk_amplitude_of_state: BulkAmplitudeOfStateType = slow_bulk_amplitude_of_state
    bulk_phi: BulkPhiType = slow_bulk_phi
    bulk_alpha: BulkAlphaType = slow_bulk_alpha
    bulk_alpha_pauli: BulkAlphaPauliType = slow_bulk_alpha_pauli 

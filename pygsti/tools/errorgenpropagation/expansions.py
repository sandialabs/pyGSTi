"""
BCH, Magnus, Zassenhaus and Taylor expansions of error generator layers, the commuting product
used by the Taylor expansion, and the layer accumulation and truncation helpers they share.
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

from itertools import chain, product
from math import factorial
from typing import Literal, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing.pool as _mppool

from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE, bel_product_str as _bel_product_str
from pygsti.tools.mptools import (
    determine_num_workers as _determine_num_workers,
    SharedWorkerPool as _SharedWorkerPool,
    lpt_partition as _lpt_partition,
    parallel_pair_reduce as _parallel_pair_reduce,
    warn_if_insufficient_memory as _warn_if_insufficient_memory,
    resolve_pool_workers as _resolve_pool_workers,
    _pool_scope,
)
from .errgentypes import _ErrorgenDict, _Rate
from .algebra import (
    error_generator_commutator,
    error_generator_anticommutator,
    pauli_conjugation_composition,
)
from . import parallelutils as _par
from .parallelutils import (
    _with_cyclic_gc_paused,
    EGPROP_PREWARM_PACKAGES,
    _estimate_magnus_peak_memory_from_layers,
    estimate_taylor_peak_memory,
    _should_parallelize_magnus,
    _should_parallelize_commuting_product,
    _lse_from_string_rep,
    _merge_string_dictionaries,
    _MagnusPairEvaluator,
    _MagnusPhase1PairEvaluator,
    _MagnusPhase2Evaluator,
    _merge_pair_indexed_string_dictionaries,
    _generate_balanced_magnus_pairs,
    _CommutingProductSliceWorker,
)

__all__ = [
    'bch_approximation',
    'magnus_expansion',
    'zassenhaus_formula',
    'error_generator_taylor_expansion',
]


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
                     truncation_threshold: float = 1e-14,
                     num_workers: Union[int, str, None] = None,
                     pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> _ErrorgenDict:
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

    num_workers : int, str, or None, optional
        Target worker count (defaults to None for sequential execution; pass 'auto'
        or an integer to enable parallel execution across layer pairs).

    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.
        
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
    # in which case no commutators are computed). Scans forward to find the first non-empty layer to determine n_q.
    identity = None
    for layer in errorgen_layers:
        if layer:
            identity = 'I'*len(next(iter(layer)).basis_element_labels[0])
            break

    # NOTE: The automatic memory headroom check for the Magnus expansion is disabled for now. Empirical
    # testing (4 to 100 qubits, orders 2 and 3, serial and parallel) found that both the worst-case
    # estimate (`estimate_magnus_peak_memory`) and the support-aware estimate
    # (`_estimate_magnus_peak_memory_from_layers`) overestimate actual memory use by too much to be
    # practically useful at present.
    #
    # if len(errorgen_layers) > 1 and magnus_order >= 2 and identity is not None:
    #     n_q = len(identity)
    #     workers = _resolve_pool_workers(pool) if pool is not None else (_determine_num_workers(num_workers) if num_workers is not None else 1)
    #     _warn_if_insufficient_memory(
    #         _estimate_magnus_peak_memory_from_layers(errorgen_layers, n_q, num_workers=workers,
    #                                                  magnus_order=magnus_order),
    #         task_name="magnus_expansion"
    #     )

    if magnus_order == 2:
        # second-order magnus term: (1/2)\sum_{t1=1}^n \sum_{t2=1}^{t1-1} [A(t1), A(t2)]
        new_errorgen_layer.append(_second_order_magnus_term(errorgen_layers, identity, truncation_threshold,
                                                            num_workers=num_workers, pool=pool))
    else:
        # the third-order term is built from the same pairwise layer commutators as the second-order one,
        # so both are produced by a single pass over the layer pairs.
        new_errorgen_layer.extend(_second_and_third_order_magnus_terms(errorgen_layers, identity, truncation_threshold,
                                                                      num_workers=num_workers, pool=pool))

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
                              truncation_threshold: float = 1e-14,
                              num_workers: Union[int, str, None] = None,
                              pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> _ErrorgenDict:
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

    num_workers : int, str, or None, optional
        Target worker count (defaults to None for sequential execution; pass 'auto'
        or an integer to enable parallel execution across layer pairs).

    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.

    Returns
    -------
    second_order_comm_dict : dict
        A dictionary with the same general structure as those in `errorgen_layers`, but with the
        rates combined according to the second order of the magnus expansion.
    """
    errorgen_pairs = []
    pair_weights = []
    for i in range(len(errorgen_layers)):
        for j in range(i):
            errorgen_pairs.append((errorgen_layers[i], errorgen_layers[j]))
            pair_weights.append(len(errorgen_layers[i]) * len(errorgen_layers[j]))
    
    # precompute an identity string for comparisons in commutator calculations if one is not provided.
    if identity is None and errorgen_layers:
        for layer in errorgen_layers:
            if layer:
                identity = 'I'*len(next(iter(layer)).basis_element_labels[0])
                break

    # If every layer is empty, all commutators evaluate trivially to zero without computation.
    if identity is None:
        return {}

    workers = _determine_num_workers(num_workers) if pool is None else _resolve_pool_workers(pool)
    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")

    n_q = len(identity)
    avg_terms = sum(len(l) for l in errorgen_layers) // max(1, len(errorgen_layers))

    if workers == 1 or num_workers is None and pool is None or not _par._should_parallelize_magnus(len(errorgen_layers), avg_terms, num_workers=workers, n_qubits=n_q):
        # accumulate the second-order correction, (1/2)[A(t1), A(t2)] over all layer pairs sequentially
        second_order_comm_dict = {}
        for errorgen_layer_1, errorgen_layer_2 in errorgen_pairs:
            _accumulate_layer_pairwise_commutators(second_order_comm_dict, errorgen_layer_1, errorgen_layer_2, identity,
                                                   addl_weight=0.5, truncation_threshold=truncation_threshold)
        return _truncated(second_order_comm_dict, truncation_threshold)

    # Parallel pairwise reduction across workers with dynamic capacity slicing and streaming reduction
    tasks, task_weights = _generate_balanced_magnus_pairs(
        errorgen_layers, workers, alpha=0.45, include_indices=False
    )
    evaluator = _MagnusPairEvaluator(identity, truncation_threshold)
    second_order_str_dict = _parallel_pair_reduce(
        evaluator, tasks, _merge_string_dictionaries, num_workers=workers, pool=pool,
        prewarm_packages=EGPROP_PREWARM_PACKAGES, pair_weights=task_weights, dynamic=True
    )
    if second_order_str_dict is None:
        return {}
    survivors = _truncated(second_order_str_dict, truncation_threshold)
    return {_lse_from_string_rep(k, n_q): v for k, v in survivors.items()}

def _second_and_third_order_magnus_terms(errorgen_layers: list[_ErrorgenDict], identity: Optional[str],
                                         truncation_threshold: float = 1e-14,
                                         num_workers: Union[int, str, None] = None,
                                         pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> tuple[_ErrorgenDict, _ErrorgenDict]:
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

    num_workers : int, str, or None, optional
        Target worker count (defaults to None for sequential execution; pass 'auto'
        or an integer to enable parallel execution across layer pairs and outer commutators).

    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.

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

    if identity is None and errorgen_layers:
        for layer in errorgen_layers:
            if layer:
                identity = 'I'*len(next(iter(layer)).basis_element_labels[0])
                break

    # If every layer is empty, all commutators evaluate trivially to zero without computation.
    if identity is None:
        return {}, {}

    workers = _determine_num_workers(num_workers) if pool is None else _resolve_pool_workers(pool)
    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")
    n_q = len(identity)
    avg_terms = sum(len(l) for l in errorgen_layers) // max(1, len(errorgen_layers))
    num_layers = len(errorgen_layers)

    if workers == 1 or num_workers is None and pool is None or not _par._should_parallelize_magnus(num_layers, avg_terms, num_workers=workers, n_qubits=n_q):
        # Sequential execution path (100% untouched)
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

    # Two-Phase Parallel Evaluation with Dynamic Capacity Slicing and Streaming Reduction
    p1_tasks, p1_weights = _generate_balanced_magnus_pairs(
        errorgen_layers, workers, alpha=0.45, include_indices=True
    )

    p1_evaluator = _MagnusPhase1PairEvaluator(identity, truncation_threshold)
    p2_evaluator = _MagnusPhase2Evaluator(identity, truncation_threshold)

    def _execute_magnus_phases(active_pool):
        p1_res = _parallel_pair_reduce(
            p1_evaluator, p1_tasks, _merge_pair_indexed_string_dictionaries,
            pool=active_pool, pair_weights=p1_weights, dynamic=True
        )
        if p1_res is None:
            p1_res = {}

        rows = [{} for _ in range(num_layers)]
        held_columns = [{} for _ in range(num_layers)]
        for (i, k), pw_dict in p1_res.items():
            r_i = rows[i]
            c_k = held_columns[k]
            for s_lbl, rate in pw_dict.items():
                r_i[s_lbl] = r_i.get(s_lbl, 0.0) + rate
                c_k[s_lbl] = c_k.get(s_lbl, 0.0) + rate

        second_order_str = {}
        for r in rows:
            for s_lbl, rate in r.items():
                second_order_str[s_lbl] = second_order_str.get(s_lbl, 0.0) + 0.5 * rate

        # Intermediate prefix row sums and suffix column sums
        R = []
        running_row = {}
        for i in range(num_layers):
            curr = running_row.copy()
            for s_lbl, rate in rows[i].items():
                curr[s_lbl] = curr.get(s_lbl, 0.0) + (1/12) * rate
            curr_trunc = {k: v for k, v in curr.items() if abs(v) > truncation_threshold}
            R.append({_lse_from_string_rep(k, n_q): v for k, v in curr_trunc.items()})
            for s_lbl, rate in rows[i].items():
                running_row[s_lbl] = running_row.get(s_lbl, 0.0) + (1/6) * rate
            running_row = {k: v for k, v in running_row.items() if abs(v) > truncation_threshold}

        C = []
        running_col = {}
        for k in range(num_layers - 2, -1, -1):
            col = held_columns[k]
            curr = running_col.copy()
            for s_lbl, rate in col.items():
                curr[s_lbl] = curr.get(s_lbl, 0.0) - (1/12) * rate
            curr_trunc = {k: v for k, v in curr.items() if abs(v) > truncation_threshold}
            C.append((k, {_lse_from_string_rep(s, n_q): v for s, v in curr_trunc.items()}))
            for s_lbl, rate in col.items():
                running_col[s_lbl] = running_col.get(s_lbl, 0.0) - (1/6) * rate
            running_col = {s: v for s, v in running_col.items() if abs(v) > truncation_threshold}

        # Phase 2: Outer commutator tasks with LPT bin-packing
        outer_tasks = []
        outer_weights = []
        for i in range(1, num_layers):
            outer_tasks.append((errorgen_layers[i], R[i], 1.0))
            outer_weights.append(len(errorgen_layers[i]) * len(R[i]))
        for k, c_k in C:
            outer_tasks.append((errorgen_layers[k], c_k, 1.0))
            outer_weights.append(len(errorgen_layers[k]) * len(c_k))

        third_order_str = _parallel_pair_reduce(
            p2_evaluator, outer_tasks, _merge_string_dictionaries,
            pool=active_pool, pair_weights=outer_weights, dynamic=True
        )
        if third_order_str is None:
            third_order_str = {}

        return second_order_str, third_order_str

    if pool is not None:
        second_order_comm_str_dict, third_order_str_dict = _execute_magnus_phases(pool)
    else:
        with _SharedWorkerPool(num_workers=workers, prewarm_packages=EGPROP_PREWARM_PACKAGES) as local_pool:
            second_order_comm_str_dict, third_order_str_dict = _execute_magnus_phases(local_pool)

    om2_survivors = _truncated(second_order_comm_str_dict, truncation_threshold)
    om3_survivors = _truncated(third_order_str_dict, truncation_threshold)
    return ({_lse_from_string_rep(k, n_q): v for k, v in om2_survivors.items()},
            {_lse_from_string_rep(k, n_q): v for k, v in om3_survivors.items()})

@_with_cyclic_gc_paused
def zassenhaus_formula(errorgen_groups: list[_ErrorgenDict], zassenhaus_order: Literal[1,2] = 1,
                      truncation_threshold: float = 1e-14,
                      num_workers: Union[int, str, None] = None,
                      pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> list[_ErrorgenDict]:
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
        second_order_comm_dict = _second_order_magnus_term(errorgen_groups, identity, truncation_threshold,
                                                           num_workers=num_workers, pool=pool)
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


def _truncated(errorgen_dict: dict[Union[_LSE, str], _Rate], truncation_threshold: float) -> dict[Union[_LSE, str], _Rate]:
    """The entries of `errorgen_dict` (label -> rate) whose rate exceeds `truncation_threshold` in magnitude."""
    return {lbl: rate for lbl, rate in errorgen_dict.items() if abs(rate) > truncation_threshold}



@_with_cyclic_gc_paused
def error_generator_taylor_expansion(errorgen_dict: _ErrorgenDict, order: int = 1,
                                     truncation_threshold: float = 1e-14,
                                     num_workers: Union[int, str, None] = None,
                                     pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> list[_ErrorgenDict]:
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
        are below this value. The intermediate powers of the generator from which the higher orders
        are built are truncated at this threshold too. In a parallel evaluation (`num_workers` or `pool`)
        the workers drop partial sums below the threshold divided by their number of tasks, so the
        result can differ from the serial one by up to about the threshold, and terms whose magnitude is
        close to the threshold may be kept by one and dropped by the other.

    num_workers : int, str, or None, optional
        Target worker count for parallel evaluation across commuting product slices.

    pool : Pool or SharedWorkerPool, optional
        Existing worker pool to reuse.

    Returns
    -------
    list of dictionaries
        List of dictionaries whose keys are error generator labels and whose values are rates (including
        whatever scaling comes from order of taylor expansion). Each list corresponds to an order
        of the taylor expansion.
    """
       
 
    taylor_order_terms = [{} for _ in range(order)]

    for lbl, rate in errorgen_dict.items():
        if abs(rate) > truncation_threshold:
            taylor_order_terms[0][lbl] = rate

    if order > 1 and errorgen_dict:
        n_q = len(next(iter(errorgen_dict))._hashable_basis_element_labels[0])
        workers = _resolve_pool_workers(pool) if pool is not None else (_determine_num_workers(num_workers) if num_workers is not None else 1)
        _warn_if_insufficient_memory(
            estimate_taylor_peak_memory(n_q, len(errorgen_dict), order=order, num_workers=workers),
            task_name="error_generator_taylor_expansion"
        )
        # The k-th order term is (1/k!) L^k with L = sum_i rate_i * L_i. Composition is bilinear, so
        # the k-th power is built from the aggregated (k-1)-th power, L^k = L o L^(k-1): every
        # generator of `errorgen_dict` is composed with every term of the previous power once,
        # instead of re-composing the tail of each k-tuple of generators (which repeats the same
        # compositions for every leading generator). L and L^(k-1) commute, so the product is
        # formed by `_commuting_product` from anticommutators, L o L^(k-1) = ½{L, L^(k-1)} (see its
        # docstring). Only the square gets the unordered-pair halving of `_commuting_product`;
        # forming even powers from two equal factors instead (L^4 = ½{L^2, L^2}) would extend it,
        # but costs ½|L^2|^2 pairs against |L||L^3| and was measured at 0.8-0.9x on inputs with few
        # generators and 2.3x slower on a 75-generator input, so every power is L o L^(k-1).
        #
        # The powers are truncated at `truncation_threshold` between orders (as are the returned,
        # 1/k!-scaled copies). Strictly, the truncated power L^(k-1) - R (R the dropped terms, each
        # of magnitude <= the threshold) no longer commutes with L, so the ½{L, .} form of the product
        # is applied to a slightly non-commuting input, and the dropped terms R change the next power
        # by terms of size ~ c * ||L||_1 * threshold per label, c a small combinatorial coefficient.
        # That can only matter if ||L||_1 is of order one or larger, where the Taylor series converges
        # too slowly for low orders to be useful anyway. Measured on 4- and 8-qubit propagated
        # generators (orders 2-4, ||L||_1 from 0.03 to 12), truncating the powers moved no reported
        # term by more than 3% of the threshold. The dropped terms are mostly cancellation residues
        # (over half of all terms of L^3 on 8 qubits), so truncating them substantially reduces the
        # size of the products at the following orders and the cost of materializing them.
        identity = 'I' * len(next(iter(errorgen_dict))._hashable_basis_element_labels[0])
        previous_power = errorgen_dict
        # A single (lazily started) pool is shared by the products of every order.
        with _pool_scope(num_workers, pool, prewarm_packages=EGPROP_PREWARM_PACKAGES) as order_pool:
            for current_order in range(2, order + 1):
                order_scale = 1 / factorial(current_order)
                current_power = _commuting_product(errorgen_dict, previous_power, identity,
                                                   truncation_threshold=truncation_threshold,
                                                   num_workers=num_workers, pool=order_pool)
                taylor_order_terms[current_order - 1] = {lbl: order_scale * rate for lbl, rate in current_power.items()
                                                         if order_scale * abs(rate) > truncation_threshold}
                previous_power = current_power

    return taylor_order_terms


def _commuting_product(errorgen_dict_1: dict[_LSE, _Rate], errorgen_dict_2: dict[_LSE, _Rate],
                       identity: str, truncation_threshold: float = 1e-14,
                       num_workers: Union[int, str, None] = None,
                       pool: Optional[Union[_SharedWorkerPool, _mppool.Pool]] = None) -> dict[_LSE, complex]:
    """
    The composition L o M of two commuting error generators (see `_raw_commuting_product`), with
    terms of magnitude at most `truncation_threshold` removed from the result.

    The serial and parallel evaluations both truncate, but are not guaranteed to agree exactly. In
    the parallel evaluation each worker drops the entries of its partial sum of magnitude at most
    `truncation_threshold / n_chunks` (to limit inter-process communication) and the merged sum is
    truncated at `truncation_threshold` before any labels are built. The merged values can differ from
    the serial ones by up to about `truncation_threshold`, and a label whose value is close to the
    threshold may be kept by one evaluation and dropped by the other.
    """
    workers = _determine_num_workers(num_workers) if pool is None else _resolve_pool_workers(pool)
    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")
    n1 = len(errorgen_dict_1)
    n2 = len(errorgen_dict_2)
    n_q = len(identity)
    if workers == 1 or num_workers is None and pool is None or not _should_parallelize_commuting_product(n_q, n1, n2, num_workers=workers):
        return _truncated(_raw_commuting_product(errorgen_dict_1, errorgen_dict_2, identity), truncation_threshold)

    # Parallel bilinear slicing with term-type load balancing
    target = errorgen_dict_1 if n1 <= n2 else errorgen_dict_2
    full = errorgen_dict_2 if n1 <= n2 else errorgen_dict_1
    items = list(target.items())
    weights = [3.0 if k.type_idx in (2, 3) else 1.0 for k, _ in items]
    chunks = [dict(b) for b in _lpt_partition(items, weights, workers)]

    # each chunk drops entries below its share of the threshold (see `_CommutingProductSliceWorker`)
    worker = _CommutingProductSliceWorker(full, identity, truncation_threshold / len(chunks))
    if pool is not None:
        chunk_results = pool.map(worker, chunks)
    else:
        with _SharedWorkerPool(num_workers=workers, prewarm_packages=EGPROP_PREWARM_PACKAGES) as local_pool:
            chunk_results = local_pool.map(worker, chunks)

    merged_str = {}
    for cr in chunk_results:
        for k, v in cr.items():
            merged_str[k] = merged_str.get(k, 0.0) + v

    # truncate before materializing labels, which dominates the cost of the merge
    survivors = _truncated(merged_str, truncation_threshold)
    return {_lse_from_string_rep(k, n_q): v for k, v in survivors.items()}


def _raw_commuting_product(errorgen_dict_1: dict[_LSE, _Rate], errorgen_dict_2: dict[_LSE, _Rate],
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

"""
Defines the EvalTree class.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations
import time as _time  # DEBUG TIMERS

import numpy as _np

from pygsti.circuits.circuit import Circuit as _Circuit, LayerTupLike
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.baseobjs.label import LabelTupTup, Label, LabelTup
import itertools
from pygsti.tools.sequencetools import (
    conduct_one_round_of_lcs_simplification,
    _compute_lcs_for_every_pair_of_sequences,
    create_tables_for_internal_LCS,
    simplify_internal_first_one_round
)
from pygsti.tools.dyadickronop import KronStructured
from pygsti.circuits.split_circuits_into_lanes import (
    compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit,
    compute_subcircuits
)

from scipy.sparse import kron as sparse_kron
from typing import List, Optional, Iterable, Union, TYPE_CHECKING, Tuple
from pygsti.tools.tqdm import our_tqdm

#region Split circuit list into lists of subcircuits

def _add_in_idle_gates_to_circuit(circuit: _Circuit, idle_gate_name: Union[str, Label] = 'I') -> _Circuit:
    """
    Add in explicit idles to the labels for each layer.
    """

    tmp = circuit.copy(editable=True)
    num_layers = circuit.num_layers

    for i in range(num_layers):
        tmp[i] = Label(tmp.layer_label_with_idles(i, idle_gate_name))

    if tmp._static:
        tmp.done_editing()
    return tmp





def setup_circuit_list_for_LCS_computations(
        circuit_list: list[_Circuit],
        implicit_idle_gate_name: Union[str, Label] = 'I'
    ) -> tuple[
        dict[int, dict[int, _Circuit]],
        dict[LayerTupLike, list[tuple[int, int]]],
        dict[tuple[int, ...],list[LayerTupLike]]
    ]:
    """
    Split a circuit list into a list of subcircuits by lanes. These lanes are non-interacting partions of a circuit.

    Also return a sequence detailing the number of lanes in each circuit.
    Then, a sequence detailing the number of qubits in each lane for a circuit.
    """

    # We want to split the circuit list into a dictionary of subcircuits where each sub_cir in the dict[key] act exclusively on the same qubits.
    # I need a mapping from subcircuit to actual circuit. This is uniquely defined by circuit_id and then lane id.

    cir_ind_and_lane_id_to_sub_cir: dict[int, dict[int, _Circuit]] = {}
    sub_cir_to_cir_id_and_lane_id:  dict[LayerTupLike, list[tuple[int, int]]] = {}
    line_labels_to_layertup_lists:  dict[tuple[int, ...], list[LayerTupLike]] = {}

    for i, cir in enumerate(circuit_list):

        if implicit_idle_gate_name:
            cir = _add_in_idle_gates_to_circuit(cir, implicit_idle_gate_name)

        qubit_to_lane, lane_to_qubits = compute_qubit_to_lane_and_lane_to_qubits_mappings_for_circuit(cir)
        sub_cirs = compute_subcircuits(cir, qubit_to_lane, lane_to_qubits)

        if not implicit_idle_gate_name:
            if not all([len(sc) == len(sub_cirs[0]) for sc in sub_cirs]):
                raise ValueError("Each lane does not have the same number of layers. Therefore, a lane has an implicit idle gate. Please add in idle gates explicitly to the circuit.")

        assert len(sub_cirs) == len(lane_to_qubits)
        for j in range(len(sub_cirs)):
            sc = _Circuit(sub_cirs[j],line_labels=tuple(lane_to_qubits[j]),)
            lbls = sc._line_labels
            if lbls in line_labels_to_layertup_lists:
                line_labels_to_layertup_lists[lbls].append(sc.layertup)
            else:
                line_labels_to_layertup_lists[lbls] = [sc.layertup]
            if sc.layertup in sub_cir_to_cir_id_and_lane_id:
                sub_cir_to_cir_id_and_lane_id[sc.layertup].append((i,j))
            else:
                sub_cir_to_cir_id_and_lane_id[sc.layertup] = [(i,j)]
            if i in cir_ind_and_lane_id_to_sub_cir:
                cir_ind_and_lane_id_to_sub_cir[i][j] = sc
            else:
                cir_ind_and_lane_id_to_sub_cir[i] = {j: sc}

    return cir_ind_and_lane_id_to_sub_cir, sub_cir_to_cir_id_and_lane_id, line_labels_to_layertup_lists

#endregion Split Circuits by lanes helpers


#region Lane Collapsing Helpers

def get_dense_representation_of_gate_with_perfect_swap_gates(model, op: LabelTup, saved: dict[int | LabelTup | LabelTupTup, _np.ndarray], swap_dense: _np.ndarray) -> _np.ndarray:
    """
    Assumes that a gate which operates on 2 qubits does not have the right orientation if label is (qu_{i+1}, qu_i).
    """
    if op.num_qubits == 2:
        # We may need to do swaps.
        op_term : _np.ndarray = _np.array([1.])
        if op in saved:
            op_term = saved[op]
        elif op.qubits[1] < op.qubits[0]:  # type: ignore
            # This is in the wrong order.
            op_term = model._layer_rules.get_dense_process_matrix_represention_for_gate(model, op)
            op_term = swap_dense @ (op_term) @ swap_dense.T
            saved[op] = op_term # Save so we only need to this operation once.
        else:
            op_term = model._layer_rules.get_dense_process_matrix_represention_for_gate(model, op)
        return op_term
    return model._layer_rules.get_dense_process_matrix_represention_for_gate(model, op)

def get_dense_op_of_gate_with_perfect_swap_gates(model, op: LabelTup, saved: dict[int | LabelTup | LabelTupTup, _np.ndarray], swap_dense: _np.ndarray):
    """
    Assumes that a gate which operates on 2 qubits does not have the right orientation if label is (qu_{i+1}, qu_i).
    """
    return model._layer_rules.get_dense_process_matrix_represention_for_gate(model, op)


#endregion Lane Collapsing Helpers


class EvalTreeBasedUponLongestCommonSubstring():

    def __init__(self, circuit_list: list[LabelTupTup], qubit_starting_loc: int = 0):
        """
        Construct an evaluation order tree for a circuit list that minimizes the number of rounds of computation.
        """

        self.circuit_to_save_location = {tuple(cir): i for i,cir in enumerate(circuit_list)}

        self.orig_circuits = {i: circuit_list[i] for i in range(len(circuit_list))}
        self.qubit_start_point = qubit_starting_loc


        internal_matches = create_tables_for_internal_LCS(circuit_list)
        best_internal_match = _np.max(internal_matches[0])

        max_rounds = best_internal_match

        C = len(circuit_list)
        sequence_intro = {0: _np.arange(C)}

        cache_pos = C
        cache = {i: circuit_list[i] for i in range(len(circuit_list))}

        new_circuit_list = [cir for cir in circuit_list] # Get a deep copy since we will modify it here.

        # Let's try simplifying internally first.
        self.internal_first = False  # TODO: Fix.
        seq_ind_to_cache_index = {i: i for i in range(C)}
        if self.internal_first:
            i = 0
            cache_pos = -1
            while max_rounds > 1:

                tmp = simplify_internal_first_one_round(new_circuit_list, 
                                                        internal_matches,
                                                        cache_pos,
                                                        cache,
                                                        seq_ind_to_cache_index)
                new_circuit_list, cache_pos, cache, sequence_intro[i-1] = tmp
                i -= 1
                internal_matches = create_tables_for_internal_LCS(new_circuit_list)

                max_rounds = _np.max(internal_matches[0])
        external_matches = _compute_lcs_for_every_pair_of_sequences(new_circuit_list,
                                                                    None,
                                                                    None,
                                                                    set(_np.arange(len(new_circuit_list))),
                                                                    max([len(cir) for cir in new_circuit_list])-1)


        best_external_match = _np.max(external_matches[0])

        max_rounds = int(max(best_external_match,best_internal_match))
        i = 0
        cache_pos = len(new_circuit_list)
        while max_rounds > 1:
            tmp = conduct_one_round_of_lcs_simplification(new_circuit_list,
                                                          external_matches,
                                                          internal_matches,
                                                          cache_pos,
                                                          cache,
                                                          seq_ind_to_cache_index)
            new_circuit_list, cache_pos, cache, sequence_intro[i+1], ext_table, external_sequences, dirty_inds = tmp
            i += 1
            dirty_inds = set(_np.arange(len(new_circuit_list))) # TODO: fix to only correct those which are actually dirty.
            external_matches = _compute_lcs_for_every_pair_of_sequences(new_circuit_list,
                                                                        ext_table,
                                                                        external_sequences,
                                                                        dirty_inds,
                                                                        max_rounds)

            if best_internal_match < best_external_match and best_external_match < 2 * best_internal_match:
                # We are not going to get a better internal match.
                pass
            elif not self.internal_first:
                internal_matches = create_tables_for_internal_LCS(new_circuit_list)

            best_external_match = _np.max(external_matches[0])
            best_internal_match = _np.max(internal_matches[0])

            max_rounds = int(max(best_external_match,best_internal_match))

        self.cache = cache
        self.num_circuits = C
        self.from_other = False

        self.sequence_intro = sequence_intro

        from pygsti.modelmembers.operations import StaticStandardOp
        self.swap_gate = StaticStandardOp('Gswap', basis='pp').to_dense().round(16)

        self.cache_ind_to_alphabet_vals_referenced: dict[int, set[LabelTupTup]] = {}


        # Useful for repeated calculations seen in a derivative calculation.
        for key in self.cache:
            self.compute_depends_on(key, self.cache_ind_to_alphabet_vals_referenced)

        alphabet_val_to_cache_inds_to_update: dict[LabelTup, set[int]] = {}

        for cache_ind, vals in self.cache_ind_to_alphabet_vals_referenced.items():
            for val in vals:
                if isinstance(val, LabelTupTup):
                    for ind_gate in val:
                        if ind_gate in alphabet_val_to_cache_inds_to_update:
                            alphabet_val_to_cache_inds_to_update[ind_gate].add(cache_ind)
                        else:
                            alphabet_val_to_cache_inds_to_update[ind_gate] = set([cache_ind])
                else:
                    if val in alphabet_val_to_cache_inds_to_update:
                        alphabet_val_to_cache_inds_to_update[val].add(cache_ind)
                    else:
                        alphabet_val_to_cache_inds_to_update[val] = set([cache_ind])

        self.results: dict[int | LabelTupTup, _np.ndarray] = {}

        self.alphabet_val_to_sorted_cache_inds: dict[LabelTup, list[int]] = {}

        self.gpindex_to_cache_vals: dict[int, tuple[list[int], list[Label]]] = {}
        # This will be filled later by _gpindex_to_cache_inds_needed_to_recompute when we have access to the model.
        # Warning that changing the model paramvec will result in this cache becoming invalidated.
        # The user is currently in charge of resetting this cache.

        for val, cache_inds in alphabet_val_to_cache_inds_to_update.items():
            rnd_nums = {}
            for cache_ind in cache_inds:
                for rnd_num in self.sequence_intro:
                    if cache_ind in self.sequence_intro[rnd_num]:
                        rnd_nums[cache_ind] = rnd_num
                        break

            sorted_inds = sorted(cache_inds, key =lambda x : rnd_nums[x])[::-1] # We want to iterate large to small.

            self.alphabet_val_to_sorted_cache_inds[val] = sorted_inds

      
    def _gpindex_to_cache_inds_needed_to_recompute(self, model, gp_index_changing: int) -> list[int]:
        """
        Given that I change the representation of a gate by modifying this index, gp_index_changing,
        what cache indices do I need to recompute and in what order.
        """
        if gp_index_changing in self.gpindex_to_cache_vals:
            return self.gpindex_to_cache_vals[gp_index_changing]

        cache_inds: list[int] = []
        all_op_inds: list[int] = []
        invalid_lbls: list[Label] = []
        for lbl in self.alphabet_val_to_sorted_cache_inds.keys():
            # my_op = get_dense_op_of_gate_with_perfect_swap_gates(model, lbl, None, None)
            try:
                my_op = model.circuit_layer_operator(lbl, "op") # Assumes that layers have the same gpindices as the gates themselves.
            except KeyError:
                # Skip to the next lbl to check. Do not immediately return None!
                continue
            op_inds = my_op.gpindices_as_array()
            if gp_index_changing in op_inds:
                cache_inds.extend(self.alphabet_val_to_sorted_cache_inds[lbl])
                invalid_lbls.append(lbl)  # We also invalidate the lbl.
                all_op_inds.extend(op_inds)

        for ind in all_op_inds:
            self.gpindex_to_cache_vals[ind] = (cache_inds, invalid_lbls)
        return cache_inds, invalid_lbls
    
    def _which_full_circuits_will_change_due_to_gpindex_changing(self, model, gp_index_changing: int) -> list[int]:

        cache_inds, _ = self._gpindex_to_cache_inds_needed_to_recompute(model, gp_index_changing)

        if len(cache_inds) == 0:
            return []
        
        answer = [ind for ind in range(self.num_circuits) if ind in cache_inds]
        return answer


    def from_other_eval_tree(self, other: EvalTreeBasedUponLongestCommonSubstring, qubit_label_exchange: dict[int, int]):
        """
        Construct a tree from another tree.
        """
        
        self.cache = other.cache
        self.num_circuits = other.num_circuits
        self.sequence_intro = other.sequence_intro
        self.swap_gate = other.swap_gate
        self.orig_circuit_list = other.orig_circuit_list
        self.circuit_to_save_location = other.circuit_to_save_location
        self.from_other = other

        for ind in self.cache:
            for i, term in enumerate(self.cache[ind]):
                if isinstance(term, int):
                    pass # The tree will stay the same.
                elif isinstance(term, LabelTupTup):
                    new_term = ()
                    for op in term:
                        new_qu = (qubit_label_exchange[qu] for qu in op.qubits)
                        new_op = (op.name, *new_qu)
                        new_term = (*new_term, new_op)
                    self.cache[ind][i] = Label(new_term)

        
        for icir in range(len(self.orig_circuit_list)):
            self.orig_circuit_list[icir] = self.trace_through_cache_to_build_circuit(icir)

        updated = {}
        for cir, loc in self.circuit_to_save_location.items():
            new_cir = ()
            for layer in cir:
                new_layer = ()
                for op in layer:
                    new_op = (op[0], *(qubit_label_exchange[qu] for qu in op[1:]))
                    new_layer = (*new_layer, new_op)
                new_cir = (*new_cir, new_layer)
            updated[new_cir] = loc
        self.circuit_to_save_location = updated

    def collapse_circuits_to_process_matrices(self, model, num_qubits_in_default: int, gp_index_changing: Optional[int] = None):
        """
        Compute the total product cache. Note that this may still have a tensor product
        structure that the operator needs to combine again if they want to have the full 'dense' matrix.

        If gp_index_changing is not None then we have already computed the results once and we only need to update
        those terms which depend on the specific gp_index.
        """

        if gp_index_changing is not None:

            # Dig through the tree to see if we have a matching

            cache_inds, invalid_lbls = self._gpindex_to_cache_inds_needed_to_recompute(model, gp_index_changing)

            if cache_inds:
                # Invalidate all gate labels that we saved just in case.
                # Invalidate every index in the which we know to be influenced by my_op.
                # local_changes = {k: v for k, v in self.results.items() \
                #                     if ((k not in cache_inds) and (not isinstance(k, Label)))} # Could just invalidate only the lbl with the index.

                # Iterating over all the cache will take too long.
                # So we need to handle the invalidness of certain cache inds when we encounter them.
                local_changes = {}

                # Ignore the last index which is the Label that matched the gpindex.
                # We assume that only one will match.
                for cache_ind in cache_inds:
                    cumulative_term = None
                    for term in self.cache[cache_ind]:
                        cumulative_term = self._collapse_cache_line(model, cumulative_term, term, local_changes,
                                                                    num_qubits_in_default, cache_inds, invalid_lbls)

                    # Save locally.
                    if cumulative_term is None:
                        local_changes[cache_ind] = _np.eye(4**num_qubits_in_default)
                        # NOTE: unclear when (if ever) this should be a noisy idle gate.
                    else:
                        local_changes[cache_ind] = cumulative_term

                return local_changes, self.circuit_to_save_location

            return self.results, self.circuit_to_save_location

        else:
            self.results = {} # We are asking to reset all the calculations.
            round_keys = sorted(_np.unique(list(self.sequence_intro.keys())))[::-1]
            # saved: dict[int | LabelTupTup, _np.ndarray] = {}

            if self.internal_first:

                round_keys = _np.unique(list(self.sequence_intro.keys()))

                pos_inds = _np.where(round_keys >0)
                pos_keys = round_keys[pos_inds]
                pos_keys = sorted(pos_keys)[::-1]

                neg_inds = _np.where(round_keys < 0)
                neg_keys = round_keys[neg_inds]
                neg_keys = sorted(neg_keys)

                round_keys = pos_keys + neg_keys + _np.array([0])
            
            empty = []
            for key in round_keys:
                for cache_ind in self.sequence_intro[key]:
                    cumulative_term = None
                    for term in self.cache[cache_ind]:
                        cumulative_term = self._collapse_cache_line(model, cumulative_term, term, self.results,
                                                                    num_qubits_in_default, empty, empty)
                            
                    if cumulative_term is None:
                        self.results[cache_ind] = _np.eye(4**num_qubits_in_default)
                        # NOTE: unclear when (if ever) this should be a noisy idle gate.
                    else:
                        self.results[cache_ind] = cumulative_term
        if __debug__:
            # We may store more in the cache in order to handle multi-qubit gates which are out of the normal order.
            for key in self.cache:
                assert key in self.results
        
        # {tuple(self.trace_through_cache_to_build_circuit(icir)): icir for icir in range(len(self.orig_circuit_list)) if icir < self.num_circuits}
        return self.results, self.circuit_to_save_location
    
    def compute_depends_on(self, val: int | LabelTupTup, visited: dict[int, set[LabelTupTup]]) -> set[LabelTupTup]:

        if not isinstance(val, int):
            return set([val])
        elif val in visited:
            return visited[val]
        else:
            tmp = set()
            for child in self.cache[val]:
                ret_val = self.compute_depends_on(child, visited)
                tmp = tmp.union(ret_val)
            visited[val] = tmp
            return tmp


    def combine_for_visualization(self, val, visited):

        if not isinstance(val, int):
            return [val]
        elif val in visited:
            return visited[val]
        else:
            tmp = []
            for child in self.cache[val]:
                tmp.append(self.combine_for_visualization(child, visited))
            visited[val] = tmp
            return tmp

    def handle_results_cache_lookup_and_product(self,
                            cumulative_term: None | _np.ndarray,
                            term_to_extend_with: int | LabelTupTup,
                            results_cache: dict[int | LabelTupTup, _np.ndarray]) -> _np.ndarray:

        if cumulative_term is None:
            return results_cache[term_to_extend_with]
        return results_cache[term_to_extend_with] @ cumulative_term

        if isinstance(term_to_extend_with, int):
            if term_to_extend_with in globally_invalid_cache_inds[:-1]:
                # look up the result in the local results cache.
                # This is just for that derivative step.
                if cumulative_term is None:
                    return results_cache[term_to_extend_with]
                return results_cache[term_to_extend_with] @ cumulative_term
        else:
            if term_to_extend_with in globally_invalid_cache_inds[-1:]:
                # Only one label gets invalidated and that is stored at the end of the list.

                # look up the result in the local results cache.
                # This is just for that derivative step.
                if cumulative_term is None:
                    return results_cache[term_to_extend_with]
                return results_cache[term_to_extend_with] @ cumulative_term
        
        # We should use the cache for all the probs calculation.
        if cumulative_term is None:
            # look up result.
            return self.results[term_to_extend_with]
        return self.results[term_to_extend_with] @ cumulative_term 


    def _collapse_cache_line(self, model, cumulative_term: None | _np.ndarray,
                            term_to_extend_with: int | LabelTupTup,
                            local_results_cache: dict[int | LabelTupTup, _np.ndarray],
                            num_qubits_in_default: int,
                            globally_invalid_cache_inds: Optional[list[int]] = None,
                            globally_invalid_labels: Optional[list[LabelTupTup]] = None
                            ) -> _np.ndarray:
        """
        Reduce a cache line to a single process matrix.

        This should really only be called from collapse_circuits_to_process_matrices.

        """

        if (term_to_extend_with in local_results_cache):
            return self.handle_results_cache_lookup_and_product(cumulative_term,
                                                                term_to_extend_with,
                                                                local_results_cache)
        elif isinstance(term_to_extend_with, int) and \
            (globally_invalid_cache_inds is not None) and \
            (term_to_extend_with not in globally_invalid_cache_inds) and \
                (term_to_extend_with in self.results):
            
            return self.handle_results_cache_lookup_and_product(cumulative_term,
                                                                term_to_extend_with,
                                                                self.results)
        elif isinstance(term_to_extend_with, LabelTupTup) and \
            (globally_invalid_labels is not None) and \
            not (any([t in globally_invalid_labels for t in term_to_extend_with])) \
                and (term_to_extend_with in self.results):        
            return self.handle_results_cache_lookup_and_product(cumulative_term,
                                                                term_to_extend_with,
                                                                self.results)
        
        # elif isinstance(term_to_extend_with, LabelTup) and \
        #     (term_to_extend_with not in globally_invalid_cache_inds[-1:]) \
        #         and (term_to_extend_with in self.results):        
        #     return self.handle_results_cache_lookup_and_product(cumulative_term, term_to_extend_with,
        #                                                         local_results_cache, globally_invalid_cache_inds)

        else:
            val = 1
            qubits_available = [i + self.qubit_start_point for i in range(num_qubits_in_default)]
            if isinstance(term_to_extend_with, int):
                breakpoint()
            matrix_reps = {op.qubits: get_dense_representation_of_gate_with_perfect_swap_gates(model, op,
                                            local_results_cache, self.swap_gate) for op in term_to_extend_with}
            qubit_used = []
            for key in matrix_reps.keys():
                qubit_used.extend(key)

            assert len(qubit_used) == len(set(qubit_used))
            unused_qubits = set(qubits_available) - set(qubit_used)

            implicit_idle_reps = {(qu,): get_dense_representation_of_gate_with_perfect_swap_gates(model,
                                        Label("Fake_Gate_To_Get_Tensor_Size_Right", qu), # A fake gate to look up and use the appropriate idle gate.
                                        local_results_cache, self.swap_gate) for qu in unused_qubits}

            while qubits_available:

                qu = qubits_available[0]
                if qu in unused_qubits:
                    val = _np.kron(val, implicit_idle_reps[(qu,)])
                    qubits_available = qubits_available[1:]
                else:
                    # It must be a part of a non-trivial gate.
                    gatekey = [key for key in matrix_reps if qu in key][0]
                    val = _np.kron(val, matrix_reps[gatekey])

                    qubits_available = qubits_available[len(gatekey):]

            local_results_cache[term_to_extend_with] = val
            if cumulative_term is None:
                return val
            # Cache if off.
            return local_results_cache[term_to_extend_with] @ cumulative_term


    def trace_through_cache_to_build_circuit(self, cache_ind: int) -> list[tuple]:

        output = ()
        for term in self.cache[cache_ind]:

            if isinstance(term, Label):
                output = (*output, term)
            elif isinstance(term, int):
                # Recurse down.
                next_term = self.trace_through_cache_to_build_circuit(term)
                output = (*output, *next_term)

        return list(output)

    def flop_cost_of_evaluating_tree(self, matrix_size: tuple[int, int], model = None, gp_index_changing: Optional[int] = None) -> int:
        """
        We assume that each matrix matrix multiply is the same size.
        """

        assert matrix_size[0] == matrix_size[1]

        total_flop_cost = 0
        if (model is not None) and (gp_index_changing is not None):

            cache_inds, invalid_lbls = self._gpindex_to_cache_inds_needed_to_recompute(model, gp_index_changing)

        else:
            cache_inds = list(self.cache.keys())

        for cache_ind in cache_inds:
            num_mm_on_this_cache_line = len(self.cache[cache_ind]) - 1
            total_flop_cost += (2* (matrix_size[0])**3) * num_mm_on_this_cache_line

        return total_flop_cost


class CollectionOfLCSEvalTrees():

    def __init__(self, line_lbls_to_circuit_list: dict[tuple[int, ...], list[LabelTupTup]],
                 sub_cir_to_full_cir_id_and_lane_id,
                 cir_id_and_lane_id_to_sub_cir):
        
        self.trees: dict[tuple[int, ...], EvalTreeBasedUponLongestCommonSubstring] = {}

        ASSUME_MATCHING_QUBIT_SIZE_MATCHING_TREE = False

        size_to_tree: dict[int, tuple[int, ...]] = {}

        self.line_lbls_to_cir_list = line_lbls_to_circuit_list

        starttime = _time.time()
        for key, vals in our_tqdm(line_lbls_to_circuit_list.items(), " Building Longest Common Substring Caches"):
            sub_cirs = []
            for cir in vals:
                sub_cirs.append(list(cir))
            if ASSUME_MATCHING_QUBIT_SIZE_MATCHING_TREE:
                if len(key) not in size_to_tree:
                    self.trees[key] = EvalTreeBasedUponLongestCommonSubstring(sub_cirs)
                    size_to_tree[len(key)] = key
                else:
                    sample = EvalTreeBasedUponLongestCommonSubstring(sub_cirs[:2]) # Build a small version to be corrected later.
                    other_key = size_to_tree[len(key)]
                    sample.from_other_eval_tree(self.trees[other_key], {other_key[i]: key[i] for i in range(len(key))})
                    self.trees[key] = sample
            else:
                self.trees[key] = EvalTreeBasedUponLongestCommonSubstring(sub_cirs, sorted(key)[0])

        endtime = _time.time()

        print(" Time to compute all the evaluation orders (s): ", endtime - starttime)


        self.sub_cir_to_full_cir_id_and_lane_id = sub_cir_to_full_cir_id_and_lane_id
        self.cir_id_and_lane_id_to_sub_cir = cir_id_and_lane_id_to_sub_cir

        self.cir_id_to_tensor_order: dict[int, list[list[int], int]] = {}
        self.compute_tensor_orders()

        self.saved_results: dict[Union[LabelTupTup, int], _np.ndarray] = {}
        self.sub_cir_to_ind_in_results: dict[tuple[int, ...], dict[_Circuit, int]] = {}
        self.original_matrices: dict[int, dict[int, _np.ndarray]] = {}
        self.full_matrices: list[KronStructured] = []
        self.process_matrices_which_will_need_to_update_for_index: _np.ndarray = []

    def do_I_need_to_recompute_portions_if_I_change_this_index(self, model, gp_index_changing: int) -> bool:

        for key in self.trees:
            inds, lbls = self.trees[key]._gpindex_to_cache_inds_needed_to_recompute(model, gp_index_changing)
            if len(inds) > 0:
                return True
        return False


    def collapse_circuits_to_process_matrices(self, model, gp_index_changing: Optional[int] = None):
        """
        Collapse all circuits to their process matrices. If alphabet_piece_changing is not None, then
        we assume we have already collapsed this system once before and so only need to update part of the eval tree.
        """
        # Just collapse all of them.

        if gp_index_changing is not None:
            # We may not need to check all of the lanes.
            pass

        else:
            self.saved_results = {}
        
        for key in self.trees:
            num_qubits = len(key)
            tree = self.trees[key]
            out1, out2 = tree.collapse_circuits_to_process_matrices(model, num_qubits, gp_index_changing)
            # self.saved_results[key], self.sub_cir_to_ind_in_results[key] = self.trees[key].collapse_circuits_to_process_matrices(model, len(key))
            self.saved_results[key] = out1
            self.sub_cir_to_ind_in_results[key] = out2

    def determine_which_circuits_will_update_for_what_gpindices(self, model):

        dirty_circuits = _np.zeros((model.num_params, len(self.trees), len(self.cir_id_and_lane_id_to_sub_cir)))
        for ind in range(model.num_params):

            for ikey, key in enumerate(self.trees):
                dirty_circuits[ind, ikey, self.trees[key]._which_full_circuits_will_change_due_to_gpindex_changing(model, ind)] = 1

        self.process_matrices_which_will_need_to_update_for_index = dirty_circuits
        return dirty_circuits

    def reset_full_matrices_to_base_probs_version(self) -> None:
        """
        Any matrix which was updated previously reset to the original version.
        """

        for icir in self.original_matrices:
            for lane_in_cir in self.original_matrices[icir]:
                self.full_matrices[icir].update_operand(lane_in_cir, self.original_matrices[icir][lane_in_cir])
        self.original_matrices = {}
        return


    def reconstruct_full_matrices(self,
                                  model = None,
                                  gp_index_changing: Optional[int] = None) -> \
                                    Optional[Tuple[List[Union[KronStructured, _np.ndarray]], List[int]]]:
        """
        Construct a tensor product structure for each individual circuit
        """

        if len(self.saved_results) == 0:
            return

        num_cirs = len(self.cir_id_and_lane_id_to_sub_cir)
        cir_inds = _np.arange(num_cirs, dtype=_np.int32)
        if (gp_index_changing is not None) and (model is not None):
            
            cir_inds = _np.where(_np.sum(self.process_matrices_which_will_need_to_update_for_index[gp_index_changing], axis=0) >= 1)[0] # At least one lane changed.

            lane_key_to_ind: dict[tuple[int, ...], int] = {key: ikey for ikey, key in enumerate(self.trees)}

            output = []
            if len(cir_inds) > 0:
                self.original_matrices = {} # Reset the cache of updated process matrices.

            for icir in cir_inds:
                lane_circuits = []
                for i in range(len(self.cir_id_and_lane_id_to_sub_cir[icir])):
                    cir = self.cir_id_and_lane_id_to_sub_cir[icir][i]
                    lblkey = cir._line_labels

                    ind_in_results = self.sub_cir_to_ind_in_results[lblkey][cir.layertup]
                    if ind_in_results not in self.saved_results[lblkey]:
                        # We have only the local changes.
                        # This will be stored in the results file of the subtree.
                        lane_circuits.append(self.trees[lblkey].results[ind_in_results])
                    else:
                        lane_circuits.append(self.saved_results[lblkey][ind_in_results])
                if len(lane_circuits) > 1:
                    output.append(self.recurse_to_build_sparse_kron_matrix(lane_circuits))
                    # output.append(KronStructured(lane_circuits))
                elif len(lane_circuits) == 1:
                    output.append(lane_circuits[0]) # gate_sequence[i] @ rho needs to work for i in range(num_circs).
                else:
                    raise ValueError()
        
            return output, cir_inds

        else:
            output = []


            # Now we can do the combination.

            for icir in cir_inds:
                lane_circuits = []
                for i in range(len(self.cir_id_and_lane_id_to_sub_cir[icir])):
                    cir = self.cir_id_and_lane_id_to_sub_cir[icir][i]
                    lblkey = cir._line_labels

                    ind_in_results = self.sub_cir_to_ind_in_results[lblkey][cir.layertup]
                    if ind_in_results not in self.saved_results[lblkey]:
                        # We have only the local changes.
                        # This will be stored in the results file of the subtree.
                        lane_circuits.append(self.trees[lblkey].results[ind_in_results])
                    else:
                        lane_circuits.append(self.saved_results[lblkey][ind_in_results])
                if len(lane_circuits) > 1:
                    output.append(self.recurse_to_build_sparse_kron_matrix(lane_circuits))
                    # output.append(KronStructured(lane_circuits))
                elif len(lane_circuits) == 1:
                    output.append(lane_circuits[0]) # gate_sequence[i] @ rho needs to work for i in range(num_circs).
                else:
                    raise ValueError()
            
        self.full_matrices = output
        return output, cir_inds


    def recurse_to_build_sparse_kron_matrix(self, operands: list[_np.ndarray]):
        if len(operands) == 1:
            return operands[0]
        return sparse_kron(operands[0], self.recurse_to_build_sparse_kron_matrix(operands[1:]))


    def flop_estimate(self, return_collapse: bool = False, return_tensor_matvec: bool = False, model = None, gp_index_changing: Optional[int] = None):


        cost_collapse = 0
        for key in self.trees:
            num_qubits = len(key) if key[0] != ('*',) else key[1] # Stored in the data structure.
            tree = self.trees[key]
            cost_collapse += tree.flop_cost_of_evaluating_tree(tuple([4**num_qubits, 4**num_qubits]), model, gp_index_changing)
        

        tensor_cost = 0
        num_cirs = len(self.cir_id_and_lane_id_to_sub_cir)
        cir_inds = _np.arange(num_cirs, dtype=_np.int32)

        if (model is not None) and (gp_index_changing is not None):

            dirty_circuits = self.determine_which_circuits_will_update_for_what_gpindices(model)
            cir_inds = _np.where(_np.sum(self.process_matrices_which_will_need_to_update_for_index[gp_index_changing], axis=0) >= 1)[0] # At least one lane changed.

        
        for cir_id in cir_inds:
            qubit_list = ()
            for lane_id in range(len(self.cir_id_and_lane_id_to_sub_cir[cir_id])):
                subcir = self.cir_id_and_lane_id_to_sub_cir[cir_id][lane_id]
                qubit_list = (*qubit_list, len(subcir._line_labels))
            qubit_list = list(qubit_list)
            total_num = _np.sum(qubit_list)

            tensor_cost += cost_to_compute_tensor_matvec_without_reordering(qubit_list, total_num)

        if return_collapse:
            return tensor_cost + cost_collapse, cost_collapse
        elif return_tensor_matvec:
            return tensor_cost + cost_collapse, tensor_cost
        elif gp_index_changing is not None:
            return tensor_cost + cost_collapse, len(cir_inds)  # Since you are not updating all of the representations we do not need to update the state props either for those.

        return tensor_cost + cost_collapse

    def compute_tensor_orders(self):

        num_cirs = len(self.cir_id_and_lane_id_to_sub_cir)

        cache_struct = {}

        for cir_id in range(num_cirs):
            qubit_list = ()
            for lane_id in range(len(self.cir_id_and_lane_id_to_sub_cir[cir_id])):
                subcir = self.cir_id_and_lane_id_to_sub_cir[cir_id][lane_id]
                qubit_list = (*qubit_list, len(subcir._line_labels))
            self.cir_id_to_tensor_order[cir_id] = self.best_order_for_tensor_contraction(qubit_list, cache_struct)

        return
            
    def best_order_for_tensor_contraction(self,
                    qubit_list: tuple[int, ...],
                    cache: dict[tuple[int, ...], tuple[list[int], int]]) -> tuple[list[int], int]:
        """
        Find the tensor contraction order that minizes the cost of contracting to a dense system with
        a total number of qubits equal to the len(qubit_list)
        """
        

        if qubit_list in cache:
            return cache[qubit_list]

        best_cost = _np.inf
        best_order = []

        for order in itertools.permutations(range(len(qubit_list)-1), len(qubit_list)-1):

            my_list = [qb for qb in qubit_list] # force deep copy.
            my_starting_points = [sp for sp in order]
            cost = 0
            early_exit = False
            while my_starting_points and not early_exit:
                sp = my_starting_points.pop(0)

                cost += self._tensor_cost_model(my_list[sp], my_list[sp+1])
                if cost <= best_cost:
                    # modify sp for future.
                    tmp = []
                    for new_val in my_starting_points:
                        tmp.append((new_val - 1)*(new_val > sp) + (new_val) * (new_val < sp))
                    my_starting_points = tmp

                    q2 = my_list.pop(sp+1)
                    my_list[sp] += q2
                else:
                    early_exit = True # This round is done because the partial sum was too big.

            if cost < best_cost and not early_exit:
                best_cost = cost
                best_order = list(order)

        # Store off the information.
        cache[qubit_list] = best_order, best_cost

        return best_order, best_cost

    def _tensor_cost_model(self, num_qubits1, num_qubits2):
        """
        Assumes kronecker product of 2 square matrices.
        """

        return (4**num_qubits1)**2 * (4**num_qubits2)**2
    
    def _flop_estimate_to_collapse_to_each_circuit_to_process_matrix(self) -> tuple[int, list[int], list[int]]:
        """
        Compute the number of flops needed to collapse each circuit into a single process matrix.

        Returns:
        ---------
            cost - int total cost to collapse and reform
            collapse_lane_cost - list[int] cost to collapse a lane
            tensor_cost - list[int] cost to recombine a circuit into its full size.
        """


        num_cirs = len(self.cir_id_and_lane_id_to_sub_cir)

        collapse_lane_cost = []

        for lbl_key, my_tree in self.trees.items():
            collapse_lane_cost.append(my_tree.flop_cost_of_evaluating_tree([4**len(lbl_key), 4**len(lbl_key)]))

        tensor_cost = []
        for icir in range(num_cirs):
            
            _order, cost = self.cir_id_to_tensor_order[icir]
            tensor_cost.append(cost)

        return sum(tensor_cost) + sum(collapse_lane_cost), collapse_lane_cost, tensor_cost
    





def cost_to_compute_tensor_matvec_without_reordering(qubit_list: list[int], total_num_qubits: int) -> int:

    assert _np.sum(qubit_list) == total_num_qubits

    if len(qubit_list) == 1:
        # Basic matvec.
        cost = 2 * (4**qubit_list[0]**2)
        return cost
    
    elif len(qubit_list) == 2:
        # vec((A \tensor B) u) = vec(B U A.T)
        term1 = 2*(4**qubit_list[1]**2) * (4**qubit_list[0]) # MM of BU.
        term2 = 2 * (4**qubit_list[0]**2) * (4**qubit_list[1]) # MM of U A.T
        return term1 + term2
    
    else:
        # Just pop off the last term
        # (B_1 \tensor B_2 ... \tensor B_n) u = (B_n \tensor B_n-1 ... \tensor B_2) U (B_1).T

        right = cost_to_compute_tensor_matvec_without_reordering(qubit_list[:1], qubit_list[0])
        right *= 4**(_np.sum(qubit_list[1:]))
        left = cost_to_compute_tensor_matvec_without_reordering(qubit_list[1:],
                                                                total_num_qubits - qubit_list[0])
        left *= 4**(qubit_list[0])
        return left + right
    
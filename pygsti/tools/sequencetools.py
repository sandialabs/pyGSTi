"""
Tools for finding and using the longest common substrings in order to cache and evaluation order.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from typing import Sequence, Any, List, Tuple, MutableSequence, Optional
import numpy as _np
from tqdm import tqdm


def len_lcp(A: Sequence, B: Sequence) -> int:
    """
    Returns:
    -----
    int - the length of the longest matching prefix between A and B.
    """
    i = 0
    n = len(A)
    m = len(B)
    while i < n and i < m:
        if A[i] != B[i]:
            return i
        i += 1
    return i


def _lcs_dp_version(A: Sequence, B: Sequence) -> _np.ndarray:
    """
    Compute the longest common substring between A and B using
    dynamic programming.

    This will use O(n \times m) space and take O(n \times m \times max(m, n)) time.
    """
    table = _np.zeros((len(A) + 1, len(B) + 1))
    n, m = table.shape
    for i in range(n-2, -1, -1):
        for j in range(m-2, -1, -1):
            opt1 = 0
            if A[i] == B[j]:
                opt1 = len_lcp(A[i:], B[j:])
            opt2 = table[i, j+1]
            opt3 = table[i+1, j]
            table[i,j] = max(opt1, opt2, opt3)
    return table


def conduct_one_round_of_lcs_simplification(sequences: MutableSequence[MutableSequence[Any]],
                                            table_data_and_sequences: tuple[_np.ndarray, dict[tuple[int, int], Sequence[Any]]],
                                            internal_tables_and_sequences: tuple[_np.ndarray, dict[tuple[int, int], Sequence[Any]]],
                                            starting_cache_num: int,
                                            cache_struct: dict[int, Any],
                                            sequence_ind_to_cache_ind: Optional[dict[int, int]] = None):
    """
    Simplify the set of sequences by contracting the set of longest common subsequences.

    Will update the list of sequences and the cache struct to hold the longest common subsequences as new sequences.
    """
    if not sequence_ind_to_cache_ind:
        sequence_ind_to_cache_ind = {i: i for i in range(len(sequences))}
    if table_data_and_sequences:
        table, external_sequences = table_data_and_sequences
    else:
        table_cache = _np.zeros((len(sequences), len(sequences)))
        table, external_sequences = _compute_lcs_for_every_pair_of_sequences(sequences, table_cache,
                                                None, set(_np.arange(len(sequences))), "Unknown")

    if internal_tables_and_sequences:
        internal_subtable, internal_subsequences = internal_tables_and_sequences
    else:
        internal_subtable, internal_subsequences = create_tables_for_internal_LCS(sequences)

    best_index = _np.where(table == _np.max(table))
    best_internal_index = _np.where(internal_subtable == _np.max(internal_subtable))
    updated_sequences = [seq for seq in sequences]
    cache_num = starting_cache_num

    # Build sequence dict
    all_subsequences_to_replace: dict[tuple, dict[int, List[int]]] = {}

    if _np.max(internal_subtable) >= _np.max(table):
        # We are only going to replace if this was the longest substring.
        for cir_ind in best_internal_index[0]:
            for seq in internal_subsequences[cir_ind]:
                key = tuple(seq)
                if key in all_subsequences_to_replace:
                    all_subsequences_to_replace[key][cir_ind] = internal_subsequences[cir_ind][seq]
                else:
                    all_subsequences_to_replace[key] = {cir_ind: internal_subsequences[cir_ind][seq]}

    if _np.max(table) >= _np.max(internal_subtable):
        for ii in range(len(best_index[0])):
            cir_index = best_index[0][ii]
            cir_index2 = best_index[1][ii]
            starting_point, starting_point_2, length = external_sequences[(cir_index, cir_index2)]
            seq = updated_sequences[cir_index][starting_point: int(starting_point + length)]

            key = tuple(seq)
            if key in all_subsequences_to_replace:
                if cir_index not in all_subsequences_to_replace[key]:
                    # We did not already handle this with internal subsequences.
                    all_subsequences_to_replace[key][cir_index] = [starting_point]
                if cir_index2 not in all_subsequences_to_replace[key]:
                    all_subsequences_to_replace[key][cir_index2] = [starting_point_2]

            else:
                all_subsequences_to_replace[key] = {cir_index: [starting_point], cir_index2: [starting_point_2]}


    # Handle the updates.
    old_cache_num = cache_num
    dirty_inds = set()
    for seq, cdict in all_subsequences_to_replace.items():
        w = len(seq)
        update_made = 0
        if  w > 1 or (not isinstance(seq[0], int)):
            # We have reached an item which we can just compute.
            for cir_ind in cdict:
                my_cir = updated_sequences[cir_ind]
                sp = 0
                while sp+w <= len(my_cir):
                    if list(my_cir[sp: sp+w]) == list(seq):
                        my_cir[sp: sp + w] = [cache_num]
                        dirty_inds.add(cir_ind)
                        update_made = 1

                    sp += 1
                updated_sequences[cir_ind] = my_cir

                cache_struct[sequence_ind_to_cache_ind[cir_ind]] = updated_sequences[cir_ind]

            if update_made:
                # There may have been multiple overlapping subsequences in the same sequence.
                # (e.g. QWEQWEQWERQWE has QWE, WEQ, and EQW all happen and all are length 3 subsequences.)
                updated_sequences.append(list(seq))
                cache_struct[cache_num] = updated_sequences[cache_num]

                # This is a new sequence index which will need to be updated.
                dirty_inds.add(cache_num)
                sequence_ind_to_cache_ind[cache_num] = cache_num
                cache_num += 1

    assert cache_num >= old_cache_num
    assert old_cache_num >=0
    sequences_introduced_in_this_round = _np.arange(cache_num - old_cache_num) + old_cache_num


    return updated_sequences, cache_num, cache_struct, sequences_introduced_in_this_round, table, external_sequences, dirty_inds

def simplify_internal_first_one_round(sequences: MutableSequence[MutableSequence[Any]],
            internal_tables_and_sequences: tuple[_np.ndarray, dict[tuple[int, int], Sequence[Any]]],
            starting_cache_num: int,
            cache_struct: dict[int, Any],
            seq_ind_to_cache_ind: Optional[dict[int, int]]):
    """
    Simplify the set of sequences by contracting the set of longest common subsequences internal subsequences.

    e.g. ["AAAA"] will be replaced with cache_num cache_num. But ["BAR", "BAC"] will not update here because "BA" is split between 2 sequences.

    Will update the list of sequences and the cache struct to hold the longest common subsequences as new sequences.
    
    Cache number will decrement so ensure that cache_struct can handle positives and negatives.
    """
    if not seq_ind_to_cache_ind:
        seq_ind_to_cache_ind = {i: i for i in range(len(sequences))}

    if internal_tables_and_sequences:
        internal_subtable, internal_subsequences = internal_tables_and_sequences
    else:
        internal_subtable, internal_subsequences = create_tables_for_internal_LCS(sequences)

    best_internal_index = _np.where(internal_subtable == _np.max(internal_subtable))
    updated_sequences = [seq for seq in sequences]
    cache_num = starting_cache_num

    # Build sequence dict
    all_subsequences_to_replace: dict[tuple, dict[int, List[int]]] = {}

    # We are only going to replace if this was the longest substring.
    for cir_ind in best_internal_index[0]:
        for seq in internal_subsequences[cir_ind]:
            key = tuple(seq)
            if key in all_subsequences_to_replace:
                all_subsequences_to_replace[key][cir_ind] = internal_subsequences[cir_ind][seq]
            else:
                all_subsequences_to_replace[key] = {cir_ind: internal_subsequences[cir_ind][seq]}

    # Handle the updates.
    old_cache_num = cache_num
    for seq, cdict in all_subsequences_to_replace.items():
        w = len(seq)
        update_made = 0
        if  w > 1 or (not isinstance(seq[0], int)):
            # We have reached an item which we can just compute.
            for cir_ind in cdict:
                my_cir = updated_sequences[cir_ind]
                sp = 0
                while sp+w <= len(my_cir):
                    if list(my_cir[sp: sp+w]) == list(seq):
                        my_cir[sp: sp + w] = [cache_num]
                        update_made = 1

                    sp += 1
                updated_sequences[cir_ind] = my_cir

                cache_struct[seq_ind_to_cache_ind[cir_ind]] = updated_sequences[cir_ind]

            if update_made:
                # There may have been multiple overlapping subsequences in the same sequence.
                # (e.g. QWEQWEQWERQWE has QWE, WEQ, and EQW all happen and all are length 3 subsequences.)
                updated_sequences.append(list(seq))
                cache_struct[cache_num] = list(seq) # Add the new sequence to the cache.

                # Add a new mapping from sequences to cache index.
                seq_ind_to_cache_ind[len(updated_sequences)-1] = cache_num

                cache_num += -1

    # Cache num and old_cache_num < 0
    assert cache_num < 0
    assert old_cache_num < 0
    assert old_cache_num > cache_num
    sequences_introduced_in_this_round = _np.arange(_np.abs(cache_num - old_cache_num))*-1 + old_cache_num

    return updated_sequences, cache_num, cache_struct, sequences_introduced_in_this_round



def _find_starting_positions_using_dp_table(
        dp_table: _np.ndarray
    ) -> tuple[int, int, int] | Tuple[None, None, None]:
    """
    Finds the starting positions for the longest common subsequence.

    Returns:
    ---------
    int - starting index in A of LCS(A,B)
    int - starting index in B of LCS(A,B)
    int - length of LCS(A,B)
    """
    n, m = dp_table.shape
    i = 0
    j = 0
    while i < n-1 and j < m -1:
        curr = dp_table[i,j]
        opt1 = dp_table[i+1, j+1] # Use
        opt2 = dp_table[i+1, j] # Eliminate A prefix
        opt3 = dp_table[i, j+1] # Eliminate B prefix
        options = [opt1, opt2, opt3]
        if _np.all(curr == options):
            i += 1
            j += 1
        elif opt2 > opt1 and opt2 > opt3:
            i += 1
        elif opt3 > opt2 and opt3 > opt1:
            j += 1
        else:
            # All three options are equal. So we should march the diagonal.
            return i, j, dp_table[0,0]
    return None, None, None

def _lookup_in_sequence_cache(seq_cache: dict[tuple[int, int], tuple], i: int, j: int) -> tuple:

    if seq_cache:
        return seq_cache[(i, j)]
    return (None, None, None)


def _compute_lcs_for_every_pair_of_sequences(sequences: MutableSequence[Any],
                                             table_cache: _np.ndarray,
                                             seq_cache: dict,
                                             dirty_inds: set,
                                             expected_best: int):
    """
    Computes the LCS for every pair of sequences A,B in sequences
    """
    best_subsequences = {}
    best_lengths = _np.zeros((len(sequences), len(sequences)))
    curr_best = 2  # We want only subsequences that have at least two characters matching.
    for i in tqdm(range(len(sequences)-1, -1, -1),
                  f"LCS_circuits Expected Val {expected_best}: ", disable = True): # Lets do this in reverse order
        cir0 = sequences[i]
        for j in range(i-1, -1, -1):
            cir1 = sequences[j]
            if i in dirty_inds or j in dirty_inds:
                if len(cir0) < curr_best or len(cir1) < curr_best:
                    # Mark pair as dirty to be computed later when it may be the longest subsequence.
                    best_lengths[i,j] = -1
                    best_subsequences[(i,j)] = (None, None, None)
                else:
                    table = _lcs_dp_version(cir0, cir1)
                    best_lengths[i,j] = table[0,0]
                    best_subsequences[(i,j)] = _find_starting_positions_using_dp_table(table)
                    curr_best = max(best_lengths[i,j], curr_best)
            else:
                best_lengths[i,j] = table_cache[i,j]
                best_subsequences[(i,j)] = _lookup_in_sequence_cache(seq_cache, i, j)

    return best_lengths, best_subsequences


def _longest_common_internal_subsequence(A: Sequence) -> tuple[int, dict[tuple, set[int]]]:
    """
    Compute the longest common subsequence within a single circuit A.

    Cost ~ O(L^3 / 8) where L is the length of A

    Returns:
    ---------
    int - length of longest common subsequences within A
    dict[tuple, set[int]] - dictionary of subsequences to starting positions within A.
    """
    n = len(A)
    best = 0
    best_ind : dict[tuple[Any,...], set[int]] = dict()
    changed = False
    for w in range(1, int(_np.floor(n / 2) + 1)):
        for sp in range(n - w):
            window = A[sp: sp + w]
            for match in range(sp+ w, n-w + 1):
                if A[match: match + w] == window:
                    if best == w:
                        if tuple(window) in best_ind:
                            best_ind[tuple(window)].add(match)
                        else:
                            best_ind[tuple(window)] = {sp, match}
                    else:
                        best_ind = {tuple(window): {sp, match}}
                        changed = True
                        best = w
        if not changed:
            return best, best_ind
    return best, best_ind


def create_tables_for_internal_LCS(
        sequences: Sequence[Sequence[Any]]
    ) -> tuple[
        _np.ndarray, List[dict[tuple[Any,...], set[int]]]
    ]:
    """
    Compute all the longest common internal sequences for each circuit A in sequences

    Total cost is O(C L^3).
    """

    C = len(sequences)
    the_table = _np.zeros(C)
    seq_table : List[dict[tuple[Any,...], set[int]]] = [dict() for _ in range(C)]

    curr_best = 1
    for i in range(C):
        if len(sequences[i]) >= 2*curr_best:
            the_table[i], seq_table[i] = _longest_common_internal_subsequence(sequences[i])
            curr_best = max(curr_best, the_table[i])
    return the_table, seq_table

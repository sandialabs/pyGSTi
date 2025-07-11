from typing import Sequence
import numpy as _np


#region Longest Common Subsequence

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


def conduct_one_round_of_lcs_simplification(sequences: list[Sequence], table_data_and_sequences,
                                            internal_tables_and_sequences,
                                            starting_cache_num,
                                            cache_struct):
    """
    Simplify the set of sequences by contracting the set of longest common subsequences.

    Will update the list of sequences and the cache struct to hold the longest common subsequences as new sequences.
    """
    if table_data_and_sequences:
        table, external_sequences = table_data_and_sequences
    else:
        table, external_sequences = _compute_lcs_for_every_pair_of_sequences(sequences)

    if internal_tables_and_sequences:
        internal_subtable, internal_subsequences = internal_tables_and_sequences
    else:
        internal_subtable, internal_subsequences = create_tables_for_internal_LCS(sequences)

    best_index = _np.where(table == _np.max(table))
    best_internal_index = _np.where(internal_subtable == _np.max(internal_subtable))
    updated_sequences = [seq for seq in sequences]
    cache_num = starting_cache_num

    # Build sequence dict
    all_subsequences_to_replace: dict[tuple, dict[int, list[int]]] = {}

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
    for seq, cdict in all_subsequences_to_replace.items():
        w = len(seq)
        if  w > 1 or (not isinstance(seq[0], int)):
            # We have reached an item which we can just compute.
            for cir_ind in cdict:
                my_cir = updated_sequences[cir_ind]
                sp = 0
                while sp+w <= len(my_cir):
                    if list(my_cir[sp: sp+w]) == list(seq):
                        my_cir[sp: sp + w] = [cache_num]

                    sp += 1
                updated_sequences[cir_ind] = my_cir

                cache_struct[cir_ind] = updated_sequences[cir_ind]

            updated_sequences.append(list(seq))
            cache_struct[cache_num] = updated_sequences[cache_num]

            cache_num += 1

    sequences_introduced_in_this_round = _np.arange(cache_num - old_cache_num) + old_cache_num

    return updated_sequences, cache_num, cache_struct, sequences_introduced_in_this_round


def _find_starting_positions_using_dp_table(dp_table: _np.ndarray) -> tuple[int, int, int]:
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


def _compute_lcs_for_every_pair_of_sequences(sequences: list):
    """
    Computes the LCS for every pair of sequences A,B in sequences
    """
    best_subsequences = {}
    best_lengths = _np.zeros((len(sequences), len(sequences)))
    curr_best = 0
    for i in range(len(sequences)-1, -1, -1): # Lets do this in reverse order
        cir0 = sequences[i]
        if len(cir0) >= curr_best:
            # Could be the best.
            for j in range(i-1, -1, -1):
                cir1 = sequences[j]
                if len(cir1) >= curr_best:
                    table = _lcs_dp_version(cir0, cir1)
                    best_lengths[i,j] = table[0,0]
                    best_subsequences[(i,j)] = _find_starting_positions_using_dp_table(table)
                    curr_best = max(best_lengths[i,j], curr_best)
                else:
                    best_lengths[i,j] = -1
                    best_subsequences[(i,j)] = (None, None, None)
        else:
            # Skipped because cannot be the best yet.
            best_lengths[i,j] = -1
            best_subsequences[(i,j)] = (None, None, None)
    return best_lengths, best_subsequences


def _longest_common_internal_subsequence(A: Sequence) -> tuple[int, dict[tuple, list[int]]]:
    """
    Compute the longest common subsequence within a single circuit A.

    Cost ~ O(L^3 / 8) where L is the length of A

    Returns:
    ---------
    int - length of longest common subsequences within A
    dict[tuple, list[int]] - dictionary of subsequences to starting positions within A.
    """
    n = len(A)
    best = 0
    best_ind = {}
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


def create_tables_for_internal_LCS(sequences: list[Sequence]) -> tuple[_np.ndarray,
                                                        list[dict[tuple, list[int]]]]:
    """
    Compute all the longest common internal sequences for each circuit A in sequences

    Total cost is O(C L^3).
    """

    C = len(sequences)
    the_table = _np.zeros(C)
    seq_table = [[] for _ in range(C)]

    curr_best = 1
    for i in range(C):
        if len(sequences[i]) >= 2*curr_best:
            the_table[i], seq_table[i] = _longest_common_internal_subsequence(sequences[i])
            curr_best = max(curr_best, the_table[i])
    return the_table, seq_table

#endregion Longest Common Subsequence

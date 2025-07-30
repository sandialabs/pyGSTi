import numpy as np
from pygsti.tools.sequencetools import _compute_lcs_for_every_pair_of_sequences, create_tables_for_internal_LCS
from pygsti.tools.sequencetools import conduct_one_round_of_lcs_simplification
from pygsti.tools.sequencetools import simplify_internal_first_one_round

def test_external_matches():

    my_strings = ["ABAARCR12LIO", "QWERTYASDFGH", "QWEELLKJAT"]

    tables, sequences = _compute_lcs_for_every_pair_of_sequences(my_strings, None, None, set([0,1,2]), 3)

    assert np.max(tables) == 3

    assert len(np.where(np.max(tables) == tables)[0]) == 1 # There is only one sequence present in this case.


    if (1,2) in sequences:
        assert sequences[(1,2)] == (0, 0, 3)
    else:
        assert (2,1) in sequences
        assert sequences[(2,1)] == (0, 0, 3)


def test_internal_matches():

    my_strings = ["RACECAR", "AAAAQAAAA", "QWERTYQWEQWEQWE"]

    tables, sequences = create_tables_for_internal_LCS(my_strings)

    assert np.max(tables) == 4


    assert sequences[1][tuple("AAAA")] == {0, 5}


    my_strings = [my_strings[0]] + [my_strings[2]]

    tables, sequences = create_tables_for_internal_LCS(my_strings)

    assert np.max(tables) == 3
    assert sequences[1][tuple("QWE")] == {0, 6, 9, 12}


def test_one_round_update_collecting_tables_first():

    example = [('R', 'A', 'C', 'E', 'C', 'A', 'R'),
    ('A', 'A', 'A', 'A', 'Q', 'A', 'A', 'A', 'A'),
    ('Q', 'W', 'E', 'R', 'T', 'Y', 'Q', 'W', 'E', 'Q', 'W', 'E', 'Q', 'W', 'E')]
    example = [list(x) for x in example]
    internal = create_tables_for_internal_LCS(example)
    external = _compute_lcs_for_every_pair_of_sequences(example, None, None, set([0,1,2]), 3)

    cache = {i: s for i,s in enumerate(example)}
    updated, num, cache, seq_intro, ext_table, ext_seq, ext_dirty = conduct_one_round_of_lcs_simplification(example, external, internal, len(example), cache)

    assert len(updated) == 4
    assert "".join(updated[3]) == "AAAA"

    assert cache[1] == [3,"Q",3]
    assert np.allclose(seq_intro, np.array(3))

    assert num == len(updated)


def test_one_round_update_without_collecting_tables_first():

    example = [('R', 'A', 'C', 'E', 'C', 'A', 'R'),
    ('A', 'A', 'A', 'A', 'Q', 'A', 'A', 'A', 'A'),
    ('Q', 'W', 'E', 'R', 'T', 'Y', 'Q', 'W', 'E', 'Q', 'W', 'E', 'Q', 'W', 'E')]
    example = [list(x) for x in example]


    cache = {i: s for i,s in enumerate(example)}
    updated, num, cache, seq_intro, ext_table, ext_seq, ext_dirty = conduct_one_round_of_lcs_simplification(example, None, None, len(example), cache)

    assert len(updated) == 4
    assert "".join(updated[3]) == "AAAA"

    assert cache[1] == [3,"Q",3]
    assert np.allclose(seq_intro, np.array(3))

    assert num == len(updated)


def test_update_only_adds_those_strings_which_are_actually_used():
    example = [('R', 'A', 'C', 'E', 'C', 'A', 'R'),
        ('A', 'A', 'A', 'A', 'Q', 'A', 'A', 'A', 'A'),
        ('Q', 'W', 'E', 'R', 'T', 'Y', 'Q', 'W', 'E', 'Q', 'W', 'E', 'Q', 'W', 'E')]
    example = [list(x) for x in example]


    cache = {i: s for i,s in enumerate(example)}
    updated, num, cache, seq_intro, ext_table, ext_seq, ext_dirty = conduct_one_round_of_lcs_simplification(example, None, None, len(example), cache)

    r2, num, c2, s2, ext_table, ext_seq, ext_dirty = conduct_one_round_of_lcs_simplification(updated, None, None, num, cache)

    assert len(r2) == num

    assert len(s2) == 1

    assert 4 in c2[2]

    assert len(c2[4]) == 3

def test_multiple_successive_internal_updates_first():

    strings_list = [list("IIIIIIAIIIIII")]

    cache = {}
    seq_ind_to_cache_ind = {}
    updated_string_list, cache_num, cache, seq_intro = simplify_internal_first_one_round(strings_list,
                                                                                         None,
                                                                                         -1,
                                                                                         cache,
                                                                                         seq_ind_to_cache_ind)

    assert -1 in cache.keys()

    assert -2 == cache_num
    assert len(updated_string_list) == 2

    updated_string_list, cache_num, cache, seq_intro = simplify_internal_first_one_round(updated_string_list, 
                                                                                         None,
                                                                                         cache_num,
                                                                                         cache,
                                                                                         seq_ind_to_cache_ind)

    assert -2 in cache.keys()
    assert -3 == cache_num

    assert len(updated_string_list) == 3
    assert np.allclose(seq_intro, [-2])
import numpy as np
import pytest

from pygsti.extras.sparsedem.singleton import (
    find_atoms_by_singletons,
    initial_p_from_counts,
    refine_p_by_parities,
    learn_atoms_and_ps,
    simulate_Y,
    find_columns_in_B,
)
from pygsti.extras.sparsedem.utils import (
    bits_to_binary_number,
    binary_number_to_bits,
    rows_to_tuples,
    parity_dot,
    estimate_polarizations,
    counts_from_samples,
)


def test_bits_roundtrip_msb_first():
    bits = [1, 0, 1, 0]
    value = bits_to_binary_number(bits)
    assert value == 10
    assert binary_number_to_bits(value, len(bits)) == bits


def test_rows_to_tuples_and_parity():
    Y = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    assert rows_to_tuples(Y) == [(0, 1, 1), (1, 0, 1)]
    mask = np.array([1, 1, 0], dtype=np.uint8)
    assert np.array_equal(parity_dot(Y, mask), np.array([1, 1], dtype=np.uint8))


def test_estimate_phi_zero_matrix():
    Y = np.zeros((4, 3), dtype=np.uint8)
    S = np.array([[1, 0, 0], [1, 1, 1]], dtype=np.uint8)
    counts = counts_from_samples(Y)
    phi = estimate_polarizations(counts, S)
    assert np.allclose(phi, 1.0)


def test_find_atoms_and_initial_p():
    Y = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    counts = counts_from_samples(Y)
    B_list, P0_hat = find_atoms_by_singletons(counts, min_count=2)
    atoms = {tuple(b.tolist()) for b in B_list}
    assert atoms == {(0, 0, 1), (0, 1, 0)}
    assert np.isclose(P0_hat, 1 / 5)
    p_init = initial_p_from_counts(counts, B_list, m=sum(counts.values()), P0_hat=P0_hat)
    assert p_init.shape == (len(B_list),)


def test_refine_and_learn_shapes():
    rng = np.random.default_rng(0)
    B = rng.integers(0, 2, size=(5, 3), dtype=np.uint8)
    p = np.array([0.02, 0.01, 0.03])
    _, Y = simulate_Y(B, p, m=400, rng=rng)
    counts = counts_from_samples(Y)
    B_list, _ = find_atoms_by_singletons(counts, min_count=2, max_atoms=5)
    p_refined = refine_p_by_parities(counts, B_list, num_masks=16, rng=rng)
    assert p_refined.shape == (len(B_list),)
    assert np.all((p_refined >= 0.0) & (p_refined <= 0.49))

    B_hat, p_init, p_refined2, meta = learn_atoms_and_ps(
        counts,
        min_count=2,
        max_atoms=5,
        num_masks=16,
        seed=0,
    )
    assert B_hat.shape[0] == B.shape[0]
    assert p_init.shape == p_refined2.shape
    assert "P0_hat" in meta


def test_find_columns_in_B():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[9, 2, 1, 3], [8, 5, 4, 6]])
    idx, present = find_columns_in_B(A, B)
    assert np.array_equal(idx, np.array([2, 1, 3]))
    assert np.array_equal(present, np.array([True, True, True]))


def test_find_columns_in_B_mismatch_rows():
    A = np.array([[1, 2, 3]])
    B = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        find_columns_in_B(A, B)

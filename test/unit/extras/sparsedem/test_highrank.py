import itertools

import numpy as np
import pytest
import stim

from pygsti.extras.sparsedem import estimation
from pygsti.extras.sparsedem.highrank import (
    ErrorEvent,
    ExclusiveBlock,
    HighRankDetectorErrorModel,
)


EXAMPLE = """
# comment
detector(1, 2) D0
error(0.0024) D0 D1 L0
exclusive(2)
    error(0.001) D0
    error(0.001) D1 L0
error(0.05) D2 ^ D3
"""


def test_parse_example():
    m = HighRankDetectorErrorModel.from_text(EXAMPLE)
    assert m.num_detectors == 4
    assert m.num_observables == 1
    assert m.detector_coords == {0: (1.0, 2.0)}
    assert len(m.instructions) == 3
    assert m.instructions[0] == ErrorEvent(0.0024, (0, 1), (0,))
    blk = m.instructions[1]
    assert isinstance(blk, ExclusiveBlock)
    assert blk.events == (ErrorEvent(0.001, (0,)), ErrorEvent(0.001, (1,), (0,)))
    assert blk.rank == 3
    # '^' separator is accepted and ignored.
    assert m.instructions[2] == ErrorEvent(0.05, (2, 3))


def test_round_trip():
    m = HighRankDetectorErrorModel.from_text(EXAMPLE)
    m2 = HighRankDetectorErrorModel.from_text(m.to_text())
    assert m == m2


def test_block_probabilities_must_sum_below_one():
    with pytest.raises(ValueError, match="sum"):
        HighRankDetectorErrorModel.from_text(
            "exclusive(2)\nerror(0.7) D0\nerror(0.4) D1\n"
        )


def test_incomplete_block_rejected():
    with pytest.raises(ValueError, match="expects"):
        HighRankDetectorErrorModel.from_text("exclusive(2)\nerror(0.1) D0\n")
    with pytest.raises(ValueError, match="expects"):
        HighRankDetectorErrorModel.from_text(
            "exclusive(2)\nerror(0.1) D0\ndetector D5\nerror(0.1) D1\n"
        )


def test_bad_target_rejected():
    with pytest.raises(ValueError, match="unrecognized target"):
        HighRankDetectorErrorModel.from_text("error(0.1) Q0\n")


def test_declared_bounds_extend_model():
    m = HighRankDetectorErrorModel.from_text(
        "error(0.1) D0\ndetector D7\nlogical_observable L2\n"
    )
    assert m.num_detectors == 8
    assert m.num_observables == 3


def test_from_stim_dem_round_trip():
    dem = stim.DetectorErrorModel(
        """
        error(0.125) D0 D1
        error(0.25) D1 L0
        detector(0, 1) D0
        """
    )
    m = HighRankDetectorErrorModel.from_stim_dem(dem)
    assert m.independent_errors == [
        ErrorEvent(0.125, (0, 1)),
        ErrorEvent(0.25, (1,), (0,)),
    ]
    assert m.exclusive_blocks == []
    assert m.detector_coords == {0: (0.0, 1.0)}


def test_approximate_stim_dem_flattens_blocks():
    m = HighRankDetectorErrorModel.from_text(EXAMPLE)
    dem = m.approximate_stim_dem()
    assert dem.num_detectors == 4
    assert dem.num_observables == 1
    n_errors = sum(1 for inst in dem.flattened() if inst.type == "error")
    assert n_errors == 4  # 2 independent + 2 block branches


def test_outcome_distribution_from_model():
    # A real HighRankDetectorErrorModel through the estimation seam that was
    # previously exercised only with duck-typed stand-ins.
    model = HighRankDetectorErrorModel.from_text(
        """
        error(0.1) D0
        exclusive(2)
            error(0.2) D0 D1
            error(0.15) D2 L0
        """
    )
    probs = estimation.compute_outcome_distribution_from_high_rank_dem(model)

    # Brute force: enumerate independent-event outcomes and block branch
    # choices; outcome index has D0 as the least significant bit.
    n = model.num_detectors
    expected = np.zeros(2 ** n)
    ind = model.independent_errors
    blocks = model.exclusive_blocks
    block_choices = [range(len(b.events) + 1) for b in blocks]
    for ind_fires in itertools.product([0, 1], repeat=len(ind)):
        for choices in itertools.product(*block_choices):
            p = 1.0
            det = [0] * n
            for fired, ev in zip(ind_fires, ind):
                p *= ev.probability if fired else (1 - ev.probability)
                if fired:
                    for d in ev.detectors:
                        det[d] ^= 1
            for choice, blk in zip(choices, blocks):
                if choice < len(blk.events):
                    ev = blk.events[choice]
                    p *= ev.probability
                    for d in ev.detectors:
                        det[d] ^= 1
                else:
                    p *= 1 - blk.total_probability
            expected[sum(bit << d for d, bit in enumerate(det))] += p

    np.testing.assert_allclose(probs, expected, atol=1e-12)

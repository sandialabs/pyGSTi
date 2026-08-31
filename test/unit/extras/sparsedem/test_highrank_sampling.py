import itertools

import numpy as np
import pytest

from pygsti.extras.sparsedem.highrank import HighRankDetectorErrorModel
from pygsti.extras.sparsedem.highrank_sampling import (
    CompiledHighRankSampler,
    NumpyReferenceSampler,
    to_stim_sampling_circuit,
)


def exact_distribution(model):
    """Brute-force joint distribution over (detector bits, observable bits).

    Only usable for tiny models; enumerates every combination of independent
    error outcomes and exclusive-block branch choices.
    """
    n_det, n_obs = model.num_detectors, model.num_observables
    ind = model.independent_errors
    blocks = model.exclusive_blocks

    dist = {}
    block_choices = [range(len(b.events) + 1) for b in blocks]
    for ind_fires in itertools.product([0, 1], repeat=len(ind)):
        for choices in itertools.product(*block_choices):
            p = 1.0
            det = [0] * n_det
            obs = [0] * n_obs
            for fired, ev in zip(ind_fires, ind):
                p *= ev.probability if fired else (1 - ev.probability)
                if fired:
                    for d in ev.detectors:
                        det[d] ^= 1
                    for o in ev.observables:
                        obs[o] ^= 1
            for choice, blk in zip(choices, blocks):
                if choice < len(blk.events):
                    ev = blk.events[choice]
                    p *= ev.probability
                    for d in ev.detectors:
                        det[d] ^= 1
                    for o in ev.observables:
                        obs[o] ^= 1
                else:
                    p *= 1 - blk.total_probability
            key = (tuple(det), tuple(obs))
            dist[key] = dist.get(key, 0.0) + p
    return dist


def empirical_distribution(dets, obs):
    dist = {}
    shots = dets.shape[0]
    for d_row, o_row in zip(dets, obs):
        key = (tuple(int(x) for x in d_row), tuple(int(x) for x in o_row))
        dist[key] = dist.get(key, 0.0) + 1.0 / shots
    return dist


TINY = """
error(0.1) D0 L0
error(0.2) D1
exclusive(2)
    error(0.3) D0
    error(0.25) D2 L0
"""

SHOTS = 1_000_000
# Worst-case std of an empirical cell probability is sqrt(.25/SHOTS) = 5e-4;
# 6 sigma keeps flake probability negligible across all cells.
ATOL = 3e-3


@pytest.mark.parametrize("make", [CompiledHighRankSampler, NumpyReferenceSampler])
def test_matches_exact_distribution(make):
    model = HighRankDetectorErrorModel.from_text(TINY)
    dets, obs = make(model, seed=1234).sample(SHOTS)
    emp = empirical_distribution(dets, obs)
    exact = exact_distribution(model)
    assert abs(sum(exact.values()) - 1.0) < 1e-12
    for key in set(exact) | set(emp):
        assert emp.get(key, 0.0) == pytest.approx(exact.get(key, 0.0), abs=ATOL), key


def test_exclusivity_is_exact():
    model = HighRankDetectorErrorModel.from_text(
        """
        exclusive(3)
            error(0.3) D0
            error(0.3) D1
            error(0.3) D2 L0
        """
    )
    dets, obs = CompiledHighRankSampler(model, seed=7).sample(SHOTS)
    # At most one branch per shot, always.
    assert dets.sum(axis=1).max() <= 1
    np.testing.assert_allclose(dets.mean(axis=0), [0.3, 0.3, 0.3], atol=ATOL)
    assert obs[:, 0].mean() == pytest.approx(0.3, abs=ATOL)
    assert np.array_equal(obs[:, 0], dets[:, 2])


def test_saturated_block_and_empty_branch():
    # Branch probabilities may sum to exactly 1, and a branch may flip
    # nothing (an explicit trivial outcome that still consumes probability).
    model = HighRankDetectorErrorModel.from_text(
        """
        exclusive(2)
            error(0.5)
            error(0.5) D0
        """
    )
    dets, _ = CompiledHighRankSampler(model, seed=5).sample(SHOTS)
    assert dets[:, 0].mean() == pytest.approx(0.5, abs=ATOL)


def test_compiled_and_reference_agree_on_random_model():
    rng = np.random.default_rng(0)
    n_det, n_obs = 40, 2
    lines = []
    for _ in range(60):
        d = rng.choice(n_det, size=2, replace=False)
        lines.append(f"error({rng.uniform(0.001, 0.05)}) D{d[0]} D{d[1]}")
    for _ in range(20):
        k = int(rng.integers(2, 5))
        lines.append(f"exclusive({k})")
        for _ in range(k):
            d = rng.choice(n_det, size=2, replace=False)
            tail = f" L{rng.integers(n_obs)}" if rng.random() < 0.3 else ""
            lines.append(f"error({rng.uniform(0.001, 0.2 / k)}) D{d[0]} D{d[1]}{tail}")
    model = HighRankDetectorErrorModel.from_text("\n".join(lines))

    # Sample in chunks: the reference sampler's temporaries scale with the
    # number of shots per call, and SHOTS at once needs ~1 GB at this width.
    chunk = 100_000
    d_mean, o_mean, corr = [], [], []
    for sampler in (
        CompiledHighRankSampler(model, seed=11),
        NumpyReferenceSampler(model, seed=22),
    ):
        d_sum = np.zeros(n_det)
        o_sum = np.zeros(n_obs)
        c_sum = np.zeros((10, 10))
        for _ in range(SHOTS // chunk):
            d, o = sampler.sample(chunk)
            d_sum += d.sum(axis=0)
            o_sum += o.sum(axis=0)
            c_sum += (d[:, :10, None] & d[:, None, :10]).sum(axis=0)
        d_mean.append(d_sum / SHOTS)
        o_mean.append(o_sum / SHOTS)
        corr.append(c_sum / SHOTS)

    np.testing.assert_allclose(d_mean[0], d_mean[1], atol=ATOL)
    np.testing.assert_allclose(o_mean[0], o_mean[1], atol=ATOL)
    # Pairwise correlations within detectors should match too (exclusive
    # blocks induce negative correlations an independent model would miss).
    np.testing.assert_allclose(corr[0], corr[1], atol=ATOL)


def test_bit_packed_output():
    model = HighRankDetectorErrorModel.from_text(TINY)
    sampler = CompiledHighRankSampler(model, seed=3)
    dets, obs = sampler.sample(1000, bit_packed=True)
    assert dets.shape == (1000, 1) and dets.dtype == np.uint8
    assert obs.shape == (1000, 1) and obs.dtype == np.uint8
    unpacked = np.unpackbits(dets, axis=1, bitorder="little")[:, : model.num_detectors]
    d2, _ = sampler.sample(1000)
    assert unpacked.shape == d2.shape


def test_sampling_circuit_structure():
    model = HighRankDetectorErrorModel.from_text(TINY)
    circuit = to_stim_sampling_circuit(model)
    assert circuit.num_detectors == model.num_detectors == 3
    assert circuit.num_observables == model.num_observables == 1
    names = [inst.name for inst in circuit]
    assert names.count("ELSE_CORRELATED_ERROR") == 1
    # Conditional probability of the second branch is 0.25 / (1 - 0.3).
    else_inst = [i for i in circuit if i.name == "ELSE_CORRELATED_ERROR"][0]
    assert else_inst.gate_args_copy()[0] == pytest.approx(0.25 / 0.7)

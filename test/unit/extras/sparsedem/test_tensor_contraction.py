"""Tests for pygsti.extras.sparsedem.tensor_contraction (exact DEM probabilities by TN contraction)."""

import itertools

import numpy as np
import pytest
import stim

from pygsti.extras.sparsedem import tensor_contraction as tc
from pygsti.extras.sparsedem.estimation import compute_outcome_distribution_from_dem
from pygsti.extras.sparsedem.io import dem_from_str
from pygsti.extras.sparsedem.utils import counts_from_samples

BACKENDS = ["numpy", pytest.param("quimb", marks=pytest.mark.skipif(
    not tc.quimb_available(), reason="quimb not installed"))]

# The two DEMs from the prototype notebook.
DEM_PROTO_1 = "error(0.01) D0 D1\nerror(0.02)"
DEM_PROTO_2 = "error(0.01) D0\nerror(0.02) D0 D1\nerror(0.01) D1\nerror(0.02) D0 D2\nerror(0.02) D2"
# Weight-3 hyperedge, duplicated event on the same detector set, detector D3 touched by no event.
DEM_HYPER = ("error(0.1) D0 D1 D2\nerror(0.05) D1\nerror(0.05) D1\nerror(0.2) D0 D2\n"
             "error(0.03) D4\nerror(0.07) D2 D4")
# Logical observable targets (marginalised by default).
DEM_LOGICAL = "error(0.1) D0 D1 L0\nerror(0.05) D1 D2\nerror(0.2) D0 L0\nerror(0.02) L0\nerror(0.03) D2"


# ---------------------------------------------------------------------------
# Independent brute-force reference: XOR convolution over events, O(events * 2^n).
# Index convention: bit d of the index = detector d (matches compute_outcome_distribution_from_dem);
# observables, if requested, occupy bits n, n+1, ... above the detectors.
# ---------------------------------------------------------------------------

def xor_convolution_distribution(dem, include_observables=False):
    n_det, n_obs = dem.num_detectors, dem.num_observables
    n = n_det + (n_obs if include_observables else 0)
    dist = np.zeros(2 ** n)
    dist[0] = 1.0
    idx = np.arange(2 ** n)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        mask = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                mask ^= 1 << t.val
            elif t.is_logical_observable_id() and include_observables:
                mask ^= 1 << (n_det + t.val)
        dist = (1 - p) * dist + p * dist[idx ^ mask]
    return dist


def dense_as_detector_axes(dist, n):
    """Reshape a mask-indexed distribution so that axis d is detector d."""
    return dist.reshape((2,) * n).transpose(list(range(n))[::-1])


def all_bits(n):
    return [tc.mask_to_detector_bits(m, n) for m in range(2 ** n)]


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

def test_bit_convention_helpers():
    # stim event `error(p) D0 D1 D4` gives sample [1,1,0,0,1]; sparsedem key '10011'; mask 19.
    bits = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
    assert tc.detector_bits_to_bitstring(bits) == "10011"
    assert np.array_equal(tc.bitstring_to_detector_bits("10011"), bits)
    assert np.array_equal(tc.mask_to_detector_bits(19, 5), bits)
    assert counts_from_samples(bits[None, :]) == {"10011": 1}


def test_dense_reference_conventions_agree():
    dem = dem_from_str(DEM_PROTO_2)
    np.testing.assert_allclose(xor_convolution_distribution(dem), compute_outcome_distribution_from_dem(dem),
                               atol=1e-14)


# ---------------------------------------------------------------------------
# Exact agreement with the dense distribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dem_str", [DEM_PROTO_1, DEM_PROTO_2, DEM_HYPER])
def test_matches_dense_distribution_all_outcomes(dem_str, backend):
    dem = dem_from_str(dem_str)
    n = dem.num_detectors
    ref = compute_outcome_distribution_from_dem(dem)
    np.testing.assert_allclose(ref, xor_convolution_distribution(dem), atol=1e-14)
    probs = [tc.detector_outcome_probability(dem, bits, backend=backend) for bits in all_bits(n)]
    np.testing.assert_allclose(probs, ref, atol=1e-13)
    assert abs(sum(probs) - 1.0) < 1e-12


def test_bitstring_keyword_uses_sparsedem_order():
    dem = dem_from_str("error(0.1) D0\nerror(0.2) D2")  # asymmetric so order matters
    ref = compute_outcome_distribution_from_dem(dem)
    # detector 0 fired, detector 2 did not -> mask 1 -> sparsedem key '001'
    assert tc.detector_outcome_probability(dem, bitstring="001") == pytest.approx(ref[1], abs=1e-14)
    assert tc.detector_outcome_probability(dem, bitstring="100") == pytest.approx(ref[4], abs=1e-14)
    assert tc.detector_outcome_probability(dem, [1, 0, 0]) == pytest.approx(ref[1], abs=1e-14)
    with pytest.raises(ValueError):
        tc.detector_outcome_probability(dem, [1, 0, 0], bitstring="001")


def test_isolated_detector_and_no_target_events():
    dem = dem_from_str(DEM_HYPER)  # D3 is never flipped, event with duplicated detector set exists
    assert dem.num_detectors == 5
    bits = np.zeros(5, dtype=np.uint8)
    bits[3] = 1
    assert tc.detector_outcome_probability(dem, bits, backend="numpy") == 0.0
    dem2 = dem_from_str("error(0.3)\nerror(0.4)\ndetector D2")  # no-target events, declared detectors only
    assert tc.detector_outcome_probability(dem2, [0, 0, 0], backend="numpy") == pytest.approx(1.0)
    assert tc.detector_outcome_probability(dem2, [0, 1, 0], backend="numpy") == 0.0
    # the network for the all-zero outcome is just rank-0 constants
    tn = tc.dem_to_tensor_network(dem2, [0, 0, 0])
    assert tn.max_rank == 0 and tn.num_events == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_logical_observables_marginalised_and_conditioned(backend):
    dem = dem_from_str(DEM_LOGICAL)
    n = dem.num_detectors
    assert dem.num_observables == 1
    ref_marg = xor_convolution_distribution(dem)
    ref_joint = xor_convolution_distribution(dem, include_observables=True)  # bit n = L0
    for m, bits in enumerate(all_bits(n)):
        p = tc.detector_outcome_probability(dem, bits, backend=backend)
        assert p == pytest.approx(ref_marg[m], abs=1e-13)
        p0 = tc.detector_outcome_probability(dem, bits, observable_bits=[0], backend=backend)
        p1 = tc.detector_outcome_probability(dem, bits, observable_bits=[1], backend=backend)
        assert p0 == pytest.approx(ref_joint[m], abs=1e-13)
        assert p1 == pytest.approx(ref_joint[m | (1 << n)], abs=1e-13)
        assert p0 + p1 == pytest.approx(p, abs=1e-13)
    # observable as an open axis of a marginal
    joint = tc.marginal_distribution(dem, [0, "L0"], backend=backend)
    ref_axes = dense_as_detector_axes(ref_joint, n + 1)  # axis 3 = L0
    np.testing.assert_allclose(joint, ref_axes.sum(axis=(1, 2)), atol=1e-13)


def test_duplicate_targets_within_one_event_cancel():
    dem = dem_from_str("error(0.3) D0 D1 D1\nerror(0.1) D1")
    ref = compute_outcome_distribution_from_dem(dem_from_str("error(0.3) D0\nerror(0.1) D1"))
    probs = [tc.detector_outcome_probability(dem, b, backend="numpy") for b in all_bits(2)]
    np.testing.assert_allclose(probs, ref, atol=1e-14)


# ---------------------------------------------------------------------------
# Marginals and conditionals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_marginal_distribution_matches_dense(backend):
    dem = dem_from_str(DEM_HYPER)
    n = dem.num_detectors
    axes = dense_as_detector_axes(compute_outcome_distribution_from_dem(dem), n)
    # full joint, in a scrambled detector order
    order = [4, 0, 2, 1, 3]
    full = tc.marginal_distribution(dem, order, backend=backend)
    np.testing.assert_allclose(full, axes.transpose(order), atol=1e-13)
    # pairwise marginal
    m = tc.marginal_distribution(dem, [2, 0], backend=backend)
    np.testing.assert_allclose(m, axes.sum(axis=(1, 3, 4)).T, atol=1e-13)
    assert m.sum() == pytest.approx(1.0)
    # single-detector marginal, string label
    m1 = tc.marginal_distribution(dem, ["D1"], backend=backend)
    np.testing.assert_allclose(m1, axes.sum(axis=(0, 2, 3, 4)), atol=1e-13)
    # empty subset -> total probability 1
    assert float(tc.marginal_distribution(dem, [], backend=backend)) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_conditional_marginal_matches_dense(backend):
    dem = dem_from_str(DEM_PROTO_2)
    axes = dense_as_detector_axes(compute_outcome_distribution_from_dem(dem), 3)
    joint = tc.marginal_distribution(dem, [2], condition={0: 1}, backend=backend)  # P(D2, D0=1)
    np.testing.assert_allclose(joint, axes[1].sum(axis=0), atol=1e-13)
    cond = joint / joint.sum()
    np.testing.assert_allclose(cond, axes[1].sum(axis=0) / axes[1].sum(), atol=1e-13)
    joint2 = tc.marginal_distribution(dem, [1, 2], condition={"D0": 0}, backend=backend)
    np.testing.assert_allclose(joint2, axes[0], atol=1e-13)


# ---------------------------------------------------------------------------
# Network structure options
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_unfused_events_and_chained_tensors_agree(backend):
    dem = dem_from_str(DEM_HYPER + "\nerror(0.01) D0 D1 D2 D4\nerror(0.02) D0 D1\nerror(0.02) D0 D2")
    ref = compute_outcome_distribution_from_dem(dem)
    for m in (0, 5, 22, 31):
        bits = tc.mask_to_detector_bits(m, 5)
        for fuse, max_rank in itertools.product([True, False], [3, 4, 8]):
            tn = tc.dem_to_tensor_network(dem, bits, fuse_events=fuse, max_rank=max_rank)
            assert tn.max_rank <= max_rank
            assert tn.contract(backend=backend) == pytest.approx(ref[m], abs=1e-13)
    tn = tc.dem_to_tensor_network(dem, tc.mask_to_detector_bits(5, 5), fuse_events=False)
    tags = {t.tag for t in tn.tensors}
    assert {"p0", "e0", "D0", "D3"} <= tags


def test_with_bits_and_plan_reuse():
    dem = dem_from_str(DEM_PROTO_2)
    ref = compute_outcome_distribution_from_dem(dem)
    base = tc.dem_to_tensor_network(dem, [0, 0, 0])
    plan = base.plan(backend="numpy")
    for m in range(8):
        assert plan.evaluate(base.with_bits(tc.mask_to_detector_bits(m, 3))) == pytest.approx(ref[m], abs=1e-13)
    with pytest.raises(ValueError):
        base.with_bits(observable_bits={0: 1})  # nothing fixed for observables
    with pytest.raises(ValueError):
        plan.evaluate(tc.dem_to_tensor_network(dem, {0: 1}))  # different structure


def test_input_validation():
    dem = dem_from_str(DEM_PROTO_2)
    with pytest.raises(ValueError):
        tc.dem_to_tensor_network(dem, [0, 0])  # wrong length
    with pytest.raises(ValueError):
        tc.dem_to_tensor_network(dem, {0: 1}, open_detectors=[0])  # fixed and open
    with pytest.raises(ValueError):
        tc.dem_to_tensor_network(dem, {7: 1})  # out of range
    with pytest.raises(ValueError):
        tc.detector_outcome_probability(dem, [0, 0, 0], backend="tensorflow")
    with pytest.raises(ValueError):
        tc.dem_to_tensor_network(dem, [0, 0, 0], max_rank=2)


# ---------------------------------------------------------------------------
# stim circuit DEMs
# ---------------------------------------------------------------------------

def test_repeat_block_is_flattened():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0
        repeat 3 {
            error(0.05) D0 D1
            shift_detectors 1
        }
        error(0.2) D0
    """)  # after the loop the shift is 3, so the last line is detector 3
    flat = dem.flattened()
    assert dem.num_detectors == 4 and any(inst.type == "repeat" for inst in dem)
    ref = xor_convolution_distribution(flat)
    probs = [tc.detector_outcome_probability(dem, b, backend="numpy") for b in all_bits(4)]
    np.testing.assert_allclose(probs, ref, atol=1e-13)
    assert tc.dem_to_tensor_network(dem, [0] * 4).num_events == 5


@pytest.fixture(scope="module")
def surface_code_dem():
    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=2,
                                     after_clifford_depolarization=0.01)
    return circuit.detector_error_model(decompose_errors=False)


@pytest.mark.parametrize("backend", BACKENDS)
def test_surface_code_dem_matches_brute_force(surface_code_dem, backend):
    dem = surface_code_dem
    assert dem.num_detectors == 16
    ref = xor_convolution_distribution(dem)
    assert ref.sum() == pytest.approx(1.0)
    det, _, _ = dem.compile_sampler(seed=7).sample(shots=6)
    det = det.astype(np.uint8)
    outcomes = np.vstack([np.zeros((1, 16), dtype=np.uint8), det])
    probs = tc.outcome_probabilities(dem, outcomes, backend=backend)
    masks = [int(sum(int(b) << d for d, b in enumerate(row))) for row in outcomes]
    np.testing.assert_allclose(probs, ref[masks], rtol=1e-10, atol=1e-15)
    # the parity tensor of the busiest detector (29 events) has been chained
    tn = tc.dem_to_tensor_network(dem, outcomes[0])
    assert tn.max_rank <= 8
    # a marginal that would need all 2^16 entries otherwise
    axes = dense_as_detector_axes(ref, 16)
    m = tc.marginal_distribution(dem, [3, 11], backend=backend)
    np.testing.assert_allclose(m, axes.sum(axis=tuple(a for a in range(16) if a not in (3, 11))), atol=1e-12)


def test_log_likelihood_prefers_true_dem():
    circuit = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=4,
                                     after_clifford_depolarization=0.02)
    dem = circuit.detector_error_model()
    det, _, _ = dem.compile_sampler(seed=3).sample(shots=400)
    counts = counts_from_samples(det.astype(np.uint8))
    ll_true, probs = tc.log_likelihood(dem, counts, backend="numpy", return_per_outcome=True)
    assert np.isfinite(ll_true) and ll_true < 0
    assert probs.shape == (len(counts),) and np.all(probs > 0)
    # cross-check one entry against the single-outcome function via the bitstring key
    key = next(iter(counts))
    assert probs[0] == pytest.approx(tc.detector_outcome_probability(dem, bitstring=key, backend="numpy"))
    # perturbed model: halve every probability
    perturbed = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            perturbed.append("error", [inst.args_copy()[0] * 0.5], inst.targets_copy())
    ll_pert = tc.log_likelihood(perturbed, counts, backend="numpy")
    assert np.isfinite(ll_pert) and ll_pert < ll_true
    # integer-mask keys are accepted too
    mask_counts = {sum(int(b) << d for d, b in enumerate(tc.bitstring_to_detector_bits(k))): v
                   for k, v in counts.items()}
    assert tc.log_likelihood(dem, mask_counts, backend="numpy") == pytest.approx(ll_true)
    # impossible outcome is floored, not -inf
    impossible = stim.DetectorErrorModel("error(0.1) D0 D1")  # detectors can only fire together
    assert np.isfinite(tc.log_likelihood(impossible, {"10": 1}, backend="numpy"))
    assert tc.detector_outcome_probability(impossible, bitstring="10", backend="numpy") == 0.0
    assert tc.log_likelihood(impossible, {}, backend="numpy") == 0.0


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def test_numpy_and_quimb_backends_agree():
    pytest.importorskip("quimb")
    circuit = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5,
                                     after_clifford_depolarization=0.01)
    dem = circuit.detector_error_model()
    det, _, _ = dem.compile_sampler(seed=11).sample(shots=8)
    det = det.astype(np.uint8)
    p_np = tc.outcome_probabilities(dem, det, backend="numpy")
    p_qb = tc.outcome_probabilities(dem, det, backend="quimb")
    np.testing.assert_allclose(p_np, p_qb, rtol=1e-12, atol=1e-300)
    m_np = tc.marginal_distribution(dem, [0, 7, 19], condition={3: 1}, backend="numpy")
    m_qb = tc.marginal_distribution(dem, [0, 7, 19], condition={3: 1}, backend="quimb")
    np.testing.assert_allclose(m_np, m_qb, rtol=1e-12)
    tn = tc.dem_to_tensor_network(dem, det[0])
    assert tn.to_quimb().num_tensors == tn.num_tensors


def test_auto_backend_resolution():
    dem = dem_from_str(DEM_PROTO_2)
    plan = tc.dem_to_tensor_network(dem, [0, 0, 0]).plan(backend="auto")
    assert plan.backend == ("quimb" if tc.quimb_available() else "numpy")


def test_contraction_tree_and_plot():
    pytest.importorskip("quimb")
    circuit = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=3,
                                     after_clifford_depolarization=0.01)
    dem = circuit.detector_error_model()
    tree = tc.contraction_tree(dem)
    assert tree.contraction_width() <= 8
    tree2 = tc.contraction_tree(tc.dem_to_tensor_network(dem, {0: 1}, open_detectors=[1]))
    assert tree2.contraction_cost() > 0
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = tc.plot_contraction_tree(tree)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batched evaluation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_outcome_probabilities_match_single(backend):
    dem = dem_from_str(DEM_LOGICAL)
    n = dem.num_detectors
    outcomes = np.array(all_bits(n) + all_bits(n)[:3])  # duplicates are fine
    single = np.array([tc.detector_outcome_probability(dem, b, backend=backend) for b in outcomes])
    for batch_size in (None, 1, 4, 100):
        batched = tc.outcome_probabilities(dem, outcomes, backend=backend, batch_size=batch_size)
        np.testing.assert_allclose(batched, single, rtol=1e-12, atol=1e-16)
    # observables fixed in batch mode
    ref = xor_convolution_distribution(dem, include_observables=True)
    with_obs = tc.outcome_probabilities(dem, outcomes, observable_bits=[1], backend=backend, batch_size=5)
    masks = [sum(int(b) << d for d, b in enumerate(row)) | (1 << n) for row in outcomes]
    np.testing.assert_allclose(with_obs, ref[masks], atol=1e-13)


def test_with_bit_batch_structure_and_errors():
    dem = dem_from_str(DEM_PROTO_2)
    base = tc.dem_to_tensor_network(dem, [0, 0, 0])
    rows = np.array(all_bits(3))
    batched = base.with_bit_batch(rows)
    assert batched.is_batched and batched.open_inds == (tc.BATCH_INDEX,)
    assert all(t.inds[-1] == tc.BATCH_INDEX for t in batched.tensors if t.bit_label is not None)
    out = batched.contract(backend="numpy")
    np.testing.assert_allclose(out, compute_outcome_distribution_from_dem(dem), atol=1e-14)
    # batched marginal: batch axis first, then the open detector
    partial = tc.dem_to_tensor_network(dem, {0: 0, 1: 0}, open_detectors=[2]).with_bit_batch(rows)
    assert partial.contract(backend="numpy").shape == (8, 2)
    with pytest.raises(ValueError):
        batched.with_bit_batch(rows)  # already batched
    with pytest.raises(ValueError):
        base.with_bit_batch(rows[:, :2])  # wrong width
    dem_l = dem_from_str(DEM_LOGICAL)
    fixed_obs = tc.dem_to_tensor_network(dem_l, [0, 0, 0], observable_bits=[0])
    with pytest.raises(ValueError):
        fixed_obs.with_bit_batch(rows)  # observable bits required
    assert fixed_obs.with_bit_batch(rows, np.zeros((8, 1), dtype=int)).contract("numpy").shape == (8,)

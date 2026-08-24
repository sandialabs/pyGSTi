import numpy as np
import pytest
import stim

from pygsti.extras.sparsedem.circuit_noise import (
    CircuitNoiseModel,
    NoiseRule,
    si1000_noisy_circuit,
)


def ops_of(circuit):
    """Flattened list of (name, qubit_targets, args) for easy assertions."""
    out = []
    for op in circuit.flattened():
        out.append((op.name,
                    tuple(t.value for t in op.targets_copy() if not t.is_combiner),
                    tuple(op.gate_args_copy())))
    return out


def noised_qubits(circuit, channel, arg):
    """All qubits hit by `channel` with probability argument `arg`.

    Fusion-tolerant: stim merges consecutive same-argument channel
    instructions into one multi-target instruction on append.
    """
    qubits = set()
    for name, qs, args in ops_of(circuit):
        if name == channel and args == (arg,):
            qubits.update(qs)
    return qubits


P = 0.01


# ---------------------------------------------------------------------------
# SI1000 rule-by-rule behavior
# ---------------------------------------------------------------------------

def test_si1000_gate_layers():
    circuit = stim.Circuit("""
        R 0 1
        TICK
        H 0
        TICK
        CZ 0 1
        TICK
        M 0 1
    """)
    noisy = si1000_noisy_circuit(circuit, P)
    ops = ops_of(noisy)
    assert ("R", (0, 1), ()) in ops
    assert noised_qubits(noisy, "X_ERROR", 2 * P) == {0, 1}
    assert ("H", (0,), ()) in ops
    # After-H depolarization on 0 and idle depolarization on 1 (may fuse).
    assert noised_qubits(noisy, "DEPOLARIZE1", P / 10) == {0, 1}
    assert ("CZ", (0, 1), ()) in ops
    assert noised_qubits(noisy, "DEPOLARIZE2", P) == {0, 1}
    assert ("M", (0, 1), (5 * P,)) in ops             # recorded-result flip
    assert noised_qubits(noisy, "DEPOLARIZE1", P) == {0, 1}  # after M


def test_si1000_measure_reset_idle():
    circuit = stim.Circuit("R 0")
    noisy = CircuitNoiseModel.si1000(P).noisy_circuit(
        circuit, system_qubits={0, 1})
    ops = ops_of(noisy)
    # Qubit 1 idles through a reset moment: ordinary idle PLUS 2p.
    assert ("DEPOLARIZE1", (1,), (P / 10,)) in ops
    assert ("DEPOLARIZE1", (1,), (2 * P,)) in ops
    # The reset qubit itself only gets the reset flip.
    assert ("X_ERROR", (0,), (2 * P,)) in ops
    assert not any(name == "DEPOLARIZE1" and qs == (0,) for name, qs, _ in ops)


def test_si1000_mr_and_mpp():
    noisy = si1000_noisy_circuit(stim.Circuit("MR 0"), P)
    ops = ops_of(noisy)
    assert ("MR", (0,), (5 * P,)) in ops
    assert ("X_ERROR", (0,), (2 * P,)) in ops

    noisy = si1000_noisy_circuit(stim.Circuit("MPP Z0*Z1 Z2*Z3"), P)
    ops = ops_of(noisy)
    mpps = [o for o in ops if o[0] == "MPP"]
    # stim may re-fuse the split products; both must carry the flip arg.
    assert sum(len(qs) for _, qs, _ in mpps) == 4
    assert all(args == (5 * P,) for _, _, args in mpps)
    assert noised_qubits(noisy, "DEPOLARIZE2", P) == {0, 1, 2, 3}

    with pytest.raises(ValueError, match="MPP basis 'XX'"):
        si1000_noisy_circuit(stim.Circuit("MPP X0*X1"), P)
    with pytest.raises(ValueError, match="No noise rule"):
        si1000_noisy_circuit(stim.Circuit("RX 0"), P)


def test_classical_control_and_mixed_pairs():
    circuit = stim.Circuit("""
        M 0
        TICK
        CX rec[-1] 1 2 3
    """)
    noisy = si1000_noisy_circuit(circuit, P)
    # The quantum pair (2, 3) is noised; the classically-controlled pair is not.
    assert noised_qubits(noisy, "DEPOLARIZE2", P) == {2, 3}
    # Feed-forward target qubit 1 counts as idle (matching the reference),
    # as does the untouched qubit 0.
    assert noised_qubits(noisy, "DEPOLARIZE1", P / 10) >= {0, 1}


def test_immune_qubits_and_double_use():
    circuit = stim.Circuit("H 0 1")
    noisy = CircuitNoiseModel.si1000(P).noisy_circuit(
        circuit, immune_qubits={1})
    ops = ops_of(noisy)
    assert ("H", (0,), ()) in ops and ("H", (1,), ()) in ops or \
        ("H", (0, 1), ()) in ops
    depol1 = [qs for name, qs, args in ops
              if name == "DEPOLARIZE1" and args == (P / 10,)]
    assert all(1 not in qs for qs in depol1)

    with pytest.raises(ValueError, match="multiple times"):
        si1000_noisy_circuit(stim.Circuit("H 0\nCZ 0 1"), P)
    with pytest.raises(ValueError, match="already contains noise"):
        si1000_noisy_circuit(stim.Circuit("X_ERROR(0.1) 0"), P)


def test_repeat_blocks_preserved():
    circuit = stim.Circuit("""
        R 0
        TICK
        REPEAT 5 {
            H 0
            TICK
            M 0
        }
    """)
    noisy = si1000_noisy_circuit(circuit, P)
    blocks = [op for op in noisy if isinstance(op, stim.CircuitRepeatBlock)]
    assert len(blocks) == 1 and blocks[0].repeat_count == 5
    body_ops = ops_of(blocks[0].body_copy())
    assert ("M", (0,), (5 * P,)) in body_ops
    assert ("DEPOLARIZE1", (0,), (P / 10,)) in body_ops


def test_noise_rule_validation():
    with pytest.raises(ValueError, match="noise channel"):
        NoiseRule(after={"H": 0.1})
    with pytest.raises(ValueError, match="probability"):
        NoiseRule(after={"X_ERROR": 1.5})
    with pytest.raises(ValueError, match="flip_result"):
        NoiseRule(flip_result=-0.1)
    with pytest.raises(ValueError, match="non-measurement"):
        CircuitNoiseModel(rules={"H": NoiseRule(flip_result=0.1)}).noisy_circuit(
            stim.Circuit("H 0"))


# ---------------------------------------------------------------------------
# Integration with stim-generated circuits and DEM extraction
# ---------------------------------------------------------------------------

def test_uniform_depolarizing_on_generated_circuit():
    clean = stim.Circuit.generated("repetition_code:memory",
                                   distance=3, rounds=3)
    noisy = CircuitNoiseModel.uniform_depolarizing(P).noisy_circuit(clean)
    dem = noisy.detector_error_model()
    assert dem.num_detectors == clean.num_detectors


def test_si1000_on_generated_surface_code():
    clean = stim.Circuit.generated("surface_code:rotated_memory_z",
                                   distance=3, rounds=3)
    noisy = si1000_noisy_circuit(clean, P)
    assert noisy.num_detectors == clean.num_detectors
    assert noisy.num_observables == clean.num_observables
    dem = noisy.detector_error_model(decompose_errors=True)
    det = noisy.compile_detector_sampler(seed=7).sample(4000)
    click_rate = det.mean()
    assert 0.001 < click_rate < 0.35
    # More noise, more clicks.
    det_hot = si1000_noisy_circuit(clean, 3 * P).compile_detector_sampler(
        seed=7).sample(4000)
    assert det_hot.mean() > 1.5 * click_rate


def test_si1000_zero_p_is_noiseless():
    clean = stim.Circuit.generated("surface_code:rotated_memory_z",
                                   distance=3, rounds=2)
    noisy = si1000_noisy_circuit(clean, 0.0)
    det = noisy.compile_detector_sampler(seed=1).sample(200)
    assert det.sum() == 0

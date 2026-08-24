"""
Circuit-level noise decoration for stim circuits.

Turns a noiseless Clifford stim circuit into a noisy one by inserting stim
noise channels according to a named noise model. The main model offered is
SI1000 ("superconducting-inspired, 1000 ns cycle") from "A Fault-Tolerant
Honeycomb Memory" (https://arxiv.org/abs/2108.10457): a single parameter p
sets the two-qubit gate error rate and everything else scales relative to
it, with measurements the noisiest operations. A uniform depolarizing model
is included as a baseline.

The decoration semantics follow the reference implementation in Craig
Gidney's `midout` repository (independently implemented here):

  * The circuit is split into moments at TICKs; a repeat block is recursed
    into and its repeat structure preserved.
  * Each operation in a moment gets its rule's noise channels appended
    after the moment's operations; measurement rules flip the *recorded*
    result (the probability argument of M/MR/MPP) rather than the qubit.
  * Qubits in `system_qubits` untouched during a moment receive idle
    depolarization; if the moment contains any measurement or reset, idle
    qubits additionally receive the (stronger) measure/reset idle
    depolarization.
  * Annotations and classically-controlled gates (e.g. ``CX rec[-1] 0``)
    receive no noise; operations touching `immune_qubits` pass through
    unchanged.
  * Operating on a qubit twice within one moment raises an error.

For SI1000, only Z-basis collapse operations are defined (M, R, MR and
Z-product MPP), matching the published model; anything else raises.
"""

import collections

import stim

_CLIFFORD_1Q = {
    "I", "X", "Y", "Z", "C_XYZ", "C_ZYX", "H", "H_XY", "H_XZ", "H_YZ", "S",
    "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "SQRT_Z", "SQRT_Z_DAG",
    "S_DAG",
}
_CLIFFORD_2Q = {
    "CNOT", "CX", "CY", "CZ", "ISWAP", "ISWAP_DAG", "CXSWAP", "SWAPCX",
    "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG", "SQRT_ZZ",
    "SQRT_ZZ_DAG", "SWAP", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "ZCX",
    "ZCY", "ZCZ",
}
_COLLAPSING = {"M", "MX", "MY", "MZ", "R", "RX", "RY", "RZ",
               "MR", "MRX", "MRY", "MRZ", "MPP"}
_MEASUREMENT = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP"}
_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS",
                "SHIFT_COORDS", "TICK", "E", "ELSE_CORRELATED_ERROR"}
_NOISE_CHANNELS = {"DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1",
                   "PAULI_CHANNEL_2", "X_ERROR", "Y_ERROR", "Z_ERROR",
                   "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"}


class NoiseRule:
    """
    Noise channels to attach to one operation.

    Parameters:
        after: dict
            Maps stim noise channel names to probability arguments; the
            channels are appended after the moment's operations, on the
            operation's targets.
        flip_result: float
            Probability that a produced measurement result is recorded
            incorrectly (becomes the probability argument of M/MR/MPP).
            Only valid on measurement operations.
    """

    def __init__(self, after=None, flip_result=0.0):
        after = dict(after or {})
        for name, prob in after.items():
            if name not in _NOISE_CHANNELS:
                raise ValueError(f"Not a stim noise channel: {name}")
            if not 0 <= prob <= 1:
                raise ValueError(f"Invalid probability {prob} for {name}")
        if not 0 <= flip_result <= 1:
            raise ValueError(f"Invalid flip_result {flip_result}")
        self.after = after
        self.flip_result = flip_result


class CircuitNoiseModel:
    """
    A named set of noise rules applied moment-by-moment to a stim circuit.

    Use the `si1000` or `uniform_depolarizing` constructors, or build a
    custom model from rules.

    Parameters:
        idle_depolarization: float
            DEPOLARIZE1 argument for qubits idle during a moment.
        measure_reset_idle: float
            Additional DEPOLARIZE1 argument for idle qubits during moments
            containing any measurement or reset.
        rules: dict
            Maps stim operation names (canonical, e.g. "M", "R", "MR",
            "CZ") or the wildcards "c1" / "c2" (any one-/two-qubit
            Clifford) or "MPP:ZZ"-style basis keys to NoiseRules.
    """

    def __init__(self, idle_depolarization=0.0, measure_reset_idle=0.0,
                 rules=None):
        self.idle_depolarization = idle_depolarization
        self.measure_reset_idle = measure_reset_idle
        self.rules = dict(rules or {})

    @staticmethod
    def si1000(p):
        """
        Superconducting-inspired noise, arXiv:2108.10457 (SI1000).

        Relative rates: two-qubit gates p, single-qubit Cliffords and idling
        p/10, reset flips 2p, measurement flips 5p with depolarization p on
        the measured qubits, and idle qubits wait through measure/reset
        layers with an extra 2p of depolarization. MR is the fused
        measurement+reset: result flip 5p, then X_ERROR(2p) from the reset
        (the post-measurement depolarization is erased by the reset).

        Parameters:
            p: float

        Returns:
            model: CircuitNoiseModel
        """
        return CircuitNoiseModel(
            idle_depolarization=p / 10,
            measure_reset_idle=2 * p,
            rules={
                "c1": NoiseRule(after={"DEPOLARIZE1": p / 10}),
                "c2": NoiseRule(after={"DEPOLARIZE2": p}),
                "R": NoiseRule(after={"X_ERROR": 2 * p}),
                "M": NoiseRule(after={"DEPOLARIZE1": p}, flip_result=5 * p),
                "MR": NoiseRule(after={"X_ERROR": 2 * p}, flip_result=5 * p),
                "MPP:ZZ": NoiseRule(after={"DEPOLARIZE2": p},
                                    flip_result=5 * p),
            },
        )

    @staticmethod
    def uniform_depolarizing(p):
        """
        Standard circuit depolarizing noise: everything at rate p.

        Parameters:
            p: float

        Returns:
            model: CircuitNoiseModel
        """
        rules = {
            "c1": NoiseRule(after={"DEPOLARIZE1": p}),
            "c2": NoiseRule(after={"DEPOLARIZE2": p}),
            "R": NoiseRule(after={"X_ERROR": p}),
            "RX": NoiseRule(after={"Z_ERROR": p}),
            "RY": NoiseRule(after={"X_ERROR": p}),
            "M": NoiseRule(after={"DEPOLARIZE1": p}, flip_result=p),
            "MX": NoiseRule(after={"DEPOLARIZE1": p}, flip_result=p),
            "MY": NoiseRule(after={"DEPOLARIZE1": p}, flip_result=p),
            "MR": NoiseRule(after={"X_ERROR": p}, flip_result=p),
        }
        for basis in ("XX", "YY", "ZZ"):
            rules[f"MPP:{basis}"] = NoiseRule(after={"DEPOLARIZE2": p},
                                              flip_result=p)
        return CircuitNoiseModel(idle_depolarization=p,
                                 measure_reset_idle=0.0, rules=rules)

    # ------------------------------------------------------------------

    def _rule_for(self, op_name, targets):
        if op_name == "MPP":
            basis = ""
            for k in range(len(targets)):
                t = targets[k]
                if t.is_combiner:
                    continue
                basis += "X" if t.is_x_target else "Y" if t.is_y_target else "Z"
            key = f"MPP:{basis}"
            if key in self.rules:
                return self.rules[key]
            raise ValueError(f"No noise rule for MPP basis '{basis}'.")
        if op_name in self.rules:
            return self.rules[op_name]
        if op_name in _CLIFFORD_1Q and "c1" in self.rules:
            return self.rules["c1"]
        if op_name in _CLIFFORD_2Q and "c2" in self.rules:
            return self.rules["c2"]
        raise ValueError(f"No noise rule for operation '{op_name}'.")

    def noisy_circuit(self, circuit, system_qubits=None, immune_qubits=None):
        """
        Return a noisy version of a (noiseless) stim circuit.

        Parameters:
            circuit: stim.Circuit
                The circuit to decorate; existing noise channels are not
                allowed (decorating twice is almost never intended).
            system_qubits: Optional[set]
                Qubits eligible for idle noise (default: all qubits in the
                circuit).
            immune_qubits: Optional[set]
                Qubits exempt from all noise, even when operated on.

        Returns:
            noisy: stim.Circuit
        """
        if system_qubits is None:
            system_qubits = set(range(circuit.num_qubits))
        if immune_qubits is None:
            immune_qubits = set()

        result = stim.Circuit()
        first = True
        for moment in _iter_moments(circuit):
            if first:
                first = False
            elif len(result) and isinstance(result[-1], stim.CircuitRepeatBlock):
                pass
            else:
                result.append("TICK")
            if isinstance(moment, stim.CircuitRepeatBlock):
                body = self.noisy_circuit(moment.body_copy(),
                                          system_qubits=system_qubits,
                                          immune_qubits=immune_qubits)
                body.append("TICK")
                result.append(stim.CircuitRepeatBlock(
                    repeat_count=moment.repeat_count, body=body))
            else:
                self._append_noisy_moment(moment, result, system_qubits,
                                          immune_qubits)
        return result

    def _append_noisy_moment(self, ops, out, system_qubits, immune_qubits):
        after = collections.defaultdict(list)
        collapse_qubits, gate_qubits = [], []
        for op in _split_ops(ops):
            name = op.name
            targets = op.targets_copy()
            if name in _NOISE_CHANNELS:
                raise ValueError(
                    f"Circuit already contains noise ({name}); refusing to "
                    "decorate a noisy circuit.")
            if name in _ANNOTATIONS:
                out.append(op)
                continue
            if _is_classically_controlled(name, targets):
                out.append(op)
                continue
            qubit_targets = [t.value for t in targets if not t.is_combiner]
            (collapse_qubits if name in _COLLAPSING
             else gate_qubits).extend(qubit_targets)
            if immune_qubits and any(q in immune_qubits for q in qubit_targets):
                out.append(op)
                continue
            rule = self._rule_for(name, targets)
            args = op.gate_args_copy()
            if rule.flip_result:
                if name not in _MEASUREMENT:
                    raise ValueError(f"flip_result rule on non-measurement {name}")
                if args:
                    raise ValueError(
                        f"{name} already has a noise argument; refusing to "
                        "decorate a noisy circuit.")
                args = [rule.flip_result]
            out.append(name, targets, args)
            for channel, prob in rule.after.items():
                after[(channel, prob)].extend(qubit_targets)

        for (channel, prob), qubits in sorted(after.items()):
            out.append(channel, qubits, prob)

        used = collections.Counter(collapse_qubits + gate_qubits)
        reused = sorted(q for q, c in used.items() if c > 1)
        if reused:
            raise ValueError(
                f"Qubits {reused} operated on multiple times within one "
                "moment (no TICK in between).")

        idle = sorted(set(system_qubits) - set(used) - set(immune_qubits))
        if idle and self.idle_depolarization:
            out.append("DEPOLARIZE1", idle, self.idle_depolarization)
        if idle and collapse_qubits and self.measure_reset_idle:
            out.append("DEPOLARIZE1", idle, self.measure_reset_idle)


def _split_ops(ops):
    """Split multi-product MPPs into one op per Pauli product, and 2q gates
    mixing classical and quantum pairs into one op per pair."""
    for op in ops:
        name = op.name
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if name == "MPP":
            k = start = 0
            while k < len(targets):
                if k + 1 == len(targets) or not targets[k + 1].is_combiner:
                    yield stim.CircuitInstruction(name, targets[start:k + 1], args)
                    k += 1
                    start = k
                else:
                    k += 2
        elif name in _CLIFFORD_2Q and any(
                t.is_measurement_record_target or t.is_sweep_bit_target
                for t in targets):
            for k in range(0, len(targets), 2):
                yield stim.CircuitInstruction(name, targets[k:k + 2], args)
        else:
            yield op


def _is_classically_controlled(name, targets):
    if name not in _CLIFFORD_2Q:
        return False
    for k in range(0, len(targets), 2):
        a, b = targets[k], targets[k + 1]
        if not (a.is_measurement_record_target or a.is_sweep_bit_target
                or b.is_measurement_record_target or b.is_sweep_bit_target):
            return False
    return True


def _iter_moments(circuit):
    """Yield TICK-delimited lists of instructions (and repeat blocks)."""
    current = []
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if current:
                yield current
                current = []
            yield op
        elif op.name == "TICK":
            yield current
            current = []
        else:
            current.append(op)
    if current:
        yield current


def si1000_noisy_circuit(circuit, p, system_qubits=None, immune_qubits=None):
    """
    Decorate a noiseless stim circuit with SI1000(p) noise.

    Convenience wrapper for CircuitNoiseModel.si1000(p).noisy_circuit(...).

    Parameters:
        circuit: stim.Circuit
        p: float
            The two-qubit gate error rate; all other rates scale with it
            (see CircuitNoiseModel.si1000).
        system_qubits: Optional[set]
        immune_qubits: Optional[set]

    Returns:
        noisy: stim.Circuit
    """
    return CircuitNoiseModel.si1000(p).noisy_circuit(
        circuit, system_qubits=system_qubits, immune_qubits=immune_qubits)

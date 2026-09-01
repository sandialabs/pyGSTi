from typing import Protocol

import numpy as np
import stim

from .rare_event import MechanismCatalog


class NoiseModel(Protocol):
    """Protocol for a circuit noise model.

    A noise model decorates a noiseless circuit with error channels
    parameterized by a physical error rate p.
    """

    def __call__(self, circuit: stim.Circuit, p: float) -> stim.Circuit:
        """Decorate the given circuit with noise at physical rate p."""
        ...


class SI1000NoiseModel:
    """SI1000 noise model parameterized by physical error rate p.

    - 1-qubit gates: DEPOLARIZE1(p)
    - 2-qubit gates: DEPOLARIZE2(p)
    - Measurements: X_ERROR(2p) before measurement
    - Resets: X_ERROR(p) after reset
    - Idles: DEPOLARIZE1(p/3) on idle qubits during TICKs (requires TICKs in circuit)

    Note: `pygsti.extras.sparsedem.circuit_noise` carries an independent
    SI1000 implementation following the Gidney `midout` reference semantics
    in more detail (recorded-result measurement flips, stronger idle noise
    during measure/reset moments). The two are deliberately not unified: this
    one is the model every published rareevent benchmark number was produced
    with. Prefer this one within rareevent, and expect small numerical
    differences from the sparsedem decoration.
    """

    def __call__(self, circuit: stim.Circuit, p: float) -> stim.Circuit:
        return self._decorate_circuit(circuit, p)

    def _decorate_circuit(self, circuit: stim.Circuit, p: float) -> stim.Circuit:
        noisy = stim.Circuit()
        active_qubits: set[int] = set()
        all_qubits = set(range(circuit.num_qubits))

        for inst in circuit:
            if isinstance(inst, stim.CircuitRepeatBlock):
                # We assume TICKs inside the block handle internal idles.
                noisy_body = self._decorate_circuit(inst.body_copy(), p)
                noisy.append(stim.CircuitRepeatBlock(inst.repeat_count, noisy_body))
                continue

            gate = stim.gate_data(inst.name)
            
            if inst.name == "TICK":
                # Apply idle noise to all qubits that weren't active since the last TICK
                idle_qubits = sorted(all_qubits - active_qubits)
                if idle_qubits and p > 0:
                    noisy.append("DEPOLARIZE1", idle_qubits, p / 3)
                noisy.append(inst)
                active_qubits.clear()
                continue

            # It's an instruction
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            active_qubits.update(targets)

            # Measurements get pre-measurement noise
            if gate.produces_measurements and p > 0:
                noisy.append("X_ERROR", targets, 2 * p)

            noisy.append(inst)

            # Post-gate noise
            if p > 0:
                if gate.is_single_qubit_gate and gate.is_unitary:
                    noisy.append("DEPOLARIZE1", targets, p)
                elif gate.is_two_qubit_gate and gate.is_unitary:
                    noisy.append("DEPOLARIZE2", targets, p)
                elif gate.is_reset:
                    noisy.append("X_ERROR", targets, p)

        return noisy


class ExactNoiseErrorModel:
    """An ErrorModel that computes exact mechanism probabilities for any p.
    
    It uses a user-provided NoiseModel to decorate the circuit at p, generates
    the detector error model, and maps the resulting exact probabilities to the
    reference MechanismCatalog.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        noise_model: NoiseModel,
        p_ref: float,
        global_dem_event_probability: float = 0.0,
    ):
        self.circuit = circuit
        self.noise_model = noise_model
        self.p_ref = p_ref
        self.global_dem_event_probability = global_dem_event_probability

        # 1. Decorate circuit at p_ref
        c_ref = self.noise_model(self.circuit, self.p_ref)
        dem_ref = c_ref.detector_error_model(decompose_errors=True, flatten_loops=True)
        
        # 2. Append global dem event if requested (using append_global_dem_event from rare_event)
        from .rare_event import append_global_dem_event
        append_global_dem_event(dem_ref, global_dem_event_probability)
        
        # 3. Build catalog
        self.catalog = MechanismCatalog.from_detector_error_model(dem_ref)
        self.num_mechanisms = len(self.catalog.mechanisms)

        # 4. Map each DEM instruction to the catalog indices it generated
        self.target_str_to_indices: dict[str, list[int]] = {}
        
        # We need to trace how MechanismCatalog.from_detector_error_model processes instructions.
        # It processes them strictly in order.
        idx = 0
        for inst in dem_ref:
            if inst.type != "error":
                continue
            
            p = float(inst.args_copy()[0])
            if p <= 0:
                continue
                
            t_str = " ".join(str(t) for t in inst.targets_copy())
            
            # Count how many components this instruction splits into
            components = 0
            has_targets = False
            for t in inst.targets_copy():
                if t.is_separator():
                    if has_targets:
                        components += 1
                        has_targets = False
                elif t.is_relative_detector_id() or t.is_logical_observable_id():
                    has_targets = True
            if has_targets:
                components += 1
                
            indices = list(range(idx, idx + components))
            self.target_str_to_indices[t_str] = indices
            idx += components
            
        assert idx == self.num_mechanisms

    def probabilities(self, p: float) -> np.ndarray:
        c_p = self.noise_model(self.circuit, p)
        dem_p = c_p.detector_error_model(decompose_errors=True, flatten_loops=True)
        
        from .rare_event import append_global_dem_event
        if self.global_dem_event_probability > 0:
            append_global_dem_event(dem_p, self.global_dem_event_probability * (p / self.p_ref))

        probs = np.zeros(self.num_mechanisms, dtype=np.float64)
        
        for inst in dem_p:
            if inst.type != "error":
                continue
            inst_p = float(inst.args_copy()[0])
            if inst_p <= 0:
                continue
            
            t_str = " ".join(str(t) for t in inst.targets_copy())
            if t_str in self.target_str_to_indices:
                for idx in self.target_str_to_indices[t_str]:
                    probs[idx] = inst_p
            # We silently ignore new error mechanisms that did not exist at p_ref.
            # This is correct behavior because the MCMC state space is fixed by p_ref.
            # (In practice, if p_ref is sufficiently large, all mechanisms will be present.)

        if np.any(probs >= 1.0):
            raise ValueError("Exact mechanism probability reached >= 1.")
            
        return probs

    def explain_mechanism(self, index: int) -> list[stim.ExplainedError]:
        """Explain a specific mechanism using stim's explain_detector_error_model_errors.
        
        Args:
            index: The index of the mechanism in the catalog.
            
        Returns:
            A list of stim.ExplainedError objects detailing the physical circuit 
            faults that cause this DEM mechanism.
        """
        if index < 0 or index >= self.num_mechanisms:
            raise IndexError(f"Mechanism index {index} out of range [0, {self.num_mechanisms})")
            
        mech = self.catalog.mechanisms[index]
        dem_filter = stim.DetectorErrorModel(str(mech))
        
        # We must explain using the circuit decorated at a non-zero physical rate
        # p_ref is guaranteed to be non-zero as it generated the catalog.
        c_ref = self.noise_model(self.circuit, self.p_ref)
        res: list[stim.ExplainedError] = c_ref.explain_detector_error_model_errors(dem_filter=dem_filter)
        return res

    def explain_malignant_set(self, active_set: set[int] | tuple[int, ...]) -> list[stim.ExplainedError]:
        """Explain a malignant set of mechanisms.
        
        Args:
            active_set: A collection of active mechanism indices.
            
        Returns:
            A list of stim.ExplainedError objects covering all active mechanisms.
        """
        dem_str = ""
        for index in active_set:
            if index < 0 or index >= self.num_mechanisms:
                raise IndexError(f"Mechanism index {index} out of range [0, {self.num_mechanisms})")
            dem_str += str(self.catalog.mechanisms[index]) + "\n"
            
        dem_filter = stim.DetectorErrorModel(dem_str.strip())
        c_ref = self.noise_model(self.circuit, self.p_ref)
        res: list[stim.ExplainedError] = c_ref.explain_detector_error_model_errors(dem_filter=dem_filter)
        return res

    def __str__(self) -> str:
        return f"ExactNoiseErrorModel(p_ref={self.p_ref}, noise_model={self.noise_model.__class__.__name__})\nCatalog: {self.catalog}"
        
    def __repr__(self) -> str:
        return self.__str__()

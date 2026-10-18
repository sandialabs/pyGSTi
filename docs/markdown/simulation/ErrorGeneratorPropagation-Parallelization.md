---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Parallelization for Error Generator Propagation

In this tutorial we provide a comprehensive guide to multicore parallelization and performance optimization options across pyGSTi's error generator propagation module.

As circuit depth, qubit count, or the number of terms in a noise model grows—particularly when evaluating higher-order BCH/Magnus expansions, computing operator Taylor expansions, or evaluating many probabilities or pauli observables—the computational workload of error generator propagation increases significantly. pyGSTi includes multicore multiprocessing parallelization across all core propagation and sensitivity routines to accelerate these workflows.

```{code-cell} ipython3
import pygsti
import stim
import numpy as np
from pygsti.tools import errgenproptools as eprop
from pygsti.tools.lindbladtools import random_CPTP_error_generator_rates
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.tools.mptools import SharedWorkerPool, warn_if_insufficient_memory
```

## Model and Circuit Setup

To demonstrate parallel execution, we first construct a 4-qubit crosstalk-free model with random weight-2 H+S (coherent + Pauli stochastic) errors.

```{code-cell} ipython3
num_qubits = 4
gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
availability = {'Gcphase': [(0, 1), (1, 2), (2, 3), (3, 0)]}
pspec = pygsti.processors.QubitProcessorSpec(num_qubits, gate_names, availability=availability)
target_model = pygsti.models.create_crosstalk_free_model(processor_spec=pspec)

qubit_labels = pspec.qubit_labels
error_rates_dict = {}
for gate, avail in pspec.availability.items():
    n = pspec.gate_num_qubits(gate)
    qubits_for_gate = qubit_labels if avail == 'all-edges' else avail
    for qs in qubits_for_gate:
        label = pygsti.baseobjs.Label(gate, qs)
        error_rates_dict[label] = random_CPTP_error_generator_rates(
            num_qubits=n, errorgen_types=('H', 'S'), label_type='local', seed=1234
        )

error_model = pygsti.models.create_crosstalk_free_model(pspec, lindblad_error_coeffs=error_rates_dict)
c = pygsti.algorithms.randomcircuit.create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)
errorgen_propagator = ErrorGeneratorPropagator(error_model)
print(c)
```

## The `num_workers` and `pool` Arguments

All primary propagation and sensitivity functions accept two optional arguments to control parallel execution:

- `num_workers`: 
  - `None` (default): Runs strictly sequentially on the main thread.
  - `int`: An explicit number of worker processes to spawn (e.g., `num_workers=4`).
  - `"auto"`: Automatically queries physical hardware cores, reserving one core on $\ge 4$-core systems for the operating system and master thread while strictly respecting container cgroup quotas.
- `pool`: An optional existing multiprocessing pool (`SharedWorkerPool` or `multiprocessing.Pool`) to reuse across calls. When `pool` is provided, `num_workers` is inferred directly from the pool.

Functions supporting parallel execution include:
- `ErrorGeneratorPropagator.propagate_errorgens` and `propagate_errorgens_bch`: Parallelizes circuit layer propagation across workers.
- `errgenproptools.magnus_expansion` and `zassenhaus_formula`: Parallelizes pairwise layer commutators and outer accumulated commutators.
- `errgenproptools.error_generator_taylor_expansion` and `_commuting_product`: Parallelizes operator powers and compositions.
- `errgenproptools.approximate_stabilizer_probabilities` and `approximate_stabilizer_pauli_expectations`: Parallelizes computations across outcomes/observables while reusing intermediate results.
- `errgenproptools.bulk_alpha` and `bulk_alpha_pauli`: Parallelizes across outcomes/observables and error generators.

## Pipeline Pool Reuse with `SharedWorkerPool`

In multi-stage workflows (such as propagating layers, computing a second- or third-order Magnus expansion, and calculating multiple outcome probabilities), spinning up and tearing down a multiprocessing pool for each function call introduces substantial process-creation overhead.

To eliminate this overhead, pyGSTi provides `SharedWorkerPool`. When initialized as a context manager, workers are spawned once and optionally pre-warmed with essential modules. These modules can be manually specified, but those relevant for error generator propagation are given by `eprop._EGPROP_PREWARM_PACKAGES` (`stim`, `numpy`, and pyGSTi Cython extensions):

```{code-cell} ipython3
# Reusing a single pre-warmed worker pool across sequential pipeline stages:
with SharedWorkerPool(num_workers="auto", prewarm_packages=_EGPROP_PREWARM_PACKAGES) as pool:
    # 1. Parallel circuit layer propagation
    propagated_layers = errorgen_propagator.propagate_errorgens(c, pool=pool)

    # 2. Parallel second-order Magnus expansion
    effective_generator = eprop.magnus_expansion(propagated_layers, magnus_order=2, pool=pool)

    # 3. Parallel third-order Magnus expansion
    effective_generator_order3 = eprop.magnus_expansion(propagated_layers, magnus_order=3, pool=pool)

    # 4. Parallel probability calculation for specific bitstrings of interest
    subset_probs = eprop.approximate_stabilizer_probabilities(
        effective_generator, c, bitstrings=["0000", "0101"], order=2, pool=pool
    )
    print("Subset probabilities:", subset_probs)

    # 5. Parallel bulk Pauli expectation value calculations
    paulis = [stim.PauliString("ZIII"), stim.PauliString("XYZI")]
    expectations = eprop.approximate_stabilizer_pauli_expectations(
        effective_generator, c, paulis=paulis, order=2, pool=pool
    )
    print("Pauli expectations:", expectations)
```

A `SharedWorkerPool` must be used inside its `with` block. By default its worker processes start as soon as the block is entered. Passing `lazy=True` defers starting them until a stage first dispatches work, so if every stage decides its workload is too small to benefit from parallelism, no processes are started at all. Functions that are given `num_workers` but no `pool` (for example `propagate_errorgens_bch`) use a lazy pool internally and share it across their stages.

## Automated Threshold Cutoffs & Manual Configuration

Multiprocessing incurs inter-process communication (IPC) overhead. If a circuit has only a few error generators or shallow depth, dispatching tasks across processes can be slower than executing sequentially on a single core.

By default, pyGSTi uses an internal multi-scale cutoff table (`_DEFAULT_THRESHOLD_TABLE`) based on system size (e.g. minimum number of layer pairs or generator-outcome product $K \times M$) to determine whether parallelization should be engaged.

If you wish to customize these threshold criteria or force parallel/serial modes for benchmarking, you can use `configure_errorgen_propagation_parallelization`:

```{code-cell} ipython3
# Check or customize parallelization thresholds
eprop.configure_errorgen_propagation_parallelization(min_km_alpha=500, min_magnus_pairs=2)

# Unconditionally force parallel execution regardless of workload size (useful for testing):
eprop.configure_errorgen_propagation_parallelization(force_parallel=True)

# Restore default pyGSTi heuristics:
eprop.configure_errorgen_propagation_parallelization(reset_to_defaults=True)
```

## Dynamic Hardware Calibration with `optimize_errorgen_propagation_parallelization` (EXPERIMENTAL)

While the built-in multi-scale threshold table provides sensible defaults across small, medium, and large qubit systems, optimal crossover thresholds depend on machine-specific hardware characteristics:
- Single-threaded CPU throughput ($\tau_\alpha$) for stabilizer tableau operations
- Inter-process communication (IPC) dispatch latency ($T_{\text{IPC}}$), which varies significantly across operating systems and start methods (`fork` on Linux vs. `spawn` on macOS/Windows)
- Memory bus bandwidth and core counts
- Etc.

To calibrate thresholds empirically for your specific host machine, you can run `optimize_errorgen_propagation_parallelization`:

```{code-cell} ipython3
# Empirically profile single-threaded throughput and IPC latency for a 4-qubit circuit:
calib_results = eprop.optimize_errorgen_propagation_parallelization(num_qubits=4, set_as_default=True)
print(f"Measured single-term evaluation time: {calib_results['tau_alpha_us']:.2f} µs")
print(f"Measured round-trip IPC latency:      {calib_results['t_ipc_s']*1000:.3f} ms")
print(f"Calibrated crossover threshold (KM):  {calib_results['threshold_KM']:,} operations")
```

`optimize_errorgen_propagation_parallelization` runs a set of lightweight microbenchmarks that can be tailored to your problem size in order to measure empirical system performance characteristics that can be used to compute when parallelization is expected to be a net win. When `set_as_default=True` (the default), the calibrated crossover thresholds are cached for the active session and supersede the built-in reference defaults. You can pass `num_qubits`, a `Circuit`, or a `Model` to tailor the profiling to your specific circuit geometry.

Note: Your configuration can be reset to the default settings at any time using `configure_errorgen_propagation_parallelization`. The empirical profiling performed by `optimize_errorgen_propagation_parallelization` is still experimental, so your mileage may vary.

```{code-cell} ipython3
# Reset back to built-in reference defaults when finished:
eprop.configure_errorgen_propagation_parallelization(reset_to_defaults=True)
```

## Peak Memory Estimation and Headroom Warnings

High-order Magnus expansions and Taylor series compositions on dense multiqubit circuits can generate large dictionaries of elementary error generators. To protect against out-of-memory (OOM) errors pyGSTi provides analytical peak memory estimators:

- `eprop.estimate_magnus_peak_memory(n_qubits, num_layers, avg_terms_per_layer, num_workers, magnus_order)`
- `eprop.estimate_taylor_peak_memory(n_qubits, num_terms, order, num_workers)`
- `eprop.estimate_bulk_alpha_peak_memory(n_qubits, n_errorgens, n_outcomes, num_workers)`
- `pygsti.errorgenpropagation.errorpropagator.estimate_propagation_peak_memory(n_qubits, n_layers, avg_errorgens_per_layer, num_workers)`

These estimate the *additional* memory an operation needs beyond what the calling process already uses, accounting for the memory of each `LocalStimErrorgenLabel` dictionary entry and for the baseline footprint and buffers of each worker process. Because the exact outputs are not known in advance, the estimates are upper bounds: they assume, for example, that every commutator of two elementary error generators produces the maximal number of terms (at most 4), and that every anticommutator produces at most 7. They can therefore substantially overestimate the memory actually needed.

For layer propagation, Taylor expansions and sensitivity (`bulk_alpha`) calculations, pyGSTi automatically issues an `InsufficientMemoryWarning` (from `pygsti.tools.exceptions`) if the estimated workload exceeds available system memory. This automatic check is currently disabled for Magnus expansions, where the estimates were found to be too conservative to be useful, but you can still check manually:

```{code-cell} ipython3
# Estimate memory footprint for a 50-layer, 4-qubit Magnus expansion:
est_bytes = eprop.estimate_magnus_peak_memory(n_qubits=4, num_layers=50, avg_terms_per_layer=25, num_workers=4, magnus_order=2)
print(f"Estimated peak memory: {est_bytes / (1024**2):.2f} MiB")

# Check against system memory safely:
warn_if_insufficient_memory(est_bytes, task_name="example_magnus_expansion")
```

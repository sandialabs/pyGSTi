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

As circuit depth, qubit count, or noise model density grows—particularly when evaluating higher-order BCH/Magnus expansions, computing operator Taylor expansions, or evaluating probabilities across many observables—the computational workload of error generator propagation increases significantly. pyGSTi includes multicore multiprocessing parallelization across all core propagation and sensitivity routines to accelerate these workflows.

Key parallelization capabilities include:
- **Zero-overhead sequential fallback**: Parallelization is opt-in; sequential paths remain lightweight and completely free of multiprocessing overhead.
- **Automated hardware detection**: Automatically infers available physical CPU cores while honoring container cgroup quotas on Linux, Windows, and macOS (`num_workers="auto"`).
- **Worker pool reuse**: Pre-warmed `SharedWorkerPool` instances can be shared across sequential pipeline stages, eliminating repetitive process spawn and teardown costs.
- **Dynamic load-balanced scheduling**: Heavy tasks like pairwise commutators in Magnus expansion are decomposed using budget capacity slicing and Longest Processing Time (LPT) bin-packing.
- **Bulk evaluation vectorization**: Taylor series expansions are computed once and reused across batches of computational basis bitstrings or Pauli observables.
- **Machine-specific calibration**: Hardware-specific profiling (`optimize_errorgen_propagation_parallelization`) calibrates exact crossover thresholds for your system's CPU speed and IPC latency.

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

To demonstrate parallel execution, we first construct a 4-qubit crosstalk-free processor and a random weight-2 H+S (coherent + Pauli stochastic) error model.

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
  - `None` (default): Runs strictly sequentially on the main thread, incurring zero IPC or serialization overhead.
  - `int`: An explicit number of worker processes to spawn (e.g., `num_workers=4`).
  - `"auto"`: Automatically queries physical hardware cores, reserving one core on $\ge 4$-core systems for the operating system and master thread while strictly respecting container cgroup quotas.
- `pool`: An optional existing multiprocessing pool (`SharedWorkerPool` or `multiprocessing.Pool`) to reuse across calls. When `pool` is provided, `num_workers` is inferred directly from the pool.

Functions supporting parallel execution include:
- `ErrorGeneratorPropagator.propagate_errorgens` and `propagate_errorgens_bch`: Parallelizes circuit layer propagation across workers.
- `errgenproptools.magnus_expansion` and `zassenhaus_formula`: Parallelizes pairwise layer commutators and outer accumulated commutators using capacity-sliced LPT bin-packing.
- `errgenproptools.error_generator_taylor_expansion` and `_commuting_product`: Parallelizes operator powers and compositions via bilinear term slicing.
- `errgenproptools.approximate_stabilizer_probabilities` and `approximate_stabilizer_pauli_expectations`: Parallelizes 2D sensitivity grids across outcomes/observables while reusing intermediate Taylor expansions.
- `errgenproptools.bulk_alpha` and `bulk_alpha_pauli`: 2D process grid partitioning across outcomes and error generators.

## Pipeline Pool Reuse with `SharedWorkerPool`

In multi-stage workflows (such as propagating layers, computing a second- or third-order Magnus expansion, and calculating multiple outcome probabilities), spinning up and tearing down a multiprocessing pool for each function call introduces substantial process-creation latency.

To eliminate this overhead, pyGSTi provides `SharedWorkerPool`. When initialized as a context manager, workers are spawned once and pre-warmed with essential modules (`stim`, `numpy`, and pyGSTi Cython extensions):

```{code-cell} ipython3
# Reusing a single pre-warmed worker pool across sequential pipeline stages:
with SharedWorkerPool(num_workers="auto") as pool:
    # 1. Parallel circuit layer propagation
    propagated_layers = errorgen_propagator.propagate_errorgens(c, pool=pool)

    # 2. Parallel second-order Magnus expansion with load-balanced task slicing
    effective_generator = eprop.magnus_expansion(propagated_layers, magnus_order=2, pool=pool)

    # 3. Parallel third-order Magnus expansion (Phase 1 pairwise reduction + Phase 2 outer commutators)
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

## Bulk Evaluations and Observable Batching

When evaluating outcome probabilities or Pauli expectation values across multiple targets, passing observables in bulk allows pyGSTi to compute the error generator Taylor series expansion once and evaluate the sensitivity grid across all requested targets:

- `approximate_stabilizer_probabilities(..., bitstrings=...)`: Rather than calling `approximate_stabilizer_probability` in a Python loop, pass a sequence of bitstrings to evaluate. If `bitstrings` is omitted, the exhaustive distribution over all $2^n$ computational basis states is computed (a warning is issued if $n > 10$ due to exponential scaling).
- `approximate_stabilizer_pauli_expectations(..., paulis=...)`: Accepts a sequence of Pauli observables (as strings or `stim.PauliString` objects) and computes all expectation values in a single vectorized pass.

```{code-cell} ipython3
# Vectorized evaluation of multiple Pauli expectations
pauli_list = ["XXXX", "YYYY", "ZZZZ", "ZIII"]
bulk_expectations = eprop.approximate_stabilizer_pauli_expectations(
    effective_generator, c, paulis=pauli_list, order=2, num_workers=2
)
for p, val in zip(pauli_list, bulk_expectations):
    print(f"< {p} > = {val:.6f}")
```

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

## Dynamic Hardware Calibration with `optimize_errorgen_propagation_parallelization`

While the built-in multi-scale threshold table provides sensible defaults across small, medium, and large qubit systems, optimal crossover thresholds depend on machine-specific hardware characteristics:
- Single-threaded CPU throughput ($\tau_\alpha$) for stabilizer tableau operations
- Inter-process communication (IPC) dispatch latency ($T_{\text{IPC}}$), which varies significantly across operating systems and start methods (`fork` on Linux vs. `spawn` on macOS/Windows)
- Memory bus bandwidth and core counts

To calibrate thresholds empirically for your specific host machine, you can run `optimize_errorgen_propagation_parallelization`:

```{code-cell} ipython3
# Empirically profile single-threaded throughput and IPC latency for a 4-qubit circuit:
calib_results = eprop.optimize_errorgen_propagation_parallelization(num_qubits=4, set_as_default=True)
print(f"Measured single-term evaluation time: {calib_results['tau_alpha_us']:.2f} µs")
print(f"Measured round-trip IPC latency:      {calib_results['t_ipc_s']*1000:.3f} ms")
print(f"Calibrated crossover threshold (KM):  {calib_results['threshold_KM']:,} operations")
```

When `set_as_default=True` (the default), the calibrated crossover thresholds are cached in the active session and supersede the built-in reference defaults for subsequent `bulk_alpha`, `bulk_alpha_pauli`, and probability calculations. You can pass `num_qubits`, a `Circuit`, or a `Model` to tailor the profiling to your specific circuit geometry.

```{code-cell} ipython3
# Reset back to built-in reference defaults when finished:
eprop.configure_errorgen_propagation_parallelization(reset_to_defaults=True)
```

## Peak Memory Estimation and Headroom Warnings

High-order Magnus expansions and Taylor series compositions on dense multiqubit circuits can generate large dictionaries of elementary error generators. To protect against out-of-memory (OOM) termination on shared compute nodes or memory-constrained CI environments, pyGSTi provides analytical peak memory estimators:

- `estimate_magnus_peak_memory(n_qubits, num_layers, avg_terms_per_layer, num_workers, magnus_order)`
- `estimate_taylor_peak_memory(n_qubits, num_terms, order, num_workers)`

These models account for dictionary bucket allocations, `LocalStimErrorgenLabel` footprint, process worker concurrency, and the theoretical upper bounds on terms emitted by commutators ($\le 4$ terms) and anticommutators ($\le 7$ terms).

If an estimated workload exceeds available system memory, pyGSTi automatically emits a descriptive `ResourceWarning`:

```{code-cell} ipython3
# Estimate memory footprint for a 50-layer, 4-qubit Magnus expansion:
est_bytes = eprop.estimate_magnus_peak_memory(n_qubits=4, num_layers=50, avg_terms_per_layer=25, num_workers=4, magnus_order=2)
print(f"Estimated peak memory: {est_bytes / (1024**2):.2f} MiB")

# Check against system memory safely:
warn_if_insufficient_memory(est_bytes, task_name="example_magnus_expansion")
```

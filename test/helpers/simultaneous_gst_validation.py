"""Shared helpers for simultaneous-GST validation runs."""

import dataclasses
import datetime
import functools
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Literal, Sequence

import numpy as np

from pygsti.algorithms.fiducialselection import find_fiducials
from pygsti.algorithms.germselection import find_germs
from pygsti.data import simulate_data
from pygsti.models import create_cloud_crosstalk_model, create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.gst import GateSetTomography, StandardGSTDesign
from pygsti.protocols.protocol import ProtocolData
from pygsti.protocols.simultaneous_gst import make_simultaneous_gst_design
from pygsti.tools import two_delta_logl, tvd


REPO = pathlib.Path(__file__).resolve().parents[2]


SPARSE_TERMS = {
    'Gi': (('H', 'Z'), ('S', 'X'), ('S', 'Y'), ('S', 'Z')),
    'Gxpi2': (('H', 'Z'), ('S', 'Z')),
    'Gypi2': (('H', 'X'), ('S', 'Z')),
    'Gcnot': (('H', 'ZZ'), ('S', 'XX'), ('S', 'ZZ')),
    'Gii': (('H', 'ZI'), ('H', 'IZ'), ('S', 'XI'), ('S', 'IX'), ('S', 'ZI'), ('S', 'IZ')),
}


def _full_hs_terms(num_qubits: int) -> tuple[tuple[str, str], ...]:
    pauli_labels = (
        ''.join(label)
        for label in itertools.product('IXYZ', repeat=num_qubits)
        if any(pauli != 'I' for pauli in label)
    )
    labels = tuple(pauli_labels)
    return tuple((term_type, label) for term_type in ('H', 'S') for label in labels)


FULL_HS_TERMS = {
    gate_name: _full_hs_terms(2 if gate_name in ('Gcnot', 'Gii') else 1)
    for gate_name in SPARSE_TERMS
}


@dataclasses.dataclass(frozen=True)
class ValidationProfile:
    """Configuration for one validation run, and the 1Q and 2Q component models that select its designs."""

    name: str
    nqubits: int
    max_lengths: tuple[int, ...]
    shots: int
    model_terms: Literal['sparse', 'full_hs']
    scenario: Literal['in_class', 'spectator_crosstalk']
    seed: int
    spectator_error: float = 0.0
    # The error injected into `Gcnot(0, 1)` for spectator-crosstalk scenarios, as a
    # `create_cloud_crosstalk_model` coefficient key.  `('S', 'Z:2')` dephases the
    # spectator.  It is weight-1, so the hypothesis class can absorb part of it, but only as
    # error on qubit 2's own gates in every layer, not just the layers containing the CNOT.
    # `('H', 'ZZ:1,2')` is a weight-2 coherent coupling of the gate's target to the
    # spectator, which no product of local errors reproduces, even to first order.
    spectator_term: tuple[str, ...] = ('S', 'Z:2')
    # The two-qubit gate connectivity: `'ring'` adds the wraparound edge to the line.
    topology: Literal['line', 'ring'] = 'line'

    @classmethod
    def sparse_line(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                    scenario: Literal['in_class', 'spectator_crosstalk'] = 'in_class',
                    seed: int = 20260901, spectator_error: float = 0.0,
                    topology: Literal['line', 'ring'] = 'line') -> 'ValidationProfile':
        return cls(f'{nqubits}q_sparse_{topology}', nqubits, tuple(max_lengths), shots,
                   'sparse', scenario, seed, spectator_error, topology=topology)

    # S(Z:2) at 0.01 is large enough that the training data reveal it -- the
    # crosstalk-free fit's 2*deltaLogL exceeds the data-generating model's by about 1200
    # at 3Q -- while that fit still removes most of the held-out error.  The earlier
    # default, H(Z:2) at 0.005, left the training likelihood unchanged.
    @classmethod
    def sparse_spectator_line(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                              seed: int = 20260901,
                              spectator_error: float = 0.01,
                              spectator_term: tuple[str, ...] = ('S', 'Z:2'),
                              topology: Literal['line', 'ring'] = 'line') -> 'ValidationProfile':
        return cls(f'{nqubits}q_sparse_spectator_{topology}', nqubits, tuple(max_lengths), shots,
                   'sparse', 'spectator_crosstalk', seed, spectator_error, tuple(spectator_term),
                   topology)

    @classmethod
    def full_hs_line(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                     scenario: Literal['in_class', 'spectator_crosstalk'] = 'in_class',
                     seed: int = 20260901, spectator_error: float = 0.0,
                     topology: Literal['line', 'ring'] = 'line') -> 'ValidationProfile':
        return cls(f'{nqubits}q_full_hs_{topology}', nqubits, tuple(max_lengths), shots,
                   'full_hs', scenario, seed, spectator_error, topology=topology)

    @functools.cached_property
    def oneq_model(self):
        return _build_model(_oneq_processor_spec(), self.model_terms, independent_gates=False)

    @functools.cached_property
    def twoq_model(self):
        return _build_model(_processor_spec(2), self.model_terms, independent_gates=False)


@dataclasses.dataclass(frozen=True)
class ValidationArtifacts:
    root: pathlib.Path
    dataset_dir: pathlib.Path
    results_dir: pathlib.Path
    manifest_path: pathlib.Path


THREE_QUBIT_SPARSE_IN_CLASS = ValidationProfile.sparse_line(
    nqubits=3, max_lengths=(1, 2, 4))
THREE_QUBIT_SPARSE_SPECTATOR = ValidationProfile.sparse_spectator_line(
    nqubits=3, max_lengths=(1, 2, 4))
FOUR_QUBIT_SPARSE_IN_CLASS = ValidationProfile.sparse_line(
    nqubits=4, max_lengths=(1, 2, 4))
FOUR_QUBIT_SPARSE_SPECTATOR = ValidationProfile.sparse_spectator_line(
    nqubits=4, max_lengths=(1, 2, 4))
THREE_QUBIT_SPARSE_RING_IN_CLASS = ValidationProfile.sparse_line(
    nqubits=3, max_lengths=(1, 2, 4), topology='ring')
THREE_QUBIT_SPARSE_RING_SPECTATOR = ValidationProfile.sparse_spectator_line(
    nqubits=3, max_lengths=(1, 2, 4), topology='ring')
FOUR_QUBIT_SPARSE_RING_IN_CLASS = ValidationProfile.sparse_line(
    nqubits=4, max_lengths=(1, 2, 4), topology='ring')
FOUR_QUBIT_SPARSE_RING_SPECTATOR = ValidationProfile.sparse_spectator_line(
    nqubits=4, max_lengths=(1, 2, 4), topology='ring')


def _oneq_processor_spec() -> QubitProcessorSpec:
    return QubitProcessorSpec(
        1, gate_names=['Gi', 'Gxpi2', 'Gypi2'], qubit_labels=(0,)
    )


def _processor_spec(nqubits: int, topology: Literal['line', 'ring'] = 'line') -> QubitProcessorSpec:
    qubits = tuple(range(nqubits))
    oneq_locations = [(qubit,) for qubit in qubits]
    edges = [(qubit, qubit + 1) for qubit in range(nqubits - 1)]
    if topology == 'ring':
        if nqubits < 3:
            raise ValueError('A ring needs at least three qubits; on two it is the line')
        # Oriented (0, n-1), the way edge colorings record it, because the stitcher places
        # the two-qubit design on each edge in the orientation its patch gives.
        edges.append((0, nqubits - 1))
    availability = {
        'Gi': oneq_locations,
        'Gxpi2': oneq_locations,
        'Gypi2': oneq_locations,
        'Gcnot': edges,
        'Gii': edges,
    }
    return QubitProcessorSpec(
        nqubits,
        gate_names=['Gi', 'Gxpi2', 'Gypi2', 'Gcnot', 'Gii'],
        nonstd_gate_unitaries={'Gii': np.eye(4)},
        availability=availability,
        qubit_labels=qubits,
    )


def _terms_by_gate(model_terms: Literal['sparse', 'full_hs']) -> dict:
    if model_terms == 'sparse':
        return SPARSE_TERMS
    return FULL_HS_TERMS


def _zero_term_dict(model_terms: Literal['sparse', 'full_hs'], gate_names) -> dict:
    terms_by_gate = _terms_by_gate(model_terms)
    return {
        gate_name: {term: 0.0 for term in terms_by_gate[gate_name]}
        for gate_name in gate_names
    }


def _build_model(processor_spec: QubitProcessorSpec,
                 model_terms: Literal['sparse', 'full_hs'], independent_gates: bool):
    return create_crosstalk_free_model(
        processor_spec,
        lindblad_error_coeffs=_zero_term_dict(model_terms, processor_spec.gate_names),
        lindblad_parameterization='H+S',
        independent_gates=independent_gates,
    )


def spectator_support(spectator_term: Sequence[str]) -> tuple:
    """Return the qubits a `create_cloud_crosstalk_model` coefficient key acts on.

    Keys carry their support in the basis-element labels, as `<paulis>:<qubits>` with
    the qubits comma-separated -- `'Z:2'` is weight-1 on qubit 2, `'ZZ:1,2'` is weight-2
    on qubits 1 and 2.
    """
    qubits = set()
    for basis_element in spectator_term[1:]:
        if ':' not in basis_element:
            continue
        for label in basis_element.split(':', maxsplit=1)[1].split(','):
            try:
                qubits.add(int(label))
            except ValueError:
                qubits.add(label)
    return tuple(sorted(qubits, key=str))


def spectator_term_text(spectator_term: Sequence[str]) -> str:
    """Render a coefficient key for manifests and reports, e.g. `H(ZZ:1,2)`."""
    return f"{spectator_term[0]}({','.join(spectator_term[1:])})"


def _set_seeded_error_coefficients(model, seed: int, spectator_gate=None,
                                   protected_support=None) -> None:
    """Set every local coefficient to a small nonzero value drawn from an RNG seeded with `seed`.

    Coefficients on `spectator_gate` whose support matches `protected_support` are left
    alone, so the deliberately injected crosstalk term is not overwritten by the
    seeded local noise.
    """
    rng = np.random.default_rng(seed)
    for op_label, coefficients in sorted(model.errorgen_coefficients().items(), key=lambda item: str(item[0])):
        if op_label not in model.operation_blks['layers']:
            continue
        updates = {}
        for coefficient_label in sorted(coefficients, key=str):
            is_spectator_gate = (
                spectator_gate is not None
                and (op_label.name,) + tuple(op_label.sslbls) == spectator_gate
            )
            if (is_spectator_gate
                    and protected_support is not None
                    and coefficient_label.support == protected_support):
                continue
            if coefficient_label.errorgen_type == 'H':
                value = rng.uniform(-1e-3, 1e-3)
                if value == 0.0:
                    value = 1e-3
            else:
                value = rng.uniform(2.5e-4, 1e-3)
            updates[coefficient_label] = value
        model._op_decomposition(op_label)[1].set_errorgen_coefficients(updates, truncate=False)


def _build_spectator_datagen_model(profile: ValidationProfile):
    if profile.nqubits < 3:
        raise ValueError('Spectator-crosstalk profiles require at least three qubits')
    processor_spec = _processor_spec(profile.nqubits, profile.topology)
    terms_by_gate = _terms_by_gate(profile.model_terms)
    coefficients = {
        (gate_name,) + tuple(placement): {
            term: 0.0 for term in terms_by_gate[gate_name]
        }
        for gate_name in processor_spec.gate_names
        for placement in processor_spec.resolved_availability(gate_name)
    }
    spectator_gate = ('Gcnot', 0, 1)
    support = spectator_support(profile.spectator_term)
    if not support or all(qubit in spectator_gate[1:] for qubit in support):
        raise ValueError(
            f'Spectator term {profile.spectator_term!r} has support {support}, which does not '
            f'reach beyond the qubits {spectator_gate[1:]} that {spectator_gate[0]} acts on; '
            'it would be an ordinary local error on the gate rather than crosstalk')
    coefficients[spectator_gate][tuple(profile.spectator_term)] = profile.spectator_error
    model = create_cloud_crosstalk_model(
        processor_spec,
        lindblad_error_coeffs=coefficients,
        lindblad_parameterization=_spectator_parameterization(profile.spectator_term),
        independent_gates=True,
    )
    _assert_spectator_term_survived(model, spectator_gate, profile)
    _set_seeded_error_coefficients(
        model, profile.seed, spectator_gate=spectator_gate, protected_support=support)
    return model


def _spectator_parameterization(spectator_term: Sequence[str]) -> str:
    """Pick the smallest Lindblad parameterization that can hold the injected term.

    `H+S` builds reduced blocks containing exactly the `H` and `S` generators asked for,
    and *discards* `C` and `A` generators without warning or error.  `GLNDU` keeps the
    same one-parameter-per-generator sparsity while admitting all four sectors, so it is
    used only when the injected term needs it -- terms drawn from `H` and `S` keep the
    original parameterization and reproduce earlier runs exactly.
    """
    return 'GLNDU' if spectator_term[0] in ('C', 'A') else 'H+S'


def _assert_spectator_term_survived(model, spectator_gate, profile: ValidationProfile) -> None:
    """Fail loudly if the injected crosstalk is not in the constructed model.

    Without this the `H+S` silent drop produces a spectator-crosstalk run containing no
    crosstalk, which is indistinguishable from an in-class run and looks entirely healthy.
    """
    wanted_type = profile.spectator_term[0]
    wanted_paulis = tuple(basis_element.split(':', maxsplit=1)[0]
                          for basis_element in profile.spectator_term[1:])
    wanted_support = spectator_support(profile.spectator_term)
    for op_label, coefficients in model.errorgen_coefficients().items():
        if (op_label.name,) + tuple(op_label.sslbls) != spectator_gate:
            continue
        for label, value in coefficients.items():
            if (label.errorgen_type != wanted_type
                    or tuple(label.basis_element_labels) != wanted_paulis
                    or label.support != wanted_support):
                continue
            if abs(np.real(value) - profile.spectator_error) <= 1e-12:
                return
            raise ValueError(
                f'Spectator term {spectator_term_text(profile.spectator_term)} was built with '
                f'coefficient {np.real(value)!r}, not the requested {profile.spectator_error!r}')
    raise ValueError(
        f'Spectator term {spectator_term_text(profile.spectator_term)} is absent from the '
        f'constructed data-generating model; the chosen Lindblad parameterization '
        f'{_spectator_parameterization(profile.spectator_term)!r} cannot represent it')


def build_fit_and_datagen_models(profile: ValidationProfile):
    """Build the crosstalk-free fit model and the scenario's seeded data-generating model.

    The fit model's parameterization defines the hypothesis class.
    """
    fit_model = _build_model(_processor_spec(profile.nqubits, profile.topology),
                             profile.model_terms, independent_gates=True)
    if profile.scenario == 'spectator_crosstalk':
        datagen_model = _build_spectator_datagen_model(profile)
    else:
        datagen_model = fit_model.copy()
        _set_seeded_error_coefficients(datagen_model, profile.seed)
    return fit_model, datagen_model


def _perturb_hamiltonian_terms(model, seed: int) -> None:
    """Move a selector model off symmetry points without changing its parameterization."""
    rng = np.random.default_rng(seed)
    for op_label, coefficients in model.errorgen_coefficients().items():
        if op_label not in model.operations:
            continue
        perturbation = {
            coefficient_label: 1e-3 * rng.random()
            for coefficient_label in coefficients
            if coefficient_label.errorgen_type == 'H'
        }
        model.operations[op_label].set_errorgen_coefficients(perturbation, truncate=False)


def _build_component_design(processor_spec: QubitProcessorSpec, model, max_lengths: tuple[int, ...],
                            seed: int) -> StandardGSTDesign:
    # Fiducial selection currently requires ExplicitOpModel.operations.  This
    # conversion preserves the reduced H+S parameterization and the component
    # model's tied parameters; it does not substitute a full model or model pack.
    explicit_model = model.to_explicit_model()
    prep_fiducials, meas_fiducials = find_fiducials(
        explicit_model,
        algorithm='greedy',
        candidate_fid_counts=3,
        candidate_seed=seed,
        verbosity=0,
    )

    # find_germs(randomize=True) converts each randomized operation to a full
    # parameterization.  Perturb only the allowed Hamiltonian coefficients and
    # disable that conversion so selection remains on the matching support.
    germ_selection_model = explicit_model.copy()
    _perturb_hamiltonian_terms(germ_selection_model, seed)
    germs = find_germs(
        germ_selection_model,
        randomize=False,
        seed=seed,
        candidate_germ_counts={3: 'all upto'},
        force=None,
        mode='all-Jac',
        verbosity=0,
    )
    if not germs:
        raise RuntimeError('Unable to select an amplificationally complete germ set')

    return StandardGSTDesign(
        processor_spec,
        prep_fiducials,
        meas_fiducials,
        germs,
        list(max_lengths),
        qubit_labels=tuple(processor_spec.qubit_labels),
    )


def build_component_designs(profile: ValidationProfile) -> tuple[StandardGSTDesign, StandardGSTDesign]:
    """Select reproducible 1Q and 2Q GST component designs.

    The component models share parameters across gate placements.
    """
    oneq_processor_spec = _oneq_processor_spec()
    twoq_processor_spec = _processor_spec(2)
    oneq_design = _build_component_design(
        oneq_processor_spec, profile.oneq_model, profile.max_lengths, profile.seed
    )
    twoq_design = _build_component_design(
        twoq_processor_spec, profile.twoq_model, profile.max_lengths, profile.seed + 1
    )
    return oneq_design, twoq_design


@dataclasses.dataclass(frozen=True)
class HeldOutCircuits:
    """Held-out circuits, and the subset first added at the largest maximum length L.

    An error injected during `Gcnot(0, 1)` accumulates with the number of times
    that gate is applied, so the deepest circuits carry the most signal about it.  The
    circuits a nested design first adds at its largest L include its deepest ones, and
    scoring them separately measures the effect where it is largest, without changing the
    data-generating model.
    """

    circuits: tuple
    deepest: tuple


def _deepest_increment(design) -> set:
    """Return the circuits a nested design first adds at its largest maximum length L."""
    circuit_lists = design.circuit_lists
    if len(circuit_lists) < 2:
        return set(circuit_lists[-1])
    return set(circuit_lists[-1]) - set(circuit_lists[-2])


def build_training_design_and_held_out_circuits(profile: ValidationProfile):
    """Build one training design and a reproducible held-out circuit set disjoint from it."""
    processor_spec = _processor_spec(profile.nqubits, profile.topology)
    oneq_design, twoq_design = build_component_designs(profile)
    training_design = make_simultaneous_gst_design(
        processor_spec, oneq_design, twoq_design, seed=profile.seed)

    held_out_profile = dataclasses.replace(profile, seed=profile.seed + 2)
    held_out_oneq_design, held_out_twoq_design = build_component_designs(held_out_profile)
    held_out_design = make_simultaneous_gst_design(
        processor_spec,
        held_out_oneq_design,
        held_out_twoq_design,
        seed=held_out_profile.seed,
    )
    training_circuits = set(training_design.all_circuits_needing_data)
    # The stitched design can list the same circuit more than once -- a circuit reached by
    # two different germ/power combinations appears once per combination.  Held-out scores
    # are per distinct circuit, so a repeat would weight that circuit twice in the mean
    # without telling anyone; deduplicate here, in the order the design produced them.
    seen = set()
    held_out = []
    for circuit in held_out_design.all_circuits_needing_data:
        if circuit in training_circuits or circuit in seen:
            continue
        seen.add(circuit)
        held_out.append(circuit)
    held_out_circuits = tuple(held_out)
    if not held_out_circuits:
        raise RuntimeError('The second seeded design produced no held-out circuits')

    deepest_increment = _deepest_increment(held_out_design)
    deepest = tuple(circuit for circuit in held_out_circuits if circuit in deepest_increment)
    if not deepest:
        raise RuntimeError(
            'The second seeded design produced no held-out circuits at its largest maximum length')
    return training_design, HeldOutCircuits(circuits=held_out_circuits, deepest=deepest)


def openmpi_extra_args() -> list[str]:
    """Return launcher arguments supported specifically by Open MPI."""
    launcher = shutil.which('mpiexec') or shutil.which('mpirun')
    if launcher is None:
        return []
    try:
        completed = subprocess.run(
            [launcher, '--version'], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    output = completed.stdout + completed.stderr
    return ['--oversubscribe'] if 'Open MPI' in output or 'OpenRTE' in output else []


def generate_finite_shot_data(profile: ValidationProfile, datagen_model, circuits):
    """Sample reproducible multinomial counts for a validation profile."""
    return simulate_data(
        datagen_model,
        circuits,
        num_samples=profile.shots,
        sample_error='multinomial',
        seed=profile.seed,
    )


def _probabilities_and_jacobian(model, circuits):
    """Stack outcome probabilities and their parameter derivatives, one row per (circuit, outcome).

    Also returns, for each row, the index in `circuits` of the circuit it belongs to.
    """
    probabilities = model.sim.bulk_probs(circuits)
    derivatives = model.sim.bulk_dprobs(circuits)
    rows = [(index, probabilities[circuit][outcome], derivatives[circuit][outcome])
            for index, circuit in enumerate(circuits) for outcome in probabilities[circuit]]
    owner, values, jacobian = zip(*rows)
    return np.array(values), np.array(jacobian), np.array(owner)


def expected_fisher_information(model, dataset) -> np.ndarray:
    """Expected multinomial Fisher information of `dataset`'s circuits and shot counts at `model`."""
    circuits = list(dataset.keys())
    values, jacobian, owner = _probabilities_and_jacobian(model, circuits)
    shots = np.array([dataset[circuit].total for circuit in circuits])[owner]
    return (jacobian.T * (shots / np.clip(values, 1e-12, None))) @ jacobian


def _identifiable_eigenpairs(fisher: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of `fisher` above a relative threshold: the directions the data identify."""
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    identifiable = eigenvalues > 1e-10 * eigenvalues.max()
    return eigenvalues[identifiable], eigenvectors[:, identifiable]


def identifiable_parameter_count(model, dataset) -> int:
    """Return the rank of the expected Fisher information of `dataset` at `model`.

    This is the number of parameter directions the data identify, and the degrees of
    freedom in Wilks' theorem; gauge and other unidentifiable directions do not count.
    """
    eigenvalues, _ = _identifiable_eigenpairs(expected_fisher_information(model, dataset))
    return len(eigenvalues)


def shot_noise_held_out_tvd_quantiles(model, dataset, held_out_circuits, quantile: float = 0.999,
                                      samples: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Return quantiles of the held-out mean and maximum TVD that shot noise alone produces.

    To first order a maximum-likelihood estimate is distributed as
    ``theta_hat ~ N(theta, F^+)``, where ``F`` is the expected Fisher information of the
    training data at the true parameters, so its held-out prediction errors are
    ``J_h (theta_hat - theta)``.  This samples that distribution and returns the `quantile`
    of the mean and of the maximum per-circuit TVD over `held_out_circuits`.

    `model` is the data-generating model, and it must have the fit's parameterization, so
    this applies only to in-class scenarios.
    """
    eigenvalues, eigenvectors = _identifiable_eigenpairs(expected_fisher_information(model, dataset))
    covariance_root = eigenvectors / np.sqrt(eigenvalues)
    _, held_out_jacobian, owner = _probabilities_and_jacobian(model, list(held_out_circuits))
    response = held_out_jacobian @ covariance_root
    circuit_starts = np.flatnonzero(np.r_[True, np.diff(owner) != 0])
    rng = np.random.default_rng(seed)
    means, maxima = [], []
    for chunk in np.array_split(np.arange(samples), max(1, samples // 500)):
        errors = np.abs(response @ rng.standard_normal((response.shape[1], len(chunk))))
        tvds = 0.5 * np.add.reduceat(errors, circuit_starts, axis=0)
        means.append(tvds.mean(axis=0))
        maxima.append(tvds.max(axis=0))
    return (float(np.quantile(np.concatenate(means), quantile)),
            float(np.quantile(np.concatenate(maxima), quantile)))


def _mpi_worker_environment(existing_environment=None) -> dict[str, str]:
    """Build worker overrides that import pyGSTi from this active checkout."""
    environment = os.environ if existing_environment is None else existing_environment
    existing_pythonpath = environment.get('PYTHONPATH')
    pythonpath_entries = [str(REPO)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    return {'PYTHONPATH': os.pathsep.join(pythonpath_entries)}


def _run_protocol_persistently(protocol, pdata, artifacts: ValidationArtifacts, mpi_ranks: int):
    """Run the fit through its serial or persistent MPI execution path."""
    started = time.perf_counter()
    if mpi_ranks == 1:
        results = protocol.run(pdata)
        results.write(artifacts.results_dir)
    else:
        results = protocol.run_mpi(
            pdata,
            num_ranks=mpi_ranks,
            persistent_dir=artifacts.results_dir,
            blas_threads_per_rank=1,
            extra_mpi_args=openmpi_extra_args(),
            env=_mpi_worker_environment(),
        )
    return results, time.perf_counter() - started


def run_validation_profile(
        profile: ValidationProfile,
        artifacts: ValidationArtifacts,
        mpi_ranks: int,
) -> dict[str, float | int | str]:
    """Run one finite-shot validation profile and persist its complete evidence."""
    if mpi_ranks < 1:
        raise ValueError('mpi_ranks must be at least one')
    artifacts.root.mkdir(parents=True, exist_ok=True)
    artifacts.manifest_path.write_text(
        json.dumps(dataclasses.asdict(profile), indent=2) + '\n')

    fit_model, datagen_model = build_fit_and_datagen_models(profile)
    training_design, held_out = build_training_design_and_held_out_circuits(profile)
    training_circuits = training_design.all_circuits_needing_data
    dataset = generate_finite_shot_data(profile, datagen_model, training_circuits)
    pdata = ProtocolData(training_design, dataset)
    pdata.write(artifacts.dataset_dir)

    protocol = GateSetTomography(
        fit_model,
        gaugeopt_suite='none',
        name='simultaneous_gst_validation',
        verbosity=1,
    )
    results, elapsed_seconds = _run_protocol_persistently(
        protocol, pdata, artifacts, mpi_ranks)

    estimate = results.estimates[protocol.name]
    fitted_model = estimate.models['final iteration estimate']
    two_delta_logl_value, nsigma, _ = two_delta_logl(
        fitted_model, dataset, dof_calc_method='modeltest')
    tvd_by_circuit = {
        circuit: float(tvd(fitted_model.probabilities(circuit),
                           datagen_model.probabilities(circuit)))
        for circuit in held_out.circuits
    }
    held_out_tvds = list(tvd_by_circuit.values())
    deep_tvds = [tvd_by_circuit[circuit] for circuit in held_out.deepest]
    # References from the same run.  The data-generating model's own 2*deltaLogL is what the
    # true model scores on exactly this dataset, and the target model's held-out TVD (the
    # fit's noiseless starting point) is the prediction error of a fit that learned nothing.
    datagen_two_delta_logl = two_delta_logl(datagen_model, dataset)
    target_tvds = [float(tvd(fit_model.probabilities(circuit), datagen_model.probabilities(circuit)))
                   for circuit in held_out.circuits]
    metrics: dict[str, float | int | str] = {
        'profile': profile.name,
        'scenario': profile.scenario,
        'spectator_term': spectator_term_text(profile.spectator_term),
        'spectator_error': float(profile.spectator_error),
        'two_delta_logl': float(two_delta_logl_value),
        'datagen_two_delta_logl': float(datagen_two_delta_logl),
        'nsigma': float(nsigma),
        'held_out_mean_tvd': float(np.mean(held_out_tvds)),
        'held_out_max_tvd': float(np.max(held_out_tvds)),
        'held_out_deep_mean_tvd': float(np.mean(deep_tvds)),
        'held_out_deep_max_tvd': float(np.max(deep_tvds)),
        'target_held_out_mean_tvd': float(np.mean(target_tvds)),
        'target_held_out_max_tvd': float(np.max(target_tvds)),
        'elapsed_seconds': float(elapsed_seconds),
        'fit_model_params': int(fit_model.num_params),
        'training_circuits': int(len(training_circuits)),
        'held_out_circuits': int(len(held_out.circuits)),
        'held_out_deep_circuits': int(len(held_out.deepest)),
        'shots': int(profile.shots),
        'mpi_ranks': int(mpi_ranks),
    }
    if profile.scenario == 'in_class':
        metrics['identifiable_params'] = identifiable_parameter_count(datagen_model, dataset)
        mean_threshold, max_threshold = shot_noise_held_out_tvd_quantiles(
            datagen_model, dataset, held_out.circuits)
        metrics['shot_noise_held_out_mean_tvd_q999'] = mean_threshold
        metrics['shot_noise_held_out_max_tvd_q999'] = max_threshold
    (artifacts.root / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def create_run_root(artifact_root: pathlib.Path, profile_name: str) -> pathlib.Path:
    """Create an artifact root and record the execution context."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    suffix = 0
    while True:
        collision_suffix = '' if suffix == 0 else f'-{suffix}'
        root = artifact_root / f'{stamp}-{profile_name}{collision_suffix}'
        try:
            root.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
    (root / 'run-manifest.json').write_text(json.dumps({
        'profile': profile_name,
        'started_at_utc': stamp,
        'git_sha': subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
        'python': sys.executable,
    }, indent=2) + '\n')
    return root

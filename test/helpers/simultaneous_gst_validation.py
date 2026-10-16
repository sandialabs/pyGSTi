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
from typing import Literal, Sequence, get_args

import numpy as np

from pygsti.algorithms.fiducialselection import find_fiducials
from pygsti.algorithms.germselection import find_germs
from pygsti.data import simulate_data
from pygsti.models import create_cloud_crosstalk_model, create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.gst import GateSetTomography, StandardGSTDesign
from pygsti.protocols.protocol import ProtocolData
from pygsti.protocols.simultaneous_gst import make_simultaneous_gst_design
from test.helpers.simultaneous_gst_statistics import score_validation_fit


REPO = pathlib.Path(__file__).resolve().parents[2]


# Both term sets hold only H and S elementary error generators, a small part of the space
# once the C and A sectors are counted.  `SOME_HS_TERMS` picks a few of them per gate;
# `ALL_HS_TERMS` holds every one supported on the gate's qubits.
SOME_HS_TERMS = {
    'Gi': (('H', 'Z'), ('S', 'X'), ('S', 'Y'), ('S', 'Z')),
    'Gxpi2': (('H', 'Z'), ('S', 'Z')),
    'Gypi2': (('H', 'X'), ('S', 'Z')),
    'Gcnot': (('H', 'ZZ'), ('S', 'XX'), ('S', 'ZZ')),
    'Gii': (('H', 'ZI'), ('H', 'IZ'), ('S', 'XI'), ('S', 'IX'), ('S', 'ZI'), ('S', 'IZ')),
}


def _all_hs_terms(num_qubits: int) -> tuple[tuple[str, str], ...]:
    pauli_labels = (
        ''.join(label)
        for label in itertools.product('IXYZ', repeat=num_qubits)
        if any(pauli != 'I' for pauli in label)
    )
    labels = tuple(pauli_labels)
    return tuple((term_type, label) for term_type in ('H', 'S') for label in labels)


ALL_HS_TERMS = {
    gate_name: _all_hs_terms(2 if gate_name in ('Gcnot', 'Gii') else 1)
    for gate_name in SOME_HS_TERMS
}

ModelTerms = Literal['somehs', 'allhs']
# Prose spells these "in-class" and "out-of-class".
Scenario = Literal['inclass', 'outclass']
Topology = Literal['line', 'ring']
DEFAULT_MODEL_TERMS: ModelTerms = 'somehs'


def _profile_name(topology: Topology, nqubits: int, model_terms: ModelTerms,
                  scenario: Scenario) -> str:
    # One token per descriptor, so every underscore separates two of them.  The default
    # term set is left out, which makes a canonical profile's name its constant's name.
    name = f'{topology}_{nqubits}q_{scenario}'
    return name if model_terms == DEFAULT_MODEL_TERMS else f'{name}_{model_terms}'


@dataclasses.dataclass(frozen=True)
class ValidationProfile:
    """Configuration for one validation run, and the 1Q and 2Q component models that select its designs."""

    name: str
    nqubits: int
    max_lengths: tuple[int, ...]
    shots: int
    model_terms: ModelTerms
    # Whether the data-generating model lies in the fit's hypothesis class.  Out of class,
    # it adds `spectator_term` to `Gcnot(0, 1)`.
    scenario: Scenario
    seed: int
    spectator_error: float = 0.0
    # The error injected into `Gcnot(0, 1)` for out-of-class scenarios, as a
    # `create_cloud_crosstalk_model` coefficient key.  `('S', 'Z:2')` dephases the
    # spectator.  It is weight-1, so the hypothesis class can absorb part of it, but only as
    # error on qubit 2's own gates in every layer, not just the layers containing the CNOT.
    # `('H', 'ZZ:1,2')` is a weight-2 coherent coupling of the gate's target to the
    # spectator, which no product of local errors reproduces, even to first order.
    spectator_term: tuple[str, ...] = ('S', 'Z:2')
    # The two-qubit gate connectivity: `'ring'` adds the wraparound edge to the line.
    topology: Topology = 'line'

    def __post_init__(self):
        # An unknown scenario would otherwise run as in-class, and an unknown term set would
        # fail only once a model is built.
        for field, allowed in (('model_terms', ModelTerms), ('scenario', Scenario),
                               ('topology', Topology)):
            if getattr(self, field) not in get_args(allowed):
                raise ValueError(f'Unknown {field} {getattr(self, field)!r}; '
                                 f'expected one of {get_args(allowed)}')

    @classmethod
    def inclass(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                seed: int = 20260901, model_terms: ModelTerms = DEFAULT_MODEL_TERMS,
                topology: Topology = 'line') -> 'ValidationProfile':
        """Build a profile whose data-generating model is a seeded point in the hypothesis class."""
        return cls(_profile_name(topology, nqubits, model_terms, 'inclass'), nqubits,
                   tuple(max_lengths), shots, model_terms, 'inclass', seed, topology=topology)

    # S(Z:2) at 0.01 is large enough that the training data reveal it -- the
    # crosstalk-free fit's 2*deltaLogL exceeds the data-generating model's by about 1200
    # at 3Q -- while that fit still removes most of the held-out error.  The earlier
    # default, H(Z:2) at 0.005, left the training likelihood unchanged.
    @classmethod
    def outclass(cls, nqubits: int, max_lengths: tuple[int, ...], shots: int = 1000,
                 seed: int = 20260901, model_terms: ModelTerms = DEFAULT_MODEL_TERMS,
                 topology: Topology = 'line', spectator_error: float = 0.01,
                 spectator_term: tuple[str, ...] = ('S', 'Z:2')) -> 'ValidationProfile':
        """Build a profile whose data-generating model adds `spectator_term` to `Gcnot(0, 1)`."""
        return cls(_profile_name(topology, nqubits, model_terms, 'outclass'), nqubits,
                   tuple(max_lengths), shots, model_terms, 'outclass', seed, spectator_error,
                   tuple(spectator_term), topology)

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


LINE_3Q_INCLASS = ValidationProfile.inclass(
    nqubits=3, max_lengths=(1, 2, 4))
LINE_3Q_OUTCLASS = ValidationProfile.outclass(
    nqubits=3, max_lengths=(1, 2, 4))
LINE_4Q_INCLASS = ValidationProfile.inclass(
    nqubits=4, max_lengths=(1, 2, 4))
LINE_4Q_OUTCLASS = ValidationProfile.outclass(
    nqubits=4, max_lengths=(1, 2, 4))
RING_3Q_INCLASS = ValidationProfile.inclass(
    nqubits=3, max_lengths=(1, 2, 4), topology='ring')
RING_3Q_OUTCLASS = ValidationProfile.outclass(
    nqubits=3, max_lengths=(1, 2, 4), topology='ring')
RING_4Q_INCLASS = ValidationProfile.inclass(
    nqubits=4, max_lengths=(1, 2, 4), topology='ring')
RING_4Q_OUTCLASS = ValidationProfile.outclass(
    nqubits=4, max_lengths=(1, 2, 4), topology='ring')


def _oneq_processor_spec() -> QubitProcessorSpec:
    return QubitProcessorSpec(
        1, gate_names=['Gi', 'Gxpi2', 'Gypi2'], qubit_labels=(0,)
    )


def _processor_spec(nqubits: int, topology: Topology = 'line') -> QubitProcessorSpec:
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


def _terms_by_gate(model_terms: ModelTerms) -> dict:
    if model_terms == 'somehs':
        return SOME_HS_TERMS
    if model_terms == 'allhs':
        return ALL_HS_TERMS
    raise ValueError(f'Unknown model terms {model_terms!r}; expected somehs or allhs')


def _zero_term_dict(model_terms: ModelTerms, gate_names) -> dict:
    terms_by_gate = _terms_by_gate(model_terms)
    return {
        gate_name: {term: 0.0 for term in terms_by_gate[gate_name]}
        for gate_name in gate_names
    }


def _build_model(processor_spec: QubitProcessorSpec,
                 model_terms: ModelTerms, independent_gates: bool):
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
        raise ValueError('Out-of-class profiles require at least three qubits')
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

    Without this the `H+S` silent drop produces an out-of-class run containing no
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
    if profile.scenario == 'outclass':
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
    metrics = score_validation_fit(
        fit_model, datagen_model, fitted_model, dataset, held_out, profile.scenario)
    metrics.update({
        'profile': profile.name,
        'scenario': profile.scenario,
        'spectator_term': spectator_term_text(profile.spectator_term),
        'spectator_error': float(profile.spectator_error),
        'elapsed_seconds': float(elapsed_seconds),
        'fit_model_params': int(fit_model.num_params),
        'training_circuits': int(len(training_circuits)),
        'held_out_circuits': int(len(held_out.circuits)),
        'held_out_deep_circuits': int(len(held_out.deepest)),
        'shots': int(profile.shots),
        'mpi_ranks': int(mpi_ranks),
    })
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

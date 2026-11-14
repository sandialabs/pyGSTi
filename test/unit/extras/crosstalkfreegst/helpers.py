"""Shared fixtures for the crosstalk-free GST tests."""
import numpy as np

from pygsti.models import create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec

ERRORS = {
    'Gi': {('H', 'Z'): 0, ('S', 'X'): 0},
    'Gxpi2': {('H', 'X'): 0, ('S', 'Z'): 0},
    'Gypi2': {('H', 'Y'): 0, ('S', 'X'): 0},
    'Gcnot': {('H', 'ZZ'): 0, ('S', 'XX'): 0, ('H', 'IX'): 0},
    'Gii': {('H', 'ZI'): 0, ('S', 'IZ'): 0},
    'prep': {('H', 'X'): 0, ('S', 'X'): 0},
    'povm': {('S', 'X'): 0, ('H', 'Y'): 0},
}


def line_pspec(n, instrument_names=()):
    qubits = tuple(range(n))
    one = [(q,) for q in qubits]
    edges = [(q, q + 1) for q in range(n - 1)]
    return QubitProcessorSpec(n, gate_names=['Gi', 'Gxpi2', 'Gypi2', 'Gcnot', 'Gii'],
                              nonstd_gate_unitaries={'Gii': np.eye(4)},
                              availability={'Gi': one, 'Gxpi2': one, 'Gypi2': one, 'Gcnot': edges, 'Gii': edges},
                              instrument_names=tuple(instrument_names), qubit_labels=qubits)


def model(n, independent_gates=True, seed=0):
    m = create_crosstalk_free_model(line_pspec(n), lindblad_error_coeffs=ERRORS, lindblad_parameterization='H+S',
                                    independent_gates=independent_gates)
    m.from_vector(np.random.default_rng(seed).uniform(-0.03, 0.05, m.num_params))
    return m


def joint_logl(m, ds, circuits):
    probs = m.sim.bulk_probs(circuits)
    return sum(n * np.log(probs[c][ol]) for c in circuits for ol, n in ds[c].counts.items())

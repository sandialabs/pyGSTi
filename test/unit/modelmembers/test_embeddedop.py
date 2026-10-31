"""
Tests for `EmbeddedOp`, in two groups.

Label-embedding regression tests for `errorgen_coefficient_labels`/`errorgen_coefficients` with
`label_type='local'` (#878). `_embed_labels` once indexed a plain list with the `EmbeddedOp`'s
target qubit labels rather than their positions within `self.state_space`. That raised `TypeError`
for string qubit labels and silently mis-embedded for permuted integer labels. The tests cover a
string-labeled space and a permuted-integer-labeled space, with an unpermuted-integer control.

`densitymx_slow` tests: `OpRepEmbedded`'s action is checked against `EmbeddedOp.to_dense()`, and
a crosstalk-free model on `densitymx_slow` is checked to get embedded reps and to match the
matrix simulator.
"""
import numpy as np
import pytest

from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.baseobjs.statespace import ExplicitStateSpace, QubitSpace
from pygsti.modelmembers.operations import EmbeddedErrorgen, EmbeddedOp, ExpErrorgenOp, LindbladErrorgen


def _h_errorgen_op(pauli, rate=0.15, num_qubits=1):
    """A single-qubit-or-more ExpErrorgenOp with one Hamiltonian-type local coefficient."""
    eg = LindbladErrorgen.from_elementary_errorgens(
        {LEEL('H', (pauli,)): rate}, parameterization='H', state_space=num_qubits)
    return ExpErrorgenOp(eg)


def test_errorgen_coefficient_labels_local_integer_qubit_labels_control():
    """Control case: an unpermuted integer-labeled space, where the bug is a no-op today."""
    ss = QubitSpace(3)
    embedded = EmbeddedOp(ss, [1], _h_errorgen_op('X'))
    labels = embedded.errorgen_coefficient_labels(label_type='local')
    assert labels == (LEEL('H', ('IXI',)),), f'Expected IXI-embedded label, got {labels}'


def test_errorgen_coefficient_labels_local_string_qubit_labels():
    """String qubit labels: `_embed_labels` must resolve each target label to its integer
    position in `state_space.qubit_labels` before indexing, not use the label itself as an index.
    """
    ss = QubitSpace(['q2', 'q16', 'q22'])
    embedded = EmbeddedOp(ss, ['q16'], _h_errorgen_op('X'))
    labels = embedded.errorgen_coefficient_labels(label_type='local')
    assert labels == (LEEL('H', ('IXI',)),), f'Expected IXI-embedded label, got {labels}'


def test_errorgen_coefficient_labels_local_permuted_integer_qubit_labels():
    """Permuted integer qubit labels: a target label equal to a valid index, but not its own
    position, would currently be silently mis-embedded (rather than raising) since `list`
    indexing with an out-of-place-but-still-integer label doesn't error.
    """
    ss = QubitSpace([2, 0, 1])
    embedded = EmbeddedOp(ss, [2], _h_errorgen_op('X'))  # qubit label 2 is at position 0
    labels = embedded.errorgen_coefficient_labels(label_type='local')
    assert labels == (LEEL('H', ('XII',)),), f'Expected XII-embedded label, got {labels}'


def test_errorgen_coefficient_labels_local_two_qubit_target_out_of_order():
    """Two-qubit target whose labels appear out of order relative to `state_space.qubit_labels`,
    to confirm each basis-element-string character is paired with its own target's resolved
    position (not, e.g., accidentally paired via enumeration order).
    """
    ss = QubitSpace(['q2', 'q16', 'q5'])
    eg = LindbladErrorgen.from_elementary_errorgens(
        {LEEL('H', ('XY',)): 0.05}, parameterization='H', state_space=2)
    embedded = EmbeddedOp(ss, ['q5', 'q2'], ExpErrorgenOp(eg))
    labels = embedded.errorgen_coefficient_labels(label_type='local')
    # 'q5' ('X') is at position 2; 'q2' ('Y') is at position 0; 'q16' untouched.
    assert labels == (LEEL('H', ('YIX',)),), f'Expected YIX-embedded label, got {labels}'


def test_errorgen_coefficients_local_string_qubit_labels_end_to_end():
    """The full `errorgen_coefficients(label_type='local')` call chain (as used by
    `pygsti.errorgenpropagation.errorpropagator.ErrorGeneratorPropagator`), not just the
    label-construction helper in isolation.
    """
    ss = QubitSpace(['q2', 'q16', 'q22'])
    embedded = EmbeddedOp(ss, ['q16'], _h_errorgen_op('X', rate=0.15))
    coeffs = embedded.errorgen_coefficients(label_type='local')
    assert set(coeffs.keys()) == {LEEL('H', ('IXI',))}, f'Expected only IXI-embedded label, got {coeffs.keys()}'
    assert np.isclose(coeffs[LEEL('H', ('IXI',))], 0.15), f'Expected rate 0.15, got {coeffs[LEEL("H", ("IXI",))]}'


def test_embeddederrorgen_coefficient_labels_local_string_qubit_labels():
    """`EmbeddedErrorgen` overrides `coefficient_labels` (not `errorgen_coefficient_labels`) but
    shares the same `_embed_labels` helper with `EmbeddedOp`, so it hits the identical bug.
    """
    eg = LindbladErrorgen.from_elementary_errorgens(
        {LEEL('H', ('X',)): 0.15}, parameterization='H', state_space=1)
    ss = QubitSpace(['q2', 'q16', 'q22'])
    embedded = EmbeddedErrorgen(ss, ['q16'], eg)
    labels = embedded.coefficient_labels(label_type='local')
    assert labels == (LEEL('H', ('IXI',)),), f'Expected IXI-embedded label, got {labels}'


def _slow_embedded_rep_and_reference(ss, targets):
    from pygsti.evotypes import Evotype
    from pygsti.modelmembers.operations import StaticArbitraryOp
    rng = np.random.default_rng(1234)
    sub = ss.create_subspace(targets)
    mx = rng.normal(size=(sub.dim, sub.dim))
    op = StaticArbitraryOp(mx, evotype='densitymx_slow', state_space=sub)
    rep = Evotype.cast('densitymx_slow').create_embedded_rep(ss, targets, op._rep)
    ref = EmbeddedOp(ss, targets, op).to_dense('HilbertSchmidt')
    return rep, ref


@pytest.mark.parametrize('ss, targets', [
    (QubitSpace(4), [3, 1]),
    (ExplicitStateSpace([('Q0',), ('Q1', 'Q2')]), ['Q2', 'Q1']),
    (ExplicitStateSpace([('Q0', 'Q1'), ('Q2',)]), ['Q2']),
])
def test_slow_opreprembedded_acton_matches_to_dense(ss, targets):
    from pygsti.evotypes.densitymx_slow.statereps import StateRepDense
    rep, ref = _slow_embedded_rep_and_reference(ss, targets)
    v = np.random.default_rng(5).normal(size=ss.dim)
    state = StateRepDense(v.copy(), ss, None)
    assert np.allclose(rep.acton(state).data, ref @ v)
    assert np.allclose(rep.adjoint_acton(state).data, ref.T @ v)
    assert np.array_equal(state.data, v)  # input untouched


def test_slow_embedded_crosstalk_free_model_uses_embedded_reps_and_matches_matrix_sim():
    from pygsti.circuits import Circuit
    from pygsti.models import modelconstruction as mc
    from pygsti.processors import QubitProcessorSpec
    pspec = QubitProcessorSpec(5, ['Gxpi2', 'Gypi2', 'Gcnot'], geometry='line')
    mdl = mc.create_crosstalk_free_model(pspec, ideal_gate_type='H+s', evotype='densitymx_slow',
                                         simulator='map')
    mdl.from_vector(np.random.default_rng(7).uniform(-0.02, 0.02, mdl.num_params))
    embedded = [op for op in mdl.operation_blks['layers'].values() if isinstance(op, EmbeddedOp)]
    assert len(embedded) > 0
    assert all(op._rep_type == 'embedded' for op in embedded)

    circ = Circuit([('Gxpi2', 0), ('Gcnot', 3, 2), ('Gypi2', 4), ('Gcnot', 0, 1), ('Gxpi2', 3)],
                   line_labels=(0, 1, 2, 3, 4))
    mdl_matrix = mdl.copy()
    mdl_matrix.sim = 'matrix'
    p_map, p_mx = mdl.probabilities(circ), mdl_matrix.probabilities(circ)
    for k, p in p_mx.items():
        assert abs(p_map[k] - p) < 1e-10

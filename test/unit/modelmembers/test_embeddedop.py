"""
Regression tests for `EmbeddedOp.errorgen_coefficient_labels`/`errorgen_coefficients` with
`label_type='local'`.

`_embed_labels`'s local-label branch indexes a plain Python list (`base_label`) directly with
entries of `self.target_labels`, which are the `EmbeddedOp`'s *target qubit labels* rather than
their integer positions within `self.state_space`. This works by coincidence for state spaces
with unpermuted integer qubit labels (where a label equals its own position), but raises
`TypeError` for any state space with string qubit labels, and would silently mis-embed for a
state space with permuted integer labels. These tests exercise both a string-labeled space (where
the bug currently raises) and a permuted-integer-labeled space (where the bug would currently
mis-embed silently), alongside an unpermuted-integer control that already passes today.
"""
import numpy as np

from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.baseobjs.statespace import QubitSpace
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

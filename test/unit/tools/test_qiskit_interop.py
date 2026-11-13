"""
Tests for pygsti.tools._qiskit_interop and the QiskitInteropWarning paths of
Circuit.from_qiskit / the mirror-edesign qiskit entry points.
"""
import sys
from unittest import mock

import pytest

try:
    import qiskit
    HAVE_QISKIT = True
except ImportError:
    HAVE_QISKIT = False

from pygsti.tools._qiskit_interop import TESTED_QISKIT_RANGE, _parse_release, check_qiskit_version
from pygsti.tools.exceptions import MissingDependencyWarning, QiskitInteropWarning

needs_qiskit = pytest.mark.skipif(not HAVE_QISKIT, reason='qiskit is not installed')


def test_parse_release():
    assert _parse_release('2.5.0') == (2, 5, 0)
    assert _parse_release('1.4') == (1, 4)
    assert _parse_release('3.0.0rc1') == (3, 0, 0)
    assert _parse_release('2.1.1.dev0+abc') == (2, 1, 1)


@needs_qiskit
def test_version_out_of_range_warns():
    for bad_version in ('1.3.9', '3.0.0'):
        with mock.patch.object(qiskit, '__version__', bad_version):
            with pytest.warns(QiskitInteropWarning, match='tested against qiskit versions'):
                check_qiskit_version('test_context()')


@needs_qiskit
def test_version_in_range_no_warning():
    import warnings
    minimum, _ = TESTED_QISKIT_RANGE
    in_range_version = '.'.join(map(str, minimum)) + '.0'
    for version in (in_range_version, '2.99.0'):
        with mock.patch.object(qiskit, '__version__', version):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                assert check_qiskit_version('test_context()') is qiskit


def test_missing_qiskit_required_raises():
    with mock.patch.dict(sys.modules, {'qiskit': None}):
        with pytest.raises(RuntimeError, match='Qiskit is required for test_context()'):
            check_qiskit_version('test_context()')


def test_missing_qiskit_optional_warns_and_returns_none():
    with mock.patch.dict(sys.modules, {'qiskit': None}):
        with pytest.warns(MissingDependencyWarning, match='does not appear to be installed'):
            assert check_qiskit_version('test_context()', required=False) is None


@needs_qiskit
class TestFromQiskitLossyWarnings:

    @staticmethod
    def _measure_free_circuit():
        qc = qiskit.QuantumCircuit(qiskit.QuantumRegister(2))
        qc.h(0)
        qc.cx(0, 1)
        return qc

    def test_clean_circuit_round_trip_is_silent(self):
        import warnings
        from pygsti.circuits import Circuit
        ps_circ = Circuit([('Gh', 'Q0'), ('Gcnot', 'Q0', 'Q1')], line_labels=('Q0', 'Q1'))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            # convert_to_qiskit always attaches an (empty) classical register named 'cr';
            # the round trip back through from_qiskit must not warn about it.
            qk_circ = ps_circ.convert_to_qiskit(qubit_conversion='remove-Q')
            assert len(qk_circ.cregs) == 1 and len(qk_circ.clbits) == 0
            Circuit.from_qiskit(qk_circ)

    def test_multiple_qregs_warn(self):
        from pygsti.circuits import Circuit
        qc = qiskit.QuantumCircuit(qiskit.QuantumRegister(1, 'a'), qiskit.QuantumRegister(1, 'b'))
        qc.h(0)
        with pytest.warns(QiskitInteropWarning, match='does not preserve Qiskit qreg structure'):
            Circuit.from_qiskit(qc)

    def test_classical_bits_warn(self):
        from pygsti.circuits import Circuit
        qc = qiskit.QuantumCircuit(qiskit.QuantumRegister(2), qiskit.ClassicalRegister(2))
        qc.h(0)
        with pytest.warns(QiskitInteropWarning, match='discards classical registers'):
            Circuit.from_qiskit(qc)

    def test_measure_warns(self):
        from pygsti.circuits import Circuit
        qc = qiskit.QuantumCircuit(qiskit.QuantumRegister(2), qiskit.ClassicalRegister(2))
        qc.h(0)
        qc.measure(0, 0)
        # both the classical-bits warning and the measure warning fire here
        with pytest.warns(QiskitInteropWarning) as record:
            Circuit.from_qiskit(qc)
        assert any('drops measure instructions' in str(w.message) for w in record)

    def test_lossy_ignore_is_silent(self):
        import warnings
        from pygsti.circuits import Circuit
        qc = qiskit.QuantumCircuit(qiskit.QuantumRegister(1, 'a'), qiskit.QuantumRegister(1, 'b'),
                                   qiskit.ClassicalRegister(2))
        qc.h(0)
        qc.measure(0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Circuit.from_qiskit(qc, lossy='ignore')

    def test_lossy_raise_raises(self):
        from pygsti.circuits import Circuit
        qc = qiskit.QuantumCircuit(qiskit.QuantumRegister(2), qiskit.ClassicalRegister(2))
        qc.h(0)
        with pytest.raises(ValueError, match='discards classical registers'):
            Circuit.from_qiskit(qc, lossy='raise')

    def test_lossy_invalid_value_rejected(self):
        from pygsti.circuits import Circuit
        with pytest.raises(AssertionError, match="'lossy' must be"):
            Circuit.from_qiskit(self._measure_free_circuit(), lossy='bogus')


@needs_qiskit
def test_fullstack_mirror_edesign_ignored_args_warn():
    from qiskit.providers.fake_provider import GenericBackendV2
    from pygsti.protocols.mirror_edesign import qiskit_circuits_to_fullstack_mirror_edesign

    backend = GenericBackendV2(num_qubits=2, basis_gates=['u', 'cz'], seed=0)
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    mirroring_kwargs = {'num_mcs_per_circ': 1, 'num_ref_per_qubit_subset': 1,
                        'rand_state': __import__('numpy').random.RandomState(0)}

    with pytest.warns(QiskitInteropWarning, match="'basis_gates' is ignored"):
        qiskit_circuits_to_fullstack_mirror_edesign(
            [qc], qk_backend=backend, basis_gates=['u', 'cz'],
            transpiler_kwargs_dict={'seed_transpiler': 0},
            mirroring_kwargs_dict=mirroring_kwargs)

    with pytest.warns(QiskitInteropWarning, match="'coupling_map' is ignored"):
        qiskit_circuits_to_fullstack_mirror_edesign(
            [qc], qk_backend=backend, coupling_map=backend.coupling_map,
            transpiler_kwargs_dict={'seed_transpiler': 0},
            mirroring_kwargs_dict=mirroring_kwargs)

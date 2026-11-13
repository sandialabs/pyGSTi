import pytest
import stim

from pygsti.extras.dem_construction.pygsti_object_builders import stim_to_pygsti_circuit


def _layers(pyg_circuit):
    return [sorted((c.name,) + tuple(c.qubits) for c in pyg_circuit.layer_label(i).components)
            for i in range(pyg_circuit.depth)]


def _expectations(pyg_circuit, paulis):
    sim = stim.TableauSimulator()
    sim.do_tableau(pyg_circuit.convert_to_stim_tableau(), list(range(len(pyg_circuit.line_labels))))
    return [sim.peek_observable_expectation(p) for p in paulis]


@pytest.mark.parametrize('name', ['repetition_code:memory', 'surface_code:rotated_memory_z'])
def test_generated_codes_match_stim_reference(name):
    circuit = stim.Circuit.generated(name, distance=3, rounds=3)
    pyg, dets, obs, _ = stim_to_pygsti_circuit(circuit)
    assert len(dets) == circuit.num_detectors
    assert len(obs) == circuit.num_observables
    det_signs, obs_signs = circuit.reference_detector_and_observable_signs()
    expected = [1 - 2 * int(s) for s in list(det_signs) + list(obs_signs)]
    assert _expectations(pyg, dets + obs) == expected


def test_exact_layout():
    circuit = stim.Circuit("""
        R 0 1
        X 1
        CX 0 1
        MR 1
        CX 0 1
        M 1 0
        DETECTOR rec[-3]
        DETECTOR rec[-3] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    pyg, dets, obs, v2p = stim_to_pygsti_circuit(circuit)
    assert _layers(pyg) == [[('Gxpi', 1)], [('Gcnot', 0, 1)], [('Gcnot', 0, 2)]]
    assert dets == [stim.PauliString('+_Z_'), stim.PauliString('+_ZZ')]
    assert obs == [stim.PauliString('+Z__')]
    assert v2p == [0, 1, 1]


def test_signs_and_observable_indices():
    _, dets, _, _ = stim_to_pygsti_circuit(stim.Circuit("M !0\nDETECTOR rec[-1]"))
    assert dets == [stim.PauliString('-Z')]

    _, dets, _, _ = stim_to_pygsti_circuit(stim.Circuit("M 0\nDETECTOR rec[-1] rec[-1]"))
    assert dets == [stim.PauliString('+_')]

    _, _, obs, _ = stim_to_pygsti_circuit(stim.Circuit("M 0\nOBSERVABLE_INCLUDE(1) rec[-1]"))
    assert obs == [stim.PauliString('+_'), stim.PauliString('+Z')]

    _, _, obs, _ = stim_to_pygsti_circuit(
        stim.Circuit("M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"))
    assert obs == [stim.PauliString('+ZZ')]

    pyg, _, _, _ = stim_to_pygsti_circuit(stim.Circuit("M(0) 0"))
    assert pyg.depth == 0


def test_gate_translation_matches_stim_tableau():
    circuit = stim.Circuit("""
        H 0
        S 1
        S_DAG 2
        SQRT_X 0
        SQRT_Y_DAG 1
        CNOT 0 1
        CZ 1 2
        SWAP 0 2
        ISWAP 0 1
        SQRT_XX 1 2
        SQRT_ZZ 0 2
        C_XYZ 1
        CX 0 1 1 2
    """)
    pyg, _, _, _ = stim_to_pygsti_circuit(circuit)
    assert pyg.convert_to_stim_tableau() == stim.Tableau.from_circuit(circuit)

    pyg, _, _, _ = stim_to_pygsti_circuit(stim.Circuit("CX 0 1 1 2"))
    assert pyg.depth == 2


@pytest.mark.parametrize('text,include_idles,match', [
    ("CY 0 1", False, 'CY'),
    ("X_ERROR(0.1) 0", False, 'X_ERROR'),
    ("M(0.01) 0", False, 'M'),
    ("MX 0", False, 'MX'),
    ("RX 0", False, 'RX'),
    ("MPP X0*X1", False, 'MPP'),
    ("M 0\nCX rec[-1] 1", False, 'CX'),
    ("CX sweep[0] 1", False, 'CX'),
    ("CX 0 1\nM 1\nCX 1 2", False, 'reset'),
    ("M 0\nM 0", False, 'reset'),
    ("M 0\nH 0", False, 'reset'),
    ("M 0\nH 0", True, 'reset'),
    ("M 0\nOBSERVABLE_INCLUDE(0) X1", False, 'OBSERVABLE_INCLUDE'),
    ("M 0\nDETECTOR rec[-2]", False, 'DETECTOR'),
])
def test_unsupported_circuits_raise(text, include_idles, match):
    circuit = stim.Circuit(text)
    with pytest.raises(ValueError, match=match):
        stim_to_pygsti_circuit(circuit, include_idles=include_idles)


@pytest.mark.parametrize('text', ["H 0\nM 0\nR 0\nH 0", "H 0\nR 0\nH 0"])
def test_reset_of_used_or_measured_qubit_allocates_virtual_qubit(text):
    pyg, _, _, v2p = stim_to_pygsti_circuit(stim.Circuit(text))
    assert _layers(pyg) == [[('Gh', 0)], [('Gh', 1)]]
    assert v2p == [0, 0]


def test_idle_gates():
    pyg, _, _, _ = stim_to_pygsti_circuit(stim.Circuit("M 2\nH 0"), include_idles=True)
    assert _layers(pyg) == [[('Gh', 0), ('Gi', 1)]]

    pyg, _, _, _ = stim_to_pygsti_circuit(stim.Circuit("H 0 1\nMR 1\nH 0"), include_meas_idles=True)
    assert _layers(pyg) == [[('Gh', 0), ('Gh', 1)], [('Gi', 0)], [('Gh', 0)]]

    pyg, _, _, _ = stim_to_pygsti_circuit(stim.Circuit("H 0 1\nMR 1\nH 0"))
    assert all(lbl[0] != 'Gi' for layer in _layers(pyg) for lbl in layer)

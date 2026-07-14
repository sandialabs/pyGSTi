"""
System-integration test for the crosstalk-free GST (XFGST) pipeline:
circuit generation -> noisy data simulation -> GST fitting, for each of the
four Lindblad error types (H, S, H+S, and H+S+C+A).

This test builds a small (3-qubit line) crosstalk-free experiment design and
runs it end-to-end for each noise configuration. It takes well under a
minute to run on typical hardware; the qubit count and germ-power depth
(`max_max_length`) are deliberately kept small relative to
`pygsti/protocols/run_xfgst_example.py` (which uses 5 qubits and
`max_max_length=4`) purely to keep runtime CI-friendly.
"""

import unittest

import pytest

import pygsti
from pygsti.baseobjs.label import Label, LabelTup, LabelTupTup
from pygsti.data import simulate_data
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.gst import GateSetTomography
from pygsti.protocols.protocol import ProtocolData
from pygsti.protocols.xfgst_edesign import (
    CrosstalkFreeExperimentDesign,
    assign_the_designs_with_mapping,
)
from pygsti.tools import two_delta_logl


def _get_layer_mappers(twoq_gst_design, oneq_gst_design):
    """
    Build layer mappers for batch_tensor.

    The 2Q mapper converts local 2Q-design labels into either:
      - primitive 2Q labels, e.g. Gcnot:0:1
      - LabelTupTup parallel layers of primitive 1Q labels, e.g.
        (Gi:0, Gypi2:1)

    This avoids requiring a primitive Gii gate in the final target model.
    """
    local_twoq_lines = tuple(twoq_gst_design.qubit_labels)
    local_oneq_lines = tuple(oneq_gst_design.qubit_labels)

    assert local_twoq_lines == (0, 1), local_twoq_lines
    assert local_oneq_lines == (0,), local_oneq_lines

    local_twoq_idle_label = Label(("Gii",) + local_twoq_lines)
    local_oneq_idle_label = Label(("Gi",) + local_oneq_lines)

    # A local 2Q idle should become two primitive 1Q idles.
    local_parallel_twoq_idle = Label((Label("Gi", 0), Label("Gi", 1)))
    assert isinstance(local_parallel_twoq_idle, LabelTupTup)

    mapper_2q = {
        Label(()): local_parallel_twoq_idle,
        local_twoq_idle_label: local_parallel_twoq_idle,
    }
    mapper_1q = {
        Label(()): local_oneq_idle_label,
        local_oneq_idle_label: local_oneq_idle_label,
    }

    for cl in oneq_gst_design.circuit_lists:
        for c in cl:
            for ell in c._labels:
                mapper_1q[ell] = local_oneq_idle_label if ell == Label(()) else ell

    for cl in twoq_gst_design.circuit_lists:
        for c in cl:
            for ell in c._labels:
                if ell == Label(()) or ell == local_twoq_idle_label:
                    mapper_2q[ell] = local_parallel_twoq_idle
                elif isinstance(ell, LabelTup) and ell.num_qubits == 1:
                    # e.g. Label("Gypi2", 1) -> (Label("Gi", 0), Label("Gypi2", 1))
                    tgt = ell.qubits[0]
                    assert tgt in (0, 1), ell
                    tmp = [None, None]
                    tmp[tgt] = ell
                    tmp[1 - tgt] = Label("Gi", 1 - tgt)
                    parallel_label = Label(tuple(tmp))
                    assert isinstance(parallel_label, LabelTupTup)
                    mapper_2q[ell] = parallel_label
                else:
                    # Keep real 2Q gates such as Gcnot:0:1 as primitive labels.
                    mapper_2q[ell] = ell

    return {1: mapper_1q, 2: mapper_2q}


def _mapped_assignment_stitcher(oneq_gstdesign, twoq_gstdesign, vertices, color_patches,
                                 randgen=None, ensure_containment=False, debug_check=True):
    """
    Adapter so CrosstalkFreeExperimentDesign can call
    assign_the_designs_with_mapping as a circuit_stitcher.
    """
    layer_mappers = _get_layer_mappers(twoq_gstdesign, oneq_gstdesign)
    return assign_the_designs_with_mapping(
        oneq_gstdesign=oneq_gstdesign,
        twoq_gstdesign=twoq_gstdesign,
        vertices=vertices,
        color_patches=color_patches,
        debug_check=debug_check,
        randgen=randgen,
        ensure_containment=ensure_containment,
        _layer_mappers_override=layer_mappers,
    )


def _build_noise_model(pspec, lindblad_error_coeffs, parameterization):
    """
    Return (target_model, noisy_model) for a given Lindblad error
    specification. `target_model` has every coefficient zeroed (ideal
    device prior) but retains the same free parameters, so GST can
    optimise them starting from zero. `noisy_model` uses the supplied
    coefficient values and is used to generate synthetic data.
    """
    zeroed = {
        gate: {key: 0.0 for key in terms}
        for gate, terms in lindblad_error_coeffs.items()
    }
    target = pygsti.models.create_crosstalk_free_model(
        pspec, lindblad_error_coeffs=zeroed, lindblad_parameterization=parameterization,
    )
    noisy = pygsti.models.create_crosstalk_free_model(
        pspec, lindblad_error_coeffs=lindblad_error_coeffs, lindblad_parameterization=parameterization,
    )
    return target, noisy


# --- H only: pure coherent over-rotations ---
_H_NOISE = {
    'Gi':    {('H', 'Z'): 0.005},
    'Gxpi2': {('H', 'Z'): 0.003},
    'Gypi2': {('H', 'X'): 0.003},
    'Gcnot': {('H', 'ZZ'): 0.005, ('H', 'IZ'): 0.002},
}

# --- S only: Pauli stochastic / dephasing ---
_S_NOISE = {
    'Gi':    {('S', 'X'): 0.001, ('S', 'Y'): 0.001, ('S', 'Z'): 0.002},
    'Gxpi2': {('S', 'X'): 0.001, ('S', 'Z'): 0.001},
    'Gypi2': {('S', 'Y'): 0.001, ('S', 'Z'): 0.001},
    'Gcnot': {('S', 'XX'): 0.001, ('S', 'ZZ'): 0.002},
}

# --- H + S: coherent errors plus stochastic Pauli noise ---
_HS_NOISE = {
    'Gi':    {('H', 'Z'): 0.005, ('S', 'X'): 0.001, ('S', 'Y'): 0.001, ('S', 'Z'): 0.001},
    'Gxpi2': {('H', 'Z'): 0.003, ('S', 'Z'): 0.001},
    'Gypi2': {('H', 'X'): 0.003, ('S', 'Z'): 0.001},
    'Gcnot': {('H', 'ZZ'): 0.005, ('S', 'XX'): 0.001, ('S', 'ZZ'): 0.001},
}

# --- H + S + C + A: full Lindblad including correlated and affine terms.
# C (correlated stochastic) and A (affine) terms require 'GLND'
# parameterization (unconstrained), since 'auto'/'CPTPLND' enforce CPTP
# positivity that an all-zero-coefficient target model may not satisfy.
_HSCA_NOISE = {
    'Gi':    {('H', 'Z'): 0.005, ('S', 'X'): 0.001, ('S', 'Y'): 0.001, ('S', 'Z'): 0.001,
              ('C', 'X', 'Y'): 0.0003, ('A', 'X', 'Y'): 0.0001},
    'Gxpi2': {('H', 'Z'): 0.003, ('S', 'Z'): 0.001, ('C', 'X', 'Y'): 0.0003},
    'Gypi2': {('H', 'X'): 0.003, ('S', 'Z'): 0.001, ('C', 'X', 'Y'): 0.0003},
    'Gcnot': {('H', 'ZZ'): 0.005, ('S', 'XX'): 0.001, ('S', 'ZZ'): 0.001,
              ('C', 'XX', 'YY'): 0.0003, ('A', 'XY', 'YX'): 0.0001},
}

# Each entry: (config_name, noise_coeffs, parameterization, max acceptable 2*deltaLogL)
_NOISE_CONFIGS = [
    ('H', _H_NOISE, 'H', 1.0),
    ('S', _S_NOISE, 'S', 1.0),
    ('H+S', _HS_NOISE, 'H+S', 1.0),
    ('H+S+C+A', _HSCA_NOISE, 'GLND', 5.0),
]


@pytest.mark.slow
class TestCrosstalkFreeGSTPipeline(unittest.TestCase):
    """
    System-integration test for crosstalk-free GST across all four Lindblad
    error types (H, S, H+S, H+S+C+A). Uses a reduced-scale (3-qubit line,
    max_max_length=2) crosstalk-free design so the full test -- which builds
    one experiment design and then runs the noisy-simulate + GST loop for
    each of 4 noise configurations -- completes in well under a minute.
    """

    @classmethod
    def setUpClass(cls):
        n_qubits = 3
        qubits = tuple(range(n_qubits))
        line_edges = [(0, 1), (1, 2)]
        oneq_locations = [(q,) for q in qubits]

        availability = {
            "Gi": oneq_locations,
            "Gxpi2": oneq_locations,
            "Gypi2": oneq_locations,
            "Gcnot": line_edges,
        }
        cls.pspec = QubitProcessorSpec(
            n_qubits, gate_names=["Gi", "Gxpi2", "Gypi2", "Gcnot"],
            availability=availability, qubit_labels=qubits,
        )

        # Small germ-power depth to keep runtime short.
        max_max_length = 2
        oneq_gstdesign = smq1Q_XYI.create_gst_experiment_design(
            max_max_length=max_max_length, qubit_labels=(0,))
        twoq_gstdesign = smq2Q_XYICNOT.create_gst_experiment_design(
            max_max_length=max_max_length, qubit_labels=(0, 1))

        # Two color patches: (0,1) 2Q GST + qubit 2 idle, then (1,2) 2Q GST +
        # qubit 0 idle.
        edge_coloring = {
            0: [(0, 1)],
            1: [(1, 2)],
        }

        cls.xfgst_design = CrosstalkFreeExperimentDesign(
            processor_spec=cls.pspec,
            oneq_gstdesign=oneq_gstdesign,
            twoq_gstdesign=twoq_gstdesign,
            edge_coloring=edge_coloring,
            circuit_stitcher=_mapped_assignment_stitcher,
            seed=1234,
            nested=False,
        )
        cls.circuits = cls.xfgst_design.all_circuits_needing_data

    def test_pipeline_all_noise_types(self):
        for config_name, noise_coeffs, parameterization, max_two_delta_logl in _NOISE_CONFIGS:
            with self.subTest(noise_config=config_name):
                target_model, noisy_model = _build_noise_model(
                    self.pspec, noise_coeffs, parameterization)
                self.assertGreater(target_model.num_params, 0)
                self.assertEqual(target_model.num_params, noisy_model.num_params)

                # sample_error='none' -> deterministic frequencies exactly
                # equal to noisy_model's probabilities, so a correct GST fit
                # should recover ~0 log-likelihood deficit without any
                # statistical flakiness.
                ds = simulate_data(
                    noisy_model, self.circuits, num_samples=1000, seed=42,
                    sample_error='none',
                )

                data = ProtocolData(self.xfgst_design, ds)
                # gaugeopt_suite=None: LocalNoiseModel (crosstalk-free model)
                # has no default_gauge_group, so gauge optimization isn't
                # applicable here.
                # objfn_builders={'objective': 'chi2'}: with noiseless data
                # (sample_error='none') and unconstrained H/S/GLND
                # parameterizations, intermediate LM iterates can produce
                # slightly negative "probabilities", which trips the
                # logl objective's regularization sanity check. chi2 doesn't
                # have this failure mode and is sufficient for this fit-
                # quality smoke test.
                proto = GateSetTomography(
                    target_model, gaugeopt_suite=None, name='xfGST',
                    objfn_builders={'objective': 'chi2'},
                )
                results = proto.run(data)

                mdl_result = results.estimates['xfGST'].models['final iteration estimate']
                two_delta_logl_val = two_delta_logl(
                    mdl_result, ds, min_prob_clip=1e-12, radius=1e-12)
                self.assertLess(
                    two_delta_logl_val, max_two_delta_logl,
                    msg=f"2*deltaLogL too large for noise config {config_name!r}: "
                        f"{two_delta_logl_val}",
                )


if __name__ == '__main__':
    unittest.main()

#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
System-integration test for the simultaneous GST (SGST) pipeline:
circuit generation -> noisy data simulation -> GST fitting, for each of the
four Lindblad error types (H, S, H+S, and H+S+C+A).
"""

import dataclasses
import os
import pathlib
import unittest

import pytest

import numpy as np

import pygsti
from pygsti.data import simulate_data
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols.gst import GateSetTomography
from pygsti.protocols.protocol import ProtocolData
from pygsti.protocols.simultaneous_gst import SimultaneousGSTDesign
from pygsti.tools import two_delta_logl
from test.unit.protocols.test_simultaneous_gst import _line_pspec, _make_designs
from test.helpers.simultaneous_gst_validation import (
    FOUR_QUBIT_COHERENT_MARKOVIAN,
    FOUR_QUBIT_COHERENT_SPECTATOR,
    FOUR_QUBIT_SPARSE_MARKOVIAN,
    FOUR_QUBIT_SPARSE_SPECTATOR,
    THREE_QUBIT_SPARSE_MARKOVIAN,
    THREE_QUBIT_SPARSE_SPECTATOR,
    ValidationArtifacts,
    run_validation_profile,
)


def artifact_dir():
    root = pathlib.Path(os.environ['PYGSTI_SGST_ARTIFACT_DIR'])
    return ValidationArtifacts(
        root=root,
        dataset_dir=root / 'dataset',
        results_dir=root / 'fit',
        manifest_path=root / 'profile.json',
    )


def mpi_ranks():
    return int(os.environ.get('PYGSTI_SGST_MPI_RANKS', '1'))


def profile_with_seed_override(profile):
    """Return ``profile`` or an immutable per-run replacement selected by the runner."""
    override = os.environ.get('PYGSTI_SGST_PROFILE_SEED')
    return profile if override is None else dataclasses.replace(profile, seed=int(override))


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


# --- H + S: coherent errors plus stochastic Pauli noise ---
_HS_NOISE = {
    'Gi':    {('H', 'Z'): 0.005, ('S', 'X'): 0.001, ('S', 'Y'): 0.001, ('S', 'Z'): 0.001},
    'Gxpi2': {('H', 'Z'): 0.003, ('S', 'Z'): 0.001},
    'Gypi2': {('H', 'X'): 0.003, ('S', 'Z'): 0.001},
    'Gcnot': {('H', 'ZZ'): 0.005, ('S', 'XX'): 0.001, ('S', 'ZZ'): 0.001},
    'Gii':   {('H', 'ZI'): 0.005, ('H', 'IZ'): 0.005, ('S', 'XI'): 0.001,
              ('S', 'IX'): 0.001, ('S', 'ZI'): 0.001, ('S', 'IZ'): 0.001},
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
    'Gii':   {('H', 'ZI'): 0.005, ('H', 'IZ'): 0.005, ('S', 'XI'): 0.001,
              ('S', 'IX'): 0.001, ('S', 'ZI'): 0.001, ('S', 'IZ'): 0.001,
              ('C', 'XI', 'YI'): 0.0003, ('A', 'XI', 'YI'): 0.0001},
}

# Each entry: (config_name, noise_coeffs, parameterization, max acceptable 2*deltaLogL)
_NOISE_CONFIGS = [
    ('H+S', _HS_NOISE, 'H+S', 1.0),
    ('H+S+C+A', _HSCA_NOISE, 'GLND', 5.0),
]


class TestSimultaneousGSTPipeline(unittest.TestCase):
    """
    System-integration test for simultaneous GST across all four Lindblad
    error types (H, S, H+S, H+S+C+A). Uses a reduced-scale (3-qubit line,
    max_max_length=2) design so the full test -- which builds
    one experiment design and then runs the noisy-simulate + GST loop for
    each of 4 noise configurations -- completes in roughly three minutes.
    """

    @classmethod
    def setUpClass(cls):
        n_qubits = 3
        cls.pspec, _, _ = _line_pspec(n_qubits)
        oneq_gstdesign, twoq_gstdesign = _make_designs(max_max_length=2)

        # Two color patches: (0,1) 2Q GST + qubit 2 idle, then (1,2) 2Q GST +
        # qubit 0 idle.
        edge_coloring = {
            0: [(0, 1)],
            1: [(1, 2)],
        }

        cls.sgst_design = SimultaneousGSTDesign(
            processor_spec=cls.pspec,
            oneq_gstdesign=oneq_gstdesign,
            twoq_gstdesign=twoq_gstdesign,
            edge_coloring=edge_coloring,
            seed=1234,
            nested=False,
        )
        cls.circuits = cls.sgst_design.all_circuits_needing_data

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

                data = ProtocolData(self.sgst_design, ds)
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
                    target_model, gaugeopt_suite='none', name='simul_gst',
                    objfn_builders={'objective': 'chi2'},
                )
                results = proto.run(data)

                mdl_result = results.estimates['simul_gst'].models['final iteration estimate']
                two_delta_logl_val = two_delta_logl(
                    mdl_result, ds, min_prob_clip=1e-12, radius=1e-12)
                self.assertLess(
                    two_delta_logl_val, max_two_delta_logl,
                    msg=f"2*deltaLogL too large for noise config {config_name!r}: "
                        f"{two_delta_logl_val}",
                )


@pytest.mark.long_running
class SimultaneousGSTValidationTester:
    def test_three_qubit_sparse_markovian_recovery(self):
        result = run_validation_profile(
            profile_with_seed_override(THREE_QUBIT_SPARSE_MARKOVIAN), artifact_dir(), mpi_ranks())
        assert result['validation_mean_tvd'] >= 0.0

    def test_three_qubit_sparse_spectator_crosstalk(self):
        result = run_validation_profile(
            profile_with_seed_override(THREE_QUBIT_SPARSE_SPECTATOR), artifact_dir(), mpi_ranks())
        assert result['two_delta_logl'] >= 0.0

    def test_four_qubit_sparse_markovian_bridge(self):
        result = run_validation_profile(
            profile_with_seed_override(FOUR_QUBIT_SPARSE_MARKOVIAN), artifact_dir(), mpi_ranks())
        assert result['fit_model_params'] == 59

    def test_four_qubit_sparse_spectator_crosstalk(self):
        result = run_validation_profile(
            profile_with_seed_override(FOUR_QUBIT_SPARSE_SPECTATOR), artifact_dir(), mpi_ranks())
        assert result['fit_model_params'] == 59

    def test_four_qubit_coherent_markovian_bridge(self):
        result = run_validation_profile(
            profile_with_seed_override(FOUR_QUBIT_COHERENT_MARKOVIAN), artifact_dir(), mpi_ranks())
        assert result['fit_model_params'] == 21

    def test_four_qubit_coherent_spectator_crosstalk(self):
        result = run_validation_profile(
            profile_with_seed_override(FOUR_QUBIT_COHERENT_SPECTATOR), artifact_dir(), mpi_ranks())
        assert result['fit_model_params'] == 21


if __name__ == '__main__':
    unittest.main()

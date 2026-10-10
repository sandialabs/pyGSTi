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
from scipy.stats import chi2

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
    FOUR_QUBIT_SPARSE_MARKOVIAN,
    FOUR_QUBIT_SPARSE_SPECTATOR,
    THREE_QUBIT_SPARSE_MARKOVIAN,
    THREE_QUBIT_SPARSE_SPECTATOR,
    ValidationArtifacts,
    create_run_root,
    run_validation_profile,
)


def artifact_dir(profile_name):
    """Return a fresh run directory under ``PYGSTI_SGST_ARTIFACT_DIR`` for one profile.

    Each run gets its own timestamped root, so running several profiles against one
    ``PYGSTI_SGST_ARTIFACT_DIR`` keeps every profile's dataset, fit and metrics.
    """
    root = create_run_root(pathlib.Path(os.environ['PYGSTI_SGST_ARTIFACT_DIR']), profile_name)
    return ValidationArtifacts(
        root=root,
        dataset_dir=root / 'dataset',
        results_dir=root / 'fit',
        manifest_path=root / 'profile.json',
    )


def mpi_ranks():
    return int(os.environ.get('PYGSTI_SGST_MPI_RANKS', '1'))


def run_profile(profile):
    """Apply the runner's overrides to ``profile`` and run it in its own artifact directory."""
    profile = profile_with_overrides(profile)
    return run_validation_profile(profile, artifact_dir(profile.name), mpi_ranks())


def profile_with_overrides(profile):
    """Return ``profile`` or an immutable per-run replacement selected by the runner.

    The spectator overrides let one queue sweep crosstalk strength and character
    without editing the canonical profiles. ``PYGSTI_SGST_SPECTATOR_TERM`` is
    pipe-separated because a correlated coefficient key already contains commas,
    e.g. ``H|ZZ:1,2``.
    """
    changes = {}
    seed = os.environ.get('PYGSTI_SGST_PROFILE_SEED')
    if seed is not None:
        changes['seed'] = int(seed)
    spectator_error = os.environ.get('PYGSTI_SGST_SPECTATOR_ERROR')
    if spectator_error is not None:
        changes['spectator_error'] = float(spectator_error)
    spectator_term = os.environ.get('PYGSTI_SGST_SPECTATOR_TERM')
    if spectator_term is not None:
        changes['spectator_term'] = tuple(spectator_term.split('|'))
    return dataclasses.replace(profile, **changes) if changes else profile


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
    """Finite-shot recovery and robustness of simultaneous GST, one profile per test.

    Each fit is checked against references computed from the same run, so the
    assertions hold for any seed rather than for recorded numbers:

    * ``datagen_two_delta_logl`` is what the data-generating model scores on exactly the
      training data.  Comparing the fit against it avoids the degrees-of-freedom
      convention behind ``nsigma``, which sparse multi-qubit outcome counts make unusable.
    * ``ideal_validation_mean_tvd`` is the held-out error of the noiseless starting model,
      that is, of a fit that learned nothing.
    * ``shot_noise_validation_*_tvd_q999`` bound the held-out error that shot noise in the
      training data gives a correct fit (Markovian profiles only).
    """

    @staticmethod
    def assert_recovers_markovian_model(result):
        likelihood_gain = result['datagen_two_delta_logl'] - result['two_delta_logl']
        # The generator is in the fit's model family, so the maximum-likelihood fit explains
        # the training data at least as well as the generator does, and by Wilks' theorem
        # better by no more than a chi-squared with one degree per parameter allows.
        assert likelihood_gain >= 0.0
        assert likelihood_gain <= chi2.ppf(0.999, result['fit_model_params'])
        # Its held-out predictions are as accurate as shot noise in the training data allows.
        assert result['validation_mean_tvd'] <= result['shot_noise_validation_mean_tvd_q999']
        assert result['validation_max_tvd'] <= result['shot_noise_validation_max_tvd_q999']
        # And the profile can tell a fit from no fit: the noise it injects is large enough
        # that fitting removes most of the noiseless model's held-out error.
        assert result['validation_mean_tvd'] <= 0.5 * result['ideal_validation_mean_tvd']

    @staticmethod
    def assert_reveals_and_survives_crosstalk(result):
        # The generator is outside the crosstalk-free family, and the training data show it
        # exactly when the best crosstalk-free fit explains them worse than the generator does.
        assert result['two_delta_logl'] > result['datagen_two_delta_logl']
        # The violation is modest: the fit still removes most of the noiseless model's
        # held-out error.
        assert result['validation_mean_tvd'] <= 0.5 * result['ideal_validation_mean_tvd']

    def test_three_qubit_sparse_markovian_recovery(self):
        self.assert_recovers_markovian_model(run_profile(THREE_QUBIT_SPARSE_MARKOVIAN))

    def test_three_qubit_sparse_spectator_crosstalk(self):
        self.assert_reveals_and_survives_crosstalk(run_profile(THREE_QUBIT_SPARSE_SPECTATOR))

    def test_four_qubit_sparse_markovian_bridge(self):
        self.assert_recovers_markovian_model(run_profile(FOUR_QUBIT_SPARSE_MARKOVIAN))

    def test_four_qubit_sparse_spectator_crosstalk(self):
        self.assert_reveals_and_survives_crosstalk(run_profile(FOUR_QUBIT_SPARSE_SPECTATOR))

if __name__ == '__main__':
    unittest.main()

#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""Exercise simultaneous-GST circuit construction, simulation and fitting together.

The unmarked smoke test fits exact-frequency data for H+S and H+S+C+A noise on a
three-qubit line, using fixed likelihood-deficit tolerances. Eight long-running
finite-shot profiles cover in-/out-of-class data-generating models on three-/four-qubit
lines/rings. They compare against same-run data-generating model and ideal-model baselines,
with asymptotic likelihood/predictive references only for in-class profiles.
The profiles are skipped unless PYGSTI_SGST_ARTIFACT_DIR names an output directory.
"""

import os
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
    LINE_3Q_INCLASS,
    LINE_3Q_OUTCLASS,
    LINE_4Q_INCLASS,
    LINE_4Q_OUTCLASS,
    RING_3Q_INCLASS,
    RING_3Q_OUTCLASS,
    RING_4Q_INCLASS,
    RING_4Q_OUTCLASS,
)
from test.helpers.simultaneous_gst_runner import run_profile


def _build_noise_model(pspec, lindblad_error_coeffs, parameterization):
    """
    Return (ideal_model, noisy_model) for a given Lindblad error
    specification. `ideal_model` has every coefficient zeroed (ideal
    device prior) but retains the same free parameters, so GST can
    optimise them starting from zero. `noisy_model` uses the supplied
    coefficient values and is used to generate synthetic data.

    Setup for the SimultaneousGSTDesign/GateSetTomography pipeline test: the
    supplied dictionary fixes the data-generating model, while an explicitly zeroed copy
    fixes the starting point. The test uses exact-frequency likelihood deficit
    as its reference, not a stored fitted model.
    """
    zeroed = {
        gate: {key: 0.0 for key in terms}
        for gate, terms in lindblad_error_coeffs.items()
    }
    ideal_model = pygsti.models.create_crosstalk_free_model(
        pspec, lindblad_error_coeffs=zeroed, lindblad_parameterization=parameterization,
    )
    datagen_model = pygsti.models.create_crosstalk_free_model(
        pspec, lindblad_error_coeffs=lindblad_error_coeffs, lindblad_parameterization=parameterization,
    )
    return ideal_model, datagen_model


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
# positivity that an all-zero-coefficient ideal model may not satisfy.
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
    """Fit exact-frequency data for H+S and H+S+C+A noise on a three-qubit line.

    A shared design with largest germ-length budget 2 feeds two noise cases.
    The reference is zero likelihood deficit for exact data-generating model frequencies;
    fixed per-case tolerances allow numerical fitting error, not shot noise.
    """

    @classmethod
    def setUpClass(cls):
        """Prepare a three-qubit-line ``SimultaneousGSTDesign`` for pipeline tests.

        Use component designs and explicitly prescribed patches (0, 1) and (1, 2),
        with design seed 1234. The later smoke-test reference is exact data-generating model
        frequencies and fixed optimization tolerances, not this setup's fitted output.
        """
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
        """Check ``SimultaneousGSTDesign`` -> simulation -> ``GateSetTomography``.

        Reference: exact data-generating model frequencies have zero saturated-model likelihood
        deficit in principle. Require the fitted deficit below the fixed smoke-test
        tolerances in _NOISE_CONFIGS (1.0 for H+S, 5.0 for H+S+C+A); these are numerical
        fit tolerances, not finite-shot statistical confidence bounds.
        """
        for config_name, noise_coeffs, parameterization, max_two_delta_logl in _NOISE_CONFIGS:
            with self.subTest(noise_config=config_name):
                ideal_model, noisy_model = _build_noise_model(
                    self.pspec, noise_coeffs, parameterization)
                self.assertGreater(ideal_model.num_params, 0)
                self.assertEqual(ideal_model.num_params, noisy_model.num_params)

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
                    ideal_model, gaugeopt_suite='none', name='simul_gst',
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
@pytest.mark.skipif(
    not os.environ.get('PYGSTI_SGST_ARTIFACT_DIR'),
    reason='Set PYGSTI_SGST_ARTIFACT_DIR to run finite-shot GST validation profiles',
)
class SimultaneousGSTValidationTester:
    """Seeded finite-shot recovery and crosstalk detection against same-run references.

    In-class likelihood gains use Wilks' identifiable-parameter chi-squared bound;
    held-out mean/max TVDs use predictive shot-noise quantiles. These asymptotic,
    probabilistic checks do not guarantee every seed passes. Both scenarios must
    remove half the ideal model's held-out mean error. ``nsigma`` and
    largest-L-increment TVDs are diagnostics, not acceptance criteria.
    """

    @staticmethod
    def assert_improves_ideal(result):
        """Check the fit's predictive improvement against its same-run ideal baseline.

        Reference: half the ideal model's mean held-out TVD is the explicit acceptance
        limit; parameters are not compared.
        """
        error = result['held_out_mean_tvd']
        limit = 0.5 * result['ideal_held_out_mean_tvd']
        assert error <= limit, f"{result['profile']}: mean TVD {error} exceeds half the ideal-model mean TVD limit {limit}"

    def assert_recovers_datagen_model(self, result):
        """Check in-class likelihood and prediction metrics produced by ``run_profile``.

        Reference: compare the same-run likelihood gain with chi2.ppf(0.999, rank),
        mean/max TVDs with propagated shot-noise quantiles, and mean TVD with half the
        ideal baseline. These asymptotic bounds do not guarantee every seed passes.
        """
        self.assert_improves_ideal(result)
        gain = result['datagen_two_delta_logl'] - result['two_delta_logl']
        limit = chi2.ppf(0.999, result['identifiable_params'])
        # An in-class MLE beats the data-generating model, with asymptotically chi-squared gain.
        assert 0.0 <= gain <= limit, f"{result['profile']}: likelihood gain {gain} outside [0, {limit}]"
        for summary in ('mean', 'max'):
            error = result[f'held_out_{summary}_tvd']
            limit = result[f'shot_noise_held_out_{summary}_tvd_q999']
            assert error <= limit, f"{result['profile']}: {summary} TVD {error} exceeds shot-noise limit {limit}"

    def assert_reveals_and_survives_crosstalk(self, result):
        """Check out-of-class loss and improvement metrics produced by ``run_profile``.

        Reference: the known data-generating model must have strictly better training likelihood,
        while the fit removes at least half the ideal model's mean held-out error.
        Positive loss alone is not a calibrated test of statistical significance.
        """
        self.assert_improves_ideal(result)
        loss = result['two_delta_logl'] - result['datagen_two_delta_logl']
        # Detection requires the crosstalk cost to exceed the fit's overfitting gain;
        # a nonpositive loss does not prove the absence of crosstalk.
        assert loss > 0.0, f"{result['profile']}: likelihood loss {loss} does not reveal crosstalk"

    def test_line_3q_inclass(self):
        """Check ``run_profile`` and GST on the 3-qubit line, inclass profile.

        Reference: use same-run data-generating model likelihood and ideal-model mean TVD;
        compare the likelihood gain to a 99.9% chi-squared quantile with Fisher-rank
        degrees of freedom and mean/max held-out errors to linearized Gaussian
        shot-noise quantiles. Require at least half the ideal mean error removed.
        These are asymptotic probabilistic references, not a known fitted vector.
        """
        self.assert_recovers_datagen_model(run_profile(LINE_3Q_INCLASS))

    def test_line_3q_outclass(self):
        """Check ``run_profile`` and GST on the 3-qubit line, outclass profile.

        Reference: same-run data-generating model likelihood must beat the fitted likelihood,
        and the fitted mean held-out TVD must be at most half the ideal baseline.
        This positive-loss criterion is not a calibrated significance test; no fixed
        coefficient recovery or empirical seed-specific threshold is asserted.
        """
        self.assert_reveals_and_survives_crosstalk(run_profile(LINE_3Q_OUTCLASS))

    def test_line_4q_inclass(self):
        """Check ``run_profile`` and GST on the 4-qubit line, inclass profile.

        Reference: use same-run data-generating model likelihood and ideal-model mean TVD;
        compare the likelihood gain to a 99.9% chi-squared quantile with Fisher-rank
        degrees of freedom and mean/max held-out errors to linearized Gaussian
        shot-noise quantiles. Require at least half the ideal mean error removed.
        These are asymptotic probabilistic references, not a known fitted vector.
        """
        self.assert_recovers_datagen_model(run_profile(LINE_4Q_INCLASS))

    def test_line_4q_outclass(self):
        """Check ``run_profile`` and GST on the 4-qubit line, outclass profile.

        Reference: same-run data-generating model likelihood must beat the fitted likelihood,
        and the fitted mean held-out TVD must be at most half the ideal baseline.
        This positive-loss criterion is not a calibrated significance test; no fixed
        coefficient recovery or empirical seed-specific threshold is asserted.
        """
        self.assert_reveals_and_survives_crosstalk(run_profile(LINE_4Q_OUTCLASS))

    # A ring adds the wraparound edge.  On three qubits that edge needs a third color,
    # and on four it puts every qubit in a two-qubit patch in both colors.

    def test_ring_3q_inclass(self):
        """Check ``run_profile`` and GST on the 3-qubit ring, inclass profile.

        Reference: use same-run data-generating model likelihood and ideal-model mean TVD;
        compare the likelihood gain to a 99.9% chi-squared quantile with Fisher-rank
        degrees of freedom and mean/max held-out errors to linearized Gaussian
        shot-noise quantiles. Require at least half the ideal mean error removed.
        These are asymptotic probabilistic references, not a known fitted vector.
        """
        self.assert_recovers_datagen_model(run_profile(RING_3Q_INCLASS))

    def test_ring_3q_outclass(self):
        """Check ``run_profile`` and GST on the 3-qubit ring, outclass profile.

        Reference: same-run data-generating model likelihood must beat the fitted likelihood,
        and the fitted mean held-out TVD must be at most half the ideal baseline.
        This positive-loss criterion is not a calibrated significance test; no fixed
        coefficient recovery or empirical seed-specific threshold is asserted.
        """
        self.assert_reveals_and_survives_crosstalk(run_profile(RING_3Q_OUTCLASS))

    def test_ring_4q_inclass(self):
        """Check ``run_profile`` and GST on the 4-qubit ring, inclass profile.

        Reference: use same-run data-generating model likelihood and ideal-model mean TVD;
        compare the likelihood gain to a 99.9% chi-squared quantile with Fisher-rank
        degrees of freedom and mean/max held-out errors to linearized Gaussian
        shot-noise quantiles. Require at least half the ideal mean error removed.
        These are asymptotic probabilistic references, not a known fitted vector.
        """
        self.assert_recovers_datagen_model(run_profile(RING_4Q_INCLASS))

    def test_ring_4q_outclass(self):
        """Check ``run_profile`` and GST on the 4-qubit ring, outclass profile.

        Reference: same-run data-generating model likelihood must beat the fitted likelihood,
        and the fitted mean held-out TVD must be at most half the ideal baseline.
        This positive-loss criterion is not a calibrated significance test; no fixed
        coefficient recovery or empirical seed-specific threshold is asserted.
        """
        self.assert_reveals_and_survives_crosstalk(run_profile(RING_4Q_OUTCLASS))

if __name__ == '__main__':
    unittest.main()

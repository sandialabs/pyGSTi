import warnings

import numpy as np

import pygsti.tools.matrixtools as mt
from pygsti.models import ExplicitOpModel
from pygsti.models import modelconstruction as mc
from pygsti.modelmembers.operations import FullArbitraryOp
from pygsti.processors import QubitProcessorSpec
from pygsti.report import reportables as rptbl
from ..util import BaseCase

# A Hadamard gate estimate taken from a failing case reported by an end user, whose error bars on the
# rotation angle and log inexactness came back thousands of times too large. Kept verbatim (up to
# zeroing the first row, which was TP to within 1e-16) because it exercises every part of the
# pathology at once: the target has -1 eigenvalues, and the estimate has two *distinct* real negative
# eigenvalues and therefore no real matrix logarithm at all.
_HADAMARD_ESTIMATE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0005827386057873005, -1.596081539413828e-06, 0.0026842386726460437, 0.9931235460359923],
    [-0.00047897732274857033, -0.0026434875205955393, -0.9881533456738786, -0.0028483281874282394],
    [-0.000516711221855366, 0.9890358028125664, 0.0028055709936753892, 0.00026052111853187057]])

_HADAMARD_TARGET = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]])


def _single_gate_model(mx):
    """A one-qubit, one-gate model wrapping `mx` in the Pauli-product basis."""
    model = ExplicitOpModel(['Q0'], 'pp')
    model.operations['Gh'] = FullArbitraryOp(mx)
    return model


def _pi_rotation_target_model():
    """A one-qubit model containing a pi rotation (Gxpi), whose superoperator has -1 eigenvalues,
    alongside a pi/2 rotation (Gzpi2), whose superoperator does not."""
    ps = QubitProcessorSpec(1, ['Gxpi', 'Gzpi2'], geometry='line')
    return mc.create_explicit_model(ps, ideal_gate_type='full')


def _damped(model, target, rates=(0.01, 0.02, 0.03)):
    """Damp every gate of `target` anisotropically. Applied to a pi rotation this splits the
    degenerate -1 eigenvalue into two distinct negative reals, so no real logarithm exists."""
    damping = np.diag([1.0] + [1.0 - r for r in rates])
    noisy = model.copy()
    for gl in list(noisy.operations.keys()):
        noisy.operations[gl] = FullArbitraryOp(
            damping @ target.operations[gl].to_dense('HilbertSchmidt'))
    return noisy


class GeneralDecompositionTester(BaseCase):
    """Covers the `-1`-eigenvalue branch of `general_decomposition`, which uses the BCH
    approximation to the matrix logarithm."""

    def setUp(self):
        self.target = _pi_rotation_target_model()
        self.noisy = _damped(self.target, self.target)

    def _decomp(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rptbl.general_decomposition(model, self.target)

    def test_pi_rotation_decomposition(self):
        # Gxpi has -1 eigenvalues and so exercises the BCH branch; Gzpi2 goes through real_matrix_log.
        self.assertTrue(np.any(np.isclose(
            np.linalg.eigvals(self.target.operations['Gxpi', 0].to_dense('HilbertSchmidt')), -1.0)))
        self.assertFalse(np.any(np.isclose(
            np.linalg.eigvals(self.target.operations['Gzpi2', 0].to_dense('HilbertSchmidt')), -1.0)))

        decomp = self._decomp(self.noisy)

        # Angles are reported in units of pi.
        self.assertAlmostEqual(decomp['Gxpi:0 angle'], 1.0, places=4)
        self.assertAlmostEqual(decomp['Gzpi2:0 angle'], 0.5, places=4)
        self.assertArraysAlmostEqual(np.abs(decomp['Gxpi:0 axis']), np.array([1.0, 0.0, 0.0]), places=4)
        self.assertArraysAlmostEqual(np.abs(decomp['Gzpi2:0 axis']), np.array([0.0, 0.0, 1.0]), places=4)
        # The two rotation axes are perpendicular (pi/2 apart, again in units of pi).
        self.assertAlmostEqual(decomp['Gxpi:0,Gzpi2:0 axis angle'], 0.5, places=4)

    def test_pi_rotation_no_real_log_inexactness(self):
        # The damped pi rotation has no real logarithm, so a nonzero inexactness is expected, but it
        # should sit at the obstruction floor set by the splitting of the two negative eigenvalues.
        decomp = self._decomp(self.noisy)
        evals = np.linalg.eigvals(self.noisy.operations['Gxpi', 0].to_dense('HilbertSchmidt'))
        neg = np.sort(evals[evals.real < 0].real)
        self.assertEqual(len(neg), 2)
        floor = np.sqrt(2) * abs(neg[1] - neg[0]) / 2
        self.assertAlmostEqual(decomp['Gxpi:0 log inexactness'], floor, places=6)

        # The pi/2 gate does have an exact real logarithm.
        self.assertAlmostEqual(decomp['Gzpi2:0 log inexactness'], 0.0, places=8)

    def test_pi_rotation_out_of_domain_yields_nan(self):
        # If a gate is nowhere near its target the BCH construction has no valid domain, and the
        # quantities for *that gate only* should degrade to NaN rather than raising.
        broken = self.noisy.copy()
        broken.operations['Gxpi', 0] = FullArbitraryOp(np.eye(4))

        with self.assertWarns(Warning):
            decomp = rptbl.general_decomposition(broken, self.target)

        self.assertTrue(np.isnan(decomp['Gxpi:0 angle']))
        self.assertTrue(np.isnan(decomp['Gxpi:0 log inexactness']))
        self.assertTrue(np.all(np.isnan(decomp['Gxpi:0 axis'])))
        self.assertFalse(np.isnan(decomp['Gzpi2:0 angle']))

    def test_pi_rotation_gradient_is_step_size_independent(self):
        """Regression test for spuriously large error bars on pi-rotation angles.

        `ConfidenceRegionFactoryView` finite differences these quantities with respect to model
        parameters. The logarithm must therefore be a smooth function of the gate. The optimizer-based
        `approximate_matrix_log` is not: its output jitters at a level set by the optimizer tolerance
        rather than by the perturbation, so the difference quotient blows up as 1/eps and the resulting
        error bars are pure noise. The BCH construction is smooth, so the difference quotient converges.
        """
        base = self.noisy
        v0 = base.to_vector()
        rng = np.random.default_rng(2024)
        direction = rng.normal(size=len(v0))
        direction /= np.linalg.norm(direction)

        def angle(eps):
            perturbed = base.copy()
            perturbed.from_vector(v0 + eps * direction)
            return self._decomp(perturbed)['Gxpi:0 angle']

        a0 = angle(0.0)
        derivs = [(angle(eps) - a0) / eps for eps in (1e-8, 1e-7, 1e-6, 1e-5)]

        # Spread across four decades of step size must be tiny relative to the derivative itself.
        # With `approximate_matrix_log` these span roughly -108 to -0.25.
        spread = max(derivs) - min(derivs)
        self.assertLess(spread, 1e-4 * abs(np.mean(derivs)))


class HadamardRegressionTester(BaseCase):
    """Regression tests pinned to a real Hadamard estimate reported by an end user.

    See `_HADAMARD_ESTIMATE`. This gate produced error bars roughly three orders of magnitude too
    large, because the quantities below were computed with an optimizer whose output was not a smooth
    function of the gate.
    """

    def setUp(self):
        self.estimate = _single_gate_model(_HADAMARD_ESTIMATE)
        self.target = _single_gate_model(_HADAMARD_TARGET)

    def _decomp(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rptbl.general_decomposition(model, self.target)

    def test_no_real_logarithm_exists(self):
        # The target is a pi rotation, and the estimate carries the resulting pathology: its two
        # negative eigenvalues are real but *distinct*, which violates the condition for a real matrix
        # logarithm to exist (they would have to be paired). real_matrix_log is forced to go complex.
        evals = np.linalg.eigvals(_HADAMARD_ESTIMATE)
        self.assertTrue(np.allclose(evals.imag, 0.0))
        neg = np.sort(evals[evals.real < 0].real)
        self.assertEqual(len(neg), 2)
        self.assertAlmostEqual(neg[0], -0.990943613941, places=10)
        self.assertAlmostEqual(neg[1], -0.988150228999, places=10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertGreater(np.linalg.norm(mt.real_matrix_log(_HADAMARD_ESTIMATE, "ignore").imag), 1.0)

    def test_decomposition_values(self):
        # The target is a pi rotation, so log(target) lands on the branch cut of the principal
        # logarithm: log(-1) = +/- i*pi, and the two branches report the same rotation as
        # (axis, angle) and (-axis, 2*pi - angle). unitary_superoperator_matrix_log's branch
        # convention fixes the choice identically on every BLAS backend (OpenBLAS and Accelerate
        # agree to ~1e-14 here), so the signed values below are pinned tightly on purpose: if the
        # convention is ever lost (e.g. by reverting to scipy.linalg.logm, where the backend picks
        # the branch), the angle crosses to just above 1 and the axis flips sign on Accelerate.
        decomp = self._decomp(self.estimate)

        # Angle is in units of pi; a Hadamard comes out just *under* 1 on the chosen branch.
        self.assertAlmostEqual(decomp['Gh angle'], 0.9999652938, places=8)

        self.assertAlmostEqual(decomp['Gh log inexactness'], 0.0021073766, places=8)

        # The middle (Y) component is ~1e-05, i.e. numerical zero for a Hadamard.
        self.assertArraysAlmostEqual(
            np.asarray(decomp['Gh axis']),
            np.array([0.7070608183, 7.9894403e-06, 0.7071527410]), places=8)

    def test_inexactness_is_near_the_obstruction_floor(self):
        # Because no real logarithm exists, a nonzero inexactness is unavoidable. Its lower bound is
        # set by the splitting of the two negative eigenvalues, and BCH should land close to it -- so
        # the reported inexactness reflects a real property of the gate rather than optimizer noise.
        decomp = self._decomp(self.estimate)
        evals = np.linalg.eigvals(_HADAMARD_ESTIMATE)
        neg = np.sort(evals[evals.real < 0].real)
        floor = np.sqrt(2) * abs(neg[1] - neg[0]) / 2
        self.assertAlmostEqual(floor, 0.001975221, places=8)
        self.assertGreaterEqual(decomp['Gh log inexactness'], floor)
        self.assertLess(decomp['Gh log inexactness'], 1.1 * floor)

    def test_gradient_is_step_size_independent(self):
        # The failure this gate was reported for. `approximate_matrix_log` gives a relative spread of
        # about 3.0 here (i.e. the finite difference is entirely noise, and diverges as 1/eps);
        # the BCH construction gives about 8e-7.
        v0 = self.estimate.to_vector()
        rng = np.random.default_rng(0)
        direction = rng.normal(size=len(v0))
        direction /= np.linalg.norm(direction)

        def angle(eps):
            perturbed = _single_gate_model(_HADAMARD_ESTIMATE)
            perturbed.from_vector(v0 + eps * direction)
            return self._decomp(perturbed)['Gh angle']

        a0 = angle(0.0)
        derivs = [(angle(eps) - a0) / eps for eps in (1e-9, 1e-8, 1e-7, 1e-6)]
        spread = (max(derivs) - min(derivs)) / abs(np.mean(derivs))
        self.assertLess(spread, 1e-4)

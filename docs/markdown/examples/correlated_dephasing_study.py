"""Joint fits of dephasing gate errors and noisy CPTP Lindblad SPAM.

Each channel has independent parameters in the same candidate hypothesis class.
The reference state is |+>^4. One unknown pre-measurement channel is shared by
the reference product-Pauli parity POVMs. There is no separate pulse-gate model.
Only K_prep + K_meas is identifiable. Run with the source checkout on PYTHONPATH.
"""

import itertools
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.special import rel_entr

from pygsti.baseobjs import Basis, QubitSpace
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.circuits import Circuit
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as LCB
from pygsti.modelmembers.povms import ComposedPOVM, StaticPOVMEffect, UnconstrainedPOVM
from pygsti.modelmembers.states import ComposedState, StaticState
from pygsti.models import ExplicitOpModel
from pygsti.tools.basistools import change_basis, stdmx_to_vec, vec_to_stdmx
from pygsti.tools.jamiolkowski import jamiolkowski_iso


DIRECTIONS = ('ZIII', 'IZII', 'IIZI', 'IIIZ')
EDGES = ((0, 1), (1, 2), (2, 3), (0, 3))
FAMILIES = ('independent', 'graph', 'dense', 'graph+common')


def parity_design(paulis, times):
    """Fourier weights for product X/Y parities on the reference |+>^4 state.

    probabilities() includes noisy preparation, idle, and pre-measurement maps.
    Odd-Y parities vanish in this real dephasing model. Padding allows a shared
    evaluator for single, pair, and higher-order product measurements.
    """
    vectors, weights, durations, labels = [], [], [], []
    for t in times:
        for word in paulis:
            sites = [i for i, p in enumerate(word) if p != 'I']
            ys = [i for i, p in enumerate(word) if p == 'Y']
            assert len(word) == 4 and len(ys) % 2 == 0 and sites
            v, w = np.zeros((16, 4)), np.zeros(16)
            for k, signs in enumerate(itertools.product((-1., 1.), repeat=len(sites))):
                v[k, sites] = signs
                w[k] = (-1.) ** (len(ys) // 2) * np.prod(v[k, ys]) / 2 ** len(sites)
            vectors.append(v)
            weights.append(w)
            durations.append(t)
            labels.append(word)
    return np.array(vectors), np.array(weights), np.array(durations), labels


def product_design(times=(0., 1., 2., 4., 8.)):
    paulis = [''.join('X' if a == i else 'I' for a in range(4)) for i in range(4)]
    for i, j in itertools.combinations(range(4), 2):
        for axis in ('X', 'Y'):
            paulis.append(''.join(axis if a in (i, j) else 'I' for a in range(4)))
    return parity_design(paulis, times)


def heldout_design():
    paulis = []
    for size in (3, 4):
        for sites in itertools.combinations(range(4), size):
            for axes in itertools.product('XY', repeat=size):
                if axes.count('Y') % 2 == 0:
                    word = ['I'] * 4
                    for i, axis in zip(sites, axes):
                        word[i] = axis
                    paulis.append(''.join(word))
    return parity_design(paulis, (.5, 1.5, 3., 6.))


def probabilities(matrices, design, matrix_jacobians=None):
    """Positive parity probabilities; matrices = (K_gate, K_prep, K_meas).

    Jacobians use a shared parameter axis containing every fitted parameter.
    All three contributions to t K_gate + K_prep + K_meas are differentiated.
    """
    gate, prep, meas = matrices
    vectors, weights, times, _ = design
    integrated = times[:, None, None] * gate + prep + meas
    rates = 2 * np.einsum('msi,mij,msj->ms', vectors, integrated, vectors)
    contrasts = np.exp(-rates)
    p = .5 * (1 + np.sum(weights * contrasts, axis=1))
    if matrix_jacobians is None:
        return p
    gate_jac, prep_jac, meas_jac = matrix_jacobians
    integrated_jac = times[:, None, None, None] * gate_jac + prep_jac + meas_jac
    rate_jac = 2 * np.einsum('msi,mijp,msj->msp', vectors, integrated_jac, vectors)
    jac = -.5 * np.einsum('ms,ms,msp->mp', weights, contrasts, rate_jac)
    return p, jac


class KossakowskiChart:
    """One physical channel; graph+common uses an example-only PSD sum."""

    def __init__(self, family, initial_rate):
        edges = () if family == 'independent' else (
            None if family == 'dense' else [(DIRECTIONS[i], DIRECTIONS[j]) for i, j in EDGES])
        # The published results use completion for the cycle graph (the default there is now
        # the contraction chart); the fill-free chart applies to the other families either way.
        errorgen = LindbladErrorgen.from_correlated_dephasing(
            QubitSpace(4), edges, initial_stochastic_rate=initial_rate,
            param_mode='cholesky completion' if family not in ('independent', 'dense') else 'auto'
        )
        self.family = family
        self.block = errorgen.coefficient_blocks[0]
        self.initial = self.block.to_vector()
        self.nblock = self.block.num_params
        if family == 'graph+common':
            self.initial = np.r_[self.initial, np.sqrt(initial_rate / 10)]
        self.nparams = len(self.initial)
        pos = {b: i for i, b in enumerate(self.block.basis_element_labels)}
        self.pairs = [(pos[label.basis_element_labels[0]], pos[label.basis_element_labels[-1]])
                      for label in self.block.elementary_errorgens]

    def evaluate(self, vector):
        self.block.from_vector(vector[:self.nblock])
        matrix = self.block.kossakowski_matrix().real
        coefficient_jac = self.block.deriv_wrt_params(vector[:self.nblock])
        jac = np.zeros((4, 4, self.nparams))
        for (i, j), derivative in zip(self.pairs, coefficient_jac):
            jac[i, j, :self.nblock] = derivative
            jac[j, i, :self.nblock] = derivative
        if self.family == 'graph+common':
            matrix += vector[-1] ** 2 * np.ones((4, 4))
            jac[:, :, -1] = 2 * vector[-1]
        return matrix, jac


class JointChart:
    """Independent, same-family channels for gate, preparation, and measurement."""

    def __init__(self, family):
        self.charts = [KossakowskiChart(family, rate) for rate in (.03, .006, .009)]
        self.nchannel = self.charts[0].nparams
        self.nparams = 3 * self.nchannel
        self.initial = np.concatenate([chart.initial for chart in self.charts])

    def evaluate(self, vector):
        matrices, jacobians = [], []
        for a, chart in enumerate(self.charts):
            indices = slice(a * self.nchannel, (a + 1) * self.nchannel)
            matrix, local_jac = chart.evaluate(vector[indices])
            jac = np.zeros((4, 4, self.nparams))
            jac[:, :, indices] = local_jac
            matrices.append(matrix)
            jacobians.append(jac)
        return matrices, jacobians


def binomial_residuals(p, positives, shots):
    """Signed sqrt likelihood-deviance residuals and derivatives with respect to p."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = positives / shots
    deviance = 2 * shots * (rel_entr(q, p) + rel_entr(1 - q, 1 - p))
    residual = np.sign(q - p) * np.sqrt(np.maximum(deviance, 0.))
    near = (q > 0) & (q < 1) & (np.abs(q - p) < 1e-7)
    far = ~near
    derivative = np.empty_like(p)
    derivative[far] = shots * (p[far] - q[far]) / (p[far] * (1 - p[far]) * residual[far])
    variance = p[near] * (1 - p[near])
    residual[near] = np.sqrt(shots) * (q[near] - p[near]) / np.sqrt(variance)
    derivative[near] = (-np.sqrt(shots) / np.sqrt(variance)
                        - .5 * np.sqrt(shots) * (q[near] - p[near]) * (1 - 2 * p[near]) / variance ** 1.5)
    return residual, derivative


def fit(family, design, positives, shots):
    chart = JointChart(family)

    def evaluate(vector, with_jacobian=False):
        matrices, derivatives = chart.evaluate(vector)
        p, jac = probabilities(matrices, design, derivatives)
        residual, slope = binomial_residuals(p, positives, shots)
        return slope[:, None] * jac if with_jacobian else residual

    optimum = least_squares(evaluate, chart.initial, jac=lambda v: evaluate(v, True),
                            max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                            tr_solver='lsmr', tr_options={'atol': 1e-12, 'btol': 1e-12})
    if not optimum.success:
        raise RuntimeError(f'{family} failed to converge: {optimum.message}')
    matrices, derivatives = chart.evaluate(optimum.x)
    _, jac = probabilities(matrices, design, derivatives)
    return matrices, {
        'parameters_per_channel': chart.nchannel,
        'parameters': chart.nparams,
        'probability_jacobian_rank': int(np.linalg.matrix_rank(jac, tol=1e-8)),
        'deviance': float(np.sum(optimum.fun ** 2)),
        'converged': bool(optimum.success),
        'status': int(optimum.status),
        'function_evaluations': int(optimum.nfev),
        'optimality': float(optimum.optimality),
    }


def observable(word):
    one = Basis.cast('PP', 4)
    out = np.ones((1, 1))
    for p in word:
        out = np.kron(out, one[p])
    return out


def make_model(matrices, paulis):
    """Independent full pyGSTi check of the analytic joint signals.

    Dense real charts reproduce the supplied matrices exactly. Family-specific
    restrictions during fitting are imposed by JointChart. One measurement map
    is shared across reference POVMs, independently of preparation and idle.
    """
    space = QubitSpace(4)
    gate, prep, meas = [
        ExpErrorgenOp(LindbladErrorgen.from_correlated_dephasing(space, initial_covariance=k))
        for k in matrices]
    model = ExplicitOpModel(space, basis='pp')
    reference = StaticState(stdmx_to_vec(np.ones((16, 16)) / 16, 'pp'),
                            basis='pp', state_space=space)
    model.preps['rho0'] = ComposedState(reference, prep)
    model.operations['Gi'] = gate  # Ideal idle followed by its error channel.
    for word in dict.fromkeys(paulis):
        effect = (np.eye(16) + observable(word)) / 2
        reference_povm = UnconstrainedPOVM([
            (label, StaticPOVMEffect(stdmx_to_vec(matrix, 'pp'), basis='pp', state_space=space))
            for label, matrix in (('+', effect), ('-', np.eye(16) - effect))
        ])
        model.povms['M' + word.lower()] = ComposedPOVM(meas, reference_povm, mx_basis='pp')
    assert model.num_params == 30
    return model


def verify_physical_signals(matrices, design):
    """Check composed SPAM, channel order, physicality, probabilities, and persistence."""
    model = make_model(matrices, design[3])
    expected = []
    for t, word in zip(design[2], design[3]):
        assert t == int(t)
        circuit = Circuit(('rho0',) + ('Gi',) * int(t) + ('M' + word.lower(),), line_labels=(0, 1, 2, 3))
        expected.append(model.probabilities(circuit)[('+',)])
    p_formula = probabilities(matrices, design)
    np.testing.assert_allclose(expected, p_formula, atol=3e-13)
    rho = vec_to_stdmx(model.preps['rho0'].to_dense(), 'pp')
    np.testing.assert_allclose(np.trace(rho), 1, atol=1e-13)
    assert np.linalg.eigvalsh(rho)[0] >= -1e-12
    min_choi = np.inf
    for channel in (model.operations['Gi'], model.preps['rho0'].error_map,
                    next(iter(model.povms.values())).error_map):
        matrix = channel.to_dense()
        np.testing.assert_allclose(matrix[0], np.eye(256)[0], atol=1e-13)
        eigmin = np.linalg.eigvalsh(jamiolkowski_iso(matrix, op_mx_basis='pp'))[0]
        min_choi = min(min_choi, eigmin)
        assert eigmin >= -1e-12
    for povm in model.povms.values():
        effects = [vec_to_stdmx(e.to_dense(), 'pp') for e in povm.values()]
        np.testing.assert_allclose(sum(effects), np.eye(16), atol=1e-13)
        assert min(np.linalg.eigvalsh(e)[0] for e in effects) >= -1e-12
    restored = ExplicitOpModel.from_nice_serialization(model.to_nice_serialization())
    assert restored.num_params == model.num_params
    np.testing.assert_allclose(restored.to_vector(), model.to_vector(), atol=1e-13)
    word = design[3][-1]
    np.testing.assert_allclose(restored.povms['M' + word.lower()]['+'].to_dense(),
                               model.povms['M' + word.lower()]['+'].to_dense(), atol=1e-13)
    return {'max_full_model_probability_error': float(np.max(np.abs(expected - p_formula))),
            'min_gate_and_spam_choi_eigenvalue': float(min_choi),
            'serialized_parameters': restored.num_params}


def dephasing_phase_cycles():
    """A-type and classical Poisson phase cycles, without preparation/measurement."""
    basis = Basis.cast('PP', 16)
    labels = [LEEL('S', ('ZI',)), LEEL('S', ('IZ',)),
              LEEL('C', ('ZI', 'IZ')), LEEL('A', ('ZI', 'IZ'))]
    block = LCB('other_sparse', basis, param_mode='cholesky', error_generator_labels=labels,
                initial_block_data=np.array([.03, .02, 0., .01]))
    errorgen = LindbladErrorgen([block], basis, 'pp', state_space=QubitSpace(2))
    generator = change_basis(errorgen.to_dense(), 'pp', 'std')
    np.testing.assert_allclose(generator - np.diag(generator.diagonal()), 0, atol=1e-14)
    phases = generator.diagonal().reshape(4, 4).imag
    cycle = phases[0, 1] + phases[1, 3] + phases[3, 0]
    assert abs(cycle) > .01
    # Classical Poisson diagonal-unitary kicks can also have a nonzero cycle.
    diagonal = np.exp(1j * np.array([0., np.pi / 2, 0., np.pi]))
    poisson_phases = .01 * np.outer(diagonal, diagonal.conj()).imag
    poisson_cycle = poisson_phases[0, 1] + poisson_phases[1, 3] + poisson_phases[3, 0]
    assert abs(poisson_cycle) > .01
    return float(cycle), float(poisson_cycle)


def truth_matrices(common=False):
    diagonals = ([.028, .033, .025, .030], [.010, .008, .011, .009], [.006, .010, .007, .008])
    edges = ((.007, -.006, .005, .004), (.0012, -.0010, .0008, .0011),
             (-.0008, .0011, -.0005, .0006))
    matrices = []
    for diagonal, phase_covariances, common_rate in zip(diagonals, edges, (.006, .0015, .0025)):
        matrix = np.diag(diagonal)
        for (i, j), value in zip(EDGES, phase_covariances):
            matrix[i, j] = matrix[j, i] = value
        if common:
            matrix += common_rate * np.ones((4, 4))
        assert np.linalg.eigvalsh(matrix)[0] > 0
        matrices.append(matrix)
    return matrices


def identifiability(matrices, design):
    """K-coordinate ranks with SPAM nuisance directions and their exact ambiguity."""
    coordinates = np.zeros((4, 4, 30))
    pairs = [(i, i) for i in range(4)] + list(itertools.combinations(range(4), 2))
    for k, (i, j) in enumerate(pairs):
        coordinates[i, j, k] = coordinates[j, i, k] = 1
    derivatives = [np.roll(coordinates, 10 * a, axis=-1) for a in range(3)]
    _, jac = probabilities(matrices, design, derivatives)
    result = {}
    for name, rows in (
            ('single_ramsey_means', np.array([p.count('X') == 1 and 'Y' not in p for p in design[3]])),
            ('all_product_parities', np.ones(len(jac), dtype=bool))):
        local = jac[rows]
        nuisance = local[:, 10:]
        u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
        nuisance_rank = np.count_nonzero(s > 1e-9)
        projected = local[:, :10] - u[:, :nuisance_rank] @ (u[:, :nuisance_rank].T @ local[:, :10])
        result[name] = {'joint_rank': int(np.linalg.matrix_rank(local, tol=1e-9)),
                        'spam_nuisance_rank': int(nuisance_rank),
                        'gate_rank_after_spam_projection': int(np.linalg.matrix_rank(projected, tol=1e-9))}
    assert result['single_ramsey_means'] == {
        'joint_rank': 8, 'spam_nuisance_rank': 4, 'gate_rank_after_spam_projection': 4}
    assert result['all_product_parities'] == {
        'joint_rank': 20, 'spam_nuisance_rank': 10, 'gate_rank_after_spam_projection': 10}
    shift = .0001 * np.eye(4)
    equivalent = (matrices[0], matrices[1] + shift, matrices[2] - shift)
    error = np.max(np.abs(probabilities(matrices, design) - probabilities(equivalent, design)))
    assert error < 1e-14
    return result, float(error)


def run_study():
    design, heldout = product_design(), heldout_design()
    truth = truth_matrices()
    ranks, ambiguity_error = identifiability(truth, design)
    chart = JointChart('graph+common')
    vector = chart.initial.copy()
    matrices, derivatives = chart.evaluate(vector)
    p, analytic = probabilities(matrices, design, derivatives)
    eps = 1e-6
    numeric = np.stack([(probabilities(chart.evaluate(vector + eps * e)[0], design)
                         - probabilities(chart.evaluate(vector - eps * e)[0], design)) / (2 * eps)
                        for e in np.eye(len(vector))], axis=-1)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=1e-9)
    observed = np.random.default_rng(90).binomial(50000, p)
    residual_jac = binomial_residuals(p, observed, 50000)[1][:, None] * analytic
    numeric_residual = np.stack([
        (binomial_residuals(probabilities(chart.evaluate(vector + eps * e)[0], design), observed, 50000)[0]
         - binomial_residuals(probabilities(chart.evaluate(vector - eps * e)[0], design), observed, 50000)[0])
        / (2 * eps) for e in np.eye(len(vector))], axis=-1)
    np.testing.assert_allclose(residual_jac, numeric_residual, rtol=3e-5, atol=3e-4)
    exact_p = probabilities(truth, design)
    exact_fit, exact_info = fit('graph', design, 50000 * exact_p, 50000)
    gate_error = float(np.max(np.abs(exact_fit[0] - truth[0])))
    spam_error = float(np.max(np.abs(exact_fit[1] + exact_fit[2] - truth[1] - truth[2])))
    assert gate_error < 2e-8 and spam_error < 2e-8

    scenarios = {}
    for seed, name in enumerate(('graph', 'graph+common'), 810):
        truth = truth_matrices(common=name == 'graph+common')
        shots = 50000
        true_p = probabilities(truth, design)
        positives = np.random.default_rng(seed).binomial(shots, true_p)
        true_deviance = float(np.sum(binomial_residuals(true_p, positives, shots)[0] ** 2))
        true_heldout = probabilities(truth, heldout)
        results = {}
        for family in FAMILIES:
            fitted, info = fit(family, design, positives, shots)
            predicted = probabilities(fitted, heldout)
            spam_sum = fitted[1] + fitted[2]
            true_spam_sum = truth[1] + truth[2]
            info.update({
                'deviance_minus_truth': info['deviance'] - true_deviance,
                'heldout_mean_tvd': float(np.mean(np.abs(predicted - true_heldout))),
                'heldout_max_tvd': float(np.max(np.abs(predicted - true_heldout))),
                'relative_gate_kossakowski_error': float(np.linalg.norm(fitted[0] - truth[0])
                                                          / np.linalg.norm(truth[0])),
                'relative_spam_sum_error': float(np.linalg.norm(spam_sum - true_spam_sum)
                                                 / np.linalg.norm(true_spam_sum)),
                'fitted_gate_kossakowski': fitted[0].tolist(),
                'fitted_spam_kossakowski_sum': spam_sum.tolist(),
                'representative_prep_kossakowski': fitted[1].tolist(),
                'representative_meas_kossakowski': fitted[2].tolist(),
                'minimum_channel_kossakowski_eigenvalues': [float(np.linalg.eigvalsh(k)[0]) for k in fitted],
            })
            assert min(info['minimum_channel_kossakowski_eigenvalues']) > 0
            if family == name or family == 'dense':
                assert info['deviance_minus_truth'] <= 1e-6
            if family in ('independent', 'graph'):
                allowed = set(EDGES) if family == 'graph' else set()
                for matrix in fitted:
                    for i, j in itertools.combinations(range(4), 2):
                        if (i, j) not in allowed:
                            assert matrix[i, j] == 0
            results[family] = info
        zero = [np.zeros((4, 4))] * 3
        scenarios[name] = {
            'true_gate_kossakowski': truth[0].tolist(),
            'true_prep_kossakowski': truth[1].tolist(),
            'true_meas_kossakowski': truth[2].tolist(),
            'true_deviance': true_deviance,
            'noiseless_model_heldout_mean_tvd': float(np.mean(np.abs(probabilities(zero, heldout) - true_heldout))),
            'prep_probability_effect': float(np.max(np.abs(
                true_p - probabilities((truth[0], zero[0], truth[2]), design)))),
            'meas_probability_effect': float(np.max(np.abs(
                true_p - probabilities((truth[0], truth[1], zero[0]), design)))),
            'fits': results,
        }
    dense = np.ones((3, 3)) + .1 * np.eye(3)
    thresholded = dense.copy()
    thresholded[0, 2] = thresholded[2, 0] = 0
    assert np.linalg.eigvalsh(dense)[0] > 0 and np.linalg.eigvalsh(thresholded)[0] < 0
    a_type_cycle, poisson_cycle = dephasing_phase_cycles()
    words = list(dict.fromkeys(design[3])) + ['XXYY', 'XXXX']
    physical = verify_physical_signals(truth_matrices(), parity_design(words, (0., 1., 4.)))
    return {
        'spam': 'independent CPTP Lindblad prep and pre-measurement channels; same family as gate; jointly fitted',
        'spam_ambiguity': 'only K_prep + K_meas is identifiable; representatives are not individual estimates',
        'measurement_sharing': 'one fitted channel shared across reference product-Pauli parity POVMs',
        'controls': 'no separate pulse-gate model; reference |+>^4 and Pauli POVMs with fitted error channels',
        'shots_per_setting': shots, 'training_settings': len(design[2]),
        'heldout_settings': len(heldout[2]), 'identifiability_ranks': ranks,
        'spam_exchange_probability_error': ambiguity_error,
        'max_analytic_probability_jacobian_error': float(np.max(np.abs(analytic - numeric))),
        'max_binomial_residual_jacobian_error': float(np.max(np.abs(residual_jac - numeric_residual))),
        'expected_data_gate_recovery_error': gate_error,
        'expected_data_spam_sum_recovery_error': spam_error,
        'expected_data_fit_converged': exact_info['converged'],
        'physical_model_checks': physical,
        'entrywise_zeroing_negative_eigenvalue': float(np.linalg.eigvalsh(thresholded)[0]),
        'a_type_phase_cycle': a_type_cycle,
        'classical_poisson_phase_cycle': poisson_cycle,
        'scenarios': scenarios,
    }


if __name__ == '__main__':
    print(json.dumps(run_study(), indent=2))

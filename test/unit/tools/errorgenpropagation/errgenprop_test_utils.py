"""
Shared helpers and fixtures for unit tests of `pygsti.tools.errorgenpropagation`.
"""
from contextlib import contextmanager
from itertools import product
from unittest import mock
import random

import numpy as np

from pygsti.baseobjs import Label, QubitSpace, BuiltinBasis
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.tools import errorgenpropagation as _eprop
from pygsti.tools.basistools import change_basis
from pygsti.tools.lindbladtools import random_cptp_errorgen_rates
from pygsti.tools.matrixtools import print_mx

from ..mptools_test_utils import assert_pool_dispatch


@contextmanager
def forced_parallel(*operations):
    """
    Context manager forcing the named error generator propagation operations (any of 'alpha',
    'magnus', 'layer_prop', 'commuting_product'; all of them if none are given) to take their
    parallel code path regardless of workload size, restoring the default configuration on exit.
    """
    _eprop.configure_errorgen_propagation_parallelization(force_parallel=list(operations) if operations else True)
    try:
        yield
    finally:
        _eprop.configure_errorgen_propagation_parallelization(reset_to_defaults=True)


@contextmanager
def parallel_path_exercised(testcase, operations, expected_dispatch):
    """
    Force the parallel path of `operations` (see `forced_parallel`) within the block, and afterwards
    assert that work was really dispatched to worker processes through each of the callables named in
    `expected_dispatch` (see `mptools_test_utils.assert_pool_dispatch`). Yields the dispatch spy.
    """
    with forced_parallel(*operations), assert_pool_dispatch(testcase, expected_dispatch) as spy:
        yield spy


def _mock_taylor_expansion():
    """
    Patch `error_generator_taylor_expansion` with a mock that records its call arguments but
    always evaluates serially (so tests checking argument forwarding do not spawn workers).
    """
    real = _eprop.error_generator_taylor_expansion

    def serial(errorgen_dict, order=1, truncation_threshold=1e-14, num_workers=None, pool=None):
        return real(errorgen_dict, order, truncation_threshold)

    return mock.patch.object(_eprop.expansions, 'error_generator_taylor_expansion', side_effect=serial)


def select_random_items_from_multiple_lists(input_lists, num_items, seed=None):
    """
    Select a specified number of items at random from multiple lists without replacement.

    Parameters:
    input_lists (list of lists): The lists from which to select items.
    num_items (int): The number of items to select.
    seed (int, optional): The seed for the random number generator. Defaults to None.

    Returns:
    list of lists: A list of lists containing the randomly selected items from each input list.
    """
    if not input_lists:
        raise ValueError("input_lists cannot be empty")

    list_length = len(input_lists[0])
    for lst in input_lists:
        if len(lst) != list_length:
            raise ValueError("All input lists must have the same length")

    if num_items > list_length:
        raise ValueError("num_items cannot be greater than the length of the input lists")

    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(list_length), num_items)

    return [[lst[i] for i in indices] for lst in input_lists]


def _all_sector_rate_subset(num_qubits, num_per_sector, seed):
    """
    A seeded random subset of a random CPTP error generator rate dictionary on `num_qubits`
    qubits, keeping `num_per_sector` labels from each of the H, S, C and A sectors. The subset
    is generally not CP (S terms supporting the kept C/A terms may be dropped), which is
    irrelevant for error generator propagation.
    """
    full = random_cptp_errorgen_rates(QubitSpace(num_qubits), errorgen_types=('H', 'S', 'C', 'A'), H_params=(0, .01),
                                             SCA_params=(0, .01), label_type='local', seed=seed)
    rng = np.random.default_rng(seed + 1)
    subset = {}
    for sector in 'HSCA':
        lbls = [lbl for lbl in full if lbl.errorgen_type == sector]
        for i in sorted(rng.choice(len(lbls), min(num_per_sector, len(lbls)), replace=False)):
            subset[lbls[i]] = float(full[lbls[i]])
    return subset


def sample_error_rates_dict(pspec, strengths, seed=None):
    """
    Sample an error rates dictionary across the gates of a processor spec.

    strengths = {1: {'S':0.001, 'H':0.01},
                 2: {'S':0.01,'H':0.1}}
    """
    qubits = pspec.qubit_labels
    errors_rates_dict = {}
    for gate, availability in pspec.availability.items():
        n = pspec.gate_num_qubits(gate)
        if availability == 'all-edges':
            assert n == 1, "Currently require all 2-qubit gates have a specified availability!"
            qubits_for_gate = qubits
        else:
            qubits_for_gate = availability
        for qs in qubits_for_gate:
            label = Label(gate, qs)
            max_stength = strengths.get(label, None)
            if max_stength is None:
                max_stength = strengths.get(gate, None)
            if max_stength is None:
                max_stength = strengths[n]
            errors_rates_dict[label] = sample_error_rates(max_stength, n, seed)
    return errors_rates_dict


def sample_error_rates(strengths, n, seed=None):
    """
    Sample an error rates dictionary for dependent gates on `n` qubits.
    """
    error_rates_dict = {}

    basis = BuiltinBasis('pp', 4**n)
    rng = np.random.default_rng(seed)

    stochastic_strength = strengths['S'] * rng.random()
    s_error_rates = rng.random(4 ** n - 1)
    s_error_rates = s_error_rates / np.sum(s_error_rates) * stochastic_strength

    hamiltonian_strength = strengths['H'] * rng.random()
    h_error_rates = rng.random(4 ** n - 1)
    h_error_rates = h_error_rates * np.sqrt(hamiltonian_strength) / np.sqrt(np.sum(h_error_rates**2))

    error_rates_dict.update({('S', basis.labels[i + 1]): s_error_rates[i] for i in range(4 ** n - 1)})
    error_rates_dict.update({('H', basis.labels[i + 1]): h_error_rates[i] for i in range(4 ** n - 1)})

    return error_rates_dict


def probabilities_errorgen_prop(error_propagator, target_model, circuit, use_bch=False, bch_order=1,
                                truncation_threshold=1e-14):
    if use_bch:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True, use_bch=use_bch,
                                                         bch_kwargs={'bch_order': bch_order,
                                                                     'truncation_threshold': truncation_threshold})
    else:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True)
    ideal_channel = target_model.sim.product(circuit)
    ideal_prep = target_model.circuit_layer_operator(Label('rho0'), typ='prep').copy()
    ideal_meas = target_model.circuit_layer_operator(Label('Mdefault'), typ='povm').copy()
    prob_vec = np.zeros(len(ideal_meas))
    for i, effect in enumerate(ideal_meas.values()):
        dense_effect = effect.to_dense().copy()
        dense_prep = ideal_prep.to_dense().copy()
        prob_vec[i] = np.linalg.multi_dot(
            [dense_effect.reshape((1, -1)), eoc_channel, ideal_channel, dense_prep.reshape((-1, 1))]
        ).item()
    return prob_vec


def pauli_expectation_errorgen_prop(error_propagator, target_model, circuit, pauli, use_bch=False, bch_order=1,
                                    truncation_threshold=1e-14):
    if use_bch:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True, use_bch=use_bch,
                                                         bch_kwargs={'bch_order': bch_order,
                                                                     'truncation_threshold': truncation_threshold})
    else:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True)
    ideal_channel = target_model.sim.product(circuit)
    ideal_prep = target_model.circuit_layer_operator(Label('rho0'), typ='prep').copy()

    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    pauli_vec = np.ravel(pauli_unitary)
    pauli_vec.reshape((len(pauli_vec), 1))
    pauli_vec = change_basis(pauli_vec, 'std', 'pp')
    dense_prep = ideal_prep.to_dense().copy()
    expectation = np.linalg.multi_dot(
        [pauli_vec.reshape((1, len(pauli_vec))), eoc_channel, ideal_channel, dense_prep.reshape((len(dense_prep), 1))]
    ).item()
    return expectation


def _compare_analytic_numeric_iterative_composition(num_qubits):
    complete_errorgen_basis = CompleteElementaryErrorgenBasis('PP', QubitSpace(num_qubits), default_label_type='local')
    complete_errorgen_lbls = complete_errorgen_basis.labels
    complete_errorgen_lbl_matrix_dict = {
        lbl: mat for lbl, mat in zip(complete_errorgen_lbls, complete_errorgen_basis.elemgen_matrices)
    }

    errorgen_label_triples = list(product(complete_errorgen_lbls, repeat=3))
    rng = np.random.default_rng(seed=1234)
    random_indices = rng.choice(len(errorgen_label_triples), 10000)
    random_triples = [errorgen_label_triples[idx] for idx in random_indices]
    random_triples_stim = [(_LSE.cast(a), _LSE.cast(b), _LSE.cast(c)) for a, b, c in random_triples]

    for i, (triple_1, triple_2) in enumerate(zip(random_triples, random_triples_stim)):
        numeric_composition = _eprop.iterative_error_generator_composition_numerical(
            triple_1, (1, 1, 1), complete_errorgen_lbl_matrix_dict
        )
        analytic_composition = _eprop.iterative_error_generator_composition(triple_2, (1, 1, 1))
        analytic_composition_dict = dict()
        for lbl, rate in analytic_composition:
            local_lbl = lbl.to_local_eel()
            if analytic_composition_dict.get(local_lbl, None) is None:
                analytic_composition_dict[local_lbl] = rate
            else:
                analytic_composition_dict[local_lbl] += rate
        analytic_composition = analytic_composition_dict
        try:
            analytic_composition_mat = _eprop.errorgen_layer_to_matrix(
                analytic_composition, num_qubits, errorgen_matrix_dict=complete_errorgen_lbl_matrix_dict
            )
        except KeyError:
            print(f'{analytic_composition=}')
        norm_diff = np.linalg.norm(numeric_composition - analytic_composition_mat)
        if norm_diff > 1e-10:
            print(f'Difference in compositions for triple {triple_1} is greater than 1e-10.')
            print(f'{triple_2=}')
            print(f'Error encountered on iteration {i}')
            print(f'{np.linalg.norm(numeric_composition-analytic_composition_mat)=}')
            print('numeric_composition=')
            print_mx(numeric_composition)

            for lbl, dual in zip(complete_errorgen_basis.labels, complete_errorgen_basis.elemgen_dual_matrices):
                rate = np.trace(dual.conj().T @ numeric_composition)
                if abs(rate) > 1e-3:
                    print(f'{lbl}: {rate}')

            print(f'{analytic_composition=}')
            print('analytic_composition_mat=')
            print_mx(analytic_composition_mat)
            raise ValueError('Numeric and analytic error generator compositions were not found to be identical!')


def _dense_zassenhaus_generators_analytic(errorgen_groups, errogen_propagator, zassenhaus_order=1):
    zassenhaus_gens = _eprop.zassenhaus_formula(errorgen_groups, zassenhaus_order)
    dense_zassenhaus_generators = [
        errogen_propagator.errorgen_layer_dict_to_errorgen(gen) for gen in zassenhaus_gens
    ]
    return dense_zassenhaus_generators

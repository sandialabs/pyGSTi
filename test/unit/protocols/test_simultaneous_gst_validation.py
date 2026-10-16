"""Tests for simultaneous-GST validation execution helpers."""

import dataclasses

import numpy as np
import pytest
from pygsti.circuits import Circuit

import test.helpers.simultaneous_gst_validation as validation
import test.integration.test_simultaneous_gst as integration
from test.helpers.simultaneous_gst_validation import (
    LINE_3Q_OUTCLASS,
    ValidationProfile,
    build_component_model,
    build_component_designs,
    build_fit_and_datagen_models,
    build_training_design_and_held_out_circuits,
    generate_finite_shot_data,
    spectator_support,
    spectator_term_text,
)


def test_somehs_component_models_have_expected_parameter_counts():
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2))
    oneq_design, twoq_design = build_component_designs(profile)
    assert build_component_model(profile, 1).num_params == 8
    assert build_component_model(profile, 2).num_params == 17
    assert oneq_design.qubit_labels == (0,)
    assert twoq_design.qubit_labels == (0, 1)


def test_line_4q_allhs_fit_model_has_independent_placement_parameters():
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2), model_terms='allhs')
    fit_model, _ = build_fit_and_datagen_models(profile)
    assert fit_model.num_params == 252


def test_inclass_datagen_model_is_a_seeded_point_in_the_hypothesis_class():
    profile = ValidationProfile.inclass(nqubits=3, max_lengths=(1, 2))
    fit_model, datagen_model = build_fit_and_datagen_models(profile)
    assert fit_model.num_params == datagen_model.num_params
    assert np.any(datagen_model.to_vector() != 0.0)
    _, repeated_datagen_model = build_fit_and_datagen_models(profile)
    assert np.array_equal(datagen_model.to_vector(), repeated_datagen_model.to_vector())


def test_spectator_profiles_default_to_calibrated_effect():
    default_profile = ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2))

    for profile in (default_profile, LINE_3Q_OUTCLASS, validation.LINE_4Q_OUTCLASS):
        assert (profile.spectator_term, profile.spectator_error) == (('S', 'Z:2'), 0.01)


def test_spectator_datagen_model_lies_outside_the_hypothesis_class():
    profile = ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.001)
    fit_model, datagen_model = build_fit_and_datagen_models(profile)
    assert fit_model.num_params < datagen_model.num_params
    cnot_coefficients = datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)]
    spectator_coefficients = {
        label: value for label, value in cnot_coefficients.items()
        if label.support == (2,)
    }
    assert len(spectator_coefficients) == 1
    spectator_label, spectator_value = next(iter(spectator_coefficients.items()))
    assert spectator_label.errorgen_type == 'S'
    assert spectator_label.basis_element_labels == ('Z',)
    assert spectator_value == pytest.approx(0.001)


def test_spectator_support_reads_qubits_from_coefficient_keys():
    assert spectator_support(('H', 'Z:2')) == (2,)
    assert spectator_support(('H', 'ZZ:1,2')) == (1, 2)
    assert spectator_support(('S', 'ZZZ:0,1,2')) == (0, 1, 2)
    assert spectator_support(('H', 'Z')) == ()
    assert spectator_term_text(('H', 'ZZ:1,2')) == 'H(ZZ:1,2)'


def test_weight_two_spectator_term_survives_the_seeded_local_noise():
    """A weight-2 key has support (1, 2), so the (2,)-only guard would not protect it."""
    profile = ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.02, spectator_term=('H', 'ZZ:1,2'))
    _, datagen_model = build_fit_and_datagen_models(profile)

    weight_two = {
        label: value
        for label, value in datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)].items()
        if label.support == (1, 2)
    }
    assert len(weight_two) == 1
    label, value = next(iter(weight_two.items()))
    assert label.errorgen_type == 'H'
    assert value == pytest.approx(0.02)


def test_correlation_spectator_terms_are_built_rather_than_silently_discarded():
    """`H+S` drops `C` coefficients without warning, which would fake a crosstalk run."""
    profile = ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.01,
        spectator_term=('C', 'Z:2', 'X:2'))
    _, datagen_model = build_fit_and_datagen_models(profile)

    injected = {
        label: value
        for label, value in datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)].items()
        if label.errorgen_type == 'C'
    }
    assert len(injected) == 1
    label, value = next(iter(injected.items()))
    assert label.support == (2,)
    assert tuple(label.basis_element_labels) == ('Z', 'X')
    assert value == pytest.approx(0.01)


def test_hamiltonian_and_stochastic_terms_keep_the_original_parameterization():
    """Promoting every data-generating model to `GLNDU` would change runs that do not need it."""
    assert validation._spectator_parameterization(('H', 'Z:2')) == 'H+S'
    assert validation._spectator_parameterization(('S', 'Z:2')) == 'H+S'
    assert validation._spectator_parameterization(('C', 'Z:2', 'X:2')) == 'GLNDU'
    assert validation._spectator_parameterization(('A', 'Z:2', 'X:2')) == 'GLNDU'


def test_held_out_circuits_are_distinct_so_the_mean_is_not_reweighted():
    """A circuit reached by two germ powers is listed twice; averaging would double it."""
    profile = ValidationProfile.inclass(nqubits=3, max_lengths=(1, 2, 4), seed=20260902)
    _, held_out = build_training_design_and_held_out_circuits(profile)

    assert len(set(held_out.circuits)) == len(held_out.circuits)
    assert len(set(held_out.deepest)) == len(held_out.deepest)


def test_spectator_term_must_reach_beyond_the_gate_it_is_attached_to():
    """A term confined to the gate's own qubits is local error, not crosstalk."""
    profile = ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2), spectator_term=('H', 'ZZ:0,1'))

    with pytest.raises(ValueError, match='does not reach beyond'):
        build_fit_and_datagen_models(profile)


def test_deepest_held_out_circuits_come_from_the_largest_max_length():
    profile = ValidationProfile.inclass(nqubits=3, max_lengths=(1, 2, 4), seed=1234)
    _, held_out = build_training_design_and_held_out_circuits(profile)

    assert held_out.deepest
    assert set(held_out.deepest) <= set(held_out.circuits)
    assert len(held_out.deepest) < len(held_out.circuits)
    # The point of the subset is that it concentrates the deep circuits.
    assert (max(circuit.depth for circuit in held_out.deepest)
            == max(circuit.depth for circuit in held_out.circuits))
    assert (min(circuit.depth for circuit in held_out.deepest)
            > min(circuit.depth for circuit in held_out.circuits))


def test_line_4q_somehs_fit_model_has_59_parameters():
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2))
    fit_model, _ = build_fit_and_datagen_models(profile)
    assert fit_model.num_params == 59


def test_ring_processor_spec_adds_the_wraparound_edge_to_the_line():
    line = validation._processor_spec(4)
    ring = validation._processor_spec(4, 'ring')
    line_edges = [(0, 1), (1, 2), (2, 3)]
    for gate_name in ('Gcnot', 'Gii'):
        assert list(line.resolved_availability(gate_name)) == line_edges
        assert list(ring.resolved_availability(gate_name)) == line_edges + [(0, 3)]
    with pytest.raises(ValueError, match='at least three qubits'):
        validation._processor_spec(2, 'ring')


def test_ring_fit_models_add_one_edge_of_parameters():
    # Each edge carries 3 Gcnot and 6 Gii parameters, so the ring adds 9 to the line.
    for nqubits, num_params in ((3, 51), (4, 68)):
        profile = ValidationProfile.inclass(nqubits=nqubits, max_lengths=(1,), topology='ring')
        fit_model, _ = build_fit_and_datagen_models(profile)
        assert fit_model.num_params == num_params


def test_profile_names_omit_only_the_default_term_set():
    assert ValidationProfile.inclass(nqubits=3, max_lengths=(1,)).name == 'line_3q_inclass'
    assert ValidationProfile.inclass(
        nqubits=4, max_lengths=(1,), model_terms='allhs', topology='ring',
    ).name == 'ring_4q_inclass_allhs'
    assert ValidationProfile.outclass(
        nqubits=4, max_lengths=(1,), topology='ring',
    ).name == 'ring_4q_outclass'
    assert ValidationProfile.outclass(
        nqubits=3, max_lengths=(1,), model_terms='allhs', spectator_term=('H', 'ZZ:1,2'),
    ).name == 'line_3q_outclass_allhs'


def test_canonical_profile_names_are_their_constant_and_test_names():
    canonical = {name: value for name, value in vars(validation).items()
                 if isinstance(value, ValidationProfile)}
    assert len(canonical) == 8
    for constant_name, profile in canonical.items():
        assert profile.name == constant_name.lower()
        assert hasattr(integration.SimultaneousGSTValidationTester, f'test_{profile.name}')


@pytest.mark.parametrize('field, value', [
    ('model_terms', 'some_hs'), ('scenario', 'out_of_class_spectator'), ('topology', 'grid'),
])
def test_profiles_reject_unknown_descriptor_values(field, value):
    with pytest.raises(ValueError, match=f'Unknown {field}'):
        dataclasses.replace(validation.LINE_3Q_OUTCLASS, **{field: value})


def test_ring_3q_design_colors_and_uses_the_wraparound_edge():
    profile = ValidationProfile.inclass(nqubits=3, max_lengths=(1,), seed=1234, topology='ring')
    training_design, _ = build_training_design_and_held_out_circuits(profile)
    # An odd cycle has no proper 2-edge-coloring.
    assert training_design.color_patches == {0: [(0, 1)], 1: [(1, 2)], 2: [(0, 2)]}
    cnot_supports = {
        component.sslbls
        for circuit in training_design.all_circuits_needing_data
        for layer in circuit
        for component in layer.components
        if component.name == 'Gcnot'
    }
    assert cnot_supports == {(0, 1), (1, 2), (0, 2)}


def test_canonical_ring_profiles_match_their_line_counterparts():
    pairs = (
        (validation.LINE_3Q_INCLASS, validation.RING_3Q_INCLASS),
        (validation.LINE_3Q_OUTCLASS, validation.RING_3Q_OUTCLASS),
        (validation.LINE_4Q_INCLASS, validation.RING_4Q_INCLASS),
        (validation.LINE_4Q_OUTCLASS, validation.RING_4Q_OUTCLASS),
    )
    for line, ring in pairs:
        assert (line.topology, ring.topology) == ('line', 'ring')
        assert ring.name == line.name.replace('line_', 'ring_', 1)
        assert ring == dataclasses.replace(line, name=ring.name, topology='ring')


def test_ring_spectator_datagen_model_keeps_the_injected_term():
    _, datagen_model = build_fit_and_datagen_models(validation.RING_4Q_OUTCLASS)
    cnot_coefficients = datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)]
    spectator_coefficients = {
        label: value for label, value in cnot_coefficients.items() if label.support == (2,)
    }
    assert len(spectator_coefficients) == 1
    assert next(iter(spectator_coefficients.values())) == pytest.approx(0.01)


def test_canonical_line_4q_inclass_profile_has_max_length_four():
    assert validation.LINE_4Q_INCLASS.max_lengths == (1, 2, 4)


def test_canonical_line_4q_outclass_profile_adds_s_z2_to_the_inclass_setup():
    profile = validation.LINE_4Q_OUTCLASS
    fit_model, datagen_model = build_fit_and_datagen_models(profile)

    assert profile.max_lengths == (1, 2, 4)
    assert profile.model_terms == 'somehs'
    assert profile.scenario == 'outclass'
    assert profile.spectator_error == pytest.approx(0.01)
    assert fit_model.num_params == 59
    cnot_coefficients = datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)]
    spectator_coefficients = {
        label: value for label, value in cnot_coefficients.items()
        if label.support == (2,)
    }
    assert len(spectator_coefficients) == 1
    spectator_label, spectator_value = next(iter(spectator_coefficients.items()))
    assert spectator_label.errorgen_type == 'S'
    assert spectator_label.basis_element_labels == ('Z',)
    assert spectator_value == pytest.approx(0.01)


def test_allhs_component_models_have_expected_parameter_counts():
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2), model_terms='allhs')
    assert build_component_model(profile, 1).num_params == 18
    assert build_component_model(profile, 2).num_params == 78


def test_component_design_selection_is_reproducible():
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2), seed=1234)
    first_designs = build_component_designs(profile)
    second_designs = build_component_designs(profile)

    def serialize(design):
        return {
            'prep_fiducials': tuple(circuit.str for circuit in design.prep_fiducials),
            'meas_fiducials': tuple(circuit.str for circuit in design.meas_fiducials),
            'germs': tuple(circuit.str for circuit in design.germs),
            'circuit_lists': tuple(
                tuple(circuit.str for circuit in circuit_list)
                for circuit_list in design.circuit_lists
            ),
        }

    assert tuple(map(serialize, first_designs)) == tuple(map(serialize, second_designs))


def test_held_out_circuits_are_reproducible_and_disjoint_from_training():
    profile = ValidationProfile.inclass(nqubits=3, max_lengths=(1, 2), seed=1234)
    training_design, held_out = build_training_design_and_held_out_circuits(profile)
    _, repeated_held_out = build_training_design_and_held_out_circuits(profile)
    training_circuits = set(training_design.all_circuits_needing_data)
    assert held_out.circuits
    assert training_circuits.isdisjoint(held_out.circuits)
    assert held_out.circuits == repeated_held_out.circuits
    assert held_out.deepest == repeated_held_out.deepest


def test_finite_shot_data_are_seeded_multinomial_counts():
    profile = ValidationProfile.inclass(
        nqubits=3, max_lengths=(1,), shots=37, seed=1234)
    _, datagen_model = build_fit_and_datagen_models(profile)
    circuits = (
        Circuit([('Gxpi2', 0)], line_labels=(0, 1, 2)),
        Circuit([('Gcnot', 0, 1)], line_labels=(0, 1, 2)),
    )
    first_dataset = generate_finite_shot_data(profile, datagen_model, circuits)
    second_dataset = generate_finite_shot_data(profile, datagen_model, circuits)
    for circuit in circuits:
        assert first_dataset[circuit].total == 37
        assert all(float(count).is_integer() for count in first_dataset[circuit].counts.values())
        assert first_dataset[circuit].counts == second_dataset[circuit].counts




@pytest.mark.parametrize('scenario, changes', [
    ('inclass', {'spectator_error': 0.01}),
    ('inclass', {'spectator_term': ('H', 'Z:2')}),
    *[('outclass', {'spectator_error': value}) for value in (0.0, np.nan, np.inf, -np.inf)],
])
def test_profiles_reject_inconsistent_injections(scenario, changes):
    profile = getattr(ValidationProfile, scenario)(nqubits=3, max_lengths=(1,))
    with pytest.raises(ValueError, match='spectator'):
        dataclasses.replace(profile, **changes)

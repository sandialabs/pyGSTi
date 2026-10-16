"""Test the models, configuration and circuits assembled by validation helpers.

Checks explicit parameter counts, injected coefficient values and supports,
line/ring graph structure, profile validation and naming, circuit uniqueness,
and seeded design reproducibility. References are known answers for specified
inputs or structural/repeatability invariants; these tests do not fit GST models.
"""

import dataclasses
import itertools

import numpy as np
import pytest

import test.helpers.simultaneous_gst_validation as validation
from test.helpers.simultaneous_gst_validation import (
    CANONICAL_PROFILES,
    LINE_3Q_OUTCLASS,
    ValidationProfile,
    build_component_model,
    build_component_designs,
    build_ideal_and_datagen_models,
    build_training_design_and_held_out_circuits,
    spectator_support,
    spectator_term_text,
)


def test_somehs_component_models_have_expected_parameter_counts():
    """Check ``build_component_model`` and ``build_component_designs`` for somehs.

    Reference: the specified tied gate-term lists contain 8 one-qubit and 17
    two-qubit parameters; component designs must use labels (0,) and (0, 1).
    """
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2))
    oneq_design, twoq_design = build_component_designs(profile)
    assert build_component_model(profile, 1).num_params == 8
    assert build_component_model(profile, 2).num_params == 17
    assert oneq_design.qubit_labels == (0,)
    assert twoq_design.qubit_labels == (0, 1)


def test_line_4q_allhs_ideal_model_has_independent_placement_parameters():
    """Check ``build_ideal_and_datagen_models`` uses independent allhs placements.

    Reference: each of four qubits contributes 3 gates times 6 H/S terms, and each
    of three edges contributes 2 gates times 30 terms: 4*18 + 3*60 = 252.
    """
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2), model_terms='allhs')
    ideal_model, _ = build_ideal_and_datagen_models(profile)
    assert ideal_model.num_params == 252


def test_inclass_datagen_model_is_a_seeded_point_in_the_hypothesis_class():
    """Check ``build_ideal_and_datagen_models`` preserves the in-class parameter space.

    Reference: ideal/generator parameter counts must match, generator coefficients
    must not all vanish, and repeated construction with the same seed must return
    an identical parameter vector. No fixed random vector is hard-coded.
    """
    profile = ValidationProfile.inclass(nqubits=3, max_lengths=(1, 2))
    ideal_model, datagen_model = build_ideal_and_datagen_models(profile)
    assert ideal_model.num_params == datagen_model.num_params
    assert np.any(datagen_model.to_vector() != 0.0)
    _, repeated_datagen_model = build_ideal_and_datagen_models(profile)
    assert np.array_equal(datagen_model.to_vector(), repeated_datagen_model.to_vector())


def test_spectator_profiles_default_to_calibrated_effect():
    """Check ``ValidationProfile.outclass`` and canonical profiles use agreed defaults.

    Reference: the configuration contract is the literal S(Z:2) coefficient 0.01.
    This test checks default selection, not the empirical calibration of its effect.
    """
    default_profile = ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2))

    for profile in (default_profile, LINE_3Q_OUTCLASS, validation.LINE_4Q_OUTCLASS):
        assert (profile.spectator_term, profile.spectator_error) == (('S', 'Z:2'), 0.01)


def test_spectator_datagen_model_lies_outside_the_hypothesis_class():
    """Check ``build_ideal_and_datagen_models`` adds the requested spectator term.

    Reference: inspect CNOT coefficients for exactly one S(Z:2) term with the
    requested value 0.001 and require more generator parameters than ideal ones.
    These are construction checks, not a proof of observable distinguishability.
    """
    profile = ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.001)
    ideal_model, datagen_model = build_ideal_and_datagen_models(profile)
    assert ideal_model.num_params < datagen_model.num_params
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
    """Check ``spectator_support`` and ``spectator_term_text`` on explicit keys.

    Reference: literal support tuples and the string H(ZZ:1,2) are known answers
    for the supplied coefficient keys, including a key with no qubit suffix.
    """
    assert spectator_support(('H', 'Z:2')) == (2,)
    assert spectator_support(('H', 'ZZ:1,2')) == (1, 2)
    assert spectator_support(('S', 'ZZZ:0,1,2')) == (0, 1, 2)
    assert spectator_support(('H', 'Z')) == ()
    assert spectator_term_text(('H', 'ZZ:1,2')) == 'H(ZZ:1,2)'


def test_weight_two_spectator_term_survives_the_seeded_local_noise():
    """Check ``build_ideal_and_datagen_models`` preserves a weight-two injection.

    Reference: inspect support (1, 2) for exactly one H coefficient with the
    requested value 0.02. A guard protecting only support (2,) would miss this case.
    """
    profile = ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.02, spectator_term=('H', 'ZZ:1,2'))
    _, datagen_model = build_ideal_and_datagen_models(profile)

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
    """Check ``build_ideal_and_datagen_models`` retains a C-sector injection.

    Reference: inspect the constructed CNOT coefficients for the sole C(Z:2,X:2)
    term, with support (2,) and value 0.01. An H+S-only model would discard it.
    """
    profile = ValidationProfile.outclass(
        nqubits=3, max_lengths=(1, 2), spectator_error=0.01,
        spectator_term=('C', 'Z:2', 'X:2'))
    _, datagen_model = build_ideal_and_datagen_models(profile)

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
    """Check ``_spectator_parameterization`` against explicit sector requirements.

    Reference: known-answer mapping H/S -> H+S and C/A -> GLNDU ensures simple
    injections retain their original parameterization while other sectors fit.
    """
    assert validation._spectator_parameterization(('H', 'Z:2')) == 'H+S'
    assert validation._spectator_parameterization(('S', 'Z:2')) == 'H+S'
    assert validation._spectator_parameterization(('C', 'Z:2', 'X:2')) == 'GLNDU'
    assert validation._spectator_parameterization(('A', 'Z:2', 'X:2')) == 'GLNDU'


def test_held_out_circuits_are_distinct_so_the_mean_is_not_reweighted():
    """Check ``build_training_design_and_held_out_circuits`` removes duplicates.

    Reference: set cardinalities must equal tuple lengths for both returned lists,
    so circuits reached by multiple germ powers cannot receive extra weight.
    """
    profile = ValidationProfile.inclass(nqubits=3, max_lengths=(1, 2, 4), seed=20260902)
    _, held_out = build_training_design_and_held_out_circuits(profile)

    assert len(set(held_out.circuits)) == len(held_out.circuits)
    assert len(set(held_out.deepest)) == len(held_out.deepest)


def test_spectator_term_must_reach_beyond_the_gate_it_is_attached_to():
    """Check ``ValidationProfile.outclass`` rejects an injection local to the CNOT.

    Reference: H(ZZ:0,1) touches only the gate's own qubits and must raise the
    specified ``ValueError`` rather than describe a spectator experiment.
    """
    with pytest.raises(ValueError, match='does not reach beyond'):
        ValidationProfile.outclass(nqubits=3, max_lengths=(1, 2), spectator_term=('H', 'ZZ:0,1'))


def test_deepest_held_out_circuits_come_from_the_largest_max_length():
    """Check the deepest subset from ``build_training_design_and_held_out_circuits``.

    Reference: inline set/length and min/max circuit-depth comparisons require a
    nonempty proper held-out subset that retains the overall maximum depth and
    excludes the shortest depths. This is a structural check for the chosen input.
    """
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


def test_line_4q_somehs_ideal_model_has_59_parameters():
    """Check ``build_ideal_and_datagen_models`` on the four-qubit somehs line.

    Reference: four independent qubits contribute 8 terms each; three edges
    contribute 3 CNOT plus 6 two-qubit-idle terms each: 4*8 + 3*9 = 59.
    """
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2))
    ideal_model, _ = build_ideal_and_datagen_models(profile)
    assert ideal_model.num_params == 59


def test_ring_processor_spec_adds_the_wraparound_edge_to_the_line():
    """Check ``_processor_spec`` line/ring connectivity and invalid ring size.

    Reference: explicitly listed edges for four vertices acquire only (0, 3) on
    closing the ring, for both two-qubit gates. A two-qubit ring must raise the
    specified error because it does not meet the minimum size.
    """
    line = validation._processor_spec(4)
    ring = validation._processor_spec(4, 'ring')
    line_edges = [(0, 1), (1, 2), (2, 3)]
    for gate_name in ('Gcnot', 'Gii'):
        assert list(line.resolved_availability(gate_name)) == line_edges
        assert list(ring.resolved_availability(gate_name)) == line_edges + [(0, 3)]
    with pytest.raises(ValueError, match='at least three qubits'):
        validation._processor_spec(2, 'ring')


def test_ring_ideal_models_add_one_edge_of_parameters():
    # Each edge carries 3 Gcnot and 6 Gii parameters, so the ring adds 9 to the line.
    """Check ``build_ideal_and_datagen_models`` counts independent ring parameters.

    Reference: n qubits each contribute 8 local terms and n edges each contribute
    9 two-qubit terms, giving 3*17 = 51 and 4*17 = 68 for the specified rings.
    """
    for nqubits, num_params in ((3, 51), (4, 68)):
        profile = ValidationProfile.inclass(nqubits=nqubits, max_lengths=(1,), topology='ring')
        ideal_model, _ = build_ideal_and_datagen_models(profile)
        assert ideal_model.num_params == num_params


def test_profile_names_omit_only_the_default_term_set():
    """Check ``ValidationProfile.inclass/outclass`` generate canonical names.

    Reference: explicit expected strings encode topology, size and scenario,
    append only the nondefault allhs term set, and omit the spectator mechanism.
    """
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


@pytest.mark.parametrize('field, value', [
    ('model_terms', 'some_hs'), ('scenario', 'out_of_class_spectator'), ('topology', 'grid'),
])
def test_profiles_reject_unknown_descriptor_values(field, value):
    """Check ``ValidationProfile`` rejects unsupported descriptor spellings.

    Reference: each listed model-term/scenario/topology value is outside the
    accepted literals and must produce ``ValueError`` naming the offending field.
    """
    with pytest.raises(ValueError, match=f'Unknown {field}'):
        dataclasses.replace(validation.LINE_3Q_OUTCLASS, **{field: value})


def test_ring_3q_design_colors_and_uses_the_wraparound_edge():
    """Check ``build_training_design_and_held_out_circuits`` covers the triangle.

    Reference: an explicit three-color edge map is the known answer for this
    seed; inline circuit traversal must find CNOTs on all three triangle edges.
    An odd cycle cannot be properly edge-colored with only two colors.
    """
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


def test_ring_spectator_datagen_model_keeps_the_injected_term():
    """Check ``build_ideal_and_datagen_models`` preserves the ring spectator term.

    Reference: inspect CNOT coefficients for exactly one support-(2,) term with
    the canonical requested strength 0.01; no fitted model is involved.
    """
    _, datagen_model = build_ideal_and_datagen_models(validation.RING_4Q_OUTCLASS)
    cnot_coefficients = datagen_model.errorgen_coefficients()[('Gcnot', 0, 1)]
    spectator_coefficients = {
        label: value for label, value in cnot_coefficients.items() if label.support == (2,)
    }
    assert len(spectator_coefficients) == 1
    assert next(iter(spectator_coefficients.values())) == pytest.approx(0.01)


def test_allhs_component_models_have_expected_parameter_counts():
    """Check ``build_component_model`` counts allhs terms with tied placements.

    Reference: nonidentity Pauli strings give 6 one-qubit or 30 two-qubit H/S
    terms: 3*6 = 18 locally and 3*6 + 2*30 = 78 for the two-qubit component.
    """
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2), model_terms='allhs')
    assert build_component_model(profile, 1).num_params == 18
    assert build_component_model(profile, 2).num_params == 78


def test_component_design_selection_is_reproducible():
    """Check ``build_component_designs`` repeats seeded fiducial/germ selection.

    Reference: serialize independently returned designs inline and compare their
    fiducials, germs and ordered circuit lists. This is repeatability, not comparison
    with a separately optimized or precomputed design.
    """
    profile = ValidationProfile.inclass(nqubits=4, max_lengths=(1, 2), seed=1234)
    first_designs = build_component_designs(profile)
    second_designs = build_component_designs(profile)

    def serialize(design):
        """Extract the comparable reference for ``build_component_designs`` repeatability.

        Return fiducials, germs and circuit strings directly from each design rather
        than relying on object identity or a golden serialized file.
        """
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
    """Check ``build_training_design_and_held_out_circuits`` hold-out invariants.

    Reference: inline set intersection must be empty; repeated seeded construction
    must return identical held-out/deepest tuples and a nonempty held-out list.
    """
    profile = ValidationProfile.inclass(nqubits=3, max_lengths=(1, 2), seed=1234)
    training_design, held_out = build_training_design_and_held_out_circuits(profile)
    _, repeated_held_out = build_training_design_and_held_out_circuits(profile)
    training_circuits = set(training_design.all_circuits_needing_data)
    assert held_out.circuits
    assert training_circuits.isdisjoint(held_out.circuits)
    assert held_out.circuits == repeated_held_out.circuits
    assert held_out.deepest == repeated_held_out.deepest



@pytest.mark.parametrize('scenario, changes', [
    ('outclass', {'nqubits': 2}),
    ('inclass', {'spectator_error': 0.01}),
    ('inclass', {'spectator_term': ('H', 'Z:2')}),
    *[('outclass', {'spectator_error': value}) for value in (0.0, np.nan, np.inf, -np.inf)],
])
def test_profiles_reject_inconsistent_injections(scenario, changes):
    """Check ``ValidationProfile`` rejects inconsistent spectator settings.

    Reference: explicit invalid cases cover too few qubits, in-class injection
    settings, and zero/nonfinite out-of-class strengths; each must raise a
    ``ValueError`` mentioning the spectator configuration.
    """
    profile = getattr(ValidationProfile, scenario)(nqubits=3, max_lengths=(1,))
    with pytest.raises(ValueError, match='spectator'):
        dataclasses.replace(profile, **changes)


def test_canonical_profiles_cover_all_eight_configurations():
    """Check ``CANONICAL_PROFILES`` and the profile constructors agree.

    Reference: build the topology-by-size-by-scenario Cartesian product inline,
    require exactly its eight combinations, and reconstruct each profile using
    its constructor and the agreed maximum lengths (1, 2, 4).
    """
    expected = set(itertools.product(('line', 'ring'), (3, 4), ('inclass', 'outclass')))
    assert len(CANONICAL_PROFILES) == len(expected)
    assert {(p.topology, p.nqubits, p.scenario) for p in CANONICAL_PROFILES} == expected
    for p in CANONICAL_PROFILES:
        assert p == getattr(ValidationProfile, p.scenario)(p.nqubits, (1, 2, 4), topology=p.topology)

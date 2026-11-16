""" Detector error models from stim circuits and pyGSTi Lindblad noise models """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import numpy as np
import itertools
import stim

from collections import defaultdict
import pygsti.tools.errgenproptools as eprop
from pygsti.baseobjs import Label
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel
from pygsti.circuits import Circuit
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel, canonicalize_errorgen_layer
from pygsti.extras.dem_construction.pygsti_object_builders import stim_to_pygsti_circuit


def format_dem_stim(dem_dict, n_detectors, n_observables):
    """
    Builds a `stim.DetectorErrorModel` from a dictionary of event bit strings (detectors, then
    observables) to probabilities. Non-positive probabilities are dropped, and the result
    always declares `n_detectors` detectors and `n_observables` observables.
    """
    dem = stim.DetectorErrorModel()
    for event, prob in dem_dict.items():
        if prob > 0:
            targets = [stim.target_relative_detector_id(i) for i in range(n_detectors) if event[i] == '1']
            targets += [stim.target_logical_observable_id(i) for i in range(n_observables)
                        if event[n_detectors + i] == '1']
            dem.append('error', float(prob), targets)
    if dem.num_detectors < n_detectors:
        dem.append('detector', [], [stim.target_relative_detector_id(n_detectors - 1)])
    if dem.num_observables < n_observables:
        dem.append('logical_observable', [], [stim.target_logical_observable_id(n_observables - 1)])
    return dem


def compute_contribution_bulk(eegs, rates, det_pauli, sim):
    """ Each error generator's rate times its sensitivity (alpha) for `det_pauli`, halved. """
    alpha_paulis = eprop.bulk_alpha_pauli(eegs, sim, [det_pauli])[0] / 2  # row 0: the one Pauli
    return [r * ap for r, ap in zip(rates, alpha_paulis)]


def sort_terms_by_effect(terms, detectors):
    """ Groups error generators by the bit string of the detectors that their first Pauli flips. """
    sorted_terms = defaultdict(list)
    for eeg in terms:
        P = eeg.basis_element_labels[0]
        det_string = ''.join('0' if P.commutes(det_pauli) else '1' for det_pauli in detectors)
        sorted_terms[det_string].append(eeg)

    return dict(sorted_terms)


def split_generator(term_sorting, eoc_eeg):
    """ Turns grouped error generators into one dictionary of rates per event, and the events. """
    generator_split = []
    base_events = []
    for k, v in term_sorting.items():
        base_events.append(k)
        generator_split.append({egen: eoc_eeg[egen] for egen in v})
    return generator_split, base_events


def _real(rate):
    """ A rate as a float; raises if its imaginary part is not negligible. """
    if abs(np.imag(rate)) > 1e-10:
        raise ValueError(f'Error generator rate {rate} has a non-negligible imaginary part.')
    return float(np.real(rate))


def add_to_dem(dem, base_events, channels, dets, sim):
    """ Adds the rate of each event's channel (a dictionary of error generator rates) to `dem`. """
    for event, egens in zip(base_events, channels):
        if event != '0' * len(event):
            first_flipped_detector = event.find('1')
            det_pauli = dets[first_flipped_detector]
            egens = {k: _real(v) for k, v in egens.items()}

            if all(k.errorgen_type == 'S' for k in egens.keys()) or (
                    len(egens.keys()) == 1 and all(k.errorgen_type == 'H' for k in egens.keys())):
                dem[event] += estimate_error_rate_exact(egens)
            else:
                #properly deal with H terms by separating SCA terms from H terms
                h_terms = [t for t in egens if t.errorgen_type == 'H']
                sca_terms = [t for t in egens if t.errorgen_type != 'H']

                h_weights = []
                h_product_eegs = []

                for (eeg1, eeg2) in itertools.product(h_terms, repeat=2):
                    weight = egens[eeg1] * egens[eeg2] / 2
                    h_weights.append(weight)

                    new_eeg = LocalStimErrorgenLabel('C', (eeg1.basis_element_labels[0], eeg2.basis_element_labels[0]))
                    h_product_eegs.append(new_eeg)

                weights = h_weights + [egens[eeg] for eeg in sca_terms]
                all_terms = h_product_eegs + sca_terms
                contributions = compute_contribution_bulk(all_terms, weights, det_pauli, sim)
                dem[event] += -1 * np.sum(np.real(np.array(contributions)))

    return dem


def compose_dems(dem1, dem2):
    """ Composes two event dictionaries as independent errors. """
    new_dem = defaultdict(float)
    events = list(set(dem1.keys()).union(set(dem2.keys())))
    for k in events:
        if k in dem1:
            if k in dem2:
                new_dem[k] = dem1[k] * (1 - dem2[k]) + (1 - dem1[k]) * dem2[k]
            else:
                new_dem[k] = dem1[k]
        else:
            new_dem[k] = dem2[k]
    return new_dem


def generate_dem_higher_order(dets_as_pauli_strings, eoc_eeg, sim, zassenhaus_order=1):
    """
    Groups the error generators by the DEM event they flip, rates each event, and composes
    the events of the Zassenhaus commutator channels. Returns a dictionary of event bit
    strings (one bit per Pauli) to probabilities. Each Pauli may have ideal value +1 or -1 on
    the state held by `sim` (its sign is folded in); a Pauli with any other ideal value
    raises a `ValueError`.
    """
    folded = []
    for i, pauli in enumerate(dets_as_pauli_strings):
        expectation = sim.peek_observable_expectation(pauli)
        if expectation == 0:
            raise ValueError(f'The Pauli {i} is not deterministic in the noiseless circuit, '
                             'so it has no error model.')
        folded.append(-pauli if expectation == -1 else pauli)
    dets_as_pauli_strings = folded

    dem = defaultdict(float)
    term_sorting = sort_terms_by_effect(eoc_eeg, dets_as_pauli_strings)
    generator_split, base_events = split_generator(term_sorting, eoc_eeg)
    if not generator_split:
        return dem

    terms_zassenhaus = eprop.zassenhaus_formula(generator_split, zassenhaus_order=zassenhaus_order)
    n_leading = len(generator_split)
    leading_order_channels = terms_zassenhaus[:n_leading]
    higher_order_channels = terms_zassenhaus[n_leading:]

    dem = add_to_dem(dem, base_events, leading_order_channels, dets_as_pauli_strings, sim)
    for channel in higher_order_channels:
        #sort by dem event
        term_sorting = sort_terms_by_effect(channel, dets_as_pauli_strings)
        generator_split, new_events = split_generator(term_sorting, channel)
        terms_zassenhaus = eprop.zassenhaus_formula(generator_split, zassenhaus_order=1)

        additional_dem = add_to_dem(defaultdict(float), new_events, terms_zassenhaus,
                                    dets_as_pauli_strings, sim)
        dem = compose_dems(dem, additional_dem)

    return dem


def estimate_error_rate_exact(edict):
    """ Exact event rate for a channel of only S terms, or of a single H term. """
    if all(k.errorgen_type == 'S' for k in edict.keys()):
        contribution = (1 - np.exp(-2 * sum(edict.values()))) / 2
    else:
        (rate,) = edict.values()  # a single H term
        contribution = np.sin(rate)**2
    return contribution


def end_of_circuit_errorgen(pyg_circuit, model, virtual_to_physical=None):
    """
    The end-of-circuit error generator of an expanded circuit under a noise model, combined
    to first order in BCH.

    Parameters
    ----------
    pyg_circuit : pygsti.circuits.Circuit
        The expanded circuit from `stim_to_pygsti_circuit`, on virtual qubits.

    model : pygsti.models.OpModel
        A model of the `stabilizer` (or `stabilizer_slow`) evotype with Lindblad noise, in one of
        two placements.
        With `virtual_to_physical`, it is on the stim qubits. Without it, it is on the
        circuit's own (virtual) qubits, which allows noise that differs between rounds.

    virtual_to_physical : list of int, optional
        As returned by `stim_to_pygsti_circuit`. If None, the identity map, which requires the
        circuit's line labels to be `0..n-1`; otherwise a `ValueError` is raised.

    Returns
    -------
    dict
        `LocalStimErrorgenLabel` to rate, on the virtual qubits, for the gates' noise and for
        preparation and measurement noise. Preparation error is applied to the start state of
        every qubit and measurement error to the final measurement of every qubit; a model
        with a one-qubit preparation or measurement operator applies its error to every
        qubit, otherwise a term on a qubit applies to that qubit only.
    stim.Tableau
        The tableau of the whole circuit.
    """
    if virtual_to_physical is None:
        virtual_to_physical = list(range(len(pyg_circuit.line_labels)))
        if list(pyg_circuit.line_labels) != virtual_to_physical:
            raise ValueError('Without virtual_to_physical, the circuit\'s line labels must be 0..n-1, not '
                             f'{list(pyg_circuit.line_labels)}.')
    n_virtual = len(virtual_to_physical)
    sslbls = list(range(n_virtual))

    if str(model.evotype) not in ('stabilizer', 'stabilizer_slow'):
        raise ValueError("Only models of the 'stabilizer' or 'stabilizer_slow' evotype are supported, "
                         f"not '{model.evotype}'.")
    missing = set(virtual_to_physical) - set(model.state_space.qubit_labels)
    if missing:
        raise ValueError(f'The model has no qubits {sorted(missing)}, which the circuit uses.')

    cache = {}

    def coefficients(layer_label):
        if layer_label not in cache:
            try:
                op = model.circuit_layer_operator(layer_label)
            except (KeyError, ValueError) as err:
                raise ValueError(f'The model has no operation for the circuit layer {layer_label}.') from err
            coeffs = op.errorgen_coefficients(label_type='global')
            cache[layer_label] = {k: v for k, v in coeffs.items() if v != 0}
        return cache[layer_label]

    def gate_layer_terms(layer_label, phys_to_virt):
        relabelled = {}
        for lbl, rate in coefficients(layer_label).items():
            if not all(q in phys_to_virt for q in lbl.sslbls):
                raise ValueError(f'In layer {layer_label}, the error generator {lbl} acts outside the gates of '
                                 'the layer; noise on other qubits cannot be placed on a virtual qubit.')
            new = GlobalElementaryErrorgenLabel(lbl.errorgen_type, lbl.basis_element_labels,
                                                [phys_to_virt[q] for q in lbl.sslbls])
            relabelled[new] = relabelled.get(new, 0) + rate
        return canonicalize_errorgen_layer(relabelled, sslbls=sslbls)

    def spam_layer_terms(layer_label):
        every_qubit = model.circuit_layer_operator(layer_label).state_space.num_qubits == 1
        terms = {}
        for lbl, rate in coefficients(layer_label).items():
            if len(lbl.sslbls) != 1:
                raise ValueError(f'In layer {layer_label}, the error generator {lbl} acts on more than one qubit; '
                                 'only single-qubit preparation and measurement error is supported.')
            for v, q in enumerate(virtual_to_physical):
                if every_qubit or q == lbl.sslbls[0]:
                    new = GlobalElementaryErrorgenLabel(lbl.errorgen_type, lbl.basis_element_labels, [v])
                    terms[new] = rate
        return canonicalize_errorgen_layer(terms, sslbls=sslbls)

    prep_label, povm_label = model.complete_circuit(Circuit([], line_labels=model.state_space.qubit_labels))
    depth = pyg_circuit.depth
    tableaus = pyg_circuit.convert_to_stim_tableau_layers() if depth > 0 else []

    total = defaultdict(float)
    for lbl, rate in spam_layer_terms(povm_label).items():
        total[lbl] += rate

    later = stim.Tableau(n_virtual)  # the product of the layers after the current one
    for i in reversed(range(depth)):
        components = pyg_circuit.layer_label(i).components
        phys_to_virt = {virtual_to_physical[v]: v for c in components for v in c.qubits}
        phys_layer = Label([Label(c.name, tuple(virtual_to_physical[v] for v in c.qubits)) for c in components])
        for lbl, rate in gate_layer_terms(phys_layer, phys_to_virt).items():
            new_lbl, new_rate = lbl.propagate_error_gen_tableau(later, rate)
            total[new_lbl] += new_rate
        later = later * tableaus[i]

    for lbl, rate in spam_layer_terms(prep_label).items():
        new_lbl, new_rate = lbl.propagate_error_gen_tableau(later, rate)
        total[new_lbl] += new_rate

    return {k: v for k, v in total.items() if abs(v) > 1e-14}, later


def build_dem_end2end(circuit, model, zassenhaus_order=1, include_idles=False, include_meas_idles=False):
    """
    Builds a detector error model from a stim circuit and a pyGSTi noise model.

    Parameters
    ----------
    circuit : stim.Circuit
        A non-adaptive circuit, as for `stim_to_pygsti_circuit`: no noise instructions, only
        Z-basis resets and measurements, no feedback, and a measured qubit is reset before
        it is used again. Every detector and observable must be deterministic in the
        noiseless circuit.

    model : pygsti.models.ExplicitOpModel or LocalNoiseModel
        A model of the `stabilizer` (or `stabilizer_slow`) evotype with Lindblad noise, on the
        circuit's own stim qubit indices. Its gates are named as `stim_to_pygsti_circuit` names them (pyGSTi's
        `internalgates` names, such as `Gcnot`, `Gh` or `Gxpi`). Noise of a gate must act
        on the gate's own qubits. A preparation or measurement error generator must act on
        one qubit; when the preparation or measurement operator is a one-qubit one, its
        error applies to every qubit, otherwise a term on a qubit applies to that qubit.

    zassenhaus_order : int, optional
        The order of the Zassenhaus formula used to split the error of an event from the
        errors of other events.

    include_idles, include_meas_idles : bool, optional
        Passed to `stim_to_pygsti_circuit`: whether idle qubits get an idle gate in each
        layer, and whether they do in the measurement layers.

    Returns
    -------
    stim.DetectorErrorModel
        With the circuit's detector and observable counts. Rates are exact for events with
        only S terms and for events with a single H term. Otherwise they are first order in
        the S, C and A rates and second order in the H rates, on top of a first-order BCH
        combination of the circuit's error generators. Rates that are not positive are
        dropped. A circuit or model outside what is described here raises a `ValueError`.
    """
    pyg_circuit, detectors, observables, virtual_to_physical = stim_to_pygsti_circuit(
        circuit, include_idles, include_meas_idles)
    n_virtual = len(virtual_to_physical)
    eoc_eeg, tableau = end_of_circuit_errorgen(pyg_circuit, model, virtual_to_physical)

    sim = stim.TableauSimulator()
    sim.do_tableau(tableau, list(range(n_virtual)))

    dem_dict = generate_dem_higher_order(list(detectors) + list(observables), eoc_eeg, sim,
                                         zassenhaus_order=zassenhaus_order)
    return format_dem_stim(dem_dict, len(detectors), len(observables))

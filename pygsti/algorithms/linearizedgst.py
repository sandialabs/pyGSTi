"""
Linearized GST algorithms
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Linearized gate set tomography (Miller et al., arXiv:2605.11158).

Linearized GST learns the rates of a *sparse* error model -- a polynomial number of elementary error
generators (Hamiltonian ``H_P`` and Pauli-stochastic ``S_P`` by default) associated with the gates of a
processor -- from shallow Clifford circuits.  Every error generator is propagated through the ideal
Clifford circuit to the end (`pygsti.errorgenpropagation`), the first-order effect of each end-of-circuit
error generator on Z-type Pauli observables of the ideal stabilizer output state is computed
(`pygsti.tools.errgenproptools`), and the resulting *design matrix* ``D`` relates the observed changes in
Pauli expectation values to the error rates, ``Δ<Q> = D ε``.  Hamiltonian rates are estimated by
pseudo-inversion and stochastic rates by non-negative least squares.

Pauli-correlation (``C_{P,Q}``) and active (``A_{P,Q}``) error generators -- needed for correlated and
non-unital noise such as relaxation -- are specified as ``('C', pauli1, pauli2)`` / ``('A', pauli1, pauli2)``
(the two Paulis may have different supports).  When they are present the H/S decoupling no longer holds and a
single bounded least-squares problem (stochastic rates >= 0) is solved for all rates; :func:`rate_to_model`
then realizes the (not necessarily completely positive) generator as given.  Note that ``A_{Q,P} = -A_{P,Q}``.

Mid-circuit measurements
------------------------
Circuits may contain computational-basis mid-circuit measurements (instrument labels such as
``Label('Iz', q)``).  Each MCM is modelled with the virtual-qubit gadget of Wysocki et al.
(arXiv:2602.03938): an ideal CNOT from the measured qubit onto a fresh virtual qubit, followed by a
post-CNOT error generator on (data qubits + virtual qubit), and a deferred ideal Z measurement of the
virtual qubit.  See :mod:`pygsti.errorgenpropagation.mcmgadget`.  The MCM's error model is specified like a
gate's, under the key ``('Iz', q)``, with two-character Pauli strings ``'PV'`` (``P`` on the measured
qubit, ``V`` on the virtual qubit) or the explicit form ``'PQV:q1,q2,v'`` (``'v'`` denoting the virtual
qubit).  Observables are Z-type Paulis on the enlarged system, so ``Z`` on a virtual qubit is the MCM
outcome and ``Z_q Z_v``-type observables capture correlations between MCM outcomes and later
measurements; the data they are estimated from are the *joint* (MCM outcomes, final bitstring)
frequencies.  Because the gadget representation is redundant (the "MCM gauge"), the recommended MCM
ansatz is :func:`pygsti.errorgenpropagation.mcmgadget.fomgi_ansatz`, whose parameters are the
first-order MCM-gauge-invariant (FOMGI) quantities ``s_read`` (pure readout error), ``s_meas``,
``s_prep``, the axis rotations ``r_*`` and the unitary weaknesses ``w0..w3``.

Example
-------
>>> lindblad_coeff = {('Gxpi2', 0): {('H', 'X'): 0.01, ('S', 'X'): 0.001, ('H', 'Z:1'): 0.002},
...                   ('Gcphase', 0, 1): {('H', 'ZZ'): 0.01},
...                   ('Iz', 0): fomgi_ansatz()}                       # doctest: +SKIP
>>> params, true_rates = build_model_parameter_indexing(lindblad_coeff, num_qubits=2)  # doctest: +SKIP
>>> designs = create_design_matrix_list(circuits, params, return_info=True)             # doctest: +SKIP
>>> rates, errs = estimate_error_rates(designs, [observed_prob_dicts, None], None, params)  # doctest: +SKIP
"""

import collections as _collections
import itertools as _itertools
import warnings as _warnings

import numpy as _np
from scipy import optimize as _optimize

from pygsti.baseobjs.label import Label as _Label
from pygsti.errorgenpropagation import localstimerrorgen as _lseg
from pygsti.errorgenpropagation import mcmgadget as _mcm
from pygsti.errorgenpropagation.mcmgadget import DEFAULT_MCM_GATE_NAMES, VIRTUAL_QUBIT_TOKEN
from pygsti.tools import errgenproptools as _egpt
from pygsti.tools import internalgates as _itgs
from pygsti.tools import sdptools as _sdptools

try:
    import stim as _stim
except ImportError:  # pragma: no cover
    _stim = None
    _warnings.warn('stim is required for linearized GST; `pip install stim`.')

_SPAM_KEYS = ('prep', 'povm')

__all__ = ['ModelParameter', 'ModelParameterList', 'build_model_parameter_indexing', 'instantiate_parameter_errorgen',
           'unmodeled_gate_keys', 'PropagatedCircuit', 'propagate_parameter_errorgens', 'build_permutation_matrix',
           'generate_ideal_state', 'default_pauli_measurements', 'calculate_sensitivity_matrix_paulis',
           'calculate_sensitivity_matrix_probs', 'calculate_sensitivity_vector_paulis',
           'calculate_sensitivity_vector_probs', 'ideal_pauli_expectations', 'CircuitDesign',
           'create_circuit_design_matrix', 'create_design_matrix_list', 'create_design_matrix',
           'create_design_matrix_list_parallel_pathos', 'compute_pauli_expectations', 'probability_dicts_from_dataset',
           'unidentifiable_directions', 'predicted_error_bars', 'select_circuits', 'neyman_shot_allocation',
           'optimal_shot_allocation', 'CircuitMoments', 'create_design_moments', 'nominal_observable_covariance',
           'nominal_rates_by_type', 'DesignInformation', 'circuit_information', 'probability_information',
           'predicted_covariance', 'predicted_error_bars_nominal', 'prediction_weight', 'design_criterion',
           'AllocationResult', 'optimal_allocation', 'round_shots', 'resample_frequencies',
           'estimate_error_rates_gls', 'observed_expectation_shifts',
           'solve_linearized_gst', 'estimate_error_rates', 'estimated_rates_dict',
           'mcm_fomgi_estimates', 'make_bit_string_vector', 'int_to_bin', 'int_to_pauli', 'rate_to_model']


# ---------------------------------------------------------------------------------------------------
# Sparse error model ("ansatz") parameterization
# ---------------------------------------------------------------------------------------------------

class ModelParameter(_collections.namedtuple('ModelParameter', ['errorgen', 'key'])):
    """
    One parameter (elementary error generator rate) of a linearized-GST error model.

    Attributes
    ----------
    errorgen : LocalStimErrorgenLabel
        Canonical error generator on ``num_qubits + 1`` qubits: the data qubits (in ansatz order) followed
        by one *virtual-qubit slot*, which is the identity for all non-MCM parameters.
    key : tuple or str
        The ansatz key this parameter belongs to: a gate key such as ``('Gxpi2', 0)``, an MCM key such as
        ``('Iz', 0)`` or one of the SPAM keys ``'prep'`` / ``'povm'``.

    Instances are 2-tuples, so ``param[0]`` / ``param[1]`` also work.
    """
    __slots__ = ()

    @property
    def errorgen_type(self):
        return self.errorgen.errorgen_type

    @property
    def is_mcm(self):
        """Whether this parameter acts on the virtual qubit slot (i.e. is an MCM gadget parameter)."""
        return any(str(p)[-1] not in ('_', 'I') for p in self.errorgen.basis_element_labels)

    @property
    def is_spam(self):
        return self.key in _SPAM_KEYS

    def __repr__(self):
        return 'ModelParameter(%s, %s)' % (str(self.errorgen), str(self.key))


class ModelParameterList(list):
    """
    A list of :class:`ModelParameter` objects that remembers the ansatz it was built from.

    Attributes
    ----------
    qubit_labels : tuple
        Data qubit labels; position ``i`` of every canonical Pauli string refers to ``qubit_labels[i]``.
    num_qubits : int
    mcm_gate_names : tuple of str
    virtual_qubit_token : str
    """

    def __init__(self, params=(), qubit_labels=None, mcm_gate_names=DEFAULT_MCM_GATE_NAMES,
                 virtual_qubit_token=VIRTUAL_QUBIT_TOKEN):
        super().__init__(params)
        self.qubit_labels = tuple(qubit_labels) if qubit_labels is not None else None
        self.mcm_gate_names = tuple(mcm_gate_names)
        self.virtual_qubit_token = virtual_qubit_token

    @property
    def num_qubits(self):
        return len(self.qubit_labels)

    @property
    def keys(self):
        """The distinct ansatz keys, in order of first appearance."""
        return list(_collections.OrderedDict.fromkeys(p.key for p in self))

    def indices_for_key(self, key):
        return [i for i, p in enumerate(self) if p.key == key]

    def errorgen_types(self):
        return [p.errorgen.errorgen_type for p in self]


def _split_basis_element_label(bel, default_support):
    """Split a basis element label of the form 'XZ' or 'XZ:0,1' into (pauli_string, support_tokens)."""
    if ':' in bel:
        pauli, support = bel.split(':')
        tokens = [t.strip() for t in support.split(',')]
    else:
        pauli, tokens = bel, None
    if tokens is None:
        if default_support is None:
            raise ValueError("Basis element label '%s' needs an explicit qubit support (e.g. 'X:0')." % bel)
        tokens = list(default_support)
        if len(pauli) != len(tokens):
            raise ValueError("Pauli string '%s' has length %d but the operation it is associated with acts on %d "
                             "qubit(s) %s; use the 'P:q1,q2,...' syntax for crosstalk terms."
                             % (pauli, len(pauli), len(tokens), str(tuple(tokens))))
    if len(pauli) != len(tokens):
        raise ValueError("Pauli string '%s' does not match the number of qubits in '%s'." % (pauli, bel))
    if any(ch not in 'IXYZ' for ch in pauli):
        raise ValueError("Invalid Pauli string '%s'" % pauli)
    return pauli, tokens


def _canonical_pauli_string(pauli, tokens, qubit_labels, virtual_qubit_token, allow_virtual):
    """Build the (num_qubits + 1)-character canonical Pauli string (virtual slot last)."""
    n = len(qubit_labels)
    str_labels = {str(lbl): i for i, lbl in enumerate(qubit_labels)}
    chars = ['I'] * (n + 1)
    for ch, tok in zip(pauli, tokens):
        tok = str(tok)
        if tok == str(virtual_qubit_token) and tok not in str_labels:
            if not allow_virtual:
                raise ValueError("Only mid-circuit-measurement error generators may act on the virtual qubit '%s'."
                                 % virtual_qubit_token)
            idx = n
        elif tok in str_labels:
            idx = str_labels[tok]
        else:
            raise ValueError("Unknown qubit label '%s' (known labels: %s)" % (tok, str(qubit_labels)))
        if chars[idx] != 'I' and ch != 'I':
            raise ValueError("Qubit '%s' appears more than once in a basis element label." % tok)
        if ch != 'I':
            chars[idx] = ch
    return ''.join(chars)


def build_model_parameter_indexing(lindblad_coeff, num_qubits=None, qubit_labels=None,
                                   mcm_gate_names=DEFAULT_MCM_GATE_NAMES, virtual_qubit_token=VIRTUAL_QUBIT_TOKEN):
    """
    Enumerate the parameters of a sparse (linearized-GST) error model.

    Parameters
    ----------
    lindblad_coeff : dict
        The error model ansatz, in the format of the `lindblad_error_coeffs` argument of
        :func:`~pygsti.models.modelconstruction.create_cloud_crosstalk_model`, extended to mid-circuit
        measurements and SPAM.  Keys are:

        * gate keys ``(gate_name, q1[, q2])``, e.g. ``('Gxpi2', 0)`` or ``('Gcphase', 0, 1)``;
        * MCM keys ``(mcm_name, q)`` with ``mcm_name`` in `mcm_gate_names`, e.g. ``('Iz', 0)``;
        * the SPAM keys ``'prep'`` and ``'povm'``.

        Values are dictionaries mapping error generator specifications ``(type, bel)`` (``type`` in
        ``'H'``, ``'S'``) or ``(type, bel1, bel2)`` (``type`` in ``'C'``, ``'A'``) to rates.  A basis element
        label ``bel`` is either a plain Pauli string acting on the operation's qubits in order (for MCMs:
        the measured qubit followed by the virtual qubit, e.g. ``'IX'`` = X on the virtual qubit = pure
        readout error) or the explicit form ``'PQ:q1,q2'`` where the qubit tokens are the string forms of
        qubit labels or `virtual_qubit_token` (MCM keys only).  Plain strings are not allowed for SPAM keys
        unless they act on all qubits.

    num_qubits : int, optional
        Number of data qubits; qubit labels default to ``range(num_qubits)``.

    qubit_labels : list, optional
        Data qubit labels (their order fixes the position of each qubit in Pauli strings).

    mcm_gate_names : iterable of str, optional
        Instrument names interpreted as computational-basis mid-circuit measurements.

    virtual_qubit_token : str, optional
        Token denoting the virtual qubit in explicit-support basis element labels.

    Returns
    -------
    model_parameter_indexing : ModelParameterList
        List of :class:`ModelParameter` objects; its index defines the column ordering of design matrices.
    ideal_error_rates : list of float
        The rates given in `lindblad_coeff`, in the same order.
    """
    if qubit_labels is None:
        if num_qubits is None:
            raise ValueError("Either `num_qubits` or `qubit_labels` must be given.")
        qubit_labels = list(range(num_qubits))
    qubit_labels = list(qubit_labels)
    if num_qubits is not None and num_qubits != len(qubit_labels):
        raise ValueError("`num_qubits` disagrees with the length of `qubit_labels`.")
    mcm_gate_names = tuple(mcm_gate_names)

    params = ModelParameterList([], qubit_labels, mcm_gate_names, virtual_qubit_token)
    rates = []
    for key, errors in lindblad_coeff.items():
        if key in _SPAM_KEYS:
            default_support = [str(l) for l in qubit_labels]
            allow_virtual = False
        else:
            key = tuple(key) if not isinstance(key, str) else (key,)
            if isinstance(key[0], _Label):  # allow Label keys
                key = (key[0].name,) + tuple(key[0].sslbls if key[0].sslbls is not None else ())
            name, qubits = key[0], key[1:]
            if name in mcm_gate_names:
                if len(qubits) != 1:
                    raise ValueError("Mid-circuit measurement keys must specify a single measured qubit; "
                                     "multi-qubit instruments are treated as parallel single-qubit MCMs.")
                default_support = [str(qubits[0]), str(virtual_qubit_token)]
                allow_virtual = True
            else:
                default_support = [str(q) for q in qubits] if len(qubits) > 0 else None
                allow_virtual = False

        for error, rate in errors.items():
            typ = error[0]
            bels = list(error[1:]) if not (len(error) == 2 and isinstance(error[1], (tuple, list))) else list(error[1])
            if typ in ('H', 'S') and len(bels) != 1 or typ in ('C', 'A') and len(bels) != 2 or typ not in 'HSCA':
                raise ValueError("Invalid error generator specification %s" % str(error))
            canonical = []
            for bel in bels:
                pauli, tokens = _split_basis_element_label(bel, default_support)
                canonical.append(_canonical_pauli_string(pauli, tokens, qubit_labels, virtual_qubit_token,
                                                         allow_virtual))
            errgen = _lseg.LocalStimErrorgenLabel(typ, [_stim.PauliString(c) for c in canonical])
            params.append(ModelParameter(errgen, key))
            rates.append(rate)
    return params, rates


def instantiate_parameter_errorgen(param, num_qubits, num_virtual=0, virtual_index=None):
    """
    Instantiate a parameter's canonical error generator on the expanded (data + virtual qubits) system.

    Parameters
    ----------
    param : ModelParameter
    num_qubits : int
        Number of data qubits.
    num_virtual : int, optional
        Number of virtual qubits in the expanded circuit.
    virtual_index : int, optional
        For MCM parameters: which virtual qubit (0-based, in MCM order) the parameter's virtual slot maps to.

    Returns
    -------
    LocalStimErrorgenLabel
    """
    new_bels = []
    for p in param.errorgen.basis_element_labels:
        s = str(p)[1:].replace('_', 'I')
        assert len(s) == num_qubits + 1, "Canonical Pauli string has the wrong length"
        chars = list(s[:num_qubits]) + ['I'] * num_virtual
        if s[num_qubits] != 'I':
            if virtual_index is None:
                raise ValueError("Parameter %s acts on the virtual qubit but no virtual index was given" % str(param))
            chars[num_qubits + virtual_index] = s[num_qubits]
        new_bels.append(_stim.PauliString(''.join(chars)))
    return _lseg.LocalStimErrorgenLabel(param.errorgen.errorgen_type, new_bels,
                                        initial_label=param.errorgen.initial_label)


def _component_key(comp):
    """The ansatz key corresponding to a circuit-layer component label."""
    sslbls = comp.sslbls if comp.sslbls is not None else ()
    return (comp.name,) + tuple(sslbls)


def unmodeled_gate_keys(circuits, model_parameter_indexing):
    """
    Ansatz keys that appear in `circuits` but have no parameters in `model_parameter_indexing`.

    Useful for catching typos in an ansatz (such gates are silently treated as error-free).

    Returns
    -------
    set
    """
    keys = set(model_parameter_indexing.keys)
    mcm_names = getattr(model_parameter_indexing, 'mcm_gate_names', DEFAULT_MCM_GATE_NAMES)
    missing = set()
    for circuit in circuits:
        for layer in circuit:
            for comp in layer.components:
                if _mcm.is_mcm_label(comp, mcm_names):
                    qubits = comp.sslbls if comp.sslbls is not None else circuit.line_labels
                    for q in qubits:
                        if (comp.name, q) not in keys:
                            missing.add((comp.name, q))
                elif _component_key(comp) not in keys:
                    missing.add(_component_key(comp))
    return missing


# ---------------------------------------------------------------------------------------------------
# Error generator propagation for a circuit (with or without mid-circuit measurements)
# ---------------------------------------------------------------------------------------------------

class PropagatedCircuit(object):
    """
    Result of propagating all ansatz error generators of a circuit to its end.

    Attributes
    ----------
    circuit : Circuit
        The original circuit.
    expanded_circuit : Circuit
        The Clifford circuit with MCMs replaced by CNOTs onto virtual qubits (equals `circuit` if it has no MCMs).
    mcm_infos : list of MCMInfo
    qubit_labels : tuple
        Data qubit labels.
    num_qubits, num_mcms : int
    tableau : stim.Tableau
        Stabilizer tableau of the ideal expanded circuit.
    eoc_errorgens : list of LocalStimErrorgenLabel
        The distinct end-of-circuit error generators.
    parameter_matrix : numpy.ndarray
        Shape ``(len(eoc_errorgens), num_params)``: entry ``[k, j]`` is the (signed) coefficient with which
        parameter ``j`` contributes to end-of-circuit error generator ``k`` (the "permutation matrix").
    """

    def __init__(self, circuit, expanded_circuit, mcm_infos, qubit_labels, tableau, eoc_errorgens, parameter_matrix):
        self.circuit = circuit
        self.expanded_circuit = expanded_circuit
        self.mcm_infos = mcm_infos
        self.qubit_labels = tuple(qubit_labels)
        self.tableau = tableau
        self.eoc_errorgens = eoc_errorgens
        self.parameter_matrix = parameter_matrix

    @property
    def num_qubits(self):
        return len(self.qubit_labels)

    @property
    def num_mcms(self):
        return len(self.mcm_infos)

    @property
    def total_num_qubits(self):
        return self.num_qubits + self.num_mcms


def _resolve_qubit_labels(circuit, model_parameter_indexing, qubit_labels):
    if qubit_labels is None:
        qubit_labels = getattr(model_parameter_indexing, 'qubit_labels', None)
    if qubit_labels is None:
        qubit_labels = circuit.line_labels
    qubit_labels = tuple(qubit_labels)
    if circuit.line_labels == ('*',):
        raise ValueError("Circuits must have explicit line labels for linearized GST.")
    if set(circuit.line_labels) != set(qubit_labels):
        raise ValueError("Circuit line labels %s do not match the error model's qubit labels %s."
                         % (str(circuit.line_labels), str(qubit_labels)))
    return qubit_labels


def propagate_parameter_errorgens(circuit, model_parameter_indexing, qubit_labels=None,
                                  mcm_gate_names=None, cnot_name='Gcnot', gate_name_conversions=None):
    """
    Propagate every ansatz error generator occurring in `circuit` to the end of the circuit.

    Post-gate error generators of layer ``j`` are pushed through the ideal Clifford layers ``j+1, ...``;
    ``'prep'`` error generators are pushed through the whole circuit and ``'povm'`` error generators are not
    propagated.  Mid-circuit measurements are handled by first expanding the circuit with
    :func:`~pygsti.errorgenpropagation.mcmgadget.expand_mcm_circuit`; an MCM parameter's virtual-qubit slot
    is mapped to the virtual qubit of the MCM instance it belongs to.

    Parameters
    ----------
    circuit : Circuit
    model_parameter_indexing : ModelParameterList
        From :func:`build_model_parameter_indexing`.
    qubit_labels : tuple, optional
        Data qubit labels; defaults to those recorded in `model_parameter_indexing`.
    mcm_gate_names : iterable of str, optional
        Defaults to those recorded in `model_parameter_indexing`.
    cnot_name : str, optional
        Gate name of the ideal CNOT used in the MCM gadget (must be in `gate_name_conversions`).
    gate_name_conversions : dict, optional
        Maps gate names to `stim.Tableau` objects; defaults to
        :func:`~pygsti.tools.internalgates.standard_gatenames_stim_conversions`.

    Returns
    -------
    PropagatedCircuit
    """
    qubit_labels = _resolve_qubit_labels(circuit, model_parameter_indexing, qubit_labels)
    if mcm_gate_names is None:
        mcm_gate_names = getattr(model_parameter_indexing, 'mcm_gate_names', DEFAULT_MCM_GATE_NAMES)
    if gate_name_conversions is None:
        gate_name_conversions = _itgs.standard_gatenames_stim_conversions()
    n = len(qubit_labels)
    num_params = len(model_parameter_indexing)

    expanded, infos = _mcm.expand_mcm_circuit(circuit, mcm_gate_names, cnot_name)
    m = len(infos)
    all_labels = tuple(qubit_labels) + tuple(info.virtual_qubit for info in infos)
    label_to_index = {lbl: i for i, lbl in enumerate(all_labels)}
    # expanded circuit lines: circuit.line_labels (possibly permuted w.r.t. qubit_labels) + virtual labels
    qubit_label_conversions = {lbl: label_to_index[lbl] for lbl in expanded.line_labels}

    stim_layers = expanded.convert_to_stim_tableau_layers(gate_name_conversions=gate_name_conversions,
                                                          num_qubits=n + m,
                                                          qubit_label_conversions=qubit_label_conversions)
    depth = len(stim_layers)
    # prop_from[k] = product of layers k..depth-1 (prop_from[depth] = identity)
    prop_from = [None] * (depth + 1)
    prop_from[depth] = _stim.Tableau(n + m)
    for k in reversed(range(depth)):
        prop_from[k] = prop_from[k + 1] * stim_layers[k]
    tableau = prop_from[0]

    # --- assemble per-layer (parameter index, instantiated error generator) pairs
    key_to_indices = _collections.defaultdict(list)
    for j, p in enumerate(model_parameter_indexing):
        key_to_indices[p.key].append(j)

    layers = []  # list of (propagation tableau, [(param_idx, errorgen), ...])
    layers.append((prop_from[0], [(j, instantiate_parameter_errorgen(model_parameter_indexing[j], n, m))
                                  for j in key_to_indices.get('prep', [])]))
    mcm_iter = iter(infos)
    for layer_index, layer_lbl in enumerate(circuit):
        entries = []
        for comp in layer_lbl.components:
            if _mcm.is_mcm_label(comp, mcm_gate_names):
                qubits = comp.sslbls if comp.sslbls is not None else circuit.line_labels
                for q in qubits:
                    info = next(mcm_iter)
                    for j in key_to_indices.get((comp.name, q), []):
                        entries.append((j, instantiate_parameter_errorgen(model_parameter_indexing[j], n, m,
                                                                          virtual_index=info.mcm_index)))
            else:
                for j in key_to_indices.get(_component_key(comp), []):
                    entries.append((j, instantiate_parameter_errorgen(model_parameter_indexing[j], n, m)))
        layers.append((prop_from[layer_index + 1], entries))
    layers.append((prop_from[depth], [(j, instantiate_parameter_errorgen(model_parameter_indexing[j], n, m))
                                      for j in key_to_indices.get('povm', [])]))

    # --- propagate and accumulate
    eoc_index = _collections.OrderedDict()
    contributions = []  # (eoc_idx, param_idx, coefficient)
    for prop_tableau, entries in layers:
        for j, errgen in entries:
            propagated, coeff = errgen.propagate_error_gen_tableau(prop_tableau, 1.0)
            if propagated not in eoc_index:
                eoc_index[propagated] = len(eoc_index)
            contributions.append((eoc_index[propagated], j, coeff))
    parameter_matrix = _np.zeros((len(eoc_index), num_params))
    for k, j, coeff in contributions:
        parameter_matrix[k, j] += coeff
    return PropagatedCircuit(circuit, expanded, infos, qubit_labels, tableau, list(eoc_index.keys()), parameter_matrix)


def build_permutation_matrix(circuit, model_parameter_indexing, **kwargs):
    """
    The matrix mapping model parameters to end-of-circuit error generators (see :func:`propagate_parameter_errorgens`).

    Returns
    -------
    permutation_matrix : numpy.ndarray
    eoc_errorgens : list of LocalStimErrorgenLabel
    """
    prop = propagate_parameter_errorgens(circuit, model_parameter_indexing, **kwargs)
    return prop.parameter_matrix, prop.eoc_errorgens


def generate_ideal_state(circ, mcm_gate_names=DEFAULT_MCM_GATE_NAMES, cnot_name='Gcnot', gate_name_conversions=None):
    """
    The ideal output stabilizer state (as a `stim.Tableau`) of a circuit, after expanding any mid-circuit measurements.

    Parameters
    ----------
    circ : Circuit
    mcm_gate_names : iterable of str, optional
    cnot_name : str, optional
    gate_name_conversions : dict, optional

    Returns
    -------
    stim.Tableau
    """
    expanded, _ = _mcm.expand_mcm_circuit(circ, mcm_gate_names, cnot_name)
    if gate_name_conversions is None:
        gate_name_conversions = _itgs.standard_gatenames_stim_conversions()
    conv = {lbl: i for i, lbl in enumerate(expanded.line_labels)}
    return expanded.convert_to_stim_tableau(gate_name_conversions=gate_name_conversions,
                                            num_qubits=len(expanded.line_labels), qubit_label_conversions=conv)


# ---------------------------------------------------------------------------------------------------
# Observables and sensitivities
# ---------------------------------------------------------------------------------------------------

def default_pauli_measurements(num_qubits, max_weight=2, num_mcms=0):
    """
    All Z-type Pauli observables of weight ``1..max_weight`` on ``num_qubits + num_mcms`` qubits.

    Parameters
    ----------
    num_qubits : int
        Number of data qubits.
    max_weight : int, optional
    num_mcms : int, optional
        Number of virtual (MCM-outcome) qubits appended after the data qubits.

    Returns
    -------
    list of stim.PauliString
        Ordered by weight, then lexicographically by support.
    """
    total = num_qubits + num_mcms
    paulis = []
    for w in range(1, min(max_weight, total) + 1):
        for support in _itertools.combinations(range(total), w):
            chars = ['I'] * total
            for i in support:
                chars[i] = 'Z'
            paulis.append(_stim.PauliString(''.join(chars)))
    return paulis


def _pad_pauli(pauli, total_num_qubits):
    pauli = pauli if isinstance(pauli, _stim.PauliString) else _stim.PauliString(str(pauli))
    if len(pauli) == total_num_qubits:
        return pauli
    if len(pauli) > total_num_qubits:
        raise ValueError("Pauli observable %s acts on more qubits than the (expanded) circuit has (%d)"
                         % (str(pauli), total_num_qubits))
    return _stim.PauliString(str(pauli) + '_' * (total_num_qubits - len(pauli)))


def _resolve_pauli_measurements(pauli_measurements, num_qubits, num_mcms, max_pauli_weight):
    total = num_qubits + num_mcms
    if pauli_measurements is None:
        return default_pauli_measurements(num_qubits, max_pauli_weight, num_mcms)
    if callable(pauli_measurements):
        pauli_measurements = pauli_measurements(num_qubits, num_mcms)
    return [_pad_pauli(p, total) for p in pauli_measurements]


def calculate_sensitivity_matrix_paulis(ideal_state, errorgen_list, num_qubits=None, pauli_subset=None):
    """
    First-order sensitivities of Pauli expectation values to end-of-circuit error generators.

    Parameters
    ----------
    ideal_state : stim.Tableau
    errorgen_list : list of LocalStimErrorgenLabel
    num_qubits : int, optional
        Total number of qubits of `ideal_state` (inferred if None).
    pauli_subset : list of stim.PauliString, optional
        Observables; defaults to all Z-type Paulis of weight <= 2.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(paulis), len(errorgen_list))``.
    """
    if num_qubits is None:
        num_qubits = len(ideal_state)
    paulis = _resolve_pauli_measurements(pauli_subset, num_qubits, 0, 2)
    if len(errorgen_list) == 0:
        return _np.zeros((len(paulis), 0))
    mat = _np.asarray(_egpt.bulk_alpha_pauli(errorgen_list, ideal_state, paulis))
    if _np.iscomplexobj(mat):
        assert _np.allclose(mat.imag, 0), "Unexpected complex sensitivities"
        mat = mat.real
    return mat.reshape((len(paulis), len(errorgen_list)))


def calculate_sensitivity_matrix_probs(ideal_state, errorgen_list, num_qubits=None):
    """
    First-order sensitivities of all computational-basis outcome probabilities to end-of-circuit error generators.

    Returns
    -------
    numpy.ndarray
        Shape ``(2**num_qubits, len(errorgen_list))``, rows ordered by :func:`int_to_bin`.
    """
    if num_qubits is None:
        num_qubits = len(ideal_state)
    bitstrings = [int_to_bin(i, num_qubits) for i in range(2**num_qubits)]
    if len(errorgen_list) == 0:
        return _np.zeros((len(bitstrings), 0))
    scale = 1 / 2**_egpt.random_support(ideal_state)
    mat = scale * _np.asarray(_egpt.bulk_alpha(errorgen_list, ideal_state, bitstrings))
    if _np.iscomplexobj(mat):
        assert _np.allclose(mat.imag, 0), "Unexpected complex sensitivities"
        mat = mat.real
    return mat.reshape((len(bitstrings), len(errorgen_list)))


def calculate_sensitivity_vector_paulis(ideal_state, error_gen_list, pauli):
    """Sensitivity of a single Pauli expectation to each error generator (column vector)."""
    return calculate_sensitivity_matrix_paulis(ideal_state, error_gen_list, pauli_subset=[pauli]).T


def calculate_sensitivity_vector_probs(ideal_state, error_gen_list, bit_string):
    """Sensitivity of a single bitstring probability to each error generator (column vector)."""
    scale = 1 / 2**_egpt.random_support(ideal_state)
    vec = _np.zeros((len(error_gen_list), 1))
    for idx, error in enumerate(error_gen_list):
        vec[idx, 0] = scale * _egpt.alpha(error, ideal_state, bit_string)
    return vec


def ideal_pauli_expectations(tableau, paulis):
    """
    Expectation values (0 or +/-1) of Pauli observables in the stabilizer state given by `tableau`.

    Returns
    -------
    numpy.ndarray
    """
    return _np.array([_egpt.stabilizer_pauli_expectation(tableau, p) for p in paulis], dtype=float)


class CircuitDesign(_collections.namedtuple('CircuitDesign', ['design_matrix', 'paulis', 'ideal_expectations',
                                                              'num_qubits', 'num_mcms', 'circuit'])):
    """
    The design matrix of a single circuit together with the observables it refers to.

    Attributes
    ----------
    design_matrix : numpy.ndarray
        Shape ``(len(paulis), num_params)`` -- or ``(2**(num_qubits+num_mcms), num_params)`` for bitstring
        probabilities, in which case `paulis` is None.
    paulis : list of stim.PauliString
        Z-type observables on the expanded (data + virtual) system, one per row.
    ideal_expectations : numpy.ndarray
        Ideal (error-free) expectation values of `paulis` (or ideal probabilities).
    num_qubits : int
    num_mcms : int
    circuit : Circuit
    """
    __slots__ = ()

    @property
    def shape(self):
        return self.design_matrix.shape

    def __array__(self, dtype=None, copy=None):
        return _np.asarray(self.design_matrix, dtype=dtype)


def create_circuit_design_matrix(circuit, model_parameter_indexing, usePaulis=True, pauli_measurements=None,
                                 max_pauli_weight=2, qubit_labels=None, mcm_gate_names=None, cnot_name='Gcnot',
                                 gate_name_conversions=None, return_info=False, include_spam=None):
    """
    Compute the linearized-GST design matrix of a single circuit.

    Row ``r`` / column ``j`` of the design matrix is the first-order sensitivity of observable ``r`` to model
    parameter ``j``:  ``Δ<Q_r> = sum_j D[r, j] ε_j``.

    Parameters
    ----------
    circuit : Circuit
        A Clifford circuit, possibly containing mid-circuit measurements (instrument labels).

    model_parameter_indexing : ModelParameterList
        The error model parameters (from :func:`build_model_parameter_indexing`).

    usePaulis : bool, optional
        If True (default) rows are Z-type Pauli expectation values; if False rows are all computational-basis
        outcome probabilities of the expanded circuit.

    pauli_measurements : list of stim.PauliString or callable, optional
        The observables (only used when `usePaulis` is True).  If None, all Z-type Paulis of weight up to
        `max_pauli_weight` on the expanded (data + virtual qubit) system are used.  A list is padded with
        identities on the virtual qubits; a callable ``f(num_qubits, num_mcms)`` may return a per-circuit list.

    max_pauli_weight : int, optional
        Maximum weight of the default Z-type observables.

    qubit_labels : tuple, optional
        Data qubit labels (default: those of `model_parameter_indexing`).

    mcm_gate_names : iterable of str, optional
        Instrument names treated as mid-circuit measurements (default: those of `model_parameter_indexing`).

    cnot_name : str, optional
        Name of the ideal CNOT used in the MCM gadget.

    gate_name_conversions : dict, optional
        Gate name to `stim.Tableau` conversions (default: pyGSTi's standard Clifford gate names).

    return_info : bool, optional
        If True return a :class:`CircuitDesign` (design matrix plus observables and ideal expectation values)
        instead of a bare array.

    include_spam : None
        Deprecated/ignored: SPAM parameters are included whenever the ansatz contains ``'prep'`` / ``'povm'`` keys.

    Returns
    -------
    numpy.ndarray or CircuitDesign
    """
    if hasattr(model_parameter_indexing, 'state_space') or hasattr(model_parameter_indexing, 'probabilities'):
        raise TypeError("The second argument must be the model parameter list from `build_model_parameter_indexing`; "
                        "pyGSTi models are no longer needed to construct design matrices.")
    prop = propagate_parameter_errorgens(circuit, model_parameter_indexing, qubit_labels, mcm_gate_names,
                                         cnot_name, gate_name_conversions)
    n, m = prop.num_qubits, prop.num_mcms
    if usePaulis:
        paulis = _resolve_pauli_measurements(pauli_measurements, n, m, max_pauli_weight)
        sensitivity = calculate_sensitivity_matrix_paulis(prop.tableau, prop.eoc_errorgens, n + m, paulis)
        ideal = ideal_pauli_expectations(prop.tableau, paulis)
    else:
        paulis = None
        sensitivity = calculate_sensitivity_matrix_probs(prop.tableau, prop.eoc_errorgens, n + m)
        ideal = _np.array([_egpt.stabilizer_probability(prop.tableau, int_to_bin(i, n + m)) for i in range(2**(n + m))])
    design_matrix = sensitivity @ prop.parameter_matrix
    if return_info:
        return CircuitDesign(design_matrix, paulis, ideal, n, m, circuit)
    return design_matrix


def create_design_matrix_list(circuit_list, model_parameter_indexing, usePaulis=True, pauli_measurements=None,
                              max_pauli_weight=2, show_progress_bar=False, return_info=False, **kwargs):
    """
    Design matrices for a list of circuits (one entry per circuit; see :func:`create_circuit_design_matrix`).

    Parameters
    ----------
    circuit_list : list of Circuit
    model_parameter_indexing : ModelParameterList
    usePaulis, pauli_measurements, max_pauli_weight, return_info, **kwargs
        Passed to :func:`create_circuit_design_matrix`.
    show_progress_bar : bool, optional
        Show a `tqdm` progress bar.

    Returns
    -------
    list of numpy.ndarray or list of CircuitDesign
    """
    iterator = circuit_list
    if show_progress_bar:
        from tqdm import tqdm
        iterator = tqdm(circuit_list, total=len(circuit_list), ascii=True,
                        desc="Processing circuits into design matrix")
    return [create_circuit_design_matrix(c, model_parameter_indexing, usePaulis=usePaulis,
                                         pauli_measurements=pauli_measurements, max_pauli_weight=max_pauli_weight,
                                         return_info=return_info, **kwargs) for c in iterator]


def create_design_matrix(circuit_list, model_parameter_indexing, **kwargs):
    """
    The stacked design matrix of a list of circuits.

    Returns
    -------
    numpy.ndarray
    """
    kwargs.pop('return_info', None)
    return _np.vstack(create_design_matrix_list(circuit_list, model_parameter_indexing, return_info=False, **kwargs))


def create_design_matrix_list_parallel_pathos(circuit_list, model_parameter_indexing, usePaulis=True,
                                              pauli_measurements=None, max_pauli_weight=2, show_progress_bar=False,
                                              do_parallel=False, num_workers=1, return_info=False, **kwargs):
    """
    Design matrices for a list of circuits, optionally computed in parallel with `pathos`.

    Parameters
    ----------
    circuit_list : list of Circuit
    model_parameter_indexing : ModelParameterList
    usePaulis, pauli_measurements, max_pauli_weight, return_info, **kwargs
        Passed to :func:`create_circuit_design_matrix`.
    show_progress_bar : bool, optional
    do_parallel : bool, optional
        Whether to use a `pathos` process pool.
    num_workers : int, optional
        Number of worker processes (and batches).

    Returns
    -------
    list of numpy.ndarray or list of CircuitDesign
    """
    common = dict(usePaulis=usePaulis, pauli_measurements=pauli_measurements, max_pauli_weight=max_pauli_weight,
                  return_info=return_info, **kwargs)
    if not do_parallel:
        return create_design_matrix_list(circuit_list, model_parameter_indexing, show_progress_bar=show_progress_bar,
                                         **common)

    from functools import partial
    from pathos.multiprocessing import ProcessingPool as Pool

    batch_size = len(circuit_list) // num_workers + 1
    batches = [circuit_list[k * batch_size:min(len(circuit_list), (k + 1) * batch_size)] for k in range(num_workers)]
    task_fn = partial(create_design_matrix_list, model_parameter_indexing=model_parameter_indexing,
                      show_progress_bar=False, **common)
    with Pool(nodes=num_workers) as pool:
        if show_progress_bar:
            from tqdm import tqdm
            nested = list(tqdm(pool.imap(task_fn, batches), ascii=True, total=len(batches),
                               desc="Processing circuit batches into design matrix"))
        else:
            nested = pool.map(task_fn, batches)
    return list(_itertools.chain.from_iterable(nested))


# ---------------------------------------------------------------------------------------------------
# Data processing: Pauli expectation values from (joint) outcome probabilities
# ---------------------------------------------------------------------------------------------------

def _calculate_Ztype_pauli_exp(pauli, probs):
    """
    Expectation value of a Z-type Pauli from an outcome-probability dictionary.

    Parameters
    ----------
    pauli : stim.PauliString or str
        Z-type Pauli on the expanded (data + virtual qubit) system.
    probs : dict
        Maps outcome labels (pyGSTi outcome tuples, possibly with MCM outcomes, or bitstrings) to probabilities.

    Returns
    -------
    float
    """
    pauli = pauli if isinstance(pauli, _stim.PauliString) else _stim.PauliString(str(pauli))
    if len(pauli.pauli_indices(included_paulis='XY')) > 0:
        raise ValueError("Only Z-type Pauli observables can be estimated from computational-basis data.")
    indices = pauli.pauli_indices(included_paulis='Z')
    exp_value = 0.0
    for outcome, p in probs.items():
        bits = _mcm.joint_outcome_to_bitstring(outcome)
        if len(bits) < len(pauli):
            raise ValueError("Outcome %s has fewer bits than the observable %s" % (str(outcome), str(pauli)))
        parity = sum(1 for i in indices if bits[i] == '1') % 2
        exp_value += -p if parity else p
    return exp_value


def _bulk_Ztype_pauli_exp(paulis, probs):
    """Expectation values of several Z-type Paulis from an outcome-probability dictionary."""
    return _np.array([_calculate_Ztype_pauli_exp(p, probs) for p in paulis], dtype=float)


def compute_pauli_expectations(prob_dict, paulis):
    """
    Estimate Z-type Pauli expectation values from (joint) outcome probabilities or frequencies.

    Parameters
    ----------
    prob_dict : dict
        Maps outcome labels to probabilities.  Outcome labels are pyGSTi outcome tuples (e.g. ``('0', '01')``
        for a circuit with one MCM), plain bitstrings or length-1 tuples of bitstrings.
    paulis : list of stim.PauliString
        Z-type observables on the expanded system (data bits followed by MCM outcome bits).

    Returns
    -------
    numpy.ndarray
    """
    return _bulk_Ztype_pauli_exp(paulis, prob_dict)


def probability_dicts_from_dataset(dataset, circuits):
    """
    Outcome-frequency dictionaries of the given circuits in a :class:`DataSet`.

    Returns
    -------
    list of dict
    """
    return [dict(dataset[c].fractions) for c in circuits]


def _resample_probability_distribution(prob_distribution, shots, rng=None):
    """
    Multinomially resample an outcome distribution.

    Parameters
    ----------
    prob_distribution : dict
        Maps outcomes to probabilities (small negative values are clipped to 0).
    shots : int
    rng : numpy.random.Generator, optional

    Returns
    -------
    dict
        Maps outcomes to resampled frequencies.
    """
    rng = _np.random.default_rng() if rng is None else rng
    keys = list(prob_distribution.keys())
    p = _np.clip(_np.array([prob_distribution[k] for k in keys], dtype=float), 0.0, None)
    p = p / p.sum()
    freqs = rng.multinomial(shots, p) / shots
    return {k: f for k, f in zip(keys, freqs)}


# ---------------------------------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------------------------------

def _as_design_arrays(design_matrices):
    """Split a list of arrays / CircuitDesign objects into (arrays, designs-or-None)."""
    arrays, infos = [], []
    for d in design_matrices:
        if isinstance(d, CircuitDesign):
            arrays.append(d.design_matrix)
            infos.append(d)
        else:
            arrays.append(_np.asarray(d))
            infos.append(None)
    return arrays, infos


def unidentifiable_directions(design_matrices, model_parameter_indexing, tol=1e-10):
    """
    Parameter combinations that the given circuits and observables cannot distinguish.

    These are the null-space directions of the stacked design matrix.  A non-empty result means the
    design matrix is rank deficient, either because the error model has a genuine gauge freedom or because the
    circuits are not diverse enough (e.g. if every layer contains a gate on every qubit, crosstalk from "the
    gate on qubit A" cannot be told apart from crosstalk from "the gate on qubit B").  In the latter case adding
    circuits with different structure restores identifiability.

    Parameters
    ----------
    design_matrices : list of CircuitDesign or numpy.ndarray, or a single 2D numpy.ndarray
    model_parameter_indexing : ModelParameterList
    tol : float, optional
        Relative singular-value cutoff passed to `scipy.linalg.null_space`.

    Returns
    -------
    list of dict
        One dictionary per null-space direction, mapping :class:`ModelParameter` objects to coefficients
        (scaled so the largest absolute coefficient is 1; entries below ``1e-8`` are dropped).  An empty list
        means the design matrix has full column rank.
    """
    if isinstance(design_matrices, _np.ndarray) and design_matrices.ndim == 2:
        design_matrix = design_matrices
    else:
        design_matrix = _np.vstack(_as_design_arrays(design_matrices)[0])
    P = len(model_parameter_indexing)
    if design_matrix.shape[1] != P:
        raise ValueError("Design matrix has %d columns but %d parameters were given"
                         % (design_matrix.shape[1], P))
    # economy SVD (a full one would allocate a rows x rows matrix; stacked designs can have 10^5 rows); padding
    # with zero rows guarantees that Vt spans all of parameter space when there are fewer rows than parameters
    if design_matrix.shape[0] < P:
        design_matrix = _np.vstack([design_matrix, _np.zeros((P - design_matrix.shape[0], P))])
    _, s, Vt = _np.linalg.svd(design_matrix, full_matrices=False)
    null_vectors = Vt[s <= tol * max(s[0], 1e-300)]
    directions = []
    for vec in null_vectors:
        vec = vec / vec[_np.argmax(_np.abs(vec))]
        directions.append(_collections.OrderedDict((model_parameter_indexing[j], float(x))
                                                   for j, x in enumerate(vec) if abs(x) > 1e-8))
    return directions


# ---------------------------------------------------------------------------------------------------
# Experiment design: choosing circuits
# ---------------------------------------------------------------------------------------------------

def _ideal_observable_covariance(design):
    """
    Leading-order covariance (per shot) of the observed Pauli expectation values of one circuit.

    For observables with ideal value 0 the covariance is that of the ideal stabilizer state,
    ``<P_a P_b> - <P_a><P_b>`` (observables from the same shots are correlated, e.g. ``Z_0`` and ``Z_1`` are
    perfectly correlated when ``Z_0 Z_1`` is a stabilizer).  Observables with ideal value ±1 have a variance of
    the order of the error rates; the worst case ``1`` is used for them instead (uncorrelated).
    """
    ideal = design.ideal_expectations
    cov = _np.eye(len(ideal))
    zero = _np.nonzero(_np.abs(ideal) < 0.5)[0]
    if len(zero) > 1:
        tableau = generate_ideal_state(design.circuit)
        for ia, a in enumerate(zero):
            for b in zero[ia + 1:]:
                c = _egpt.stabilizer_pauli_expectation(tableau, design.paulis[a] * design.paulis[b])
                cov[a, b] = cov[b, a] = c
    return cov


def predicted_error_bars(design_matrices, shots=1, tol=1e-10):
    """
    Predicted standard errors of the error-rate estimates for a set of circuits, before any data is taken.

    The least-squares estimate ``ε = D⁺ Δ<Q>`` has covariance ``D⁺ Σ D⁺ᵀ``, where ``Σ`` is the covariance of
    the observed Pauli expectation values, which is ``(1/shots)`` times the covariance of the observables in the
    output state.  For :class:`CircuitDesign` inputs this covariance is evaluated in the ideal stabilizer
    state -- this is exact to leading order for observables with ideal value 0 (and hence for the Hamiltonian
    rates), and for observables with ideal value ±1, whose true variance is of the order of the error rates and
    hence much smaller, the worst case ``1/shots`` is used (so the error bars of the stochastic rates are
    conservative upper bounds).  For bare design-matrix arrays all observables are taken to be independent
    with variance ``1/shots``.  Only shot noise is considered; the systematic error of the linearization is not.

    Parameters
    ----------
    design_matrices : list of CircuitDesign or numpy.ndarray, or a single 2D numpy.ndarray
    shots : int, float, or array-like, optional
        Number of shots per circuit -- either one number used for every circuit, or a sequence with one entry
        per circuit (so that an unequal, e.g. optimized, shot allocation can be evaluated; see
        :func:`neyman_shot_allocation` and :func:`optimal_shot_allocation`).
    tol : float, optional
        Relative singular-value cutoff of ``D`` below which a direction is considered unidentifiable.

    Returns
    -------
    numpy.ndarray
        One predicted standard error per model parameter; ``inf`` for parameters with a component along an
        unidentifiable direction (i.e. when the stacked design matrix is rank deficient).
    """
    if isinstance(design_matrices, _np.ndarray) and design_matrices.ndim == 2:
        arrays, infos = [design_matrices], [None]
    else:
        arrays, infos = _as_design_arrays(design_matrices)
    covariances = [_ideal_observable_covariance(info) if _has_circuit_info(info) else None for info in infos]
    shots_per_circuit = _np.broadcast_to(_np.asarray(shots, dtype=float), (len(arrays),))
    if _np.any(shots_per_circuit <= 0):
        raise ValueError("shots must be positive for every circuit")
    return _np.sqrt(_predicted_variances(arrays, covariances, tol, shots=shots_per_circuit))


def _has_circuit_info(info):
    return info is not None and info.paulis is not None and info.circuit is not None


def _predicted_variances(arrays, covariances, tol=1e-10, shots=None):
    """
    Per-parameter variance of the least-squares estimate; `covariances[i]` may be None (identity).

    `shots`, if given, is a per-circuit array of shot counts (default: unit shots for every circuit, i.e. the
    caller is expected to divide by a single scalar shot count itself, as :func:`select_circuits` does).
    """
    design_matrix = _np.vstack(arrays)
    U, s, Vt = _np.linalg.svd(design_matrix, full_matrices=False)
    identifiable = s > tol * max(s[0], 1e-300)
    pinv = (Vt[identifiable].T / s[identifiable]) @ U[:, identifiable].T      # rows: parameters, cols: observables
    variance = _np.zeros(design_matrix.shape[1])
    start = 0
    for c, (arr, cov) in enumerate(zip(arrays, covariances)):
        block = pinv[:, start:start + arr.shape[0]]
        start += arr.shape[0]
        contribution = (block ** 2).sum(axis=1) if cov is None else _np.einsum('ia,ab,ib->i', block, cov, block)
        variance += contribution if shots is None else contribution / shots[c]
    # a parameter is unidentifiable if it has a component outside the row space of the design matrix
    variance[1.0 - (Vt[identifiable] ** 2).sum(axis=0) > tol] = _np.inf
    return variance


_SELECTION_CRITERIA = ('A', 'D', 'E', 'G')


def _selection_scores(criterion, M_inv, arrays, sign):
    """
    New value of the selection criterion (larger is better) when each of the circuits with design matrices
    `arrays` (all of the same shape) is added (sign=+1) to / removed (sign=-1) from the information matrix.

    With ``W = D M⁻¹`` and ``B = D M⁻¹ Dᵀ`` the Woodbury identity gives ``(M ± DᵀD)⁻¹ = M⁻¹ ∓ Wᵀ (I ± B)⁻¹ W``;
    all candidates are scored in one batched computation.  A removal that makes M singular scores ``-inf``.
    """
    W = arrays @ M_inv                                   # (n, k, P)
    K = _np.eye(arrays.shape[1]) + sign * (W @ arrays.transpose(0, 2, 1))
    if criterion == 'D':   # log det(M ± DᵀD) = log det M + log det K
        signs, logdets = _np.linalg.slogdet(K)
        return _np.where(signs > 0, logdets, -_np.inf)
    if sign > 0:           # K = I + B is positive definite
        valid = _np.ones(len(K), dtype=bool)
    else:
        evals = _np.linalg.eigvalsh(K)
        valid = evals[:, 0] > 1e-12 * _np.maximum(evals[:, -1], 1e-300)
    K_inv = _np.zeros_like(K)
    K_inv[valid] = _np.linalg.inv(K[valid])
    if criterion == 'A':   # trace(Wᵀ K⁻¹ W) = sum(K⁻¹ ∘ W Wᵀ)
        out = -(_np.trace(M_inv) - sign * (K_inv * (W @ W.transpose(0, 2, 1))).sum(axis=(1, 2)))
    else:                  # 'G': diag(Wᵀ K⁻¹ W)
        out = -_np.max(_np.diag(M_inv)[None, :] - sign * (W * (K_inv @ W)).sum(axis=1), axis=1)
    out[~valid] = -_np.inf
    return out


def select_circuits(design_matrices, num_circuits=None, max_error_bar=None, shots=1, criterion='A',
                    method='forward', ridge=1e-6, include=None, show_progress_bar=False):
    """
    Choose a subset of circuits whose design matrix is (nearly) as informative as that of all of them.

    The error-rate estimates are least-squares solutions, so their covariance is proportional to
    ``(DᵀD)⁻¹`` with ``D`` the stacked design matrix of the chosen circuits (see :func:`predicted_error_bars`).
    This function greedily optimizes a scalar figure of merit of that covariance (a classical
    optimal-experiment-design criterion), either by adding circuits one at a time (`method='forward'`) or
    by removing them one at a time (`method='backward'`).  Candidates can be any set of circuits -- typically a
    large pool of random circuits, from which a much smaller set with the same (or, at equal total shot budget,
    better) precision is selected.

    Full column rank of ``D`` is only a prerequisite: it says that the estimate is finite, not that it is
    precise.  Of the criteria below, ``'A'`` (the default) most directly targets the precision of the
    estimates; ``'E'`` only improves the single least-constrained parameter combination and is the slowest.

    Parameters
    ----------
    design_matrices : list of CircuitDesign or numpy.ndarray
        The candidate circuits' design matrices (one per circuit).
    num_circuits : int, optional
        Number of circuits to select.  Required unless `max_error_bar` is given.
    max_error_bar : float, optional
        Stop as soon as every predicted error bar (:func:`predicted_error_bars` with `shots` shots) is below
        this value (`'forward'`), or stop removing circuits before it would be exceeded (`'backward'`).
        If `num_circuits` is also given it acts as a cap on the number of circuits (forward; a warning is
        issued if the target is not reached within the cap) or as a floor (backward).  A `ValueError` is
        raised if the target cannot be reached with all candidates.
    shots : int or float, optional
        Shots per circuit used for `max_error_bar`.
    criterion : {'A', 'D', 'E', 'G'}, optional
        ``'A'``: minimize the sum of the predicted variances, ``trace((DᵀD)⁻¹)``;
        ``'D'``: maximize ``log det(DᵀD)`` (the volume of the confidence ellipsoid);
        ``'E'``: maximize the smallest singular value of ``D``;
        ``'G'``: minimize the largest predicted variance.
    method : {'forward', 'backward'}, optional
        Greedy addition starting from no circuits, or greedy elimination starting from all candidates.
    ridge : float, optional
        Small multiple of the identity added to ``DᵀD`` so that the criteria are finite before the design
        has full column rank (then the criteria first drive the rank up).
    include : iterable of int, optional
        Indices of candidates that are always part of the design: `'forward'` starts from them (they count
        towards `num_circuits`), `'backward'` never removes them.
    show_progress_bar : bool, optional
        Show a `tqdm` progress bar over the greedy steps (one step per circuit added or removed).  The bar's
        total is the number of steps needed to reach `num_circuits`; when `max_error_bar` stops the search
        earlier the bar simply ends early.

    Returns
    -------
    list of int
        Indices into `design_matrices` of the selected circuits -- in order of selection for `'forward'`
        (`include` first, then every prefix is itself a greedy design) and in the original order for
        `'backward'`.
    """
    if criterion not in _SELECTION_CRITERIA:
        raise ValueError("Unknown criterion '%s'; expected one of %s" % (criterion, _SELECTION_CRITERIA))
    if method not in ('forward', 'backward'):
        raise ValueError("Unknown method '%s'; expected 'forward' or 'backward'" % method)
    if num_circuits is None and max_error_bar is None:
        raise ValueError("Either num_circuits or max_error_bar must be given")
    arrays, infos = _as_design_arrays(design_matrices)
    n = len(arrays)
    if n == 0:
        return []
    P = arrays[0].shape[1]
    if num_circuits is None:
        num_circuits = n if method == 'forward' else 0
    if not 0 <= num_circuits <= n:
        raise ValueError("num_circuits must be between 0 and the number of candidates (%d)" % n)
    include = list(dict.fromkeys(int(i) for i in include)) if include is not None else []
    if any(not 0 <= i < n for i in include):
        raise ValueError("include contains an index outside 0..%d" % (n - 1))
    if len(include) > num_circuits:
        raise ValueError("num_circuits (%d) is smaller than the number of included circuits (%d)"
                         % (num_circuits, len(include)))
    gram = [a.T @ a for a in arrays]
    if max_error_bar is not None:
        covariances = [_ideal_observable_covariance(info) if _has_circuit_info(info) else None for info in infos]

    def meets_target(subset):
        if max_error_bar is None or len(subset) == 0:
            return False
        variances = _predicted_variances([arrays[i] for i in subset], [covariances[i] for i in subset])
        return _np.sqrt(variances.max() / shots) <= max_error_bar

    by_shape = _collections.defaultdict(list)          # candidates are scored in batches of equal shape
    for i in range(n):
        by_shape[arrays[i].shape].append(i)
    groups = [(_np.array(idx), _np.stack([arrays[i] for i in idx])) for idx in by_shape.values()]

    def scores(M, candidates, sign):
        if criterion == 'E':
            return [_np.linalg.eigvalsh(M + sign * gram[i])[0] for i in candidates]
        M_inv = _np.linalg.inv(M)
        all_scores = _np.empty(n)
        for idx, stacked in groups:
            all_scores[idx] = _selection_scores(criterion, M_inv, stacked, sign)
        return all_scores[candidates]

    def progress_bar(total, desc):
        if not show_progress_bar:
            return None
        from tqdm import tqdm
        return tqdm(total=total, ascii=True, desc=desc, unit="circuit")

    M = ridge * _np.eye(P)
    if method == 'forward':
        chosen, remaining = list(include), [i for i in range(n) if i not in include]
        M = M + sum((gram[i] for i in include), _np.zeros((P, P)))
        pbar = progress_bar(num_circuits - len(chosen), "Selecting circuits (%s-optimal, forward)" % criterion)
        while len(chosen) < num_circuits and remaining and not meets_target(chosen):
            j = remaining[int(_np.argmax(scores(M, remaining, +1)))]
            chosen.append(j)
            remaining.remove(j)
            M = M + gram[j]
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        if max_error_bar is not None and not meets_target(chosen):
            if len(chosen) == n:
                raise ValueError("The target error bar %g cannot be reached even with all %d candidates x %g shots"
                                 % (max_error_bar, n, shots))
            _warnings.warn("The target error bar %g is not reached with num_circuits=%d circuits x %g shots"
                           % (max_error_bar, num_circuits, shots))
        return chosen

    chosen = list(range(n))
    M = M + sum(gram)
    if max_error_bar is not None and not meets_target(chosen):
        raise ValueError("The target error bar %g cannot be reached even with all %d candidates x %g shots"
                         % (max_error_bar, n, shots))
    removable = [i for i in chosen if i not in include]
    pbar = progress_bar(len(chosen) - num_circuits, "Selecting circuits (%s-optimal, backward)" % criterion)
    while len(chosen) > num_circuits and removable:
        j = removable[int(_np.argmax(scores(M, removable, -1)))]
        if max_error_bar is not None and not meets_target([i for i in chosen if i != j]):
            break
        chosen.remove(j)
        removable.remove(j)
        M = M - gram[j]
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    return chosen


# ---------------------------------------------------------------------------------------------------
# Experiment design: allocating a fixed shot budget across (possibly all of) the candidates
# ---------------------------------------------------------------------------------------------------
#
# `select_circuits` answers "which circuits?" by picking a subset and implicitly giving every member of the
# subset the same number of shots.  The functions below answer "how many shots does each candidate get?",
# allowing the allocation to vary continuously over the whole candidate pool at a fixed *total* shot budget --
# the classical "approximate design" relaxation of optimal experimental design (Kiefer & Wolfowitz, 1960;
# Fedorov, *Theory of Optimal Experiments*, 1972; Pukelsheim, *Optimal Design of Experiments*, 1993).  A
# circuit the optimizer does not want is simply given (near-)zero shots, so there is no separate combinatorial
# selection step: `optimal_shot_allocation` can be handed the entire candidate pool (structured probes,
# random circuits, or both) and it performs selection and allocation jointly.
#
# Two estimators are in play, and they call for two different allocation methods.
#
# `estimate_error_rates`/`solve_linearized_gst` fit by plain pseudo-inversion (H) and NNLS (S, or bounded
# least squares for S+C+A) of the *stacked, unweighted* design matrix -- every circuit's rows count equally
# regardless of how many shots measured them. Once shot counts are unequal this is no longer even sensible
# (a circuit measured with 10 shots is treated as being known exactly as well as one measured with 10^6), so
# `neyman_shot_allocation` instead predicts and optimizes a *per-circuit inverse-shot-count-weighted* least
# squares fit (``Σ_c N_c‖y_c - D_c θ‖²``), which reduces to the current estimator when all `N_c` happen to be
# equal and is otherwise the obvious and safe fix. It works for a *fixed* circuit list, in closed form, and
# *does* exploit the cross-observable correlations within a circuit (`_ideal_observable_covariance`) because
# propagating a fixed pseudo-inverse through a covariance (as `predicted_error_bars` already does) never
# requires inverting that covariance.
#
# `optimal_shot_allocation` goes further and also chooses *which* circuits to use, which requires a per-shot
# Fisher information that is additive over circuits -- true of a properly weighted-least-squares estimator,
# not of plain pseudo-inversion (whose effective weighting on a given row depends on the whole, variable,
# circuit list). We deliberately do *not* fold the cross-observable correlations into that information
# matrix: `_ideal_observable_covariance` is exactly singular along directions where two observables are
# perfectly correlated under the *ideal*, zeroth-order circuit (e.g. the MCM record and a later measurement
# of the same, undisturbed qubit), and naively inverting it (e.g. with a Moore-Penrose pseudo-inverse) would
# silently discard that direction's contribution to the Fisher information entirely -- the opposite of a
# conservative approximation, since in reality the circuit's *true* (non-ideal) covariance is nonsingular
# there, with a variance that depends on the very error rates being estimated. `select_circuits` already
# makes the same simplifying choice for its own (equal-shot) greedy criterion (`gram = [a.T @ a for a in
# arrays]`); `optimal_shot_allocation` is best understood as its continuous, unequal-shot generalization,
# using the same per-shot information. Exploiting the within-circuit correlations in a convex, non-circular
# way (e.g. by bounding the singular directions' true variance rather than discarding them) is a natural but
# unresolved extension, left to future work.

def _fisher_information(arrays):
    """
    Per-shot Fisher information ``F_c = D_cᵀ D_c`` of every circuit, i.e. the information of a weighted
    least-squares fit that treats each circuit's observables as independent and unit-variance (so that
    circuits are weighted only by how many shots they are given, not by their internal observable
    correlations; see the module-level note above). Used by :func:`optimal_shot_allocation`.
    """
    return [arr.T @ arr for arr in arrays]


def neyman_shot_allocation(design_matrices, budget, weight=None, tol=1e-10):
    """
    Closed-form optimal shot allocation, at a fixed total budget, for a *fixed* list of circuits -- optimal
    for the plain least-squares / NNLS estimator that :func:`estimate_error_rates` actually uses (contrast
    :func:`optimal_shot_allocation`, which assumes a weighted-least-squares estimator but can also choose
    *which* circuits to use).

    With `N_c` shots given to circuit `c`, the (unweighted) least-squares estimate has covariance
    ``Cov = Σ_c M_c / N_c`` with ``M_c`` the (shot-independent) per-circuit contribution computed by
    :func:`predicted_error_bars` (``M_c = D⁺_c Σ_c D⁺_cᵀ`` in block notation).  Minimizing
    ``trace(weight @ Cov)`` subject to ``Σ_c N_c = budget``, ``N_c ≥ 0`` is a classical allocation problem
    (Neyman allocation in stratified sampling; Elfving, *Ann. Math. Statist.* 23, 255 (1952); Studden,
    *Ann. Math. Statist.* 42, 1613 (1971), for this trace/quadratic-loss generalization to several
    parameters at once). By the Cauchy-Schwarz equality condition its exact solution is
    ``N_c ∝ sqrt(trace(weight @ M_c))`` -- circuits are weighted by the square root of how much variance
    they remove, not the variance itself, so the allocation is less aggressive than "give everything to the
    single most informative circuit."  No numerical optimizer is needed.

    Parameters
    ----------
    design_matrices : list of CircuitDesign or numpy.ndarray
        The circuits to allocate shots to (already selected, e.g. by :func:`select_circuits`; this function
        does not add or remove circuits, only splits the budget between them).
    budget : int or float
        Total number of shots to distribute.
    weight : numpy.ndarray, optional
        A positive semi-definite ``(P, P)`` matrix (``P`` the number of parameters); defaults to the
        identity (ordinary A-optimality, i.e. the expected total squared error of the rate vector). Use a
        diagonal matrix to equalize sectors measured in different units, or a projector to target only a
        gauge-invariant subspace (e.g. the FOMGI quantities of an MCM).

    Returns
    -------
    shots : numpy.ndarray
        Real-valued optimal shot count per circuit, summing to `budget` (round before use).
    objective : float
        The resulting ``trace(weight @ Cov)``, i.e. ``(Σ_c sqrt(trace(weight @ M_c)))² / budget``.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    arrays, infos = _as_design_arrays(design_matrices)
    covariances = [_ideal_observable_covariance(info) if _has_circuit_info(info) else None for info in infos]
    design_matrix = _np.vstack(arrays)
    P = design_matrix.shape[1]
    U, s, Vt = _np.linalg.svd(design_matrix, full_matrices=False)
    identifiable = s > tol * max(s[0], 1e-300)
    if not _np.all(identifiable):
        raise ValueError("The stacked design matrix is rank deficient (%d of %d directions unidentifiable); "
                         "add circuits (see `unidentifiable_directions`) before allocating shots."
                         % (int((~identifiable).sum()), P))
    pinv = (Vt.T / s) @ U.T
    W = _np.eye(P) if weight is None else _np.asarray(weight)
    k = _np.zeros(len(arrays))
    start = 0
    for c, (arr, cov) in enumerate(zip(arrays, covariances)):
        block = pinv[:, start:start + arr.shape[0]]
        start += arr.shape[0]
        Mc = block @ (_np.eye(arr.shape[0]) if cov is None else cov) @ block.T
        k[c] = max(float(_np.trace(W @ Mc)), 0.0)
    sqrt_k = _np.sqrt(k)
    total = sqrt_k.sum()
    if total <= 0:
        return _np.full(len(arrays), budget / len(arrays)), 0.0
    return budget * sqrt_k / total, float(total ** 2 / budget)


_DESIGN_CRITERIA = ('A', 'D', 'E')


def _solve_continuous_allocation(arrays, budget, criterion, weight, ridge, solver_kwargs):
    cp = _sdptools._get_cvxpy()
    F = _fisher_information(arrays)
    n, P = len(arrays), arrays[0].shape[1]

    w = cp.Variable(n, nonneg=True, name='w')
    # `budget * w` is the (continuous) number of shots given to each circuit, so `budget * sum(w_c F_c)` is
    # the total information at this budget -- scaling it in here means `M` directly represents the
    # information of one unit of budget, so the problem is well-conditioned (O(1) entries) independent of
    # `budget`, which can be in the millions; the criteria are rescaled to the actual budget afterwards,
    # where that rescaling is a one-line closed form rather than something the solver has to discover.
    M = ridge * _np.eye(P) + sum((w[c] * F[c] for c in range(n)), start=_np.zeros((P, P)))
    constraints = [cp.sum(w) == 1]
    if criterion == 'D':
        objective = cp.Maximize(cp.log_det(M))
    elif criterion == 'E':
        t = cp.Variable(name='t')
        constraints.append(M - t * _np.eye(P) >> 0)
        objective = cp.Maximize(t)
    else:
        W = _np.eye(P) if weight is None else _np.asarray(weight)
        Y = cp.Variable((P, P), symmetric=True, name='Y')
        constraints.append(cp.bmat([[M, _np.eye(P)], [_np.eye(P), Y]]) >> 0)
        objective = cp.Minimize(cp.trace(W @ Y))

    value, varvals = _sdptools.solve_sdp(cp.Problem(objective, constraints), **(solver_kwargs or {}))
    if 'w' not in varvals or varvals['w'] is None:
        raise RuntimeError("optimal_shot_allocation: all SDP solvers failed to find a solution; "
                           "see the warnings issued above for solver-specific errors.")
    w_val = _np.clip(varvals['w'], 0, None)
    w_val = w_val / w_val.sum()
    # rescale the unit-budget criterion value to the actual budget: information scales linearly with the
    # number of shots, so trace(inverse) scales as 1/budget, log det as +P log(budget), and the smallest
    # eigenvalue as budget (the optimal `w` itself is budget-independent, which is why solving at unit
    # budget and rescaling afterwards is exact, not an approximation).
    if criterion == 'A':
        rescaled_value = float(value) / budget
    elif criterion == 'D':
        rescaled_value = float(value) + P * _np.log(budget)
    else:
        rescaled_value = float(value) * budget
    return budget * w_val, rescaled_value


def optimal_shot_allocation(design_matrices, budget, criterion='A', weight=None, ridge=1e-6, min_shots=0,
                            solver_kwargs=None):
    """
    Jointly choose which circuits to run and how many shots to give each of them, at a fixed total shot
    budget, by solving the continuous ("approximate design") relaxation of the classical
    optimal-experimental-design problem over the *entire* candidate pool.

    Unlike :func:`select_circuits` -- which greedily picks a *subset* of circuits that then (implicitly) get
    *equal* numbers of shots -- this function lets the number of shots vary continuously over the whole pool.
    Candidates that are not worth running are simply driven to (near-)zero weight by the optimizer, so
    "which circuits" and "how many shots" are solved as one convex program rather than two separate
    heuristics applied in sequence.  The relaxation is standard: Kiefer and Wolfowitz's equivalence theorem
    guarantees that for large budgets, rounding the continuous optimum to integers costs essentially nothing
    (the shortfall from the unrounded optimum is ``O(n_circuits)`` shots, negligible against a budget in the
    thousands or millions).

    This predicts the performance of a least-squares fit *weighted by shot count*
    (``Σ_c N_cǁy_c - D_c θǁ²``), which is the minimal, obviously-necessary fix to plain OLS/NNLS once shot
    counts are unequal (a circuit run with 10 shots cannot be trusted as much as one run with 10⁶): its
    Fisher information is additive over circuits (``Σ_c N_c D_cᵀD_c``), which is what makes selection and
    allocation jointly convex. :func:`solve_linearized_gst` does not currently implement this weighting (it
    pools every circuit's rows with equal weight regardless of its shot count); for the allocation that is
    optimal for the *exact* estimator used today, with a *fixed* circuit list, see
    :func:`neyman_shot_allocation` (which also, unlike this function, exploits the correlations between
    observables measured in the same circuit; see the module-level note above for why that is not done
    here). Until `solve_linearized_gst` is updated to weight by shot count, treat this function's output as
    a design heuristic -- it still favors circuits that are genuinely more informative per shot -- whose
    numerical prediction is exactly realized by that estimator, not quite by the current one.

    The formulation is agnostic to the error-generator sectors present: it only uses each circuit's design
    matrix, computed identically whether the ansatz is H+S or the fully general H+S+C+A, with or without
    mid-circuit measurements or SPAM parameters.

    Parameters
    ----------
    design_matrices : list of CircuitDesign or numpy.ndarray
        The candidate pool -- e.g. a structured probe family, a batch of random circuits, or both
        concatenated (a "hybrid pool", as in the second tutorial's Section 5).
    budget : int or float
        Total number of shots to distribute across the candidates.
    criterion : {'A', 'D', 'E'}, optional
        ``'A'``: minimize ``trace(weight @ Cov)``, the (weighted) expected squared error of the full rate
        vector -- with `weight=None` this is exactly the ACES figure of merit's informational content,
        ``trace(Cov)``.
        ``'D'``: maximize ``log det(information)``, i.e. minimize the volume of the confidence ellipsoid.
        Invariant under any invertible linear reparametrization of the rates (e.g. switching the MCM's
        parameters to FOMGI coordinates does not change the optimal design), unlike `'A'` and `'E'`.
        ``'E'``: maximize the smallest eigenvalue of the information matrix, i.e. minimize the *worst-case*
        parameter combination's variance (a minimax guarantee, matching the "max" error bars the tutorials
        otherwise track by trial and error).
    weight : numpy.ndarray, optional
        A positive semi-definite ``(P, P)`` matrix used only by `criterion='A'` (see
        :func:`neyman_shot_allocation` for the same option); defaults to the identity.
    ridge : float, optional
        Small multiple of the identity added to the information matrix for numerical conditioning before the
        pool's design has full rank, as in `select_circuits`.
    min_shots : int or float, optional
        After solving, candidates allocated fewer than `min_shots` are dropped and the (smaller) problem is
        re-solved on the remainder, so that every circuit that is used receives at least `min_shots` --
        useful when per-circuit overhead makes very small shot counts impractical.
    solver_kwargs : dict, optional
        Extra keyword arguments forwarded to `pygsti.tools.sdptools.solve_sdp` (hence to `cvxpy`'s
        `Problem.solve`), e.g. `dict(verbose=True)`.

    Returns
    -------
    shots : numpy.ndarray
        Shot count for each candidate circuit, in the order of `design_matrices` (zero for circuits not
        used; real-valued -- round before use).
    objective : float
        The optimized criterion value (for `'A'`, `trace(weight @ Cov)`; for `'D'`, `log det(information)`;
        for `'E'`, the smallest eigenvalue of `information`).
    """
    if not _sdptools.CVXPY_ENABLED:
        raise ImportError("optimal_shot_allocation requires cvxpy; `pip install cvxpy`.")
    if criterion not in _DESIGN_CRITERIA:
        raise ValueError("Unknown criterion '%s'; expected one of %s" % (criterion, _DESIGN_CRITERIA))
    if weight is not None and criterion != 'A':
        _warnings.warn("`weight` is only used by criterion='A'; ignoring it for criterion='%s'" % criterion)
    if budget <= 0:
        raise ValueError("budget must be positive")
    arrays, _ = _as_design_arrays(design_matrices)
    n = len(arrays)
    shots, objective = _solve_continuous_allocation(arrays, budget, criterion, weight, ridge, solver_kwargs)
    if min_shots > 0:
        active = shots >= min_shots
        if not _np.any(active):
            raise ValueError("No candidate can be allocated at least %g shots from a budget of %g"
                             % (min_shots, budget))
        if not _np.all(active):
            kept = [i for i in range(n) if active[i]]
            sub_shots, objective = optimal_shot_allocation(
                [design_matrices[i] for i in kept], budget, criterion, weight, ridge, min_shots, solver_kwargs)
            shots = _np.zeros(n)
            shots[kept] = sub_shots
    return shots, objective


# ---------------------------------------------------------------------------------------------------
# Experiment design under a nominal error model
# ---------------------------------------------------------------------------------------------------
#
# The functions above score a circuit by ``D_cᵀ D_c`` -- every observable independent with unit variance -- and
# the module-level note before `_fisher_information` explains what that leaves on the table: the data of a
# circuit are a multinomial sample, and the Z-type observables estimated from it have a covariance that (i)
# correlates observables measured in the same shots and (ii) is *much* smaller than 1 for observables whose
# ideal value is ±1, where the variance is of the order of the error rates themselves.  Both effects are
# first-order computable from the same design machinery: with ``m = <Q>`` the vector of nominal expectation
# values and ``M_ab = <Q_a Q_b>`` the nominal expectation values of the *product* observables (which are
# again Z-type Paulis, so their ideal values and design rows are available), the per-shot covariance is
# ``Σ_c = M - m mᵀ``.  Evaluating it at a *nominal* error model (a guess of the rates, or a previous estimate)
# makes it non-singular, resolves the circularity noted above, and turns the shot-noise statistics of every
# candidate circuit into a per-shot Fisher information ``F_c = D_cᵀ Σ_c⁻¹ D_c`` -- the information of the
# generalized-least-squares (GLS) estimator, which attains it.  The resulting design problem,
#
#     minimize   Φ_W(N) = trace( W · (Σ_c N_c F_c)⁻¹ )     subject to   Σ_c N_c = budget,  N_c ≥ 0,
#
# is the classical (locally) A-optimal approximate design problem (Kiefer & Wolfowitz 1960; Fedorov 1972;
# Pukelsheim 1993; for quantum tomography Kosut, Walmsley & Rabitz, quant-ph/0411093), with ``W`` a
# positive-semidefinite weight that says what the rates are for (identity: every rate counts equally; a
# diagonal matrix of inverse precision targets; or the "prediction" weight of :func:`prediction_weight`, the
# expected squared error of the predicted observable shifts of application circuits).  Φ_I is exactly the
# squared ACES figure of merit of Hockings, Doherty & Harper (PRX Quantum 6, 010334 (2025)), who likewise
# evaluate a nominal-noise GLS covariance to allocate shots over a candidate set of circuits.  The problem is
# convex; :func:`optimal_allocation` solves it with a multiplicative algorithm that needs no external solver
# and returns an optimality certificate from the Kiefer-Wolfowitz equivalence theorem (or with `cvxpy`), and
# :func:`estimate_error_rates_gls` is the estimator whose covariance the design predicts.

class CircuitMoments(_collections.namedtuple('CircuitMoments', ['design', 'product_paulis', 'pair_index',
                                                                'product_design_matrix',
                                                                'product_ideal_expectations'])):
    """
    First *and second* moments of the observables of one circuit, to first order in the error rates.

    Attributes
    ----------
    design : CircuitDesign
        The circuit's design matrix, observables and ideal expectation values (first moments).
    product_paulis : list of stim.PauliString
        The distinct pairwise products ``Q_a Q_b`` (``a < b``) of the observables -- Z-type Paulis of up to
        twice the observable weight.
    pair_index : numpy.ndarray
        ``(k, k)`` integer array mapping an observable pair to its row in `product_design_matrix`
        (``-1`` on the diagonal, where ``Q_a Q_a = I``).
    product_design_matrix : numpy.ndarray
        Design rows (first-order sensitivities) of the product observables.
    product_ideal_expectations : numpy.ndarray
        Ideal expectation values of the product observables.
    """
    __slots__ = ()

    @property
    def circuit(self):
        return self.design.circuit

    @property
    def shape(self):
        return self.design.shape

    @property
    def num_observables(self):
        return self.design.shape[0]

    def expectations(self, rates):
        """Nominal expectation values of the observables, ``<Q> = <Q>_ideal + D ε``, clipped to [-1, 1]."""
        return _np.clip(self.design.ideal_expectations + self.design.design_matrix @ _np.asarray(rates, float),
                        -1.0, 1.0)

    def second_moments(self, rates):
        """Nominal matrix ``M_ab = <Q_a Q_b>`` (``M_aa = 1``)."""
        k = self.num_observables
        M = _np.eye(k)
        if len(self.product_paulis) > 0:
            prod = _np.clip(self.product_ideal_expectations + self.product_design_matrix @ _np.asarray(rates, float),
                            -1.0, 1.0)
            iu = _np.triu_indices(k, 1)
            M[iu] = prod[self.pair_index[iu]]
            M = M + M.T - _np.eye(k)
        return M

    def covariance(self, rates, min_variance=1e-3):
        """
        Per-shot covariance ``Σ = M - m mᵀ`` of the observable estimates under the nominal `rates`.

        The matrix is made positive definite by raising every eigenvalue below `min_variance` to
        `min_variance` -- the exact first-order covariance is singular along directions in which no error
        generator of the ansatz has an effect (e.g. an observable whose nominal value is ±1 when all rates that
        could flip it are zero), and the floor states the smallest variance (i.e. the smallest flip
        probability, ``min_variance/2``) the design is allowed to count on.
        """
        m = self.expectations(rates)
        Sigma = self.second_moments(rates) - _np.outer(m, m)
        Sigma = 0.5 * (Sigma + Sigma.T)
        evals, U = _np.linalg.eigh(Sigma)
        evals = _np.maximum(evals, min_variance)
        return (U * evals) @ U.T

    def whitened_design(self, rates, min_variance=1e-3):
        """``B = Σ^{-1/2} D`` such that the per-shot Fisher information is ``Bᵀ B``."""
        m = self.expectations(rates)
        Sigma = self.second_moments(rates) - _np.outer(m, m)
        Sigma = 0.5 * (Sigma + Sigma.T)
        evals, U = _np.linalg.eigh(Sigma)
        evals = _np.maximum(evals, min_variance)
        return (U.T @ self.design.design_matrix) / _np.sqrt(evals)[:, None]


def _product_observables(paulis):
    """Distinct pairwise products of a list of Z-type Paulis and the (k, k) index map into them."""
    k = len(paulis)
    pair_index = -_np.ones((k, k), dtype=int)
    products, where = [], {}
    for a in range(k):
        for b in range(a + 1, k):
            prod = paulis[a] * paulis[b]
            if prod.sign != 1:
                prod = prod / prod.sign  # cannot happen for Z-type Paulis, but keep the sign out of the key
            key = str(prod)
            if key not in where:
                where[key] = len(products)
                products.append(prod)
            pair_index[a, b] = pair_index[b, a] = where[key]
    return products, pair_index


def create_design_moments(designs_or_circuits, model_parameter_indexing, **design_kwargs):
    """
    First- and second-moment design information of circuits (needed for shot-noise statistics beyond the
    worst case): one :class:`CircuitMoments` per circuit.

    Parameters
    ----------
    designs_or_circuits : list of CircuitDesign or list of Circuit
        :class:`CircuitDesign` objects (from ``create_design_matrix_list(..., return_info=True)``; their
        observables and design matrices are reused) or circuits, whose design matrices are computed first.
    model_parameter_indexing : ModelParameterList
    **design_kwargs
        Passed to :func:`create_circuit_design_matrix` (``pauli_measurements``, ``max_pauli_weight``, ...).

    Returns
    -------
    list of CircuitMoments
    """
    out = []
    for item in designs_or_circuits:
        if isinstance(item, CircuitDesign):
            design = item
            if design.paulis is None or design.circuit is None:
                raise ValueError("create_design_moments needs CircuitDesign objects with observables and circuits "
                                 "(use create_design_matrix_list(..., return_info=True)).")
        else:
            design = create_circuit_design_matrix(item, model_parameter_indexing, return_info=True, **design_kwargs)
        prop_kwargs = {k: design_kwargs[k] for k in ('qubit_labels', 'mcm_gate_names', 'cnot_name',
                                                      'gate_name_conversions') if k in design_kwargs}
        prop = propagate_parameter_errorgens(design.circuit, model_parameter_indexing, **prop_kwargs)
        paulis = list(design.paulis)
        products, pair_index = _product_observables(paulis)
        total = prop.num_qubits + prop.num_mcms
        if len(products) > 0:
            sens = calculate_sensitivity_matrix_paulis(prop.tableau, prop.eoc_errorgens, total, products)
            product_design = sens @ prop.parameter_matrix
            product_ideal = ideal_pauli_expectations(prop.tableau, products)
        else:
            product_design = _np.zeros((0, prop.parameter_matrix.shape[1]))
            product_ideal = _np.zeros(0)
        out.append(CircuitMoments(design, products, pair_index, product_design, product_ideal))
    return out


def nominal_observable_covariance(moments, nominal_rates=None, min_variance=1e-3):
    """
    Per-shot covariance of the observable estimates of one circuit under a nominal error model.

    See :meth:`CircuitMoments.covariance`.  With ``nominal_rates=None`` the ideal (error-free) statistics are
    used, floored at `min_variance`.
    """
    rates = _np.zeros(moments.shape[1]) if nominal_rates is None else _np.asarray(nominal_rates, float)
    return moments.covariance(rates, min_variance)


def nominal_rates_by_type(model_parameter_indexing, rates_by_type):
    """
    A nominal rate vector that assigns one value per error-generator sector, e.g. ``{'H': 0.0, 'S': 1e-3}``.

    Sectors not listed get rate 0.  Useful as the "what we expect of the device" input of a design when
    nothing better than typical error magnitudes is known.
    """
    types = model_parameter_indexing.errorgen_types()
    return _np.array([float(rates_by_type.get(t, 0.0)) for t in types])


class DesignInformation(object):
    """
    Per-shot Fisher information of a pool of candidate circuits, stored in whitened form.

    For circuit ``c`` the information is ``F_c = B_cᵀ B_c`` with ``B_c`` of shape ``(k_c, P)``; storing the
    blocks rather than the ``P x P`` matrices keeps the memory linear in the number of observables and makes
    the information of any shot allocation ``Σ_c N_c F_c`` a single weighted Gram matrix.

    Attributes
    ----------
    blocks : list of numpy.ndarray
    cost : numpy.ndarray
        Cost of one shot of each circuit (default 1); budgets in :func:`optimal_allocation` are in these units.
    """

    def __init__(self, blocks, cost=None, labels=None):
        self.blocks = [_np.asarray(b, dtype=float) for b in blocks]
        if len(self.blocks) == 0:
            raise ValueError("DesignInformation needs at least one circuit")
        self.num_params = self.blocks[0].shape[1]
        if any(b.shape[1] != self.num_params for b in self.blocks):
            raise ValueError("All information blocks must have the same number of columns (parameters)")
        self.cost = _np.ones(len(self.blocks)) if cost is None else _np.asarray(cost, dtype=float)
        if self.cost.shape != (len(self.blocks),) or _np.any(self.cost <= 0):
            raise ValueError("cost must be a positive number for every circuit")
        self.labels = labels
        self._stacked = _np.vstack(self.blocks)
        self._owner = _np.repeat(_np.arange(len(self.blocks)), [b.shape[0] for b in self.blocks])

    def __len__(self):
        return len(self.blocks)

    @property
    def num_circuits(self):
        return len(self.blocks)

    def subset(self, indices):
        """The information of a subset of the candidates (same order as `indices`)."""
        indices = list(indices)
        return DesignInformation([self.blocks[i] for i in indices], self.cost[indices],
                                 None if self.labels is None else [self.labels[i] for i in indices])

    def matrices(self):
        """The explicit per-shot information matrices ``F_c`` (one ``P x P`` array per circuit)."""
        return [b.T @ b for b in self.blocks]

    def information(self, shots):
        """The information matrix ``Σ_c shots_c F_c`` of an allocation (`shots` scalar or per circuit)."""
        shots = _np.broadcast_to(_np.asarray(shots, dtype=float), (len(self.blocks),))
        if _np.any(shots < 0):
            raise ValueError("shots must be non-negative")
        weighted = self._stacked * shots[self._owner][:, None]
        return self._stacked.T @ weighted

    def _scores(self, A):
        """``trace(A F_c)`` for every circuit, for a symmetric matrix ``A``."""
        G = self._stacked @ A
        row = _np.einsum('rj,rj->r', G, self._stacked)
        return _np.bincount(self._owner, weights=row, minlength=len(self.blocks))


def circuit_information(moments, nominal_rates=None, min_variance=1e-3, cost=None):
    """
    The per-shot Fisher information of every candidate circuit about the error rates, under a nominal model.

    For a circuit with observables ``Q`` (design matrix ``D_c``) measured with ``N_c`` shots, the observed
    expectation values have covariance ``Σ_c / N_c`` with ``Σ_c`` the nominal per-shot covariance of
    :func:`nominal_observable_covariance`, so the information they carry about the rates is
    ``N_c D_cᵀ Σ_c⁻¹ D_c``.  This is the information of the generalized-least-squares estimator
    (:func:`estimate_error_rates_gls`), whose covariance is its inverse.

    Parameters
    ----------
    moments : list of CircuitMoments
        From :func:`create_design_moments`.
    nominal_rates : array-like, optional
        Nominal error rates (one per model parameter).  The statistics of observables with ideal value ±1
        depend on them to first order (their variance is of the order of the rates that can flip them), the
        statistics of observables with ideal value 0 only weakly.  ``None`` means zero rates, i.e. the ideal
        statistics floored at `min_variance`.
    min_variance : float, optional
        Floor on the eigenvalues of every circuit's per-shot covariance; see :meth:`CircuitMoments.covariance`.
    cost : array-like, optional
        Cost of one shot of each circuit (e.g. its duration), used by :func:`optimal_allocation` when the budget
        is not simply a number of shots.

    Returns
    -------
    DesignInformation
    """
    P = moments[0].shape[1]
    rates = _np.zeros(P) if nominal_rates is None else _np.asarray(nominal_rates, dtype=float)
    if rates.shape != (P,):
        raise ValueError("nominal_rates must have one entry per model parameter (%d)" % P)
    blocks = [m.whitened_design(rates, min_variance) for m in moments]
    return DesignInformation(blocks, cost)


def probability_information(probability_designs, nominal_rates=None, min_probability=5e-4, cost=None):
    """
    The per-shot Fisher information of the *full* outcome distribution of every circuit (all bitstring
    probabilities rather than a set of Z-type observables).

    This is the Cramér-Rao bound for any estimator that uses all of a circuit's data; comparing it with
    :func:`circuit_information` for the default weight-≤2 observables shows what the truncation to few
    observables costs.  It needs ``2**(num_qubits + num_mcms)`` probabilities per circuit, so it is only
    practical for small systems.

    Parameters
    ----------
    probability_designs : list of CircuitDesign
        From ``create_design_matrix_list(..., usePaulis=False, return_info=True)``.
    nominal_rates : array-like, optional
    min_probability : float, optional
        Floor on the nominal outcome probabilities (ideally-impossible outcomes have a nominal probability of
        the order of the error rates; the floor bounds the information the design may count on from them).
    cost : array-like, optional

    Returns
    -------
    DesignInformation
    """
    blocks = []
    for d in probability_designs:
        if d.paulis is not None:
            raise ValueError("probability_information needs designs created with usePaulis=False")
        J = d.design_matrix
        rates = _np.zeros(J.shape[1]) if nominal_rates is None else _np.asarray(nominal_rates, dtype=float)
        p = _np.maximum(d.ideal_expectations + J @ rates, min_probability)
        blocks.append(J / _np.sqrt(p)[:, None])
    return DesignInformation(blocks, cost)


def predicted_covariance(information, shots, nominal_rates=None, estimator='gls', min_variance=1e-3, tol=1e-10):
    """
    Covariance of the error-rate estimate for an allocation of shots, under a nominal error model.

    Parameters
    ----------
    information : DesignInformation or list of CircuitMoments
        A :class:`DesignInformation` (then `estimator` must be ``'gls'`` and the nominal model is the one it was
        built with) or the circuits' :class:`CircuitMoments`, from which the nominal statistics are evaluated.
    shots : float or array-like
        Shots per circuit (scalar or one entry per circuit; zero entries are allowed).
    nominal_rates : array-like, optional
        Only used with :class:`CircuitMoments` input.
    estimator : {'gls', 'ols'}, optional
        ``'gls'``: the generalized-least-squares estimator (:func:`estimate_error_rates_gls`), whose covariance
        is the inverse information ``(Σ_c N_c F_c)⁻¹``.  ``'ols'``: the plain pseudo-inverse estimator of
        :func:`estimate_error_rates`, which weights every observable of every circuit equally regardless of
        its shot count and variance; its covariance ``D⁺ Σ D⁺ᵀ`` is evaluated with the same nominal statistics,
        so the two can be compared like for like.
    min_variance : float, optional
    tol : float, optional
        Relative singular-value cutoff below which a direction is considered unidentifiable (its variance is
        reported as ``inf``).

    Returns
    -------
    numpy.ndarray
        ``(P, P)`` covariance matrix.
    """
    if isinstance(information, DesignInformation):
        if estimator != 'gls':
            raise ValueError("estimator='ols' needs CircuitMoments input (the OLS weighting depends on D itself)")
        n = len(information)
        shots = _np.broadcast_to(_np.asarray(shots, dtype=float), (n,))
        used = shots > 0
        if not _np.any(used):
            raise ValueError("at least one circuit must receive shots")
        A = _np.vstack([information.blocks[c] * _np.sqrt(shots[c]) for c in range(n) if used[c]])
        return _pinv_covariance(A, tol)
    moments = list(information)
    P = moments[0].shape[1]
    rates = _np.zeros(P) if nominal_rates is None else _np.asarray(nominal_rates, dtype=float)
    n = len(moments)
    shots = _np.broadcast_to(_np.asarray(shots, dtype=float), (n,))
    used = [c for c in range(n) if shots[c] > 0]
    if len(used) == 0:
        raise ValueError("at least one circuit must receive shots")
    if estimator == 'gls':
        A = _np.vstack([moments[c].whitened_design(rates, min_variance) * _np.sqrt(shots[c]) for c in used])
        return _pinv_covariance(A, tol)
    if estimator != 'ols':
        raise ValueError("estimator must be 'gls' or 'ols'")
    D = _np.vstack([moments[c].design.design_matrix for c in used])
    U, s, Vt = _np.linalg.svd(D, full_matrices=False)
    identifiable = s > tol * max(s[0], 1e-300)
    pinv = (Vt[identifiable].T / s[identifiable]) @ U[:, identifiable].T
    cov = _np.zeros((P, P))
    start = 0
    for c in used:
        k = moments[c].num_observables
        block = pinv[:, start:start + k]
        start += k
        cov += block @ moments[c].covariance(rates, min_variance) @ block.T / shots[c]
    bad = 1.0 - (Vt[identifiable] ** 2).sum(axis=0) > tol
    cov[bad, :] = cov[:, bad] = _np.inf
    return cov


def _pinv_covariance(A, tol):
    """``(AᵀA)⁻¹`` via the SVD of ``A``, with ``inf`` on the unidentifiable rows/columns."""
    U, s, Vt = _np.linalg.svd(A, full_matrices=False)
    identifiable = s > tol * max(s[0], 1e-300)
    V = Vt[identifiable].T
    cov = (V / s[identifiable] ** 2) @ V.T
    bad = 1.0 - (Vt[identifiable] ** 2).sum(axis=0) > tol
    cov[bad, :] = cov[:, bad] = _np.inf
    return cov


def predicted_error_bars_nominal(information, shots, nominal_rates=None, estimator='gls', min_variance=1e-3):
    """Square roots of the diagonal of :func:`predicted_covariance` -- one predicted standard error per rate."""
    return _np.sqrt(_np.diag(predicted_covariance(information, shots, nominal_rates, estimator, min_variance)))


def prediction_weight(application_designs, ridge=1e-3):
    """
    The weight matrix that makes the A-criterion the mean squared error of *predictions*.

    For application circuits with design matrices ``D_a``, the first-order prediction error of the fitted
    model for observable ``r`` of circuit ``a`` is ``D_a[r] · (ε̂ - ε)``, so the mean (over all observables of
    all application circuits) squared prediction error is ``(ε̂ - ε)ᵀ W (ε̂ - ε)`` with
    ``W = mean_r D[r]ᵀ D[r]`` -- and its expectation is ``trace(W Cov)``.  This weight is invariant under
    reparametrization of the rates and automatically counts a parameter as often as the application exercises
    it.

    Parameters
    ----------
    application_designs : list of CircuitDesign or numpy.ndarray
    ridge : float, optional
        Multiple of the mean diagonal entry of ``W`` added to its diagonal: the weight given to every rate
        individually, so that parameters the application never exercises are still (weakly) wanted.  Without
        it the criterion ignores such parameters entirely and a design optimized for it need not identify them
        at all; the default costs of the order of a percent of the application objective and keeps every rate
        reasonably determined.

    Returns
    -------
    numpy.ndarray
    """
    arrays, _ = _as_design_arrays(application_designs)
    D = _np.vstack(arrays)
    W = D.T @ D / D.shape[0]
    return W + ridge * _np.mean(_np.diag(W)) * _np.eye(W.shape[0])


def _as_weight(weight, P):
    if weight is None:
        return _np.eye(P)
    W = _np.asarray(weight, dtype=float)
    if W.ndim == 1:
        if W.shape != (P,):
            raise ValueError("a 1-D weight must have one entry per parameter")
        return _np.diag(W)
    if W.shape != (P, P):
        raise ValueError("weight must be a (P, P) matrix or a length-P vector")
    return 0.5 * (W + W.T)


def design_criterion(information, shots, criterion='A', weight=None, tol=1e-10):
    """
    Value of an optimal-design criterion for an allocation of shots.

    ``'A'``: ``trace(weight @ Cov)`` -- the expected (weighted) squared error of the rate vector (its square root
    divided by the number of parameters is the expected RMS error, i.e. the ACES figure of merit);
    ``'D'``: ``log det(information)``; ``'E'``: the smallest eigenvalue of the information matrix.

    Parameters
    ----------
    information : DesignInformation
    shots : float or array-like
    criterion : {'A', 'D', 'E'}, optional
    weight : numpy.ndarray, optional
        Weight for ``'A'`` (``(P, P)`` matrix or length-``P`` diagonal).

    Returns
    -------
    float
    """
    P = information.num_params
    if criterion == 'A':
        cov = predicted_covariance(information, shots, tol=tol)
        return float(_np.trace(_as_weight(weight, P) @ cov))
    M = information.information(shots)
    if criterion == 'D':
        sign, logdet = _np.linalg.slogdet(M)
        return float(logdet) if sign > 0 else -_np.inf
    if criterion == 'E':
        return float(_np.linalg.eigvalsh(M)[0])
    raise ValueError("Unknown criterion '%s'; expected 'A', 'D' or 'E'" % criterion)


AllocationResult = _collections.namedtuple('AllocationResult', ['objective', 'gap', 'iterations', 'method',
                                                                'support_size'])
AllocationResult.__doc__ = """
Summary of :func:`optimal_allocation`.

objective : float
    The criterion value at the returned allocation (``trace(weight @ Cov)`` for ``'A'``, ``log det`` of the
    information for ``'D'``, its smallest eigenvalue for ``'E'``).
gap : float
    Relative optimality certificate from the Kiefer-Wolfowitz equivalence theorem (multiplicative method):
    the objective is within ``gap`` (relative) of the optimum over the candidate pool.  ``nan`` for the
    `cvxpy` method, which converges to the optimum by itself.
iterations : int
method : str
support_size : int
    Number of circuits with a nonzero number of shots.
"""


def _multiplicative_allocation(information, criterion, W, tol, max_iter, ridge, w0=None):
    """
    Multiplicative weight algorithm for A- and D-optimal approximate designs (Titterington 1976; Silvey,
    Titterington & Torsney 1978; monotonic convergence: Yu, Ann. Statist. 38, 1593 (2010)).

    At each step the weight of every candidate is multiplied by (its directional derivative / the average
    directional derivative) ** λ, with λ = 1/2 for 'A' and λ = 1 for 'D' -- the exponents for which the
    objective is guaranteed to decrease monotonically (λ = 1 for 'A' can cycle between designs of equal
    objective, e.g. on a two-circuit problem).  A backtracking safeguard halves the step whenever the objective
    would increase anyway.  The stopping rule is the equivalence-theorem gap: for a convex criterion ``φ(w)``
    on the simplex, ``φ(w) - φ* <= max_c g_c - Σ_c w_c g_c`` with ``g_c = -∂φ/∂w_c``.

    For speed on large pools, candidates whose weight has decayed below ``1e-10`` are removed from the working
    set (a multiplicative update cannot revive them in any reasonable number of iterations anyway); the final
    gap is always evaluated over the *whole* pool, and any removed candidate that the equivalence theorem says
    is still wanted is put back and the iteration continued, so the certificate refers to the full pool.
    """
    from scipy import linalg as _sla
    n, P = len(information), information.num_params
    Bt_all = information._stacked / _np.sqrt(information.cost[information._owner])[:, None]  # information per unit cost
    owner_all = information._owner
    w = _np.full(n, 1.0 / n) if w0 is None else _np.asarray(w0, float) / _np.sum(w0)
    lam = 0.5 if criterion == 'A' else 1.0
    I = _np.eye(P)
    prune_every, prune_below = 25, 1e-10

    def working_set(active):
        rows = active[owner_all]
        return Bt_all[rows], owner_all[rows]

    def evaluate(w, Bt, owner):
        M = Bt.T @ (Bt * w[owner][:, None]) + ridge * I
        try:
            L = _sla.cholesky(M, lower=True, check_finite=False)
        except _np.linalg.LinAlgError:
            return None, _np.inf, None
        Minv = _sla.cho_solve((L, True), I, check_finite=False)
        if criterion == 'A':
            A = Minv @ W @ Minv
            phi = float(_np.sum(W * Minv))              # trace(W Minv) for symmetric W
        else:
            A = Minv
            phi = -2.0 * float(_np.sum(_np.log(_np.diag(L))))
        return A, phi, Minv

    def derivatives(A, Bt, owner):
        G = Bt @ A
        row = _np.einsum('rj,rj->r', G, Bt)
        return _np.maximum(_np.bincount(owner, weights=row, minlength=n), 0.0)   # trace(A F_c) >= 0 up to rounding

    active = w > 0
    Bt, owner = working_set(active)
    evals0 = _np.linalg.eigvalsh(Bt.T @ (Bt * w[owner][:, None]) + ridge * I)
    A, phi, _ = evaluate(w, Bt, owner)
    if A is None or evals0[0] <= 1e-12 * max(evals0[-1], 1e-300):
        raise ValueError("The candidate pool's total information is singular (some parameter combination is "
                         "unidentifiable); add circuits (see `unidentifiable_directions`) or use a ridge.")
    g = derivatives(A, Bt, owner)
    norm = phi if criterion == 'A' else float(P)
    gap = (g.max() - norm) / norm
    it = stalled = 0
    certified = False
    while it < max_iter:
        if gap <= tol:
            # converged on the working set: certify over the whole pool, reviving candidates that are wanted
            g_all = derivatives(A, Bt_all, owner_all)
            gap = (g_all.max() - norm) / norm
            revive = (~active) & (g_all > norm * (1 + tol))
            if gap <= tol or not revive.any():
                certified = True
                break
            w[revive] = w[active].mean()
            w /= w.sum()
            active |= revive
            Bt, owner = working_set(active)
            A, phi, _ = evaluate(w, Bt, owner)
            g = derivatives(A, Bt, owner)
            norm = phi if criterion == 'A' else float(P)
            gap = (g.max() - norm) / norm
            continue
        it += 1
        step = lam
        for _ in range(8):   # backtracking safeguard (the theory says it is never needed for these λ)
            w_new = w * (g / norm) ** step
            w_new /= w_new.sum()
            A_new, phi_new, _ = evaluate(w_new, Bt, owner)
            if A_new is not None and phi_new <= phi:
                break
            step *= 0.5
        else:
            break
        stalled = stalled + 1 if phi - phi_new <= 1e-15 * abs(phi) else 0
        w, A, phi = w_new, A_new, phi_new
        if it % prune_every == 0 and (w[active] < prune_below).any():
            w[active & (w < prune_below)] = 0.0
            w /= w.sum()
            active = w > 0
            Bt, owner = working_set(active)
            A, phi, _ = evaluate(w, Bt, owner)
        g = derivatives(A, Bt, owner)
        norm = phi if criterion == 'A' else float(P)
        gap = (g.max() - norm) / norm
        if stalled >= 20:    # the objective no longer moves at machine precision
            break
    if not certified:        # stopped early: still report the gap over the whole pool
        g_all = derivatives(A, Bt_all, owner_all)
        gap = (g_all.max() - norm) / norm
    return w, phi, max(gap, 0.0), it


def _cvxpy_allocation(information, criterion, W, ridge, max_fraction, solver_kwargs):
    cp = _sdptools._get_cvxpy()
    n, P = len(information), information.num_params
    Bt = information._stacked / _np.sqrt(information.cost[information._owner])[:, None]
    owner = information._owner
    w = cp.Variable(n, nonneg=True, name='w')
    M = Bt.T @ cp.multiply(cp.reshape(w[owner], (Bt.shape[0], 1), order='C'), Bt) + ridge * _np.eye(P)
    constraints = [cp.sum(w) == 1]
    if max_fraction is not None:
        constraints.append(w <= max_fraction)
    if criterion == 'D':
        objective = cp.Maximize(cp.log_det(M))
    elif criterion == 'E':
        t = cp.Variable(name='t')
        constraints.append(M - t * _np.eye(P) >> 0)
        objective = cp.Maximize(t)
    else:
        Y = cp.Variable((P, P), symmetric=True, name='Y')
        constraints.append(cp.bmat([[M, _np.eye(P)], [_np.eye(P), Y]]) >> 0)
        objective = cp.Minimize(cp.trace(W @ Y))
    value, varvals = _sdptools.solve_sdp(cp.Problem(objective, constraints), **(solver_kwargs or {}))
    if 'w' not in varvals or varvals['w'] is None:
        raise RuntimeError("optimal_allocation: the SDP solver did not return a solution")
    w_val = _np.clip(varvals['w'], 0, None)
    return w_val / w_val.sum(), float(value)


def optimal_allocation(information, budget, criterion='A', weight=None, min_shots=0, max_shots=None,
                       method='auto', tol=1e-3, max_iter=20000, ridge=0.0, drop_below=None, solver_kwargs=None):
    """
    Jointly choose which candidate circuits to run and how many shots to give each, at a fixed budget, by
    solving the convex ("approximate design") problem for the per-shot information of the candidates.

    With ``F_c`` the per-shot information of candidate ``c`` (:func:`circuit_information`) and ``κ_c`` the
    cost of one of its shots, the allocation ``N`` minimizes

    * ``'A'``: ``trace(W (Σ_c N_c F_c)⁻¹)`` -- the expected weighted squared error of the rate estimate, i.e. the
      sum of the predicted variances (``W = I``, the squared ACES figure of merit), their weighted sum, or the
      mean squared prediction error for application circuits (:func:`prediction_weight`);
    * ``'D'``: ``-log det(Σ_c N_c F_c)`` -- the volume of the confidence ellipsoid (reparametrization invariant);
    * ``'E'``: ``-λ_min(Σ_c N_c F_c)`` -- the variance of the worst-determined parameter combination,

    subject to ``Σ_c κ_c N_c = budget`` and ``N_c ≥ 0``.  Candidates the optimum does not need get zero shots,
    so selection and allocation are one problem.  The continuous optimum is returned; :func:`round_shots`
    rounds it to integers (by the equivalence theorem the loss is negligible at realistic budgets).

    Parameters
    ----------
    information : DesignInformation
        From :func:`circuit_information` (or :func:`probability_information`).
    budget : float
        Total budget, in units of the information's per-shot `cost` (plain shots by default).
    criterion : {'A', 'D', 'E'}, optional
    weight : numpy.ndarray, optional
        ``(P, P)`` positive-semidefinite matrix or length-``P`` diagonal, used by ``'A'`` only.
    min_shots : float, optional
        Candidates allocated fewer shots than this are dropped and the problem is re-solved on the remainder
        (repeatedly), so that every circuit that is used gets at least `min_shots` -- for per-circuit overheads.
    max_shots : float, optional
        Upper bound on the shots of any single circuit (a diversity safeguard; `cvxpy` method only).
    method : {'auto', 'multiplicative', 'cvxpy'}, optional
        ``'multiplicative'`` (no dependencies; ``'A'`` and ``'D'``; returns an optimality certificate) or
        ``'cvxpy'`` (semidefinite programming; all criteria and `max_shots`).  ``'auto'`` uses the
        multiplicative algorithm when it applies and `cvxpy` otherwise.
    tol : float, optional
        Relative equivalence-theorem gap at which the multiplicative algorithm stops.
    max_iter : int, optional
    ridge : float, optional
        Added to the information matrix (per unit budget) for conditioning; normally 0.
    drop_below : float, optional
        Fraction of the budget below which a candidate's allocation is set to exactly zero (and the problem
        re-solved without it); defaults to one shot's worth, ``1/budget``.
    solver_kwargs : dict, optional
        Forwarded to `pygsti.tools.sdptools.solve_sdp` (`cvxpy` method).

    Returns
    -------
    shots : numpy.ndarray
        Real-valued shots per candidate (zero for unused candidates); ``Σ_c cost_c shots_c = budget``.
    result : AllocationResult
    """
    if criterion not in _DESIGN_CRITERIA:
        raise ValueError("Unknown criterion '%s'; expected one of %s" % (criterion, _DESIGN_CRITERIA))
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not isinstance(information, DesignInformation):
        raise TypeError("information must be a DesignInformation (see circuit_information)")
    if method == 'auto':
        method = 'multiplicative' if (criterion in ('A', 'D') and max_shots is None) else 'cvxpy'
    if method not in ('multiplicative', 'cvxpy'):
        raise ValueError("method must be 'auto', 'multiplicative' or 'cvxpy'")
    n, P = len(information), information.num_params
    W = _as_weight(weight, P) if criterion == 'A' else None
    if weight is not None and criterion != 'A':
        _warnings.warn("`weight` is only used by criterion='A'; ignoring it for criterion='%s'" % criterion)
    if W is not None:
        # A singular weight (e.g. a projector onto a few parameters) lets the optimum starve every other
        # direction, which makes the information matrix arbitrarily ill-conditioned.  Optimizing with a tiny
        # ridge on the weight keeps the problem well posed at a negligible change of the optimum; the returned
        # objective is evaluated with the weight as given.
        evals_W = _np.linalg.eigvalsh(W)
        if evals_W[0] <= 1e-9 * max(evals_W[-1], 1e-300):
            W = W + 1e-6 * max(_np.mean(_np.diag(W)), 1e-300) * _np.eye(P)

    total_iterations = 0

    def solve(info, w0=None):
        nonlocal total_iterations
        if method == 'multiplicative':
            if criterion == 'E':
                raise ValueError("criterion='E' needs method='cvxpy'")
            w, phi, gap, its = _multiplicative_allocation(info, criterion, W, tol, max_iter, ridge, w0)
            total_iterations += its
            if gap > tol:
                _warnings.warn("optimal_allocation did not reach the tolerance %g (gap %.2e) in %d iterations"
                               % (tol, gap, its))
            return w, gap
        if not _sdptools.CVXPY_ENABLED:
            raise ImportError("method='cvxpy' requires cvxpy; `pip install cvxpy`.")
        max_fraction = None
        if max_shots is not None:
            max_fraction = max_shots * info.cost / budget
            if _np.sum(max_fraction) < 1:
                raise ValueError("max_shots is too small to spend the budget on the available candidates")
        w, _ = _cvxpy_allocation(info, criterion, W, ridge, max_fraction, solver_kwargs)
        return w, _np.nan

    if drop_below is None:
        drop_below = 1.0 / budget          # less than one shot
    active = _np.arange(n)
    info = information
    w, gap = solve(info)
    while True:
        shots_active = budget * w / info.cost
        drop = shots_active * info.cost < drop_below * budget
        if min_shots > 0:
            # prune gradually -- at most half of the sub-floor candidates per round, smallest first -- so that
            # the re-solve can raise the others above the floor instead of losing what they measured
            under = _np.nonzero((shots_active < min_shots) & ~drop)[0]
            if len(under) > 0:
                under = under[_np.argsort(shots_active[under])]
                drop[under[:max(1, (len(under) + 1) // 2)]] = True
        if not drop.any() or drop.all():
            break
        w_kept = None
        while True:
            try:
                w_kept, gap_kept = solve(information.subset(active[~drop]), w0=w[~drop])
                break
            except ValueError:
                # the dropped candidates were the only ones informing some direction: drop fewer
                dropped = _np.nonzero(drop)[0]
                if len(dropped) <= 1:
                    break
                drop[:] = False
                drop[dropped[_np.argmin(shots_active[dropped])]] = True
        if w_kept is None:
            if min_shots > 0:
                _warnings.warn("optimal_allocation: min_shots=%g cannot be met without losing identifiability; "
                               "some used circuits receive fewer shots" % min_shots)
            break
        active, info, w, gap = active[~drop], information.subset(active[~drop]), w_kept, gap_kept
    shots = _np.zeros(n)
    shots[active] = budget * w / info.cost
    objective = design_criterion(information, shots, criterion, weight)
    return shots, AllocationResult(objective, float(gap), int(total_iterations), method, int((shots > 0).sum()))


def round_shots(shots, budget=None):
    """
    Round a real-valued shot allocation to integers that sum to `budget` (default: the rounded total), by the
    largest-remainder method: floors first, then the remaining shots go to the largest fractional parts.
    Candidates with zero shots stay at zero.
    """
    shots = _np.asarray(shots, dtype=float)
    if budget is None:
        budget = int(round(shots.sum()))
    budget = int(budget)
    base = _np.floor(shots).astype(int)
    remainder = budget - int(base.sum())
    if remainder < 0:
        raise ValueError("budget is smaller than the sum of the floors of the allocation")
    frac = shots - base
    frac[shots <= 0] = -1.0
    order = _np.argsort(-frac, kind='stable')
    out = base.copy()
    for i in order[:remainder]:
        if shots[i] > 0:
            out[i] += 1
    return out


def resample_frequencies(probability_dicts, shots, seed=None):
    """
    Multinomially sample outcome frequencies: one dictionary per circuit, with `shots` (scalar or per-circuit)
    shots drawn from each circuit's outcome distribution.  Circuits with zero shots get an empty dictionary.
    """
    rng = seed if isinstance(seed, _np.random.Generator) else _np.random.default_rng(seed)
    shots = _np.broadcast_to(_np.asarray(shots), (len(probability_dicts),))
    return [_resample_probability_distribution(d, int(round(s)), rng) if s > 0 else {}
            for d, s in zip(probability_dicts, shots)]


def estimate_error_rates_gls(moments, probability_dict_list, model_parameter_indexing, shots, nominal_rates=None,
                             iterations=2, min_variance=1e-3, stochastic_solver='nnls', error_bars='asymptotic',
                             bootstrap_trials=50, seed=None):
    """
    Estimate the error rates by generalized least squares -- the estimator whose precision
    :func:`predicted_covariance` / :func:`optimal_allocation` predict and optimize.

    Every circuit's vector of observed expectation-value shifts ``y_c = <Q>_obs - <Q>_ideal`` is weighted by
    the inverse of its covariance ``Σ_c / N_c`` (shot count *and* the within-circuit covariance structure,
    including the small variances of observables with ideal value ±1), and the weighted least-squares problem
    ``min Σ_c N_c (y_c - D_c ε)ᵀ Σ_c⁻¹ (y_c - D_c ε)`` is solved with the stochastic rates constrained to be
    non-negative.  Since ``Σ_c`` depends on the rates, the fit is iterated (feasible GLS): the covariance is
    first evaluated at `nominal_rates` (or zero rates), then at the current estimate, `iterations` times.

    Parameters
    ----------
    moments : list of CircuitMoments
        The circuits' moment designs (:func:`create_design_moments`), one per data dictionary.
    probability_dict_list : list
        ``[observed_dicts, ideal_dicts]`` as for :func:`estimate_error_rates` (``ideal_dicts`` may be None).
    model_parameter_indexing : ModelParameterList
    shots : float or array-like
        Shots per circuit (scalar or one entry per circuit).  Circuits with zero shots are ignored.
    nominal_rates : array-like, optional
        Starting point for the covariance (default: zero rates, i.e. ideal statistics floored at `min_variance`).
    iterations : int, optional
        Number of re-weighting iterations after the first solve.
    min_variance : float, optional
    stochastic_solver : {'nnls', 'inversion'}, optional
        Whether the stochastic rates are constrained to be non-negative.
    error_bars : {'asymptotic', 'bootstrap', None}, optional
        ``'asymptotic'``: square roots of the diagonal of the GLS covariance ``(Σ_c N_c D_cᵀ Σ_c⁻¹ D_c)⁻¹`` at the
        final estimate (the Cramér-Rao bound; it ignores the non-negativity constraints, which can only reduce
        the error of stochastic rates near zero).  ``'bootstrap'``: non-parametric bootstrap, re-sampling every
        circuit's outcome distribution with its own shot count and re-solving with the final weights.
    bootstrap_trials : int, optional
    seed : int or numpy.random.Generator, optional

    Returns
    -------
    error_rates : numpy.ndarray
    error_uncertainty : numpy.ndarray or None
    """
    if stochastic_solver not in ('nnls', 'inversion'):
        raise ValueError("Unknown stochastic_solver '%s'" % stochastic_solver)
    moments = list(moments)
    designs = [m.design for m in moments]
    n = len(moments)
    shots = _np.broadcast_to(_np.asarray(shots, dtype=float), (n,)).copy()
    observed_dicts = probability_dict_list[0]
    if len(observed_dicts) != n:
        raise ValueError("Expected %d observed probability dictionaries, got %d" % (n, len(observed_dicts)))
    used = [c for c in range(n) if shots[c] > 0 and len(observed_dicts[c]) > 0]
    if len(used) == 0:
        raise ValueError("No circuit has both shots and data")
    sub = [moments[c] for c in used]
    ideal_dicts = None if probability_dict_list[1] is None else [probability_dict_list[1][c] for c in used]
    y, per_circuit_paulis = observed_expectation_shifts([designs[c] for c in used],
                                                        [[observed_dicts[c] for c in used], ideal_dicts])
    P = len(model_parameter_indexing)
    if sub[0].shape[1] != P:
        raise ValueError("Design matrices have %d columns but %d parameters were given" % (sub[0].shape[1], P))
    types = _np.array(model_parameter_indexing.errorgen_types())
    lb = _np.full(P, -_np.inf)
    if stochastic_solver == 'nnls':
        lb[types == 'S'] = 0.0
    sizes = [m.num_observables for m in sub]
    splits = _np.cumsum(sizes)[:-1]
    sqrt_shots = _np.sqrt(shots[used])

    def whiten(rates):
        rows_A, rows_b = [], []
        for m, yc, s in zip(sub, _np.split(y, splits), sqrt_shots):
            Sigma = m.covariance(rates, min_variance)
            evals, U = _np.linalg.eigh(Sigma)
            scale = s / _np.sqrt(_np.maximum(evals, min_variance))
            rows_A.append((U.T @ m.design.design_matrix) * scale[:, None])
            rows_b.append((U.T @ yc) * scale)
        return _np.vstack(rows_A), _np.concatenate(rows_b)

    def solve(A, b):
        if _np.all(_np.isinf(lb)):
            return _np.linalg.pinv(A) @ b
        return _optimize.lsq_linear(A, b, bounds=(lb, _np.inf)).x

    rates = _np.zeros(P) if nominal_rates is None else _np.asarray(nominal_rates, dtype=float)
    for _ in range(max(int(iterations), 0) + 1):
        A, b = whiten(rates)
        rates = solve(A, b)

    if error_bars is None or error_bars is False:
        return rates, None
    if error_bars == 'asymptotic':
        return rates, _np.sqrt(_np.diag(_pinv_covariance(A, 1e-10)))
    if error_bars != 'bootstrap':
        raise ValueError("Unknown error_bars option '%s'" % str(error_bars))
    rng = seed if isinstance(seed, _np.random.Generator) else _np.random.default_rng(seed)
    # the whitening of the final iteration is kept fixed; only the data are resampled
    ideal_part = y - _np.concatenate([_bulk_Ztype_pauli_exp(p, observed_dicts[c])
                                      for p, c in zip(per_circuit_paulis, used)])
    whiten_ops = []
    for m, s in zip(sub, sqrt_shots):
        Sigma = m.covariance(rates, min_variance)
        evals, U = _np.linalg.eigh(Sigma)
        whiten_ops.append(U.T * (s / _np.sqrt(_np.maximum(evals, min_variance)))[:, None])
    estimates = _np.zeros((int(bootstrap_trials), P))
    for t in range(int(bootstrap_trials)):
        resampled = [_resample_probability_distribution(observed_dicts[c], int(round(shots[c])), rng) for c in used]
        obs = _np.concatenate([_bulk_Ztype_pauli_exp(p, d) for p, d in zip(per_circuit_paulis, resampled)])
        yb = obs + ideal_part
        b = _np.concatenate([op @ yc for op, yc in zip(whiten_ops, _np.split(yb, splits))])
        estimates[t] = solve(A, b)
    return rates, _np.std(estimates, axis=0)


def observed_expectation_shifts(design_matrices, probability_dict_list, measurement_list=None):
    """
    The vector of observed minus ideal Pauli expectation values, stacked over circuits.

    Parameters
    ----------
    design_matrices : list of CircuitDesign or numpy.ndarray
    probability_dict_list : list
        ``[observed_dicts, ideal_dicts]`` where each element is a list (over circuits) of outcome-probability
        dictionaries.  ``ideal_dicts`` may be None when `design_matrices` are :class:`CircuitDesign` objects,
        in which case their `ideal_expectations` are used.
    measurement_list : list, optional
        Observables: a single list of Paulis (used for every circuit) or a list of per-circuit lists.  Not
        needed when `design_matrices` are :class:`CircuitDesign` objects.

    Returns
    -------
    observed_values : numpy.ndarray
    per_circuit_paulis : list of list
    """
    arrays, infos = _as_design_arrays(design_matrices)
    observed_dicts, ideal_dicts = probability_dict_list[0], probability_dict_list[1]
    num_circuits = len(arrays)
    if len(observed_dicts) != num_circuits:
        raise ValueError("Number of observed probability dictionaries (%d) does not match number of design "
                         "matrices (%d)" % (len(observed_dicts), num_circuits))

    per_circuit_paulis = []
    for i in range(num_circuits):
        if measurement_list is not None and len(measurement_list) > 0 and \
           not isinstance(measurement_list[0], (list, tuple)):
            paulis = list(measurement_list)  # a single list of Paulis for all circuits
        elif measurement_list is not None:
            paulis = list(measurement_list[i])
        elif infos[i] is not None and infos[i].paulis is not None:
            paulis = list(infos[i].paulis)
        else:
            raise ValueError("`measurement_list` is required when design matrices are bare arrays.")
        if len(paulis) != arrays[i].shape[0]:
            raise ValueError("Circuit %d: %d observables but design matrix has %d rows"
                             % (i, len(paulis), arrays[i].shape[0]))
        per_circuit_paulis.append(paulis)

    shifts = []
    for i in range(num_circuits):
        obs = _bulk_Ztype_pauli_exp(per_circuit_paulis[i], observed_dicts[i])
        if ideal_dicts is not None:
            ideal = _bulk_Ztype_pauli_exp(per_circuit_paulis[i], ideal_dicts[i])
        elif infos[i] is not None:
            ideal = _np.asarray(infos[i].ideal_expectations, dtype=float)
        else:
            raise ValueError("Ideal probabilities are required when design matrices are bare arrays.")
        shifts.append(obs - ideal)
    return _np.concatenate(shifts) if len(shifts) > 0 else _np.zeros(0), per_circuit_paulis


def solve_linearized_gst(design_matrix, observed_values, errorgen_types, coherent_solver='inversion',
                         stochastic_solver='nnls'):
    """
    Solve ``design_matrix @ rates = observed_values`` for the error rates.

    For error models containing only 'H' and 'S' parameters each row is sensitive to only one of the two
    sectors, so the problem decouples: Hamiltonian rates are obtained by pseudo-inversion and stochastic
    rates by non-negative least squares (or pseudo-inversion).  If 'C'/'A' parameters are present a joint
    bounded least-squares problem is solved instead (stochastic rates constrained to be non-negative when
    `stochastic_solver` is ``'nnls'``).

    Parameters
    ----------
    design_matrix : numpy.ndarray
    observed_values : numpy.ndarray
    errorgen_types : list of str
        The sector ('H', 'S', 'C', 'A') of each column.
    coherent_solver : {'inversion'}
    stochastic_solver : {'nnls', 'inversion'}

    Returns
    -------
    numpy.ndarray
    """
    if coherent_solver not in ('inversion',):
        raise ValueError("Unknown coherent_solver '%s'" % coherent_solver)
    if stochastic_solver not in ('nnls', 'inversion'):
        raise ValueError("Unknown stochastic_solver '%s'" % stochastic_solver)
    types = _np.array(list(errorgen_types))
    rates = _np.zeros(design_matrix.shape[1])
    if design_matrix.shape[1] == 0:
        return rates
    h_cols = _np.nonzero(types == 'H')[0]
    s_cols = _np.nonzero(types == 'S')[0]
    other_cols = _np.nonzero((types != 'H') & (types != 'S'))[0]

    if len(other_cols) == 0:
        if len(h_cols) > 0:
            rates[h_cols] = _np.linalg.pinv(design_matrix[:, h_cols]) @ observed_values
        if len(s_cols) > 0:
            if stochastic_solver == 'nnls':
                rates[s_cols], _ = _optimize.nnls(design_matrix[:, s_cols], observed_values)
            else:
                rates[s_cols] = _np.linalg.pinv(design_matrix[:, s_cols]) @ observed_values
    else:
        if stochastic_solver == 'nnls':
            lb = _np.full(design_matrix.shape[1], -_np.inf)
            lb[s_cols] = 0.0
            res = _optimize.lsq_linear(design_matrix, observed_values, bounds=(lb, _np.inf))
            rates = res.x
        else:
            rates = _np.linalg.pinv(design_matrix) @ observed_values
    return rates


def estimate_error_rates(design_matrices, probability_dict_list, measurement_list=None, error_indexing_list=None,
                         error_bar_params=None, coherent_solver='inversion', stochastic_solver='nnls',
                         error_bars=None, seed=None):
    """
    Estimate the error rates of a sparse error model from circuit data (linearized GST).

    Parameters
    ----------
    design_matrices : list
        One :class:`CircuitDesign` (recommended; from ``create_design_matrix_list(..., return_info=True)``) or
        design-matrix array per circuit.

    probability_dict_list : list
        ``[observed_dicts, ideal_dicts]``: lists (over circuits, in the same order as `design_matrices`) of
        outcome-probability (or frequency) dictionaries.  Outcome labels may be pyGSTi outcome tuples of
        circuits with mid-circuit measurements, e.g. ``('0', '01')``.  ``ideal_dicts`` may be None when
        :class:`CircuitDesign` objects are given (their ideal expectation values are used).

    measurement_list : list, optional
        Z-type Pauli observables: a single list used for every circuit, or a list of per-circuit lists.  Not
        needed for :class:`CircuitDesign` inputs.

    error_indexing_list : ModelParameterList
        The model parameters (from :func:`build_model_parameter_indexing`).

    error_bar_params : list, optional
        ``[bootstrap_trials, shots]`` for ``error_bars='bootstrap'``.

    coherent_solver : {'inversion'}, optional
    stochastic_solver : {'nnls', 'inversion'}, optional
        See :func:`solve_linearized_gst`.

    error_bars : {None, 'bootstrap'}, optional
        If ``'bootstrap'`` a non-parametric bootstrap (multinomial resampling of every circuit's outcome
        distribution with `shots` shots) is used to estimate the standard error of each rate.

    seed : int or numpy.random.Generator, optional
        Random seed for the bootstrap.

    Returns
    -------
    error_rates : numpy.ndarray
        Estimated rates, ordered like `error_indexing_list`.
    error_uncertainty : numpy.ndarray or None
        Bootstrap standard errors (None if `error_bars` is None).
    """
    if error_indexing_list is None:
        raise ValueError("`error_indexing_list` (the model parameter list) is required.")
    arrays, _ = _as_design_arrays(design_matrices)
    observed_values, per_circuit_paulis = observed_expectation_shifts(design_matrices, probability_dict_list,
                                                                      measurement_list)
    design_matrix = _np.vstack(arrays)
    if design_matrix.shape[1] != len(error_indexing_list):
        raise ValueError("Design matrices have %d columns but %d parameters were given"
                         % (design_matrix.shape[1], len(error_indexing_list)))
    types = [p[0].errorgen_type for p in error_indexing_list]
    error_rates = solve_linearized_gst(design_matrix, observed_values, types, coherent_solver, stochastic_solver)

    if error_bars is None or error_bars is False:
        return error_rates, None
    if error_bars != 'bootstrap':
        raise ValueError("Unknown error_bars option '%s'" % str(error_bars))
    if error_bar_params is None or len(error_bar_params) < 2:
        raise ValueError("error_bar_params=[bootstrap_trials, shots] is required for bootstrapped error bars")
    bootstrap_trials, shots = int(error_bar_params[0]), int(error_bar_params[1])
    rng = seed if isinstance(seed, _np.random.Generator) else _np.random.default_rng(seed)
    observed_dicts = probability_dict_list[0]
    # ideal expectation values are fixed; pre-compute the ideal part once
    ideal_part = observed_values - _np.concatenate([_bulk_Ztype_pauli_exp(p, d)
                                                    for p, d in zip(per_circuit_paulis, observed_dicts)])
    estimates = _np.zeros((bootstrap_trials, len(error_indexing_list)))
    for t in range(bootstrap_trials):
        resampled = [_resample_probability_distribution(d, shots, rng) for d in observed_dicts]
        obs = _np.concatenate([_bulk_Ztype_pauli_exp(p, d) for p, d in zip(per_circuit_paulis, resampled)])
        estimates[t, :] = solve_linearized_gst(design_matrix, obs + ideal_part, types, coherent_solver,
                                               stochastic_solver)
    return error_rates, _np.std(estimates, axis=0)


def estimated_rates_dict(model_parameter_indexing, error_rates):
    """
    Arrange estimated rates into a nested ``{key: {(type, pauli[, pauli]): rate}}`` dictionary.

    The inner keys use the explicit-support syntax (``'X:0'``, ``'ZY:0,v'``) so the result can be fed back into
    :func:`build_model_parameter_indexing` or :func:`rate_to_model`.  The two Paulis of a ``'C'`` or ``'A'``
    term are written on the union of their supports (e.g. ``('C', 'ZI:0,1', 'IZ:0,1')``), as required by pyGSTi's
    model constructors.

    Returns
    -------
    OrderedDict
    """
    n = model_parameter_indexing.num_qubits
    labels = list(model_parameter_indexing.qubit_labels)
    vtoken = model_parameter_indexing.virtual_qubit_token
    ret = _collections.OrderedDict()
    for p, rate in zip(model_parameter_indexing, error_rates):
        strings = [str(ps)[1:].replace('_', 'I') for ps in p.errorgen.basis_element_labels]
        support = sorted(set(i for s in strings for i, ch in enumerate(s) if ch != 'I'))
        tokens = [str(labels[i]) if i < n else str(vtoken) for i in support]
        bels = [''.join(s[i] for i in support) + ':' + ','.join(tokens) for s in strings]
        ret.setdefault(p.key, _collections.OrderedDict())[(p.errorgen_type,) + tuple(bels)] = float(rate)
    return ret


def mcm_fomgi_estimates(model_parameter_indexing, error_rates, sectors='auto'):
    """
    Convert the estimated rates of each mid-circuit measurement into named FOMGI quantities.

    Only MCM parameters whose error generators act on the measured qubit and/or the virtual qubit (the
    standard single-qubit gadget) are decomposed; parameters with crosstalk onto other data qubits are skipped.

    Parameters
    ----------
    model_parameter_indexing : ModelParameterList
    error_rates : numpy.ndarray
    sectors : see :func:`~pygsti.errorgenpropagation.mcmgadget.fomgi_decomposition_matrix`

    Returns
    -------
    OrderedDict
        Maps each MCM key to an OrderedDict of FOMGI name -> value.
    """
    n = model_parameter_indexing.num_qubits
    labels = list(model_parameter_indexing.qubit_labels)
    ret = _collections.OrderedDict()
    for key in model_parameter_indexing.keys:
        if key in _SPAM_KEYS or key[0] not in model_parameter_indexing.mcm_gate_names:
            continue
        q_idx = labels.index(key[1])
        gadget_rates = _collections.OrderedDict()
        for j in model_parameter_indexing.indices_for_key(key):
            p = model_parameter_indexing[j]
            bels = []
            ok = True
            for ps in p.errorgen.basis_element_labels:
                s = str(ps)[1:].replace('_', 'I')
                if any(ch != 'I' for i, ch in enumerate(s[:n]) if i != q_idx):
                    ok = False
                bels.append(s[q_idx] + s[n])
            if ok:
                gadget_rates[(p.errorgen_type,) + tuple(bels)] = error_rates[j]
        ret[key] = _mcm.fomgi_quantities(gadget_rates, sectors=sectors)
    return ret


# ---------------------------------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------------------------------

def make_bit_string_vector(bitstring):
    """
    The Pauli-product-basis superket of a computational basis state.

    Parameters
    ----------
    bitstring : str

    Returns
    -------
    numpy.ndarray
    """
    vector_dict = {'0': _np.sqrt(1 / 2) * _np.array([1, 0, 0, 1]),
                   '1': _np.sqrt(1 / 2) * _np.array([1, 0, 0, -1])}
    vector = vector_dict[bitstring[0]]
    for char in bitstring[1:]:
        vector = _np.kron(vector, vector_dict[char])
    return vector


def int_to_bin(integer, qbts):
    """Zero-padded binary string (length `qbts`) of an integer."""
    return format(integer, 'b').zfill(qbts)


def int_to_pauli(integer, qbts):
    """The `integer`-th `qbts`-qubit Pauli string in the ordering I, X, Y, Z (most significant qubit first)."""
    pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    pauli = ''
    temp_int = integer
    while temp_int > 0:
        pauli = pauli_dict[temp_int % 4] + pauli
        temp_int = temp_int // 4
    if len(pauli) < qbts:
        pauli = 'I' * (qbts - len(pauli)) + pauli
    return _stim.PauliString(pauli)


# ---------------------------------------------------------------------------------------------------
# Building pyGSTi models from (estimated) rates -- mostly for simulation and testing
# ---------------------------------------------------------------------------------------------------

def rate_to_model(pspec, lindblad_coeff, rate_ests=None, mcm_gate_names=DEFAULT_MCM_GATE_NAMES,
                  virtual_qubit_token=VIRTUAL_QUBIT_TOKEN, simulator='auto', model_type='cloud',
                  multiqubit_mcm_labels=(), **model_kwargs):
    """
    Build a pyGSTi noise model from a linearized-GST ansatz and (estimated) rates.

    Gate and SPAM error generators are given to :func:`~pygsti.models.modelconstruction.create_cloud_crosstalk_model`
    (or :func:`create_crosstalk_free_model`) and every mid-circuit measurement key is realized by inserting a
    noisy instrument built from the virtual-qubit gadget
    (:func:`~pygsti.errorgenpropagation.mcmgadget.insert_mcm_gadget_instrument`).

    Parameters
    ----------
    pspec : QubitProcessorSpec
    lindblad_coeff : dict
        The ansatz (see :func:`build_model_parameter_indexing`).  Not modified.
    rate_ests : array-like or dict, optional
        Rates ordered like the parameters of `lindblad_coeff` (as enumerated by
        :func:`build_model_parameter_indexing`), or a dict mapping parameter indices to rates.  If None, the rates
        stored in `lindblad_coeff` are used.
    mcm_gate_names : iterable of str, optional
    virtual_qubit_token : str, optional
    simulator : str, optional
        Forward simulator type.  ``'auto'`` selects ``'matrix'`` when the ansatz contains mid-circuit measurements
        (the map simulator does not support instruments in implicit models) and pyGSTi's default otherwise.
    model_type : {'cloud', 'crosstalk-free'}, optional
        Which model constructor to use for the gates.  Crosstalk-free models cannot represent crosstalk terms or
        per-qubit SPAM error generators.
    multiqubit_mcm_labels : iterable of tuple, optional
        Qubit tuples ``(q1, q2, ...)`` for which a simultaneous multi-qubit instrument (the tensor product of the
        single-qubit MCM instruments, keyed ``(mcm_name, q1, q2, ...)``) should also be inserted, so that
        circuits using labels like ``Label('Iz', (q1, q2))`` can be simulated.  Linearized GST itself treats such
        labels as parallel single-qubit MCMs, whose error models must be given under the single-qubit keys.
    **model_kwargs
        Additional arguments for the model constructor.  If the ansatz contains ``'C'`` or ``'A'`` parameters,
        ``lindblad_parameterization`` defaults to ``'GLND'`` so that the error generator is realized exactly as
        given (pyGSTi's default would insist on a completely positive generator, which estimated rates need not
        be); pass ``lindblad_parameterization='CPTPLND'`` to enforce complete positivity instead.

    Returns
    -------
    Model
    """
    from pygsti.models import modelconstruction as _mc

    params, stored_rates = build_model_parameter_indexing(lindblad_coeff, qubit_labels=pspec.qubit_labels,
                                                          mcm_gate_names=mcm_gate_names,
                                                          virtual_qubit_token=virtual_qubit_token)
    if rate_ests is None:
        rates = list(stored_rates)
    elif isinstance(rate_ests, dict):
        rates = [rate_ests[i] for i in range(len(params))]
    else:
        rates = list(_np.asarray(rate_ests, dtype=float))
    if len(rates) != len(params):
        raise ValueError("Expected %d rates but got %d" % (len(params), len(rates)))

    n = params.num_qubits
    labels = list(params.qubit_labels)
    explicit = estimated_rates_dict(params, rates)  # explicit-support syntax, understood by the cloud model
    gate_coeffs = _collections.OrderedDict()
    mcm_coeffs = _collections.OrderedDict()
    for key, errors in explicit.items():
        if key not in _SPAM_KEYS and key[0] in mcm_gate_names:
            mcm_coeffs[key] = errors
        elif model_type == 'crosstalk-free':
            # crosstalk-free models need *local* Pauli strings on the gate's own qubits
            if key in _SPAM_KEYS:
                raise ValueError("Crosstalk-free models only support identical single-qubit SPAM noise on every qubit; "
                                 "use model_type='cloud' for 'prep'/'povm' error generators.")
            gate_qubit_idx = [labels.index(q) for q in key[1:]]
            local = _collections.OrderedDict()
            for j in params.indices_for_key(key):
                p = params[j]
                bels = []
                for ps in p.errorgen.basis_element_labels:
                    s = str(ps)[1:].replace('_', 'I')
                    if any(ch != 'I' for i, ch in enumerate(s[:n]) if i not in gate_qubit_idx):
                        raise ValueError("Parameter %s is a crosstalk term; use model_type='cloud'." % str(p))
                    bels.append(''.join(s[i] for i in gate_qubit_idx))
                local[(p.errorgen_type,) + tuple(bels)] = rates[j]
            gate_coeffs[key] = local
        else:
            gate_coeffs[key] = errors

    has_mcms = len(mcm_coeffs) > 0
    if simulator == 'auto':
        simulator = 'matrix' if has_mcms else 'map'
    if any(p.errorgen_type in ('C', 'A') for p in params):
        # pyGSTi's 'auto' parameterization switches to CPTPLND when C/A terms are present, which insists on a
        # completely positive generator (C^2 + A^2 <= S_P S_Q for every pair).  Ansatz rates -- and in particular
        # *estimated* rates -- need not satisfy this, so realize the generator as given instead.
        model_kwargs.setdefault('lindblad_parameterization', 'GLND')
    if model_type == 'cloud':
        model_kwargs.setdefault('errcomp_type', 'errorgens')
        mdl = _mc.create_cloud_crosstalk_model(pspec, lindblad_error_coeffs=gate_coeffs, simulator=simulator,
                                               **model_kwargs)
    elif model_type == 'crosstalk-free':
        mdl = _mc.create_crosstalk_free_model(pspec, lindblad_error_coeffs=gate_coeffs, simulator=simulator,
                                              **model_kwargs)
    else:
        raise ValueError("Unknown model_type '%s'" % model_type)

    gadget_rates_by_key = _collections.OrderedDict()
    gadget_qubits_by_key = _collections.OrderedDict()
    for key in mcm_coeffs.keys():
        measured_qubit = key[1]
        # gadget data qubits = all data qubits touched by this MCM's error generators (measured qubit first)
        support = set()
        for j in params.indices_for_key(key):
            for ps in params[j].errorgen.basis_element_labels:
                s = str(ps)[1:].replace('_', 'I')
                support.update(i for i, ch in enumerate(s[:n]) if ch != 'I')
        gadget_idx = [labels.index(measured_qubit)] + sorted(i for i in support if i != labels.index(measured_qubit))
        gadget_rates = _collections.OrderedDict()
        for j in params.indices_for_key(key):
            p = params[j]
            bels = []
            for ps in p.errorgen.basis_element_labels:
                s = str(ps)[1:].replace('_', 'I')
                bels.append(''.join(s[i] for i in gadget_idx) + s[n])
            gadget_rates[(p.errorgen_type,) + tuple(bels)] = rates[j]
        gadget_rates_by_key[key] = gadget_rates
        gadget_qubits_by_key[key] = [labels[i] for i in gadget_idx]
        _mcm.insert_mcm_gadget_instrument(mdl, measured_qubit, gadget_rates, gadget_qubits=gadget_qubits_by_key[key],
                                          mcm_gate_name=key[0])

    for qubits in multiqubit_mcm_labels:
        qubits = tuple(qubits)
        for name in mcm_gate_names:
            rates_by_qubit = {q: gadget_rates_by_key.get((name, q), {}) for q in qubits}
            gq_by_qubit = {q: gadget_qubits_by_key.get((name, q), [q]) for q in qubits}
            _mcm.insert_mcm_gadget_instrument(mdl, qubits, rates_by_qubit, gadget_qubits=gq_by_qubit,
                                              mcm_gate_name=name)
    return mdl

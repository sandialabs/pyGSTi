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
           'observed_expectation_shifts', 'solve_linearized_gst', 'estimate_error_rates', 'estimated_rates_dict',
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
    :func:`build_model_parameter_indexing` or :func:`rate_to_model`.

    Returns
    -------
    OrderedDict
    """
    n = model_parameter_indexing.num_qubits
    labels = list(model_parameter_indexing.qubit_labels)
    vtoken = model_parameter_indexing.virtual_qubit_token
    ret = _collections.OrderedDict()
    for p, rate in zip(model_parameter_indexing, error_rates):
        bels = []
        for ps in p.errorgen.basis_element_labels:
            s = str(ps)[1:].replace('_', 'I')
            support = [i for i, ch in enumerate(s) if ch != 'I']
            pauli = ''.join(s[i] for i in support)
            tokens = [str(labels[i]) if i < n else str(vtoken) for i in support]
            bels.append(pauli + ':' + ','.join(tokens))
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
        Additional arguments for the model constructor.

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

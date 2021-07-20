#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
""" Idle Tomography utility routines """

import itertools as _itertools

import numpy as _np

from . import pauliobjs as _pobjs
from ... import tools as _tools
from ...modelmembers import operations as _op
from ...circuits import cloudcircuitconstruction as _nqn


# maybe need to restructure in future - "tools" usually doesn't import "objects"


def alloutcomes(prep, meas, maxweight):
    """
    Lists every "error bit string" that co1uld be caused by an error of weight
    up to `maxweight` when performing prep & meas (must be in same basis, but may
    have different signs).

    Parameters
    ----------
    prep, meas : NQPauliState

    maxweight : int

    Returns
    -------
    list
        A list of :class:`NQOutcome` objects.
    """
    if not (0 < maxweight <= 2): raise NotImplementedError("Only maxweight <= 2 is currently supported")
    assert(prep.rep == meas.rep), "`prep` and `meas` must specify the same basis!"
    expected = ["0" if s1 == s2 else "1" for s1, s2 in zip(prep.signs, meas.signs)]
    #whether '0' or '1' outcome is expected, i.e. what is an "error"

    N = len(prep)  # == len(meas)
    eoutcome = _pobjs.NQOutcome(''.join(expected))
    if maxweight == 1:
        return [eoutcome.flip(i) for i in range(N)]
    else:
        return [eoutcome.flip(i) for i in range(N)] + \
               [eoutcome.flip(i, j) for i in range(N) for j in range(i + 1, N)]


def allerrors(nqubits, maxweight):
    """
    Lists every Pauli error operator for `nqubits` qubits with weight <= `maxweight`

    Parameters
    ----------
    nqubits, maxweight : int

    Returns
    -------
    list
        A list of :class:`NQPauliOp` objects.
    """
    if not (0 < maxweight <= 2): raise NotImplementedError("Only maxweigth <= 2 is currently supported")
    if maxweight == 1:
        return [_pobjs.NQPauliOp.weight_1_pauli(nqubits, loc, p) for loc in range(nqubits) for p in range(3)]
    else:
        return [_pobjs.NQPauliOp.weight_1_pauli(nqubits, loc, p) for loc in range(nqubits) for p in range(3)] + \
               [_pobjs.NQPauliOp.weight_2_pauli(nqubits, loc1, loc2, p1, p2) for loc1 in range(nqubits)
                for loc2 in range(loc1 + 1, nqubits)
                for p1 in range(3) for p2 in range(3)]


def allobservables(meas, maxweight):
    """
    Lists every weight <= `maxweight` observable whose expectation value can be
    extracted from the local Pauli measurement described by `meas`.

    Parameters
    ----------
    meas : NQPauliState

    maxweight : int

    Returns
    -------
    list
        A list of :class:`NQPauliOp` objects.
    """
    if not (0 < maxweight <= 2): raise NotImplementedError("Only maxweight <= 2 is currently supported")
    #Note: returned observables always have '+' sign (i.e. .sign == +1).  We're
    # not interested in meas.signs - this is take into account when we compute the
    # expectation value of our observable given a prep & measurement fiducial.
    if maxweight == 1:
        return [_pobjs.NQPauliOp(meas.rep).subpauli([i]) for i in range(len(meas))]
    else:
        return [_pobjs.NQPauliOp(meas.rep).subpauli([i]) for i in range(len(meas))] + \
               [_pobjs.NQPauliOp(meas.rep).subpauli([i, j]) for i in range(len(meas)) for j in range(i + 1, len(meas))]


def tile_pauli_fidpairs(base_fidpairs, nqubits, maxweight):
    """
    Tiles a set of base fiducial pairs on `maxweight` qubits to a
    set of fiducial pairs on `nqubits` qubits such that every set
    of `maxweight` qubits takes on the values in each base pair in
    at least one of the returned pairs.

    Parameters
    ----------
    base_fidpairs : list
        A list of 2-tuples of :class:`NQPauliState` objects (on `maxweight`
        qubits).

    nqubits : int
        The number of qubits.

    maxweight : int
        The maximum weight errors the base qubits are meant to
        detect.  Equal to the number of qubits in the base pairs.

    Returns
    -------
    list
        A list of 2-tuples of :class:`NQPauliState` objects (on `nqubits`
        qubits).
    """
    nqubit_fidpairs = []
    tmpl = _nqn.get_kcoverage_template(nqubits, maxweight)
    for base_prep, base_meas in base_fidpairs:
        for tmpl_row in tmpl:
            #Replace 0...weight-1 integers in tmpl_row with Pauli basis
            # designations (e.g. +X) to construct NQPauliState objects.
            prep = _pobjs.NQPauliState([base_prep.rep[i] for i in tmpl_row],
                                       [base_prep.signs[i] for i in tmpl_row])
            meas = _pobjs.NQPauliState([base_meas.rep[i] for i in tmpl_row],
                                       [base_meas.signs[i] for i in tmpl_row])
            nqubit_fidpairs.append((prep, meas))

    _tools.remove_duplicates_in_place(nqubit_fidpairs)
    return nqubit_fidpairs


# ----------------------------------------------------------------------------
# Testing tools (only used in testing, not for running idle tomography)
# ----------------------------------------------------------------------------

def nontrivial_paulis(wt):
    """
    List all nontrivial paulis of given weight `wt`.

    Parameters
    ----------
    wt : int

    Returns
    -------
    list
        A list of tuples of 'X', 'Y', and 'Z', e.g. `('X','Z')`.
    """
    ret = []
    for tup in _itertools.product(*([['X', 'Y', 'Z']] * wt)):
        ret.append(tup)
    return ret


def set_idle_errors(nqubits, model, errdict, rand_default=None,
                    hamiltonian=True, stochastic=True, affine=True):
    """
    Set specific or random error terms (typically for a data-generating model)
    within a noise model (a :class:`CloudNoiseModel` object).

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    model : CloudNoiseModel
        The model, to set the idle errors of.

    errdict : dict
        A dictionary of errors to include.  Keys are `"S(<>)"`, `"H(<>)"`, and
        `"A(<>)"` where <> is a string of 'X','Y','Z',and 'I' (e.g. `"S(XIZ)"`)
        and values are floating point error rates.

    rand_default : float or numpy array, optional
        Random error rates to insert into values not specified in `errdict`.
        If a floating point number, a random value between 0 and `rand_default`
        is used.  If an array, then values are taken directly and sequentially
        from this array (typically of random rates).  The array must be long
        enough to provide values for all unspecified rates.

    hamiltonian, stochastic, affine : bool, optional
        Whether `model` includes Hamiltonian, Stochastic, and/or Affine
        errors (e.g. if the model was built with "H+S" parameterization,
        then only `hamiltonian` and `stochastic` should be set to True).

    Returns
    -------
    numpy.ndarray
        The random rates the were used.
    """
    rand_rates = []; i_rand_default = 0
    v = model.to_vector()
    #assumes Implicit model w/'globalIdle' as a composed gate...
    # each factor applies to some set of the qubits (of size 1 to the max-error-weight)
    global_idle_lbl = model.processor_spec.global_idle_layer_label
    global_idle = model.circuit_layer_operator(global_idle_lbl, typ='op')
    factorops = global_idle.factorops if isinstance(global_idle, _op.ComposedOp) else (global_idle,)
    for i, factor in enumerate(factorops):
        #print("Factor %d: target = %s, gpindices=%s" % (i,str(factor.targetLabels),str(factor.gpindices)))
        if isinstance(factor, _op.EmbeddedOp):
            experrgen_op = factor.embedded_op
            targetLabels = factor.target_labels
        else:
            experrgen_op = factor
            targetLabels = model.state_space.qubit_labels

        assert(isinstance(experrgen_op, _op.ExpErrorgenOp)), \
            "Expected idle op to be a composition of possibly embedded exp(errorgen) gates!"

        sub_v = v[factor.gpindices]
        bsH = experrgen_op.errorgen.ham_basis_size
        bsO = experrgen_op.errorgen.other_basis_size
        if hamiltonian: hamiltonian_sub_v = sub_v[0:bsH - 1]  # -1s b/c bsH, bsO include identity in basis
        if stochastic: stochastic_sub_v = sub_v[bsH - 1:bsH - 1 + bsO - 1]
        if affine: affine_sub_v = sub_v[bsH - 1 + bsO - 1:bsH - 1 + 2 * (bsO - 1)]

        for k, tup in enumerate(nontrivial_paulis(len(targetLabels))):
            lst = ['I'] * nqubits
            for ii, i in enumerate(targetLabels):
                indx = i if isinstance(i, int) else int(i[1:])  # i is something like "Q0" so int(i[1:]) extracts the 0
                lst[indx] = tup[ii]
            label = "".join(lst)

            if "S(%s)" % label in errdict:
                Srate = errdict["S(%s)" % label]
            elif rand_default is None:
                Srate = 0.0
            elif isinstance(rand_default, float):
                Srate = rand_default * _np.random.random()
                rand_rates.append(Srate)
            else:  # assume rand_default is array-like, and gives default rates
                Srate = rand_default[i_rand_default]
                i_rand_default += 1

            if "H(%s)" % label in errdict:
                Hrate = errdict["H(%s)" % label]
            elif rand_default is None:
                Hrate = 0.0
            elif isinstance(rand_default, float):
                Hrate = rand_default * _np.random.random()
                rand_rates.append(Hrate)
            else:  # assume rand_default is array-like, and gives default rates
                Hrate = rand_default[i_rand_default]
                i_rand_default += 1

            if "A(%s)" % label in errdict:
                Arate = errdict["A(%s)" % label]
            elif rand_default is None:
                Arate = 0.0
            elif isinstance(rand_default, float):
                Arate = rand_default * _np.random.random()
                rand_rates.append(Arate)
            else:  # assume rand_default is array-like, and gives default rates
                Arate = rand_default[i_rand_default]
                i_rand_default += 1

            if hamiltonian: hamiltonian_sub_v[k] = Hrate
            if stochastic: stochastic_sub_v[k] = _np.sqrt(Srate)  # b/c param gets squared
            if affine: affine_sub_v[k] = Arate

    model.from_vector(v)
    return _np.array(rand_rates, 'd')  # the random rates that were chosen (to keep track of them for later)


def get_idle_errors(nqubits, model, hamiltonian=True, stochastic=True, affine=True, scale_for_idt=True):
    """
    Get error rates on the global idle operation withina :class:`CloudNoiseModel` object.

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    model : CloudNoiseModel
        The model, to get the idle errors of.

    hamiltonian, stochastic, affine : bool, optional
        Whether `model` includes Hamiltonian, Stochastic, and/or Affine
        errors (e.g. if the model was built with "H+S" parameterization,
        then only `hamiltonian` and `stochastic` should be set to True).

    scale_for_idt : bool, optional
        Whether rates should be scaled to match the intrinsic rates
        output by idle tomography.  If `False`, then the rates are
        simply the coefficients of corresponding terms in the
        error generator.

    Returns
    -------
    hamiltonian_rates, stochastic_rates, affine_rates : dict
        Dictionaries of error rates.  Keys are Pauli labels of length `nqubits`,
        e.g. `"XIX"`, `"IIX"`, `"XZY"`.  Only nonzero rates are returned.
    """
    ham_rates = {}
    sto_rates = {}
    aff_rates = {}
    v = model.to_vector()
    #assumes Implicit model w/'globalIdle' as a composed gate...
    idleop = model.circuit_layer_operator(model.processor_spec.global_idle_layer_label, 'op')
    for i, factor in enumerate(idleop.factorops):
        # each factor applies to some set of the qubits (of size 1 to the max-error-weight)

        #print("Factor %d: target = %s, gpindices=%s" % (i,str(factor.targetLabels),str(factor.gpindices)))
        assert(isinstance(factor, _op.EmbeddedOp)), "Expected Gi to be a composition of embedded gates!"
        sub_v = v[factor.gpindices]
        bsH = factor.embedded_op.errorgen.ham_basis_size
        bsO = factor.embedded_op.errorgen.other_basis_size
        if hamiltonian: hamiltonian_sub_v = sub_v[0:bsH - 1]  # -1s b/c bsH, bsO include identity in basis
        if stochastic: stochastic_sub_v = sub_v[bsH - 1:bsH - 1 + bsO - 1]
        if affine: affine_sub_v = sub_v[bsH - 1 + bsO - 1:bsH - 1 + 2 * (bsO - 1)]

        nTargetQubits = len(factor.targetLabels)

        for k, tup in enumerate(nontrivial_paulis(len(factor.targetLabels))):
            lst = ['I'] * nqubits
            for ii, i in enumerate(factor.targetLabels):
                lst[int(i[1:])] = tup[ii]  # i is something like "Q0" so int(i[1:]) extracts the 0
            label = "".join(lst)

            #For explanation of why `scale` is set as it is, see comments in
            # the `predicted_intrinsic_rates(...)` function.
            if hamiltonian and abs(hamiltonian_sub_v[k]) > 1e-6:
                scale = _np.sqrt(2**(2 - nTargetQubits)) if scale_for_idt else 1.0
                ham_rates[label] = hamiltonian_sub_v[k] * scale
            if stochastic and abs(stochastic_sub_v[k]) > 1e-6:
                scale = 1. / (2**nTargetQubits) if scale_for_idt else 1.0
                sto_rates[label] = stochastic_sub_v[k]**2 * scale
            if affine and abs(affine_sub_v[k]) > 1e-6:
                scale = 1. / (_np.sqrt(2)**nTargetQubits) if scale_for_idt else 1.0
                aff_rates[label] = affine_sub_v[k] * scale

    return ham_rates, sto_rates, aff_rates


def predicted_intrinsic_rates(nqubits, maxweight, model,
                              hamiltonian=True, stochastic=True, affine=True):
    """
    Get the exact intrinsic rates that would be produced by simulating `model`
    (for comparison with idle tomography results).

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    maxweight : int, optional
        The maximum weight of errors to consider.

    model : CloudNoiseModel
        The model to extract intrinsic error rates from.

    hamiltonian, stochastic, affine : bool, optional
        Whether `model` includes Hamiltonian, Stochastic, and/or Affine
        errors (e.g. if the model was built with "H+S" parameterization,
        then only `hamiltonian` and `stochastic` should be set to True).

    Returns
    -------
    ham_intrinsic_rates, sto_intrinsic_rates, aff_intrinsic_rates : numpy.ndarray
        Arrays of intrinsic rates.  None if corresponding `hamiltonian`,
        `stochastic` or `affine` is set to False.
    """
    error_labels = [str(pauliOp.rep) for pauliOp in allerrors(nqubits, maxweight)]
    #v = model.to_vector()

    if hamiltonian:
        ham_intrinsic_rates = _np.zeros(len(error_labels), 'd')
    else: ham_intrinsic_rates = None

    if stochastic:
        sto_intrinsic_rates = _np.zeros(len(error_labels), 'd')
    else: sto_intrinsic_rates = None

    if affine:
        aff_intrinsic_rates = _np.zeros(len(error_labels), 'd')
    else: aff_intrinsic_rates = None

    # assumes this is a composed op of embedded lindblad ops
    idleop = model.circuit_layer_operator(model.processor_spec.global_idle_layer_label, 'op')
    factorops = idleop.factorops if isinstance(idleop, _op.ComposedOp) else (idleop,)
    for i, factor in enumerate(factorops):
        #print("Factor %d: target = %s, gpindices=%s" % (i,str(factor.targetLabels),str(factor.gpindices)))

        if isinstance(factor, _op.EmbeddedOp):
            experrgen_op = factor.embedded_op
            targetLabels = factor.target_labels
        else:
            experrgen_op = factor
            targetLabels = model.state_space.qubit_labels

        assert(isinstance(experrgen_op, _op.ExpErrorgenOp)), \
            "Expected idle op to be a composition of possibly embedded exp(errorgen) gates!"
            
        errgen_coeffs = experrgen_op.errorgen_coefficients()
        nTargetQubits = len(targetLabels)

        #OLD - before get_errgen_coeffs
        #sub_v = v[factor.gpindices]
        #bsH = factor.embedded_op.errorgen.ham_basis_size
        #bsO = factor.embedded_op.errorgen.other_basis_size
        #if hamiltonian: hamiltonian_sub_v = sub_v[0:bsH - 1]  # -1s b/c bsH, bsO include identity in basis
        #if stochastic: stochastic_sub_v = sub_v[bsH - 1:bsH - 1 + bsO - 1]
        #if affine: affine_sub_v = sub_v[bsH - 1 + bsO - 1:bsH - 1 + 2 * (bsO - 1)]

        for k, tup in enumerate(nontrivial_paulis(len(targetLabels))):
            lst = ['I'] * nqubits
            for ii, i in enumerate(targetLabels):
                indx = i if isinstance(i, int) else int(i[1:])  # i is something like "Q0" so int(i[1:]) extracts the 0
                lst[indx] = tup[ii]
            label = "".join(lst)  # label on *all* qubits (with 'I's)
            P = ''.join(tup)  # nontrivial pauli on target qubits (no 'I's)

            result_index = error_labels.index(label)
            if hamiltonian and ('H', P) in errgen_coeffs:
                ham_intrinsic_rates[result_index] = errgen_coeffs[('H', P)]
            if stochastic and ('S', P) in errgen_coeffs:
                sto_intrinsic_rates[result_index] = errgen_coeffs[('S', P)]
            if affine and ('A', P) in errgen_coeffs:
                scale = 1 / (_np.sqrt(2)**nTargetQubits)  # not exactly sure how this is derived
                aff_intrinsic_rates[result_index] = errgen_coeffs[('A', P)] * scale

            #TODO REMOVE (OLD)
            #if stochastic: sval = stochastic_sub_v[k]
            #if hamiltonian: hval = hamiltonian_sub_v[k]
            #if affine: aval = affine_sub_v[k]
            #
            #nTargetQubits = len(factor.targetLabels)
            #
            #if stochastic:
            #    # each Stochastic term has two Paulis in it (on either side of rho), each of which is
            #    # scaled by 1/sqrt(d), so 1/d in total, where d = 2**nqubits
            #    sscaled_val = sval**2 / (2**nTargetQubits)  # val**2 b/c it's a *stochastic* term parameter
            #
            #if hamiltonian:
            #    # each Hamiltonian term, to fix missing scaling factors in Hamiltonian jacobian
            #    # elements, needs a sqrt(d) for each trivial ('I') Pauli... ??
            #    hscaled_val = hval * _np.sqrt(2**(2 - nTargetQubits))  # TODO: figure this out...
            #    # 1Q: sqrt(2)
            #    # 2Q: nqubits-targetqubits (sqrt(2) on 1Q)
            #    # 4Q: sqrt(2)**-2
            #
            #if affine:
            #    ascaled_val = aval * 1 / (_np.sqrt(2)**nTargetQubits)  # not exactly sure how this is derived
            #    # 1Q: sqrt(2)/6
            #    # 2Q: 1/3 * 10-2
            #
            #result_index = error_labels.index(label)
            #if hamiltonian: ham_intrinsic_rates[result_index] = hscaled_val
            #if stochastic: sto_intrinsic_rates[result_index] = sscaled_val
            #if affine: aff_intrinsic_rates[result_index] = ascaled_val

    return ham_intrinsic_rates, sto_intrinsic_rates, aff_intrinsic_rates


def predicted_observable_rates(idtresults, typ, nqubits, maxweight, model):
    """
    Get the exact observable rates that would be produced by simulating
    `model` (for comparison with idle tomography results).

    Parameters
    ----------
    idtresults : IdleTomographyResults
        The idle tomography results object used to determing which observable
        rates should be computed, and the provider of the Jacobian relating
        the intrinsic rates internal to `model` to these observable rates.

    typ : {"samebasis","diffbasis"}
        The type of observable rates to predict and return.

    nqubits : int
        The number of qubits.

    maxweight : int
        The maximum weight of errors to consider.

    model : CloudNoiseModel
        The noise model to extract error rates from.

    Returns
    -------
    rates : dict
        A dictionary of the form: `rate = rates[pauli_fidpair][obsORoutcome]`,
        to match the structure of an IdleTomographyResults object's
        `
    """
    intrinsic = None
    ret = {}

    if typ == "samebasis":

        Ne = len(idtresults.error_list)
        for fidpair, dict_of_infos in zip(idtresults.pauli_fidpairs[typ],
                                          idtresults.observed_rate_infos[typ]):
            ret[fidpair] = {}
            for obsORoutcome, info_dict in dict_of_infos.items():
                #Get jacobian row and compute predicted observed rate
                Jrow = info_dict['jacobian row']

                if intrinsic is None:
                    # compute intrinsic (wait for jac row to check length)
                    affine = bool(len(Jrow) == 2 * Ne)  # affine included?
                    _, sto_intrinsic_rates, aff_intrinsic_rates = \
                        predicted_intrinsic_rates(nqubits, maxweight, model, False, True, affine)
                    intrinsic = _np.concatenate([sto_intrinsic_rates, aff_intrinsic_rates]) \

                predicted_rate = _np.dot(Jrow, intrinsic)
                ret[fidpair][obsORoutcome] = predicted_rate

    elif typ == "diffbasis":

        # J_ham * Hintrinsic = observed_rates - J_aff * Aintrinsic
        # so: observed_rates = J_ham * Hintrinsic + J_aff * Aintrinsic
        for fidpair, dict_of_infos in zip(idtresults.pauli_fidpairs[typ],
                                          idtresults.observed_rate_infos[typ]):
            ret[fidpair] = {}
            for obsORoutcome, info_dict in dict_of_infos.items():
                #Get jacobian row and compute predicted observed rate
                Jrow = info_dict['jacobian row']

                if intrinsic is None:
                    # compute intrinsic (wait for jac row to check for affine)
                    affine = bool('affine jacobian row' in info_dict)
                    ham_intrinsic_rates, _, aff_intrinsic_rates = \
                        predicted_intrinsic_rates(nqubits, maxweight, model, True, False, affine)

                predicted_rate = _np.dot(Jrow, ham_intrinsic_rates)
                if 'affine jacobian row' in info_dict:
                    affJrow = info_dict['affine jacobian row']
                    predicted_rate += _np.dot(affJrow, aff_intrinsic_rates)
                ret[fidpair][obsORoutcome] = predicted_rate

    else:
        raise ValueError("Unknown `typ` argument: %s" % typ)

    return ret

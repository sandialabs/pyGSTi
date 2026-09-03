"""
Experiment design for character gate set tomography (cGST) of arbitrary finite-order gate sets
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools
import math as _math
import warnings as _warnings

import numpy as _np

from pygsti.algorithms import germselection as _germsel
from pygsti.circuits import Circuit as _Circuit
from pygsti.circuits import circuitconstruction as _cc
from pygsti.tools import basistools as _bt
from pygsti.tools import chartools as _ct

# This module builds cGST experiment designs for *arbitrary* gate sets, as opposed
# to the hand-built one-qubit {S, sqrt(Y)} design of
# :func:`pygsti.protocols.cgst.create_1q_szy_cgst_design`.  The pipeline is
#
#   1. enumerate germ candidates whose IDEAL superoperator has finite order
#      (only such germs generate a finite cyclic group and hence admit character
#      filtering) -- :func:`finite_order_candidate_germs`;
#   2. run pyGSTi's ordinary germ-selection machinery restricted to those
#      candidates, so the resulting germ set is amplificationally complete
#      -- :func:`find_cgst_germs`;
#   3. for each germ, work out which irreps of its cyclic group its ideal
#      superoperator supports and with what multiplicity
#      -- :func:`germ_irreps` / :func:`nonconjugate_irreps`;
#   4. for each (germ, irrep), choose fiducial circuits that maximize the ideal
#      decaying signal -- :func:`select_cgst_fiducials`;
#   5. assemble one :class:`~pygsti.protocols.cgst.CharacterGSTGermDesign` per
#      (germ, irrep, fiducial pair) -- :func:`create_cgst_design`.


# -----------------------------------------------------------------------------
# Germ candidates and germ selection
# -----------------------------------------------------------------------------

def finite_order_candidate_germs(target_model, max_length, max_order=24, tol=1e-8,
                                 qubit_labels=None):
    """
    All finite-order germ candidates of a target model up to a maximum length.

    Candidate circuits are all words in the target model's primitive operation
    labels of length 1 to `max_length`, deduplicated up to cyclic rotation and
    up to powers (see
    :func:`pygsti.circuits.circuitconstruction.list_all_circuits_without_powers_and_cycles`),
    keeping only those whose *ideal* superoperator has finite order at least 2
    and at most `max_order`.  Only finite-order germs generate a finite cyclic
    group, which is what cGST's character filtering requires; order-1 germs
    (an idle gate, or any word whose ideal product is the identity) generate
    the trivial group, whose only irrep is the trivial one with multiplicity
    equal to the full superoperator dimension -- a matrix-valued decay that the
    cGST analysis does not support -- so they are excluded too.  An idle's
    errors are amplified instead by the finite-order germs that *contain* it
    (e.g. `Gzpi2 Gi`).

    Parameters
    ----------
    target_model : Model
        The ideal model.  Must support `sim.product` (e.g. an
        :class:`~pygsti.models.ExplicitOpModel` with a matrix forward
        simulator).

    max_length : int
        The maximum germ length (number of circuit layers) to consider.

    max_order : int, optional
        The largest cyclic-group order a germ is allowed to generate.  Germs
        whose ideal superoperator has no power up to this one equal to the
        identity are discarded.

    tol : float, optional
        Tolerance used when testing whether a power of the germ's
        superoperator is the identity.

    qubit_labels : tuple, optional
        If not None, the line labels given to the returned circuits.  By
        default the line labels implied by the model's operation labels are
        used.

    Returns
    -------
    list of Circuit
    """
    op_labels = list(target_model.primitive_op_labels)
    candidates = _cc.list_all_circuits_without_powers_and_cycles(op_labels, max_length)
    if qubit_labels is not None:
        candidates = [_Circuit(c.layertup, line_labels=tuple(qubit_labels)) for c in candidates]

    finite = []
    for circuit in candidates:
        try:
            order = _ct.germ_group_order(target_model.sim.product(circuit), max_order=max_order,
                                         tol=tol)
        except ValueError:
            continue  # infinite order (or order > max_order): not a cGST germ
        if order < 2:
            continue  # ideal identity: no nontrivial characters to filter on
        finite.append(circuit)
    return finite


def _dedupe_candidate_germs(target_model, candidates, tol=1e-6):
    """
    Drop candidate germs that are interchangeable for germ selection.

    Two candidates are treated as duplicates only if their ideal superoperators
    agree *and* they contain the same multiset of gate labels.  Deduplicating
    on the superoperator alone (as
    :func:`~pygsti.algorithms.germselection.clean_germ_list` does) is wrong
    whenever different gates share an ideal action -- with an idle `Gi` in the
    gate set, `Gzpi2 Gi` would be discarded as a duplicate of `Gzpi2` although
    only the former amplifies the idle's errors -- and leaves no
    amplificationally complete candidate set at all.
    """
    kept, keys = [], []
    for circuit in candidates:
        superop = _np.asarray(target_model.sim.product(circuit))
        content = tuple(sorted(str(lbl) for lbl in circuit.layertup))
        if any(content == c and _np.linalg.norm(superop - m) < tol for c, m in keys):
            continue
        kept.append(circuit)
        keys.append((content, superop))
    return kept


def find_cgst_germs(target_model, max_length=7, max_order=24, force='singletons',
                    randomization_strength=1e-3, num_gs_copies=3, seed=None,
                    verbosity=0, **kwargs):
    """
    An amplificationally complete germ set drawn from finite-order germs only.

    This is :func:`pygsti.algorithms.germselection.find_germs` with its
    candidate list replaced by the finite-order candidates of
    :func:`finite_order_candidate_germs`, deduplicated by ideal superoperator
    *and* gate content (unlike `find_germs`'s own superoperator-only
    deduplication, which discards every germ that amplifies an idle gate's
    errors, so that `find_germs` itself fails on gate sets with an idle).

    Parameters
    ----------
    target_model : Model
        The ideal model.  A matrix forward simulator is required; a copy with
        `sim='matrix'` is made automatically if needed.

    max_length : int, optional
        The maximum candidate germ length.

    max_order : int, optional
        The largest cyclic-group order a candidate germ may generate.

    force : str or list, optional
        Germs that must appear in the returned set.  The special value
        `'singletons'` means "all length-1 candidates"; note that, unlike in
        :func:`~pygsti.algorithms.germselection.find_germs`, it is expanded
        against the *finite-order* candidates, so a gate that is not of finite
        order is never forced into the germ set.

    randomization_strength : float, optional
        The size of the random unitary perturbations used to build the model
        ensemble the germ set must amplify.

    num_gs_copies : int, optional
        The number of randomized model copies in that ensemble.

    seed : int, optional
        Seed for the random unitary perturbations.

    verbosity : int, optional
        Verbosity level passed to
        :func:`~pygsti.algorithms.germselection.find_germs`.

    Returns
    -------
    list of Circuit

    Raises
    ------
    ValueError
        If there are no finite-order candidates, or if no amplificationally
        complete germ set exists among them (in which case
        `germselection` aborts its search; the message suggests raising
        `max_length` / `max_order`).

    Other Parameters
    ----------------
    kwargs : dict
        Additional keyword arguments forwarded to
        :func:`~pygsti.algorithms.germselection.find_germs` (e.g. `algorithm`,
        `algorithm_kwargs`, `num_nongauge_params`).
    """
    candidates = finite_order_candidate_germs(target_model, max_length, max_order=max_order)
    if len(candidates) == 0:
        raise ValueError("No finite-order germ candidates of length <= %d (max_order=%d)!"
                         % (max_length, max_order))
    candidates = _dedupe_candidate_germs(target_model, candidates)

    if isinstance(force, str) and force == 'singletons':
        force = [c for c in candidates if len(c) == 1]

    algorithm_kwargs = dict(kwargs.pop('algorithm_kwargs', None) or {})
    algorithm_kwargs.setdefault('germs_list', candidates)

    # candidate_germ_counts={1: 'all upto'} keeps find_germs from building a
    # (discarded) candidate list of its own; `germs_list` is what is used.
    kwargs.setdefault('candidate_germ_counts', {1: 'all upto'})
    germs = _germsel.find_germs(target_model, randomize=True,
                                randomization_strength=randomization_strength,
                                num_gs_copies=num_gs_copies, seed=seed, force=force,
                                algorithm_kwargs=algorithm_kwargs, verbosity=verbosity,
                                **kwargs)
    if germs is None or len(germs) == 0:
        # germselection prints "Complete initial candidate germ set FAILS ... Aborting
        # search." and returns an empty list when even the full candidate set is not
        # amplificationally complete.
        raise ValueError("No amplificationally complete cGST germ set exists among the %d "
                         "finite-order candidate germs of length <= %d with group order <= %d "
                         "(pyGSTi's germ selection aborted because the complete candidate set "
                         "itself fails to amplify every non-gauge parameter).  Try raising "
                         "`max_length` and/or `max_order`, or supply germs explicitly."
                         % (len(candidates), max_length, max_order))
    return list(germs)


# -----------------------------------------------------------------------------
# Irreps of a germ
# -----------------------------------------------------------------------------

def germ_irreps(target_model, germ, tol=1e-8, max_order=24):
    """
    The cyclic-group order and irrep multiplicities of an ideal germ.

    Parameters
    ----------
    target_model : Model
        The ideal model.

    germ : Circuit
        The germ circuit.

    tol : float, optional
        Tolerance used for the group-order test and for matching eigenvalues
        to characters.

    max_order : int, optional
        The largest cyclic-group order to test for.

    Returns
    -------
    order : int
        The order of the cyclic group generated by the ideal germ.

    multiplicities : dict
        Maps irrep index to the dimension of the corresponding eigenspace of
        the ideal germ's superoperator (see
        :func:`pygsti.tools.chartools.germ_irrep_multiplicities`).  The trivial
        irrep's multiplicity includes the non-decaying identity direction.
    """
    superop = target_model.sim.product(germ)
    order = _ct.germ_group_order(superop, max_order=max_order, tol=tol)
    return order, _ct.germ_irrep_multiplicities(superop, order, tol=tol)


def nonconjugate_irreps(order, multiplicities):
    """
    One representative of each complex-conjugate pair of irreps.

    A germ's Ramsey-type decays on irreps `j` and `order - j` are complex
    conjugates of one another and so carry identical information; only one of
    each pair needs to be measured.  This function returns the trivial irrep
    (index 0) together with the smaller member `j <= order - j` of every
    conjugate pair (which includes the self-conjugate irrep `order/2` when
    `order` is even), restricted to irreps the germ actually supports.

    Parameters
    ----------
    order : int
        The order of the germ's cyclic group.

    multiplicities : dict
        Maps irrep index to multiplicity, as returned by :func:`germ_irreps`.
        Irreps absent from this dict (or with zero multiplicity) are omitted.

    Returns
    -------
    list of int
        The irrep indices to design experiments for, in increasing order.
    """
    return [j for j in range(order)
            if multiplicities.get(j, 0) > 0 and j <= (order - j) % order]


# -----------------------------------------------------------------------------
# Fiducial selection
# -----------------------------------------------------------------------------

def _identity_direction(dim, basis):
    """The normalized vectorized identity `|I>>/||I||` in the given (Hermitian, orthonormal) basis."""
    hdim = int(round(_np.sqrt(dim)))
    if hdim * hdim != dim:
        raise ValueError("Superoperator dimension %d is not a perfect square" % dim)
    vec = _np.asarray(_bt.change_basis(_np.identity(hdim, complex).ravel(), 'std', basis)).ravel()
    vec = _np.real_if_close(vec)
    return vec / _np.linalg.norm(vec)


def _decaying_projector(germ_superop, group_order, irrep_index, ident_vec=None):
    """
    The ideal character projector with the non-decaying identity direction removed.

    `Pi' = Q Pi Q` where `Pi` is
    :func:`pygsti.tools.chartools.fourier_operator` of the ideal germ and
    `Q = 1 - |I>><<I|` projects out the (normalized) identity direction, which
    is a fixed point of every trace-preserving unital ideal germ and therefore
    contributes a constant, non-decaying background rather than signal.  In a
    normalized Pauli-product basis this simply zeroes row and column 0.
    """
    proj = _ct.fourier_operator(germ_superop, group_order, irrep_index)
    dim = proj.shape[0]
    if ident_vec is None:
        ident_vec = _np.zeros(dim); ident_vec[0] = 1.0  # normalized pp basis convention
    q = _np.identity(dim) - _np.outer(ident_vec, _np.conj(ident_vec))
    return q @ proj @ q


def _fiducial_overlap(germ_superop, group_order, irrep_index, prep_superop, meas_superop,
                      rho=None, evec=None, ident_vec=None):
    """
    The ideal decaying-signal amplitude `|<<E| M_meas Pi' M_prep |rho>>|`.

    `Pi'` is the character projector of the targeted irrep with the identity
    direction removed (see :func:`_decaying_projector`), so this is the ideal
    amplitude of the part of the character-weighted signal that actually
    decays.  `rho` and `evec` default to the one-qubit `|0><0|` superket/superbra
    in the normalized Pauli-product basis.
    """
    dim = germ_superop.shape[0]
    if rho is None:
        rho = _np.array([1., 0., 0., 1.]) / _np.sqrt(2)
    if evec is None:
        evec = _np.array([1., 0., 0., 1.]) / _np.sqrt(2)
    assert dim == len(rho) == len(evec), "Dimension mismatch in fiducial overlap"
    proj = _decaying_projector(germ_superop, group_order, irrep_index, ident_vec)
    return abs(evec @ meas_superop @ proj @ prep_superop @ rho)


def _sort_key(word):
    """Tie-breaking key for a fiducial word: shorter first, then lexicographic by label string."""
    return (len(word), tuple(str(lbl) for lbl in word))


def _select_fiducial_words(germ_superop, group_order, irrep_index, words, word_superops,
                           rho=None, evec=None, ident_vec=None, num_pairs=None):
    """
    The shared fiducial-selection kernel used by both cGST design helpers.

    Parameters
    ----------
    germ_superop : numpy.ndarray
        The ideal germ superoperator.

    group_order, irrep_index : int
        The germ's cyclic-group order and the targeted irrep.

    words : list
        The candidate fiducial words; each is a tuple of (hashable) layer
        labels, including the empty tuple.

    word_superops : dict
        Maps each element of `words` to its ideal superoperator.

    rho, evec : numpy.ndarray, optional
        The native preparation superket and measurement superbra.

    ident_vec : numpy.ndarray, optional
        The normalized vectorized identity (defaults to `e_0`, correct in a
        normalized Pauli-product basis).

    num_pairs : int, optional
        The number of (prep, meas) pairs to return.  Defaults to `d**2`, where
        `d` is the dimension of the *decaying* part of the targeted irrep's
        eigenspace.

    Returns
    -------
    list of tuple
        `(prep_word, meas_word)` pairs.  Empty if the targeted irrep has no
        decaying directions at all.

    Notes
    -----
    The degenerate (`d > 1`) branch assumes `germ_superop` is a *normal*
    superoperator -- true of any ideal unitary germ in an orthonormal basis --
    so that its character projectors are orthogonal projectors and the signal
    factorizes as `(<<E| M_meas B) . (B^dag M_prep |rho>>)` for an orthonormal
    basis `B` of the decaying block.
    """
    if rho is None:
        rho = _np.array([1., 0., 0., 1.]) / _np.sqrt(2)
    if evec is None:
        evec = _np.array([1., 0., 0., 1.]) / _np.sqrt(2)

    proj = _decaying_projector(germ_superop, group_order, irrep_index, ident_vec)
    #  rank of the decaying block = irrep multiplicity, minus the identity
    #  direction when the irrep is trivial
    u, s, _ = _np.linalg.svd(proj)
    block_dim = int(_np.count_nonzero(s > 0.5))
    if block_dim == 0:
        return []  # nothing decays on this irrep (e.g. trivial irrep of multiplicity 1)

    num_words = block_dim if (num_pairs is None) else int(_math.ceil(_math.sqrt(num_pairs)))
    num_words = max(1, num_words)

    if num_words == 1 and block_dim == 1:
        # scalar case: search prep/meas words jointly, preferring larger ideal
        # amplitude, then shorter total length, then lexicographic order
        best, best_key = None, None
        for prep_word, meas_word in _itertools.product(words, repeat=2):
            overlap = abs(evec @ word_superops[meas_word] @ proj @ word_superops[prep_word] @ rho)
            key = (-round(overlap, 10), len(prep_word) + len(meas_word),
                   tuple(str(lbl) for lbl in prep_word), tuple(str(lbl) for lbl in meas_word))
            if best_key is None or key < best_key:
                best_key, best = key, (prep_word, meas_word)
        return [best]

    # degenerate (or over-sampled) case: pick words whose projections onto the
    # decaying block are as linearly independent as possible
    block = u[:, :block_dim]  # orthonormal basis of the decaying block
    prep_rows = _np.array([block.conj().T @ (word_superops[w] @ rho) for w in words])
    meas_rows = _np.array([(evec @ word_superops[w]) @ block for w in words])

    def _greedy(rows, side):
        chosen = []
        for _ in range(num_words):
            best_idx, best_key = None, None
            for i, word in enumerate(words):
                if i in chosen: continue
                trial = rows[chosen + [i], :]
                svals = _np.linalg.svd(trial, compute_uv=False)
                score = svals[-1] if len(svals) > 0 else 0.0
                key = (-round(float(score), 10),) + _sort_key(word)
                if best_key is None or key < best_key:
                    best_key, best_idx = key, i
            if best_idx is None:
                raise ValueError("Cannot choose %d distinct %s fiducial words for a %d-dimensional "
                                 "decaying block from only %d candidate word(s); raise "
                                 "`max_length`." % (num_words, side, block_dim, len(words)))
            chosen.append(best_idx)
        return [words[i] for i in chosen]

    prep_words = _greedy(prep_rows, 'preparation')
    meas_words = _greedy(meas_rows, 'measurement')
    pairs = [(p, m) for p in prep_words for m in meas_words]
    return pairs if (num_pairs is None) else pairs[:num_pairs]


def _model_line_labels(target_model, default=None):
    """The full set of line labels a circuit for this model should carry."""
    try:
        labels = tuple(target_model.state_space.state_space_labels)
    except AttributeError:  # pragma: no cover - unusual state space
        labels = None
    return labels if labels else default


def _model_fiducial_words(target_model, max_length):
    """All words of the model's primitive op labels up to `max_length`, plus their superoperators."""
    op_labels = tuple(target_model.primitive_op_labels)
    words = [()]
    for length in range(1, max_length + 1):
        words.extend(_itertools.product(op_labels, repeat=length))

    dim = target_model.dim
    line_labels = _model_line_labels(target_model)
    superops = {(): _np.identity(dim)}
    single = {lbl: _np.asarray(target_model.sim.product(_Circuit((lbl,), line_labels=line_labels)))
              for lbl in op_labels}
    for word in words:
        if word in superops: continue
        superops[word] = single[word[-1]] @ superops[word[:-1]]
    return words, superops


def _model_spam(target_model):
    """The model's native prep superket, all-zeros effect superbra, and identity direction."""
    prep_key = list(target_model.preps.keys())[0]
    rho = _np.asarray(target_model.preps[prep_key].to_dense()).ravel()

    povm_key = list(target_model.povms.keys())[0]
    povm = target_model.povms[povm_key]
    effect_keys = list(povm.keys())
    zeros_key = '0' * len(str(effect_keys[0]))
    if zeros_key not in effect_keys:
        raise ValueError("POVM '%s' has no all-zeros effect (keys: %s)" % (povm_key, str(effect_keys)))
    evec = _np.asarray(povm[zeros_key].to_dense()).ravel()

    ident_vec = _identity_direction(target_model.dim, target_model.basis)
    return rho, evec, ident_vec


def select_cgst_fiducials(target_model, germ, group_order, irrep_index, max_length=3,
                          num_pairs=None):
    """
    Choose cGST fiducial circuits maximizing the ideal decaying signal.

    Fiducials are words in the target model's primitive operations of length up
    to `max_length`; the state preparation and the all-zeros effect of the
    model's first POVM are used, so this works for any gate names and any
    dimension.

    When the targeted irrep's *decaying* block is one-dimensional -- i.e. a
    nontrivial irrep of multiplicity 1, or the trivial irrep of multiplicity 2
    (identity plus one decaying direction) -- a single (prep, meas) pair is
    returned: the one with the largest ideal amplitude
    `|<<E| M_meas Pi' M_prep |rho>>|`, breaking ties toward shorter and then
    lexicographically smaller words.

    When the decaying block has dimension `d > 1` a single pair cannot resolve
    the block, so `d` preparation words and `d` measurement words are chosen
    greedily to maximize the smallest singular value of their projections onto
    the block, and all `d*d` pairs are returned.

    Parameters
    ----------
    target_model : Model
        The ideal model.

    germ : Circuit
        The germ circuit.

    group_order : int
        The order of the cyclic group generated by the ideal germ.

    irrep_index : int
        The targeted irrep of `Z_group_order`.

    max_length : int, optional
        The maximum fiducial length.

    num_pairs : int, optional
        Override the number of returned pairs.  `ceil(sqrt(num_pairs))` words
        are selected per side and the first `num_pairs` pairs of the resulting
        grid are returned.

    Returns
    -------
    list of tuple
        `(prep_fiducial, meas_fiducial)` pairs of :class:`Circuit` objects.  An
        empty list is returned when the irrep supports no decaying direction
        (the trivial irrep of a germ with multiplicity 1).
    """
    rho, evec, ident_vec = _model_spam(target_model)
    words, superops = _model_fiducial_words(target_model, max_length)
    germ_superop = _np.asarray(target_model.sim.product(germ))
    pairs = _select_fiducial_words(germ_superop, group_order, irrep_index, words, superops,
                                   rho=rho, evec=evec, ident_vec=ident_vec, num_pairs=num_pairs)
    line_labels = germ.line_labels
    return [(_Circuit(p, line_labels=line_labels), _Circuit(m, line_labels=line_labels))
            for p, m in pairs]


# -----------------------------------------------------------------------------
# Full design construction
# -----------------------------------------------------------------------------

def _sanitize_name(name):
    """Strip characters that are awkward in file/directory names."""
    return ''.join(ch for ch in name if ch.isalnum() or ch in '_-')


def germ_name(germ):
    """
    A filesystem-safe name for a germ: its layer names concatenated.

    Qubit labels are omitted for single-qubit germs (so the {S, sqrt(Y)}
    triangle germ is simply `"Gzpi2Gypi2"`) and appended to each gate name for
    germs on more than one qubit (e.g. `"Gxpi2Q0GcnotQ0Q1"`), where they are
    needed to tell `Gxpi2:Q0` from `Gxpi2:Q1`; the components of a parallel
    layer are joined with `'-'`.

    Parameters
    ----------
    germ : Circuit
        The germ circuit.

    Returns
    -------
    str
    """
    with_qubits = len(germ.line_labels) > 1

    def component_name(c):
        name = str(getattr(c, 'name', c))
        sslbls = getattr(c, 'sslbls', None)
        if with_qubits and sslbls:
            name += ''.join(str(q) for q in sslbls)
        return _sanitize_name(name)

    parts = []
    for layer in germ.layertup:
        components = getattr(layer, 'components', (layer,))
        if len(components) <= 1:
            parts.append(component_name(layer))
        else:
            parts.append('-'.join(component_name(c) for c in components))
    return ''.join(parts) or 'empty'


def create_cgst_design(target_model, depths, circuits_per_depth, germs=None,
                       mode='exact', num_projection_rounds=4, irreps='nonconjugate',
                       max_fiducial_length=3, max_germ_length=7, max_order=24,
                       qubit_labels=None, seed=None, germ_search_kwargs=None,
                       descriptor=None, include_degenerate_blocks=True):
    """
    Build a cGST experiment design for an arbitrary finite-order gate set.

    One :class:`~pygsti.protocols.cgst.CharacterGSTGermDesign` child is created
    for every (germ, irrep, fiducial pair) combination, named
    `"<germ_name>_irrep<j>_pair<p>"` (see :func:`germ_name`).  The returned
    design's `germ_table` records the germ, group order, irrep, multiplicity
    and fiducials of each child.

    Parameters
    ----------
    target_model : Model
        The ideal model.  Needs a matrix forward simulator (`sim.product`).

    depths : list of ints
        The germ depths of each child design.

    circuits_per_depth : int
        The number of randomized circuits at each depth (ignored in 'exact'
        mode).

    germs : list of Circuit, optional
        The germs to use.  By default :func:`find_cgst_germs` is run.  Germs
        whose ideal product is the identity (order 1, e.g. a bare idle) are
        skipped with a warning: they have no character structure for cGST to
        filter on (see :func:`finite_order_candidate_germs`).

    mode : {'exact', 'reduced', 'full'}, optional
        The cGST sampling mode (see :mod:`pygsti.protocols.cgst`).  The
        default, `'exact'`, evaluates the synthetic irrep projector by
        deterministic quadrature over the germ powers, with zero sampling
        error.  That matters beyond variance: the `'linear'` gate-set
        inversion differentiates the fitted decays by finite differences, which
        needs a deterministic, ripple-free response -- the Monte-Carlo projector
        of `'reduced'` mode leaves an `O(error)` ripple (periodic in the depth
        modulo the group order) in every decay curve that the fits cannot
        separate from a genuine decay, and `'full'` mode's Fourier-operator
        eigenvalues are not supported by the linear inversion at all.

    num_projection_rounds : int, optional
        The number of random projection rounds in 'reduced' / 'exact' mode.

    irreps : {'nonconjugate', 'all'} or dict, optional
        Which irreps to design experiments for.  `'nonconjugate'` uses
        :func:`nonconjugate_irreps`; `'all'` uses every irrep the germ
        supports; a dict maps a germ (Circuit) to an explicit list of irrep
        indices.

    max_fiducial_length : int, optional
        The maximum fiducial length (see :func:`select_cgst_fiducials`).

    max_germ_length : int, optional
        The maximum germ length used when `germs` is None.

    max_order : int, optional
        The largest cyclic-group order a germ may generate.

    qubit_labels : tuple, optional
        The qubits the design applies to.  Defaults to the germs' line labels.

    seed : int, optional
        Base seed; each child design gets a distinct derived seed.

    germ_search_kwargs : dict, optional
        Extra keyword arguments for :func:`find_cgst_germs`.

    descriptor : str, optional
        A description used for the child designs.

    include_degenerate_blocks : bool, optional
        Whether to emit children for (germ, irrep) blocks the analysis cannot
        currently use: nontrivial irreps of multiplicity greater than 1 and
        trivial irreps of multiplicity other than 2 (matrix-valued decays).
        With the default `True` they are designed -- the fiducial grid needed
        to resolve the block is emitted -- but skipped, with a warning, by the
        analysis; with `False` only children the analysis supports (nontrivial
        multiplicity 1, trivial multiplicity 2) are created.

    Returns
    -------
    CharacterGSTDesign
    """
    from pygsti.protocols.cgst import CharacterGSTDesign as _CharacterGSTDesign
    from pygsti.protocols.cgst import CharacterGSTGermDesign as _CharacterGSTGermDesign

    if germs is None:
        kwargs = dict(germ_search_kwargs or {})
        kwargs.setdefault('seed', seed)
        germs = find_cgst_germs(target_model, max_length=max_germ_length,
                                max_order=max_order, **kwargs)
    germs = list(germs)
    if len(germs) == 0:
        raise ValueError("No germs to build a cGST design from (the `germs` argument is empty)!")

    if qubit_labels is None:
        qubit_labels = _model_line_labels(target_model, default=germs[0].line_labels)
    qubit_labels = tuple(qubit_labels)
    # make every germ (and hence every fiducial) carry the full set of lines
    def _relabel(c):
        return c if c.line_labels == qubit_labels else _Circuit(c.layertup, line_labels=qubit_labels)
    germs = [_relabel(g) for g in germs]
    if isinstance(irreps, dict):
        irreps = {_relabel(g): v for g, v in irreps.items()}
    if descriptor is None:
        descriptor = 'A cGST germ-decay experiment'

    designs, germ_table, used_names = {}, [], set()
    for germ in germs:
        order, multiplicities = germ_irreps(target_model, germ, max_order=max_order)
        if order < 2:
            _warnings.warn("Skipping germ %s: its ideal product is the identity (group order 1), "
                           "so it has no nontrivial characters for cGST to filter on.  An "
                           "idle's errors are amplified by finite-order germs that contain it, "
                           "e.g. `Gzpi2 Gi`." % germ.str)
            continue
        if isinstance(irreps, dict):
            irrep_indices = list(irreps[germ])
        elif irreps == 'all':
            irrep_indices = sorted(multiplicities.keys())
        elif irreps == 'nonconjugate':
            irrep_indices = nonconjugate_irreps(order, multiplicities)
        else:
            raise ValueError("`irreps` must be 'nonconjugate', 'all' or a dict, not %s" % str(irreps))

        gname = germ_name(germ)
        for irrep_index in irrep_indices:
            mult = multiplicities.get(irrep_index, 0)
            supported = (mult == 1) if irrep_index != 0 else (mult == 2)
            if not (supported or include_degenerate_blocks):
                continue
            pairs = select_cgst_fiducials(target_model, germ, order, irrep_index,
                                          max_length=max_fiducial_length)
            for pair_index, (prep_fid, meas_fid) in enumerate(pairs):
                name = "%s_irrep%d_pair%d" % (gname, irrep_index, pair_index)
                suffix = 1
                while name in used_names:  # keep child names unique whatever the germ names
                    name = "%s-%d_irrep%d_pair%d" % (gname, suffix, irrep_index, pair_index)
                    suffix += 1
                used_names.add(name)

                child_seed = None if (seed is None) else seed + len(designs)
                designs[name] = _CharacterGSTGermDesign(
                    germ, order, irrep_index, depths, circuits_per_depth,
                    prep_fiducial=prep_fid, meas_fiducial=meas_fid, mode=mode,
                    num_projection_rounds=num_projection_rounds, qubit_labels=qubit_labels,
                    seed=child_seed, descriptor=descriptor)
                germ_table.append({'name': name,
                                   'germ': germ.str,
                                   'group_order': int(order),
                                   'irrep_index': int(irrep_index),
                                   'multiplicity': int(mult),
                                   'prep_fiducial': prep_fid.str,
                                   'meas_fiducial': meas_fid.str,
                                   'pair_index': int(pair_index)})

    if len(designs) == 0:
        raise ValueError("None of the %d germ(s) produced a cGST sub-experiment!" % len(germs))
    return _CharacterGSTDesign(designs, qubit_labels=qubit_labels, germ_table=germ_table)

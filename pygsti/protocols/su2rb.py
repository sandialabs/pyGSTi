"""
Synthetic SPAM randomized benchmarking for arbitrary spin (SU(2)) systems.

This module ports the SU(2) synthetic-SPAM RB protocols (SSRB, SSchiRB, SSR1RB) from
the `su2-rb-conservative` branch into pyGSTi's `Design -> DataSimulator -> Protocol ->
Results` architecture, generalized from the branch's hardcoded spin-7/2 tooling to
arbitrary spin `j` via `pygsti.tools.su2tools.SpinJ`.

See `active-project/su2-synthetic-spam-rb-plan.md` for the normative conventions
(Section 3) and phase-by-phase deliverables (Section 4), and
`active-project/su2rb-label-spike-notes.md` for the Task 3.0 label/factory ergonomics
spike that this module's circuit conventions are built on.

Circuit conventions (see the spike notes for the full rationale):

- A single gate name, `'Gu'`, carries the ZXZ Euler angles `(alpha, beta, gamma)` of
  one SU(2) group element as its three `args`, on a single qudit line.
- Each sampled `(m+1, 3)` angle sequence (rows `0..m-1` Haar-random, row `m` the
  inverting gate) is turned into `2j+1` circuits -- one per prep -- that share the
  same `Gu` layers but differ in their explicit `rho{ell}` prep label (`ell = 0..2j`,
  in `SpinJ.spins` order, i.e. `rho0` prepares `|j>` and `rho{2j}` prepares `|-j>`)
  and a shared `Mdefault` POVM label.
- `prep_index`, `seq_index`, and `euler_angles` (plus, for the character design,
  `characters`/`charcores`) are carried as `paired_with_circuit_attrs` JSON aux data
  rather than being recovered by re-parsing the embedded `rho{ell}`/`Mdefault` labels.
"""
# ***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.protocols import vb as _vb
from pygsti.baseobjs.label import Label as _Label
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.tools import su2tools as _su2
from pygsti.tools.su2tools import _validate_spin

GATE_NAME = 'Gu'
"""The name of the single arg-carrying gate label used by SU(2) RB circuits."""

POVM_NAME = 'Mdefault'
"""The name of the (2j+1)-outcome POVM label appended to SU(2) RB circuits."""


def _prep_name(prep_index):
    """The name of the prep label for the `prep_index`-th Jz eigenstate (`spins[prep_index]`)."""
    return 'rho%d' % prep_index


def circuit_from_euler_angles(angles, j, qudit_label='Q0', prep_index=None):
    """
    Build a `Circuit` of `Gu` gate layers from a sequence of ZXZ Euler angles.

    Parameters
    ----------
    angles : numpy array
        Shape `(m+1, 3)`. Row `i` gives the ZXZ Euler angles `(alpha, beta, gamma)`
        of the `i`-th layer's `Gu` gate.

    j : int, float, or Fraction
        The spin (used only to validate `prep_index`, if given).

    qudit_label : str, optional
        The line label of the single qudit this circuit acts on.

    prep_index : int, optional
        If given, an integer in `range(2*j + 1)` selecting one of the `2j+1` Jz
        eigenstates (`0` is `|j>`, `2*j` is `|-j>`, matching `SpinJ.spins`). When
        given, the returned circuit is prefixed with an explicit `rho{prep_index}`
        prep label and suffixed with the shared `Mdefault` POVM label. When `None`
        (the default), the circuit consists of only the `Gu` gate layers.

    Returns
    -------
    Circuit
    """
    angles = _np.asarray(angles, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError("`angles` must have shape (m+1, 3); got %s" % (angles.shape,))

    layers = [_Label(GATE_NAME, qudit_label, args=tuple(float(x) for x in row)) for row in angles]

    if prep_index is not None:
        _, two_j = _validate_spin(j)
        dim = two_j + 1
        if not (0 <= prep_index < dim):
            raise ValueError("prep_index=%r is out of range for j=%s (dim=%d)" % (prep_index, j, dim))
        layers = [_Label(_prep_name(prep_index))] + layers + [_Label(POVM_NAME)]

    return _Circuit(layers, line_labels=(qudit_label,))


def euler_angles_from_circuit(circuit):
    """
    Recover the `(m+1, 3)` array of ZXZ Euler angles from a `Circuit`'s `Gu` layers.

    Any non-`Gu` layers (e.g. explicit `rho{ell}` prep or `Mdefault` POVM layers) are
    ignored, so this is the inverse of `circuit_from_euler_angles` regardless of
    whether `prep_index` was given.

    Parameters
    ----------
    circuit : Circuit
        A circuit built by `circuit_from_euler_angles` (or with the same `Gu`-layer
        convention).

    Returns
    -------
    numpy array
        Shape `(m+1, 3)`.
    """
    rows = []
    for layer in circuit:
        if layer.name == GATE_NAME:
            args = layer.args
            if len(args) != 3:
                raise ValueError("Gate layer %s does not carry exactly 3 args" % (layer,))
            rows.append([float(a) for a in args])
    if not rows:
        raise ValueError("Circuit %s contains no '%s' gate layers" % (circuit, GATE_NAME))
    return _np.array(rows, dtype=float)


def _sample_su2rb_circuits(j_float, dim, depths, circuits_per_depth, seed, qudit_label, invert_from):
    """
    Core RB circuit sampler shared by `SU2RBDesign` and `SU2CharacterRBDesign`.

    Ports `raw_su2_rb_design` from the `su2-rb-conservative` branch: for each depth
    `m`, samples `circuits_per_depth` length-`m` Haar-random Euler-angle sequences and
    appends an inverting final gate computed from the composition of rows
    `invert_from..m-1` (so the net ideal composition of all `m+1` rows is the identity
    when `invert_from == 0`, or the "hidden" first gate when `invert_from == 1`).
    Each sampled sequence yields `dim` circuits (one per prep).

    `idealout_lists` entries are `(str(ell),)`, i.e. the prep index of the circuit
    they're attached to. This is a genuine deterministic ideal outcome when
    `invert_from == 0` (net composition is the identity, so state `ell` should be
    recovered), but for `invert_from == 1` (`SU2CharacterRBDesign`) the net
    composition is the random "hidden" first gate, which generally does *not* fix
    `ell`, so these entries are placeholders rather than true ideal outcomes -- see
    `SU2CharacterRBDesign`'s docstring. They are populated uniformly here (rather than
    being `None` or omitted) only because `BenchmarkingDesign.__init__` requires an
    `ideal_outs` entry for every circuit; generic idealout-consuming analyses (e.g.
    anything that computes a "success probability" from `idealout_lists`) should not
    be silently applied to `SU2CharacterRBDesign` data on this basis.

    Returns
    -------
    dict
        Keys `circuit_lists`, `idealout_lists`, `euler_angles_lists`, `seq_index_lists`,
        `prep_index_lists`, `firstgate_angles_lists` -- each a list over `depths` (the
        first five are further indexed over the `dim * circuits_per_depth` circuits at
        that depth; `firstgate_angles_lists` is indexed over the `circuits_per_depth`
        sampled sequences at that depth, and holds each sequence's row-0 angles).
    """
    rng = _np.random.default_rng(seed)

    circuit_lists, idealout_lists = [], []
    euler_angles_lists, seq_index_lists, prep_index_lists = [], [], []
    firstgate_angles_lists = []

    for m in depths:
        if m < 1:
            raise ValueError("SU(2) RB depths must be >= 1 (got %r)" % (m,))

        circuits_at_depth, idealouts_at_depth = [], []
        euler_angles_at_depth, seq_index_at_depth, prep_index_at_depth = [], [], []
        firstgate_angles_at_seq = []

        for s in range(circuits_per_depth):
            angles = _np.zeros((m + 1, 3))
            a, b, g = _su2.random_euler_angles(m, rng=rng)
            angles[:m, 0], angles[:m, 1], angles[:m, 2] = a, b, g
            a_inv, b_inv, g_inv = _su2.composition_inverse(
                angles[invert_from:m, 0], angles[invert_from:m, 1], angles[invert_from:m, 2])
            angles[m, :] = (a_inv, b_inv, g_inv)
            firstgate_angles_at_seq.append(angles[0, :].copy())

            angles_as_list = angles.tolist()
            for ell in range(dim):
                c = circuit_from_euler_angles(angles, j_float, qudit_label, prep_index=ell)
                circuits_at_depth.append(c)
                # Placeholder ideal outcome (see this function's docstring): only a
                # genuine deterministic ideal outcome when invert_from == 0.
                idealouts_at_depth.append((str(ell),))
                euler_angles_at_depth.append(angles_as_list)
                seq_index_at_depth.append(s)
                prep_index_at_depth.append(ell)

        circuit_lists.append(circuits_at_depth)
        idealout_lists.append(idealouts_at_depth)
        euler_angles_lists.append(euler_angles_at_depth)
        seq_index_lists.append(seq_index_at_depth)
        prep_index_lists.append(prep_index_at_depth)
        firstgate_angles_lists.append(firstgate_angles_at_seq)

    return dict(circuit_lists=circuit_lists, idealout_lists=idealout_lists,
                euler_angles_lists=euler_angles_lists, seq_index_lists=seq_index_lists,
                prep_index_lists=prep_index_lists, firstgate_angles_lists=firstgate_angles_lists)


class SU2RBDesign(_vb.BenchmarkingDesign):
    """
    Experiment design for synthetic SPAM randomized benchmarking (SSRB) of an
    arbitrary-spin SU(2) system.

    For each depth `m` in `depths`, samples `circuits_per_depth` length-`m`
    Haar-random SU(2) Euler-angle sequences, appends an inverting final gate so that
    the net ideal composition of the full `m+1`-gate sequence is the identity, and
    emits `2*j + 1` circuits per sampled sequence -- one per Jz-eigenstate prep,
    sharing the same `Gu` gate layers (see the module docstring for the full circuit
    convention).

    Parameters
    ----------
    j : int, float, or Fraction
        The spin. `2*j` must be a non-negative integer.

    depths : list or tuple of int
        The RB circuit depths `m` (number of Haar-random gates before the inverting
        gate is appended). Each depth must be >= 1.

    circuits_per_depth : int
        The number of (Haar-)random gate sequences sampled per depth (shared across
        all depths). The number of circuits at a given depth is
        `circuits_per_depth * (2*j + 1)`.

    seed : int, optional
        Seed for the `numpy.random.default_rng` used to sample circuits. If `None`,
        a random seed is chosen (and stored on `self.seed`).

    qudit_label : str, optional
        The line label of the single qudit these circuits act on.

    descriptor : str, optional
        A string describing the generated experiment.

    Attributes
    ----------
    j : float
        The spin.

    dim : int
        `2*j + 1`, the number of preps (and POVM effects).

    invert_from : int
        Class attribute; `0` for `SU2RBDesign` (net ideal composition is the
        identity), overridden to `1` by `SU2CharacterRBDesign`.
    """

    invert_from = 0

    def __init__(self, j, depths, circuits_per_depth, seed=None, qudit_label='Q0',
                 descriptor='An SU(2) synthetic SPAM RB experiment'):
        self._populate(j, depths, circuits_per_depth, seed, qudit_label, descriptor)

    def _populate(self, j, depths, circuits_per_depth, seed, qudit_label, descriptor):
        """
        Shared implementation of `__init__`, factored out so that `SU2CharacterRBDesign`
        can reuse it and then extend the freshly-built `self` with `characters`/
        `charcores` computed from the (locally-scoped, not stored on `self`) per-sequence
        first-gate angles that `_sample_su2rb_circuits` returns alongside everything
        that *does* get stored.

        Returns
        -------
        dict
            The `_sample_su2rb_circuits` return dict (including `firstgate_angles_lists`,
            which is not otherwise retained on `self` since it is not JSON-serializable
            and is only needed transiently by `SU2CharacterRBDesign.__init__`).
        """
        j_float, two_j = _validate_spin(j)
        dim = two_j + 1
        depths = list(depths)

        if seed is None:
            seed = _np.random.randint(1, 1000000)
        self.seed = seed

        sampled = _sample_su2rb_circuits(j_float, dim, depths, circuits_per_depth, self.seed,
                                          qudit_label, self.invert_from)

        # Set before calling super().__init__ so that BenchmarkingDesign inserts
        # 'idealout_lists' at the front and registers 'json' auxfile types for all of
        # these (mirrors CliffordRBDesign._init_foundation / BinaryRBDesign._init_foundation).
        self.paired_with_circuit_attrs = ["euler_angles", "seq_index", "prep_index"]

        super(SU2RBDesign, self).__init__(depths, sampled['circuit_lists'], sampled['idealout_lists'],
                                           qubit_labels=(qudit_label,), remove_duplicates=False)

        self.euler_angles = sampled['euler_angles_lists']
        self.seq_index = sampled['seq_index_lists']
        self.prep_index = sampled['prep_index_lists']

        self.j = j_float
        self.dim = dim
        self.qudit_label = qudit_label
        self.circuits_per_depth = circuits_per_depth
        self.descriptor = descriptor

        return sampled


class SU2CharacterRBDesign(SU2RBDesign):
    """
    Experiment design for the character variants of synthetic SPAM RB (SSchiRB and
    SSR1RB) of an arbitrary-spin SU(2) system.

    As `SU2RBDesign`, except the final gate only inverts the composition of rows
    `1..m-1` (not row 0), so the net ideal composition of the full circuit is the
    "hidden" first gate rather than the identity. Additionally stores, per circuit,
    the irrep character `chi_k` (`characters`) and Legendre "character core"
    `P_k(cos(beta))` (`charcores`) of the hidden first gate, for `k = 0..2j` -- these
    support both SSchiRB (via `characters`) and SSR1RB (via `charcores`) from the same
    design.

    Note on `idealout_lists`: because the net ideal composition here is the random
    hidden first gate rather than the identity, `idealout_lists` entries (inherited
    unchanged from `SU2RBDesign`, one `(str(prep_index),)` per circuit) are *not*
    genuine deterministic ideal outcomes for this design -- they are placeholders
    required only because `BenchmarkingDesign.__init__` needs some `ideal_outs` value
    per circuit. `SyntheticSPAMCharacterRB`/`SyntheticSPAMRank1RB` (Phase 6) reconstruct
    probabilities directly from the `euler_angles`/`characters`/`charcores` aux data,
    not from `idealout_lists`. Generic RB analyses that infer a "success probability"
    from `idealout_lists` should not be pointed at `SU2CharacterRBDesign` data.

    Parameters
    ----------
    j, depths, circuits_per_depth, seed, qudit_label, descriptor
        As `SU2RBDesign`.
    """

    invert_from = 1

    def __init__(self, j, depths, circuits_per_depth, seed=None, qudit_label='Q0',
                 descriptor='An SU(2) synthetic SPAM character RB experiment'):
        sampled = self._populate(j, depths, circuits_per_depth, seed, qudit_label, descriptor)

        irrep_labels = _np.arange(self.dim)
        characters_lists, charcores_lists = [], []
        for firstgate_angles_at_seq in sampled['firstgate_angles_lists']:
            # Shape (3, circuits_per_depth), as required by characters_from_euler_angles.
            angles_arr = _np.array(firstgate_angles_at_seq).T
            chars = _su2.characters_from_euler_angles(irrep_labels, angles_arr)
            cores = _su2.charactercores_from_euler_angles(irrep_labels, angles_arr)
            # Broadcast each sampled sequence's (shared) character/charcore row out to
            # the `dim` consecutive per-prep circuits generated from it, matching the
            # (seq-major, prep-minor) circuit ordering produced by _sample_su2rb_circuits.
            characters_lists.append(_np.repeat(chars, self.dim, axis=0).tolist())
            charcores_lists.append(_np.repeat(cores, self.dim, axis=0).tolist())

        self.paired_with_circuit_attrs = self.paired_with_circuit_attrs + ["characters", "charcores"]
        self.characters = characters_lists
        self.charcores = charcores_lists
        self.auxfile_types['characters'] = 'json'
        self.auxfile_types['charcores'] = 'json'

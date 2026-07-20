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
- `prep_index`, `seq_index`, `euler_angles`, and `charcores` are carried as
  `paired_with_circuit_attrs` JSON aux data rather than being recovered by
  re-parsing the embedded `rho{ell}`/`Mdefault` labels.
"""
# ***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import math as _math
import warnings as _warnings

import numpy as _np
import pandas as _pd
import scipy.linalg as _spl
from scipy.optimize import curve_fit as _curve_fit
from scipy.optimize import OptimizeWarning as _OptimizeWarning

from pygsti.protocols import vb as _vb
from pygsti.protocols import protocol as _proto
from pygsti.algorithms import rbfit as _rbfit
from pygsti.baseobjs.label import Label as _Label
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.data.dataset import DataSet as _DataSet
from pygsti.tools import su2tools as _su2
from pygsti.tools import wignersymbols as _wg
from pygsti.tools.su2tools import _validate_spin
from pygsti.tools.optools import unitary_to_std_process_mx as _unitary_to_std_process_mx

__all__ = [
    'GATE_NAME',
    'POVM_NAME',
    'circuit_from_euler_angles',
    'euler_angles_from_circuit',
    'SU2RBDesign',
    'jz_dephasing',
    'jz_rotation',
    'compose_noise_channels',
    'SU2RBDataSimulator',
    'predicted_zero_noise_variance',
    'SyntheticSPAMRank1RB',
    'SyntheticSPAMRBResults',
]

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


def _sample_su2rb_circuits(j_float, dim, depths, circuits_per_depth, seed, qudit_label):
    """
    Core RB circuit sampler for `SU2RBDesign`.

    Ports `raw_su2_rb_design` from the `su2-rb-conservative` branch: for each depth
    `m`, samples `circuits_per_depth` length-`m` Haar-random Euler-angle sequences and
    appends an inverting final gate computed from the composition of rows `1..m-1`, so
    the net ideal composition of all `m+1` rows is the random "hidden" first gate (row
    0), not the identity. Each sampled sequence yields `dim` circuits (one per prep).

    `idealout_lists` entries are `(str(ell),)`, i.e. the prep index of the circuit
    they're attached to. Because the net ideal composition is the random hidden first
    gate (not the identity), these are placeholders rather than genuine deterministic
    ideal outcomes -- see `SU2RBDesign`'s docstring. They are populated uniformly here
    (rather than being `None` or omitted) only because `BenchmarkingDesign.__init__`
    requires an `ideal_outs` entry for every circuit; generic idealout-consuming
    analyses (e.g. anything that computes a "success probability" from
    `idealout_lists`) should not be silently applied to this design's data on that
    basis.

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
                angles[1:m, 0], angles[1:m, 1], angles[1:m, 2])
            angles[m, :] = (a_inv, b_inv, g_inv)
            firstgate_angles_at_seq.append(angles[0, :].copy())

            angles_as_list = angles.tolist()
            for ell in range(dim):
                c = circuit_from_euler_angles(angles, j_float, qudit_label, prep_index=ell)
                circuits_at_depth.append(c)
                # Placeholder ideal outcome (see this function's docstring).
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
    Experiment design for synthetic SPAM randomized benchmarking of an arbitrary-spin
    SU(2) system.

    For each depth `m` in `depths`, samples `circuits_per_depth` length-`m`
    Haar-random SU(2) Euler-angle sequences, appends a final gate that inverts the
    composition of rows `1..m-1` (not row 0), so the net ideal composition of the full
    `m+1`-gate sequence is the random "hidden" first gate rather than the identity,
    and emits `2*j + 1` circuits per sampled sequence -- one per Jz-eigenstate prep,
    sharing the same `Gu` gate layers (see the module docstring for the full circuit
    convention). Additionally stores, per circuit, the Legendre "character core"
    `P_k(cos(beta))` of the hidden first gate (`charcores`), for `k = 0..2j`, which
    `SyntheticSPAMRank1RB` uses to weight its per-irrep estimator.

    Note on `idealout_lists`: because the net ideal composition is the random hidden
    first gate rather than the identity, `idealout_lists` entries (one
    `(str(prep_index),)` per circuit) are *not* genuine deterministic ideal outcomes
    -- they are placeholders required only because `BenchmarkingDesign.__init__`
    needs some `ideal_outs` value per circuit. `SyntheticSPAMRank1RB` reconstructs
    probabilities directly from the `euler_angles`/`charcores` aux data, not from
    `idealout_lists`. Generic RB analyses that infer a "success probability" from
    `idealout_lists` should not be pointed at this design's data.

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
    """

    def __init__(self, j, depths, circuits_per_depth, seed=None, qudit_label='Q0',
                 descriptor='An SU(2) synthetic SPAM RB experiment'):
        j_float, two_j = _validate_spin(j)
        dim = two_j + 1
        depths = list(depths)

        if seed is None:
            seed = _np.random.randint(1, 1000000)
        self.seed = seed

        sampled = _sample_su2rb_circuits(j_float, dim, depths, circuits_per_depth, self.seed, qudit_label)

        # Set before calling super().__init__ so that BenchmarkingDesign inserts
        # 'idealout_lists' at the front and registers 'json' auxfile types for all of
        # these (mirrors CliffordRBDesign._init_foundation / BinaryRBDesign._init_foundation).
        self.paired_with_circuit_attrs = ["euler_angles", "seq_index", "prep_index", "charcores"]

        super(SU2RBDesign, self).__init__(depths, sampled['circuit_lists'], sampled['idealout_lists'],
                                           qubit_labels=(qudit_label,), remove_duplicates=False)

        self.euler_angles = sampled['euler_angles_lists']
        self.seq_index = sampled['seq_index_lists']
        self.prep_index = sampled['prep_index_lists']

        irrep_labels = _np.arange(dim)
        charcores_lists = []
        for firstgate_angles_at_seq in sampled['firstgate_angles_lists']:
            # Shape (3, circuits_per_depth), as required by charactercores_from_euler_angles.
            angles_arr = _np.array(firstgate_angles_at_seq).T
            cores = _su2.charactercores_from_euler_angles(irrep_labels, angles_arr)
            # Broadcast each sampled sequence's (shared) charcore row out to the `dim`
            # consecutive per-prep circuits generated from it, matching the (seq-major,
            # prep-minor) circuit ordering produced by _sample_su2rb_circuits.
            charcores_lists.append(_np.repeat(cores, dim, axis=0).tolist())
        self.charcores = charcores_lists

        self.j = j_float
        self.dim = dim
        self.qudit_label = qudit_label
        self.circuits_per_depth = circuits_per_depth
        self.descriptor = descriptor


# ***************************************************************************************************
# Noise channels for SU2RBDataSimulator
# ***************************************************************************************************

def jz_dephasing(spinj, gamma, power=1.0):
    """
    A gate-independent dephasing noise channel, diagonal in the standard (matrix-unit)
    basis: `E[ell, ell] = exp(-gamma * |i - j|**power)`, where `ell = i + j*dim` are the
    row-major-raveled matrix-unit indices of a `dim`-by-`dim` density matrix (`i` is the
    column index, `j` the row index -- since `|i - j|` is symmetric in the two, this
    labeling ambiguity doesn't matter). Ports `SU2RBSim._set_error_channel_Jz_dephasing`
    from the `su2-rb-conservative` branch.

    Parameters
    ----------
    spinj : SpinJ
        The representation whose (`dim`-by-`dim`) density matrices this channel acts on.

    gamma : float
        The dephasing rate. `gamma == 0` gives the identity channel.

    power : float, optional
        The power `p` in `|i - j|**p` (`p=1` is the branch's "exponential" profile,
        `p=2` its "gaussian" profile).

    Returns
    -------
    numpy array
        Shape `(dim**2, dim**2)`, a diagonal superoperator acting on row-major-raveled
        density matrices.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative (got %r)" % gamma)
    dim = spinj.dim
    ell = _np.arange(dim * dim)
    i = ell % dim
    j = ell // dim
    diag = _np.exp(-gamma * _np.abs(i - j).astype(float) ** power)
    return _np.diag(diag).astype(complex)


def jz_rotation(spinj, theta, power=1.0):
    """
    A gate-independent unitary "Z-rotation" noise channel: `U = diag(exp(1j * theta *
    spins**power))`, in the `spinj.spins`-ordered basis. Ports
    `SU2RBSim.set_error_channel_rotate_Jz_pow` from the `su2-rb-conservative` branch.

    Parameters
    ----------
    spinj : SpinJ
        The representation whose `spins` this channel's generator is diagonal in.

    theta : float
        The rotation angle.

    power : float, optional
        The power `p` in `spins**p` (`p=1` is a rotation about the "physical" Jz axis;
        `p=2` is the branch's `rotate_Jz2` variant).

    Returns
    -------
    numpy array
        Shape `(dim, dim)`, a unitary.
    """
    return _np.diag(_np.exp(1j * theta * (spinj.spins ** power)))


def compose_noise_channels(spinj, *channels):
    """
    Compose a sequence of noise channels (each either a `(dim, dim)` unitary or a
    `(dim**2, dim**2)` superoperator) into a single superoperator, applied in the order
    given (`channels[0]` first).

    Parameters
    ----------
    spinj : SpinJ
        The representation the channels act on.

    *channels : numpy array
        Each either shape `(dim, dim)` (a unitary) or `(dim**2, dim**2)` (a
        superoperator).

    Returns
    -------
    numpy array
        Shape `(dim**2, dim**2)`.
    """
    dim = spinj.dim
    composed = _np.eye(dim * dim, dtype=complex)
    for channel in channels:
        channel = _np.asarray(channel)
        if channel.shape == (dim, dim):
            channel = _unitary_to_std_process_mx(channel)
        elif channel.shape != (dim * dim, dim * dim):
            raise ValueError("Each channel must have shape (%d, %d) or (%d, %d); got %s"
                              % (dim, dim, dim * dim, dim * dim, channel.shape))
        composed = channel @ composed
    return composed


# ***************************************************************************************************
# SU2RBDataSimulator
# ***************************************************************************************************

class SU2RBDataSimulator(_proto.DataSimulator):
    """
    Simulates SU(2) synthetic SPAM RB data for `SU2RBDesign`/`SU2CharacterRBDesign`
    experiment designs, without going through a full pyGSTi `Model`.

    Reads the design's `euler_angles` (and, for `SU2CharacterRBDesign`, `invert_from`)
    aux data directly -- it never parses circuit labels -- and builds the ZXZ-angle
    unitaries in batch via the instance's `SpinJ`'s precomputed eigendecompositions,
    inserting a gate-independent noise channel after every gate (except, per the
    `su2-rb-conservative` branch's convention, immediately after the "hidden" first
    gate of a design with `invert_from == 1`; ports `SU2RBSim.compute_probabilities` /
    `SU2CharacterRBSim._process_circuits`).

    Parameters
    ----------
    spinj : SpinJ or (int, float, or Fraction)
        The representation to simulate in, or a spin `j` from which one is built.

    noise_channel : None, numpy array, optional
        The gate-independent noise channel applied after each gate. `None` means no
        noise (the identity channel). A `(dim, dim)` array is treated as a unitary and
        converted to a superoperator. A `(dim**2, dim**2)` array is treated as a
        superoperator directly (acting on row-major-raveled density matrices, i.e. in
        the same convention as `pygsti.tools.optools.unitary_to_std_process_mx`). Use
        `jz_dephasing`, `jz_rotation`, and/or `compose_noise_channels` (this module) to
        build one.

    shots : int, optional
        If `None` (the default), the returned `DataSet` records exact probabilities
        (as floating-point "counts" that sum to 1 per circuit), following
        `DataCountsSimulator`'s `sample_error='none'` convention. If an integer, that
        many shots per circuit are drawn from a multinomial distribution.

    seed : int, optional
        Seed for the `numpy.random.default_rng` used when `shots` is not `None`.

    Attributes
    ----------
    dim : int
        `spinj.dim`.
    """

    def __init__(self, spinj, noise_channel=None, shots=None, seed=None):
        super(SU2RBDataSimulator, self).__init__()
        if not isinstance(spinj, _su2.SpinJ):
            spinj = _su2.SpinJ(spinj)
        self.spinj = spinj
        self.dim = spinj.dim
        self.shots = shots
        self.seed = seed
        self._noise_superop = self._resolve_noise_channel(noise_channel)

    def _resolve_noise_channel(self, noise_channel):
        if noise_channel is None:
            return None
        noise_channel = _np.asarray(noise_channel)
        dim = self.dim
        if noise_channel.shape == (dim, dim):
            return _unitary_to_std_process_mx(noise_channel)
        elif noise_channel.shape == (dim * dim, dim * dim):
            return noise_channel.astype(complex)
        else:
            raise ValueError(
                "noise_channel must be None, a (%d, %d) unitary, or a (%d, %d) superoperator; got shape %s"
                % (dim, dim, dim * dim, dim * dim, noise_channel.shape))

    @property
    def is_noiseless(self):
        """True if this simulator's noise channel is the identity (`noise_channel=None`)."""
        return self._noise_superop is None

    # -----------------------------------------------------------------
    # Core composition machinery
    # -----------------------------------------------------------------

    def _compose_full(self, angles, skip_first_noise):
        """
        The general (O(depth)) path: compose the `(dim**2, dim**2)` superoperator for
        the full gate sequence `angles` (shape `(m+1, 3)`), inserting `self.
        _noise_superop` after every gate except (if `skip_first_noise`) the first.
        """
        dim = self.dim
        Us = self.spinj.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
        S = _np.eye(dim * dim, dtype=complex)
        skip_next_noise = skip_first_noise
        for U in Us:
            S = _unitary_to_std_process_mx(U) @ S
            if not skip_next_noise and self._noise_superop is not None:
                S = self._noise_superop @ S
            skip_next_noise = False
        return S

    def _compose_shortcut(self, angles):
        """
        The noiseless-character O(1) shortcut: the composition of a full character-
        design gate sequence (with noise skipped after the hidden first gate) is, in
        the absence of any noise, exactly the hidden first gate's superoperator (the
        appended inverting gate cancels rows `1..m-1`). Ports
        `SU2CharacterRBSim._process_circuits`'s fast path.
        """
        U0 = self.spinj.unitaries_from_angles(angles[0, 0], angles[0, 1], angles[0, 2])[0]
        return _unitary_to_std_process_mx(U0)

    def _compose(self, angles, skip_first_noise):
        if self._noise_superop is None and skip_first_noise:
            return self._compose_shortcut(angles)
        return self._compose_full(angles, skip_first_noise)

    @staticmethod
    def _skip_first_noise_for(edesign):
        return getattr(edesign, 'invert_from', 0) == 1

    def _unique_sequences_at_depth(self, edesign, depth_idx):
        """
        The (m+1, 3) angle arrays of the `circuits_per_depth` distinct sampled
        sequences at depth index `depth_idx`, one per `seq_index` value, ordered by
        `seq_index`.
        """
        seen = {}
        for angles, seq_idx in zip(edesign.euler_angles[depth_idx], edesign.seq_index[depth_idx]):
            if seq_idx not in seen:
                seen[seq_idx] = _np.asarray(angles, dtype=float)
        return [seen[k] for k in sorted(seen.keys())]

    # -----------------------------------------------------------------
    # Ideal (computational-basis) SPAM, as row-major-raveled superkets
    # -----------------------------------------------------------------

    def _ideal_computational_ops(self):
        """
        Shape `(dim, dim**2)`: row `ell` is the row-major-raveled `|ell><ell|`
        projector (used as both the default state preps and the default POVM
        effects).
        """
        dim = self.dim
        ops = _np.zeros((dim, dim * dim), dtype=complex)
        for ell in range(dim):
            ops[ell, ell * dim + ell] = 1.0
        return ops

    # -----------------------------------------------------------------
    # Composition cache / SPAM-swap API
    # -----------------------------------------------------------------

    def compute_nonspam_compositions(self, edesign, depth_indices=None):
        """
        Compute the composed noisy (non-SPAM) superoperator for each distinct sampled
        circuit sequence, at the requested depths.

        This is the capability behind the paper's SPAM-robustness figures: once
        computed, `probabilities_from_compositions` can re-derive probabilities under
        arbitrary (e.g. perturbed) state preps/POVMs without re-simulating circuits.

        Parameters
        ----------
        edesign : SU2RBDesign or SU2CharacterRBDesign
            The design whose `euler_angles`/`seq_index` aux data to read.

        depth_indices : iterable of int, optional
            Which depth indices (into `edesign.depths`) to compute compositions for.
            If `None` (the default), all depths are computed -- note that the full
            cache at the paper's scale (10**4 sequences x 12 depths x 64**2 complex
            doubles) is approximately 4 GB, so pass an explicit subset to bound memory
            usage for large designs.

        Returns
        -------
        dict
            Maps depth index -> numpy array of shape
            `(circuits_per_depth, dim**2, dim**2)`, the composed superoperator for
            each sampled sequence at that depth (ordered by `seq_index`).
        """
        skip_first_noise = self._skip_first_noise_for(edesign)
        if depth_indices is None:
            depth_indices = range(len(edesign.depths))
        out = {}
        for depth_idx in depth_indices:
            sequences = self._unique_sequences_at_depth(edesign, depth_idx)
            out[depth_idx] = _np.array([self._compose(angles, skip_first_noise) for angles in sequences])
        return out

    def probabilities_from_compositions(self, compositions, statepreps=None, povm=None):
        """
        Compute prep-by-effect outcome probabilities from precomputed non-SPAM
        compositions (see `compute_nonspam_compositions`), for post-hoc SPAM sweeps
        without re-simulating circuits.

        Parameters
        ----------
        compositions : dict
            As returned by `compute_nonspam_compositions`: depth index -> array of
            shape `(n_sequences, dim**2, dim**2)`.

        statepreps : numpy array, optional
            Shape `(num_preps, dim**2)`; row `i` is the row-major-raveled density
            matrix of the `i`-th state preparation. Defaults to the ideal
            computational-basis preps `|ell><ell|` (`ell = 0..dim-1`, in
            `SpinJ.spins` order).

        povm : numpy array, optional
            Shape `(num_effects, dim**2)`; row `e` is the row-major-raveled effect
            matrix of the `e`-th POVM outcome. Defaults to the ideal
            computational-basis POVM (same convention as `statepreps`).

        Returns
        -------
        dict
            Maps depth index -> numpy array of shape
            `(n_sequences, num_preps, num_effects)`.
        """
        if statepreps is None:
            statepreps = self._ideal_computational_ops()
        if povm is None:
            povm = self._ideal_computational_ops()
        statepreps = _np.asarray(statepreps, dtype=complex)
        povm = _np.asarray(povm, dtype=complex)

        out = {}
        for depth_idx, compositions_at_depth in compositions.items():
            # evolved[s, i, :] = compositions_at_depth[s] @ statepreps[i]
            evolved = _np.einsum('sab,ib->sia', compositions_at_depth, statepreps)
            # probs[s, i, e] = <povm[e] | evolved[s, i]> = Tr(povm_mx[e] . prep_mx[i] after composition s)
            probs = _np.einsum('ek,sik->sie', _np.conj(povm), evolved)
            imag_norm = _np.linalg.norm(probs.imag)
            if imag_norm > 1e-8:
                raise ValueError("probabilities_from_compositions produced non-negligible imaginary "
                                  "part (norm %g); statepreps/povm may not be Hermitian" % imag_norm)
            out[depth_idx] = probs.real
        return out

    # -----------------------------------------------------------------
    # run()
    # -----------------------------------------------------------------

    def _sample_counts(self, probs, outcome_labels, rng):
        if self.shots is None:
            return {ol: float(p) for ol, p in zip(outcome_labels, probs)}
        clipped = _np.clip(probs, 0, None)
        total = clipped.sum()
        clipped = clipped / total if total > 0 else _np.full(len(probs), 1.0 / len(probs))
        counts = rng.multinomial(self.shots, clipped)
        return {ol: int(c) for ol, c in zip(outcome_labels, counts)}

    def run(self, edesign, memlimit=None, comm=None):
        """
        Simulate SU(2) synthetic SPAM RB data for `edesign`.

        Parameters
        ----------
        edesign : SU2RBDesign or SU2CharacterRBDesign
            The design to simulate. Its `qudit_label` and `j`/`dim` must be consistent
            with this simulator's `spinj`.

        memlimit : int, optional
            Unused (present for `DataSimulator` interface compatibility).

        comm : mpi4py.MPI.Comm, optional
            Unused (present for `DataSimulator` interface compatibility).

        Returns
        -------
        ProtocolData
        """
        if edesign.dim != self.dim:
            raise ValueError("edesign.dim=%d does not match this simulator's spinj.dim=%d"
                              % (edesign.dim, self.dim))

        skip_first_noise = self._skip_first_noise_for(edesign)
        outcome_labels = [str(ell) for ell in range(self.dim)]
        rng = _np.random.default_rng(self.seed) if self.shots is not None else None
        ideal_ops = self._ideal_computational_ops()

        ds = _DataSet(collision_action="aggregate")
        # DataSet's default repetition-count dtype is float32 (dataset.py's
        # `Repcount_type`), which is far too coarse for the exact-probability
        # "counts" recorded when `shots is None`: float32 has ~1e-7 relative
        # precision, which shows up as an artifact-level discrepancy against
        # independently-computed probabilities (e.g. from a full pyGSTi Model).
        # Use float64 so the shots=None counts really are exact.
        ds.repType = _np.float64
        for depth_idx, circuits_at_depth in enumerate(edesign.circuit_lists):
            seq_compositions = {}
            angles_at_depth = edesign.euler_angles[depth_idx]
            seq_at_depth = edesign.seq_index[depth_idx]
            prep_at_depth = edesign.prep_index[depth_idx]
            for circuit, angles, seq_idx, prep_idx in zip(
                    circuits_at_depth, angles_at_depth, seq_at_depth, prep_at_depth):
                if seq_idx not in seq_compositions:
                    seq_compositions[seq_idx] = self._compose(
                        _np.asarray(angles, dtype=float), skip_first_noise)
                S = seq_compositions[seq_idx]

                evolved = S @ ideal_ops[prep_idx]
                probs = _np.real(_np.diag(evolved.reshape(self.dim, self.dim)))
                # ^ Tr(|ell'><ell'| . evolved) == evolved.reshape(dim, dim)[ell', ell'],
                #   for the diagonal computational-basis effects.

                counts = self._sample_counts(probs, outcome_labels, rng)
                ds.add_count_dict(circuit, counts)
        ds.done_adding_data()
        return _proto.ProtocolData(edesign, ds)


# ***************************************************************************************************
# Zero-noise sample-complexity (variance) diagnostics
# ***************************************************************************************************

def _variance_M_entry(j, k, ell):
    """
    `M[k, ell] = sqrt((2k+1)/(2j+1)) * <j ell; k 0 | j ell>`, the same formula as
    `SpinJ.synthetic_spam_matrix`, but evaluated directly via `wignersymbols.
    clebsch_gordan` for *any* non-negative integer `k` -- not just `k` in `0..2j`.

    This matters here because the paper's zero-noise variance sums (Eqs.
    `normalizedvariance`/`SSvariance`) range the inner irrep label `k'` over
    `0..2*k`, which exceeds `2*j` once `k > j`; `SpinJ.synthetic_spam_matrix` only has
    rows `0..2j`. The corresponding entries are exactly zero once `k > 2*j` (the
    Clebsch-Gordan triangle inequality `|j - k| <= j` fails), and `clebsch_gordan`
    already returns `0.0` in that case, so this direct evaluation is well-defined for
    every `k >= 0` and agrees with `SpinJ.synthetic_spam_matrix` wherever both are
    defined (verified against the paper's golden Tables `variancewithk`/`variancewithj`
    in `test_su2rb.py`).
    """
    return _math.sqrt((2 * k + 1) / (2 * j + 1)) * _wg.clebsch_gordan(j, ell, k, 0, j, ell)


def _variance_C(k, kp):
    """
    `C(k, k') = <k 0; k 0 | k' 0>**2` from Eq. `SSvariance`, the rank-1 synthetic-SPAM
    variance weight.
    """
    return _wg.clebsch_gordan(k, 0, k, 0, kp, 0) ** 2


def predicted_zero_noise_variance(j, k):
    """
    The paper's exact zero-noise (perfect-gate) sampling variance of the irrep-`k`
    estimator for the rank-1 synthetic-SPAM randomized benchmarking protocol
    (SSR1RB; Section "Sample Complexity", Eq. `SSvariance`). This is a diagnostic
    for the protocol's sample complexity -- smaller is better -- and does not depend
    on gate noise (the paper's variance result is exact only in the zero-noise
    limit).

    Parameters
    ----------
    j : int, float, or Fraction
        The spin. `2*j` must be a non-negative integer.

    k : int
        The irrep label. Must satisfy `0 <= k <= 2*j`.

    Returns
    -------
    float
    """
    j_float, two_j = _validate_spin(j)
    if not (isinstance(k, (int, _np.integer)) and 0 <= k <= two_j):
        raise ValueError("k must be an integer with 0 <= k <= 2*j=%d; got %r" % (two_j, k))

    spins = [j_float - i for i in range(two_j + 1)]
    total = 0.0
    for kp in range(0, 2 * k + 1):
        inner = sum(_variance_M_entry(j_float, k, ell) ** 2 * _variance_M_entry(j_float, kp, ell)
                    for ell in spins)
        total += _variance_C(k, kp) / (2 * kp + 1) * inner ** 2
    quartic = sum(_variance_M_entry(j_float, k, ell) ** 4 for ell in spins)
    return (2 * k + 1) ** 2 * total - quartic


# ***************************************************************************************************
# SyntheticSPAMRank1RB
# ***************************************************************************************************

def _decay_model(x, a, f):
    """`A_k * f_k**x` -- the per-irrep decay model (Section 3's "no additive offset"
    convention: the trivial irrep k=0 is itself one of the fitted series)."""
    return a * f ** x


def _fit_irrep_decay(x, y, p0=(1.0, 0.9)):
    """
    Fit `y ~ a * f**x` (no additive offset) via `scipy.optimize.curve_fit`, with `f`
    (the per-irrep decay parameter) bounded to `[0, 1]` and `a` (the amplitude)
    unbounded. Ports the `a * b**x` parameterization of the `su2-rb-conservative`
    branch's `Analysis.fit_exp` (`fitlog=False`).

    Returns
    -------
    a, f, sigma_a, sigma_f : float
        The fitted amplitude/decay and their estimated standard errors (`nan` if the
        fit failed).

    success : bool
    """
    x = _np.asarray(x, dtype=float)
    y = _np.asarray(y, dtype=float)
    try:
        # Promote "covariance could not be estimated" (e.g. too few depths for the
        # 2-parameter fit) from a warning to an exception, caught below, so this
        # function's own success/failure contract doesn't depend on the caller's
        # ambient warnings configuration (pyGSTi's pytest.ini turns
        # scipy.optimize.OptimizeWarning into an error project-wide).
        with _warnings.catch_warnings():
            _warnings.simplefilter('error', _OptimizeWarning)
            # Tight xtol/ftol/gtol: the default (~1e-8) `trf` stopping tolerances leave
            # a ~1e-6-level residual on (near-)degenerate series -- e.g. the always-
            # exactly-1 k=0 series -- that this module's analytic cross-validation
            # tests (test_su2rb.py) need to resolve well below that level.
            popt, pcov = _curve_fit(_decay_model, x, y, p0=p0,
                                     bounds=([-_np.inf, 0.0], [_np.inf, 1.0]), maxfev=10000,
                                     xtol=1e-14, ftol=1e-14, gtol=1e-14)
        a, f = float(popt[0]), float(popt[1])
        sigma_a, sigma_f = (float(s) for s in _np.sqrt(_np.diag(pcov)))
        success = True
    except (RuntimeError, _np.linalg.LinAlgError, ValueError, _OptimizeWarning):
        a, f, sigma_a, sigma_f, success = _np.nan, _np.nan, _np.nan, _np.nan, False
    return a, f, sigma_a, sigma_f, success


class SyntheticSPAMRank1RB(_proto.Protocol):
    """
    The rank-1 ("Legendre-weighted") synthetic-SPAM randomized benchmarking protocol
    (SSR1RB) for an arbitrary-spin SU(2) system: reconstructs, per depth, the
    `dim`-by-`dim` [prep, effect] probability matrix for each sampled circuit
    sequence, weights the per-irrep-`k` `diag(M @ P @ M.T)` sandwich by `(2k+1) *
    P_k(cos(beta_hidden))` -- the Legendre "character core" of that sequence's hidden
    first gate (the design's `charcores` aux data) -- averages over sequences (with
    standard errors), fits each irrep's averaged series to `A_k * f_k**x` (`x` =
    depth + 1, no additive offset), and recovers per-irrep rates `p = solve(F, f)`
    with propagated covariance `Sigma_p = F^-1 diag(sigma_f**2) F^-T`.

    The rank-1 weighting makes the resulting per-irrep decay estimate robust to
    arbitrary (fixed, gate-independent) SPAM error, unlike an unweighted `diag(M @ P
    @ M.T)` sandwich, which requires SPAM diagonal in the Jz eigenbasis.

    Requires `SU2RBDesign` data (raises `TypeError` on `run` otherwise, via
    `_per_sequence_irrep_values`).

    Ports `Analysis.fit_and_get_rates` / `fit_and_plot_with_rates_stderrors` and
    `SU2CharacterRBSim.character_transform` / `synspam_character_transform` from the
    `su2-rb-conservative` branch (minus the plotting). The latter two are normative
    for the weight-normalization factors: `character_transform` scales each irrep's
    contribution by `irrep_sizes[k] = 2k + 1` before averaging over circuit sequences
    (see that function's `wchars = _chars * irrep_sizes[...]` line), and
    `synspam_character_transform` then sandwiches the per-irrep-`k` (weighted)
    transition-probability matrix with `M`'s `k`-th row on both sides (`row_M_k @
    block @ row_M_k`) -- algebraically identical, for a fixed depth, to averaging
    `irrep_sizes[k] * charcore[s, k] * (M @ P^(s) @ M.T)[k, k]` over sampled
    sequences `s`, which is what `_per_sequence_irrep_values` below computes
    per-sequence (before `run` averages over `s`).

    Parameters
    ----------
    seed : (float, float), optional
        The `(a, f)` seed passed to `scipy.optimize.curve_fit` for each irrep's decay
        fit.

    name : str, optional
        The name of this protocol.
    """

    def __init__(self, seed=(1.0, 0.9), name=None):
        super(SyntheticSPAMRank1RB, self).__init__(name)
        self.seed = tuple(seed)

    @staticmethod
    def _reconstruct_prep_effect_probs(edesign, ds, depth_idx):
        """
        Shape `(circuits_per_depth, dim, dim)`: `P[s, l, l']` is the probability of
        measuring effect `l'` given preparation `l`, for the `s`-th sampled circuit
        sequence at depth index `depth_idx`, reconstructed from `ds`'s recorded
        outcome fractions for the `dim` circuits (one per prep) sharing that sequence.
        """
        dim = edesign.dim
        circuits_per_depth = edesign.circuits_per_depth
        circuits = edesign.circuit_lists[depth_idx]
        seq_at_depth = edesign.seq_index[depth_idx]
        prep_at_depth = edesign.prep_index[depth_idx]
        outcome_labels = [(str(ell),) for ell in range(dim)]

        P = _np.zeros((circuits_per_depth, dim, dim))
        for circuit, seq_idx, prep_idx in zip(circuits, seq_at_depth, prep_at_depth):
            fracs = ds[circuit].fractions
            P[seq_idx, prep_idx, :] = [fracs.get(ol, 0.0) for ol in outcome_labels]
        return P

    def _per_sequence_irrep_values(self, edesign, ds, depth_idx, M):
        """
        Shape `(circuits_per_depth, dim)`:
        `X[s, k] = irrep_sizes[k] * charcores[s, k] * diag(M @ P^(s) @ M.T)[k]`, where
        `P^(s)` is the `dim`-by-`dim` [prep, effect] probability matrix for the `s`-th
        sampled sequence at depth index `depth_idx` (`_reconstruct_prep_effect_probs`),
        `charcores` is the design's per-sequence Legendre-character-core aux data
        (broadcast to each of that sequence's `dim` prep circuits), and
        `irrep_sizes[k] = 2k + 1`.
        """
        if not hasattr(edesign, 'charcores'):
            raise TypeError(
                "SyntheticSPAMRank1RB requires an SU2RBDesign (with a "
                "'charcores' aux list); got %s" % type(edesign).__name__)

        dim = edesign.dim
        circuits_per_depth = edesign.circuits_per_depth
        P = self._reconstruct_prep_effect_probs(edesign, ds, depth_idx)

        seq_at_depth = edesign.seq_index[depth_idx]
        charcores_at_depth = edesign.charcores[depth_idx]
        W = _np.zeros((circuits_per_depth, dim))
        for seq_idx, charcore_row in zip(seq_at_depth, charcores_at_depth):
            # Every one of the `dim` circuits sharing a sampled sequence carries the
            # same (hidden-first-gate-derived) charcore row, so this repeatedly
            # overwrites W[seq_idx, :] with an identical value -- harmless.
            W[seq_idx, :] = charcore_row

        irrep_sizes = 2 * _np.arange(dim) + 1
        MPM = _np.einsum('kl,slm,km->sk', M, P, M)
        return irrep_sizes[_np.newaxis, :] * W * MPM

    def _fit_and_get_rates(self, x, per_irrep_series, F):
        """
        Fit each irrep's averaged series (`per_irrep_series[k, :]`, aligned with `x`)
        to `A_k * f_k**x`, then recover per-irrep rates and their covariance.

        Returns
        -------
        fits : list of rbfit.FitResults
            One per irrep, in `FitResults`-compatible form: `estimates = {'a': 0.0
            (fixed), 'b': A_k, 'p': f_k}` (matching `rbfit`'s `a + b*p**m` convention
            with the offset `a` fixed to `0`).

        f, sigma_f : numpy array, shape (dim,)
        p, cov_p : numpy array, shapes (dim,) and (dim, dim)
        """
        dim = per_irrep_series.shape[0]
        fits = []
        f = _np.zeros(dim)
        sigma_f = _np.zeros(dim)
        failed_irreps = []
        for k in range(dim):
            a, fk, sigma_a, sigma_fk, success = _fit_irrep_decay(x, per_irrep_series[k, :], p0=self.seed)
            f[k], sigma_f[k] = fk, sigma_fk
            if not success:
                failed_irreps.append(k)
            estimates = {'a': 0.0, 'b': a, 'p': fk}
            variable = {'a': False, 'b': True, 'p': True}
            stds = {'a': 0.0, 'b': sigma_a, 'p': sigma_fk}
            fits.append(_rbfit.FitResults('synspam-no-offset', list(self.seed), None, success,
                                           estimates, variable, stds=stds))

        if failed_irreps:
            # A failed fit leaves `nan`s in `f`/`sigma_f` for just the failed irrep(s),
            # but `p = solve(F, f)` and `cov_p` mix *every* irrep together (F is not
            # block-diagonal), so a single failure silently poisons every entry of
            # `p`/`cov_p` with `nan`, not just the failed irrep's. Surface that loudly
            # rather than letting it look like an ordinary (if strange) all-nan result.
            _warnings.warn(
                "SyntheticSPAMRank1RB: the per-irrep decay fit failed for irrep(s) %s; "
                "the recovered rates and covariance (which mix all irreps via F) are "
                "therefore entirely nan, not just the failed irrep(s)." % failed_irreps,
                UserWarning)

        invF = _spl.inv(F)
        p = invF @ f
        cov_p = invF @ _np.diag(sigma_f ** 2) @ invF.T
        return fits, f, sigma_f, p, cov_p

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data, from an `SU2RBDesign` and a `DataSet` with matching
            circuits.

        memlimit : int, optional
            Unused (present for `Protocol` interface compatibility).

        comm : mpi4py.MPI.Comm, optional
            Unused (present for `Protocol` interface compatibility).

        Returns
        -------
        SyntheticSPAMRBResults
        """
        edesign = data.edesign
        ds = data.dataset
        j, dim = edesign.j, edesign.dim
        spinj = _su2.SpinJ(j)
        M = spinj.synthetic_spam_matrix
        F = spinj.decay_recoupling_matrix

        depths = list(edesign.depths)
        num_depths = len(depths)

        means = _np.zeros((dim, num_depths))
        stderrs = _np.zeros((dim, num_depths))
        variances = _np.zeros((dim, num_depths))
        for depth_idx in range(num_depths):
            X = self._per_sequence_irrep_values(edesign, ds, depth_idx, M)
            n = X.shape[0]
            means[:, depth_idx] = X.mean(axis=0)
            var = X.var(axis=0, ddof=1) if n > 1 else _np.zeros(dim)
            variances[:, depth_idx] = var
            stderrs[:, depth_idx] = _np.sqrt(var / n)

        x = _np.array(depths, dtype=float) + 1.0
        fits, f, sigma_f, p, cov_p = self._fit_and_get_rates(x, means, F)

        return SyntheticSPAMRBResults(data, self, depths, means, stderrs, variances, fits, f, sigma_f, p, cov_p)


class SyntheticSPAMRBResults(_proto.ProtocolResults):
    """
    The results of running `SyntheticSPAMRank1RB` on synthetic-SPAM RB data.

    Parameters
    ----------
    data : ProtocolData
        The data these results were computed from.

    protocol_instance : SyntheticSPAMRank1RB
        The protocol that produced these results.

    depths : list of int
        The RB circuit depths (in the same order as the columns of `means`/`stderrs`/
        `variances`).

    means, stderrs, variances : numpy array
        Shape `(dim, len(depths))`: the per-irrep, per-depth sample mean, standard
        error, and (sequence-count-unnormalized) sample variance of the per-sequence
        estimator `X_k` (`SyntheticSPAMRank1RB._per_sequence_irrep_values`).

    fits : list of rbfit.FitResults
        One per irrep `k = 0..dim-1`.

    f, sigma_f : numpy array, shape (dim,)
        The fitted per-irrep decay parameters and their standard errors.

    p, cov_p : numpy array, shapes (dim,) and (dim, dim)
        The recovered per-irrep rates (`solve(F, f)`) and their propagated covariance
        (`F^-1 diag(sigma_f**2) F^-T`).

    Attributes
    ----------
    j : float
        The spin.

    dim : int
        `2*j + 1`.
    """

    def __init__(self, data, protocol_instance, depths, means, stderrs, variances, fits, f, sigma_f, p, cov_p):
        super(SyntheticSPAMRBResults, self).__init__(data, protocol_instance)
        self.j = data.edesign.j
        self.dim = data.edesign.dim
        self.depths = list(depths)
        self.per_irrep_means = means
        self.per_irrep_stderrs = stderrs
        self.per_irrep_variances = variances
        self.fits = fits
        self.decays = f
        self.decay_stderrs = sigma_f
        self.rates = p
        self.rates_covariance = cov_p

        # Register the ndarray-valued and FitResults-list-valued attributes so
        # write()/read-from-dir round-trip: the base ProtocolResults.auxfile_types
        # only covers 'data'/'protocol', and plain JSON serialization (the
        # default for un-registered attributes) cannot handle numpy arrays or
        # rbfit.FitResults objects.
        for attr in ('per_irrep_means', 'per_irrep_stderrs', 'per_irrep_variances',
                     'decays', 'decay_stderrs', 'rates', 'rates_covariance'):
            self.auxfile_types[attr] = 'numpy-array'
        self.auxfile_types['fits'] = 'list:serialized-object'

    def rates_dataframe(self):
        """
        A `pandas.DataFrame` rate-table export, with one row per irrep `k`.

        Returns
        -------
        pandas.DataFrame
            Columns `irrep`, `decay_f`, `decay_f_stderr`, `rate_p`, `rate_p_stderr`.
        """
        return _pd.DataFrame({
            'irrep': _np.arange(self.dim),
            'decay_f': self.decays,
            'decay_f_stderr': self.decay_stderrs,
            'rate_p': self.rates,
            'rate_p_stderr': _np.sqrt(_np.diag(self.rates_covariance)),
        })

    def variance_diagnostic(self, depth_index=0):
        """
        An order-of-magnitude sanity check comparing SSR1RB's predicted zero-noise
        `Var(X_k)` (`predicted_zero_noise_variance`) against the empirical sample
        variance of the per-sequence irrep estimator at depth
        `self.depths[depth_index]`. Since the prediction is exact only at zero gate
        noise, this is a sanity check (expected order-of-magnitude agreement at low
        noise/short depths), not an exact-agreement test.

        Parameters
        ----------
        depth_index : int, optional
            Which of `self.depths` to compare the empirical variance at.

        Returns
        -------
        dict
            Maps irrep label `k` -> `(predicted, empirical)`.
        """
        return {k: (predicted_zero_noise_variance(self.j, k),
                    float(self.per_irrep_variances[k, depth_index]))
                for k in range(self.dim)}

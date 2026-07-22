# ***************************************************************************************************
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Efficient randomized benchmarking over SU(2) for arbitrary spin qudit systems,
following pyGSTi's `Design -> DataSimulator -> Protocol -> Results` architecture.

This module implements the *rank-1 synthetic SPAM RB* protocol introduced in

    Randomized Benchmarking with Synthetic Quantum Circuits, by Yale Fan,
    Riley Murray, Thaddeus D Ladd, Kevin Young and Robin Blume-Kohout.
    Published in Quantum Science and Technology, 2026.

The protocol generates circuits in batches of size `2j+1`. Circuits within a batch
differ only in their state prep, each of which is a distinct Jz-eigenstate.
The gate sequence for a batch consists of `m+1` SU(2) elements for some m >= 1;
we represent it by an array of shape `(m+1, 3)` whose rows are ZXZ Euler angles.
Rows `0..m-1` are Haar-random, while row `m` encodes the inverse of gates `1..m-1`.

The numeric metadata needed to run RB analysis on these circuits is carried as
`paired_with_circuit_attrs` JSON aux data.

Notation
--------
We denote the state prep targets as `rho{ell}` for `ell = 0..2j`, so `rho0`
prepares `|j>` and `rho{2j}` prepares `|-j>`. We denote the ZXZ Euler
angles of an SU(2) element by `(alpha, beta, gamma)`.
"""

import math as _math
import warnings as _warnings
from fractions import Fraction as _Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as _np
import pandas as _pd
import scipy.linalg as _spl
from scipy.optimize import curve_fit as _curve_fit

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
from pygsti.tools.exceptions import RBFitFailureWarning as _RBFitFailureWarning
from pygsti.tools.su2tools import circuit_from_euler_angles, euler_angles_from_circuit, GATE_NAME

__all__ = [
    'circuit_from_euler_angles',
    'euler_angles_from_circuit',
    'SU2QuditRBDesign',
    'jz_dephasing',
    'jz_rotation',
    'SU2QuditRBSimulator',
    'predicted_zero_noise_variance',
    'SU2QuditRB',
    'SU2QuditRBResults',
]

ChannelFactory_t = Callable[[float, float, float], _np.ndarray]
"""A gate-dependent noise factory: maps a gate's ZXZ Euler angles to the
`(dim**2, dim**2)` noise superoperator applied after that gate."""


def _r1rb_circuit(angles: _np.ndarray, prep_index: int, qudit_label: str) -> _Circuit:
    """
    The full R1RB circuit for one (sequence, prep) pair: a `rho{prep_index}` prep
    layer, one `Gu` layer per row of `angles`, and an `Mdefault` POVM layer.
    """
    layers = [_Label(f'rho{prep_index}')]
    layers += [_Label(GATE_NAME, qudit_label, args=tuple(float(x) for x in row)) for row in angles]
    layers += [_Label('Mdefault')]
    circuit = _Circuit(layers, line_labels=(qudit_label,))
    return circuit


class SU2QuditRBDesign(_vb.BenchmarkingDesign):
    """
    Experiment design for "rank-1 synthetic SPAM RB" (R1RB) of an arbitrary-spin SU(2) system.

    The design generates circuits in batches of size `2j+1`. Circuits within a batch
    differ only in their state prep, each of which is a distinct Jz-eigenstate.
    The gate sequence for a batch consists of `m+1` SU(2) elements for some m >= 1;
    we represent it by an array of shape `(m+1, 3)` whose rows are ZXZ Euler angles.
    Rows `0..m-1` are Haar-random, while row `m` encodes the inverse of gates `1..m-1`.

    Additionally stores, per circuit, the Legendre "character core" `P_k(cos(beta))`
    of the hidden first gate (`charcores`), for `k = 0..2j`, which `SU2QuditRB` uses
    to weight its per-irrep estimator.

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

    Notes
    -----
    Because the net ideal composition is the random hidden first gate rather than the identity,
    `idealout_lists` entries are *not* genuine deterministic ideal outcomes. They are
    just placeholders required by `BenchmarkingDesign.__init__`, which needs some `ideal_outs`
    value per circuit. `SU2QuditRB` reconstructs probabilities directly from `euler_angles` and
    `charcores` aux data, not from `idealout_lists`.
    """

    def __init__(self, j: _su2.SpinSpec_t, depths: Sequence[int], circuits_per_depth: int,
                 seed: Optional[int] = None, qudit_label: str = 'Q0',
                 descriptor: str = 'An SU(2) R1RB experiment') -> None:
        j_float, two_j = _validate_spin(j)
        dim = two_j + 1
        depths = list(depths)

        if seed is None:
            seed = _np.random.randint(1, 1000000)
        self.seed = seed

        sampled = SU2QuditRBDesign._sample_circuits(dim, depths, circuits_per_depth, self.seed, qudit_label)

        self.paired_with_circuit_attrs = ["euler_angles", "seq_index", "prep_index", "charcores"]
        # ^ Set before calling super().__init__ so that BenchmarkingDesign inserts 'idealout_lists'
        #   at the front and registers 'json' auxfile types for these members.

        super(SU2QuditRBDesign, self).__init__(
            depths, sampled['circuit_lists'], sampled['idealout_lists'],
            qubit_labels=(qudit_label,), remove_duplicates=False
        )

        self.euler_angles = sampled['euler_angles_lists']
        self.seq_index    = sampled['seq_index_lists']
        self.prep_index   = sampled['prep_index_lists']
        self.charcores    = sampled['charcores_lists']

        self.j = j_float
        self.dim = dim
        self.qudit_label = qudit_label
        self.circuits_per_depth = circuits_per_depth
        self.descriptor = descriptor

    @staticmethod
    def _sample_circuits(dim: int, depths: Sequence[int], circuits_per_depth: int, seed: Any, qudit_label: str):
        """
        Core RB circuit sampler for SU2QuditRBDesign; returns a dict holding the
        per-depth member lists of an SU2QuditRBDesign (circuits, placeholder ideal
        outcomes, and the euler_angles/seq_index/prep_index/charcores aux data).
        """
        rng = _np.random.default_rng(seed)
        irrep_labels = _np.arange(dim)

        circuit_lists, idealout_lists = [], []
        euler_angles_lists, seq_index_lists, prep_index_lists = [], [], []
        charcores_lists = []

        for m in depths:
            if m < 1:
                raise ValueError(f"SU(2) RB depths must be >= 1 (got {m!r})")

            circuits_at_depth, idealouts_at_depth = [], []
            euler_angles_at_depth, seq_index_at_depth, prep_index_at_depth = [], [], []
            charcores_at_depth = []

            for s in range(circuits_per_depth):
                angles = _np.zeros((m + 1, 3))
                a, b, g = _su2.random_euler_angles(m, rng=rng)
                angles[:m, 0], angles[:m, 1], angles[:m, 2] = a, b, g
                a_inv, b_inv, g_inv = _su2.composition_inverse(
                    angles[1:m, 0], angles[1:m, 1], angles[1:m, 2])
                angles[m, :] = (a_inv, b_inv, g_inv)

                angles_as_list = angles.tolist()
                charcore_row = _su2.charactercores_from_euler_angles(irrep_labels, angles[:1, :])[0].tolist()
                # ^ The Legendre character cores P_k(cos(beta)) of the hidden first gate,
                #   shared by the `dim` per-prep circuits generated from this sequence.
                for ell in range(dim):
                    circuits_at_depth.append(_r1rb_circuit(angles, ell, qudit_label))
                    idealouts_at_depth.append((str(ell),))
                    euler_angles_at_depth.append(angles_as_list)
                    seq_index_at_depth.append(s)
                    prep_index_at_depth.append(ell)
                    charcores_at_depth.append(charcore_row)

            circuit_lists.append(circuits_at_depth)
            idealout_lists.append(idealouts_at_depth)
            euler_angles_lists.append(euler_angles_at_depth)
            seq_index_lists.append(seq_index_at_depth)
            prep_index_lists.append(prep_index_at_depth)
            charcores_lists.append(charcores_at_depth)

        result = dict(circuit_lists=circuit_lists, idealout_lists=idealout_lists,
                      euler_angles_lists=euler_angles_lists, seq_index_lists=seq_index_lists,
                      prep_index_lists=prep_index_lists, charcores_lists=charcores_lists)
        return result


# ***************************************************************************************************
# Noise channels for SU2QuditRBSimulator
# ***************************************************************************************************

def jz_dephasing(spinj: _su2.SpinJ, gamma: float, power: float = 1.0) -> _np.ndarray:
    """
    A dephasing noise channel, diagonal in the standard (matrix-unit) basis. It damps
    element (i, j) of a density matrix by `exp(-gamma * |i - j|**power)`.

    Parameters
    ----------
    spinj : SpinJ
        The representation whose (`dim`-by-`dim`) density matrices this channel acts on.

    gamma : float
        The dephasing rate. `gamma == 0` gives the identity channel.

    power : float, optional
        The power `p` in `|i - j|**p` (`p=1` gives an exponential decay profile,
        `p=2` a Gaussian one).

    Returns
    -------
    numpy array
        Shape `(dim**2, dim**2)`, a diagonal superoperator acting on row-major-raveled
        density matrices.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative (got {gamma!r})")
    dim = spinj.dim
    ell = _np.arange(dim * dim)
    i = ell % dim
    j = ell // dim
    diag = _np.exp(-gamma * _np.abs(i - j).astype(float) ** power)
    superop = _np.diag(diag).astype(complex)
    return superop


def jz_rotation(spinj: _su2.SpinJ, theta: float, power: float = 1.0) -> _np.ndarray:
    """
    A unitary "Z-rotation" noise channel: `U = diag(exp(1j * theta * spins**power))`,
    in the `spinj.spins`-ordered basis for Hilbert space. Returned as a superoperator
    in the standard (matrix-unit) basis for Hilbert--Schmidt space.

    Parameters
    ----------
    spinj : SpinJ
        The representation whose `spins` this channel's generator is diagonal in.

    theta : float
        The rotation angle.

    power : float, optional
        The power `p` in `spins**p` (`p=1` rotates about the physical Jz axis; `p=2`
        gives a quadratic-in-m variant).

    Returns
    -------
    numpy array
        Shape `(dim**2, dim**2)`, the superoperator of a unitary channel.
    """
    U = _np.diag(_np.exp(1j * theta * (spinj.spins ** power)))
    S = _unitary_to_std_process_mx(U)
    return S


# ***************************************************************************************************
# SU2QuditRBSimulator
# ***************************************************************************************************

class SU2QuditRBSimulator(_proto.DataSimulator):
    """
    Simulates data for an SU(2) RB experiment design (more specifically, a design
    using rank-1 synthetic SPAM RB). 

    The simulation can be noiseless or use a provided post-gate error channel.
    
    Parameters
    ----------
    spinj : SpinJ or (int, float, or Fraction)
        The representation to simulate in, or a spin `j` from which one is built.

    noise_channel : None, numpy array, or callable, optional
        The noise channel applied after each gate other than the first.
        
        Array values indicate a fixed post-gate error and must be superoperator
        that acts on a row-major vectorization of a density matrix.

        Callable values will be invoked with a given gate's Euler angles as
        `noise_channel(alpha, beta, gamma)`, and must return a valid array in
        the sense above.

        Passing `None` is equivalent to the identity channel.

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

    Notes
    -----
    Reads the design's `euler_angles` aux data directly -- it never parses circuit
    labels -- and builds the ZXZ-angle unitaries in batch via the instance's
    `SpinJ`'s precomputed eigendecompositions, inserting a noise channel after every
    gate except the "hidden" first gate.

    See also
    --------
    This module's `jz_dephasing` and `jz_rotation` functions for simple post-gate
    error channels.
    """

    def __init__(self, spinj: _su2.SpinJCastable_t,
                 noise_channel: Union[ChannelFactory_t, _np.ndarray, None] = None,
                 shots: Optional[int] = None, seed: Optional[int] = None
        ) -> None:
        super(SU2QuditRBSimulator, self).__init__()
        self.spinj = _su2.SpinJ.cast(spinj)
        self.dim   = self.spinj.dim
        self.shots = shots
        self.seed  = seed

        self._noise_superop = None
        self._noise_factory = None
        if callable(noise_channel):
            self._noise_factory = noise_channel
        elif isinstance(noise_channel, _np.ndarray):
            dim2 = self.dim ** 2
            if noise_channel.shape != (dim2, dim2):
                raise ValueError(f"noise_channel must have shape ({dim2}, {dim2}); got {noise_channel.shape}")
            self._noise_superop = noise_channel
        elif noise_channel is not None:
            raise ValueError(f'Unsupported argument noise_channel={noise_channel}.')
        return

    @property
    def is_noiseless(self) -> bool:
        """True if this simulator applies no noise (`noise_channel=None`)."""
        return self._noise_superop is None and self._noise_factory is None

    # -----------------------------------------------------------------
    # Core composition machinery
    # -----------------------------------------------------------------

    def _compose_full(self, angles: _np.ndarray) -> _np.ndarray:
        """
        Explicitly compose the `(dim**2, dim**2)` superoperators for the full gate
        sequence, inserting a noise superoperator after every gate except the first.
        
        If `noise_channel` was a fixed array, the same superoperator is inserted after
        every gate; if it was a callable factory, the factory is called once per gate
        with that gate's own `(alpha, beta, gamma)` row to get the channel inserted
        after it.
        """
        dim2 = self.dim * self.dim
        Us = self.spinj.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
        S = _np.eye(dim2, dtype=complex)
        skip_next_noise = True
        for row, U in zip(angles, Us):
            S = _unitary_to_std_process_mx(U) @ S
            if not skip_next_noise:
                if self._noise_factory is not None:
                    S = self._noise_factory(row[0], row[1], row[2]) @ S
                elif self._noise_superop is not None:
                    S = self._noise_superop @ S
            skip_next_noise = False
        return S

    def _compose(self, angles: _np.ndarray) -> _np.ndarray:
        if self.is_noiseless:
            # The composition reduces to the action of the hidden layer.
            U0 = self.spinj.unitaries_from_angles(angles[0, 0], angles[0, 1], angles[0, 2])[0]
            superop = _unitary_to_std_process_mx(U0)
            return superop
        return self._compose_full(angles)

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
        sequences = [seen[k] for k in sorted(seen.keys())]
        return sequences

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
        dim2 = dim * dim
        ops = _np.zeros((dim, dim2), dtype=complex)
        for ell in range(dim):
            ops[ell, ell * dim + ell] = 1.0
        return ops

    # -----------------------------------------------------------------
    # Composition cache / SPAM-swap API
    # -----------------------------------------------------------------

    def compute_nonspam_compositions(self, edesign: SU2QuditRBDesign,
                                      depth_indices: Optional[Sequence[int]] = None) -> Dict[int, _np.ndarray]:
        """
        Compute the composed noisy (non-SPAM) superoperator for each distinct sampled
        circuit sequence, at the requested depths.

        This is the capability behind the paper's SPAM-robustness figures: once
        computed, `probabilities_from_compositions` can re-derive probabilities under
        arbitrary (e.g. perturbed) state preps/POVMs without re-simulating circuits.

        Parameters
        ----------
        edesign : SU2QuditRBDesign
            The design whose `euler_angles`/`seq_index` aux data to read.

        depth_indices : iterable of int, optional
            Which depth indices (into `edesign.depths`) to compute compositions for.
            If `None` (the default), all depths are computed. Each depth's cache holds
            `circuits_per_depth * dim**4` complex doubles, so pass an explicit subset
            to bound memory usage for large designs.

        Returns
        -------
        dict
            Maps depth index -> numpy array of shape
            `(circuits_per_depth, dim**2, dim**2)`, the composed superoperator for
            each sampled sequence at that depth (ordered by `seq_index`).
        """
        if depth_indices is None:
            depth_indices = range(len(edesign.depths))
        out = {}
        for depth_idx in depth_indices:
            sequences = self._unique_sequences_at_depth(edesign, depth_idx)
            out[depth_idx] = _np.array([self._compose(angles) for angles in sequences])
        return out

    def probabilities_from_compositions(self, compositions: Dict[int, _np.ndarray],
                                         statepreps: Optional[_np.ndarray] = None,
                                         povm: Optional[_np.ndarray] = None) -> Dict[int, _np.ndarray]:
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
            evolved = _np.einsum('sab,ib->sia', compositions_at_depth, statepreps)
            # ^ evolved[s, i, :] = compositions_at_depth[s] @ statepreps[i]
            probs = _np.einsum('ek,sik->sie', _np.conj(povm), evolved)
            # ^ probs[s, i, e] = <povm[e] | evolved[s, i]>
            #                  = Tr(povm_mx[e] . prep_mx[i] after composition s)
            imag_norm = _np.linalg.norm(probs.imag)
            if imag_norm > 1e-8:
                message = (
                    f"probabilities_from_compositions produced non-negligible "
                    f"imaginary part (norm {imag_norm:g}); statepreps/povm may not "
                    f"be Hermitian"
                )
                raise ValueError(message)
            out[depth_idx] = probs.real
        return out

    # -----------------------------------------------------------------
    # run()
    # -----------------------------------------------------------------

    def _sample_counts(self, probs, outcome_labels, rng):
        if self.shots is None:
            exact_counts = {ol: float(p) for ol, p in zip(outcome_labels, probs)}
            return exact_counts
        clipped = _np.clip(probs, 0, None)
        total = clipped.sum()
        clipped = clipped / total if total > 0 else _np.full(len(probs), 1.0 / len(probs))
        counts = rng.multinomial(self.shots, clipped)
        sampled_counts = {ol: int(c) for ol, c in zip(outcome_labels, counts)}
        return sampled_counts

    def run(self, edesign: SU2QuditRBDesign, memlimit: Optional[int] = None,
            comm: Optional[Any] = None) -> _proto.ProtocolData:
        """
        Simulate SU(2) synthetic SPAM RB data for `edesign`.

        Parameters
        ----------
        edesign : SU2QuditRBDesign
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
            message = (
                f"edesign.dim={edesign.dim} does not match this simulator's "
                f"spinj.dim={self.dim}"
            )
            raise ValueError(message)

        outcome_labels = [str(ell) for ell in range(self.dim)]
        rng = _np.random.default_rng(self.seed) if self.shots is not None else None
        ideal_ops = self._ideal_computational_ops()

        ds = _DataSet(collision_action="aggregate")
        ds.repType = _np.float64
        # ^ float32 repcounts (DataSet's default) are too coarse for exact-probability counts.
        for depth_idx, circuits_at_depth in enumerate(edesign.circuit_lists):
            seq_compositions = {}
            angles_at_depth = edesign.euler_angles[depth_idx]
            seq_at_depth = edesign.seq_index[depth_idx]
            prep_at_depth = edesign.prep_index[depth_idx]
            for circuit, angles, seq_idx, prep_idx in zip(
                    circuits_at_depth, angles_at_depth, seq_at_depth, prep_at_depth):
                if seq_idx not in seq_compositions:
                    seq_compositions[seq_idx] = self._compose(_np.asarray(angles, dtype=float))
                S = seq_compositions[seq_idx]

                evolved = S @ ideal_ops[prep_idx]
                probs = _np.real(_np.diag(evolved.reshape(self.dim, self.dim)))
                # ^ Tr(|ell'><ell'| . evolved) == evolved.reshape(dim, dim)[ell', ell'],
                #   for the diagonal computational-basis effects.

                counts = self._sample_counts(probs, outcome_labels, rng)
                ds.add_count_dict(circuit, counts)
        ds.done_adding_data()
        protocol_data = _proto.ProtocolData(edesign, ds)
        return protocol_data


# ***************************************************************************************************
# Zero-noise sample-complexity (variance) diagnostics
# ***************************************************************************************************

def _variance_M_entry(j, k, ell):
    """
    `M[k, ell] = sqrt((2k+1)/(2j+1)) * <j ell; k 0 | j ell>`, the same formula as
    `SpinJ.synthetic_spam_matrix`, but evaluated directly via
    `wignersymbols.clebsch_gordan` so it is well-defined for *any* non-negative
    integer `k`, not just `k` in `0..2j`. That matters because the paper's variance
    sums (Eqs. `normalizedvariance`/`SSvariance`) range the inner irrep label `k'`
    over `0..2k`, which exceeds `2j` once `k > j`; entries with `k > 2j` are exactly
    zero by the Clebsch-Gordan triangle inequality.
    """
    value = _math.sqrt((2 * k + 1) / (2 * j + 1)) * _wg.clebsch_gordan(j, ell, k, 0, j, ell)
    return value


def _variance_C(k, kp):
    """
    `C(k, k') = <k 0; k 0 | k' 0>**2` from Eq. `SSvariance`, the rank-1 synthetic-SPAM
    variance weight.
    """
    value = _wg.clebsch_gordan(k, 0, k, 0, kp, 0) ** 2
    return value


def predicted_zero_noise_variance(j: _su2.SpinSpec_t, k: int) -> float:
    """
    The paper's exact zero-noise (perfect-gate) sampling variance of the irrep-`k`
    estimator for the rank-1 synthetic-SPAM randomized benchmarking protocol
    (R1RB; Section "Sample Complexity", Eq. `SSvariance`). This is a diagnostic for
    the protocol's sample complexity -- smaller is better; it does not depend on gate
    noise, since the paper's variance result is exact only in the zero-noise limit.

    Parameters
    ----------
    j : int, float, or Fraction
        The spin. `2*j` must be a non-negative integer.

    k : int
        The irrep label. Must satisfy `0 <= k <= 2*j`.

    Returns
    -------
    float
        The zero-noise variance `Var(X_k)`; finite and nonnegative.
    """
    j_float, two_j = _validate_spin(j)
    if not (isinstance(k, (int, _np.integer)) and 0 <= k <= two_j):
        raise ValueError(f"k must be an integer with 0 <= k <= 2*j={two_j}; got {k!r}")

    spins = [j_float - i for i in range(two_j + 1)]
    total = 0.0
    for kp in range(0, 2 * k + 1):
        inner = sum(_variance_M_entry(j_float, k, ell) ** 2 * _variance_M_entry(j_float, kp, ell)
                    for ell in spins)
        total += _variance_C(k, kp) / (2 * kp + 1) * inner ** 2
    quartic = sum(_variance_M_entry(j_float, k, ell) ** 4 for ell in spins)
    variance = (2 * k + 1) ** 2 * total - quartic
    return variance


# ***************************************************************************************************
# SU2QuditRB
# ***************************************************************************************************

def _decay_model(x, a, f):
    """`A_k * f_k**x`, the per-irrep decay model (no additive offset -- the trivial
    irrep k=0 is itself one of the fitted series)."""
    value = a * f ** x
    return value


def _fit_irrep_decay(x, y, p0=(1.0, 0.9)):
    """
    Fit `y ~ a * f**x` (no additive offset) via `scipy.optimize.curve_fit`, with `f`
    (the per-irrep decay parameter) bounded to `[0, 1]` and `a` (the amplitude)
    unbounded.

    `success=False` is reserved for genuine fit exceptions (`RuntimeError`,
    `LinAlgError`, `ValueError`). If `curve_fit` converges but cannot estimate a
    covariance (e.g. too few depths), it emits `scipy.optimize.OptimizeWarning` and
    this function reports `success=True` with `sigma_a`/`sigma_f` set to `inf`.

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
        popt, pcov = _curve_fit(_decay_model, x, y, p0=p0,
                                 bounds=([-_np.inf, 0.0], [_np.inf, 1.0]), maxfev=10000,
                                 xtol=1e-14, ftol=1e-14, gtol=1e-14)
        # ^ Tight xtol/ftol/gtol: the default (~1e-8) `trf` stopping tolerances leave a
        #   ~1e-6-level residual on (near-)degenerate series -- e.g. the always-
        #   exactly-1 k=0 series -- that this module's analytic cross-validation tests
        #   (test_su2rb.py) need to resolve well below that level.
        a, f = float(popt[0]), float(popt[1])
        sigma_a, sigma_f = (float(s) for s in _np.sqrt(_np.diag(pcov)))
        success = True
    except (RuntimeError, _np.linalg.LinAlgError, ValueError):
        a, f, sigma_a, sigma_f, success = _np.nan, _np.nan, _np.nan, _np.nan, False
    return a, f, sigma_a, sigma_f, success


class SU2QuditRB(_proto.Protocol):
    """
    The rank-1 ("Legendre-weighted") synthetic-SPAM randomized benchmarking protocol
    (R1RB) for an arbitrary-spin SU(2) system, from the paper's "Rotationally
    Invariant Randomized Benchmarking" section.

    For each depth and each sampled circuit sequence, this protocol reconstructs the
    `dim`-by-`dim` [prep, effect] probability matrix `P` and computes the per-irrep
    estimator `X_k = (2k+1) * P_k(cos(beta_hidden)) * diag(M @ P @ M.T)[k]`, where
    `P_k(cos(beta_hidden))` is the Legendre "character core" of the sequence's hidden
    first gate (the design's `charcores` aux data) and `M` is the SPAM-synthesis
    matrix (paper eq. `Mmaintext`). It averages `X_k` over sequences (with standard
    errors), fits each irrep's series to `A_k * f_k**x` (`x` = depth + 1, no additive
    offset), and recovers per-irrep rates `p = solve(F, f)` with propagated
    covariance `Sigma_p = F^-1 diag(sigma_f**2) F^-T`.

    The rank-1 weighting makes the decay estimates robust to arbitrary (fixed,
    gate-independent) SPAM error, unlike an unweighted `diag(M @ P @ M.T)` sandwich,
    which requires SPAM diagonal in the Jz eigenbasis.

    Requires `SU2QuditRBDesign` data (raises `TypeError` on `run` otherwise).

    Parameters
    ----------
    fit_p0 : (float, float), optional
        The `(a, f)` initial guess passed to `scipy.optimize.curve_fit` for each
        irrep's decay fit.

    name : str, optional
        The name of this protocol.
    """

    def __init__(self, fit_p0: Tuple[float, float] = (1.0, 0.9), name: Optional[str] = None) -> None:
        super(SU2QuditRB, self).__init__(name)
        self.fit_p0 = tuple(fit_p0)

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
            message = (
                f"SU2QuditRB requires an SU2QuditRBDesign (with a 'charcores' "
                f"aux list); got {type(edesign).__name__}"
            )
            raise TypeError(message)

        dim = edesign.dim
        circuits_per_depth = edesign.circuits_per_depth
        P = self._reconstruct_prep_effect_probs(edesign, ds, depth_idx)

        seq_at_depth = edesign.seq_index[depth_idx]
        charcores_at_depth = edesign.charcores[depth_idx]
        W = _np.zeros((circuits_per_depth, dim))
        for seq_idx, charcore_row in zip(seq_at_depth, charcores_at_depth):
            W[seq_idx, :] = charcore_row
            # ^ Every one of the `dim` circuits sharing a sampled sequence carries
            #   the same (hidden-first-gate-derived) charcore row, so this repeatedly
            #   overwrites W[seq_idx, :] with an identical value -- harmless.

        irrep_sizes = 2 * _np.arange(dim) + 1
        MPM = _np.einsum('kl,slm,km->sk', M, P, M)
        weighted = irrep_sizes[_np.newaxis, :] * W * MPM
        return weighted

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
            a, fk, sigma_a, sigma_fk, success = _fit_irrep_decay(x, per_irrep_series[k, :], p0=self.fit_p0)
            f[k], sigma_f[k] = fk, sigma_fk
            if not success:
                failed_irreps.append(k)
            estimates = {'a': 0.0, 'b': a, 'p': fk}
            variable = {'a': False, 'b': True, 'p': True}
            stds = {'a': 0.0, 'b': sigma_a, 'p': sigma_fk}
            fits.append(_rbfit.FitResults('synspam-no-offset', list(self.fit_p0), None, success,
                                           estimates, variable, stds=stds))

        if failed_irreps:
            message = (
                f"SU2QuditRB: the per-irrep decay fit failed for irrep(s) "
                f"{failed_irreps}; the recovered rates and covariance (which mix all "
                f"irreps via F) are therefore entirely nan, not just the failed "
                f"irrep(s)."
            )
            _warnings.warn(message, _RBFitFailureWarning)
            # ^ A failed fit leaves `nan`s in `f`/`sigma_f` for just the failed
            #   irrep(s), but `p = solve(F, f)` and `cov_p` mix *every* irrep together
            #   (F is not block-diagonal), so a single failure silently poisons every
            #   entry of `p`/`cov_p` with `nan`, not just the failed irrep's. Surface
            #   that loudly rather than letting it look like an ordinary (if strange)
            #   all-nan result.

        invF = _spl.inv(F)
        p = invF @ f
        cov_p = invF @ _np.diag(sigma_f ** 2) @ invF.T
        return fits, f, sigma_f, p, cov_p

    def run(self, data: _proto.ProtocolData, memlimit: Optional[int] = None,
            comm: Optional[Any] = None) -> 'SU2QuditRBResults':
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data, from an `SU2QuditRBDesign` and a `DataSet` with matching
            circuits.

        memlimit : int, optional
            Unused (present for `Protocol` interface compatibility).

        comm : mpi4py.MPI.Comm, optional
            Unused (present for `Protocol` interface compatibility).

        Returns
        -------
        SU2QuditRBResults
        """
        edesign = data.edesign
        assert isinstance(edesign, SU2QuditRBDesign)
        ds = data.dataset
        j, dim = edesign.j, edesign.dim
        spinj = _su2.SpinJ(j)
        M = spinj.synthetic_spam_matrix
        F = spinj.decay_recoupling_matrix

        depths = list(edesign.depths)
        num_depths = len(depths)

        means     = _np.zeros((dim, num_depths))
        stderrs   = _np.zeros((dim, num_depths))
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

        results = SU2QuditRBResults(data, self, depths, means, stderrs, variances, fits, f, sigma_f, p, cov_p)
        return results


class SU2QuditRBResults(_proto.ProtocolResults):
    """
    The results of running `SU2QuditRB` on synthetic-SPAM RB data.

    Parameters
    ----------
    data : ProtocolData
        The data these results were computed from.

    protocol_instance : SU2QuditRB
        The protocol that produced these results.

    depths : list of int
        The RB circuit depths (in the same order as the columns of `means`/`stderrs`/
        `variances`).

    means, stderrs, variances : numpy array
        Shape `(dim, len(depths))`: the per-irrep, per-depth sample mean, standard
        error, and (sequence-count-unnormalized) sample variance of the per-sequence
        estimator `X_k` (`SU2QuditRB._per_sequence_irrep_values`).

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

    def __init__(self, data: _proto.ProtocolData, protocol_instance: SU2QuditRB, depths: Sequence[int],
                 means: _np.ndarray, stderrs: _np.ndarray, variances: _np.ndarray, fits: List[_rbfit.FitResults],
                 f: _np.ndarray, sigma_f: _np.ndarray, p: _np.ndarray, cov_p: _np.ndarray) -> None:
        super(SU2QuditRBResults, self).__init__(data, protocol_instance)
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

    def rates_dataframe(self) -> _pd.DataFrame:
        """
        A `pandas.DataFrame` rate-table export, with one row per irrep `k`.

        Returns
        -------
        pandas.DataFrame
            Columns `irrep`, `decay_f`, `decay_f_stderr`, `rate_p`, `rate_p_stderr`.
        """
        df = _pd.DataFrame({
            'irrep': _np.arange(self.dim),
            'decay_f': self.decays,
            'decay_f_stderr': self.decay_stderrs,
            'rate_p': self.rates,
            'rate_p_stderr': _np.sqrt(_np.diag(self.rates_covariance)),
        })
        return df

    def variance_diagnostic(self, depth_index: int = 0) -> Dict[int, Tuple[float, float]]:
        """
        An order-of-magnitude sanity check comparing R1RB's predicted zero-noise
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
        diagnostics = {k: (predicted_zero_noise_variance(self.j, k),
                            float(self.per_irrep_variances[k, depth_index]))
                       for k in range(self.dim)}
        return diagnostics

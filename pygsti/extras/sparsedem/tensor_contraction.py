"""
Exact detector-outcome probabilities by tensor-network contraction.

A detector error model (DEM) is a list of independent error events ``i``, each
firing with probability ``p_i`` and flipping the detectors in a set ``T_i``
(and possibly some logical observables).  The probability of observing a
particular detector outcome ``s`` is

    P(s) = sum over all subsets E of events  prod_{i in E} p_i prod_{i not in E} (1 - p_i)
           * [ for every detector d: parity of |{i in E : d in T_i}| == s_d ]

which is a sum over 2^(number of events) terms.  This module evaluates it
exactly by writing it as a tensor network and contracting the network:

1. an event tensor per event ``i``: the probability vector ``[1 - p_i, p_i]``
   on a binary "main" index ``e{i}`` ...
2. ... fanned out by a copy (delta / all-equal) tensor to one auxiliary index
   ``e{i}_D{d}`` per detector ``d`` in ``T_i`` (by default the two are fused
   into a single weighted copy tensor on the auxiliary indices only);
3. a parity (XOR) tensor per detector ``d`` acting on all auxiliary indices of
   the events touching ``d``.  It equals 1 iff the parity of those indices is
   the requested bit ``s_d`` (or it carries the parity on an extra open output
   index ``D{d}_out`` when a marginal over ``d`` is requested).

Summing every index of the network enumerates the error subsets, the copy
tensors force the same event bit to be seen by every detector it touches,
and the parity tensors implement the indicator, so the full contraction
equals ``P(s)`` exactly.  Leaving the output index of some parity tensors
open returns the joint marginal over those detectors; dropping a detector
from the network altogether (equivalent to closing its parity tensor with an
all-ones tensor) marginalises it.

Complexity
----------
The number of detectors never enters as ``2^n``.  Contraction cost is
linear in the number of tensors and exponential in the *contraction width*
(a treewidth-like quantity) of the detector--event hypergraph of the DEM.
A repetition-code DEM is a ``(d-1) x rounds`` ladder, so its width scales
with ``min(d, rounds)`` -- roughly ``2 d`` with the path optimisers used here
-- and the cost is linear in the number of rounds: ``d = 7`` with 7 rounds
(48 detectors) contracts in milliseconds.  A surface-code DEM is a
``d x d x rounds`` lattice whose width grows like ``d^2``: ``d = 3`` is fine,
``d = 5`` needs much better path optimisation (kahypar-based hyper-optimisers
and slicing) than the greedy / ``'auto-hq'`` search used here.  Parity and
copy tensors are split into chains of at most ``max_rank`` indices so that a
detector touched by many events never materialises a ``2^k`` tensor.
Compare :func:`pygsti.extras.sparsedem.estimation.compute_outcome_distribution_from_dem`,
which always costs ``O(n 2^n)`` (and a dense ``2^n x 2^n`` Hadamard matrix) but
returns the entire distribution at once.

Bit conventions (read carefully)
--------------------------------
Every public function takes detector outcomes primarily as a sequence or
array ``bits`` in *detector-index order*: ``bits[d]`` is the outcome of
detector ``d`` (stim's sampling order).  The sparsedem package keys its
counts dictionaries with the *reversed* bitstring (see
:mod:`pygsti.extras.sparsedem.utils`); use :func:`bitstring_to_detector_bits`
/ :func:`detector_bits_to_bitstring` or the ``bitstring=`` keyword of
:func:`detector_outcome_probability`.  :func:`log_likelihood` consumes the
sparsedem counts dictionary directly.  Integer masks follow the package
convention ``bit d = detector d``, i.e. ``mask = sum(bits[d] << d)`` -- which
is also the index into the dense array returned by
``compute_outcome_distribution_from_dem``.

Backends
--------
``backend='numpy'`` depends on numpy only and contracts the network pairwise
with ``np.tensordot`` following a greedy path (smallest memory increase
first); it is not limited by the 52-letter ``np.einsum`` alphabet.
``backend='quimb'`` builds a :class:`quimb.tensor.TensorNetwork` and uses
cotengra's ``'auto-hq'`` path finder.  ``backend='auto'`` (default) prefers
quimb when it is importable.  quimb, cotengra and matplotlib are imported
lazily and only when needed.
"""

from __future__ import annotations

import heapq
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # stim is a hard dependency of the sparsedem package
    import stim
except ImportError:  # pragma: no cover - only for import-time robustness
    stim = None

__all__ = [
    "TensorSpec",
    "DEMTensorNetwork",
    "ContractionPlan",
    "dem_to_tensor_network",
    "detector_outcome_probability",
    "outcome_probabilities",
    "marginal_distribution",
    "log_likelihood",
    "contraction_tree",
    "plot_contraction_tree",
    "bitstring_to_detector_bits",
    "detector_bits_to_bitstring",
    "mask_to_detector_bits",
    "quimb_available",
    "BATCH_INDEX",
]

Label = Union[int, str]
BitSpec = Union[None, Sequence[int], np.ndarray, Mapping[Label, int]]

#: Name of the index shared by all bit-dependent tensors in a batched network
#: (see :meth:`DEMTensorNetwork.with_bit_batch`).  It behaves like a broadcast
#: ("batch") dimension: intermediates carry it, the result has it as first axis.
BATCH_INDEX = "__batch__"


# ---------------------------------------------------------------------------
# Bit-convention helpers
# ---------------------------------------------------------------------------

def bitstring_to_detector_bits(bitstring: str) -> np.ndarray:
    """Convert a sparsedem bitstring key to detector-order bits.

    sparsedem keys syndromes by the *reversed* stim sample, so ``bitstring[-1-d]``
    is the outcome of detector ``d``.

    Parameters:
        bitstring: str
            String of '0'/'1' characters in sparsedem (decreasing detector) order.

    Returns:
        bits: np.ndarray
            uint8 array with ``bits[d]`` = outcome of detector ``d``.
    """
    return np.array([int(b) for b in reversed(bitstring)], dtype=np.uint8)


def detector_bits_to_bitstring(bits: Sequence[int]) -> str:
    """Convert detector-order bits (``bits[d]`` = detector ``d``) to a sparsedem key."""
    return "".join(str(int(b)) for b in reversed(list(bits)))


def mask_to_detector_bits(mask: int, num_detectors: int) -> np.ndarray:
    """Convert an integer mask (bit ``d`` = detector ``d``) to detector-order bits."""
    return np.array([(mask >> d) & 1 for d in range(num_detectors)], dtype=np.uint8)


def _detector_label(label: Label) -> str:
    """Normalise a detector/observable reference to the canonical 'D{k}'/'L{k}' label."""
    if isinstance(label, (int, np.integer)):
        return f"D{int(label)}"
    if isinstance(label, str) and len(label) > 1 and label[0] in "DL" and label[1:].isdigit():
        return label
    raise ValueError(f"Cannot interpret {label!r} as a detector (int or 'Dk') or observable ('Lk').")


def _normalise_bits(spec: BitSpec, count: int, kind: str) -> Dict[str, int]:
    """Turn a bit specification into a dict of canonical label -> bit.

    ``spec`` may be None (nothing fixed), a full-length sequence/array of 0/1
    (everything fixed) or a mapping label -> bit (those labels fixed).
    """
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        out = {}
        for k, v in spec.items():
            lab = _detector_label(k) if kind == "D" else _detector_label(k if isinstance(k, str) else f"L{k}")
            if lab[0] != kind:
                raise ValueError(f"Expected a {kind!r} label, got {k!r}.")
            out[lab] = int(v) & 1
        return out
    arr = np.asarray(spec).ravel()
    if arr.size != count:
        raise ValueError(f"Expected {count} bits for {kind!r} but got {arr.size}.")
    return {f"{kind}{i}": int(arr[i]) & 1 for i in range(count)}


# ---------------------------------------------------------------------------
# Elementary tensors
# ---------------------------------------------------------------------------

_PARITY_CACHE: Dict[int, np.ndarray] = {}


def _parity_open(k: int) -> np.ndarray:
    """Parity tensor with ``k`` binary inputs and an output index appended.

    ``T[b_1, ..., b_k, o] = 1`` iff ``b_1 ^ ... ^ b_k == o``; for ``k = 0`` it is
    the vector ``[1, 0]`` (the parity of nothing is 0).
    """
    if k not in _PARITY_CACHE:
        if k == 0:
            data = np.array([1.0, 0.0])
        else:
            grid = np.indices((2,) * k).reshape(k, -1)
            parity = grid.sum(axis=0) % 2
            flat = np.zeros((2 ** k, 2))
            flat[np.arange(2 ** k), parity] = 1.0
            data = flat.reshape((2,) * k + (2,))
        _PARITY_CACHE[k] = data
    return _PARITY_CACHE[k]


def _copy_tensor(k: int, w0: float = 1.0, w1: float = 1.0) -> np.ndarray:
    """All-equal (delta) tensor on ``k`` binary indices with weights on the two diagonal entries."""
    data = np.zeros((2,) * k)
    data[(0,) * k] = w0
    data[(1,) * k] = w1
    return data


# ---------------------------------------------------------------------------
# Network description
# ---------------------------------------------------------------------------

@dataclass
class TensorSpec:
    """One tensor of the network: dense data with one named index per axis.

    Attributes:
        data: np.ndarray
            Tensor entries, shape ``(2,) * len(inds)``.
        inds: tuple[str, ...]
            Index names, one per axis.  Every index appears in at most two tensors.
        tag: str
            Human-readable tag (``'e3'`` event, ``'D5'`` detector parity, ...).
        bit_label: str | None
            For parity tensors closed on a fixed bit: the ``'D{d}'``/``'L{k}'`` label
            whose bit selects ``data = open_data[..., bit]``.
        open_data: np.ndarray | None
            The corresponding parity tensor with the output index still open.
    """

    data: np.ndarray
    inds: Tuple[str, ...]
    tag: str = ""
    bit_label: Optional[str] = None
    open_data: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def rank(self) -> int:
        return len(self.inds)


@dataclass
class DEMTensorNetwork:
    """Backend-independent tensor network for a DEM outcome probability / marginal.

    Attributes:
        tensors: list[TensorSpec]
        open_inds: tuple[str, ...]
            Output indices, in the order of the requested open labels; the contraction
            result has one axis per entry (index value = bit of that detector/observable).
        open_labels: tuple[str, ...]
            Canonical labels ``'D{d}'``/``'L{k}'`` matching ``open_inds``.
        fixed_bits: dict[str, int]
            Labels closed on a fixed bit (parity constraints).
        num_detectors, num_observables, num_events: int
    """

    tensors: List[TensorSpec]
    open_inds: Tuple[str, ...]
    open_labels: Tuple[str, ...]
    fixed_bits: Dict[str, int]
    num_detectors: int
    num_observables: int
    num_events: int

    # -- introspection ----------------------------------------------------
    @property
    def num_tensors(self) -> int:
        return len(self.tensors)

    @property
    def max_rank(self) -> int:
        return max((t.rank for t in self.tensors), default=0)

    @property
    def num_indices(self) -> int:
        return len({i for t in self.tensors for i in t.inds})

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"DEMTensorNetwork(num_tensors={self.num_tensors}, num_indices={self.num_indices}, "
                f"max_rank={self.max_rank}, open={self.open_labels}, fixed={len(self.fixed_bits)})")

    # -- re-targeting -------------------------------------------------------
    def with_bits(self, detector_bits: BitSpec = None, observable_bits: BitSpec = None) -> "DEMTensorNetwork":
        """Return a copy with new values for (a subset of) the already-fixed bits.

        The network structure (indices, shapes) is unchanged, so a
        :class:`ContractionPlan` made for ``self`` remains valid.
        """
        new = dict(_normalise_bits(detector_bits, self.num_detectors, "D"))
        new.update(_normalise_bits(observable_bits, self.num_observables, "L"))
        unknown = set(new) - set(self.fixed_bits)
        if unknown:
            raise ValueError(f"Labels {sorted(unknown)} are not fixed in this network; rebuild it instead.")
        tensors = []
        for t in self.tensors:
            if t.bit_label is not None and t.bit_label in new:
                tensors.append(replace(t, data=t.open_data[..., new[t.bit_label]]))
            else:
                tensors.append(t)
        fixed = dict(self.fixed_bits)
        fixed.update(new)
        return replace(self, tensors=tensors, fixed_bits=fixed)

    def with_bit_batch(self, detector_bits: np.ndarray, observable_bits: Optional[np.ndarray] = None
                       ) -> "DEMTensorNetwork":
        """Return a batched copy: one contraction evaluating ``B`` bit assignments at once.

        Every parity tensor closed on a fixed bit acquires an extra axis, named
        :data:`BATCH_INDEX`, indexed by the row of ``detector_bits`` / ``observable_bits``;
        the contraction result then has shape ``(B,) + (2,) * len(open_inds)``.

        Parameters:
            detector_bits: np.ndarray, shape (B, num_detectors)
                Rows are outcomes in detector-index order.  Only the columns of detectors
                fixed in this network are used.
            observable_bits: np.ndarray, shape (B, num_observables), optional
                Required iff observables are fixed in this network.
        """
        if self.is_batched:
            raise ValueError("Network is already batched.")
        detector_bits = np.asarray(detector_bits, dtype=np.intp)
        if detector_bits.ndim != 2 or detector_bits.shape[1] != self.num_detectors:
            raise ValueError(f"detector_bits must have shape (B, {self.num_detectors}).")
        if observable_bits is not None:
            observable_bits = np.asarray(observable_bits, dtype=np.intp)
            if observable_bits.shape != (detector_bits.shape[0], self.num_observables):
                raise ValueError(f"observable_bits must have shape (B, {self.num_observables}).")
        tensors = []
        for t in self.tensors:
            if t.bit_label is None:
                tensors.append(t)
                continue
            k = int(t.bit_label[1:])
            if t.bit_label[0] == "D":
                column = detector_bits[:, k]
            elif observable_bits is None:
                raise ValueError(f"{t.bit_label} is fixed in this network; observable_bits is required.")
            else:
                column = observable_bits[:, k]
            tensors.append(replace(t, data=t.open_data[..., column & 1], inds=t.inds + (BATCH_INDEX,)))
        return replace(self, tensors=tensors, open_inds=(BATCH_INDEX,) + self.open_inds,
                       open_labels=(BATCH_INDEX,) + self.open_labels)

    @property
    def is_batched(self) -> bool:
        return bool(self.open_inds) and self.open_inds[0] == BATCH_INDEX

    # -- contraction ----------------------------------------------------------
    def plan(self, backend: str = "auto", optimize=None) -> "ContractionPlan":
        """Find (and cache in a :class:`ContractionPlan`) a contraction path for this structure."""
        return ContractionPlan(self, backend=backend, optimize=optimize)

    def contract(self, backend: str = "auto", optimize=None) -> Union[float, np.ndarray]:
        """Contract the network.

        Returns:
            float if there are no open indices, else an ndarray of shape
            ``(2,) * len(open_inds)`` with axis ``j`` indexed by the bit of ``open_labels[j]``.
        """
        return self.plan(backend=backend, optimize=optimize).evaluate(self)

    def to_quimb(self):
        """Build a :class:`quimb.tensor.TensorNetwork` (requires quimb)."""
        qtn = _import_quimb()
        return qtn.TensorNetwork([qtn.Tensor(data=np.asarray(t.data, dtype=float), inds=t.inds, tags={t.tag})
                                  for t in self.tensors])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _dem_events(dem: "stim.DetectorErrorModel") -> List[Tuple[float, Tuple[str, ...]]]:
    """Flatten a DEM into (probability, labels) pairs.

    Labels are ``'D{d}'``/``'L{k}'``; a label repeated an even number of times in
    one instruction cancels (stim XOR semantics), an odd number keeps one copy.
    ``detector``/``logical_observable`` declarations carry no probability and
    are skipped (they only affect ``num_detectors``/``num_observables``).
    """
    events = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        counts: Dict[str, int] = defaultdict(int)
        order: List[str] = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                lab = f"D{t.val}"
            elif t.is_logical_observable_id():
                lab = f"L{t.val}"
            else:  # separators '^' carry no information for probabilities
                continue
            if lab not in counts:
                order.append(lab)
            counts[lab] += 1
        labels = tuple(lab for lab in order if counts[lab] % 2 == 1)
        events.append((p, labels))
    return events


def dem_to_tensor_network(
    dem: "stim.DetectorErrorModel",
    detector_bits: BitSpec = None,
    observable_bits: BitSpec = None,
    *,
    open_detectors: Iterable[Label] = (),
    open_observables: Iterable[Label] = (),
    fuse_events: bool = True,
    max_rank: int = 8,
) -> DEMTensorNetwork:
    """Build the tensor network whose contraction is a DEM outcome probability or marginal.

    Every detector (observable) is in exactly one of three states:

    * **fixed** to a bit -- listed in ``detector_bits`` (``observable_bits``):
      its parity tensor is closed on that bit (a constraint);
    * **open** -- listed in ``open_detectors`` (``open_observables``): its parity
      tensor keeps an output index, giving one axis of the result;
    * **marginalised** -- neither: it is removed from the network, which is
      exactly equivalent to closing its parity tensor with an all-ones tensor.

    Parameters:
        dem: stim.DetectorErrorModel
            Flattened internally, so ``repeat``/``shift_detectors`` blocks are fine.
        detector_bits: None | sequence of 0/1 | dict
            Full-length sequence in detector-index order (``bits[d]`` = detector ``d``),
            or a dict ``{detector: bit}`` fixing a subset, or None (fix nothing).
        observable_bits: None | sequence of 0/1 | dict
            Same for logical observables (``L`` targets).  None (default) marginalises
            them, i.e. ``L`` targets are simply ignored.
        open_detectors, open_observables: iterable of labels
            Detectors (ints or ``'Dk'``) / observables (ints or ``'Lk'``) to leave open,
            in the order the result axes should appear (detectors first, then observables).
        fuse_events: bool
            If True (default) the probability vector and the copy tensor of each event
            are fused into one weighted copy tensor.  If False they are kept as separate
            tensors ``p{i}`` (vector on index ``e{i}``) and ``e{i}`` (copy tensor), which
            mirrors the textbook construction.
        max_rank: int
            Parity and copy tensors with more than this many indices are split into
            chains of tensors of rank <= ``max_rank`` (carry indices ``..._c{j}``).  Must be >= 3.

    Returns:
        DEMTensorNetwork
    """
    if max_rank < 3:
        raise ValueError("max_rank must be at least 3.")
    n_det, n_obs = dem.num_detectors, dem.num_observables
    fixed = dict(_normalise_bits(detector_bits, n_det, "D"))
    fixed.update(_normalise_bits(observable_bits, n_obs, "L"))
    open_labels = tuple(_detector_label(d) for d in open_detectors) + tuple(
        _detector_label(o if isinstance(o, str) else f"L{o}") for o in open_observables)
    if len(set(open_labels)) != len(open_labels):
        raise ValueError("Open labels must be distinct.")
    clash = set(open_labels) & set(fixed)
    if clash:
        raise ValueError(f"Labels {sorted(clash)} are both fixed and open.")
    for lab in list(fixed) + list(open_labels):
        k = int(lab[1:])
        limit = n_det if lab[0] == "D" else n_obs
        if k >= limit:
            raise ValueError(f"Label {lab} out of range (DEM has {n_det} detectors, {n_obs} observables).")
    active = set(fixed) | set(open_labels)

    events = _dem_events(dem)
    tensors: List[TensorSpec] = []
    legs_by_label: Dict[str, List[str]] = defaultdict(list)  # label -> auxiliary indices

    for i, (p, labels) in enumerate(events):
        kept = [lab for lab in labels if lab in active]
        if not kept:  # marginalised out entirely: contributes (1-p) + p = 1
            continue
        aux = [f"e{i}_{lab}" for lab in kept]
        for lab, ind in zip(kept, aux):
            legs_by_label[lab].append(ind)
        if fuse_events:
            _append_copy_chain(tensors, aux, 1.0 - p, p, tag=f"e{i}", prefix=f"e{i}", max_rank=max_rank)
        else:
            main = f"e{i}"
            tensors.append(TensorSpec(np.array([1.0 - p, p]), (main,), tag=f"p{i}"))
            _append_copy_chain(tensors, [main] + aux, 1.0, 1.0, tag=f"e{i}", prefix=f"e{i}", max_rank=max_rank)

    open_inds = tuple(f"{lab}_out" for lab in open_labels)
    for lab in sorted(active, key=lambda s: (s[0], int(s[1:]))):
        inputs = legs_by_label.get(lab, [])
        if lab in fixed:
            _append_parity_chain(tensors, inputs, tag=lab, prefix=lab, max_rank=max_rank,
                                 fixed_bit=fixed[lab], out_ind=None)
        else:
            _append_parity_chain(tensors, inputs, tag=lab, prefix=lab, max_rank=max_rank,
                                 fixed_bit=None, out_ind=f"{lab}_out")

    if not tensors:  # everything marginalised: the network is the scalar 1
        tensors.append(TensorSpec(np.array(1.0), (), tag="one"))

    return DEMTensorNetwork(tensors=tensors, open_inds=open_inds, open_labels=open_labels,
                            fixed_bits=fixed, num_detectors=n_det, num_observables=n_obs,
                            num_events=len(events))


def _append_copy_chain(tensors: List[TensorSpec], legs: List[str], w0: float, w1: float,
                       *, tag: str, prefix: str, max_rank: int) -> None:
    """Append a (weighted) all-equal tensor on ``legs``, split into a chain if too large."""
    remaining = list(legs)
    carry = None
    j = 0
    weights = (w0, w1)
    while True:
        head = [carry] if carry else []
        cap = max_rank - len(head) - 1  # leave one slot for a carry-out
        if len(remaining) <= cap + 1:  # final piece, no carry-out needed
            inds = tuple(head + remaining)
            tensors.append(TensorSpec(_copy_tensor(len(inds), *weights), inds, tag=tag))
            return
        take, remaining = remaining[:cap], remaining[cap:]
        new_carry = f"{prefix}_c{j}"
        inds = tuple(head + take + [new_carry])
        tensors.append(TensorSpec(_copy_tensor(len(inds), *weights), inds, tag=tag))
        weights = (1.0, 1.0)  # weights applied exactly once
        carry, j = new_carry, j + 1


def _append_parity_chain(tensors: List[TensorSpec], inputs: List[str], *, tag: str, prefix: str,
                         max_rank: int, fixed_bit: Optional[int], out_ind: Optional[str]) -> None:
    """Append the parity constraint over ``inputs`` (fixed bit or open output), split if too large.

    With zero inputs this is the vector ``[1, 0]`` on the output (open) or the
    scalar ``1``/``0`` for fixed bit 0/1 -- so a detector no event touches needs
    no special casing.
    """
    remaining = list(inputs)
    carry = None
    j = 0
    while True:
        head = [carry] if carry else []
        cap = max_rank - len(head) - 1  # inputs per piece, one slot reserved for out/carry
        if len(remaining) <= cap:
            inds_in = tuple(head + remaining)
            open_data = _parity_open(len(inds_in))
            if fixed_bit is None:
                tensors.append(TensorSpec(open_data, inds_in + (out_ind,), tag=tag))
            else:
                tensors.append(TensorSpec(open_data[..., fixed_bit], inds_in, tag=tag,
                                          bit_label=tag, open_data=open_data))
            return
        take, remaining = remaining[:cap], remaining[cap:]
        new_carry = f"{prefix}_c{j}"
        inds_in = tuple(head + take)
        tensors.append(TensorSpec(_parity_open(len(inds_in)), inds_in + (new_carry,), tag=tag))
        carry, j = new_carry, j + 1


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def quimb_available() -> bool:
    """True if ``quimb.tensor`` can be imported."""
    try:
        _import_quimb()
        return True
    except ImportError:
        return False


def _import_quimb():
    """Import ``quimb.tensor`` lazily, keeping quimb's thread pools disabled.

    Importing quimb monkeypatches ``scipy.sparse`` CSR matrix-vector products with a
    numba-parallel kernel (used when ``nnz > 50000`` and more than one thread worker
    is configured).  That kernel is not needed for tensor contraction and has been
    observed to segfault (aarch64, numba 0.67) when other code in the same process --
    e.g. :mod:`pygsti.extras.sparsedem.logical_decoration` -- multiplies large sparse
    matrices.  Unless the user set ``QUIMB_NUM_THREAD_WORKERS`` explicitly, quimb is
    therefore run with a single thread worker.
    """
    import os
    os.environ.setdefault("QUIMB_NUM_THREAD_WORKERS", "1")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*kahypar.*")
            import quimb.tensor as qtn  # noqa: WPS433 (lazy import by design)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "backend='quimb' requires the optional packages quimb and cotengra "
            "(pip install quimb cotengra); use backend='numpy' otherwise."
        ) from exc
    if os.environ.get("QUIMB_NUM_THREAD_WORKERS") == "1":
        import quimb.core  # quimb may have been imported before the env var was set
        quimb.core._NUM_THREAD_WORKERS = 1
    return qtn


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "quimb" if quimb_available() else "numpy"
    if backend not in ("numpy", "quimb"):
        raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'numpy' or 'quimb'.")
    return backend


def _greedy_path(inds_list: Sequence[Tuple[str, ...]], size: int = 2) -> List[Tuple[int, int]]:
    """Greedy pairwise contraction order (SSA ids: the k-th contraction creates id ``N + k``).

    Candidate pairs are tensors sharing at least one index; the pair whose result
    removes the most memory (``size(result) - size(a) - size(b)`` smallest) is
    contracted first.  Disconnected pieces are finally multiplied together,
    smallest first.
    """
    alive: Dict[int, frozenset] = {i: frozenset(inds) for i, inds in enumerate(inds_list)}
    owners: Dict[str, set] = defaultdict(set)
    for i, inds in alive.items():
        for ind in inds:
            owners[ind].add(i)

    def cost(a: frozenset, b: frozenset) -> Tuple[int, int]:
        shared = a & b
        out = (a | b) - shared
        return (size ** len(out) - size ** len(a) - size ** len(b), size ** len(out))

    heap: List[Tuple[Tuple[int, int], int, int]] = []
    seen = set()
    for ind, ts in owners.items():
        if len(ts) == 2:
            i, j = sorted(ts)
            if (i, j) not in seen:
                seen.add((i, j))
                heapq.heappush(heap, (cost(alive[i], alive[j]), i, j))

    path: List[Tuple[int, int]] = []
    nxt = len(inds_list)
    while heap:
        _, i, j = heapq.heappop(heap)
        if i not in alive or j not in alive:
            continue
        a, b = alive.pop(i), alive.pop(j)
        shared = a & b
        new = (a | b) - shared
        neighbours = set()
        for ind in a | b:
            owners[ind].discard(i)
            owners[ind].discard(j)
            if ind in new:
                owners[ind].add(nxt)
                neighbours |= owners[ind] - {nxt}
        alive[nxt] = new
        for k in neighbours:
            heapq.heappush(heap, (cost(alive[k], new), min(k, nxt), max(k, nxt)))
        path.append((i, j))
        nxt += 1

    # outer products of disconnected remainders, smallest first
    rest = sorted(alive, key=lambda k: (len(alive[k]), k))
    while len(rest) > 1:
        i, j = rest[0], rest[1]
        alive[nxt] = alive[i] | alive[j]
        path.append((i, j))
        rest = sorted([k for k in rest[2:]] + [nxt], key=lambda k: (len(alive[k]), k))
        nxt += 1
    return path


def _linear_to_ssa(path: Sequence[Sequence[int]], n: int) -> List[Tuple[int, int]]:
    """Convert an opt_einsum-style path (positions in a shrinking list) to SSA pairs."""
    ids = list(range(n))
    out = []
    nxt = n
    for step in path:
        pos = sorted(step, reverse=True)
        chosen = [ids.pop(p) for p in pos]
        if len(chosen) != 2:  # pragma: no cover - cotengra paths are pairwise
            raise ValueError("Only pairwise contraction paths are supported.")
        out.append((chosen[1], chosen[0]))
        ids.append(nxt)
        nxt += 1
    return out


def _path_width(inds_list: Sequence[Tuple[str, ...]], path: Sequence[Tuple[int, int]]) -> int:
    """Largest number of indices of any intermediate tensor along an SSA path."""
    alive = {i: frozenset(inds) for i, inds in enumerate(inds_list)}
    width = max((len(v) for v in alive.values()), default=0)
    nxt = len(inds_list)
    for i, j in path:
        a, b = alive.pop(i), alive.pop(j)
        alive[nxt] = (a | b) - (a & b)
        width = max(width, len(alive[nxt]))
        nxt += 1
    return width


def _contract_pair(a_inds: Tuple[str, ...], a: np.ndarray, b_inds: Tuple[str, ...], b: np.ndarray):
    """Contract two tensors over all shared indices (each index occurs in at most two tensors).

    :data:`BATCH_INDEX` is the exception: it is a broadcast axis that survives the
    contraction, handled with a batched ``np.matmul``.
    """
    a_batched, b_batched = BATCH_INDEX in a_inds, BATCH_INDEX in b_inds
    if a_batched or b_batched:
        shared = [ind for ind in a_inds if ind in b_inds and ind != BATCH_INDEX]
        free_a = [ind for ind in a_inds if ind not in shared and ind != BATCH_INDEX]
        free_b = [ind for ind in b_inds if ind not in shared and ind != BATCH_INDEX]

        def as_matrix(x, inds, batched, rows, cols):
            perm = ([inds.index(BATCH_INDEX)] if batched else []) + [inds.index(i) for i in rows + cols]
            x = np.transpose(x, perm)
            return x.reshape((x.shape[0] if batched else 1, 2 ** len(rows), 2 ** len(cols)))

        product = np.matmul(as_matrix(a, a_inds, a_batched, free_a, shared),
                            as_matrix(b, b_inds, b_batched, shared, free_b))
        inds = (BATCH_INDEX,) + tuple(free_a) + tuple(free_b)
        return inds, product.reshape((product.shape[0],) + (2,) * (len(free_a) + len(free_b)))
    shared = [ind for ind in a_inds if ind in b_inds]
    if shared:
        axes_a = [a_inds.index(s) for s in shared]
        axes_b = [b_inds.index(s) for s in shared]
        data = np.tensordot(a, b, axes=(axes_a, axes_b))
    else:
        data = np.multiply.outer(a, b)
    inds = tuple(i for i in a_inds if i not in shared) + tuple(i for i in b_inds if i not in shared)
    return inds, data


def _numpy_execute(network: DEMTensorNetwork, path: Sequence[Tuple[int, int]]) -> Union[float, np.ndarray]:
    pool: Dict[int, Tuple[Tuple[str, ...], np.ndarray]] = {
        i: (t.inds, np.asarray(t.data, dtype=float)) for i, t in enumerate(network.tensors)}
    nxt = len(pool)
    for i, j in path:
        (ai, ad), (bi, bd) = pool.pop(i), pool.pop(j)
        pool[nxt] = _contract_pair(ai, ad, bi, bd)
        nxt += 1
    if not pool:
        result_inds, result = (), np.array(1.0)
    else:
        (result_inds, result), = pool.values()
    if set(result_inds) != set(network.open_inds):  # pragma: no cover - internal consistency
        raise RuntimeError(f"Contraction left indices {result_inds}, expected {network.open_inds}.")
    if result_inds:
        result = np.transpose(result, [result_inds.index(ind) for ind in network.open_inds])
        return np.asarray(result)
    return float(result)


class ContractionPlan:
    """A contraction path for one network structure, reusable across bit assignments.

    Build with :meth:`DEMTensorNetwork.plan` and call :meth:`evaluate` on the
    original network or any :meth:`DEMTensorNetwork.with_bits` variant of it.
    """

    def __init__(self, network: DEMTensorNetwork, backend: str = "auto", optimize=None):
        self.backend = _resolve_backend(backend)
        self.batched = network.is_batched
        self.open_inds = network.open_inds
        self._signature = tuple(t.inds for t in network.tensors)
        unbatched = tuple(tuple(i for i in inds if i != BATCH_INDEX) for inds in self._signature)
        if self.backend == "numpy":
            self.optimize = optimize if optimize is not None else "greedy"
            if self.optimize != "greedy":
                raise ValueError("The numpy backend only supports optimize='greedy'.")
            self.path = _greedy_path(unbatched)
            #: largest intermediate rank along the path (batch axis excluded)
            self.width = _path_width(unbatched, self.path)
        else:
            qtn = _import_quimb()
            self.optimize = optimize if optimize is not None else "auto-hq"
            self._qtn = qtn
            if len(network.tensors) <= 1:  # nothing to contract; cotengra rejects an empty path
                self.path = []
            else:
                # plan on the unbatched structure: the batch axis is a broadcast dimension
                # that should not influence the path
                plain = replace(network, open_inds=tuple(i for i in self.open_inds if i != BATCH_INDEX),
                                tensors=[replace(t, inds=inds, data=t.data[..., 0] if BATCH_INDEX in t.inds else t.data)
                                         for t, inds in zip(network.tensors, unbatched)])  # batch axis is last
                tn = plain.to_quimb()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*kahypar.*")
                    self.path = tn.contraction_path(optimize=self.optimize, output_inds=plain.open_inds)
            self._ssa_path = _linear_to_ssa(self.path, len(unbatched))
            self.width = _path_width(unbatched, self._ssa_path)

    def evaluate(self, network: DEMTensorNetwork) -> Union[float, np.ndarray]:
        """Contract ``network`` (same structure as the planning network) along the stored path."""
        if tuple(t.inds for t in network.tensors) != self._signature:
            raise ValueError("Network structure differs from the one this plan was made for "
                             "(build the plan from a network with the same fixed/open/batched labels).")
        if self.backend == "numpy" or len(network.tensors) <= 1:
            return _numpy_execute(network, self.path)
        if network.is_batched:
            # quimb/cotengra found the path; batched execution uses the numpy matmul
            # executor, which handles the broadcast batch axis far more efficiently
            # than an einsum over a size-B hyperindex.
            return _numpy_execute(network, self._ssa_path)
        tn = network.to_quimb()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*kahypar.*")
            res = tn.contract(all, optimize=self.path, output_inds=self.open_inds)
        if isinstance(res, self._qtn.Tensor):
            return np.asarray(res.transpose(*self.open_inds).data, dtype=float)
        if self.open_inds:  # pragma: no cover - quimb returned a scalar for an open network
            return np.asarray(res, dtype=float)
        return float(res)


# ---------------------------------------------------------------------------
# Public convenience API
# ---------------------------------------------------------------------------

def detector_outcome_probability(
    dem: "stim.DetectorErrorModel",
    detector_bits: Optional[Sequence[int]] = None,
    *,
    bitstring: Optional[str] = None,
    observable_bits: Optional[Sequence[int]] = None,
    backend: str = "auto",
    optimize=None,
    max_rank: int = 8,
) -> float:
    """Exact probability ``P(detectors = bits)`` under a DEM.

    Parameters:
        dem: stim.DetectorErrorModel
        detector_bits: sequence of 0/1, length ``dem.num_detectors``
            Outcome in detector-index order: ``detector_bits[d]`` is detector ``d``
            (stim sampling order).
        bitstring: str, optional
            Alternative to ``detector_bits``: a sparsedem-convention bitstring key
            (reversed order, as in ``utils.counts_from_samples``).
        observable_bits: sequence of 0/1, optional
            If given, additionally condition on the logical observables having these
            values (joint probability of detectors and observables).  Default: observables
            are marginalised.
        backend: 'auto' | 'numpy' | 'quimb'
        optimize:
            Path optimiser for the quimb backend (default ``'auto-hq'``); ignored by numpy.
        max_rank: int
            Maximum tensor rank before parity/copy tensors are chained.

    Returns:
        float
    """
    if (detector_bits is None) == (bitstring is None):
        raise ValueError("Give exactly one of detector_bits or bitstring.")
    if bitstring is not None:
        detector_bits = bitstring_to_detector_bits(bitstring)
    tn = dem_to_tensor_network(dem, detector_bits, observable_bits, max_rank=max_rank)
    return float(tn.contract(backend=backend, optimize=optimize))


def outcome_probabilities(
    dem: "stim.DetectorErrorModel",
    outcomes: Union[np.ndarray, Sequence[Sequence[int]]],
    *,
    observable_bits: Optional[Sequence[int]] = None,
    backend: str = "auto",
    optimize=None,
    max_rank: int = 8,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Exact probabilities of many detector outcomes, batched over one contraction path.

    The network structure is identical for every outcome (only the parity tensors'
    data change), so a single path is planned and the outcomes are evaluated in
    chunks with a shared batch axis (:data:`BATCH_INDEX`), i.e. each pairwise
    contraction is a batched matrix product.

    Parameters:
        dem: stim.DetectorErrorModel
        outcomes: array-like, shape (m, num_detectors)
            One row per outcome, in detector-index order (as returned by stim samplers).
        observable_bits, backend, optimize, max_rank:
            As in :func:`detector_outcome_probability`.
        batch_size: int, optional
            Outcomes per chunk.  Default: chosen from the contraction width so that the
            largest batched intermediate stays around 2^24 entries.

    Returns:
        np.ndarray, shape (m,)
    """
    outcomes = np.asarray(outcomes, dtype=np.uint8)
    if outcomes.ndim == 1:
        outcomes = outcomes[None, :]
    m = outcomes.shape[0]
    if m == 0:
        return np.zeros(0)
    base = dem_to_tensor_network(dem, outcomes[0], observable_bits, max_rank=max_rank)
    if m == 1:
        return np.array([base.contract(backend=backend, optimize=optimize)])
    if batch_size is None:
        # keep the largest batched intermediate around 2^24 doubles (128 MB)
        width = base.plan(backend="numpy").width
        batch_size = int(max(1, min(m, 1024, 2 ** max(0, 24 - width))))
    obs_row = None if observable_bits is None else np.asarray(observable_bits, dtype=np.intp).ravel()

    def batched(rows):
        obs = None if obs_row is None else np.broadcast_to(obs_row, (len(rows), obs_row.size))
        return base.with_bit_batch(rows, obs)

    plan = batched(outcomes[:batch_size]).plan(backend=backend, optimize=optimize)
    probs = np.empty(m)
    for start in range(0, m, batch_size):
        rows = outcomes[start:start + batch_size]
        probs[start:start + len(rows)] = plan.evaluate(batched(rows))
    return probs


def marginal_distribution(
    dem: "stim.DetectorErrorModel",
    detectors: Sequence[Label],
    *,
    condition: Optional[Mapping[Label, int]] = None,
    observable_bits: Optional[Sequence[int]] = None,
    backend: str = "auto",
    optimize=None,
    max_rank: int = 8,
) -> np.ndarray:
    """Joint marginal (optionally conditional) distribution of a subset of detectors.

    Detectors in ``detectors`` are left open, detectors in ``condition`` are fixed,
    all other detectors are marginalised (their parity tensor is dropped, which is
    equivalent to closing it with an all-ones tensor).  Labels may be ints or ``'Dk'``
    strings; ``'Lk'`` strings address logical observables.

    Parameters:
        dem: stim.DetectorErrorModel
        detectors: sequence of labels
            Detectors (or observables) to keep, in output-axis order.
        condition: dict label -> bit, optional
            Detectors/observables to fix.  The result is then the *joint* probability
            ``P(detectors, condition)``; divide by its sum for the conditional.
        observable_bits: sequence, optional
            Fix all observables at once (alternative to ``'Lk'`` keys in ``condition``).
        backend, optimize, max_rank:
            As in :func:`detector_outcome_probability`.

    Returns:
        np.ndarray, shape ``(2,) * len(detectors)``
            ``out[b_0, ..., b_{k-1}] = P(detectors[0] = b_0, ..., detectors[k-1] = b_{k-1} [, condition])``.
    """
    fixed_d: Dict[str, int] = {}
    fixed_l: Dict[str, int] = {}
    for k, v in (condition or {}).items():
        lab = _detector_label(k)
        (fixed_d if lab[0] == "D" else fixed_l)[lab] = int(v)
    if observable_bits is not None:
        fixed_l.update(_normalise_bits(observable_bits, dem.num_observables, "L"))
    open_d = [d for d in detectors if _detector_label(d)[0] == "D"]
    open_l = [d for d in detectors if _detector_label(d)[0] == "L"]
    tn = dem_to_tensor_network(dem, fixed_d, fixed_l, open_detectors=open_d, open_observables=open_l,
                               max_rank=max_rank)
    result = tn.contract(backend=backend, optimize=optimize)
    if len(detectors) == 0:
        return np.asarray(result)
    # restore the caller's interleaving of detectors and observables
    labels = [_detector_label(d) for d in detectors]
    order = [tn.open_labels.index(lab) for lab in labels]
    return np.transpose(np.asarray(result), order)


def log_likelihood(
    dem: "stim.DetectorErrorModel",
    syndrome_counts: Mapping[Union[str, int], int],
    *,
    observable_bits: Optional[Sequence[int]] = None,
    backend: str = "auto",
    optimize=None,
    max_rank: int = 8,
    prob_floor: float = 1e-300,
    return_per_outcome: bool = False,
    batch_size: Optional[int] = None,
):
    """Log-likelihood ``sum_s counts[s] log P(s)`` of syndrome counts under a DEM.

    Parameters:
        dem: stim.DetectorErrorModel
        syndrome_counts: dict
            sparsedem counts dictionary: keys are bitstrings in the package's *reversed*
            order (``utils.counts_from_samples``) or integer masks (bit ``d`` = detector ``d``);
            values are counts.
        observable_bits, backend, optimize, max_rank:
            As in :func:`detector_outcome_probability`.
        prob_floor: float
            Probabilities below this (including exact zeros / negative round-off) are
            clipped before taking the log, so the result is always finite.
        return_per_outcome: bool
            If True also return the array of per-outcome probabilities (in the
            iteration order of ``syndrome_counts``).
        batch_size: int, optional
            Forwarded to :func:`outcome_probabilities`.

    Returns:
        float, or (float, np.ndarray) if ``return_per_outcome``.
    """
    if len(syndrome_counts) == 0:
        return (0.0, np.zeros(0)) if return_per_outcome else 0.0
    n = dem.num_detectors
    rows = []
    for key in syndrome_counts:
        if isinstance(key, str):
            rows.append(bitstring_to_detector_bits(key))
        else:
            rows.append(mask_to_detector_bits(int(key), n))
    outcomes = np.array(rows, dtype=np.uint8)
    counts = np.fromiter(syndrome_counts.values(), dtype=float, count=len(syndrome_counts))
    probs = outcome_probabilities(dem, outcomes, observable_bits=observable_bits, backend=backend,
                                  optimize=optimize, max_rank=max_rank, batch_size=batch_size)
    ll = float(np.dot(counts, np.log(np.clip(probs, prob_floor, None))))
    return (ll, probs) if return_per_outcome else ll


# ---------------------------------------------------------------------------
# Contraction-tree inspection (quimb / cotengra)
# ---------------------------------------------------------------------------

def contraction_tree(
    dem_or_network: Union["stim.DetectorErrorModel", DEMTensorNetwork],
    detector_bits: BitSpec = None,
    *,
    optimize="greedy",
    max_rank: int = 8,
    **network_kwargs,
):
    """Return the cotengra :class:`ContractionTree` for a DEM network (requires quimb).

    Parameters:
        dem_or_network: stim.DetectorErrorModel | DEMTensorNetwork
            A DEM (a network is built with ``detector_bits`` and ``network_kwargs``,
            all-zeros outcome if ``detector_bits`` is None) or a prebuilt network.
        optimize:
            cotengra/opt_einsum optimiser preset, default ``'greedy'``.

    Returns:
        cotengra.ContractionTree -- use ``.contraction_width()``, ``.contraction_cost()``,
        ``.plot_rubberband()`` etc.
    """
    _import_quimb()
    if isinstance(dem_or_network, DEMTensorNetwork):
        network = dem_or_network
    else:
        if detector_bits is None:
            detector_bits = np.zeros(dem_or_network.num_detectors, dtype=np.uint8)
        network = dem_to_tensor_network(dem_or_network, detector_bits, max_rank=max_rank, **network_kwargs)
    tn = network.to_quimb()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*kahypar.*")
        return tn.contraction_tree(optimize=optimize, output_inds=network.open_inds)


def plot_contraction_tree(tree_or_dem, *, kind: str = "rubberband", **plot_kwargs):
    """Plot a contraction tree with cotengra (requires quimb + matplotlib).

    Parameters:
        tree_or_dem:
            A cotengra ``ContractionTree`` (from :func:`contraction_tree`), or a DEM /
            :class:`DEMTensorNetwork` from which one is built with ``optimize='greedy'``.
        kind: str
            One of cotengra's plotters: ``'rubberband'`` (default), ``'tent'``, ``'ring'``,
            ``'flat'``, ``'span'``, ``'circuit'``, ``'contractions'``.
        plot_kwargs:
            Forwarded to the cotengra plotting function.

    Returns:
        (fig, ax) as returned by cotengra.
    """
    if not hasattr(tree_or_dem, "plot_rubberband"):
        tree_or_dem = contraction_tree(tree_or_dem)
    tree = tree_or_dem
    plotter = getattr(tree, f"plot_{kind}", None)
    if plotter is None:
        raise ValueError(f"Unknown plot kind {kind!r}.")
    # cotengra <= 0.8 resolves string colormaps through matplotlib.cm.get_cmap, which
    # matplotlib 3.9 removed; passing a Colormap object sidesteps that lookup.
    import inspect
    import matplotlib
    param = inspect.signature(plotter).parameters.get("colormap")
    if param is not None and "colormap" not in plot_kwargs and isinstance(param.default, str):
        plot_kwargs["colormap"] = matplotlib.colormaps[param.default]
    elif isinstance(plot_kwargs.get("colormap"), str):
        plot_kwargs["colormap"] = matplotlib.colormaps[plot_kwargs["colormap"]]
    try:
        return plotter(**plot_kwargs)
    except TypeError:
        if kind != "rubberband":
            raise
        # cotengra 0.8.x's plot_rubberband still iterates tree nodes as sets although
        # nodes became integer ids; redo the (short) routine with the current node API.
        import matplotlib.pyplot as plt
        plt.close()
        return _plot_rubberband_fallback(tree, **plot_kwargs)


def _plot_rubberband_fallback(tree, colormap=None, **hypergraph_kwargs):
    """Rubber-band plot of a cotengra tree: nested loops around progressively contracted leaves."""
    import matplotlib
    from cotengra.plot import plot_hypergraph
    from cotengra.schematic import Drawing

    cmap = matplotlib.colormaps["Spectral"] if colormap is None else colormap
    info = {"pos": None}
    fig, ax = plot_hypergraph(tree.get_hypergraph(), info=info, show_and_close=False, **hypergraph_kwargs)
    pos, r0 = info["pos"], info["node_size"]
    drawing = Drawing(ax=ax)
    counts: Dict[int, int] = defaultdict(int)
    steps = list(tree.traverse())
    for i, (parent, _, _) in enumerate(steps):
        leaves = sorted(leaf for leaf in tree.get_subgraph(parent) if leaf in pos)  # scalars have no position
        if not leaves:
            continue
        for leaf in leaves:
            counts[leaf] += 1
        prog = i / max(len(steps) - 1, 1)
        drawing.patch_around([pos[leaf] for leaf in leaves], resolution=20,
                             radius=[r0 + 0.01 * counts[leaf] for leaf in leaves],
                             edgecolor=cmap(prog), facecolor="none", linestyle="-", zorder=-prog)
    return fig, ax

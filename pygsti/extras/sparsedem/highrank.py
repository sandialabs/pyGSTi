"""Data model and text format for high-rank detector error models.

A high-rank DEM extends stim's detector error model format with one new
directive::

    exclusive(k)
        error(p_1) <targets...>
        ...
        error(p_k) <targets...>

The ``exclusive(k)`` directive declares that the next ``k`` ``error`` lines
form a single stochastic event of rank ``k + 1``: in any given shot at most
one of the ``k`` listed errors occurs (each with its stated probability),
and with probability ``1 - sum(p_i)`` none of them occurs.  The listed
probabilities must therefore sum to at most 1.

Everything outside an ``exclusive`` block follows the ordinary stim DEM
semantics: each ``error(p) ...`` line is an independent rank-2 event, and
``detector`` / ``logical_observable`` lines declare coordinates / index
bounds.  ``repeat`` blocks and ``shift_detectors`` are not supported; models
using them should be flattened first.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Iterator, Sequence, Union

import stim

__all__ = [
    "ErrorEvent",
    "ExclusiveBlock",
    "HighRankDetectorErrorModel",
]

_PROB_TOL = 1e-9


@dataclasses.dataclass(frozen=True)
class ErrorEvent:
    """A single stochastic error mechanism.

    If the event occurs it flips the given detector bits and logical
    observable bits.  On its own it is a rank-2 stochastic channel; inside an
    :class:`ExclusiveBlock` it is one branch of a higher-rank channel.
    """

    probability: float
    detectors: tuple[int, ...] = ()
    observables: tuple[int, ...] = ()

    def __post_init__(self):
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability {self.probability} not in [0, 1]")
        if len(set(self.detectors)) != len(self.detectors):
            raise ValueError(f"repeated detector target in {self.detectors}")
        if len(set(self.observables)) != len(self.observables):
            raise ValueError(f"repeated observable target in {self.observables}")
        object.__setattr__(self, "detectors", tuple(sorted(self.detectors)))
        object.__setattr__(self, "observables", tuple(sorted(self.observables)))

    def targets_str(self) -> str:
        parts = [f"D{d}" for d in self.detectors] + [f"L{o}" for o in self.observables]
        return " ".join(parts)

    def to_dem_line(self) -> str:
        targets = self.targets_str()
        return f"error({self.probability!r}) {targets}".rstrip()


@dataclasses.dataclass(frozen=True)
class ExclusiveBlock:
    """A rank-(k+1) stochastic event: at most one of ``events`` occurs.

    The probabilities of the branches must sum to at most 1; the remaining
    probability mass is the trivial "nothing happened" outcome.
    """

    events: tuple[ErrorEvent, ...]

    def __post_init__(self):
        if len(self.events) < 1:
            raise ValueError("exclusive block must contain at least one event")
        total = sum(e.probability for e in self.events)
        if total > 1.0 + _PROB_TOL:
            raise ValueError(
                f"probabilities in exclusive block sum to {total} > 1"
            )

    @property
    def total_probability(self) -> float:
        return sum(e.probability for e in self.events)

    @property
    def rank(self) -> int:
        return len(self.events) + 1


Instruction = Union[ErrorEvent, ExclusiveBlock]

_ERROR_RE = re.compile(r"^error\(([^()]+)\)((?:\s+\S+)*)\s*$")
_EXCL_RE = re.compile(r"^exclusive\((\d+)\)\s*$")
_DET_RE = re.compile(r"^detector(?:\(([^()]*)\))?((?:\s+\S+)*)\s*$")
_OBS_RE = re.compile(r"^logical_observable((?:\s+\S+)*)\s*$")


def _parse_targets(text: str, line_no: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    detectors: list[int] = []
    observables: list[int] = []
    for tok in text.split():
        if tok == "^":
            # Decomposition separators are irrelevant for sampling.
            continue
        if tok.startswith("D") and tok[1:].isdigit():
            detectors.append(int(tok[1:]))
        elif tok.startswith("L") and tok[1:].isdigit():
            observables.append(int(tok[1:]))
        else:
            raise ValueError(f"line {line_no}: unrecognized target {tok!r}")
    return tuple(detectors), tuple(observables)


class HighRankDetectorErrorModel:
    """A detector error model whose events may have rank > 2.

    The model is an ordered list of instructions, each either an independent
    :class:`ErrorEvent` or an :class:`ExclusiveBlock`, plus optional detector
    coordinate annotations.
    """

    def __init__(
        self,
        instructions: Sequence[Instruction] = (),
        *,
        num_detectors: int | None = None,
        num_observables: int | None = None,
        detector_coords: dict[int, tuple[float, ...]] | None = None,
    ):
        self.instructions: list[Instruction] = list(instructions)
        self.detector_coords: dict[int, tuple[float, ...]] = dict(detector_coords or {})

        max_d = -1
        max_o = -1
        for ev in self.iter_events():
            if ev.detectors:
                max_d = max(max_d, ev.detectors[-1])
            if ev.observables:
                max_o = max(max_o, ev.observables[-1])
        if self.detector_coords:
            max_d = max(max_d, max(self.detector_coords))
        self.num_detectors = max(max_d + 1, num_detectors or 0)
        self.num_observables = max(max_o + 1, num_observables or 0)

    # ------------------------------------------------------------------ views

    def iter_events(self) -> Iterator[ErrorEvent]:
        """Yields every ErrorEvent, flattening exclusive blocks."""
        for inst in self.instructions:
            if isinstance(inst, ExclusiveBlock):
                yield from inst.events
            else:
                yield inst

    @property
    def independent_errors(self) -> list[ErrorEvent]:
        return [i for i in self.instructions if isinstance(i, ErrorEvent)]

    @property
    def exclusive_blocks(self) -> list[ExclusiveBlock]:
        return [i for i in self.instructions if isinstance(i, ExclusiveBlock)]

    # ---------------------------------------------------------------- parsing

    @classmethod
    def from_text(cls, text: str) -> "HighRankDetectorErrorModel":
        instructions: list[Instruction] = []
        detector_coords: dict[int, tuple[float, ...]] = {}
        num_detectors = 0
        num_observables = 0

        pending_block: list[ErrorEvent] | None = None
        pending_remaining = 0
        pending_start_line = 0

        for line_no, raw in enumerate(text.splitlines(), start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue

            m = _ERROR_RE.match(line)
            if m is not None:
                try:
                    p = float(m.group(1))
                except ValueError:
                    raise ValueError(f"line {line_no}: bad probability {m.group(1)!r}")
                dets, obs = _parse_targets(m.group(2), line_no)
                ev = ErrorEvent(p, dets, obs)
                if pending_block is not None:
                    pending_block.append(ev)
                    pending_remaining -= 1
                    if pending_remaining == 0:
                        instructions.append(ExclusiveBlock(tuple(pending_block)))
                        pending_block = None
                else:
                    instructions.append(ev)
                continue

            if pending_block is not None:
                raise ValueError(
                    f"line {line_no}: exclusive block starting at line "
                    f"{pending_start_line} expects {pending_remaining} more "
                    f"error line(s), got {line!r}"
                )

            m = _EXCL_RE.match(line)
            if m is not None:
                k = int(m.group(1))
                if k < 1:
                    raise ValueError(f"line {line_no}: exclusive({k}) needs k >= 1")
                pending_block = []
                pending_remaining = k
                pending_start_line = line_no
                continue

            m = _DET_RE.match(line)
            if m is not None:
                coords = tuple(
                    float(x) for x in (m.group(1) or "").split(",") if x.strip()
                )
                dets, obs = _parse_targets(m.group(2), line_no)
                if obs:
                    raise ValueError(f"line {line_no}: detector line with L target")
                for d in dets:
                    if coords:
                        detector_coords[d] = coords
                    num_detectors = max(num_detectors, d + 1)
                continue

            m = _OBS_RE.match(line)
            if m is not None:
                dets, obs = _parse_targets(m.group(1), line_no)
                if dets:
                    raise ValueError(
                        f"line {line_no}: logical_observable line with D target"
                    )
                for o in obs:
                    num_observables = max(num_observables, o + 1)
                continue

            raise ValueError(f"line {line_no}: unrecognized instruction {line!r}")

        if pending_block is not None:
            raise ValueError(
                f"exclusive block starting at line {pending_start_line} still "
                f"expects {pending_remaining} more error line(s) at end of input"
            )

        return cls(
            instructions,
            num_detectors=num_detectors,
            num_observables=num_observables,
            detector_coords=detector_coords,
        )

    @classmethod
    def from_file(cls, path) -> "HighRankDetectorErrorModel":
        with open(path) as f:
            return cls.from_text(f.read())

    @classmethod
    def from_stim_dem(
        cls, dem: stim.DetectorErrorModel
    ) -> "HighRankDetectorErrorModel":
        """Imports an ordinary stim DEM (all events independent / rank-2)."""
        flat = dem.flattened()
        instructions: list[Instruction] = []
        detector_coords = {
            d: tuple(c) for d, c in flat.get_detector_coordinates().items() if c
        }
        for inst in flat:
            if inst.type == "error":
                dets: list[int] = []
                obs: list[int] = []
                for t in inst.targets_copy():
                    if t.is_relative_detector_id():
                        dets.append(t.val)
                    elif t.is_logical_observable_id():
                        obs.append(t.val)
                instructions.append(
                    ErrorEvent(inst.args_copy()[0], tuple(dets), tuple(obs))
                )
        return cls(
            instructions,
            num_detectors=flat.num_detectors,
            num_observables=flat.num_observables,
            detector_coords=detector_coords,
        )

    # ------------------------------------------------------------- rendering

    def to_text(self) -> str:
        lines: list[str] = []
        for d in sorted(self.detector_coords):
            coords = ", ".join(repr(c) for c in self.detector_coords[d])
            lines.append(f"detector({coords}) D{d}")
        for inst in self.instructions:
            if isinstance(inst, ExclusiveBlock):
                lines.append(f"exclusive({len(inst.events)})")
                for ev in inst.events:
                    lines.append(f"    {ev.to_dem_line()}")
            else:
                lines.append(inst.to_dem_line())
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        n_blk = len(self.exclusive_blocks)
        n_ind = len(self.independent_errors)
        return (
            f"<HighRankDetectorErrorModel: {n_ind} independent errors, "
            f"{n_blk} exclusive blocks, {self.num_detectors} detectors, "
            f"{self.num_observables} observables>"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, HighRankDetectorErrorModel):
            return NotImplemented
        return (
            self.instructions == other.instructions
            and self.num_detectors == other.num_detectors
            and self.num_observables == other.num_observables
            and self.detector_coords == other.detector_coords
        )

    # ----------------------------------------------------------- conversions

    def approximate_stim_dem(self) -> stim.DetectorErrorModel:
        """Returns an ordinary stim DEM that treats every branch of every
        exclusive block as an independent error.

        This discards the exclusivity constraint (useful for handing the
        model to decoders such as pymatching, which only understand rank-2
        events).  It is a good approximation when branch probabilities are
        small, since then P(two branches of one block both firing) ~ p^2.
        """
        lines = []
        for d in sorted(self.detector_coords):
            coords = ", ".join(repr(c) for c in self.detector_coords[d])
            lines.append(f"detector({coords}) D{d}")
        for ev in self.iter_events():
            lines.append(ev.to_dem_line())
        if self.num_detectors:
            lines.append(f"detector D{self.num_detectors - 1}")
        if self.num_observables:
            lines.append(f"logical_observable L{self.num_observables - 1}")
        return stim.DetectorErrorModel("\n".join(lines))

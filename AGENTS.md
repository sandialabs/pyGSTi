# AGENTS.md — pyGSTi orientation for coding agents

> **NOTE FOR HUMANS:** these docs use [Mermaid](https://mermaid.js.org/) diagrams. To view them rendered, install a Mermaid-aware Markdown viewer — e.g., the **"Markdown Preview Mermaid Support"** extension in VS Code, or read the files on GitHub (which renders Mermaid in `.md` files natively). Without one, the diagram code blocks will display as plain text.

This file and the [agent-docs/](agent-docs/) folder beside it are hierarchical orientation material for agents (and new group developers) working on **pyGSTi** itself. It covers architecture, key abstractions, and non-obvious gotchas. It does *not* teach pyGSTi as a library — for that, read the jupyter-book at [docs/markdown/](docs/markdown/) in the source tree.

Read this file first, then jump into the subsystem doc(s) relevant to your task.

> **A note to agents on doc accuracy.** These docs are intentionally maintained at a "not overtly wrong" bar, not at a "perfectly accurate" one — they're a hint system to reduce trial-and-error, not a contract. **If, while doing real work, you find that what these docs say contradicts what the code actually does, treat the code as authoritative and flag the discrepancy to the user.** Be specific: name the file, the section, the claim, and the contradicting code path. Don't silently work around it, and don't edit the docs yourself unless the user asks you to — if they do, read [agent-docs/META.md](agent-docs/META.md) first.

## What pyGSTi is

pyGSTi is a Python framework for **modeling and analyzing collections of qudits** (commonly qubits, but the package supports higher-dimensional qudit spaces). Its most prominent capability is **gate set tomography (GST)** — fitting a [`Model`](pygsti/models/model.py) of a noisy quantum device from circuit-outcome count data. Beyond that, it covers randomized benchmarking (RB), robust phase estimation (RPE), drift characterization, and standalone circuit simulation. Some protocols (notably the RB family) intentionally do *not* produce a full Model and instead estimate scalar performance metrics on structured random circuits. 

The "data" pyGSTi consumes can come from a real experiment or be sampled from a simulation — there is nothing experiment-specific about the data path. pyGSTi's **report generator**, which renders fit results and diagnostics into interactive HTML/PDF/notebook reports, is itself a major user-facing capability.

## The mental model

The central triad is **`Model` ↔ `Circuit` ↔ `DataSet`**:

- A [`Model`](pygsti/models/model.py) describes a (possibly noisy) quantum device — a parameter vector with structure, typically equipped with a forward simulator.
- A [`Circuit`](pygsti/circuits/circuit.py) is a program that can run on such a device — a sequence of `Label` objects.
- A [`DataSet`](pygsti/data/dataset.py) records circuit outcomes (real or simulated).

A [`Protocol`](pygsti/protocols/protocol.py) is the user-facing object for estimating a model from data: it consumes a `ProtocolData` (which pairs an `ExperimentDesign` with a `DataSet`) and produces a `ProtocolResults`. For GST, the result contains updated Models; for RB, a scalar error rate; for any protocol with reportable outputs, a `Report` can be generated downstream.

If you internalize that triad and the Protocol-on-top picture, the rest of the subsystem docs are about *how* each piece is implemented.

## Layout of `agent-docs/`

| Doc | Subpackages | Read when you need to… |
|---|---|---|
| [01-representation.md](agent-docs/01-representation.md) | `baseobjs`, `circuits`, `models`, `modelmembers`, `evotypes` | add/change a gate parameterization, state, POVM, instrument, or a parameterization mode; understand Model internals |
| [02-forward-simulation.md](agent-docs/02-forward-simulation.md) | `forwardsims`, `layouts` | change how circuit outcomes get computed; tune MPI configuration |
| [03-data-and-fitting.md](agent-docs/03-data-and-fitting.md) | `data`, `objectivefns`, `optimize`, `algorithms` | change how data is stored, or how a fit's objective / optimizer / loop behaves |
| [04-orchestration.md](agent-docs/04-orchestration.md) | `protocols`, `drivers` | add a new high-level workflow; understand the Protocol class API users actually call |
| [05-reporting-and-persistence.md](agent-docs/05-reporting-and-persistence.md) | `report`, `io`, `serialization` | touch reports, HTML/PDF/notebook output, or persistence formats |
| [06-modelpacks-and-processors.md](agent-docs/06-modelpacks-and-processors.md) | `modelpacks`, `processors` | add a named (and annotated) gate set, or a device description |
| [07-tools-library.md](agent-docs/07-tools-library.md) | `tools` | reach for a basis transform, channel conversion, MPI helper, or other utility |
| [08-domain-plugins.md](agent-docs/08-domain-plugins.md) | `extras`, `errorgenpropagation` | work on RB / RPE / drift / error-generator propagation |
| [known-debt.md](agent-docs/known-debt.md) | — | the area you're touching has a known smell or in-flight redesign |
| [META.md](agent-docs/META.md) | — | you've been asked to **update these docs** — the correctness bar, the invariants that break silently, and the link checker |

Files are numbered for stable referencing only — not as a recommended reading order.

## Cross-cutting concerns

A handful of topics span essentially every subsystem. Read these once; the subsystem docs cross-reference them rather than restating.

### Gauge freedom and gauge optimization

This is the single most important cross-cutting concept in pyGSTi. A `Model` is defined only up to a **gauge transformation** — a similarity transform that preserves all measurable circuit-outcome probabilities. Two `Model` objects with very different `to_vector()` outputs can describe identical physical devices; the parameter vector is not a unique fingerprint.

Before reporting, comparing, or extracting metrics from a `Model`, you almost always need to **gauge-optimize** it — pick the representative closest to a target Model. The machinery lives in [pygsti/algorithms/gaugeopt.py](pygsti/algorithms/gaugeopt.py), and gauge optimization is automatically wired into `Protocol` classes via `gaugeopt_suite` / `gaugeopt_params`. Reports run a gauge-optimization pass per estimate before rendering metrics.

If you find yourself writing code that compares two Models, computes a fidelity, or surfaces an error rate, **check whether gauge optimization has been applied first**. Forgetting this produces meaningless numbers.

The representation of a gauge-optimization suite is itself complicated — sometimes a `list[list[dict]]`, sometimes a [`GSTGaugeOptSuite`](pygsti/protocols/gst.py#L857) object. See doc 04 for the patterns, and [known-debt.md](agent-docs/known-debt.md#12-gaugeopt_suite-representation-duality) for the rough edge.

### Parameterization modes

When constructing a Model, you pick a **parameterization mode**: `"full"`, `"Full TP"`, `"CPTP"`, `"H+S"`, `"S"`, `"Target"`, or — most importantly for noise modeling — `"CPTPLND"`. This decides the map from model parameters to the entries of the model's constituent superoperator matrices (or "superbra" / "superket" vectors).

**CPTPLND** is worth singling out because it's what we use for CPTP noise modeling in practice. It wraps a `LindbladErrorgen` in an `ExpErrorgenOp` for each gate, and represents noisy SPAM as "perfect SPAM composed with a noisy gate." This pattern is also used for reduced-order modeling (e.g., `"H+S"`). It introduces some **representation degeneracies in state prep** (and to a lesser extent measurement): multiple parameter assignments map to the same physical noisy SPAM.

Gauge optimization interacts nontrivially with model parameterizations. There are many codepaths in pyGSTi that coerce a model to a `"Full TP"`before the gauge optimization step. This can create problems in evaluating goodness of fit, since it amounts to changing the hypothesis class for a statistical estimation problem _after_ doing the model fit.

### MPI / parallelization

Many code paths fork on `comm is None` vs. an `mpi4py.MPI.Comm`. The object that gets sharded across MPI ranks is the [`Layout`](agent-docs/02-forward-simulation.md), which both flattens `(circuit, outcome)` pairs into a 1-D array *and* plans memory for forward simulation. Helpers live in [pygsti/tools/mpitools.py](pygsti/tools/mpitools.py). MPI is opt-in but the plumbing is everywhere — most fit and forward-sim entry points accept a `comm` kwarg.

### Optional dependencies

pyGSTi guards several heavy or platform-specific dependencies with try-imports and degrades gracefully:

- `cvxpy` — SDP routines in [pygsti/tools/sdptools.py](pygsti/tools/sdptools.py), parts of [optools.py](pygsti/tools/optools.py) and [jamiolkowski.py](pygsti/tools/jamiolkowski.py).
- `stim` — [pygsti/errorgenpropagation/](pygsti/errorgenpropagation/) and some of [pygsti/tools/errgenproptools.py](pygsti/tools/errgenproptools.py).
- `pymongo` — [pygsti/io/mongodb.py](pygsti/io/mongodb.py).
- `pdflatex` (external binary) — required for PDF report generation.
- `plotly` and `matplotlib` — required for HTML and PDF reports respectively.

When editing code in these areas, check the import guard rather than assuming the function exists. Functions inside guard blocks raise informative `ImportError`s (or return `None`) when the optional dep is missing.

## What's not in here

- **User-facing usage docs** for pyGSTi as a library: see the jupyter-book at [docs/markdown/](docs/markdown/). Notebooks are organized in reader tiers -- `docs/markdown/start/`, `docs/markdown/guides/`, and `docs/markdown/advanced/` -- and are the canonical tutorials.
- **In-flight project state and active threads**: see [sandialabs/pyGSTi](https://github.com/sandialabs/pyGSTi/issues) on GitHub.
- **Code conventions, build/test environment, lint rules**: see the repository's `README.md`, `CONTRIBUTING.md`, and `pyproject.toml`.

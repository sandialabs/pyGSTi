---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Live-Monitoring a Running GST Fit

Long-sequence GST proceeds through a sequence of iterations, one per circuit-list "depth" (`L`), fitting a model at each depth before moving on to the next, longer circuit list. Only *after* the last of these iterations does pyGSTi perform gauge optimization, and gauge optimization is what makes familiar quantities like process fidelity to a target, SPAM tables, or diamond norm meaningful in the first place: a gauge transformation is a similarity transform on the model's gates, and reportable quantities that depend on a *choice* of gauge are essentially arbitrary until that final gauge-optimization step has actually happened.

That creates a real tension for anyone who wants visibility into a long-running fit *while it's running*: most of the quantities you'd normally want to look at aren't meaningful yet.

Two families of quantities *are* safe to look at before gauge optimization, however:

1. **Goodness-of-fit statistics** (the objective function value at each iteration's optimum, degrees of freedom, and the corresponding p-value). These describe how well the *model family* explains the *data* - a property of predicted probabilities, which don't depend on gauge at all.
2. **Gate eigenvalues.** A gauge transformation acts on a gate's dense superoperator matrix $G$ as a similarity transform $G \to M G M^{-1}$, which by definition leaves eigenvalues unchanged.

`pygsti.report.livemetrics` and `pygsti.report.livemonitor` build on this: they let you watch a `GateSetTomography` or `StandardGST` fit's progress, live, by polling the checkpoint files that `Protocol.run` already writes to disk once per completed iteration - without ever showing you a gauge-dependent quantity before it's actually meaningful.

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI
```

## Setting up some data to fit

As usual, we need a `ProtocolData` (an experiment design plus a dataset) to run GST on.

```{code-cell} ipython3
edesign = smq1Q_XYI.create_gst_experiment_design(max_max_length=8)
target_model = smq1Q_XYI.target_model()
datagen_model = target_model.depolarize(op_noise=0.01, spam_noise=0.01)

ds = pygsti.data.simulate_data(datagen_model, edesign.all_circuits_needing_data,
                               num_samples=1000, sample_error='binomial', seed=1234)
data = pygsti.protocols.ProtocolData(edesign, ds)
```

## Option 1: `Protocol.run_live` (spawn-and-watch)

The simplest way to get a live view is `Protocol.run_live`, a convenience wrapper around `LiveGSTMonitor`. It launches the fit as a separate (single) process and then blocks, polling that process's checkpoint directory and updating a live `plotly` figure, until the fit finishes:

```{code-cell} ipython3
:tags: [nbval-skip]

proto = pygsti.protocols.GateSetTomography(target_model, 'stdgaugeopt', name='exampleGST')

results = proto.run_live(
    data,
    checkpoint_path='./gst_checkpoints/exampleGST',  # optional; this is also the default
    poll_interval=2.0,  # seconds between checks of the checkpoint directory
)
```

While this cell is running, the notebook displays a two-panel figure that updates every `poll_interval` seconds:

- **Left panel**: p-value vs. circuit-depth iteration, with a reference line at the commonly-used $p=0.05$ "this fit looks statistically implausible" threshold.
- **Right panel**: each gate's eigenvalues (in the complex plane, with a unit-circle reference) at the most recently completed iteration.

Both panels are explicitly labeled **"PROVISIONAL: pre-gauge-optimization"**, and the figure title also shows when it was last updated - a reminder that everything shown is deliberately restricted to gauge-invariant quantities from a fit that hasn't been gauge optimized yet.

`run_live` returns the same final, gauge-optimized `ProtocolResults` object an ordinary `proto.run(data)` call would - the live view doesn't change what's ultimately computed, only what you can see while waiting for it.

```{code-cell} ipython3
:tags: [nbval-skip]

final_model = results.estimates['exampleGST'].models['stdgaugeopt']
print(pygsti.tools.two_delta_logl(final_model, ds))
```

## Option 2: `LiveGSTMonitor.watch` (attach to an already-running job)

`run_live`/`LiveGSTMonitor.run` only launches a single (non-MPI) process. If you're running a larger fit via `Protocol.run_mpi` or via a SLURM job staged with `Protocol.stage_slurm`, launch that job however you normally would, and separately attach a monitor to its checkpoint directory with `LiveGSTMonitor.watch`:

```{code-cell} ipython3
:tags: [nbval-skip]

from pygsti.report.livemonitor import LiveGSTMonitor

# Point this at the *same* checkpoint_path used by the already-running (or
# already-finished) job - e.g. the directory a SLURM job is writing
# checkpoints into on a shared filesystem.
monitor = LiveGSTMonitor(
    './gst_checkpoints/exampleGST',
    n_iterations=len(edesign.circuit_lists),  # optional; lets watch() know when to stop
    poll_interval=2.0,
)
monitor.watch()
```

`watch` doesn't launch or control the fit at all - it only ever needs read access to the checkpoint directory, so it works identically whether the job is local, remote, or running under MPI/SLURM on a shared filesystem. If `n_iterations` isn't known ahead of time, `watch` polls indefinitely; stop it with a `timeout=` argument or by interrupting the cell.

## Monitoring a `StandardGST` run

`StandardGST` fits several named "modes" (e.g. `'full TP'`, `'CPTPLND'`) as child protocols. Point `checkpoint_path`/`mode` at the specific mode you want to watch:

```{code-cell} ipython3
:tags: [nbval-skip]

std_proto = pygsti.protocols.StandardGST(modes=['full TP'], name='exampleStdGST')
results = std_proto.run_live(
    data,
    checkpoint_path='./gst_checkpoints/exampleStdGST',
    mode='full TP',  # only needed if more than one GST mode is being run
)
```

## Limitations

- Only `GateSetTomography` and `StandardGST` are supported - `LinearGateSetTomography` is a single non-iterative fit with no per-depth progression to show, and `ModelTest` writes an incompatible checkpoint format (it isn't fitting a model at all).
- `run_live` only spawns a single (non-MPI) process; for multi-rank runs, launch the fit yourself (`run_mpi`/`stage_slurm`) and use `LiveGSTMonitor.watch` to monitor it.
- Rendering a live-updating figure requires `plotly`'s `FigureWidget`, which in turn needs either the `anywidget` package (plotly &ge; 6) or `ipywidgets` (older plotly) to be installed alongside plotly.

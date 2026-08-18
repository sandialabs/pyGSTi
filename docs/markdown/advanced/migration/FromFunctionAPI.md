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

# Migrating from the function-based API

Older pyGSTi code drives analyses through module-level *driver functions*: `run_long_sequence_gst`, `run_stdpractice_gst`, `run_model_test`, and friends. Most of them still work.

**They are no longer the recommended way to use pyGSTi, and we expect to deprecate them in the near future.** New code should use the protocol objects instead. If you are writing something you intend to keep, write it against the protocol API; if you maintain a script built on the driver functions, this page is the translation.

```{warning}
`run_linear_gst` is currently broken: it passes a `sequenceRules` argument that `StandardGSTDesign` no longer accepts, so every call raises `TypeError`. Use `pygsti.protocols.LinearGateSetTomography` instead. The other four driver functions run.
```

## Why the protocol objects

The driver functions bundle four separate concerns — describing the experiment, holding the data, choosing the analysis, and configuring the optimizer — into one call with a long tail of keyword arguments. The protocol API separates them into objects you can build, inspect, save and reuse:

| Concern | Object |
|---|---|
| Which circuits the experiment needs | an **experiment design** (`StandardGSTDesign`, `GSTDesign`, …) |
| The counts you collected | a **`ProtocolData`** (an experiment design plus a `DataSet`) |
| The analysis to run | a **protocol** (`GateSetTomography`, `StandardGST`, `ModelTest`, …) |
| What came out | a **results** object, written to and read back from a directory |

The practical wins: an experiment design can be written to disk before you take data and read back afterwards; the same data object can be handed to several protocols; and results carry their own provenance rather than living in whatever variable you assigned them to.

## Function → protocol map

| Driver function | Use instead | Notes |
|---|---|---|
| `pygsti.run_long_sequence_gst` | `pygsti.protocols.GateSetTomography` | One GST estimate from one starting model. The starting model's parameterization *is* the constraint: pass `target_model("full TP")` to constrain the fit to TP gate sets. |
| `pygsti.run_stdpractice_gst` | `pygsti.protocols.StandardGST` | Runs several parameterizations in one pass. The old `modes="full TP,CPTP,Target"` comma-string still parses but warns; pass a tuple, e.g. `modes=('full TP','CPTPLND','Target')`. |
| `pygsti.run_linear_gst` | `pygsti.protocols.LinearGateSetTomography` | LGST only. The driver function is broken (see the warning above); the protocol works. |
| `pygsti.run_model_test` | `pygsti.protocols.ModelTest` | See [model testing](../../guides/analysis/ModelTesting). |
| `pygsti.run_long_sequence_gst_base` | `pygsti.protocols.GateSetTomography` with an explicit design | The `_base` variant existed to accept a pre-built circuit structure; build a `GSTDesign` instead. |

Keyword arguments move too. `advanced_options={'estimate_label': ...}` becomes the protocol's `name`, and bad-fit handling, which the drivers exposed piecemeal through `advanced_options`, is now a `GSTBadFitOptions` object passed as `badfit_options`.

Gauge optimization is the one worth care. The driver's `gauge_opt_params` dictionary is wrapped internally into a `GSTGaugeOptSuite` under the key `'go0'`, and it is ignored unless you also pass `gauge_opt_suite_name=None`, because that argument defaults to `'stdgaugeopt'`. On the protocol side you pass `gaugeopt_suite` directly and there is no second argument to remember.

## The same analysis, both ways

Before — one call that hides the experiment design:

```python
results = pygsti.run_long_sequence_gst(
    ds, target_model, prep_fiducials, meas_fiducials, germs, maxLengths)
```

After — the design is a real object, and the data is separable from the analysis:

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI

# 1. What circuits does the experiment need?
edesign = smq1Q_XYI.create_gst_experiment_design(max_max_length=8)

# 2. Pair those circuits with counts. (Simulated here; see the data-loading
#    tutorial for reading counts you collected yourself.)
noisy = smq1Q_XYI.target_model().depolarize(op_noise=0.01, spam_noise=0.001)
ds = pygsti.data.simulate_data(noisy, edesign.all_circuits_needing_data,
                               num_samples=1000, seed=1234)
data = pygsti.protocols.ProtocolData(edesign, ds)

# 3. Choose the analysis. The starting model carries the parameterization.
protocol = pygsti.protocols.GateSetTomography(
    smq1Q_XYI.target_model("full TP"), verbosity=1)

# 4. Run it.
results = protocol.run(data)
print(results)
```

The estimate is reached the same way it always was:

```{code-cell} ipython3
estimate = results.estimates['GateSetTomography']
print(estimate.models['final iteration estimate'].num_params, "parameters")
```

## Where to go next

- [Running GST](../../guides/gst/RunningGST) — the protocol API in full, including parameterization choices, gauge optimization and bad-fit handling.
- [GST overview](../../start/FirstGST) — start here if you are new to GST rather than porting existing code.
- [Model testing](../../guides/analysis/ModelTesting) — the `ModelTest` protocol, which replaces `run_model_test`.

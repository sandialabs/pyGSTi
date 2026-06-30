# 05 — Reporting and persistence

**Covers:** [pygsti/report/](../pygsti/report/), [pygsti/io/](../pygsti/io/), [pygsti/serialization/](../pygsti/serialization/).

`report/` is the largest of this doc's covered subpackages, and is also pyGSTi's most common source of contributor confusion.
This doc's coverage of `io/` and `serialization/` is smaller and deferred to the end.

## What lives here

- [`report/`](../pygsti/report/) — turn a `ProtocolResults` (typically a `ModelEstimateResults`) into an interactive HTML report, a static PDF, or a Jupyter notebook of plots and tables. ~27k lines across 41 files; the largest single subpackage besides `tools/`.
- [`io/`](../pygsti/io/) — DataSet text-format I/O ([readers.py](../pygsti/io/readers.py), [writers.py](../pygsti/io/writers.py)), directory-based object-graph persistence ([metadir.py](../pygsti/io/metadir.py)), and an optional MongoDB backend ([mongodb.py](../pygsti/io/mongodb.py)).
- [`serialization/`](../pygsti/serialization/) — the low-level JSON codec ([jsoncodec.py](../pygsti/serialization/jsoncodec.py)) used by `metadir`. Small (~600 lines).

## Mental model for `report/`

There are four ideas to hold in your head before you touch anything in `report/`.

### 1. The user-facing path is `construct_standard_report` → `Report` → `.write_html(...)`

```python
report = pygsti.report.construct_standard_report(results, title="my fit")
report.write_html("path/to/output_dir")
```

[`construct_standard_report`](../pygsti/report/factory.py#L1149) is the canonical entry; it's what the notebooks teach. It returns a [`Report`](../pygsti/report/report.py) object holding templates, results, and a [`Workspace`](../pygsti/report/workspace.py#L177). Call `report.write_html(path)` (or `.write_pdf`, `.write_notebook`) to materialize.

The other public factories are:

| Function | File:line | When |
|---|---|---|
| [`construct_standard_report`](../pygsti/report/factory.py#L1149) | factory.py:1149 | **Canonical entry.** 1–2 qubit Models with dense matrices. |
| [`create_standard_report`](../pygsti/report/factory.py#L602) | factory.py:602 | Lower-level disk-writing variant of standard report. |
| [`create_report_notebook`](../pygsti/report/factory.py#L976) | factory.py:976 | Generate a Jupyter notebook of the report cells. |
| [`create_drift_report`](../pygsti/report/factory.py#L1644) | factory.py:1644 | Specialized for drift-characterization output. |

Two other factories — [`construct_nqnoise_report`](../pygsti/report/factory.py#L1429) and [`create_general_report`](../pygsti/report/factory.py#L579) — exist in the file but are rarely seen in user workflows. Don't elevate them to canonical status.

### 2. `Workspace` is a factory plus a smart cache

[`Workspace`](../pygsti/report/workspace.py#L177) is the engine that produces plots and tables. It's two things at once:

- **A factory** for [`WorkspacePlot`](../pygsti/report/workspace.py#L2428), [`WorkspaceTable`](../pygsti/report/workspace.py#L2012), and [`WorkspaceText`](../pygsti/report/workspace.py#L2773) objects (all subclasses of [`WorkspaceOutput`](../pygsti/report/workspace.py#L1449)). The factory methods are generated **dynamically via `exec()`** inside `Workspace._makefactory()` — they aren't statically defined on the class. Calling `ws.GatesTable(model)` invokes one of these exec-generated wrappers that constructs a `GatesTable` instance backed by the cache.
- **A smart MD5-keyed cache** for the outputs. `Workspace` carries a `SmartCache` and a custom digest function ([`ws_custom_digest`](../pygsti/report/workspace.py#L141)) that knows how to fingerprint the special objects involved (Plotly figures, `NotApplicable` placeholders, `SwitchValue` arrays).

The reason `Workspace` isn't just a simple `@functools.lru_cache`: outputs depend not just on inputs but on *switch positions* (see point 3), and Plotly figures aren't trivially hashable.

### 3. Switchboards: the same plot, indexed by switch position

A standard report is interactive: the reader can switch between estimates, gauge optimizations, datasets, and so on. The mechanism is a [`Switchboard`](../pygsti/report/workspace.py#L725) — an `OrderedDict`-derived holder for switch definitions, plus [`SwitchValue`](../pygsti/report/workspace.py#L1372) objects which are *multi-dimensional numpy arrays of values*, one axis per switch.

So a plot doesn't depend directly on "this Model" — it depends on a `SwitchValue` that holds `[N_estimates × N_gauge_opts × N_datasets]` models, and a tuple-of-switch-positions tells you which one is current. The cache key includes the switch-position tuple via `ws_custom_digest`.

[`SwitchboardView`](../pygsti/report/workspace.py#L1270) is a linked sub-view of a Switchboard so that two parts of a report can stay in sync as the reader flips switches.

**Why this matters for you:** a single plot/table object can correspond to dozens of rendered variants. Cache hits across variants are what makes report generation tractable.

### 4. Reportables are the diagnostic-metric library

[`reportables.py`](../pygsti/report/reportables.py) (2868 lines) defines ~100+ classes, each a subclass of `ModelFunction`. Each represents one diagnostic metric — `GateEigenvalues`, `CircuitEigenvalues`, `HalfDiamondNorm`, fidelity variants, error-rate variants, etc.

A reportable is a callable: it takes a Model (and optional auxiliary arguments) and returns a scalar or array quantity, optionally wrapped in a [`ReportableQty`](../pygsti/report/reportableqty.py#L64) carrying an error bar. When a plot or table needs a metric, it calls into reportables.

The reportables file is **predominantly hand-written**, not generated. Adding a new diagnostic metric means adding a new reportable class (and almost always a new plot or table to render it).

## Public entry-point inventory

(See the table in the "User-facing path" section above.) The reduced inventory — `construct_standard_report`, `create_standard_report`, `create_report_notebook`, `create_drift_report` — is what you should see in real user code.

## Key abstractions

| Class / function | File:line | Role |
|---|---|---|
| [`Report`](../pygsti/report/report.py) | report.py | Wraps templates + a `Workspace` + results; exposes `write_html` / `write_pdf` / `write_notebook`. |
| [`Workspace`](../pygsti/report/workspace.py#L177) | workspace.py:177 | Factory + smart cache. |
| [`Switchboard`](../pygsti/report/workspace.py#L725) | workspace.py:725 | Multi-axis switch holder. |
| [`SwitchValue`](../pygsti/report/workspace.py#L1372) | workspace.py:1372 | Array of values indexed by switch position. |
| [`SwitchboardView`](../pygsti/report/workspace.py#L1270) | workspace.py:1270 | Linked sub-view of a Switchboard. |
| [`WorkspaceOutput`](../pygsti/report/workspace.py#L1449) and subclasses | workspace.py:1449 ff | `WorkspacePlot` (2428), `WorkspaceTable` (2012), `WorkspaceText` (2773), `NotApplicable` (1961). |
| [`ws_custom_digest`](../pygsti/report/workspace.py#L141) | workspace.py:141 | The cache-key digest function. Where bugs about stale caches usually live. |
| Reportables — `ModelFunction` subclasses | [reportables.py](../pygsti/report/reportables.py) | Diagnostic metric library. |
| [`ReportableQty`](../pygsti/report/reportableqty.py#L64) | reportableqty.py:64 | Value + error-bar wrapper. |
| Plot classes (~100) | [workspaceplots.py](../pygsti/report/workspaceplots.py) | Plotly-based figure renderers, all `WorkspacePlot` subclasses. |
| Table classes (~50) | [workspacetables.py](../pygsti/report/workspacetables.py) | `ReportTable`-backed table renderers. |
| Output formatters | [html.py](../pygsti/report/html.py), [latex.py](../pygsti/report/latex.py), [python.py](../pygsti/report/python.py), [notebook.py](../pygsti/report/notebook.py) | Format-specific rendering. |
| Templates | [pygsti/report/templates/](../pygsti/report/templates/) | Plain HTML with string substitution (**not** Jinja2). |
| FOGI diagrams | [fogidiagram.py](../pygsti/report/fogidiagram.py) | Specialized FOGI-decomposition diagnostic. |

## Tricky bits

1. **Switchboard indirection.** New contributors expect a plot to depend on "this Model." It depends on a *switch position into a SwitchValue array of Models.* The tuple-of-positions ↔ cache-key mapping is implicit. If you're adding a plot that needs to vary with a new switch, you have to (a) define the switch on the Switchboard, (b) shape your value array accordingly, (c) make sure `ws_custom_digest` knows how to fingerprint your axis values.

2. **Reportables → Plot/Table chain is non-obvious.** A plot fetches a metric via `self.ws.SomeTable(...)` or via direct reportable construction. Where is "process fidelity" actually computed? Somewhere in 2868 lines of `reportables.py` with minimal cross-references. There is no central index. The practical advice: grep for the metric name across both `reportables.py` and `workspaceplots.py` / `workspacetables.py`.

3. **Cache invalidation through `ws_custom_digest`.** The custom digest at [workspace.py:141](../pygsti/report/workspace.py#L141) handles `NotApplicable`, `SwitchValue`, and Plotly figures specially. If you introduce a new "input type" to a reportable or plot and don't extend the digest, stale outputs will be served from cache.

4. **Factory pattern via `exec()`.** [`Workspace._makefactory()`](../pygsti/report/workspace.py) generates factory methods dynamically. Calling `ws.GatesTable(...)` does *not* dispatch to a statically defined method; the function is constructed at runtime. Static analysis (IDE "go to definition", `mypy`, ...) will not find it. To find the underlying class, look in `workspacetables.py` / `workspaceplots.py` for `class GatesTable(...)`.

## Where to start (and where not to)

If you're new to `report/` and have a change to make:

1. Read [docs/markdown/reporting/Workspace.md](../docs/markdown/reporting/Workspace.md), then [WorkspaceSwitchboards.md](../docs/markdown/reporting/WorkspaceSwitchboards.md). These are the canonical conceptual intros.
2. In code, read the `Workspace` class skeleton ([workspace.py:177–270](../pygsti/report/workspace.py#L177)) — focus on `__init__`, `_makefactory`, and the digest interaction.
3. Then read one simple factory function — [`create_general_report` at factory.py:579](../pygsti/report/factory.py#L579) is a reasonable choice because it's shorter than `construct_standard_report`.
4. **Do not start in [reportables.py](../pygsti/report/reportables.py).** 2868 lines of metric formulas with no scaffolding to orient on. Only go there to add or fix a specific metric.

## Debugging roadmap

| Symptom | First place to look |
|---|---|
| Plot isn't rendering / shows blank | [workspaceplots.py](../pygsti/report/workspaceplots.py); find the class, inspect its `_create()`. |
| Quantity is numerically wrong | [reportables.py](../pygsti/report/reportables.py); find the metric class, check the formula. |
| Cache is stale (plot doesn't update when underlying Model changes) | [`ws_custom_digest`](../pygsti/report/workspace.py#L141); check whether your input type is being hashed correctly. |
| Switchboard updates don't propagate | `SwitchboardView` and cache-key generation at [workspace.py:141–163](../pygsti/report/workspace.py#L141). |
| PDF generation fails | Confirm `pdflatex` is on PATH; the route is Plotly → matplotlib → LaTeX → PDF. |
| HTML missing styles or interactivity | Check `templates/standard_html_report/` and the `offline/` resources directory. |

## Output formats and templates

Templates live in [pygsti/report/templates/](../pygsti/report/templates/). They are **plain HTML with string substitution** — *not* Jinja2 (despite the appearance). The substitution is done in [html.py](../pygsti/report/html.py).

- **HTML** (primary, fully interactive): Plotly JSON embedded directly, JS for switchboard interaction.
- **PDF** (static, single-variant): Plotly → matplotlib conversion, then LaTeX, then `pdflatex`. **Requires `pdflatex` on PATH**; will fail noisily if missing.
- **Notebook**: Jupyter cells, each containing one `WorkspacePlot` or `WorkspaceTable`.

## `io/` — DataSet I/O, metadir persistence, MongoDB backend

[`io/`](../pygsti/io/) splits into roughly three concerns:

- **Dataset text-format I/O.** [`read_dataset`](../pygsti/io/readers.py) and [`write_dataset`](../pygsti/io/writers.py) handle the human-readable text format used to ship experimental data. This is the format most tutorials use.
- **Directory-based object persistence.** [`metadir.py`](../pygsti/io/metadir.py) implements the convention of "object graph as a directory of `meta.json` + auxiliary files." This is what `Protocol`, `ProtocolData`, and `ProtocolResults` use for `.write()` / `.read_dir()` round-trips. It calls into `serialization/jsoncodec.py` for the JSON layer.
- **MongoDB backend.** [`mongodb.py`](../pygsti/io/mongodb.py) is an optional alternative persistence target. Guarded by a try-import of `pymongo`.

**Gotcha.** [pygsti/io/metadir.py:96](../pygsti/io/metadir.py#L96) marks several formats as `# DEPRECATED formats! REMOVE LATER`. The code still reads them (so old saved object graphs continue to load) but you should not extend them. See [known-debt.md #6](known-debt.md#6-iometadirpy-deprecated-formats).

## `serialization/` — JSON codec

[`serialization/jsoncodec.py`](../pygsti/serialization/jsoncodec.py) is the low-level codec: it knows how to JSON-encode numpy arrays, scipy sparse matrices, complex numbers, and pyGSTi `NicelySerializable` objects. Used by `io/metadir.py` for the JSON-bearing parts of an object-graph directory. The two functions of interest are `encode_obj` and `decode_obj`.

This subpackage is small (~600 lines, 2 files) and stable. You only need to touch it if you're adding a new type that needs custom JSON serialization.

## Architectural debt

- [`Report` "needs rewrite" note](known-debt.md#12-reportreportpy-needs-rewrite-note) — comment at the top of [report.py](../pygsti/report/report.py) flags that `Report` should arguably be a base class with derived classes per report type rather than one class configured many ways.
- [`io/metadir.py` deprecated formats](known-debt.md#6-iometadirpy-deprecated-formats).
- Tangentially, [#205](https://github.com/sandialabs/pyGSTi/issues/205) — JupyterLab-compatibility rethink for the report-system inline JavaScript.

## Canonical examples

Notebooks under [docs/markdown/reporting/](../docs/markdown/reporting/) are the only good teaching material. There are no high-quality code-level examples inside the `report/` subpackage itself.

- [docs/markdown/reporting/ReportGeneration.md](../docs/markdown/reporting/ReportGeneration.md) — top-level report-generation walkthrough.
- [docs/markdown/reporting/Workspace.md](../docs/markdown/reporting/Workspace.md) — the Workspace cache model.
- [docs/markdown/reporting/WorkspaceExamples.md](../docs/markdown/reporting/WorkspaceExamples.md) — concrete plot/table examples.
- [docs/markdown/reporting/WorkspaceSwitchboards.md](../docs/markdown/reporting/WorkspaceSwitchboards.md) — switchboard mechanics.
- [docs/markdown/Reporting.md](../docs/markdown/Reporting.md) — top-level entry point in the jupyter-book.

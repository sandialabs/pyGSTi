# Known architectural debt and in-flight redesigns

This file is an **index**, not a redesign proposal. Each entry says *what* the debt is, *where* it bites you in practice, and links a tracker on `sandialabs/pyGSTi` when one exists. If you hit one of these and discover it's already burned you, please file or comment on the linked issue. If an entry says "no tracker — please open one," that's not a hint, it's the policy: open an issue so the next person to hit this lands somewhere instead of bouncing.

Don't treat any entry here as a redesign plan in itself — the linked GitHub issue (when present) is the authoritative discussion thread.

## How to find a tracker for a smell you've hit

```
# Search by keyword:
gh issue list --repo sandialabs/pyGSTi --search "<keywords>" --state all

# Or via the GitHub web UI:
https://github.com/sandialabs/pyGSTi/issues?q=is%3Aissue+<keywords>
```

If you find one not listed below, please update this file.

---

## 1. `tools/` misplacement and namespace pollution

**What it is.** Several files under [pygsti/tools/](../pygsti/tools/) are domain features, not utilities — they import "upward" from `models`, `protocols`, `report`, etc. Concretely: `leakage.py`, `chi2fns.py`, `likelihoodfns.py`, `rbtheory.py`, `edesigntools.py` are the main offenders. Separately, no `tools/` module defines `__all__`, so `pygsti.tools.*` re-exports every public name from every file with no curation.

**Where it bites.** Adding "just one more utility" to `tools/` tends to deepen the coupling; touching a `tools/` file can pull `models` or `protocols` along for the ride at import time. The lack of `__all__` makes it hard to deprecate names — you can't tell what's actually part of the public API.

**Tracker.** Partially covered by the broader subpackage-restructuring discussion in [sandialabs/pyGSTi#715](https://github.com/sandialabs/pyGSTi/issues/715). A focused `tools/` issue would be welcome — please open one if you start work here.

## 2. `tools/leakage.py` → `pygsti.leakage` move

**What it is.** `tools/leakage.py` is the most flagrant of the misplaced-module offenders: it contains complete domain features (leaky-qubit GST + LAGO gauge optimization + report generation), with lazy imports from `models`, `protocols`, and `report`. There is an in-flight plan to extract it into a top-level `pygsti.leakage` subpackage.

**Where it bites.** Don't add new code to `tools/leakage.py` — anything you write will need to be migrated. New leakage-related code should land in the new subpackage layout once it exists.

**Tracker.** No standalone tracker issue — please open one if you're picking up the migration.

## 3. `baseobjs` ↔ `protocols` circular import

**What it is.** [pygsti/baseobjs/](../pygsti/baseobjs/) is conceptually the bottom of the dependency stack, but there's a residual circular import with [pygsti/protocols/](../pygsti/protocols/) somewhere in the load order. Routinely sidestepped by deferred imports.

**Where it bites.** Refactoring `baseobjs` is harder than it should be; certain reorderings of imports trip on the cycle.

**Tracker.** No standalone tracker — please open one if you decide to break the cycle. Likely related to [#715](https://github.com/sandialabs/pyGSTi/issues/715) (subpackage restructuring).

## 4. `MDCObjectiveFunction` multi-inheritance smell

**What it is.** [pygsti/objectivefns/objectivefns.py:1126](../pygsti/objectivefns/objectivefns.py#L1126) inherits both `ObjectiveFunction` *and* `EvaluatedModelDatasetCircuitsStore`. That's "is-an-objective-function" *and* "is-a-Model-+-Dataset-+-Circuits-cache" mashed into the same class. Most callers want only one of these roles.

**Where it bites.** Anyone changing the objective-function interface (adding a method, renaming a kwarg) has to reason about both bases simultaneously. The class is also a hard read.

**Related historical bugs.** [#718](https://github.com/sandialabs/pyGSTi/issues/718) (TimeDependent variant constructor bug) and [#719](https://github.com/sandialabs/pyGSTi/issues/719) (Cached variant serialization) both involved confusion about which base class supplied which attribute.

**Tracker.** No standalone refactor issue — please open one.

## 5. `modelpacks/legacy/` — 19 old-style `std*` files

**What it is.** [pygsti/modelpacks/legacy/](../pygsti/modelpacks/legacy/) holds 19 files using the older `std*` naming convention (predating the `smq*` convention used by current packs). They still work but use a different (older) API.

**Where it bites.** If you pattern-match on `legacy/` to build a new modelpack, you'll inherit the older API. The current canonical example is [pygsti/modelpacks/smq1Q_XY.py](../pygsti/modelpacks/smq1Q_XY.py).

**Tracker.** No standalone tracker — covered indirectly by [#715](https://github.com/sandialabs/pyGSTi/issues/715).

## 6. `io/metadir.py` deprecated formats

**What it is.** [pygsti/io/metadir.py:96](../pygsti/io/metadir.py#L96) still reads several persistence formats annotated `# DEPRECATED formats! REMOVE LATER` (`'text-circuit-lists'`, `'list-of-protocolobjs'`, etc.). They linger because removing them breaks loading of old serialized object graphs.

**Where it bites.** Touching `metadir.py` requires deciding whether your change extends the deprecated branch or only the supported one. Easy to accidentally keep deprecated formats alive longer.

**Tracker.** No standalone tracker — please open one when you start removing them.

## 7. `tools/chi2fns.py` deprecated function names

**What it is.** [pygsti/tools/chi2fns.py](../pygsti/tools/chi2fns.py) still exports older chi-squared function names alongside the modern `RawChi2Function` class-based interface. Several are marked `@deprecate`.

**Where it bites.** New callers find both styles in the namespace and may pick the deprecated one. Removing the deprecated names requires migrating all internal callers first.

**Tracker.** Part of the test-suite warnings cleanup tracked at [#706](https://github.com/sandialabs/pyGSTi/issues/706).

## 8. `extras/__init__.py` imports commented out

**What it is.** [pygsti/extras/__init__.py](../pygsti/extras/__init__.py) has its sub-module imports commented out. Result: `pygsti.extras.rpe` etc. are not auto-imported by `import pygsti`; users must use the full module path.

**Where it bites.** Code that assumes `pygsti.extras.rpe` is available after `import pygsti` will silently fail. Signals that several `extras/` subdirectories are not fully integrated and the maintainers haven't decided what to do about it.

**Tracker.** No standalone tracker — please open one to drive a decision.

## 9. `CustomLMOptimizer` is legacy-only

**What it is.** [pygsti/optimize/customlm.py:33](../pygsti/optimize/customlm.py#L33) is kept for backward compatibility. The path forward is [pygsti/optimize/simplerlm.py:109](../pygsti/optimize/simplerlm.py#L109)'s `SimplerLMOptimizer`.

**Where it bites.** Don't add new callers of `CustomLMOptimizer`. Existing callers should migrate when convenient.

**Tracker.** No standalone tracker — please open one.

## 10. `report/report.py` "needs rewrite" note

**What it is.** [pygsti/report/report.py](../pygsti/report/report.py) carries an inline comment near the top of the file noting "this whole thing needs to be rewritten with different reports as derived classes." Today `Report` is one class instantiated by several factory functions in [pygsti/report/factory.py](../pygsti/report/factory.py); the comment is suggesting subclassing instead.

**Where it bites.** Adding a new report variant (e.g., `create_X_report`) means adding another factory that ad-hoc configures the same `Report` instance, rather than subclassing. Tangentially related: [#205](https://github.com/sandialabs/pyGSTi/issues/205) ("Rethink inline javascript for reports for JupyterLab compatibility") covers a different aspect of report-system architecture.

**Tracker.** No standalone tracker for the per-report-type subclassing refactor — please open one.

## 11. `LogLOptions`-style parameter bundling not yet implemented

**What it is.** Likelihood/chi-squared function signatures in [pygsti/tools/likelihoodfns.py](../pygsti/tools/likelihoodfns.py) and [pygsti/tools/chi2fns.py](../pygsti/tools/chi2fns.py) still pass `min_prob_clip`, `prob_clip_interval`, `radius`, `op_label_aliases`, `comm` individually — repeatedly, across many functions. A `LogLOptions` (or similar) dataclass would consolidate the bag.

**Where it bites.** Long signatures are easy to misuse; default values drift between call sites; refactoring is tedious.

**Tracker.** No standalone tracker — please open one.

## 12. `gaugeopt_suite` representation duality

**What it is.** A gauge-optimization suite is variously represented as a `list[list[dict]]` *or* as a [pygsti/protocols/gst.py:857](../pygsti/protocols/gst.py#L857) `GSTGaugeOptSuite` object. Different entry points accept different shapes; some accept both.

**Where it bites.** Constructing the suite for a non-trivial gauge-optimization configuration requires knowing which shape the caller expects. Worth covering with a concrete example in any code that touches gauge configuration. Related: [#620](https://github.com/sandialabs/pyGSTi/issues/620) (parameterization-preserving gauge optimization).

**Tracker.** No standalone tracker for the duality itself — please open one.

## 13. `extras/idletomography/` is broken

**What it is.** The idle-tomography subsystem under [pygsti/extras/idletomography/](../pygsti/extras/idletomography/) is known-broken. Maintainers are aware. **Do not attempt to introspect, run, or fix it as part of unrelated work** — it has its own scope.

**Tracker.** Related open issues: [#711](https://github.com/sandialabs/pyGSTi/issues/711) (IDT with custom qubit labels), [#737](https://github.com/sandialabs/pyGSTi/issues/737) (error simulating IDT circuits), [#576](https://github.com/sandialabs/pyGSTi/issues/576) (view IDT results).

## 14. Towards 1.0: subpackage restructuring meta-thread

**What it is.** A meta-discussion of pyGSTi's maintainability and the structural changes needed before a 1.0 release. Motivated explicitly by making the codebase tractable for LLM-based coding agents — i.e., the same goal as `agent-docs/`. Several proposals (move `pygsti.layouts` under `pygsti.forwardsims`, etc.) live in the issue body.

**Tracker.** [sandialabs/pyGSTi#715](https://github.com/sandialabs/pyGSTi/issues/715).

## 15. POVM inheritance structure refactor

**What it is.** [pygsti/modelmembers/povms/povm.py](../pygsti/modelmembers/povms/povm.py)'s `POVM` base class and `_BasePOVM` have inverted-ish roles: `POVM` actually implements a fully-wired-up zero-parameter POVM, but most concrete subclasses do have parameters. This forces awkward overrides.

**Tracker.** [sandialabs/pyGSTi#727](https://github.com/sandialabs/pyGSTi/issues/727).

## 16. Inconsistent ModelMember copy/deepcopy/pickle semantics around `_parent`

**What it is.** Three distinct code paths handle `_parent` differently when copying a `ModelMember`:

1. **`ModelChild.copy(parent=...)`** (`modelmember.py`) — seeds the deepcopy memo with `memo[id(self.parent)] = None`, so the parent model is *not* copied; caller sets the new parent explicitly.
2. **`ModelChild.__getstate__`** — nulls `_parent` for the pickle/JSON path; `relink_parent` restores it on deserialization.
3. **`_DenseCopyMixin.__deepcopy__`** (`modelmember.py`) — copies `__dict__` verbatim, *including* `_parent`. A bare `copy.deepcopy(dense_member)` therefore deep-copies the entire parent model as a side effect.

Path 3 is *intentionally* inconsistent with paths 1 and 2 because the regression test for [#651](https://github.com/sandialabs/pyGSTi/issues/651) (`test/unit/modelmembers/test_operation.py::test_deepcopy`) requires it: the test calls `copy.deepcopy(op)` where `op._parent` is a live model, then asserts that `o2.parent` is a distinct but equivalent model.

**Where it bites.** Any future maintainer who "normalises" `_DenseCopyMixin.__deepcopy__` to null or drop `_parent` (for consistency with `__getstate__` or `ModelChild.copy`) will silently break the #651 test with `AttributeError: 'NoneType' object has no attribute 'create_modelmember_graph'`. The `_DenseCopyMixin` docstring in `modelmember.py` carries a prominent warning about this, but it is easy to miss during a refactor.

Additionally, the path-3 behaviour is arguably surprising to callers: `copy.deepcopy(op)` has an invisible, potentially expensive side effect (copying the parent model) that `ModelChild.copy()` deliberately avoids.

**Do not change `_DenseCopyMixin.__deepcopy__` without first resolving the tracking issue.**

**Tracker.** [sandialabs/pyGSTi#804](https://github.com/sandialabs/pyGSTi/issues/804)

## 17. `Label.IS_SIMPLE` conflates "layer-atom" with "primitive op"

**What it is.** [pygsti/baseobjs/label.py:173](../pygsti/baseobjs/label.py#L173) defines one class flag, read via the `is_simple` property ([label.py:464](../pygsti/baseobjs/label.py#L464)), that is asked two independent questions:

- **A — is this a layer-atom?** Does it occupy a single layer slot as an opaque unit, so the within-layer machinery must *not* splay its `.components` out as parallel co-layer siblings? This is what `circuit.py` wants.
- **B — is this a primitive op?** Is it one operation with no nested sub-label structure, reconstructible from `name`/`sslbls`(/`args`) and resolvable as a single op in a model? This is what the model, objective-function, and budget layers want.

The two agree for every label class except [`CircuitLabel`](../pygsti/baseobjs/label.py#L1419), which is a genuine hybrid: a sub-circuit box occupies one layer slot (A = true) but wraps a whole subcircuit (B = false). One boolean cannot hold both, so `CircuitLabel.IS_SIMPLE = True` ([label.py:1430](../pygsti/baseobjs/label.py#L1430)) is right for A and wrong for B — and note it flips the flag back to `True` under a `LabelTupTup` ancestry whose whole branch is `False`, so the True/False boundary does not follow the type hierarchy and no consumer can reason from the class alone.

**Where it bites.** Any B-consumer needs an `isinstance(..., CircuitLabel)` bypass or carries a latent bug. It has already shipped one silent data-loss bug: `map_names_inplace` / `map_state_space_labels_inplace` took the simple-label branch and rebuilt a box as a bare `Label(name, sslbls)`, *discarding the subcircuit*; commit `2aff09799` patched it with explicit `isinstance` guards ([circuit.py:3064](../pygsti/circuits/circuit.py#L3064), [:3109](../pygsti/circuits/circuit.py#L3109)) — a workaround, not a resolution. Two guards remain mis-aimed for the same reason: [wildcardbudget.py:598](../pygsti/objectivefns/wildcardbudget.py#L598)/[:658](../pygsti/objectivefns/wildcardbudget.py#L658) and [oplessmodel.py:408](../pygsti/models/oplessmodel.py#L408) all do `assert(not lbl.is_simple)` before recursing into `.components`, so a `CircuitLabel` that isn't a model-registered primitive raises instead of decomposing. Separately, the metrics family (`size`, `num_gates`, `num_nq_gates`, `num_multiq_gates`) short-circuits on `IS_SIMPLE` while `depth` is hand-overridden to descend, so a box of five 2-qubit layers × 3 reps reports `size == 2`, `num_gates == 1`, `depth == 15` — an inconsistency that is a side effect of the flag rather than a decision.

To see any of this you must retain a box in a static circuit, which means the editable path: `Circuit.default_expand_subcircuits` is `True`, so `Circuit([cl], ...)` flattens it. Build with `editable=True` and call `done_editing()`.

**Don't** add new `is_simple` consumers without deciding which axis you mean. If you want "won't corrupt a layer if I don't descend," that's A and `is_simple` is correct today. If you want "I can look this up as one op," that's B and `is_simple` will lie to you about boxes.

**Tracker.** [sandialabs/pyGSTi#766](https://github.com/sandialabs/pyGSTi/issues/766) has the full analysis, the per-class A/B table, and the proposed `IS_LAYER_ATOM` / `IS_PRIMITIVE_OP` split with a call-site migration table. The design is accepted; implementation is deliberately held for the `Circuit`/`Label` rewrite, because the metrics-family row is a policy decision that would move the golden characterization fixtures under `test/unit/circuits/golden/`.

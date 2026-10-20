# Expiry guide: removing `TermForwardSimulator` and its term machinery (pyGSTi 0.12)

> **Audience.** An agent (or developer) asked to *expire* the `TermForwardSimulator` deprecation, i.e. delete the code that was deprecated in 0.11. This guide is self-contained: you shouldn't need the PR history. Background and tracker: [sandialabs/pyGSTi#916](https://github.com/sandialabs/pyGSTi/issues/916). Summary entry: [known-debt #19](known-debt.md#19-termforwardsimulator-deprecation-expires-in-012).
>
> **Line numbers drift.** Every item below is named by file *and* symbol. Locate it with `grep`, not by line number. The grep commands in [§9](#9-final-audit-greps) are the authoritative completeness check.

## 0. What was deprecated, and how to recognise it

In 0.11, `TermForwardSimulator` and the machinery that exists **solely** to support it were deprecated. Every deprecated item is marked in one of four ways, so these greps find (almost) everything:

| Marker | Grep |
|---|---|
| Runtime warning on a method/property | `grep -rn "_TERM_SIM_DEPRECATION_MESSAGE" pygsti` |
| Runtime warning in `TermForwardSimulator.__init__` | `grep -rn "warn_deprecated(\"TermForwardSimulator\"" pygsti` |
| Docstring note | `grep -rn "deprecated:: 0.11" pygsti` |
| Comment | `grep -rn "916" pygsti` |

**Caution:** `deprecated:: 0.11` is also used by *unrelated* 0.11 deprecations (e.g. `StaticStandardOp`). Only remove items whose note mentions `TermForwardSimulator`, term machinery, or #916.

## 1. Do NOT remove these (shared / general-purpose)

These were reviewed and deliberately **kept**. Several look term-related but are used elsewhere, including by **user code outside the repo** built on error-generator propagation, so "no in-repo callers" is *not* a reason to delete them.

| Keep | Why |
|---|---|
| `pygsti/baseobjs/polynomial.py`: `Polynomial` and everything on it (`compact`, `map_indices(_inplace)`, `mapvec_indices(_inplace)`, `to_rep`/`from_rep`, `deriv`, `evaluate`, …), `bulk_load_compact_polynomials`, `compact_polynomial_list` | Error-generator propagation (`pygsti/tools/errgenpolytools.py`) and user code |
| `PolynomialRep` (Python fallback in `evotypes/basereps.py`, Cython in `evotypes/basereps_cython.pyx/.pxd`) and `PolynomialCRep` (`evotypes/basecreps.cpp/.h`) | Back `Polynomial` |
| `pygsti/baseobjs/opcalc` compact-polynomial evaluators: `bulk_eval_compact_polynomials{,_real,_complex}`, `bulk_eval_compact_polynomials_derivs{,_real,_complex}`, `compact_deriv`, `abs_sum_bulk_eval_compact_polynomials_complex` (both `fastopcalc.pyx` and `slowopcalc.py`) | Used by user code built on error-generator propagation (maintainer decision) |
| `opcalc.float_product`; `fastopcalc` `test_map`, `check_polynomials`, `dot`, `stabilizer_measurement_prob` | Not term-specific (`float_product` is used by `models/oplessmodel.py`). The others are dead but were explicitly left out of scope |
| `tools/listtools.py`: `partition_into`, `_partition_into_slow`, `incd_product` | General-purpose combinatorics (maintainer decision), even though only term code calls them today |
| `OpRepExpErrorgen` / `OpRepLindbladErrorgen` classes in `evotypes/{statevec,stabilizer}/opreps.pyx` and `evotypes/{statevec,stabilizer}_slow/opreps.py` | Must stay constructible: user code builds stabilizer/statevec models with Lindblad ops for error-generator propagation. **Only** their `Lterms`/`Lterm_coeffs` attributes are removed (§3.3) |
| `create_lindblad_term_superoperators`, `combined_lindblad_term_superops`, `_compute_term_generators` (`lindbladcoefficients.py`/`lindbladerrorgen.py`), `tools/lindbladtools.py:create_lindbladian_term_errorgen`, Lindblad coefficient weights | "term" here means a Lindblad generator term; used by dense simulation |
| `objectivefns.TermWeighted` | Least-squares objective-term weighting; unrelated |
| Optimizer `oob_check_interval` / `oob_check_mode` / `oob_action` (`optimize/customlm.py`, `simplerlm.py`) | Used by gauge optimization (`algorithms/gaugeopt.py`) |
| `circuits/cloudcircuitconstruction.py`, including `_FirstOrderProbabilityJacobian` and `_linearized_lindblad_parameterization` | Rewritten in 0.11 to **not** use the term simulator. Keep it and its tests (`test_nqubit.py` sequence-selection tests, `test_nqnoiseconstruction.py::CloudNoiseFirstOrderJacobianTester`) |
| `tools/legacytools.py`: `deprecate(..., message=, suppress_nested=)`, `warn_deprecated(..., message=, stacklevel=)` and their tests | General deprecation tooling added alongside this work. Only the `TERM_SIM_DEPRECATION_MESSAGE` constant goes |
| `Evotype.supports()` / `supported_reptypes()` | Generic. Only the `'term'`/`'direct term'` dict entries go |

## 2. Recommended order of work

Delete from the top of the dependency stack down, so that every intermediate state still imports:

1. Tests and reference files (§7). Removing them first means failing tests can't mislead you.
2. Callers outside the term modules: `algorithms/core.py`, `objectivefns.py`, modelpacks, package `__init__` re-exports (§4).
3. The simulator, its kernels and its layout (§3.1).
4. ModelMember term methods, helpers, caches and the `term.py` module (§3.2).
5. Evotype term reps, the factories, and the `Lterms` attributes (§3.3).
6. Miscellaneous (§3.4), build config (§5), docs (§6), agent-docs/CHANGELOG (§8).
7. Rebuild extensions, **delete stale compiled files** (§5), run the verification (§10).

## 3. Code to delete or edit

### 3.1 Simulator, kernels, layout — delete whole files

| File (git-tracked) | Notes |
|---|---|
| `pygsti/forwardsims/termforwardsim.py` | `TermForwardSimulator`, `_TermPathSetBase`, `_AtomicTermPathSet`, `TermPathSet`, `_ignore_term_deprecation_warnings` |
| `pygsti/forwardsims/termforwardsim_calc_generic.py` | |
| `pygsti/forwardsims/termforwardsim_calc_statevec.pyx` | Cython extension (see §5) |
| `pygsti/forwardsims/termforwardsim_calc_stabilizer.pyx` | Cython extension (see §5) |
| `pygsti/layouts/termlayout.py` | `TermCOPALayout`, `_TermCOPALayoutAtom` |

### 3.2 ModelMembers

**Delete the whole module** `pygsti/modelmembers/term.py` (all `RankOne*Term` classes, `compose_terms*`, `exponentiate_terms*`, and the `TERM_SIM_DEPRECATION_MESSAGE` re-export). Then remove `term as _term` / `import term as _term` from these 17 files:
`operations/{composedop, experrorgenop, identitypluserrorgenop, lindbladcoefficients, staticarbitraryop, staticcliffordop, staticunitaryop, stochasticop}.py`, `povms/{composedeffect, computationaleffect, conjugatedeffect, tensorprodeffect}.py`, `states/{composedstate, computationalstate, purestate, staticpurestate, tensorprodstate}.py` (all under `pygsti/modelmembers/`).

**Delete the decorated public term methods/properties.** They carry `@_deprecate(message=_TERM_SIM_DEPRECATION_MESSAGE, suppress_nested=True)`; for properties the decorator sits under `@property`, so delete the `@property` line too. 60 definitions in 24 files:

| Member | `taylor_order_terms` | `taylor_order_terms_above_mag` | `highmagnitude_terms` | `total_term_magnitude` (prop) | `total_term_magnitude_deriv` (prop) |
|---|---|---|---|---|---|
| `operations/linearop.py` `LinearOperator` (base) | ✓ | ✓ | ✓ | | |
| `operations/composedop.py` `ComposedOp` | ✓ | ✓ | | ✓ | ✓ |
| `operations/embeddedop.py` `EmbeddedOp` | ✓ | ✓ | | ✓ | ✓ |
| `operations/experrorgenop.py` `ExpErrorgenOp` | ✓ | ✓ | | ✓ | ✓ |
| `operations/identitypluserrorgenop.py` `IdentityPlusErrorgenOp` | ✓ | ✓ | | ✓ | ✓ |
| `operations/lindbladerrorgen.py` `LindbladErrorgen` | ✓ | | | ✓ | ✓ |
| `operations/composederrorgen.py` `ComposedErrorgen` | ✓ | | | ✓ | ✓ |
| `operations/staticarbitraryop.py`, `staticcliffordop.py`, `staticunitaryop.py` | ✓ | | | ✓ | ✓ |
| `operations/stochasticop.py` `StochasticNoiseOp` | ✓ | | | ✓ | ✓ |
| `operations/depolarizeop.py` `DepolarizeOp` | | | | | ✓ |
| `states/state.py` `State` (base) | ✓ | ✓ | ✓ | | |
| `states/composedstate.py` `ComposedState` | ✓ | ✓ | | ✓ | ✓ |
| `states/{computationalstate, purestate, staticpurestate, tensorprodstate}.py` | ✓ | | | | |
| `povms/effect.py` `POVMEffect` (base) | ✓ | ✓ | ✓ | | |
| `povms/composedeffect.py` `ComposedEffect` | ✓ | ✓ | | ✓ | ✓ |
| `povms/{computationaleffect, conjugatedeffect, tensorprodeffect}.py` | ✓ | | | | |

Plus `LindbladCoefficientBlock.create_lindblad_term_objects` in `operations/lindbladcoefficients.py`, also decorated.

**Delete the private term helpers** (each has a `.. deprecated:: 0.11` note):
- `_compute_taylor_order_terms` in `ComposedOp`, `ExpErrorgenOp`, `IdentityPlusErrorgenOp`;
- `LindbladErrorgen._init_terms`;
- `lindbladcoefficients.py`: all 4 `_create_lindblad_term_objects_impl` (ham, other-diagonal, other, other-unconstrained blocks) and all 8 `_coefficient_polynomial` definitions (`_BlockParameterization` base plus `_StaticParam`, `_RealVectorElements`, `_DiagCholesky`, `_Depol`, `_RelDepol`, `_OtherElements`, `_OtherCholesky`). Check whether any nested helpers inside these impls (e.g. `O_terms`, `coeff_poly`) become orphaned;
- `StochasticNoiseOp._get_rate_poly_dicts` (`stochasticop.py`) and its override `DepolarizeOp._get_rate_poly_dicts` (`depolarizeop.py`).

**Delete the term caches.** Each site has the comment `# term caches: used only by the deprecated TermForwardSimulator machinery ...`. Delete the comment and the assignments of `self.terms`, `self.local_term_poly_coeffs` and `self.exp_terms_cache` in `__init__` **and** `set_gpindices` of `ComposedOp`, `ExpErrorgenOp`, `IdentityPlusErrorgenOp`, `ComposedEffect`, `ComposedState`, plus the orphaned `self.terms = {}` in `ComposedPOVM.set_gpindices` (`povms/composedpovm.py`). If a `set_gpindices` override then only calls the base class, consider deleting the override.
  - **Check serialization:** grep for `'terms'`, `local_term_poly_coeffs` and `exp_terms_cache` in `__getstate__`/`__setstate__`/`to_memoized_dict` of those classes, so old pickles don't trip on missing attributes.

**Clean up imports that become unused** (run `flake8 --select=F401` on every edited file):
- `_Polynomial` imports in the modelmember files;
- `pygsti.baseobjs.opcalc` imports in `lindbladerrorgen.py` (`_compact_deriv`, `_abs_sum_bulk_eval_compact_polynomials_complex`, `_bulk_eval_compact_polynomials_complex`) and in `linearop.py`, `states/state.py` and `povms/effect.py` (`_bulk_eval_compact_polynomials_complex`). The *functions* stay in opcalc (§1); only these imports go;
- `listtools as _lt` imports in `composedop.py`, `tensorprodeffect.py`, `tensorprodstate.py` if no longer used;
- `from pygsti.tools.legacytools import deprecate as _deprecate` and `... TERM_SIM_DEPRECATION_MESSAGE as _TERM_SIM_DEPRECATION_MESSAGE` from all 24 modelmember files.

### 3.3 Evotypes

| File / symbol | Action |
|---|---|
| `evotypes/statevec/termreps.pyx`, `termreps.pxd`, `termcreps.cpp`, `termcreps.h` | Delete (git-tracked sources) |
| `evotypes/stabilizer/termreps.pyx`, `termreps.pxd`, `termcreps.cpp`, `termcreps.h` | Delete |
| `evotypes/statevec/__init__.py`, `evotypes/stabilizer/__init__.py` | Remove `from .termreps import *  # term-sim machinery ...` |
| `evotypes/basereps.py` | Delete the fallback `class TermRep` (in the `except ImportError` branch), `class StockTermRep`, and `StockTermDirectRep = StockTermRep`. Remove `TermRep` from `from .basereps_cython import OpRep, StateRep, EffectRep, TermRep, PolynomialRep` |
| `evotypes/basereps_cython.pyx` / `.pxd` | Delete `cdef class TermRep` in both. Keep everything else (`PolynomialRep` etc.) |
| `evotypes/evotype.py` | Delete `create_term_rep`, `create_direct_term_rep`, `conjugate_state_term_rep`, the `# TERM REPS` comment, and the `'term'` / `'direct term'` entries (plus their comment) in `Evotype._reptype_to_attrs` |
| `evotypes/{statevec,stabilizer}/opreps.pyx`, `evotypes/{statevec,stabilizer}_slow/opreps.py` | In `OpRepLindbladErrorgen`, remove the `Lterms` / `Lterm_coeffs` attributes (`cdef public object ...` declarations and their initialisation). **Keep both classes.** Update their "not deprecated" docstring paragraph so it no longer mentions term attributes or the term simulator, e.g. "cannot act on states; exists so Lindblad-errorgen ops can be constructed with this evotype (e.g. for error generator propagation)" |

Grep the remaining Cython sources (`termforwardsim_calc_*.pyx` will already be gone) for `cimport` of anything `term*` to make sure nothing else depends on these `.pxd` files.

### 3.4 Other places

| File / symbol | Action |
|---|---|
| `pygsti/forwardsims/__init__.py` | Remove `from .termforwardsim import TermForwardSimulator  # term-sim machinery ...` |
| `pygsti/algorithms/core.py`, `run_gst_fit` | Replace the `_TermFSim` import and `if isinstance(objective.model.sim, _TermFSim): ... else:` dispatch with a plain `opt_result = _do_runopt(objective, optimizer, printer)` |
| `pygsti/algorithms/core.py`, `_do_term_runopt` | Delete the function |
| `pygsti/objectivefns/objectivefns.py`, `TimeIndependentMDCObjectiveFunction.terms` | Delete the `if oob_check:  # Only used for (deprecated) TermForwardSimulator ...` block, which calls `bulk_test_if_paths_are_sufficient`. **Keep the `oob_check` parameter** in the signature: the optimizers pass it generically. Optionally document it as a no-op for this objective, or follow up separately |
| `pygsti/modelpacks/_modelpack.py` | Delete `ModelPack._get_cachefile_names`, the module-level `_load_calccache`, and the import `bulk_load_compact_polynomials as _bulk_load_compact_polys` (the function itself stays, §1). Check that `_gzip`/`_pickle`/`_Path` imports are still used |
| `pygsti/modelpacks/stdtarget.py` | Delete `_get_cachefile_names`, `_write_calccache`, `_load_calccache`, the commented-out `"H+S terms"` blocks in `_copy_target`, and the `_polynomial` import if unused. `_copy_target` itself stays |
| `pygsti/baseobjs/opcalc/fastopcalc.pyx` | Delete `def prs_as_polynomials` and its `cdef` helpers `pr_as_poly_innerloop` and `extract_term_coeffs`. (`opcalc/__init__.py` re-exports via `from .fastopcalc import *`, so no export list needs editing; `slowopcalc.py` has no counterpart.) Keep everything listed in §1 |
| `pygsti/tools/legacytools.py` | Delete `TERM_SIM_DEPRECATION_MESSAGE` and its comment block. Keep `deprecate` / `warn_deprecated` |

## 4. Callers that should already be gone

After §3, this must return nothing outside the files you deleted:

```bash
grep -rn -E "TermForwardSimulator|termforwardsim|TermCOPALayout|termlayout|taylor_order_terms|highmagnitude_terms|total_term_magnitude|create_lindblad_term_objects|_coefficient_polynomial|RankOne\w*Term|modelmembers\.term|term as _term|TermRep\b|TermDirectRep|StockTerm|create_term_rep|create_direct_term_rep|conjugate_state_term_rep|Lterms|Lterm_coeffs|_do_term_runopt|bulk_test_if_paths_are_sufficient|prs_as_polynomials|TERM_SIM_DEPRECATION_MESSAGE|_get_rate_poly_dicts|local_term_poly_coeffs|exp_terms_cache" pygsti test docs/markdown docs/api.rst setup.py
```

Expected leftovers you may ignore: none in `pygsti/`. In `CHANGELOG`, the historical 0.11 entries stay.

## 5. Build configuration and compiled artifacts

`setup.py`:
- remove the four `Extension(...)` entries `pygsti.evotypes.statevec.termreps`, `pygsti.evotypes.stabilizer.termreps`, `pygsti.forwardsims.termforwardsim_calc_statevec`, `pygsti.forwardsims.termforwardsim_calc_stabilizer`;
- remove from `package_data` the `termreps.pxd`, `termreps.pyx`, `termcreps.cpp`, `termcreps.h` entries under `pygsti.evotypes.statevec` and `pygsti.evotypes.stabilizer`, and `termforwardsim_calc_statevec.pyx`, `termforwardsim_calc_stabilizer.pyx` under `pygsti.forwardsims`.

Repository config:
- `.github/CODEOWNERS`: remove `pygsti/forwardsims/termforwardsim* ...`.
- `.flake8-critical`: remove `pygsti/forwardsims/termforwardsim_calc_generic.py:F824`.

**Stale compiled files (important).** Editable/in-place builds leave generated files next to the sources: `*.cpython-*.so` and Cython-generated `*.cpp` (e.g. `pygsti/evotypes/statevec/termreps.cpp`, `pygsti/forwardsims/termforwardsim_calc_statevec.cpp`; these are **git-ignored** and different from the tracked hand-written `termcreps.cpp`). A leftover `termreps*.so` still imports and can mask a missed reference. Delete them before testing:

```bash
find pygsti -name "termreps*.so" -o -name "termforwardsim_calc_*.so" -o -name "termreps.cpp" -o -name "termforwardsim_calc_*.cpp" | xargs rm -f
python setup.py build_ext --inplace   # or: pip install -e .  (rebuilds basereps_cython, fastopcalc, opreps)
```

`basereps_cython`, `fastopcalc`, and the statevec/stabilizer `opreps` extensions **must** be rebuilt because their sources change.

> Note for whoever tests the no-Cython path: as of 0.11, `setup.py` checks the environment variable `PYGSTI_SKIP_CYTHON`, while CI and the error message use `PYGSTI_CYTHON_SKIP`. Use `PYGSTI_SKIP_CYTHON=1` to really skip extension builds (or check whether this has since been fixed).

## 6. User documentation

- `docs/api.rst`: remove the six entries `pygsti.forwardsims.termforwardsim`, `pygsti.forwardsims.termforwardsim_calc_generic`, `pygsti.forwardsims.termforwardsim_calc_stabilizer`, `pygsti.forwardsims.termforwardsim_calc_statevec`, `pygsti.layouts.termlayout`, `pygsti.modelmembers.term`.
- `docs/markdown/simulation/ForwardSimulationTypes.md`: delete the `` `TermForwardSimulator` (**deprecated**, ...) `` bullet and the ```` ```{warning} ```` admonition after the simulator list. These are the only `.md` sources; the paired `.ipynb` files are git-ignored and generated by `jupytext`.
- `docs/markdown/Simulation.md`: drop the parenthetical "(the term-based simulator is deprecated)" from the *Forward simulation types* entry.

## 7. Tests and reference files

| File | Remove |
|---|---|
| `test/unit/objects/test_termsim_deprecation.py` | **Whole file** |
| `test/test_packages/drivers/test_calcmethods1Q.py` | The 9 tests decorated `@ignore_termsim_deprecation`: `test_stdgst_terms`, `test_stdgst_prunedpath`, `test_reducedmod_svterm`, `test_reducedmod_svterm_errorgens`, `test_reducedmod_prunedpath_svterm_errorgens`, `test_reducedmod_cterm`, `test_reducedmod_cterm_errorgens`, `test_circuitsim_svterm`, `test_circuitsim_cterm`; then the `ignore_termsim_deprecation` marker definition and its comment |
| `test/test_packages/drivers/test_nqubit.py` | `test_2Q_terms` and the marker definition. **Keep** `test_sequential_sequenceselection` / `test_greedy_sequenceselection` (cloud-noise, not term-based any more) |
| `test/unit/modelmembers/test_lindbladcoefficients.py` | `test_create_lindblad_term_objects_snapshot_1q`, `test_term_objects_reconstruct_dense_errorgen_1q`, `test_term_objects_reconstruct_dense_errorgen_other_elements_evotypes` (marked), **and** `test_coefficient_polynomial_contract` (unmarked; tests the private `_coefficient_polynomial`); helpers and data used only by them (as of 0.11: `_TERM_SNAPSHOT_1Q`, `_TERM_SNAPSHOT_CONFIGS`, `_term_eval_point`, `_sorted_coeffs`, `_evaluated_term_coeffs`, `_std_superop_from_terms`, `_dense_std_superop_from_block`, `_setup_1q_block_with_deterministic_data`, and the case data feeding `test_coefficient_polynomial_contract`; grep each before deleting); the marker definition. Update the module docstring if it advertises term coverage |
| `test/unit/modelmembers/test_other_unconstrained_coefficients.py` | `test_term_objects_reconstruct_dense_errorgen_1q`, its helper `_std_superop_from_terms`, the marker definition (keep `_errorgen_dense`, which other tests use) |
| `test/unit/objects/test_cython_build.py` | The `pygsti.evotypes.statevec.termreps` line in `test_statevec_modules_exist`, the `pygsti.evotypes.stabilizer.termreps` line in `test_stabilizer_modules_exist`, and the whole `test_termforwardsim_calc_modules_exist` |
| `test/unit/tools/test_legacytools.py` | `test_term_sim_deprecation_message` only. Keep the other legacytools tests |
| `test/test_packages/extras/test_idt.py` (quarantined, module-level skip; idle tomography is known-broken, see [known-debt #13](known-debt.md#13-extrasidletomography-is-broken)) | `test_idletomography_1Q` and `test_idletomography_2Q` construct `TermForwardSimulator`. Either delete them or switch them to a non-term simulator. Coordinate with whoever owns IDT; don't try to fix IDT itself |
| `test/test_packages/cmp_chk_files/` | Delete `test1Qcalc_std_terms.json`, `test1Qcalc_std_prunedpath.json`, `test1Qcalc_redmod_terms.json` (used only by the removed tests) and `nqubit_2Qterms.cache` (already unreferenced) |

Use `grep -rn "ignore_termsim_deprecation" test` to confirm nothing marked remains.

## 8. agent-docs and CHANGELOG

Read [META.md](META.md) first.

- **`known-debt.md` #19**: mark it **RESOLVED in place**. Don't delete or renumber it; other docs anchor on the number. Use the template in META.md §3, e.g. `## 19. RESOLVED — TermForwardSimulator deprecation (expired in 0.12)`, with a short **Status** paragraph. **Changing the heading text changes the anchor** (`#19-termforwardsimulator-deprecation-expires-in-012`), so update every inbound link (grep for `19-termforwardsimulator`) or keep the anchor-forming text identical. Run the link checker.
- **This guide**: delete `agent-docs/termsim-expiry-guide.md` and remove its row from the *Layout of `agent-docs/`* table in the repo-root [AGENTS.md](../AGENTS.md) in the same commit (META.md Level 4). Remove the link to it from known-debt #19.
- **`02-forward-simulation.md`**: remove `termlayout.py` from the layouts sentence; the "(e.g., TermForwardSimulator)" aside about non-identical probabilities; the `TermForwardSimulator` rows in the simulator and architecture tables; the two `termforwardsim_calc_*.pyx` entries in the Cython-kernels row; `termlayout.py` in the simulator-specific-layouts row; "Term" in the Mermaid `forwardsims` node; and the "Forward-simulator swap can change which fit-loop wrapper runs" pitfall bullet (it only exists because of `_do_term_runopt`).
- **`03-data-and-fitting.md`**: delete the `_do_term_runopt` row of the fitting table, and check for other mentions of path sets.
- **`CHANGELOG`**: under the 0.12 section add `### Expired Deprecations` (the heading style used in earlier releases), e.g. "* Removed `TermForwardSimulator` and the term-based (path-integral) machinery deprecated in 0.11: ... (#916)". List the removed public names: `TermForwardSimulator`, `TermCOPALayout`, `pygsti.modelmembers.term`, the ModelMember `taylor_order_terms`/`taylor_order_terms_above_mag`/`highmagnitude_terms`/`total_term_magnitude`/`total_term_magnitude_deriv`, `LindbladCoefficientBlock.create_lindblad_term_objects`, the statevec/stabilizer term reps, the `Evotype` term-rep factories, `opcalc.prs_as_polynomials`.

## 9. Final audit greps

```bash
grep -rn "916" pygsti test docs/markdown setup.py .github .flake8-critical   # only unrelated hits should remain
grep -rn "deprecated:: 0.11" pygsti | grep -i -E "term|916"                  # expect nothing
grep -rn -i "termforwardsim\|termlayout\|termreps\|termcreps" .  --include=*.py --include=*.pyx --include=*.pxd --include=*.rst --include=*.md --include=*.txt --include=*.toml --include=*.cfg
python3 agent-docs/check-links.py
```

The "unrelated hits" for `916` should be none. If an unrelated issue #916 reference shows up, it is not this deprecation.

## 10. Verification

```bash
# Environment note: on many-core boxes cap BLAS threads, or pytest-xdist workers can exhaust thread limits
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python setup.py build_ext --inplace                         # after deleting stale artifacts (§5)
python -c "import pygsti, pygsti.forwardsims, pygsti.evotypes.statevec, pygsti.evotypes.stabilizer"
pytest -n auto --dist loadscope -m "not slow" test/unit
pytest -n auto --dist loadscope test/integration
pytest -n auto test/test_packages/drivers                   # includes the cloud-noise sequence-selection tests
flake8 --count --config .flake8-critical .
```

Also check that:
- **Error-generator-propagation-style models still build**: e.g. `pygsti.modelpacks.smq1Q_XY.target_model("H+S", evotype="statevec")` and a stabilizer-evotype cloud/crosstalk model construct without error. This exercises the kept `OpRepExpErrorgen`/`OpRepLindbladErrorgen` reps.
- **`create_cloudnoise_circuits` is unchanged**: the `test_nqubit.py` sequence-selection tests compare against stored reference files and must pass without regenerating them.
- **`Polynomial` / opcalc evaluators are intact**: `test/unit/objects/test_polynomial.py`, `test/unit/objects/test_opcalc.py` and `test/unit/tools/test_errgenpolytools.py` pass.
- **Old serialized models** (e.g. JSON files that recorded a `TermForwardSimulator` as `sim`) now fail to load. That's the intended breaking change, but mention it in the CHANGELOG. A model saved with a term simulator can't be loaded in 0.12 unless the loader falls back to a default simulator; decide whether that fallback is wanted.

## 11. Known term-simulator issues that disappear with the removal

Recorded so nobody "fixes" them first (see known-debt #19):
- the POVM-Hamiltonian sign error in first-order term polynomials when SPAM has parameters;
- `bulk_sopm_gaps_jacobian` raises `NotImplementedError`, and its atom-level path needs a layout field that is never populated.

# Producing the inputs: learn → refit → decorate

The report skill starts from a DEM. This is how to get one out of shot data
with `pygsti.extras.sparsedem`, and the three places the process goes wrong.

Requires the `feature-dem-estimation` branch of pyGSTi.

## 0. Load the data

```python
import numpy as np, stim
from pygsti.extras.sparsedem import io as sdio

circuit = stim.Circuit.from_file(f"{D}/circuit_noisy_si1000.stim")
det = np.asarray(stim.read_shot_data_file(
    path=f"{D}/detection_events.b8", format="b8",
    num_detectors=circuit.num_detectors), dtype=np.uint8)
obs = np.asarray(stim.read_shot_data_file(
    path=f"{D}/obs_flips_actual.b8", format="b8", num_observables=1),
    dtype=np.uint8)[:, 0]
```

Events are integer bitmasks over detector indices throughout — `1 << d`. The
fit consumes syndrome *counts*, not raw shots:

```python
from pygsti.extras.sparsedem import utils as sdutils
syndrome_counts = dict(sdutils.counts_from_samples(det))
```

## 1. Learn the support (lattice pruning)

`bitmask_trie_search` walks the subset lattice, calling a predicate on each
candidate detector set and pruning branches whose statistics are consistent
with no error. Set the confidence from the number of masks you expect to test,
not by intuition: at confidence `1 - 1e-7` (z ≈ 5.2) with ~350k masks tested
you expect ~0.04 false discoveries.

```python
from pygsti.extras.sparsedem.lattice import (bitmask_trie_search,
                                             make_fast_event_check)
check = make_fast_event_check(syndrome_counts, alpha=1e-7)
masks = bitmask_trie_search(n_det, check)
```

Wrap `check` in a counter that logs every 60 s — the search gives no progress
output of its own and you want to know both that it is alive and how many
masks it tested (that count sets your false-discovery expectation).
`lattice_pruning_dem_estimation` in the same module does search + fit in one
call, but the two-stage form is what you want here, because of step 2.

Cost scales with the number of surviving branches, not with `2**n_det`. Log the
weight histogram of the result — a healthy surface-code run gives a pyramid
peaking at weight 2.

## 2. Refit with backward elimination

**This is the step people skip, and it is not optional.** The trie search is a
*discovery* procedure: it returns every mask whose statistic is significant,
including high-weight masks that alias combinations of true lower-weight
events. Feed that set to a single unconstrained joint fit and a few hundred
events come back with negative probabilities — the set is over-complete.

Fix: drop the non-positive events and refit the survivors, repeating to
convergence.

```python
from pygsti.extras.sparsedem.estimation import (fit_specified_dem,
                                                default_polarization_masks)
ATOL = 1e-4
for _ in range(40):
    pol = default_polarization_masks(masks, n_det)
    dem_masks, probs = fit_specified_dem(syndrome_counts, masks,
                                         return_probs=True, pol_masks=pol)
    keep = probs > ATOL
    if keep.all():
        break
    masks = sorted(int(m) for m, k in zip(dem_masks, keep) if k)
# final pass, this time paying for the covariance
pol = default_polarization_masks(masks, n_det)
dem, dem_masks, probs, cov = fit_specified_dem(
    syndrome_counts, masks, atol=ATOL, return_covariance=True, pol_masks=pol)
```

Two practical notes:

- **Do not compute the covariance inside the loop.** It is
  O(n_syndromes × n_events²) and dominates the runtime — on a 2300-event
  problem it turned 25 s iterations into 85 s ones. Compute it once at the end.
- Write the refit outputs to different filenames than the learn outputs
  (`learned_dem_refit.dem`, …) so a failed refit does not destroy hours of
  search.

Sanity check the result: `total_error_mass × mean_event_weight` should equal
the observed mean clicks per shot. If it does not, the fit did not converge.

## 3. Decorate with logical flips

The learned DEM knows which detectors fire together but nothing about the
logical observable. `assign_logical_flags` recovers that from the data by
solving `Y = B·L` over GF(2) — decode each shot, compare the residual to the
measured observable.

```python
from pygsti.extras.sparsedem.logical_decoration import assign_logical_flags
decorated, flags, residual, diag = assign_logical_flags(
    dem, det, obs, num_detectors=n_det,
    decoder="pymatching", on_nongraphlike="drop",
    residual_threshold=0.20,      # ~ the expected decoder logical error rate
    max_ransac_iterations=4, seed=0,
    row_order="confidence", refine=True)
```

- `on_nongraphlike="drop"` is required for pymatching: weight-3/4 events are
  not graph-like and cannot be decoded. Their flags stay undetermined. **This
  is a much bigger loss than it sounds**, because the decoder decides which
  columns the GF(2) solve even sees: on the Willow d=5 run it left 1240 of 2305
  flags undetermined and gave the solve 1065 columns instead of 2305.
- **`decoder="tesseract"` is worth the runtime for the decoration**, even if
  you go on to decode with a matcher. It handles hyperedges, so no events are
  dropped and no `on_nongraphlike` argument is needed. On the same Willow run
  it took the solve to rank 2304/2305 with 1 undetermined flag, cut the
  inconsistent-row fraction from 0.1245 to 0.0830, and raised the circuit-flag
  cross-check from 915/915 to 1595/1595 — the extra 680 checks being precisely
  the weight-3/4 events pymatching could not reach. It costs ~5 ms/shot
  (~4 minutes for 50k shots) against pymatching's seconds.
- `row_order="confidence"` orders the elimination by how well-determined each
  row is; `refine=True` follows the solve with greedy bit-flips against the
  residual.
- `residual_threshold` is the residual below which the solve is accepted
  without RANSAC. Set it near the logical error rate you expect: too tight and
  you burn RANSAC iterations chasing noise that the decoder itself produces.
- Check `diag["rank"]` against `diag["n_events"]`, and `diag["converged"]`.
  Full rank with `ransac_iterations == 0` means the solve was unique; anything
  else means the flags are partly a guess.

**Free cross-check:** if you have a circuit-level DEM, every learned event
whose detector mask also appears in it should get the same logical flag.
Agreement is the strongest evidence the decoration is right, and it costs
nothing — the solver never sees the circuit DEM.

## 4. Hand off to the report

```
detection_events.b8 ──┐                              ┌─> report.html
obs_flips_actual.b8 ──┤                              │
circuit.stim ─────────┼──> scripts/run_report.py ────┼─> report_brief.md ──┐
learned_dem_refit.dem ┤                              │                     │
decorated_dem.dem ────┘                              └─> report_state.pkl  │
                                                            │              │
                                        Fable subagent <────┼──────────────┘
                                              │             │
                                        commentary.json ────┴─> annotate_report.py
                                                                       └─> report.html
```

Keep `learned_events_refit.npz` (masks/probs/stderr) — pass it as `--events`
and the probability figure gets error bars.

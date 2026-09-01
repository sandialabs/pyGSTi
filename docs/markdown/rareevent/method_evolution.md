# Method evolution: how each estimator got here, and why

Every method in `pygsti.extras.rareevent` exists because a predecessor failed in
a specific, diagnosed way. This note tells that story — organized by *problem*,
not by file — so that when you reach for a method you know what pathology it
fixes, what it inherits, and what it still leaves open. It is the companion to:

- [`methods.md`](methods.md) — the mathematics of every method, in full.
- [`malignant_set_counting.md`](malignant_set_counting.md) — the counting
  deep-dive (methods 1–4 and their cross-validation).
- the standalone repo's `benchmarks/REPORT.md` — every measured number
  cited below.

Three lineages run through the package: **splitting** (sampling the
conditional-failure distribution down a $p$ schedule), the **failure
spectrum** (weight-resolved failure fractions plus a transform), and
**counting/peeling** (the low-weight malignant structure both of the others
need). The July 2026 additions adapt four techniques from Mullan, Weippert &
Brown, *"Improved Methods for Determining Quantum Error Correcting Code
Performance and Fault Tolerance"* (arXiv:2607.27153) — the subregion proposal,
its core-resampling region-rate heuristic, $\hat R$-driven level stopping, and
random-subset peeling — plus the Bennett acceptance-ratio level estimator
(Bennett 1976) used in the same spirit. **Every pre-existing implementation is
kept byte-identical as a frozen baseline** so the comparisons stay honest; see
the summary table in §5.

## 1. The shared problem

All three lineages fight the same enemy. At low physical rate $p$, logical
failures are carried by *near-minimal malignant sets* — fault configurations
of weight near the onset $w_{\min} = \lceil d/2 \rceil$ that make the decoder
mispredict — and these are astronomically rare under the sampling
distribution. Direct Monte Carlo costs $\sim 100/P_{\mathrm{fail}}$ decodes;
at $P_{\mathrm{fail}} \sim 10^{-12}$ that is the wall. Every generation below
is a different way of steering computation toward those rare light sets
without biasing the answer.

## 2. The splitting lineage

### 2.1 Generation 0 — uniform toggles (`rare_event.RareEventSplittingEstimator`)

**How it works.** Bravyi–Vargo splitting: telescope
$P_{\mathrm{fail}}(p_L) = P_{\mathrm{fail}}(p_0)\prod_k
P_{\mathrm{fail}}(p_{k+1})/P_{\mathrm{fail}}(p_k)$ over a descending schedule
`p_scales`; anchor $p_0$ by direct MC; estimate each ratio as
$\mathbb{E}_{\pi_k}[w_k(E)]$ over MCMC samples of the conditional-failure
distribution $\pi_k$ ([`methods.md`](methods.md) §4.1). The chain
(`ConditionalFailureMCMC`) proposes toggling one mechanism chosen uniformly
from all $n$; the Metropolis ratio is the odds factor
$\mathrm{odds}_i^{\pm 1}$, and failure conditioning rejects any move whose
result does not fail. The oracle is consulted only after the cheap ratio test
("ratio first, decode second").

**The failure mode.** Freeze-out at $d \ge 7$. Sitting on a near-minimal
malignant set, every *remove* breaks failure (rejected by conditioning) and
every *add* passes with probability $\sim q_i \ll 1$. Worse, a uniform
proposal touches a mechanism *relevant* to the current set only with
probability $|E|/n$, and $n$ grows to ~48k at $d = 11$ — the chain spends
almost every step proposing irrelevant far-away toggles. It explores a single
malignant basin; the level ratios come out biased low, and the bias compounds
multiplicatively across levels (the standalone repo's `benchmarks/REPORT.md`
diagnostics section).

**Status.** Frozen legacy baseline; do not use at $d \ge 7$.

*Two prototyped fixes from the same diagnosis are retained for reference:
`splitting_swap.py` (Variant B, a detector-adjacent swap move — its
weight-preserving kernel found its real home in gap-splitting, §3) and
`splitting_smc.py` (Variant C, a population/SMC architecture around the
unchanged toggle kernel). The lineage below is the line that became the
defaults.*

### 2.2 Generation 1 — local toggles (`splitting_local.LocalSplittingEstimator`)

**What changed.** Propose the toggled mechanism from a mixture of the uniform
distribution ($\beta = 0.1$, for irreducibility) and the *detector
neighborhood* $N(E)$ of the current set, with the exact Metropolis–Hastings
correction $q(i \mid E')/q(i \mid E)$ for the state-dependent proposal
([`methods.md`](methods.md) §4.2). This raises the relevant-touch rate from
$|E|/n$ to $O(1)$. A second repair in the same module: **external low
anchoring** (`anchor_failure_rate`/`anchor_state`) — since per-level bias
compounds and the biased levels are the low-$p$ ones, anchor at the lowest
$p_0$ catalog MC can still measure and walk only the last few levels.

**What it fixed.** The wasted-proposal problem. The chain now actually moves
within a basin, and kernel exactness is unit-tested against brute-force
enumeration of the stationary distribution.

**What remained.** It is still a *single-toggle* chain: add moves are still
$q$-suppressed (only their targeting improved), and hopping between distinct
malignant basins still requires passing through non-failing intermediate
states, which the conditioning indicator rejects. A single chain started from
an anchor-typical (heavy) state can therefore never reach the light,
near-onset basins that dominate the low-$p$ ratios.

**Status.** Frozen baseline for the subregion comparison; still the fallback
building block when the gap construction does not apply.

### 2.3 Generation 2 — gap-seeded multi-chain levels (`pipelines.gap_seeded_splitting_estimate`)

**What changed.** Attack basin coverage from the *seeding* side. Fixed-weight
gap-splitting (§3.2) visits exactly the light failing states the local chain
cannot reach, and harvests them (`harvest_states=`). Each level then runs
`num_chains` chains — chain 0 from the anchor, the rest from harvested
near-onset seeds, valid at every level because failure is $p$-independent —
and pools all samples for the ratio. A cross-chain split-$\hat R$ on the
log-weight-ratio series is reported per level: $\hat R \gg 1$ means chains
sit in basins with different ratio statistics, i.e. a single chain was
missing mass.

**What it fixed.** Light-basin coverage, and it made the missing-mass problem
*visible* (at $d = 5$ the cross-chain $\hat R$ of 1.1–1.6 directly confirmed
multi-basin structure; see
[`malignant_set_counting.md`](malignant_set_counting.md) §7).

**What remained.** Three separate inefficiencies, all downstream of the fact
that the kernel and the level machinery were untouched:

1. *The kernel still cannot jump.* Seeding places chains in several basins,
   but within a level each chain is still confined to the basin it started
   in; coverage is only as good as the harvest.
2. *Fixed step budgets.* Every level gets the same step count, so easy
   (high-$p$) levels waste decodes and hard levels may underspend, with no
   principled stopping rule.
3. *One-sided level ratios.* Each ratio $Z_{k+1}/Z_k$ is estimated from
   level-$k$ samples only, even though the descent produces samples at level
   $k+1$ anyway.

**Status.** Recommended default splitting pipeline (unchanged, frozen as the
baseline the subregion stack is benchmarked against).

### 2.4 Generation 3 — the subregion kernel (`splitting_subregion`, arXiv:2607.27153)

**What changed.** Attack the basin problem from the *kernel* side.
`SubregionConditionalFailureMCMC` replaces the single toggle with a partial
resample: each step draws a region $R$ (every index included independently
with probability `region_rate`, independent of the current state), keeps the
configuration outside $R$, and redraws every coordinate inside $R$ from a
resample distribution $f$. The Hastings ratio collapses to
$\prod_{i \in R:\, x_i' \ne x_i} [\mathrm{odds}_q(i)/\mathrm{odds}_f(i)]^{x_i'-x_i}$,
and with the default $f = q$ it is *identically 1*: the proposal is
rejection-free with respect to $\pi$, and acceptance reduces to the failure
indicator alone — one decode per proposing step, never a $q$-suppressed ratio
test (full math in [`methods.md`](methods.md) §4.5). The paper's
core-resampling heuristic sets `region_rate` $= 1/w_{\min}$
(`default_region_rate`), so each of the $\sim w_{\min}$ core errors of the
current failing set is hit with probability $1/w_{\min}$ — the proposal
replaces about one core error per step, a basin-to-basin jump in a single
move. Proposals that change nothing are counted as accepted no-ops *without*
an oracle call (`KernelCounters.noop_steps`), so acceptance diagnostics refer
to real moves and decodes are accounted exactly.

`pipelines.gap_seeded_subregion_estimate` / `GapSeededSubregionEstimator`
wraps this in the identical two-stage flow as Generation 2 (gap-harvest light
seeds, multi-chain descent), so seeding-side and kernel-side coverage
compose.

**What it fixed.** Inefficiency 1 of §2.3: $q$-suppressed growth and the
inability to cross non-failing valleys between basins.

### 2.5 Generation 3a — $\hat R$-driven level stopping (`stop_rhat`)

**What changed.** Following the paper's practice, levels run their chains in
blocks (`block_steps`) instead of a fixed budget; after each block (past
`min_steps_per_chain`) the cross-chain split-$\hat R$ of the log
weight-ratio series — first half of each chain discarded as burn-in — is
compared to the threshold, and the level stops as soon as it is met or a hard
cap (`max_steps_per_chain`) is reached. The level estimate then uses exactly
the second-half series the diagnostic certified; hard levels that hit the cap
are flagged (`SubregionLevelDiagnostics.rhat_threshold_met`). Requires
`num_chains >= 2` (the diagnostic is cross-chain).

**What it fixed.** Inefficiency 2: fixed budgets that waste steps on easy
levels and give hard levels no more than everyone else.

### 2.6 Generation 3b — Bennett acceptance ratio level estimator (`ratio_estimator="bar"`)

**What changed.** Each level ratio is estimated from *both* adjacent levels'
sample sets by solving the Bennett (1976) self-consistent equation
(`bennett_log_ratio`; the exact equation and its $\log(n_R/n_F)$ shift are in
[`methods.md`](methods.md) §4.7) — the minimum-variance choice in a wide
class of two-sided estimators. The reverse sample sets are free: the descent
already samples at every level, except the final $p$, where one extra
sampling run is performed. The one-sided estimate is always computed
alongside for comparison (`forward_log_ratio` in the diagnostics).

**What it fixed.** Inefficiency 3: variance thrown away by ignoring the
level-$(k+1)$ samples that already exist.

**Status of Generation 3 (all three toggles).** Candidate default; the
like-for-like campaign against the frozen Generation 1/2 baselines
(`benchmarks/runners/run_paper_methods_campaign.py`, measured verdict in
the standalone repo's `benchmarks/REPORT.md` §10) is in. The paper
reports 2–10× fewer decodes to reach $\hat R \le 1.05$, growing with
distance (arXiv:2607.27153 — the paper's numbers, on its own benchmarks);
our measurement realizes that gain as ≈10× tighter per-seed scatter at
equal cost at $d \ge 9$ (log-SD 0.34/0.35 → 0.05/0.03 at $d = 9/11$) for the
subregion kernel alone, and ≈9× fewer decodes at matched $\hat R \le 1.05$
with adaptive stopping on top. BAR agreed with the one-sided estimator
(mean per-level $|\Delta\log| = 0.06$ over 120 levels) without a measurable
variance win at this schedule, so it stays an off-by-default cross-check.
Promotion over `gap_seeded_splitting_estimate` as the package-wide default
is deliberately left as a separate decision; the baselines stay frozen.

## 3. The spectrum lineage

### 3.1 Generation 0 — rejection-sampled spectrum (`failure_spectrum.FailureSpectrumEstimator`, arXiv:2511.15177)

**How it works.** Write $P_{\mathrm{fail}}(p) = \sum_w f(w)\Pr[W=w; q(p)]$
with $f(w)$ the failure fraction at fixed weight and $\Pr[W=w]$ the
Poisson-binomial weight distribution; measure $f(w)$ by exact conditional
fixed-weight sampling (exponential tilting + rejection), fit the paper's
low-parameter ansatz with onset $w_0 = \lceil d/2\rceil$, transform
([`methods.md`](methods.md) §6).

**The failure mode.** The measurement floor: rejection sampling cannot see
$f(w) \lesssim 1/\text{max\_trials} \approx 5\times 10^{-5}$. At $d \ge 9$
the lowest weight with any counted failure sits far above $w_0$ (first
failures at $w = 17$ against $w_0 = 5$ at $d = 9$), yet the low-$p$
transform is dominated by weights near $w_0$ — the fit extrapolates blindly
across the gap and errs high, with error bars that underprice the
extrapolation.

### 3.2 Generation 1 — fixed-weight gap-splitting (`gap_splitting.GapSplittingEstimator`)

**What changed.** Measure $f(w)$ *below* the floor by subset simulation over
PyMatching's complementary gap $G(E) = w_{1-t}(E) - w_t(E)$, a continuous
"distance to failure" score; nested gap-level sets each resolve only a factor
`quantile`, so cost is linear in $-\log f(w)$ instead of $1/f(w)$, reaching
$f \sim 10^{-10}$ in minutes ([`methods.md`](methods.md) §7). The
rejuvenation kernel is the weight-preserving swap of `splitting_swap.py` —
Generation 0's Variant B, repurposed. Side effect: it harvests light failing
states, which became the seeds of splitting Generation 2 (§2.3).

**What remained.** Gap-splitting alone yields points, not a curve.

### 3.3 Generation 2 — gap-aux spectrum (`pipelines.gap_spectrum_estimate`)

**What changed.** Feed gap-splitting measurements at
$w_0, \dots, w_0 + \texttt{gap\_weight\_span}$ into the ansatz fit as
auxiliary log-space residuals with their own standard errors, pinning the
onset by data instead of extrapolation.

**What remained.** The fixed-ratio *transport* assumption (reusing
$f(w)$ measured at $p_{\mathrm{ref}}$ at other $p$), now the dominant
residual systematic — including its structural breakdown for
non-proportional schedules and the stratified repair, all quantified in
[`methods.md`](methods.md) §6.5 and REPORT.md §9.

**Status.** Recommended default spectrum pipeline.

## 4. The counting / peeling lineage

### 4.1 Generation 0 — brute force (`malignant.MalignantSetEstimator`)

Enumerate all $\binom{n}{v}$ subsets up to `max_weight`; exact lower bound on
$P_{\mathrm{fail}}$, tight as $p \to 0$; combinatorially dead beyond $d = 3$
(weight 3 at $d = 5$ is already $\sim 10^{10}$ oracle calls).

### 4.2 Generation 1 — the four counting routines

`connected_enumeration.py` (exact $m(v)$ census over connected clusters),
`knuth_counting.py` (unbiased tree-size estimation with error bars where the
census explodes), `gap_splitting.py` (direct sub-floor $f(w)$), and
`core_planting.py` (certified lower bound + coverage meter from harvested
cores). Their division of labor, cross-validation, and the
disconnected-cluster surprise are the subject of
[`malignant_set_counting.md`](malignant_set_counting.md).

### 4.3 Generation 2 — random-subset peeling (`core_planting.peel_to_minimal_subset`, arXiv:2607.27153 Alg. 1)

**The cost problem.** Core planting (and any core-jump proposal built on it)
needs harvested failing sets *peeled* to 1-minimal cores. The baseline peel
(`_peel_to_minimal`) removes one element at a time — one oracle call per
element per pass — so a heavy failing set of weight $W$ hiding a weight-$v$
core costs $\Theta(W)$ calls just for the first pass.

**What changed.** The paper's Algorithm 1: each round draws a subset size $s$
uniform on $[1, \max(1, \lfloor|E|/2\rfloor)]$ and a uniform random
$s$-subset $S$, tests $E \setminus S$ with a *single* oracle call, and
commits the removal iff the reduced set is nonempty and still fails. Large
draws strip many "fluff" mechanisms per call while the set is heavy; the cap
shrinks with the set, degrading gracefully to single-element moves near the
core. The subset phase exits after `max_rounds` total or `max_stall_rounds`
consecutive uncommitted rounds, then the baseline single-element polish pass
runs to *guarantee* 1-minimality of the result. Exposed as
`peel_method="subset"` on `harvest_cores`; the default `"single"` reproduces
the pre-existing behavior byte-identically (same results, same RNG stream).
`CountingOracle` wraps any `ForwardSimulator` to count `fails` calls for cost
accounting.

**Measured so far.** The unit test
(`tests/test_core_planting.py::test_fewer_oracle_calls_than_single_element_baseline`)
pins the subset peel at $\le 0.7\times$ the single-element baseline's
deterministic 127 calls on a synthetic weight-123 pattern hiding a weight-3
core; harvest-scale comparisons on the real pipelines are in
the standalone repo's `benchmarks/REPORT.md` §10: a clean crossover
at pattern weight $|E| \approx 30$–70 — the subset peel costs 1.9× the
single-element peel on light $d = 3$ patterns but 0.59–0.62× on the heavy
$d = 9/11$ ones (|E| ≈ 141–257), finding cores of the same weight.

## 5. Summary table

Status legend: **default** = recommended default pipeline; **candidate** =
measured best configuration in REPORT.md §10, promotion to default pending a
deliberate switch; **building block**
= use directly for its specific job; **baseline** = frozen for comparison
(byte-identical to its pre-adaptation state); **legacy** = superseded, kept
only for benchmarking; **prototype** = historical variant retained for
reference.

| Method | Module / class | Problem it fixed | Status |
|---|---|---|---|
| Uniform-toggle splitting | `rare_event.RareEventSplittingEstimator` | (original Bravyi–Vargo implementation) | legacy — freezes at $d \ge 7$ |
| Swap-kernel splitting (Variant B) | `splitting_swap.py` | freeze-out via same-weight swaps | prototype; kernel reused by gap-splitting |
| SMC splitting (Variant C) | `splitting_smc.py` | freeze-out via population resampling | prototype |
| Local-toggle splitting | `splitting_local.LocalSplittingEstimator` | $O(1)$ relevant-touch rate; low anchoring | baseline (and fallback when no gap oracle) |
| Gap-seeded splitting | `pipelines.gap_seeded_splitting_estimate` / `GapSeededSplittingEstimator` | light-basin coverage via seeding; $\hat R$ visibility | **default** (splitting); baseline for the subregion stack |
| Subregion-kernel splitting | `splitting_subregion.SubregionSplittingEstimator` | rejection-free kernel; basin-to-basin jumps | candidate (arXiv:2607.27153) |
| + $\hat R$ stopping | `subregion_splitting_estimate(stop_rhat=...)` | fixed per-level budgets | candidate toggle |
| + BAR level ratios | `subregion_splitting_estimate(ratio_estimator="bar")` | one-sided ratio variance | candidate toggle |
| Gap-seeded subregion pipeline | `pipelines.gap_seeded_subregion_estimate` / `GapSeededSubregionEstimator` | seeding + kernel coverage combined | candidate |
| Rejection-sampled spectrum | `failure_spectrum.FailureSpectrumEstimator` | (original ansatz implementation) | building block; baseline |
| Fixed-weight gap-splitting | `gap_splitting.GapSplittingEstimator` | $f(w)$ below the rejection floor | building block |
| Gap-aux spectrum | `pipelines.gap_spectrum_estimate` / `GapSpectrumEstimator` | onset pinned by data | **default** (spectrum) |
| Brute-force enumeration | `malignant.MalignantSetEstimator` | (original exact bound) | building block (small catalogs only) |
| Connected census / Knuth / core planting | `connected_enumeration.py`, `knuth_counting.py`, `core_planting.py` | $m(v)$ and $f(w)$ structure at scale | building blocks |
| Subset peeling | `core_planting.peel_to_minimal_subset` (`peel_method="subset"`) | oracle cost of core harvesting | candidate toggle (default `"single"` = baseline) |

**Explicitly frozen baselines.** `RareEventSplittingEstimator`,
`LocalSplittingEstimator`, `gap_seeded_splitting_estimate` /
`GapSeededSplittingEstimator`, `FailureSpectrumEstimator`, and the
single-element peel (`peel_method="single"`, byte-identical results and RNG
stream) are all unchanged by the arXiv:2607.27153 integration. Any
improvement claimed for the new stack is measured against these, on identical
schedules, anchors, seeds, and budgets — see
the standalone repo's `benchmarks/REPORT.md` (paper-methods
campaign).

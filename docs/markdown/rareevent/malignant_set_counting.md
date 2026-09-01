# Malignant set counting: the low-weight failure structure of a QEC decoder

*A methods note for `pygsti.extras.rareevent` (originally the standalone error-rate-estimation package). Companion to the standalone repo's `benchmarks/REPORT.md`
(which benchmarks the full logical-error-rate estimators); this note explains the low-weight
counting problem those estimators reduce to, the pre-existing tools, and the four new counting
routines added in July 2026: `gap_splitting.py`, `connected_enumeration.py`,
`knuth_counting.py`, and `core_planting.py`.*

---

## 1. The problem

### 1.1 Setting

Everything operates on the shared vocabulary defined in
`pygsti/extras/rareevent/rare_event.py`: a `MechanismCatalog` of $n$ independent Bernoulli
**error mechanisms** (one per hyperedge of the flattened Stim detector error model), each with
detector targets, observable targets, and a probability $q_i(p)$ at physical rate $p$. A
**fault set** $E \subseteq \{0,\dots,n-1\}$ is a set of simultaneously active mechanisms. The
`FailureOracle` (catalog + fixed `pymatching` decoder) answers a single yes/no question:

> `fails(E)` — does the decoder, given the syndrome produced by $E$, predict the wrong value
> for a logical observable?

A fault set with `fails(E) = True` is called **malignant**. The logical failure rate is

$$P_{\mathrm{fail}}(p) \;=\; \sum_{E} \Pr[E;\,q(p)]\;\mathbf{1}[\text{fails}(E)],$$

and the whole difficulty of the low-$p$ regime is that the malignant sets carrying this sum
become astronomically rare under the sampling distribution.

### 1.2 Why both extrapolation methods hit the same wall

The package's two extrapolation estimators fail at high distance in opposite directions, for
reasons that turn out to be the *same* missing information:

- **Failure-spectrum ansatz** (`failure_spectrum.py`, arXiv:2511.15177). Writes
  $P_{\mathrm{fail}}(p) = \sum_w f(w)\,\Pr[W{=}w;q(p)]$, where $f(w)$ is the failure fraction
  of weight-$w$ fault sets and $W$ is Poisson-binomial. It *measures* $f(w)$ by conditional
  sampling, but only down to $f \sim 1/\text{max\_trials} \approx 5\times10^{-5}$. At $d=9$
  the smallest weight with any observed failure was 17 while the weights that dominate
  $P_{\mathrm{fail}}(10^{-4})$ are near 8; at $d=11$, 31 versus ~15. Everything below the
  floor is filled in by the smooth ansatz, whose onset cannot fall as steeply as the true,
  combinatorially crushed $f(w)$ near the minimum failing weight — so the fit errs **high**.
- **Rare-event splitting** (`rare_event.py`, `splitting_local.py`). Walks a Markov chain over
  malignant sets down a descending $p$ schedule. Each level ratio is an average of
  $(p_{k+1}/p_k)^{|E|}$-type weights over the malignant sets the chain visits; the chain can
  reach the *heavy* malignant sets but not the rare, light, near-minimal ones (every removal
  breaks the failure, every hop between light basins is $q$-suppressed), so the sample is
  overweight and every ratio comes out **low**, compounding per level.

Both methods are blind to the same object: the light, near-minimal malignant sets. The
spectrum replaces them with an optimistic guess; splitting substitutes the heavy sets it can
reach. Estimating the low-weight failure structure directly repairs both at once.

### 1.3 The counting reframing

At low weight, a fault set fails (to leading order) because it *contains* one small malignant
core, plus benign spectators. Define:

- $f(w) = \Pr[\text{fails}(E) \mid |E| = w]$ under the exact conditional distribution
  $\pi_w(E) \propto \prod_{i\in E} \mathrm{odds}_i$ at the reference rate
  ($\mathrm{odds}_i = q_i/(1-q_i)$) — the **failure spectrum**;
- $m(v)$ = the number of **minimal malignant clusters** of weight $v$: sets with
  `fails = True`, no failing proper subset, connected in the shared-detector graph
  (mechanisms adjacent iff they share a detector target).

To first order (uniform-probability approximation, one core plus spectators),

$$f(w) \;\gtrsim\; \sum_v m(v)\,\frac{\binom{n-v}{\,w-v\,}}{\binom{n}{w}},$$

with an exact heterogeneous-probability version obtained by weighting each cluster by its
probability mass (implemented as `predicted_f_w_weighted` in `connected_enumeration.py`).
The combinatorial factor is trivial; all the physics is in $m(v)$ for $v$ from the onset
weight $w_{\min} = \lceil d/2 \rceil$ up to $w_{\min}+2$ or so. Equivalently, a handful of
sub-floor $f(w)$ measurements pin the same onset. **Malignant set counting** is the problem of
producing those few numbers — $m(v)$ at small $v$, or $f(w)$ at small $w$ — with quantified
uncertainty, at distances where brute force is impossible.

An important empirical caveat discovered during validation (§7): not all minimal malignant
sets are connected clusters, so the display above is a lower bound on the onset, not an
identity. The four methods are chosen so that this gap is itself measurable.

---

## 2. The existing implementation

Three pre-existing pieces are the baseline the new routines extend or plug into.

### 2.1 `MalignantSetEstimator` (`malignant.py`) — exhaustive enumeration

The original counting method: iterate over **all** $\binom{n}{w}$ combinations up to
`max_weight`, call the oracle on each, and sum the exact configuration probabilities
$P(E) = \prod_{i\in E} q_i \prod_{i\notin E}(1-q_i)$ of the malignant ones. This gives a
strict lower bound on $P_{\mathrm{fail}}(p)$ that becomes tight as $p \to 0$, and — as a side
product — the complete list of malignant sets up to the weight cap, which is the exact ground
truth every new routine is unit-tested against on small instances.

Its limitation is absolute: cost is $\binom{n}{w}$ oracle calls. At $d=5$ the catalog has
$n = 3{,}706$ mechanisms, so even $w=3$ is $8.5\times10^{9}$ combinations — already out of
reach. It remains usable only for the repetition-code and $d=3$ test pipelines.

### 2.2 Fixed-weight rejection sampling (`failure_spectrum.py`) — the measurement floor

`sample_fixed_weight_failure_fraction` draws i.i.d. samples from the exact conditional
distribution $\pi_w$ by **exponential tilting + rejection**: tilt each Bernoulli so the mean
weight equals $w$ (tilting preserves the conditional distribution given the total weight),
then reject draws whose weight isn't exactly $w$. Unbiased and simple, but the cost of one
observed failure is $1/f(w)$ evaluated sets — this is precisely the $\sim 5\times10^{-5}$
floor of §1.2. The new routines reuse both the tilting utility (for seeding and for planting
fills) and the sampler itself (as the cross-check wherever $f(w)$ is measurable).

### 2.3 The MCMC kernels (`rare_event.py`, `splitting_swap.py`, `splitting_local.py`)

The splitting estimators contribute two reusable ingredients: `build_detector_adjacency`
(the shared-detector graph that defines cluster connectivity) and the **detector-adjacent
swap move** of `splitting_swap.py` — remove an active mechanism $i$, add an inactive $j$ that
shares a detector with $i$, with Metropolis–Hastings acceptance
$\min\!\big(1, \tfrac{\mathrm{odds}_j}{\mathrm{odds}_i}\cdot\tfrac{|C(i,E)|}{|C(j,E')|}\big)$.
Crucially, a swap **preserves cardinality**, which makes it exactly the right kernel for
fixed-weight sampling (method 1) — in its original habitat it was only a mixing accelerant
for the variable-weight conditional-failure chain.

### 2.4 The common output contract (`weight_points.py`)

All four new routines report through one shape: a `WeightPoint(method, kind, weight,
estimate, rel_err, exact, lower_bound, meta)` with `kind="f_w"` (failure fraction) or
`kind="m_v"` (cluster count), serialized to a shared JSONL record by `weight_point_record`.
Downstream consumers (spectrum fit constraints, splitting core-jump proposals, plots) can
therefore pool points from any mix of methods without caring which produced them.

---

## 3. Method 1 — Fixed-weight gap-splitting (`gap_splitting.py`)

**What it estimates:** $f(w)$ at fixed small $w$, arbitrarily far below the rejection floor.

**Idea.** "A uniform(-ish) weight-$w$ set fails" is itself a rare event — so estimate it with
subset simulation (multilevel splitting), the same trust-tested idea as the $p$-schedule
splitting estimator, but along a different coordinate. That needs a *continuous* measure of
"how close to failing" a non-failing set is. The decoder supplies one: the **complementary
gap**.

**The gap score.** Following the construction prototyped in the archived exploratory script
`benchmarks/diagnostics/complementary_gap.py` (lifted into the module as `make_gap_matching_from_vanilla_dem`), the
matching graph is rebuilt with every boundary edge that carries the logical observable
re-terminated at one explicit extra detector node. Forcing that node's syndrome bit to 0 or 1
and decoding with `return_weight=True` yields the minimum-weight correction in each logical
class. For a fault set $E$ with true observable class $t$,

$$G(E) \;=\; w_{1-t}(E) - w_t(E)$$

is the weight margin by which the *correct* class wins: $G > 0$ means the decoder succeeds,
$G < 0$ means it fails, and $G$ decreases continuously toward failure. (Ties $G=0$ are
resolved by the real oracle; the final failure event always is.) One evaluation costs two
decodes. The construction requires the logical to live on boundary edges only and a single
observable; both hold for the repetition-code and rotated-surface-code memory pipelines and
are checked at build time.

**The chain.** The target is $\pi_w$ restricted to a level set $\{G \le g\}$. The kernel is a
mixture of two weight-preserving swaps: a global swap (uniform $i \in E$, uniform
$j \notin E$; symmetric, accept with $\min(1,\mathrm{odds}_j/\mathrm{odds}_i)$) and the
detector-adjacent swap of §2.3 for local moves along the failure surface.

**The estimator.** Standard adaptive multilevel splitting: draw $N$ i.i.d. weight-$w$ seeds by
tilting + rejection; repeatedly set the next gap threshold at the empirical
$\rho$-quantile of the population's $G$ values, keep the survivors, resample and rejuvenate
them with constrained MCMC steps; stop when a $\ge\rho$ fraction of the population actually
fails. Then $\hat f(w) = \rho_1\rho_2\cdots\rho_L \times$ (final failure fraction). The
reported uncertainty is the log-space spread over independent repetitions (default 3).

**Cost scaling — the point of the method.** Each level resolves a factor $\sim\rho$, so the
number of levels grows like $\log(1/f(w))$ and total cost is *linear* in $-\log f(w)$, versus
the $1/f(w)$ of rejection sampling. Reaching $f \sim 10^{-12}$ is a budget knob, not a new
algorithm.

**Validation.** At $d=3$ and $d=5$, gap-splitting agreed with direct rejection sampling within
a factor 0.87–1.56 at all eight $(d,w)$ points tested (tolerance $\log 2.5$), including
$d{=}5, w{=}3$, where rejection sampling could not reach its 50-failure target in 20,000
trials but gap-splitting returned $f(3) = 8.3\times10^{-4}$ (rel. err. 0.16) in 7 s and ~93k
decodes. The gap sign matched the oracle on 200/200 random sets, and the swap kernel's
stationary distribution was verified against exact enumeration on a 6-mechanism ring.

---

## 4. Method 2 — Connected-cluster enumeration (`connected_enumeration.py`)

**What it estimates:** $m(v)$ **exactly** (plus the cluster lists themselves), for $v$ and $d$
where the connected search space fits a node budget.

**Idea.** `MalignantSetEstimator` dies because it enumerates all $\binom{n}{v}$ sets, almost
none of which are even candidates. Restricting to sets that are *connected in the
shared-detector graph* shrinks the space to roughly $n\,\Delta^{v-1}$ (graph degree
$\Delta$) — still exponential in $v$, but with a base of ~50–90 instead of $n$.

**Enumeration.** The classic duplicate-free connected-induced-subgraph enumeration: for each
root $r$ (the minimum element of the sets it generates), grow the set along an ordered
extension frontier with a "visited" discipline — each vertex is offered as an extension at
most once per root — which guarantees every connected set of size $\le K$ appears at exactly
one node of the search forest. Correctness (exact set-of-sets equality with brute force, no
duplicates) is unit-tested on synthetic ring/path/isolated-vertex graphs and against the
`MalignantSetEstimator` ground truth on the repetition code.

**Malignancy and minimality.** Each size-$v$ set is tested with the oracle; failing sets are
then checked for minimality by testing **all** $2^v - 2$ proper nonempty subsets. The full
subset check matters: decoder failure is *not monotone* in the fault set (adding a mechanism
can un-fail a failing set), so checking only $(v{-}1)$-subsets would be unsound. There is
also deliberately *no* "can't touch the observable" pruning — a set none of whose members
touch an observable can still flip the *prediction* through the decoder.

**Validation.** Complete censuses within a 4.5-minute budget:

| $d$ | $v$ | $m(v)$ (minimal) | non-minimal malignant | tree nodes | oracle calls | time |
|---|---|---|---|---|---|---|
| 3 | 2 | 2,815 | 0 | 15.7k | 21k | 0.3 s |
| 3 | 3 | 14,316 | 71,320 | 518k | 931k | 12 s |
| 5 | 2 | 800 | 0 | 173k | 175k | 2.8 s |
| 5 | 3 | **529,838** | 61,282 | 10.4M | 13.9M | 233 s |

The enumerated minimal clusters are stored in the result — they double as the core lists
methods 4 (and eventually the splitting chain's core-jump proposals) consume. $d=7$ at
$v = w_{\min} = 4$ will need the pruning hooks (geometric reach, decoder-gap bounds) noted as
future work in the module docstring, or method 3.

---

## 5. Method 3 — Knuth tree-size estimation (`knuth_counting.py`)

**What it estimates:** $m(v)$ **unbiasedly**, with error bars, when the enumeration tree of
§4 is too large to walk.

**Idea (Knuth 1975).** You don't have to visit a tree to count its leaves. Take one random
root-to-depth-$v$ path, at each node multiplying a running factor $X$ by the number of
children before descending into a uniformly chosen one; at depth $v$ return
$X \cdot \varphi(S)$ where $\varphi$ is the property being counted (here: fails **and**
minimal, same full-subset minimality test as §4). Because each set occupies exactly one node
of the duplicate-free forest, $\mathbb{E}[X\varphi] = m(v)$ exactly — one probe is an
unbiased estimate of the whole count. Variance comes from tree imbalance, and is driven down
by averaging probes.

**Root stratification.** The single biggest variance component is *which root* a probe starts
from (bulk vs. boundary subtrees differ enormously). The implementation therefore loops
deterministically over all $n$ roots as strata, runs `probes_per_root` probes in each, and
sums per-root means; the standard error follows from the within-root probe variances only.
The module is deliberately self-contained (it re-implements the same tree discipline rather
than importing `connected_enumeration.py`), which turns agreement between the two modules
into a genuine independent cross-check rather than a shared-bug tautology.

**Validation.**

| $d$ | $v$ | Knuth $\hat m(v)$ ± SE | exact (§4) | z |
|---|---|---|---|---|
| 3 | 2 | 2,817.6 ± 13.9 | 2,815 | +0.2 |
| 3 | 3 | 13,784 ± 419 | 14,316 | −1.3 |
| 5 | 3 | 531,370 ± 3,820 | 529,838 | **+0.4** |
| 5 | 4 | 9.40M ± 0.17M | (tree too large) | — |

The $d{=}5, v{=}3$ row is the headline: two independently implemented trees, exact vs.
sampled, agreeing to 0.4σ at 0.7% relative error from ~20 s of probes — and the $v=4$ row is
a number nothing else in the repo can currently produce. Diagnostic worth remembering:
96–99% of $d=5$ probes return zero (dead-end path or non-failing leaf), so pruning or probe
importance sampling has an order of magnitude of headroom for the $d \ge 7$ campaigns.

---

## 6. Method 4 — Core-planting importance sampling (`core_planting.py`)

**What it estimates:** a **certified lower bound** on $f(w)$ — unbiased for the part of the
failure event covered by a list of known malignant cores — plus a direct measurement of how
complete that core list is.

**Harvesting.** Any source of failing sets works: fixed-weight sampling at moderate weights,
the recorded `failing_states` of `../benchmarks/results/anchor_mc.jsonl`, MCMC visits. Each
failing set is **greedy-peeled** — repeatedly try removing single elements while the set
still fails — down to a 1-minimal failing core; cores are deduplicated and pruned to an
inclusion-minimal antichain. (Validity below requires nothing of the cores; small failing
cores just buy coverage.)

**Subset peeling (July 2026, arXiv:2607.27153 Algorithm 1).** The single-element peel costs
one oracle call per element per pass — $\Theta(W)$ calls just for the first pass over a
weight-$W$ harvested set. `peel_to_minimal_subset` instead peels by **random-subset
removal**: each round draws a subset size $s$ uniform on
$[1, \max(1, \lfloor |E|/2 \rfloor)]$ and a uniform $s$-subset $S$ of the current elements,
tests $E \setminus S$ with a *single* oracle call, and commits the removal iff the reduced
set is nonempty and still fails. Large draws strip many fluff mechanisms per call while the
set is heavy; the size cap shrinks with the set, degrading gracefully to single-element
moves near the core (removing more than half the set rarely leaves a failing
configuration). The subset phase exits after `max_rounds` total rounds or
`max_stall_rounds` consecutive uncommitted rounds, and a final single-element polish pass
(the baseline loop) then *guarantees* 1-minimality of the returned core. Select it with
`harvest_cores(..., peel_method="subset")`; the default `"single"` reproduces the
pre-existing peel byte-identically (same cores, same RNG stream), and `CountingOracle`
wraps any `ForwardSimulator` to count `fails` calls when comparing strategies. The unit
test pins the subset peel at $\le 0.7\times$ the baseline's deterministic 127 calls on a
synthetic weight-123 pattern hiding a weight-3 core; harvest-scale numbers on the real
pipelines belong to the paper-methods campaign — see `../benchmarks/REPORT.md`
(paper-methods campaign) and `method_evolution.md` §4.3 for the lineage.

**The proposal.** To sample weight-$w$ sets that are likely to fail: pick a core $c$ with
probability $\alpha_c$ (default: proportional to its planted probability mass), then fill
with $w - |c|$ mechanisms drawn from the *exact* conditional fixed-weight distribution on the
complement (tilting + rejection again). The mixture density of a sample $E$ sums over every
known core contained in it, and the importance weight collapses to a numerically clean form —
the $\prod_{i \in E}\mathrm{odds}_i$ cancels between target and proposal:

$$\frac{\pi_w(E)}{Q(E)} \;=\; \frac{1/Z_w}{\displaystyle\sum_{c\,\subseteq\,E}
\alpha_c \,\big/\, \big(Z^{(c)}\textstyle\prod_{i\in c}\mathrm{odds}_i\big)},$$

where $Z_w$ and each $Z^{(c)}$ are elementary symmetric polynomials of the odds, computed
stably from `poisson_binomial_pmf`. All terms are evaluated in log space.

**Semantics.** The estimator $\tfrac1M\sum \mathbf{1}[\text{fails}]\,\pi_w/Q$ is unbiased for
$\Pr[\text{fails} \wedge E \text{ contains a known core} \mid W{=}w] \le f(w)$: a certified
lower bound whose only systematic is core-list coverage — and that coverage is *itself
measurable* (the fraction of independently-sampled failing sets that contain a known core).
Note the failure indicator is always evaluated: containing a malignant core does **not**
imply failure (non-monotonicity again). A pleasing special case: at $w = w_{\min}$ with
mass-weighted $\alpha$, the importance weight is provably constant across samples — the
covered-part estimate has literally zero variance.

**Validation.** With the complete minimal-core list on the repetition code, the estimator
reproduces exact $f(w)$ within error; with half the cores deleted, it matches the exactly
computed covered fraction (i.e., it is honest about what it covers). On the benchmark
pipelines with ~130–160 harvested cores it behaved exactly as a lower bound should — every
estimate at or below the rejection cross-check — while the coverage diagnostic exposed the
real story: harvested cores covered only 26–34% of failing sets at $d=3$ and the bound sat at
0.4–1.5% of true $f(w)$ at $d=5$. Out of the box this method is a *cheap consistency check
and coverage meter*; fed the complete cluster lists of §4–5 it becomes tight near onset.

---

## 7. Cross-validation, the disconnected-cluster surprise, and how the pieces fit

**Agreement matrix (this session's runs).** Knuth vs. exact enumeration: 0.2–1.3σ everywhere
both exist, including the independent-implementation check at $d{=}5, v{=}3$ (0.4σ).
Gap-splitting vs. rejection sampling: within 0.87–1.56× at all eight points. Core planting
vs. exact/brute force: within 5 SE in all regimes, always from below. All 72 repo tests pass;
each module also carries an exact-ground-truth unit test on the repetition-code pipeline.

**The surprise.** Plugging the *complete* $d=5$ cluster census into the onset prediction
under-shoots the directly measured $f(w)$: ratio 0.83 at $w = w_{\min} = 3$, degrading to
0.61 and 0.49 at $w = 4, 5$ (and much worse, ~0.2–0.3, at $d=3$). At $w = w_{\min}$ there are
no spectators to blame, so the conclusion is structural: **a material fraction of minimal
malignant sets are not connected in the shared-detector graph** — spatially separated
mechanisms whose defects the decoder mis-pairs jointly, plausibly via boundary-assisted
matchings. Consequences:

- $m(v)$ from methods 2–3 is an exact **lower-bound structure** for the onset, not the whole
  onset; treat `predicted_f_w*` as a floor.
- Direct $f(w)$ measurement (method 1) is connectivity-agnostic and therefore the primary
  tool for rebuilding the spectrum onset; methods 2–3 anchor and cross-check it.
- The disconnected fraction shrank from $d=3$ to $d=5$ (0.2 → 0.83 at onset) but must be
  re-measured, not assumed away, at $d \ge 7$ — gap-splitting vs. the cluster prediction at
  $w_{\min}$ is exactly that measurement.

**Division of labor going forward.**

| Question | Tool |
|---|---|
| $f(w)$ below the rejection floor ($d\ge7$, near $w_{\min}$) | gap-splitting |
| Exact $m(v)$, cluster lists / cores | connected enumeration (small $d,v$) |
| $\hat m(v) \pm$ SE where enumeration explodes | Knuth counting |
| Coverage meter, cheap cross-check, certified floor | core planting |

**Integration status (July 2026).** Two integrations are implemented:

1. *Spectrum*: `failure_spectrum_estimate(..., aux_points=...)` accepts gap-splitting
   `WeightPoint`s as auxiliary fit data — each enters `fit_failure_spectrum` as a log-space
   residual with its own standard error, and an auxiliary point at a low weight tightens the
   upper bound on a fitted onset $w_0$ exactly like a counted failure would. The $d{=}5$ demo
   (`benchmarks/runners/run_gap_integration.py`) also surfaced the next systematic in line: with the
   onset $f(w)$ pinned accurately by gap points, the prediction at $p=10^{-4}$ lands at
   ~0.46× the catalog-MC truth — the *transform's* fixed-ratio assumption (relative
   mechanism probabilities independent of $p$, inexact for SI1000 via `ExactNoiseErrorModel`)
   becomes the dominant error once the fit itself is right.
2. *Splitting*: `local_splitting_estimate(..., num_chains=..., seed_states=...)` runs extra
   chains per level from light failing states harvested by gap-splitting
   (`harvest_states=`), pools their samples for each level ratio, and reports per-chain
   ratios plus a cross-chain split-$\hat R$. At $d=5$ the cross-chain $\hat R$ of 1.1–1.6
   confirms directly that chains started in different basins sample different level-ratio
   distributions — the multi-basin structure the single-chain estimator could not see.

**Remaining next steps.** (1) The heavier $d\ge7$ runs of all four routines plus the two
integrations, sequentially (the light validations above deliberately stopped at $d=5$) —
that is the regime where the spectrum floor actually binds and where multi-chain seeding
should move the splitting answer, not just the diagnostics. (2) A $p$-dependent transform
(re-deriving the weight distribution and/or $f(w)$ reference at each target $p$) to attack
the fixed-ratio systematic exposed in the demo. (3) Core-jump independence proposals in
`LocalConditionalFailureMCMC` (from the enumerated cluster lists, with the exact MH
correction) as a sharper alternative to seed-only multi-chain coverage.

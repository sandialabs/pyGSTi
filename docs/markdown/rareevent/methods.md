# The mathematics of the estimation methods

This note explains, from first principles, what every estimator in
`pygsti.extras.rareevent` computes and why it works. It is the companion to:

- the package docstring of `pygsti.extras.rareevent` — quickstart and API surface.
- [`method_evolution.md`](method_evolution.md) — the lineage story: which
  failure mode each generation of every method fixed, and what is frozen as a
  baseline.
- [`malignant_set_counting.md`](malignant_set_counting.md) — a deeper dive
  into the malignant-set *counting* problem and the four counting routines.
- the standalone repo's `benchmarks/REPORT.md` — measured accuracy and
  cost of every method on rotated surface codes up to distance 11.

## 1. The problem

A quantum error-correcting (QEC) memory experiment has a **logical failure
rate** $P_{\mathrm{fail}}(p)$: the probability, per shot, that after decoding
the syndrome the corrected observable disagrees with the true one, when every
physical operation fails at rate ~$p$. Below threshold, $P_{\mathrm{fail}}$
falls roughly as $p^{\lceil d/2\rceil}$ for code distance $d$, so the regime
one actually cares about (say $d = 11$, $p = 10^{-4}$) has failure rates of
order $10^{-12}$. Direct Monte Carlo needs $\sim 100/P_{\mathrm{fail}}$ shots
for 10% relative precision — at $10^{-12}$ that is $10^{14}$ shots, decades of
CPU time. Everything in this package exists to predict such numbers without
paying that cost.

## 2. The model: independent Bernoulli error mechanisms

All estimators operate on one shared abstraction, built once per circuit:

1. Start from a noiseless `stim.Circuit` and decorate it with a
   **noise model** (e.g. `SI1000NoiseModel`: 1q gates fail at $p$, 2q gates at
   $p$, measurement at $2p$, reset at $p$, idling at $p/3$).
2. Convert to a **detector error model** (DEM) with
   `decompose_errors=True, flatten_loops=True`. Each DEM error is an
   independent hyperedge: it fires with probability $q_i$, flipping a fixed
   set of detectors and possibly the logical observable(s). Decomposition
   separators (`^`) are split into separate independent mechanisms.
3. Collect the hyperedges into a **`MechanismCatalog`**: mechanism $i$ is
   identified by its index, with detector targets $D_i$, observable targets
   $O_i$, and reference probability $q_i(p_{\mathrm{ref}})$.

A **fault configuration** is a subset $E \subseteq \{1,\dots,n\}$ of
simultaneously fired mechanisms. Under the model, configurations are drawn as
independent Bernoullis,

$$
\Pr_p[E] \;=\; \prod_{i \in E} q_i(p) \prod_{j \notin E} \bigl(1 - q_j(p)\bigr),
$$

where $q_i(p)$ comes from an `ErrorModel`:

- **`ScaledMechanismErrorModel`** scales linearly,
  $q_i(p) = q_i(p_{\mathrm{ref}})\cdot p/p_{\mathrm{ref}}$ — cheap, exact for
  homogeneous first-order noise, inexact over wide $p$ ranges.
- **`ExactNoiseErrorModel`** re-decorates the circuit and regenerates the DEM
  at every requested $p$, giving exact $q_i(p)$ (the catalog's *index space*
  stays fixed at $p_{\mathrm{ref}}$, so pick $p_{\mathrm{ref}}$ large enough
  that no mechanism has probability 0 there).

The **failure event** is a deterministic function of the configuration: a
fixed decoder (`pymatching.Matching` built from the DEM at
$p_{\mathrm{ref}}$, wrapped in `FailureOracle`) receives the syndrome
$\sigma(E) = \oplus_{i\in E} D_i$ and fails iff its correction's observable
flips differ from the true flips $\oplus_{i\in E} O_i$. Write

$$
\mathcal{F} \;=\; \{E : \text{decoding } E \text{ fails}\},
\qquad
P_{\mathrm{fail}}(p) \;=\; \Pr_p[\mathcal{F}]
\;=\; \sum_{E \in \mathcal{F}} \Pr_p[E].
$$

Two things are worth internalizing:

- **Failure is $p$-independent.** Whether a given $E$ fails depends only on
  the decoder, never on $p$. Only the *measure* $\Pr_p$ moves with $p$. This
  is what makes every method below possible: they all re-weight or re-count
  the fixed set $\mathcal{F}$ under different measures.
- **The decoder is fixed for all methods, including Monte Carlo**, so every
  method estimates exactly the same quantity and cross-comparisons are
  meaningful.

## 3. Monte Carlo (the ground truth, where affordable)

Two tiers:

- **Circuit MC**: sample stim detector data directly, decode in batch. Ground
  truth for the full circuit-level problem.
- **Catalog MC**: sample the *catalog model* itself — per mechanism, draw
  Binomial counts per batch and place them on distinct shot rows, then decode
  the resulting sparse syndromes in batch. This samples exactly the
  independent-Bernoulli measure $\Pr_p$ that every rare-event method targets,
  so it is the correct referee for them; comparing it against circuit MC
  separately bounds the cost of the DEM-decomposition approximation itself
  (measured: ratios 0.74–1.16 across all overlap points).

With $F$ observed failures the relative standard error is $\approx
1/\sqrt{F}$, hence the "100 failures ⇒ 10%" rule, and hence the wall: time
scales as $1/P_{\mathrm{fail}}$.

## 4. Rare-event splitting (Bravyi–Vargo)

### 4.1 The identity

Pick a descending schedule $p_0 > p_1 > \dots > p_L$ (`p_scales`). The
telescoping identity

$$
P_{\mathrm{fail}}(p_L)
\;=\;
P_{\mathrm{fail}}(p_0)\,
\prod_{k=0}^{L-1}
\frac{P_{\mathrm{fail}}(p_{k+1})}{P_{\mathrm{fail}}(p_k)}
$$

reduces the problem to (a) an **anchor** $P_{\mathrm{fail}}(p_0)$ cheap enough
for direct MC, and (b) per-level **ratios**. Each ratio is an expectation
under the *conditional failure distribution* at the current level,
$\pi_k(E) = \Pr_{p_k}[E]/P_{\mathrm{fail}}(p_k)$ for $E \in \mathcal{F}$:

$$
\frac{P_{\mathrm{fail}}(p_{k+1})}{P_{\mathrm{fail}}(p_k)}
= \sum_{E\in\mathcal{F}} \frac{\Pr_{p_{k+1}}[E]}{\Pr_{p_k}[E]}\,\pi_k(E)
= \mathbb{E}_{E \sim \pi_k}\!\left[ w_k(E) \right],
\qquad
w_k(E) = \prod_{i \in E}\frac{q_i(p_{k+1})}{q_i(p_k)}
         \prod_{j \notin E}\frac{1-q_j(p_{k+1})}{1-q_j(p_k)}.
$$

Consecutive levels are chosen close enough that $w_k$ has moderate variance
under $\pi_k$; the ratio is estimated by a log-space mean
(`logmeanexp`) over MCMC samples from $\pi_k$.

### 4.2 Sampling $\pi_k$: Metropolis over failing sets

$\pi_k$ is known only up to normalization, which is exactly the MCMC setting.
The baseline chain (`ConditionalFailureMCMC`) proposes toggling one mechanism
$i$ chosen uniformly from all $n$; the Metropolis ratio for the unconditioned
Bernoulli measure is the odds factor $\mathrm{odds}_i^{\pm 1}$ with
$\mathrm{odds}_i = q_i/(1-q_i)$ ($+1$ when adding, $-1$ when removing), and
the failure conditioning is enforced by rejecting any accepted move whose
result does not fail (the oracle is consulted only *after* the cheap ratio
test passes).

**Why the baseline freezes at high distance.** At low $p$, $\pi_k$
concentrates on near-minimal *malignant sets* (failing sets with no failing
proper subset). Sitting on one, every remove breaks failure (rejected by
conditioning) and every add is suppressed by $\mathrm{odds}_i \sim q_i \ll 1$.
Worse, a uniform proposal touches a mechanism *relevant* to the current set
(sharing a detector with it) only with probability $|E|/n$, and $n$ grows to
~48k at $d=11$. The chain therefore explores a single malignant basin and the
level ratios are biased low — by orders of magnitude at $d \ge 9$.

**The local-proposal fix (`LocalSplittingEstimator`).** Define the detector
neighborhood $N(E) = \bigcup_{i \in E} \mathrm{nbr}(i)$, where
$\mathrm{nbr}(i)$ is the set of mechanisms sharing at least one detector with
$i$ (including $i$). Propose from the mixture

$$
q(i \mid E) \;=\; \frac{\beta}{n} \;+\; (1-\beta)\,
\frac{\mathbf{1}[i \in N(E)]}{|N(E)|},
\qquad \beta = 0.1,
$$

which raises the relevant-touch rate to $O(1)$ while the $\beta$ component
preserves irreducibility. Because the proposal now depends on the state, the
correct acceptance is Metropolis–Hastings,

$$
\alpha \;=\; \min\!\left(1,\;
\mathrm{odds}_i^{\pm 1}\,
\frac{q(i \mid E')}{q(i \mid E)}\right),
$$

with $E'$ the toggled state. $N(E)$ is maintained incrementally with a
cover-count structure so each step is $O(\mathrm{degree})$. Kernel exactness
is unit-tested against brute-force enumeration of the stationary distribution.

### 4.3 Anchoring low, and seeding chains from gap harvests

Two further repairs, both measured in `benchmarks/REPORT.md`:

- **Low anchor.** Per-level ratio bias compounds multiplicatively, and the
  biased levels are the *low-p* ones. So rather than anchoring at an easy
  $p_0$ and walking many levels, anchor at the **lowest $p_0$ that
  catalog MC can still measure directly** (pass `anchor_failure_rate` and a
  recorded failing `anchor_state`) and walk only the last few levels.
- **Gap-seeded multi-chain levels** (`num_chains`, `seed_states` — the
  package default via `gap_seeded_splitting_estimate`). Even the local chain
  started from an anchor-typical state can fail to reach *light* (near-onset,
  low-weight) malignant basins. Fixed-weight gap-splitting (§7) *visits*
  exactly those states, and harvests them as a side effect. Each level then
  runs $C$ chains — chain 0 from the anchor state, chains $1..C-1$ from
  harvested light states (valid at every level because failure is
  $p$-independent) — and pools all samples for the ratio. A cross-chain
  split-$\hat R$ on the log-weight samples is reported per level: $\hat R \gg
  1$ means chains sit in basins with different ratio statistics, i.e. the
  single-chain estimate was missing mass.

### 4.4 Diagnostics and error bars

Each level reports acceptance rates, per-chain log-ratios, and split-$\hat R$
of both the log-weights and the active-set weights. Curve-level uncertainty is
the seed-to-seed spread of the log-estimate over 3 independent runs
(log-SEM = log-SD/$\sqrt{3}$), with the anchor's binomial relative error
folded in quadrature.

### 4.5 The subregion (partial-resampling) proposal (arXiv:2607.27153)

Even the local chain of §4.2 is a *single-toggle* kernel: add moves pass with
probability $\sim q_i \ll 1$, and moving between distinct malignant basins
requires traversing non-failing intermediates that the conditioning rejects.
`SubregionConditionalFailureMCMC` (`splitting_subregion.py`) replaces the
toggle with a partial resample. Each step draws a **region**
$R \subseteq \{1,\dots,n\}$, including every index independently with
probability $r$ (`region_rate`), *independently of the current state*; keeps
$x_i' = x_i$ for $i \notin R$; and redraws every coordinate inside $R$ from a
resample distribution $f$:

$$
x_i' \sim \mathrm{Bernoulli}(f_i) \quad (i \in R),
\qquad
x_i' = x_i \quad (i \notin R).
$$

Given $R$, the proposal density is $g(E' \mid E, R) = \prod_{i \in R}
f_i^{x_i'}(1-f_i)^{1-x_i'}$, so the Hastings ratio against the target
$\pi(E) \propto \prod_i q_i^{x_i}(1-q_i)^{1-x_i}$ (restricted to failing $E$)
is

$$
A \;=\; \min\!\Bigl(1,\;
\frac{\pi(E')}{\pi(E)}\,
\frac{g(E \mid E', R)}{g(E' \mid E, R)}\Bigr)
\;=\;
\min\!\Bigl(1,
\prod_{i \in R:\, x_i' \ne x_i}
\bigl[\mathrm{odds}_q(i)/\mathrm{odds}_f(i)\bigr]^{x_i' - x_i}
\Bigr),
$$

with $\mathrm{odds}(i) = p_i/(1-p_i)$; coordinates with $x_i' = x_i$ cancel.
**With the default $f = q$ (resample at the current level's probabilities)
the product is identically 1**: the proposal is rejection-free with respect
to $\pi$, and acceptance reduces to the failure-conditioning indicator alone
— one call to `fails` per proposing step, never a $q$-suppressed ratio test.
Resampling part of the pattern at the current rate is already a valid $\pi$
move; Metropolis rejections are needed only to enforce the conditioning.
(With an explicit $f \ne q$ the per-step ratio above is evaluated before the
oracle — ratio first, decode second, as in the baselines.)

The paper's **core-resampling heuristic** sets $r = 1/w_{\min}$
(`default_region_rate`), with $w_{\min}$ the minimum failing weight
($\lceil d/2\rceil$). $r$ is a per-mechanism inclusion probability over all
$n$ indices, so the expected region size is $n/w_{\min}$; the point is that
each of the $\sim w_{\min}$ core errors of the current failing set is hit
with probability $1/w_{\min}$ — the proposal replaces about one core error
per step (while redrawing the fluff around it), a basin-to-basin jump in a
single move. A proposal that changes nothing ($R$ missed the active set and
the resample added nothing — common at low $p$) is counted as an accepted
no-op *without* consulting the oracle: $E' = E$ trivially still fails. No-op
fractions and oracle calls are tracked separately
(`KernelCounters`/`SubregionLevelDiagnostics`), so acceptance diagnostics
refer to real moves and decode costs are exact.

`subregion_splitting_estimate` / `SubregionSplittingEstimator` embed this
kernel in the same anchor-then-descend, multi-chain flow as §4.2–4.3, and
`gap_seeded_subregion_estimate` is the gap-harvest-seeded pipeline analogous
to `gap_seeded_splitting_estimate`. Two further opt-in components:

### 4.6 $\hat R$-driven adaptive level stopping (`stop_rhat`)

The baseline spends a fixed step budget at every level; easy levels waste
decodes, hard levels get no more. With `stop_rhat` (requires
`num_chains` $\ge 2$), each level's chains run in blocks of `block_steps`.
After each block, once `min_steps_per_chain` is reached, compute the
cross-chain **split-$\hat R$** of the per-chain series of log weight-ratios
$\log w_k(E_t)$ — the very statistic the level ratio averages — with the
first half of each chain discarded as burn-in. Stop as soon as
$\hat R \le$ `stop_rhat`, or at the hard cap `max_steps_per_chain`. The level
estimate is then computed from exactly the second-half series the diagnostic
certified, and levels that hit the cap without converging are flagged
(`SubregionLevelDiagnostics.rhat_threshold_met = False`).

### 4.7 Two-sided level ratios: the Bennett acceptance ratio (`ratio_estimator="bar"`)

The baseline estimates each $Z_{k+1}/Z_k$ one-sidedly from level-$k$ samples
via $\widehat{\log \text{ratio}} = \operatorname{logmeanexp}_E \log w_k(E)$.
But the descent produces samples at level $k+1$ anyway, and Bennett (1976)
showed how to combine the two sample sets with minimum variance among a wide
class of two-sided estimators. Write $\ell(E) = \log w_k(E) =
\log \Pr_{p_{k+1}}[E] - \log \Pr_{p_k}[E]$, evaluated on $n_F$ "forward"
samples $E \sim \pi_k$ and $n_R$ "reverse" samples $E \sim \pi_{k+1}$. With
$C = \log(Z_{k+1}/Z_k)$ and the sample-size shift $M = \log(n_R/n_F)$, BAR is
the root of the monotone scalar equation

$$
\sum_{E \sim \pi_k} s\bigl(\ell(E) - C - M\bigr)
\;=\;
\sum_{E \sim \pi_{k+1}} s\bigl(-(\ell(E) - C - M)\bigr),
\qquad
s(x) = \frac{1}{1 + e^{-x}} .
$$

The left side is strictly decreasing and the right side increasing in $C$, so
`bennett_log_ratio` solves it by bisection on a bracket expanded around the
two one-sided estimates ($\operatorname{logmeanexp}$ of the forward deltas
and its reverse counterpart); convergence is unconditional. The reverse sets
are free for every interior level; only the final $p_L$ needs one extra
sampling run, which `subregion_splitting_estimate` performs when
`ratio_estimator="bar"`. The forward estimate is always reported alongside
(`SubregionLevelDiagnostics.forward_log_ratio` vs. `bar_log_ratio`) so the
two can be compared per level.

The measured effect of §4.5–4.7 against the frozen §4.2–4.3 baselines (same
schedules, anchors, seeds, budgets) is the subject of the paper-methods
campaign — see the standalone repo's `benchmarks/REPORT.md`
(paper-methods campaign); the paper itself reports 2–10× fewer decodes to
reach $\hat R \le 1.05$, growing with distance (arXiv:2607.27153). The full
lineage narrative is in [`method_evolution.md`](method_evolution.md).

## 5. Malignant-set enumeration (exact lower bound)

`MalignantSetEstimator` enumerates every configuration of weight
$\le w_{\max}$ and sums the exact probabilities of the failing ones:

$$
P_{\mathrm{fail}}(p) \;\ge\;
\sum_{\substack{E \in \mathcal{F} \\ |E| \le w_{\max}}} \Pr_p[E]
\;=\;
\prod_j (1-q_j(p)) \sum_{\substack{E \in \mathcal{F},\, |E| \le w_{\max}}}
\;\prod_{i \in E} \mathrm{odds}_i(p).
$$

This is a certified lower bound for every $p$, and it converges to the truth
as $p \to 0$ because configurations of weight $> w_{\max}$ contribute
$O(p^{\,w_{\max}+1})$. The cost is $\sum_{v \le w_{\max}} \binom{n}{v}$ oracle
calls, so it is practical only for small catalogs (at $d=3$, weight 2 runs in
seconds and lands within ~5% of MC at $p = 10^{-4}$; at $d = 5$ weight 3 is
already ~$10^{10}$ calls). For structure-exploiting ways to count the
low-weight failing sets at larger $d$ — connected-cluster enumeration, Knuth
tree-size estimation, core-planting importance sampling — see
[`malignant_set_counting.md`](malignant_set_counting.md).

## 6. The failure-spectrum ansatz (arXiv:2511.15177)

### 6.1 Exact weight decomposition

Let $W = |E|$ be the fault weight. Conditioning on the weight,

$$
P_{\mathrm{fail}}(p) \;=\; \sum_{w} f(w)\;\Pr[W = w;\, q(p)],
$$

where $f(w) = \Pr[\mathcal{F} \mid W = w]$ is the **failure spectrum** and
$\Pr[W=w; q(p)]$ is the **Poisson-binomial** weight distribution of the
independent-but-non-identical Bernoullis $q_i(p)$ (computed exactly by the
standard $O(nw)$ convolution recursion). This generalizes the paper's uniform
binomial transform to heterogeneous mechanism probabilities.

The decomposition is exact at the reference rate. Reusing the *same* $f(w)$
at other $p$ assumes the conditional distribution of *which* mechanisms are
active given the weight does not change with $p$ — true when relative
mechanism probabilities are $p$-independent (exact for linear scaling;
approximate for SI1000 under `ExactNoiseErrorModel`, where the
$p$ / $2p$ / $p/3$ channel mix shifts with $p$). This **fixed-ratio
(transport) assumption** is the method's main systematic; the benchmark
campaign bounds it at roughly a factor ~2 at $p=10^{-4}$ for $d \le 7$
(measured against catalog-MC truth).

### 6.2 Measuring $f(w)$: exact fixed-weight conditional sampling

To sample $E$ from $\Pr[\,\cdot \mid W = w\,]$ exactly: **exponentially
tilt** each Bernoulli, $q_i \mapsto \tilde q_i = c\,q_i/(1-q_i+c\,q_i)$ with
$c$ chosen so the mean weight is $w$, then **reject** unless the sampled
weight is exactly $w$. Tilting multiplies every odds by the same $c$, which
cancels in the conditional distribution, so acceptance-conditioned samples
are exact draws. $f(w)$ is then the failure fraction of these draws —
measurable down to roughly $1/\text{max\_trials}$ and no further.

### 6.3 The ansatz fit

$f(w)$ is measured at ~12 log-spaced weights and fitted with the low-parameter
forms of the paper (Eq. 10), e.g. the 3-parameter form

$$
f^{(3)}(w) \;=\; a\left(1 - \exp\!\left[-\tfrac{f_0}{a}\,
\bigl(w/w_0\bigr)^{\gamma_1}\right]\right),\qquad f(w<w_0) = 0,
$$

with onset weight $w_0 = \lceil d/2\rceil$ (the minimum failing weight under
min-weight decoding), and asymptote fixed at $a = 1 - 2^{-K}$ for $K$
observables (a random very-heavy fault set fails with that probability). The
fit is weighted least squares in $\log f$, with per-point standard errors from
the binomial counts.

### 6.4 Auxiliary gap points (the package default)

At $d \ge 9$ the lowest weight with *any* counted failure sits far above
$w_0$ (e.g. first failures at $w = 17$ when $w_0 = 5$), yet the low-$p$
transform is dominated by weights near $w_0$ — the fit is pure extrapolation
across the gap, and its error bars underprice this because all ansatz forms
share the onset shape. The fix (`gap_spectrum_estimate`,
`aux_points=` in `failure_spectrum_estimate`): measure $f(w)$ at
$w_0, \dots, w_0 + 4$ directly with fixed-weight gap-splitting (§7), which
reaches $f \sim 10^{-10}$ in minutes, and add those measurements to the fit
as log-space residuals $\bigl(\log f_{\mathrm{gap}}(w),\,
\sigma_{\log}\bigr)$, with $\sigma_{\log}$ = the gap point's relative error
(floored at 0.05). The onset is then pinned by data instead of extrapolation.

### 6.5 Non-proportional schedules: when transport breaks, and the stratified repair

The transport assumption of §6.1 can be stated exactly. The fixed-weight
conditional distribution is

$$
\pi_w^{(p)}(E) \;\propto\; \prod_{i \in E} \mathrm{odds}_i(p)
\quad\text{on } |E| = w,
$$

which is invariant under a *global* rescaling of all odds (the same
cancellation that makes exponential tilting exact, §6.2). So $f(w)$ is
$p$-independent **iff the odds ratios $\mathrm{odds}_i(p)/\mathrm{odds}_j(p)$
are $p$-independent** — i.e. all mechanisms share one common schedule.
Proportional models ($q_i(p) = c_i\,p$) satisfy this to first order in $q$;
the SI1000 channel mix violates it mildly (the measured "transport drift").

A mechanism on a **non-proportional schedule** violates it structurally. The
benchmark case (`benchmarks/REPORT.md` §9) appends one DEM event
$\mathrm{error}(q_f)\ L_0$ — a logical flip with no detectors — whose
probability asymptotes to $q_f = 10^{-6}$ as $p \to 0$ instead of vanishing.
Its relative odds grow like $1/p$ as $p$ falls, so the composition of a fixed
weight class tilts toward it, and one scalar per weight cannot follow. The
failure spectrum acquires an additive floor component (the floor flips only
the observable, so a floor-containing set fails whenever its circuit part
alone would not):

$$
f_{\mathrm{floor}}(w; p) \;\approx\; \frac{w\, q_f}{\sum_j q_j(p)},
\qquad
\sum_w \Pr[W{=}w]\,\frac{w\,q_f}{\sum_j q_j} = q_f\,\frac{\mathbb{E}[W]}{\sum_j q_j} = q_f .
$$

The correct $f(w; p)$ therefore reproduces the floor exactly — but the
*transported* $f(w; p_{\mathrm{ref}})$ yields
$q_f \cdot \sum_j q_j(p) / \sum_j q_j(p_{\mathrm{ref}}) \approx q_f\, p/p_{\mathrm{ref}}$:
**transport silently converts a constant floor into a proportional one.** In
practice the miss is threefold: the floor component of $f(w)$ at
$p_{\mathrm{ref}}$ sits far below the rejection floor (unmeasurable), the
ansatz enforces $f(w < w_0) = 0$ while the floor makes weight-1 sets fail, and
the transform scales away whatever survived. Measured effect: the gap-aux
spectrum lands at $2\times 10^{-12}$ against a catalog-MC truth of
$8.8\times 10^{-7}$ at $d=11$, $p=10^{-4}$.

**The stratified repair.** Condition on the state of the specially-scheduled
set $S$ and keep the scalar spectrum only for the well-behaved remainder:

$$
P_{\mathrm{fail}}(p) \;=\; \sum_{S' \subseteq S} \Pr[S' \text{ active};\, p]\,
\sum_w f_{S'}(w)\, \Pr[W_{\mathrm{rest}} = w;\, q_{\mathrm{rest}}(p)],
$$

where each $f_{S'}$ is measured with $S'$ forced active, under the remaining
mechanisms' conditional — which is $p$-invariant again, so ordinary transport
applies stratum by stratum. For an *observable-only* special mechanism no new
measurement is needed at all: $\mathrm{fails}(E \cup \{\mathrm{floor}\}) =
\lnot\,\mathrm{fails}(E)$ identically (the syndrome is unchanged, the true
observable flips), hence $f_{\mathrm{floor}}(w) = 1 - f_\varnothing(w)$ and
the estimator collapses to the closed form

$$
P_{\mathrm{fail}}(p) \;=\; P_c(p) \;+\; q_f(p)\,\bigl(1 - 2\,P_c(p)\bigr),
$$

with $P_c$ the ordinary spectrum transform over the circuit-only
probabilities. On the benchmark this lands at 0.85–1.13× of catalog MC at
$p=10^{-4}$ for $d = 5$–$11$ (from 0.26×–0.000002× before the repair), with
±0.3% per-seed scatter and zero new decodes; the residuals are the circuit
stratum's own pre-existing systematics. One subtlety works in the repair's
favor: gap-splitting aux points approximate the *circuit-only* $f(w)$, because
the fixed-weight swap chain effectively never samples the floor mechanism —
exactly the stratum the fit needs.

Generalizations and limits: the stratum sum is $2^{|S|}$, so explicit
stratification suits a handful of special mechanisms (ones that also touch
detectors need their $f_{S'}$ actually measured — same tilted sampler, one
spectrum run per stratum). For broad schedule heterogeneity the options are a
**multivariate spectrum** $f(w_1, \dots, w_K)$ over per-schedule-class weights
(p-invariant under within-class proportionality; transform = product of
per-class Poisson-binomials), **importance-reweighting** stored fixed-weight
samples to the target $p$ (zero new decodes, but cannot create support the
reference measure never sampled — it cannot rescue a floor), or **re-measuring
$f(w; p)$ at each target $p$** (tilting makes the cost $p$-independent; this
removes the assumption entirely and is the general answer for complex
$q_i(p)$). A cheap a-priori diagnostic for whether transport is safe: the
spread of $\mathrm{d}\log q_i / \mathrm{d}\log p$ across mechanisms — order-unity
spread means tens of percent; orders-of-magnitude spread means stratify or
re-measure.

Implementation: `benchmarks/common.py` (`LogicalFloorErrorModel`,
`smooth_floor_schedule`) and the `spectrum_stratified` stage of
`benchmarks/runners/run_l0floor_campaign.py`.

## 7. The complementary gap, and fixed-weight gap-splitting

### 7.1 The gap score

For matchable codes with one logical observable carried only by
boundary edges, rebuild the matching graph with the logical boundary
re-terminated at one explicit **gap detector** node. Decoding the same
syndrome twice — once with the gap detector's bit pinned to the true logical
class $t$, once to $1-t$ — returns the min-weight correction weight in each
logical sector, and the **signed complementary gap** is

$$
G(E) \;=\; w_{1-t}(E) - w_{t}(E).
$$

MWPM picks the lighter class, so $G < 0 \Rightarrow E$ fails and $G > 0
\Rightarrow E$ does not (ties broken by implementation detail; the package
always resolves the actual failure event with the real oracle, never with
$\operatorname{sign} G$). The gap is a *continuous measure of how close a
configuration is to failing* — precisely the graded score that rare-event
subset simulation needs.

### 7.2 Subset simulation for $f(w)$

To estimate $f(w) = \Pr[\mathcal{F} \mid W=w]$ when it is far below the
rejection-sampling floor, run classical subset simulation over nested level
sets of the gap:

1. Draw a population of exact conditional weight-$w$ samples (§6.2).
2. Pick the next threshold $g_1$ as the `quantile` (default 0.25) order
   statistic of the population's gaps; the survival fraction is a factor of
   the estimate.
3. Resample survivors and **rejuvenate** with weight-preserving MCMC
   restricted to $\{G \le g_1\}$; repeat until the population's true failure
   fraction (by the real oracle) is $\ge$ the quantile, which becomes the
   final factor.

The estimate is the product of level survival fractions times the final
failure fraction; each level only resolves a probability of order `quantile`,
so cost grows *linearly* in $-\log f(w)$ rather than as $1/f(w)$. Repeats
(default 3) give a log-space spread as the error bar.

The rejuvenation kernel (`WeightPreservingSwapKernel`) targets
$\pi_w(E) \propto \prod_{i\in E}\mathrm{odds}_i$ on $\{|E| = w\}$ with a
mixture of two reversible swaps: a **global swap** ($i \in E$ out, $j \notin
E$ in, both uniform — symmetric proposal, acceptance
$\min(1, \mathrm{odds}_j/\mathrm{odds}_i)$) and a **detector-adjacent local
swap** with the Hastings factor $|C(i,E)| / |C(j,E')|$, where $C(i,E) =
\mathrm{nbr}(i)\setminus E$ is the candidate set the forward move drew from.
Level membership is enforced by an extra rejection after the ratio test.

As a side effect, the failing states encountered at the final level are
**harvested** (`harvest_states=`) — these are the light malignant sets used
to seed the splitting chains in §4.3.

### 7.3 What the campaign showed

With gap-derived onset data in place (both pipelines), the three
gap-founded estimates (low-anchor splitting, gap-seeded splitting,
gap-aux spectrum) agree within a factor ~2 at $d\le 9$, $p=10^{-4}$, and
within ~7× at $d = 11$ — down from the ~90× bracket the un-enhanced methods
left. Residual known systematics: the spectrum's fixed-ratio transport
assumption (§6.1), and possible residual downward bias in splitting levels
below the anchor.

## 8. Which method, when

| Regime | Recommended |
|---|---|
| $P_{\mathrm{fail}} \gtrsim 10^{-6}$ | Direct MC (catalog MC to validate estimators; circuit MC to check the DEM approximation). |
| Small catalogs ($\lesssim$ 1k mechanisms), low $p$ | Malignant enumeration: certified lower bound in seconds. |
| Below the MC wall, matchable single-observable DEM | **The two default pipelines**: `gap_seeded_splitting_estimate` (central/lower estimate) and `gap_spectrum_estimate` (independent cross-check; upper side at high $d$). Read disagreement between them as the honest uncertainty. |
| Gap construction not applicable | `LocalSplittingEstimator` (low-anchored) + plain `failure_spectrum_estimate`; treat as a bracket, not a point estimate. |
| $f(w)$ or $m(v)$ structure itself | `gap_splitting`, plus the counting routines in [`malignant_set_counting.md`](malignant_set_counting.md). |

The legacy uniform-toggle `RareEventSplittingEstimator` is retained only as a
benchmarking baseline; do not use it at $d \ge 7$.

The subregion stack of §4.5–4.7 (`SubregionSplittingEstimator`,
`gap_seeded_subregion_estimate`) is a candidate replacement for the splitting
row: same flow, rejection-free kernel, optional adaptive stopping and BAR.
It becomes the recommendation only once the paper-methods campaign in
the standalone repo's `benchmarks/REPORT.md` confirms it against the
frozen baselines; until then, prefer `gap_seeded_splitting_estimate` and see
[`method_evolution.md`](method_evolution.md) for the lineage.

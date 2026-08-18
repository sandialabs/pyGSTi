# Key concepts

pyGSTi's documentation assumes a handful of ideas that are easy to half-know: gauge freedom, error generators, parameterizations, and the statistics used to judge a fit. This page states them once, in prose, so the rest of the documentation can point here instead of re-explaining. There is no code on this page.

## Superoperators and the Liouville representation

A noisy quantum operation is a map from density matrices to density matrices, not from states to states. pyGSTi represents such a map by flattening the $d \times d$ density matrix into a length-$d^2$ vector and writing the operation as a $d^2 \times d^2$ matrix acting on it. That matrix is the *superoperator*, and this is the Liouville (or "process matrix") representation.

Two consequences follow, and both surface constantly in the API. First, a one-qubit gate is a $4 \times 4$ object, not $2 \times 2$. Second, the choice of basis for that flattened space is a real choice with real consequences for how numbers look: pyGSTi defaults to the normalized Pauli-product basis, in which a perfect gate's matrix has a recognizable sparse structure and error terms read off cleanly. See [bases](../advanced/conventions/Bases) if you need the conventions spelled out.

State preparations and measurements live in the same space. A prepared state is a vector in it, and a POVM effect is a covector, so probabilities come out of ordinary matrix products.

## Gauge freedom

This is the concept that most often trips up people reading GST output for the first time.

Every circuit outcome probability pyGSTi can predict has the form $\langle\langle E | G_k \cdots G_1 | \rho \rangle\rangle$. Pick any invertible $M$ and send $G \to M G M^{-1}$, $|\rho\rangle\rangle \to M |\rho\rangle\rangle$ and $\langle\langle E| \to \langle\langle E| M^{-1}$. Every adjacent $M^{-1}M$ cancels, so you get a completely different set of matrices that predicts *exactly the same probability for every circuit*. Nothing you can measure distinguishes the two. That freedom is the gauge.

So a gate set estimated from data is only determined up to gauge. This is not an artifact of the fitting procedure or a sign that the fit went badly; it is a property of what circuit data can possibly determine. It has a practical consequence that catches people out: **quantities like process fidelity and diamond distance are gauge-variant.** Compute the fidelity between your estimate and your target, apply a gauge transformation to the estimate, and the fidelity changes, even though the estimate still predicts every measurable quantity identically.

pyGSTi handles this by *gauge optimization*: after fitting, it searches the gauge orbit for the representative closest to your target model, and reports metrics against that. That makes the numbers comparable and interpretable, but it means the reported fidelity depends on the gauge-fixing choice as well as on your device. When you report such a number, report how it was gauge-fixed. Gauge-invariant alternatives exist and are preferable when they answer your question; see [gauge freedom](../guides/analysis/GaugeFreedom) and [metrics](../guides/analysis/Metrics).

## Error generators

Writing a noisy gate as $G = e^{L} G_{\text{target}}$ makes $L$ the *error generator*: the thing that vanishes when the gate is perfect. Error generators are more useful than the maps themselves for most purposes, because they compose additively to first order and their pieces have physical names.

pyGSTi sorts those pieces into four types, labeled by their first tuple element:

- **H** — Hamiltonian (coherent) errors. A systematic over- or under-rotation is an H term.
- **S** — stochastic errors. The diagonal of the non-Hamiltonian block; a depolarizing channel is all-S.
- **C** — correlation errors: the symmetric (real) half of that block's off-diagonal part.
- **A** — active errors: the antisymmetric (imaginary) half of the same off-diagonal part.

The names are pyGSTi's, following the taxonomy of [arXiv:2103.01928](https://arxiv.org/abs/2103.01928). Note that C and A are two halves of one thing, not a diagonal term and an off-diagonal term: both index a *pair* of basis elements, and they split the off-diagonal block between them.

Each term carries basis-element labels. The labels are objects rather than bare tuples, and they come in two flavours: `errorgen_coefficients()` returns global labels that print like `('H', ('X',), (0,))`, while `label_type='local'` gives the shorter `H(X)` form. An H term names one basis element; C and A name two.

**Coefficients are not error rates.** An error generator is exponentiated to produce a map, so a stochastic coefficient $C$ in the generator produces a map whose error rate is $(1 - e^{-d^2 C}) / d^2$, not $C$. Expanding, that is $C - d^2C^2/2 + \dots$, so the two agree to first order and diverge as the errors grow: the gap is negligible at $C = 10^{-4}$ and matters once you are constructing models with percent-level noise. If you build a test model by setting stochastic coefficients to the fidelity you want, check the fidelity you actually got.

pyGSTi exposes both views. `errorgen_coefficients` gives the generator's coefficients; `error_rates` applies the rescaling above, but **only to `S`-type terms** — H, C and A pass through untouched, and an H "rate" is a rotation angle rather than a rate at all. The setters mirror both. Use whichever matches what you mean, and do not assume `error_rates` has made every entry comparable.

## Parameterization

A pyGSTi `Model` is not just a collection of matrices; it is a collection of matrices together with a rule for which of them are reachable. That rule is the *parameterization*, and it is the constraint set the fit optimizes over.

`full` lets every entry of every superoperator vary, which is the least constrained and the most prone to overfitting. `full TP` constrains the gates to be trace-preserving and the state preparation and effects to be consistent with that. `CPTPLND` (still accepted under its old name `CPTP`) constrains the maps to be completely positive and trace-preserving by construction, via a Lindblad-form error generator.

The cost is substantial, which is the point. For the one-qubit XYI model pack, `full` has 60 parameters and `full TP` has 43; counting only the non-gauge parameters that actually affect the fit, the drop is 44 to 31. `CPTPLND` keeps 60 raw parameters but only 10 non-gauge ones.

The practical point is that **choosing a parameterization is choosing a physical assumption**, and GST is only as constrained as you make it. A `full` fit that produces a non-physical map is not necessarily a bug; it may be telling you your data does not pin the gate down. See [running GST](../guides/gst/RunningGST) for how to set this, and [operators](../advanced/models/Operators#choosing-types-when-you-build-a-model) for the full inventory of parameterization names.

## Judging a fit

GST fits a model by maximizing a likelihood, and the natural question afterward is whether the fit is any good. Two statistics do most of the work.

$\chi^2$ compares observed counts to predicted frequencies in the usual way. $2\Delta\log\mathcal{L}$ — twice the difference between the log-likelihood of your fitted model and that of the best possible model of the data — plays the same role and is what pyGSTi actually optimizes. Under the hypothesis that your model class contains the truth and you have enough counts, $2\Delta\log\mathcal{L}$ is approximately $\chi^2$-distributed with degrees of freedom equal to the number of independent circuit outcomes minus the number of **non-gauge** model parameters. The gauge distinction is not a technicality here: gauge directions do not change any predicted probability, so they cannot absorb misfit and must not be counted. pyGSTi uses `num_modeltest_params` for this, not `num_params`.

That distributional statement is what makes the number interpretable. pyGSTi converts it into $N_\sigma$: how many standard deviations the observed $2\Delta\log\mathcal{L}$ sits above its expected value. $N_\sigma$ near zero means the model describes the data as well as one could expect from statistical fluctuation alone. Large $N_\sigma$ means *model violation* — your data contains structure your model class cannot express.

Model violation is common with real devices, and it is information rather than failure. It usually means non-Markovian behavior: drift, context dependence, or crosstalk that a fixed gate set cannot represent. Note that $N_\sigma$ grows with the number of counts, so a large, well-sampled experiment on a slightly-imperfect model will report large $N_\sigma$; the size of the violation matters as much as its significance. See [judging the fit](../guides/gst/JudgingTheFit) for how to read these in practice, and [bad fits](../guides/gst/BadFits) for what to do about a large one.

## Where these show up

- [Your first GST run](FirstGST) is where you choose a parameterization for the first time.
- [Reading your results](ReadingResults) is where the gauge caveat first becomes concrete.
- [Model noise](../guides/models/ModelNoise) builds models out of the H/S/C/A terms above.

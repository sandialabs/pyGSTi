# Specialist protocols

These pages cover protocols and modelling techniques with a narrow audience: each one answers a question that matters a great deal to some people and not at all to most. They are grouped here so the main [characterization guides](../../guides/workflow/Workflow) stay short, not because they are unfinished or unsupported.

Read them individually — there is no reading order, and nothing here depends on anything else here.

## Beyond a fixed gate set

- [Time-dependent GST](TimeDependentGST) — when the gates themselves drift during the experiment, so a single set of process matrices cannot fit the data. Adds time dependence to the model and fits it.
- [Context-dependent GST](ContextDependence) — when a gate behaves differently depending on what surrounds it. Introduces distinct operation labels per context and tests whether the split is warranted.

## Systems that are not qubits

- [Qutrit GST](QutritGST) — GST on a three-level system.
- [Leakage modelled by hand](LeakageByHand) — modelling a leaky qubit as an explicit three-level system, built by hand rather than through the automatic path in [leakage](../../guides/gst/Leakage).
- [SU(2) qudit RB](SU2QuditRB) — benchmarking global SU(2) rotations on a single spin-$j$ qudit via rank-1 synthetic-SPAM RB.

## Narrow-purpose estimators

- [Robust phase estimation](RobustPhaseEstimation.md) — a small number of circuits gives a precise estimate of a few rotation angles. Much cheaper than GST when those angles are all you want.
- [Idle tomography](IdleTomography) — characterizes the errors in a multi-qubit idle from a small, intuitive circuit set. **Read its warning first:** the reported intrinsic stochastic rates currently disagree with injected rates by a factor of two, and the discrepancy is unresolved.
- [Parity benchmarking](ParityBenchmarking) — the weight-$X$ disturbance metric between a reference and a test dataset, applied to a four-qubit parity check.
- [Robust GST via TVD](RobustGST-TVD) — swapping the final-stage objective function away from the likelihood, which makes the fit less sensitive to a small number of badly-behaved circuits.
